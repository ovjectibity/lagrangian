import os

os.environ["JAX_LOG_COMPILES"] = "0"
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_cuda_malloc_async=false"

import warp

if os.getenv("CUDA_DEBUG_MODE") == "1":
    print(dir(warp))
    print(warp.__path__)
    print(warp.__version__)

    # Enable before the affected module is compiled/first launched.
    warp.config.mode = "debug"

    # Attribute asynchronous CUDA errors to the launch that caused them.
    warp.config.verify_cuda = True

    # Print every Warp kernel launch and its arguments.
    warp.config.print_launches = True

# warp.config.cache_kernels = False

# # # Warp compiler, codegen, module-loading, and cache diagnostics.
# warp.config.log_level = warp.LOG_DEBUG

# # # Include Python source locations in Warp warnings.
# warp.config.verbose_warnings = True

# # # Optional: detect NaN/Inf production.
# warp.config.verify_fp = True

import asyncio
import base64
import hmac
import io
import json
import logging
import ssl
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import jax
import websockets
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed

from commondef.remoteworkermessages import (
    ControlModelVLA,
    RemoteWorkerServerMessageTypes,
    StartTestMessage,
    StartTrainingMessage,
    StopTestMessage,
    StopTrainingMessage,
    TrainingAccelerationMode,
)
from commondef.training import ModelInputSpec
from models.mlp import MLPModel
from models.model import TestingModel
from models.pi0.pi0 import Pi0Model
from simulator.simulator import MjxSimulator, MujocoSimulator, Simulator
from workflows.testing import CallbackTestingTaskSink, TestingWorkflow
from workflows.training import CallbackTrainingTaskSink, TrainingWorkflow

logger = logging.getLogger(__name__)

Array = jax.Array


@dataclass
class TrainingTaskState:
    task: asyncio.Task[None]
    stop_event: threading.Event


@dataclass
class TestingTaskState:
    task: asyncio.Task[None]
    stop_event: threading.Event


def _load_secret() -> str:
    secret_file = os.getenv("WORKER_SHARED_SECRET_FILE")
    if secret_file:
        content = Path(secret_file).read_text(encoding="utf-8").strip()
        if content:
            return content
    secret = os.getenv("WORKER_SHARED_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "Missing worker auth secret. Configure WORKER_SHARED_SECRET "
            "or WORKER_SHARED_SECRET_FILE."
        )
    return secret


def _extract_presented_secret(websocket: Any) -> str | None:
    request = getattr(websocket, "request", None)
    headers = getattr(request, "headers", None)
    if headers is None:
        headers = getattr(websocket, "request_headers", None)
    if headers is None:
        return None
    x_secret = headers.get("X-Worker-Secret")
    if x_secret:
        return x_secret
    auth_header = headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]
    path = getattr(request, "path", None)
    if path is None:
        path = getattr(websocket, "path", "")
    if path:
        secret_q = parse_qs(urlparse(path).query).get("secret")
        if secret_q:
            return secret_q[0]
    return None


def _build_model_for_testing(
    control_model: Any,
    rng_key: Any = None,
) -> TestingModel:
    _null_spec = ModelInputSpec(
        all_cameras=False, qpos_state=False, qvel_state=False,
        xpos_state=False, xquat_state=False, contact_pairs=False,
    )
    if isinstance(control_model, ControlModelVLA):
        vla_spec = control_model.content
        return Pi0Model(
            vla_spec,
            noise_mlp_spec=None,
            control_input_spec=vla_spec.input_spec,
            value_input_spec=_null_spec,
            reward_input_spec=_null_spec,
            rng_key=rng_key,
            n_action_steps=1,
        )

    control_mlp = control_model.content
    policy, policy_store = MLPModel.init_policy(
        input_dimension=int(control_mlp.input_dim),
        output_dimension=int(control_mlp.output_dim),
        hidden_layer_dimension=int(control_mlp.hidden_layer_dim),
        num_layers=int(control_mlp.num_hidden_layers),
        add_softmax=False,
        control_policy=True,
    )
    value, value_store = MLPModel.init_policy(
        input_dimension=int(control_mlp.input_dim),
        output_dimension=1,
        hidden_layer_dimension=int(control_mlp.hidden_layer_dim),
        num_layers=1,
        add_softmax=False,
        control_policy=False,
    )
    return MLPModel.init_ppo_trained_model(
        policy=policy,
        val_estimator=value,
        pol_store=policy_store,
        val_estimator_store=value_store,
        control_input_spec=control_mlp.input_spec,
        value_input_spec=_null_spec,
        reward_input_spec=_null_spec,
        reward_fn=None,
        rng_key=rng_key,
    )


def _extract_sim_model_from_zip(
    model_zip_base64: str,
) -> tuple[tempfile.TemporaryDirectory, str]:
    tempdir = tempfile.TemporaryDirectory(prefix="lagrangian-sim-model-")
    root = Path(tempdir.name).resolve()
    try:
        zip_bytes = base64.b64decode(model_zip_base64)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for member in zf.namelist():
                member_path = (root / member).resolve()
                if member_path != root and root not in member_path.parents:
                    raise ValueError(f"Unsafe zip member: {member}")
            zf.extractall(root)
            xml_files = [m for m in zf.namelist() if "/" not in m and m.endswith(".xml")]
            if not xml_files:
                raise FileNotFoundError("No XML found at zip root")
        return tempdir, str((root / xml_files[0]).resolve())
    except Exception:
        tempdir.cleanup()
        raise


def _extract_checkpoint_from_zip(
    checkpoint_zip_base64: str,
) -> tuple[tempfile.TemporaryDirectory, str]:
    tempdir = tempfile.TemporaryDirectory(prefix="lagrangian-checkpoint-")
    root = Path(tempdir.name).resolve()
    try:
        zip_bytes = base64.b64decode(checkpoint_zip_base64)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for member in zf.namelist():
                member_path = (root / member).resolve()
                if member_path != root and root not in member_path.parents:
                    raise ValueError(f"Unsafe zip member: {member}")
            zf.extractall(root)
        return tempdir, str(root)
    except Exception:
        tempdir.cleanup()
        raise


def _log_message_payload(msg: str, training_id: str = "system") -> dict[str, Any]:
    return {
        "type": "TrainingUpdatesMessage",
        "content": {
            "id": training_id,
            "updates": [{
                "training_run_state": {
                    "type": "PPOTrainingRunState",
                    "content": {"epoch_index": 0, "update_index": 0},
                },
                "logs": [msg],
                "metrics": [],
                "checkpoints": [],
            }],
        },
    }


async def _sender_loop(
    websocket: Any,
    send_queue: asyncio.Queue[dict[str, Any] | bytes],
) -> None:
    while True:
        payload = await send_queue.get()
        # Pre-serialized protobuf updates are sent as raw binary frames; JSON
        # control/log messages are sent as text frames.
        if isinstance(payload, (bytes, bytearray)):
            await websocket.send(payload)
            continue
        logger.debug("Sending payload: type=%s", payload.get("type"))
        await websocket.send(json.dumps(payload))


async def handle_connection(
    websocket: Any,
    shared_secret: str,
) -> None:
    remote_addr = getattr(websocket, "remote_address", None)
    logger.info("Client connected: %s", remote_addr)
    presented_secret = _extract_presented_secret(websocket)
    if not presented_secret or not hmac.compare_digest(presented_secret, shared_secret):
        await websocket.close(code=1008, reason="unauthorized")
        logger.warning("Unauthorized connection rejected: %s", remote_addr)
        return

    training_tasks: dict[str, TrainingTaskState] = {}
    testing_tasks: dict[str, TestingTaskState] = {}
    send_queue: asyncio.Queue[dict[str, Any] | bytes] = asyncio.Queue()
    sender_task = asyncio.create_task(_sender_loop(websocket, send_queue))
    loop = asyncio.get_running_loop()

    def thread_safe_send(payload: dict[str, Any] | bytes) -> None:
        loop.call_soon_threadsafe(send_queue.put_nowait, payload)

    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await send_queue.put(_log_message_payload("Invalid JSON message received"))
                continue
            if not isinstance(payload, dict):
                await send_queue.put(_log_message_payload("Invalid message payload type"))
                continue

            msg_type = payload.get("type")
            content = payload.get("content")
            logger.info("Received message: type=%s", msg_type)

            # ── START TRAINING ────────────────────────────────────────────
            if msg_type == RemoteWorkerServerMessageTypes.START_TRAINING_MESSAGE.value:
                try:
                    start_msg = StartTrainingMessage.model_validate(content)
                except ValidationError as exc:
                    logger.warning("Invalid StartTrainingMessage: %s", exc)
                    await send_queue.put(_log_message_payload(f"Invalid StartTrainingMessage: {exc}"))
                    continue

                training_id = start_msg.id
                if training_id in training_tasks and not training_tasks[training_id].task.done():
                    await send_queue.put(_log_message_payload(
                        f"Training task '{training_id}' already running", training_id=training_id
                    ))
                    continue

                training_controller = start_msg.training_graph.training_job_controller
                ppo_params = training_controller.training_method.content
                sim_model_zip = start_msg.simulation_models.get(ppo_params.simulation_model)

                if sim_model_zip is None:
                    await send_queue.put(_log_message_payload(
                        f"Training not started: simulation model "
                        f"'{ppo_params.simulation_model}' was not provided",
                        training_id=training_id,
                    ))
                    continue

                try:
                    model_tmpdir, model_path = _extract_sim_model_from_zip(sim_model_zip)
                except Exception as exc:
                    logger.exception(
                        "Simulation model extraction failed for training_id=%s",
                        training_id,
                    )
                    await send_queue.put(_log_message_payload(
                        f"Simulation model extraction failed: "
                        f"{type(exc).__name__}: {exc}",
                        training_id=training_id,
                    ))
                    continue

                checkpoint_tmpdirs: list[tempfile.TemporaryDirectory] = []
                checkpoint_paths: list[str] = []
                for ckpt in start_msg.checkpoints:
                    if ckpt.model_zip_base_64 is not None:
                        try:
                            checkpoint_tmpdir, checkpoint_path = (
                                _extract_checkpoint_from_zip(ckpt.model_zip_base_64)
                            )
                            checkpoint_tmpdirs.append(checkpoint_tmpdir)
                            checkpoint_paths.append(checkpoint_path)
                        except Exception as exc:
                            logger.exception(
                                "Checkpoint extraction failed for training_id=%s",
                                training_id,
                            )
                            await send_queue.put(_log_message_payload(
                                f"Checkpoint extraction failed: "
                                f"{type(exc).__name__}: {exc}",
                                training_id=training_id,
                            ))

                try:
                    workflow = TrainingWorkflow(
                        training_id=training_id,
                        training_controller=training_controller,
                        model_path=model_path,
                        checkpoint_paths=checkpoint_paths,
                        rng_key=jax.random.PRNGKey(0),
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to initialize training workflow for training_id=%s",
                        training_id,
                    )
                    await send_queue.put(_log_message_payload(
                        f"Failed to initialize training: "
                        f"{type(exc).__name__}: {exc}",
                        training_id=training_id,
                    ))
                    model_tmpdir.cleanup()
                    for checkpoint_tmpdir in checkpoint_tmpdirs:
                        checkpoint_tmpdir.cleanup()
                    continue

                stop_event = threading.Event()
                sink = CallbackTrainingTaskSink(thread_safe_send)
                tmpdir_refs = (model_tmpdir, *checkpoint_tmpdirs)

                async def run_training(
                    wf: TrainingWorkflow = workflow,
                    s: CallbackTrainingTaskSink = sink,
                    ev: threading.Event = stop_event,
                    tempdirs: tuple[tempfile.TemporaryDirectory, ...] = tmpdir_refs,
                ) -> None:
                    try:
                        await asyncio.to_thread(wf.run_training_blocking, s, ev.is_set)
                    finally:
                        for tempdir in tempdirs:
                            tempdir.cleanup()

                task = asyncio.create_task(run_training(), name=f"training-{training_id}")

                def _on_training_done(
                    done_task: asyncio.Task[None],
                    tid: str = training_id,
                ) -> None:
                    training_tasks.pop(tid, None)
                    try:
                        done_task.result()
                        logger.info("Training task completed: %s", tid)
                    except asyncio.CancelledError:
                        logger.info("Training task cancelled: %s", tid)
                    except Exception:
                        logger.exception("Training task failed: %s", tid)
                        thread_safe_send(_log_message_payload(
                            f"Training task '{tid}' failed unexpectedly", training_id=tid
                        ))

                task.add_done_callback(_on_training_done)
                training_tasks[training_id] = TrainingTaskState(task=task, stop_event=stop_event)
                await send_queue.put(_log_message_payload(
                    f"Started training task '{training_id}'", training_id=training_id
                ))
                continue

            # ── STOP TRAINING ─────────────────────────────────────────────
            if msg_type == RemoteWorkerServerMessageTypes.STOP_TRAINING_MESSAGE.value:
                try:
                    stop_msg = StopTrainingMessage.model_validate(content)
                except ValidationError as exc:
                    logger.warning("Invalid StopTrainingMessage: %s", exc)
                    await send_queue.put(_log_message_payload(f"Invalid StopTrainingMessage: {exc}"))
                    continue
                task_state = training_tasks.get(stop_msg.id)
                if task_state is None:
                    await send_queue.put(_log_message_payload(
                        f"No running training task for id '{stop_msg.id}'"
                    ))
                    continue
                task_state.stop_event.set()
                await send_queue.put(_log_message_payload(
                    f"Stop signal sent for training task '{stop_msg.id}'", training_id=stop_msg.id
                ))
                continue

            # ── START TESTING ─────────────────────────────────────────────
            if msg_type == RemoteWorkerServerMessageTypes.START_TESTING_MESSAGE.value:
                try:
                    start_msg = StartTestMessage.model_validate(content)
                except ValidationError as exc:
                    logger.warning("Invalid StartTestMessage: %s", exc)
                    await send_queue.put(_log_message_payload(f"Invalid StartTestMessage: {exc}"))
                    continue

                test_id = start_msg.id
                if test_id in testing_tasks and not testing_tasks[test_id].task.done():
                    await send_queue.put(_log_message_payload(f"Testing task '{test_id}' already running"))
                    continue

                test_controller = start_msg.test_graph.test_job_controller
                test_params = test_controller.test_params
                test_acceleration = test_controller.acceleration_mode

                sim_model_zip = start_msg.simulation_models.get(test_params.simulation_model)
                test_model_path: str | None = None
                test_model_tmpdir: tempfile.TemporaryDirectory | None = None
                if sim_model_zip:
                    try:
                        test_model_tmpdir, test_model_path = _extract_sim_model_from_zip(sim_model_zip)
                    except Exception as exc:
                        logger.exception("Failed to extract sim model for test_id=%s", test_id)
                        await send_queue.put(_log_message_payload(
                            f"Test not started: sim model extraction failed: {exc}"
                        ))
                        continue

                use_mjx = test_acceleration != TrainingAccelerationMode.MLCPUSIMULATIONCPU

                # Single RNG seed for testing, split for the model (eager
                # sequential sampling) and the workflow (batched scan).
                test_model_rng, test_workflow_rng = jax.random.split(jax.random.PRNGKey(0))

                try:
                    test_model = _build_model_for_testing(
                        test_params.control_model, rng_key=test_model_rng
                    )
                except Exception as exc:
                    logger.exception("Model build failed for test_id=%s", test_id)
                    await send_queue.put(_log_message_payload(
                        f"Test not started: model build failed: {exc}"
                    ))
                    if test_model_tmpdir is not None:
                        test_model_tmpdir.cleanup()
                    continue

                try:
                    test_simulator: Simulator = (
                        MjxSimulator(model_path=test_model_path, mjcf=None)
                        if use_mjx
                        else MujocoSimulator(model_path=test_model_path, mjcf=None)
                    )
                except Exception as exc:
                    logger.exception("Simulator creation failed for test_id=%s", test_id)
                    await send_queue.put(_log_message_payload(
                        f"Test not started: simulator creation failed: {exc}"
                    ))
                    if test_model_tmpdir is not None:
                        test_model_tmpdir.cleanup()
                    continue

                workflow = TestingWorkflow(
                    test_id=test_id,
                    test_controller=test_controller,
                    simulator=test_simulator,
                    model=test_model,
                    rng_key=test_workflow_rng,
                )
                stop_event = threading.Event()
                sink = CallbackTestingTaskSink(thread_safe_send)
                _test_tmpdir_ref = test_model_tmpdir

                async def run_testing(
                    wf: TestingWorkflow = workflow,
                    s: CallbackTestingTaskSink = sink,
                    ev: threading.Event = stop_event,
                    td: tempfile.TemporaryDirectory | None = _test_tmpdir_ref,
                ) -> None:
                    try:
                        await asyncio.to_thread(wf.run_test_blocking, s, ev.is_set)
                    finally:
                        if td is not None:
                            td.cleanup()

                task = asyncio.create_task(run_testing(), name=f"testing-{test_id}")

                def _on_test_done(
                    done_task: asyncio.Task[None],
                    tid: str = test_id,
                ) -> None:
                    testing_tasks.pop(tid, None)
                    try:
                        done_task.result()
                        logger.info("Testing task completed: %s", tid)
                    except asyncio.CancelledError:
                        logger.info("Testing task cancelled: %s", tid)
                    except Exception:
                        logger.exception("Testing task failed: %s", tid)
                        thread_safe_send(_log_message_payload(
                            f"Testing task '{tid}' failed unexpectedly"
                        ))

                task.add_done_callback(_on_test_done)
                testing_tasks[test_id] = TestingTaskState(task=task, stop_event=stop_event)
                continue

            # ── STOP TESTING ──────────────────────────────────────────────
            if msg_type == RemoteWorkerServerMessageTypes.STOP_TESTING_MESSAGE.value:
                try:
                    stop_msg = StopTestMessage.model_validate(content)
                except ValidationError as exc:
                    logger.warning("Invalid StopTestMessage: %s", exc)
                    await send_queue.put(_log_message_payload(f"Invalid StopTestMessage: {exc}"))
                    continue
                task_state = testing_tasks.get(stop_msg.id)
                if task_state is None:
                    await send_queue.put(_log_message_payload(
                        f"No running testing task for id '{stop_msg.id}'"
                    ))
                    continue
                task_state.stop_event.set()
                continue

            await send_queue.put(_log_message_payload(f"Unsupported message type: {msg_type}"))

    except ConnectionClosed as exc:
        logger.info(
            "Websocket closed: %s code=%s reason=%s", remote_addr, exc.code, exc.reason
        )
    finally:
        for tid, ts in list(training_tasks.items()):
            ts.stop_event.set()
            logger.info("Stop signal sent to training task '%s' on disconnect", tid)
        for tid, ts in list(testing_tasks.items()):
            ts.stop_event.set()
            logger.info("Stop signal sent to testing task '%s' on disconnect", tid)
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        logger.info("Client disconnected: %s", remote_addr)


def _maybe_ssl_context() -> ssl.SSLContext | None:
    cert_path = os.getenv("WS_SSL_CERT")
    key_path = os.getenv("WS_SSL_KEY")
    if not cert_path or not key_path:
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return context


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"", "none", "null", "unlimited", "inf", "infinite", "-1", "0"}:
        return None
    return int(value)


async def run_server(host: str, port: int, shared_secret: str) -> None:
    ssl_context = _maybe_ssl_context()
    protocol = "wss" if ssl_context else "ws"
    max_size = _env_optional_int("WS_MAX_MESSAGE_BYTES", 256 * 1024 * 1024)
    ping_interval = _env_optional_int("WS_PING_INTERVAL_SECONDS", None)
    ping_timeout = _env_optional_int("WS_PING_TIMEOUT_SECONDS", 20)
    logger.info("Starting websocket server on %s://%s:%s", protocol, host, port)
    async with websockets.serve(
        lambda ws: handle_connection(ws, shared_secret),
        host,
        port,
        ssl=ssl_context,
        max_size=max_size,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    ):
        await asyncio.Future()


def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    host = os.getenv("WS_HOST", "0.0.0.0")
    port = int(os.getenv("WS_PORT", "8765"))
    logger.info("JAX devices: %s", jax.devices())
    logger.info("JAX default backend: %s", jax.default_backend())
    shared_secret = _load_secret()
    logger.info("Worker shared secret loaded")
    asyncio.run(run_server(host, port, shared_secret))


if __name__ == "__main__":
    main()
