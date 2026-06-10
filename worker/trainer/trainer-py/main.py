import os
os.environ["JAX_LOG_COMPILES"] = "0"

import asyncio
import base64
import hmac
import io
import json
import logging
import os
import ssl
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import jax
import orbax.checkpoint as ocp
import websockets
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed

from commondef.remoteworkermessages import (
    Checkpoint,
    ControlModelCheckpoint,
    ControlModelFFN,
    ControlModelPretrainedVLA,
    ControlOutputModeFlowMatchingAction,
    LoadModelMessage,
    PPOTrainingMethod,
    StartTrainingMessage,
    StopTrainingMessage,
    TrainingAccelerationMode,
    TrainingRemoteWorkerServerMessageTypes,
    ValueModelCheckpoint,
    ValueModelFFN,
)
from samples.arm import ArmModel
from simulator import MjxSimulator, MujocoSimulator, Simulator
from training import CallbackTrainingTaskSink, TrainingWorkflow

logger = logging.getLogger(__name__)


@dataclass
class WorkerRuntime:
    simulator: Simulator
    model: ArmModel
    model_path: str | None = None
    mjcf: str | None = None
    model_tmpdir: tempfile.TemporaryDirectory | None = None


@dataclass
class TrainingTaskState:
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


def _build_default_model(simulator: Simulator) -> ArmModel:
    input_dim = simulator.nq() + simulator.nv()
    policy_output_dim = simulator.nu() * 2
    hidden_dim = 256
    num_hidden_layers = 3
    policy, policy_store = ArmModel.init_policy(
        input_dimension=input_dim,
        output_dimension=policy_output_dim,
        hidden_layer_dimension=hidden_dim,
        num_layers=num_hidden_layers,
        add_softmax=False,
        control_policy=True,
    )
    value, value_store = ArmModel.init_policy(
        input_dimension=input_dim,
        output_dimension=1,
        hidden_layer_dimension=hidden_dim,
        num_layers=num_hidden_layers,
        add_softmax=False,
        control_policy=False,
    )
    return ArmModel.init_ppo_trained_model(
        policy=policy,
        val_estimator=value,
        pol_store=policy_store,
        val_estimator_store=value_store,
        nbody=simulator.nbody(),
    )


def _build_model_for_training(
    simulator: Simulator,
    params: PPOTrainingMethod,
    checkpoints: list[Checkpoint],
) -> ArmModel:
    checkpoints_by_id = {c.id: c for c in checkpoints if c.id is not None}

    if isinstance(params.control_model, ControlModelPretrainedVLA):
        if not isinstance(params.control_output_mode, ControlOutputModeFlowMatchingAction):
            raise ValueError(
                "PretrainedVLA control model requires control_output_mode=FlowMatchingAction(FlowNoise)"
            )
        from pretrained.pi0 import Pi0Model
        vla_spec = params.control_model.content
        noise_ffn_spec = params.control_output_mode.content.content.noise_model
        return Pi0Model(vla_spec.content, noise_ffn_spec)

    if isinstance(params.control_model, ControlModelFFN):
        control_ffn = params.control_model.content
    else:
        ckpt_id = params.control_model.content
        if not ckpt_id or ckpt_id not in checkpoints_by_id:
            raise ValueError(f"Checkpoint ID '{ckpt_id}' for control model not found in provided checkpoints")
        control_ffn = checkpoints_by_id[ckpt_id].control_model_ffn
        if control_ffn is None:
            raise ValueError(f"Checkpoint '{ckpt_id}' is missing control_model_ffn architecture params")

    if isinstance(params.value_model, ValueModelFFN):
        value_ffn = params.value_model.content
    else:
        ckpt_id = params.value_model.content
        if not ckpt_id or ckpt_id not in checkpoints_by_id:
            raise ValueError(f"Checkpoint ID '{ckpt_id}' for value model not found in provided checkpoints")
        value_ffn = checkpoints_by_id[ckpt_id].value_model_ffn
        if value_ffn is None:
            raise ValueError(f"Checkpoint '{ckpt_id}' is missing value_model_ffn architecture params")

    policy, policy_store = ArmModel.init_policy(
        input_dimension=int(control_ffn.input_dim),
        output_dimension=int(control_ffn.output_dim),
        hidden_layer_dimension=int(control_ffn.hidden_layer_dim),
        num_layers=int(control_ffn.num_hidden_layers),
        add_softmax=False,
        control_policy=True,
    )
    value, value_store = ArmModel.init_policy(
        input_dimension=int(value_ffn.input_dim),
        output_dimension=int(value_ffn.output_dim),
        hidden_layer_dimension=int(value_ffn.hidden_layer_dim),
        num_layers=int(value_ffn.num_hidden_layers),
        add_softmax=False,
        control_policy=False,
    )
    return ArmModel.init_ppo_trained_model(
        policy=policy,
        val_estimator=value,
        pol_store=policy_store,
        val_estimator_store=value_store,
        nbody=simulator.nbody(),
    )


def _load_checkpoint_into_model(
    model: ArmModel,
    model_zip_base64: str,
    training_id: str,
) -> None:
    restore_dir = f"/tmp/checkpoints/{training_id}/loaded"
    zip_bytes = base64.b64decode(model_zip_base64)
    os.makedirs(restore_dir, exist_ok=True)
    zip_path = f"/tmp/checkpoints/{training_id}/init.zip"
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(restore_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    policy_store = checkpointer.restore(
        os.path.join(restore_dir, "policy"),
        item=model.get_policy_store(),
    )
    value_store = checkpointer.restore(
        os.path.join(restore_dir, "value"),
        item=model.get_value_store(),
    )
    model.policy_store = policy_store
    model.value_store = value_store
    logger.info(
        "Checkpoint loaded into model: training_id=%s restore_dir=%s",
        training_id,
        restore_dir,
    )


def _extract_model_bundle_to_tempdir(
    model_zip_base64: str,
    mjcf_entrypoint: str,
) -> tuple[tempfile.TemporaryDirectory, str]:
    tempdir = tempfile.TemporaryDirectory(prefix="lagrangian-model-")
    root = Path(tempdir.name).resolve()
    try:
        zip_bytes = base64.b64decode(model_zip_base64)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for member in zf.namelist():
                member_path = (root / member).resolve()
                if member_path != root and root not in member_path.parents:
                    raise ValueError(f"Unsafe zip member path: {member}")
            zf.extractall(root)

        entrypoint_path = (root / mjcf_entrypoint).resolve()
        if entrypoint_path != root and root not in entrypoint_path.parents:
            raise ValueError("Invalid mjcf_entrypoint path")
        if not entrypoint_path.exists():
            raise FileNotFoundError(f"mjcf_entrypoint not found in package: {mjcf_entrypoint}")
        return tempdir, str(entrypoint_path)
    except Exception:
        tempdir.cleanup()
        raise


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
            xml_files = [
                m for m in zf.namelist()
                if "/" not in m and m.endswith(".xml")
            ]
            if not xml_files:
                raise FileNotFoundError("No XML found at zip root")
        entrypoint_path = (root / xml_files[0]).resolve()
        return tempdir, str(entrypoint_path)
    except Exception:
        tempdir.cleanup()
        raise


def _load_user_reward_fn(code: str) -> Any:
    import jax.numpy as jnp
    import numpy as np
    import math
    import functools
    import itertools
    ns: dict[str, Any] = {
        "jax": jax,
        "jnp": jnp,
        "numpy": np,
        "np": np,
        "math": math,
        "functools": functools,
        "itertools": itertools,
        "__builtins__": {},
    }
    exec(compile(code, "<reward_fn>", "exec"), ns)
    fn = ns.get("reward")
    if fn is None or not callable(fn):
        raise ValueError(
            "reward function not found in code"
        )
    return fn


def _log_message_payload(msg: str, training_id: str = "system") -> dict[str, Any]:
    return {
        "type": "TrainingUpdatesMessage",
        "content": {
            "id": training_id,
            "updates": [
                {
                    "training_run_state": {"type": "PPOTrainingRunState", "content": {"epoch_index": 0, "update_index": 0}},
                    "logs": [msg],
                    "metrics": [],
                    "checkpoints": [],
                }
            ],
        },
    }


async def _sender_loop(
    websocket: Any,
    send_queue: asyncio.Queue[dict[str, Any]],
) -> None:
    while True:
        payload = await send_queue.get()
        logger.debug("Sending payload to websocket: type=%s", payload.get("type"))
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

    runtime: WorkerRuntime | None = None
    training_tasks: dict[str, TrainingTaskState] = {}
    send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    sender_task = asyncio.create_task(_sender_loop(websocket, send_queue))
    loop = asyncio.get_running_loop()

    def thread_safe_send(payload: dict[str, Any]) -> None:
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

            if msg_type == TrainingRemoteWorkerServerMessageTypes.LOAD_MODEL_MESSAGE.value:
                try:
                    load_msg = LoadModelMessage.model_validate(content)
                    logger.info("Loading model for training_id=%s", load_msg.id)
                    if runtime is not None and runtime.model_tmpdir is not None:
                        runtime.model_tmpdir.cleanup()
                    model_tmpdir: tempfile.TemporaryDirectory | None = None
                    # DEBUG: use minimal model to test put_model performance
                    # _MINIMAL_MJCF = '<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
                    # simulator = MjxSimulator(mjcf=_MINIMAL_MJCF)
                    resolved_model_path: str | None = None
                    resolved_mjcf: str | None = None
                    if load_msg.model_zip_base_64 and load_msg.mjcf_entrypoint:
                        model_tmpdir, resolved_model_path = _extract_model_bundle_to_tempdir(
                            model_zip_base64=load_msg.model_zip_base_64,
                            mjcf_entrypoint=load_msg.mjcf_entrypoint,
                        )
                        logger.info(
                            "Extracted model bundle from zip for training_id=%s (entrypoint=%s)",
                            load_msg.id,
                            load_msg.mjcf_entrypoint,
                        )
                    else:
                        resolved_mjcf = load_msg.mjcf
                        logger.info(
                            "Using raw MJCF for training_id=%s",
                            load_msg.id,
                        )
                    simulator = MjxSimulator(model_path=resolved_model_path, mjcf=resolved_mjcf)
                    model = _build_default_model(simulator)
                    runtime = WorkerRuntime(
                        simulator=simulator,
                        model=model,
                        model_path=resolved_model_path,
                        mjcf=resolved_mjcf,
                        model_tmpdir=model_tmpdir,
                    )
                    await send_queue.put(
                        _log_message_payload(
                            "Loaded model from LoadModelMessage",
                            training_id=load_msg.id,
                        )
                    )
                except Exception as exc:
                    logger.exception("LoadModelMessage failed")
                    await send_queue.put(
                        _log_message_payload(f"LoadModelMessage failed: {type(exc).__name__}: {exc}")
                    )
                continue

            if msg_type == TrainingRemoteWorkerServerMessageTypes.START_TRAINING_MESSAGE.value:
                if runtime is None:
                    await send_queue.put(
                        _log_message_payload(
                            "LoadModelMessage must be sent before StartTrainingMessage"
                        )
                    )
                    continue
                try:
                    start_msg = StartTrainingMessage.model_validate(content)
                except ValidationError as exc:
                    logger.warning("Invalid StartTrainingMessage: %s", exc)
                    await send_queue.put(
                        _log_message_payload(f"Invalid StartTrainingMessage: {exc}")
                    )
                    continue

                training_id = start_msg.id
                logger.info("StartTrainingMessage accepted for training_id=%s", training_id)
                if training_id in training_tasks and not training_tasks[training_id].task.done():
                    await send_queue.put(
                        _log_message_payload(
                            f"Training task '{training_id}' already running",
                            training_id=training_id,
                        )
                    )
                    continue

                training_controller = start_msg.training_graph.training_job_controller
                acceleration_mode = training_controller.acceleration_mode
                ppo_params = training_controller.training_method.content
                sim_model_zip = start_msg.simulation_models.get(ppo_params.simulation_model)
                if sim_model_zip:
                    if runtime.model_tmpdir is not None:
                        runtime.model_tmpdir.cleanup()
                    sim_tmpdir, sim_model_path = _extract_sim_model_from_zip(sim_model_zip)
                    runtime.model_tmpdir = sim_tmpdir
                    runtime.model_path = sim_model_path
                    runtime.mjcf = None
                if acceleration_mode == TrainingAccelerationMode.MLCPUSIMULATIONCPU:
                    logger.info("Using MujocoSimulator (CPU) for training_id=%s", training_id)
                    runtime.simulator = MujocoSimulator(
                        model_path=runtime.model_path, mjcf=runtime.mjcf
                    )
                else:
                    logger.info("Using MjxSimulator (MJX) for training_id=%s acceleration_mode=%s", training_id, acceleration_mode)
                    runtime.simulator = MjxSimulator(
                        model_path=runtime.model_path, mjcf=runtime.mjcf
                    )
                try:
                    runtime.model = _build_model_for_training(runtime.simulator, ppo_params, start_msg.checkpoints)
                except Exception as exc:
                    logger.exception("Failed to build model for training_id=%s: %s", training_id, exc)
                    await send_queue.put(_log_message_payload(f"Failed to build model: {type(exc).__name__}: {exc}", training_id=training_id))
                    continue
                for ckpt in start_msg.checkpoints:
                    if ckpt.model_zip_base_64 is not None:
                        try:
                            _load_checkpoint_into_model(
                                runtime.model,
                                ckpt.model_zip_base_64,
                                training_id,
                            )
                        except Exception as exc:
                            logger.exception(
                                "Failed to load checkpoint for training_id=%s: %s",
                                training_id,
                                exc,
                            )
                            await send_queue.put(
                                _log_message_payload(
                                    f"Checkpoint load failed: {type(exc).__name__}: {exc}",
                                    training_id=training_id,
                                )
                            )
                user_reward_fn = _load_user_reward_fn(
                    ppo_params.reward_function.reward_function_code
                )
                workflow = TrainingWorkflow(
                    training_id=training_id,
                    training_controller=training_controller,
                    simulator=runtime.simulator,
                    model=runtime.model,
                    user_reward_fn=user_reward_fn,
                )
                stop_event = threading.Event()
                sink = CallbackTrainingTaskSink(thread_safe_send)

                async def run_training(
                    workflow_instance: TrainingWorkflow = workflow,
                    sink_instance: CallbackTrainingTaskSink = sink,
                    stop_event_instance: threading.Event = stop_event,
                ) -> None:
                    await asyncio.to_thread(
                        workflow_instance.run_training_blocking,
                        sink_instance,
                        stop_event_instance.is_set,
                    )

                task = asyncio.create_task(run_training(), name=f"training-{training_id}")

                def _on_task_done(
                    done_task: asyncio.Task[None],
                    training_id_local: str = training_id,
                ) -> None:
                    training_tasks.pop(training_id_local, None)
                    try:
                        done_task.result()
                        logger.info("Training task completed: training_id=%s", training_id_local)
                    except asyncio.CancelledError:
                        logger.info("Training task cancelled: training_id=%s", training_id_local)
                    except Exception as exc:
                        logger.exception(
                            "Training task failed: training_id=%s",
                            training_id_local,
                        )
                        thread_safe_send(
                            _log_message_payload(
                                f"Training task '{training_id_local}' failed: {type(exc).__name__}: {exc}",
                                training_id=training_id_local,
                            )
                        )

                task.add_done_callback(_on_task_done)
                training_tasks[training_id] = TrainingTaskState(task=task, stop_event=stop_event)
                await send_queue.put(
                    _log_message_payload(
                        f"Started training task '{training_id}'",
                        training_id=training_id,
                    )
                )
                continue

            if msg_type == TrainingRemoteWorkerServerMessageTypes.STOP_TRAINING_MESSAGE.value:
                try:
                    stop_msg = StopTrainingMessage.model_validate(content)
                except ValidationError as exc:
                    logger.warning("Invalid StopTrainingMessage: %s", exc)
                    await send_queue.put(
                        _log_message_payload(f"Invalid StopTrainingMessage: {exc}")
                    )
                    continue
                logger.info("StopTrainingMessage received for training_id=%s", stop_msg.id)
                task_state = training_tasks.get(stop_msg.id)
                if task_state is None:
                    await send_queue.put(
                        _log_message_payload(
                            f"No running training task for id '{stop_msg.id}'",
                            training_id=stop_msg.id,
                        )
                    )
                    continue
                task_state.stop_event.set()
                await send_queue.put(
                    _log_message_payload(
                        f"Stop signal sent for training task '{stop_msg.id}'",
                        training_id=stop_msg.id,
                    )
                )
                continue

            await send_queue.put(
                _log_message_payload(f"Unsupported message type: {msg_type}")
            )
    except ConnectionClosed as exc:
        logger.info(
            "Websocket closed for client: %s code=%s reason=%s",
            remote_addr,
            exc.code,
            exc.reason,
        )
    finally:
        if training_tasks:
            logger.warning(
                "Client %s disconnected with %d training task(s) still running — sending stop signals: %s",
                remote_addr,
                len(training_tasks),
                list(training_tasks.keys()),
            )
        for training_id, task_state in training_tasks.items():
            task_state.stop_event.set()
            logger.info("Stop signal sent to training task '%s' due to client disconnect", training_id)
        if runtime is not None and runtime.model_tmpdir is not None:
            runtime.model_tmpdir.cleanup()
        logger.info("Client disconnected: %s", remote_addr)
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


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
    logger.info(
        "Websocket settings: max_size=%s ping_interval=%s ping_timeout=%s",
        max_size,
        ping_interval,
        ping_timeout,
    )
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
    logger.info("Initializing worker runtime")
    shared_secret = _load_secret()
    logger.info("Worker shared secret loaded")
    asyncio.run(run_server(host, port, shared_secret))


if __name__ == "__main__":
    main()
