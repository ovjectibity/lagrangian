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
    # warp.config.verify_cuda = True

    # Print every Warp kernel launch and its arguments.
    warp.config.print_launches = True

    # # # Warp compiler, codegen, module-loading, and cache diagnostics.
    # warp.config.log_level = warp.LOG_DEBUG

# warp.config.cache_kernels = False

# # # Include Python source locations in Warp warnings.
# warp.config.verbose_warnings = True

# # # Optional: detect NaN/Inf production.
# warp.config.verify_fp = True

import asyncio
import hmac
import json
import logging
import shutil
import ssl
import stat
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import jax
import websockets
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed

from commondef.remoteworker.remoteworkermessages import (
    RemoteWorkerServerMessageTypes,
    StartTestWorkerMessage,
    StartTrainingWorkerMessage,
    StopWorkloadServer,
    StopWorkloadWorker,
    WorkloadStopReasonWorkerCompletedSuccess,
    WorkloadStopReasonWorkerRequestedByAppServer,
    WorkloadStopReasonWorkerWorkerFailure,
)
from commondef.records.assets import CheckpointRef, SimulationModelType
from commondef.records.workload import LogStream, LogType, WorkloadLogs
from commondef.records.workload_graph import TestMethodCustom, TestMethodDataset
from utils.asset_storage import download_file
from workflows.testing import CallbackTestingTaskSink, TestingWorkflow
from workflows.training import CallbackTrainingTaskSink, TrainingWorkflow

logger = logging.getLogger(__name__)
# IDs also identify local asset directories, so connections must not run the
# same workload concurrently and replace one another's downloaded assets.
_active_workloads: set[str] = set()


@dataclass
class WorkloadTaskState:
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


def _asset_id(value: str) -> str:
    if not value or value in {".", ".."} or any(c in value for c in ("/", "\\", "\x00")):
        raise ValueError("Workload and asset IDs must be single path components")
    return value


def _extract_asset_zip(archive_path: Path) -> None:
    root = archive_path.parent.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if (
                (target != root and root not in target.parents)
                or target == archive_path.resolve()
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(root)


def _download_workload_assets(
    message: StartTrainingWorkerMessage | StartTestWorkerMessage,
) -> tuple[dict[str, str], dict[str, tuple[CheckpointRef, Path]]]:
    """Retain downloaded ZIPs and extracted assets in the workload asset tree."""
    root = Path("/tmp/workload_assets") / _asset_id(message.workload_id)
    models: dict[str, str] = {}
    checkpoints: dict[str, tuple[CheckpointRef, Path]] = {}

    def prepare(
        uri: str, directory: Path, archive_name: str,
        required: list[str], *, directories: bool = False,
    ) -> None:
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Failed downloads/extraction must never reuse stale extracted weights.
        with tempfile.TemporaryDirectory(prefix=".download-", dir=directory.parent) as staging:
            stage = Path(staging)
            archive = stage / archive_name
            download_file(uri, archive)
            _extract_asset_zip(archive)
            for name in required:
                item = (stage / name).resolve()
                valid = item.is_dir() if directories else item.is_file()
                if stage.resolve() not in item.parents or not valid:
                    raise ValueError(f"Missing or unsafe asset content: {name}")
            if directory.exists():
                shutil.rmtree(directory)
            stage.rename(directory)

    for ref in message.simulation_models:
        asset_id = _asset_id(ref.simulation_model_id)
        if asset_id in models:
            raise ValueError(f"Duplicate simulation model reference: {asset_id}")
        if ref.model_type != SimulationModelType.MJCF:
            raise ValueError(f"Unsupported simulation model format: {ref.model_type}")
        directory = root / "models" / asset_id
        prepare(ref.uri_path, directory, f"m_{asset_id}.zip", [ref.base_file_name])
        base_file = (directory / ref.base_file_name).resolve()
        if directory.resolve() not in base_file.parents or not base_file.is_file():
            raise ValueError(f"Missing or unsafe simulation model base file: {ref.base_file_name}")
        models[asset_id] = str(base_file)
    for ref in message.checkpoints:
        asset_id = _asset_id(ref.checkpoint_id)
        if asset_id in checkpoints:
            raise ValueError(f"Duplicate checkpoint reference: {asset_id}")
        directory = root / "checkpoints" / asset_id
        prepare(
            ref.uri_path, directory, f"cp_{asset_id}.zip",
            ["weights", "optimizer"], directories=True,
        )
        checkpoints[asset_id] = (ref, directory)
    return models, checkpoints


def _log_message_payload(
    msg: str, workload_id: str = "system", testing: bool = False,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    log = WorkloadLogs(
        workload_id=workload_id, created_at=now, log_timestamp=now,
        log_id=str(uuid4()), log_line_text=msg, user_facing=True,
        log_type=LogType.ERROR, stream=LogStream.WORKER,
    )
    return {
        "type": "TestUpdatesMessage" if testing else "TrainingUpdatesMessage",
        "content": {
            "workload_id": workload_id, "logs": [log.model_dump(mode="json")],
            "metrics": [], "checkpoints": [], "camera_frames": [], "model_states": [],
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

    workload_tasks: dict[str, WorkloadTaskState] = {}
    send_queue: asyncio.Queue[dict[str, Any] | bytes] = asyncio.Queue()
    sender_task = asyncio.create_task(_sender_loop(websocket, send_queue))
    loop = asyncio.get_running_loop()

    def thread_safe_send(payload: dict[str, Any] | bytes) -> None:
        loop.call_soon_threadsafe(send_queue.put_nowait, payload)

    async def run_workload(
        message: StartTrainingWorkerMessage | StartTestWorkerMessage,
        testing: bool,
        stop_event: threading.Event,
    ) -> None:
        workload_id = message.workload_id

        def execute() -> None:
            workflow = None
            try:
                models, checkpoints = _download_workload_assets(message)
                if stop_event.is_set():
                    return
                if testing:
                    controller = message.test_graph.test_job_controller
                    method = controller.test_method
                    if isinstance(method, TestMethodCustom):
                        model_path = models.get(method.content.simulation_model_id)
                        if model_path is None:
                            raise ValueError("Required testing simulation model reference was not supplied")
                    elif isinstance(method, TestMethodDataset):
                        model_path = None
                    else:
                        raise ValueError("Unsupported testing method")
                    workflow = TestingWorkflow(
                        test_id=workload_id, test_controller=controller,
                        model_path=model_path, checkpoints=checkpoints,
                        rng_key=jax.random.PRNGKey(0),
                    )
                    if not stop_event.is_set():
                        workflow.run_test_blocking(
                            CallbackTestingTaskSink(thread_safe_send), stop_event.is_set,
                        )
                else:
                    controller = message.training_graph.training_job_controller
                    model_id = controller.training_method.content.simulation_model_id
                    model_path = models.get(model_id)
                    if model_path is None:
                        raise ValueError(f"Required training simulation model {model_id!r} was not supplied")
                    workflow = TrainingWorkflow(
                        training_id=workload_id, training_controller=controller,
                        model_path=model_path, checkpoints=checkpoints,
                        rng_key=jax.random.PRNGKey(0),
                    )
                    if not stop_event.is_set():
                        workflow.run_training_blocking(
                            CallbackTrainingTaskSink(thread_safe_send), stop_event.is_set,
                        )
            finally:
                if workflow is not None:
                    workflow.close()

        try:
            await asyncio.to_thread(execute)
            reason = (
                WorkloadStopReasonWorkerRequestedByAppServer()
                if stop_event.is_set() else WorkloadStopReasonWorkerCompletedSuccess()
            )
        except Exception as exc:
            logger.exception("Workload failed: workload_id=%s", workload_id)
            await send_queue.put(_log_message_payload(
                f"Workload failed: {type(exc).__name__}: {exc}", workload_id, testing,
            ))
            reason = WorkloadStopReasonWorkerWorkerFailure(content=f"{type(exc).__name__}: {exc}")
        finally:
            workload_tasks.pop(workload_id, None)
            _active_workloads.discard(workload_id)
        stopped = StopWorkloadWorker(workload_id=workload_id, reason=reason)
        await send_queue.put({
            "type": "StopTestingMessage" if testing else "StopTrainingMessage",
            "content": stopped.model_dump(mode="json"),
        })

    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue
            try:
                payload = json.loads(message)
                if not isinstance(payload, dict):
                    raise ValueError("Message must be a JSON object")
                msg_type = payload.get("type")
                content = payload.get("content")
                if msg_type in {
                    RemoteWorkerServerMessageTypes.START_TRAINING_MESSAGE.value,
                    RemoteWorkerServerMessageTypes.START_TESTING_MESSAGE.value,
                }:
                    testing = msg_type == RemoteWorkerServerMessageTypes.START_TESTING_MESSAGE.value
                    message_type = StartTestWorkerMessage if testing else StartTrainingWorkerMessage
                    start = message_type.model_validate(content)
                    _asset_id(start.workload_id)
                    if start.workload_id in _active_workloads:
                        raise ValueError(f"Workload {start.workload_id!r} is already running")
                    stop_event = threading.Event()
                    task = asyncio.create_task(
                        run_workload(start, testing, stop_event),
                        name=f"workload-{start.workload_id}",
                    )
                    _active_workloads.add(start.workload_id)
                    workload_tasks[start.workload_id] = WorkloadTaskState(task, stop_event)
                elif msg_type in {
                    RemoteWorkerServerMessageTypes.STOP_TRAINING_MESSAGE.value,
                    RemoteWorkerServerMessageTypes.STOP_TESTING_MESSAGE.value,
                }:
                    stop = StopWorkloadServer.model_validate(content)
                    state = workload_tasks.get(stop.workload_id)
                    if state is None:
                        raise ValueError(f"No running workload for {stop.workload_id!r}")
                    state.stop_event.set()
                else:
                    raise ValueError(f"Unsupported message type: {msg_type}")
            except (ValueError, ValidationError) as exc:
                logger.warning("Invalid worker message: %s", exc)
                await send_queue.put(_log_message_payload(str(exc)))
    except ConnectionClosed:
        logger.info("Websocket closed: %s", remote_addr)
    finally:
        states = list(workload_tasks.values())
        for state in states:
            state.stop_event.set()
        if states:
            # A to_thread task must finish before its assets or model resources
            # can be released. Cancelling it would leave its thread running.
            await asyncio.gather(*(state.task for state in states), return_exceptions=True)
        sender_task.cancel()
        try:
            await sender_task
        except (asyncio.CancelledError, ConnectionClosed):
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
