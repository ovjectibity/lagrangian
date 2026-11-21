"""Direct API helpers for the Python training smoke scripts.

Install tests/requirements.txt. Set ACCESS_KEY and MODEL_PATH;
PROJECT_ID optionally reuses a project. TEST_DRY_RUN=1 prints
request fixtures without accessing services. Artifacts are kept
under tmp/tests/test_run_assets, including downloaded camera PNGs.
Large ZIPs may need ROCKET_LIMITS='{json=128MiB}' on the app.
"""

import io
import json
import math
import os
from pathlib import Path
import re
import struct
import time
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit
import zipfile

import requests
from websockets.sync.client import connect

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/api").rstrip("/") + "/"
DRY_RUN = os.environ.get("TEST_DRY_RUN") == "1"
GROUPS = (
    "logs",
    "metrics",
    "checkpoints",
    "simulation_models",
    "camera_frames",
    "model_states",
)


def positive_integer(name, fallback):
    value = int(os.environ.get(name, fallback))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def api_request(path, *, method="GET", body=None, query=None, accepted=(200,)):
    key = os.environ.get("ACCESS_KEY")
    if not key:
        raise ValueError("ACCESS_KEY is required")
    response = requests.request(
        method,
        urljoin(BASE_URL, path),
        json=body,
        params=query,
        headers={"Authorization": f"Bearer {key}"},
        timeout=positive_integer("HTTP_TIMEOUT_SECONDS", 300),
    )
    if response.status_code not in accepted:
        hint = (
            " Increase the app's JSON body limit for model uploads."
            if response.status_code == 413
            else ""
        )
        raise RuntimeError(
            f"{method} {path}: {response.status_code} {response.text[:1500]}{hint}"
        )
    return response.json() if response.status_code == 200 else None


def prepare_model(label):
    if DRY_RUN:
        return "project-fixture", "model-fixture"
    if not os.environ.get("MODEL_PATH"):
        raise ValueError("MODEL_PATH is required")
    source = Path(os.environ["MODEL_PATH"]).resolve(strict=True)
    if not source.is_file():
        raise ValueError("MODEL_PATH must be a file")
    project = os.environ.get("PROJECT_ID")
    if not project:
        project = api_request("create_project", method="POST", body={"name": label})[
            "project_id"
        ]
    print(f"project_id={project}")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.parent.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent).as_posix())
    model = api_request(
        "save_model",
        method="POST",
        body={
            "project_id": project,
            "name": source.name,
            "model_type": "mjcf",
            "file_type": "zip",
            "base_file_name": source.name,
            "file_contents": list(output.getvalue()),
        },
    )["model_id"]
    print(f"model_id={model}")
    return project, model


def record_key(group, record):
    field = {
        "logs": "log_id",
        "checkpoints": "checkpoint_id",
        "simulation_models": "simulation_model_id",
        "camera_frames": "camera_frame_id",
        "model_states": "sim_model_state_entry_id",
    }.get(group)
    value = (
        record[field]
        if field
        else json.dumps(
            [
                record["created_at"],
                record["metric_name"],
                record["workload_structure_state_values"],
                record["metric_values"],
            ]
        )
    )
    if not value:
        raise ValueError(f"Missing identifier in {group}")
    return group, value


def coordinate(record):
    names = record["workload_structure_index_names"]
    values = record["workload_structure_state"]
    if (
        not names
        or len(names) != len(values)
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise ValueError("Invalid workload coordinates or LUT names")
    value = dict(zip(names, values))
    return (
        value.get("rollout", 0),
        value.get("trajectory", value.get("rollout_trajectory")),
        value.get("trajectory_timestep", value.get("rollout_trajectory_timestep")),
    )


def validate_world(world):
    if not world.get("bodies"):
        raise ValueError("Empty WorldBody")
    pending = list(world["bodies"])
    for body in pending:
        for field, size in (("pos", 3), ("quat", 4)):
            values = body.get(field)
            if (
                not values
                or len(values) != size
                or not all(math.isfinite(v) for v in values)
            ):
                raise ValueError(f"Invalid WorldBody {field}")
        pending.extend(body.get("bodies") or [])


def save_frame(frame, directory, label, timestamp):
    url = urljoin(BASE_URL, frame["uri_path"])
    target, base = urlsplit(url), urlsplit(BASE_URL)
    if (target.scheme, target.netloc) != (
        base.scheme,
        base.netloc,
    ) or not target.path.startswith(base.path + "assets/download/"):
        raise ValueError("Expected an app-owned camera download link")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.content
    if (
        len(data) < 24
        or data[:8] != b"\x89PNG\r\n\x1a\n"
        or data[12:16] != b"IHDR"
        or not all(struct.unpack(">II", data[16:24]))
    ):
        raise ValueError("Downloaded camera asset is not a valid PNG")
    parts = (
        timestamp,
        label,
        *coordinate(frame),
        frame["simulation_model_id"],
        frame["camera_id"],
    )
    filename = (
        "__".join(re.sub(r"[^a-zA-Z0-9_.-]", "_", str(part)) for part in parts) + ".png"
    )
    with (directory / filename).open("xb") as output:
        output.write(data)


def verify_history(workload_id, seen):
    historical, cursors = set(), set()
    cursor = None
    while True:
        page = api_request(
            "get_workload_updates",
            query={
                "workload_id": workload_id,
                "limit": 37,
                "cursor": cursor,
            },
        )
        if page["workload_id"] != workload_id:
            raise ValueError("History belongs to another workload")
        count = 0
        for update in page["updates"]:
            for group in GROUPS:
                count += len(update[group])
                historical.update(record_key(group, item) for item in update[group])
        if count > 37:
            raise ValueError("History page exceeds its limit")
        cursor = page["next_cursor"]
        if not cursor:
            break
        if cursor in cursors:
            raise ValueError("History cursor did not advance")
        cursors.add(cursor)
    if historical != seen:
        raise ValueError("Persisted history differs from live records")


def run_workload(project, graph, label):
    request = {
        "project_id": project,
        "training_graph": graph,
        "name": label,
        "description": f"API smoke test: {label}",
        "tags": ["smoke-test"],
    }
    if DRY_RUN:
        print(json.dumps({"dry_run": True, "request": request}))
        return
    deadline = time.monotonic() + positive_integer("TIMEOUT_SECONDS", 3600)
    max_updates = positive_integer("MAX_UPDATES", 10000)
    workload_id = api_request("start_training", method="POST", body=request)[
        "workload_id"
    ]
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = Path(
        os.environ.get(
            "TEST_ARTIFACT_DIR",
            Path(__file__).resolve().parent.parent / "tmp/tests/test_run_assets",
        )
    )
    directory = root / f"{timestamp}__{workload_id}"
    seen, states, cameras, checkpoints = set(), {}, {}, set()
    metric_count = 0
    stopped = False
    try:
        directory.mkdir(parents=True, exist_ok=False)
        print(f"workload_id={workload_id} artifacts={directory}")
        ticket = api_request(
            "get_workload_live_ticket",
            method="POST",
            query={"workload_id": workload_id},
        )["ticket"]
        parts = urlsplit(urljoin(BASE_URL, "get_workload_live_updates"))
        url = urlunsplit(
            (
                "wss" if parts.scheme == "https" else "ws",
                parts.netloc,
                parts.path,
                urlencode({"workload_id": workload_id, "ticket": ticket}),
                "",
            )
        )
        with connect(url, open_timeout=60, max_size=64 * 1024 * 1024) as socket:
            messages = 0
            with (directory / "updates.jsonl").open("w") as output:
                while not stopped:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Workload timed out")
                    raw = socket.recv(timeout=remaining)
                    if not isinstance(raw, str):
                        raise ValueError("Expected JSON text on the WebSocket")
                    message = json.loads(raw)
                    output.write(json.dumps(message) + "\n")
                    output.flush()
                    if "WorkloadStoppedMessage" in message:
                        terminal = message["WorkloadStoppedMessage"]
                        if terminal["workload_id"] != workload_id:
                            raise ValueError("Stop event belongs to another workload")
                        stopped = True
                        if terminal["reason"] != "CompletedSuccess":
                            raise RuntimeError(f"Workload failed: {terminal['reason']}")
                        break
                    messages += 1
                    if messages > max_updates:
                        raise ValueError("MAX_UPDATES reached before completion")
                    for update in message["WorkloadUpdatesMessage"]:
                        if update["workload_id"] != workload_id:
                            raise ValueError("Update belongs to another workload")
                        for group in GROUPS:
                            for record in update[group]:
                                if record.get("workload_id") not in (None, workload_id):
                                    raise ValueError(
                                        "Record belongs to another workload"
                                    )
                                key = record_key(group, record)
                                if key in seen:
                                    continue
                                seen.add(key)
                                if group == "logs":
                                    print(record["log_line_text"])
                                elif group == "metrics":
                                    values = record["metric_values"]
                                    if len(values) != len(
                                        record["workload_structure_state_values"]
                                    ) or not all(
                                        math.isfinite(value) for value in values
                                    ):
                                        raise ValueError("Invalid metric batch")
                                    metric_count += len(values)
                                elif group == "checkpoints":
                                    checkpoints.add(record["model_type"])
                                elif group == "model_states":
                                    validate_world(record["state"])
                                    position = coordinate(record)
                                    if position in states:
                                        raise ValueError(
                                            f"Repeated state coordinate {position}"
                                        )
                                    states[position] = record["simulation_model_id"]
                                elif group == "camera_frames":
                                    save_frame(record, directory, label, timestamp)
                                    camera = (
                                        record["simulation_model_id"],
                                        record["camera_id"],
                                    )
                                    position = coordinate(record)
                                    entries = cameras.setdefault(camera, set())
                                    if position in entries:
                                        raise ValueError(
                                            f"Repeated camera coordinate {position}"
                                        )
                                    entries.add(position)
        params = graph["training_job_controller"]["training_method"]["content"]
        expected = {
            (rollout, trajectory, timestep)
            for rollout in range(params["num_rollouts"])
            for trajectory in range(params["num_trajs"])
            for timestep in range(params["num_steps_per_traj"])
        }
        if set(states) != expected:
            raise ValueError("Incomplete or unexpected rollout state coordinates")
        for (model_id, camera), entries in cameras.items():
            required = {
                position for position, model in states.items() if model == model_id
            }
            if entries != required:
                raise ValueError(f"Incomplete camera coverage: {camera}")
        if not metric_count or not {"policy", "value"} <= checkpoints:
            raise ValueError("Training metrics or scheduled checkpoints are missing")
        saved = api_request(
            "get_workload_checkpoints", query={"workload_ids": [workload_id]}
        )["checkpoints"]
        if not {"policy", "value"} <= {item["model_type"] for item in saved}:
            raise ValueError("Persisted training checkpoints are missing")
        verify_history(workload_id, seen)
        workloads = api_request("list_workloads", query={"project_id": project})[
            "workloads"
        ]
        workload = next(
            item for item in workloads if item["workload_id"] == workload_id
        )
        if workload["status"] != "completed":
            raise ValueError("Final workload status is not completed")
        print(f"PASS {label}: {len(states)} states, {messages} updates")
    finally:
        if not stopped:
            try:
                api_request(
                    "stop_workload",
                    method="POST",
                    body={"workload_id": workload_id},
                    accepted=(202, 409),
                )
            except Exception as error:
                print(f"Stop cleanup: {error}")
