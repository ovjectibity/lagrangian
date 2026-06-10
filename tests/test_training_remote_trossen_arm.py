"""
Remote (MLGPUSimulationGPU) training test with a Python reward function.
Requires a live GPU worker running and reachable by the server.

Usage:
    MODEL_PATH=app/assets/trossen_vx300s/vx300s.xml python tests/test_training_remote.py
    BASE_URL=http://localhost:8000/api MODEL_PATH=... python tests/test_training_remote.py
"""
import io
import os
import zipfile

import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/api")
MODEL_PATH = os.environ.get("MODEL_PATH", "")

PYTHON_REWARD = '''
import jax.numpy as jnp

def reward(prev_cstate, curr_cstate, action, contact_matrix):
    target_cube = 3
    gripper = 10
    left_finger = 12
    right_finger = 13
    left_box = 1
    right_box = 2
    target_xy = (0.35, -0.22)
    left_box_start = (0.35, 0.22)
    right_box_start = (0.35, -0.22)

    def get_pos(s, i):
        return s[i * 3], s[i * 3 + 1], s[i * 3 + 2]

    nx, ny, nz = get_pos(curr_cstate, target_cube)
    px, py, _ = get_pos(prev_cstate, target_cube)
    gx, gy, _ = get_pos(curr_cstate, gripper)

    next_dist = jnp.sqrt((nx - target_xy[0]) ** 2 + (ny - target_xy[1]) ** 2)
    prev_dist = jnp.sqrt((px - target_xy[0]) ** 2 + (py - target_xy[1]) ** 2)
    r_prog = (prev_dist - next_dist) * 10.0

    gripper_xy_dist = jnp.sqrt((gx - nx) ** 2 + (gy - ny) ** 2)
    r_gripper = -2.0 * gripper_xy_dist

    lx, ly, _ = get_pos(curr_cstate, left_box)
    rx, ry, _ = get_pos(curr_cstate, right_box)
    left_disp = jnp.sqrt((lx - left_box_start[0]) ** 2 + (ly - left_box_start[1]) ** 2)
    right_disp = jnp.sqrt((rx - right_box_start[0]) ** 2 + (ry - right_box_start[1]) ** 2)
    r_box = -20.0 * (left_disp + right_disp)

    cube_drop = jnp.where(nz < 0.10, -20.0, 0.0)

    grip_contact = (
        contact_matrix[target_cube, left_finger]
        + contact_matrix[target_cube, right_finger]
    ) * 20.0

    return r_prog + r_gripper + r_box + cube_drop + grip_contact
'''


def upload_model() -> str:
    model_dir = os.path.dirname(os.path.abspath(MODEL_PATH))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(model_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, model_dir)
                zf.write(fpath, arcname=arcname)
    buf.seek(0)
    r = requests.post(
        f"{BASE_URL}/models/save_model",
        files={"file": ("vx300s.zip", buf, "application/zip")},
    )
    assert r.status_code == 200, f"upload model: {r.status_code} {r.text}"
    model_id = r.json()["id"]
    print(f"  model_id={model_id}")
    return model_id


def create_experiment() -> str:
    r = requests.post(
        f"{BASE_URL}/training/experiments",
        json={"name": "reward-fn-remote-test"},
    )
    assert r.status_code == 200, f"create experiment: {r.status_code} {r.text}"
    exp_id = r.json()["experiment"]["training_experiment_id"]
    print(f"  exp_id={exp_id}")
    return exp_id


def create_run(exp_id: str) -> str:
    r = requests.post(
        f"{BASE_URL}/training/experiments/{exp_id}/runs",
        json={},
    )
    assert r.status_code == 200, f"create run: {r.status_code} {r.text}"
    run_id = r.json()["run"]["id"]
    print(f"  run_id={run_id}")
    return run_id


def build_training_graph(model_id: str) -> dict:
    return {
        "training_job_controller": {
            "training_method": {
                "type": "PPO",
                "content": {
                    "num_trajs": 2,
                    "num_steps_per_traj": 10,
                    "num_epochs": 1,
                    "num_updates": 1,
                    "epsilon": 0.2,
                    "gamma": 0.99,
                    "lambda": 0.95,
                    "reward_function": {
                        "language": "Python",
                        "reward_function_code": PYTHON_REWARD,
                    },
                    "control_model": {
                        "type": "FFN",
                        "content": {
                            "num_hidden_layers": 2,
                            "hidden_layer_dim": 64,
                            "output_dim": 14,
                            "input_dim": 55,
                        },
                    },
                    "value_model": {
                        "type": "FFN",
                        "content": {
                            "num_hidden_layers": 2,
                            "hidden_layer_dim": 64,
                            "output_dim": 1,
                            "input_dim": 55,
                        },
                    },
                    "simulation_model": model_id,
                    "control_output_mode": {"type": "ActionMean"},
                },
            },
            "acceleration_mode": "MLGPUSimulationGPU",
            "compute_instance": "AWSG4dn",
            "checkpoint_frequency": {"type": "LastOnly"},
        }
    }


def stream_sse(exp_id: str, run_id: str, max_events: int = 3) -> None:
    url = f"{BASE_URL}/training/experiments/{exp_id}/runs/{run_id}/updates"
    with requests.get(url, stream=True, timeout=300) as resp:
        count = 0
        for line in resp.iter_lines():
            if line and line.startswith(b"data:"):
                data = line[5:].strip()
                print(f"    SSE[{count}]: {data[:120].decode()}")
                count += 1
                if count >= max_events:
                    break


def main() -> None:
    print("=== Uploading model ===")
    model_id = upload_model()

    print("=== Creating experiment ===")
    exp_id = create_experiment()

    print("\n=== Remote training with Python reward ===")
    run_id = create_run(exp_id)
    graph = build_training_graph(model_id)
    r = requests.post(
        f"{BASE_URL}/training/experiments/{exp_id}/runs/{run_id}/start",
        json={"training_graph": graph},
    )
    print(f"  start_training status: {r.status_code}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    print("  Streaming SSE events:")
    stream_sse(exp_id, run_id, max_events=3)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
