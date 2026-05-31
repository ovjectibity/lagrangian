"""
Local (AllOnServer) training test with a Rust reward function.

Usage:
    MODEL_PATH=app/assets/trossen_vx300s/vx300s.xml python tests/test_training_trossen_arm.py
    BASE_URL=http://localhost:8000/api MODEL_PATH=... python tests/test_training_trossen_arm.py
"""
import io
import os
import zipfile

import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/api")
MODEL_PATH = os.environ.get("MODEL_PATH", "")

RUST_REWARD = '''
pub fn reward(
    prev_cstate: &[f64],
    curr_cstate: &[f64],
    action: &[f64],
    contact_pairs: &[(usize, usize)],
) -> f64 {
    let target_cube: usize = 3;
    let gripper: usize = 10;
    let left_finger: usize = 12;
    let right_finger: usize = 13;
    let left_box: usize = 1;
    let right_box: usize = 2;

    let get_pos = |s: &[f64], i: usize| -> (f64, f64, f64) {
        (s[i * 3], s[i * 3 + 1], s[i * 3 + 2])
    };

    let target_xy = (0.35_f64, -0.22_f64);
    let left_box_start = (0.35_f64, 0.22_f64);
    let right_box_start = (0.35_f64, -0.22_f64);

    let (nx, ny, nz) = get_pos(curr_cstate, target_cube);
    let (px, py, _) = get_pos(prev_cstate, target_cube);
    let (gx, gy, _) = get_pos(curr_cstate, gripper);

    let next_dist = ((nx - target_xy.0).powi(2) + (ny - target_xy.1).powi(2)).sqrt();
    let prev_dist = ((px - target_xy.0).powi(2) + (py - target_xy.1).powi(2)).sqrt();
    let r_prog = (prev_dist - next_dist) * 10.0;

    let gripper_xy_dist = ((gx - nx).powi(2) + (gy - ny).powi(2)).sqrt();
    let r_gripper = -2.0 * gripper_xy_dist;

    let (lx, ly, _) = get_pos(curr_cstate, left_box);
    let (rx, ry, _) = get_pos(curr_cstate, right_box);
    let left_disp = ((lx - left_box_start.0).powi(2) + (ly - left_box_start.1).powi(2)).sqrt();
    let right_disp = ((rx - right_box_start.0).powi(2) + (ry - right_box_start.1).powi(2)).sqrt();
    let r_box = -20.0 * (left_disp + right_disp);

    let cube_drop = if nz < 0.10 { -20.0 } else { 0.0 };

    let grip_contact: f64 = contact_pairs.iter().filter(|(a, b)| {
        (*a == target_cube && (*b == left_finger || *b == right_finger))
            || (*b == target_cube && (*a == left_finger || *a == right_finger))
    }).count() as f64 * 20.0;

    r_prog + r_gripper + r_box + cube_drop + grip_contact
}
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
        json={"name": "reward-fn-test"},
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
                        "language": "Rust",
                        "reward_function_code": RUST_REWARD,
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
                },
            },
            "acceleration_mode": "AllOnServer",
            "compute_instance": "AWSG4dn",
            "checkpoint_frequency": {"type": "LastOnly"},
        }
    }


def stream_sse(exp_id: str, run_id: str, max_events: int = 3) -> None:
    url = f"{BASE_URL}/training/experiments/{exp_id}/runs/{run_id}/updates"
    with requests.get(url, stream=True, timeout=120) as resp:
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

    print("\n=== Local training with Rust reward ===")
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
