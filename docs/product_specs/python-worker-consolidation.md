# Spec: Consolidate to the Python worker for local + remote (axe Rust local stack)

## Context

The server currently maintains **two execution stacks**: a Rust/`tch` local stack (`app/src/workflows/` + `app/src/models/`) and a Python/JAX worker (`worker/worker-py`) used for remote jobs. This is duplicate work and the Rust local path can't run VLA (Pi0) models at all.

Goal: **use the Python worker for both local and remote** workloads. Remove the Rust local execution entirely. Both local and remote drive the same Python worker over websocket; the only differences are the worker's host and lifecycle. Refactor the controllers so shared worker-interaction logic lives in one place (`CommonWorkerController`), with thin local/remote wrappers on top.

Decisions locked with the user:
- **Local lifecycle = connect-only**: the local controller assumes a Python worker is already running on `localhost:PORT` and just connects. It does not spawn/manage the process.
- **`CommonWorkerController`, composed**: a shared struct holding the ws + shared methods; Local and Remote controllers each hold one as a field.
- **Delete the now-dead Rust `models/`** module too.

## Findings (from exploration)

- Rust `workflows/` (`WorkflowRuntime`, `TestingWorkflow`, `TrainingWorkflow`, Rust `ppo_trainer.rs`) is consumed **only** by the local branches of `api/testing.rs` and `api/training.rs`.
- Rust `models/` (`ControlPolicyModel`/`PPOTrainedModel` traits, tch `MLPModel`) is consumed **only** by `workflows/` → fully dead after removal. (`commondef::training::MLPModel` is a separate typeshare config struct — keep it.)
- `MjSimulator` stays — `api/models.rs` uses it for the world-body preview, independent of `workflows/`.
- The Python worker (`worker-py/main.py`) is already a websocket server (Bearer auth, `WS_PORT` default 8765) — same protocol for local and remote.
- `RemoteWorkerController` splits cleanly: **common** = `connect_worker_wss`, `send_to_worker`, `monitor_training_until_stop`, `monitor_testing_until_stop`, `run_training_job`, `run_test_job`, `stop_test`, plus `load_mjcf_package_zip_base64`/`find_xml_in_zip_root`; **remote-only** = EC2 lifecycle (`launch_spot_instance`, `wait_until_instance_ready`, `resolve_existing_instance`, `terminate_instance`, `build_user_data_script`, `Drop`).
- The `run_*_job` bodies already package the model into a zip and send it in the Start message — works unchanged for a local worker, so no divergence there.

## Changes

### 1. Delete the Rust local execution stack
- Delete `app/src/workflows/{mod.rs,testing.rs,training.rs,ppo_trainer.rs}` (the current local module).
- Delete `app/src/models/{mod.rs,mlp.rs}`.
- `app/src/main.rs`: remove `pub mod models;`.

### 2. Rename `remote_workflows/` → `workflows/`
- Move `app/src/remote_workflows/remote_worker_controller.rs` into the new `app/src/workflows/`.
- `main.rs`: replace `pub mod remote_workflows;` + old `pub mod workflows;` with a single `pub mod workflows;`.
- New `app/src/workflows/mod.rs`:
  ```rust
  pub mod common_worker_controller;
  pub mod local_worker_controller;
  pub mod remote_worker_controller;
  ```
- Update import paths `remote_workflows::remote_worker_controller::…` → `workflows::remote_worker_controller::…` in `main.rs` (~30, 90, 96), `api/testing.rs` (~332–339), `api/training.rs` (~707–729).

### 3. `workflows/common_worker_controller.rs` — `CommonWorkerController`
- `CommonWorkerControllerConfig { host, port, ws_path, shared_secret, websocket_connect_timeout }`.
- Struct holds the config + `worker_ws: Option<WorkerWs>`.
- Move the **common** methods here (from `RemoteWorkerController`), with `connect_worker_wss` building the URL from `config.host` instead of the EC2 public DNS:
  `connect_worker_wss`, `send_to_worker`, `monitor_training_until_stop`, `monitor_testing_until_stop`, `run_training_job`, `run_test_job`, `stop_test`, `stop_training` (send-StopTraining only).
- Move free fns `load_mjcf_package_zip_base64`, `find_xml_in_zip_root` here.

### 4. `workflows/remote_worker_controller.rs` — `RemoteWorkerController` (composes common)
- Fields: `common: CommonWorkerController` + EC2 bits (`ec2`, AWS config, `instance_id`, `instance_public_dns`, `skip_instance_lifecycle`).
- Keep EC2-only methods + `Drop`.
- `RemoteWorkerControllerConfig` keeps AWS fields; builds a `CommonWorkerControllerConfig` (host = public DNS) for the inner controller.
- `initialize()`: launch/resolve instance → public DNS → construct `common` (host = public DNS) → `common.connect_worker_wss()`.
- `run_test_job`/`run_training_job`/`stop_test` delegate to `self.common.*`; `stop_training_and_terminate` = `common.stop_training` + `self.terminate_instance`.

### 5. `workflows/local_worker_controller.rs` — `LocalWorkerController` (connect-only)
- Field: `common: CommonWorkerController`.
- Config from env: host `127.0.0.1` (override via env), port = worker port, `shared_secret` from the existing worker-secret env. Reuse the secret/port source already used by `build_remote_worker_config()`.
- `initialize()`: construct `common` pointed at localhost. (Connection is lazy — `run_*_job` reconnects anyway, mirroring remote.)
- `run_test_job`/`run_training_job`/`stop_test` delegate to `self.common.*`. No instance teardown.
- **Prerequisite:** a local Python worker must already be running on the configured port (connect-only).

### 6. Rewire `api/testing.rs` and `api/training.rs` local branches
- Remove `WorkflowRuntime`/`TestingWorkflow`/`TrainingWorkflow` imports and the `spawn_blocking` Rust-execution blocks.
- `is_local` branch: spawn an async task (mirror `spawn_remote_test_task`/`spawn_remote_training_task`) that builds a `LocalWorkerController`, calls `run_test_job`/`run_training_job` with an `on_update` closure feeding the **existing** `tx` (mpsc → broadcast → SSE relay). Both branches converge on `controller.run_*_job(on_update)`; keep `spawn_local_*` / `spawn_remote_*` helpers that differ only in controller construction.
- **Reward gating change:** local now runs in the Python worker, so local also requires **Python** reward functions. The reward spec rides along in the Start message (training_graph), as it already does for remote. See §8.

### 7. `main.rs` helpers
- Keep `remote_worker_slot` / `build_remote_worker_config` / `ensure_remote_worker_shutdown_hook` for remote; update module paths. No local slot (connect-only, per-job controller).

### 8. Remove the Rust reward C-ABI path + local-only rejections (confirmed)
- **`api/reward_validation.rs`**: delete `call_reward_c_abi` and `validate_and_compile_rust_reward`. Keep `validate_python_reward`. Remove now-unused imports (tch `Tensor`, libloading, etc.).
- **`api/training.rs`**: remove the local-only reward gate (`is_local` ⇒ `RewardLanguage::Rust`, `!is_local` ⇒ Python) and replace with a single **Python-required** check for all jobs. Delete the `user_reward_lib` Rust-compile arm and the `reward_fn` C-ABI boxing in the (now-removed) local execution block.
- **Local-only rejections in the api module**: the only api-level local-only rejection is that reward gate (above); remove it. The VLA-local and non-MLP-local rejections lived in the deleted `WorkflowRuntime::for_{testing,training}`, so they vanish with §1. **Keep** the general validations that are *not* local-only: `validate_model_types` (VLA must be `Pretrained(Pi0)`, no Pretrained value model) and the unconditional `contact_pairs` rejections.

## Verification
- `cargo build` (app) compiles with no references to the removed `workflows`/`models`; no dead-code errors.
- Start a local Python worker: `WS_PORT=8765 WORKER_SHARED_SECRET=… python worker/worker-py/main.py`.
- POST a local test job (`AllOnServer`) → `LocalWorkerController` connects → worker streams `TestUpdateMessage` frames → existing SSE relay delivers them to the client (unchanged client contract).
- POST a local PPO training job with a **Python** reward → trains via the worker; updates stream back.
- POST a remote test + training job → unchanged behavior via `RemoteWorkerController`.
- Confirm `api/models.rs` world-body preview still works (MjSimulator retained).

## Open considerations (note, not blocking)
- Local jobs now **require** a running local Python worker (connect-only) — document the dev/run setup (script or compose service on `localhost:WS_PORT`).
- Side benefit: local VLA (Pi0) testing/training now works (the Rust local path rejected VLA, and its reward gate forced Rust rewards) — both restrictions are removed per §8.
