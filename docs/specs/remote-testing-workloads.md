# Workflow #2 Phase-2 — Remote Testing Workload

## Overview

Phase-2 adds a remote GPU testing workload (alongside the existing local one) and wires up
full DB persistence for test runs. A user submits a `TestGraph` (mirrors `TrainingGraph`)
specifying the control model, simulator, parallelism, acceleration mode, and compute
instance. The app server routes to local `TestingWorkflow` or `RemoteWorkerController`
accordingly. Both paths persist a `TestRun` Postgres record and a `TestParamsRecord` MongoDB
document.

---

## Pre-conditions Already Done (do NOT redo)

- `TestParams` already has `control_model: ControlModel`, `simulation_model: String`,
  `num_parallel_envs: u32`, `num_timesteps: i32`
- `TestRun` and `TestParamsRecord` structs exist in `app/src/commondef/record.rs`
- `TestRunStatus` enum exists in `record.rs`
- `RemoteWorkerServerMessage` / `RemoteWorkerClientMessage` already include:
  - `StartTestingMessage(StartTestMessage)`, `StopTestingMessage(StopTestMessage)`
  - `TestUpdatesMessage(TestUpdateMessage)`, `StopTestingMessage(StopTestMessage)`
- Struct consolidation (`StopTrainingMessage` etc. → `training.rs`) and `LoadModelMessage`
  removal are done
- Python worker has already been renamed/moved to `worker/worker-py/` (NOT `trainer-py`)
- `TestTaskMessage` in `workermessages.rs` has been cleaned up (no `StartTestMessage` variant)

---

## Part A — New Structs: `TestGraph` and `TestJobController`

**File:** `app/src/commondef/testing.rs`

Add after the existing `TestParams` struct:

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct TestGraph {
    pub test_job_controller: TestJobController,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct TestJobController {
    pub test_params: TestParams,
    pub compute_instance: ComputeInstance,            // from training.rs
    pub acceleration_mode: TrainingAccelerationMode,  // from training.rs
}
```

Add imports at the top of `testing.rs`:
```rust
use crate::commondef::training::{ComputeInstance, TrainingAccelerationMode};
```

`TestParams` already has `#[derive(JsonSchema)]` — verify this stays.

**Update `StartTestMessage`** (already in `testing.rs`):
```rust
// Before:
pub struct StartTestMessage {
    pub simulation_params: TestParams,
    pub id: String,
}
// After:
pub struct StartTestMessage {
    pub id: String,
    pub test_graph: TestGraph,
}
```

---

## Part B — Update `TestParamsRecord` in `record.rs`

**File:** `app/src/commondef/record.rs`

1. Change the `params` field to embed the full `TestGraph`:
   ```rust
   // Before: pub params: TestParams,
   pub graph: TestGraph,
   ```

2. Add `#[derive(JsonSchema)]` to the following structs (currently missing):
   - `LogStream`
   - `MetricsStream`
   - `TestParamsRecord`
   - (`TrainingGraphRecord` already has it)

3. Add import: `use crate::commondef::testing::TestGraph;`

---

## Part C — MongoDB Schema Generation for All Collections

**File:** `scripts/gen_mongo_schema/src/main.rs`

Currently generates schema only for `TrainingGraphRecord`. Extend to emit all 4 schemas.
Change `main()` to output to files (not stdout) since we now have multiple schemas:

```rust
fn main() {
    generate::<TrainingGraphRecord>("schemas/training_graph.json");
    generate::<LogStream>("schemas/log_stream.json");
    generate::<MetricsStream>("schemas/metrics_stream.json");
    generate::<TestParamsRecord>("schemas/test_params.json");
}

fn generate<T: JsonSchema>(path: &str) {
    let schema = schemars::schema_for!(T);
    let mut value: Value = serde_json::to_value(&schema).expect("serialize");
    rewrite_defs(&mut value);
    std::fs::write(path, serde_json::to_string_pretty(&value).expect("pretty")).expect("write");
}
```

Add imports for `LogStream`, `MetricsStream`, `TestParamsRecord` from `lagrangian::commondef::record`.

**File:** `scripts/gen_types.sh`

Replace the single `cargo run` schema line with:
```bash
mkdir -p schemas
cargo run --manifest-path scripts/gen_mongo_schema/Cargo.toml -q
echo "Generated schemas: training_graph.json, log_stream.json, metrics_stream.json, test_params.json"
```
(No longer needs `> schemas/training_graph.json` redirect since main.rs writes files directly.)

---

## Part D — Rename Controller File + Structs

**Rename:** `app/src/remote_workflows/training_controller.rs`
→ `app/src/remote_workflows/remote_worker_controller.rs`

**Rename structs within the file:**
- `TrainingWorkerControllerConfig` → `RemoteWorkerControllerConfig`
- `TrainingWorkerController` → `RemoteWorkerController`

**Update `app/src/remote_workflows/mod.rs`:**
```rust
// Before: pub mod training_controller;
pub mod remote_worker_controller;
```

**Update `app/src/api/training.rs`** imports accordingly.

**Add to `RemoteWorkerController`:**

```rust
pub async fn run_test_job<F>(
    &mut self,
    test_id: String,
    test_graph: TestGraph,
    mut on_update: F,
) -> Result<()>
where
    F: FnMut(&TestUpdateMessage) + Send,
{
    self.send_to_worker(&RemoteWorkerServerMessage::StartTestingMessage(
        StartTestMessage { id: test_id.clone(), test_graph }
    )).await?;
    self.monitor_testing_until_stop(&test_id, &mut on_update).await?;
    Ok(())
}

async fn monitor_testing_until_stop<F>(
    &mut self,
    test_id: &str,
    on_update: &mut F,
) -> Result<()>
where
    F: FnMut(&TestUpdateMessage) + Send,
{
    // same ws loop pattern as monitor_training_until_stop
    // dispatch TestUpdatesMessage → on_update(&m)
    // dispatch StopTestingMessage if s.id == test_id → return Ok(())
    // _ => {}
}

pub async fn stop_test(&mut self, test_id: &str) -> Result<()> {
    self.send_to_worker(&RemoteWorkerServerMessage::StopTestingMessage(
        StopTestMessage { id: test_id.to_string() }
    )).await
    // Does NOT call terminate_instance — instance is reused
}
```

Add imports: `TestGraph`, `TestUpdateMessage`, `StartTestMessage`, `StopTestMessage` from
`crate::commondef::testing` (they're not in `remoteworkermessages`).

---

## Part E — `TestingWorkflow` Change

**File:** `app/src/workflows/testing.rs`

Change `TestingWorkflow` to hold `TestGraph` instead of `TestParams`:

```rust
pub struct TestingWorkflow {
    id: String,
    simulator: Arc<Mutex<dyn Simulator + Send>>,
    model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
    graph: TestGraph,   // was: params: TestParams
}

impl TestingWorkflow {
    pub fn new(
        id: String,
        simulator: Arc<Mutex<dyn Simulator + Send>>,
        model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
        graph: TestGraph,   // was: params: TestParams
    ) -> Self { ... }
```

Inside `run_test`, replace `self.params.num_timesteps` →
`self.graph.test_job_controller.test_params.num_timesteps`. Similarly for `num_parallel_envs`.

Add import: `use crate::commondef::testing::TestGraph;`

---

## Part F — API Route Renames + Routing Logic

**File:** `app/src/api/testing.rs`

### Route renames
```
POST /testing/simulations           → POST /testing/start
POST /testing/simulations/<id>/stop → POST /testing/<test_id>/stop
GET  /testing/simulations/<id>/updates → GET /testing/<test_id>/updates
```
Rename parameter name `simul_id` → `test_id` throughout.

### Request body
```rust
pub struct StartTestBody { pub test_graph: TestGraph }
```

### DB writes — BOTH local and remote paths

Before dispatching to local/remote, always do:
```rust
let test_params_id = uuid::Uuid::new_v4().to_string();
let logs_id = uuid::Uuid::new_v4().to_string();
let metrics_id = uuid::Uuid::new_v4().to_string();

pg.create_test_run(&project_id, &test_id, &TestRunStatus::NotStarted,
    &logs_id, &metrics_id, &test_params_id).await?;
mongo.create_test_params(TestParamsRecord {
    id: test_params_id,
    test_run_id: test_id.clone(),
    created_at: Utc::now(),
    graph: test_graph.clone(),
}).await?;
pg.update_test_run_status(&test_id, &TestRunStatus::Ongoing).await?;
```

On completion/stop:
```rust
pg.update_test_run_status(&test_id, &TestRunStatus::Completed).await?;
// or StoppedByUser for explicit stop
```

### Routing logic
```rust
match test_graph.test_job_controller.compute_instance {
    ComputeInstance::LocalCPU => {
        // existing TestingWorkflow path, pass test_graph directly
        let mut wf = TestingWorkflow::new(test_id, sim, model, test_graph);
        wf.run_test(tx_test);
    }
    ComputeInstance::RemoteEC2(config) => {
        // init RemoteWorkerController, call run_test_job
        let mut worker = RemoteWorkerController::initialize(
            RemoteWorkerControllerConfig { ... from config ... }
        ).await?;
        worker.run_test_job(test_id, test_graph, |update| {
            // relay TestUpdateMessage frames to SSE tx
        }).await?;
    }
}
```

**`app/src/main.rs`** — update route symbol names to match renamed Rocket handlers.

---

## Part G — Database Layer

### `scripts/init_db.sh`

Add after `checkpoint_references` table:
```sql
CREATE TABLE IF NOT EXISTS test_runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_start_at TIMESTAMPTZ,
    run_end_at TIMESTAMPTZ,
    run_status TEXT NOT NULL DEFAULT 'not_started',
    logs_id TEXT NOT NULL,
    metrics_id TEXT NOT NULL,
    test_params_id TEXT NOT NULL
);
```

`TestRunStatus` values on the Rust side: `Ongoing`, `Completed`, `StoppedByUser`,
`NotStarted` — serialised as `snake_case` text (see `TrainingRunStatus` pattern in
`record.rs` for the `#[sqlx(type_name = "text", rename_all = "snake_case")]` attribute).

### `app/src/db/mod.rs` — PostgresOrm additions

Follow the exact SQL pattern of `create_run` / `update_run_status`. Imports needed:
`use crate::commondef::record::{TestRun, TestRunStatus};`

```rust
pub async fn create_test_run(
    &self, project_id: &str, id: &str, run_status: &TestRunStatus,
    logs_id: &str, metrics_id: &str, test_params_id: &str
) -> Result<TestRun, sqlx::Error>

pub async fn get_test_run(&self, id: &str) -> Result<Option<TestRun>, sqlx::Error>

pub async fn get_test_runs_by_project(&self, project_id: &str)
    -> Result<Vec<TestRun>, sqlx::Error>

pub async fn update_test_run_status(&self, id: &str, status: &TestRunStatus)
    -> Result<TestRun, sqlx::Error>
```

### `app/src/db/mod.rs` — MongoOrm additions

Collection name: `"test_params"`. Follow `create_training_graph` pattern exactly.
Imports needed: `use crate::commondef::record::TestParamsRecord;`

```rust
pub async fn create_test_params(&self, record: &TestParamsRecord)
    -> Result<(), mongodb::error::Error>

pub async fn get_test_params(&self, id: &str)
    -> Result<Option<TestParamsRecord>, mongodb::error::Error>
```

---

## Part H — Python Worker Testing Support

Testing logic is split across three files following the same pattern as training:

### `ppo_trainer.py` — `TestingModel` abstract class

Added after `PPOTrainedModel`:
```python
class TestingModel(ABC):
    @abstractmethod
    def get_test_action(self, state: Array) -> Array: ...

    @abstractmethod
    def get_pure_test_fn(self) -> Callable[[Params, Array], Array]: ...

    @abstractmethod
    def get_test_params(self) -> Params: ...
```

`MLPPPOModel` (in `ppo_model.py`) and `Pi0Model` (in `pretrained/pi0.py`) both inherit
`TestingModel` in addition to `PPOTrainedModel`. The three methods delegate to already-existing
methods (`get_policy_output`, `get_pure_policy_fn`, `get_policy_store`) so no new logic.

### `testing.py` — `TestingWorkflow` (mirrors `training.py`)

New file `worker/worker-py/testing.py` contains:

- `TestingTaskSink(ABC)` — `send_test_updates(TestUpdateMessage)` + `send_stop_testing(StopTestMessage)`
- `CallbackTestingTaskSink(TestingTaskSink)` — wraps payloads in `RemoteWorkerClientMessageTestUpdatesMessage` / `RemoteWorkerClientMessageStopTestingMessage` and calls the provided `send_message` callback
- `TestingWorkflow` — `__init__(test_id, test_controller, simulator, model: TestingModel)` + `run_test_blocking(tx, should_stop)`

`run_test_blocking` delegates to one of two private methods:

**MJX batched path** (`_run_mjx_batched`) — `num_parallel_envs > 1` and `MjxSimulator`:
Uses **chunked `jax.lax.scan`** (`UPDATE_CHUNK_SIZE = 10` steps per chunk). Each chunk runs
fully on GPU with no Python interruption; one `TestUpdateMessage` per chunk is sent, giving
regular streaming updates. XLA re-uses the compiled kernel after the first chunk.

```python
def scan_body(bd, _):
    states  = jax.vmap(lambda d: jnp.concatenate([d.qpos[:nq], d.qvel[:nv]]))(bd)
    actions = jax.vmap(lambda s: pure_fn(params, s))(states)
    bd_next = jax.vmap(lambda d, a: mjx_sim.forward_dyn_pure(d, a)[0])(bd, actions)
    return bd_next, (bd_next.xpos[:, :nbody], bd_next.xquat[:, :nbody])

while remaining > 0 and not should_stop():
    batched_data, (chunk_xpos, chunk_xquat) = jax.lax.scan(
        scan_body, batched_data, None, length=chunk
    )
    tx.send_test_updates(TestUpdateMessage(id=self.id, updates=[...]))
```

**Sequential path** (`_run_sequential`) — `MujocoSimulator` or single-env MJX:
```python
for _ in range(num_timesteps):
    action = model.get_test_action(simulator.get_model_state())
    simulator.forward_dyn(action)
    xpos, xquat = simulator.get_simul_update()
    tx.send_test_updates(TestUpdateMessage(id=self.id, updates=[[ModelForward(...)]]))
```

**Model construction** (`_build_model_for_testing` in `main.py`):
- `ControlModelMLP` → `MLPPPOModel` (1-output dummy value head, `reward_fn=None`)
- `ControlModelVLA` → `Pi0Model(vla_spec, noise_mlp_spec=None)` — inference-only, no FlowNoise

### `main.py` — START/STOP TESTING handlers

`main.py` handles message dispatch only. START_TESTING_MESSAGE handler:
1. Validates `StartTestMessage`, extracts `test_controller`
2. Extracts sim model zip from `simulation_models`, calls `_extract_sim_model_from_zip`
3. Builds simulator (`MjxSimulator` or `MujocoSimulator` based on `acceleration_mode`)
4. Calls `_build_model_for_testing` → `TestingModel`
5. Constructs `TestingWorkflow`, wraps `run_test_blocking` in `asyncio.to_thread`
6. Stores `TestingTaskState(task, stop_event)` in `testing_tasks` dict

STOP_TESTING_MESSAGE handler sets `stop_event` for the running task.

On client disconnect, stop signals are sent to all active testing tasks.

### Post-regeneration type names

After `gen_types.sh` is run, `commondef/remoteworkermessages.py` will contain:
- `RemoteWorkerServerMessageTypes` enum (replaces `TrainingRemoteWorkerServerMessageTypes`)
  with variants `START_TRAINING_MESSAGE`, `STOP_TRAINING_MESSAGE`, `START_TESTING_MESSAGE`, `STOP_TESTING_MESSAGE`
- `RemoteWorkerClientMessageTestUpdatesMessage` wrapper for `TestUpdateMessage`
- `RemoteWorkerClientMessageStopTestingMessage` wrapper for `StopTestMessage`
- `TestUpdateMessage`, `StopTestMessage`, `ModelForward`, `TestGraph`, `TestJobController`, `TestParams`

Note: `training.py` also imports stale names (`TrainingRemoteWorkerClientMessage*`,
`TrainingUpdatesMessage`, `TrainingUpdate`) that will break after regeneration and need updating.

---

## Part I — `scripts/gen_types.sh` Updates

1. **Python testing types** — add typeshare run for `testing.rs`:
   ```bash
   typeshare app/src/commondef/testing.rs \
     --lang=python \
     --output-file=worker/worker-py/commondef/testing.py
   ```
   This generates Python classes for `TestParams`, `TestGraph`, `TestJobController`,
   `StartTestMessage`, `StopTestMessage`, `TestUpdateMessage`, `ModelForward`.

2. **Verify path** — all Python output paths should reference `worker/worker-py/` (not the
   old `trainer-py` path). Audit the entire script.

3. **MongoDB schema** — after the schema binary changes in Part C, the script line becomes:
   ```bash
   cargo run --manifest-path scripts/gen_mongo_schema/Cargo.toml -q
   ```
   (Writes 4 files directly; no stdout redirect needed.)

---

## Part J — TypeScript SDK (`sdk/src/resources/simulations.ts`)

Update endpoint strings only (types auto-regenerated by `gen_types.sh`):
- `POST /testing/simulations` → `POST /testing/start`
- `POST /testing/simulations/{id}/stop` → `POST /testing/{id}/stop`
- `GET /testing/simulations/{id}/updates` → `GET /testing/{id}/updates`

Update `start()` method signature to accept `TestGraph` instead of `SimulationParams`.

---

## Part K — Test Script (`tests/test_testing_remote.py`)

Uses the SDK (follow pattern of `tests/test_training_remote_trossen_arm.py`):

```python
# Construct TestGraph with RemoteEC2 compute instance and MLP control model
# Call client.simulations.start(test_graph)  → returns { id }
# Stream updates via client.simulations.stream_updates(id)
# Assert at least N frames received with non-empty xpos/xquat
# Call client.simulations.stop(id)
```

Use an MLP control model for the initial test (not VLA — VLA not yet supported for remote
test). The simulation model should reference an already-saved model from `/models`.

---

## Implementation Order

1. Part A — `testing.rs` new structs + `StartTestMessage` update
2. Part B — `record.rs` `TestParamsRecord.graph` + `JsonSchema` derives
3. Part D — rename `training_controller.rs` → `remote_worker_controller.rs`, rename structs
4. Part E — `TestingWorkflow` change to accept `TestGraph`
5. Part F — API route renames + routing logic + DB writes
6. Part G — DB layer (`init_db.sh`, `db/mod.rs`)
7. Part C — Schema gen binary + `gen_types.sh`
8. Compile: `cargo build --manifest-path app/Cargo.toml --lib`
9. Part H — Python worker testing handlers
10. Part I — `gen_types.sh` Python types update
11. Part J — SDK endpoint updates
12. Part K — Test script

---

## Key Constraints

- Local test: no VLA support, no checkpoint support (limitation noted in Notion spec)
- Remote test Pi-0: vanilla inference only — no FlowNoise noise model at test time
- `stop_test` does NOT terminate EC2 instance (instance reused across jobs)
- `TestParamsRecord` stores `TestGraph` (not bare `TestParams`) — full config for replay
- `TestRunStatus` SQL serialization: `snake_case` text, matching `TrainingRunStatus` pattern
- Python worker is at `worker/worker-py/` — `trainer-py` name is gone
