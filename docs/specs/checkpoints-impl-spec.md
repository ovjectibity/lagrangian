# Plan: Implement Checkpoints Feature

## Context
`instructions/specs/checkpoints.md` defines the full checkpoints layer. The DB schema, ORM basics (`create_checkpoint_ref`, `get_checkpoint_ref`), and type stubs (`CheckpointReference`, `Checkpoint`, `CheckpointFrequency`) are already in place. This plan wires them end-to-end.

**Scope decisions:**
- Base64 encoding retained (no binary WSS this iteration)
- Checkpoint viewer tab shows active run only
- MLP params stored as TEXT at save time, deserialized into `Option<MLPModel>` by the ORM (no sqlx json feature; no Cargo.toml changes beyond `base64` which is already present)
- `CheckpointsForRunMessage` is a separate explicit message (sent after `RunLoadedMessage`)
- Checkpoint loading is remote-job only (AllOnServer excluded)
- `LastOnly` checkpoint also fires when user stops training mid-run

---

## Step 1 — Backend: extend `CheckpointReference` with MLP metadata

**File:** `app/src/commondef/experiments.rs`

Add two strongly-typed optional fields to `CheckpointReference`. Because `sqlx::FromRow` can't auto-map TEXT → struct, we drop the `FromRow` derive from `CheckpointReference` and write a manual `impl sqlx::FromRow` (or use an intermediate raw struct). The public API field types are `Option<MLPModel>` from `crate::commondef::training`.

```rust
pub control_model_params: Option<MLPModel>,
pub value_model_params: Option<MLPModel>,
```

**Approach for manual FromRow:**
Introduce a private `CheckpointReferenceRow` struct with `control_model_params: Option<String>` / `value_model_params: Option<String>` that still derives `FromRow`, then convert via:
```rust
impl From<CheckpointReferenceRow> for CheckpointReference {
    fn from(r: CheckpointReferenceRow) -> Self {
        CheckpointReference {
            // ... other fields ...
            control_model_params: r.control_model_params
                .and_then(|s| serde_json::from_str(&s).ok()),
            value_model_params: r.value_model_params
                .and_then(|s| serde_json::from_str(&s).ok()),
        }
    }
}
```

Add `use crate::commondef::training::MLPModel;` import. Also add `#[typeshare]` support — the TS type should have `control_model_params: MLPModel | null`.

---

## Step 2 — Backend: extend DB schema + ORM

**File:** `scripts/init_db.sh`

Add columns to `checkpoint_references` CREATE TABLE:
```sql
control_model_params TEXT,
value_model_params   TEXT
```

**File:** `app/src/db/db_orm.rs`

All ORM methods that query `checkpoint_references` must use `CheckpointReferenceRow` and convert via `.into()`.

1. Update `create_checkpoint_ref` signature:
   - Add `control_model_params: Option<&str>` and `value_model_params: Option<&str>`.
   - Bind them (positions $6, $7) in the INSERT, return `CheckpointReference` via conversion.

2. Update `get_checkpoint_ref` to use `CheckpointReferenceRow` → `CheckpointReference`.

3. Add `get_checkpoint_refs_for_run`:
```rust
pub async fn get_checkpoint_refs_for_run(
    &self,
    training_experiment_id: &str,
    training_run_id: &str,
) -> Result<Vec<CheckpointReference>, sqlx::Error>
```
`SELECT * FROM checkpoint_references WHERE training_experiment_id = $1 AND training_run_id = $2 ORDER BY created_at ASC`

4. Add `get_all_checkpoint_refs`:
```rust
pub async fn get_all_checkpoint_refs(
    &self,
) -> Result<Vec<CheckpointReference>, sqlx::Error>
```
`SELECT * FROM checkpoint_references ORDER BY training_experiment_id, training_run_id, created_at ASC`

5. Add `add_checkpoint_ref_to_run` (mirrors `add_run_to_experiment`):
```rust
pub async fn add_checkpoint_ref_to_run(
    &self,
    training_experiment_id: &str,
    run_id: &str,
    checkpoint_ref_id: &str,
) -> Result<TrainingExperimentRun, sqlx::Error>
```
`UPDATE training_runs SET checkpoints_refs = array_append(checkpoints_refs, $3) WHERE training_experiment_id = $1 AND id = $2 RETURNING *`

---

## Step 3 — Backend: new WebSocket message types

**File:** `app/src/commondef/clientmessages.rs`

Add to `ServerMessage` enum:
```rust
GetAllCheckpointsMessage,
GetCheckpointInfoMessage(GetCheckpointInfoMessage),
```

Add to `ClientMessage` enum:
```rust
AllCheckpointsMessage(AllCheckpointsMessage),
CheckpointInfoMessage(CheckpointInfoMessage),
CheckpointsForRunMessage(CheckpointsForRunMessage),
```

New structs:
```rust
pub struct GetCheckpointInfoMessage {
    pub checkpoint_id: String,
}
pub struct AllCheckpointsMessage {
    pub checkpoints: Vec<CheckpointReference>,
}
pub struct CheckpointInfoMessage {
    pub checkpoint: CheckpointReference,
}
pub struct CheckpointsForRunMessage {
    pub training_run_id: String,
    pub checkpoints: Vec<CheckpointReference>,
}
```

**File:** `client/src/commondef/messages.ts` (hand-edited)

Add interfaces and extend `ServerMessage` / `ClientMessage` unions to match. `CheckpointReference` is already in `training.ts`; import it.

---

## Step 4 — Backend: save checkpoints in `handle_start_training_remote`

**File:** `app/src/main.rs`

Before `tokio::spawn`, lock `ACTIVE_CONTEXT` and clone `exp_id` and `run_id` for capture into the closure. Also capture the `training_graph` for MLP param extraction.

Inside the closure replacing `checkpoints: vec![]`:
For each `Checkpoint { model_zip_base64: Some(b64), .. }` in an update:
1. `base64::engine::general_purpose::STANDARD.decode(&b64)` → bytes (crate already present at v0.22.1).
2. Generate `ckpt_id = Uuid::new_v4().to_string()`.
3. `std::fs::create_dir_all(format!("checkpoints/{exp_id}/{run_id}"))`, then `std::fs::write(format!("checkpoints/{exp_id}/{run_id}/{ckpt_id}.zip"), &bytes)`.
4. Extract `control_model_params_json` / `value_model_params_json`:
   - If `ControlModel::MLP(m)` → `serde_json::to_string(&m).ok()`
   - If `ControlModel::Checkpoint(src_id)` → fetch `get_checkpoint_ref(src_id)` and re-serialize the returned `MLPModel`.
5. `postgres_orm().create_checkpoint_ref(&ckpt_id, &path, "ppo", &exp_id, &run_id, ctrl_json.as_deref(), val_json.as_deref()).await`
6. `postgres_orm().add_checkpoint_ref_to_run(&exp_id, &run_id, &ckpt_id).await`
7. Push returned `CheckpointReference` into the `clientmessages::TrainingUpdate.checkpoints` vec.

Note: the closure is `move`, so `postgres_orm()` (returns a `'static` ref) is fine to call inside.

---

## Step 5 — Backend: handle new WS messages

**File:** `app/src/main.rs`

### `GetAllCheckpointsMessage`
`postgres_orm().get_all_checkpoint_refs().await` → send `ClientMessage::AllCheckpointsMessage`.

### `GetCheckpointInfoMessage(m)`
`postgres_orm().get_checkpoint_ref(&m.checkpoint_id).await` → if found, send `ClientMessage::CheckpointInfoMessage`.

### Extend `LoadRunMessage` handler
After sending `RunLoadedMessage`, also:
1. `postgres_orm().get_checkpoint_refs_for_run(&exp_id, &run_id).await`
2. Send `ClientMessage::CheckpointsForRunMessage { training_run_id: run.id, checkpoints }`.

---

## Step 6 — Backend: pass checkpoint to remote worker

**File:** `app/src/main.rs` — `handle_start_training_remote`

When building `remoteworkermessages::StartTrainingMessage`, inspect `training_graph.training_job_controller.training_method`:
- If `TrainingMethod::PPO(ppo)` and `ppo.control_model == ControlModel::Checkpoint(id)`:
  1. `postgres_orm().get_checkpoint_ref(&id).await` to get the path.
  2. Read zip bytes from disk at `ref.path`.
  3. `base64::engine::general_purpose::STANDARD.encode(&bytes)`.
  4. Set `checkpoint: Some(Checkpoint { model_zip_base64: Some(encoded), model_zip_path: None })`.
- Otherwise `checkpoint: None`.

---

## Step 7 — Python worker: `LastOnly` checkpoint frequency

**File:** `worker/trainer/trainer-py/commondef/training.py`

Add `LAST_ONLY = "LastOnly"` to `CheckpointFrequencyTypes`. Define:
```python
class CheckpointFrequencyLastOnly(BaseModel):
    type: Literal[CheckpointFrequencyTypes.LAST_ONLY] = CheckpointFrequencyTypes.LAST_ONLY
```
Update `CheckpointFrequency` union to include `CheckpointFrequencyLastOnly`.

**File:** `worker/trainer/trainer-py/training.py`

**On loop exit (normal finish) AND early stop:** Use a `last_epoch_run` flag to handle both cases:
```python
last_epoch_run = -1
for i in range(self.params.num_epochs):
    if should_stop is not None and should_stop():
        break
    last_epoch_run = i
    # ... training ...

if freq.type == CheckpointFrequencyTypes.LAST_ONLY and last_epoch_run >= 0:
    self._save_and_send_checkpoint(
        tx, label="last",
        epoch_index=last_epoch_run,
        update_index=self.params.num_updates - 1,
    )
tx.send_stop_training(StopTrainingMessage(id=self.id))
```

Also fix typo on existing line 149: `model_zip_base_64=model_zip_base64` → `model_zip_base64=model_zip_base64`.

---

## Step 8 — Python worker: load checkpoint on startup

**File:** `worker/trainer/trainer-py/main.py`

In the `StartTrainingMessage` handler, before constructing `TrainingWorkflow`, if `message.checkpoint` is not None and `model_zip_base64` is set:
1. Decode base64 → raw bytes.
2. Write to `/tmp/checkpoints/{id}/init.zip`, unzip to `/tmp/checkpoints/{id}/loaded`.
3. Restore policy + value stores:
   ```python
   checkpointer = ocp.PyTreeCheckpointer()
   policy_store = checkpointer.restore(os.path.join(restore_dir, "policy"), item=model.get_policy_store())
   value_store  = checkpointer.restore(os.path.join(restore_dir, "value"), item=model.get_value_store())
   model.load_policy_store(policy_store)
   model.load_value_store(value_store)
   ```
   The `item=` target shape must match how `_save_and_send_checkpoint` saved it (pytree of the same structure). Check `ppo_trainer.py` for `get_policy_store`/`get_value_store` return type to confirm. Optimizer states are **not** restored.

---

## Step 9 — Frontend: `CheckpointFrequency` setting in training graph

**File:** `client/src/commondef/training.ts`

Update `CheckpointFrequency` type:
```ts
export type CheckpointFrequency =
  | { type: "PostEpoch", content: number }
  | { type: "PostUpdate", content: number }
  | { type: "LastOnly" };
```

**File:** `client/src/workflows/workflowgraph.ts`

Add two settings to `TrainingJobControllerNode.settings`:
```ts
{ key: "checkpoint_frequency_type", label: "checkpoint_frequency_type", value: "PostEpoch",
  inputType: "select", options: [
    { label: "PostEpoch", value: "PostEpoch" },
    { label: "PostUpdate", value: "PostUpdate" },
    { label: "LastOnly", value: "LastOnly" },
  ]},
{ key: "checkpoint_frequency_n", label: "checkpoint_frequency_n", value: 1, min: 1, step: 1 },
```

Update `buildTrainingController`:
```ts
const freqType = getStringSetting('training-job-controller', 'checkpoint_frequency_type', 'PostEpoch');
const freqN = getNumericSetting('training-job-controller', 'checkpoint_frequency_n', 1);
checkpoint_frequency: freqType === 'LastOnly'
  ? { type: 'LastOnly' }
  : { type: freqType as 'PostEpoch' | 'PostUpdate', content: freqN },
```

---

## Step 10 — Frontend: `Checkpoint` model type in graph nodes

**File:** `client/src/workflows/workflowgraph.ts`

- Add `{ label: "Checkpoint", value: "Checkpoint" }` to `type` select in `ControlModelNode` and `ValueModelNode`.
- Add `checkpoint_id` setting to both nodes: `{ key: "checkpoint_id", label: "Checkpoint ID", value: "", inputType: "select", options: [] }`.
- Update `buildTrainingWorkflow` to read model type and checkpoint_id:
  ```ts
  const controlType = getStringSetting('control-model', 'type', 'MLP');
  control_model: controlType === 'Checkpoint'
    ? { type: 'Checkpoint', content: getStringSetting('control-model', 'checkpoint_id', '') }
    : { type: 'MLP', content: buildMlpModel('control-model') },
  // same for value_model
  ```

---

## Step 11 — Frontend: checkpoint UI in `nodegrapheditor.svelte`

**File:** `client/src/lib/nodegrapheditor.svelte`

- Add props:
  - `checkpointOptionsByNode?: Record<string, Array<{label: string; value: string}>>` — options for the `checkpoint_id` select per node.
  - `onCheckpointTypeSelected?: (nodeId: string) => void` — called when type changes to `"Checkpoint"`.
  - `onCheckpointSelected?: (nodeId: string, checkpointId: string) => void` — called when a checkpoint is chosen.
- In the settings render: when a node's `type` setting value is `"Checkpoint"`:
  - Disable inputs for `num_hidden_layers`, `hidden_layer_dim`, `output_dim`, `input_dim`.
  - Show the `checkpoint_id` select with options from `checkpointOptionsByNode?.[nodeId]`.
  - On type changing to `"Checkpoint"`: fire `onCheckpointTypeSelected?.(nodeId)`.
  - On `checkpoint_id` changing: fire `onCheckpointSelected?.(nodeId, newValue)`.

**File:** `client/src/lib/traininggrapheditor.svelte`

Accept and pass through the three new props to `NodeGraphEditor`.

---

## Step 12 — Frontend: Checkpoints viewer tab

**File:** `client/src/lib/checkpointsviewer.svelte` ← **new component** (no existing component covers this)

A scrollable list of checkpoint cards. Each card shows: Checkpoint ID, Experiment ID, Run ID, Timestamp. Styling mirrors `experimentruncard.svelte`.

```svelte
<script lang="ts">
  import type { CheckpointReference } from '../commondef/training';
  const { checkpoints = [] } = $props<{ checkpoints?: CheckpointReference[] }>();
</script>
```

**File:** `client/src/App.svelte`

- Import `CheckpointsViewer`.
- Add `runCheckpoints = $state<CheckpointReference[]>([])` and `allCheckpoints = $state<CheckpointReference[]>([])`.
- Handle `AllCheckpointsMessage` → `allCheckpoints = content.checkpoints`.
- Handle `CheckpointInfoMessage` → auto-fill MLP fields of the relevant model node (control or value, distinguished by a `nodeId` passed from `onCheckpointSelected`) from `checkpoint.control_model_params` / `checkpoint.value_model_params`; update `checkpoint_id` options.
- Handle `CheckpointsForRunMessage` → `runCheckpoints = content.checkpoints`.
- On `TrainingUpdatesMessage`: for each update with non-empty `checkpoints`, append to `runCheckpoints`.
- On `RunLoadedMessage`: reset `runCheckpoints = []` (server sends `CheckpointsForRunMessage` immediately after).
- Build `checkpointOptionsByNode`: group `allCheckpoints` by experiment_id + run_id. Options labels include a header row per group (e.g. `"Exp: {exp_id} / Run: {run_id}"` as a disabled option, followed by actual checkpoint IDs). Pass as `checkpointOptionsByNode={{ 'control-model': opts, 'value-model': opts }}` to `TrainingGraphEditor`.
- `onCheckpointTypeSelected(nodeId)`: send `GetAllCheckpointsMessage`.
- `onCheckpointSelected(nodeId, ckptId)`: send `GetCheckpointInfoMessage { checkpoint_id: ckptId }`.
- Add `checkpoints` tab to training mode `editorTabs`:
  ```ts
  { id: 'checkpoints', name: 'Checkpoints', component: CheckpointsViewer, props: { checkpoints: runCheckpoints } }
  ```

---

## Critical files

| File | Change |
|------|--------|
| `scripts/init_db.sh` | Add `control_model_params`, `value_model_params` TEXT cols |
| `app/src/commondef/experiments.rs` | `Option<MLPModel>` fields on `CheckpointReference`; private `CheckpointReferenceRow` + `From` impl |
| `app/src/db/db_orm.rs` | Update `create_checkpoint_ref` + `get_checkpoint_ref`; add `get_checkpoint_refs_for_run`, `get_all_checkpoint_refs`, `add_checkpoint_ref_to_run` |
| `app/src/commondef/clientmessages.rs` | 3 new server→client messages + 2 new client→server messages + structs |
| `app/src/main.rs` | Save checkpoints from remote updates (Steps 4, 5, 6) |
| `worker/trainer/trainer-py/commondef/training.py` | `LastOnly` variant |
| `worker/trainer/trainer-py/training.py` | `LastOnly` handling (normal + early stop); base64 typo fix |
| `worker/trainer/trainer-py/main.py` | Load checkpoint before training starts |
| `client/src/commondef/training.ts` | `LastOnly` in `CheckpointFrequency`; `MLPModel` in `CheckpointReference` |
| `client/src/commondef/messages.ts` | New TS message types |
| `client/src/workflows/workflowgraph.ts` | `checkpoint_frequency` settings; `Checkpoint` model type; `buildTrainingController` + `buildTrainingWorkflow` updates |
| `client/src/lib/nodegrapheditor.svelte` | Conditional disable/show for checkpoint mode; new props |
| `client/src/lib/traininggrapheditor.svelte` | Pass-through checkpoint props |
| `client/src/lib/checkpointsviewer.svelte` | New component |
| `client/src/App.svelte` | `runCheckpoints`, `allCheckpoints` state; message handlers; Checkpoints tab; checkpoint node props |

---

## Verification

1. `cargo check` in `app/` after Steps 1–6.
2. `npm run build` in `client/` after Steps 9–12.
3. End-to-end remote training with `PostEpoch(1)`:
   - Checkpoint zip on disk, `checkpoint_references` row in Postgres, `training_runs.checkpoints_refs` updated, Checkpoints tab shows card live.
4. `LastOnly` — normal finish: checkpoint appears after all epochs complete.
5. `LastOnly` — user stops early: checkpoint still fires for the last completed epoch.
6. Checkpoint reuse (remote path):
   - Select `Checkpoint` in ControlModel → dropdown shows saved IDs grouped by experiment/run → select one → MLP fields auto-populate → start remote training → worker receives zip and trains from it.
