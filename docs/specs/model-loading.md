# MJCF Model Import

**Scope:** Server-side only. No client changes.

---

## POST `/api/models/save_model` (multipart/form-data)

- **Input:** `file` field — either a single `.xml` file or a `.zip` archive
- **XML path:** Validate the file content contains `<mujoco` as the root element (after any XML declaration/comments)
- **ZIP path:** Find the first `.xml` file at the archive root (no path separator in its name); apply the same `<mujoco` check to that file's bytes. If no root-level XML found → 400.
- **On success:**
  1. Generate UUID as model `id`
  2. Create directory `$MODELS_DIR/{id}/`
  3. Persist uploaded file to `$MODELS_DIR/{id}/{original_filename}`
  4. Insert row into `model_references` with `id`, `name` (original filename), `path` (full path), `model_type = "mjcf"`, `created_at`
  5. Return `{ id: String }`
- **On validation failure:** Return HTTP 400

---

## GET `/api/models`

- **Response:** `{ models: ModelReference[] }` — all rows in `model_references`, ordered by `created_at DESC`
- Returns metadata only; no file content

---

## Schema change

`model_references` table: add `name TEXT NOT NULL DEFAULT ''` column.

---

## Training graph uses saved model

**Scope:** Server-side (Rust) + Python remote worker. No client changes.

### `PPOTrainingMethod.simulation_model` (new mandatory field)

- Type: `String` — a model ID referencing a `model_references` row
- Mandatory — the training start API returns HTTP 400 if the ID is not found in `model_references`

### Training start API behaviour changes

- Extract `simulation_model` ID from `body.training_graph.training_job_controller.training_method` (PPO variant)
- Call `get_model_reference(sim_model_id)` on the DB → 400 if not found
- Use `model_ref.path` for both local and remote training (removes `TEST_MJCF_MODEL` env var dependency)
- Local (`AllOnServer`): `WorkflowRuntime::load(model_ref.path)` — XML paths only; ZIP models require remote mode
- Remote: model is packaged (zip + entrypoint) and sent to worker

### `StartTrainingMessage.simulation_models` (new field)

- Type: `HashMap<String, String>` — key: model ID, value: base64-encoded model zip
- Populated by the server for all models referenced in the training graph
- The Python worker uses this to reinitialize its MuJoCo simulator on `StartTrainingMessage`

### Remote worker behaviour changes

- On `StartTrainingMessage`: look up `ppo_params.simulation_model` in `start_msg.simulation_models`. If present, extract the zip, auto-detect the XML entrypoint, and reinitialize `runtime.model_path` and `runtime.mjcf`. The simulator re-init that follows then picks up the updated paths.
- `ArmModel` (policy/value network) is unaffected by simulation model changes.
