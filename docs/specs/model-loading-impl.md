# MJCF Model Import — Implementation

## Files to modify

| File | Change |
|---|---|
| `scripts/init_db.sh` | Add `name` column to `model_references` CREATE TABLE; add `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for existing DBs |
| `app/Cargo.toml` | Add `zip = "2"` (no existing dependency covers ZIP extraction) |
| `app/src/commondef/experiments.rs` | Add `name: String` field to `ModelReference` |
| `app/src/db/db_orm.rs` | Add `name` param to `create_model_reference`; add `get_all_model_references` |
| `app/src/api/models.rs` | Add `list_models` and `save_model` handlers |
| `app/src/main.rs` | Mount two new routes |
| `AGENTS.md` | Add instruction: write spec + impl plan to `docs/specs/` before writing any code for a structured feature |

## Files to create

| File | Purpose |
|---|---|
| `docs/specs/model-loading.md` | High-level spec |
| `docs/specs/model-loading-impl.md` | This implementation plan |

---

## Step 1 — Update `scripts/init_db.sh`

Add `name TEXT NOT NULL DEFAULT ''` to the `model_references` CREATE TABLE block:

```sql
CREATE TABLE IF NOT EXISTS model_references (
    id           TEXT        PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL,
    name         TEXT        NOT NULL DEFAULT '',
    path         TEXT        NOT NULL,
    model_type   TEXT        NOT NULL
);
```

Add immediately after the CREATE TABLE block to handle existing DBs:

```sql
ALTER TABLE model_references
    ADD COLUMN IF NOT EXISTS
    name TEXT NOT NULL DEFAULT '';
```

No changes needed to `scripts/seed_db.sh`.

---

## Step 2 — Add `zip` crate

`app/Cargo.toml`:
```toml
zip = "2"
```

No existing dependency in the project handles ZIP archive extraction.

---

## Step 3 — Update `ModelReference` struct

`app/src/commondef/experiments.rs` — add `name: String` between `id` and `path`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
#[typeshare]
pub struct ModelReference {
    pub created_at: Timestamp,
    pub id: String,
    pub name: String,
    pub path: String,
    pub model_type: String,
}
```

---

## Step 4 — Update ORM (`app/src/db/db_orm.rs`)

**Update `create_model_reference`** — add `name: &str` parameter and include in INSERT:
```sql
INSERT INTO model_references (id, created_at, name, path, model_type)
VALUES ($1, NOW(), $2, $3, $4) RETURNING *
```

**Add `get_all_model_references`** (follows same pattern as `get_all_checkpoint_refs`):
```rust
pub async fn get_all_model_references(
    &self,
) -> Result<Vec<ModelReference>, sqlx::Error> {
    sqlx::query_as::<_, ModelReference>(
        "SELECT * FROM model_references
        ORDER BY created_at DESC",
    )
    .fetch_all(&self.pool)
    .await
}
```

---

## Step 5 — Implement route handlers (`app/src/api/models.rs`)

**Response types:**
```rust
pub struct ModelsResponse { pub models: Vec<ModelReference> }
pub struct SaveModelResponse { pub id: String }
```

**Upload form:**
```rust
#[derive(FromForm)]
pub struct ModelUploadForm<'r> {
    pub file: TempFile<'r>,
}
```

**`GET /models`** — calls `get_all_model_references`, wraps in `ModelsResponse`.

**`POST /models/save_model`** handler logic:
1. Read file bytes from `form.file.path()` via `std::fs::read`
2. Detect ZIP by checking first 4 bytes for PK magic (`[0x50, 0x4B, 0x03, 0x04]`)
3. **ZIP branch:** open archive, find first entry with no `/` in name ending in `.xml`, read bytes, validate
4. **XML branch:** validate bytes directly
5. Validation failure → return `Status::BadRequest`
6. Get `original_name` from `form.file.name()` (fallback: `"model.xml"`)
7. Generate `id = uuid::Uuid::new_v4().to_string()`
8. Read `models_dir` from `env::var("MODELS_DIR")`
9. Create `{models_dir}/{id}/` via `std::fs::create_dir_all`
10. Persist file via `form.file.persist_to(dest).await`
11. Call `postgres_orm().create_model_reference(&id, &original_name, &full_path, "mjcf").await`
12. Return `Json(SaveModelResponse { id })`

**Validation helper (file-private):**
```rust
fn has_mujoco_root(content: &[u8]) -> bool {
    let s = std::str::from_utf8(content).unwrap_or("");
    s.contains("<mujoco")
}
```

---

## Step 6 — Mount routes in `main.rs`

Add to `.mount("/api", routes![...])`:
```rust
api::models::list_models,
api::models::save_model,
```

---

## Step 7 — Run typeshare

```bash
typeshare app/src/ --lang=typescript --output-file=client/src/common/types.ts
```

---

## Verification (points 1–3)

1. Start server with `MODELS_DIR=/tmp/models DATABASE_URL=...`
2. `POST /api/models/save_model` with valid MJCF `.xml` → `{ id: "..." }`
3. `POST /api/models/save_model` with `.zip` containing valid MJCF XML at root → `{ id: "..." }`
4. `POST /api/models/save_model` with non-MJCF XML → HTTP 400
5. `GET /api/models` → `{ models: [...] }` with entries for both uploads, no file bytes
6. Confirm `$MODELS_DIR/{id}/` exists on disk with the uploaded file
7. Confirm DB row has correct `id`, `name`, `path`, `model_type`, `created_at`

---

# Training graph uses saved model — Implementation

## Files to modify

| File | Change |
|---|---|
| `app/src/commondef/training.rs` | Add `pub simulation_model: String` to `PPOTrainingMethod` |
| `app/src/commondef/remoteworkermessages.rs` | Fix missing struct name bug; add `HashMap` import; add `simulation_models: HashMap<String, String>` to `StartTrainingMessage` |
| `app/src/api/training.rs` | Validate `simulation_model` ID from DB; use `model_ref.path`; remove `TEST_MJCF_MODEL` usage |
| `app/src/remote_workflows/training_controller.rs` | Add `find_xml_in_zip_root` helper; update `load_mjcf_package_zip_base64` for ZIP paths; add `TrainingMethod` import; populate `simulation_models` in `run_training_job`; update `StartTrainingMessage` construction |
| `worker/trainer/trainer-py/commondef/training.py` | Add `simulation_model: str` to `PPOTrainingMethod` |
| `worker/trainer/trainer-py/commondef/remoteworkermessages.py` | Add `Dict` to imports; add `simulation_models: Dict[str, str]` to `StartTrainingMessage` |
| `worker/trainer/trainer-py/main.py` | Add `_extract_sim_model_from_zip` helper; in `StartTrainingMessage` handling, extract sim model zip and update `runtime` before simulator re-init |
| `docs/specs/model-loading.md` | Append point #4 spec (done) |
| `docs/specs/model-loading-impl.md` | This section |

---

## Step 8 — Add `simulation_model` to `PPOTrainingMethod` (Rust)

`app/src/commondef/training.rs` — add as last field of `PPOTrainingMethod`:
```rust
pub simulation_model: String,
```

---

## Step 9 — Fix and extend `StartTrainingMessage` (Rust)

`app/src/commondef/remoteworkermessages.rs`:

1. Add at top (std imports first): `use std::collections::HashMap;`
2. Fix line with missing struct name: `pub struct   {` → `pub struct StartTrainingMessage {`
3. Add field:
```rust
pub simulation_models: HashMap<String, String>,
```

---

## Step 10 — Training start API: validate model + use path (Rust)

`app/src/api/training.rs`:

1. Remove `use std::env;` import (no longer used)
2. Remove `let model_path = env::var("TEST_MJCF_MODEL").ok();`
3. Before `if is_local {`, insert DB lookup:
```rust
let sim_model_id = match &controller
    .training_method
{
    TrainingMethod::PPO(ref p) =>
        p.simulation_model.clone(),
};
let model_ref = crate::postgres_orm()
    .get_model_reference(&sim_model_id)
    .await
    .map_err(|_| Status::InternalServerError)?
    .ok_or(Status::BadRequest)?;
let model_path = model_ref.path.clone();
```
4. Local path: change `let path = model_path.unwrap_or_default();` → `let path = model_path;`
5. Remote path: remove `let model_path_str = ...` and `let mjcf = ...`; replace `mjcf` argument with `model_path` in `spawn_remote_training_task` call
6. `spawn_remote_training_task` signature: rename `mjcf: String` → `model_path: String`
7. Inside `spawn_remote_training_task`: compute `mjcf` locally via `std::fs::read_to_string(&model_path).unwrap_or_default()`; pass `Some(model_path)` instead of `None` to `run_training_job`

---

## Step 11 — Support ZIP model paths + populate `simulation_models` (Rust)

`app/src/remote_workflows/training_controller.rs`:

Add `TrainingMethod` to the existing crate import:
```rust
use crate::commondef::training::{
    TrainingGraph, TrainingJobController, TrainingMethod,
};
```

Add private helper before `load_mjcf_package_zip_base64`:
```rust
fn find_xml_in_zip_root(zip_bytes: &[u8]) -> Result<String> {
    use std::io::Cursor;
    let cursor = Cursor::new(zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if !name.contains('/') && name.ends_with(".xml") {
            return Ok(name);
        }
    }
    Err("No XML file at zip root".into())
}
```

Add ZIP-path branch at the top of `load_mjcf_package_zip_base64` (before the existing XML-path logic):
```rust
if path.extension().and_then(|e| e.to_str()) == Some("zip") {
    let zip_bytes = fs::read(path)?;
    let entrypoint = find_xml_in_zip_root(&zip_bytes)?;
    println!(
        "Using model zip directly {:?} ({} bytes), entrypoint={}",
        path, zip_bytes.len(), entrypoint
    );
    return Ok((BASE64.encode(&zip_bytes), entrypoint));
}
```

In `run_training_job`, after computing `model_zip_base64` and `mjcf_entrypoint`, build `simulation_models`:
```rust
let sim_model_id = match &training_controller.training_method {
    TrainingMethod::PPO(ref p) => p.simulation_model.clone(),
};
let mut simulation_models = std::collections::HashMap::new();
if let Some(ref zip_b64) = model_zip_base64 {
    simulation_models.insert(sim_model_id, zip_b64.clone());
}
```

Update `StartTrainingMessage` construction to include `simulation_models`.

---

## Step 12 — Update Python types

`worker/trainer/trainer-py/commondef/training.py` — add to `PPOTrainingMethod`:
```python
simulation_model: str
```

`worker/trainer/trainer-py/commondef/remoteworkermessages.py`:
- Add `Dict` to `from typing import ...`
- Add to `StartTrainingMessage`:
```python
simulation_models: Dict[str, str]
```

---

## Step 13 — Python worker uses `simulation_models`

`worker/trainer/trainer-py/main.py`:

Add helper after `_extract_model_bundle_to_tempdir`:
```python
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
```

In `handle_connection`, in the `StartTrainingMessage` block — before the simulator re-init lines, insert (and define `ppo_params` here instead of after the re-init):
```python
ppo_params = training_controller.training_method.content
sim_model_zip = start_msg.simulation_models.get(ppo_params.simulation_model)
if sim_model_zip:
    if runtime.model_tmpdir is not None:
        runtime.model_tmpdir.cleanup()
    sim_tmpdir, sim_model_path = _extract_sim_model_from_zip(sim_model_zip)
    runtime.model_tmpdir = sim_tmpdir
    runtime.model_path = sim_model_path
    runtime.mjcf = None
```

Remove the now-duplicate `ppo_params = training_controller.training_method.content` line that comes after the simulator re-init.

---

## Step 14 — Run typeshare

```bash
typeshare app/src/ --lang=typescript --output-file=client/src/common/types.ts
```

---

## Verification (point #4)

1. Import a model via `POST /api/models/save_model` → get back an `id`
2. Start training via the training start API with `training_graph.training_job_controller.training_method.content.simulation_model` set to that `id` → HTTP 200
3. Start training with a non-existent model ID → HTTP 400
4. (Remote) Confirm `StartTrainingMessage` received by the Python worker contains `simulation_models` with the correct ID and a non-empty base64 string
5. (Remote) Confirm the Python worker logs show the extracted model path being used for simulator initialization
