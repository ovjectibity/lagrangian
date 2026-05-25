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

## Verification

1. Start server with `MODELS_DIR=/tmp/models DATABASE_URL=...`
2. `POST /api/models/save_model` with valid MJCF `.xml` → `{ id: "..." }`
3. `POST /api/models/save_model` with `.zip` containing valid MJCF XML at root → `{ id: "..." }`
4. `POST /api/models/save_model` with non-MJCF XML → HTTP 400
5. `GET /api/models` → `{ models: [...] }` with entries for both uploads, no file bytes
6. Confirm `$MODELS_DIR/{id}/` exists on disk with the uploaded file
7. Confirm DB row has correct `id`, `name`, `path`, `model_type`, `created_at`
