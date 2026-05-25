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
