use std::env;
use std::io::{Cursor, Read as _};
use std::path::PathBuf;

use rocket::form::Form;
use rocket::fs::TempFile;
use rocket::http::Status;
use rocket::serde::json::Json;
use serde::Serialize;
use zip::ZipArchive;

use crate::commondef::experiments::ModelReference;
use crate::commondef::model::WorldBody;
use crate::workflows::WorkflowRuntime;

// ── Response types ───────────────────────────

#[derive(Serialize)]
pub struct DefaultModelResponse {
    pub id: String,
    pub model: WorldBody,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub models: Vec<ModelReference>,
}

#[derive(Serialize)]
pub struct SaveModelResponse {
    pub id: String,
}

// ── Upload form ──────────────────────────────

#[derive(FromForm)]
pub struct ModelUploadForm<'r> {
    pub file: TempFile<'r>,
}

// ── Handlers ─────────────────────────────────

#[get("/models/default")]
pub async fn get_default_model()
-> Result<Json<DefaultModelResponse>, Status>
{
    let model_path =
        env::var("TEST_MJCF_MODEL")
        .unwrap_or_default();

    let world_body =
        rocket::tokio::task::spawn_blocking(
            move || {
                let runtime =
                    WorkflowRuntime::load(model_path)
                    .map_err(|e| {
                        println!(
                            "WorkflowRuntime load \
                             failed: {:?}", e
                        );
                    })?;
                runtime.world_body().map_err(|e| {
                    println!(
                        "world_body failed: {:?}", e
                    );
                })
            },
        )
        .await
        .map_err(|_| Status::InternalServerError)?
        .map_err(|_| Status::InternalServerError)?;

    Ok(Json(DefaultModelResponse {
        id: "default".to_string(),
        model: world_body,
    }))
}

#[get("/models")]
pub async fn list_models()
-> Result<Json<ModelsResponse>, Status>
{
    let models = crate::postgres_orm()
        .get_all_model_references()
        .await
        .map_err(|_| Status::InternalServerError)?;
    Ok(Json(ModelsResponse { models }))
}

#[post("/models/save_model", data = "<form>")]
pub async fn save_model(
    mut form: Form<ModelUploadForm<'_>>,
) -> Result<Json<SaveModelResponse>, Status>
{
    let original_name = form
        .file
        .name()
        .unwrap_or("model.xml")
        .to_string();

    let id = uuid::Uuid::new_v4().to_string();

    let models_dir = env::var("MODELS_DIR")
        .map_err(|_| Status::InternalServerError)?;

    let dest_dir: PathBuf =
        [models_dir.as_str(), id.as_str()]
        .iter()
        .collect();

    rocket::tokio::fs::create_dir_all(&dest_dir)
        .await
        .map_err(|_| Status::InternalServerError)?;

    let dest = dest_dir.join(&original_name);

    // persist_to works for both in-memory and
    // on-disk temp files (small files stay in
    // memory and path() would return None)
    form.file
        .persist_to(&dest)
        .await
        .map_err(|_| Status::InternalServerError)?;

    let bytes =
        rocket::tokio::fs::read(&dest)
        .await
        .map_err(|_| Status::InternalServerError)?;

    let is_zip = bytes.len() >= 4
        && bytes[..4] == [0x50, 0x4B, 0x03, 0x04];

    if is_zip {
        let cursor = Cursor::new(bytes.as_slice());
        let mut archive =
            ZipArchive::new(cursor)
            .map_err(|_| Status::BadRequest)?;

        let xml_bytes: Option<Vec<u8>> =
            (0..archive.len()).find_map(|i| {
                let mut entry =
                    archive.by_index(i).ok()?;
                let name =
                    entry.name().to_string();
                if name.contains('/')
                    || !name.ends_with(".xml")
                {
                    return None;
                }
                let mut buf = Vec::new();
                entry.read_to_end(&mut buf).ok()?;
                Some(buf)
            });

        let xml_bytes =
            xml_bytes.ok_or(Status::BadRequest)?;
        if !has_mujoco_root(&xml_bytes) {
            let _ = std::fs::remove_dir_all(
                &dest_dir,
            );
            return Err(Status::BadRequest);
        }
    } else if !has_mujoco_root(&bytes) {
        let _ = std::fs::remove_dir_all(&dest_dir);
        return Err(Status::BadRequest);
    }

    let full_path =
        dest.to_string_lossy().into_owned();

    crate::postgres_orm()
        .create_model_reference(
            &id,
            &original_name,
            &full_path,
            "mjcf",
        )
        .await
        .map_err(|_| Status::InternalServerError)?;

    Ok(Json(SaveModelResponse { id }))
}

fn has_mujoco_root(content: &[u8]) -> bool {
    std::str::from_utf8(content)
        .map(|s| s.contains("<mujoco"))
        .unwrap_or(false)
}
