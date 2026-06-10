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

    // Persist to a scratch path first; works for
    // both in-memory and on-disk temp files.
    let scratch = dest_dir.join("upload.tmp");
    form.file
        .persist_to(&scratch)
        .await
        .map_err(|_| {
            let _ = std::fs::remove_dir_all(&dest_dir);
            Status::InternalServerError
        })?;

    let bytes =
        rocket::tokio::fs::read(&scratch)
        .await
        .map_err(|_| {
            let _ = std::fs::remove_dir_all(&dest_dir);
            Status::InternalServerError
        })?;

    let is_zip = bytes.len() >= 4
        && bytes[..4] == [0x50, 0x4B, 0x03, 0x04];

    // xml_path: path stored in DB for MuJoCo to load
    let (xml_path, xml_name) = if is_zip {
        let cursor = Cursor::new(bytes.as_slice());
        let mut archive =
            ZipArchive::new(cursor).map_err(|_| {
                let _ =
                    std::fs::remove_dir_all(&dest_dir);
                Status::BadRequest
            })?;

        // Extract all entries; track root XML path.
        let mut found_xml: Option<PathBuf> = None;
        for i in 0..archive.len() {
            let mut entry =
                archive.by_index(i).map_err(|_| {
                    Status::BadRequest
                })?;
            let entry_name =
                entry.name().to_string();
            // Skip directory entries
            if entry_name.ends_with('/') {
                continue;
            }
            let out =
                dest_dir.join(&entry_name);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|_| {
                        Status::InternalServerError
                    })?;
            }
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .map_err(|_| Status::BadRequest)?;
            std::fs::write(&out, &buf)
                .map_err(|_| {
                    Status::InternalServerError
                })?;
            // A root-level XML is the MJCF entry.
            if !entry_name.contains('/')
                && entry_name.ends_with(".xml")
                && has_mujoco_root(&buf)
            {
                found_xml = Some(out);
            }
        }
        let xml = found_xml.ok_or_else(|| {
            let _ =
                std::fs::remove_dir_all(&dest_dir);
            Status::BadRequest
        })?;
        let name = xml
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.xml")
            .to_string();
        let stem = xml
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let zip_dest = dest_dir.join(format!("{stem}.zip"));
        let _ = rocket::tokio::fs::rename(&scratch, &zip_dest).await;
        (xml, name)
    } else {
        if !has_mujoco_root(&bytes) {
            let _ =
                std::fs::remove_dir_all(&dest_dir);
            return Err(Status::BadRequest);
        }
        let original_name = form
            .file
            .name()
            .unwrap_or("model")
            .to_string()
            + ".xml";
        let dest = dest_dir.join(&original_name);
        rocket::tokio::fs::rename(&scratch, &dest)
            .await
            .map_err(|_| {
                let _ =
                    std::fs::remove_dir_all(&dest_dir);
                Status::InternalServerError
            })?;
        (dest, original_name)
    };

    let full_path =
        xml_path.to_string_lossy().into_owned();

    crate::postgres_orm()
        .create_model_reference(
            &id,
            &xml_name,
            &full_path,
            "mjcf",
        )
        .await
        .map_err(|_| {
            let _ =
                std::fs::remove_dir_all(&dest_dir);
            Status::InternalServerError
        })?;

    Ok(Json(SaveModelResponse { id }))
}

fn has_mujoco_root(content: &[u8]) -> bool {
    std::str::from_utf8(content)
        .map(|s| s.contains("<mujoco"))
        .unwrap_or(false)
}
