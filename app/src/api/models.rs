use std::env;

use rocket::http::Status;
use rocket::serde::json::Json;
use serde::Serialize;

use crate::commondef::model::WorldBody;
use crate::workflows::WorkflowRuntime;

#[derive(Serialize)]
pub struct DefaultModelResponse {
    pub id: String,
    pub model: WorldBody,
}

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
