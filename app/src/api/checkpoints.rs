use rocket::http::Status;
use rocket::serde::json::Json;
use serde::Serialize;
use utoipa::ToSchema;

use crate::api::route_doc;
use crate::commondef::record::CheckpointReference;

// ── Response types ───────────────────────────

#[derive(Serialize, ToSchema)]
pub struct CheckpointsResponse {
    pub checkpoints: Vec<CheckpointReference>,
}

#[derive(Serialize, ToSchema)]
pub struct CheckpointResponse {
    pub checkpoint: CheckpointReference,
}

// ── Handlers ─────────────────────────────────

#[route_doc(tag = "Checkpoints", ok = CheckpointsResponse)]
#[get("/checkpoints")]
pub async fn list_checkpoints() -> Result<Json<CheckpointsResponse>, Status> {
    let checkpoints = crate::postgres_orm()
        .get_all_checkpoint_refs()
        .await
        .map_err(|_| Status::InternalServerError)?;
    Ok(Json(CheckpointsResponse { checkpoints }))
}

#[route_doc(
    tag = "Checkpoints",
    ok = CheckpointResponse,
    errors(404),
    params(checkpoint_id = "Checkpoint UUID"),
)]
#[get("/checkpoints/<checkpoint_id>")]
pub async fn get_checkpoint(checkpoint_id: String) -> Result<Json<CheckpointResponse>, Status> {
    let checkpoint = crate::postgres_orm()
        .get_checkpoint_ref(&checkpoint_id)
        .await
        .map_err(|_| Status::InternalServerError)?
        .ok_or(Status::NotFound)?;
    Ok(Json(CheckpointResponse { checkpoint }))
}
