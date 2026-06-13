use rocket::http::Status;
use rocket::serde::json::Json;
use serde::Serialize;

use crate::commondef::record::CheckpointReference;

// ── Response types ───────────────────────────

#[derive(Serialize)]
pub struct CheckpointsResponse {
    pub checkpoints: Vec<CheckpointReference>,
}

#[derive(Serialize)]
pub struct CheckpointResponse {
    pub checkpoint: CheckpointReference,
}

// ── Handlers ─────────────────────────────────

#[get("/checkpoints")]
pub async fn list_checkpoints()
-> Result<Json<CheckpointsResponse>, Status>
{
    let checkpoints = crate::postgres_orm()
        .get_all_checkpoint_refs()
        .await
        .map_err(|_| Status::InternalServerError)?;
    Ok(Json(CheckpointsResponse { checkpoints }))
}

#[get("/checkpoints/<checkpoint_id>")]
pub async fn get_checkpoint(
    checkpoint_id: String,
) -> Result<Json<CheckpointResponse>, Status>
{
    let checkpoint = crate::postgres_orm()
        .get_checkpoint_ref(&checkpoint_id)
        .await
        .map_err(|_| Status::InternalServerError)?
        .ok_or(Status::NotFound)?;
    Ok(Json(CheckpointResponse { checkpoint }))
}
