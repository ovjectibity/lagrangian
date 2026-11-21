//! Training workload entry point.

use super::{ApiResult, ApiState, Authenticated, workloads};
use crate::commondef::api::apimessages::{
    StartTrainingRequestBody, StartTrainingResponseBody,
};
use crate::commondef::records::workload_graph::WorkloadGraph;
use rocket::State;
use rocket::serde::json::Json;
use std::sync::Arc;

/// Start training with project-scoped assets and checkpoints.
#[utoipa::path(
    post, path = "/api/start_training",
    tag = "Workloads",
    request_body = StartTrainingRequestBody,
    responses(
        (status = 200, body = StartTrainingResponseBody),
        (
            status = 400,
            description = "Invalid request",
        ),
        (
            status = 401,
            description = "Invalid or revoked access key",
        ),
        (
            status = 403,
            description = "Resource belongs to another user",
        ),
        (
            status = 404,
            description = "Resource not found",
        ),
        (
            status = 409,
            description = "Resource conflict or workload active",
        ),
        (
            status = 413,
            description = "JSON body exceeds the size limit",
        ),
        (
            status = 422,
            description = "Request fields missing or malformed",
        ),
        (
            status = 500,
            description = "Persistence or storage failed",
        ),
        (
            status = 503,
            description = "App server is shutting down",
        ),
    )
)]
#[post("/start_training", format = "json", data = "<body>")]
pub async fn start_training(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    body: Json<StartTrainingRequestBody>,
) -> ApiResult<Json<StartTrainingResponseBody>> {
    let body = body.into_inner();
    let id = workloads::start_workload(
        state.inner().clone(),
        auth,
        body.project_id,
        WorkloadGraph::TrainingGraph(body.training_graph),
        body.name,
        body.description,
        body.tags,
    )
    .await?;
    Ok(Json(StartTrainingResponseBody { workload_id: id }))
}
