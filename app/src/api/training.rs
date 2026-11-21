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
#[utoipa::path(post, path = "/api/start_training",
    request_body = StartTrainingRequestBody,
    responses((status = 200, body = StartTrainingResponseBody),
              (status = 409)))]
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
