//! Testing workload entry point.

use super::{ApiResult, ApiState, Authenticated, workloads};
use crate::commondef::api::apimessages::{
    StartTestRequestBody, StartTestResponseBody,
};
use crate::commondef::records::workload_graph::WorkloadGraph;
use rocket::State;
use rocket::serde::json::Json;
use std::sync::Arc;

/// Start custom or dataset testing with referenced checkpoints.
#[utoipa::path(
    post, path = "/api/start_test",
    tag = "Workloads",
    request_body = StartTestRequestBody,
    responses(
        (status = 200, body = StartTestResponseBody),
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
#[post("/start_test", format = "json", data = "<body>")]
pub async fn start_test(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    body: Json<StartTestRequestBody>,
) -> ApiResult<Json<StartTestResponseBody>> {
    let body = body.into_inner();
    let id = workloads::start_workload(
        state.inner().clone(),
        auth,
        body.project_id,
        WorkloadGraph::TestGraph(body.test_graph),
        body.name,
        body.description,
        body.tags,
    )
    .await?;
    Ok(Json(StartTestResponseBody { workload_id: id }))
}
