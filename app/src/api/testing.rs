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
#[utoipa::path(post, path = "/api/start_test",
    request_body = StartTestRequestBody,
    responses((status = 200, body = StartTestResponseBody),
              (status = 409)))]
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
