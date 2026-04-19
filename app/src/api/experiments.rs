use rocket::http::Status;
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::commondef::experiments::{
    CheckpointReference,
    TrainingExperiment,
    TrainingExperimentRun,
    TrainingRunStatus,
};

// ── Response types ───────────────────────────

#[derive(Serialize)]
pub struct ExperimentsResponse {
    pub experiments: Vec<TrainingExperiment>,
}

#[derive(Serialize)]
pub struct ExperimentResponse {
    pub experiment: TrainingExperiment,
}

#[derive(Serialize)]
pub struct RunsResponse {
    pub runs: Vec<TrainingExperimentRun>,
}

#[derive(Serialize)]
pub struct RunResponse {
    pub run: TrainingExperimentRun,
}

#[derive(Serialize)]
pub struct CheckpointsResponse {
    pub checkpoints: Vec<CheckpointReference>,
}

// ── Request types ────────────────────────────

#[derive(Deserialize)]
pub struct CreateExperimentBody {
    pub name: Option<String>,
}

// ── Handlers ─────────────────────────────────

#[get("/training/experiments")]
pub async fn list_experiments()
-> Result<Json<ExperimentsResponse>, Status>
{
    let experiments = crate::postgres_orm()
        .get_experiments_by_project(
            crate::PROJECT_ID,
        )
        .await
        .map_err(|_| Status::InternalServerError)?;
    Ok(Json(ExperimentsResponse { experiments }))
}

#[post(
    "/training/experiments",
    data = "<body>",
    format = "json"
)]
pub async fn create_experiment(
    body: Json<CreateExperimentBody>,
) -> Result<Json<ExperimentResponse>, Status>
{
    let id = Uuid::new_v4().to_string();
    let name = body
        .into_inner()
        .name
        .unwrap_or_else(|| id.clone());
    let row = crate::postgres_orm()
        .create_experiment(
            crate::PROJECT_ID,
            &id,
            &name,
            &[],
        )
        .await
        .map_err(|_| Status::InternalServerError)?;
    Ok(Json(ExperimentResponse { experiment: row }))
}

#[get(
    "/training/experiments/<experiment_id>/runs"
)]
pub async fn list_runs(
    experiment_id: String,
) -> Result<Json<RunsResponse>, Status>
{
    let runs = crate::postgres_orm()
        .get_runs_by_experiment(
            crate::PROJECT_ID,
            &experiment_id,
        )
        .await
        .map_err(|_| Status::InternalServerError)?;
    Ok(Json(RunsResponse { runs }))
}

#[post(
    "/training/experiments/<experiment_id>/runs",
    format = "json"
)]
pub async fn create_run(
    experiment_id: String,
) -> Result<Json<RunResponse>, Status>
{
    let run_id = Uuid::new_v4().to_string();
    let logs_id = format!("logs-{}", run_id);
    let metrics_id =
        format!("metrics-{}", run_id);
    let graph_id =
        format!("graph-{}", run_id);
    let pg = crate::postgres_orm();
    let row = pg
        .create_run(
            crate::PROJECT_ID,
            &experiment_id,
            &run_id,
            &TrainingRunStatus::NotStarted,
            &logs_id,
            &metrics_id,
            &graph_id,
            &[],
            &[],
        )
        .await
        .map_err(|_| Status::InternalServerError)?;
    let _ = pg
        .add_run_to_experiment(
            &experiment_id,
            &run_id,
        )
        .await;
    Ok(Json(RunResponse { run: row }))
}

#[get(
    "/training/experiments\
     /<experiment_id>/runs/<run_id>"
)]
pub async fn get_run(
    experiment_id: String,
    run_id: String,
) -> Result<Json<RunResponse>, Status>
{
    let row = crate::postgres_orm()
        .get_run(
            crate::PROJECT_ID,
            &experiment_id,
            &run_id,
        )
        .await
        .map_err(|_| Status::InternalServerError)?
        .ok_or(Status::NotFound)?;
    Ok(Json(RunResponse { run: row }))
}

#[get(
    "/training/experiments/<experiment_id>\
     /runs/<run_id>/checkpoints"
)]
pub async fn get_run_checkpoints(
    experiment_id: String,
    run_id: String,
) -> Result<Json<CheckpointsResponse>, Status>
{
    let checkpoints = crate::postgres_orm()
        .get_checkpoint_refs_for_run(
            &experiment_id,
            &run_id,
        )
        .await
        .map_err(|_| Status::InternalServerError)?;
    Ok(Json(CheckpointsResponse { checkpoints }))
}
