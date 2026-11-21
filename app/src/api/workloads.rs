//! Workload lifecycle, update ingestion, replay and subscriptions.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use rocket::State;
use rocket::http::Status;
use rocket::serde::json::Json;
use rocket_ws as ws;
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use uuid::Uuid;

use super::{
    ApiResult, ApiState, Authenticated, internal, sql_error,
};
use crate::commondef::api::apimessages::*;
use crate::commondef::records::assets::*;
use crate::commondef::records::product::Project;
use crate::commondef::records::workload::*;
use crate::commondef::records::workload_graph::*;
use crate::commondef::remoteworker::remoteworkermessages::*;
use crate::db::{self, validations};
use crate::workflows::common_worker_controller::Result as WorkerResult;
use crate::workflows::local_worker_controller::LocalWorkerController;
use crate::workflows::remote_worker_controller::RemoteWorkerController;
use crate::workflows::{self, final_record_id, model_key};

/// Create a named project for the authenticated user.
#[utoipa::path(post, path = "/api/create_project",
    request_body = CreateProjectRequestBody,
    responses((status = 200, body = CreateProjectResponseBody)))]
#[post("/create_project", format = "json", data = "<body>")]
pub async fn create_project(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    body: Json<CreateProjectRequestBody>,
) -> ApiResult<Json<CreateProjectResponseBody>> {
    let now = Utc::now();
    let id = Uuid::new_v4().to_string();
    state
        .postgres
        .create_project(&Project {
            project_id: id.clone(),
            user_id: auth.user.user_id,
            created_at: now,
            updated_at: now,
            name: body.into_inner().name,
        })
        .await
        .map_err(sql_error)?;
    Ok(Json(CreateProjectResponseBody { project_id: id }))
}

/// List only the authenticated user's projects.
#[utoipa::path(get, path = "/api/list_projects",
    responses((status = 200, body = GetProjectsResponseBody)))]
#[get("/list_projects")]
pub async fn list_projects(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
) -> ApiResult<Json<GetProjectsResponseBody>> {
    let mut ids = Vec::new();
    let mut offset = 0;
    loop {
        let page = state
            .postgres
            .list_projects(&auth.user.user_id, (500, offset))
            .await
            .map_err(sql_error)?;
        let count = page.len();
        ids.extend(page.into_iter().map(|p| p.project_id));
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    Ok(Json(GetProjectsResponseBody { project_ids: ids }))
}

/// Foreign-key references prevent deleting nonempty projects.
#[utoipa::path(delete, path = "/api/delete_project",
    request_body = DeleteProjectsRequestBody,
    responses((status = 204), (status = 409)))]
#[delete("/delete_project", format = "json", data = "<body>")]
pub async fn delete_project(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    body: Json<DeleteProjectsRequestBody>,
) -> ApiResult<Status> {
    let _delivery = state.delivery.lock().await;
    for id in &body.project_ids {
        state.authorize_project(&auth, id).await?;
    }
    let mut transaction =
        state.postgres.pool().begin().await.map_err(sql_error)?;
    for id in &body.project_ids {
        sqlx::query(
            "DELETE FROM product.project WHERE project_id = $1",
        )
        .bind(id)
        .execute(&mut *transaction)
        .await
        .map_err(sql_error)?;
    }
    transaction.commit().await.map_err(sql_error)?;
    Ok(Status::NoContent)
}

/// Assemble workload metadata and graphs through the ORM.
#[utoipa::path(get, path = "/api/list_workloads",
    params(("project_id" = String, Query, description = "Request filter")),
    responses((status = 200, body = ListWorkloadsResponseBody)))]
#[get("/list_workloads?<project_id>")]
pub async fn list_workloads(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    project_id: String,
) -> ApiResult<Json<ListWorkloadsResponseBody>> {
    state.authorize_project(&auth, &project_id).await?;
    let mut workloads = Vec::new();
    let mut offset = 0;
    loop {
        let page = db::list_workloads(
            &state.postgres,
            &state.mongo,
            Some(&project_id),
            Some(&auth.user.user_id),
            None,
            None,
            (500, offset),
        )
        .await
        .map_err(internal)?;
        let count = page.len();
        workloads.extend(page.into_iter().map(workload_response));
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    Ok(Json(ListWorkloadsResponseBody { workloads }))
}

/// Signal the executing job without waiting for the worker lock.
#[utoipa::path(post, path = "/api/stop_workload",
    request_body = StopWorkloadRequestBody,
    responses((status = 202), (status = 409)))]
#[post("/stop_workload", format = "json", data = "<body>")]
pub async fn stop_workload(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    body: Json<StopWorkloadRequestBody>,
) -> ApiResult<Status> {
    state.authorize_workload(&auth, &body.workload_id).await?;
    let jobs = state.jobs.lock().await;
    let (_, stop) =
        jobs.get(&body.workload_id).ok_or(Status::Conflict)?;
    stop.send(true).map_err(|_| Status::Conflict)?;
    Ok(Status::Accepted)
}

pub(super) async fn start_workload(
    state: Arc<ApiState>,
    auth: Authenticated,
    project_id: String,
    graph: WorkloadGraph,
    name: String,
    description: String,
    tags: Vec<String>,
) -> ApiResult<String> {
    state.authorize_project(&auth, &project_id).await?;
    let (models, checkpoints) =
        resolve_assets(&state, &auth, &graph).await?;
    let remote = is_remote(&graph);
    let id = Uuid::new_v4().to_string();
    let now = Utc::now();
    let workload = Workload {
        workload_id: id.clone(),
        workload_type: match &graph {
            WorkloadGraph::TrainingGraph(_) => {
                WorkloadType::Training
            }
            WorkloadGraph::TestGraph(_) => WorkloadType::Testing,
        },
        project_id,
        user_id: auth.user.user_id,
        created_at: now,
        updated_at: now,
        launched_at: None,
        completed_at: None,
        scheduled_at: None,
        launch_origin: LaunchOrigin::Api,
        status: WorkloadStatus::Scheduled,
        workload_graph_id: Uuid::new_v4().to_string(),
        simulation_model_ids: models
            .iter()
            .map(|r| r.simulation_model_id.clone())
            .collect(),
        workload_graph: graph,
        name,
        description,
        tags,
    };
    let delivery = state.delivery.lock().await;
    validations::validate_workload(&state.postgres, &workload)
        .await
        .map_err(|_| Status::BadRequest)?;
    let (stop_tx, stop_rx) = watch::channel(false);
    {
        let mut jobs = state.jobs.lock().await;
        validate_capacity(&jobs, remote)?;
        jobs.insert(id.clone(), (remote, stop_tx));
    }
    if let Err(error) = db::create_workload(
        &state.postgres,
        &state.mongo,
        &workload,
    )
    .await
    {
        state.jobs.lock().await.remove(&id);
        return Err(internal(error));
    }
    drop(delivery);
    tokio::spawn(run_workload(
        state,
        workload,
        models,
        checkpoints,
        stop_rx,
    ));
    Ok(id)
}

fn validate_capacity(
    jobs: &std::collections::HashMap<
        String,
        (bool, watch::Sender<bool>),
    >,
    remote: bool,
) -> ApiResult<()> {
    if jobs.values().any(|(r, _)| *r == remote) {
        Err(Status::Conflict)
    } else {
        Ok(())
    }
}

fn is_remote(graph: &WorkloadGraph) -> bool {
    let mode = match graph {
        WorkloadGraph::TrainingGraph(g) => {
            &g.training_job_controller.acceleration_mode
        }
        WorkloadGraph::TestGraph(g) => {
            &g.test_job_controller.acceleration_mode
        }
    };
    matches!(
        mode,
        TrainingAccelerationMode::MLGPUSimulationCPU
            | TrainingAccelerationMode::MLGPUSimulationGPU
    )
}

async fn resolve_assets(
    state: &ApiState,
    auth: &Authenticated,
    graph: &WorkloadGraph,
) -> ApiResult<(Vec<SimulationModelRef>, Vec<CheckpointRef>)> {
    let mut model_ids = Vec::new();
    let mut checkpoint_types = Vec::new();
    match graph {
        WorkloadGraph::TrainingGraph(g) => {
            let TrainingMethod::PPO(p) =
                &g.training_job_controller.training_method;
            model_ids.push(&p.simulation_model_id);
            checkpoint_types.extend([
                &p.control_model.model_type,
                &p.value_model.model_type,
            ]);
            if let ControlOutputMode::FlowMatchingAction(
                FlowMatchingAction::FlowNoise(flow),
            ) = &p.control_output_mode
            {
                checkpoint_types.push(&flow.model_type);
            }
        }
        WorkloadGraph::TestGraph(g) => {
            let control = match &g.test_job_controller.test_method
            {
                TestMethod::Custom(m) => {
                    model_ids.push(&m.simulation_model_id);
                    &m.control_model
                }
                TestMethod::Dataset(m) => &m.control_model,
            };
            checkpoint_types.push(&control.model_type);
        }
    }
    let mut models = Vec::new();
    for id in model_ids {
        models.push(state.authorize_model(auth, id).await?);
    }
    let mut checkpoints = Vec::new();
    for kind in checkpoint_types {
        if let ModelType::Checkpoint(
            CheckpointType::CheckpointID(id),
        ) = kind
            && !checkpoints
                .iter()
                .any(|c: &CheckpointRef| &c.checkpoint_id == id)
        {
            checkpoints.push(
                state.authorize_checkpoint(auth, id).await?,
            );
        }
    }
    Ok((models, checkpoints))
}

async fn execute_worker(
    start: RemoteWorkerServerMessage,
    remote: bool,
    stop: watch::Receiver<bool>,
    updates: mpsc::Sender<RemoteWorkerClientMessage>,
    launched: oneshot::Sender<()>,
) -> WorkerResult<StopWorkloadWorker> {
    if remote {
        crate::ensure_remote_worker_shutdown_hook();
        let mut slot = crate::remote_worker_slot().lock().await;
        if slot.is_none() {
            let config = crate::build_remote_worker_config()?;
            *slot = Some(
                RemoteWorkerController::initialize(config)
                    .await?,
            );
        }
        let worker =
            slot.as_mut().ok_or("Missing remote worker")?;
        worker.run_job(start, stop, updates, launched).await
    } else {
        LocalWorkerController::from_env()?
            .run_job(start, stop, updates, launched)
            .await
    }
}

async fn run_workload(
    state: Arc<ApiState>,
    mut workload: Workload,
    models: Vec<SimulationModelRef>,
    checkpoints: Vec<CheckpointRef>,
    stop: watch::Receiver<bool>,
) {
    let start = match &workload.workload_graph {
        WorkloadGraph::TrainingGraph(graph) => {
            RemoteWorkerServerMessage::StartTrainingMessage(
                StartTrainingWorkerMessage {
                    project_id: workload.project_id.clone(),
                    workload_id: workload.workload_id.clone(),
                    training_graph: graph.clone(),
                    checkpoints,
                    simulation_models: models,
                },
            )
        }
        WorkloadGraph::TestGraph(graph) => {
            RemoteWorkerServerMessage::StartTestingMessage(
                StartTestWorkerMessage {
                    project_id: workload.project_id.clone(),
                    workload_id: workload.workload_id.clone(),
                    test_graph: graph.clone(),
                    checkpoints,
                    simulation_models: models,
                },
            )
        }
    };
    let (tx, mut rx) = mpsc::channel(8);
    let (launched_tx, mut launched_rx) = oneshot::channel();
    let worker = tokio::spawn(execute_worker(
        start,
        is_remote(&workload.workload_graph),
        stop,
        tx,
        launched_tx,
    ));
    let mut launch_received = false;
    let mut failure = None;
    loop {
        tokio::select! {
            launch = &mut launched_rx, if !launch_received => {
                launch_received = true;
                if launch.is_ok() {
                    workload.launched_at = Some(Utc::now());
                    workload.status = WorkloadStatus::Ongoing;
                    if let Err(error) = set_status(&state, &workload, None).await {
                        failure = Some(error.to_string());
                        request_stop(&state, &workload.workload_id).await;
                    }
                }
            }
            message = rx.recv() => {
                let Some(message) = message else { break; };
                let update = match message {
                    RemoteWorkerClientMessage::TrainingUpdatesMessage(u)
                    | RemoteWorkerClientMessage::TestUpdatesMessage(u) => u,
                    _ => continue,
                };
                if failure.is_none()
                    && let Err(error) = ingest_update(
                        &state, &workload, update,
                    ).await
                {
                    failure = Some(error.to_string());
                    request_stop(
                        &state, &workload.workload_id,
                    ).await;
                }
            }
        }
    }
    if !launch_received && launched_rx.try_recv().is_ok() {
        workload.launched_at = Some(Utc::now());
    }
    let result = worker.await;
    let reason = if let Some(error) = failure {
        WorkloadStopReasonWorker::WorkerFailure(error)
    } else {
        match result {
            Ok(Ok(stopped)) => stopped.reason,
            Ok(Err(error)) => {
                WorkloadStopReasonWorker::WorkerFailure(
                    error.to_string(),
                )
            }
            Err(error) => {
                WorkloadStopReasonWorker::WorkerFailure(
                    error.to_string(),
                )
            }
        }
    };
    workload.status = match &reason {
        WorkloadStopReasonWorker::CompletedSuccess => {
            WorkloadStatus::Completed
        }
        WorkloadStopReasonWorker::RequestedByAppServer => {
            WorkloadStatus::UserStopped
        }
        WorkloadStopReasonWorker::WorkerFailure(_)
            if workload.launched_at.is_none() =>
        {
            WorkloadStatus::LaunchFailed
        }
        _ => WorkloadStatus::ExecutionFailed,
    };
    if let WorkloadStopReasonWorker::WorkerFailure(error) =
        &reason
    {
        eprintln!(
            "Workload {} failed: {error}",
            workload.workload_id
        );
        let now = Utc::now();
        let mut update = WorkloadUpdates {
            workload_id: workload.workload_id.clone(),
            logs: Vec::new(),
            metrics: Vec::new(),
            checkpoints: Vec::new(),
            camera_frames: Vec::new(),
            simulation_models: Vec::new(),
            model_states: Vec::new(),
        };
        update.logs.push(WorkloadLogs {
            workload_id: workload.workload_id.clone(),
            created_at: now,
            log_timestamp: now,
            log_id: Uuid::new_v4().to_string(),
            log_line_text: error.clone(),
            user_facing: false,
            log_type: LogType::Error,
            stream: LogStream::AppServer,
        });
        let mut public = update.logs[0].clone();
        public.log_id = Uuid::new_v4().to_string();
        public.user_facing = true;
        public.log_line_text = "Workload failed. Check the app-server logs for details.".into();
        update.logs.push(public);
        if let Err(error) =
            ingest_update(&state, &workload, update).await
        {
            eprintln!("Could not persist failure logs: {error}");
        }
    }
    let _delivery = state.delivery.lock().await;
    if let Err(error) =
        set_status(&state, &workload, Some(Utc::now())).await
    {
        eprintln!(
            "Could not persist terminal workload status: {error}"
        );
    } else {
        let stopped = StopWorkloadWorker {
            workload_id: workload.workload_id.clone(),
            reason,
        };
        let event = match workload.workload_type {
            WorkloadType::Training => {
                RemoteWorkerClientMessage::StopTrainingMessage(
                    stopped,
                )
            }
            WorkloadType::Testing => {
                RemoteWorkerClientMessage::StopTestingMessage(
                    stopped,
                )
            }
        };
        let _ = state.events.send(event);
    }
    state.jobs.lock().await.remove(&workload.workload_id);
}

async fn request_stop(state: &ApiState, id: &str) {
    if let Some((_, stop)) = state.jobs.lock().await.get(id) {
        let _ = stop.send(true);
    }
}

async fn set_status(
    state: &ApiState,
    workload: &Workload,
    completed_at: Option<chrono::DateTime<Utc>>,
) -> WorkerResult<()> {
    state
        .postgres
        .update_workload_status(
            &workload.workload_id,
            &workload.status,
            Utc::now(),
            workload.launched_at,
            completed_at,
        )
        .await?;
    Ok(())
}

async fn ingest_update(
    state: &ApiState,
    workload: &Workload,
    mut update: WorkloadUpdates,
) -> WorkerResult<()> {
    validations::validate_update_scope(&update, workload)?;
    let wid = &workload.workload_id;
    for model in &mut update.simulation_models {
        let temporary = model.simulation_model_id.clone();
        validations::validate_asset_identifier(&temporary)?;
        let id = final_record_id(wid, "models", &temporary);
        model.uri_path = workflows::promote_asset(
            &state.s3, &state.bucket, &model.uri_path,
            &format!("worker/{wid}/models/{temporary}/m_{temporary}.zip"),
            &model_key(&workload.project_id, &id),
        ).await?;
        model.simulation_model_id = id;
    }
    for record in &mut update.checkpoints {
        let temporary = record.checkpoint_id.clone();
        validations::validate_asset_identifier(&temporary)?;
        let id = final_record_id(wid, "checkpoints", &temporary);
        record.uri_path = workflows::promote_asset(
            &state.s3, &state.bucket, &record.uri_path,
            &format!(
                "worker/workloads/{wid}/checkpoints/\
                 {temporary}/cp_{temporary}.zip",
            ),
            &format!(
                "app/workloads/{wid}/checkpoints/{id}/cp_{id}.zip",
            ),
        ).await?;
        record.checkpoint_id = id;
    }
    for record in &mut update.camera_frames {
        let temporary = record.camera_frame_id.clone();
        validations::validate_asset_identifier(&temporary)?;
        let id = final_record_id(wid, "frames", &temporary);
        record.uri_path = workflows::promote_asset(
            &state.s3,
            &state.bucket,
            &record.uri_path,
            &format!(
                "worker/workloads/{wid}/cameras/\
                 cf_{temporary}.png",
            ),
            &format!(
                "app/workloads/{wid}/cameras/cf_{id}.png",
            ),
        )
        .await?;
        record.camera_frame_id = id;
        record.simulation_model_id = canonical_model_id(
            workload,
            &record.simulation_model_id,
        );
    }
    for record in &mut update.model_states {
        record.sim_model_state_entry_id = final_record_id(
            wid,
            "states",
            &record.sim_model_state_entry_id,
        );
        record.simulation_model_id = canonical_model_id(
            workload,
            &record.simulation_model_id,
        );
    }
    for record in &mut update.logs {
        record.log_id =
            final_record_id(wid, "logs", &record.log_id);
    }
    for record in &mut update.metrics {
        record.metrics_entry_id = final_record_id(
            wid,
            "metrics",
            &record.metrics_entry_id,
        );
    }
    let _delivery = state.delivery.lock().await;
    // Stores do not share a transaction. Stable IDs and existence
    // checks let a retry complete a previously written prefix.
    let mut attempt = 0;
    loop {
        match persist_update(state, &update).await {
            Ok(()) => break,
            Err(error) if attempt < 2 => {
                eprintln!(
                    "Retrying workload persistence: {error}"
                );
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(200))
                    .await;
            }
            Err(error) => return Err(error),
        }
    }
    let event = match workload.workload_type {
        WorkloadType::Training => {
            RemoteWorkerClientMessage::TrainingUpdatesMessage(
                update,
            )
        }
        WorkloadType::Testing => {
            RemoteWorkerClientMessage::TestUpdatesMessage(update)
        }
    };
    let _ = state.events.send(event);
    Ok(())
}

fn canonical_model_id(workload: &Workload, id: &str) -> String {
    if workload.simulation_model_ids.iter().any(|r| r == id) {
        id.to_owned()
    } else {
        final_record_id(&workload.workload_id, "models", id)
    }
}

async fn persist_update(
    state: &ApiState,
    update: &WorkloadUpdates,
) -> WorkerResult<()> {
    let pg = &state.postgres;
    let mongo = &state.mongo;
    for record in &update.simulation_models {
        if pg
            .get_simulation_model_ref(&record.simulation_model_id)
            .await?
            .is_none()
        {
            pg.create_simulation_model_ref(record).await?;
        }
    }
    let ids: Vec<_> = update
        .simulation_models
        .iter()
        .map(|m| m.simulation_model_id.clone())
        .collect();
    if !ids.is_empty() {
        pg.attach_workload_models(&update.workload_id, &ids)
            .await?;
    }
    for record in &update.checkpoints {
        if pg
            .get_checkpoint_ref(&record.checkpoint_id)
            .await?
            .is_none()
        {
            pg.create_checkpoint_ref(mongo, record).await?;
        }
    }
    for record in &update.camera_frames {
        if pg
            .get_trajectory_camera_frame_ref(
                &record.camera_frame_id,
            )
            .await?
            .is_none()
        {
            pg.create_trajectory_camera_frame_ref(mongo, record)
                .await?;
        }
    }
    let mut logs = Vec::new();
    for record in &update.logs {
        if mongo.get_workload_log(&record.log_id).await?.is_none()
        {
            logs.push(record.clone());
        }
    }
    mongo.create_workload_logs(pg, &logs).await?;
    let mut metrics = Vec::new();
    for record in &update.metrics {
        if mongo
            .get_workload_user_metrics(&record.metrics_entry_id)
            .await?
            .is_none()
        {
            metrics.push(record.clone());
        }
    }
    mongo
        .create_workload_user_metrics_batch(pg, &metrics)
        .await?;
    let mut states = Vec::new();
    for record in &update.model_states {
        if mongo
            .get_simulation_model_trajectory_state(
                &record.sim_model_state_entry_id,
            )
            .await?
            .is_none()
        {
            states.push(record.clone());
        }
    }
    mongo
        .create_simulation_model_trajectory_states(pg, &states)
        .await?;
    Ok(())
}

async fn user_update(
    state: &ApiState,
    auth: &Authenticated,
    update: WorkloadUpdates,
) -> ApiResult<WorkloadUpdateUser> {
    let structures = state
        .postgres
        .list_workload_structure_lut(None)
        .await
        .map_err(sql_error)?;
    let metrics_lut = state
        .postgres
        .list_user_metrics_lut(None)
        .await
        .map_err(sql_error)?;
    let names = |id, excluded: &[u32]| {
        structures
            .iter()
            .find(|s| s.workload_structure_id == id)
            .map(|s| {
                s.workload_state_index_names
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        !excluded.contains(&(*i as u32))
                    })
                    .map(|(_, name)| name.clone())
                    .collect()
            })
    };
    let logs = update
        .logs
        .into_iter()
        .filter(|l| l.user_facing)
        .map(|r| WorkloadLogsUser {
            workload_id: r.workload_id,
            created_at: r.created_at,
            log_timestamp: r.log_timestamp,
            log_id: r.log_id,
            log_line_text: r.log_line_text,
            log_type: r.log_type,
        })
        .collect();
    let mut metrics = Vec::new();
    for r in update.metrics {
        let definition = metrics_lut
            .iter()
            .find(|m| m.metric_id == r.metric_id)
            .ok_or(Status::InternalServerError)?;
        let excluded: Vec<_> = definition
            .aggregation_indices
            .iter()
            .chain(&definition.na_indices)
            .copied()
            .collect();
        metrics.push(WorkloadUserMetricsUser {
            created_at: r.created_at,
            workload_id: r.workload_id,
            workload_structure_state_values: r
                .workload_structure_state_values,
            metric_values: r.metric_values,
            metric_name: definition.name.clone(),
            workload_structure_index_names: names(
                definition.workload_structure_id,
                &excluded,
            ),
        });
    }
    let mut camera_frames = Vec::new();
    for r in update.camera_frames {
        camera_frames.push(TrajectoryCameraFrameRefUser {
            uri_path: state
                .download_link(auth, &r.uri_path)
                .await?,
            workload_structure_index_names: names(
                r.workload_structure_id,
                &[],
            ),
            camera_frame_id: r.camera_frame_id,
            workload_id: r.workload_id,
            workload_structure_state: r.workload_structure_state,
            camera_id: r.camera_id,
            model_camera_name: r.model_camera_name,
            simulation_model_id: r.simulation_model_id,
        });
    }
    let model_states = update
        .model_states
        .into_iter()
        .map(|r| WorkloadSimulationModelTrajectoryStateUser {
            workload_structure_index_names: names(
                r.workload_structure_id,
                &[],
            ),
            created_at: r.created_at,
            sim_model_state_entry_id: r.sim_model_state_entry_id,
            workload_id: r.workload_id,
            workload_structure_state: r.workload_structure_state,
            simulation_model_id: r.simulation_model_id,
            state: r.state,
        })
        .collect();
    let mut checkpoints = Vec::new();
    for record in update.checkpoints {
        checkpoints.push(
            super::checkpoints::checkpoint_response(
                state, record,
            )
            .await?,
        );
    }
    let mut simulation_models = Vec::new();
    for record in update.simulation_models {
        simulation_models.push(
            super::models::model_response(state, auth, record)
                .await?,
        );
    }
    Ok(WorkloadUpdateUser {
        workload_id: update.workload_id,
        logs,
        metrics,
        checkpoints,
        simulation_models,
        camera_frames,
        model_states,
    })
}

fn workload_response(r: Workload) -> WorkloadUser {
    WorkloadUser {
        workload_id: r.workload_id,
        workload_type: r.workload_type,
        project_id: r.project_id,
        user_id: r.user_id,
        launched_at: r.launched_at,
        launch_origin: r.launch_origin,
        scheduled_at: r.scheduled_at,
        completed_at: r.completed_at,
        status: r.status,
        workload_graph_id: r.workload_graph_id,
        workload_graph: r.workload_graph,
        tags: r.tags,
        simulation_model_ids: r.simulation_model_ids,
        name: r.name,
        description: r.description,
    }
}

/// Read persisted logs, metrics and assets, with fresh links.
#[utoipa::path(get, path = "/api/get_workload_updates",
    params(("workload_id" = String, Query, description = "Request filter")),
    responses((status = 200, body = WorkloadUpdatesResponseBody)))]
#[get("/get_workload_updates?<workload_id>")]
pub async fn get_workload_updates(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    workload_id: String,
) -> ApiResult<Json<WorkloadUpdatesResponseBody>> {
    let _delivery = state.delivery.lock().await;
    let workload =
        state.authorize_workload(&auth, &workload_id).await?;
    let update = history(state, &workload).await?;
    Ok(Json(WorkloadUpdatesResponseBody {
        workload_id,
        updates: vec![user_update(state, &auth, update).await?],
    }))
}

async fn history(
    state: &ApiState,
    workload: &Workload,
) -> ApiResult<WorkloadUpdates> {
    let id = &workload.workload_id;
    let mut update = WorkloadUpdates {
        workload_id: id.clone(),
        logs: Vec::new(),
        metrics: Vec::new(),
        checkpoints: Vec::new(),
        simulation_models: Vec::new(),
        camera_frames: Vec::new(),
        model_states: Vec::new(),
    };
    // Each collection has independent pagination and ordering.
    let mut offset = 0;
    loop {
        let page = state
            .mongo
            .list_workload_logs(id, (500, offset))
            .await
            .map_err(internal)?;
        let count = page.len();
        update.logs.extend(page);
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    offset = 0;
    loop {
        let page = state
            .mongo
            .list_workload_user_metrics(id, None, (500, offset))
            .await
            .map_err(internal)?;
        let count = page.len();
        update.metrics.extend(page);
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    offset = 0;
    loop {
        let page = state
            .mongo
            .list_simulation_model_trajectory_states(
                id,
                None,
                None,
                (500, offset),
            )
            .await
            .map_err(internal)?;
        let count = page.len();
        update.model_states.extend(page);
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    offset = 0;
    loop {
        let page = state
            .postgres
            .list_checkpoint_refs(Some(id), None, (500, offset))
            .await
            .map_err(sql_error)?;
        let count = page.len();
        update.checkpoints.extend(page);
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    offset = 0;
    loop {
        let page = state
            .postgres
            .list_trajectory_camera_frame_refs(
                id,
                None,
                None,
                None,
                (500, offset),
            )
            .await
            .map_err(sql_error)?;
        let count = page.len();
        update.camera_frames.extend(page);
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    for id in &workload.simulation_model_ids {
        let model = state
            .postgres
            .get_simulation_model_ref(id)
            .await
            .map_err(sql_error)?
            .ok_or(Status::InternalServerError)?;
        update.simulation_models.push(model);
    }
    Ok(update)
}

/// Issue a one-minute, single-use browser WebSocket ticket.
#[utoipa::path(post, path = "/api/get_workload_live_ticket",
    params(("workload_id" = String, Query, description = "Request filter")),
    responses((status = 200, body = WorkloadLiveTicketResponseBody)))]
#[post("/get_workload_live_ticket?<workload_id>")]
pub async fn live_ticket(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    workload_id: String,
) -> ApiResult<Json<WorkloadLiveTicketResponseBody>> {
    state.authorize_workload(&auth, &workload_id).await?;
    let ticket = Uuid::new_v4().to_string();
    let mut tickets = state.tickets.lock().await;
    let now = Instant::now();
    tickets.retain(|_, (expiry, _, _)| *expiry > now);
    tickets.insert(
        ticket.clone(),
        (now + Duration::from_secs(60), auth.key_id, workload_id),
    );
    Ok(Json(WorkloadLiveTicketResponseBody {
        ticket,
        expires_at: Utc::now() + chrono::Duration::seconds(60),
    }))
}

/// Replay persisted updates, then stream new events for one workload.
#[utoipa::path(get, path = "/api/get_workload_live_updates",
    params(("workload_id" = String, Query, description = "Request filter"), ("ticket" = String, Query, description = "Request filter")),
    responses((status = 101), (status = 401)))]
#[get("/get_workload_live_updates?<workload_id>&<ticket>")]
pub async fn live_updates(
    state: &State<Arc<ApiState>>,
    socket: ws::WebSocket,
    workload_id: String,
    ticket: String,
) -> ApiResult<ws::Channel<'static>> {
    let (expiry, key_id, scope) = state
        .tickets
        .lock()
        .await
        .remove(&ticket)
        .ok_or(Status::Unauthorized)?;
    validate_live_ticket(expiry, &scope, &workload_id)?;
    let auth = state.authenticate_ticket(&key_id).await?;
    state.authorize_workload(&auth, &workload_id).await?;
    let state = state.inner().clone();
    let mut events = state.events.subscribe();
    Ok(socket.channel(move |mut socket| Box::pin(async move {
        let mut seen = HashSet::new();
        let mut pending = replay(&state, &auth, &workload_id, &mut seen)
            .await.map_err(|_| ws::result::Error::ConnectionClosed)?;
        loop {
            for message in pending.drain(..) {
                let stopped = matches!(message, WorkloadUpdatesServer::WorkloadStoppedMessage(_));
                socket.send(ws::Message::Text(
                    serde_json::to_string(&message)
                        .map_err(|_| ws::result::Error::ConnectionClosed)?,
                )).await?;
                if stopped {
                    socket.close(None).await?;
                    return Ok(());
                }
            }
            tokio::select! {
                incoming = socket.next() => match incoming {
                    Some(Ok(ws::Message::Text(text))) => {
                        match serde_json::from_str::<WorkloadUpdatesClient>(&text) {
                            Ok(WorkloadUpdatesClient::RequestAllUpdatesMessage) => {
                                seen.clear();
                                pending = replay(&state, &auth, &workload_id, &mut seen)
                                    .await.map_err(|_| ws::result::Error::ConnectionClosed)?;
                            }
                            Err(_) => return Err(ws::result::Error::ConnectionClosed),
                        }
                    }
                    Some(Ok(ws::Message::Ping(bytes))) => {
                        socket.send(ws::Message::Pong(bytes)).await?;
                    }
                    Some(Ok(ws::Message::Pong(_))) => {}
                    _ => return Ok(()),
                },
                event = events.recv() => {
                    match event {
                        Ok(RemoteWorkerClientMessage::TrainingUpdatesMessage(mut u))
                        | Ok(RemoteWorkerClientMessage::TestUpdatesMessage(mut u))
                            if u.workload_id == workload_id => {
                            retain_unseen(&mut u, &mut seen);
                            if has_records(&u) {
                                let u = user_update(&state, &auth, u).await
                                    .map_err(|_| ws::result::Error::ConnectionClosed)?;
                                pending.push(WorkloadUpdatesServer::WorkloadUpdatesMessage(vec![u]));
                            }
                        }
                        Ok(RemoteWorkerClientMessage::StopTrainingMessage(s))
                        | Ok(RemoteWorkerClientMessage::StopTestingMessage(s))
                            if s.workload_id == workload_id => {
                            pending.push(stopped_response(s));
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            pending = replay(&state, &auth, &workload_id, &mut seen)
                                .await.map_err(|_| ws::result::Error::ConnectionClosed)?;
                        }
                        Err(broadcast::error::RecvError::Closed) => return Ok(()),
                        _ => {}
                    }
                }
            }
        }
    })))
}

fn validate_live_ticket(
    expiry: Instant,
    scope: &str,
    id: &str,
) -> ApiResult<()> {
    if expiry <= Instant::now() || scope != id {
        Err(Status::Unauthorized)
    } else {
        Ok(())
    }
}

async fn replay(
    state: &ApiState,
    auth: &Authenticated,
    id: &str,
    seen: &mut HashSet<String>,
) -> ApiResult<Vec<WorkloadUpdatesServer>> {
    let _delivery = state.delivery.lock().await;
    let workload = state.authorize_workload(auth, id).await?;
    let mut update = history(state, &workload).await?;
    retain_unseen(&mut update, seen);
    let mut messages = Vec::new();
    if has_records(&update) {
        messages.push(
            WorkloadUpdatesServer::WorkloadUpdatesMessage(vec![
                user_update(state, auth, update).await?,
            ]),
        );
    }
    let reason = match workload.status {
        WorkloadStatus::Completed => {
            Some(WorkloadStopReasonWorker::CompletedSuccess)
        }
        WorkloadStatus::UserStopped => {
            Some(WorkloadStopReasonWorker::RequestedByAppServer)
        }
        WorkloadStatus::LaunchFailed
        | WorkloadStatus::ExecutionFailed => {
            Some(WorkloadStopReasonWorker::WorkerFailure(
                "Workload failed; see its logs".into(),
            ))
        }
        _ => None,
    };
    if let Some(reason) = reason {
        messages.push(stopped_response(StopWorkloadWorker {
            workload_id: id.to_owned(),
            reason,
        }));
    }
    Ok(messages)
}

fn stopped_response(
    stopped: StopWorkloadWorker,
) -> WorkloadUpdatesServer {
    WorkloadUpdatesServer::WorkloadStoppedMessage(WorkloadStopped {
        workload_id: stopped.workload_id,
        reason: match stopped.reason {
            WorkloadStopReasonWorker::CompletedSuccess => WorkloadStoppedReasonUser::CompletedSuccess,
            WorkloadStopReasonWorker::RequestedByAppServer => WorkloadStoppedReasonUser::RequestedByUser,
            WorkloadStopReasonWorker::WorkerFailure(_) => WorkloadStoppedReasonUser::WorkerFailure(
                "Workload failed; see its logs".into(),
            ),
        },
    })
}

fn has_records(update: &WorkloadUpdates) -> bool {
    !update.logs.is_empty()
        || !update.metrics.is_empty()
        || !update.checkpoints.is_empty()
        || !update.camera_frames.is_empty()
        || !update.simulation_models.is_empty()
        || !update.model_states.is_empty()
}

fn retain_unseen(
    update: &mut WorkloadUpdates,
    seen: &mut HashSet<String>,
) {
    update
        .logs
        .retain(|r| seen.insert(format!("log:{}", r.log_id)));
    update.metrics.retain(|r| {
        seen.insert(format!("metric:{}", r.metrics_entry_id))
    });
    update.checkpoints.retain(|r| {
        seen.insert(format!("checkpoint:{}", r.checkpoint_id))
    });
    update.camera_frames.retain(|r| {
        seen.insert(format!("frame:{}", r.camera_frame_id))
    });
    update.simulation_models.retain(|r| {
        seen.insert(format!("model:{}", r.simulation_model_id))
    });
    update.model_states.retain(|r| {
        seen.insert(format!(
            "state:{}",
            r.sim_model_state_entry_id
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn empty_update(id: &str) -> WorkloadUpdates {
        WorkloadUpdates {
            workload_id: id.into(),
            logs: Vec::new(),
            metrics: Vec::new(),
            checkpoints: Vec::new(),
            simulation_models: Vec::new(),
            camera_frames: Vec::new(),
            model_states: Vec::new(),
        }
    }

    fn test_graph() -> WorkloadGraph {
        serde_json::from_value(json!({"TestGraph": {
            "test_job_controller": {
                "test_method": {"type": "Dataset", "content": {
                    "dataset": {"type": "Libero", "content": {
                        "tasks_index": [0], "libero_dataset_type": "Libero10",
                        "num_envs_per_task": 1, "num_timesteps": 2
                    }},
                    "control_model": {
                        "model_type": {"type": "New"},
                        "model": {"type": "VLA"},
                        "input_spec": {"type": "StandardInputSpec", "content": {
                            "base_body_name": null, "states": []
                        }},
                        "output_spec": {"type": "ProprioceptiveOutputSpec", "content": {
                            "base_body_name": "robot", "state": {
                                "is_delta": false, "rebase_body": "robot",
                                "state_item": {"type": "QPos"}
                            }
                        }}
                    }
                }},
                "compute_instance": "AWSG4dn",
                "acceleration_mode": "MLCPUSimulationCPU"
            }
        }})).unwrap()
    }

    #[test]
    fn replay_deduplicates_records_independently() {
        let mut update = empty_update("w");
        update.logs.push(WorkloadLogs {
            workload_id: "w".into(),
            created_at: Utc::now(),
            log_timestamp: Utc::now(),
            log_id: "same".into(),
            log_line_text: "hello".into(),
            user_facing: true,
            log_type: LogType::Info,
            stream: LogStream::Worker,
        });
        let mut seen = HashSet::new();
        let original = update.clone();
        retain_unseen(&mut update, &mut seen);
        assert_eq!(update.logs.len(), 1);
        let mut duplicate = original;
        retain_unseen(&mut duplicate, &mut seen);
        assert!(!has_records(&duplicate));
    }

    #[test]
    fn live_tickets_are_scoped_and_expire() {
        let future = Instant::now() + Duration::from_secs(60);
        assert!(validate_live_ticket(future, "a", "a").is_ok());
        assert!(validate_live_ticket(future, "a", "b").is_err());
        assert!(
            validate_live_ticket(Instant::now(), "a", "a")
                .is_err()
        );
    }

    #[test]
    fn capacity_is_reserved_per_worker_location() {
        let mut jobs = std::collections::HashMap::new();
        let (stop, _) = watch::channel(false);
        jobs.insert("w".into(), (false, stop));
        assert!(validate_capacity(&jobs, false).is_err());
        assert!(validate_capacity(&jobs, true).is_ok());
    }

    /// Uses disposable DBs and a local S3-compatible stub, never AWS.
    #[tokio::test]
    #[ignore = "requires API_TEST_DATABASE_URL, API_TEST_MONGODB_URI and API_TEST_S3_ENDPOINT"]
    async fn persistence_auth_and_asset_downloads() {
        use crate::commondef::records::product::*;
        use crate::db::db_orm::PostgresOrm;
        use crate::db::mongo_orm::MongoOrm;
        use aws_sdk_s3::config::retry::RetryConfig;
        use aws_sdk_s3::config::{
            BehaviorVersion, Credentials, Region,
        };
        use rocket::http::Header;
        use rocket::local::asynchronous::Client;

        let pg = Arc::new(
            PostgresOrm::new(
                &std::env::var("API_TEST_DATABASE_URL").unwrap(),
            )
            .await
            .unwrap(),
        );
        let mongo = Arc::new(
            MongoOrm::new(
                &std::env::var("API_TEST_MONGODB_URI").unwrap(),
            )
            .await
            .unwrap(),
        );
        let config = aws_sdk_s3::config::Builder::new()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .credentials_provider(Credentials::new(
                "test", "test", None, None, "test",
            ))
            .retry_config(RetryConfig::disabled())
            .endpoint_url(
                std::env::var("API_TEST_S3_ENDPOINT").unwrap(),
            )
            .force_path_style(true)
            .build();
        let state = Arc::new(ApiState::new(
            pg.clone(),
            mongo.clone(),
            aws_sdk_s3::Client::from_conf(config),
            "test-assets".into(),
        ));
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        pg.create_group(&Group {
            group_id: id.clone(),
            name: "Test".into(),
            created_at: now,
            updated_at: now,
            login_type: LoginType::Direct,
        })
        .await
        .unwrap();
        let user = User {
            user_id: id.clone(),
            group_id: id.clone(),
            name: "Owner".into(),
            email: format!("{id}@test.local"),
            created_at: now,
            updated_at: now,
            role: UserRole::Admin,
        };
        pg.create_user(&user).await.unwrap();
        pg.create_access_key(&AccessKey {
            access_key_id: id.clone(),
            access_key_value: id.clone(),
            user_id: id.clone(),
            name: "Test".into(),
            created_at: now,
            status: AccessKeyStatus::Active,
        })
        .await
        .unwrap();
        pg.create_project(&Project {
            project_id: id.clone(),
            user_id: id.clone(),
            name: "Project".into(),
            created_at: now,
            updated_at: now,
        })
        .await
        .unwrap();
        let auth = Authenticated {
            user: user.clone(),
            key_id: id.clone(),
        };
        let workload = Workload {
            workload_id: id.clone(),
            workload_type: WorkloadType::Testing,
            project_id: id.clone(),
            user_id: id.clone(),
            created_at: now,
            updated_at: now,
            launched_at: Some(now),
            completed_at: None,
            scheduled_at: None,
            launch_origin: LaunchOrigin::Api,
            status: WorkloadStatus::Ongoing,
            workload_graph_id: id.clone(),
            workload_graph: test_graph(),
            simulation_model_ids: vec![],
            tags: vec!["test".into()],
            name: "Evaluation".into(),
            description: "Round trip".into(),
        };
        db::create_workload(&pg, &mongo, &workload)
            .await
            .unwrap();
        let mut stored = db::get_workload(&pg, &mongo, &id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.name, "Evaluation");
        assert_eq!(stored.description, "Round trip");
        stored.name = "Updated".into();
        stored.description = "Changed".into();
        db::update_workload(&pg, &mongo, &stored).await.unwrap();
        assert_eq!(
            db::get_workload(&pg, &mongo, &id)
                .await
                .unwrap()
                .unwrap()
                .name,
            "Updated"
        );

        let model_id = Uuid::new_v4().to_string();
        let frame_id = Uuid::new_v4().to_string();
        let source = format!(
            "worker/{id}/models/{model_id}/m_{model_id}.zip"
        );
        state
            .s3
            .put_object()
            .bucket(&state.bucket)
            .key(&source)
            .body(
                aws_sdk_s3::primitives::ByteStream::from_static(
                    b"model",
                ),
            )
            .send()
            .await
            .unwrap();
        let frame_source =
            format!(
                "worker/workloads/{id}/cameras/cf_{frame_id}.png",
            );
        state
            .s3
            .put_object()
            .bucket(&state.bucket)
            .key(&frame_source)
            .body(
                aws_sdk_s3::primitives::ByteStream::from_static(
                    b"png",
                ),
            )
            .send()
            .await
            .unwrap();
        let structure = pg
            .list_workload_structure_lut(Some(
                &WorkloadType::Testing,
            ))
            .await
            .unwrap()
            .into_iter()
            .find(|s| {
                s.method_applicability == WorkloadMethod::Dataset
            })
            .unwrap();
        let mut update = empty_update(&id);
        update.simulation_models.push(SimulationModelRef {
            project_id: id.clone(),
            simulation_model_id: model_id.clone(),
            created_at: now,
            updated_at: now,
            name: "Generated".into(),
            uri_path: format!("s3://test-assets/{source}"),
            model_type: SimulationModelType::Mjcf,
            file_type: SimulationModelFileType::Zip,
            base_file_name: "model.xml".into(),
        });
        update.camera_frames.push(TrajectoryCameraFrameRef {
            camera_frame_id: frame_id,
            workload_id: id.clone(),
            workload_structure_id: structure
                .workload_structure_id,
            workload_structure_state: vec![0, 0, 0],
            camera_id: "camera".into(),
            model_camera_name: "camera".into(),
            uri_path: format!("s3://test-assets/{frame_source}"),
            simulation_model_id: model_id.clone(),
        });
        update.logs.push(WorkloadLogs {
            workload_id: id.clone(),
            created_at: now,
            log_timestamp: now,
            log_id: "public".into(),
            log_line_text: "Visible".into(),
            user_facing: true,
            log_type: LogType::Info,
            stream: LogStream::Worker,
        });
        let mut hidden = update.logs[0].clone();
        hidden.log_id = "hidden".into();
        hidden.user_facing = false;
        update.logs.push(hidden);
        let mut events = state.events.subscribe();
        ingest_update(&state, &workload, update.clone())
            .await
            .unwrap();
        assert!(events.try_recv().is_ok());
        ingest_update(&state, &workload, update).await.unwrap();
        let stored = db::get_workload(&pg, &mongo, &id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.simulation_model_ids.len(), 1);
        while events.try_recv().is_ok() {}
        let mut missing = empty_update(&id);
        let mut missing_model = pg
            .get_simulation_model_ref(
                &stored.simulation_model_ids[0],
            )
            .await
            .unwrap()
            .unwrap();
        let missing_id = Uuid::new_v4().to_string();
        missing_model.simulation_model_id = missing_id.clone();
        missing_model.uri_path = format!(
            "s3://test-assets/worker/{id}/models/{missing_id}/m_{missing_id}.zip",
        );
        missing.simulation_models.push(missing_model);
        assert!(
            ingest_update(&state, &workload, missing)
                .await
                .is_err()
        );
        assert!(events.try_recv().is_err());
        assert!(
            pg.get_simulation_model_ref(&final_record_id(
                &id,
                "models",
                &missing_id
            ),)
                .await
                .unwrap()
                .is_none()
        );

        let raw = history(&state, &stored).await.unwrap();
        assert_eq!(raw.logs.len(), 2);
        assert_eq!(raw.camera_frames.len(), 1);
        let frame_prefix = format!(
            "s3://test-assets/app/workloads/{id}/cameras/cf_",
        );
        assert!(raw.camera_frames[0].uri_path
            .starts_with(&frame_prefix));
        let response =
            user_update(&state, &auth, raw).await.unwrap();
        assert_eq!(response.logs.len(), 1);
        assert_eq!(response.simulation_models.len(), 1);
        assert_eq!(
            response.camera_frames[0].simulation_model_id,
            final_record_id(&id, "models", &model_id)
        );
        let link = response.simulation_models[0].uri_path.clone();
        assert!(link.starts_with("/api/assets/download/"));
        assert!(
            !serde_json::to_string(&response)
                .unwrap()
                .contains("s3://")
        );
        let global_id = Uuid::new_v4().to_string();
        pg.create_checkpoint_ref(
            &mongo,
            &CheckpointRef {
                checkpoint_id: global_id.clone(),
                created_at: now,
                updated_at: now,
                name: "Predefined".into(),
                uri_path: "s3://test-assets/predefined.zip"
                    .into(),
                workload_id: None,
                workload_structure_id: None,
                workload_structure_state: None,
                na_indices: vec![],
                model_type: CheckpointModelType::Policy,
                model_spec: MLModel::MLP(MLPModelSpec {
                    num_hidden_layers: 1,
                    hidden_layer_dim: 2,
                    input_dim: 1,
                    output_dim: 2,
                }),
            },
        )
        .await
        .unwrap();
        let shared = pg
            .list_project_checkpoint_refs(&id, (500, 0))
            .await
            .unwrap();
        assert!(
            shared.iter().any(|c| c.checkpoint_id == global_id)
        );
        let mut wrong = auth.clone();
        wrong.user.user_id = "other".into();
        assert!(
            state
                .authorize_checkpoint(&wrong, &global_id)
                .await
                .is_ok()
        );
        assert_eq!(
            state
                .authorize_model(
                    &wrong,
                    &stored.simulation_model_ids[0]
                )
                .await
                .unwrap_err(),
            Status::Forbidden
        );
        assert!(
            validations::validate_model_unreferenced(
                &pg,
                &mongo,
                &stored.simulation_model_ids[0],
            )
            .await
            .is_err()
        );

        let client = Client::tracked(
            rocket::build().manage(state.clone()).mount(
                "/api",
                routes![
                    list_projects,
                    live_ticket,
                    get_workload_updates,
                    super::super::models::download_asset,
                ],
            ),
        )
        .await
        .unwrap();
        assert_eq!(
            client
                .get("/api/list_projects")
                .dispatch()
                .await
                .status(),
            Status::Unauthorized
        );
        let bearer =
            Header::new("Authorization", format!("Bearer {id}"));
        assert_eq!(
            client
                .get("/api/list_projects")
                .header(bearer.clone())
                .dispatch()
                .await
                .status(),
            Status::Ok
        );
        let download = client.get(link).dispatch().await;
        assert_eq!(download.status(), Status::Ok);
        assert_eq!(
            download.into_bytes().await.unwrap(),
            b"model"
        );
        let ticket = client
            .post(format!(
                "/api/get_workload_live_ticket?workload_id={id}"
            ))
            .header(bearer.clone())
            .dispatch()
            .await;
        assert_eq!(ticket.status(), Status::Ok);
        let ticket: WorkloadLiveTicketResponseBody =
            ticket.into_json().await.unwrap();
        assert!(
            state
                .tickets
                .lock()
                .await
                .remove(&ticket.ticket)
                .is_some()
        );
        assert!(
            state
                .tickets
                .lock()
                .await
                .remove(&ticket.ticket)
                .is_none()
        );
        pg.update_workload_status(
            &id,
            &WorkloadStatus::UserStopped,
            Utc::now(),
            Some(now),
            Some(Utc::now()),
        )
        .await
        .unwrap();
        assert!(matches!(
            db::get_workload(&pg, &mongo, &id)
                .await
                .unwrap()
                .unwrap()
                .status,
            WorkloadStatus::UserStopped
        ));
        pg.update_access_key_status(
            &id,
            &AccessKeyStatus::Expired,
        )
        .await
        .unwrap();
        assert_eq!(
            client
                .get("/api/list_projects")
                .header(bearer)
                .dispatch()
                .await
                .status(),
            Status::Unauthorized
        );
    }
}
