//! Workload lifecycle, update ingestion, replay and subscriptions.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use rocket::State;
use rocket::http::Status;
use rocket::serde::json::Json;
use rocket_ws as ws;
use tokio::sync::broadcast;
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
use crate::db;
#[cfg(test)]
use crate::db::validations;
use crate::workflows;
#[cfg(test)]
use crate::workflows::{final_record_id, ingest_update};

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
    let _delivery = state.catalog.lock().await;
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
    let task =
        jobs.get(&body.workload_id).ok_or(Status::Conflict)?;
    task.user_stopped.store(true, Ordering::SeqCst);
    task.stop.send(true).map_err(|_| Status::Conflict)?;
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
    workflows::launch_workload(
        state,
        workload,
        models,
        checkpoints,
    )
    .await?;
    Ok(id)
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

/// Read a bounded, independently ordered page with fresh links.
/// A 409 means concurrent ingestion invalidated the cursor.
#[utoipa::path(get, path = "/api/get_workload_updates",
    params(
        ("workload_id" = String, Query, description = "Workload ID"),
        ("cursor" = Option<String>, Query, description = "Opaque next-page cursor"),
        ("limit" = Option<usize>, Query, description = "Records per page, 1 to 500; default 200")
    ),
    responses((status = 200, body = WorkloadUpdatesResponseBody),
        (status = 409, description = "Start a fresh history snapshot")))]
#[get("/get_workload_updates?<workload_id>&<cursor>&<limit>")]
pub async fn get_workload_updates(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    workload_id: String,
    cursor: Option<String>,
    limit: Option<usize>,
) -> ApiResult<Json<WorkloadUpdatesResponseBody>> {
    let limit = validate_history_limit(limit)?;
    let _guard = state.lock_workload(&workload_id).await;
    let workload =
        state.authorize_workload(&auth, &workload_id).await?;
    let mut cursor = if let Some(value) = cursor {
        workflows::HistoryCursor::decode(&value)
            .map_err(|_| Status::BadRequest)?
    } else {
        workflows::HistoryCursor::new(state, &workload_id).await
    };
    cursor.validate(state, &workload_id).await?;
    let update = workflows::history_chunk(
        state,
        &workload,
        &mut cursor,
        limit,
    )
    .await
    .map_err(internal)?;
    let updates = if has_records(&update) {
        vec![user_update(state, &auth, update).await?]
    } else {
        Vec::new()
    };
    let next_cursor = if cursor.stream < 6 {
        Some(cursor.encode().map_err(internal)?)
    } else {
        None
    };
    Ok(Json(WorkloadUpdatesResponseBody {
        workload_id,
        updates,
        next_cursor,
    }))
}

fn validate_history_limit(
    limit: Option<usize>,
) -> ApiResult<usize> {
    let limit = limit.unwrap_or(200);
    if !(1..=500).contains(&limit) {
        return Err(Status::BadRequest);
    }
    Ok(limit)
}

// Tests can assemble their small fixtures; production never
// collects all historical records into one response.
#[cfg(test)]
async fn history(
    state: &ApiState,
    workload: &Workload,
) -> ApiResult<WorkloadUpdates> {
    let mut cursor = workflows::HistoryCursor::new(
        state,
        &workload.workload_id,
    )
    .await;
    let mut all =
        workflows::empty_workload_update(&workload.workload_id);
    while cursor.stream < 6 {
        let page = workflows::history_chunk(
            state,
            workload,
            &mut cursor,
            200,
        )
        .await
        .map_err(internal)?;
        all.logs.extend(page.logs);
        all.metrics.extend(page.metrics);
        all.checkpoints.extend(page.checkpoints);
        all.camera_frames.extend(page.camera_frames);
        all.simulation_models.extend(page.simulation_models);
        all.model_states.extend(page.model_states);
    }
    Ok(all)
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
        let (mut revision, stopped) = replay(
            &mut socket, &state, &auth, &workload_id, &mut seen,
        ).await?;
        if stopped {
            socket.close(None).await?;
            return Ok(());
        }
        loop {
            tokio::select! {
                incoming = socket.next() => match incoming {
                    Some(Ok(ws::Message::Text(text))) => {
                        match serde_json::from_str::<WorkloadUpdatesClient>(&text) {
                            Ok(WorkloadUpdatesClient::RequestAllUpdatesMessage) => {
                                seen.clear();
                                let result = replay(&mut socket, &state, &auth,
                                    &workload_id, &mut seen).await?;
                                revision = result.0;
                                if result.1 { break; }
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
                        Ok((version, RemoteWorkerClientMessage::TrainingUpdatesMessage(mut u)))
                        | Ok((version, RemoteWorkerClientMessage::TestUpdatesMessage(mut u)))
                            if u.workload_id == workload_id && version > revision => {
                            retain_unseen(&mut u, &mut seen);
                            while has_records(&u) {
                                let chunk = workflows::take_update_chunk(&mut u, 200);
                                let chunk = user_update(&state, &auth, chunk).await
                                    .map_err(|_| ws::result::Error::ConnectionClosed)?;
                                send_message(&mut socket,
                                    WorkloadUpdatesServer::WorkloadUpdatesMessage(vec![chunk]),
                                ).await?;
                            }
                            revision = version;
                        }
                        Ok((version, RemoteWorkerClientMessage::StopTrainingMessage(s)))
                        | Ok((version, RemoteWorkerClientMessage::StopTestingMessage(s)))
                            if s.workload_id == workload_id && version > revision => {
                            send_message(&mut socket, stopped_response(s)).await?;
                            break;
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            let result = replay(&mut socket, &state, &auth,
                                &workload_id, &mut seen).await?;
                            revision = result.0;
                            if result.1 { break; }
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                        _ => {}
                    }
                }
            }
        }
        socket.close(None).await?;
        Ok(())
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

async fn send_message<S>(
    socket: &mut S,
    message: WorkloadUpdatesServer,
) -> ws::result::Result<()>
where
    S: futures_util::Sink<ws::Message, Error = ws::result::Error>
        + Unpin,
{
    let text = serde_json::to_string(&message)
        .map_err(|_| ws::result::Error::ConnectionClosed)?;
    socket.send(ws::Message::Text(text)).await
}

async fn replay<S>(
    socket: &mut S,
    state: &ApiState,
    auth: &Authenticated,
    id: &str,
    seen: &mut HashSet<String>,
) -> ws::result::Result<(u64, bool)>
where
    S: futures_util::Sink<ws::Message, Error = ws::result::Error>
        + Unpin,
{
    let snapshot = async {
        let _guard = state.lock_workload(id).await;
        let workload =
            state.authorize_workload(auth, id).await.map_err(
                |_| ws::result::Error::ConnectionClosed,
            )?;
        let mut cursor =
            workflows::HistoryCursor::new(state, id).await;
        while cursor.stream < 6 {
            let mut update = workflows::history_chunk(
                state,
                &workload,
                &mut cursor,
                200,
            )
            .await
            .map_err(|_| ws::result::Error::ConnectionClosed)?;
            retain_unseen(&mut update, seen);
            if has_records(&update) {
                let update = user_update(state, auth, update)
                    .await
                    .map_err(|_| {
                        ws::result::Error::ConnectionClosed
                    })?;
                send_message(
                    socket,
                    WorkloadUpdatesServer::WorkloadUpdatesMessage(
                        vec![update],
                    ),
                )
                .await?;
            }
        }
        let reason = match workload.status {
            WorkloadStatus::Completed => {
                Some(WorkloadStopReasonWorker::CompletedSuccess)
            }
            WorkloadStatus::UserStopped => Some(
                WorkloadStopReasonWorker::RequestedByAppServer,
            ),
            WorkloadStatus::LaunchFailed
            | WorkloadStatus::ExecutionFailed => {
                Some(WorkloadStopReasonWorker::WorkerFailure(
                    "Workload failed; see its logs".into(),
                ))
            }
            _ => None,
        };
        let stopped = reason.is_some();
        if let Some(reason) = reason {
            send_message(
                socket,
                stopped_response(StopWorkloadWorker {
                    workload_id: id.to_owned(),
                    reason,
                }),
            )
            .await?;
        }
        Ok((cursor.revision, stopped))
    };
    // Slow clients cannot hold up this workload's ingestion forever.
    tokio::time::timeout(Duration::from_secs(30), snapshot)
        .await
        .map_err(|_| ws::result::Error::ConnectionClosed)?
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
pub(crate) mod tests {
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
        let (stop, _) = tokio::sync::watch::channel(false);
        let (_, finished) = tokio::sync::watch::channel(false);
        jobs.insert(
            "w".into(),
            workflows::WorkloadTaskState {
                remote: false,
                stop,
                finished,
                user_stopped: Arc::new(
                    std::sync::atomic::AtomicBool::new(false),
                ),
                handle: None,
            },
        );
        assert!(
            workflows::validate_capacity(&jobs, false).is_err()
        );
        assert!(
            workflows::validate_capacity(&jobs, true).is_ok()
        );
    }

    pub(crate) async fn fixture()
    -> (Arc<ApiState>, Authenticated, Workload) {
        use crate::commondef::records::product::*;
        use crate::db::db_orm::PostgresOrm;
        use crate::db::mongo_orm::MongoOrm;
        use aws_sdk_s3::config::retry::RetryConfig;
        use aws_sdk_s3::config::{
            BehaviorVersion, Credentials, Region,
        };

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
        (state, auth, workload)
    }

    /// Uses disposable DBs and a local S3-compatible stub, never AWS.
    #[tokio::test]
    #[ignore = "requires API_TEST_DATABASE_URL, API_TEST_MONGODB_URI and API_TEST_S3_ENDPOINT"]
    async fn persistence_auth_and_asset_downloads() {
        use crate::commondef::records::product::AccessKeyStatus;
        use rocket::http::Header;
        use rocket::local::asynchronous::Client;
        let (state, auth, workload) = fixture().await;
        let pg = state.postgres.clone();
        let mongo = state.mongo.clone();
        let id = workload.workload_id.clone();
        let now = workload.created_at;
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
        let frame_source = format!(
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
        assert!(
            raw.camera_frames[0]
                .uri_path
                .starts_with(&frame_prefix)
        );
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
    #[tokio::test]
    #[ignore = "requires disposable DBs and local S3 stub"]
    async fn history_is_chunked_and_invalidated_by_new_updates() {
        use rocket::futures::StreamExt;
        use rocket::http::Header;
        use rocket::local::asynchronous::Client;
        let (state, auth, workload) = fixture().await;
        let id = workload.workload_id.clone();
        let now = workload.created_at;
        let mut update = empty_update(&id);
        for index in (0..405).rev() {
            update.logs.push(WorkloadLogs {
                workload_id: id.clone(),
                created_at: now,
                log_timestamp: now
                    + chrono::Duration::milliseconds(index),
                log_id: format!("log-{index}"),
                log_line_text: index.to_string(),
                user_facing: true,
                log_type: LogType::Info,
                stream: LogStream::Worker,
            });
        }
        ingest_update(&state, &workload, update).await.unwrap();
        let client = Client::tracked(
            rocket::build()
                .manage(state.clone())
                .mount("/api", routes![get_workload_updates]),
        )
        .await
        .unwrap();
        let bearer =
            Header::new("Authorization", format!("Bearer {id}"));
        let mut cursor = None;
        let mut received = Vec::new();
        loop {
            let mut url = format!(
                "/api/get_workload_updates?workload_id={id}&limit=37"
            );
            if let Some(token) = &cursor {
                url.push_str(&format!("&cursor={token}"));
            }
            let response = client
                .get(url)
                .header(bearer.clone())
                .dispatch()
                .await;
            assert_eq!(response.status(), Status::Ok);
            let page: WorkloadUpdatesResponseBody =
                response.into_json().await.unwrap();
            let mut count = 0;
            for update in page.updates {
                count += update.logs.len();
                received.extend(
                    update
                        .logs
                        .into_iter()
                        .map(|log| log.log_line_text),
                );
            }
            assert!(count <= 37);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(
            received,
            (0..405).map(|i| i.to_string()).collect::<Vec<_>>()
        );
        let first = client.get(format!("/api/get_workload_updates?workload_id={id}&limit=1"))
            .header(bearer.clone()).dispatch().await
            .into_json::<WorkloadUpdatesResponseBody>().await.unwrap();
        let stale = first.next_cursor.unwrap();
        let mut update = empty_update(&id);
        update.logs.push(WorkloadLogs {
            workload_id: id.clone(),
            created_at: now,
            log_timestamp: now,
            log_id: "new-log".into(),
            log_line_text: "new".into(),
            user_facing: true,
            log_type: LogType::Info,
            stream: LogStream::Worker,
        });
        ingest_update(&state, &workload, update).await.unwrap();
        let response = client.get(format!("/api/get_workload_updates?workload_id={id}&cursor={stale}"))
            .header(bearer).dispatch().await;
        assert_eq!(response.status(), Status::Conflict);
        let (tx, mut rx) =
            rocket::futures::channel::mpsc::unbounded::<
                ws::Message,
            >();
        let mut sink = tx.sink_map_err(|_| {
            ws::result::Error::ConnectionClosed
        });
        let mut seen = HashSet::new();
        let (revision, stopped) =
            replay(&mut sink, &state, &auth, &id, &mut seen)
                .await
                .unwrap();
        assert_eq!(revision, state.revision(&id).await);
        assert!(!stopped);
        drop(sink);
        let mut total = 0;
        while let Some(ws::Message::Text(text)) = rx.next().await
        {
            let message: WorkloadUpdatesServer =
                serde_json::from_str(&text).unwrap();
            if let WorkloadUpdatesServer::WorkloadUpdatesMessage(
                updates,
            ) = message
            {
                let count = updates
                    .into_iter()
                    .map(|u| u.logs.len())
                    .sum::<usize>();
                assert!(count <= 200);
                total += count;
            }
        }
        assert_eq!(total, 406);
    }
}
