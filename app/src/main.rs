use std::collections::HashMap;
use std::env;
use std::fs;
use std::sync::{Arc, OnceLock};

#[macro_use]
extern crate rocket;
use rocket::fairing::AdHoc;
use rocket::fs::FileServer;
use rocket::futures::{SinkExt, StreamExt};
use rocket::tokio;
use rocket::tokio::sync::Mutex as AsyncMutex;
use rocket::tokio::sync::mpsc;
use rocket_ws as ws;
use chrono::Utc;
use uuid::Uuid;

use crate::commondef::{clientmessages, workermessages};
use crate::commondef::experiments::{
    TrainingExperiment, TrainingExperimentRun,
    TrainingRunStatus,
};
use crate::db::db_orm::PostgresOrm;
use crate::db::mongo_orm::MongoOrm;
use crate::workflows::testing::TestingWorkflow;
use crate::workflows::training::TrainingWorkflow;
use crate::workflows::{WorkflowMode, WorkflowRuntime};

pub mod commondef;
pub mod db;
pub mod mujoco;
pub mod plugin;
pub mod remote_workflows;
pub mod simulator;
pub mod usr;
pub mod workflows;

// ── Project/User constants ──────────────────
const PROJECT_ID: &str = "1";

// ── Globals ─────────────────────────────────

type RemoteWorkerSlot = Arc<
    AsyncMutex<
        Option<
            remote_workflows::training_controller
            ::TrainingWorkerController
        >
    >
>;

static REMOTE_WORKER_SLOT: OnceLock<RemoteWorkerSlot> =
    OnceLock::new();
static REMOTE_WORKER_SHUTDOWN_TASK_STARTED: OnceLock<()> =
    OnceLock::new();

struct ActiveContext {
    experiment_id: Option<String>,
    run_id: Option<String>,
}

static ACTIVE_CONTEXT: OnceLock<
    Arc<AsyncMutex<ActiveContext>>
> = OnceLock::new();

static POSTGRES_ORM: OnceLock<Arc<PostgresOrm>> =
    OnceLock::new();
static MONGO_ORM: OnceLock<Arc<MongoOrm>> =
    OnceLock::new();

// ── Global accessors ────────────────────────

fn remote_worker_slot() -> &'static RemoteWorkerSlot {
    REMOTE_WORKER_SLOT.get_or_init(|| {
        Arc::new(AsyncMutex::new(None))
    })
}

fn active_context()
-> &'static Arc<AsyncMutex<ActiveContext>> {
    ACTIVE_CONTEXT.get_or_init(|| {
        Arc::new(AsyncMutex::new(ActiveContext {
            experiment_id: None,
            run_id: None,
        }))
    })
}

fn postgres_orm() -> &'static Arc<PostgresOrm> {
    POSTGRES_ORM.get()
        .expect("PostgresOrm not initialised")
}

fn mongo_orm() -> &'static Arc<MongoOrm> {
    MONGO_ORM.get()
        .expect("MongoOrm not initialised")
}

// ── Remote worker helpers ────────────────────

fn ensure_remote_worker_shutdown_hook() {
    if REMOTE_WORKER_SHUTDOWN_TASK_STARTED
        .set(()).is_err()
    {
        return;
    }

    let slot = remote_worker_slot().clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let mut guard = slot.lock().await;
            if let Some(worker) = guard.as_mut() {
                let _ = worker
                    .terminate_instance()
                    .await;
            }
            *guard = None;
        }
    });
}

fn build_remote_training_worker_config()
-> Result<
    remote_workflows::training_controller
    ::TrainingWorkerControllerConfig,
    String,
> {
    let mut cfg =
        remote_workflows::training_controller
        ::TrainingWorkerControllerConfig::default();

    if let Ok(region) =
        env::var("TRAINING_WORKER_AWS_REGION")
    {
        if !region.is_empty() {
            cfg.aws_region = region;
        }
    }
    cfg.ami_id = env::var("TRAINING_WORKER_AMI_ID")
        .map_err(|_| {
            "Missing required env var \
             TRAINING_WORKER_AMI_ID".to_string()
        })?;
    cfg.worker_image_uri =
        env::var("TRAINING_WORKER_IMAGE_URI")
        .map_err(|_| {
            "Missing required env var \
             TRAINING_WORKER_IMAGE_URI".to_string()
        })?;
    cfg.worker_shared_secret =
        env::var("TRAINING_WORKER_SHARED_SECRET")
        .map_err(|_| {
            "Missing required env var \
             TRAINING_WORKER_SHARED_SECRET".to_string()
        })?;

    if let Ok(v) =
        env::var("TRAINING_WORKER_SUBNET_ID")
    {
        if !v.is_empty() {
            cfg.subnet_id = Some(v);
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_SECURITY_GROUP_IDS")
    {
        let values: Vec<String> = v
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect();
        if !values.is_empty() {
            cfg.security_group_ids = values;
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_KEY_NAME")
    {
        if !v.is_empty() {
            cfg.key_name = Some(v);
        }
    }
    if let Ok(v) = env::var(
        "TRAINING_WORKER_IAM_INSTANCE_PROFILE_ARN",
    ) {
        if !v.is_empty() {
            cfg.iam_instance_profile_arn = Some(v);
        }
    }
    if let Ok(v) = env::var(
        "TRAINING_WORKER_INSTANCE_NAME_TAG",
    ) {
        if !v.is_empty() {
            cfg.instance_name_tag = v;
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_SPOT_MAX_PRICE_USD")
    {
        if !v.is_empty() {
            cfg.spot_max_price_usd = Some(v);
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_CONTAINER_NAME")
    {
        if !v.is_empty() {
            cfg.worker_container_name = v;
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_PORT")
    {
        if let Ok(port) = v.parse::<u16>() {
            cfg.worker_port = port;
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_WS_PATH")
    {
        if !v.is_empty() {
            cfg.worker_ws_path = v;
        }
    }

    Ok(cfg)
}

fn default_training_job_controller()
-> crate::commondef::training::TrainingJobController
{
    use crate::commondef::training::*;
    TrainingJobController {
        training_method: TrainingMethod::PPO(
            PPOTrainingMethod {
                num_trajs: 0,
                num_steps_per_traj: 0,
                num_epochs: 0,
                num_updates: 0,
                epsilon: 0.0,
                gamma: 0.0,
                lambda: 0.0,
                reward_function: RewardSystem {
                    reward_function_code: String::new(),
                },
                control_model: ControlModel::FFN(
                    FFNModel {
                        num_hidden_layers: 0,
                        hidden_layer_dim: 0,
                        output_dim: 0,
                        input_dim: 0,
                    },
                ),
                value_model: ValueModel::FFN(
                    FFNModel {
                        num_hidden_layers: 0,
                        hidden_layer_dim: 0,
                        output_dim: 0,
                        input_dim: 0,
                    },
                ),
            },
        ),
        acceleration_mode:
            TrainingAccelerationMode::AllOnServer,
        compute_instance:
            ComputeInstance::AWSG4dn,
        checkpoint_frequency:
            CheckpointFrequency::PostEpoch(1),
    }
}

// ── Channel handler helpers ──────────────────

/// Sends the current active experiment and run
/// to a newly connected client.
async fn send_active_context(
    stream: &mut ws::stream::DuplexStream,
) {
    let ctx = active_context().lock().await;
    let exp_id = match ctx.experiment_id.clone() {
        Some(id) => id,
        None => return,
    };
    let pg = postgres_orm().clone();
    drop(ctx);

    let exp_opt = pg
        .get_experiment(&exp_id)
        .await
        .ok()
        .flatten();
    let exp_row = match exp_opt {
        Some(r) => r,
        None => return,
    };

    let mut exp_runs: Vec<TrainingExperimentRun> =
        vec![];
    for rid in &exp_row.training_run_ids {
        if let Ok(Some(r)) =
            pg.get_run(PROJECT_ID, &exp_id, rid).await
        {
            exp_runs.push(r);
        }
    }

    let experiment = exp_row.clone();
    let ctx2 = active_context().lock().await;
    let active_run =
        ctx2.run_id.as_ref().and_then(|rid| {
            exp_runs
                .iter()
                .find(|r| &r.id == rid)
                .cloned()
        });
    drop(ctx2);

    let msg =
        clientmessages::ClientMessage
        ::ActiveContextMessage(
            clientmessages::ActiveContextMessage {
                experiment,
                run: active_run,
            },
        );
    if let Ok(json) = serde_json::to_string(&msg) {
        let _ = stream
            .send(ws::Message::Text(json))
            .await;
    }
}

/// Handles LoadModelMessage: loads the workflow
/// runtime and sends the world body to the client.
async fn handle_load_model(
    stream: &mut ws::stream::DuplexStream,
    model_path: String,
    workflow_runtime: &mut Option<WorkflowRuntime>,
    loaded_model_path: &mut Option<String>,
) {
    match WorkflowRuntime::load(model_path.clone()) {
        Ok(runtime) => {
            println!("Loaded shared workflow runtime");
            match runtime.world_body() {
                Ok(world_body) => {
                    let msg =
                        clientmessages::ClientMessage
                        ::new_load_model_message(
                            "1".to_string(),
                            world_body,
                        );
                    let json =
                        serde_json::to_string(&msg)
                        .unwrap_or_else(|e| {
                            format!(
                                "{{\"error\":\"{}\"}}",
                                e
                            )
                        });
                    let _ = stream
                        .send(ws::Message::Text(json))
                        .await;
                }
                Err(e) => println!(
                    "Couldn't extract world body {:?}",
                    e
                ),
            }
            *workflow_runtime = Some(runtime);
            *loaded_model_path = Some(model_path);
        }
        Err(e) => println!(
            "Couldn't load workflow runtime {:?}",
            e
        ),
    }
}

/// Handles StartTrainingMessage for the local
/// AllOnServer + PPO path.
/// If no run is currently loaded but an experiment
/// is active, a new run is auto-created first.
async fn handle_start_training_local(
    stream: &mut ws::stream::DuplexStream,
    training_id: String,
    ppomethod: crate::commondef::training
        ::PPOTrainingMethod,
    runtime: &mut WorkflowRuntime,
    tx_train: mpsc::Sender<
        workermessages::TrainingTaskMessage
    >,
    pending_flush: Arc<AsyncMutex<Vec<(
        Vec<crate::commondef::training::TrainingMetrics>,
        Vec<String>,
    )>>>,
    training_tasks: &mut HashMap<
        String,
        tokio::task::JoinHandle<()>,
    >,
) {
    // Auto-create a run if none is loaded
    {
        let ctx = active_context().lock().await;
        let has_run = ctx.run_id.is_some();
        let exp_id = ctx.experiment_id.clone();
        drop(ctx);
        if !has_run {
            match exp_id {
                Some(exp_id) => {
                    handle_create_run(
                        clientmessages::CreateRunMessage {
                            training_experiment_id:
                                exp_id,
                        },
                        stream,
                    ).await;
                }
                None => {
                    println!(
                        "Cannot start training: \
                         no active experiment"
                    );
                    return;
                }
            }
        }
    }

    if let Err(e) =
        runtime.configure_model_for_training(&ppomethod)
    {
        println!(
            "Couldn't configure model {:?}", e
        );
        return;
    }
    runtime.set_mode(WorkflowMode::Training);
    let simulator_h = runtime.simulator_handle();
    let model_h = runtime.ppo_model_handle();
    let tx_tc = tx_train.clone();
    let id_c = training_id.clone();

    // Mark active run as Ongoing in Postgres
    {
        let ctx = active_context().lock().await;
        if let (Some(exp_id), Some(run_id)) = (
            ctx.experiment_id.clone(),
            ctx.run_id.clone(),
        ) {
            drop(ctx);
            let _ = postgres_orm()
                .update_run_status(
                    PROJECT_ID,
                    &exp_id,
                    &run_id,
                    &TrainingRunStatus::Ongoing,
                )
                .await;
        }
    }

    // Spawn periodic Mongo flush task
    let ctx_snap = {
        let ctx = active_context().lock().await;
        (
            ctx.experiment_id.clone(),
            ctx.run_id.clone(),
        )
    };
    if let (Some(exp_id), Some(run_id)) = ctx_snap {
        let logs_id = format!("logs-{}", run_id);
        let metrics_id =
            format!("metrics-{}", run_id);
        let flush_buf = pending_flush.clone();
        let mg = mongo_orm().clone();
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(
                    std::time::Duration::from_secs(30)
                );
            loop {
                interval.tick().await;
                let mut buf =
                    flush_buf.lock().await;
                if buf.is_empty() {
                    continue;
                }
                let drained: Vec<_> =
                    buf.drain(..).collect();
                drop(buf);
                for (metrics, logs) in drained {
                    if !metrics.is_empty() {
                        let _ = mg
                            .append_metrics(
                                &metrics_id,
                                &metrics,
                            )
                            .await;
                    }
                    if !logs.is_empty() {
                        let _ = mg
                            .append_logs(
                                &logs_id,
                                &logs,
                            )
                            .await;
                    }
                }
                let _ = exp_id.as_str();
            }
        });
    }

    // Spawn blocking training task
    let h = tokio::task::spawn_blocking(move || {
        let wf = TrainingWorkflow::new(
            id_c,
            ppomethod,
            simulator_h,
            model_h,
        );
        if let Ok(mut wf) = wf {
            wf.run_training_blocking(tx_tc);
        }
    });
    training_tasks.insert(training_id, h);
}

/// Handles StartTrainingMessage for the remote
/// worker path.
async fn handle_start_training_remote(
    stream: &mut ws::stream::DuplexStream,
    training_id: String,
    training_controller: crate::commondef::training
        ::TrainingJobController,
    loaded_model_path: &Option<String>,
    workflow_runtime: &mut Option<WorkflowRuntime>,
    tx_train: mpsc::Sender<
        workermessages::TrainingTaskMessage
    >,
    training_tasks: &mut HashMap<
        String,
        tokio::task::JoinHandle<()>,
    >,
) -> bool {
    // Auto-create a run if none is loaded
    {
        let ctx = active_context().lock().await;
        let has_run = ctx.run_id.is_some();
        let exp_id = ctx.experiment_id.clone();
        drop(ctx);
        if !has_run {
            match exp_id {
                Some(exp_id) => {
                    handle_create_run(
                        clientmessages::CreateRunMessage {
                            training_experiment_id:
                                exp_id,
                        },
                        stream,
                    ).await;
                }
                None => {
                    println!(
                        "Cannot start training: \
                         no active experiment"
                    );
                    return false;
                }
            }
        }
    }

    let model_path = match loaded_model_path {
        Some(p) => p.clone(),
        None => {
            println!(
                "LoadModelMessage must be sent first"
            );
            return false;
        }
    };
    let mjcf = match std::fs::read_to_string(
        &model_path
    ) {
        Ok(c) => c,
        Err(e) => {
            println!(
                "Couldn't read MJCF {:?}: {:?}",
                model_path, e,
            );
            return false;
        }
    };
    let remote_cfg = match
        build_remote_training_worker_config()
    {
        Ok(c) => c,
        Err(e) => {
            println!(
                "Couldn't build remote config: {:?}",
                e
            );
            return false;
        }
    };
    ensure_remote_worker_shutdown_hook();
    if let Some(rt) = workflow_runtime.as_mut() {
        rt.set_mode(WorkflowMode::Training);
    }
    let tx_tc = tx_train.clone();
    let id_c = training_id.clone();
    let worker_slot = remote_worker_slot().clone();
    let h = tokio::spawn(async move {
        let tx_updates = tx_tc.clone();
        let mut guard = worker_slot.lock().await;
        if guard.is_none() {
            match remote_workflows
                ::training_controller
                ::TrainingWorkerController
                ::initialize(remote_cfg).await
            {
                Ok(worker) => {
                    *guard = Some(worker);
                }
                Err(e) => {
                    println!(
                        "Couldn't init remote \
                         worker: {:?}",
                        e
                    );
                    let _ = tx_tc
                        .send(
                            workermessages
                            ::TrainingTaskMessage
                            ::StopTraining(
                                clientmessages
                                ::StopTrainingMessage {
                                    id: id_c.clone(),
                                },
                            ),
                        )
                        .await;
                    return;
                }
            }
        }
        let worker = guard
            .as_mut()
            .expect("worker should be initialised");
        let result = worker
            .run_training_job(
                id_c.clone(),
                training_controller,
                Some(model_path.clone()),
                mjcf,
                move |update| {
                    let _ = tx_updates.try_send(
                        workermessages
                        ::TrainingTaskMessage
                        ::TrainingUpdates(
                            clientmessages
                            ::TrainingUpdatesMessage {
                                id: update.id.clone(),
                                updates: update
                                    .updates
                                    .clone(),
                            },
                        ),
                    );
                },
            )
            .await;
        drop(guard);
        if let Err(e) = result {
            println!(
                "Remote training failed: {:?}", e
            );
        }
        let _ = tx_tc
            .send(
                workermessages::TrainingTaskMessage
                ::StopTraining(
                    clientmessages::StopTrainingMessage {
                        id: id_c.clone(),
                    },
                ),
            )
            .await;
    });
    training_tasks.insert(training_id, h);
    true
}

/// Handles StopTrainingMessage: removes the task,
/// marks the run Completed, and flushes Mongo.
async fn handle_stop_training(
    t: clientmessages::StopTrainingMessage,
    stream: &mut ws::stream::DuplexStream,
    workflow_runtime: &mut Option<WorkflowRuntime>,
    training_tasks: &mut HashMap<
        String,
        tokio::task::JoinHandle<()>,
    >,
    pending_flush: Arc<AsyncMutex<Vec<(
        Vec<crate::commondef::training::TrainingMetrics>,
        Vec<String>,
    )>>>,
) {
    println!("Stopping training {:?}", t);
    training_tasks.remove(&t.id);
    if let Some(rt) = workflow_runtime.as_mut() {
        rt.set_mode(WorkflowMode::Idle);
    }

    // Mark run Completed in Postgres
    {
        let ctx = active_context().lock().await;
        if let (Some(exp_id), Some(run_id)) = (
            ctx.experiment_id.clone(),
            ctx.run_id.clone(),
        ) {
            drop(ctx);
            let _ = postgres_orm()
                .update_run_status(
                    PROJECT_ID,
                    &exp_id,
                    &run_id,
                    &TrainingRunStatus::Completed,
                )
                .await;
        }
    }

    // Final Mongo flush
    {
        let ctx = active_context().lock().await;
        if let (Some(_exp_id), Some(run_id)) = (
            ctx.experiment_id.clone(),
            ctx.run_id.clone(),
        ) {
            drop(ctx);
            let logs_id =
                format!("logs-{}", run_id);
            let metrics_id =
                format!("metrics-{}", run_id);
            let mg = mongo_orm().clone();
            let mut buf = pending_flush.lock().await;
            let drained: Vec<_> =
                buf.drain(..).collect();
            drop(buf);
            for (metrics, logs) in drained {
                if !metrics.is_empty() {
                    let _ = mg
                        .append_metrics(
                            &metrics_id, &metrics,
                        )
                        .await;
                }
                if !logs.is_empty() {
                    let _ = mg
                        .append_logs(&logs_id, &logs)
                        .await;
                }
            }
        }
    }

    let msg = clientmessages::ClientMessage
        ::StopTrainingMessage(t);
    let json = serde_json::to_string(&msg)
        .unwrap_or_else(|e| {
            format!("{{\"error\":\"{}\"}}", e)
        });
    let _ = stream
        .send(ws::Message::Text(json))
        .await;
}

/// Handles StartSimulationMessage: starts a
/// testing workflow task.
async fn handle_start_simulation(
    t: clientmessages::StartSimulationMessage,
    workflow_runtime: &mut Option<WorkflowRuntime>,
    tx_test: mpsc::Sender<
        workermessages::TestingTaskMessage
    >,
    testing_tasks: &mut HashMap<
        String,
        tokio::task::JoinHandle<()>,
    >,
) -> bool {
    println!("Start simulation {:?}", t.id);
    if workflow_runtime.is_none() {
        println!(
            "LoadModelMessage must be sent first"
        );
        return false;
    }
    let runtime = workflow_runtime
        .as_mut()
        .expect("Runtime should be available");
    runtime.set_mode(WorkflowMode::Testing);
    let simulator_h = runtime.simulator_handle();
    let model_h = runtime.control_model_handle();
    let tx_sc = tx_test.clone();
    let params = t.simulation_params;
    let id_c = t.id.clone();
    let h = tokio::task::spawn_blocking(move || {
        let mut wf = TestingWorkflow::new(
            id_c, simulator_h, model_h, params,
        );
        wf.run_test(tx_sc);
    });
    testing_tasks.insert(t.id, h);
    true
}

/// Handles StopSimulationMessage.
async fn handle_stop_simulation(
    t: clientmessages::StopSimulationMessage,
    stream: &mut ws::stream::DuplexStream,
    workflow_runtime: &mut Option<WorkflowRuntime>,
    testing_tasks: &mut HashMap<
        String,
        tokio::task::JoinHandle<()>,
    >,
) {
    println!("Stop simulation {:?}", t.id);
    testing_tasks.remove(&t.id);
    if let Some(rt) = workflow_runtime.as_mut() {
        rt.set_mode(WorkflowMode::Idle);
    }
    let msg = clientmessages::ClientMessage
        ::StopSimulationMessage(t);
    let json = serde_json::to_string(&msg)
        .unwrap_or_else(|e| {
            format!("{{\"error\":\"{}\"}}", e)
        });
    let _ = stream
        .send(ws::Message::Text(json))
        .await;
}

/// Handles CreateExperimentMessage: creates a new
/// experiment row and updates the active context.
async fn handle_create_experiment(
    stream: &mut ws::stream::DuplexStream,
) {
    let new_id = Uuid::new_v4().to_string();
    let pg = postgres_orm().clone();
    match pg.create_experiment(
        PROJECT_ID, &new_id, &new_id, &[],
    ).await {
        Ok(row) => {
            {
                let mut ctx =
                    active_context().lock().await;
                ctx.experiment_id = Some(
                    row.training_experiment_id.clone()
                );
                ctx.run_id = None;
            }
            let msg =
                clientmessages::ClientMessage
                ::ExperimentCreatedMessage(
                    clientmessages::ExperimentCreatedMessage {
                        experiment: row,
                    },
                );
            if let Ok(json) =
                serde_json::to_string(&msg)
            {
                let _ = stream
                    .send(ws::Message::Text(json))
                    .await;
            }
        }
        Err(e) => println!(
            "create_experiment failed: {:?}", e
        ),
    }
}

/// Handles CreateRunMessage: creates a new run row,
/// appends it to the experiment, and updates the
/// active context.
async fn handle_create_run(
    m: clientmessages::CreateRunMessage,
    stream: &mut ws::stream::DuplexStream,
) {
    let run_id = Uuid::new_v4().to_string();
    let logs_id = format!("logs-{}", run_id);
    let metrics_id = format!("metrics-{}", run_id);
    let graph_id = format!("graph-{}", run_id);
    let exp_id = m.training_experiment_id.clone();
    let pg = postgres_orm().clone();
    match pg.create_run(
        PROJECT_ID,
        &exp_id,
        &run_id,
        &TrainingRunStatus::NotStarted,
        &logs_id,
        &metrics_id,
        &graph_id,
        &[],
        &[],
    ).await {
        Ok(row) => {
            let _ = pg
                .add_run_to_experiment(&exp_id, &run_id)
                .await;
            {
                let mut ctx =
                    active_context().lock().await;
                ctx.experiment_id =
                    Some(exp_id.clone());
                ctx.run_id = Some(run_id.clone());
            }
            let msg =
                clientmessages::ClientMessage
                ::RunCreatedMessage(
                    clientmessages::RunCreatedMessage {
                        run: row,
                    },
                );
            if let Ok(json) =
                serde_json::to_string(&msg)
            {
                let _ = stream
                    .send(ws::Message::Text(json))
                    .await;
            }
        }
        Err(e) => println!(
            "create_run failed: {:?}", e
        ),
    }
}

/// Handles LoadRunMessage: fetches the run and
/// updates the active context.
async fn handle_load_run(
    m: clientmessages::LoadRunMessage,
    stream: &mut ws::stream::DuplexStream,
) {
    let pg = postgres_orm().clone();
    match pg.get_run(
        PROJECT_ID,
        &m.training_experiment_id,
        &m.run_id,
    ).await {
        Ok(Some(row)) => {
            {
                let mut ctx =
                    active_context().lock().await;
                ctx.experiment_id = Some(
                    m.training_experiment_id.clone()
                );
                ctx.run_id = Some(m.run_id.clone());
            }
            let msg =
                clientmessages::ClientMessage
                ::RunLoadedMessage(
                    clientmessages::RunLoadedMessage {
                        run: row,
                    },
                );
            if let Ok(json) =
                serde_json::to_string(&msg)
            {
                let _ = stream
                    .send(ws::Message::Text(json))
                    .await;
            }
        }
        Ok(None) => println!(
            "Run not found: {}", m.run_id
        ),
        Err(e) => println!(
            "get_run failed: {:?}", e
        ),
    }
}

/// Handles messages arriving from the simulator
/// thread channel.
async fn handle_simulator_msg(
    msg: workermessages::SimulatorThreadMessage,
    stream: &mut ws::stream::DuplexStream,
) {
    match msg {
        workermessages::SimulatorThreadMessage
        ::LoadModel(m) => {
            let client_msg =
                clientmessages::ClientMessage
                ::new_load_model_message(
                    "1".to_string(), m.model,
                );
            let json =
                serde_json::to_string(&client_msg)
                .unwrap_or_else(|e| {
                    format!(
                        "{{\"error\":\"{}\"}}",
                        e
                    )
                });
            let _ = stream
                .send(ws::Message::Text(json))
                .await;
        }
        workermessages::SimulatorThreadMessage
        ::SimulUpdate(m) => {
            let binary =
                clientmessages::ClientMessage
                ::to_binary_simul_update(
                    "1".to_string(), m.update,
                );
            let _ = stream
                .send(ws::Message::Binary(binary))
                .await;
        }
    }
}

/// Handles messages arriving from the training
/// task channel.
async fn handle_training_msg(
    msg: workermessages::TrainingTaskMessage,
    stream: &mut ws::stream::DuplexStream,
    workflow_runtime: &mut Option<WorkflowRuntime>,
    training_tasks: &mut HashMap<
        String,
        tokio::task::JoinHandle<()>,
    >,
    pending_flush: Arc<AsyncMutex<Vec<(
        Vec<crate::commondef::training::TrainingMetrics>,
        Vec<String>,
    )>>>,
) {
    println!(
        "Sending training updates to client {:?}",
        msg
    );
    match msg {
        workermessages::TrainingTaskMessage
        ::TrainingUpdates(m) => {
            {
                let mut buf =
                    pending_flush.lock().await;
                for upd in &m.updates {
                    buf.push((
                        upd.metrics.clone(),
                        upd.logs.clone(),
                    ));
                }
            }
            let client_msg =
                clientmessages::ClientMessage
                ::TrainingUpdatesMessage(m);
            let json =
                serde_json::to_string(&client_msg)
                .unwrap_or_else(|e| {
                    format!(
                        "{{\"error\":\"{}\"}}",
                        e
                    )
                });
            let _ = stream
                .send(ws::Message::Text(json))
                .await;
        }
        workermessages::TrainingTaskMessage
        ::StopTraining(m) => {
            training_tasks.remove(&m.id);
            if let Some(rt) =
                workflow_runtime.as_mut()
            {
                rt.set_mode(WorkflowMode::Idle);
            }
            let client_msg =
                clientmessages::ClientMessage
                ::StopTrainingMessage(m);
            let json =
                serde_json::to_string(&client_msg)
                .unwrap_or_else(|e| {
                    format!(
                        "{{\"error\":\"{}\"}}",
                        e
                    )
                });
            let _ = stream
                .send(ws::Message::Text(json))
                .await;
        }
    }
}

/// Handles messages arriving from the testing
/// task channel.
async fn handle_testing_msg(
    msg: workermessages::TestingTaskMessage,
    stream: &mut ws::stream::DuplexStream,
    workflow_runtime: &mut Option<WorkflowRuntime>,
    testing_tasks: &mut HashMap<
        String,
        tokio::task::JoinHandle<()>,
    >,
) {
    println!(
        "Sending testing updates to client {:?}",
        msg
    );
    match msg {
        workermessages::TestingTaskMessage
        ::StartSimulation(_) => {}
        workermessages::TestingTaskMessage
        ::StopSimulation(m) => {
            testing_tasks.remove(&m.id);
            if let Some(rt) =
                workflow_runtime.as_mut()
            {
                rt.set_mode(WorkflowMode::Idle);
            }
            let client_msg =
                clientmessages::ClientMessage
                ::StopSimulationMessage(m);
            let json =
                serde_json::to_string(&client_msg)
                .unwrap_or_else(|e| {
                    format!(
                        "{{\"error\":\"{}\"}}",
                        e
                    )
                });
            let _ = stream
                .send(ws::Message::Text(json))
                .await;
        }
        workermessages::TestingTaskMessage
        ::SimulationUpdates(m) => {
            let client_msg =
                clientmessages::ClientMessage
                ::SimulUpdateMessage(m);
            let json =
                serde_json::to_string(&client_msg)
                .unwrap_or_else(|e| {
                    format!(
                        "{{\"error\":\"{}\"}}",
                        e
                    )
                });
            let _ = stream
                .send(ws::Message::Text(json))
                .await;
        }
    }
}

// ── API Routes ───────────────────────────────

#[get("/hello")]
fn api_hello() -> &'static str {
    "Hello from Rocket!"
}

#[get("/workloads")]
fn workloads_channel(
    ws: ws::WebSocket,
) -> ws::Channel<'static> {
    let (tx_simul, mut rx_simul) =
        mpsc::channel::<
            workermessages::SimulatorThreadMessage
        >(32);
    let (tx_test, mut rx_test) =
        mpsc::channel::<
            workermessages::TestingTaskMessage
        >(32);
    let (tx_train, mut rx_train) =
        mpsc::channel::<
            workermessages::TrainingTaskMessage
        >(32);
    let mut training_tasks = HashMap::new();
    let mut testing_tasks = HashMap::new();
    let mut workflow_runtime: Option<WorkflowRuntime> =
        None;
    let mut loaded_model_path: Option<String> = None;
    let test_mjcf_model_path =
        env::var("TEST_MJCF_MODEL")
        .expect("Expected test mjcf model path \
                 to be set");

    // Shared buffer for periodic Mongo flush
    // during training.
    type PendingFlush = Vec<(
        Vec<crate::commondef::training::TrainingMetrics>,
        Vec<String>,
    )>;
    let pending_flush: Arc<AsyncMutex<PendingFlush>> =
        Arc::new(AsyncMutex::new(vec![]));

    ws.channel(move |mut stream| {
        Box::pin(async move {
            // ── On connect ───────────────────────
            send_active_context(&mut stream).await;

            // ── Main event loop ──────────────────
            loop {
                tokio::select! {
                    Some(message) = stream.next() => {
                        match message {
                            Ok(ws::Message::Text(text)) => {
                                // Legacy raw-text commands
                                if text == "get_test_model" {
                                    let world_body =
                                        simulator::get_test_model();
                                    let msg =
                                        clientmessages::ClientMessage
                                        ::new_set_model_message(
                                            world_body
                                        );
                                    let json =
                                        serde_json::to_string(&msg)
                                        .unwrap_or_else(|_| {
                                            "{}".to_string()
                                        });
                                    let _ = stream.send(
                                        ws::Message::Text(json)
                                    ).await;
                                } else if text == "get_test_model_2" {
                                    let world_body;
                                    unsafe {
                                        world_body =
                                            simulator::load_test_model(
                                                test_mjcf_model_path
                                                .clone()
                                            );
                                    }
                                    if let Some(wb) = world_body {
                                        let msg =
                                            clientmessages::ClientMessage
                                            ::new_set_model_message(wb);
                                        let json =
                                            serde_json::to_string(&msg)
                                            .unwrap_or_else(|e| {
                                                format!(
                                                    "{{\"error\":\"{}\"}}",
                                                    e
                                                )
                                            });
                                        if let Err(e) = fs::write(
                                            "target/test_model_2.json",
                                            json.clone(),
                                        ) {
                                            println!(
                                                "Could not write test \
                                                 model 2 JSON {:?}",
                                                e
                                            );
                                        }
                                        let _ = stream.send(
                                            ws::Message::Text(json)
                                        ).await;
                                    }
                                } else if text == "get_test_model_3" {
                                    if let Ok(json) = fs::read_to_string(
                                        "target/test_model_3.json"
                                    ) {
                                        let _ = stream.send(
                                            ws::Message::Text(json)
                                        ).await;
                                    }
                                } else if text == "run_simul" {
                                    simulator::spawn_simulator(
                                        test_mjcf_model_path.clone(),
                                        tx_simul.clone(),
                                    );
                                } else {
                                    // Typed JSON messages
                                    match serde_json::from_str::<
                                        clientmessages::ServerMessage
                                    >(&text) {
                                        Ok(m) => match m {
                                            clientmessages::ServerMessage
                                            ::LoadModelMessage => {
                                                handle_load_model(
                                                    &mut stream,
                                                    test_mjcf_model_path
                                                        .clone(),
                                                    &mut workflow_runtime,
                                                    &mut loaded_model_path,
                                                ).await;
                                            }
                                            clientmessages::ServerMessage
                                            ::StartTrainingMessage(t) => {
                                                println!(
                                                    "Start training ask \
                                                     from client {:?}",
                                                    t
                                                );
                                                let training_id =
                                                    t.id.clone();
                                                let controller =
                                                    t.training_graph
                                                    .training_job_controller;
                                                match (
                                                    &controller
                                                        .acceleration_mode,
                                                    &controller
                                                        .training_method,
                                                ) {
                                                    (
                                                        commondef::training
                                                        ::TrainingAccelerationMode
                                                        ::AllOnServer,
                                                        commondef::training
                                                        ::TrainingMethod
                                                        ::PPO(ppo),
                                                    ) => {
                                                        if workflow_runtime
                                                            .is_none()
                                                        {
                                                            println!(
                                                                "LoadModelMessage \
                                                                 must be sent first"
                                                            );
                                                            continue;
                                                        }
                                                        let ppo = ppo.clone();
                                                        let runtime =
                                                            workflow_runtime
                                                            .as_mut()
                                                            .expect(
                                                                "Runtime should \
                                                                 be available"
                                                            );
                                                        handle_start_training_local(
                                                            &mut stream,
                                                            training_id,
                                                            ppo,
                                                            runtime,
                                                            tx_train.clone(),
                                                            pending_flush.clone(),
                                                            &mut training_tasks,
                                                        ).await;
                                                    }
                                                    (_, _) => {
                                                        handle_start_training_remote(
                                                            &mut stream,
                                                            training_id,
                                                            controller,
                                                            &loaded_model_path,
                                                            &mut workflow_runtime,
                                                            tx_train.clone(),
                                                            &mut training_tasks,
                                                        ).await;
                                                    }
                                                }
                                            }
                                            clientmessages::ServerMessage
                                            ::StopTrainingMessage(t) => {
                                                handle_stop_training(
                                                    t,
                                                    &mut stream,
                                                    &mut workflow_runtime,
                                                    &mut training_tasks,
                                                    pending_flush.clone(),
                                                ).await;
                                            }
                                            clientmessages::ServerMessage
                                            ::StartSimulationMessage(t) => {
                                                handle_start_simulation(
                                                    t,
                                                    &mut workflow_runtime,
                                                    tx_test.clone(),
                                                    &mut testing_tasks,
                                                ).await;
                                            }
                                            clientmessages::ServerMessage
                                            ::StopSimulationMessage(t) => {
                                                handle_stop_simulation(
                                                    t,
                                                    &mut stream,
                                                    &mut workflow_runtime,
                                                    &mut testing_tasks,
                                                ).await;
                                            }
                                            clientmessages::ServerMessage
                                            ::CreateExperimentMessage => {
                                                handle_create_experiment(
                                                    &mut stream,
                                                ).await;
                                            }
                                            clientmessages::ServerMessage
                                            ::CreateRunMessage(m) => {
                                                handle_create_run(
                                                    m,
                                                    &mut stream,
                                                ).await;
                                            }
                                            clientmessages::ServerMessage
                                            ::LoadRunMessage(m) => {
                                                handle_load_run(
                                                    m,
                                                    &mut stream,
                                                ).await;
                                            }
                                        },
                                        Err(e) => println!(
                                            "Couldn't deserialise msg \
                                             from client {:?}",
                                            e
                                        ),
                                    }
                                }
                            }
                            Ok(ws::Message::Binary(bin)) => {
                                let _ = stream.send(
                                    ws::Message::Binary(bin)
                                ).await;
                            }
                            Ok(ws::Message::Close(_)) => {
                                println!("Client disconnected");
                                break;
                            }
                            Err(e) => {
                                eprintln!(
                                    "WebSocket error: {:?}",
                                    e
                                );
                                break;
                            }
                            _ => {}
                        }
                    }
                    Some(msg) = rx_simul.recv() => {
                        handle_simulator_msg(
                            msg, &mut stream,
                        ).await;
                    }
                    Some(msg) = rx_train.recv() => {
                        handle_training_msg(
                            msg,
                            &mut stream,
                            &mut workflow_runtime,
                            &mut training_tasks,
                            pending_flush.clone(),
                        ).await;
                    }
                    Some(msg) = rx_test.recv() => {
                        handle_testing_msg(
                            msg,
                            &mut stream,
                            &mut workflow_runtime,
                            &mut testing_tasks,
                        ).await;
                    }
                    else => break
                }
            }
            Ok(())
        })
    })
}

#[launch]
fn rocket() -> _ {
    let client_dist_dir =
        env::var("CLIENT_DIST_DIR")
        .expect("Expected CLIENT_DIST_DIR to be set");

    // Ensure ActiveContext global is ready
    let _ = active_context();

    rocket::build()
        .attach(AdHoc::on_liftoff(
            "DB Init",
            |_rocket| Box::pin(async move {
                // Postgres
                match env::var("DATABASE_URL") {
                    Ok(url) => {
                        match PostgresOrm::new(&url).await {
                            Ok(orm) => {
                                let _ = POSTGRES_ORM
                                    .set(Arc::new(orm));
                                println!(
                                    "PostgresOrm initialised"
                                );
                            }
                            Err(e) => println!(
                                "WARNING: PostgresOrm \
                                 init failed: {:?}", e
                            ),
                        }
                    }
                    Err(_) => println!(
                        "WARNING: DATABASE_URL not set; \
                         PostgresOrm unavailable"
                    ),
                }
                // MongoDB
                match env::var("MONGODB_URI") {
                    Ok(uri) => {
                        match MongoOrm::new(&uri).await {
                            Ok(orm) => {
                                let _ = MONGO_ORM
                                    .set(Arc::new(orm));
                                println!(
                                    "MongoOrm initialised"
                                );
                            }
                            Err(e) => println!(
                                "WARNING: MongoOrm \
                                 init failed: {:?}", e
                            ),
                        }
                    }
                    Err(_) => println!(
                        "WARNING: MONGODB_URI not set; \
                         MongoOrm unavailable"
                    ),
                }
            }),
        ))
        .mount("/api", routes![api_hello, workloads_channel])
        .mount(
            "/",
            FileServer::from(client_dist_dir).rank(1),
        )
}
