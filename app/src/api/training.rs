use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rocket::State;
use rocket::http::Status;
use rocket::serde::json::Json;
use rocket::tokio::sync::{broadcast, mpsc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use rocket_ws as ws;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD
    as BASE64;

use crate::api::{route_doc, ApiState};
use crate::commondef::clientmessages as cm;
use crate::commondef::record::TrainingRunStatus;

use crate::commondef::remoteworkermessages::WorkerTrainingUpdatesMessage;
use crate::commondef::training::{
    Checkpoint,
    CheckpointType,
    ControlModel,
    ModelType,
    RewardLanguage,
    StopTrainingMessage,
    TrainingGraph,
    TrainingJobController,
    TrainingMethod,
    TrainingUpdate,
    TrainingUpdatesMessage,
    ValueModel,
};
use crate::workflows::local_worker_controller::LocalWorkerController;

/// Internal task→relay message for the training mpsc (API-local; carries the
/// client-facing update with checkpoint refs already persisted).
#[derive(Debug, Clone)]
enum TrainingTaskMessage {
    TrainingUpdates(TrainingUpdatesMessage),
    StopTraining(StopTrainingMessage),
}

// ── Request / response types ─────────────────

#[derive(Deserialize, ToSchema)]
pub struct StartTrainingBody {
    pub training_graph: TrainingGraph,
}

#[derive(Serialize, ToSchema)]
pub struct TrainingStartedResponse {
    pub id: String,
}

#[derive(Serialize, ToSchema)]
pub struct TrainingStoppedResponse {
    pub id: String,
}

// ── SSE relay ────────────────────────────────

/// Reads from the training mpsc channel, flushes
/// metrics/logs to Mongo, and broadcasts JSON
/// events to SSE subscribers.
async fn run_training_relay(
    run_id: String,
    exp_id: String,
    mut rx: mpsc::Receiver<TrainingTaskMessage>,
    sse_tx: broadcast::Sender<String>,
    state: Arc<(
        tokio::sync::Mutex<
            std::collections::HashMap<
                String,
                broadcast::Sender<String>,
            >
        >,
    )>,
) {
    let logs_id = format!("logs-{}", run_id);
    let metrics_id = format!("metrics-{}", run_id);

    while let Some(msg) = rx.recv().await {
        match msg {
            TrainingTaskMessage::TrainingUpdates(m) =>
            {
                // Flush to Mongo immediately
                let mg = crate::mongo_orm().clone();
                for u in &m.updates {
                    if !u.metrics.is_empty() {
                        let _ = mg
                            .append_metrics(
                                &metrics_id,
                                &u.metrics,
                            )
                            .await;
                    }
                    if !u.logs.is_empty() {
                        let _ = mg
                            .append_logs(
                                &logs_id,
                                &u.logs,
                            )
                            .await;
                    }
                }
                let ev = cm::ClientMessage
                    ::TrainingUpdatesMessage(m);
                if let Ok(json) =
                    serde_json::to_string(&ev)
                {
                    sse_tx.send(json).ok();
                }
            }
            TrainingTaskMessage::StopTraining(m) => {
                let _ = crate::postgres_orm()
                    .update_run_status(
                        crate::PROJECT_ID,
                        &exp_id,
                        &run_id,
                        &TrainingRunStatus::Completed,
                    )
                    .await;
                let ev = cm::ClientMessage
                    ::StopTrainingMessage(m);
                if let Ok(json) =
                    serde_json::to_string(&ev)
                {
                    sse_tx.send(json).ok();
                }
                break;
            }
        }
    }
}

// ── Checkpoint helpers ────────────────────────

/// Resolves the serialised MLP JSON for a model
/// parameter (used when creating checkpoint DB
/// records on the remote training path).
async fn resolve_mlp_json<M: serde::Serialize>(
    model: Option<M>,
    src_id: Option<&str>,
) -> Option<String>
{
    if let Some(m) = model {
        return serde_json::to_string(&m).ok();
    }
    if let Some(id) = src_id {
        return crate::postgres_orm()
            .get_checkpoint_ref(id)
            .await
            .ok()
            .flatten()
            .and_then(|r| {
                r.control_model_params
                    .as_ref()
                    .and_then(|m| {
                        serde_json::to_string(m).ok()
                    })
            });
    }
    None
}

// ── Model-type validation ─────────────────────

fn validate_model_types(
    controller: &TrainingJobController,
) -> Result<(), &'static str> {
    let TrainingMethod::PPO(ref p) =
        controller.training_method;

    let control_spec = match &p.control_model {
        ControlModel::MLP(m) => &m.input_spec,
        ControlModel::VLA(m) => &m.input_spec,
    };
    if control_spec.contact_pairs {
        return Err(
            "contact_pairs is not allowed in control model input_spec",
        );
    }
    let ValueModel::MLP(ref vm) = p.value_model;
    if vm.input_spec.contact_pairs {
        return Err(
            "contact_pairs is not allowed in value model input_spec",
        );
    }

    match &p.control_model {
        ControlModel::MLP(m) => {
            if matches!(
                &m.model_type,
                ModelType::Checkpoint(
                    CheckpointType::Pretrained(_)
                )
            ) {
                return Err(
                    "Pretrained model type is not \
                     allowed for MLP control models",
                );
            }
        }
        ControlModel::VLA(m) => {
            match &m.model_type {
                ModelType::New => {
                    return Err(
                        "VLA model must specify a \
                         pretrained checkpoint, not New",
                    );
                }
                ModelType::Checkpoint(
                    CheckpointType::Checkpoint(_),
                ) => {
                    return Err(
                        "Checkpoint (non-pretrained) is \
                         not supported for VLA models; \
                         only Pretrained(Pi0) is valid",
                    );
                }
                _ => {}
            }
        }
    }

    if matches!(
        &vm.model_type,
        ModelType::Checkpoint(
            CheckpointType::Pretrained(_)
        )
    ) {
        return Err(
            "Pretrained model type is not allowed \
             for value models",
        );
    }

    Ok(())
}

// ── Handlers ─────────────────────────────────

#[route_doc(
    tag = "Training",
    ok = TrainingStartedResponse,
    body = StartTrainingBody,
    errors(400, 422),
    params(
        experiment_id = "Experiment UUID",
        run_id = "Run UUID",
    ),
)]
#[post(
    "/training/experiments\
     /<experiment_id>/runs/<run_id>/start",
    format = "json",
    data = "<body>"
)]
pub async fn start_training(
    experiment_id: String,
    run_id: String,
    body: Json<StartTrainingBody>,
    state: &State<Arc<ApiState>>,
) -> Result<Json<TrainingStartedResponse>, Status>
{
    let training_graph =
        body.into_inner().training_graph;
    let controller =
        training_graph.training_job_controller.clone();

    if let Err(e) = validate_model_types(&controller) {
        println!("start_training: invalid model types: {e}");
        return Err(Status::UnprocessableEntity);
    }

    let is_local = matches!(
        controller.acceleration_mode,
        crate::commondef::training
            ::TrainingAccelerationMode::AllOnServer,
    );

    // Mark run Ongoing
    let _ = crate::postgres_orm()
        .update_run_status(
            crate::PROJECT_ID,
            &experiment_id,
            &run_id,
            &TrainingRunStatus::Ongoing,
        )
        .await;

    // SSE broadcast channel
    let (sse_tx, _) =
        broadcast::channel::<String>(256);
    state
        .training_senders
        .lock()
        .await
        .insert(run_id.clone(), sse_tx.clone());

    // Internal mpsc for training workflow
    let (tx_train, rx_train) =
        mpsc::channel::<TrainingTaskMessage>(64);

    // Relay: mpsc → broadcast + Mongo
    let relay_sse_tx = sse_tx.clone();
    let relay_run_id = run_id.clone();
    let relay_exp_id = experiment_id.clone();
    // Pass a plain clone of the sender map so the
    // relay can remove its entry on exit.
    let relay_senders_guard = {
        // We just need a way to drop the entry; pass
        // a reference via a channel message instead
        // of Arc to avoid a dependency on ApiState.
        // Removal is done below via a cleanup task.
        ()
    };
    let _ = relay_senders_guard;

    // let run_id_cleanup = run_id.clone();
    // Spawn cleanup task that removes the sender
    // from the map once the relay finishes.
    // let training_senders_ptr =
    //     state.inner() as *const ApiState as usize;
    // let _ = training_senders_ptr;

    rocket::tokio::spawn(async move {
        run_training_relay(
            relay_run_id,
            relay_exp_id,
            rx_train,
            relay_sse_tx,
            Arc::new(Default::default()),
        )
        .await;
    });

    // Stop flag
    let stop_flag =
        Arc::new(AtomicBool::new(false));
    let stop_flag_c = stop_flag.clone();

    let sim_model_id = match &controller
        .training_method
    {
        TrainingMethod::PPO(p) =>
            p.simulation_model.clone(),
    };
    let model_ref = crate::postgres_orm()
        .get_model_reference(&sim_model_id)
        .await
        .map_err(|_| Status::InternalServerError)?
        .ok_or(Status::BadRequest)?;
    let model_path = model_ref.path.clone();

    let (reward_lang, reward_code) =
        match &controller.training_method {
            TrainingMethod::PPO(p) => (
                p.reward_function.language.clone(),
                p.reward_function
                    .reward_function_code
                    .clone(),
            ),
        };
    // Both local and remote run in the Python worker, so reward functions must
    // be Python and are validated as such; the spec is forwarded in the Start
    // message (training_graph) and executed inside the worker.
    if !matches!(reward_lang, RewardLanguage::Python) {
        return Err(Status::BadRequest);
    }
    {
        let code = reward_code.clone();
        rocket::tokio::task
            ::spawn_blocking(move || {
                crate::api::reward_validation
                    ::validate_python_reward(&code)
            })
            .await
            .map_err(|_| {
                Status::InternalServerError
            })?
            .map_err(|e| {
                println!(
                    "Reward validation \
                     failed: {e}"
                );
                Status::UnprocessableEntity
            })?;
    }

    // Build ctrl/val JSON for checkpoint records (used by both local and remote).
    let (ctrl_json, val_json) =
        build_ctrl_val_json(&controller).await;

    if is_local {
        let tx_c = tx_train.clone();
        let run_id_c = run_id.clone();
        let exp_id_c = experiment_id.clone();
        let h = rocket::tokio::spawn(async move {
            spawn_local_training_task(
                run_id_c,
                exp_id_c,
                controller,
                model_path,
                tx_c,
                stop_flag_c,
                ctrl_json,
                val_json,
            )
            .await;
        });
        state
            .training_stops
            .lock()
            .await
            .insert(run_id.clone(), (h, stop_flag));
    } else {
        // Remote training
        let remote_cfg =
            match crate::build_remote_worker_config()
            {
                Ok(c) => c,
                Err(e) => {
                    println!(
                        "remote cfg failed: {:?}",
                        e
                    );
                    return Err(
                        Status::InternalServerError,
                    );
                }
            };
        crate::ensure_remote_worker_shutdown_hook();

        let tx_c = tx_train.clone();
        let run_id_c = run_id.clone();
        let exp_id_c = experiment_id.clone();
        let worker_slot =
            crate::remote_worker_slot().clone();
        let h = rocket::tokio::spawn(async move {
            spawn_remote_training_task(
                run_id_c,
                exp_id_c,
                controller,
                model_path,
                remote_cfg,
                worker_slot,
                tx_c,
                stop_flag_c,
                ctrl_json,
                val_json,
            )
            .await;
        });
        state
            .training_stops
            .lock()
            .await
            .insert(run_id.clone(), (h, stop_flag));
    }

    Ok(Json(TrainingStartedResponse {
        id: run_id,
    }))
}

#[route_doc(
    tag = "Training",
    ok = TrainingStoppedResponse,
    params(
        experiment_id = "Experiment UUID",
        run_id = "Run UUID",
    ),
)]
#[post(
    "/training/experiments\
     /<experiment_id>/runs/<run_id>/stop",
    format = "json"
)]
pub async fn stop_training(
    experiment_id: String,
    run_id: String,
    state: &State<Arc<ApiState>>,
) -> Result<Json<TrainingStoppedResponse>, Status>
{
    if let Some((h, flag)) = state
        .training_stops
        .lock()
        .await
        .remove(&run_id)
    {
        flag.store(true, Ordering::Relaxed);
        h.abort();
    }
    let _ = crate::postgres_orm()
        .update_run_status(
            crate::PROJECT_ID,
            &experiment_id,
            &run_id,
            &TrainingRunStatus::StoppedByUser,
        )
        .await;
    Ok(Json(TrainingStoppedResponse {
        id: run_id,
    }))
}

#[route_doc(
    tag = "Training",
    params(
        experiment_id = "Experiment UUID",
        run_id = "Run UUID",
    ),
)]
#[get(
    "/training/experiments\
     /<_experiment_id>/runs/<run_id>/updates"
)]
pub async fn training_updates(
    _experiment_id: String,
    run_id: String,
    state: &State<Arc<ApiState>>,
    ws: ws::WebSocket,
) -> ws::Channel<'static>
{
    let rx = state
        .training_senders
        .lock()
        .await
        .get(&run_id)
        .map(|tx| tx.subscribe());

    ws.channel(move |mut stream| Box::pin(async move {
        use rocket::futures::{SinkExt, StreamExt};
        if let Some(mut rx) = rx {
            loop {
                rocket::tokio::select! {
                    msg = rx.recv() => match msg {
                        Ok(json) => {
                            if stream.send(ws::Message::Text(json)).await.is_err() {
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    },
                    incoming = stream.next() => match incoming {
                        None
                        | Some(Ok(ws::Message::Close(_)))
                        | Some(Err(_)) => break,
                        Some(Ok(_)) => {}
                    },
                }
            }
        }
        Ok(())
    }))
}

// ── Remote training helpers ───────────────────

async fn build_ctrl_val_json(
    controller: &TrainingJobController,
) -> (Option<String>, Option<String>)
{
    let TrainingMethod::PPO(ref p) =
        controller.training_method;

    let ctrl = match &p.control_model {
        ControlModel::MLP(m) => match &m.model_type {
            ModelType::Checkpoint(
                CheckpointType::Checkpoint(id),
            ) => crate::postgres_orm()
                .get_checkpoint_ref(id)
                .await
                .ok()
                .flatten()
                .and_then(|r| {
                    r.control_model_params
                        .as_ref()
                        .and_then(|m| {
                            serde_json::to_string(m).ok()
                        })
                }),
            _ => serde_json::to_string(m).ok(),
        },
        ControlModel::VLA(_) => None,
    };

    let val = {
        let ValueModel::MLP(ref m) = p.value_model;
        match &m.model_type {
            ModelType::Checkpoint(
                CheckpointType::Checkpoint(id),
            ) => crate::postgres_orm()
                .get_checkpoint_ref(id)
                .await
                .ok()
                .flatten()
                .and_then(|r| {
                    r.value_model_params
                        .as_ref()
                        .and_then(|m| {
                            serde_json::to_string(m).ok()
                        })
                }),
            _ => serde_json::to_string(m).ok(),
        }
    };

    (ctrl, val)
}

/// Builds the checkpoint payloads to seed a training job from any pretrained
/// control/value checkpoints referenced by the controller.
async fn build_worker_checkpoints(
    controller: &TrainingJobController,
) -> Vec<Checkpoint> {
    use crate::commondef::training::{
        CheckpointType as CT,
        ControlModel as CM,
        ModelType as MT,
        TrainingMethod as TM,
        ValueModel as VM,
    };
    use std::collections::HashMap;

    let mut map: HashMap<String, Checkpoint> = HashMap::new();
    let TM::PPO(ref ppo) = controller.training_method;

    if let CM::MLP(ref m) = ppo.control_model {
        if let MT::Checkpoint(CT::Checkpoint(ref src_id)) = m.model_type {
            if let Some(r) = crate::postgres_orm()
                .get_checkpoint_ref(src_id)
                .await
                .ok()
                .flatten()
            {
                if let Ok(bytes) = std::fs::read(&r.path) {
                    map.insert(
                        src_id.clone(),
                        Checkpoint {
                            id: Some(src_id.clone()),
                            model_zip_base64: Some(BASE64.encode(&bytes)),
                            model_zip_path: None,
                            control_model_mlp: r.control_model_params.clone(),
                            value_model_mlp: r.value_model_params.clone(),
                        },
                    );
                }
            }
        }
    }

    let VM::MLP(ref vm) = ppo.value_model;
    if let MT::Checkpoint(CT::Checkpoint(ref src_id)) = vm.model_type {
        if !map.contains_key(src_id) {
            if let Some(r) = crate::postgres_orm()
                .get_checkpoint_ref(src_id)
                .await
                .ok()
                .flatten()
            {
                if let Ok(bytes) = std::fs::read(&r.path) {
                    map.insert(
                        src_id.clone(),
                        Checkpoint {
                            id: Some(src_id.clone()),
                            model_zip_base64: Some(BASE64.encode(&bytes)),
                            model_zip_path: None,
                            control_model_mlp: r.control_model_params.clone(),
                            value_model_mlp: r.value_model_params.clone(),
                        },
                    );
                }
            }
        }
    }

    map.into_values().collect()
}

/// Builds the per-update handler shared by local and remote training: persists
/// any returned checkpoint zips to disk + DB and forwards a client update via tx.
fn make_training_on_update(
    run_id: String,
    exp_id: String,
    tx: mpsc::Sender<TrainingTaskMessage>,
    ctrl_json: Option<String>,
    val_json: Option<String>,
) -> impl FnMut(&WorkerTrainingUpdatesMessage) {
    move |update| {
        let mut client_updates = vec![];
        for u in &update.updates {
            let mut saved_refs = vec![];
            for ckpt in &u.checkpoints {
                if let Some(ref b64) = ckpt.model_zip_base64 {
                    match BASE64.decode(b64) {
                        Ok(bytes) => {
                            let ckpt_id = uuid::Uuid::new_v4().to_string();
                            let base = std::env::var("CHECKPOINTS_DIR")
                                .unwrap_or_else(|_| "checkpoints".to_string());
                            let dir = format!("{}/{}/{}", base, exp_id, run_id);
                            let path = format!("{}/{}.zip", dir, ckpt_id);
                            let _ = std::fs::create_dir_all(&dir);
                            let _ = std::fs::write(&path, &bytes);
                            let pg = crate::postgres_orm();
                            let create_res = tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(
                                    pg.create_checkpoint_ref(
                                        &ckpt_id,
                                        &path,
                                        "ppo",
                                        &exp_id,
                                        &run_id,
                                        ctrl_json.as_deref(),
                                        val_json.as_deref(),
                                    ),
                                )
                            });
                            if let Ok(r) = create_res {
                                let _ = tokio::task::block_in_place(|| {
                                    tokio::runtime::Handle::current().block_on(
                                        pg.add_checkpoint_ref_to_run(&exp_id, &run_id, &ckpt_id),
                                    )
                                });
                                saved_refs.push(r);
                            }
                        }
                        Err(e) => println!("b64 decode: {:?}", e),
                    }
                }
            }
            client_updates.push(TrainingUpdate {
                training_run_state: u.training_run_state.clone(),
                logs: u.logs.clone(),
                metrics: u.metrics.clone(),
                checkpoints: saved_refs,
            });
        }
        let _ = tx.try_send(TrainingTaskMessage::TrainingUpdates(TrainingUpdatesMessage {
            id: update.id.clone(),
            updates: client_updates,
        }));
    }
}

async fn spawn_local_training_task(
    run_id: String,
    exp_id: String,
    controller: TrainingJobController,
    model_path: String,
    tx: mpsc::Sender<TrainingTaskMessage>,
    _stop_flag: Arc<AtomicBool>,
    ctrl_json: Option<String>,
    val_json: Option<String>,
) {
    // Connect-only: assumes a Python worker is already running on the local host.
    let mut local = match LocalWorkerController::from_env() {
        Ok(c) => c,
        Err(e) => {
            println!("Local worker config failed: {:?}", e);
            let _ = tx
                .send(TrainingTaskMessage::StopTraining(StopTrainingMessage {
                    id: run_id.clone(),
                }))
                .await;
            return;
        }
    };

    let worker_checkpoints = build_worker_checkpoints(&controller).await;
    let on_update = make_training_on_update(
        run_id.clone(),
        exp_id,
        tx.clone(),
        ctrl_json,
        val_json,
    );

    let result = local
        .run_training_job(
            run_id.clone(),
            controller,
            Some(model_path),
            worker_checkpoints,
            on_update,
        )
        .await;

    if let Err(e) = result {
        println!("Local training failed: {:?}", e);
    }
    let _ = tx
        .send(TrainingTaskMessage::StopTraining(StopTrainingMessage {
            id: run_id.clone(),
        }))
        .await;
}

async fn spawn_remote_training_task(
    run_id: String,
    exp_id: String,
    controller: TrainingJobController,
    model_path: String,
    remote_cfg: crate::workflows
        ::remote_worker_controller
        ::RemoteWorkerControllerConfig,
    worker_slot: crate::RemoteWorkerSlot,
    tx: mpsc::Sender<TrainingTaskMessage>,
    _stop_flag: Arc<AtomicBool>,
    ctrl_json: Option<String>,
    val_json: Option<String>,
) {
    let mut guard = worker_slot.lock().await;
    if guard.is_none() {
        match crate::workflows
            ::remote_worker_controller
            ::RemoteWorkerController
            ::initialize(remote_cfg)
            .await
        {
            Ok(w) => *guard = Some(w),
            Err(e) => {
                println!("Remote worker init: {:?}", e);
                let _ = tx
                    .send(TrainingTaskMessage::StopTraining(StopTrainingMessage {
                        id: run_id.clone(),
                    }))
                    .await;
                return;
            }
        }
    }
    let worker = guard.as_mut().expect("worker initialised");

    let worker_checkpoints = build_worker_checkpoints(&controller).await;
    let on_update = make_training_on_update(
        run_id.clone(),
        exp_id,
        tx.clone(),
        ctrl_json,
        val_json,
    );

    let result = worker
        .run_training_job(
            run_id.clone(),
            controller,
            Some(model_path),
            worker_checkpoints,
            on_update,
        )
        .await;

    drop(guard);
    if let Err(e) = result {
        println!("Remote training failed: {:?}", e);
    }
    let _ = tx
        .send(TrainingTaskMessage::StopTraining(StopTrainingMessage {
            id: run_id.clone(),
        }))
        .await;
}
