use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rocket::State;
use rocket::http::Status;
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::Json;
use rocket::tokio::sync::{broadcast, mpsc};
use serde::{Deserialize, Serialize};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD
    as BASE64;

use crate::api::ApiState;
use crate::commondef::clientmessages::{
    self as cm,
    StopTrainingMessage,
    TrainingUpdatesMessage,
};
use crate::commondef::experiments::TrainingRunStatus;
use tch::Tensor;

use crate::commondef::training::{
    ControlModel,
    RewardLanguage,
    TrainingGraph,
    TrainingJobController,
    TrainingMethod,
    ValueModel,
};
use crate::commondef::workermessages::TrainingTaskMessage;
use crate::workflows::{WorkflowMode, WorkflowRuntime};
use crate::workflows::training::TrainingWorkflow;

// ── Request / response types ─────────────────

#[derive(Deserialize)]
pub struct StartTrainingBody {
    pub training_graph: TrainingGraph,
}

#[derive(Serialize)]
pub struct TrainingStartedResponse {
    pub id: String,
}

#[derive(Serialize)]
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

/// Resolves the serialised FFN JSON for a model
/// parameter (used when creating checkpoint DB
/// records on the remote training path).
async fn resolve_ffn_json<M: serde::Serialize>(
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

// ── Handlers ─────────────────────────────────

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
    state: &State<ApiState>,
) -> Result<Json<TrainingStartedResponse>, Status>
{
    let training_graph =
        body.into_inner().training_graph;
    let controller =
        training_graph.training_job_controller.clone();
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

    let run_id_cleanup = run_id.clone();
    // Spawn cleanup task that removes the sender
    // from the map once the relay finishes.
    let training_senders_ptr =
        state.inner() as *const ApiState as usize;
    let _ = training_senders_ptr;

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
    if is_local
        && !matches!(
            reward_lang,
            RewardLanguage::Rust,
        )
    {
        return Err(Status::BadRequest);
    }
    if !is_local
        && !matches!(
            reward_lang,
            RewardLanguage::Python,
        )
    {
        return Err(Status::BadRequest);
    }
    let user_reward_lib = if is_local {
        let code = reward_code.clone();
        let lib = rocket::tokio::task
            ::spawn_blocking(move || {
                crate::api::reward_validation
                    ::validate_and_compile_rust_reward(
                        &code,
                    )
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
        Some(lib)
    } else {
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
        None
    };

    if is_local {
        let ppo_params = match controller
            .training_method
        {
            TrainingMethod::PPO(ref p) => p.clone(),
        };
        let tx_c = tx_train.clone();
        let run_id_c = run_id.clone();
        let h = rocket::tokio::task::spawn_blocking(
            move || {
                let path = model_path;
                let mut runtime =
                    match WorkflowRuntime::load(
                        path,
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            println!(
                                "WorkflowRuntime \
                                 load failed: {:?}",
                                e
                            );
                            return;
                        }
                    };
                if let Err(e) = runtime
                    .configure_model_for_training(
                        &ppo_params,
                    )
                {
                    println!(
                        "configure model: {:?}",
                        e
                    );
                    return;
                }
                runtime.set_mode(
                    WorkflowMode::Training,
                );
                let sim = runtime.simulator_handle();
                let model =
                    runtime.ppo_model_handle();
                use std::sync::Arc as StdArc;
                let reward_fn: Option<
                    Box<
                        dyn Fn(
                            &Tensor,
                            &Tensor,
                            &Tensor,
                            Option<
                                &[(usize, usize)]
                            >,
                        ) -> f64
                            + Send,
                    >,
                > = user_reward_lib.map(|lib| {
                    let lib = StdArc::new(lib);
                    let lib_c = lib.clone();
                    Box::new(
                        move |prev: &Tensor,
                              curr: &Tensor,
                              act: &Tensor,
                              contacts: Option<
                                  &[(usize, usize)]
                              >| {
                            unsafe {
                                type RawFn =
                                    unsafe extern "C"
                                    fn(
                                        *const f64,
                                        usize,
                                        *const f64,
                                        usize,
                                        *const f64,
                                        usize,
                                        *const (
                                            usize,
                                            usize,
                                        ),
                                        usize,
                                    ) -> f64;
                                let f: libloading
                                    ::Symbol<RawFn> =
                                    lib_c
                                    .get(
                                        b"reward_c_abi"
                                    )
                                    .expect(
                                        "reward_c_abi"
                                    );
                                let pv =
                                    Vec::<f64>
                                    ::try_from(
                                        prev
                                        .shallow_clone()
                                    )
                                    .unwrap_or_default();
                                let cv =
                                    Vec::<f64>
                                    ::try_from(
                                        curr
                                        .shallow_clone()
                                    )
                                    .unwrap_or_default();
                                let av =
                                    Vec::<f64>
                                    ::try_from(
                                        act
                                        .shallow_clone()
                                    )
                                    .unwrap_or_default();
                                let cs = contacts
                                    .unwrap_or(&[]);
                                f(
                                    pv.as_ptr(),
                                    pv.len(),
                                    cv.as_ptr(),
                                    cv.len(),
                                    av.as_ptr(),
                                    av.len(),
                                    cs.as_ptr(),
                                    cs.len(),
                                )
                            }
                        },
                    ) as Box<
                        dyn Fn(
                            &Tensor,
                            &Tensor,
                            &Tensor,
                            Option<
                                &[(usize, usize)]
                            >,
                        ) -> f64
                            + Send,
                    >
                });
                let wf = TrainingWorkflow::new(
                    run_id_c,
                    ppo_params,
                    sim,
                    model,
                    reward_fn,
                );
                if let Ok(mut wf) = wf {
                    wf.run_training_blocking(
                        tx_c,
                        stop_flag_c,
                    );
                }
            },
        );
        state
            .training_stops
            .lock()
            .await
            .insert(run_id.clone(), (h, stop_flag));
    } else {
        // Remote training
        let remote_cfg =
            match crate::build_remote_training_worker_config()
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

        // Build ctrl/val JSON for checkpoint records
        let (ctrl_json, val_json) =
            build_ctrl_val_json(&controller).await;

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

#[post(
    "/training/experiments\
     /<experiment_id>/runs/<run_id>/stop",
    format = "json"
)]
pub async fn stop_training(
    experiment_id: String,
    run_id: String,
    state: &State<ApiState>,
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

#[get(
    "/training/experiments\
     /<_experiment_id>/runs/<run_id>/updates"
)]
pub async fn training_updates(
    _experiment_id: String,
    run_id: String,
    state: &State<ApiState>,
) -> EventStream![]
{
    let rx = state
        .training_senders
        .lock()
        .await
        .get(&run_id)
        .map(|tx| tx.subscribe());

    EventStream! {
        if let Some(mut rx) = rx {
            loop {
                match rx.recv().await {
                    Ok(data) =>
                        yield Event::data(data),
                    Err(
                        broadcast::error::RecvError
                        ::Closed
                    ) => break,
                    Err(
                        broadcast::error::RecvError
                        ::Lagged(_)
                    ) => continue,
                }
            }
        }
    }
}

// ── Remote training helpers ───────────────────

async fn build_ctrl_val_json(
    controller: &TrainingJobController,
) -> (Option<String>, Option<String>)
{
    let ctrl = if let TrainingMethod::PPO(ref p) =
        controller.training_method
    {
        match &p.control_model {
            ControlModel::FFN(m) =>
                serde_json::to_string(m).ok(),
            ControlModel::Checkpoint(id) =>
                crate::postgres_orm()
                    .get_checkpoint_ref(id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|r| {
                        r.control_model_params
                            .as_ref()
                            .and_then(|m| {
                                serde_json::to_string(m)
                                    .ok()
                            })
                    }),
        }
    } else {
        None
    };

    let val = if let TrainingMethod::PPO(ref p) =
        controller.training_method
    {
        match &p.value_model {
            ValueModel::FFN(m) =>
                serde_json::to_string(m).ok(),
            ValueModel::Checkpoint(id) =>
                crate::postgres_orm()
                    .get_checkpoint_ref(id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|r| {
                        r.value_model_params
                            .as_ref()
                            .and_then(|m| {
                                serde_json::to_string(m)
                                    .ok()
                            })
                    }),
        }
    } else {
        None
    };

    (ctrl, val)
}

async fn spawn_remote_training_task(
    run_id: String,
    exp_id: String,
    controller: TrainingJobController,
    model_path: String,
    remote_cfg: crate::remote_workflows
        ::training_controller
        ::TrainingWorkerControllerConfig,
    worker_slot: crate::RemoteWorkerSlot,
    tx: mpsc::Sender<TrainingTaskMessage>,
    stop_flag: Arc<AtomicBool>,
    ctrl_json: Option<String>,
    val_json: Option<String>,
) {
    use crate::commondef::remoteworkermessages
        ::Checkpoint as WC;
    use crate::commondef::training::{
        ControlModel as CM,
        TrainingMethod as TM,
        ValueModel as VM,
    };
    use std::collections::HashMap;

    let mut guard = worker_slot.lock().await;
    if guard.is_none() {
        match crate::remote_workflows
            ::training_controller
            ::TrainingWorkerController
            ::initialize(remote_cfg)
            .await
        {
            Ok(w) => *guard = Some(w),
            Err(e) => {
                println!(
                    "Remote worker init: {:?}",
                    e
                );
                let _ = tx
                    .send(
                        TrainingTaskMessage
                        ::StopTraining(
                            StopTrainingMessage {
                                id: run_id.clone(),
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
        .expect("worker initialised");

    // Build checkpoint payloads
    let worker_checkpoints: Vec<WC> = {
        let mut map: HashMap<String, WC> =
            HashMap::new();
        if let TM::PPO(ref ppo) =
            controller.training_method
        {
            if let CM::Checkpoint(ref src_id) =
                ppo.control_model
            {
                if let Some(r) =
                    crate::postgres_orm()
                        .get_checkpoint_ref(src_id)
                        .await
                        .ok()
                        .flatten()
                {
                    if let Ok(bytes) =
                        std::fs::read(&r.path)
                    {
                        map.insert(
                            src_id.clone(),
                            WC {
                                id: Some(
                                    src_id.clone(),
                                ),
                                model_zip_base64:
                                    Some(BASE64
                                        .encode(&bytes)),
                                model_zip_path: None,
                                control_model_ffn:
                                    r.control_model_params
                                    .clone(),
                                value_model_ffn:
                                    r.value_model_params
                                    .clone(),
                            },
                        );
                    }
                }
            }
            if let VM::Checkpoint(ref src_id) =
                ppo.value_model
            {
                if !map.contains_key(src_id) {
                    if let Some(r) =
                        crate::postgres_orm()
                            .get_checkpoint_ref(src_id)
                            .await
                            .ok()
                            .flatten()
                    {
                        if let Ok(bytes) =
                            std::fs::read(&r.path)
                        {
                            map.insert(
                                src_id.clone(),
                                WC {
                                    id: Some(
                                        src_id.clone()
                                    ),
                                    model_zip_base64:
                                        Some(BASE64
                                            .encode(
                                                &bytes,
                                            )),
                                    model_zip_path:
                                        None,
                                    control_model_ffn:
                                        r.control_model_params
                                        .clone(),
                                    value_model_ffn:
                                        r.value_model_params
                                        .clone(),
                                },
                            );
                        }
                    }
                }
            }
        }
        map.into_values().collect()
    };

    let mjcf =
        std::fs::read_to_string(&model_path)
        .unwrap_or_default();
    let result = worker
        .run_training_job(
            run_id.clone(),
            controller.clone(),
            Some(model_path),
            mjcf,
            worker_checkpoints,
            |update| {
                let mut client_updates = vec![];
                for u in &update.updates {
                    let mut saved_refs = vec![];
                    for ckpt in &u.checkpoints {
                        if let Some(ref b64) =
                            ckpt.model_zip_base64
                        {
                            match BASE64.decode(&b64)
                            {
                                Ok(bytes) => {
                                    let ckpt_id =
                                        uuid::Uuid
                                        ::new_v4()
                                        .to_string();
                                    let dir = format!(
                                        "checkpoints\
                                         /{}/{}",
                                        exp_id,
                                        run_id,
                                    );
                                    let path =
                                        format!(
                                            "{}/{}.zip",
                                            dir,
                                            ckpt_id,
                                        );
                                    let _ =
                                        std::fs
                                        ::create_dir_all(
                                            &dir,
                                        );
                                    let _ =
                                        std::fs::write(
                                            &path,
                                            &bytes,
                                        );
                                    let pg =
                                        crate::postgres_orm();
                                    let create_res =
                                        tokio::task
                                        ::block_in_place(
                                            || {
                                                tokio::runtime
                                                ::Handle
                                                ::current()
                                                .block_on(
                                                    pg
                                                    .create_checkpoint_ref(
                                                        &ckpt_id,
                                                        &path,
                                                        "ppo",
                                                        &exp_id,
                                                        &run_id,
                                                        ctrl_json
                                                            .as_deref(),
                                                        val_json
                                                            .as_deref(),
                                                    )
                                                )
                                            }
                                        );
                                    if let Ok(r) =
                                        create_res
                                    {
                                        let _ =
                                            tokio::task
                                            ::block_in_place(
                                                || {
                                                    tokio::runtime
                                                    ::Handle
                                                    ::current()
                                                    .block_on(
                                                        pg
                                                        .add_checkpoint_ref_to_run(
                                                            &exp_id,
                                                            &run_id,
                                                            &ckpt_id,
                                                        )
                                                    )
                                                }
                                            );
                                        saved_refs
                                            .push(r);
                                    }
                                }
                                Err(e) => println!(
                                    "b64 decode: {:?}",
                                    e
                                ),
                            }
                        }
                    }
                    client_updates.push(
                        cm::TrainingUpdate {
                            training_run_state:
                                u.training_run_state
                                .clone(),
                            logs: u.logs.clone(),
                            metrics: u.metrics.clone(),
                            checkpoints: saved_refs,
                        },
                    );
                }
                let _ = tx.try_send(
                    TrainingTaskMessage
                    ::TrainingUpdates(
                        TrainingUpdatesMessage {
                            id: update.id.clone(),
                            updates: client_updates,
                        },
                    ),
                );
            },
        )
        .await;

    drop(guard);
    if let Err(e) = result {
        println!("Remote training failed: {:?}", e);
    }
    let _ = tx
        .send(TrainingTaskMessage::StopTraining(
            StopTrainingMessage {
                id: run_id.clone(),
            },
        ))
        .await;
}
