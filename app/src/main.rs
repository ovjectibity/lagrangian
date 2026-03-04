use std::collections::HashMap;
use std::env;
use std::fs;
use std::sync::{Arc, OnceLock};

#[macro_use]
extern crate rocket;
use rocket::fs::FileServer;
use rocket::futures::{SinkExt, StreamExt};
use rocket::tokio;
use rocket::tokio::sync::Mutex as AsyncMutex;
use rocket::tokio::sync::mpsc;
use rocket_ws as ws;

use crate::commondef::{clientmessages, workermessages};
use crate::workflows::testing::TestingWorkflow;
use crate::workflows::training::TrainingWorkflow;
use crate::workflows::{WorkflowMode, WorkflowRuntime};

pub mod commondef;
pub mod mujoco;
pub mod plugin;
pub mod remote_workflows;
pub mod simulator;
pub mod usr;
pub mod workflows;

type RemoteWorkerSlot =
    Arc<AsyncMutex<Option<remote_workflows::training_controller::TrainingWorkerController>>>;

static REMOTE_WORKER_SLOT: OnceLock<RemoteWorkerSlot> = OnceLock::new();
static REMOTE_WORKER_SHUTDOWN_TASK_STARTED: OnceLock<()> = OnceLock::new();

fn remote_worker_slot() -> &'static RemoteWorkerSlot {
    REMOTE_WORKER_SLOT.get_or_init(|| Arc::new(AsyncMutex::new(None)))
}

fn ensure_remote_worker_shutdown_hook() {
    if REMOTE_WORKER_SHUTDOWN_TASK_STARTED.set(()).is_err() {
        return;
    }

    let slot = remote_worker_slot().clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let mut guard = slot.lock().await;
            if let Some(worker) = guard.as_mut() {
                let _ = worker.terminate_instance().await;
            }
            *guard = None;
        }
    });
}

fn build_remote_training_worker_config()
-> Result<remote_workflows::training_controller::TrainingWorkerControllerConfig, String> {
    let mut cfg = remote_workflows::training_controller::TrainingWorkerControllerConfig::default();

    if let Ok(region) = env::var("TRAINING_WORKER_AWS_REGION") {
        if !region.is_empty() {
            cfg.aws_region = region;
        }
    }
    cfg.ami_id = env::var("TRAINING_WORKER_AMI_ID")
        .map_err(|_| "Missing required env var TRAINING_WORKER_AMI_ID".to_string())?;
    cfg.worker_image_uri = env::var("TRAINING_WORKER_IMAGE_URI")
        .map_err(|_| "Missing required env var TRAINING_WORKER_IMAGE_URI".to_string())?;
    cfg.worker_shared_secret = env::var("TRAINING_WORKER_SHARED_SECRET")
        .map_err(|_| "Missing required env var TRAINING_WORKER_SHARED_SECRET".to_string())?;

    if let Ok(v) = env::var("TRAINING_WORKER_SUBNET_ID") {
        if !v.is_empty() {
            cfg.subnet_id = Some(v);
        }
    }
    if let Ok(v) = env::var("TRAINING_WORKER_SECURITY_GROUP_IDS") {
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
    if let Ok(v) = env::var("TRAINING_WORKER_KEY_NAME") {
        if !v.is_empty() {
            cfg.key_name = Some(v);
        }
    }
    if let Ok(v) = env::var("TRAINING_WORKER_IAM_INSTANCE_PROFILE_ARN") {
        if !v.is_empty() {
            cfg.iam_instance_profile_arn = Some(v);
        }
    }
    if let Ok(v) = env::var("TRAINING_WORKER_INSTANCE_NAME_TAG") {
        if !v.is_empty() {
            cfg.instance_name_tag = v;
        }
    }
    if let Ok(v) = env::var("TRAINING_WORKER_SPOT_MAX_PRICE_USD") {
        if !v.is_empty() {
            cfg.spot_max_price_usd = Some(v);
        }
    }
    if let Ok(v) = env::var("TRAINING_WORKER_CONTAINER_NAME") {
        if !v.is_empty() {
            cfg.worker_container_name = v;
        }
    }
    if let Ok(v) = env::var("TRAINING_WORKER_PORT") {
        if let Ok(port) = v.parse::<u16>() {
            cfg.worker_port = port;
        }
    }
    if let Ok(v) = env::var("TRAINING_WORKER_WS_PATH") {
        if !v.is_empty() {
            cfg.worker_ws_path = v;
        }
    }

    Ok(cfg)
}

// --- API Routes ---
#[get("/hello")]
fn api_hello() -> &'static str {
    "Hello from Rocket!"
}

#[get("/viewport")]
fn viewport_channel(ws: ws::WebSocket) -> ws::Channel<'static> {
    let (tx_simul, mut rx_simul) = mpsc::channel::<workermessages::SimulatorThreadMessage>(32);
    let (tx_test, mut rx_test) = mpsc::channel::<workermessages::TestingTaskMessage>(32);
    let (tx_train, mut rx_train) = mpsc::channel::<workermessages::TrainingTaskMessage>(32);
    let mut training_tasks = HashMap::new();
    let mut testing_tasks = HashMap::new();
    let mut workflow_runtime: Option<WorkflowRuntime> = None;
    let mut loaded_model_path: Option<String> = None;
    ws.channel(move |mut stream| {
        Box::pin(async move {
            loop {
                tokio::select! {
                    Some(message) = stream.next() => {
                        match message {
                            Ok(ws::Message::Text(text)) => {
                                if text == "get_test_model" {
                                    let world_body =
                                        simulator::get_test_model();
                                    let response_message =
                                        clientmessages::ClientMessage::new_set_model_message(world_body);
                                    let json_response =
                                        serde_json::to_string(&response_message)
                                        .unwrap_or_else(|_| "{}".to_string());
                                    let _ = stream.send(
                                        ws::Message::Text(json_response)).await;
                                } else if text == "get_test_model_2" {
                                    let world_body;
                                    unsafe {
                                        world_body =
                                        // simulator::load_test_model(
                                        //     "/Users/avi/Documents/manual/lagrangian/app/assets/universal_robots_ur10e/ur10e.xml".
                                        //     to_string());
                                        simulator::load_test_model(
                                            "/Users/avi/Documents/manual/lagrangian/app/assets/trossen_vx300s/vx300s.xml".
                                            to_string());
                                    }
                                    if let Some(wb) = world_body {
                                        let response_message =
                                            clientmessages::ClientMessage::new_set_model_message(wb);
                                        let json_response =
                                            serde_json::to_string(&response_message)
                                            .unwrap_or_else(|_| "{}".to_string());
                                        if let Err(e) =
                                            fs::write("target/test_model_2.json", json_response.clone()) {
                                            println!("Could not write the test model 2 JSON to file {:?}",e);
                                        }
                                        let _ = stream.send(
                                            ws::Message::Text(json_response)).await;
                                    }
                                } else if text == "get_test_model_3" {
                                    if let Ok(json_response) =
                                        fs::read_to_string("target/test_model_3.json") {
                                        let _ = stream.send(
                                            ws::Message::Text(json_response)).await;
                                    }
                                } else if text == "run_simul" {
                                    //TODO: First send the load_message here
                                    //Setup the map ID of the model
                                    //Pass id to the simulator thread as well
                                    simulator::spawn_simulator(
                                        "/Users/avi/Documents/manual/lagrangian/app/assets/trossen_vx300s/vx300s.xml"
                                        .to_string(),
                                        tx_simul.clone());
                                } else {
                                    match serde_json::from_str::<clientmessages::ServerMessage>(&text) {
                                        Ok(m) => {
                                            match m {
                                                clientmessages::ServerMessage::LoadModelMessage => {
                                                    let model_path = "/Users/avi/Documents/manual/lagrangian/app/assets/trossen_vx300s/vx300s.xml"
                                                        .to_string();
                                                    match WorkflowRuntime::load(model_path.clone()) {
                                                        Ok(runtime) => {
                                                            println!("Loaded shared workflow runtime");
                                                            match runtime.world_body() {
                                                                Ok(world_body) => {
                                                                    let response_message = clientmessages::ClientMessage::new_load_model_message(
                                                                        "1".to_string(),
                                                                        world_body,
                                                                    );
                                                                    let json_response =
                                                                        serde_json::to_string(&response_message)
                                                                            .unwrap_or_else(|e| {
                                                                                format!(
                                                                                    "{{\"error\":\"{}\"}}",
                                                                                    e
                                                                                )
                                                                            });
                                                                    let _ = stream
                                                                        .send(ws::Message::Text(
                                                                            json_response,
                                                                        ))
                                                                        .await;
                                                                }
                                                                Err(e) => {
                                                                    println!(
                                                                        "Couldn't extract world body from runtime {:?}",
                                                                        e
                                                                    );
                                                                }
                                                            }
                                                            workflow_runtime = Some(runtime);
                                                            loaded_model_path = Some(model_path);
                                                        },
                                                        Err(e) => {
                                                            println!("Couldn't load shared workflow runtime {:?}",e);
                                                        }
                                                    }
                                                },
                                                clientmessages::ServerMessage::StartTrainingMessage(t) => {
                                                    println!("Start training ask from client {:?}",t);
                                                    let training_id = t.id.clone();
                                                    let training_controller = t.training_controller;
                                                    match (
                                                        training_controller.acceleration_mode,
                                                        training_controller.training_method,
                                                    ) {
                                                        (
                                                            commondef::training::TrainingAccelerationMode::AllOnServer,
                                                            commondef::training::TrainingMethod::PPO(ppomethod),
                                                        ) => {
                                                            if workflow_runtime.is_none() {
                                                                println!("LoadModelMessage must be sent before StartTrainingMessage");
                                                                continue;
                                                            }
                                                            let runtime = workflow_runtime
                                                                .as_mut()
                                                                .expect("Runtime should be available");
                                                            if let Err(e) = runtime
                                                                .configure_model_for_training(&ppomethod) {
                                                                println!("Couldn't configure shared model for training {:?}",e);
                                                                continue;
                                                            }
                                                            runtime.set_mode(WorkflowMode::Training);
                                                            let simulator_h = runtime.simulator_handle();
                                                            let model_h = runtime.ppo_model_handle();
                                                            let tx_tc = tx_train.clone();
                                                            let id_c = training_id.clone();
                                                            let h = tokio::task::spawn_blocking(move || {
                                                                let workflow_r = TrainingWorkflow::new(
                                                                    id_c,
                                                                    ppomethod,
                                                                    simulator_h,
                                                                    model_h,
                                                                );
                                                                if let Ok(mut workflow) = workflow_r {
                                                                    workflow.run_training_blocking(tx_tc);
                                                                }
                                                            });
                                                            training_tasks.insert(training_id, h);
                                                        }
                                                        (_, training_method) => {
                                                            let model_path = match loaded_model_path.clone() {
                                                                Some(path) => path,
                                                                None => {
                                                                    println!("LoadModelMessage must be sent before StartTrainingMessage");
                                                                    continue;
                                                                }
                                                            };
                                                            let mjcf = match fs::read_to_string(&model_path) {
                                                                Ok(content) => content,
                                                                Err(e) => {
                                                                    println!("Couldn't read model MJCF from path {:?}: {:?}", model_path, e);
                                                                    continue;
                                                                }
                                                            };
                                                            let remote_cfg = match build_remote_training_worker_config() {
                                                                Ok(cfg) => cfg,
                                                                Err(e) => {
                                                                    println!("Couldn't build remote training worker config: {:?}", e);
                                                                    continue;
                                                                }
                                                            };
                                                            ensure_remote_worker_shutdown_hook();

                                                            if let Some(runtime) = workflow_runtime.as_mut() {
                                                                runtime.set_mode(WorkflowMode::Training);
                                                            }
                                                            let tx_tc = tx_train.clone();
                                                            let id_c = training_id.clone();
                                                            let worker_slot = remote_worker_slot().clone();
                                                            let h = tokio::spawn(async move {
                                                                let tx_updates = tx_tc.clone();
                                                                let mut guard = worker_slot.lock().await;
                                                                if guard.is_none() {
                                                                    match remote_workflows::training_controller::TrainingWorkerController::initialize(remote_cfg).await {
                                                                        Ok(worker) => {
                                                                            *guard = Some(worker);
                                                                        }
                                                                        Err(e) => {
                                                                            println!("Couldn't initialize remote training worker: {:?}", e);
                                                                            let _ = tx_tc.send(workermessages::TrainingTaskMessage::StopTraining(
                                                                                clientmessages::StopTrainingMessage { id: id_c.clone() },
                                                                            )).await;
                                                                            return;
                                                                        }
                                                                    }
                                                                }
                                                                let worker = guard
                                                                    .as_mut()
                                                                    .expect("worker should be initialized");
                                                                let result = worker
                                                                    .run_training_job(
                                                                        id_c.clone(),
                                                                        training_method,
                                                                        mjcf,
                                                                        move |update| {
                                                                            let _ = tx_updates.try_send(
                                                                                workermessages::TrainingTaskMessage::TrainingUpdates(
                                                                                    clientmessages::TrainingUpdatesMessage {
                                                                                        id: update.id.clone(),
                                                                                        metrics: update.metrics.clone(),
                                                                                        logs: update.logs.clone(),
                                                                                    },
                                                                                ),
                                                                            );
                                                                        },
                                                                    )
                                                                    .await;
                                                                drop(guard);
                                                                if let Err(e) = result {
                                                                    println!("Remote training execution failed: {:?}", e);
                                                                }
                                                                let _ = tx_tc
                                                                    .send(workermessages::TrainingTaskMessage::StopTraining(
                                                                        clientmessages::StopTrainingMessage {
                                                                            id: id_c.clone(),
                                                                        },
                                                                    ))
                                                                    .await;
                                                            });
                                                            training_tasks.insert(training_id, h);
                                                        }
                                                    }
                                                },
                                                clientmessages::ServerMessage::StopTrainingMessage(t) => {
                                                     println!("Stopping training as asked by client {:?}",t);
                                                    training_tasks.remove(&t.id.clone());
                                                    if let Some(runtime) = workflow_runtime.as_mut() {
                                                        runtime.set_mode(WorkflowMode::Idle);
                                                    }
                                                },
                                                clientmessages::ServerMessage::StartSimulationMessage(t) => {
                                                    println!("Start simulation ask from client {:?}",t.id);
                                                    if workflow_runtime.is_none() {
                                                        println!("LoadModelMessage must be sent before StartSimulationMessage");
                                                        continue;
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
                                                        let mut workflow = TestingWorkflow::new(
                                                            id_c,
                                                            simulator_h,
                                                            model_h,
                                                            params,
                                                        );
                                                        workflow.run_test(tx_sc);
                                                    });
                                                    testing_tasks.insert(t.id, h);
                                                },
                                                clientmessages::ServerMessage::StopSimulationMessage(t) => {
                                                    println!("Stop simulation ask from client {:?}",t.id);
                                                    testing_tasks.remove(&t.id.clone());
                                                    if let Some(runtime) = workflow_runtime.as_mut() {
                                                        runtime.set_mode(WorkflowMode::Idle);
                                                    }
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            println!("Couldn't deserialise the msg from client to a server message {:?}",e);
                                        }
                                    }
                                }
                            }
                            Ok(ws::Message::Binary(bin)) => {
                                let _ = stream.send(
                                    ws::Message::Binary(bin)).await;
                            }
                            Ok(ws::Message::Close(_)) => {
                                println!("Client disconnected");
                                break;
                            }
                            Err(e) => {
                                eprintln!("WebSocket error: {:?}", e);
                                break;
                            }
                            _ => {}
                        }
                    }
                    Some(msg) = rx_simul.recv() => {
                        match msg {
                            workermessages::SimulatorThreadMessage::LoadModel(m) => {
                                let response_message =
                                    clientmessages::ClientMessage::new_load_model_message(
                                    "1".to_string(), m.model);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                            workermessages::SimulatorThreadMessage::SimulUpdate(m) => {
                                let binary_data = clientmessages::ClientMessage::to_binary_simul_update(
                                    "1".to_string(), m.update
                                );
                                let _ = stream.send(ws::Message::Binary(binary_data)).await;
                            }
                        };
                    },
                    Some(msg) = rx_train.recv() => {
                        // println!("Sending training updates to client {:?}",msg);
                        println!("Sending testing updates to client");
                        match msg {
                            workermessages::TrainingTaskMessage::TrainingUpdates(m) => {
                                let response_message =
                                    clientmessages::ClientMessage::TrainingUpdatesMessage(m);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                            workermessages::TrainingTaskMessage::StopTraining(m) => {
                                training_tasks.remove(&m.id);
                                if let Some(runtime) = workflow_runtime.as_mut() {
                                    runtime.set_mode(WorkflowMode::Idle);
                                }
                                let response_message =
                                    clientmessages::ClientMessage::StopTrainingMessage(m);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                        }
                    },
                    Some(msg) = rx_test.recv() => {
                        // println!("Sending testing updates to client {:?}",msg);
                        println!("Sending testing updates to client");
                        match msg {
                            workermessages::TestingTaskMessage::StartSimulation(_) => {},
                            workermessages::TestingTaskMessage::StopSimulation(m) => {
                                testing_tasks.remove(&m.id);
                                if let Some(runtime) = workflow_runtime.as_mut() {
                                    runtime.set_mode(WorkflowMode::Idle);
                                }
                                let response_message =
                                    clientmessages::ClientMessage::StopSimulationMessage(m);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                            workermessages::TestingTaskMessage::SimulationUpdates(m) => {
                                let response_message =
                                    clientmessages::ClientMessage::SimulUpdateMessage(m);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                        }
                    },
                    else => break
                }
            }
            Ok(())
        })
    })
}

#[launch]
fn rocket() -> _ {
    // renderer::render();
    rocket::build()
        // 1. API Routes
        .mount("/api", routes![api_hello, viewport_channel])
        // 2. Static Files (JS/CSS/Images)
        // Serve the 'dist' folder at the root path "/"
        // 'rank = 1' tries to match exact files first
        .mount(
            "/",
            FileServer::from("/Users/avi/Documents/manual/lagrangian/client/dist").rank(1),
        )

    // 3. Fallback (for client-side routing)
    // 'rank = 2' is matched only if FileServer finds nothing
    // .mount("/", routes![index_fallback])
}
