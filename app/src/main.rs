use std::collections::HashMap;
use std::fs;

#[macro_use]
extern crate rocket;
use rocket::fs::FileServer;
use rocket::futures::{SinkExt, StreamExt};
use rocket::tokio;
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
                                                    match WorkflowRuntime::load(model_path) {
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
                                                        },
                                                        Err(e) => {
                                                            println!("Couldn't load shared workflow runtime {:?}",e);
                                                        }
                                                    }
                                                },
                                                clientmessages::ServerMessage::StartTrainingMessage(t) => {
                                                    println!("Start training ask from client {:?}",t);
                                                    if workflow_runtime.is_none() {
                                                        println!("LoadModelMessage must be sent before StartTrainingMessage");
                                                        continue;
                                                    }
                                                    let commondef::training::TrainingMethod::PPO(ppomethod) =
                                                        t.training_controller.training_method;
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
                                                    let id_c = t.id.clone();
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
                                                    training_tasks.insert(t.id.clone(),h);
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
