use std::fs;
use std::collections::HashMap;

#[macro_use]
extern crate rocket;
use rocket::tokio;
use rocket::tokio::sync::mpsc;
use rocket_ws as ws;
use rocket::fs::FileServer;
use rocket::futures::{SinkExt, StreamExt};

use crate::commondef::{messages, workermessages};
use crate::training::TrainingWorkflow;

pub mod simulator;
pub mod mujoco;
pub mod training;
pub mod commondef;
pub mod plugin;
pub mod usr;

// --- API Routes ---
#[get("/hello")]
fn api_hello() -> &'static str {
    "Hello from Rocket!"
}

#[get("/viewport")]
fn viewport_channel(ws: ws::WebSocket) -> ws::Channel<'static> {
    let (tx,mut rx) = 
        mpsc::channel::<workermessages::SimulatorThreadMessage>(32);
    let (tx_t,mut rx_t) = 
        mpsc::channel::<workermessages::TrainingTaskMessage>(32);
    let mut training_tasks = HashMap::new();
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
                                        messages::ClientMessage::new_set_model_message(world_body);
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
                                            messages::ClientMessage::new_set_model_message(wb);
                                        let json_response = 
                                            serde_json::to_string(&response_message)
                                            .unwrap_or_else(|_| "{}".to_string());
                                        if let Err(e) = fs::write("target/test_model_2.json", json_response.clone()) {
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
                                        "/Users/avi/Documents/manual/lagrangian/app/assets/trossen_vx300s/vx300s.xml".to_string(),
                                        tx.clone());
                                } else {
                                    match serde_json::from_str::<messages::ServerMessage>(&text) {
                                        Ok(m) => {
                                            match m {
                                                messages::ServerMessage::StartTrainingMessage(t) => {
                                                    println!("Start training ask from client {:?}",t);
                                                    let commondef::training::TrainingMethod::PPO(ppomethod) = 
                                                        t.training_workflow;
                                                    let tx_tc = tx_t.clone();
                                                    let id_c = t.id.clone();
                                                    let h = tokio::spawn(async move {
                                                        let workflow_r = TrainingWorkflow::new(
                                                            "/Users/avi/Documents/manual/lagrangian/app/assets/trossen_vx300s/vx300s.xml".to_string(),
                                                            id_c,ppomethod);
                                                        if let Ok(mut workflow) = workflow_r {
                                                            workflow.run_training(tx_tc).await;
                                                        }
                                                    });
                                                    training_tasks.insert(t.id.clone(),h);
                                                },
                                                messages::ServerMessage::StopTrainingMessage(t) => {
                                                     println!("Stopping training as asked by client {:?}",t);
                                                    training_tasks.remove(&t.id.clone());
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
                    Some(msg) = rx.recv() => {
                        match msg {
                            workermessages::SimulatorThreadMessage::LoadModel(m) => {
                                let response_message = messages::ClientMessage::new_load_model_message(
                                    "1".to_string(), m.model);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                            workermessages::SimulatorThreadMessage::SimulUpdate(m) => {
                                let binary_data = messages::ClientMessage::to_binary_simul_update(
                                    "1".to_string(), m.update
                                );
                                let _ = stream.send(ws::Message::Binary(binary_data)).await;
                            }
                        };
                    },
                    Some(msg) = rx_t.recv() => {
                        println!("Sending training updates to client {:?}",msg);
                        match msg {
                            workermessages::TrainingTaskMessage::TrainingUpdates(m) => {
                                let response_message =
                                    messages::ClientMessage::TrainingUpdatesMessage(m);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                            workermessages::TrainingTaskMessage::StopTraining(m) => {
                                training_tasks.remove(&m.id);
                                let response_message =
                                    messages::ClientMessage::StopTrainingMessage(m);
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
