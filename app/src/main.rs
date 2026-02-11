use std::fs;

#[macro_use]
extern crate rocket;
use rocket::tokio;
use rocket::tokio::sync::mpsc;
use rocket_ws as ws;
use rocket::fs::FileServer;
use rocket::futures::{SinkExt, StreamExt};

pub mod simulator;
pub mod mujoco;
pub mod model;
pub mod messages;
pub mod training;
pub mod workflowgraph;
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
        mpsc::channel::<messages::SimulatorThreadMessage>(32);
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
                                        simulator::load_test_model(
                                            "/Users/avi/Documents/manual/lagrangian/app/assets/universal_robots_ur10e/ur10e.xml".
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
                            messages::SimulatorThreadMessage::LoadModel(m) => {
                                let response_message = messages::ClientMessage::new_load_model_message(
                                    "1".to_string(), m.model);
                                let json_response = serde_json::to_string(&response_message)
                                    .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e));
                                let _ = stream.send(ws::Message::Text(json_response)).await;
                            },
                            messages::SimulatorThreadMessage::SimulUpdate(m) => {
                                let binary_data = messages::ClientMessage::to_binary_simul_update(
                                    "1".to_string(), m.update
                                );
                                let _ = stream.send(ws::Message::Binary(binary_data)).await;
                            }
                        };
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
