#![feature(coroutines)]
#![feature(coroutine_trait)]
#![feature(stmt_expr_attributes)]

use std::thread;
use std::sync::mpsc;
use std::fs;

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

#[macro_use]
extern crate rocket;
use rocket_ws as ws;
use rocket::fs::FileServer;
use rocket::futures::{SinkExt, StreamExt};

pub mod simulator;
pub mod mujoco;
pub mod model_def;
pub mod messages_def;

// --- API Routes ---
#[get("/hello")]
fn api_hello() -> &'static str {
    "Hello from Rocket!"
}

#[get("/viewport")]
fn viewport_channel(ws: ws::WebSocket) -> ws::Channel<'static> {
    let mut run_dynamics = #[coroutine] || {
        let (tx,rx) = 
            mpsc::channel::<Result<model_def::WorldBody,rocket::Error>>();
        let dyn_th = thread::spawn(move || {
            unsafe {
                if let Some((m, d)) = 
                simulator::load_model(
                    "/Users/avi/Documents/manual/lagrangian/app/assets/universal_robots_ur10e/ur10e.xml".
                    to_string()) {
                    while (*d).time < 20.0 {
                        //TODO: Get the model data as messages at each simulation step
                        simulator::run_dynamics_step(m, d);
                        if let Some(b) = 
                        simulator::create_model(m, d) {
                            tx.send(Ok(b));
                        } else {
                            println!("Could not get the model from create_model");
                        }
                    }
                    simulator::cleanup_model(m, d);
                }
            }
        });
        for msg in rx {
            // yield msg
            if let Ok(body) = msg {
                yield body;
            } else {
                println!("Got an error extracting the body in the main thread");
            }
        }
        return;
    };

    ws.channel(move |mut stream| {
        Box::pin(async move {
            while let Some(message) = 
                stream.next().await {
                match message {
                    Ok(ws::Message::Text(text)) => {
                        if text == "get_test_model" {
                            let world_body = 
                                simulator::get_test_model();
                            let response_message = 
                            messages_def::SetModelMessage {
                                set_model: world_body,
                                msg_type: messages_def::SetModelMessageType::TestModel
                            };
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
                                messages_def::SetModelMessage {
                                    set_model: wb,
                                    msg_type: messages_def::SetModelMessageType::TestModel
                                };
                                let json_response = 
                                    serde_json::to_string(&response_message)
                                    .unwrap_or_else(|_| "{}".to_string());
                                if let Err(e) = fs::write("target/test_model_2.json", json_response.clone()) {
                                    println!("Could not write the test model 2 JSON to file {:?}",e);
                                }
                                let _ = stream.send(
                                    ws::Message::Text(json_response)).await;
                            }
                        }  else if text == "get_test_model_3" {
                            if let Ok(json_response) = 
                                fs::read_to_string("target/test_model_3.json") {
                                let _ = stream.send(
                                    ws::Message::Text(json_response)).await;
                            }
                        } else if text == "run_simul" {
                            //TODO: Can the loop run async?
                            let mut simul_complete = false;
                            while !simul_complete {
                                match Pin::new(&mut run_dynamics).resume(()) {
                                    CoroutineState::Yielded(b) =>  {
                                    let response_message = 
                                    messages_def::SetModelMessage {
                                        set_model: b,
                                        msg_type: messages_def::SetModelMessageType::SimulUpdate
                                    };
                                    let json_response = 
                                        serde_json::to_string(&response_message)
                                        .unwrap_or_else(|_| "{}".to_string());
                                    let _ = stream.send(
                                        ws::Message::Text(json_response)).await;
                                    },
                                    CoroutineState::Complete(()) => {
                                        simul_complete = true;
                                    }
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
