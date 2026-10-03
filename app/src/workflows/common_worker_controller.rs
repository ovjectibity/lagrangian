//! Shared worker-interaction logic for both local and remote controllers.
//!
//! `CommonWorkerController` owns the websocket connection to a Python worker and
//! the protocol on top of it (connect/send/monitor + the run_*_job bodies that
//! package a model, send a Start message, and stream updates until stop). It is
//! transport-agnostic about *where* the worker runs: the `host` in its config is
//! `localhost` for the local controller and the EC2 public DNS for the remote
//! controller. Lifecycle (spawning/terminating the worker) lives in the wrappers.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use http::HeaderValue;
use tokio::net::TcpStream;
use tokio::time::sleep;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::MaybeTlsStream;
use tokio_tungstenite::WebSocketStream;

use crate::commondef::remoteworkermessages::{
    RemoteWorkerClientMessage, RemoteWorkerServerMessage, WorkerTrainingUpdatesMessage,
};
use crate::commondef::testing::{StartTestMessage, StopTestMessage, TestGraph};
use crate::commondef::training::{
    Checkpoint, StartTrainingMessage, StopTrainingMessage, TrainingGraph, TrainingJobController,
    TrainingMethod,
};

pub type DynError = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, DynError>;
pub type WorkerWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn find_xml_in_zip_root(zip_bytes: &[u8]) -> Result<String> {
    use std::io::Cursor;
    let cursor = Cursor::new(zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if !name.contains('/') && name.ends_with(".xml") {
            return Ok(name);
        }
    }
    Err("No XML file at zip root".into())
}

/// Loads a MuJoCo model package as a base64 zip + the root entrypoint XML name.
pub fn load_mjcf_package_zip_base64(model_path: &str) -> Result<(String, String)> {
    let path = Path::new(model_path);

    if path.extension().and_then(|e| e.to_str()) == Some("zip") {
        let zip_bytes = fs::read(path)?;
        let entrypoint = find_xml_in_zip_root(&zip_bytes)?;
        println!(
            "Using model zip directly {:?} ({} bytes), entrypoint={}",
            path,
            zip_bytes.len(),
            entrypoint
        );
        return Ok((BASE64.encode(&zip_bytes), entrypoint));
    }

    let model_path = Path::new(model_path);
    let model_dir = model_path
        .parent()
        .ok_or("model_path has no parent directory")?;
    let default_entrypoint = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("model_path has no valid UTF-8 filename")?;
    let entrypoint = env::var("REMOTE_WORKER_MJCF_ENTRYPOINT")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default_entrypoint.to_string());

    let zip_path = match env::var("REMOTE_WORKER_MODEL_ZIP_PATH")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        Some(path) => PathBuf::from(path),
        None => {
            let explicit_default = model_dir.join("model.zip");
            if explicit_default.exists() {
                explicit_default
            } else {
                let stem = model_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or("model_path has no valid UTF-8 stem")?;
                model_dir.join(format!("{stem}.zip"))
            }
        }
    };

    if !zip_path.exists() {
        return Err(format!(
            "Prebuilt zip not found at {:?}. Set REMOTE_WORKER_MODEL_ZIP_PATH or place model.zip/{:?}.zip in {:?}.",
            zip_path,
            model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model"),
            model_dir
        )
        .into());
    }

    let zip_bytes = fs::read(&zip_path)?;
    println!(
        "Using prebuilt model zip {:?} ({} bytes), entrypoint={}",
        zip_path,
        zip_bytes.len(),
        entrypoint
    );
    Ok((BASE64.encode(zip_bytes), entrypoint))
}

#[derive(Debug, Clone)]
pub struct CommonWorkerControllerConfig {
    pub host: String,
    pub port: u16,
    pub ws_path: String,
    pub shared_secret: String,
    pub websocket_connect_timeout: Duration,
}

/// Owns the websocket to a Python worker and the job protocol on top of it.
pub struct CommonWorkerController {
    config: CommonWorkerControllerConfig,
    worker_ws: Option<WorkerWs>,
}

impl CommonWorkerController {
    pub fn new(config: CommonWorkerControllerConfig) -> Self {
        Self {
            config,
            worker_ws: None,
        }
    }

    pub fn set_host(&mut self, host: String) {
        self.config.host = host;
    }

    /// Connects to `ws://{host}:{port}{ws_path}` with `Authorization: Bearer <secret>`,
    /// retrying until the connect timeout elapses.
    pub async fn connect_worker_wss(&mut self) -> Result<()> {
        let mut path = self.config.ws_path.clone();
        if !path.starts_with('/') {
            path = format!("/{path}");
        }
        let url = format!("ws://{}:{}{}", self.config.host, self.config.port, path);

        // Cap each individual connect attempt so a blackholed SYN (e.g. a
        // security group silently dropping packets rather than sending RST)
        // can't block past the loop's own deadline check — without this, a
        // single hung attempt can exceed websocket_connect_timeout entirely
        // since the deadline is only re-checked *between* attempts.
        const PER_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(10);

        let deadline = tokio::time::Instant::now() + self.config.websocket_connect_timeout;
        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(format!("Timed out connecting to worker WSS at {url}").into());
            }

            let mut req = url.clone().into_client_request()?;
            let auth = format!("Bearer {}", self.config.shared_secret);
            req.headers_mut()
                .insert("Authorization", HeaderValue::from_str(&auth)?);

            match tokio::time::timeout(PER_ATTEMPT_TIMEOUT, connect_async(req)).await {
                Ok(Ok((ws, _))) => {
                    println!("Successfully connected to worker node at {url}");
                    self.worker_ws = Some(ws);
                    return Ok(());
                }
                Ok(Err(e)) => {
                    println!("Connect attempt to {url} failed: {e:?}");
                    sleep(Duration::from_secs(3)).await;
                }
                Err(_) => {
                    println!(
                        "Connect attempt to {url} did not complete within {:?} (likely network/firewall issue)",
                        PER_ATTEMPT_TIMEOUT
                    );
                }
            }
        }
    }

    /// Closes and drops the current websocket, if any.
    pub async fn close(&mut self) {
        if let Some(ws) = self.worker_ws.as_mut() {
            let _ = ws.close(None).await;
        }
        self.worker_ws = None;
    }

    async fn send_to_worker(&mut self, message: &RemoteWorkerServerMessage) -> Result<()> {
        let ws = self
            .worker_ws
            .as_mut()
            .ok_or("Worker websocket is not connected")?;
        let payload = serde_json::to_string(message)?;
        ws.send(Message::Text(payload.into())).await?;
        Ok(())
    }

    pub async fn run_training_job<F>(
        &mut self,
        training_id: String,
        training_controller: TrainingJobController,
        model_path: Option<String>,
        checkpoints: Vec<Checkpoint>,
        mut on_update: F,
    ) -> Result<()>
    where
        F: FnMut(&WorkerTrainingUpdatesMessage) + Send,
    {
        // Always reconnect before each job — the previous connection may be in a
        // half-open state where pings succeed but data sends hang.
        self.close().await;
        self.connect_worker_wss().await?;

        let model_zip_base64 = match model_path {
            Some(path) => match load_mjcf_package_zip_base64(&path) {
                Ok((zip_b64, _entrypoint)) => Some(zip_b64),
                Err(e) => {
                    println!(
                        "Could not load prebuilt zipped model package from path {:?}: {:?}. Falling back to simulation_models.",
                        path, e
                    );
                    None
                }
            },
            None => None,
        };

        let sim_model_id = match &training_controller.training_method {
            TrainingMethod::PPO(p) => p.simulation_model.clone(),
        };
        let mut simulation_models = std::collections::HashMap::new();
        if let Some(ref zip_b64) = model_zip_base64 {
            simulation_models.insert(sim_model_id, zip_b64.clone());
        }

        self.send_to_worker(&RemoteWorkerServerMessage::StartTrainingMessage(
            StartTrainingMessage {
                id: training_id.clone(),
                training_graph: TrainingGraph {
                    training_job_controller: training_controller,
                },
                checkpoints,
                simulation_models,
            },
        ))
        .await?;

        self.monitor_training_until_stop(&training_id, &mut on_update)
            .await?;
        Ok(())
    }

    pub async fn run_test_job<F>(
        &mut self,
        test_id: String,
        test_graph: TestGraph,
        model_path: Option<String>,
        mut on_update: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<u8>) + Send,
    {
        self.close().await;
        self.connect_worker_wss().await?;

        let model_zip_base64 = match model_path {
            Some(path) => match load_mjcf_package_zip_base64(&path) {
                Ok((zip_b64, _entrypoint)) => Some(zip_b64),
                Err(e) => {
                    println!(
                        "Could not load model zip for test from {:?}: {:?}. \
                         Falling back to empty simulation_models.",
                        path, e
                    );
                    None
                }
            },
            None => None,
        };

        let mut simulation_models = std::collections::HashMap::new();
        if let (crate::commondef::testing::TestMethod::Custom(method), Some(zip_b64)) = (
            &test_graph.test_job_controller.test_method,
            model_zip_base64,
        ) {
            simulation_models.insert(method.simulation_model.clone(), zip_b64);
        }

        println!(
            "Sending StartTestingMessage to remote worker {:?}",
            self.config
        );
        self.send_to_worker(&RemoteWorkerServerMessage::StartTestingMessage(
            StartTestMessage {
                id: test_id.clone(),
                test_graph,
                simulation_models,
            },
        ))
        .await?;

        self.monitor_testing_until_stop(&test_id, &mut on_update)
            .await?;
        Ok(())
    }

    pub async fn stop_test(&mut self, test_id: &str) -> Result<()> {
        self.send_to_worker(&RemoteWorkerServerMessage::StopTestingMessage(
            StopTestMessage {
                id: test_id.to_string(),
            },
        ))
        .await
    }

    pub async fn stop_training(&mut self, training_id: &str) -> Result<()> {
        self.send_to_worker(&RemoteWorkerServerMessage::StopTrainingMessage(
            StopTrainingMessage {
                id: training_id.to_string(),
            },
        ))
        .await
    }

    async fn monitor_training_until_stop<F>(
        &mut self,
        training_id: &str,
        on_update: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&WorkerTrainingUpdatesMessage) + Send,
    {
        let ws = self
            .worker_ws
            .as_mut()
            .ok_or("Worker websocket is not connected")?;

        while let Some(msg) = ws.next().await {
            let msg = msg?;
            match msg {
                Message::Text(text) => {
                    let parsed = serde_json::from_str::<RemoteWorkerClientMessage>(&text);
                    if let Ok(ev) = parsed {
                        match ev {
                            RemoteWorkerClientMessage::TrainingUpdatesMessage(update) => {
                                on_update(&update);
                            }
                            RemoteWorkerClientMessage::StopTrainingMessage(stop) => {
                                if stop.id == training_id {
                                    return Ok(());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Message::Close(_) => {
                    return Err("Worker websocket closed before training completion".into());
                }
                Message::Ping(payload) => {
                    ws.send(Message::Pong(payload)).await?;
                }
                _ => {}
            }
        }

        Err("Worker websocket stream ended before training completion".into())
    }

    async fn monitor_testing_until_stop<F>(
        &mut self,
        test_id: &str,
        on_update: &mut F,
    ) -> Result<()>
    where
        F: FnMut(Vec<u8>) + Send,
    {
        let ws = self
            .worker_ws
            .as_mut()
            .ok_or("Worker websocket is not connected")?;

        while let Some(msg) = ws.next().await {
            let msg = msg?;
            match msg {
                // Protobuf `TestingStreamMessage` frame — forwarded verbatim to
                // SDK subscribers; the server never decodes it.
                Message::Binary(payload) => {
                    on_update(payload.to_vec());
                }
                // Control messages (e.g. StopTesting) stay JSON text.
                Message::Text(text) => {
                    if let Ok(RemoteWorkerClientMessage::StopTestingMessage(stop)) =
                        serde_json::from_str::<RemoteWorkerClientMessage>(&text)
                    {
                        if stop.id == test_id {
                            return Ok(());
                        }
                    }
                }
                Message::Close(_) => {
                    return Err("Worker websocket closed before test completion".into());
                }
                Message::Ping(payload) => {
                    ws.send(Message::Pong(payload)).await?;
                }
                _ => {}
            }
        }

        Err("Worker websocket stream ended before test completion".into())
    }
}
