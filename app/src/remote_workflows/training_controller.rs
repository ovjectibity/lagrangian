//!
//! Assumptions:
//! - The worker image is already built from `worker/trainer` Dockerfile and published
//!   to a registry reachable by the EC2 instance (for private ECR, instance role should
//!   have pull permissions).
//! - The worker process inside the container starts `main.py`.
//! - Worker websocket auth uses `Authorization: Bearer <secret>`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use aws_config::Region;
use aws_sdk_ec2::Client as Ec2Client;
use aws_sdk_ec2::types::{
    Filter, IamInstanceProfileSpecification, InstanceInterruptionBehavior,
    InstanceMarketOptionsRequest, InstanceStateName, InstanceType, MarketType, ResourceType,
    SpotInstanceType, SpotMarketOptions, Tag, TagSpecification,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_util::{SinkExt, StreamExt};
use http::HeaderValue;
use tokio::net::TcpStream;
use tokio::time::sleep;
use tokio_tungstenite::MaybeTlsStream;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

use crate::commondef::remoteworkermessages::{
    Checkpoint, LoadModelMessage, StartTrainingMessage, StopTrainingMessage,
    TrainingRemoteWorkerClientMessage, TrainingRemoteWorkerServerMessage, TrainingUpdatesMessage,
};
use crate::commondef::training::{TrainingGraph, TrainingJobController};

type DynError = Box<dyn std::error::Error + Send + Sync>;
type Result<T> = std::result::Result<T, DynError>;
type WorkerWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn load_mjcf_package_zip_base64(model_path: &str) -> Result<(String, String)> {
    let model_path = Path::new(model_path);
    let model_dir = model_path.parent().ok_or("model_path has no parent directory")?;
    let default_entrypoint = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("model_path has no valid UTF-8 filename")?;
    let entrypoint = env::var("TRAINING_WORKER_MJCF_ENTRYPOINT")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default_entrypoint.to_string());

    let zip_path = match env::var("TRAINING_WORKER_MODEL_ZIP_PATH")
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
            "Prebuilt zip not found at {:?}. Set TRAINING_WORKER_MODEL_ZIP_PATH or place model.zip/{:?}.zip in {:?}.",
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

fn skip_instance_launch_terminate() -> bool {
    matches!(
        env::var("AWS_SKIP_INSTANCE_LAUNCH_TERMINATE").as_deref(),
        Ok("1")
    )
}

#[derive(Debug, Clone)]
pub struct TrainingWorkerControllerConfig {
    pub aws_region: String,
    pub ami_id: String,
    pub subnet_id: Option<String>,
    pub security_group_ids: Vec<String>,
    pub key_name: Option<String>,
    pub iam_instance_profile_arn: Option<String>,
    pub instance_name_tag: String,
    pub spot_max_price_usd: Option<String>,
    pub worker_image_uri: String,
    pub worker_container_name: String,
    pub worker_port: u16,
    pub worker_ws_path: String,
    pub worker_shared_secret: String,
    pub instance_boot_timeout: Duration,
    pub websocket_connect_timeout: Duration,
}

impl Default for TrainingWorkerControllerConfig {
    fn default() -> Self {
        Self {
            aws_region: "us-east-1".to_string(),
            ami_id: String::new(),
            subnet_id: None,
            security_group_ids: Vec::new(),
            key_name: None,
            iam_instance_profile_arn: None,
            instance_name_tag: "lagrangian-training-worker".to_string(),
            spot_max_price_usd: None,
            worker_image_uri: String::new(),
            worker_container_name: "lagrangian-training-worker".to_string(),
            worker_port: 8765,
            worker_ws_path: "/".to_string(),
            worker_shared_secret: String::new(),
            instance_boot_timeout: Duration::from_secs(8 * 60),
            websocket_connect_timeout: Duration::from_secs(2 * 60),
        }
    }
}

pub struct TrainingWorkerController {
    ec2: Ec2Client,
    config: TrainingWorkerControllerConfig,
    skip_instance_lifecycle: bool,
    instance_id: Option<String>,
    instance_public_dns: Option<String>,
    worker_ws: Option<WorkerWs>,
}

impl TrainingWorkerController {
    /// Creates controller, launches a `g4dn.xlarge` spot instance, waits for worker bootstrap,
    /// and connects to worker WSS.
    pub async fn initialize(config: TrainingWorkerControllerConfig) -> Result<Self> {
        let skip_instance_lifecycle = skip_instance_launch_terminate();
        if !skip_instance_lifecycle && config.ami_id.is_empty() {
            return Err("TrainingWorkerControllerConfig.ami_id is required".into());
        }
        if !skip_instance_lifecycle && config.worker_image_uri.is_empty() {
            return Err("TrainingWorkerControllerConfig.worker_image_uri is required".into());
        }
        if config.worker_shared_secret.is_empty() {
            return Err("TrainingWorkerControllerConfig.worker_shared_secret is required".into());
        }

        let shared_cfg = aws_config::from_env()
            .region(Region::new(config.aws_region.clone()))
            .load()
            .await;
        let ec2 = Ec2Client::new(&shared_cfg);
        let mut controller = Self {
            ec2,
            config,
            skip_instance_lifecycle,
            instance_id: None,
            instance_public_dns: None,
            worker_ws: None,
        };

        if controller.skip_instance_lifecycle {
            let (instance_id, public_dns) = controller.resolve_existing_instance().await?;
            controller.instance_id = Some(instance_id);
            controller.instance_public_dns = Some(public_dns);
        } else {
            let instance_id = controller.launch_spot_instance().await?;
            let public_dns = controller.wait_until_instance_ready(&instance_id).await?;
            controller.instance_id = Some(instance_id);
            controller.instance_public_dns = Some(public_dns);
        }
        controller.connect_worker_wss().await?;

        Ok(controller)
    }

    pub fn instance_id(&self) -> Option<&str> {
        self.instance_id.as_deref()
    }

    pub fn worker_public_dns(&self) -> Option<&str> {
        self.instance_public_dns.as_deref()
    }

    pub async fn run_training_job<F>(
        &mut self,
        training_id: String,
        training_controller: TrainingJobController,
        model_path: Option<String>,
        mjcf: String,
        checkpoint: Option<Checkpoint>,
        mut on_update: F,
    ) -> Result<()>
    where
        F: FnMut(&TrainingUpdatesMessage) + Send,
    {
        // Reconnect WS if the previous connection was closed
        if self.worker_ws.is_none() {
            self.connect_worker_wss().await?;
        } else {
            // Probe the existing connection with a ping; reconnect if stale
            let ws = self.worker_ws.as_mut().unwrap();
            if ws.send(Message::Ping(Vec::new().into())).await.is_err() {
                self.worker_ws = None;
                self.connect_worker_wss().await?;
            }
        }

        let (model_zip_base64, mjcf_entrypoint) = match model_path {
            Some(path) => match load_mjcf_package_zip_base64(&path) {
                Ok((zip_b64, entrypoint)) => (Some(zip_b64), Some(entrypoint)),
                Err(e) => {
                    println!(
                        "Could not load prebuilt zipped model package from path {:?}: {:?}. Falling back to raw mjcf.",
                        path, e
                    );
                    (None, None)
                }
            },
            None => (None, None),
        };

        self.send_to_worker(&TrainingRemoteWorkerServerMessage::LoadModelMessage(
            LoadModelMessage {
                id: training_id.clone(),
                mjcf,
                model_zip_base64,
                mjcf_entrypoint,
            },
        ))
        .await?;

        self.send_to_worker(&TrainingRemoteWorkerServerMessage::StartTrainingMessage(
            StartTrainingMessage {
                id: training_id.clone(),
                training_graph: TrainingGraph {
                    training_job_controller: training_controller,
                },
                checkpoint,
            },
        ))
        .await?;

        self.monitor_training_until_stop(&training_id, &mut on_update)
            .await?;
        Ok(())
    }

    pub async fn stop_training_and_terminate(&mut self, training_id: &str) -> Result<()> {
        let _ = self
            .send_to_worker(&TrainingRemoteWorkerServerMessage::StopTrainingMessage(
                StopTrainingMessage {
                    id: training_id.to_string(),
                },
            ))
            .await;
        self.terminate_instance().await
    }

    pub async fn terminate_instance(&mut self) -> Result<()> {
        if let Some(ws) = self.worker_ws.as_mut() {
            let _ = ws.close(None).await;
        }
        self.worker_ws = None;

        if self.skip_instance_lifecycle {
            self.instance_id = None;
            self.instance_public_dns = None;
            return Ok(());
        }

        if let Some(instance_id) = self.instance_id.take() {
            self.ec2
                .terminate_instances()
                .instance_ids(instance_id)
                .send()
                .await?;
        }
        self.instance_public_dns = None;
        Ok(())
    }

    async fn resolve_existing_instance(&self) -> Result<(String, String)> {
        let response = self
            .ec2
            .describe_instances()
            .filters(
                Filter::builder()
                    .name("tag:Name")
                    .values(self.config.instance_name_tag.clone())
                    .build(),
            )
            .filters(
                Filter::builder()
                    .name("instance-state-name")
                    .values("running")
                    .build(),
            )
            .send()
            .await?;

        let instance = response
            .reservations()
            .iter()
            .flat_map(|reservation| reservation.instances())
            .find(|instance| {
                !instance.public_dns_name().unwrap_or_default().is_empty()
                    && instance.instance_id().is_some()
            })
            .ok_or_else(|| {
                format!(
                    "AWS_SKIP_INSTANCE_LAUNCH_TERMINATE=1 but no running instance found with Name tag '{}'",
                    self.config.instance_name_tag
                )
            })?;

        let instance_id = instance
            .instance_id()
            .ok_or("Running instance is missing instance_id")?
            .to_string();
        let public_dns = instance
            .public_dns_name()
            .ok_or("Running instance is missing public DNS")?
            .to_string();
        Ok((instance_id, public_dns))
    }

    async fn launch_spot_instance(&self) -> Result<String> {
        let user_data_plain = self.build_user_data_script()?;
        let user_data_base64 = BASE64.encode(user_data_plain.as_bytes());

        let mut req = self
            .ec2
            .run_instances()
            .image_id(self.config.ami_id.clone())
            // .instance_type(InstanceType::G4dnXlarge)
            .instance_type(InstanceType::M7iFlexLarge)
            .min_count(1)
            .max_count(1)
            .user_data(user_data_base64)
            .instance_market_options(
                InstanceMarketOptionsRequest::builder()
                    .market_type(MarketType::Spot)
                    .spot_options(
                        SpotMarketOptions::builder()
                            .spot_instance_type(SpotInstanceType::OneTime)
                            .instance_interruption_behavior(InstanceInterruptionBehavior::Terminate)
                            .set_max_price(self.config.spot_max_price_usd.clone())
                            .build(),
                    )
                    .build(),
            )
            .tag_specifications(
                TagSpecification::builder()
                    .resource_type(ResourceType::Instance)
                    .tags(
                        Tag::builder()
                            .key("Name")
                            .value(self.config.instance_name_tag.clone())
                            .build(),
                    )
                    .build(),
            );

        if let Some(subnet_id) = &self.config.subnet_id {
            req = req.subnet_id(subnet_id.clone());
        }
        if !self.config.security_group_ids.is_empty() {
            req = req.set_security_group_ids(Some(self.config.security_group_ids.clone()));
        }
        if let Some(key_name) = &self.config.key_name {
            req = req.key_name(key_name.clone());
        }
        if let Some(profile_arn) = &self.config.iam_instance_profile_arn {
            req = req.iam_instance_profile(
                IamInstanceProfileSpecification::builder()
                    .arn(profile_arn.clone())
                    .build(),
            );
        }

        let resp = req.send().await?;
        let instance_id = resp
            .instances()
            .first()
            .and_then(|inst| inst.instance_id())
            .ok_or("EC2 did not return an instance id for run_instances")?
            .to_string();

        Ok(instance_id)
    }

    async fn wait_until_instance_ready(&self, instance_id: &str) -> Result<String> {
        let deadline = tokio::time::Instant::now() + self.config.instance_boot_timeout;
        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(format!(
                    "Timed out waiting for instance {instance_id} to become ready"
                )
                .into());
            }

            let resp = self
                .ec2
                .describe_instances()
                .instance_ids(instance_id)
                .send()
                .await?;

            let instance = resp
                .reservations()
                .iter()
                .flat_map(|r| r.instances())
                .next();

            if let Some(inst) = instance {
                let is_running = inst
                    .state()
                    .and_then(|s| s.name())
                    .is_some_and(|n| match n {
                        InstanceStateName::Running => true,
                        _ => false,
                    });
                let public_dns = inst.public_dns_name().unwrap_or_default().to_string();
                if is_running && !public_dns.is_empty() {
                    return Ok(public_dns);
                }
            }

            sleep(Duration::from_secs(5)).await;
        }
    }

    async fn connect_worker_wss(&mut self) -> Result<()> {
        let public_dns = self
            .instance_public_dns
            .as_deref()
            .ok_or("Missing public dns; instance is not ready")?;

        let mut path = self.config.worker_ws_path.clone();
        if !path.starts_with('/') {
            path = format!("/{path}");
        }
        let url = format!("ws://{public_dns}:{}{}", self.config.worker_port, path);

        let deadline = tokio::time::Instant::now() + self.config.websocket_connect_timeout;
        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(format!("Timed out connecting to worker WSS at {url}").into());
            }

            let mut req = url.clone().into_client_request()?;
            let auth = format!("Bearer {}", self.config.worker_shared_secret);
            req.headers_mut()
                .insert("Authorization", HeaderValue::from_str(&auth)?);

            match connect_async(req).await {
                Ok((ws, _)) => {
                    self.worker_ws = Some(ws);
                    return Ok(());
                }
                Err(_) => {
                    sleep(Duration::from_secs(3)).await;
                }
            }
        }
    }

    async fn send_to_worker(&mut self, message: &TrainingRemoteWorkerServerMessage) -> Result<()> {
        let ws = self
            .worker_ws
            .as_mut()
            .ok_or("Worker websocket is not connected")?;
        let payload = serde_json::to_string(message)?;
        ws.send(Message::Text(payload.into())).await?;
        Ok(())
    }

    async fn monitor_training_until_stop<F>(
        &mut self,
        training_id: &str,
        on_update: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&TrainingUpdatesMessage) + Send,
    {
        let ws = self
            .worker_ws
            .as_mut()
            .ok_or("Worker websocket is not connected")?;

        while let Some(msg) = ws.next().await {
            let msg = msg?;
            match msg {
                Message::Text(text) => {
                    let parsed = serde_json::from_str::<TrainingRemoteWorkerClientMessage>(&text);
                    if let Ok(ev) = parsed {
                        match ev {
                            TrainingRemoteWorkerClientMessage::TrainingUpdatesMessage(update) => {
                                on_update(&update);
                            }
                            TrainingRemoteWorkerClientMessage::StopTrainingMessage(stop) => {
                                if stop.id == training_id {
                                    return Ok(());
                                }
                            }
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

    fn build_user_data_script(&self) -> Result<String> {
        let registry = self
            .config
            .worker_image_uri
            .split('/')
            .next()
            .unwrap_or_default()
            .to_string();
        let is_ecr = registry.contains(".dkr.ecr.");

        let ecr_login = if is_ecr {
            format!(
                "aws ecr get-login-password --region {} | docker login --username AWS --password-stdin {}",
                self.config.aws_region, registry
            )
        } else {
            "true".to_string()
        };

        let scripts_root = env::var("LAGRANGIAN_SCRIPTS_ROOT")
            .map_err(|_| "Missing required env var LAGRANGIAN_SCRIPTS_ROOT")?;
        let script_path = PathBuf::from(scripts_root).join("bootstrap_training_worker.sh");
        let script_template = fs::read_to_string(&script_path).map_err(|e| {
            format!(
                "Couldn't read bootstrap training worker script at {}: {e}",
                script_path.display()
            )
        })?;

        Ok(script_template
            .replace("__ECR_LOGIN__", &ecr_login)
            .replace("__IMAGE_URI__", &self.config.worker_image_uri)
            .replace("__CONTAINER_NAME__", &self.config.worker_container_name)
            .replace("__WORKER_PORT__", &self.config.worker_port.to_string())
            .replace(
                "__WORKER_SHARED_SECRET__",
                &self.config.worker_shared_secret,
            ))
    }
}

impl Drop for TrainingWorkerController {
    fn drop(&mut self) {
        // Best-effort fail-safe notice; explicit `terminate_instance` should be preferred.
        if self.instance_id.is_some() && !self.skip_instance_lifecycle {
            eprintln!(
                "TrainingWorkerController dropped with live instance. \
                 Call `terminate_instance()` explicitly to avoid leaks."
            );
        }
    }
}
