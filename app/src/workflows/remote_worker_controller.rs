//! Remote worker controller: launches an EC2 spot instance running the Python
//! worker, then drives jobs over the shared `CommonWorkerController`.
//!
//! Assumptions:
//! - The worker image is already built from `worker/worker-py` Dockerfile and published
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
    Filter, IamInstanceProfileSpecification,
    InstanceInterruptionBehavior, InstanceMarketOptionsRequest,
    InstanceStateName, InstanceType, MarketType, ResourceType,
    SpotInstanceType, SpotMarketOptions, Tag, TagSpecification,
};
use aws_sdk_ec2::types::{
    HttpTokensState, InstanceMetadataOptionsRequest,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use tokio::time::sleep;

use crate::commondef::remoteworker::remoteworkermessages::{
    RemoteWorkerClientMessage, RemoteWorkerServerMessage,
    StopWorkloadWorker,
};
use crate::workflows::common_worker_controller::{
    CommonWorkerController, CommonWorkerControllerConfig, Result,
};
use tokio::sync::{mpsc, oneshot, watch};

fn skip_instance_launch_terminate() -> bool {
    matches!(
        env::var("AWS_SKIP_INSTANCE_LAUNCH_TERMINATE").as_deref(),
        Ok("1")
    )
}

#[derive(Debug, Clone)]
pub struct RemoteWorkerControllerConfig {
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

impl Default for RemoteWorkerControllerConfig {
    fn default() -> Self {
        Self {
            aws_region: "us-east-1".to_string(),
            ami_id: String::new(),
            subnet_id: None,
            security_group_ids: Vec::new(),
            key_name: None,
            iam_instance_profile_arn: None,
            instance_name_tag: "lagrangian-training-worker"
                .to_string(),
            spot_max_price_usd: None,
            worker_image_uri: String::new(),
            worker_container_name: "lagrangian-training-worker"
                .to_string(),
            worker_port: 8765,
            worker_ws_path: "/".to_string(),
            worker_shared_secret: String::new(),
            instance_boot_timeout: Duration::from_secs(8 * 60),
            websocket_connect_timeout: Duration::from_secs(
                2 * 60,
            ),
        }
    }
}

impl RemoteWorkerControllerConfig {
    /// Connection-only config for the inner `CommonWorkerController`. `host` is
    /// filled in once the instance's public DNS is known.
    fn common_config(&self) -> CommonWorkerControllerConfig {
        CommonWorkerControllerConfig {
            host: String::new(),
            port: self.worker_port,
            ws_path: self.worker_ws_path.clone(),
            shared_secret: self.worker_shared_secret.clone(),
            websocket_connect_timeout: self
                .websocket_connect_timeout,
        }
    }
}

pub struct RemoteWorkerController {
    common: CommonWorkerController,
    ec2: Ec2Client,
    config: RemoteWorkerControllerConfig,
    skip_instance_lifecycle: bool,
    instance_id: Option<String>,
    instance_public_dns: Option<String>,
}

impl RemoteWorkerController {
    /// Creates controller, launches a spot instance, waits for worker bootstrap,
    /// and connects to worker WSS.
    pub async fn initialize(
        config: RemoteWorkerControllerConfig,
    ) -> Result<Self> {
        let skip_instance_lifecycle =
            skip_instance_launch_terminate();
        if !skip_instance_lifecycle && config.ami_id.is_empty() {
            return Err(
                "RemoteWorkerControllerConfig.ami_id is required"
                    .into(),
            );
        }
        if !skip_instance_lifecycle
            && config.worker_image_uri.is_empty()
        {
            return Err("RemoteWorkerControllerConfig.worker_image_uri is required".into());
        }
        if config.worker_shared_secret.is_empty() {
            return Err("RemoteWorkerControllerConfig.worker_shared_secret is required".into());
        }

        let shared_cfg = aws_config::defaults(
            aws_config::BehaviorVersion::latest(),
        )
        .region(Region::new(config.aws_region.clone()))
        .load()
        .await;
        let ec2 = Ec2Client::new(&shared_cfg);
        let common =
            CommonWorkerController::new(config.common_config());
        let mut controller = Self {
            common,
            ec2,
            config,
            skip_instance_lifecycle,
            instance_id: None,
            instance_public_dns: None,
        };

        if controller.skip_instance_lifecycle {
            let (instance_id, public_dns) =
                controller.resolve_existing_instance().await?;
            controller.instance_id = Some(instance_id);
            controller.instance_public_dns = Some(public_dns);
        } else {
            let instance_id =
                controller.launch_spot_instance().await?;
            controller.instance_id = Some(instance_id.clone());
            match controller
                .wait_until_instance_ready(&instance_id)
                .await
            {
                Ok(public_dns) => {
                    controller.instance_public_dns =
                        Some(public_dns)
                }
                Err(error) => {
                    let _ = controller.terminate_instance().await;
                    return Err(error);
                }
            }
        }

        let public_dns =
            controller.instance_public_dns.clone().ok_or(
                "Missing public dns; instance is not ready",
            )?;
        controller.common.set_host(public_dns);
        if let Err(error) =
            controller.common.connect_worker_wss().await
        {
            let _ = controller.terminate_instance().await;
            return Err(error);
        }

        Ok(controller)
    }

    pub fn instance_id(&self) -> Option<&str> {
        self.instance_id.as_deref()
    }

    pub fn worker_public_dns(&self) -> Option<&str> {
        self.instance_public_dns.as_deref()
    }

    /// Execute the shared worker protocol with cancellation.
    pub async fn run_job(
        &mut self,
        start: RemoteWorkerServerMessage,
        stop: watch::Receiver<bool>,
        updates: mpsc::Sender<RemoteWorkerClientMessage>,
        launched: oneshot::Sender<()>,
    ) -> Result<StopWorkloadWorker> {
        self.common.run_job(start, stop, updates, launched).await
    }

    pub async fn terminate_instance(&mut self) -> Result<()> {
        self.common.close().await;

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

    async fn resolve_existing_instance(
        &self,
    ) -> Result<(String, String)> {
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
        let user_data_base64 =
            BASE64.encode(user_data_plain.as_bytes());

        let mut req = self
            .ec2
            .run_instances()
            .image_id(self.config.ami_id.clone())
            .instance_type(InstanceType::G4dnXlarge)
            .metadata_options(
                InstanceMetadataOptionsRequest::builder()
                    .http_tokens(HttpTokensState::Required)
                    // Containerized boto3 must reach the instance role.
                    .http_put_response_hop_limit(2)
                    .build(),
            )
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
            req = req.set_security_group_ids(Some(
                self.config.security_group_ids.clone(),
            ));
        }
        if let Some(key_name) = &self.config.key_name {
            req = req.key_name(key_name.clone());
        }
        if let Some(profile_arn) =
            &self.config.iam_instance_profile_arn
        {
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

    async fn wait_until_instance_ready(
        &self,
        instance_id: &str,
    ) -> Result<String> {
        let deadline = tokio::time::Instant::now()
            + self.config.instance_boot_timeout;
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
                let public_dns = inst
                    .public_dns_name()
                    .unwrap_or_default()
                    .to_string();
                if is_running && !public_dns.is_empty() {
                    return Ok(public_dns);
                }
            }

            sleep(Duration::from_secs(5)).await;
        }
    }

    fn build_user_data_script(&self) -> Result<String> {
        let scripts_root =
            PathBuf::from(env::var("LAGRANGIAN_SCRIPTS_ROOT")?);
        render_worker_bootstrap(
            &self.config,
            &scripts_root,
            &env::var("S3_BUCKET")?,
            &env::var("HF_TOKEN").unwrap_or_default(),
            &env::var("CUDA_DEBUG_MODE")
                .unwrap_or_else(|_| "0".into()),
        )
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn render_worker_bootstrap(
    config: &RemoteWorkerControllerConfig,
    scripts_root: &Path,
    bucket: &str,
    hf_token: &str,
    cuda_debug: &str,
) -> Result<String> {
    for value in [hf_token, config.worker_shared_secret.as_str()]
    {
        if value.contains(['\n', '\r', '\0']) {
            return Err(
                "Worker credentials must be single-line values"
                    .into(),
            );
        }
    }
    let directory = scripts_root.join("worker");
    let template = fs::read_to_string(
        directory.join("bootstrap_remote_worker.sh"),
    )?;
    let downloader = fs::read_to_string(
        directory.join("download_pi0_model.sh"),
    )?;
    let registry = config
        .worker_image_uri
        .split('/')
        .next()
        .ok_or("Invalid worker image URI")?;
    let login = if registry.contains(".dkr.ecr.") {
        format!(
            "aws ecr get-login-password --region {} | \
             docker login --username AWS --password-stdin {}",
            shell_quote(&config.aws_region),
            shell_quote(registry),
        )
    } else {
        "true".into()
    };
    let script =
        template
            .replace("set -euxo pipefail", "set -euo pipefail")
            .replace("__ECR_LOGIN__", &login)
            .replace(
                "__IMAGE_URI__",
                &shell_quote(&config.worker_image_uri),
            )
            .replace(
                "__CONTAINER_NAME__",
                &shell_quote(&config.worker_container_name),
            )
            .replace(
                "__WORKER_PORT__",
                &config.worker_port.to_string(),
            )
            .replace(
                "__WORKER_SHARED_SECRET__",
                &config.worker_shared_secret,
            )
            .replace(
                "export HF_TOKEN=__HF_TOKEN__",
                &format!(
                    "export HF_TOKEN={}",
                    shell_quote(hf_token),
                ),
            )
            .replace("__HF_TOKEN__", hf_token)
            .replace(
                "__CUDA_DEBUG_MODE__",
                if cuda_debug == "1" { "1" } else { "0" },
            );
    let script = format!(
        "#!/bin/bash\nset -euo pipefail\nexport S3_BUCKET={}\n\
         cat >/tmp/download_pi0_model.sh <<'LAGRANGIAN_PI0_DOWNLOAD'\n\
         {}\nLAGRANGIAN_PI0_DOWNLOAD\n{}",
        shell_quote(bucket),
        downloader,
        script.strip_prefix("#!/bin/bash\n").unwrap_or(&script),
    );
    if script.len() > 16_384 {
        return Err(
            "Worker user data exceeds the EC2 size limit".into(),
        );
    }
    Ok(script)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_bootstrap_from_the_worker_script_directory() {
        let config = RemoteWorkerControllerConfig {
            worker_image_uri: "registry.local/worker:latest"
                .into(),
            worker_shared_secret: "test-secret".into(),
            ..Default::default()
        };
        let script = render_worker_bootstrap(
            &config,
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../scripts"),
            "test-bucket",
            "hf_test",
            "0",
        )
        .unwrap();
        assert!(!script.contains("__WORKER_"));
        assert!(!script.contains("__HF_TOKEN__"));
        assert!(!script.contains("__CUDA_DEBUG_MODE__"));
        assert!(
            script.contains("export S3_BUCKET='test-bucket'")
        );
        assert!(
            script.contains("cat >/tmp/download_pi0_model.sh")
        );
        assert!(!script.contains("set -euxo"));
        let mut child = std::process::Command::new("bash")
            .arg("-n")
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(script.as_bytes())
            .unwrap();
        assert!(child.wait().unwrap().success());
    }
}

impl Drop for RemoteWorkerController {
    fn drop(&mut self) {
        // Best-effort fail-safe notice; explicit `terminate_instance` should be preferred.
        if self.instance_id.is_some()
            && !self.skip_instance_lifecycle
        {
            eprintln!(
                "RemoteWorkerController dropped with live instance. \
                 Call `terminate_instance()` explicitly to avoid leaks."
            );
        }
    }
}
