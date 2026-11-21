//! Local worker controller: connects to a Python worker already running on the
//! local host (connect-only — it does not spawn or terminate the worker process).
//! All job logic is delegated to the shared `CommonWorkerController`.

use std::env;
use std::time::Duration;

use crate::commondef::remoteworker::remoteworkermessages::{
    RemoteWorkerClientMessage, RemoteWorkerServerMessage,
    StopWorkloadWorker,
};
use crate::workflows::common_worker_controller::{
    CommonWorkerController, CommonWorkerControllerConfig, Result,
};
use tokio::sync::{mpsc, oneshot, watch};

#[derive(Debug, Clone)]
pub struct LocalWorkerControllerConfig {
    pub host: String,
    pub port: u16,
    pub ws_path: String,
    pub shared_secret: String,
    pub websocket_connect_timeout: Duration,
}

impl Default for LocalWorkerControllerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8765,
            ws_path: "/".to_string(),
            shared_secret: String::new(),
            websocket_connect_timeout: Duration::from_secs(30),
        }
    }
}

pub struct LocalWorkerController {
    common: CommonWorkerController,
}

impl LocalWorkerController {
    pub fn new(config: LocalWorkerControllerConfig) -> Self {
        let common = CommonWorkerController::new(
            CommonWorkerControllerConfig {
                host: config.host,
                port: config.port,
                ws_path: config.ws_path,
                shared_secret: config.shared_secret,
                websocket_connect_timeout: config
                    .websocket_connect_timeout,
            },
        );
        Self { common }
    }

    /// Builds a controller from env. Connect-only: assumes a Python worker is
    /// already listening on the local host. Reuses the same port / ws-path /
    /// shared-secret env conventions as the remote worker; host defaults to
    /// 127.0.0.1 (override with `LOCAL_WORKER_HOST`).
    pub fn from_env() -> Result<Self> {
        let mut cfg = LocalWorkerControllerConfig::default();
        if let Ok(v) = env::var("LOCAL_WORKER_HOST") {
            if !v.is_empty() {
                cfg.host = v;
            }
        }
        if let Ok(v) = env::var("REMOTE_WORKER_PORT") {
            if let Ok(port) = v.parse::<u16>() {
                cfg.port = port;
            }
        }
        if let Ok(v) = env::var("REMOTE_WORKER_WS_PATH") {
            if !v.is_empty() {
                cfg.ws_path = v;
            }
        }
        cfg.shared_secret = env::var(
            "REMOTE_WORKER_SHARED_SECRET",
        )
        .map_err(|_| {
            "Missing required env var REMOTE_WORKER_SHARED_SECRET"
                .to_string()
        })?;
        Ok(Self::new(cfg))
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
}
