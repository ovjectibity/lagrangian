//! Local worker controller: connects to a Python worker already running on the
//! local host (connect-only — it does not spawn or terminate the worker process).
//! All job logic is delegated to the shared `CommonWorkerController`.

use std::env;
use std::time::Duration;

use crate::commondef::remoteworkermessages::WorkerTrainingUpdatesMessage;
use crate::commondef::testing::TestGraph;
use crate::commondef::training::{Checkpoint, TrainingJobController};
use crate::workflows::common_worker_controller::{
    CommonWorkerController, CommonWorkerControllerConfig, Result,
};

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
        let common = CommonWorkerController::new(CommonWorkerControllerConfig {
            host: config.host,
            port: config.port,
            ws_path: config.ws_path,
            shared_secret: config.shared_secret,
            websocket_connect_timeout: config.websocket_connect_timeout,
        });
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
        cfg.shared_secret = env::var("REMOTE_WORKER_SHARED_SECRET")
            .map_err(|_| "Missing required env var REMOTE_WORKER_SHARED_SECRET".to_string())?;
        Ok(Self::new(cfg))
    }

    pub async fn run_training_job<F>(
        &mut self,
        training_id: String,
        training_controller: TrainingJobController,
        model_path: Option<String>,
        checkpoints: Vec<Checkpoint>,
        on_update: F,
    ) -> Result<()>
    where
        F: FnMut(&WorkerTrainingUpdatesMessage) + Send,
    {
        self.common
            .run_training_job(training_id, training_controller, model_path, checkpoints, on_update)
            .await
    }

    pub async fn run_test_job<F>(
        &mut self,
        test_id: String,
        test_graph: TestGraph,
        model_path: Option<String>,
        on_update: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<u8>) + Send,
    {
        self.common
            .run_test_job(test_id, test_graph, model_path, on_update)
            .await
    }

    pub async fn stop_test(&mut self, test_id: &str) -> Result<()> {
        self.common.stop_test(test_id).await
    }

    pub async fn stop_training(&mut self, training_id: &str) -> Result<()> {
        self.common.stop_training(training_id).await
    }
}
