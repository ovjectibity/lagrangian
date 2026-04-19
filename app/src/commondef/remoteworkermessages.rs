use serde;
use typeshare::typeshare;

use crate::commondef::training::{
    FFNModel, TrainingGraph, TrainingMetrics, TrainingRunState,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingRemoteWorkerServerMessage {
    LoadModelMessage(LoadModelMessage),
    StartTrainingMessage(StartTrainingMessage),
    StopTrainingMessage(StopTrainingMessage),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTrainingMessage {
    pub id: String,
    pub training_graph: TrainingGraph,
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StopTrainingMessage {
    pub id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingRemoteWorkerClientMessage {
    TrainingUpdatesMessage(TrainingUpdatesMessage),
    StopTrainingMessage(StopTrainingMessage),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct LoadModelMessage {
    pub id: String,
    pub mjcf: String,
    pub model_zip_base64: Option<String>,
    pub mjcf_entrypoint: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingUpdatesMessage {
    pub id: String,
    pub updates: Vec<TrainingUpdate>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct Checkpoint {
    pub id: Option<String>,
    pub model_zip_base64: Option<String>,
    pub model_zip_path: Option<String>,
    pub control_model_ffn: Option<FFNModel>,
    pub value_model_ffn: Option<FFNModel>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingUpdate {
    pub training_run_state: TrainingRunState,
    pub logs: Vec<String>,
    pub metrics: Vec<TrainingMetrics>,
    pub checkpoints: Vec<Checkpoint>,
}
