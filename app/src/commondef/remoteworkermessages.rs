use serde;
use typeshare::typeshare;

use crate::commondef::training::{TrainingGraph, TrainingUpdate};

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
