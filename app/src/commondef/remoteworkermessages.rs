use serde;
use typeshare::typeshare;

use crate::commondef::model;
use crate::commondef::training::{TrainingMethod, TrainingMetrics};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingRemoteWorkerServerMessage {
    LoadModelMessage,
    StartTrainingMessage(StartTrainingMessage),
    StopTrainingMessage(StopTrainingMessage)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTrainingMessage {
    pub id: String,
    pub training_workflow: TrainingMethod,
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
    pub model: model::WorldBody,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingUpdatesMessage {
    pub id: String,
    pub metrics: Vec<TrainingMetrics>,
    pub logs: Vec<String>,
}
