use serde;
use typeshare::typeshare;

use crate::commondef::{
    testing::{
        StartTestMessage,
        StopTestMessage,
        TestUpdateMessage
    },
    training::{
        Checkpoint, TrainingMetrics,
        TrainingRunState, StopTrainingMessage,
        StartTrainingMessage
    }};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct WorkerTrainingUpdatesMessage {
    pub id: String,
    pub updates: Vec<WorkerTrainingUpdate>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct WorkerTrainingUpdate {
    pub training_run_state: TrainingRunState,
    pub logs: Vec<String>,
    pub metrics: Vec<TrainingMetrics>,
    pub checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum RemoteWorkerServerMessage {
    StartTrainingMessage(StartTrainingMessage),
    StopTrainingMessage(StopTrainingMessage),
    StartTestingMessage(StartTestMessage),
    StopTestingMessage(StopTestMessage),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum RemoteWorkerClientMessage {
    TrainingUpdatesMessage(WorkerTrainingUpdatesMessage),
    StopTrainingMessage(StopTrainingMessage),
    TestUpdatesMessage(TestUpdateMessage),
    StopTestingMessage(StopTestMessage)
}
