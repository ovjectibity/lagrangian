use serde;
use typeshare::typeshare;

use crate::commondef::testing::SimulationParams;
use crate::commondef::training::{
    TrainingMetrics, TrainingRunState,
};
use crate::commondef::experiments::CheckpointReference;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ClientMessage {
    SimulUpdateMessage(SimulUpdateMessage),
    TrainingUpdatesMessage(TrainingUpdatesMessage),
    StopTrainingMessage(StopTrainingMessage),
    StopSimulationMessage(StopSimulationMessage),
}

impl ClientMessage {
    pub fn to_binary_simul_update(id: String, update: ModelForward) -> Vec<u8> {
        let id_bytes = id.as_bytes();
        let nbody = update.xpos.len() / 3;

        let mut binary_data = Vec::new();
        // Message type
        binary_data.push(7u8);
        // ID length
        binary_data.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
        // ID
        binary_data.extend_from_slice(id_bytes);
        // Number of bodies
        binary_data.extend_from_slice(&(nbody as u16).to_le_bytes());
        // Xpos data
        for &val in &update.xpos {
            binary_data.extend_from_slice(&val.to_le_bytes());
        }
        // Xquat data
        for &val in &update.xquat {
            binary_data.extend_from_slice(&val.to_le_bytes());
        }
        binary_data
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct SimulUpdateMessage {
    pub id: String,
    pub updates: Vec<ModelForward>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ModelForward {
    pub xpos: Vec<f64>,  //nbody x 3
    pub xquat: Vec<f64>, //nbody x 4
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingUpdatesMessage {
    pub id: String,
    pub updates: Vec<TrainingUpdate>
}

#[derive(
    Debug, Clone,
    serde::Serialize, serde::Deserialize,
)]
#[typeshare]
pub struct TrainingUpdate {
    pub training_run_state: TrainingRunState,
    pub logs: Vec<String>,
    pub metrics: Vec<TrainingMetrics>,
    pub checkpoints: Vec<CheckpointReference>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartSimulationMessage {
    pub simulation_params: SimulationParams,
    pub id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StopSimulationMessage {
    pub id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StopTrainingMessage {
    pub id: String,
}
