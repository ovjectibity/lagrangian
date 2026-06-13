use serde;
use typeshare::typeshare;

use crate::commondef::testing::{
    ModelForward, StopTestMessage, TestUpdateMessage,
};
use crate::commondef::training::{
    StopTrainingMessage, TrainingUpdatesMessage,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ClientMessage {
    TrainingUpdatesMessage(TrainingUpdatesMessage),
    StopTrainingMessage(StopTrainingMessage),
    TestUpdatesMessage(TestUpdateMessage),
    StopTestMessage(StopTestMessage),
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
