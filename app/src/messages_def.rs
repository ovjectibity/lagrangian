use serde::Serialize;

use crate::model_def;

#[derive(Serialize)]
pub enum SetModelMessageType {
    TestModel,
    SimulUpdate
}

#[derive(Serialize)]
pub struct SetModelMessage {
    pub set_model: model_def::WorldBody,
    pub msg_type: SetModelMessageType
}