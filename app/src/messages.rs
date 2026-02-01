use serde::Serialize;
use crate::model;

#[derive(Serialize)]
pub enum ClientMessage {
    SetModelMessage(SetModelMessage),
    LoadModelMessage(LoadModelMessage),
    SimulUpdateMessage(SimulUpdateMessage)
}

impl ClientMessage {
    pub fn new_set_model_message(wb: model::WorldBody) -> Self {
        ClientMessage::SetModelMessage(
            SetModelMessage { 
                set_model: wb 
            }
        )
    }

    pub fn new_load_model_message(id: String, 
        wb: model::WorldBody) -> Self {
        ClientMessage::LoadModelMessage(
            LoadModelMessage {
                id: id,
                model: wb
            }
        )
    }

    pub fn new_simul_update_message(id: String, 
        updates: Vec<ModelForward>) -> Self {
        ClientMessage::SimulUpdateMessage(
            SimulUpdateMessage {
                id: id,
                updates: updates
            }
        )
    }
}

#[derive(Serialize)]
pub struct SetModelMessage {
    pub set_model: model::WorldBody
}

#[derive(Serialize)]
pub struct LoadModelMessage {
    pub id: String,
    pub model: model::WorldBody
}

#[derive(Serialize)]
pub struct SimulUpdateMessage {
    pub id: String,
    pub updates: Vec<ModelForward>
}

#[derive(Serialize)]
pub struct ModelForward {
    pub xpos: Vec<f64>, //nbody x 3 
    pub xquat: Vec<f64>, //nbody x 4
}

pub enum SimulatorThreadMessage {
    LoadModel(SimulatorThreadLoadModelMessage),
    SimulUpdate(SimulatorThreadSimulUpdateMessage)
}

impl SimulatorThreadMessage {
    pub fn new_load_model_message(wb: model::WorldBody) -> Self {
        SimulatorThreadMessage::LoadModel(
            SimulatorThreadLoadModelMessage { 
                model: wb 
            }
        )
    }

    pub fn new_simul_update_message(xpos: Vec<f64>, xquat: Vec<f64>) -> Self {
        SimulatorThreadMessage::SimulUpdate(
            SimulatorThreadSimulUpdateMessage { 
                update: ModelForward { 
                    xpos: xpos, 
                    xquat: xquat
                } 
            } 
        )
    }
}

pub struct SimulatorThreadLoadModelMessage {
    pub model: model::WorldBody
}

pub struct SimulatorThreadSimulUpdateMessage {
    pub update: ModelForward
}