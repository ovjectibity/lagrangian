use crate::commondef::model;
use crate::commondef::messages::
    { ModelForward, StopTrainingMessage, TrainingUpdatesMessage };

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct SimulatorThreadLoadModelMessage {
    pub model: model::WorldBody
}

#[derive(Debug, Clone)]
pub struct SimulatorThreadSimulUpdateMessage {
    pub update: ModelForward
}

#[derive(Debug, Clone)]
pub enum TrainingTaskMessage {
    TrainingUpdates(TrainingUpdatesMessage),
    StopTraining(StopTrainingMessage)
}