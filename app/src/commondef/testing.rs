use std::collections::HashMap;

use schemars::JsonSchema;
use serde;
use typeshare::typeshare;

use crate::commondef::training::{
    ComputeInstance, ControlModel, TrainingAccelerationMode,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct TestParams {
    pub num_timesteps: i32,
    pub control_model: ControlModel,
    pub simulation_model: String,
    pub num_parallel_envs: u32
    // pub control_output_mode: ControlOutputMode,
    // pub reward_function: Option<RewardSystem>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct TestJobController {
    pub test_params: TestParams,
    pub compute_instance: ComputeInstance,
    pub acceleration_mode: TrainingAccelerationMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct TestGraph {
    pub test_job_controller: TestJobController,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTestMessage {
    pub id: String,
    pub test_graph: TestGraph,
    pub simulation_models: HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StopTestMessage {
    pub id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TestUpdateMessage {
    pub id: String,
    //The first dimension here is for the env index and 
    //the second one for the sequence of update 
    pub updates: Vec<Vec<ModelForward>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ModelForward {
    pub xpos: Vec<f64>,  //nbody x 3
    pub xquat: Vec<f64>, //nbody x 4
}