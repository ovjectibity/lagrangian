use std::collections::HashMap;

use schemars::JsonSchema;
use serde;
use typeshare::typeshare;
use utoipa::ToSchema;

use crate::commondef::training::{
    ComputeInstance, ControlModel, TrainingAccelerationMode,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct CustomTestMethodTestParams {
    pub num_timesteps: i32,
    pub num_parallel_envs: u32
    //TODO: Add reward function for eval
}

// #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
// #[typeshare]
// pub enum DatasetTaskSequencing {
//     ParallelTasksParallelEnvs,
//     SequentialTasksParallelEnvs,
//     FullySequential
// }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub enum LiberoDatasetType {
    Libero10, 
    Libero90,
    LiberoGoal,
    LiberoObject,
    LiberoSequential
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct LiberoDatasetParams {
    // pub tasks: Option<Vec<i32>>,
    pub tasks_index: Option<Vec<i32>>,
    pub libero_dataset_type: LiberoDatasetType,
    // pub tasks_sequencing: DatasetTaskSequencing,
    pub num_envs_per_task: i32
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum Dataset {
    Libero(LiberoDatasetParams)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct DatasetMethod {
    pub dataset: Dataset,
    pub control_model: ControlModel
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct CustomTestMethod {
    pub test_params: CustomTestMethodTestParams,
    pub control_model: ControlModel,
    pub simulation_model: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TestMethod {
    Custom(CustomTestMethod),
    Dataset(DatasetMethod)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct TestJobController {
    pub test_method: TestMethod, 
    pub compute_instance: ComputeInstance,
    pub acceleration_mode: TrainingAccelerationMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct StopTestMessage {
    pub id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct TestUpdateMessage {
    pub id: String,
    //The first dimension here is for the env index and 
    //the second one for the sequence of update 
    pub simulation_state: Vec<Vec<ModelForward>>,
    pub frames: Option<CameraFrames>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct CameraFrames {
    // Shape: [nenv, t, ncam, h, w, 3]
    pub frames: Vec<u8>,
    pub nenv: i32,
    pub ncam: i32,
    pub t: i32,
    pub h: i32,
    pub w: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct ModelForward {
    pub xpos: Vec<f64>,  //nbody x 3
    pub xquat: Vec<f64>, //nbody x 4
}