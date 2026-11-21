use serde;
use typeshare::typeshare;
use utoipa::ToSchema;

use crate::commondef::{
    records::{ 
        assets::{
            CheckpointRef, TrajectoryCameraFrameRef, SimulationModelRef
        }, 
        workload::{
            WorkloadLogs, WorkloadSimulationModelTrajectoryState, WorkloadUserMetrics
        }, 
        workload_graph:: {
            TestGraph,
            TrainingGraph
        }
    }
};

//For structures from records being used, the id fields will 
// remain invalid when the app server inserts it into the DB
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct WorkloadUpdates {
    pub workload_id: String,
    pub logs: Vec<WorkloadLogs>,
    pub metrics: Vec<WorkloadUserMetrics>, 
    pub checkpoints: Vec<CheckpointRef>,
    pub camera_frames:  Vec<TrajectoryCameraFrameRef>,
    pub simulation_models: Vec<SimulationModelRef>,
    pub model_states: Vec<WorkloadSimulationModelTrajectoryState>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct StopWorkloadServer {
    pub workload_id: String, 
    pub reason: WorkloadStopReasonServer
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum WorkloadStopReasonServer {
    RequestedByUser
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct StopWorkloadWorker {
    pub workload_id: String, 
    pub reason: WorkloadStopReasonWorker
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum WorkloadStopReasonWorker {
    CompletedSuccess,
    WorkerFailure(String), 
    RequestedByAppServer
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTrainingWorkerMessage {
    pub workload_id: String,
    pub training_graph: TrainingGraph,
    pub checkpoints: Vec<CheckpointRef>,
    pub simulation_models: Vec<SimulationModelRef>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTestWorkerMessage {
    pub workload_id: String,
    pub test_graph: TestGraph,
    pub checkpoints: Vec<CheckpointRef>,
    pub simulation_models: Vec<SimulationModelRef>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum RemoteWorkerServerMessage {
    StartTrainingMessage(StartTrainingWorkerMessage),
    StopTrainingMessage(StopWorkloadServer),
    StartTestingMessage(StartTestWorkerMessage),
    StopTestingMessage(StopWorkloadServer),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum RemoteWorkerClientMessage {
    TrainingUpdatesMessage(WorkloadUpdates),
    StopTrainingMessage(StopWorkloadWorker),
    TestUpdatesMessage(WorkloadUpdates),
    StopTestingMessage(StopWorkloadWorker)
}
