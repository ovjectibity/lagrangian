use chrono::{DateTime, Utc};
use serde;
use typeshare::typeshare;

use crate::commondef::records::{ 
    assets::{
        CheckpointModelType, SimulationModelFileType, 
        SimulationModelType, checkpoint_ref_variant, 
        simulation_model_ref_variant, trajectory_camera_frame_ref_variant
    }, model::WorldBody, workload::{
        LaunchOrigin, LogType, WorkloadStatus, 
        WorkloadType, workload_logs_variant, 
        workload_simulation_model_trajectory_state_variant, 
        workload_user_metrics_variant, workload_variant
    }, workload_graph::{
        MLModel, TestGraph, TrainingGraph, WorkloadGraph
    }
};

//Projects API requests/responses ----------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct DeleteProjectsRequestBody {
    pub project_ids: Vec<String>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct CreateProjectResponseBody {
    pub project_id: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct GetProjectsResponseBody {
    pub project_ids: Vec<String>
}

//Models API requests/responses ----------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ListModelsRequestBody {
    pub project_id: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct GetModelsRequestBody {
    pub model_ids: Vec<String>
}

simulation_model_ref_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct SaveModelRequestBody {
        remove: [created_at, updated_at, uri_path],
        add: {
            pub file_contents: Vec<u8>
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct DeleteModelsRequestBody {
    pub model_ids: Vec<String>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ListModelsResponseBody {
    pub model_data: Vec<SimulationModelDataUser> 
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct GetModelsResponseBody {
    pub model_refs: Vec<SimulationModelRefUser>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct SaveModelResponseBody {
    pub model_id: String
}

//Checkpoints API requests/responses ----------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ListCheckpointsRequestBody {
    pub project_id: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct GetWorkloadCheckpointsRequestBody {
    pub workload_ids: Vec<String>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ListCheckpointsResponseBody {
    pub checkpoints: Vec<CheckpointRefUser> 
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct GetWorkloadCheckpointsResponseBody {
    pub checkpoints: Vec<CheckpointRefUser>
}

//Workloads API requests/responses ----------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ListWorkloadsRequestBody {
    pub project_id: Vec<String>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ListWorkloadsResponseBody {
    pub workloads: Vec<WorkloadUser>
}

workload_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct WorkloadUser {
        remove: [created_at, updated_at],
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTrainingRequestBody {
    pub project_id: String,
    pub training_graph: TrainingGraph,
    pub tags: Vec<String> 
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTestRequestBody {
    pub project_id: String,
    pub test_graph: TestGraph,
    pub tags: Vec<String> 
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTrainingResponseBody {
    pub workload_id: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTestResponseBody {
    pub workload_id: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StopWorkloadRequestBody {
    pub workload_id: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct WorkloadUpdatesResponseBody {
    pub workload_id: String,
    pub reason: Vec<WorkloadUpdateUser>
}

// WSS workload updates ----------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum WorkloadUpdatesServer {
    WorkloadStoppedMessage(WorkloadStopped),
    WorkloadUpdatesMessage(Vec<WorkloadUpdateUser>)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum WorkloadUpdatesClient {
    RequestAllUpdatesMessage
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct WorkloadStopped {
    pub workload_id: String,
    pub reason: WorkloadStoppedReasonUser
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum WorkloadStoppedReasonUser {
    CompletedSuccess,
    WorkerFailure(String), 
    RequestedByUser
}

//For structures from records being used, the id fields will 
// remain invalid when the app server inserts it into the DB
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct WorkloadUpdateUser {
    pub workload_id: String,
    pub logs: Vec<WorkloadLogsUser>,
    pub metrics: Vec<WorkloadUserMetricsUser>, 
    pub checkpoints: Vec<CheckpointRefUser>,
    pub camera_frames:  Vec<TrajectoryCameraFrameRefUser>,
    pub model_states: Vec<WorkloadSimulationModelTrajectoryStateUser>
}

// Modified workload & assets structures for user responses ----------------------------------

workload_logs_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct WorkloadLogsUser {
        remove: [user_facing, stream]
    }
}

// Simulation model state captured at a workload coordinate.
workload_user_metrics_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct WorkloadUserMetricsUser {
        remove: [metric_id, metrics_entry_id],
        add: {
            pub workload_structure_index_names: Option<Vec<String>>, 
            pub metric_name: String
        }
    }
}

// Simulation model state captured at a workload coordinate.
workload_simulation_model_trajectory_state_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct WorkloadSimulationModelTrajectoryStateUser {
        remove: [workload_structure_id], 
        add: {
            pub workload_structure_index_names: Option<Vec<String>>
        }
    }
}

// Reference to a camera frame captured at a workload state.
trajectory_camera_frame_ref_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct TrajectoryCameraFrameRefUser {
        remove: [workload_structure_id], 
        add: {
            pub workload_structure_index_names: Option<Vec<String>>
        }
    }
}

checkpoint_ref_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct CheckpointRefUser {
        remove: [created_at, updated_at, uri_path, workload_structure_id], 
        add: {
            pub workload_structure_index_names: Option<Vec<String>>
        }
    }
}

// Reference to a stored simulation model with uri path.
simulation_model_ref_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct SimulationModelRefUser {
        remove: [created_at, updated_at, uri_path],
    }
}

simulation_model_ref_variant! {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct SimulationModelDataUser {
        remove: [created_at, updated_at],
    }
}
