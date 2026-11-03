pub mod checkpoints;
pub mod experiments;
pub mod models;
pub mod reward_validation;
pub mod testing;
pub mod training;

pub use app_proc_macros::route_doc;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use rocket::serde::json::Json;
use rocket::tokio::sync::{broadcast, Mutex as AsyncMutex};
use rocket::tokio::task::JoinHandle;
use utoipa::OpenApi;

fn has_contact_pairs(
    input: &crate::commondef::training::PolicyInputSpec,
) -> bool {
    use crate::commondef::training::{
        InputState, InputStateItemSensors, PolicyInputSpec,
    };

    let PolicyInputSpec::StandardInputSpec(spec) = input;
    spec.states.iter().any(|state| {
        matches!(
            state,
            InputState::InputStateSensors(
                InputStateItemSensors::ContactPairs
            )
        )
    })
}

#[cfg(test)]
mod policy_input_tests {
    use super::has_contact_pairs;
    use crate::commondef::training::{
        InputState, InputStateItemSensors, PolicyInputSpec,
        StandardInputSpec,
    };

    #[test]
    fn detects_contact_pairs_in_sensor_list() {
        let mut spec = StandardInputSpec {
            base_body_name: None,
            states: vec![InputState::InputStateSensors(
                InputStateItemSensors::Cameras,
            )],
        };
        assert!(!has_contact_pairs(
            &PolicyInputSpec::StandardInputSpec(spec.clone())
        ));
        spec.states.push(InputState::InputStateSensors(
            InputStateItemSensors::ContactPairs,
        ));
        assert!(has_contact_pairs(
            &PolicyInputSpec::StandardInputSpec(spec)
        ));
    }
}

// ── OpenAPI spec ──────────────────────────────

#[derive(OpenApi)]
#[openapi(
    paths(
        models::get_default_model,
        models::list_models,
        models::save_model,
        checkpoints::list_checkpoints,
        checkpoints::get_checkpoint,
        experiments::list_experiments,
        experiments::create_experiment,
        experiments::list_runs,
        experiments::create_run,
        experiments::get_run,
        experiments::get_run_checkpoints,
        training::start_training,
        training::stop_training,
        training::training_updates,
        testing::start_test,
        testing::stop_test,
        testing::test_updates,
    ),
    components(schemas(
        models::DefaultModelResponse,
        models::ModelsResponse,
        models::SaveModelResponse,
        checkpoints::CheckpointsResponse,
        checkpoints::CheckpointResponse,
        experiments::ExperimentsResponse,
        experiments::ExperimentResponse,
        experiments::RunsResponse,
        experiments::RunResponse,
        experiments::CheckpointsResponse,
        experiments::CreateExperimentBody,
        training::StartTrainingBody,
        training::TrainingStartedResponse,
        training::TrainingStoppedResponse,
        testing::StartTestBody,
        testing::TestStartedResponse,
        testing::TestStoppedResponse,
        crate::commondef::model::WorldBody,
        crate::commondef::model::Body,
        crate::commondef::model::Geom,
        crate::commondef::model::GeomType,
        crate::commondef::record
            ::TrainingExperiment,
        crate::commondef::record
            ::TrainingExperimentRun,
        crate::commondef::record::TrainingRunStatus,
        crate::commondef::record::ModelReference,
        crate::commondef::record
            ::CheckpointReference,
        crate::commondef::training::MLPModel,
        crate::commondef::training::VLAModel,
        crate::commondef::training::ControlModel,
        crate::commondef::training::ValueModel,
        crate::commondef::training::MLModel,
        crate::commondef::training::PolicyInputSpec,
        crate::commondef::training::PolicyOutputSpec,
        crate::commondef::training::StandardInputSpec,
        crate::commondef::training::ProprioceptiveOutputSpec,
        crate::commondef::training::EEFOutputSpec,
        crate::commondef::training::InputState,
        crate::commondef::training::InputStateControllable,
        crate::commondef::training::InputStateItemControllable,
        crate::commondef::training::InputStateItemSensors,
        crate::commondef::training::OutputState,
        crate::commondef::training::OutputStateItem,
        crate::commondef::training::XPoseSpec,
        crate::commondef::training::IOStateXFrame,
        crate::commondef::training::IOStateXOrientationType,
        crate::commondef::training::ModelType,
        crate::commondef::training::CheckpointType,
        crate::commondef::training::Pretrained,
        crate::commondef::training::RewardSystem,
        crate::commondef::training::RewardLanguage,
        crate::commondef::training::TrainingGraph,
        crate::commondef::training
            ::TrainingJobController,
        crate::commondef::training::TrainingMethod,
        crate::commondef::training
            ::PPOTrainingMethod,
        crate::commondef::training
            ::TrainingAccelerationMode,
        crate::commondef::training::ComputeInstance,
        crate::commondef::training
            ::CheckpointFrequency,
        crate::commondef::training
            ::ControlOutputMode,
        crate::commondef::training
            ::FlowMatchingAction,
        crate::commondef::training::FlowNoiseConfig,
        crate::commondef::testing::TestGraph,
        crate::commondef::testing::TestJobController,
        crate::commondef::testing::TestMethod,
        crate::commondef::testing::CustomTestMethod,
        crate::commondef::testing::CustomTestMethodTestParams,
        crate::commondef::testing::DatasetMethod,
        crate::commondef::testing::Dataset,
        crate::commondef::testing::LiberoDatasetParams,
        crate::commondef::testing::LiberoDatasetType,
        crate::commondef::testing::StopTestMessage,
        crate::commondef::testing::TestUpdateMessage,
        crate::commondef::testing::CameraFrames,
        crate::commondef::testing::ModelForward,
        crate::commondef::clientmessages::ClientMessage,
        crate::commondef::training::StopTrainingMessage,
        crate::commondef::training
            ::TrainingUpdatesMessage,
        crate::commondef::training::TrainingUpdate,
        crate::commondef::training::TrainingRunState,
        crate::commondef::training::PPOTrainingRunState,
        crate::commondef::training::TrainingMetrics,
        crate::commondef::training::MetricsData,
    )),
    tags(
        (
            name = "Models",
            description =
                "Simulation model management",
        ),
        (
            name = "Checkpoints",
            description =
                "Training checkpoint management",
        ),
        (
            name = "Training",
            description =
                "Training experiments and runs",
        ),
        (
            name = "Testing",
            description =
                "Simulation testing workloads",
        ),
    )
)]
pub struct ApiDoc;

#[get("/openapi.json")]
pub async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

// ── Shared API state ──────────────────────────

pub struct ApiState {
    pub training_senders: AsyncMutex<HashMap<String, broadcast::Sender<String>>>,
    pub training_stops: AsyncMutex<HashMap<String, (JoinHandle<()>, Arc<AtomicBool>)>>,
    pub simul_senders: AsyncMutex<HashMap<String, broadcast::Sender<Vec<u8>>>>,
    pub simul_tasks: AsyncMutex<HashMap<String, JoinHandle<()>>>,
}

impl ApiState {
    pub fn new() -> Self {
        ApiState {
            training_senders: AsyncMutex::new(HashMap::new()),
            training_stops: AsyncMutex::new(HashMap::new()),
            simul_senders: AsyncMutex::new(HashMap::new()),
            simul_tasks: AsyncMutex::new(HashMap::new()),
        }
    }
}
