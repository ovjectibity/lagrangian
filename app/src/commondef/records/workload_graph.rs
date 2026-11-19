use serde;
use typeshare::typeshare;
use utoipa::ToSchema;

/// Execution graph for a training or testing workload.
#[derive(
    Debug, Clone, serde::Serialize, serde::Deserialize,
    ToSchema,
)]
#[typeshare]
pub enum WorkloadGraph {
    TestGraph(TestGraph),
    TrainingGraph(TrainingGraph)
}

// ---------------------TESTING WORKLOAD TYPES--------------------------------------
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct CustomTestMethodTestParams {
    pub num_timesteps: i32,
    pub num_parallel_envs: u32
    //TODO: Add reward function for eval
}

// #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
// #[typeshare]
// pub enum DatasetTaskSequencing {
//     ParallelTasksParallelEnvs,
//     SequentialTasksParallelEnvs,
//     FullySequential
// }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum LiberoDatasetType {
    Libero10, 
    Libero90,
    LiberoGoal,
    LiberoObject,
    LiberoSpatial
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct LiberoDatasetParams {
    // pub tasks: Option<Vec<i32>>,
    pub tasks_index: Vec<i32>,
    pub libero_dataset_type: LiberoDatasetType,
    // pub tasks_sequencing: DatasetTaskSequencing,
    pub num_envs_per_task: i32,
    pub num_timesteps: i32
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum Dataset {
    Libero(LiberoDatasetParams)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct DatasetMethod {
    pub dataset: Dataset,
    pub control_model: ControlModel
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct CustomTestMethod {
    pub test_params: CustomTestMethodTestParams,
    pub control_model: ControlModel,
    pub simulation_model_id: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TestMethod {
    Custom(CustomTestMethod),
    Dataset(DatasetMethod)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct TestJobController {
    pub test_method: TestMethod, 
    pub compute_instance: ComputeInstance,
    pub acceleration_mode: TrainingAccelerationMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct TestGraph {
    pub test_job_controller: TestJobController,
}

// ---------------------TRAINING WORKLOAD TYPES--------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum IOStateXOrientationType {
    Quaternion,
    AxisAngle
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum IOStateXFrame { 
    WorldFrame
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct XPoseSpec {
    pub frame_spec: IOStateXFrame,
    pub orientation_spec: IOStateXOrientationType, 
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum OutputStateItem {
    QPos,
    QVel,
    GeneralisedForces,
    //Represents the Cartesian pose as position, 
    //orientation in specific frame & orientation representation. 
    XPose(XPoseSpec)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum InputStateItemControllable {
    QPos,
    QVel,
    GeneralisedForces,
    XPose(XPoseSpec)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum InputStateItemSensors {
    Cameras,
    ContactPairs
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct OutputState {
    pub is_delta: bool, 
    pub rebase_body: String,
    pub state_item: OutputStateItem
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct InputStateControllable { 
    //This is to be interpreted as rebase
    // body tree to be included from the earlier specified base's tree
    // This inclusion operation is to be applied before the exclusion list 
    // If this list is empty it is interpreted as 
    // meaning all bodies from the prior base need to be included 
    pub rebase_body_tree: Option<String>,
    //This list is to be interpreted as the list of 
    // body trees to be excluded after the inclusion operation
    pub excluded_body_trees: Vec<String>,
    pub state_item: InputStateItemControllable
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum InputState { 
    InputStateControllable(InputStateControllable),
    InputStateSensors(InputStateItemSensors)

}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct StandardInputSpec {
    //When null the entire model's state is returned otherwise only the 
    // state of the bodies encapsulated by the robot body base
    pub base_body_name: Option<String>,
    pub states: Vec<InputState>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct ProprioceptiveOutputSpec {
    //When null the entire model's state is returned otherwise only the 
    // state of the bodies encapsulated by the robot body base
    pub base_body_name: String,
    pub state: OutputState
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct EEFOutputSpec {
    //When null the entire model's state is returned otherwise only the 
    // state of the bodies encapsulated by the robot body base
    pub base_body_name: Option<String>,
    pub eef_state: OutputState,
    pub gripper: OutputState
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum PolicyOutputSpec {
    ProprioceptiveOutputSpec(ProprioceptiveOutputSpec),
    EEFOutputSpec(EEFOutputSpec)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum PolicyInputSpec {
    StandardInputSpec(StandardInputSpec)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum Pretrained {
    Pi0
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum CheckpointType {
    Pretrained(Pretrained),
    CheckpointID(String)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ModelType {
    Checkpoint(CheckpointType),
    New
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct MLPModelSpec {
    pub num_hidden_layers: i32,
    pub hidden_layer_dim: i32,
    pub output_dim: i32,
    pub input_dim: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum MLModel {
    MLP(MLPModelSpec),
    VLA
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct ControlModel {
    pub model_type: ModelType,
    pub model: MLModel,
    pub input_spec: PolicyInputSpec,
    pub output_spec: PolicyOutputSpec,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct ValueModel {
    pub model_type: ModelType,
    pub input_spec: PolicyInputSpec,
    pub model: MLModel
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingMethod {
    PPO(PPOTrainingMethod),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum FlowMatchingAction {
    FlowNoise(FlowNoiseConfig),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct FlowNoiseConfig {
    pub noise_model: MLPModelSpec,
    pub model_type: ModelType
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlOutputMode {
    ActionMean,
    FlowMatchingAction(FlowMatchingAction),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct PPOTrainingMethod {
    pub num_trajs: i32,
    pub num_steps_per_traj: i32,
    pub num_epochs: i32,
    pub num_updates: i32,
    pub epsilon: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub reward_function: RewardSystem,
    pub control_model: ControlModel,
    pub value_model: ValueModel,
    pub simulation_model_id: String,
    pub control_output_mode: ControlOutputMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum RewardLanguage {
    Rust,
    Python,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct RewardSystem {
    pub language: RewardLanguage,
    pub reward_function_code: String,
    pub input_spec: PolicyInputSpec
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum TrainingAccelerationMode {
    MLGPUSimulationCPU,
    MLGPUSimulationGPU,
    MLCPUSimulationCPU,
    AllOnServer,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub enum ComputeInstance {
    AWSG4dn,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct TrainingJobController {
    pub training_method: TrainingMethod,
    pub acceleration_mode: TrainingAccelerationMode,
    pub compute_instance: ComputeInstance,
    pub checkpoint_frequency: CheckpointFrequency,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct TrainingGraph {
    pub training_job_controller: TrainingJobController,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum CheckpointFrequency {
    PostEpoch(i32),
    PostUpdate(i32),
    LastOnly
}
