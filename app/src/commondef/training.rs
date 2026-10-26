use std::collections::HashMap;

use schemars::JsonSchema;
use serde;
use typeshare::typeshare;
use utoipa::ToSchema;

use crate::commondef::record::CheckpointReference;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlPolicyOutputSpec {
    OrdinaryConfiguration(OrdinaryIOStateConfiguration),
    EEFOnlyConfiguration(EEFOnlyIOStateConfiguration),
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct OrdinaryIOStateConfiguration {
    pub all_cameras: bool,
    pub qpos_state: bool,
    pub qvel_state: bool,
    pub genforces_state: bool,
    pub xpos_state: bool,
    pub xquat_state: bool,
    /// Only valid in reward_function.input_spec; forbidden in control/value model specs.
    pub contact_pairs: bool,
    pub is_diff: bool,
    //When null the entire model's state is returned otherwise only the 
    // state of the bodies encapsulated by the robot body base
    pub robot_base_body_name: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub enum IOStateOrientationType {
    Quaternion, 
    AxisAngle,
    //Same as Yaw-Pitch-Roll:
    TaitBryan,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct EEFOnlyIOStateConfiguration {
    //When set this results in only the end effector state being returned.
    //The eef body should be marked with a clear label.
    pub robot_base_body_name: String,
    pub gripper_body_name: String,
    pub end_effector_body_name: String,
    pub xpos_state: bool, 
    pub xquat_state: bool,
    pub eef_state_orientation_type: IOStateOrientationType, 
    pub qpos_gripper: bool,
    pub is_diff: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlPolicyInputSpec {
    OrdinaryConfiguration(OrdinaryIOStateConfiguration),
    EEFOnlyConfiguration(EEFOnlyIOStateConfiguration),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub enum Pretrained {
    Pi0
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum CheckpointType {
    Pretrained(Pretrained),
    Checkpoint(String)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ModelType {
    Checkpoint(CheckpointType),
    New
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct VLAModel {
    pub model_type: ModelType,
    pub input_spec: ControlPolicyInputSpec,
    pub output: ControlPolicyOutputSpec,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct MLPModel {
    pub model_type: ModelType,
    pub num_hidden_layers: i32,
    pub hidden_layer_dim: i32,
    pub input_spec: ControlPolicyInputSpec,
    pub output_dim: i32,
    pub input_dim: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlModel {
    MLP(MLPModel),
    VLA(VLAModel)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ValueModel {
    MLP(MLPModel)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingMethod {
    PPO(PPOTrainingMethod),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum FlowMatchingAction {
    FlowNoise(FlowNoiseConfig),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct FlowNoiseConfig {
    pub noise_model: MLPModel,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlOutputMode {
    ActionMean,
    FlowMatchingAction(FlowMatchingAction),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
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
    pub simulation_model: String,
    pub control_output_mode: ControlOutputMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub enum RewardLanguage {
    Rust,
    Python,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct RewardSystem {
    pub language: RewardLanguage,
    pub reward_function_code: String,
    pub input_spec: ControlPolicyInputSpec
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingMetrics {
    MeanReward(MetricsData),
    MeanPolicyLoss(MetricsData),
    ValueLoss(MetricsData),
    MeanEntropy(MetricsData),
    PolicyGradientNorm(MetricsData),
    ValueGradientNorm(MetricsData),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct MetricsData {
    pub x_axis: Vec<f64>,
    pub y_axis: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub enum TrainingAccelerationMode {
    MLGPUSimulationCPU,
    MLGPUSimulationGPU,
    MLCPUSimulationCPU,
    AllOnServer,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub enum ComputeInstance {
    AWSG4dn,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct TrainingJobController {
    pub training_method: TrainingMethod,
    pub acceleration_mode: TrainingAccelerationMode,
    pub compute_instance: ComputeInstance,
    pub checkpoint_frequency: CheckpointFrequency,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[typeshare]
pub struct TrainingGraph {
    pub training_job_controller: TrainingJobController,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum CheckpointFrequency {
    PostEpoch(i32),
    PostUpdate(i32),
    LastOnly
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingRunState {
    PPOTrainingRunState(PPOTrainingRunState),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct PPOTrainingRunState {
    pub epoch_index: i32,
    pub update_index: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct StopTrainingMessage {
    pub id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[typeshare]
pub struct TrainingUpdatesMessage {
    pub id: String,
    pub updates: Vec<TrainingUpdate>
}

#[derive(
    Debug, Clone,
    serde::Serialize, serde::Deserialize,
    ToSchema,
)]
#[typeshare]
pub struct TrainingUpdate {
    pub training_run_state: TrainingRunState,
    pub logs: Vec<String>,
    pub metrics: Vec<TrainingMetrics>,
    pub checkpoints: Vec<CheckpointReference>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StartTrainingMessage {
    pub id: String,
    pub training_graph: TrainingGraph,
    pub checkpoints: Vec<Checkpoint>,
    pub simulation_models: HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct Checkpoint {
    pub id: Option<String>,
    pub model_zip_base64: Option<String>,
    pub model_zip_path: Option<String>,
    pub control_model_mlp: Option<MLPModel>,
    pub value_model_mlp: Option<MLPModel>,
}