use std::collections::HashMap;

use schemars::JsonSchema;
use serde;
use typeshare::typeshare;

use crate::commondef::record::CheckpointReference;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub enum ControlPolicyOutputMode {
    TargetPositions,
    TargetVelocities,
    TargetTorques,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct ModelInputSpec {
    pub all_cameras: bool,
    pub qpos_state: bool,
    pub qvel_state: bool,
    pub xpos_state: bool,
    pub xquat_state: bool,
    pub qdof_state: bool,
    /// Only valid in reward_function.input_spec; forbidden in control/value model specs.
    pub contact_pairs: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub enum Pretrained {
    Pi0
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum CheckpointType {
    Pretrained(Pretrained),
    Checkpoint(String)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ModelType {
    Checkpoint(CheckpointType),
    New
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct VLAModel {
    pub model_type: ModelType,
    pub input_spec: ModelInputSpec,
    pub output: ControlPolicyOutputMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct FFNModel {
    pub model_type: ModelType,
    pub num_hidden_layers: i32,
    pub hidden_layer_dim: i32,
    pub input_spec: ModelInputSpec,
    pub output_dim: i32,
    pub input_dim: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlModel {
    FFN(FFNModel),
    VLA(VLAModel)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ValueModel {
    FFN(FFNModel)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingMethod {
    PPO(PPOTrainingMethod),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum FlowMatchingAction {
    FlowNoise(FlowNoiseConfig),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct FlowNoiseConfig {
    pub noise_model: FFNModel,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlOutputMode {
    ActionMean,
    FlowMatchingAction(FlowMatchingAction),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub enum RewardLanguage {
    Rust,
    Python,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct RewardSystem {
    pub language: RewardLanguage,
    pub reward_function_code: String,
    pub input_spec: ModelInputSpec
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct MetricsData {
    pub x_axis: Vec<f64>,
    pub y_axis: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub enum TrainingAccelerationMode {
    MLGPUSimulationCPU,
    MLGPUSimulationGPU,
    MLCPUSimulationCPU,
    AllOnServer,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub enum ComputeInstance {
    AWSG4dn,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct TrainingJobController {
    pub training_method: TrainingMethod,
    pub acceleration_mode: TrainingAccelerationMode,
    pub compute_instance: ComputeInstance,
    pub checkpoint_frequency: CheckpointFrequency,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[typeshare]
pub struct TrainingGraph {
    pub training_job_controller: TrainingJobController,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum CheckpointFrequency {
    PostEpoch(i32),
    PostUpdate(i32),
    LastOnly
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingRunState {
    PPOTrainingRunState(PPOTrainingRunState),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct PPOTrainingRunState {
    pub epoch_index: i32,
    pub update_index: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct StopTrainingMessage {
    pub id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingUpdatesMessage {
    pub id: String,
    pub updates: Vec<TrainingUpdate>
}

#[derive(
    Debug, Clone,
    serde::Serialize, serde::Deserialize,
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
    pub control_model_ffn: Option<FFNModel>,
    pub value_model_ffn: Option<FFNModel>,
}