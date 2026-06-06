use serde;
use typeshare::typeshare;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum ControlPolicyOutputMode {
    TargetPositions,
    TargetVelocities,
    TargetTorques,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct ControlModelInput {
    pub all_cameras: bool,
    pub qpos_state: bool,
    pub qvel_state: bool,
    pub xpos_state: bool,
    pub xquat_state: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct VLAModel {
    pub input: ControlModelInput,
    pub output: ControlPolicyOutputMode,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum PretrainedVLA {
    Pi0(VLAModel),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlModel {
    FFN(FFNModel),
    Checkpoint(String),
    PretrainedVLA(PretrainedVLA),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ValueModel {
    FFN(FFNModel),
    Checkpoint(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct FFNModel {
    pub num_hidden_layers: i32,
    pub hidden_layer_dim: i32,
    pub output_dim: i32,
    pub input_dim: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingMethod {
    PPO(PPOTrainingMethod),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum FlowMatchingAction {
    FlowNoise(FlowNoiseConfig),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct FlowNoiseConfig {
    pub noise_model: FFNModel,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlOutputMode {
    ActionMean,
    FlowMatchingAction(FlowMatchingAction),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum RewardLanguage {
    Rust,
    Python,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct RewardSystem {
    pub language: RewardLanguage,
    pub reward_function_code: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct MetricsData {
    pub x_axis: Vec<f64>,
    pub y_axis: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum TrainingAccelerationMode {
    MLGPUSimulationCPU,
    MLGPUSimulationGPU,
    MLCPUSimulationCPU,
    AllOnServer,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum ComputeInstance {
    AWSG4dn,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingJobController {
    pub training_method: TrainingMethod,
    pub acceleration_mode: TrainingAccelerationMode,
    pub compute_instance: ComputeInstance,
    pub checkpoint_frequency: CheckpointFrequency,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingGraph {
    pub training_job_controller: TrainingJobController,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
