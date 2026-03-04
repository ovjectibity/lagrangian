use serde;
use typeshare::typeshare;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ControlModel {
    FFN(FFNModel),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ValueModel {
    FFN(FFNModel),
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
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct RewardSystem {
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
    MLCPUSimulationGPU,
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
}
