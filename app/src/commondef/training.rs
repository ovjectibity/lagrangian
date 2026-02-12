use typeshare::typeshare;
use serde;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content="content")]
#[typeshare]
pub enum ControlModel {
    FFN(FFNModel)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content="content")]
#[typeshare]
pub enum ValueModel {
    FFN(FFNModel)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct FFNModel {
    num_hidden_layers: i32,
    hidden_layer_dim: i32,
    output_dim: i32,
    input_dim: i32
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content="content")]
#[typeshare]
pub enum TrainingMethod {
    PPO(PPOTrainingMethod)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct PPOTrainingMethod {
    num_epochs: i32,
    num_updates: i32,
    epsilon: f64,
    gamma: f64,
    lambda: f64,
    reward_function: RewardSystem,
    control_model: ControlModel,
    value_model: ValueModel
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct RewardSystem {
    reward_function_code: String
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content="content")]
#[typeshare]
pub enum TrainingMetrics {
    AvgReward(MetricsData),
    MeanPolicyLoss(MetricsData),
    ValueLoss(MetricsData),
    MeanEntropy(MetricsData)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct MetricsData {
    x_axis: Vec<f64>,
    y_axis: Vec<f64>
}