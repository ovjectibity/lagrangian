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
    pub checkpoint_frequency: CheckpointFrequency
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingGraph {
    pub training_job_controller: TrainingJobController
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum CheckpointFrequency {
    PostEpoch(i32),
    PostUpdate(i32),
    // PostMinibatch(i64)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct Checkpoint {
    pub model_zip_base64: Option<String>,
    pub model_zip_path: Option<String>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum TrainingRunState {
    PPOTrainingRunState(PPOTrainingRunState)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct PPOTrainingRunState {
    pub epoch_index: i32,
    pub update_index: i32
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingExperiment {
    pub experiment_id: String, 
    pub experiment_name: String, 
    pub created_at: String, 
    pub experiment_runs: Vec<TrainingExperimentRun>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingExperimentRun {
    pub experiment_id: String, 
    pub run_name: String, 
    pub run_id: String,
    pub created_at: String, 
    pub run_start_at: String, 
    pub run_end_at: String, 
    pub training_graph: TrainingGraph,
    pub training_update: Vec<TrainingUpdate>
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct TrainingUpdate {
    pub training_run_state: TrainingRunState,
    pub logs: Vec<String>,
    pub metrics: Vec<TrainingMetrics>,
    pub checkpoints: Vec<Checkpoint>,
}