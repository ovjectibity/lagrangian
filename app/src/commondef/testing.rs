use serde;
use typeshare::typeshare;

use crate::commondef::training::{ControlModel, RewardSystem};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct SimulationParams {
    pub policy_id: String,
    pub num_timesteps: i32,
    // pub control_model: ControlModel,
    // pub simulation_model: String, 
    // pub control_output_mode: ControlOutputMode,
    // pub reward_function: Option<RewardSystem>
}
