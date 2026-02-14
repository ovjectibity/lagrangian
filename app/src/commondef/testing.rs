use serde;
use typeshare::typeshare;

use crate::commondef::training::ControlModel;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct SimulationParams {
    pub policy_id: String,
    pub num_timesteps: i32,
    control_model: ControlModel
}
