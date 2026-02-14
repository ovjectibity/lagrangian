use typeshare::typeshare;
use serde;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub struct SimulationParams {
    pub policy_id: String,
    pub num_timesteps: i64,
}
