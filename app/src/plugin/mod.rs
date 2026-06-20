use std::collections::HashMap;

use tch::{Tensor, nn};

use crate::commondef::model::WorldBody;
use crate::commondef::training::ModelInputSpec;
pub use crate::commondef::training::ControlPolicyOutputMode;

pub mod mj_simulator;
pub mod ppo_trainer;

pub trait Simulator {
    fn reset_state(&mut self);
    fn nq(&self) -> i64;
    fn nu(&self) -> i64;
    fn nv(&self) -> i64;
    //Shape of u: [nu]
    fn forward_dyn(&self, controls: &Tensor);
    //Shape of acc: [nq]
    //Shape of output: [nv]
    fn inverse_dyn(&self, acc: &Tensor) -> Tensor;
    fn get_model_state(&self, input_spec: &ModelInputSpec) -> HashMap<String, Tensor>;
    fn get_model_cartesian_state(&self) -> Tensor;
    fn get_world_body(&self) -> Option<WorldBody>;
    fn get_simul_update(&self) -> (Vec<f64>, Vec<f64>);
    fn get_contact_body_index_pairs(&self) -> Vec<(usize, usize)>;
    fn modify_target_position(&mut self, target_pos_diff: &Tensor) -> Tensor;
    fn get_target_controls_from_target_positions(&self, target_pos_diff: &Tensor) -> Tensor;
    fn get_target_controls_from_target_velocities(&self, target_vel_diff: &Tensor) -> Tensor;
}

pub trait ControlPolicyModel {
    fn get_policy_output(&self, state: &Tensor) -> Tensor;
    fn get_policy(&self) -> &nn::Sequential;
    fn get_policy_store(&self) -> &nn::VarStore;
    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore;
    fn get_output_mode(&self) -> ControlPolicyOutputMode;
    fn get_control_input_spec(&self) -> ModelInputSpec;
    fn assemble_control_state(&self, parts: &HashMap<String, Tensor>) -> Tensor;
}

