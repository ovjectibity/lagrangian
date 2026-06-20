use tch::Tensor;
use tch::nn;
use std::collections::HashMap;

use crate::commondef::training::{
    ModelInputSpec, ControlPolicyOutputMode
};

pub mod mlp;

pub trait ControlPolicyModel {
    fn get_policy_output(&self, state: &Tensor) -> Tensor;
    fn get_policy(&self) -> &nn::Sequential;
    fn get_policy_store(&self) -> &nn::VarStore;
    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore;
    fn get_output_mode(&self) -> ControlPolicyOutputMode;
    fn get_control_input_spec(&self) -> ModelInputSpec;
    fn assemble_control_state(&self, parts: &HashMap<String, Tensor>) -> Tensor;
}

pub trait PPOTrainedModel {
    //TODO: Replace NQ with richer model info
    fn init_ppo_trained_model(
        policy: nn::Sequential,
        val_estimator: nn::Sequential,
        pol_store: nn::VarStore,
        val_estimator_store: nn::VarStore,
        control_input_spec: ModelInputSpec,
        value_input_spec: ModelInputSpec,
        reward_input_spec: ModelInputSpec,
        reward_fn: Option<Box<dyn Fn(&HashMap<String, Tensor>, &HashMap<String, Tensor>) -> f64 + Send>>,
    ) -> Self
    where
        Self: Sized;
    fn get_reward(
        &self,
        parts_pre: &HashMap<String, Tensor>,
        parts_post: &HashMap<String, Tensor>,
    ) -> f64;
    fn get_policy_output(&self, state: &Tensor) -> Tensor;
    fn get_value_output(&self, state: &Tensor) -> f64;
    fn get_policy(&self) -> &nn::Sequential;
    fn get_value_func(&self) -> &nn::Sequential;
    fn get_policy_store(&self) -> &nn::VarStore;
    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore;
    fn get_value_store(&self) -> &nn::VarStore;
    fn get_value_store_mut(&mut self) -> &mut nn::VarStore;
    fn get_control_policy_output_mode(&self) -> ControlPolicyOutputMode;
    fn get_control_input_spec(&self) -> ModelInputSpec;
    fn get_value_input_spec(&self) -> ModelInputSpec;
    fn get_reward_input_spec(&self) -> ModelInputSpec;
    fn assemble_control_state(&self, parts: &HashMap<String, Tensor>) -> Tensor;
    fn assemble_value_state(&self, parts: &HashMap<String, Tensor>) -> Tensor;
}