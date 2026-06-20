use std::collections::HashMap;

use tch::{
    Kind, Tensor,
    nn::{self, Module},
};

use crate::commondef::training::{ModelInputSpec,ControlPolicyOutputMode};
use crate::models::{PPOTrainedModel, ControlPolicyModel};

const STATE_KEY_ORDER: &[&str] = &["cameras", "qdof", "qpos", "qvel", "xpos", "xquat"];

pub struct MLPModel {
    control_output_mode: ControlPolicyOutputMode,
    policy: nn::Sequential,
    value: nn::Sequential,
    policy_store: nn::VarStore,
    value_store: nn::VarStore,
    control_input_spec: ModelInputSpec,
    value_input_spec: ModelInputSpec,
    reward_input_spec: ModelInputSpec,
}

impl MLPModel {
    pub fn init_policy(
        input_dimension: i64,
        output_dimension: i64,
        hidden_layer_dimension: i64,
        num_layers: i64,
        add_softmax: bool,
        control_policy: bool,
    ) -> (nn::Sequential, nn::VarStore) {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let vs_p = vs.root();
        let mut y = nn::seq()
            .add(nn::linear(
                vs_p.clone(),
                input_dimension,
                hidden_layer_dimension,
                Default::default(),
            ))
            .add_fn(Tensor::relu);
        for _ in 0..num_layers.saturating_sub(1) {
            y = y
                .add(nn::linear(
                    vs_p.clone(),
                    hidden_layer_dimension,
                    hidden_layer_dimension,
                    Default::default(),
                ))
                .add_fn(Tensor::relu);
        }
        y = y.add(nn::linear(
            vs_p.clone(),
            hidden_layer_dimension,
            output_dimension,
            Default::default(),
        ));
        if control_policy {
            if output_dimension % 2 != 0 {
                panic!(
                    "Control policy output dim must be even, got {}",
                    output_dimension
                );
            }
            let action_dim = output_dimension / 2;
            y = y.add_fn(move |xs| {
                let parts = xs.split_with_sizes([action_dim, action_dim], -1);
                let mean = parts
                    .get(0)
                    .expect("missing mean head")
                    .tanh()
                    * 0.01;
                let log_std = parts
                    .get(1)
                    .expect("missing log_std head")
                    .clamp(-5.2, -4.2);
                Tensor::concat(&[mean, log_std], -1)
            });
        }
        if add_softmax {
            y = y
                .add_fn(Tensor::relu)
                .add_fn(|xs| xs.softmax(-1, xs.kind()));
        }
        (y, vs)
    }

    pub fn set_input_specs(
        &mut self,
        control: ModelInputSpec,
        value: ModelInputSpec,
        reward: ModelInputSpec,
    ) {
        self.control_input_spec = control;
        self.value_input_spec = value;
        self.reward_input_spec = reward;
    }

    fn forward_policy_impl(&self, state: &Tensor) -> Tensor {
        let state = state
            .to_device(self.policy_store.device())
            .to_kind(self.policy_store.kind());
        self.policy.forward(&state)
    }

    fn get_policy_impl(&self) -> &nn::Sequential {
        &self.policy
    }

    fn get_policy_store_impl(&self) -> &nn::VarStore {
        &self.policy_store
    }

    fn get_policy_store_mut_impl(&mut self) -> &mut nn::VarStore {
        &mut self.policy_store
    }

    fn assemble_from_parts(parts: &HashMap<String, Tensor>) -> Tensor {
        let tensors: Vec<Tensor> = STATE_KEY_ORDER
            .iter()
            .filter_map(|k| parts.get(*k).map(|t| t.to_kind(Kind::Double)))
            .collect();
        Tensor::concat(&tensors, 0)
    }
}

impl PPOTrainedModel for MLPModel {
    fn init_ppo_trained_model(
        policy: nn::Sequential,
        val_estimator: nn::Sequential,
        pol_store: nn::VarStore,
        val_estimator_store: nn::VarStore,
    ) -> Self {
        MLPModel {
            control_output_mode: ControlPolicyOutputMode::TargetPositions,
            policy,
            value: val_estimator,
            policy_store: pol_store,
            value_store: val_estimator_store,
            control_input_spec: ModelInputSpec::default(),
            value_input_spec: ModelInputSpec::default(),
            reward_input_spec: ModelInputSpec::default(),
        }
    }

    fn get_reward(
        &self,
        _parts_pre: &HashMap<String, Tensor>,
        _parts_post: &HashMap<String, Tensor>,
    ) -> f64 {
        0.0
    }

    fn get_control_input_spec(&self) -> ModelInputSpec {
        self.control_input_spec.clone()
    }

    fn get_value_input_spec(&self) -> ModelInputSpec {
        self.value_input_spec.clone()
    }

    fn get_reward_input_spec(&self) -> ModelInputSpec {
        self.reward_input_spec.clone()
    }

    fn assemble_control_state(&self, parts: &HashMap<String, Tensor>) -> Tensor {
        Self::assemble_from_parts(parts)
    }

    fn assemble_value_state(&self, parts: &HashMap<String, Tensor>) -> Tensor {
        Self::assemble_from_parts(parts)
    }

    fn get_policy_output(&self, state: &Tensor) -> Tensor {
        self.forward_policy_impl(state)
    }

    fn get_value_output(&self, state: &Tensor) -> f64 {
        let state = state
            .to_device(self.value_store.device())
            .to_kind(self.value_store.kind());
        Vec::<f64>::try_from(self.value.forward(&state))
            .expect("Expected a 1-d tensor as value output")
            .get(0)
            .expect("Expected 0 index of value output")
            .clone()
    }

    fn get_policy(&self) -> &nn::Sequential {
        self.get_policy_impl()
    }

    fn get_value_func(&self) -> &nn::Sequential {
        &self.value
    }

    fn get_policy_store(&self) -> &nn::VarStore {
        self.get_policy_store_impl()
    }

    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore {
        self.get_policy_store_mut_impl()
    }

    fn get_value_store(&self) -> &nn::VarStore {
        &self.value_store
    }

    fn get_value_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.value_store
    }

    fn get_control_policy_output_mode(&self) -> ControlPolicyOutputMode {
        self.control_output_mode.clone()
    }
}

impl ControlPolicyModel for MLPModel {
    fn get_policy_output(&self, state: &Tensor) -> Tensor {
        self.forward_policy_impl(state)
    }

    fn get_policy(&self) -> &nn::Sequential {
        self.get_policy_impl()
    }

    fn get_policy_store(&self) -> &nn::VarStore {
        self.get_policy_store_impl()
    }

    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore {
        self.get_policy_store_mut_impl()
    }

    fn get_output_mode(&self) -> ControlPolicyOutputMode {
        self.control_output_mode.clone()
    }

    fn get_control_input_spec(&self) -> ModelInputSpec {
        self.control_input_spec.clone()
    }

    fn assemble_control_state(&self, parts: &HashMap<String, Tensor>) -> Tensor {
        Self::assemble_from_parts(parts)
    }
}
