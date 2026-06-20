use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tch::Tensor;

use crate::commondef::testing::TestGraph;
use crate::commondef::training::{
    ControlModel, ModelInputSpec, ModelType, PPOTrainingMethod, ValueModel,
};
use crate::simulator::mj_simulator::MjSimulator;
use crate::simulator::{Simulator};
use crate::models::{ControlPolicyModel, PPOTrainedModel};
use crate::models::mlp::MLPModel;

pub mod testing;
pub mod training;
pub mod ppo_trainer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowMode {
    Idle,
    Training,
    Testing,
}

pub struct WorkflowRuntime {
    simulator: Arc<Mutex<MjSimulator>>,
    model: Arc<Mutex<MLPModel>>,
    mode: WorkflowMode,
}

impl WorkflowRuntime {
    pub fn for_training(
        model_path: String,
        params: &PPOTrainingMethod,
        reward_fn: Option<Box<dyn Fn(&HashMap<String, Tensor>, &HashMap<String, Tensor>) -> f64 + Send>>,
    ) -> Result<Self, String> {
        if matches!(&params.control_model, ControlModel::VLA(_)) {
            return Err("VLA control model is not supported for local training".into());
        }
        let control_params = match &params.control_model {
            ControlModel::MLP(m) => {
                if matches!(&m.model_type, ModelType::Checkpoint(_)) {
                    return Err(
                        "Checkpoint-based MLP control model is not supported for local training"
                            .into(),
                    );
                }
                m
            }
            _ => unreachable!(),
        };
        let value_params = match &params.value_model {
            ValueModel::MLP(m) => {
                if matches!(&m.model_type, ModelType::Checkpoint(_)) {
                    return Err(
                        "Checkpoint-based MLP value model is not supported for local training"
                            .into(),
                    );
                }
                m
            }
        };
        let simulator = MjSimulator::new(model_path)?;
        let (policy, policy_store) = MLPModel::init_policy(
            control_params.input_dim as i64,
            control_params.output_dim as i64,
            control_params.hidden_layer_dim as i64,
            control_params.num_hidden_layers as i64,
            false,
            true,
        );
        let (value, value_store) = MLPModel::init_policy(
            value_params.input_dim as i64,
            value_params.output_dim as i64,
            value_params.hidden_layer_dim as i64,
            value_params.num_hidden_layers as i64,
            false,
            false,
        );
        let model = MLPModel::init_ppo_trained_model(
            policy,
            value,
            policy_store,
            value_store,
            control_params.input_spec.clone(),
            value_params.input_spec.clone(),
            params.reward_function.input_spec.clone(),
            reward_fn,
        );
        Ok(WorkflowRuntime {
            simulator: Arc::new(Mutex::new(simulator)),
            model: Arc::new(Mutex::new(model)),
            mode: WorkflowMode::Idle,
        })
    }

    pub fn for_testing(
        model_path: String,
        test_graph: &TestGraph,
    ) -> Result<Self, String> {
        let control_params =
            match &test_graph.test_job_controller.test_params.control_model {
                ControlModel::MLP(m) => m,
                ControlModel::VLA(_) => {
                    return Err(
                        "VLA control model is not supported for local testing".into(),
                    );
                }
            };
        let simulator = MjSimulator::new(model_path)?;
        let (policy, policy_store) = MLPModel::init_policy(
            control_params.input_dim as i64,
            control_params.output_dim as i64,
            control_params.hidden_layer_dim as i64,
            control_params.num_hidden_layers as i64,
            false,
            true,
        );
        let (value, value_store) = MLPModel::init_policy(1, 1, 1, 1, false, false);
        let model = MLPModel::init_ppo_trained_model(
            policy,
            value,
            policy_store,
            value_store,
            control_params.input_spec.clone(),
            ModelInputSpec::default(),
            ModelInputSpec::default(),
            None,
        );
        Ok(WorkflowRuntime {
            simulator: Arc::new(Mutex::new(simulator)),
            model: Arc::new(Mutex::new(model)),
            mode: WorkflowMode::Idle,
        })
    }

    pub fn simulator_handle(&self) -> Arc<Mutex<dyn Simulator + Send>> {
        self.simulator.clone()
    }

    pub fn ppo_model_handle(&self) -> Arc<Mutex<dyn PPOTrainedModel + Send>> {
        self.model.clone()
    }

    pub fn control_model_handle(&self) -> Arc<Mutex<dyn ControlPolicyModel + Send>> {
        self.model.clone()
    }

    pub fn world_body(&self) -> Result<crate::commondef::model::WorldBody, String> {
        let simulator = self
            .simulator
            .lock()
            .map_err(|_| "Simulator mutex poisoned".to_string())?;
        simulator
            .get_world_body()
            .ok_or_else(|| "Couldn't build world body from simulator".to_string())
    }

    pub fn mode(&self) -> WorkflowMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: WorkflowMode) {
        self.mode = mode;
    }
}
