use std::sync::{Arc, Mutex};

use crate::commondef::training::{ControlModel, PPOTrainingMethod, ValueModel};
use crate::plugin::mj_simulator::MjSimulator;
use crate::plugin::ppo_trainer::PPOTrainedModel;
use crate::plugin::{ControlPolicyModel, Simulator};
use crate::usr::arm::ArmModel;

pub mod testing;
pub mod training;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowMode {
    Idle,
    Training,
    Testing,
}

pub struct WorkflowRuntime {
    simulator: Arc<Mutex<MjSimulator>>,
    model: Arc<Mutex<ArmModel>>,
    mode: WorkflowMode,
}

impl WorkflowRuntime {
    pub fn load(model_path: String) -> Result<Self, String> {
        let simulator = MjSimulator::new(model_path)?;
        let nq = simulator.nq();
        let model = Self::build_default_model(nq);
        Ok(WorkflowRuntime {
            simulator: Arc::new(Mutex::new(simulator)),
            model: Arc::new(Mutex::new(model)),
            mode: WorkflowMode::Idle,
        })
    }

    fn build_default_model(nq: i64) -> ArmModel {
        let input_dim = nq * 2;
        let policy_output_dim = nq * 2;
        let hidden_dim = 64;
        let num_hidden_layers = 2;
        let (policy, policy_store) = ArmModel::init_policy(
            input_dim,
            policy_output_dim,
            hidden_dim,
            num_hidden_layers,
            false,
        );
        let (value, value_store) =
            ArmModel::init_policy(input_dim, 1, hidden_dim, num_hidden_layers, false);
        ArmModel::init_ppo_trained_model(policy, value, policy_store, value_store, nq)
    }

    pub fn configure_model_for_training(&self, params: &PPOTrainingMethod) -> Result<(), String> {
        let control_params = match &params.control_model {
            ControlModel::FFN(m) => m,
        };
        let value_params = match &params.value_model {
            ValueModel::FFN(m) => m,
        };
        let nq = self
            .simulator
            .lock()
            .map_err(|_| "Simulator mutex poisoned".to_string())?
            .nq();
        let (policy, policy_store) = ArmModel::init_policy(
            control_params.input_dim as i64,
            control_params.output_dim as i64,
            control_params.hidden_layer_dim as i64,
            control_params.num_hidden_layers as i64,
            false,
        );
        let (value, value_store) = ArmModel::init_policy(
            value_params.input_dim as i64,
            value_params.output_dim as i64,
            value_params.hidden_layer_dim as i64,
            value_params.num_hidden_layers as i64,
            false,
        );
        let next_model =
            ArmModel::init_ppo_trained_model(policy, value, policy_store, value_store, nq);
        let mut model = self
            .model
            .lock()
            .map_err(|_| "Model mutex poisoned".to_string())?;
        *model = next_model;
        Ok(())
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
