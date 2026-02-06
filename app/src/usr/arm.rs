use tch::{nn, Tensor};

use crate::{plugin::ppo_trainer::PPOTrainedModel, usr::UsrDefined};

pub struct ArmModel {

}

impl PPOTrainedModel for ArmModel {
    fn init_ppo_trained_model(
        _policy: nn::Sequential,
        _val_estimator: nn::Sequential,
        _pol_store: nn::VarStore,
        _val_estimator_store: nn::VarStore,
    ) -> Self {
        todo!("ArmModel is not implemented yet")
    }

    fn get_nq(&self) -> i64 {
        todo!("ArmModel is not implemented yet")
    }

    fn get_reward(&self, _wstate: &Tensor, _wstatenext: Option<&Tensor>, _action: &Tensor) -> f64 {
        todo!("ArmModel is not implemented yet")
    }

    fn get_policy_output(&self, _state: &Tensor) -> Tensor {
        todo!("ArmModel is not implemented yet")
    }

    fn get_value_output(&self, _state: &Tensor) -> f64 {
        todo!("ArmModel is not implemented yet")
    }

    fn get_policy(&self) -> &&nn::Sequential {
        todo!("ArmModel is not implemented yet")
    }

    fn get_value_func(&self) -> &&nn::Sequential {
        todo!("ArmModel is not implemented yet")
    }

    fn get_policy_store(&self) -> &nn::VarStore {
        todo!("ArmModel is not implemented yet")
    }

    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore {
        todo!("ArmModel is not implemented yet")
    }
}

impl UsrDefined for ArmModel {

}
