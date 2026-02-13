use tch::{Tensor, nn::{self, Module}};

use crate::{plugin::ppo_trainer::PPOTrainedModel, usr::UsrDefined};

pub struct ArmModel {
    policy: nn::Sequential,
    value: nn::Sequential,
    policy_store: nn::VarStore,
    value_store: nn::VarStore,
    nq: i64,
    bq: i64,
    start_center_xy: (f64, f64),
    target_center_xy: (f64, f64),
    start_radius: f64,
    target_radius: f64,
    plane_z: f64,
    w_dist: f64,
    w_prog: f64,
    w_success: f64,
    w_act: f64,
}

impl ArmModel {
    pub fn init_policy(
        input_dimension: i64,
        output_dimension: i64,
        hidden_layer_dimension: i64,
        num_layers: i64,
        add_softmax: bool) -> (nn::Sequential, nn::VarStore) {
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
        y = y
            .add(nn::linear(
                vs_p.clone(),
                hidden_layer_dimension,
                output_dimension,
                Default::default(),
            ));
        //TODO: This needs to output the 
        //mean & log(std-dev) n_q's for target positions: 
        if add_softmax {
            y = y.add_fn(Tensor::relu)
                .add_fn(|xs| xs.softmax(-1, xs.kind()));
        }

        (y, vs)
    }
}

impl PPOTrainedModel for ArmModel {
    fn init_ppo_trained_model(
        policy: nn::Sequential,
        val_estimator: nn::Sequential,
        pol_store: nn::VarStore,
        val_estimator_store: nn::VarStore,
        nq: i64
    ) -> Self {
        ArmModel {
            policy: policy,
            value: val_estimator,
            policy_store: pol_store,
            value_store: val_estimator_store,
            nq: nq,
            bq: 7,
            start_center_xy: (0.0, 0.0),
            target_center_xy: (1.0, 0.0),
            start_radius: 0.1,
            target_radius: 0.1,
            plane_z: 0.0,
            w_dist: 1.0,
            w_prog: 1.0,
            w_success: 5.0,
            w_act: 1e-3,
        }
    }

    fn get_nq(&self) -> i64 {
        self.nq
    }

    //TODO: Add contact bonus here
    fn get_reward(
        &self,
        wstate: &Tensor,
        wstatenext: Option<&Tensor>,
        action: &Tensor,
    ) -> f64 {
        let curr = wstatenext.unwrap_or(wstate);
        let curr_pos = curr.narrow(0, 0, 3);
        let curr_pos_v: Vec<f64> = Vec::<f64>::try_from(curr_pos)
            .expect("Expected curr_pos to be a 1D f64 tensor");
        let dx = curr_pos_v[0] - self.target_center_xy.0;
        let dy = curr_pos_v[1] - self.target_center_xy.1;
        let curr_dist = (dx * dx + dy * dy).sqrt();
        let r_dist = -curr_dist * self.w_dist;

        let r_prog = if let Some(next) = wstatenext {
            let prev_pos = wstate.narrow(0, 0, 3);
            let prev_pos_v: Vec<f64> = Vec::<f64>::try_from(prev_pos)
                .expect("Expected prev_pos to be a 1D f64 tensor");
            let prev_dx = prev_pos_v[0] - self.target_center_xy.0;
            let prev_dy = prev_pos_v[1] - self.target_center_xy.1;
            let prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy).sqrt();
            (prev_dist - curr_dist) * self.w_prog
        } else {
            0.0
        };

        let r_success = if curr_dist <= self.target_radius {
            self.w_success
        } else {
            0.0
        };

        let act_penalty = {
            let act_sq = action.pow_tensor_scalar(2.0).sum(tch::Kind::Double);
            let act_sq_v = f64::try_from(act_sq).expect("Expected action to be f64 tensor");
            -self.w_act * act_sq_v
        };

        r_dist + r_prog + r_success + act_penalty
    }

    fn get_policy_output(&self, _state: &Tensor) -> Tensor {
        let state = _state
            .to_device(self.policy_store.device())
            .to_kind(self.policy_store.kind());
        self.policy.forward(&state)
    }

    fn get_value_output(&self, state: &Tensor) -> f64 {
        let state = state
            .to_device(self.value_store.device())
            .to_kind(self.value_store.kind());
        Vec::<f64>::try_from(self.value.forward(&state)).
            expect("Expected a 1-d tensor as value output").
            get(0).expect("Expected 0 index of value output").
            clone()
    }

    fn get_policy(&self) -> &nn::Sequential {
        &self.policy
    }

    fn get_value_func(&self) -> &nn::Sequential {
        &self.value
    }

    fn get_policy_store(&self) -> &nn::VarStore {
        &self.policy_store
    }

    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.policy_store
    }

    fn get_value_store(&self) -> &nn::VarStore {
        &self.value_store
    }

    fn get_value_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.value_store
    }
}

impl UsrDefined for ArmModel {

}
