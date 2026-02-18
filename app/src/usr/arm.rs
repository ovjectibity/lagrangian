use tch::{
    nn::{self, Module},
    Tensor,
};

use crate::plugin::{ControlPolicyOutputMode, ControlPolicyModel, ppo_trainer::PPOTrainedModel};

pub struct ArmModel {
    control_output_mode: ControlPolicyOutputMode,
    policy: nn::Sequential,
    value: nn::Sequential,
    policy_store: nn::VarStore,
    value_store: nn::VarStore,
    bq: i64,
    target_cube_body_index: usize,
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
                let parts = xs.split_with_sizes(
                    [action_dim, action_dim],
                    -1,
                );
                let mean = parts
                    .get(0)
                    .expect("missing mean head")
                    .tanh()
                    * 0.05;
                let log_std = parts
                    .get(1)
                    .expect("missing log_std head")
                    .clamp(-5.5, -3.0);
                Tensor::concat(&[mean, log_std], -1)
            });
        }
        //TODO: This needs to output the
        //mean & log(std-dev) n_q's for target positions:
        if add_softmax {
            y = y
                .add_fn(Tensor::relu)
                .add_fn(|xs| xs.softmax(-1, xs.kind()));
        }

        (y, vs)
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
}

impl PPOTrainedModel for ArmModel {
    fn init_ppo_trained_model(
        policy: nn::Sequential,
        val_estimator: nn::Sequential,
        pol_store: nn::VarStore,
        val_estimator_store: nn::VarStore
    ) -> Self {
        ArmModel {
            control_output_mode: 
                ControlPolicyOutputMode::TargetPositions,
            policy: policy,
            value: val_estimator,
            policy_store: pol_store,
            value_store: val_estimator_store,
            bq: 7,
            // vx300 worldbody order assumption:
            // 0=world, 1=left_box_body, 2=right_box_body, 3=target_cube_body
            target_cube_body_index: 3,
            // Scene-tuned defaults for the vx300 setup:
            // left_box:  (0.35,  0.22), right_box: (0.35, -0.22)
            // target_cube sits on left_box at z ~ 0.125 (top ~ 0.15)
            start_center_xy: (0.35, -0.22),
            target_center_xy: (0.35, 0.22),
            start_radius: 0.09,
            target_radius: 0.04,
            plane_z: 0.15,
            w_dist: 1.0,
            w_prog: 5.0,
            w_success: 50.0,
            w_act: 0.0 //1e-3,
        }
    }

    //TODO: Add contact bonus here
    fn get_reward(&self, 
        cstate: &Tensor, 
        cstate_next: Option<&Tensor>, 
        action: &Tensor) -> f64 {
        let cube_pos = 
        |state: &Tensor| -> (f64, f64, f64) {
            let state = state.to_kind(tch::Kind::Double);
            let state_vals: Vec<f64> = Vec::<f64>::try_from(state)
                .expect("Expected cartesian state to be a 1D f64 tensor");
            if state_vals.len() % 7 != 0 {
                panic!(
                    "Expected cartesian state len to be 7*nbody, got {}",
                    state_vals.len()
                );
            }
            let nbody = state_vals.len() / 7;
            if self.target_cube_body_index >= nbody {
                panic!(
                    "target_cube body index {} out of bounds for nbody {}",
                    self.target_cube_body_index, nbody
                );
            }
            let cube_x_idx = self.target_cube_body_index * 3;
            (
                state_vals[cube_x_idx],
                state_vals[cube_x_idx + 1],
                state_vals[cube_x_idx + 2],
            )
        };

        let curr_state = cstate_next.unwrap_or(cstate);
        let (curr_x, curr_y, curr_z) = cube_pos(curr_state);
        let dx = curr_x - self.target_center_xy.0;
        let dy = curr_y - self.target_center_xy.1;
        let dz = curr_z - self.plane_z;
        let curr_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let r_dist = -curr_dist * self.w_dist;

        let r_prog = if cstate_next.is_some() {
            let (prev_x, prev_y, prev_z) = cube_pos(cstate);
            let prev_dx = prev_x - self.target_center_xy.0;
            let prev_dy = prev_y - self.target_center_xy.1;
            let prev_dz = prev_z - self.plane_z;
            let prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy + prev_dz * prev_dz).sqrt();
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

impl ControlPolicyModel for ArmModel {
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
}
