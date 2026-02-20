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
    plane_z: f64
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
                    * 0.1;
                let log_std = parts
                    .get(1)
                    .expect("missing log_std head")
                    .clamp(-4.0, -2.0);
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

    fn get_body_pos(
        &self,
        state: &Tensor,
        body_index: usize,
    ) -> (f64, f64, f64) {
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
        if body_index >= nbody {
            panic!(
                "Body index {} out of bounds for nbody {}",
                body_index,
                nbody
            );
        }
        let pos_offset = body_index * 3;
        (
            state_vals[pos_offset],
            state_vals[pos_offset + 1],
            state_vals[pos_offset + 2],
        )
    }

    fn get_body_quat(
        &self,
        state: &Tensor,
        body_index: usize,
    ) -> (f64, f64, f64, f64) {
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
        if body_index >= nbody {
            panic!(
                "Body index {} out of bounds for nbody {}",
                body_index,
                nbody
            );
        }
        let quat_offset = nbody * 3 + body_index * 4;
        (
            state_vals[quat_offset],
            state_vals[quat_offset + 1],
            state_vals[quat_offset + 2],
            state_vals[quat_offset + 3],
        )
    }

    fn get_reward_2(
        &self,
        cstate: &Tensor,
        cstate_next: &Tensor,
        contact_pairs: Option<&[(usize, usize)]>,
    ) -> f64 {
        // vx300 body index assumptions:
        // 0=world, 1=left_box, 2=right_box, 3=target_cube, 10=gripper_link
        let left_box_body_index = 1usize;
        let right_box_body_index = 2usize;
        let gripper_body_index = 10usize;
        let gripper_prop_body_index = 11usize;
        let left_finger_body_index = 12usize;
        let right_finger_body_index = 13usize;
        let left_box_start = (0.35_f64, 0.22_f64, 0.0_f64);
        let right_box_start = (0.35_f64, -0.22_f64, 0.0_f64);
        let w_prog = 10.0_f64;
        let w_gripper_above = 2.0_f64;
        let w_gripper_below = 4.0_f64;
        let w_gripper_down = 1.5_f64;
        let w_box_move = 20.0_f64;
        let w_cube_drop = 20.0_f64;
        let w_grip_contact = 10.0_f64;
        let w_cube_contact_clean = 20.0_f64;
        let cube_drop_z_threshold = 0.10_f64;
        let gripper_xy_reward_radius = 0.035_f64;

        //Target cube xy position reward
        let (next_cx, next_cy, next_cz) =
            self.get_body_pos(cstate_next, self.target_cube_body_index);
        let next_cube_dx = next_cx - self.target_center_xy.0;
        let next_cube_dy = next_cy - self.target_center_xy.1;
        // let cube_dz = curr_cz - self.plane_z;
        //Using the box position instead of plan_z
        let next_cube_dz = next_cz - 0.025;
        let next_cube_dist = (
            next_cube_dx * next_cube_dx + 
            next_cube_dy * next_cube_dy 
            // next_cube_dz * next_cube_dz
        )
        .sqrt();

        let (prev_cx, prev_cy, prev_cz) =
            self.get_body_pos(cstate, self.target_cube_body_index);
        let prev_cube_dx = prev_cx - self.target_center_xy.0;
        let prev_cube_dy = prev_cy - self.target_center_xy.1;
        let prev_cube_dz = prev_cz - 0.025;
        let prev_cube_dist = (
            prev_cube_dx * prev_cube_dx + prev_cube_dy * prev_cube_dy
            // + prev_cube_dz * prev_cube_dz
        ).sqrt();
        let r_progress = (prev_cube_dist - next_cube_dist) * w_prog;

        //Gripper movement towards cube: 
        let (gripper_next_x, gripper_next_y, gripper_next_z) =
            self.get_body_pos(cstate_next, gripper_body_index);
        let gripper_xy_dist = ((gripper_next_x - next_cx).powi(2) + 
            (gripper_next_y - next_cy).powi(2)).sqrt();
        let gripper_xy = -w_gripper_above * gripper_xy_dist;

        //Gripper above the target cube reward
        // let (gripper_prev_x, gripper_prev_y, gripper_prev_z) =
        //     self.get_body_pos(cstate, gripper_body_index);
        let z_clearance = 0.03_f64;
        let z_sat_k = 12.0_f64;
        let z_gap = gripper_next_z - next_cz;
        let z_above = (z_gap - z_clearance).max(0.0);
        let z_below = (z_clearance - z_gap).max(0.0);
        let r_above_z = w_gripper_above
            * (1.0 - (-z_sat_k * z_above).exp());
        let p_below_z = -w_gripper_below * z_below;
        let r_gripper_above = 
        // r_above_z + p_below_z;
        if gripper_xy_dist <= gripper_xy_reward_radius {
            r_above_z + p_below_z
        } else {
            0.0
        };
        let (gripper_qw, gripper_qx, gripper_qy, gripper_qz) =
            self.get_body_quat(cstate_next, gripper_body_index);
        // local +X axis of gripper_link, rotated to world frame
        let gripper_x_world_z =
            2.0 * (gripper_qx * gripper_qz - gripper_qw * gripper_qy);
        let down_alignment = (-gripper_x_world_z).clamp(0.0, 1.0);
        let r_gripper_down = w_gripper_down * down_alignment;

        let (r_grip_contact, r_cube_contact_clean) =
            if let Some(contact_pairs) = contact_pairs {
                let mut left_contact = false;
                let mut right_contact = false;
                let mut prop_contact = false;
                let mut has_disallowed_cube_contact = false;
                for (body_a, body_b) in contact_pairs.iter() {
                    let cube_other = if *body_a == self.target_cube_body_index {
                        Some(*body_b)
                    } else if *body_b == self.target_cube_body_index {
                        Some(*body_a)
                    } else {
                        None
                    };
                    if let Some(other_body) = cube_other {
                        if other_body == left_finger_body_index {
                            left_contact = true;
                        } else if other_body == right_finger_body_index {
                            right_contact = true;
                        } else if other_body == gripper_prop_body_index {
                            prop_contact = true;
                        } else {
                            has_disallowed_cube_contact = true;
                        }
                    }
                }
                let contact_count =
                    left_contact as i32
                    + right_contact as i32
                    + prop_contact as i32;
                let has_allowed_contact = contact_count > 0;
                let r_grip_contact =
                    w_grip_contact * contact_count as f64;
                let r_cube_contact_clean =
                    if has_allowed_contact
                    && !has_disallowed_cube_contact {
                        w_cube_contact_clean
                    } else {
                        0.0
                    };
                (r_grip_contact, r_cube_contact_clean)
            } else {
                (0.0, 0.0)
            };

        //Boxes moved -ve reward
        let (left_x, left_y, left_z) =
            self.get_body_pos(cstate_next, left_box_body_index);
        let (right_x, right_y, right_z) =
            self.get_body_pos(cstate_next, right_box_body_index);
        let left_disp = (
            (left_x - left_box_start.0).powi(2) +
            (left_y - left_box_start.1).powi(2)
            // (left_z - left_box_start.2).powi(2)
        )
        .sqrt();
        let right_disp = (
            (right_x - right_box_start.0).powi(2) +
            (right_y - right_box_start.1).powi(2)
            // (right_z - right_box_start.2).powi(2)
        )
        .sqrt();
        // println!("DEBUG: Getting this box displacement: {:?} {:?} {:?}",
        //     left_disp,
        //     right_disp,
        //     left_disp + right_disp);
        let r_box_moved = -w_box_move * (left_disp + right_disp);

        //Cube drop -ve reward
        let p_cube_drop = if next_cz < cube_drop_z_threshold {
            -w_cube_drop //* (cube_drop_z_threshold - next_cz)
        } else {
            0.0
        };

        println!("Returning this reward: r_gripper_above: {:?} r_gripper_down: {:?} gripper_xy: {:?} r_box_moved: {:?} p_cube_drop: {:?} r_grip_contact: {:?} r_cube_contact_clean: {:?}",
            r_gripper_above,
            r_gripper_down,
            gripper_xy,
            r_box_moved,
            p_cube_drop,
            r_grip_contact,
            r_cube_contact_clean);

        // r_progress + r_gripper_above + gripper_xy + r_box_moved + p_cube_drop
        r_gripper_above
            + r_gripper_down
            + gripper_xy
            + r_box_moved
            + p_cube_drop
            + r_grip_contact
            + r_cube_contact_clean
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
            start_center_xy: (0.35, 0.22),
            target_center_xy: (0.35, -0.22),
            start_radius: 0.09,
            target_radius: 0.04,
            plane_z: 0.15
        }
    }

    //TODO: Add contact bonus here
    fn get_reward(&self, 
        cstate: &Tensor, 
        cstate_next: &Tensor, 
        action: &Tensor,
        contact_pairs: Option<&[(usize, usize)]>) -> f64 {
        let _ = action;
        self.get_reward_2(cstate, cstate_next, contact_pairs)
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

    fn get_reward(&self, 
        cstate: &Tensor, 
        cstate_next: &Tensor, 
        action: &Tensor,
        contact_pairs: Option<&[(usize, usize)]>) -> f64 {
        let _ = action;
        self.get_reward_2(cstate, cstate_next, contact_pairs)
    }
}
