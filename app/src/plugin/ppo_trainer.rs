use std::sync::{Arc, Mutex};

use tch::{
    Kind, Tensor,
    nn::{self, Module, OptimizerConfig},
};

use crate::plugin::{ControlPolicyOutputMode, Simulator};

pub struct Trajectory {
    //Shape of indv element: [nq*2]
    //Number of world_states: traj_depth + 1
    world_states: Vec<Tensor>,
    //Shape of indv element: [nu*2]
    //Note: the std devs are logs
    model_outputs: Vec<Tensor>,
    //Shape of indv element: [nu]
    model_output_samples: Vec<Tensor>,
    model_output_sample_log_probs: Vec<f64>,
    rewards: Vec<f64>,
    rtgs: Vec<f64>,
}

pub struct PPOTrainerRuntimeInfo {
    mean_policy_loss: Vec<f64>,
    mean_value_loss: Vec<f64>,
    policy_gradient_norm: Vec<f64>,
    value_gradient_norm: Vec<f64>,
}

impl PPOTrainerRuntimeInfo {
    pub fn new(
        mean_policy_loss: Vec<f64>,
        mean_value_loss: Vec<f64>,
        policy_gradient_norm: Vec<f64>,
        value_gradient_norm: Vec<f64>,
    ) -> Self {
        PPOTrainerRuntimeInfo {
            mean_policy_loss: mean_policy_loss,
            mean_value_loss: mean_value_loss,
            policy_gradient_norm: policy_gradient_norm,
            value_gradient_norm: value_gradient_norm,
        }
    }

    pub fn mean_policy_loss(&self) -> &[f64] {
        &self.mean_policy_loss
    }

    pub fn mean_value_loss(&self) -> &[f64] {
        &self.mean_value_loss
    }

    pub fn policy_gradient_norm(&self) -> &[f64] {
        &self.policy_gradient_norm
    }

    pub fn value_gradient_norm(&self) -> &[f64] {
        &self.value_gradient_norm
    }
}

pub struct PPOTrainer {
    num_trajs: i32,
    traj_depth: i32,
    num_updates: i32,
    epsilon: f64,
    //Used for GAE:
    gamma_lambda: (f64, f64),
    simulator: Arc<Mutex<dyn Simulator + Send>>,
    model: Arc<Mutex<dyn PPOTrainedModel + Send>>,
    policy_opt: nn::Optimizer,
    value_opt: nn::Optimizer,
}

pub trait PPOTrainedModel {
    //TODO: Replace NQ with richer model info
    fn init_ppo_trained_model(
        policy: nn::Sequential,
        val_estimator: nn::Sequential,
        pol_store: nn::VarStore,
        val_estimator_store: nn::VarStore,
    ) -> Self
    where
        Self: Sized;
    // Using SpinningUp's formulation of reward function:
    // https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
    fn get_reward(
        &self,
        cstate: &Tensor,
        cstate_next: &Tensor,
        action: &Tensor,
        contact_pairs: Option<&[(usize, usize)]>,
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
}

//TODO: Optimising with aggregating all
// computations to single tensor where possible
impl PPOTrainer {
    fn get_action_output_size(&self) -> i64 {
        self.simulator
            .lock()
            .expect("Simulator mutex poisoned")
            .nu()
    }

    pub fn new(
        simulator: Arc<Mutex<dyn Simulator + Send>>,
        model: Arc<Mutex<dyn PPOTrainedModel + Send>>,
        num_trajs: i32,
        traj_depth: i32,
        num_updates: i32,
        epsilon: f64,
        gamma_lambda: (f64, f64),
    ) -> Result<Self, tch::TchError> {
        let policy_opt = {
            let m = model.lock().expect("Model mutex poisoned");
            nn::Adam::default().build(m.get_policy_store(), 1e-3)?
        };
        let value_opt = {
            let m = model.lock().expect("Model mutex poisoned");
            nn::Adam::default().build(m.get_value_store(), 1e-3)?
        };
        Ok(PPOTrainer {
            num_trajs: num_trajs,
            traj_depth: traj_depth,
            num_updates: num_updates,
            epsilon: epsilon,
            gamma_lambda: gamma_lambda,
            simulator: simulator,
            model: model,
            policy_opt: policy_opt,
            value_opt: value_opt,
        })
    }

    fn gen_trajectory(&self) -> Trajectory {
        let action_output_size = self.get_action_output_size();
        let mut wstates = Vec::new();
        let mut model_outputs = Vec::new();
        let mut model_output_samples = Vec::new();
        let mut rewards = Vec::new();
        tch::no_grad(|| {
            self.simulator
                .lock()
                .expect("Simulator mutex poisoned")
                .reset_state();
            for i in 0..self.traj_depth {
                let (wstate, cstate): (Tensor, Tensor) = {
                    let simulator = self.simulator.lock().expect("Simulator mutex poisoned");
                    (
                        simulator.get_model_state(),
                        simulator.get_model_cartesian_state(),
                    )
                };
                let action_gaussian_params = self
                    .model
                    .lock()
                    .expect("Model mutex poisoned")
                    .get_policy_output(&wstate);
                //Output action needs to be sampled &
                // then passed on to the simulator
                // println!("DEBUG: nu is {:?}",nu);
                let s_action_gaussian_params = action_gaussian_params
                    .split_with_sizes([action_output_size, action_output_size], 0);
                let action_gaussian_mean = s_action_gaussian_params
                    .get(0)
                    .expect("Couldn't extract the action_gaussian_mean");
                let action_gaussian_stdev = s_action_gaussian_params
                    .get(1)
                    .expect("Couldn't extract the action_gaussian_stddev")
                    .exp();
                let eps =
                    Tensor::randn([action_output_size], (tch::Kind::Double, tch::Device::Cpu));
                let sampled_action = action_gaussian_mean + &eps * &action_gaussian_stdev;
                // println!("DEBUG: Sampled action for traj rollout as follows: ");
                // sampled_action.print();
                let controls = match &self
                    .model
                    .lock()
                    .expect("Couldn't lock model during ppo traj gen")
                    .get_control_policy_output_mode()
                {
                    ControlPolicyOutputMode::TargetTorques => sampled_action.shallow_clone(),
                    ControlPolicyOutputMode::TargetPositions => self
                        .simulator
                        .lock()
                        .expect("Simulator mutex poisoned")
                        .modify_target_position(&sampled_action),
                    // ControlPolicyOutputMode::TargetPositions => self
                    //     .simulator
                    //     .lock()
                    //     .expect("Simulator mutex poisoned")
                    //     .get_target_controls_from_target_positions(&sampled_action),
                    ControlPolicyOutputMode::TargetVelocities => self
                        .simulator
                        .lock()
                        .expect("Simulator mutex poisoned")
                        .get_target_controls_from_target_velocities(&sampled_action),
                };
                // println!("DEBUG: Controls for the same sampled action as follows: ");
                // controls.print();
                self.simulator
                    .lock()
                    .expect("Simulator mutex poisoned")
                    .forward_dyn(&controls);
                let (cstate_next, contact_pairs) = {
                    let simulator = self.simulator.lock().expect("Simulator mutex poisoned");
                    (
                        simulator.get_model_cartesian_state(),
                        simulator.get_contact_body_index_pairs(),
                    )
                };
                let reward = self.model.lock().expect("Model mutex poisoned").get_reward(
                    &cstate,
                    &cstate_next,
                    &sampled_action,
                    Some(&contact_pairs),
                );
                println!("Got reward during gen_traj as: {:?}", reward);
                rewards.push(reward);
                wstates.push(wstate);
                model_outputs.push(action_gaussian_params);
                model_output_samples.push(sampled_action);
            }
            //Adding a terminal wstate given it's used for terminal val calculations
            let wstate_end: Tensor = self
                .simulator
                .lock()
                .expect("Simulator mutex poisoned")
                .get_model_state();
            wstates.push(wstate_end);
            let v_terminal = self
                .model
                .lock()
                .expect("Model mutex poisoned")
                .get_value_output(wstates.last().expect("missing terminal world state"));
            let rtgs = self.compute_rtg(&rewards, v_terminal);
            let action_sample_log_probs = self.compute_action_log_probs(
                &Tensor::stack(&model_output_samples, 0),
                &Tensor::stack(&model_outputs, 0),
            );
            Trajectory {
                world_states: wstates,
                model_outputs: model_outputs,
                model_output_samples: model_output_samples,
                rewards: rewards,
                rtgs: rtgs,
                model_output_sample_log_probs: Vec::<f64>::try_from(action_sample_log_probs)
                    .expect("wrong type of tensor"),
            }
        })
    }

    pub fn gen_trajectories(&self, num_trajs: i32) -> Vec<Trajectory> {
        let mut trajs = Vec::new();
        for i in 0..num_trajs {
            trajs.push(self.gen_trajectory());
        }
        trajs
    }

    //Output shape: [traj_depth]
    fn compute_gae(&self, traj: &Trajectory) -> Tensor {
        let mut advs = Vec::new();
        //Step-1: Get all rewards for the trajectory:
        //Step-2: Compute the advantage:
        for i in (0..self.traj_depth).rev() {
            let wstate = traj.world_states.get(i as usize).expect("");
            let reward = traj.rewards.get(i as usize).expect("");
            let val = self
                .model
                .lock()
                .expect("Model mutex poisoned")
                .get_value_output(wstate);
            let wstate_next = traj.world_states.get((i + 1) as usize).expect("");
            let val_next = self
                .model
                .lock()
                .expect("Model mutex poisoned")
                .get_value_output(wstate_next);
            //for terminal values,
            //don't assume there's no future value
            let adv = reward + (self.gamma_lambda.0 * val_next) - val;
            advs.push(adv);
        }
        advs.reverse();
        let mut gaes = Tensor::from_slice::<f64>(&[]);
        //Step-3: Cal generalised advantages after discounting:
        for i in 0..self.traj_depth {
            let exps = Tensor::arange(
                (self.traj_depth - i) as f64,
                (tch::Kind::Double, tch::Device::Cpu),
            );
            let weights = Tensor::full(
                [(self.traj_depth - i) as i64],
                self.gamma_lambda.0 * self.gamma_lambda.1,
                (tch::Kind::Double, tch::Device::Cpu),
            )
            .pow(&exps);
            let advs_t = Tensor::from_slice(advs.split_at(i as usize).1);
            let gae = (advs_t * weights).sum(tch::Kind::Double).unsqueeze(0);
            gaes = Tensor::concat(&[gaes, gae], 0);
        }
        gaes
    }

    fn compute_rtg(&self, rewards: &Vec<f64>, v_terminal: f64) -> Vec<f64> {
        let mut rtgs = vec![0.0_f64; self.traj_depth as usize];
        let mut running_rtg = v_terminal;
        for i in (0..self.traj_depth).rev() {
            let reward = rewards.get(i as usize).expect("");
            running_rtg = reward + (self.gamma_lambda.0 * running_rtg);
            rtgs[i as usize] = running_rtg;
        }
        rtgs
    }

    //Shape model_output_sample - [traj_depth,nu]
    //Shape model_output - [traj_depth,nu*2]
    //Output provides a log prob - [traj_depth]
    fn compute_action_log_probs(
        &self,
        model_output_sample: &Tensor,
        model_output: &Tensor,
    ) -> Tensor {
        let action_output_size = self.get_action_output_size();
        let s_action_gaussian_params =
            model_output.split_with_sizes([action_output_size, action_output_size], -1);
        let action_gaussian_mean = s_action_gaussian_params
            .get(0)
            .expect("Couldn't extract the action_gaussian_mean");
        let action_gaussian_log_stddev = s_action_gaussian_params
            .get(1)
            .expect("Couldn't extract the action_gaussian_log_stddev");
        let action_gaussian_stddev = action_gaussian_log_stddev.exp();
        let normalized = (model_output_sample - action_gaussian_mean) / action_gaussian_stddev;
        let log_two_pi = (2.0_f64 * std::f64::consts::PI).ln();
        let log_probs: Tensor = -0.5
            * (normalized.pow_tensor_scalar(2.0) + (action_gaussian_log_stddev * 2.0) + log_two_pi);
        let summed_log_probs = log_probs.sum_dim_intlist(vec![-1], false, Kind::Double);
        summed_log_probs
    }

    fn get_ppo_clip_objective(&self, trajs: &[Trajectory], gaes: &Tensor) -> Tensor {
        //Collect everything in singular tensors first:
        let start_log_probs_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| Tensor::from_slice(&traj.model_output_sample_log_probs))
            .collect();
        let wstates_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| Tensor::stack(&traj.world_states[..self.traj_depth as usize], 0))
            .collect();
        let action_samples_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| Tensor::stack(&traj.model_output_samples, 0))
            .collect();
        let start_log_probs = Tensor::stack(&start_log_probs_v, 0).detach();
        let wstates = Tensor::stack(&wstates_v, 0);
        let action_samples = Tensor::stack(&action_samples_v, 0).detach();

        let action_gaussian_params = self
            .model
            .lock()
            .expect("Model mutex poisoned")
            .get_policy_output(&wstates);
        let action_log_probs =
            self.compute_action_log_probs(&action_samples, &action_gaussian_params);

        let ratio = (action_log_probs - start_log_probs).exp();
        let clipped = ratio.clamp(1.0 - self.epsilon as f64, 1.0 + self.epsilon as f64) * gaes;
        (ratio * gaes).minimum(&clipped).mean(tch::Kind::Double)
    }

    fn get_value_function_objective(&self, trajs: &[Trajectory]) -> Tensor {
        let rtgs_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| Tensor::from_slice(&traj.rtgs))
            .collect();
        let wstates_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| Tensor::stack(&traj.world_states[..self.traj_depth as usize], 0))
            .collect();
        let rtgs = Tensor::stack(&rtgs_v, 0).detach();
        let (value_device, value_kind) = {
            let model = self.model.lock().expect("Model mutex poisoned");
            (
                model.get_value_store().device(),
                model.get_value_store().kind(),
            )
        };
        let wstates = Tensor::stack(&wstates_v, 0)
            .to_device(value_device)
            .to_kind(value_kind);
        let values = self
            .model
            .lock()
            .expect("Model mutex poisoned")
            .get_value_func()
            .forward(&wstates);
        let values = values.squeeze_dim(-1);
        let rtgs = rtgs.to_device_(values.device(), values.kind(), false, false);
        (values - rtgs).pow_tensor_scalar(2.0).mean(Kind::Double)
    }

    fn get_mean_policy_loss(&self, trajs: &[Trajectory], gaes: &Tensor) -> Tensor {
        -self.get_ppo_clip_objective(trajs, gaes)
    }

    fn get_mean_value_loss(&self, trajs: &[Trajectory]) -> Tensor {
        self.get_value_function_objective(trajs)
    }

    pub fn train_an_epoch(&mut self, trajs: &[Trajectory]) -> PPOTrainerRuntimeInfo {
        let gae_v: Vec<Tensor> = trajs.iter().map(|traj| self.compute_gae(traj)).collect();
        let gaes = Tensor::stack(&gae_v, 0).detach();
        let gae_mean = gaes.mean(Kind::Double);
        let gae_std = gaes.std(true);
        let gaes = (gaes - &gae_mean) / (&gae_std + 1e-8);
        let mut mean_policy_loss: Vec<f64> = Vec::new();
        let mut mean_value_loss: Vec<f64> = Vec::new();
        let mut policy_gradient_norm: Vec<f64> = Vec::new();
        let mut value_gradient_norm: Vec<f64> = Vec::new();
        //TODO: Expose iterations as param here
        for i in 0..self.num_updates {
            println!("Running epoch's update iteration {:?}", i);
            let policy_loss = self.get_mean_policy_loss(&trajs, &gaes);
            let policy_loss_v = f64::try_from(policy_loss.detach())
                .expect("Policy loss should be a scalar f64 tensor");
            mean_policy_loss.push(policy_loss_v);
            self.policy_opt.zero_grad();
            policy_loss.backward();
            let pol_variables = self
                .model
                .lock()
                .expect("Model mutex poisoned")
                .get_policy_store()
                .trainable_variables();
            policy_gradient_norm.push(Self::get_gradient_norm(pol_variables));
            self.policy_opt.step();

            let value_loss = self.get_mean_value_loss(&trajs);
            let value_loss_v = f64::try_from(value_loss.detach())
                .expect("Value loss should be a scalar f64 tensor");
            mean_value_loss.push(value_loss_v);
            self.value_opt.zero_grad();
            value_loss.backward();
            let val_variables = self
                .model
                .lock()
                .expect("Model mutex poisoned")
                .get_value_store()
                .trainable_variables();
            value_gradient_norm.push(Self::get_gradient_norm(val_variables));
            self.value_opt.step();
        }
        PPOTrainerRuntimeInfo::new(
            mean_policy_loss,
            mean_value_loss,
            policy_gradient_norm,
            value_gradient_norm,
        )
    }

    pub fn get_gradient_norm(variables: Vec<Tensor>) -> f64 {
        let mut total_norm_sq = 0.0;
        for tensor in variables {
            let grad = tensor.grad();
            if grad.defined() {
                let grad_sq = grad.pow_tensor_scalar(2.0).sum(Kind::Double);
                let grad_sq_v = f64::try_from(grad_sq).expect("Gradient norm should be scalar f64");
                total_norm_sq += grad_sq_v;
            }
        }
        total_norm_sq.sqrt()
    }

    pub fn get_rollout_mean_entropy(&self, trajs: &[Trajectory]) -> f64 {
        if trajs.is_empty() {
            return 0.0;
        }
        let action_output_size = self.get_action_output_size();
        let model_outputs_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| Tensor::stack(&traj.model_outputs, 0))
            .collect();
        let model_outputs = Tensor::stack(&model_outputs_v, 0);
        let s_action_gaussian_params =
            model_outputs.split_with_sizes([action_output_size, action_output_size], -1);
        let action_gaussian_log_stddev = s_action_gaussian_params
            .get(1)
            .expect("Couldn't extract the action_gaussian_log_stddev");
        let log_two_pi = (2.0_f64 * std::f64::consts::PI).ln();
        let entropy = (action_gaussian_log_stddev + 0.5 * (1.0 + log_two_pi))
            .sum_dim_intlist(vec![-1], false, Kind::Double)
            .mean(Kind::Double);
        f64::try_from(entropy).expect("wrong type of tensor")
    }

    pub fn get_rollout_mean_reward(&self, trajs: &[Trajectory]) -> f64 {
        let reward_count: usize = trajs.iter().map(|traj| traj.rewards.len()).sum();
        if reward_count == 0 {
            return 0.0;
        }
        let reward_sum: f64 = trajs.iter().flat_map(|traj| traj.rewards.iter()).sum();
        reward_sum / reward_count as f64
    }
}
