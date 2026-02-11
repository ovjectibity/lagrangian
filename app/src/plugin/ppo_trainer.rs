use tch::{Kind, Tensor, nn::{self, Module, OptimizerConfig}};

use crate::plugin::{Simulator};

struct Trajectory {
    //Shape of indv element: [nq] 
    world_states: Vec<Tensor>,
    //Shape of indv element: [nq+nq]
    //Note: the std devs are logs
    model_outputs: Vec<Tensor>,
    //Shape of indv element: [nq]
    model_output_samples: Vec<Tensor>,
    model_output_sample_log_probs: Vec<f64>,
    rewards: Vec<f64>,
    rtgs: Vec<f64>
}

pub struct PPOTrainer {
    num_trajs: i32,
    epsilon: f64,
    //Used for GAE: 
    gamma_lambda: (f64,f64),
    traj_depth: i32,
    simulator: Box<dyn Simulator>,
    model: Box<dyn PPOTrainedModel>,
    policy_opt: nn::Optimizer,
    value_opt: nn::Optimizer
}

pub trait PPOTrainedModel {
    fn init_ppo_trained_model(
        policy: nn::Sequential, 
        val_estimator: nn::Sequential,
        pol_store: nn::VarStore,
        val_estimator_store: nn::VarStore) -> Self where Self: Sized;
    fn get_nq(&self) -> i64;
    // Using SpinningUp's formulation of reward function: 
    // https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
    fn get_reward(&self,wstate: &Tensor, 
        wstatenext: Option<&Tensor>, action: &Tensor) -> f64;
    fn get_policy_output(&self,state: &Tensor) -> Tensor;
    fn get_value_output(&self,state: &Tensor) -> f64;
    fn get_policy(&self) -> &nn::Sequential;
    fn get_value_func(&self) -> &nn::Sequential;
    fn get_policy_store(&self) -> &nn::VarStore;
    fn get_policy_store_mut(&mut self) -> &mut nn::VarStore;
    fn get_value_store(&self) -> &nn::VarStore;
    fn get_value_store_mut(&mut self) -> &mut nn::VarStore;
}

//TODO: Optimising with aggregating all 
// computations to single tensor where possible 
impl PPOTrainer {
    fn new(
        simulator: Box<dyn Simulator>,
        model: Box<dyn PPOTrainedModel>,
        num_trajs: i32,
        epsilon: f64,
    ) -> Result<Self, tch::TchError> {
        let policy_opt = nn::Adam::default().build(
            model.get_policy_store(), 1e-3)?;
        let value_opt = nn::Adam::default().build(
            model.get_value_store(), 1e-3)?;
        Ok(PPOTrainer {
            num_trajs: num_trajs,
            epsilon: epsilon,
            gamma_lambda: (0.1,0.1),
            traj_depth: 10,
            simulator: simulator,
            model: model,
            policy_opt: policy_opt,
            value_opt: value_opt
        })
    }

    fn gen_trajectory(&self) -> Trajectory {
        let nq = self.model.get_nq();
        let mut wstates = Vec::new();
        let mut model_outputs = Vec::new();
        let mut model_output_samples = Vec::new();
        let mut rewards = Vec::new();
        for i in 0..self.traj_depth {
            let wstate = self.simulator.get_model_state();
            let action_gaussian_params = 
                self.model.get_policy_output(&wstate);
            //Output action needs to be sampled & 
            // then passed on to the simulator
            let s_action_gaussian_params = 
                action_gaussian_params.split_with_sizes([nq,nq], 0);
            let action_gaussian_mean = s_action_gaussian_params.get(0).
                expect("Couldn't extract the action_gaussian_mean");
            let action_gaussian_stdev = s_action_gaussian_params.get(1).
                expect("Couldn't extract the action_gaussian_stddev").exp();
            let eps = Tensor::rand([nq],
                (tch::Kind::Float, tch::Device::Cpu));
            let sampled_action = action_gaussian_mean + &eps * &action_gaussian_stdev;
            let controls = 
                self.simulator.get_target_controls_from_target_positions(
                    &sampled_action);
            self.simulator.forward_dyn(&controls);
            //TODO: Assuming no PPOTrained model is using wstatenext for now: 
            rewards.push(self.model.get_reward(&wstate, None,
                    &sampled_action));
            wstates.push(wstate);
            model_outputs.push(action_gaussian_params);
            model_output_samples.push(sampled_action);
        }
        let rtgs = self.compute_rtg(&rewards);
        let action_sample_log_probs = self.compute_action_log_probs(
            &Tensor::stack(&model_output_samples,0), 
            &Tensor::stack(&model_outputs,0));
        Trajectory { 
            world_states: wstates, 
            model_outputs: model_outputs,
            model_output_samples: model_output_samples,
            rewards: rewards,
            rtgs: rtgs,
            model_output_sample_log_probs: 
                Vec::<f64>::try_from(action_sample_log_probs).
                expect("wrong type of tensor")
        }
    }

    fn gen_trajectories(&self, num_trajs: i32) -> Vec<Trajectory> {
        let mut trajs = Vec::new();
        for i in 0..num_trajs {
            trajs.push(self.gen_trajectory());
        }
        trajs
    }

    //Output shape: [traj_depth]
    fn compute_gae(&self,traj: &Trajectory) -> Tensor {
        let mut advs =  Vec::new();
        //Step-1: Get all rewards for the trajectory: 
        //Step-2: Compute the advantage: 
        for i in (0..self.traj_depth).rev() {
            let wstate = 
                traj.world_states.get(i as usize).expect("");
            let reward = traj.rewards.
                get(i as usize).expect("");
            let val = self.model.get_value_output(wstate);
            if i != (self.traj_depth - 1) {
                let wstate_next = traj.world_states.get(
                    (i + 1) as usize).expect("");
                let val_next = self.model.get_value_output(wstate_next);
                let adv = reward + (self.gamma_lambda.0 * val_next) - val;
                advs.push(adv);
            } else {
                let adv = reward - val;
                advs.push(adv);
            }
        }
        advs.reverse();
        let mut gaes = Tensor::from_slice::<f64>(&[]);
        //Step-3: Cal generalised advantages after discounting: 
        for i in 0..self.traj_depth {
            let exps = Tensor::arange(
                (self.traj_depth - i) as f64, 
                (tch::Kind::Float, tch::Device::Cpu));
            let weights = Tensor::full(
                [(self.traj_depth - i) as i64],
                self.gamma_lambda.0 * self.gamma_lambda.1,
                (tch::Kind::Float,tch::Device::Cpu)).
                pow(&exps);
            let advs_t = Tensor::from_slice(
                advs.split_at(i as usize).1);
            let gae = (advs_t * weights).
                sum(tch::Kind::Float);
            gaes = Tensor::concat(&[gaes,gae], 0);
        }
        gaes
    }

    fn compute_rtg(&self,rewards: &Vec<f64>) -> Vec<f64> {
        let mut rtgs = Tensor::from_slice::<f64>(&[]);
        for i in 0..self.traj_depth {
            let exps = Tensor::arange(
                (self.traj_depth - i) as f64, 
                (tch::Kind::Float, tch::Device::Cpu));
            let weights = Tensor::full(
                [(self.traj_depth - i) as i64],
                self.gamma_lambda.0,
                (tch::Kind::Float,tch::Device::Cpu)).
                pow(&exps);
            let rewards_t = Tensor::from_slice(
                rewards.split_at(i as usize).1);
            let rtg = (rewards_t * weights).
                sum(tch::Kind::Float);
            rtgs = Tensor::concat(&[rtgs,rtg], 0);
        }
        Vec::<f64>::try_from(rtgs).expect("wrong type of tensor")
    }

    //Shape model_output_sample - [traj_depth,nq]
    //Shape model_output - [traj_depth,nq+nq]
    //Output provides a log prob - [traj_depth]
    fn compute_action_log_probs(&self,
        model_output_sample: &Tensor, 
        model_output: &Tensor) -> Tensor {
        let nq = self.model.get_nq();
        let s_action_gaussian_params = 
            model_output.split_with_sizes([nq, nq], -1);
        let action_gaussian_mean = 
            s_action_gaussian_params
            .get(0)
            .expect("Couldn't extract the action_gaussian_mean");
        let action_gaussian_log_stddev = 
            s_action_gaussian_params
            .get(1)
            .expect("Couldn't extract the action_gaussian_log_stddev");
        let action_gaussian_stddev = 
            action_gaussian_log_stddev.exp();
        let normalized = 
            (model_output_sample - action_gaussian_mean) / 
            action_gaussian_stddev;
        let log_two_pi = (2.0_f64 * std::f64::consts::PI).ln();
        let log_probs: Tensor = -0.5
            * (normalized.pow_tensor_scalar(2.0)
                + (action_gaussian_log_stddev * 2.0)
                + log_two_pi);
        let summed_log_probs = 
            log_probs.sum_dim_intlist(vec![-1], 
                false, Kind::Float);
        summed_log_probs
    }

    fn get_ppo_clip_objective(&self, 
        trajs: &[Trajectory]) -> Tensor {
        //Collect everything in singular tensors first:
        let start_log_probs_v: Vec<Tensor> = 
        trajs.iter().map(|traj| {
            Tensor::from_slice(&traj.model_output_sample_log_probs)
        }).collect();
        let gae_v: Vec<Tensor> = trajs.iter().map(|traj| {
            self.compute_gae(traj)
        }).collect();
        let wstates_v: Vec<Tensor> = trajs.iter().map(|traj| {
            Tensor::stack(&traj.world_states,0)
        }).collect();
        let action_samples_v: Vec<Tensor> = trajs.iter().map(|traj| {
            Tensor::stack(&traj.model_output_samples,0)
        }).collect();
        let start_log_probs = 
                Tensor::stack(&start_log_probs_v,0).detach();
        let wstates = Tensor::stack(&wstates_v,0);
        let action_samples = Tensor::stack(&action_samples_v,0);
        let gaes = Tensor::stack(&gae_v,0).detach();
        
        let action_gaussian_params = 
                self.model.get_policy_output(&wstates);
        let action_log_probs = self.compute_action_log_probs(
            &action_samples,
            &action_gaussian_params);
        
        let ratio = (action_log_probs - start_log_probs).exp();
        let clipped = ratio.clamp(
            1.0 - self.epsilon as f64, 
            1.0 + self.epsilon as f64) * &gaes;
        (ratio * gaes).minimum(&clipped).sum(tch::Kind::Float)
    }

    fn get_value_function_objective(&self, 
        trajs: &[Trajectory]) -> Tensor {
        let nq = self.model.get_nq();
        let rtgs_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| 
                Tensor::from_slice(&traj.rtgs))
            .collect();
        let wstates_v: Vec<Tensor> = trajs
            .iter()
            .map(|traj| 
                Tensor::stack(&traj.world_states, 0))
            .collect();
        let rtgs = Tensor::stack(&rtgs_v, 0).detach();
        let wstates = Tensor::stack(&wstates_v, 0);
        let values = self.model.get_value_func().forward(&wstates);
        let rtgs = rtgs.to_device_(
            values.device(),
            values.kind(),
            false,
            false,
        );
        (values - rtgs).pow_tensor_scalar(2.0).mean(Kind::Float)
    }

    fn train_an_epoch(&mut self) {
        let trajs = self.gen_trajectories(self.num_trajs);
        for _ in 0..10 {
            let objective = self.get_ppo_clip_objective(&trajs);
            let loss = -objective;
            self.policy_opt.backward_step(&loss);
            let value_loss = self.get_value_function_objective(&trajs);
            self.value_opt.backward_step(&value_loss);
        }
    }

    fn train_n_epochs(&mut self, n: u32) {
        for _ in 0..n {
            self.train_an_epoch();
        }
    }
}
