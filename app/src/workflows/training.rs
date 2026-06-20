use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use rocket::tokio::sync::mpsc;
use tch::Tensor;

use crate::commondef::training::{
    MetricsData, PPOTrainingMethod, PPOTrainingRunState, TrainingMetrics, TrainingRunState,
    StopTrainingMessage, TrainingUpdatesMessage, TrainingUpdate,
};
use crate::commondef::workermessages::TrainingTaskMessage;
use crate::simulator::{Simulator};
use crate::models::{PPOTrainedModel, ControlPolicyModel};
use crate::workflows::ppo_trainer::{PPOTrainer};

pub struct TrainingWorkflow {
    id: String,
    simulator: Arc<Mutex<dyn Simulator + Send>>,
    model: Arc<Mutex<dyn PPOTrainedModel + Send>>,
    trainer: PPOTrainer,
    params: PPOTrainingMethod,
}

impl TrainingWorkflow {
    fn log_and_send_blocking(
        tx: &mpsc::Sender<TrainingTaskMessage>,
        training_id: String,
        training_run_state: TrainingRunState,
        msg: String,
    ) {
        println!("{msg}");
        let log_update = TrainingTaskMessage::TrainingUpdates(TrainingUpdatesMessage {
            id: training_id,
            updates: vec![TrainingUpdate {
                training_run_state,
                logs: vec![msg],
                metrics: vec![],
                checkpoints: vec![],
            }],
        });
        let _ = tx.blocking_send(log_update);
    }

    pub fn new(
        id: String,
        params: PPOTrainingMethod,
        simulator: Arc<Mutex<dyn Simulator + Send>>,
        model: Arc<Mutex<dyn PPOTrainedModel + Send>>,
        reward_fn: Option<
            Box<
                dyn Fn(
                    &HashMap<String, Tensor>,
                    &HashMap<String, Tensor>,
                ) -> f64
                    + Send,
            >,
        >,
    ) -> Result<Self, String> {
        let trainer = PPOTrainer::new(
            simulator.clone(),
            model.clone(),
            params.num_trajs,
            params.num_steps_per_traj,
            params.num_updates,
            params.epsilon,
            (params.gamma, params.lambda),
            reward_fn,
        );

        if let Ok(t) = trainer {
            Ok(TrainingWorkflow {
                id: id,
                simulator: simulator,
                trainer: t,
                model: model,
                params: params,
            })
        } else {
            Err("Couldn't create trainer".to_string())
        }
    }

    pub fn run_training_blocking(
        &mut self,
        tx: mpsc::Sender<TrainingTaskMessage>,
        stop_flag: Arc<AtomicBool>,
    ) {
        let params = &self.params;
        for i in 0..params.num_epochs {
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }
            let epoch_state = TrainingRunState::PPOTrainingRunState(PPOTrainingRunState {
                epoch_index: i,
                update_index: 0,
            });
            Self::log_and_send_blocking(
                &tx,
                self.id.clone(),
                epoch_state,
                format!("Running the training epoch {i}"),
            );
            let trajs = self.trainer.gen_trajectories(params.num_trajs);
            let runtime_info = self.trainer.train_an_epoch(&trajs);
            let mean_entropy = self.trainer.get_rollout_mean_entropy(&trajs);
            let mean_reward = self.trainer.get_rollout_mean_reward(&trajs);
            let mean_entropy_m = TrainingMetrics::MeanEntropy(MetricsData {
                x_axis: vec![i as f64],
                y_axis: vec![mean_entropy],
            });
            let mean_reward_m = TrainingMetrics::MeanReward(MetricsData {
                x_axis: vec![i as f64],
                y_axis: vec![mean_reward],
            });
            let num_updates = runtime_info.mean_policy_loss().len();
            let base_update_step = (i as usize) * num_updates;
            let update_steps: Vec<f64> = (0..num_updates)
                .map(|u| (base_update_step + u) as f64)
                .collect();
            let mean_policy_loss_m = TrainingMetrics::MeanPolicyLoss(MetricsData {
                x_axis: update_steps.clone(),
                y_axis: runtime_info.mean_policy_loss().to_vec(),
            });
            let mean_value_loss_m = TrainingMetrics::ValueLoss(MetricsData {
                x_axis: update_steps.clone(),
                y_axis: runtime_info.mean_value_loss().to_vec(),
            });
            let policy_grad_norm_m = TrainingMetrics::PolicyGradientNorm(MetricsData {
                x_axis: update_steps.clone(),
                y_axis: runtime_info.policy_gradient_norm().to_vec(),
            });
            let value_grad_norm_m = TrainingMetrics::ValueGradientNorm(MetricsData {
                x_axis: update_steps,
                y_axis: runtime_info.value_gradient_norm().to_vec(),
            });
            let epoch_end_state = TrainingRunState::PPOTrainingRunState(PPOTrainingRunState {
                epoch_index: i,
                update_index: num_updates as i32 - 1,
            });
            let update = TrainingTaskMessage::TrainingUpdates(TrainingUpdatesMessage {
                id: self.id.clone(),
                updates: vec![TrainingUpdate {
                    training_run_state: epoch_end_state,
                    logs: vec![],
                    metrics: vec![
                        mean_entropy_m,
                        mean_reward_m,
                        mean_policy_loss_m,
                        mean_value_loss_m,
                        policy_grad_norm_m,
                        value_grad_norm_m,
                    ],
                    checkpoints: vec![],
                }],
            });
            let _ = tx.blocking_send(update);
        }
        let _ = tx.blocking_send(TrainingTaskMessage::StopTraining(StopTrainingMessage {
            id: self.id.clone(),
        }));
    }
}
