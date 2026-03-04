use std::sync::{Arc, Mutex};

use rocket::tokio::sync::mpsc;

use crate::commondef::clientmessages::{
    StopTrainingMessage, TrainingUpdatesMessage
};
use crate::commondef::training::{
    MetricsData, PPOTrainingMethod, TrainingMetrics
};
use crate::commondef::workermessages::TrainingTaskMessage;
use crate::plugin::ppo_trainer::{
    PPOTrainedModel, PPOTrainer
};
use crate::plugin::{
    ControlPolicyOutputMode, Simulator
};

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
        msg: String,
    ) {
        println!("{msg}");
        let log_update = 
            TrainingTaskMessage::TrainingUpdates(
                TrainingUpdatesMessage {
                    id: training_id,
                    metrics: vec![],
                    logs: vec![msg],
                });
        let _ = tx.blocking_send(log_update);
    }

    pub fn new(
        id: String,
        params: PPOTrainingMethod,
        simulator: Arc<Mutex<dyn Simulator + Send>>,
        model: Arc<Mutex<dyn PPOTrainedModel + Send>>,
    ) -> Result<Self, String> {
        let trainer = PPOTrainer::new(
            simulator.clone(),
            model.clone(),
            100,
            10000,
            10,
            // params.num_trajs,
            params.epsilon
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
        tx: mpsc::Sender<TrainingTaskMessage>) {
        let params = &self.params;
        for i in 0..params.num_epochs {
            Self::log_and_send_blocking(
                &tx,
                self.id.clone(),
                format!("Running the training epoch {i}"),
            );
            //TODO: Expose num trajs as a param
            let trajs = 
                self.trainer.gen_trajectories(10);
            let runtime_info = 
                self.trainer.train_an_epoch(&trajs);
            let mean_entropy = 
                self.trainer.get_rollout_mean_entropy(&trajs);
            let mean_reward = 
                self.trainer.get_rollout_mean_reward(&trajs);
            let mean_entropy_m = 
            TrainingMetrics::MeanEntropy(MetricsData {
                x_axis: vec![i as f64],
                y_axis: vec![mean_entropy],
            });
            let mean_reward_m = 
            TrainingMetrics::MeanReward(MetricsData {
                x_axis: vec![i as f64],
                y_axis: vec![mean_reward],
            });
            let num_updates = runtime_info.mean_policy_loss().len();
            let base_update_step = (i as usize) * num_updates;
            let update_steps: Vec<f64> = (0..num_updates)
                .map(|u| (base_update_step + u) as f64)
                .collect();
            let mean_policy_loss_m = 
            TrainingMetrics::MeanPolicyLoss(MetricsData {
                x_axis: update_steps.clone(),
                y_axis: runtime_info.mean_policy_loss().to_vec(),
            });
            let mean_value_loss_m = 
            TrainingMetrics::ValueLoss(MetricsData {
                x_axis: update_steps.clone(),
                y_axis: runtime_info.mean_value_loss().to_vec(),
            });
            let policy_grad_norm_m =
            TrainingMetrics::PolicyGradientNorm(MetricsData {
                x_axis: update_steps.clone(),
                y_axis: runtime_info.policy_gradient_norm().to_vec(),
            });
            let value_grad_norm_m =
            TrainingMetrics::ValueGradientNorm(MetricsData {
                x_axis: update_steps,
                y_axis: runtime_info.value_gradient_norm().to_vec(),
            });
            let update = 
            TrainingTaskMessage::TrainingUpdates(
                TrainingUpdatesMessage {
                    id: self.id.clone(),
                    metrics: vec![
                        mean_entropy_m,
                        mean_reward_m,
                        mean_policy_loss_m,
                        mean_value_loss_m,
                        policy_grad_norm_m,
                        value_grad_norm_m,
                    ],
                    logs: vec![],
                });
            let _ = tx.blocking_send(update);
        }
        let _ = tx.blocking_send(
            TrainingTaskMessage::StopTraining(
                StopTrainingMessage {
                    id: self.id.clone(),
                }));
    }
}
