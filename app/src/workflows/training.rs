use std::sync::{Arc, Mutex};

use rocket::tokio::sync::mpsc;

use crate::commondef::messages::{StopTrainingMessage, TrainingUpdatesMessage};
use crate::commondef::training::{
    ControlModel, MetricsData, PPOTrainingMethod, TrainingMetrics, ValueModel,
};
use crate::commondef::workermessages::TrainingTaskMessage;
use crate::plugin::ppo_trainer::{PPOTrainedModel, PPOTrainer};
use crate::plugin::{mj_simulator::MjSimulator, ControlModelOutput, Simulator};
use crate::usr::arm::ArmModel;

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

    pub fn new(model_path: String, 
        id: String, 
        params: PPOTrainingMethod) -> 
        Result<Self, String> {
        let simulator = 
            MjSimulator::new(model_path);
        let control_params = match &params.control_model {
            ControlModel::FFN(m) => m,
        };
        let value_params = match &params.value_model {
            ValueModel::FFN(m) => m,
        };
        let (control_model, 
            control_model_store) = 
            ArmModel::init_policy(
            control_params.input_dim as i64,
            control_params.output_dim as i64,
            control_params.hidden_layer_dim as i64,
            value_params.num_hidden_layers as i64,
            false,
        );
        let (value_model, 
            value_model_store) = 
            ArmModel::init_policy(
            value_params.input_dim as i64,
            value_params.output_dim as i64,
            value_params.hidden_layer_dim as i64,
            value_params.num_hidden_layers as i64,
            false,
        );

        if let Ok(s) = simulator {
            let model = ArmModel::init_ppo_trained_model(
                control_model,
                value_model,
                control_model_store,
                value_model_store,
                s.nq() as i64,
            );
            let simulator_p: Arc<Mutex<dyn Simulator + Send>> = 
                Arc::new(Mutex::new(s));
            let model_p: Arc<Mutex<dyn PPOTrainedModel + Send>> = 
                Arc::new(Mutex::new(model));
            let trainer = PPOTrainer::new(
                simulator_p.clone(),
                model_p.clone(),
                20,
                1000,
                100,
                // params.num_trajs,
                params.epsilon,
                ControlModelOutput::TargetPositions,
            );

            if let Ok(t) = trainer {
                Ok(TrainingWorkflow {
                    //TODO: Fix the identifier here
                    id: id,
                    simulator: simulator_p.clone(),
                    trainer: t,
                    model: model_p.clone(),
                    params: params,
                })
            } else {
                Err("Couldn't create trainer".to_string())
            }
        } else {
            Err("Couldn't create simulator".to_string())
        }
    }

    pub fn run_training_blocking(&mut self, 
        tx: mpsc::Sender<TrainingTaskMessage>) {
        let params = &self.params;
        for i in 0..params.num_epochs {
            Self::log_and_send_blocking(
                &tx,
                self.id.clone(),
                format!("Running the training epoch {i}"),
            );
            //TODO: Expose num trajs as a param
            let trajs = self.trainer.gen_trajectories(10);
            let runtime_info = self.trainer.train_an_epoch(&trajs);
            let mean_entropy = self.trainer.get_rollout_mean_entropy(&trajs);
            let mean_reward = self.trainer.get_rollout_mean_reward(&trajs);
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
                x_axis: update_steps,
                y_axis: runtime_info.mean_value_loss().to_vec(),
            });
            let update = 
                TrainingTaskMessage::TrainingUpdates(
                    TrainingUpdatesMessage {
                        id: self.id.clone(),
                        metrics: vec![
                            mean_entropy_m, 
                            mean_reward_m,
                            mean_policy_loss_m,
                            mean_value_loss_m
                        ],
                        logs: vec![],
                    });
            let _ = tx.blocking_send(update);
        }
        let _ = tx.blocking_send(
            TrainingTaskMessage::StopTraining(StopTrainingMessage {
                id: self.id.clone(),
            })
        );
    }
}
