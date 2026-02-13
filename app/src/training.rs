use std::rc::Rc;
use std::cell::RefCell;

use rocket::tokio::sync::mpsc;

use crate::commondef::training::{
    ControlModel, PPOTrainingMethod,
    MetricsData, ValueModel, 
    TrainingMetrics
};
use crate::plugin::{mj_simulator::MjSimulator};
use crate::plugin::ppo_trainer::{PPOTrainedModel, PPOTrainer};
use crate::usr::arm::ArmModel;
use crate::commondef::workermessages::{TrainingTaskMessage};
use crate::commondef::messages::TrainingUpdatesMessage;

pub struct TrainingWorkflow {
    id: String,
    simulator: Rc<RefCell<MjSimulator>>,
    model: Rc<RefCell<ArmModel>>,
    trainer: PPOTrainer,
    params: PPOTrainingMethod
}

impl TrainingWorkflow {
    fn new(params: PPOTrainingMethod) -> Result<Self,String> {
        let simulator =  MjSimulator::new("".to_string());
        let control_params = match &params.control_model {
            ControlModel::FFN(m) => m
        };
        let value_params = match &params.value_model {
            ValueModel::FFN(m) => m
        };
        let (control_model,control_model_store) = 
            ArmModel::init_policy(
                control_params.input_dim as i64,
                control_params.output_dim as i64,
                control_params.hidden_layer_dim as i64,
                // TODO: Possible bug here, hidden vs total layers
                control_params.num_hidden_layers as i64,
                true
            );
        let (value_model,value_model_store) = 
            ArmModel::init_policy(
                value_params.input_dim as i64,
                value_params.output_dim as i64,
                value_params.hidden_layer_dim as i64,
                // TODO: Possible bug here, hidden vs total layers
                value_params.num_hidden_layers as i64,
                true
            );
        let model = 
            ArmModel::init_ppo_trained_model(
                control_model, value_model, 
                control_model_store, value_model_store);
        
        if let Ok(s) = simulator {
            let simulator_p = 
                Rc::new(RefCell::new(s));
            let model_p = 
                Rc::new(RefCell::new(model));
            let trainer = 
                PPOTrainer::new(simulator_p.clone(),
                model_p.clone(),
                10,
                // params.num_trajs,
                params.epsilon
            );

            if let Ok(t) = trainer {
                Ok(TrainingWorkflow {
                    //TODO: Fix the identifier here
                    id: "".to_string(),
                    simulator: simulator_p.clone(),
                    trainer: t, 
                    model: model_p.clone(),
                    params: params
                })
            } else {
                Err("Couldn't create trainer".to_string())
            }
        } else {
            Err("Couldn't create simulator".to_string())
        }
    }

    fn run_training(&mut self,
        tx: mpsc::Sender<TrainingTaskMessage>) {
        let params = &self.params;
        for i in 0..params.num_epochs {
            //TODO: Expose num trajs as a param
            let trajs = self.trainer.gen_trajectories(10);
            self.trainer.train_an_epoch(&trajs);
            let mean_entropy = self.trainer.get_rollout_mean_entropy(&trajs);
            let mean_reward = self.trainer.get_rollout_mean_reward(&trajs);
            let mean_entropy_m = 
            TrainingMetrics::MeanEntropy(MetricsData {
                x_axis: vec![i as f64],
                y_axis: vec![mean_entropy]
            });
            let mean_reward_m = 
            TrainingMetrics::MeanReward(MetricsData {
                x_axis: vec![i as f64],
                y_axis: vec![mean_reward]
            });
            let update = 
                TrainingTaskMessage::TrainingUpdates(
                    TrainingUpdatesMessage {
                        id: self.id.clone(),
                        metrics: vec![mean_entropy_m,mean_reward_m],
                        logs: vec![]
                    });
            tx.send(update);
        }
    }
}