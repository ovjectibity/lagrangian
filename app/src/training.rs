use crate::plugin::{Simulator, ppo_trainer::{PPOTrainedModel, PPOTrainer}};

pub struct TrainingWorkflow {
    simulator: Box<dyn Simulator>,
    trainer: Box<PPOTrainer>,
    model: Box<dyn PPOTrainedModel>
}

impl TrainingWorkflow {
    fn run_training() {

    }
}