use crate::commondef::training::{
    StopTrainingMessage, TrainingUpdatesMessage,
};
use crate::commondef::testing::{
    StopTestMessage, TestUpdateMessage,
};

#[derive(Debug, Clone)]
pub enum TrainingTaskMessage {
    TrainingUpdates(TrainingUpdatesMessage),
    StopTraining(StopTrainingMessage),
}

#[derive(Debug, Clone)]
pub enum TestTaskMessage {
    StopTestMessage(StopTestMessage),
    TestUpdateMessage(TestUpdateMessage),
}
