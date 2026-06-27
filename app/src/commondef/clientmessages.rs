use serde;
use typeshare::typeshare;
use utoipa::ToSchema;

use crate::commondef::testing::{
    StopTestMessage, TestUpdateMessage,
};
use crate::commondef::training::{
    StopTrainingMessage, TrainingUpdatesMessage,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, ToSchema)]
#[serde(tag = "type", content = "content")]
#[typeshare]
pub enum ClientMessage {
    TrainingUpdatesMessage(TrainingUpdatesMessage),
    StopTrainingMessage(StopTrainingMessage),
    TestUpdatesMessage(TestUpdateMessage),
    StopTestMessage(StopTestMessage),
}
