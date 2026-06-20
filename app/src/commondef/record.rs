use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use typeshare::typeshare;

use crate::commondef::{testing::TestGraph, training::{
    MLPModel, TrainingGraph, TrainingMetrics,
}};

#[typeshare(serialized_as = "string")]
pub type Timestamp = DateTime<Utc>;

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
#[typeshare]
pub struct Project {
    pub project_id: String,
    pub user_id: i32,
    pub created_at: Timestamp,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
#[typeshare]
pub struct TrainingExperiment {
    pub project_id: String,
    pub experiment_name: String,
    pub training_experiment_id: String,
    pub created_at: Timestamp,
    pub training_run_ids: Vec<String>,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, sqlx::Type, PartialEq,
)]
#[sqlx(
    type_name = "text",
    rename_all = "snake_case"
)]
#[typeshare]
pub enum TrainingRunStatus {
    Ongoing,
    Completed,
    StoppedByUser,
    NotStarted,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
#[typeshare]
pub struct TrainingExperimentRun {
    pub project_id: String,
    pub training_experiment_id: String,
    pub id: String,
    pub created_at: Timestamp,
    pub run_start_at: Option<Timestamp>,
    pub run_end_at: Option<Timestamp>,
    pub run_status: TrainingRunStatus,
    pub logs_id: String,
    pub metrics_id: String,
    pub training_graph_id: String,
    pub checkpoints_refs: Vec<String>,
    pub model_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[typeshare]
pub struct LogStream {
    pub created_at: Timestamp,
    pub id: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[typeshare]
pub struct TrainingGraphRecord {
    pub created_at: Timestamp,
    pub id: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub graph: TrainingGraph,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, sqlx::Type, PartialEq,
)]
#[sqlx(
    type_name = "text",
    rename_all = "snake_case"
)]
#[typeshare]
pub enum TestRunStatus {
    Ongoing,
    Completed,
    StoppedByUser,
    NotStarted,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
#[typeshare]
pub struct TestRun {
    pub project_id: String,
    pub id: String,
    pub created_at: Timestamp,
    pub run_start_at: Option<Timestamp>,
    pub run_end_at: Option<Timestamp>,
    pub run_status: TestRunStatus,
    pub logs_id: String,
    pub metrics_id: String,
    pub test_graph_id: String
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[typeshare]
pub struct TestGraphRecord {
    pub created_at: Timestamp,
    pub id: String,
    pub test_run_id: String,
    pub graph: TestGraph,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
#[typeshare]
pub struct ModelReference {
    pub created_at: Timestamp,
    pub id: String,
    pub name: String,
    pub path: String,
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct CheckpointReference {
    pub created_at: Timestamp,
    pub id: String,
    pub path: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub checkpoint_type: String,
    pub control_model_params: Option<MLPModel>,
    pub value_model_params: Option<MLPModel>,
}

impl<'r, R> sqlx::FromRow<'r, R> for CheckpointReference
where
    R: sqlx::Row,
    &'r str: sqlx::ColumnIndex<R>,
    Timestamp: sqlx::decode::Decode<'r, R::Database>
        + sqlx::types::Type<R::Database>,
    String: sqlx::decode::Decode<'r, R::Database>
        + sqlx::types::Type<R::Database>,
    Option<String>: sqlx::decode::Decode<'r, R::Database>
        + sqlx::types::Type<R::Database>,
{
    fn from_row(row: &'r R) -> Result<Self, sqlx::Error> {
        let created_at: Timestamp =
            row.try_get("created_at")?;
        let id: String = row.try_get("id")?;
        let path: String = row.try_get("path")?;
        let training_experiment_id: String =
            row.try_get("training_experiment_id")?;
        let training_run_id: String =
            row.try_get("training_run_id")?;
        let checkpoint_type: String =
            row.try_get("checkpoint_type")?;
        let control_model_params_raw: Option<String> =
            row.try_get("control_model_params")?;
        let value_model_params_raw: Option<String> =
            row.try_get("value_model_params")?;
        Ok(CheckpointReference {
            created_at,
            id,
            path,
            training_experiment_id,
            training_run_id,
            checkpoint_type,
            control_model_params: control_model_params_raw
                .and_then(|s| serde_json::from_str(&s).ok()),
            value_model_params: value_model_params_raw
                .and_then(|s| serde_json::from_str(&s).ok()),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[typeshare]
pub struct MetricsStream {
    pub created_at: Timestamp,
    pub id: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub training_metrics: Vec<TrainingMetrics>,
}
