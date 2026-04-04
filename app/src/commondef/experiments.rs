use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

use crate::commondef::training::{
    TrainingGraph, TrainingMetrics,
};

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
pub struct Project {
    pub project_id: String,
    pub user_id: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
pub struct TrainingExperiment {
    pub project_id: String,
    pub experiment_name: String,
    pub training_experiment_id: String,
    pub created_at: DateTime<Utc>,
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
pub struct TrainingExperimentRun {
    pub project_id: String,
    pub training_experiment_id: String,
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub run_start_at: Option<DateTime<Utc>>,
    pub run_end_at: Option<DateTime<Utc>>,
    pub run_status: TrainingRunStatus,
    pub logs_id: String,
    pub metrics_id: String,
    pub training_graph_id: String,
    pub checkpoints_refs: Vec<String>,
    pub model_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStream {
    pub created_at: DateTime<Utc>,
    pub id: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingGraphRecord {
    pub created_at: DateTime<Utc>,
    pub id: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub graph: TrainingGraph,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
pub struct ModelReference {
    pub created_at: DateTime<Utc>,
    pub id: String,
    pub path: String,
    pub model_type: String,
}

#[derive(
    Debug, Clone, Serialize,
    Deserialize, FromRow,
)]
pub struct CheckpointReference {
    pub created_at: DateTime<Utc>,
    pub id: String,
    pub path: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub checkpoint_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsStream {
    pub created_at: DateTime<Utc>,
    pub id: String,
    pub training_experiment_id: String,
    pub training_run_id: String,
    pub training_metrics: Vec<TrainingMetrics>,
}
