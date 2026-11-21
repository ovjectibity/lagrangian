use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use typeshare::typeshare;
use utoipa::ToSchema;

use app_proc_macros::variant_source;

use crate::commondef::records::{model::WorldBody, workload_graph::WorkloadGraph};

/// Kind of workload executed by the platform.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum WorkloadType {
    Training,
    Testing,
}

/// Origin from which a workload was launched.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum LaunchOrigin {
    Api,
}

/// Lifecycle status of a workload.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum WorkloadStatus {
    Completed,
    ExecutionFailed,
    LaunchFailed,
    Scheduled,
    Ongoing,
    UserStopped
}

/// Workload assembled from Postgres metadata and a MongoDB graph.
#[variant_source]
#[derive(
    Debug, Clone, Serialize, Deserialize, ToSchema,
)]
#[typeshare]
pub struct Workload {
    pub workload_id: String,
    pub workload_type: WorkloadType,
    pub project_id: String,
    pub user_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub launched_at: Option<DateTime<Utc>>,
    pub launch_origin: LaunchOrigin,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: WorkloadStatus,
    pub workload_graph_id: String,
    pub workload_graph: WorkloadGraph,
    /// Uploaded and generated models associated with this workload.
    pub simulation_model_ids: Vec<String>,
    pub tags: Vec<String>,
    pub name: String,
    pub description: String,
}

/// Reduction already computed by the worker for a metric.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum AggregationFunction {
    Mean,
    Sum,
}

/// Method to which a workload structure applies.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum WorkloadMethod {
    Ppo,
    Dataset,
    Custom,
}

/// Immutable definition used to interpret worker metric values.
///
/// Aggregation functions and indices pair by position and describe
/// successive reductions already performed by the worker. Their
/// lengths must match. Aggregated and non-applicable indices must
/// be disjoint and refer to the associated structure's dimensions.
/// The last remaining dimension is the x-axis; preceding remaining
/// dimensions identify separate series.
#[derive(
    Debug, Clone, Serialize, Deserialize, ToSchema,
)]
#[typeshare]
pub struct UserMetricsLut {
    pub metric_id: i32,
    pub name: String,
    pub workload_structure_id: i32,
    pub workload_type: WorkloadType,
    pub aggregation_functions: Vec<AggregationFunction>,
    pub aggregation_indices: Vec<u32>,
    pub na_indices: Vec<u32>,
}

/// Immutable ordered dimensions for a workload method.
///
/// Dimension positions define the indices used by metric and
/// asset records. The declared count must equal the names' length.
#[derive(
    Debug, Clone, Serialize, Deserialize, ToSchema,
)]
#[typeshare]
pub struct WorkloadStructureLut {
    pub workload_structure_id: i32,
    pub workload_type: WorkloadType,
    pub workload_state_index_names: Vec<String>,
    pub number_of_state_indices: u32,
    pub method_applicability: WorkloadMethod,
}

/// Severity of a workload log entry.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum LogType {
    Info,
    Debug,
    Trace,
    Error,
    Warning,
}

/// Component that emitted a workload log entry.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum LogStream {
    AppServer,
    Worker,
}

/// A document in the logs collection.
#[variant_source]
#[derive(
    Debug, Clone, Serialize, Deserialize, ToSchema,
)]
#[typeshare]
pub struct WorkloadLogs {
    pub workload_id: String,
    pub created_at: DateTime<Utc>,
    pub log_timestamp: DateTime<Utc>,
    pub log_id: String,
    pub log_line_text: String,
    pub user_facing: bool,
    pub log_type: LogType,
    pub stream: LogStream,
}

/// A batch of worker-computed values for one workload metric.
///
/// Each x-values entry holds coordinates in the metric's remaining
/// dimension order after aggregation and non-applicable dimensions
/// are excluded. The final coordinate is the chart x-axis.
/// The x-values and y-values arrays must have equal lengths.
#[variant_source]
#[derive(
    Debug, Clone, Serialize, Deserialize, ToSchema,
)]
#[typeshare]
pub struct WorkloadUserMetrics {
    pub metrics_entry_id: String,
    pub created_at: DateTime<Utc>,
    pub workload_id: String,
    pub metric_id: i32,
    pub workload_structure_state_values: Vec<Vec<u32>>,
    pub metric_values: Vec<f64>,
}

/// Simulation model state captured at a workload coordinate.
#[variant_source]
#[derive(
    Debug, Clone, Serialize, Deserialize, ToSchema,
)]
#[typeshare]
pub struct WorkloadSimulationModelTrajectoryState {
    pub created_at: DateTime<Utc>,
    pub sim_model_state_entry_id: String,
    pub workload_id: String,
    pub workload_structure_id: i32,
    /// Coordinates in the structure LUT's declared order.
    pub workload_structure_state: Vec<u32>,
    pub simulation_model_id: String,
    pub state: WorldBody,
}
