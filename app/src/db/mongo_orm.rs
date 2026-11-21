//! Mongo persistence for graphs, logs, metrics and poses.

use chrono::{DateTime, SecondsFormat, Utc};
use futures_util::TryStreamExt;
use mongodb::bson::{self, Bson, Document, doc};
use mongodb::error::{Error, Result};
use mongodb::{Client, Database};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::commondef::records::workload::{
    Workload, WorkloadLogs,
    WorkloadSimulationModelTrajectoryState, WorkloadType,
    WorkloadUserMetrics,
};
use crate::commondef::records::workload_graph::WorkloadGraph;

use super::db_orm::PostgresOrm;
use super::validations;

const GRAPHS: &str = "workload_graph";
const LOGS: &str = "workload_logs";
const METRICS: &str = "workload_user_metrics";
const STATES: &str = "simulation_model_trajectory_state";

/// Document persistence; asset bytes are managed by the caller.
pub struct MongoOrm {
    db: Database,
}

impl MongoOrm {
    /// Use the URI database, or MONGODB_DB (default: lagrangian).
    pub async fn new(uri: &str) -> Result<Self> {
        let client = Client::with_uri_str(uri).await?;
        let name = std::env::var("MONGODB_DB")
            .unwrap_or_else(|_| "lagrangian".into());
        let db = client
            .default_database()
            .unwrap_or_else(|| client.database(&name));
        Ok(Self { db })
    }

    /// Insert a graph envelope using the supplied workload IDs.
    pub async fn create_workload_graph(
        &self,
        postgres: &PostgresOrm,
        workload: &Workload,
    ) -> Result<()> {
        validations::validate_workload(postgres, workload)
            .await
            .map_err(validation_error)?;
        self.db
            .collection::<Document>(GRAPHS)
            .insert_one(graph_document(workload)?)
            .await?;
        Ok(())
    }

    /// Read a graph and verify its workload identity and type.
    pub async fn get_workload_graph(
        &self,
        graph_id: &str,
        workload_id: &str,
        workload_type: &WorkloadType,
    ) -> Result<Option<WorkloadGraph>> {
        let record = self
            .db
            .collection::<Document>(GRAPHS)
            .find_one(doc! { "workload_graph_id": graph_id })
            .await?;
        let Some(record) = record else {
            return Ok(None);
        };
        let stored_type: WorkloadType = bson::from_bson(
            record.get("workload_type").cloned().ok_or_else(
                || Error::custom("Missing graph type"),
            )?,
        )?;
        validations::validate_graph_owner(
            record.get_str("workload_id").map_err(|error| {
                Error::custom(error.to_string())
            })?,
            workload_id,
        )
        .map_err(validation_error)?;
        validations::validate_graph_envelope_type(
            &stored_type,
            workload_type,
        )
        .map_err(validation_error)?;
        // Reject malformed storage timestamps even though the
        // Workload uses its Postgres creation timestamp.
        record
            .get_datetime("created_at")
            .map_err(|error| Error::custom(error.to_string()))?;
        let value =
            record.get("graph").cloned().ok_or_else(|| {
                Error::custom("Missing workload graph")
            })?;
        let graph: WorkloadGraph = bson::from_bson(value)?;
        validations::validate_graph_type(&graph, workload_type)
            .map_err(validation_error)?;
        Ok(Some(graph))
    }

    /// Replace graph content without changing its ownership.
    pub async fn update_workload_graph(
        &self,
        postgres: &PostgresOrm,
        workload: &Workload,
    ) -> Result<()> {
        validations::validate_workload(postgres, workload)
            .await
            .map_err(validation_error)?;
        let result = self
            .db
            .collection::<Document>(GRAPHS)
            .replace_one(
                doc! {
                    "workload_graph_id":
                        &workload.workload_graph_id,
                    "workload_id": &workload.workload_id,
                },
                graph_document(workload)?,
            )
            .await?;
        validations::validate_affected_rows(result.matched_count)
            .map_err(validation_error)
    }

    /// Delete only the graph with the matching workload owner.
    pub async fn delete_workload_graph(
        &self,
        graph_id: &str,
        workload_id: &str,
    ) -> Result<bool> {
        let result = self
            .db
            .collection::<Document>(GRAPHS)
            .delete_one(doc! {
                "workload_graph_id": graph_id,
                "workload_id": workload_id,
            })
            .await?;
        Ok(result.deleted_count != 0)
    }

    /// Insert the supplied record without changing its identity.
    pub async fn create_workload_log(
        &self,
        postgres: &PostgresOrm,
        record: &WorkloadLogs,
    ) -> Result<()> {
        self.create_workload_logs(
            postgres,
            std::slice::from_ref(record),
        )
        .await
    }

    /// Insert supplied records; database write errors propagate.
    /// Mongo insert-many may write a prefix before an error.
    pub async fn create_workload_logs(
        &self,
        postgres: &PostgresOrm,
        records: &[WorkloadLogs],
    ) -> Result<()> {
        validations::validate_logs(postgres, self, records)
            .await
            .map_err(validation_error)?;
        self.insert_records(
            LOGS,
            records,
            &["created_at", "log_timestamp"],
        )
        .await
    }

    /// Fetch one stored record by ID.
    pub async fn get_workload_log(
        &self,
        id: &str,
    ) -> Result<Option<WorkloadLogs>> {
        let record = self
            .db
            .collection::<Document>(LOGS)
            .find_one(doc! { "log_id": id })
            .await?;
        record
            .map(|document| {
                decode_record(
                    document,
                    &["created_at", "log_timestamp"],
                )
            })
            .transpose()
    }

    /// Delete one record by ID.
    pub async fn delete_workload_log(
        &self,
        id: &str,
    ) -> Result<bool> {
        let result = self
            .db
            .collection::<Document>(LOGS)
            .delete_one(doc! { "log_id": id })
            .await?;
        Ok(result.deleted_count != 0)
    }

    /// Delete this collection's records for a workload.
    pub async fn delete_workload_logs_by_workload(
        &self,
        workload_id: &str,
    ) -> Result<u64> {
        let result = self
            .db
            .collection::<Document>(LOGS)
            .delete_many(doc! { "workload_id": workload_id })
            .await?;
        Ok(result.deleted_count)
    }

    /// Insert the supplied record without changing its identity.
    pub async fn create_workload_user_metrics(
        &self,
        postgres: &PostgresOrm,
        record: &WorkloadUserMetrics,
    ) -> Result<()> {
        self.create_workload_user_metrics_batch(
            postgres,
            std::slice::from_ref(record),
        )
        .await
    }

    /// Insert supplied records; database write errors propagate.
    /// Mongo insert-many may write a prefix before an error.
    pub async fn create_workload_user_metrics_batch(
        &self,
        postgres: &PostgresOrm,
        records: &[WorkloadUserMetrics],
    ) -> Result<()> {
        validations::validate_metrics(postgres, self, records)
            .await
            .map_err(validation_error)?;
        self.insert_records(METRICS, records, &["created_at"])
            .await
    }

    /// Fetch one stored record by ID.
    pub async fn get_workload_user_metrics(
        &self,
        id: &str,
    ) -> Result<Option<WorkloadUserMetrics>> {
        let record = self
            .db
            .collection::<Document>(METRICS)
            .find_one(doc! { "metrics_entry_id": id })
            .await?;
        record
            .map(|document| {
                decode_record(document, &["created_at"])
            })
            .transpose()
    }

    /// Delete one record by ID.
    pub async fn delete_workload_user_metrics(
        &self,
        id: &str,
    ) -> Result<bool> {
        let result = self
            .db
            .collection::<Document>(METRICS)
            .delete_one(doc! { "metrics_entry_id": id })
            .await?;
        Ok(result.deleted_count != 0)
    }

    /// Delete this collection's records for a workload.
    pub async fn delete_workload_user_metrics_by_workload(
        &self,
        workload_id: &str,
    ) -> Result<u64> {
        let result = self
            .db
            .collection::<Document>(METRICS)
            .delete_many(doc! { "workload_id": workload_id })
            .await?;
        Ok(result.deleted_count)
    }

    /// Insert the supplied record without changing its identity.
    pub async fn create_simulation_model_trajectory_state(
        &self,
        postgres: &PostgresOrm,
        record: &WorkloadSimulationModelTrajectoryState,
    ) -> Result<()> {
        self.create_simulation_model_trajectory_states(
            postgres,
            std::slice::from_ref(record),
        )
        .await
    }

    /// Insert supplied records; database write errors propagate.
    /// Mongo insert-many may write a prefix before an error.
    pub async fn create_simulation_model_trajectory_states(
        &self,
        postgres: &PostgresOrm,
        records: &[WorkloadSimulationModelTrajectoryState],
    ) -> Result<()> {
        validations::validate_model_states(
            postgres, self, records,
        )
        .await
        .map_err(validation_error)?;
        self.insert_records(STATES, records, &["created_at"])
            .await
    }

    /// Fetch one stored record by ID.
    pub async fn get_simulation_model_trajectory_state(
        &self,
        id: &str,
    ) -> Result<Option<WorkloadSimulationModelTrajectoryState>>
    {
        let record = self
            .db
            .collection::<Document>(STATES)
            .find_one(doc! { "sim_model_state_entry_id": id })
            .await?;
        record
            .map(|document| {
                decode_record(document, &["created_at"])
            })
            .transpose()
    }

    /// Delete one record by ID.
    pub async fn delete_simulation_model_trajectory_state(
        &self,
        id: &str,
    ) -> Result<bool> {
        let result = self
            .db
            .collection::<Document>(STATES)
            .delete_one(doc! { "sim_model_state_entry_id": id })
            .await?;
        Ok(result.deleted_count != 0)
    }

    /// Delete this collection's records for a workload.
    pub async fn delete_model_states_by_workload(
        &self,
        workload_id: &str,
    ) -> Result<u64> {
        let result = self
            .db
            .collection::<Document>(STATES)
            .delete_many(doc! { "workload_id": workload_id })
            .await?;
        Ok(result.deleted_count)
    }

    /// List logs by timestamp/ID, using (limit, offset).
    pub async fn list_workload_logs(
        &self,
        workload_id: &str,
        pagination: (i64, i64),
    ) -> Result<Vec<WorkloadLogs>> {
        self.find_records(
            LOGS,
            doc! { "workload_id": workload_id },
            doc! { "log_timestamp": 1, "log_id": 1 },
            pagination,
            &["created_at", "log_timestamp"],
        )
        .await
    }

    /// List metric batches, optionally for one metric ID.
    pub async fn list_workload_user_metrics(
        &self,
        workload_id: &str,
        metric_id: Option<i32>,
        pagination: (i64, i64),
    ) -> Result<Vec<WorkloadUserMetrics>> {
        let mut filter = doc! { "workload_id": workload_id };
        if let Some(id) = metric_id {
            filter.insert("metric_id", id);
        }
        self.find_records(
            METRICS,
            filter,
            doc! { "created_at": 1, "metrics_entry_id": 1 },
            pagination,
            &["created_at"],
        )
        .await
    }

    /// List poses by timestamp and ID, with optional model/state.
    /// Coordinates are matched as a complete array.
    pub async fn list_simulation_model_trajectory_states(
        &self,
        workload_id: &str,
        simulation_model_id: Option<&str>,
        coordinate: Option<&[u32]>,
        pagination: (i64, i64),
    ) -> Result<Vec<WorkloadSimulationModelTrajectoryState>> {
        let mut filter = doc! { "workload_id": workload_id };
        if let Some(id) = simulation_model_id {
            filter.insert("simulation_model_id", id);
        }
        if let Some(coordinate) = coordinate {
            filter.insert(
                "workload_structure_state",
                bson::to_bson(coordinate)?,
            );
        }
        self.find_records(
            STATES,
            filter,
            doc! {
                "created_at": 1,
                "sim_model_state_entry_id": 1,
            },
            pagination,
            &["created_at"],
        )
        .await
    }

    /// Graph references do not have SQL foreign keys.
    pub async fn simulation_model_is_referenced(
        &self,
        id: &str,
    ) -> Result<bool> {
        let count = self.db.collection::<Document>(GRAPHS)
            .count_documents(doc! { "$or": [
                { "graph.TrainingGraph.training_job_controller.training_method.content.simulation_model_id": id },
                { "graph.TestGraph.test_job_controller.test_method.content.simulation_model_id": id },
            ] }).await?;
        Ok(count != 0)
    }

    /// Order metric batches by metric and their first coordinate.
    pub async fn list_history_metrics(
        &self,
        workload_id: &str,
        dimensions: u32,
        pagination: (i64, i64),
    ) -> Result<Vec<WorkloadUserMetrics>> {
        let mut sort = doc! { "metric_id": 1 };
        for index in 0..dimensions {
            sort.insert(
                format!(
                    "workload_structure_state_values.0.{index}"
                ),
                1,
            );
        }
        sort.insert("metrics_entry_id", 1);
        self.find_records(
            METRICS,
            doc! { "workload_id": workload_id },
            sort,
            pagination,
            &["created_at"],
        )
        .await
    }

    /// Array elements are separate sort keys; Mongo's default
    /// array sort would otherwise use the minimum array element.
    pub async fn list_history_states(
        &self,
        workload_id: &str,
        dimensions: u32,
        pagination: (i64, i64),
    ) -> Result<Vec<WorkloadSimulationModelTrajectoryState>> {
        let mut sort = doc! { "workload_structure_id": 1 };
        for index in 0..dimensions {
            sort.insert(
                format!("workload_structure_state.{index}"),
                1,
            );
        }
        sort.insert("simulation_model_id", 1);
        sort.insert("sim_model_state_entry_id", 1);
        self.find_records(
            STATES,
            doc! { "workload_id": workload_id },
            sort,
            pagination,
            &["created_at"],
        )
        .await
    }

    async fn insert_records<T: Serialize>(
        &self,
        collection: &str,
        records: &[T],
        dates: &[&str],
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        // Serialize every record before starting any writes.
        let documents = records
            .iter()
            .map(|record| encode_record(record, dates))
            .collect::<Result<Vec<_>>>()?;
        self.db
            .collection::<Document>(collection)
            .insert_many(documents)
            .ordered(true)
            .await?;
        Ok(())
    }

    async fn find_records<T: DeserializeOwned>(
        &self,
        collection: &str,
        filter: Document,
        sort: Document,
        pagination: (i64, i64),
        dates: &[&str],
    ) -> Result<Vec<T>> {
        let (limit, offset) = pagination;
        validations::validate_pagination(pagination)
            .map_err(validation_error)?;
        let mut cursor = self
            .db
            .collection::<Document>(collection)
            .find(filter)
            .allow_disk_use(true)
            .sort(sort)
            .limit(limit)
            .skip(offset as u64)
            .await?;
        let mut records = Vec::new();
        while let Some(document) = cursor.try_next().await? {
            records.push(decode_record(document, dates)?);
        }
        Ok(records)
    }
}

fn graph_document(workload: &Workload) -> Result<Document> {
    Ok(doc! {
        "workload_graph_id": &workload.workload_graph_id,
        "graph": bson::to_bson(&workload.workload_graph)?,
        "workload_type": bson::to_bson(&workload.workload_type)?,
        "workload_id": &workload.workload_id,
        "created_at": bson::DateTime::from_millis(
            workload.created_at.timestamp_millis(),
        ),
    })
}

fn validation_error(error: sqlx::Error) -> Error {
    Error::custom(error.to_string())
}

fn encode_record<T: Serialize>(
    record: &T,
    dates: &[&str],
) -> Result<Document> {
    let mut document = bson::to_document(record)?;
    for &name in dates {
        let text = document
            .get_str(name)
            .map_err(|error| Error::custom(error.to_string()))?;
        let timestamp = DateTime::parse_from_rfc3339(text)
            .map_err(|error| Error::custom(error.to_string()))?;
        document.insert(
            name,
            bson::DateTime::from_millis(
                timestamp.timestamp_millis(),
            ),
        );
    }
    Ok(document)
}

fn decode_record<T: DeserializeOwned>(
    mut document: Document,
    dates: &[&str],
) -> Result<T> {
    document.remove("_id");
    for &name in dates {
        let stored = document
            .get_datetime(name)
            .map_err(|error| Error::custom(error.to_string()))?;
        let timestamp = DateTime::<Utc>::from_timestamp_millis(
            stored.timestamp_millis(),
        )
        .ok_or_else(|| {
            Error::custom("BSON date is out of range")
        })?;
        document.insert(
            name,
            Bson::String(
                timestamp
                    .to_rfc3339_opts(SecondsFormat::Millis, true),
            ),
        );
    }
    Ok(bson::from_document(document)?)
}
