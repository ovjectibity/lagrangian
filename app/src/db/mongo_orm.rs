//! MongoDB ORM for TrainingGraphRecord,
//! LogStream, and MetricsStream.

use mongodb::bson::doc;
use mongodb::options::ClientOptions;
use mongodb::{Client, Collection, Database};

use crate::commondef::experiments::{
    LogStream, MetricsStream,
    TrainingGraphRecord,
};

const DB_NAME: &str = "lagrangian";
const TRAINING_GRAPHS_COL: &str =
    "training_graphs";
const LOG_STREAMS_COL: &str = "log_streams";
const METRICS_STREAMS_COL: &str =
    "metrics_streams";

/// Holds a MongoDB client and provides
/// CRUD operations for document-oriented
/// data.
pub struct MongoOrm {
    db: Database,
}

impl MongoOrm {
    /// Connect to MongoDB and return a
    /// new MongoOrm instance.
    pub async fn new(
        connection_uri: &str,
    ) -> Result<Self, mongodb::error::Error>
    {
        let opts =
            ClientOptions::parse(
                connection_uri,
            )
            .await?;
        let client =
            Client::with_options(opts)?;
        let db = client.database(DB_NAME);
        Ok(Self { db })
    }

    fn training_graphs_col(
        &self,
    ) -> Collection<TrainingGraphRecord> {
        self.db
            .collection(TRAINING_GRAPHS_COL)
    }

    fn log_streams_col(
        &self,
    ) -> Collection<LogStream> {
        self.db.collection(LOG_STREAMS_COL)
    }

    fn metrics_streams_col(
        &self,
    ) -> Collection<MetricsStream> {
        self.db
            .collection(METRICS_STREAMS_COL)
    }

    // ── TrainingGraphRecord CRUD ──

    /// Insert a training graph record.
    pub async fn create_training_graph(
        &self,
        record: &TrainingGraphRecord,
    ) -> Result<(), mongodb::error::Error>
    {
        self.training_graphs_col()
            .insert_one(record)
            .await?;
        Ok(())
    }

    /// Fetch a training graph record by id.
    pub async fn get_training_graph(
        &self,
        id: &str,
    ) -> Result<
        Option<TrainingGraphRecord>,
        mongodb::error::Error,
    > {
        self.training_graphs_col()
            .find_one(doc! { "id": id })
            .await
    }

    /// Fetch all graph records for a run.
    pub async fn get_training_graphs_by_run(
        &self,
        training_experiment_id: &str,
        training_run_id: &str,
    ) -> Result<
        Vec<TrainingGraphRecord>,
        mongodb::error::Error,
    > {
        use futures_util::TryStreamExt;
        self.training_graphs_col()
            .find(doc! {
                "training_experiment_id":
                    training_experiment_id,
                "training_run_id":
                    training_run_id,
            })
            .await?
            .try_collect()
            .await
    }

    /// Replace a training graph record.
    pub async fn update_training_graph(
        &self,
        record: &TrainingGraphRecord,
    ) -> Result<(), mongodb::error::Error>
    {
        self.training_graphs_col()
            .replace_one(
                doc! { "id": &record.id },
                record,
            )
            .await?;
        Ok(())
    }

    /// Delete a training graph record.
    pub async fn delete_training_graph(
        &self,
        id: &str,
    ) -> Result<(), mongodb::error::Error>
    {
        self.training_graphs_col()
            .delete_one(doc! { "id": id })
            .await?;
        Ok(())
    }

    // ── LogStream CRUD ──

    /// Insert a log stream.
    pub async fn create_log_stream(
        &self,
        log_stream: &LogStream,
    ) -> Result<(), mongodb::error::Error>
    {
        self.log_streams_col()
            .insert_one(log_stream)
            .await?;
        Ok(())
    }

    /// Fetch a log stream by id.
    pub async fn get_log_stream(
        &self,
        id: &str,
    ) -> Result<
        Option<LogStream>,
        mongodb::error::Error,
    > {
        self.log_streams_col()
            .find_one(doc! { "id": id })
            .await
    }

    /// Fetch all log streams for a run.
    pub async fn get_log_streams_by_run(
        &self,
        training_experiment_id: &str,
        training_run_id: &str,
    ) -> Result<
        Vec<LogStream>,
        mongodb::error::Error,
    > {
        use futures_util::TryStreamExt;
        self.log_streams_col()
            .find(doc! {
                "training_experiment_id":
                    training_experiment_id,
                "training_run_id":
                    training_run_id,
            })
            .await?
            .try_collect()
            .await
    }

    /// Append log entries to a stream.
    pub async fn append_logs(
        &self,
        id: &str,
        new_logs: &[String],
    ) -> Result<(), mongodb::error::Error>
    {
        self.log_streams_col()
            .update_one(
                doc! { "id": id },
                doc! {
                    "$push": {
                        "logs": {
                            "$each": new_logs
                        }
                    }
                },
            )
            .await?;
        Ok(())
    }

    /// Delete a log stream.
    pub async fn delete_log_stream(
        &self,
        id: &str,
    ) -> Result<(), mongodb::error::Error>
    {
        self.log_streams_col()
            .delete_one(doc! { "id": id })
            .await?;
        Ok(())
    }

    // ── MetricsStream CRUD ──

    /// Insert a metrics stream.
    pub async fn create_metrics_stream(
        &self,
        stream: &MetricsStream,
    ) -> Result<(), mongodb::error::Error>
    {
        self.metrics_streams_col()
            .insert_one(stream)
            .await?;
        Ok(())
    }

    /// Fetch a metrics stream by id.
    pub async fn get_metrics_stream(
        &self,
        id: &str,
    ) -> Result<
        Option<MetricsStream>,
        mongodb::error::Error,
    > {
        self.metrics_streams_col()
            .find_one(doc! { "id": id })
            .await
    }

    /// Fetch all metrics streams for a run.
    pub async fn get_metrics_streams_by_run(
        &self,
        training_experiment_id: &str,
        training_run_id: &str,
    ) -> Result<
        Vec<MetricsStream>,
        mongodb::error::Error,
    > {
        use futures_util::TryStreamExt;
        self.metrics_streams_col()
            .find(doc! {
                "training_experiment_id":
                    training_experiment_id,
                "training_run_id":
                    training_run_id,
            })
            .await?
            .try_collect()
            .await
    }

    /// Append metrics to an existing stream.
    pub async fn append_metrics(
        &self,
        id: &str,
        new_metrics: &[
            crate::commondef::training
            ::TrainingMetrics
        ],
    ) -> Result<(), mongodb::error::Error>
    {
        self.metrics_streams_col()
            .update_one(
                doc! { "id": id },
                doc! {
                    "$push": {
                        "training_metrics": {
                            "$each": new_metrics
                        }
                    }
                },
            )
            .await?;
        Ok(())
    }

    /// Delete a metrics stream.
    pub async fn delete_metrics_stream(
        &self,
        id: &str,
    ) -> Result<(), mongodb::error::Error>
    {
        self.metrics_streams_col()
            .delete_one(doc! { "id": id })
            .await?;
        Ok(())
    }
}
