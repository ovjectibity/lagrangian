//! MongoDB ORM for TrainingGraphRecord
//! and LogStream.

use mongodb::bson::doc;
use mongodb::options::ClientOptions;
use mongodb::{Client, Collection, Database};

use crate::commondef::experiments::{
    LogStream, TrainingGraphRecord,
};

const DB_NAME: &str = "lagrangian";
const TRAINING_GRAPHS_COL: &str =
    "training_graphs";
const LOG_STREAMS_COL: &str = "log_streams";

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
            ClientOptions::parse(connection_uri)
                .await?;
        let client = Client::with_options(opts)?;
        let db = client.database(DB_NAME);
        Ok(Self { db })
    }

    fn training_graphs_col(
        &self,
    ) -> Collection<TrainingGraphRecord> {
        self.db.collection(TRAINING_GRAPHS_COL)
    }

    fn log_streams_col(
        &self,
    ) -> Collection<LogStream> {
        self.db.collection(LOG_STREAMS_COL)
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

    /// Update a training graph record.
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

    /// Append logs to an existing stream.
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
}
