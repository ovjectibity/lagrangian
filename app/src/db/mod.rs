//! Persistence for relational records and workload documents.
//! Asset references are accepted as supplied. Storage operations
//! such as S3 copies, moves and deletion belong to the caller.

use std::error::Error;
use std::io;

use sqlx::Row;
use sqlx::postgres::PgRow;

use crate::commondef::records::workload::{
    Workload, WorkloadSimulationModelTrajectoryState,
    WorkloadStatus, WorkloadType, WorkloadUserMetrics,
};

use self::db_orm::PostgresOrm;
use self::mongo_orm::MongoOrm;

pub mod db_orm;
pub mod mongo_orm;
pub mod validations;

/// Create a graph and its SQL metadata with the supplied IDs.
/// A failed SQL insert removes the newly inserted graph. These
/// databases do not share an atomic transaction.
pub async fn create_workload(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    record: &Workload,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    mongo.create_workload_graph(postgres, record).await?;
    if let Err(error) =
        postgres.create_workload_metadata(record).await
    {
        if let Err(cleanup) = mongo
            .delete_workload_graph(
                &record.workload_graph_id,
                &record.workload_id,
            )
            .await
        {
            return Err(io::Error::other(format!(
                "Workload insert failed: {error}; \
                 graph cleanup also failed: {cleanup}"
            ))
            .into());
        }
        return Err(error.into());
    }
    Ok(())
}

/// Return the complete workload; a missing graph is an error.
/// None means that the SQL workload itself does not exist.
pub async fn get_workload(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    id: &str,
) -> Result<Option<Workload>, Box<dyn Error + Send + Sync>> {
    let Some(row) = postgres.workload_row(id).await? else {
        return Ok(None);
    };
    Ok(Some(assemble_workload(&row, mongo).await?))
}

/// List complete workloads; pagination is (limit, offset).
pub async fn list_workloads(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    project_id: Option<&str>,
    user_id: Option<&str>,
    workload_type: Option<&WorkloadType>,
    status: Option<&WorkloadStatus>,
    pagination: (i64, i64),
) -> Result<Vec<Workload>, Box<dyn Error + Send + Sync>> {
    let rows = postgres
        .workload_rows(
            project_id,
            user_id,
            workload_type,
            status,
            pagination,
        )
        .await?;
    let mut records = Vec::with_capacity(rows.len());
    for row in rows {
        records.push(assemble_workload(&row, mongo).await?);
    }
    Ok(records)
}

/// Update graph content and metadata under their existing IDs.
/// If the SQL update fails, restore the previous graph content.
/// Concurrent aggregate edits must be serialized by the caller.
pub async fn update_workload(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    record: &Workload,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let previous =
        get_workload(postgres, mongo, &record.workload_id)
            .await?
            .ok_or(sqlx::Error::RowNotFound)?;
    validations::validate_graph_id_unchanged(&previous, record)?;
    mongo.update_workload_graph(postgres, record).await?;
    if let Err(error) =
        postgres.update_workload_metadata(record).await
    {
        if let Err(restore) =
            mongo.update_workload_graph(postgres, &previous).await
        {
            return Err(io::Error::other(format!(
                "Workload update failed: {error}; \
                 graph restoration also failed: {restore}"
            ))
            .into());
        }
        return Err(error.into());
    }
    Ok(())
}

/// Validate metric coordinates against the immutable SQL LUTs,
/// then insert the supplied Mongo records unchanged.
pub async fn create_workload_metrics(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    records: &[WorkloadUserMetrics],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    mongo
        .create_workload_user_metrics_batch(postgres, records)
        .await?;
    Ok(())
}

/// Validate pose coordinates and relational references, then
/// insert the supplied Mongo records unchanged.
pub async fn create_simulation_model_trajectory_states(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    records: &[WorkloadSimulationModelTrajectoryState],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    mongo
        .create_simulation_model_trajectory_states(
            postgres, records,
        )
        .await?;
    Ok(())
}

async fn assemble_workload(
    row: &PgRow,
    mongo: &MongoOrm,
) -> Result<Workload, Box<dyn Error + Send + Sync>> {
    let graph = mongo
        .get_workload_graph(
            row.try_get("workload_graph_id")?,
            row.try_get("workload_id")?,
            &row.try_get("workload_type")?,
        )
        .await?;
    let graph = validations::validate_graph_exists(graph)?;
    Ok(db_orm::workload_from_row(row, graph)?)
}
