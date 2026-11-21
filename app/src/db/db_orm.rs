//! Postgres persistence for the commondef records.
//! Asset references are stored as supplied.
//! File storage is managed by the caller.

use chrono::{DateTime, Utc};
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::types::Json;
use sqlx::{Executor, FromRow, PgPool, Postgres, Row};

use crate::commondef::records::assets::{
    CheckpointModelType, CheckpointRef, SimulationModelRef,
    TrajectoryCameraFrameRef,
};
use crate::commondef::records::product::{
    AccessKey, AccessKeyStatus, Group, Project, User,
};
use crate::commondef::records::workload::{
    UserMetricsLut, Workload, WorkloadStatus,
    WorkloadStructureLut, WorkloadType,
};
use crate::commondef::records::workload_graph::{
    MLModel, WorkloadGraph,
};

use super::mongo_orm::MongoOrm;
use super::validations;

/// Connection pool for relational records and read-only LUTs.
pub struct PostgresOrm {
    pool: PgPool,
}

impl PostgresOrm {
    /// Connect using the application's Postgres URL.
    pub async fn new(url: &str) -> sqlx::Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await?;
        Ok(Self { pool })
    }

    /// Access the pool for caller-managed transactions.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Store every supplied group field.
    pub async fn create_group(
        &self,
        record: &Group,
    ) -> sqlx::Result<Group> {
        validations::validate_modified_at(
            record.created_at,
            record.updated_at,
        )?;
        sqlx::query_as::<_, Group>(
            "INSERT INTO product.\"group\" (group_id, name,
             created_at, updated_at, login_type) VALUES ($1,
             $2, $3, $4, $5) RETURNING group_id, name,
             created_at, updated_at, login_type",
        )
        .bind(&record.group_id)
        .bind(&record.name)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.login_type)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a group by its identifier.
    pub async fn get_group(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<Group>> {
        sqlx::query_as::<_, Group>(
            "SELECT group_id, name, created_at, updated_at,
             login_type FROM product.\"group\" WHERE group_id
             = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Replace supplied fields, retaining the identifier.
    pub async fn update_group(
        &self,
        record: &Group,
    ) -> sqlx::Result<Group> {
        validations::validate_modified_at(
            record.created_at,
            record.updated_at,
        )?;
        sqlx::query_as::<_, Group>(
            "UPDATE product.\"group\" SET name = $2,
             created_at = $3, updated_at = $4, login_type =
             $5 WHERE group_id = $1 RETURNING group_id, name,
             created_at, updated_at, login_type",
        )
        .bind(&record.group_id)
        .bind(&record.name)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.login_type)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete one row; existing foreign keys remain enforced.
    pub async fn delete_group(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM product.\"group\" WHERE group_id =
             $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Store every supplied user field.
    pub async fn create_user(
        &self,
        record: &User,
    ) -> sqlx::Result<User> {
        validations::validate_modified_at(
            record.created_at,
            record.updated_at,
        )?;
        sqlx::query_as::<_, User>(
            "INSERT INTO product.\"user\" (user_id, group_id,
             created_at, updated_at, name, email, role)
             VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING
             user_id, group_id, created_at, updated_at, name,
             email, role",
        )
        .bind(&record.user_id)
        .bind(&record.group_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.name)
        .bind(&record.email)
        .bind(&record.role)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a user by its identifier.
    pub async fn get_user(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<User>> {
        sqlx::query_as::<_, User>(
            "SELECT user_id, group_id, created_at,
             updated_at, name, email, role FROM
             product.\"user\" WHERE user_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Replace supplied fields, retaining the identifier.
    pub async fn update_user(
        &self,
        record: &User,
    ) -> sqlx::Result<User> {
        validations::validate_modified_at(
            record.created_at,
            record.updated_at,
        )?;
        sqlx::query_as::<_, User>(
            "UPDATE product.\"user\" SET group_id = $2,
             created_at = $3, updated_at = $4, name = $5,
             email = $6, role = $7 WHERE user_id = $1
             RETURNING user_id, group_id, created_at,
             updated_at, name, email, role",
        )
        .bind(&record.user_id)
        .bind(&record.group_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.name)
        .bind(&record.email)
        .bind(&record.role)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete one row; existing foreign keys remain enforced.
    pub async fn delete_user(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM product.\"user\" WHERE user_id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Store every supplied access key field.
    pub async fn create_access_key(
        &self,
        record: &AccessKey,
    ) -> sqlx::Result<AccessKey> {
        sqlx::query_as::<_, AccessKey>(
            "INSERT INTO product.access_key (access_key_id,
             access_key_value, user_id, name, created_at,
             status) VALUES ($1, $2, $3, $4, $5, $6)
             RETURNING access_key_id, access_key_value,
             user_id, name, created_at, status",
        )
        .bind(&record.access_key_id)
        .bind(&record.access_key_value)
        .bind(&record.user_id)
        .bind(&record.name)
        .bind(record.created_at)
        .bind(&record.status)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a access key by its identifier.
    pub async fn get_access_key(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<AccessKey>> {
        sqlx::query_as::<_, AccessKey>(
            "SELECT access_key_id, access_key_value, user_id,
             name, created_at, status FROM product.access_key
             WHERE access_key_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Replace supplied fields, retaining the identifier.
    pub async fn update_access_key(
        &self,
        record: &AccessKey,
    ) -> sqlx::Result<AccessKey> {
        sqlx::query_as::<_, AccessKey>(
            "UPDATE product.access_key SET access_key_value =
             $2, user_id = $3, name = $4, created_at = $5,
             status = $6 WHERE access_key_id = $1 RETURNING
             access_key_id, access_key_value, user_id, name,
             created_at, status",
        )
        .bind(&record.access_key_id)
        .bind(&record.access_key_value)
        .bind(&record.user_id)
        .bind(&record.name)
        .bind(record.created_at)
        .bind(&record.status)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete one row; existing foreign keys remain enforced.
    pub async fn delete_access_key(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM product.access_key WHERE
             access_key_id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Store every supplied project field.
    pub async fn create_project(
        &self,
        record: &Project,
    ) -> sqlx::Result<Project> {
        validations::validate_modified_at(
            record.created_at,
            record.updated_at,
        )?;
        sqlx::query_as::<_, Project>(
            "INSERT INTO product.project (project_id,
             user_id, created_at, updated_at, name) VALUES
             ($1, $2, $3, $4, $5) RETURNING project_id,
             user_id, created_at, updated_at, name",
        )
        .bind(&record.project_id)
        .bind(&record.user_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.name)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a project by its identifier.
    pub async fn get_project(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<Project>> {
        sqlx::query_as::<_, Project>(
            "SELECT project_id, user_id, created_at,
             updated_at, name FROM product.project WHERE
             project_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Replace supplied fields, retaining the identifier.
    pub async fn update_project(
        &self,
        record: &Project,
    ) -> sqlx::Result<Project> {
        validations::validate_modified_at(
            record.created_at,
            record.updated_at,
        )?;
        sqlx::query_as::<_, Project>(
            "UPDATE product.project SET user_id = $2,
             created_at = $3, updated_at = $4, name = $5
             WHERE project_id = $1 RETURNING project_id,
             user_id, created_at, updated_at, name",
        )
        .bind(&record.project_id)
        .bind(&record.user_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.name)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete one row; existing foreign keys remain enforced.
    pub async fn delete_project(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM product.project WHERE project_id =
             $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Store every supplied simulation model ref field.
    pub async fn create_simulation_model_ref(
        &self,
        record: &SimulationModelRef,
    ) -> sqlx::Result<SimulationModelRef> {
        insert_simulation_model_ref(&self.pool, record).await
    }

    /// Fetch a simulation model ref by its identifier.
    pub async fn get_simulation_model_ref(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<SimulationModelRef>> {
        sqlx::query_as::<_, SimulationModelRef>(
            "SELECT project_id, simulation_model_id, created_at,
             updated_at, name, uri_path, model_type,
             file_type, base_file_name FROM
             assets.simulation_model_ref WHERE
             simulation_model_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Replace supplied fields, retaining the identifier.
    pub async fn update_simulation_model_ref(
        &self,
        record: &SimulationModelRef,
    ) -> sqlx::Result<SimulationModelRef> {
        validations::validate_modified_at(
            record.created_at,
            record.updated_at,
        )?;
        sqlx::query_as::<_, SimulationModelRef>(
            "UPDATE assets.simulation_model_ref SET
             created_at = $2, updated_at = $3, name = $4,
             uri_path = $5, model_type = $6, file_type = $7,
             base_file_name = $8, project_id = $9 WHERE simulation_model_id =
             $1 RETURNING project_id, simulation_model_id, created_at,
             updated_at, name, uri_path, model_type,
             file_type, base_file_name",
        )
        .bind(&record.simulation_model_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.name)
        .bind(&record.uri_path)
        .bind(&record.model_type)
        .bind(&record.file_type)
        .bind(&record.base_file_name)
        .bind(&record.project_id)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete one row; existing foreign keys remain enforced.
    pub async fn delete_simulation_model_ref(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM assets.simulation_model_ref WHERE
             simulation_model_id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Insert a batch atomically without changing references.
    pub async fn create_simulation_model_refs(
        &self,
        records: &[SimulationModelRef],
    ) -> sqlx::Result<()> {
        let mut transaction = self.pool.begin().await?;
        for record in records {
            insert_simulation_model_ref(
                &mut *transaction,
                record,
            )
            .await?;
        }
        transaction.commit().await
    }

    /// Store every supplied checkpoint ref field.
    pub async fn create_checkpoint_ref(
        &self,
        mongo: &MongoOrm,
        record: &CheckpointRef,
    ) -> sqlx::Result<CheckpointRef> {
        validations::validate_checkpoint_ref(self, mongo, record)
            .await?;
        insert_checkpoint_ref(&self.pool, record).await
    }

    /// Fetch a checkpoint ref by its identifier.
    pub async fn get_checkpoint_ref(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<CheckpointRef>> {
        sqlx::query_as::<_, CheckpointRef>(
            "SELECT checkpoint_id, created_at, updated_at,
             name, uri_path, workload_id,
             workload_structure_id, workload_structure_state,
             na_indices, model_type, model_spec FROM
             assets.checkpoint_ref WHERE checkpoint_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Replace supplied fields, retaining the identifier.
    pub async fn update_checkpoint_ref(
        &self,
        mongo: &MongoOrm,
        record: &CheckpointRef,
    ) -> sqlx::Result<CheckpointRef> {
        validations::validate_checkpoint_ref(self, mongo, record)
            .await?;
        sqlx::query_as::<_, CheckpointRef>(
            "UPDATE assets.checkpoint_ref SET created_at =
             $2, updated_at = $3, name = $4, uri_path = $5,
             workload_id = $6, workload_structure_id = $7,
             workload_structure_state = $8, na_indices = $9,
             model_type = $10, model_spec = $11 WHERE
             checkpoint_id = $1 RETURNING checkpoint_id,
             created_at, updated_at, name, uri_path,
             workload_id, workload_structure_id,
             workload_structure_state, na_indices,
             model_type, model_spec",
        )
        .bind(&record.checkpoint_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(&record.name)
        .bind(&record.uri_path)
        .bind(&record.workload_id)
        .bind(record.workload_structure_id)
        .bind(
            record
                .workload_structure_state
                .as_deref()
                .map(encode_coordinates),
        )
        .bind(encode_coordinates(&record.na_indices))
        .bind(&record.model_type)
        .bind(Json(&record.model_spec))
        .fetch_one(&self.pool)
        .await
    }

    /// Delete one row; existing foreign keys remain enforced.
    pub async fn delete_checkpoint_ref(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM assets.checkpoint_ref WHERE
             checkpoint_id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Insert a batch atomically without changing references.
    pub async fn create_checkpoint_refs(
        &self,
        mongo: &MongoOrm,
        records: &[CheckpointRef],
    ) -> sqlx::Result<()> {
        for record in records {
            validations::validate_checkpoint_ref(
                self, mongo, record,
            )
            .await?;
        }
        let mut transaction = self.pool.begin().await?;
        for record in records {
            insert_checkpoint_ref(&mut *transaction, record)
                .await?;
        }
        transaction.commit().await
    }

    /// Store every supplied trajectory camera frame ref field.
    pub async fn create_trajectory_camera_frame_ref(
        &self,
        mongo: &MongoOrm,
        record: &TrajectoryCameraFrameRef,
    ) -> sqlx::Result<TrajectoryCameraFrameRef> {
        validations::validate_camera_frame(self, mongo, record)
            .await?;
        insert_trajectory_camera_frame_ref(&self.pool, record)
            .await
    }

    /// Fetch a trajectory camera frame ref by its identifier.
    pub async fn get_trajectory_camera_frame_ref(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<TrajectoryCameraFrameRef>> {
        sqlx::query_as::<_, TrajectoryCameraFrameRef>(
            "SELECT camera_frame_id, workload_id,
             workload_structure_id, workload_structure_state,
             camera_id, model_camera_name, uri_path,
             simulation_model_id FROM
             assets.trajectory_camera_frame_ref WHERE
             camera_frame_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Replace supplied fields, retaining the identifier.
    pub async fn update_trajectory_camera_frame_ref(
        &self,
        mongo: &MongoOrm,
        record: &TrajectoryCameraFrameRef,
    ) -> sqlx::Result<TrajectoryCameraFrameRef> {
        validations::validate_camera_frame(self, mongo, record)
            .await?;
        sqlx::query_as::<_, TrajectoryCameraFrameRef>(
            "UPDATE assets.trajectory_camera_frame_ref SET
             workload_id = $2, workload_structure_id = $3,
             workload_structure_state = $4, camera_id = $5,
             model_camera_name = $6, uri_path = $7,
             simulation_model_id = $8 WHERE camera_frame_id =
             $1 RETURNING camera_frame_id, workload_id,
             workload_structure_id, workload_structure_state,
             camera_id, model_camera_name, uri_path,
             simulation_model_id",
        )
        .bind(&record.camera_frame_id)
        .bind(&record.workload_id)
        .bind(record.workload_structure_id)
        .bind(encode_coordinates(
            &record.workload_structure_state,
        ))
        .bind(&record.camera_id)
        .bind(&record.model_camera_name)
        .bind(&record.uri_path)
        .bind(&record.simulation_model_id)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete one row; existing foreign keys remain enforced.
    pub async fn delete_trajectory_camera_frame_ref(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM assets.trajectory_camera_frame_ref
             WHERE camera_frame_id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Insert a batch atomically without changing references.
    pub async fn create_trajectory_camera_frame_refs(
        &self,
        mongo: &MongoOrm,
        records: &[TrajectoryCameraFrameRef],
    ) -> sqlx::Result<()> {
        for record in records {
            validations::validate_camera_frame(
                self, mongo, record,
            )
            .await?;
        }
        let mut transaction = self.pool.begin().await?;
        for record in records {
            insert_trajectory_camera_frame_ref(
                &mut *transaction,
                record,
            )
            .await?;
        }
        transaction.commit().await
    }

    /// List matching records; pagination is (limit, offset).
    pub async fn list_groups(
        &self,

        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<Group>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, Group>(
            "SELECT group_id, name, created_at, updated_at,
             login_type FROM product.\"group\" ORDER BY
             created_at DESC, group_id LIMIT $1 OFFSET $2",
        )
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// List matching records; pagination is (limit, offset).
    pub async fn list_users(
        &self,
        group_id: &str,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<User>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, User>(
            "SELECT user_id, group_id, created_at,
             updated_at, name, email, role FROM
             product.\"user\" WHERE group_id = $1 ORDER BY
             created_at DESC, user_id LIMIT $2 OFFSET $3",
        )
        .bind(group_id)
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// List matching records; pagination is (limit, offset).
    pub async fn list_access_keys(
        &self,
        user_id: &str,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<AccessKey>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, AccessKey>(
            "SELECT access_key_id, access_key_value, user_id,
             name, created_at, status FROM product.access_key
             WHERE user_id = $1 ORDER BY created_at DESC,
             access_key_id LIMIT $2 OFFSET $3",
        )
        .bind(user_id)
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// List matching records; pagination is (limit, offset).
    pub async fn list_projects(
        &self,
        user_id: &str,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<Project>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, Project>(
            "SELECT project_id, user_id, created_at,
             updated_at, name FROM product.project WHERE
             user_id = $1 ORDER BY created_at DESC,
             project_id LIMIT $2 OFFSET $3",
        )
        .bind(user_id)
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// List matching records; pagination is (limit, offset).
    pub async fn list_simulation_model_refs(
        &self,
        project_id: &str,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<SimulationModelRef>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, SimulationModelRef>(
            "SELECT project_id, simulation_model_id, created_at,
             updated_at, name, uri_path, model_type,
             file_type, base_file_name FROM
             assets.simulation_model_ref WHERE project_id = $1
             ORDER BY created_at DESC, simulation_model_id
             LIMIT $2 OFFSET $3",
        )
        .bind(project_id)
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// List matching records; pagination is (limit, offset).
    pub async fn list_checkpoint_refs(
        &self,
        workload_id: Option<&str>,
        model_type: Option<&CheckpointModelType>,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<CheckpointRef>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, CheckpointRef>(
            "SELECT checkpoint_id, created_at, updated_at,
             name, uri_path, workload_id,
             workload_structure_id, workload_structure_state,
             na_indices, model_type, model_spec FROM
             assets.checkpoint_ref WHERE ($1::text IS NULL OR
             workload_id = $1) AND ($2::text IS NULL OR
             model_type = $2) ORDER BY created_at,
             checkpoint_id LIMIT $3 OFFSET $4",
        )
        .bind(workload_id)
        .bind(model_type)
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// Include globally shared predefined checkpoints alongside
    /// checkpoints produced by the requested project's workloads.
    pub async fn list_project_checkpoint_refs(
        &self,
        project_id: &str,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<CheckpointRef>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, CheckpointRef>(
            "SELECT c.* FROM assets.checkpoint_ref c LEFT JOIN
             workloads.workload w ON c.workload_id = w.workload_id
             WHERE c.workload_id IS NULL OR w.project_id = $1
             ORDER BY c.created_at, c.checkpoint_id
             LIMIT $2 OFFSET $3",
        )
        .bind(project_id)
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// List matching records; pagination is (limit, offset).
    pub async fn list_trajectory_camera_frame_refs(
        &self,
        workload_id: &str,
        simulation_model_id: Option<&str>,
        camera_id: Option<&str>,
        workload_structure_state: Option<&[u32]>,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<TrajectoryCameraFrameRef>> {
        validations::validate_pagination(pagination)?;
        sqlx::query_as::<_, TrajectoryCameraFrameRef>(
            "SELECT camera_frame_id, workload_id,
             workload_structure_id, workload_structure_state,
             camera_id, model_camera_name, uri_path,
             simulation_model_id FROM
             assets.trajectory_camera_frame_ref WHERE
             workload_id = $1 AND ($2::text IS NULL OR
             simulation_model_id = $2) AND ($3::text IS NULL
             OR camera_id = $3) AND ($4::bigint[] IS NULL OR
             workload_structure_state = $4) ORDER BY
             workload_structure_state, camera_frame_id LIMIT
             $5 OFFSET $6",
        )
        .bind(workload_id)
        .bind(simulation_model_id)
        .bind(camera_id)
        .bind(workload_structure_state.map(encode_coordinates))
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// Fetch records by the supplied lookup value.
    pub async fn get_users_by_email(
        &self,
        email: &str,
    ) -> sqlx::Result<Vec<User>> {
        sqlx::query_as::<_, User>(
            "SELECT user_id, group_id, created_at,
             updated_at, name, email, role FROM
             product.\"user\" WHERE email = $1 ORDER BY
             user_id",
        )
        .bind(email)
        .fetch_all(&self.pool)
        .await
    }

    /// Fetch records by the supplied lookup value.
    pub async fn get_access_key_by_value(
        &self,
        value: &str,
    ) -> sqlx::Result<Option<AccessKey>> {
        sqlx::query_as::<_, AccessKey>(
            "SELECT access_key_id, access_key_value, user_id,
             name, created_at, status FROM product.access_key
             WHERE access_key_value = $1",
        )
        .bind(value)
        .fetch_optional(&self.pool)
        .await
    }

    /// Change key status without changing its value or identity.
    pub async fn update_access_key_status(
        &self,
        id: &str,
        status: &AccessKeyStatus,
    ) -> sqlx::Result<AccessKey> {
        sqlx::query_as::<_, AccessKey>(
            "UPDATE product.access_key SET status = $2 WHERE
             access_key_id = $1 RETURNING access_key_id,
             access_key_value, user_id, name, created_at,
             status",
        )
        .bind(id)
        .bind(status)
        .fetch_one(&self.pool)
        .await
    }

    /// Insert related asset references in one SQL transaction.
    /// Models precede frames, satisfying their foreign keys.
    pub async fn create_asset_refs(
        &self,
        mongo: &MongoOrm,
        models: &[SimulationModelRef],
        checkpoints: &[CheckpointRef],
        frames: &[TrajectoryCameraFrameRef],
    ) -> sqlx::Result<()> {
        for record in checkpoints {
            validations::validate_checkpoint_ref(
                self, mongo, record,
            )
            .await?;
        }
        for record in frames {
            validations::validate_camera_frame(
                self, mongo, record,
            )
            .await?;
        }
        let mut transaction = self.pool.begin().await?;
        for record in models {
            insert_simulation_model_ref(
                &mut *transaction,
                record,
            )
            .await?;
        }
        for record in checkpoints {
            insert_checkpoint_ref(&mut *transaction, record)
                .await?;
        }
        for record in frames {
            insert_trajectory_camera_frame_ref(
                &mut *transaction,
                record,
            )
            .await?;
        }
        transaction.commit().await
    }

    /// Create SQL metadata only; the graph is stored separately.
    pub(super) async fn create_workload_metadata(
        &self,
        record: &Workload,
    ) -> sqlx::Result<()> {
        validations::validate_workload_timestamps(record)?;
        validations::validate_graph_type(
            &record.workload_graph,
            &record.workload_type,
        )?;
        let result = sqlx::query(
            "INSERT INTO workloads.workload (workload_id,
             workload_type, project_id, user_id, created_at,
             updated_at, launched_at, launch_origin,
             scheduled_at, completed_at, status,
             workload_graph_id, tags, name, description, simulation_model_ids) VALUES ($1, $2, $3, $4,
             $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)",
        )
        .bind(&record.workload_id)
        .bind(&record.workload_type)
        .bind(&record.project_id)
        .bind(&record.user_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(record.launched_at)
        .bind(&record.launch_origin)
        .bind(record.scheduled_at)
        .bind(record.completed_at)
        .bind(&record.status)
        .bind(&record.workload_graph_id)
        .bind(&record.tags)
        .bind(&record.name)
        .bind(&record.description)
        .bind(&record.simulation_model_ids)
        .execute(&self.pool)
        .await?;
        validations::validate_affected_rows(
            result.rows_affected(),
        )
    }

    /// Update SQL metadata only; the graph is stored separately.
    pub(super) async fn update_workload_metadata(
        &self,
        record: &Workload,
    ) -> sqlx::Result<()> {
        validations::validate_workload_timestamps(record)?;
        validations::validate_graph_type(
            &record.workload_graph,
            &record.workload_type,
        )?;
        let result = sqlx::query(
            "UPDATE workloads.workload SET workload_type =
             $2, project_id = $3, user_id = $4, created_at =
             $5, updated_at = $6, launched_at = $7,
             launch_origin = $8, scheduled_at = $9,
             completed_at = $10, status = $11,
             workload_graph_id = $12, tags = $13, name = $14,
             description = $15, simulation_model_ids = $16 WHERE
             workload_id = $1",
        )
        .bind(&record.workload_id)
        .bind(&record.workload_type)
        .bind(&record.project_id)
        .bind(&record.user_id)
        .bind(record.created_at)
        .bind(record.updated_at)
        .bind(record.launched_at)
        .bind(&record.launch_origin)
        .bind(record.scheduled_at)
        .bind(record.completed_at)
        .bind(&record.status)
        .bind(&record.workload_graph_id)
        .bind(&record.tags)
        .bind(&record.name)
        .bind(&record.description)
        .bind(&record.simulation_model_ids)
        .execute(&self.pool)
        .await?;
        validations::validate_affected_rows(
            result.rows_affected(),
        )
    }

    pub(super) async fn workload_row(
        &self,
        id: &str,
    ) -> sqlx::Result<Option<PgRow>> {
        sqlx::query(
            "SELECT workload_id, workload_type, project_id,
             user_id, created_at, updated_at, launched_at,
             launch_origin, scheduled_at, completed_at,
             status, workload_graph_id, tags, name, description, simulation_model_ids FROM
             workloads.workload WHERE workload_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    pub(super) async fn workload_rows(
        &self,
        project_id: Option<&str>,
        user_id: Option<&str>,
        workload_type: Option<&WorkloadType>,
        status: Option<&WorkloadStatus>,
        pagination: (i64, i64),
    ) -> sqlx::Result<Vec<PgRow>> {
        validations::validate_pagination(pagination)?;
        sqlx::query(
            "SELECT workload_id, workload_type, project_id,
             user_id, created_at, updated_at, launched_at,
             launch_origin, scheduled_at, completed_at,
             status, workload_graph_id, tags, name, description, simulation_model_ids FROM
             workloads.workload WHERE ($1::text IS NULL OR
             project_id = $1) AND ($2::text IS NULL OR
             user_id = $2) AND ($3::text IS NULL OR
             workload_type = $3) AND ($4::text IS NULL OR
             status = $4) ORDER BY created_at DESC,
             workload_id LIMIT $5 OFFSET $6",
        )
        .bind(project_id)
        .bind(user_id)
        .bind(workload_type)
        .bind(status)
        .bind(pagination.0)
        .bind(pagination.1)
        .fetch_all(&self.pool)
        .await
    }

    /// Attach generated model IDs after their references are stored.
    pub async fn attach_workload_models(
        &self,
        id: &str,
        model_ids: &[String],
    ) -> sqlx::Result<()> {
        validations::validate_generated_model_associations(
            self, id, model_ids,
        )
        .await?;
        sqlx::query(
            "UPDATE workloads.workload SET simulation_model_ids =
             ARRAY(SELECT DISTINCT unnest(simulation_model_ids ||
             $2::text[])), updated_at = now() WHERE workload_id = $1",
        )
        .bind(id)
        .bind(model_ids)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Set lifecycle values without inferring stop reasons.
    pub async fn update_workload_status(
        &self,
        id: &str,
        status: &WorkloadStatus,
        updated_at: DateTime<Utc>,
        launched_at: Option<DateTime<Utc>>,
        completed_at: Option<DateTime<Utc>>,
    ) -> sqlx::Result<()> {
        validations::validate_workload_status_update(
            self,
            id,
            updated_at,
            launched_at,
            completed_at,
        )
        .await?;
        let result = sqlx::query(
            "UPDATE workloads.workload SET status = $2,
             updated_at = $3, launched_at = $4, completed_at
             = $5 WHERE workload_id = $1",
        )
        .bind(id)
        .bind(status)
        .bind(updated_at)
        .bind(launched_at)
        .bind(completed_at)
        .execute(&self.pool)
        .await?;
        validations::validate_affected_rows(
            result.rows_affected(),
        )
    }

    /// Update tags using the caller's modification timestamp.
    pub async fn update_workload_tags(
        &self,
        id: &str,
        tags: &[String],
        updated_at: DateTime<Utc>,
    ) -> sqlx::Result<()> {
        validations::validate_workload_modified_at(
            self, id, updated_at,
        )
        .await?;
        let result = sqlx::query(
            "UPDATE workloads.workload SET tags = $2,
             updated_at = $3 WHERE workload_id = $1",
        )
        .bind(id)
        .bind(tags)
        .bind(updated_at)
        .execute(&self.pool)
        .await?;
        validations::validate_affected_rows(
            result.rows_affected(),
        )
    }

    /// Delete SQL metadata, enforcing asset foreign keys.
    pub async fn delete_workload_metadata(
        &self,
        id: &str,
    ) -> sqlx::Result<bool> {
        let result = sqlx::query(
            "DELETE FROM workloads.workload WHERE workload_id
             = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() != 0)
    }

    /// Read an immutable LUT entry.
    pub async fn get_workload_structure_lut(
        &self,
        id: i32,
    ) -> sqlx::Result<Option<WorkloadStructureLut>> {
        sqlx::query_as::<_, WorkloadStructureLut>(
            "SELECT workload_structure_id, workload_type,
             workload_state_index_names,
             number_of_state_indices, method_applicability
             FROM workloads.workload_structure_lut WHERE
             workload_structure_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Read immutable LUT entries, optionally by workload type.
    pub async fn list_workload_structure_lut(
        &self,
        workload_type: Option<&WorkloadType>,
    ) -> sqlx::Result<Vec<WorkloadStructureLut>> {
        sqlx::query_as::<_, WorkloadStructureLut>(
            "SELECT workload_structure_id, workload_type,
             workload_state_index_names,
             number_of_state_indices, method_applicability
             FROM workloads.workload_structure_lut WHERE
             ($1::text IS NULL OR workload_type = $1) ORDER
             BY workload_structure_id",
        )
        .bind(workload_type)
        .fetch_all(&self.pool)
        .await
    }

    /// Read an immutable LUT entry.
    pub async fn get_user_metrics_lut(
        &self,
        id: i32,
    ) -> sqlx::Result<Option<UserMetricsLut>> {
        sqlx::query_as::<_, UserMetricsLut>(
            "SELECT metric_id, name, workload_structure_id,
             workload_type, aggregation_functions,
             aggregation_indices, na_indices FROM
             workloads.user_metrics_lut WHERE metric_id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Read immutable LUT entries, optionally by workload type.
    pub async fn list_user_metrics_lut(
        &self,
        workload_type: Option<&WorkloadType>,
    ) -> sqlx::Result<Vec<UserMetricsLut>> {
        sqlx::query_as::<_, UserMetricsLut>(
            "SELECT metric_id, name, workload_structure_id,
             workload_type, aggregation_functions,
             aggregation_indices, na_indices FROM
             workloads.user_metrics_lut WHERE ($1::text IS
             NULL OR workload_type = $1) ORDER BY metric_id",
        )
        .bind(workload_type)
        .fetch_all(&self.pool)
        .await
    }
}

async fn insert_simulation_model_ref<'e>(
    executor: impl Executor<'e, Database = Postgres>,
    record: &SimulationModelRef,
) -> sqlx::Result<SimulationModelRef> {
    validations::validate_modified_at(
        record.created_at,
        record.updated_at,
    )?;
    sqlx::query_as::<_, SimulationModelRef>(
        "INSERT INTO assets.simulation_model_ref
             (simulation_model_id, created_at, updated_at,
             name, uri_path, model_type, file_type,
             base_file_name, project_id) VALUES ($1, $2, $3, $4, $5, $6,
             $7, $8, $9) RETURNING project_id, simulation_model_id,
             created_at, updated_at, name, uri_path,
             model_type, file_type, base_file_name",
    )
    .bind(&record.simulation_model_id)
    .bind(record.created_at)
    .bind(record.updated_at)
    .bind(&record.name)
    .bind(&record.uri_path)
    .bind(&record.model_type)
    .bind(&record.file_type)
    .bind(&record.base_file_name)
    .bind(&record.project_id)
    .fetch_one(executor)
    .await
}

impl<'r> FromRow<'r, PgRow> for SimulationModelRef {
    fn from_row(row: &'r PgRow) -> sqlx::Result<Self> {
        Ok(Self {
            project_id: row.try_get("project_id")?,
            simulation_model_id: row
                .try_get("simulation_model_id")?,
            created_at: row.try_get("created_at")?,
            updated_at: row.try_get("updated_at")?,
            name: row.try_get("name")?,
            uri_path: row.try_get("uri_path")?,
            model_type: row.try_get("model_type")?,
            file_type: row.try_get("file_type")?,
            base_file_name: row.try_get("base_file_name")?,
        })
    }
}

async fn insert_checkpoint_ref<'e>(
    executor: impl Executor<'e, Database = Postgres>,
    record: &CheckpointRef,
) -> sqlx::Result<CheckpointRef> {
    sqlx::query_as::<_, CheckpointRef>(
        "INSERT INTO assets.checkpoint_ref
             (checkpoint_id, created_at, updated_at, name,
             uri_path, workload_id, workload_structure_id,
             workload_structure_state, na_indices,
             model_type, model_spec) VALUES ($1, $2, $3, $4,
             $5, $6, $7, $8, $9, $10, $11) RETURNING
             checkpoint_id, created_at, updated_at, name,
             uri_path, workload_id, workload_structure_id,
             workload_structure_state, na_indices,
             model_type, model_spec",
    )
    .bind(&record.checkpoint_id)
    .bind(record.created_at)
    .bind(record.updated_at)
    .bind(&record.name)
    .bind(&record.uri_path)
    .bind(&record.workload_id)
    .bind(record.workload_structure_id)
    .bind(
        record
            .workload_structure_state
            .as_deref()
            .map(encode_coordinates),
    )
    .bind(encode_coordinates(&record.na_indices))
    .bind(&record.model_type)
    .bind(Json(&record.model_spec))
    .fetch_one(executor)
    .await
}

impl<'r> FromRow<'r, PgRow> for CheckpointRef {
    fn from_row(row: &'r PgRow) -> sqlx::Result<Self> {
        Ok(Self {
            checkpoint_id: row.try_get("checkpoint_id")?,
            created_at: row.try_get("created_at")?,
            updated_at: row.try_get("updated_at")?,
            name: row.try_get("name")?,
            uri_path: row.try_get("uri_path")?,
            workload_id: row.try_get("workload_id")?,
            workload_structure_id: row
                .try_get("workload_structure_id")?,
            workload_structure_state:
                decode_optional_coordinates(
                    row,
                    "workload_structure_state",
                )?,
            na_indices: decode_coordinates(row, "na_indices")?,
            model_type: row.try_get("model_type")?,
            model_spec: row
                .try_get::<Json<MLModel>, _>("model_spec")?
                .0,
        })
    }
}

async fn insert_trajectory_camera_frame_ref<'e>(
    executor: impl Executor<'e, Database = Postgres>,
    record: &TrajectoryCameraFrameRef,
) -> sqlx::Result<TrajectoryCameraFrameRef> {
    sqlx::query_as::<_, TrajectoryCameraFrameRef>(
        "INSERT INTO assets.trajectory_camera_frame_ref
             (camera_frame_id, workload_id,
             workload_structure_id, workload_structure_state,
             camera_id, model_camera_name, uri_path,
             simulation_model_id) VALUES ($1, $2, $3, $4, $5,
             $6, $7, $8) RETURNING camera_frame_id,
             workload_id, workload_structure_id,
             workload_structure_state, camera_id,
             model_camera_name, uri_path, simulation_model_id",
    )
    .bind(&record.camera_frame_id)
    .bind(&record.workload_id)
    .bind(record.workload_structure_id)
    .bind(encode_coordinates(&record.workload_structure_state))
    .bind(&record.camera_id)
    .bind(&record.model_camera_name)
    .bind(&record.uri_path)
    .bind(&record.simulation_model_id)
    .fetch_one(executor)
    .await
}

impl<'r> FromRow<'r, PgRow> for TrajectoryCameraFrameRef {
    fn from_row(row: &'r PgRow) -> sqlx::Result<Self> {
        Ok(Self {
            camera_frame_id: row.try_get("camera_frame_id")?,
            workload_id: row.try_get("workload_id")?,
            workload_structure_id: row
                .try_get("workload_structure_id")?,
            workload_structure_state: decode_coordinates(
                row,
                "workload_structure_state",
            )?,
            camera_id: row.try_get("camera_id")?,
            model_camera_name: row
                .try_get("model_camera_name")?,
            uri_path: row.try_get("uri_path")?,
            simulation_model_id: row
                .try_get("simulation_model_id")?,
        })
    }
}

impl<'r> FromRow<'r, PgRow> for WorkloadStructureLut {
    fn from_row(row: &'r PgRow) -> sqlx::Result<Self> {
        Ok(Self {
            workload_structure_id: row
                .try_get("workload_structure_id")?,
            workload_type: row.try_get("workload_type")?,
            workload_state_index_names: row
                .try_get("workload_state_index_names")?,
            number_of_state_indices: decode_u32(
                row.try_get("number_of_state_indices")?,
                "number_of_state_indices",
            )?,
            method_applicability: row
                .try_get("method_applicability")?,
        })
    }
}

impl<'r> FromRow<'r, PgRow> for UserMetricsLut {
    fn from_row(row: &'r PgRow) -> sqlx::Result<Self> {
        let functions: Vec<String> =
            row.try_get("aggregation_functions")?;
        let aggregation_functions =
            serde_json::from_value(serde_json::json!(functions))
                .map_err(|error| sqlx::Error::ColumnDecode {
                    index: "aggregation_functions".into(),
                    source: Box::new(error),
                })?;
        Ok(Self {
            metric_id: row.try_get("metric_id")?,
            name: row.try_get("name")?,
            workload_structure_id: row
                .try_get("workload_structure_id")?,
            workload_type: row.try_get("workload_type")?,
            aggregation_functions,
            aggregation_indices: decode_coordinates(
                row,
                "aggregation_indices",
            )?,
            na_indices: decode_coordinates(row, "na_indices")?,
        })
    }
}

pub(super) fn workload_from_row(
    row: &PgRow,
    graph: WorkloadGraph,
) -> sqlx::Result<Workload> {
    Ok(Workload {
        workload_id: row.try_get("workload_id")?,
        workload_type: row.try_get("workload_type")?,
        project_id: row.try_get("project_id")?,
        user_id: row.try_get("user_id")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        launched_at: row.try_get("launched_at")?,
        launch_origin: row.try_get("launch_origin")?,
        scheduled_at: row.try_get("scheduled_at")?,
        completed_at: row.try_get("completed_at")?,
        status: row.try_get("status")?,
        workload_graph_id: row.try_get("workload_graph_id")?,
        tags: row.try_get("tags")?,
        simulation_model_ids: row
            .try_get("simulation_model_ids")?,
        name: row.try_get("name")?,
        description: row.try_get("description")?,
        workload_graph: graph,
    })
}

fn encode_coordinates(values: &[u32]) -> Vec<i64> {
    values.iter().map(|&value| i64::from(value)).collect()
}

fn decode_u32(value: i64, column: &str) -> sqlx::Result<u32> {
    u32::try_from(value).map_err(|error| {
        sqlx::Error::ColumnDecode {
            index: column.into(),
            source: Box::new(error),
        }
    })
}

fn decode_coordinates(
    row: &PgRow,
    column: &str,
) -> sqlx::Result<Vec<u32>> {
    row.try_get::<Vec<i64>, _>(column)?
        .into_iter()
        .map(|v| decode_u32(v, column))
        .collect()
}

fn decode_optional_coordinates(
    row: &PgRow,
    column: &str,
) -> sqlx::Result<Option<Vec<u32>>> {
    row.try_get::<Option<Vec<i64>>, _>(column)?
        .map(|values| {
            values
                .into_iter()
                .map(|v| decode_u32(v, column))
                .collect()
        })
        .transpose()
}
