//! Postgres ORM for User, Project,
//! TrainingExperiment, TrainingExperimentRun,
//! ModelReference, and CheckpointReference.

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::commondef::experiments::{
    CheckpointReference, ModelReference,
    Project, TrainingExperiment,
    TrainingExperimentRun, TrainingRunStatus,
};
use crate::commondef::users::User;

/// Holds a Postgres connection pool and
/// provides CRUD operations.
pub struct PostgresOrm {
    pool: PgPool,
}

impl PostgresOrm {
    /// Connect to Postgres and return a
    /// new PostgresOrm instance.
    pub async fn new(
        database_url: &str,
    ) -> Result<Self, sqlx::Error> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;
        Ok(Self { pool })
    }

    /// Return a reference to the pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    // ── User CRUD ──

    /// Insert a new user.
    pub async fn create_user(
        &self,
        user_id: &str,
        name: &str,
        email: &str,
    ) -> Result<User, sqlx::Error> {
        sqlx::query_as::<_, User>(
            "INSERT INTO users (
                user_id, created_at,
                name, email
            )
            VALUES ($1, NOW(), $2, $3)
            RETURNING *",
        )
        .bind(user_id)
        .bind(name)
        .bind(email)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a user by id.
    pub async fn get_user(
        &self,
        user_id: &str,
    ) -> Result<Option<User>, sqlx::Error> {
        sqlx::query_as::<_, User>(
            "SELECT * FROM users
            WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Update a user's name and email.
    pub async fn update_user(
        &self,
        user_id: &str,
        name: &str,
        email: &str,
    ) -> Result<User, sqlx::Error> {
        sqlx::query_as::<_, User>(
            "UPDATE users
            SET name = $2, email = $3
            WHERE user_id = $1
            RETURNING *",
        )
        .bind(user_id)
        .bind(name)
        .bind(email)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete a user by id.
    pub async fn delete_user(
        &self,
        user_id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "DELETE FROM users
            WHERE user_id = $1",
        )
        .bind(user_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── Project CRUD ──

    /// Insert a new project.
    pub async fn create_project(
        &self,
        project_id: &str,
        user_id: i64,
    ) -> Result<Project, sqlx::Error> {
        sqlx::query_as::<_, Project>(
            "INSERT INTO projects (
                project_id, user_id,
                created_at
            )
            VALUES ($1, $2, NOW())
            RETURNING *",
        )
        .bind(project_id)
        .bind(user_id)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a project by id.
    pub async fn get_project(
        &self,
        project_id: &str,
    ) -> Result<Option<Project>, sqlx::Error>
    {
        sqlx::query_as::<_, Project>(
            "SELECT * FROM projects
            WHERE project_id = $1",
        )
        .bind(project_id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Fetch all projects for a user.
    pub async fn get_projects_by_user(
        &self,
        user_id: i64,
    ) -> Result<Vec<Project>, sqlx::Error> {
        sqlx::query_as::<_, Project>(
            "SELECT * FROM projects
            WHERE user_id = $1
            ORDER BY created_at DESC",
        )
        .bind(user_id)
        .fetch_all(&self.pool)
        .await
    }

    /// Delete a project by id.
    pub async fn delete_project(
        &self,
        project_id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "DELETE FROM projects
            WHERE project_id = $1",
        )
        .bind(project_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── TrainingExperiment CRUD ──

    /// Insert a new training experiment.
    pub async fn create_experiment(
        &self,
        project_id: &str,
        training_experiment_id: &str,
        experiment_name: &str,
        training_run_ids: &[String],
    ) -> Result<
        TrainingExperiment,
        sqlx::Error,
    > {
        sqlx::query_as::<
            _,
            TrainingExperiment,
        >(
            "INSERT INTO
            training_experiments (
                project_id,
                training_experiment_id,
                experiment_name,
                created_at,
                training_run_ids
            )
            VALUES ($1, $2, $3, NOW(), $4)
            RETURNING *",
        )
        .bind(project_id)
        .bind(training_experiment_id)
        .bind(experiment_name)
        .bind(training_run_ids)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a training experiment by id.
    pub async fn get_experiment(
        &self,
        training_experiment_id: &str,
    ) -> Result<
        Option<TrainingExperiment>,
        sqlx::Error,
    > {
        sqlx::query_as::<
            _,
            TrainingExperiment,
        >(
            "SELECT *
            FROM training_experiments
            WHERE
            training_experiment_id = $1",
        )
        .bind(training_experiment_id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Fetch experiments for a project.
    pub async fn get_experiments_by_project(
        &self,
        project_id: &str,
    ) -> Result<
        Vec<TrainingExperiment>,
        sqlx::Error,
    > {
        sqlx::query_as::<
            _,
            TrainingExperiment,
        >(
            "SELECT *
            FROM training_experiments
            WHERE project_id = $1
            ORDER BY created_at DESC",
        )
        .bind(project_id)
        .fetch_all(&self.pool)
        .await
    }

    /// Append a run id to an experiment's
    /// training_run_ids array.
    pub async fn add_run_to_experiment(
        &self,
        training_experiment_id: &str,
        run_id: &str,
    ) -> Result<
        TrainingExperiment,
        sqlx::Error,
    > {
        sqlx::query_as::<
            _,
            TrainingExperiment,
        >(
            "UPDATE training_experiments
            SET training_run_ids =
                array_append(
                    training_run_ids, $2
                )
            WHERE
            training_experiment_id = $1
            RETURNING *",
        )
        .bind(training_experiment_id)
        .bind(run_id)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete a training experiment.
    pub async fn delete_experiment(
        &self,
        training_experiment_id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "DELETE
            FROM training_experiments
            WHERE
            training_experiment_id = $1",
        )
        .bind(training_experiment_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── TrainingExperimentRun CRUD ──

    /// Insert a new training run.
    pub async fn create_run(
        &self,
        project_id: &str,
        training_experiment_id: &str,
        id: &str,
        run_status: &TrainingRunStatus,
        logs_id: &str,
        metrics_id: &str,
        training_graph_id: &str,
        checkpoints_refs: &[String],
        model_refs: &[String],
    ) -> Result<TrainingExperimentRun, sqlx::Error> {
        sqlx::query_as::<_, TrainingExperimentRun>(
            "INSERT INTO training_runs (
                project_id,
                training_experiment_id,
                id, created_at,
                run_start_at,
                run_end_at,
                run_status, logs_id,
                metrics_id,
                training_graph_id,
                checkpoints_refs,
                model_refs
            )
            VALUES (
                $1, $2, $3, NOW(),
                NULL, NULL,
                $4, $5, $6, $7,
                $8, $9
            )
            RETURNING *",
        )
        .bind(project_id)
        .bind(training_experiment_id)
        .bind(id)
        .bind(run_status)
        .bind(logs_id)
        .bind(metrics_id)
        .bind(training_graph_id)
        .bind(checkpoints_refs)
        .bind(model_refs)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a training run by key.
    pub async fn get_run(
        &self,
        project_id: &str,
        training_experiment_id: &str,
        id: &str,
    ) -> Result<
        Option<TrainingExperimentRun>,
        sqlx::Error,
    > {
        sqlx::query_as::<_, TrainingExperimentRun>(
            "SELECT * FROM training_runs
            WHERE project_id = $1
            AND training_experiment_id = $2
            AND id = $3",
        )
        .bind(project_id)
        .bind(training_experiment_id)
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Fetch all runs for an experiment.
    pub async fn get_runs_by_experiment(
        &self,
        project_id: &str,
        training_experiment_id: &str,
    ) -> Result<
        Vec<TrainingExperimentRun>,
        sqlx::Error,
    > {
        sqlx::query_as::<_, TrainingExperimentRun>(
            "SELECT * FROM training_runs
            WHERE project_id = $1
            AND training_experiment_id = $2
            ORDER BY created_at DESC",
        )
        .bind(project_id)
        .bind(training_experiment_id)
        .fetch_all(&self.pool)
        .await
    }

    /// Update a run's status.
    pub async fn update_run_status(
        &self,
        project_id: &str,
        training_experiment_id: &str,
        id: &str,
        run_status: &TrainingRunStatus,
    ) -> Result<TrainingExperimentRun, sqlx::Error> {
        let query = match run_status {
            TrainingRunStatus::Ongoing => {
                "UPDATE training_runs
                SET run_status = $4,
                    run_start_at = NOW()
                WHERE project_id = $1
                AND
                training_experiment_id = $2
                AND id = $3
                RETURNING *"
            }
            TrainingRunStatus::Completed => {
                "UPDATE training_runs
                SET run_status = $4,
                    run_end_at = NOW()
                WHERE project_id = $1
                AND
                training_experiment_id = $2
                AND id = $3
                RETURNING *"
            }
            TrainingRunStatus::NotStarted => {
                "UPDATE training_runs
                SET run_status = $4,
                    run_start_at = NULL,
                    run_end_at = NULL
                WHERE project_id = $1
                AND
                training_experiment_id = $2
                AND id = $3
                RETURNING *"
            }
        };
        sqlx::query_as::<_, TrainingExperimentRun>(
            query,
        )
        .bind(project_id)
        .bind(training_experiment_id)
        .bind(id)
        .bind(run_status)
        .fetch_one(&self.pool)
        .await
    }

    /// Delete a training run.
    pub async fn delete_run(
        &self,
        project_id: &str,
        training_experiment_id: &str,
        id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "DELETE FROM training_runs
            WHERE project_id = $1
            AND training_experiment_id = $2
            AND id = $3",
        )
        .bind(project_id)
        .bind(training_experiment_id)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── ModelReference CRUD ──

    /// Insert a new model reference.
    pub async fn create_model_reference(
        &self,
        id: &str,
        path: &str,
        model_type: &str,
    ) -> Result<ModelReference, sqlx::Error>
    {
        sqlx::query_as::<_, ModelReference>(
            "INSERT INTO model_references (
                id, created_at,
                path, model_type
            )
            VALUES ($1, NOW(), $2, $3)
            RETURNING *",
        )
        .bind(id)
        .bind(path)
        .bind(model_type)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a model reference by id.
    pub async fn get_model_reference(
        &self,
        id: &str,
    ) -> Result<
        Option<ModelReference>,
        sqlx::Error,
    > {
        sqlx::query_as::<_, ModelReference>(
            "SELECT *
            FROM model_references
            WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Delete a model reference by id.
    pub async fn delete_model_reference(
        &self,
        id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "DELETE FROM model_references
            WHERE id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ── CheckpointReference CRUD ──

    /// Insert a new checkpoint reference.
    pub async fn create_checkpoint_ref(
        &self,
        id: &str,
        path: &str,
        checkpoint_type: &str,
    ) -> Result<
        CheckpointReference,
        sqlx::Error,
    > {
        sqlx::query_as::<
            _,
            CheckpointReference,
        >(
            "INSERT INTO
            checkpoint_references (
                id, created_at,
                path, checkpoint_type
            )
            VALUES ($1, NOW(), $2, $3)
            RETURNING *",
        )
        .bind(id)
        .bind(path)
        .bind(checkpoint_type)
        .fetch_one(&self.pool)
        .await
    }

    /// Fetch a checkpoint reference by id.
    pub async fn get_checkpoint_ref(
        &self,
        id: &str,
    ) -> Result<
        Option<CheckpointReference>,
        sqlx::Error,
    > {
        sqlx::query_as::<
            _,
            CheckpointReference,
        >(
            "SELECT *
            FROM checkpoint_references
            WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
    }

    /// Delete a checkpoint reference.
    pub async fn delete_checkpoint_ref(
        &self,
        id: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            "DELETE
            FROM checkpoint_references
            WHERE id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
