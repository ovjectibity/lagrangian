//! Independently auditable ORM validations.
//! Numbers refer to the agreed 37-rule validation table.
//! LUT definitions are trusted; only supplied records are
//! checked.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use sqlx::{Result, Row};

use crate::commondef::records::assets::{
    CheckpointModelType, CheckpointRef, TrajectoryCameraFrameRef,
};
use crate::commondef::records::model::{Body, Geom, WorldBody};
use crate::commondef::records::workload::{
    UserMetricsLut, Workload, WorkloadLogs, WorkloadMethod,
    WorkloadSimulationModelTrajectoryState, WorkloadStructureLut,
    WorkloadType, WorkloadUserMetrics,
};
use crate::commondef::records::workload_graph::{
    CheckpointType, ControlOutputMode, Dataset,
    FlowMatchingAction, MLModel, ModelType, TestMethod,
    TrainingMethod, WorkloadGraph,
};

use super::db_orm::{self, PostgresOrm};
use super::mongo_orm::MongoOrm;

/// Rule 1: A SQL workload must have its referenced Mongo graph.
pub fn validate_graph_exists(
    graph: Option<WorkloadGraph>,
) -> Result<WorkloadGraph> {
    graph.ok_or_else(|| {
        invalid(
            "Workload metadata references a missing Mongo graph",
        )
    })
}

/// Rule 2: The Mongo graph belongs to the expected SQL workload.
pub fn validate_graph_owner(
    stored_id: &str,
    expected_id: &str,
) -> Result<()> {
    require(stored_id == expected_id, "Graph owner mismatch")
}

/// Rule 3: The graph variant matches the SQL workload type.
pub fn validate_graph_type(
    graph: &WorkloadGraph,
    workload_type: &WorkloadType,
) -> Result<()> {
    require(
        matches!(
            (graph, workload_type),
            (
                WorkloadGraph::TrainingGraph(_),
                WorkloadType::Training
            ) | (
                WorkloadGraph::TestGraph(_),
                WorkloadType::Testing
            )
        ),
        "Graph variant does not match the workload type",
    )
}

/// The stored envelope's type also matches the SQL workload.
pub fn validate_graph_envelope_type(
    stored: &WorkloadType,
    expected: &WorkloadType,
) -> Result<()> {
    require(stored == expected, "Graph envelope type mismatch")
}

/// Rule 4: A model referenced by a graph exists in Postgres.
/// Dataset models are generated later, rather than referenced
/// here.
pub async fn validate_graph_simulation_model(
    postgres: &PostgresOrm,
    graph: &WorkloadGraph,
) -> Result<()> {
    if let Some(id) = graph_simulation_model_id(graph) {
        required(
            postgres.get_simulation_model_ref(id).await?,
            "Graph simulation model does not exist",
        )?;
    }
    Ok(())
}

/// Rule 5: A requested checkpoint ID exists in Postgres.
pub async fn validate_requested_checkpoint_exists(
    postgres: &PostgresOrm,
    id: &str,
) -> Result<CheckpointRef> {
    required(
        postgres.get_checkpoint_ref(id).await?,
        "Requested checkpoint does not exist",
    )
}

/// Rule 6: A requested checkpoint has the component's model role.
pub fn validate_checkpoint_role(
    checkpoint: &CheckpointRef,
    expected: &CheckpointModelType,
) -> Result<()> {
    use CheckpointModelType::{FlowNoise, Policy, Value};
    require(
        matches!(
            (&checkpoint.model_type, expected),
            (Policy, Policy)
                | (Value, Value)
                | (FlowNoise, FlowNoise)
        ),
        "Checkpoint role does not match the graph component",
    )
}

/// Rule 7: The stored model specification matches the
/// requested one.
pub fn validate_checkpoint_model_spec(
    checkpoint: &CheckpointRef,
    expected: &MLModel,
) -> Result<()> {
    let stored = serde_json::to_value(&checkpoint.model_spec)
        .map_err(|error| invalid(&error.to_string()))?;
    let requested = serde_json::to_value(expected)
        .map_err(|error| invalid(&error.to_string()))?;
    require(
        stored == requested,
        "Checkpoint specification does not match the component",
    )
}

/// Rule 8: A Mongo record references an existing, complete
/// workload.
pub async fn validate_workload_exists(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    id: &str,
) -> Result<Workload> {
    let row = required(
        postgres.workload_row(id).await?,
        "Referenced workload does not exist",
    )?;
    let graph = mongo
        .get_workload_graph(
            row.try_get("workload_graph_id")?,
            id,
            &row.try_get("workload_type")?,
        )
        .await
        .map_err(|error| invalid(&error.to_string()))?;
    db_orm::workload_from_row(&row, validate_graph_exists(graph)?)
}

/// Rule 9: A metric record references an existing metric
/// definition.
pub async fn validate_metric_exists(
    postgres: &PostgresOrm,
    id: i32,
) -> Result<UserMetricsLut> {
    required(
        postgres.get_user_metrics_lut(id).await?,
        "Metric definition does not exist",
    )
}

/// Rule 10: A pose references an existing structure definition.
/// The same lookup is used for other coordinate-bearing records.
pub async fn validate_structure_exists(
    postgres: &PostgresOrm,
    id: i32,
) -> Result<WorkloadStructureLut> {
    required(
        postgres.get_workload_structure_lut(id).await?,
        "Structure definition does not exist",
    )
}

/// Rule 11: A Mongo pose references an existing SQL
/// simulation model.
pub async fn validate_pose_model_exists(
    postgres: &PostgresOrm,
    id: &str,
) -> Result<()> {
    required(
        postgres.get_simulation_model_ref(id).await?,
        "Pose simulation model does not exist",
    )?;
    Ok(())
}

/// Rule 12: The structure's workload type matches the workload.
pub fn validate_structure_workload_type(
    structure: &WorkloadStructureLut,
    workload: &Workload,
) -> Result<()> {
    require(
        structure.workload_type == workload.workload_type,
        "Structure workload type does not match the workload",
    )
}

/// Rule 13: The structure applies to the graph's execution
/// method.
pub fn validate_structure_method(
    structure: &WorkloadStructureLut,
    graph: &WorkloadGraph,
) -> Result<()> {
    let method = match graph {
        WorkloadGraph::TrainingGraph(_) => WorkloadMethod::Ppo,
        WorkloadGraph::TestGraph(graph) => {
            match graph.test_job_controller.test_method {
                TestMethod::Custom(_) => WorkloadMethod::Custom,
                TestMethod::Dataset(_) => WorkloadMethod::Dataset,
            }
        }
    };
    require(
        structure.method_applicability == method,
        "Structure does not apply to the workload method",
    )
}

/// Rule 14: A structure describes the record's execution stage.
/// This compares a record's required layout with trusted LUT
/// data.
pub fn validate_structure_stage(
    structure: &WorkloadStructureLut,
    expected_dimensions: &[&str],
) -> Result<()> {
    require(
        structure
            .workload_state_index_names
            .iter()
            .map(String::as_str)
            .eq(expected_dimensions.iter().copied()),
        "Structure stage does not match the record",
    )
}

/// Rule 15: A frame/pose contains every trajectory coordinate.
pub fn validate_trajectory_coordinate_count(
    structure: &WorkloadStructureLut,
    coordinate: &[u32],
) -> Result<()> {
    require(
        coordinate.len()
            == structure.number_of_state_indices as usize,
        "Trajectory coordinate count mismatch",
    )
}

/// Rule 16: A metric contains its remaining dimensions, in
/// LUT order.
pub fn validate_metric_coordinate_count(
    record: &WorkloadUserMetrics,
    retained_indices: &[u32],
) -> Result<()> {
    require(
        record
            .workload_structure_state_values
            .iter()
            .all(|state| state.len() == retained_indices.len()),
        "Metric coordinate count does not match its definition",
    )
}

/// Rule 17: A checkpoint contains dimensions retained after
/// NA removal.
pub fn validate_checkpoint_coordinate_count(
    coordinate: &[u32],
    retained_indices: &[u32],
) -> Result<()> {
    require(
        coordinate.len() == retained_indices.len(),
        "Checkpoint coordinate count mismatch",
    )
}

/// Rule 18: Checkpoint NA indices identify existing LUT
/// positions.
pub fn validate_checkpoint_na_bounds(
    checkpoint: &CheckpointRef,
    structure: &WorkloadStructureLut,
) -> Result<()> {
    require(
        checkpoint
            .na_indices
            .iter()
            .all(|&i| i < structure.number_of_state_indices),
        "Checkpoint NA index exceeds its structure dimensions",
    )
}

/// Rule 19: Checkpoint NA positions do not repeat.
pub fn validate_checkpoint_na_uniqueness(
    checkpoint: &CheckpointRef,
) -> Result<()> {
    let distinct: HashSet<_> =
        checkpoint.na_indices.iter().collect();
    require(
        distinct.len() == checkpoint.na_indices.len(),
        "Checkpoint NA positions must be unique",
    )
}

/// Rule 20: Structure ID and checkpoint coordinates occur
/// together.
pub fn validate_checkpoint_structure_pair(
    checkpoint: &CheckpointRef,
) -> Result<()> {
    require(
        checkpoint.workload_structure_id.is_some()
            == checkpoint.workload_structure_state.is_some(),
        "Checkpoint structure ID and coordinates must pair",
    )
}

/// Rule 21: Checkpoint coordinates require an originating
/// workload.
pub fn validate_checkpoint_coordinate_workload(
    checkpoint: &CheckpointRef,
) -> Result<()> {
    require(
        checkpoint.workload_structure_state.is_none()
            || checkpoint.workload_id.is_some(),
        "Checkpoint coordinates require a workload ID",
    )
}

/// Rule 22: A checkpoint without coordinates has no excluded
/// positions.
pub fn validate_checkpoint_na_presence(
    checkpoint: &CheckpointRef,
) -> Result<()> {
    require(
        checkpoint.workload_structure_state.is_some()
            || checkpoint.na_indices.is_empty(),
        "NA indices require checkpoint coordinates",
    )
}

/// Rule 23: The originating workload of a checkpoint is training.
pub fn validate_checkpoint_origin_type(
    workload: &Workload,
) -> Result<()> {
    require(
        workload.workload_type == WorkloadType::Training,
        "Checkpoint origin must be a training workload",
    )
}

/// Rule 24: A generated checkpoint matches its originating
/// component.
pub fn validate_checkpoint_origin_spec(
    checkpoint: &CheckpointRef,
    graph: &WorkloadGraph,
) -> Result<()> {
    let WorkloadGraph::TrainingGraph(graph) = graph else {
        return Err(invalid("Checkpoint origin is not training"));
    };
    let TrainingMethod::PPO(params) =
        &graph.training_job_controller.training_method;
    match &checkpoint.model_type {
        CheckpointModelType::Policy => {
            validate_checkpoint_model_spec(
                checkpoint,
                &params.control_model.model,
            )
        }
        CheckpointModelType::Value => {
            validate_checkpoint_model_spec(
                checkpoint,
                &params.value_model.model,
            )
        }
        CheckpointModelType::FlowNoise => {
            let ControlOutputMode::FlowMatchingAction(
                FlowMatchingAction::FlowNoise(noise),
            ) = &params.control_output_mode
            else {
                return Err(invalid(
                    "Origin graph has no flow-noise component",
                ));
            };
            validate_checkpoint_model_spec(
                checkpoint,
                &MLModel::MLP(noise.noise_model.clone()),
            )
        }
    }
}

/// Rule 25: Retained coordinates obey their original
/// dimension bounds.
/// Call after the record-specific coordinate-count validator.
pub fn validate_coordinate_bounds(
    structure: &WorkloadStructureLut,
    retained_indices: &[u32],
    coordinate: &[u32],
    graph: &WorkloadGraph,
) -> Result<()> {
    for (&index, &value) in
        retained_indices.iter().zip(coordinate)
    {
        let dimension =
            &structure.workload_state_index_names[index as usize];
        let limit = match (graph, dimension.as_str()) {
            (WorkloadGraph::TrainingGraph(graph), name) => {
                let TrainingMethod::PPO(params) = &graph
                    .training_job_controller
                    .training_method;
                match name {
                    "rollout" => i64::from(params.num_rollouts),
                    "epoch" => i64::from(params.num_epochs),
                    "minibatch" => 1,
                    "trajectory" | "rollout_trajectory" => {
                        i64::from(params.num_trajs)
                    }
                    "trajectory_timestep"
                    | "rollout_trajectory_timestep" => {
                        i64::from(params.num_steps_per_traj)
                    }
                    _ => {
                        return Err(invalid(
                            "Unknown PPO dimension",
                        ));
                    }
                }
            }
            (WorkloadGraph::TestGraph(graph), name) => {
                match &graph.test_job_controller.test_method {
                    TestMethod::Custom(method) => match name {
                        "task_index" => {
                            validate_custom_task_index(value)?;
                            continue;
                        }
                        "trajectory" => i64::from(
                            method.test_params.num_parallel_envs,
                        ),
                        "trajectory_timestep" => i64::from(
                            method.test_params.num_timesteps,
                        ),
                        _ => {
                            return Err(invalid(
                                "Unknown custom dimension",
                            ));
                        }
                    },
                    TestMethod::Dataset(method) => {
                        let Dataset::Libero(params) =
                            &method.dataset;
                        match name {
                            "task_index" => {
                                validate_dataset_task_index(
                                    value,
                                    &params.tasks_index,
                                )?;
                                continue;
                            }
                            "trajectory" => i64::from(
                                params.num_envs_per_task,
                            ),
                            "trajectory_timestep" => {
                                i64::from(params.num_timesteps)
                            }
                            _ => {
                                return Err(invalid(
                                    "Unknown dataset dimension",
                                ));
                            }
                        }
                    }
                }
            }
        };
        require(
            i64::from(value) < limit,
            &format!("Coordinate {dimension} is out of bounds"),
        )?;
    }
    Ok(())
}

/// Rule 26: Dataset task coordinates are selected IDs, not
/// list offsets.
pub fn validate_dataset_task_index(
    value: u32,
    selected_tasks: &[i32],
) -> Result<()> {
    require(
        selected_tasks
            .iter()
            .any(|&task| i64::from(task) == i64::from(value)),
        "Task index is not selected by the workload",
    )
}

/// Rule 27: Custom testing reserves task coordinate zero.
pub fn validate_custom_task_index(value: u32) -> Result<()> {
    require(value == 0, "Custom-testing task index must be zero")
}

/// Rule 28: PPO/custom trajectory records use their graph's
/// model.
/// Dataset-generated task models have no association field yet.
pub fn validate_trajectory_model(
    graph: &WorkloadGraph,
    simulation_model_id: &str,
) -> Result<()> {
    if let Some(expected) = graph_simulation_model_id(graph) {
        require(
            expected == simulation_model_id,
            "Trajectory model does not match its graph",
        )?;
    }
    Ok(())
}

/// Rule 29: Persisted metric values are finite.
pub fn validate_metric_values(
    record: &WorkloadUserMetrics,
) -> Result<()> {
    require(
        record.metric_values.iter().all(|x| x.is_finite()),
        "Metric values must be finite",
    )
}

/// Rule 30: All supplied body and geometry numerical values
/// are finite.
pub fn validate_model_state_values(
    world: &WorldBody,
) -> Result<()> {
    for body in bodies(world) {
        for vector in [&body.pos, &body.quat, &body.euler]
            .into_iter()
            .flatten()
        {
            require(
                vector.iter().all(|x| x.is_finite()),
                "Body numerical values must be finite",
            )?;
        }
        for geom in body.geoms.iter().flatten() {
            for vector in [&geom.pos, &geom.quat, &geom.euler]
                .into_iter()
                .flatten()
            {
                require(
                    vector.iter().all(|x| x.is_finite()),
                    "Geometry numerical values must be finite",
                )?;
            }
            require(
                geom.size.iter().all(|x| x.is_finite()),
                "Geometry sizes must be finite",
            )?;
        }
    }
    Ok(())
}

/// Rule 31: Supplied body and geometry positions have three
/// components.
pub fn validate_model_state_positions(
    world: &WorldBody,
) -> Result<()> {
    validate_vector_lengths(
        world,
        3,
        |body| body.pos.as_deref(),
        |geom| geom.pos.as_deref(),
    )
}

/// Rule 32: Supplied quaternions have four components.
pub fn validate_model_state_quaternions(
    world: &WorldBody,
) -> Result<()> {
    validate_vector_lengths(
        world,
        4,
        |body| body.quat.as_deref(),
        |geom| geom.quat.as_deref(),
    )
}

/// Rule 33: Supplied Euler orientations have three components.
pub fn validate_model_state_eulers(
    world: &WorldBody,
) -> Result<()> {
    validate_vector_lengths(
        world,
        3,
        |body| body.euler.as_deref(),
        |geom| geom.euler.as_deref(),
    )
}

/// Rule 34: Supplied body IDs are unique across the whole
/// hierarchy.
pub fn validate_model_state_body_ids(
    world: &WorldBody,
) -> Result<()> {
    let mut seen = HashSet::new();
    for body in bodies(world) {
        if let Some(id) = body.body_id {
            require(
                seen.insert(id),
                "Duplicate body ID in model state",
            )?;
        }
    }
    Ok(())
}

/// Rule 35: A modification timestamp cannot precede creation.
pub fn validate_modified_at(
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
) -> Result<()> {
    require(
        updated_at >= created_at,
        "Modification timestamp precedes creation",
    )
}

/// Rule 36: A supplied workload launch cannot precede creation.
pub fn validate_launched_at(
    created_at: DateTime<Utc>,
    launched_at: Option<DateTime<Utc>>,
) -> Result<()> {
    require(
        launched_at.is_none_or(|time| time >= created_at),
        "Launch timestamp precedes workload creation",
    )
}

/// Rule 37: Completion cannot precede launch when both are
/// supplied.
pub fn validate_completed_at(
    launched_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
) -> Result<()> {
    require(
        launched_at.zip(completed_at).is_none_or(
            |(launch, completion)| completion >= launch,
        ),
        "Completion timestamp precedes workload launch",
    )
}

/// Apply graph, checkpoint-reference and workload timestamp
/// rules.
pub async fn validate_workload(
    postgres: &PostgresOrm,
    record: &Workload,
) -> Result<()> {
    validate_graph_type(
        &record.workload_graph,
        &record.workload_type,
    )?;
    validate_workload_timestamps(record)?;
    validate_graph_simulation_model(
        postgres,
        &record.workload_graph,
    )
    .await?;
    validate_graph_checkpoints(postgres, &record.workload_graph)
        .await
}

/// Apply the three independent workload timestamp validators.
pub fn validate_workload_timestamps(
    record: &Workload,
) -> Result<()> {
    validate_modified_at(record.created_at, record.updated_at)?;
    validate_launched_at(record.created_at, record.launched_at)?;
    validate_completed_at(record.launched_at, record.completed_at)
}

/// Validate every checkpoint request in a graph.
pub async fn validate_graph_checkpoints(
    postgres: &PostgresOrm,
    graph: &WorkloadGraph,
) -> Result<()> {
    match graph {
        WorkloadGraph::TrainingGraph(graph) => {
            let TrainingMethod::PPO(params) =
                &graph.training_job_controller.training_method;
            validate_checkpoint_request(
                postgres,
                &params.control_model.model_type,
                &CheckpointModelType::Policy,
                &params.control_model.model,
            )
            .await?;
            validate_checkpoint_request(
                postgres,
                &params.value_model.model_type,
                &CheckpointModelType::Value,
                &params.value_model.model,
            )
            .await?;
            if let ControlOutputMode::FlowMatchingAction(
                FlowMatchingAction::FlowNoise(noise),
            ) = &params.control_output_mode
            {
                validate_checkpoint_request(
                    postgres,
                    &noise.model_type,
                    &CheckpointModelType::FlowNoise,
                    &MLModel::MLP(noise.noise_model.clone()),
                )
                .await?;
            }
        }
        WorkloadGraph::TestGraph(graph) => {
            let control =
                match &graph.test_job_controller.test_method {
                    TestMethod::Custom(method) => {
                        &method.control_model
                    }
                    TestMethod::Dataset(method) => {
                        &method.control_model
                    }
                };
            validate_checkpoint_request(
                postgres,
                &control.model_type,
                &CheckpointModelType::Policy,
                &control.model,
            )
            .await?;
        }
    }
    Ok(())
}

/// Validate one supplied checkpoint before any SQL writes.
pub async fn validate_checkpoint_ref(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    record: &CheckpointRef,
) -> Result<()> {
    validate_modified_at(record.created_at, record.updated_at)?;
    validate_checkpoint_structure_pair(record)?;
    validate_checkpoint_coordinate_workload(record)?;
    validate_checkpoint_na_presence(record)?;
    validate_checkpoint_na_uniqueness(record)?;
    if let Some(id) = &record.workload_id {
        let workload =
            validate_workload_exists(postgres, mongo, id).await?;
        validate_checkpoint_origin_type(&workload)?;
        validate_checkpoint_origin_spec(
            record,
            &workload.workload_graph,
        )?;
        if let (Some(id), Some(state)) = (
            record.workload_structure_id,
            record.workload_structure_state.as_deref(),
        ) {
            let structure =
                validate_structure_exists(postgres, id).await?;
            validate_structure_workload_type(
                &structure, &workload,
            )?;
            validate_structure_method(
                &structure,
                &workload.workload_graph,
            )?;
            validate_structure_stage(
                &structure,
                &[
                    "rollout",
                    "epoch",
                    "minibatch",
                    "trajectory",
                    "trajectory_timestep",
                ],
            )?;
            validate_checkpoint_na_bounds(record, &structure)?;
            let retained = retained_indices(
                &structure,
                &[],
                &record.na_indices,
            );
            validate_checkpoint_coordinate_count(
                state, &retained,
            )?;
            validate_coordinate_bounds(
                &structure,
                &retained,
                state,
                &workload.workload_graph,
            )?;
        }
    }
    Ok(())
}

/// Validate frame stage, model association and coordinate bounds.
/// SQL foreign keys validate the existence of its asset
/// reference.
pub async fn validate_camera_frame(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    record: &TrajectoryCameraFrameRef,
) -> Result<()> {
    let workload = validate_workload_exists(
        postgres,
        mongo,
        &record.workload_id,
    )
    .await?;
    let structure = validate_structure_exists(
        postgres,
        record.workload_structure_id,
    )
    .await?;
    validate_trajectory_record(
        &structure,
        &workload,
        &record.workload_structure_state,
        &record.simulation_model_id,
    )
}

/// Validate an entire log batch before its first write.
pub async fn validate_logs(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    records: &[WorkloadLogs],
) -> Result<()> {
    for record in records {
        validate_workload_exists(
            postgres,
            mongo,
            &record.workload_id,
        )
        .await?;
    }
    Ok(())
}

/// Validate an entire metric batch before its first write.
pub async fn validate_metrics(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    records: &[WorkloadUserMetrics],
) -> Result<()> {
    for record in records {
        validate_metric_array_lengths(record)?;
        validate_metric_values(record)?;
        let workload = validate_workload_exists(
            postgres,
            mongo,
            &record.workload_id,
        )
        .await?;
        let metric =
            validate_metric_exists(postgres, record.metric_id)
                .await?;
        let structure = validate_structure_exists(
            postgres,
            metric.workload_structure_id,
        )
        .await?;
        validate_structure_workload_type(&structure, &workload)?;
        validate_structure_method(
            &structure,
            &workload.workload_graph,
        )?;
        let retained = retained_indices(
            &structure,
            &metric.aggregation_indices,
            &metric.na_indices,
        );
        validate_metric_coordinate_count(record, &retained)?;
        for state in &record.workload_structure_state_values {
            validate_coordinate_bounds(
                &structure,
                &retained,
                state,
                &workload.workload_graph,
            )?;
        }
    }
    Ok(())
}

/// Validate an entire pose batch before its first write.
pub async fn validate_model_states(
    postgres: &PostgresOrm,
    mongo: &MongoOrm,
    records: &[WorkloadSimulationModelTrajectoryState],
) -> Result<()> {
    for record in records {
        validate_model_state_values(&record.state)?;
        validate_model_state_positions(&record.state)?;
        validate_model_state_quaternions(&record.state)?;
        validate_model_state_eulers(&record.state)?;
        validate_model_state_body_ids(&record.state)?;
        let workload = validate_workload_exists(
            postgres,
            mongo,
            &record.workload_id,
        )
        .await?;
        let structure = validate_structure_exists(
            postgres,
            record.workload_structure_id,
        )
        .await?;
        validate_pose_model_exists(
            postgres,
            &record.simulation_model_id,
        )
        .await?;
        validate_trajectory_record(
            &structure,
            &workload,
            &record.workload_structure_state,
            &record.simulation_model_id,
        )?;
    }
    Ok(())
}

/// Existing pagination rule shared by both stores.
pub fn validate_pagination(
    (limit, offset): (i64, i64),
) -> Result<()> {
    require(
        limit > 0 && offset >= 0,
        "Pagination requires limit > 0 and offset >= 0",
    )
}

/// Existing update rule: a workload keeps its graph identity.
pub fn validate_graph_id_unchanged(
    previous: &Workload,
    next: &Workload,
) -> Result<()> {
    require(
        previous.workload_graph_id == next.workload_graph_id,
        "Updating a workload cannot change its graph ID",
    )
}

/// Existing write-result rule, kept out of ORM query code.
pub fn validate_affected_rows(count: u64) -> Result<()> {
    if count == 0 {
        return Err(sqlx::Error::RowNotFound);
    }
    Ok(())
}

/// Load and check the timestamp fields for a lifecycle update.
pub async fn validate_workload_status_update(
    postgres: &PostgresOrm,
    id: &str,
    updated_at: DateTime<Utc>,
    launched_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
) -> Result<()> {
    let row = required(
        postgres.workload_row(id).await?,
        "Workload does not exist",
    )?;
    let created_at = row.try_get("created_at")?;
    validate_modified_at(created_at, updated_at)?;
    validate_launched_at(created_at, launched_at)?;
    validate_completed_at(launched_at, completed_at)
}

/// Load and check a workload's supplied modification timestamp.
pub async fn validate_workload_modified_at(
    postgres: &PostgresOrm,
    id: &str,
    updated_at: DateTime<Utc>,
) -> Result<()> {
    let row = required(
        postgres.workload_row(id).await?,
        "Workload does not exist",
    )?;
    validate_modified_at(row.try_get("created_at")?, updated_at)
}

fn validate_trajectory_record(
    structure: &WorkloadStructureLut,
    workload: &Workload,
    state: &[u32],
    model_id: &str,
) -> Result<()> {
    validate_structure_workload_type(structure, workload)?;
    validate_structure_method(
        structure,
        &workload.workload_graph,
    )?;
    let expected: &[&str] = match &workload.workload_graph {
        WorkloadGraph::TrainingGraph(_) => &[
            "rollout",
            "rollout_trajectory",
            "rollout_trajectory_timestep",
        ],
        WorkloadGraph::TestGraph(_) => {
            &["task_index", "trajectory", "trajectory_timestep"]
        }
    };
    validate_structure_stage(structure, expected)?;
    validate_trajectory_coordinate_count(structure, state)?;
    validate_coordinate_bounds(
        structure,
        &retained_indices(structure, &[], &[]),
        state,
        &workload.workload_graph,
    )?;
    validate_trajectory_model(&workload.workload_graph, model_id)
}

async fn validate_checkpoint_request(
    postgres: &PostgresOrm,
    request: &ModelType,
    role: &CheckpointModelType,
    model: &MLModel,
) -> Result<()> {
    if let ModelType::Checkpoint(CheckpointType::CheckpointID(
        id,
    )) = request
    {
        let checkpoint =
            validate_requested_checkpoint_exists(postgres, id)
                .await?;
        validate_checkpoint_role(&checkpoint, role)?;
        validate_checkpoint_model_spec(&checkpoint, model)?;
    }
    Ok(())
}

fn graph_simulation_model_id(
    graph: &WorkloadGraph,
) -> Option<&str> {
    match graph {
        WorkloadGraph::TrainingGraph(graph) => {
            let TrainingMethod::PPO(params) =
                &graph.training_job_controller.training_method;
            Some(&params.simulation_model_id)
        }
        WorkloadGraph::TestGraph(graph) => {
            match &graph.test_job_controller.test_method {
                TestMethod::Custom(method) => {
                    Some(&method.simulation_model_id)
                }
                TestMethod::Dataset(_) => None,
            }
        }
    }
}

fn retained_indices(
    structure: &WorkloadStructureLut,
    aggregation: &[u32],
    na: &[u32],
) -> Vec<u32> {
    (0..structure.number_of_state_indices)
        .filter(|i| !aggregation.contains(i) && !na.contains(i))
        .collect()
}

fn bodies(world: &WorldBody) -> impl Iterator<Item = &Body> {
    let mut stack: Vec<_> = world.bodies.iter().collect();
    std::iter::from_fn(move || {
        let body = stack.pop()?;
        stack.extend(body.bodies.iter().flatten());
        Some(body)
    })
}

fn validate_vector_lengths(
    world: &WorldBody,
    length: usize,
    body_field: fn(&Body) -> Option<&[f32]>,
    geom_field: fn(&Geom) -> Option<&[f32]>,
) -> Result<()> {
    for body in bodies(world) {
        require(
            body_field(body).is_none_or(|v| v.len() == length),
            "Invalid body vector length",
        )?;
        for geom in body.geoms.iter().flatten() {
            require(
                geom_field(geom)
                    .is_none_or(|v| v.len() == length),
                "Invalid geometry vector length",
            )?;
        }
    }
    Ok(())
}

fn validate_metric_array_lengths(
    record: &WorkloadUserMetrics,
) -> Result<()> {
    require(
        record.workload_structure_state_values.len()
            == record.metric_values.len(),
        "Metric coordinate/value array lengths do not match",
    )
}

fn required<T>(value: Option<T>, message: &str) -> Result<T> {
    value.ok_or_else(|| invalid(message))
}

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(invalid(message));
    }
    Ok(())
}

fn invalid(message: &str) -> sqlx::Error {
    sqlx::Error::Protocol(message.into())
}
