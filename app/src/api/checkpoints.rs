//! Workload checkpoints and globally available pretrained references.

use super::{ApiResult, ApiState, Authenticated, sql_error};
use crate::commondef::api::apimessages::*;
use crate::commondef::records::assets::CheckpointRef;
use rocket::State;
use rocket::serde::json::Json;
use std::sync::Arc;

/// List project checkpoints and globally shared predefined ones.
#[utoipa::path(get, path = "/api/list_checkpoints",
    params(("project_id" = String, Query, description = "Request filter")),
    responses((status = 200, body = ListCheckpointsResponseBody)))]
#[get("/list_checkpoints?<project_id>")]
pub async fn list_checkpoints(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    project_id: String,
) -> ApiResult<Json<ListCheckpointsResponseBody>> {
    state.authorize_project(&auth, &project_id).await?;
    let mut checkpoints = Vec::new();
    let mut offset = 0;
    loop {
        let page = state
            .postgres
            .list_project_checkpoint_refs(
                &project_id,
                (500, offset),
            )
            .await
            .map_err(sql_error)?;
        let count = page.len();
        for record in page {
            checkpoints
                .push(checkpoint_response(state, record).await?);
        }
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    Ok(Json(ListCheckpointsResponseBody { checkpoints }))
}

/// Check every workload's project before returning checkpoints.
#[utoipa::path(get, path = "/api/get_workload_checkpoints",
    params(("workload_ids" = Vec<String>, Query, description = "Request filter")),
    responses((status = 200, body = GetWorkloadCheckpointsResponseBody)))]
#[get("/get_workload_checkpoints?<workload_ids>")]
pub async fn get_workload_checkpoints(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    workload_ids: Vec<String>,
) -> ApiResult<Json<GetWorkloadCheckpointsResponseBody>> {
    let mut checkpoints = Vec::new();
    for id in workload_ids {
        state.authorize_workload(&auth, &id).await?;
        let mut offset = 0;
        loop {
            let page = state
                .postgres
                .list_checkpoint_refs(
                    Some(&id),
                    None,
                    (500, offset),
                )
                .await
                .map_err(sql_error)?;
            let count = page.len();
            for record in page {
                checkpoints.push(
                    checkpoint_response(state, record).await?,
                );
            }
            if count < 500 {
                break;
            }
            offset += count as i64;
        }
    }
    Ok(Json(GetWorkloadCheckpointsResponseBody { checkpoints }))
}

pub(super) async fn checkpoint_response(
    state: &ApiState,
    record: CheckpointRef,
) -> ApiResult<CheckpointRefUser> {
    let names = if let Some(id) = record.workload_structure_id {
        state
            .postgres
            .get_workload_structure_lut(id)
            .await
            .map_err(sql_error)?
            .map(|structure| {
                structure
                    .workload_state_index_names
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        !record.na_indices.contains(&(*i as u32))
                    })
                    .map(|(_, name)| name)
                    .collect()
            })
    } else {
        None
    };
    Ok(CheckpointRefUser {
        checkpoint_id: record.checkpoint_id,
        name: record.name,
        workload_id: record.workload_id,
        workload_structure_state: record.workload_structure_state,
        na_indices: record.na_indices,
        model_type: record.model_type,
        model_spec: record.model_spec,
        workload_structure_index_names: names,
    })
}
