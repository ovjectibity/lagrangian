//! Project-scoped simulation models and temporary downloads.

use std::io::Cursor;
use std::sync::Arc;

use rocket::State;
use rocket::http::{ContentType, Status};
use rocket::response::stream::ReaderStream;
use rocket::serde::json::Json;
use uuid::Uuid;
use zip::ZipArchive;

use super::{
    ApiResult, ApiState, Authenticated, internal, sql_error,
};
use crate::commondef::api::apimessages::*;
use crate::commondef::records::assets::SimulationModelRef;
use crate::workflows;

/// List model metadata belonging to an authorized project.
#[utoipa::path(get, path = "/api/list_models",
    params(("project_id" = String, Query, description = "Request filter")),
    responses((status = 200, body = ListModelsResponseBody)))]
#[get("/list_models?<project_id>")]
pub async fn list_models(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    project_id: String,
) -> ApiResult<Json<ListModelsResponseBody>> {
    state.authorize_project(&auth, &project_id).await?;
    let mut models = Vec::new();
    let mut offset = 0;
    loop {
        let page = state
            .postgres
            .list_simulation_model_refs(
                &project_id,
                (500, offset),
            )
            .await
            .map_err(sql_error)?;
        let count = page.len();
        models.extend(page.into_iter().map(|r| {
            SimulationModelRefUser {
                project_id: r.project_id,
                simulation_model_id: r.simulation_model_id,
                name: r.name,
                model_type: r.model_type,
                file_type: r.file_type,
                base_file_name: r.base_file_name,
            }
        }));
        if count < 500 {
            break;
        }
        offset += count as i64;
    }
    Ok(Json(ListModelsResponseBody { model_data: models }))
}

/// Resolve model IDs to authorized temporary app download links.
#[utoipa::path(get, path = "/api/get_models",
    params(("model_ids" = Vec<String>, Query, description = "Request filter")),
    responses((status = 200, body = GetModelsResponseBody)))]
#[get("/get_models?<model_ids>")]
pub async fn get_models(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    model_ids: Vec<String>,
) -> ApiResult<Json<GetModelsResponseBody>> {
    let mut models = Vec::new();
    for id in model_ids {
        let model = state.authorize_model(&auth, &id).await?;
        models.push(model_response(state, &auth, model).await?);
    }
    Ok(Json(GetModelsResponseBody { model_refs: models }))
}

/// Upload a ZIP, then store its project-owned reference.
#[utoipa::path(post, path = "/api/save_model",
    request_body = SaveModelRequestBody,
    responses((status = 200, body = SaveModelResponseBody)))]
#[post("/save_model", format = "json", data = "<body>")]
pub async fn save_model(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    body: Json<SaveModelRequestBody>,
) -> ApiResult<Json<SaveModelResponseBody>> {
    state.authorize_project(&auth, &body.project_id).await?;
    let body = body.into_inner();
    let body = tokio::task::spawn_blocking(move || {
        validate_model_zip(&body)?;
        Ok::<_, Status>(body)
    })
    .await
    .map_err(internal)??;
    let id = Uuid::new_v4().to_string();
    let uri = workflows::upload_model(
        &state.s3,
        &state.bucket,
        &body.project_id,
        &id,
        body.file_contents,
    )
    .await
    .map_err(internal)?;
    let now = chrono::Utc::now();
    state
        .postgres
        .create_simulation_model_ref(&SimulationModelRef {
            project_id: body.project_id,
            simulation_model_id: id.clone(),
            created_at: now,
            updated_at: now,
            uri_path: uri,
            name: body.name,
            base_file_name: body.base_file_name,
            model_type: body.model_type,
            file_type: body.file_type,
        })
        .await
        .map_err(sql_error)?;
    Ok(Json(SaveModelResponseBody { model_id: id }))
}

/// Reject deletion of models referenced by any workload or frame.
#[utoipa::path(delete, path = "/api/delete_model",
    request_body = DeleteModelsRequestBody,
    responses((status = 204), (status = 409)))]
#[delete("/delete_model", format = "json", data = "<body>")]
pub async fn delete_model(
    auth: Authenticated,
    state: &State<Arc<ApiState>>,
    body: Json<DeleteModelsRequestBody>,
) -> ApiResult<Status> {
    let _delivery = state.delivery.lock().await;
    for id in &body.model_ids {
        state.authorize_model(&auth, id).await?;
        crate::db::validations::validate_model_unreferenced(
            &state.postgres,
            &state.mongo,
            id,
        )
        .await
        .map_err(|_| Status::Conflict)?;
    }
    let mut transaction =
        state.postgres.pool().begin().await.map_err(sql_error)?;
    for id in &body.model_ids {
        sqlx::query("DELETE FROM assets.simulation_model_ref WHERE simulation_model_id = $1")
            .bind(id).execute(&mut *transaction)
            .await.map_err(sql_error)?;
    }
    transaction.commit().await.map_err(sql_error)?;
    Ok(Status::NoContent)
}

/// Stream an authorized object through the app, never redirecting
/// to a bucket URL. Download tickets expire after five minutes.
#[utoipa::path(get, path = "/api/assets/download/{token}",
    params(("token" = String, Path, description = "Download ticket")),
    responses((status = 200, body = Vec<u8>), (status = 401)))]
#[get("/assets/download/<token>")]
pub async fn download_asset(
    state: &State<Arc<ApiState>>,
    token: String,
) -> ApiResult<
    impl for<'r> rocket::response::Responder<'r, 'r> + use<>,
> {
    let (_, key_id, key) = {
        let mut links = state.downloads.lock().await;
        let now = std::time::Instant::now();
        links.retain(|_, (expiry, _, _)| *expiry > now);
        links.get(&token).cloned().ok_or(Status::Unauthorized)?
    };
    state.authenticate_ticket(&key_id).await?;
    let response = state
        .s3
        .get_object()
        .bucket(&state.bucket)
        .key(&key)
        .send()
        .await
        .map_err(internal)?;
    let content_type = if key.ends_with(".png") {
        ContentType::PNG
    } else {
        ContentType::ZIP
    };
    Ok((
        content_type,
        ReaderStream::one(response.body.into_async_read()),
    ))
}

pub(super) async fn model_response(
    state: &ApiState,
    auth: &Authenticated,
    model: SimulationModelRef,
) -> ApiResult<SimulationModelDataUser> {
    Ok(SimulationModelDataUser {
        uri_path: state
            .download_link(auth, &model.uri_path)
            .await?,
        project_id: model.project_id,
        simulation_model_id: model.simulation_model_id,
        name: model.name,
        model_type: model.model_type,
        file_type: model.file_type,
        base_file_name: model.base_file_name,
    })
}

fn validate_model_zip(
    body: &SaveModelRequestBody,
) -> ApiResult<()> {
    let mut archive =
        ZipArchive::new(Cursor::new(&body.file_contents))
            .map_err(|_| Status::BadRequest)?;
    let mut found = false;
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .map_err(|_| Status::BadRequest)?;
        if entry.enclosed_name().is_none() {
            return Err(Status::BadRequest);
        }
        found |= entry.name() == body.base_file_name
            && entry.is_file();
    }
    if found {
        Ok(())
    } else {
        Err(Status::BadRequest)
    }
}
