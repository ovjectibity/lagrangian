//! Authenticated API, shared state and asset download tickets.

pub mod checkpoints;
pub mod models;
pub mod reward_validation;
pub mod testing;
pub mod training;
pub mod workloads;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use aws_sdk_s3::Client as S3Client;
use rocket::Request;
use rocket::http::Status;
use rocket::request::{FromRequest, Outcome};
use rocket::serde::json::Json;
use tokio::sync::{Mutex, OwnedMutexGuard, broadcast};
use tokio::task::JoinHandle;
use utoipa::OpenApi;
use uuid::Uuid;

use crate::commondef::api::apimessages::*;
use crate::commondef::records::assets::*;
use crate::commondef::records::model::*;
use crate::commondef::records::product::User;
use crate::commondef::records::workload::*;
use crate::commondef::records::workload_graph::*;
use crate::commondef::remoteworker::remoteworkermessages::RemoteWorkerClientMessage;
use crate::db::db_orm::PostgresOrm;
use crate::db::mongo_orm::MongoOrm;
use crate::db::{self, validations};

pub type ApiResult<T> = Result<T, Status>;

/// Identity obtained exclusively from an active access key.
#[derive(Clone)]
pub struct Authenticated {
    pub user: User,
    pub key_id: String,
}

#[rocket::async_trait]
impl<'r> FromRequest<'r> for Authenticated {
    type Error = ();

    async fn from_request(
        request: &'r Request<'_>,
    ) -> Outcome<Self, Self::Error> {
        let Some(state) =
            request.rocket().state::<Arc<ApiState>>()
        else {
            return Outcome::Error((
                Status::ServiceUnavailable,
                (),
            ));
        };
        let Some(value) = request
            .headers()
            .get_one("Authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
        else {
            return Outcome::Error((Status::Unauthorized, ()));
        };
        match state.authenticate(value).await {
            Ok(auth) => Outcome::Success(auth),
            Err(status) => Outcome::Error((status, ())),
        }
    }
}

/// Dependencies and transient state; durable data lives in the DB.
pub struct ApiState {
    pub postgres: Arc<PostgresOrm>,
    pub mongo: Arc<MongoOrm>,
    pub s3: S3Client,
    pub bucket: String,
    pub jobs: Mutex<
        HashMap<String, crate::workflows::WorkloadTaskState>,
    >,
    pub events:
        broadcast::Sender<(u64, RemoteWorkerClientMessage)>,
    /// Catalog mutations are short; workload data has separate locks.
    pub catalog: Mutex<()>,
    workload_locks: Mutex<HashMap<String, Weak<Mutex<()>>>>,
    revisions: Mutex<HashMap<String, u64>>,
    pub instance_id: String,
    pub shutting_down: AtomicBool,
    pub cleanup_scopes: Mutex<HashSet<String>>,
    pub assets_inflight: Mutex<HashMap<String, Weak<()>>>,
    pub maintenance: Mutex<Option<JoinHandle<()>>>,
    tickets: Mutex<HashMap<String, (Instant, String, String)>>,
    downloads: Mutex<HashMap<String, (Instant, String, String)>>,
}

impl ApiState {
    /// Construct the API dependencies without contacting S3.
    pub fn new(
        postgres: Arc<PostgresOrm>,
        mongo: Arc<MongoOrm>,
        s3: S3Client,
        bucket: String,
    ) -> Self {
        let (events, _) = broadcast::channel(64);
        Self {
            postgres,
            mongo,
            s3,
            bucket,
            events,
            jobs: Mutex::new(HashMap::new()),
            catalog: Mutex::new(()),
            workload_locks: Mutex::new(HashMap::new()),
            revisions: Mutex::new(HashMap::new()),
            instance_id: Uuid::new_v4().to_string(),
            shutting_down: AtomicBool::new(false),
            cleanup_scopes: Mutex::new(HashSet::new()),
            assets_inflight: Mutex::new(HashMap::new()),
            maintenance: Mutex::new(None),
            tickets: Mutex::new(HashMap::new()),
            downloads: Mutex::new(HashMap::new()),
        }
    }

    /// Only writers and snapshots of the same workload serialize.
    pub async fn lock_workload(
        &self,
        id: &str,
    ) -> OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.workload_locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() != 0);
            if let Some(lock) =
                locks.get(id).and_then(Weak::upgrade)
            {
                lock
            } else {
                let lock = Arc::new(Mutex::new(()));
                locks
                    .insert(id.to_owned(), Arc::downgrade(&lock));
                lock
            }
        };
        lock.lock_owned().await
    }

    /// Revision checks reject HTTP cursors invalidated by new data.
    pub async fn revision(&self, id: &str) -> u64 {
        self.revisions.lock().await.get(id).copied().unwrap_or(0)
    }

    /// Called while holding the workload lock, after durable writes.
    pub async fn publish(
        &self,
        message: RemoteWorkerClientMessage,
    ) {
        use RemoteWorkerClientMessage::*;
        let id =
            match &message {
                TrainingUpdatesMessage(m)
                | TestUpdatesMessage(m) => &m.workload_id,
                StopTrainingMessage(m)
                | StopTestingMessage(m) => &m.workload_id,
            };
        let revision = {
            let mut revisions = self.revisions.lock().await;
            let revision =
                revisions.entry(id.clone()).or_default();
            *revision += 1;
            *revision
        };
        let _ = self.events.send((revision, message));
    }

    /// Leases disappear on cancellation as well as ordinary returns.
    pub async fn lease_asset(&self, uri: &str) -> Arc<()> {
        let lease = Arc::new(());
        self.assets_inflight
            .lock()
            .await
            .insert(uri.to_owned(), Arc::downgrade(&lease));
        lease
    }

    pub fn accepting_workloads(&self) -> bool {
        !self.shutting_down.load(Ordering::SeqCst)
    }

    async fn authenticate(
        &self,
        value: &str,
    ) -> ApiResult<Authenticated> {
        let key = self
            .postgres
            .get_access_key_by_value(value)
            .await
            .map_err(sql_error)?
            .ok_or(Status::Unauthorized)?;
        validations::validate_active_access_key(&key)
            .map_err(|_| Status::Unauthorized)?;
        let user = self
            .postgres
            .get_user(&key.user_id)
            .await
            .map_err(sql_error)?
            .ok_or(Status::Unauthorized)?;
        Ok(Authenticated {
            user,
            key_id: key.access_key_id,
        })
    }

    pub(super) async fn authenticate_ticket(
        &self,
        key_id: &str,
    ) -> ApiResult<Authenticated> {
        let key = self
            .postgres
            .get_access_key(key_id)
            .await
            .map_err(sql_error)?
            .ok_or(Status::Unauthorized)?;
        self.authenticate(&key.access_key_value).await
    }

    /// Authorize a project before loading dependent resources.
    pub async fn authorize_project(
        &self,
        auth: &Authenticated,
        id: &str,
    ) -> ApiResult<()> {
        let project = self
            .postgres
            .get_project(id)
            .await
            .map_err(sql_error)?
            .ok_or(Status::NotFound)?;
        validations::validate_project_owner(
            &project.user_id,
            &auth.user.user_id,
        )
        .map_err(|_| Status::Forbidden)
    }

    /// Resolve and authorize a workload's owning project.
    pub async fn authorize_workload(
        &self,
        auth: &Authenticated,
        id: &str,
    ) -> ApiResult<Workload> {
        let record =
            db::get_workload(&self.postgres, &self.mongo, id)
                .await
                .map_err(internal)?
                .ok_or(Status::NotFound)?;
        self.authorize_project(auth, &record.project_id).await?;
        Ok(record)
    }

    /// Resolve and authorize a model's owning project.
    pub async fn authorize_model(
        &self,
        auth: &Authenticated,
        id: &str,
    ) -> ApiResult<SimulationModelRef> {
        let record = self
            .postgres
            .get_simulation_model_ref(id)
            .await
            .map_err(sql_error)?
            .ok_or(Status::NotFound)?;
        self.authorize_project(auth, &record.project_id).await?;
        Ok(record)
    }

    /// Predefined checkpoints are shared across all users.
    pub async fn authorize_checkpoint(
        &self,
        auth: &Authenticated,
        id: &str,
    ) -> ApiResult<CheckpointRef> {
        let record = self
            .postgres
            .get_checkpoint_ref(id)
            .await
            .map_err(sql_error)?
            .ok_or(Status::NotFound)?;
        if let Some(id) = &record.workload_id {
            self.authorize_workload(auth, id).await?;
        }
        Ok(record)
    }

    pub(super) async fn download_link(
        &self,
        auth: &Authenticated,
        uri: &str,
    ) -> ApiResult<String> {
        let key = self.object_key(uri)?.to_owned();
        let token = Uuid::new_v4().to_string();
        let now = Instant::now();
        let mut links = self.downloads.lock().await;
        links.insert(
            token.clone(),
            (
                now + Duration::from_secs(300),
                auth.key_id.clone(),
                key,
            ),
        );
        Ok(format!("/api/assets/download/{token}"))
    }

    pub async fn sweep_ephemeral(&self) {
        let now = Instant::now();
        self.downloads
            .lock()
            .await
            .retain(|_, (expiry, _, _)| *expiry > now);
        self.tickets
            .lock()
            .await
            .retain(|_, (expiry, _, _)| *expiry > now);
        self.assets_inflight
            .lock()
            .await
            .retain(|_, lease| lease.strong_count() != 0);
    }

    pub(crate) fn object_key<'a>(
        &self,
        uri: &'a str,
    ) -> ApiResult<&'a str> {
        uri.strip_prefix(&format!("s3://{}/", self.bucket))
            .ok_or(Status::InternalServerError)
    }
}

/// Map SQL constraints and validation failures to HTTP errors.
pub(super) fn sql_error(error: sqlx::Error) -> Status {
    match &error {
        sqlx::Error::Database(e)
            if e.code().is_some_and(|code| {
                matches!(code.as_ref(), "23503" | "23505")
            }) =>
        {
            Status::Conflict
        }
        sqlx::Error::Protocol(_) => Status::BadRequest,
        _ => internal(error),
    }
}

pub(super) fn internal(error: impl std::fmt::Display) -> Status {
    eprintln!("API operation failed: {error}");
    Status::InternalServerError
}

/// The application API routes, shared with coverage checks.
pub fn routes() -> Vec<rocket::Route> {
    routes![
        workloads::create_project,
        workloads::list_projects,
        workloads::delete_project,
        workloads::list_workloads,
        workloads::stop_workload,
        workloads::get_workload_updates,
        workloads::live_ticket,
        workloads::live_updates,
        models::list_models,
        models::get_models,
        models::save_model,
        models::delete_model,
        models::download_asset,
        checkpoints::list_checkpoints,
        checkpoints::get_workload_checkpoints,
        training::start_training,
        testing::start_test,
    ]
}

#[derive(OpenApi)]
#[openapi(
    paths(
        workloads::create_project,
        workloads::list_projects,
        workloads::delete_project,
        workloads::list_workloads,
        workloads::stop_workload,
        workloads::get_workload_updates,
        workloads::live_ticket,
        workloads::live_updates,
        models::list_models,
        models::get_models,
        models::save_model,
        models::delete_model,
        models::download_asset,
        checkpoints::list_checkpoints,
        checkpoints::get_workload_checkpoints,
        training::start_training,
        testing::start_test,
    ),
    components(schemas(
        CreateProjectRequestBody,
        WorkloadLiveTicketResponseBody,
        DeleteProjectsRequestBody,
        CreateProjectResponseBody,
        GetProjectsResponseBody,
        SaveModelRequestBody,
        DeleteModelsRequestBody,
        ListModelsResponseBody,
        GetModelsResponseBody,
        SaveModelResponseBody,
        ListCheckpointsResponseBody,
        GetWorkloadCheckpointsResponseBody,
        ListWorkloadsResponseBody,
        WorkloadUser,
        StartTrainingRequestBody,
        StartTestRequestBody,
        StartTrainingResponseBody,
        StartTestResponseBody,
        StopWorkloadRequestBody,
        WorkloadUpdatesResponseBody,
        WorkloadUpdatesServer,
        WorkloadUpdatesClient,
        WorkloadStopped,
        WorkloadStoppedReasonUser,
        WorkloadUpdateUser,
        WorkloadLogsUser,
        WorkloadUserMetricsUser,
        WorkloadSimulationModelTrajectoryStateUser,
        TrajectoryCameraFrameRefUser,
        CheckpointRefUser,
        SimulationModelRefUser,
        SimulationModelDataUser,
        WorkloadGraph,
        CustomTestMethodTestParams,
        LiberoDatasetType,
        LiberoDatasetParams,
        Dataset,
        DatasetMethod,
        CustomTestMethod,
        TestMethod,
        TestJobController,
        TestGraph,
        IOStateXOrientationType,
        IOStateXFrame,
        XPoseSpec,
        OutputStateItem,
        InputStateItemControllable,
        InputStateItemSensors,
        OutputState,
        InputStateControllable,
        InputState,
        StandardInputSpec,
        ProprioceptiveOutputSpec,
        EEFOutputSpec,
        PolicyOutputSpec,
        PolicyInputSpec,
        Pretrained,
        CheckpointType,
        ModelType,
        MLPModelSpec,
        MLModel,
        ControlModel,
        ValueModel,
        TrainingMethod,
        FlowMatchingAction,
        FlowNoiseConfig,
        ControlOutputMode,
        PPOTrainingMethod,
        RewardLanguage,
        RewardSystem,
        TrainingAccelerationMode,
        ComputeInstance,
        TrainingJobController,
        TrainingGraph,
        CheckpointFrequency,
        SimulationModelType,
        SimulationModelFileType,
        CheckpointModelType,
        WorkloadType,
        LaunchOrigin,
        WorkloadStatus,
        LogType,
        GeomType,
        Geom,
        Body,
        WorldBody,
    ))
)]
pub struct ApiDoc;

/// Serve the schema generated from the actual API handlers.
#[get("/openapi.json")]
pub fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(openapi_document())
}

/// Build the same document for HTTP serving and offline export.
pub fn openapi_document() -> utoipa::openapi::OpenApi {
    use utoipa::openapi::{RefOr, Schema};
    use utoipa::openapi::security::{
        Http, HttpAuthScheme, SecurityRequirement, SecurityScheme,
    };
    let mut document = ApiDoc::openapi();
    if let Some(components) = document.components.as_mut() {
        components.add_security_scheme(
            "access_key",
            SecurityScheme::Http(Http::new(
                HttpAuthScheme::Bearer,
            )),
        );
    }
    for (path, item) in &mut document.paths.paths {
        if path == "/api/assets/download/{token}"
            || path == "/api/get_workload_live_updates"
        {
            continue;
        }
        for operation in item.operations.values_mut() {
            operation.security =
                Some(vec![SecurityRequirement::new(
                    "access_key",
                    Vec::<String>::new(),
                )]);
            // Utoipa 4 path parameters cannot declare defaults.
            if path == "/api/get_workload_updates"
                && let Some(parameters) =
                    operation.parameters.as_mut()
            {
                for parameter in parameters {
                    if parameter.name == "limit"
                        && let Some(RefOr::T(
                            Schema::Object(schema),
                        )) = parameter.schema.as_mut()
                    {
                        schema.default =
                            Some(serde_json::json!(200));
                    }
                }
            }
        }
    }
    document
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::{Value, json};

    use super::*;

    fn document() -> Value {
        serde_json::to_value(openapi_document()).unwrap()
    }

    fn check_refs(value: &Value, root: &Value) {
        match value {
            Value::Object(fields) => {
                if let Some(reference) =
                    fields.get("$ref").and_then(Value::as_str)
                {
                    assert!(
                        reference.starts_with("#/"),
                        "External schema reference: {reference}"
                    );
                    assert!(
                        root.pointer(&reference[1..]).is_some(),
                        "Missing schema: {reference}"
                    );
                }
                for field in fields.values() {
                    check_refs(field, root);
                }
            }
            Value::Array(values) => {
                for value in values {
                    check_refs(value, root);
                }
            }
            _ => {}
        }
    }

    fn schema_refs(value: &Value, pending: &mut Vec<String>) {
        match value {
            Value::Object(fields) => {
                if let Some(name) = fields
                    .get("$ref")
                    .and_then(Value::as_str)
                    .and_then(|r| {
                        r.strip_prefix("#/components/schemas/")
                    })
                {
                    pending.push(name.to_owned());
                }
                for field in fields.values() {
                    schema_refs(field, pending);
                }
            }
            Value::Array(values) => {
                for value in values {
                    schema_refs(value, pending);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn openapi_matches_registered_routes_and_resolves_refs() {
        let document = document();
        let registered: BTreeSet<_> = routes()
            .into_iter()
            .map(|route| {
                let path = format!("/api{}", route.uri.path())
                    .replace('<', "{")
                    .replace('>', "}");
                (path, route.method.to_string().to_lowercase())
            })
            .collect();
        let mut documented = BTreeSet::new();
        for (path, item) in document["paths"].as_object().unwrap()
        {
            for method in item.as_object().unwrap().keys() {
                documented.insert((path.clone(), method.clone()));
            }
        }
        assert_eq!(registered, documented);
        check_refs(&document, &document);
    }

    #[test]
    fn openapi_contains_only_reachable_contract_schemas() {
        let document = document();
        let schemas = document["components"]["schemas"]
            .as_object()
            .unwrap();
        // WebSocket messages are roots outside HTTP body schemas.
        let mut pending = vec![
            "WorkloadUpdatesServer".to_owned(),
            "WorkloadUpdatesClient".to_owned(),
        ];
        schema_refs(&document["paths"], &mut pending);
        let mut reachable = BTreeSet::new();
        while let Some(name) = pending.pop() {
            if reachable.insert(name.clone()) {
                schema_refs(&schemas[&name], &mut pending);
            }
        }
        let registered = schemas.keys().cloned().collect();
        assert_eq!(reachable, registered);
    }

    #[test]
    fn openapi_security_errors_and_statuses_match_handlers() {
        let document = document();
        for (path, item) in document["paths"].as_object().unwrap()
        {
            let ticket_auth = matches!(
                path.as_str(),
                "/api/assets/download/{token}"
                    | "/api/get_workload_live_updates"
            );
            for operation in item.as_object().unwrap().values() {
                if ticket_auth {
                    assert!(operation.get("security").is_none());
                } else {
                    assert_eq!(
                        operation["security"],
                        json!([{"access_key": []}])
                    );
                }
                for code in ["401", "500"] {
                    assert!(
                        operation["responses"][code].is_object(),
                        "Missing {code} response on {path}"
                    );
                }
            }
        }
        let paths = &document["paths"];
        for path in ["/api/start_training", "/api/start_test"] {
            for code in ["400", "403", "404", "409", "503"] {
                assert!(
                    paths[path]["post"]["responses"][code]
                        .is_object()
                );
            }
        }
        assert!(paths["/api/stop_workload"]["post"]["responses"]
            ["202"]
            .is_object());
        for path in ["/api/delete_project", "/api/delete_model"] {
            let response = &paths[path]["delete"]["responses"];
            assert!(response["204"].is_object());
            assert!(response["204"].get("content").is_none());
        }
    }

    #[test]
    fn openapi_upload_and_download_representations_match_wire() {
        let document = document();
        let schemas = &document["components"]["schemas"];
        let bytes = &schemas["SaveModelRequestBody"]
            ["properties"]["file_contents"];
        assert_eq!(bytes["type"], "array");
        assert_eq!(bytes["items"]["type"], "integer");
        assert_eq!(bytes["items"]["minimum"], 0.0);
        assert_eq!(bytes["items"]["maximum"], 255.0);
        let input = json!({
            "project_id": "project-1",
            "name": "Example model",
            "model_type": "mjcf",
            "file_type": "zip",
            "base_file_name": "model.xml",
            "file_contents": [0, 255]
        });
        let model: SaveModelRequestBody =
            serde_json::from_value(input.clone()).unwrap();
        assert_eq!(model.file_contents, [0, 255]);
        assert_eq!(serde_json::to_value(model).unwrap(), input);
        for invalid in [json!([-1]), json!([256]), json!("zip")] {
            let mut invalid_input = input.clone();
            invalid_input["file_contents"] = invalid;
            assert!(
                serde_json::from_value::<SaveModelRequestBody>(
                    invalid_input
                )
                .is_err()
            );
        }
        let download_path = "/api/assets/download/{token}";
        let content = &document["paths"][download_path]["get"]
            ["responses"]["200"]["content"];
        assert_eq!(content.as_object().unwrap().len(), 2);
        for media in ["image/png", "application/zip"] {
            assert_eq!(
                content[media]["schema"]["type"],
                "string"
            );
            assert_eq!(
                content[media]["schema"]["format"],
                "binary"
            );
        }
    }

    #[test]
    fn openapi_history_and_array_parameters_match_handlers() {
        let document = document();
        let history_path = "/api/get_workload_updates";
        let history = &document["paths"][history_path]["get"];
        let parameters =
            history["parameters"].as_array().unwrap();
        let limit = parameters
            .iter()
            .find(|p| p["name"] == "limit")
            .unwrap();
        assert_eq!(limit["required"], false);
        assert_eq!(limit["schema"]["minimum"], 1.0);
        assert_eq!(limit["schema"]["maximum"], 500.0);
        assert_eq!(limit["schema"]["default"], 200);
        assert!(history["responses"]["409"].is_object());
        for (path, name) in [
            ("/api/get_models", "model_ids"),
            ("/api/get_workload_checkpoints", "workload_ids"),
        ] {
            let parameter =
                &document["paths"][path]["get"]["parameters"][0];
            assert_eq!(parameter["name"], name);
            assert_eq!(parameter["style"], "form");
            assert_eq!(parameter["explode"], true);
            assert_eq!(parameter["required"], false);
        }
        assert!(document["components"]["schemas"]
            ["WorkloadUpdatesResponseBody"]["properties"]
            .get("next_cursor")
            .is_some());
    }

    #[test]
    fn openapi_preserves_enum_and_timestamp_representations() {
        let document = document();
        let schemas = &document["components"]["schemas"];
        assert_eq!(
            schemas["CheckpointModelType"]["enum"],
            json!(["policy", "value", "flow_noise"])
        );
        let ticket = &schemas["WorkloadLiveTicketResponseBody"];
        let expires_at = &ticket["properties"]["expires_at"];
        assert_eq!(expires_at["format"], "date-time");
        assert_eq!(
            serde_json::to_value(
                WorkloadUpdatesClient::RequestAllUpdatesMessage
            )
            .unwrap(),
            "RequestAllUpdatesMessage"
        );
        let message =
            WorkloadUpdatesServer::WorkloadStoppedMessage(
                WorkloadStopped {
                    workload_id: "workload-1".into(),
                    reason:
                        WorkloadStoppedReasonUser::WorkerFailure(
                            "Example failure".into(),
                        ),
                },
            );
        assert_eq!(
            serde_json::to_value(message).unwrap(),
            json!({"WorkloadStoppedMessage": {
                "workload_id": "workload-1",
                "reason": {"WorkerFailure": "Example failure"}
            }})
        );
        assert!(
            schemas["WorkloadUpdatesServer"]["oneOf"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["properties"]
                    .get("WorkloadStoppedMessage")
                    .is_some())
        );
    }

    #[test]
    fn committed_openapi_matches_current_document() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../docs/user/api_spec/spec.json"
        );
        let saved: Value = serde_json::from_str(
            &std::fs::read_to_string(path).unwrap(),
        )
        .unwrap();
        assert_eq!(
            saved,
            document(),
            "Regenerate with scripts/typegen/gen_api_spec.sh"
        );
    }
}
