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
        ListModelsRequestBody,
        GetModelsRequestBody,
        SaveModelRequestBody,
        DeleteModelsRequestBody,
        ListModelsResponseBody,
        GetModelsResponseBody,
        SaveModelResponseBody,
        ListCheckpointsRequestBody,
        GetWorkloadCheckpointsRequestBody,
        ListCheckpointsResponseBody,
        GetWorkloadCheckpointsResponseBody,
        ListWorkloadsRequestBody,
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
        SimulationModelRef,
        CheckpointModelType,
        CheckpointRef,
        TrajectoryCameraFrameRef,
        WorkloadType,
        LaunchOrigin,
        WorkloadStatus,
        Workload,
        AggregationFunction,
        WorkloadMethod,
        UserMetricsLut,
        WorkloadStructureLut,
        LogType,
        LogStream,
        WorkloadLogs,
        WorkloadUserMetrics,
        WorkloadSimulationModelTrajectoryState,
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
        }
    }
    Json(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openapi_contains_all_routes_and_resolves_schema_refs() {
        fn check_refs(
            value: &serde_json::Value,
            root: &serde_json::Value,
        ) {
            match value {
                serde_json::Value::Object(fields) => {
                    if let Some(reference) = fields
                        .get("$ref")
                        .and_then(|v| v.as_str())
                    {
                        assert!(
                            root.pointer(
                                reference.trim_start_matches('#')
                            )
                            .is_some(),
                            "Missing schema: {reference}"
                        );
                    }
                    for field in fields.values() {
                        check_refs(field, root);
                    }
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        check_refs(value, root);
                    }
                }
                _ => {}
            }
        }
        let document =
            serde_json::to_value(openapi_json().into_inner())
                .unwrap();
        assert_eq!(
            document["paths"].as_object().unwrap().len(),
            17
        );
        assert!(
            document["paths"]["/api/start_training"]["post"]
                .is_object()
        );
        assert!(
            document["paths"]["/api/delete_model"]["delete"]
                .is_object()
        );
        assert!(document["paths"]["/api/list_projects"]["get"]["security"].is_array());
        check_refs(&document, &document);
    }
}
