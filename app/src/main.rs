use std::env;
use std::sync::{Arc, OnceLock};

#[macro_use]
extern crate rocket;
use rocket::fairing::AdHoc;
use rocket::fs::FileServer;
use rocket::tokio;
use rocket::tokio::sync::Mutex as AsyncMutex;

use crate::db::db_orm::PostgresOrm;
use crate::db::mongo_orm::MongoOrm;

pub mod api;
pub mod commondef;
pub mod db;
pub mod simulator;
pub mod workflows;

// ── Globals ─────────────────────────────────

pub type RemoteWorkerSlot = Arc<
    AsyncMutex<
        Option<
            workflows::remote_worker_controller
            ::RemoteWorkerController
        >
    >
>;

static REMOTE_WORKER_SLOT: OnceLock<RemoteWorkerSlot> =
    OnceLock::new();
static REMOTE_WORKER_SHUTDOWN_TASK_STARTED: OnceLock<()> =
    OnceLock::new();

// ── Global accessors ────────────────────────

pub fn remote_worker_slot() -> &'static RemoteWorkerSlot {
    REMOTE_WORKER_SLOT
        .get_or_init(|| Arc::new(AsyncMutex::new(None)))
}

// ── Remote worker helpers ────────────────────

pub fn ensure_remote_worker_shutdown_hook() {
    if REMOTE_WORKER_SHUTDOWN_TASK_STARTED.set(()).is_err() {
        return;
    }

    let slot = remote_worker_slot().clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let mut guard = slot.lock().await;
            if let Some(worker) = guard.as_mut() {
                let _ = worker.terminate_instance().await;
            }
            *guard = None;
        }
    });
}

pub fn build_remote_worker_config()
-> Result<
    workflows::remote_worker_controller
    ::RemoteWorkerControllerConfig,
    String,
>
{
    let mut cfg =
        workflows::remote_worker_controller
        ::RemoteWorkerControllerConfig::default();

    if let Ok(region) = env::var("REMOTE_WORKER_AWS_REGION") {
        if !region.is_empty() {
            cfg.aws_region = region;
        }
    }
    cfg.ami_id =
        env::var("REMOTE_WORKER_AMI_ID").map_err(|_| {
            "Missing required env var \
             REMOTE_WORKER_AMI_ID"
                .to_string()
        })?;
    cfg.worker_image_uri = env::var("REMOTE_WORKER_IMAGE_URI")
        .map_err(|_| {
            "Missing required env var \
             REMOTE_WORKER_IMAGE_URI"
                .to_string()
        })?;
    cfg.worker_shared_secret = env::var(
        "REMOTE_WORKER_SHARED_SECRET",
    )
    .map_err(|_| {
        "Missing required env var \
             REMOTE_WORKER_SHARED_SECRET"
            .to_string()
    })?;

    if let Ok(v) = env::var("REMOTE_WORKER_SUBNET_ID") {
        if !v.is_empty() {
            cfg.subnet_id = Some(v);
        }
    }
    if let Ok(v) = env::var("REMOTE_WORKER_SECURITY_GROUP_IDS") {
        let values: Vec<String> = v
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect();
        if !values.is_empty() {
            cfg.security_group_ids = values;
        }
    }
    if let Ok(v) = env::var("REMOTE_WORKER_KEY_NAME") {
        if !v.is_empty() {
            cfg.key_name = Some(v);
        }
    }
    if let Ok(v) =
        env::var("REMOTE_WORKER_IAM_INSTANCE_PROFILE_ARN")
    {
        if !v.is_empty() {
            cfg.iam_instance_profile_arn = Some(v);
        }
    }
    if let Ok(v) = env::var("REMOTE_WORKER_INSTANCE_NAME_TAG") {
        if !v.is_empty() {
            cfg.instance_name_tag = v;
        }
    }
    if let Ok(v) = env::var("REMOTE_WORKER_SPOT_MAX_PRICE_USD") {
        if !v.is_empty() {
            cfg.spot_max_price_usd = Some(v);
        }
    }
    if let Ok(v) = env::var("REMOTE_WORKER_CONTAINER_NAME") {
        if !v.is_empty() {
            cfg.worker_container_name = v;
        }
    }
    if let Ok(v) = env::var("REMOTE_WORKER_PORT") {
        if let Ok(port) = v.parse::<u16>() {
            cfg.worker_port = port;
        }
    }
    if let Ok(v) = env::var("REMOTE_WORKER_WS_PATH") {
        if !v.is_empty() {
            cfg.worker_ws_path = v;
        }
    }

    Ok(cfg)
}

// ── API Routes ───────────────────────────────

#[get("/hello")]
fn api_hello() -> &'static str {
    "Hello from Rocket!"
}

#[launch]
fn rocket() -> _ {
    let client_dist_dir = env::var("CLIENT_DIST_DIR")
        .unwrap_or_else(|_| {
            format!(
                "{}/../client/build",
                env!("CARGO_MANIFEST_DIR")
            )
        });

    rocket::build()
        .attach(AdHoc::try_on_ignite("API dependencies", |rocket| {
            Box::pin(async move {
                let dependencies = async {
                    let postgres = PostgresOrm::new(
                        &env::var("DATABASE_URL")?,
                    ).await?;
                    let mongo = MongoOrm::new(
                        &env::var("MONGODB_URI")?,
                    ).await?;
                    let bucket = env::var("S3_BUCKET")?;
                    let config = aws_config::defaults(
                        aws_config::BehaviorVersion::latest(),
                    ).load().await;
                    Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                        api::ApiState::new(
                            Arc::new(postgres), Arc::new(mongo),
                            aws_sdk_s3::Client::new(&config), bucket,
                        ),
                    )
                }.await;
                match dependencies {
                    Ok(state) => Ok(rocket.manage(Arc::new(state))),
                    Err(error) => {
                        eprintln!("API initialization failed: {error}");
                        Err(rocket)
                    }
                }
            })
        }))
        .attach(AdHoc::on_shutdown("Stop workloads", |rocket| {
            Box::pin(async move {
                if let Some(state) = rocket.state::<Arc<api::ApiState>>() {
                    for (_, stop) in state.jobs.lock().await.values() {
                        let _ = stop.send(true);
                    }
                }
            })
        }))
        .mount("/api", routes![
            api_hello, api::openapi_json,
            api::workloads::create_project,
            api::workloads::list_projects,
            api::workloads::delete_project,
            api::workloads::list_workloads,
            api::workloads::stop_workload,
            api::workloads::get_workload_updates,
            api::workloads::live_ticket,
            api::workloads::live_updates,
            api::models::list_models, api::models::get_models,
            api::models::save_model, api::models::delete_model,
            api::models::download_asset,
            api::checkpoints::list_checkpoints,
            api::checkpoints::get_workload_checkpoints,
            api::training::start_training,
            api::testing::start_test,
        ])
        .mount("/", FileServer::from(client_dist_dir).rank(1))
}
