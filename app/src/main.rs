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
pub mod mujoco;
pub mod plugin;
pub mod remote_workflows;
pub mod simulator;
pub mod usr;
pub mod workflows;

// ── Project/User constants ──────────────────
pub const PROJECT_ID: &str = "1";

// ── Globals ─────────────────────────────────

pub type RemoteWorkerSlot = Arc<
    AsyncMutex<
        Option<
            remote_workflows::training_controller
            ::TrainingWorkerController
        >
    >
>;

static REMOTE_WORKER_SLOT: OnceLock<RemoteWorkerSlot> =
    OnceLock::new();
static REMOTE_WORKER_SHUTDOWN_TASK_STARTED: OnceLock<()> =
    OnceLock::new();

static POSTGRES_ORM: OnceLock<Arc<PostgresOrm>> =
    OnceLock::new();
static MONGO_ORM: OnceLock<Arc<MongoOrm>> =
    OnceLock::new();

// ── Global accessors ────────────────────────

pub fn remote_worker_slot()
-> &'static RemoteWorkerSlot {
    REMOTE_WORKER_SLOT.get_or_init(|| {
        Arc::new(AsyncMutex::new(None))
    })
}

pub fn postgres_orm() -> &'static Arc<PostgresOrm> {
    POSTGRES_ORM.get()
        .expect("PostgresOrm not initialised")
}

pub fn mongo_orm() -> &'static Arc<MongoOrm> {
    MONGO_ORM.get()
        .expect("MongoOrm not initialised")
}

// ── Remote worker helpers ────────────────────

pub fn ensure_remote_worker_shutdown_hook() {
    if REMOTE_WORKER_SHUTDOWN_TASK_STARTED
        .set(()).is_err()
    {
        return;
    }

    let slot = remote_worker_slot().clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let mut guard = slot.lock().await;
            if let Some(worker) = guard.as_mut() {
                let _ = worker
                    .terminate_instance()
                    .await;
            }
            *guard = None;
        }
    });
}

pub fn build_remote_training_worker_config()
-> Result<
    remote_workflows::training_controller
    ::TrainingWorkerControllerConfig,
    String,
>
{
    let mut cfg =
        remote_workflows::training_controller
        ::TrainingWorkerControllerConfig::default();

    if let Ok(region) =
        env::var("TRAINING_WORKER_AWS_REGION")
    {
        if !region.is_empty() {
            cfg.aws_region = region;
        }
    }
    cfg.ami_id = env::var("TRAINING_WORKER_AMI_ID")
        .map_err(|_| {
            "Missing required env var \
             TRAINING_WORKER_AMI_ID".to_string()
        })?;
    cfg.worker_image_uri =
        env::var("TRAINING_WORKER_IMAGE_URI")
        .map_err(|_| {
            "Missing required env var \
             TRAINING_WORKER_IMAGE_URI".to_string()
        })?;
    cfg.worker_shared_secret =
        env::var("TRAINING_WORKER_SHARED_SECRET")
        .map_err(|_| {
            "Missing required env var \
             TRAINING_WORKER_SHARED_SECRET".to_string()
        })?;

    if let Ok(v) =
        env::var("TRAINING_WORKER_SUBNET_ID")
    {
        if !v.is_empty() {
            cfg.subnet_id = Some(v);
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_SECURITY_GROUP_IDS")
    {
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
    if let Ok(v) =
        env::var("TRAINING_WORKER_KEY_NAME")
    {
        if !v.is_empty() {
            cfg.key_name = Some(v);
        }
    }
    if let Ok(v) = env::var(
        "TRAINING_WORKER_IAM_INSTANCE_PROFILE_ARN",
    ) {
        if !v.is_empty() {
            cfg.iam_instance_profile_arn = Some(v);
        }
    }
    if let Ok(v) = env::var(
        "TRAINING_WORKER_INSTANCE_NAME_TAG",
    ) {
        if !v.is_empty() {
            cfg.instance_name_tag = v;
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_SPOT_MAX_PRICE_USD")
    {
        if !v.is_empty() {
            cfg.spot_max_price_usd = Some(v);
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_CONTAINER_NAME")
    {
        if !v.is_empty() {
            cfg.worker_container_name = v;
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_PORT")
    {
        if let Ok(port) = v.parse::<u16>() {
            cfg.worker_port = port;
        }
    }
    if let Ok(v) =
        env::var("TRAINING_WORKER_WS_PATH")
    {
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
    let client_dist_dir =
        env::var("CLIENT_DIST_DIR")
        .expect("Expected CLIENT_DIST_DIR to be set");

    rocket::build()
        .attach(AdHoc::on_liftoff(
            "DB Init",
            |_rocket| Box::pin(async move {
                match env::var("DATABASE_URL") {
                    Ok(url) => {
                        match PostgresOrm::new(&url).await {
                            Ok(orm) => {
                                let _ = POSTGRES_ORM
                                    .set(Arc::new(orm));
                                println!(
                                    "PostgresOrm initialised"
                                );
                            }
                            Err(e) => println!(
                                "WARNING: PostgresOrm \
                                 init failed: {:?}", e
                            ),
                        }
                    }
                    Err(_) => println!(
                        "WARNING: DATABASE_URL not set; \
                         PostgresOrm unavailable"
                    ),
                }
                match env::var("MONGODB_URI") {
                    Ok(uri) => {
                        match MongoOrm::new(&uri).await {
                            Ok(orm) => {
                                let _ = MONGO_ORM
                                    .set(Arc::new(orm));
                                println!(
                                    "MongoOrm initialised"
                                );
                            }
                            Err(e) => println!(
                                "WARNING: MongoOrm \
                                 init failed: {:?}", e
                            ),
                        }
                    }
                    Err(_) => println!(
                        "WARNING: MONGODB_URI not set; \
                         MongoOrm unavailable"
                    ),
                }
            }),
        ))
        .manage(api::ApiState::new())
        .mount("/api", routes![
            api_hello,
            api::models::get_default_model,
            api::models::list_models,
            api::models::save_model,
            api::checkpoints::list_checkpoints,
            api::checkpoints::get_checkpoint,
            api::experiments::list_experiments,
            api::experiments::create_experiment,
            api::experiments::list_runs,
            api::experiments::create_run,
            api::experiments::get_run,
            api::experiments::get_run_checkpoints,
            api::training::start_training,
            api::training::stop_training,
            api::training::training_updates,
            api::simulations::start_simulation,
            api::simulations::stop_simulation,
            api::simulations::simulation_updates,
        ])
        .mount(
            "/",
            FileServer::from(client_dist_dir).rank(1),
        )
}
