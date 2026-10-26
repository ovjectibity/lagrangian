use rocket::http::Status;
use rocket::serde::json::Json;
use rocket::tokio::sync::{broadcast, mpsc};
use rocket::State;
use rocket_ws as ws;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::api::{route_doc, ApiState};
use crate::commondef::record::{TestGraphRecord, TestRunStatus};
use crate::commondef::testing::{StopTestMessage, TestGraph, TestMethod};
use crate::commondef::training::{
    ControlPolicyInputSpec, TrainingAccelerationMode,
};
use crate::workflows::local_worker_controller::LocalWorkerController;

/// Internal task→relay message for the testing mpsc (API-local). Test updates
/// are opaque protobuf `TestingStreamMessage` frames forwarded verbatim to SDK
/// subscribers; the server never decodes them.
#[derive(Debug, Clone)]
enum TestTaskMessage {
    StopTestMessage(StopTestMessage),
    TestUpdateBytes(Vec<u8>),
}

// ── Request / response types ─────────────────

#[derive(Deserialize, ToSchema)]
pub struct StartTestBody {
    pub test_graph: TestGraph,
}

#[derive(Serialize, ToSchema)]
pub struct TestStartedResponse {
    pub id: String,
}

#[derive(Serialize, ToSchema)]
pub struct TestStoppedResponse {
    pub id: String,
}

// ── Handlers ─────────────────────────────────

#[route_doc(
    tag = "Testing",
    ok = TestStartedResponse,
    body = StartTestBody,
    errors(400),
)]
#[post("/testing/start", format = "json", data = "<body>")]
pub async fn start_test(
    body: Json<StartTestBody>,
    state: &State<Arc<ApiState>>,
) -> Result<Json<TestStartedResponse>, Status> {
    let test_id = Uuid::new_v4().to_string();
    let test_graph = body.into_inner().test_graph;

    let control_model = match &test_graph.test_job_controller.test_method {
        TestMethod::Custom(method) => &method.control_model,
        TestMethod::Dataset(method) => &method.control_model,
    };
    let control_spec = match control_model {
        crate::commondef::training::ControlModel::MLP(m) => &m.input_spec,
        crate::commondef::training::ControlModel::VLA(m) => &m.input_spec,
    };
    if matches!(
        control_spec,
        ControlPolicyInputSpec::OrdinaryConfiguration(
            spec
        ) if spec.contact_pairs
    ) {
        return Err(Status::BadRequest);
    }

    let is_local = matches!(
        test_graph.test_job_controller.acceleration_mode,
        TrainingAccelerationMode::AllOnServer | TrainingAccelerationMode::MLCPUSimulationCPU,
    );

    // Dataset tests provide task MJCFs from the dataset. Only custom tests
    // need a user-provided simulation model to be resolved and uploaded.
    let model_path = match &test_graph.test_job_controller.test_method {
        TestMethod::Custom(method) => {
            let model_ref = crate::postgres_orm()
                .get_model_reference(&method.simulation_model)
                .await
                .map_err(|_| Status::InternalServerError)?
                .ok_or(Status::BadRequest)?;
            Some(model_ref.path.clone())
        }
        TestMethod::Dataset(_) => None,
    };

    // DB records
    let logs_id = format!("logs-{}", test_id);
    let metrics_id = format!("metrics-{}", test_id);
    let test_graph_id = format!("tg-{}", test_id);

    let _ = crate::postgres_orm()
        .create_test_run(
            crate::PROJECT_ID,
            &test_id,
            &TestRunStatus::NotStarted,
            &logs_id,
            &metrics_id,
            &test_graph_id,
        )
        .await;

    let now = chrono::Utc::now();
    let _ = crate::mongo_orm()
        .create_test_graph(&TestGraphRecord {
            id: test_graph_id.clone(),
            test_run_id: test_id.clone(),
            created_at: now,
            graph: test_graph.clone(),
        })
        .await;

    let _ = crate::postgres_orm()
        .update_test_run_status(&test_id, &TestRunStatus::Ongoing)
        .await;

    // Broadcast channel of raw binary simul frames (fanned out to WS subscribers)
    let (sse_tx, _) = broadcast::channel::<Vec<u8>>(512);
    state
        .simul_senders
        .lock()
        .await
        .insert(test_id.clone(), sse_tx.clone());

    // Internal mpsc for the testing workflow
    let (tx_test, mut rx_test) = mpsc::channel::<TestTaskMessage>(64);

    // Relay: mpsc → broadcast. Protobuf frames are forwarded verbatim.
    let relay_sse_tx = sse_tx.clone();
    let test_id_c = test_id.clone();
    let state_c = Arc::clone(state);
    // let state_c = Arc::clone(state.
    rocket::tokio::spawn(async move {
        while let Some(msg) = rx_test.recv().await {
            match msg {
                TestTaskMessage::TestUpdateBytes(bytes) => {
                    relay_sse_tx.send(bytes).ok();
                }
                //TODO: This needs to be handled.
                TestTaskMessage::StopTestMessage(_) => {
                    if let Some(x) = state_c.simul_senders.lock().await.remove(&test_id_c) {
                        drop(x);
                    }
                    state_c.simul_tasks.lock().await.remove(&test_id_c);
                    let _ = crate::postgres_orm()
                        .update_test_run_status(&test_id_c, &TestRunStatus::Completed)
                        .await;
                }
            }
        }
    });

    if is_local {
        let tx_c = tx_test.clone();
        let id_c = test_id.clone();
        let h = rocket::tokio::spawn(async move {
            spawn_local_test_task(id_c, test_graph, model_path, tx_c).await;
        });

        state.simul_tasks.lock().await.insert(test_id.clone(), h);
    } else {
        let remote_cfg = match crate::build_remote_worker_config() {
            Ok(c) => c,
            Err(e) => {
                println!("remote cfg failed: {:?}", e);
                return Err(Status::InternalServerError);
            }
        };
        crate::ensure_remote_worker_shutdown_hook();

        let tx_c = tx_test.clone();
        let id_c = test_id.clone();
        let worker_slot = crate::remote_worker_slot().clone();
        print!("Spawning remote test task for start_test");
        let h = rocket::tokio::spawn(async move {
            spawn_remote_test_task(id_c, test_graph, model_path, remote_cfg, worker_slot, tx_c)
                .await;
        });

        state.simul_tasks.lock().await.insert(test_id.clone(), h);
    }

    Ok(Json(TestStartedResponse { id: test_id }))
}

#[route_doc(
    tag = "Testing",
    ok = TestStoppedResponse,
    params(test_id = "Test run UUID"),
)]
#[post("/testing/<test_id>/stop")]
pub async fn stop_test(
    test_id: String,
    state: &State<Arc<ApiState>>,
) -> Result<Json<TestStoppedResponse>, Status> {
    if let Some(h) = state.simul_tasks.lock().await.remove(&test_id) {
        h.abort();
    }
    let _ = crate::postgres_orm()
        .update_test_run_status(&test_id, &TestRunStatus::StoppedByUser)
        .await;
    Ok(Json(TestStoppedResponse { id: test_id }))
}

#[route_doc(tag = "Testing", params(test_id = "Test run UUID"))]
#[get("/testing/<test_id>/updates")]
pub async fn test_updates(
    test_id: String,
    state: &State<Arc<ApiState>>,
    ws: ws::WebSocket,
) -> ws::Channel<'static> {
    let rx = state
        .simul_senders
        .lock()
        .await
        .get(&test_id)
        .map(|tx| tx.subscribe());

    ws.channel(move |mut stream| {
        Box::pin(async move {
            use rocket::futures::{SinkExt, StreamExt};
            if let Some(mut rx) = rx {
                loop {
                    rocket::tokio::select! {
                        msg = rx.recv() => match msg {
                            Ok(bytes) => {
                                if stream.send(ws::Message::Binary(bytes)).await.is_err() {
                                    break;
                                }
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                //Close the websocket
                                let _ = stream.close(None).await;
                                break;
                            },
                            Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        },
                        incoming = stream.next() => match incoming {
                            None
                            | Some(Ok(ws::Message::Close(_)))
                            | Some(Err(_)) => break,
                            Some(Ok(_)) => {}
                        },
                    }
                }
            }
            Ok(())
        })
    })
}

// ── Remote test helpers ───────────────────────

async fn spawn_local_test_task(
    test_id: String,
    test_graph: TestGraph,
    model_path: Option<String>,
    tx: mpsc::Sender<TestTaskMessage>,
) {
    // Connect-only: assumes a Python worker is already running on the local host.
    let mut controller = match LocalWorkerController::from_env() {
        Ok(c) => c,
        Err(e) => {
            println!("Local worker config failed: {:?}", e);
            let _ = tx
                .send(TestTaskMessage::StopTestMessage(StopTestMessage {
                    id: test_id.clone(),
                }))
                .await;
            return;
        }
    };

    let result = controller
        .run_test_job(test_id.clone(), test_graph, model_path, |frame: Vec<u8>| {
            let _ = tx.try_send(TestTaskMessage::TestUpdateBytes(frame));
        })
        .await;

    if let Err(e) = result {
        println!("Local test failed: {:?}", e);
    }
    let _ = tx
        .send(TestTaskMessage::StopTestMessage(StopTestMessage {
            id: test_id.clone(),
        }))
        .await;

    let _ = crate::postgres_orm()
        .update_test_run_status(&test_id, &TestRunStatus::Completed)
        .await;
}

async fn spawn_remote_test_task(
    test_id: String,
    test_graph: TestGraph,
    model_path: Option<String>,
    remote_cfg: crate::workflows::remote_worker_controller::RemoteWorkerControllerConfig,
    worker_slot: crate::RemoteWorkerSlot,
    tx: mpsc::Sender<TestTaskMessage>,
) {
    use crate::workflows::remote_worker_controller::RemoteWorkerController;

    let mut guard = worker_slot.lock().await;
    if guard.is_none() {
        match RemoteWorkerController::initialize(remote_cfg).await {
            Ok(w) => *guard = Some(w),
            Err(e) => {
                println!("Remote worker init: {:?}", e);
                let _ = tx
                    .send(TestTaskMessage::StopTestMessage(StopTestMessage {
                        id: test_id.clone(),
                    }))
                    .await;
                return;
            }
        }
    }
    let worker = guard.as_mut().expect("worker initialised");

    let result = worker
        .run_test_job(test_id.clone(), test_graph, model_path, |frame: Vec<u8>| {
            let _ = tx.try_send(TestTaskMessage::TestUpdateBytes(frame));
        })
        .await;

    drop(guard);
    if let Err(e) = result {
        println!("Remote test failed: {:?}", e);
    }
    let _ = tx
        .send(TestTaskMessage::StopTestMessage(StopTestMessage {
            id: test_id.clone(),
        }))
        .await;

    let _ = crate::postgres_orm()
        .update_test_run_status(&test_id, &TestRunStatus::Completed)
        .await;
}
