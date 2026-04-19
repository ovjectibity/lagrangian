use std::env;

use rocket::State;
use rocket::http::Status;
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::Json;
use rocket::tokio::sync::{broadcast, mpsc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD
    as BASE64;

use crate::api::ApiState;
use crate::commondef::clientmessages::ClientMessage;
use crate::commondef::testing::SimulationParams;
use crate::commondef::workermessages::TestingTaskMessage;
use crate::workflows::{WorkflowMode, WorkflowRuntime};
use crate::workflows::testing::TestingWorkflow;

// ── Request / response types ─────────────────

#[derive(Deserialize)]
pub struct StartSimulationBody {
    pub simulation_params: SimulationParams,
}

#[derive(Serialize)]
pub struct SimulationStartedResponse {
    pub id: String,
}

#[derive(Serialize)]
pub struct SimulationStoppedResponse {
    pub id: String,
}

// ── Handlers ─────────────────────────────────

#[post(
    "/testing/simulations",
    format = "json",
    data = "<body>"
)]
pub async fn start_simulation(
    body: Json<StartSimulationBody>,
    state: &State<ApiState>,
) -> Result<
    Json<SimulationStartedResponse>,
    Status,
>
{
    let simul_id = Uuid::new_v4().to_string();
    let params =
        body.into_inner().simulation_params;
    let model_path =
        env::var("TEST_MJCF_MODEL").ok();

    // SSE broadcast channel
    let (sse_tx, _) =
        broadcast::channel::<String>(512);
    state
        .simul_senders
        .lock()
        .await
        .insert(simul_id.clone(), sse_tx.clone());

    // Internal mpsc for the testing workflow
    let (tx_test, mut rx_test) =
        mpsc::channel::<TestingTaskMessage>(64);

    // Relay: mpsc → broadcast
    // Simulation frames are base64-encoded binary.
    let relay_sse_tx = sse_tx.clone();
    let relay_id = simul_id.clone();
    rocket::tokio::spawn(async move {
        while let Some(msg) = rx_test.recv().await {
            match msg {
                TestingTaskMessage
                ::SimulationUpdates(m) => {
                    for update in m.updates {
                        let binary =
                            ClientMessage
                            ::to_binary_simul_update(
                                relay_id.clone(),
                                update,
                            );
                        relay_sse_tx
                            .send(BASE64.encode(&binary))
                            .ok();
                    }
                }
                TestingTaskMessage
                ::StopSimulation(_) => break,
                TestingTaskMessage
                ::StartSimulation(_) => {}
            }
        }
    });

    // Spawn blocking testing workflow
    let tx_c = tx_test.clone();
    let id_c = simul_id.clone();
    let h = rocket::tokio::task::spawn_blocking(
        move || {
            let path =
                model_path.unwrap_or_default();
            let mut runtime =
                match WorkflowRuntime::load(path) {
                    Ok(r) => r,
                    Err(e) => {
                        println!(
                            "WorkflowRuntime load \
                             failed: {:?}",
                            e
                        );
                        return;
                    }
                };
            runtime.set_mode(WorkflowMode::Testing);
            let sim = runtime.simulator_handle();
            let model =
                runtime.control_model_handle();
            let mut wf = TestingWorkflow::new(
                id_c,
                sim,
                model,
                params,
            );
            wf.run_test(tx_c);
        },
    );

    state
        .simul_tasks
        .lock()
        .await
        .insert(simul_id.clone(), h);

    Ok(Json(SimulationStartedResponse {
        id: simul_id,
    }))
}

#[post("/testing/simulations/<simul_id>/stop")]
pub async fn stop_simulation(
    simul_id: String,
    state: &State<ApiState>,
) -> Result<
    Json<SimulationStoppedResponse>,
    Status,
>
{
    if let Some(h) = state
        .simul_tasks
        .lock()
        .await
        .remove(&simul_id)
    {
        h.abort();
    }
    Ok(Json(SimulationStoppedResponse {
        id: simul_id,
    }))
}

#[get("/testing/simulations/<simul_id>/updates")]
pub async fn simulation_updates(
    simul_id: String,
    state: &State<ApiState>,
) -> EventStream![]
{
    let rx = state
        .simul_senders
        .lock()
        .await
        .get(&simul_id)
        .map(|tx| tx.subscribe());

    EventStream! {
        if let Some(mut rx) = rx {
            loop {
                match rx.recv().await {
                    Ok(data) =>
                        yield Event::data(data),
                    Err(
                        broadcast::error::RecvError
                        ::Closed
                    ) => break,
                    Err(
                        broadcast::error::RecvError
                        ::Lagged(_)
                    ) => continue,
                }
            }
        }
    }
}
