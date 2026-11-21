//! Typed, cancellable communication with a Python worker.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::{Instant, sleep, sleep_until, timeout};
use tokio_tungstenite::tungstenite::{
    Message, client::IntoClientRequest, http::HeaderValue,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
};

use crate::commondef::remoteworker::remoteworkermessages::{
    RemoteWorkerClientMessage, RemoteWorkerServerMessage,
    StopWorkloadServer, StopWorkloadWorker,
    WorkloadStopReasonServer, WorkloadStopReasonWorker,
};

pub type DynError = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, DynError>;
pub type WorkerWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Connection settings shared by local and EC2 workers.
pub struct CommonWorkerControllerConfig {
    pub host: String,
    pub port: u16,
    pub ws_path: String,
    pub shared_secret: String,
    pub websocket_connect_timeout: Duration,
    pub stop_ack_timeout: Duration,
}

/// Owns one worker connection for the duration of a workload.
pub struct CommonWorkerController {
    config: CommonWorkerControllerConfig,
    worker_ws: Option<WorkerWs>,
}

impl CommonWorkerController {
    /// Construct a disconnected controller.
    pub fn new(config: CommonWorkerControllerConfig) -> Self {
        Self {
            config,
            worker_ws: None,
        }
    }

    /// Set the provisioned worker's address.
    pub fn set_host(&mut self, host: String) {
        self.config.host = host;
    }

    /// Connect with the worker secret and a bounded retry period.
    pub async fn connect_worker_wss(&mut self) -> Result<()> {
        let path = self.config.ws_path.trim_start_matches('/');
        let url = format!(
            "ws://{}:{}/{path}",
            self.config.host, self.config.port,
        );
        let deadline = Instant::now()
            + self.config.websocket_connect_timeout;
        loop {
            let mut request =
                url.clone().into_client_request()?;
            request.headers_mut().insert(
                "Authorization",
                HeaderValue::from_str(&format!(
                    "Bearer {}",
                    self.config.shared_secret,
                ))?,
            );
            let attempt = timeout(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_secs(10)),
                connect_async(request),
            )
            .await;
            if let Ok(Ok((socket, _))) = attempt {
                self.worker_ws = Some(socket);
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err("Worker connection timed out".into());
            }
            sleep(Duration::from_secs(1)).await;
        }
    }

    /// Close the connection without terminating its instance.
    pub async fn close(&mut self) {
        if let Some(socket) = self.worker_ws.as_mut() {
            let _ = timeout(
                Duration::from_secs(2),
                socket.close(None),
            )
            .await;
        }
        self.worker_ws = None;
    }

    /// Run a typed start message until the worker acknowledges
    /// completion. Backpressure never prevents a stop write.
    pub async fn run_job(
        &mut self,
        start: RemoteWorkerServerMessage,
        stop: watch::Receiver<bool>,
        updates: mpsc::Sender<RemoteWorkerClientMessage>,
        launched: oneshot::Sender<()>,
    ) -> Result<StopWorkloadWorker> {
        let result = self
            .run_connected_job(start, stop, updates, launched)
            .await;
        self.close().await;
        result
    }

    async fn run_connected_job(
        &mut self,
        start: RemoteWorkerServerMessage,
        mut stop: watch::Receiver<bool>,
        updates: mpsc::Sender<RemoteWorkerClientMessage>,
        launched: oneshot::Sender<()>,
    ) -> Result<StopWorkloadWorker> {
        let (id, testing) = match &start {
            RemoteWorkerServerMessage::StartTrainingMessage(
                m,
            ) => (m.workload_id.clone(), false),
            RemoteWorkerServerMessage::StartTestingMessage(m) => {
                (m.workload_id.clone(), true)
            }
            _ => return Err("Expected a start message".into()),
        };
        self.close().await;
        tokio::select! {
            result = self.connect_worker_wss() => result?,
            _ = wait_for_stop(&mut stop) => {
                return Ok(requested_stop(id));
            }
        }
        let socket = self
            .worker_ws
            .as_mut()
            .ok_or("Worker socket unavailable")?;
        timeout(
            self.config.websocket_connect_timeout,
            socket.send(Message::Text(
                serde_json::to_string(&start)?.into(),
            )),
        )
        .await
        .map_err(|_| "Worker start write timed out")??;
        let _ = launched.send(());
        let mut stop_sent = false;
        let mut stop_deadline = Instant::now();
        let mut pending = None;
        loop {
            tokio::select! {
                _ = wait_for_stop(&mut stop), if !stop_sent => {
                    let message = StopWorkloadServer {
                        workload_id: id.clone(),
                        reason: WorkloadStopReasonServer::RequestedByUser,
                    };
                    let message = if testing {
                        RemoteWorkerServerMessage::StopTestingMessage(message)
                    } else {
                        RemoteWorkerServerMessage::StopTrainingMessage(message)
                    };
                    stop_deadline = Instant::now()
                        + self.config.stop_ack_timeout;
                    timeout(self.config.stop_ack_timeout,
                        socket.send(Message::Text(
                            serde_json::to_string(&message)?.into(),
                        )),
                    ).await.map_err(|_| "Worker stop write timed out")??;
                    stop_sent = true;
                }
                _ = sleep_until(stop_deadline), if stop_sent => {
                    return Err("Worker stop acknowledgement timed out".into());
                }
                permit = updates.reserve(), if pending.is_some() => {
                    permit.map_err(|_| "Update processor closed")?
                        .send(pending.take().ok_or("Missing pending update")?);
                }
                frame = socket.next(), if pending.is_none() => {
                    match frame {
                        Some(Ok(Message::Text(text))) => {
                            let message: RemoteWorkerClientMessage =
                                serde_json::from_str(&text)?;
                            validate_worker_message(&message, &id, testing)?;
                            match message {
                                RemoteWorkerClientMessage::StopTrainingMessage(m)
                                | RemoteWorkerClientMessage::StopTestingMessage(m) => {
                                    return Ok(m);
                                }
                                message => pending = Some(message),
                            }
                        }
                        Some(Ok(Message::Ping(bytes))) => {
                            socket.send(Message::Pong(bytes)).await?;
                        }
                        Some(Ok(Message::Pong(_))) => {}
                        Some(Ok(Message::Close(_))) | None => {
                            return Err("Worker disconnected before completion".into());
                        }
                        Some(Err(error)) => return Err(error.into()),
                        _ => return Err("Unexpected worker frame".into()),
                    }
                }
            }
        }
    }
}

async fn wait_for_stop(stop: &mut watch::Receiver<bool>) {
    while !*stop.borrow_and_update() {
        if stop.changed().await.is_err() {
            return;
        }
    }
}

fn requested_stop(id: String) -> StopWorkloadWorker {
    StopWorkloadWorker {
        workload_id: id,
        reason: WorkloadStopReasonWorker::RequestedByAppServer,
    }
}

fn validate_worker_message(
    message: &RemoteWorkerClientMessage,
    expected: &str,
    testing: bool,
) -> Result<()> {
    use RemoteWorkerClientMessage::*;
    let (id, is_test) = match message {
        TrainingUpdatesMessage(m) => (&m.workload_id, false),
        TestUpdatesMessage(m) => (&m.workload_id, true),
        StopTrainingMessage(m) => (&m.workload_id, false),
        StopTestingMessage(m) => (&m.workload_id, true),
    };
    if id != expected || is_test != testing {
        return Err("Worker message scope mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_workload_and_message_kind() {
        let message =
            RemoteWorkerClientMessage::StopTestingMessage(
                requested_stop("one".into()),
            );
        assert!(
            validate_worker_message(&message, "one", true)
                .is_ok()
        );
        assert!(
            validate_worker_message(&message, "two", true)
                .is_err()
        );
        assert!(
            validate_worker_message(&message, "one", false)
                .is_err()
        );
    }
    use crate::commondef::records::workload_graph::*;
    use crate::commondef::remoteworker::remoteworkermessages::{
        StartTestWorkerMessage, WorkloadUpdates,
    };
    use serde_json::json;
    fn test_graph() -> WorkloadGraph {
        serde_json::from_value(json!({"TestGraph": {
            "test_job_controller": {
                "test_method": {"type": "Dataset", "content": {
                    "dataset": {"type": "Libero", "content": {
                        "tasks_index": [0], "libero_dataset_type": "Libero10",
                        "num_envs_per_task": 1, "num_timesteps": 2
                    }},
                    "control_model": {
                        "model_type": {"type": "New"},
                        "model": {"type": "VLA"},
                        "input_spec": {"type": "StandardInputSpec", "content": {
                            "base_body_name": null, "states": []
                        }},
                        "output_spec": {"type": "ProprioceptiveOutputSpec", "content": {
                            "base_body_name": "robot", "state": {
                                "is_delta": false, "rebase_body": "robot",
                                "state_item": {"type": "QPos"}
                            }
                        }}
                    }
                }},
                "compute_instance": "AWSG4dn",
                "acceleration_mode": "MLCPUSimulationCPU"
            }
        }})).unwrap()
    }

    // Tungstenite fixes the callback's error type to an HTTP response.
    #[allow(clippy::result_large_err)]
    #[tokio::test]
    async fn sends_stop_while_update_queue_is_full() {
        let listener =
            tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
        let port = listener.local_addr().unwrap().port();
        let (seen_tx, seen_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_hdr_async(
                stream,
                |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
                 response| {
                    assert_eq!(request.headers()["Authorization"], "Bearer test-secret");
                    Ok(response)
                },
            ).await.unwrap();
            let start = socket.next().await.unwrap().unwrap();
            let Message::Text(start) = start else {
                panic!("Expected JSON");
            };
            let start: RemoteWorkerServerMessage =
                serde_json::from_str(&start).unwrap();
            assert!(matches!(
                start,
                RemoteWorkerServerMessage::StartTestingMessage(_)
            ));
            for _ in 0..3 {
                let update =
                    RemoteWorkerClientMessage::TestUpdatesMessage(
                        WorkloadUpdates {
                            workload_id: "w".into(),
                            logs: vec![],
                            metrics: vec![],
                            checkpoints: vec![],
                            camera_frames: vec![],
                            simulation_models: vec![],
                            model_states: vec![],
                        },
                    );
                socket
                    .send(Message::Text(
                        serde_json::to_string(&update)
                            .unwrap()
                            .into(),
                    ))
                    .await
                    .unwrap();
            }
            let stop = socket.next().await.unwrap().unwrap();
            let Message::Text(stop) = stop else {
                panic!("Expected JSON stop");
            };
            let stop: RemoteWorkerServerMessage =
                serde_json::from_str(&stop).unwrap();
            assert!(matches!(
                stop,
                RemoteWorkerServerMessage::StopTestingMessage(_)
            ));
            seen_tx.send(()).unwrap();
            socket.send(Message::Text(serde_json::to_string(
                &RemoteWorkerClientMessage::StopTestingMessage(requested_stop("w".into())),
            ).unwrap().into())).await.unwrap();
        });
        let mut controller = CommonWorkerController::new(
            CommonWorkerControllerConfig {
                host: "127.0.0.1".into(),
                port,
                ws_path: "/".into(),
                shared_secret: "test-secret".into(),
                websocket_connect_timeout: Duration::from_secs(2),
                stop_ack_timeout: Duration::from_secs(2),
            },
        );
        let WorkloadGraph::TestGraph(graph) = test_graph() else {
            unreachable!();
        };
        let start =
            RemoteWorkerServerMessage::StartTestingMessage(
                StartTestWorkerMessage {
                    workload_id: "w".into(),
                    project_id: "p".into(),
                    test_graph: graph,
                    checkpoints: vec![],
                    simulation_models: vec![],
                },
            );
        let (tx, mut rx) = mpsc::channel(1);
        let (stop_tx, stop_rx) = watch::channel(false);
        let (launched_tx, launched_rx) = oneshot::channel();
        let job = tokio::spawn(async move {
            controller
                .run_job(start, stop_rx, tx, launched_tx)
                .await
        });
        timeout(Duration::from_secs(5), async move {
            launched_rx.await.unwrap();
            while rx.is_empty() {
                tokio::task::yield_now().await;
            }
            stop_tx.send(true).unwrap();
            // The stop reaches the worker before this consumer drains.
            seen_rx.await.unwrap();
            for _ in 0..3 {
                rx.recv().await.unwrap();
            }
            assert!(matches!(
                job.await.unwrap().unwrap().reason,
                WorkloadStopReasonWorker::RequestedByAppServer
            ));
            server.await.unwrap();
        })
        .await
        .unwrap();
    }
    #[tokio::test]
    async fn stop_deadline_closes_an_unresponsive_worker() {
        let listener =
            tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket =
                tokio_tungstenite::accept_async(stream)
                    .await
                    .unwrap();
            while let Some(message) = socket.next().await {
                if matches!(
                    message,
                    Ok(Message::Close(_)) | Err(_)
                ) {
                    break;
                }
            }
        });
        let mut controller = CommonWorkerController::new(
            CommonWorkerControllerConfig {
                host: "127.0.0.1".into(),
                port,
                ws_path: "/".into(),
                shared_secret: "test-secret".into(),
                websocket_connect_timeout: Duration::from_secs(2),
                stop_ack_timeout: Duration::from_millis(30),
            },
        );
        let WorkloadGraph::TestGraph(graph) = test_graph() else {
            unreachable!();
        };
        let start =
            RemoteWorkerServerMessage::StartTestingMessage(
                StartTestWorkerMessage {
                    workload_id: "w".into(),
                    project_id: "p".into(),
                    test_graph: graph,
                    checkpoints: vec![],
                    simulation_models: vec![],
                },
            );
        let (tx, _rx) = mpsc::channel(1);
        let (stop_tx, stop_rx) = watch::channel(false);
        let (launched_tx, launched_rx) = oneshot::channel();
        let signal = tokio::spawn(async move {
            launched_rx.await.unwrap();
            stop_tx.send(true).unwrap();
        });
        let result = timeout(
            Duration::from_secs(5),
            controller.run_job(start, stop_rx, tx, launched_tx),
        )
        .await
        .unwrap();
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("acknowledgement timed out")
        );
        assert!(controller.worker_ws.is_none());
        signal.await.unwrap();
        server.await.unwrap();
    }
}
