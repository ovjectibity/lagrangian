# API Specification: REST + SSE

All endpoints are under the `/api` prefix.

---

## Experiments & Runs

| Endpoint | Method | Response body | Notes |
|---|---|---|---|
| `/training/experiments` | GET | `{ experiments: TrainingExperiment[] }` | Replaces proactive `ExperimentsListMessage` on WS connect |
| `/training/experiments` | POST | `{ experiment: TrainingExperiment }` | Replaces `CreateExperimentMessage` + `ExperimentCreatedMessage` |
| `/training/experiments/:experiment_id/runs` | GET | `{ runs: TrainingExperimentRun[] }` | Replaces `ExperimentRunsMessage` |
| `/training/experiments/:experiment_id/runs` | POST | `{ run: TrainingExperimentRun }` | Replaces `CreateRunMessage` + `RunCreatedMessage` |
| `/training/experiments/:experiment_id/runs/:run_id` | GET | `{ run: TrainingExperimentRun }` | Replaces `LoadRunMessage` (stateless; no server-side context change) |
| `/training/experiments/:experiment_id/runs/:run_id/checkpoints` | GET | `{ checkpoints: CheckpointReference[] }` | Replaces `CheckpointsForRunMessage` |

---

## Training

| Endpoint | Method | Body / Response | Notes |
|---|---|---|---|
| `/training/experiments/:experiment_id/runs/:run_id/start` | POST | Body: `{ training_graph: TrainingGraph }` → `{ id: string }` | Replaces `StartTrainingMessage`. Server uses default vx300 model. |
| `/training/experiments/:experiment_id/runs/:run_id/stop` | POST | `{ id: string }` | Replaces `StopTrainingMessage` (both directions) |
| `/training/experiments/:experiment_id/runs/:run_id/updates` | GET (SSE) | SSE stream of `TrainingUpdate` events | Replaces `TrainingUpdatesMessage`. Scoped to `run_id`. Stream closes when training stops. |

---

## Simulations

| Endpoint | Method | Body / Response | Notes |
|---|---|---|---|
| `/testing/simulations` | POST | Body: `{ simulation_params: SimulationParams }` → `{ id: string }` | Replaces `StartSimulationMessage`. Server uses default vx300 model. |
| `/testing/simulations/:id/stop` | POST | `{ id: string }` | Replaces `StopSimulationMessage` (both directions) |
| `/testing/simulations/:id/updates` | GET (SSE) | SSE stream of binary frames | Replaces `SimulUpdateMessage`. Binary payload is base64-encoded in the SSE `data` field. Client: `Uint8Array.from(atob(event.data), c => c.charCodeAt(0))` → ArrayBuffer. |

---

## Models

| Endpoint | Method | Response | Notes |
|---|---|---|---|
| `/models/default` | GET | `{ id: string, model: WorldBody }` | Replaces `LoadModelMessage` + `SetModelMessage`. Hardcoded to vx300. Called once on client startup. |

---

## Checkpoints

| Endpoint | Method | Response | Notes |
|---|---|---|---|
| `/checkpoints` | GET | `{ checkpoints: CheckpointReference[] }` | Replaces `GetAllCheckpointsMessage` + `AllCheckpointsMessage` |
| `/checkpoints/:id` | GET | `{ checkpoint: CheckpointReference }` | Replaces `GetCheckpointInfoMessage` + `CheckpointInfoMessage` |

---

## Dropped / removed

| Item | Reason |
|---|---|
| `ActiveContextMessage` | Never sent by server; active context is client-side state only |
| `LoadRunMessage` (WS) | Replaced by stateless `GET .../runs/:run_id` |
| `SetModelMessage` | Legacy raw-command path; replaced by `GET /models/default` |
| `ExperimentsListMessage` (proactive push) | Client calls `GET /training/experiments` on startup |
| `RunLoadedMessage` | Becomes the response body of `GET .../runs/:run_id` |
| Global `ACTIVE_CONTEXT` server state | No server-side session state in REST model |
