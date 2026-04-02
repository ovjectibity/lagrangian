# Experiments & Runs Implementation Spec

## Context
`instructions/specs/experiments.md` defines the product requirements for the experiments & runs layer. The ORM groundwork (`PostgresOrm`, `MongoOrm`) and UI components (`trainingexperimentstable.svelte`, `experimentruncardlayout.svelte`, `experimentruncard.svelte`, `breadcrumb.svelte`, `playpausecta.svelte`) were already in place. This document describes what was implemented to wire them together end-to-end.

**Scope exclusions (deferred):**
- Remote worker code paths — no changes to the remote worker/EC2 branch of `StartTrainingMessage`
- Checkpoint implementation — `checkpoints` field remains but no checkpoint logic added

**Decisions:**
- Active context: in-memory only (`Arc<AsyncMutex<ActiveContext>>`), not persisted across restarts
- Type sync: hand-edited `client/src/commondef/messages.ts` to match new Rust structs

---

## Backend changes

### New WebSocket message types
**File:** `app/src/commondef/clientmessages.rs`

Added to `ServerMessage` enum (from browser → server):
- `CreateExperimentMessage` — create a new experiment and load it
- `CreateRunMessage(CreateRunMessage)` — create a new run under the active experiment
- `LoadRunMessage(LoadRunMessage)` — load an existing run

Added to `ClientMessage` enum (from server → browser):
- `ExperimentCreatedMessage(ExperimentCreatedMessage)` — sent after a new experiment is created
- `RunCreatedMessage(RunCreatedMessage)` — sent after a new run is created
- `RunLoadedMessage(RunLoadedMessage)` — sent after a run is loaded
- `ActiveContextMessage(ActiveContextMessage)` — sent on WS connect with last active experiment + run

New message structs:
```rust
pub struct CreateRunMessage {
    pub training_experiment_id: String,
}
pub struct LoadRunMessage {
    pub training_experiment_id: String,
    pub run_id: String,
}
pub struct ExperimentCreatedMessage {
    pub experiment: TrainingExperiment,
}
pub struct RunCreatedMessage {
    pub run: TrainingExperimentRun,
}
pub struct RunLoadedMessage {
    pub run: TrainingExperimentRun,
}
pub struct ActiveContextMessage {
    pub experiment: TrainingExperiment,
    pub run: Option<TrainingExperimentRun>,
}
```

### ORM globals + active context
**File:** `app/src/main.rs`

Three new `OnceLock`-backed globals:
```rust
static POSTGRES_ORM: OnceLock<Arc<PostgresOrm>>
static MONGO_ORM: OnceLock<Arc<MongoOrm>>
static ACTIVE_CONTEXT: OnceLock<Arc<AsyncMutex<ActiveContext>>>
```

`ActiveContext` holds `experiment_id: Option<String>` and `run_id: Option<String>` — in-memory only, reset on server restart.

Both ORMs are initialised in `rocket()` from env vars `DATABASE_URL` and `MONGODB_URI` via `tokio::runtime::Handle::current().block_on(...)`.

Project is hardcoded as `PROJECT_ID = "1"`.

### WebSocket message handlers
**File:** `app/src/main.rs` — `workloads_channel`

**On WS connect:** reads `ACTIVE_CONTEXT`; if an experiment is active, loads the experiment + run from Postgres and sends `ActiveContextMessage` to the client before entering the event loop.

**`CreateExperimentMessage`:**
1. Generates a UUID for `training_experiment_id`
2. Calls `postgres_orm().create_experiment(PROJECT_ID, &new_id, &[])`
3. Updates `ACTIVE_CONTEXT` (new experiment, no run)
4. Sends `ExperimentCreatedMessage`

**`CreateRunMessage`:**
1. Generates UUIDs for `run_id`, `logs_id`, `metrics_id`, `training_graph_id`
2. Calls `postgres_orm().create_run(...)` with `TrainingRunStatus::NotStarted`
3. Calls `postgres_orm().add_run_to_experiment(...)`
4. Updates `ACTIVE_CONTEXT`
5. Sends `RunCreatedMessage`

**`LoadRunMessage`:**
1. Fetches run from Postgres via `postgres_orm().get_run(...)`
2. Updates `ACTIVE_CONTEXT`
3. Sends `RunLoadedMessage`

**`StartTrainingMessage` (local AllOnServer path — extended):**
- After spawning the training task, marks run as `Ongoing` in Postgres
- Spawns a periodic Tokio task (30 s interval) that drains accumulated `(metrics, logs)` from a shared `Arc<AsyncMutex<PendingFlush>>` buffer into MongoDB via `mongo_orm().append_metrics(...)` / `mongo_orm().append_logs(...)`
- Each `TrainingUpdatesMessage` received from the training worker pushes its metrics/logs into the buffer

**`StopTrainingMessage` (local path — extended):**
- Marks run as `Completed` in Postgres
- Drains remaining buffer to MongoDB (final flush)

### New dependency
**File:** `app/Cargo.toml`

```toml
uuid = { version = "1", features = ["v4"] }
```

---

## Frontend changes

### New TypeScript message types
**File:** `client/src/commondef/messages.ts`

Added interfaces: `CreateRunMessage`, `LoadRunMessage`, `ExperimentCreatedMessage`, `RunCreatedMessage`, `RunLoadedMessage`, `ActiveContextMessage`.

Extended `ClientMessage` union with the four new inbound message types.

Extended `ServerMessage` union with the three new outbound message types.

### App-level state and wiring
**File:** `client/src/App.svelte`

New state:
- `activeExperiment: TrainingExperiment | null` — the currently loaded experiment
- `activeRun: TrainingExperimentRun | null` — the currently loaded run
- `trainingGraphEdited: boolean` — set to `true` when training graph or reward fn is edited; controls split-button visibility

New WS message handlers in `onmessage`:
- `ActiveContextMessage` → sets `activeExperiment` and `activeRun`
- `ExperimentCreatedMessage` → sets `activeExperiment`, clears `activeRun`
- `RunCreatedMessage` → sets `activeRun`
- `RunLoadedMessage` → sets `activeRun`; resets metrics and logs

New functions: `createExperiment`, `createRun`, `loadRun`, `startNewRun`.

`Breadcrumb` now receives `experimentName` and `runName` from live state.

`PlayPauseCta` receives `showNewRunOption` and `onStartNewRun` for the split-button behaviour.

The `training-experiments` tab now passes `experiments`, `activeExperimentId`, `onCreateExperiment`, `onCreateRun`, `onLoadRun` as props.

### Experiments table
**File:** `client/src/lib/trainingexperimentstable.svelte`

- Replaced hardcoded dummy data with `experiments` prop
- Added `Create & load new experiment` toolbar button (calls `onCreateExperiment`)
- Passes `experimentId`, `onCreateRun`, `onLoadRun` through to `ExperimentRunCardLayout`

### Run card layout
**File:** `client/src/lib/experimentruncardlayout.svelte`

- Added `experimentId`, `onCreateRun`, `onLoadRun` props
- Added `Create & load new run` button in the header
- Passes `experimentId` and `onLoadRun` to each `ExperimentRunCard`

### Run card
**File:** `client/src/lib/experimentruncard.svelte`

- Added `experimentId` prop
- `onLoadRun` signature changed to `(expId: string, runId: string) => void`
- "Load run" button calls `onLoadRun(experimentId, run.run_id)`

### Play/pause CTA — split button
**File:** `client/src/lib/playpausecta.svelte`

- Added `showNewRunOption` and `onStartNewRun` props
- When `showNewRunOption` is true and not running: renders a split button — the main area triggers `onToggle` (Start training), a chevron `▾` button opens a dropdown with "Start new run" (triggers `onStartNewRun`)
- Dropdown closes on outside click

### Styles
**File:** `client/src/app.css`

Added styles for: `.playpause-split`, `.playpause-split-main`, `.playpause-split-chevron`, `.playpause-dropdown`, `.playpause-dropdown-item`, `.experiments-table-toolbar`, `.experiments-table-cta`.

---

## Critical files

| File | Change |
|---|---|
| `app/Cargo.toml` | Added `uuid` dep |
| `app/src/commondef/clientmessages.rs` | New message types + structs |
| `app/src/main.rs` | ORM globals, init, new WS message handlers, periodic Mongo flush |
| `client/src/commondef/messages.ts` | New TS message types |
| `client/src/App.svelte` | Active experiment/run state, handlers, prop forwarding |
| `client/src/lib/trainingexperimentstable.svelte` | Props, removed dummy data, added CTAs |
| `client/src/lib/experimentruncardlayout.svelte` | Added "Create & load new run" CTA |
| `client/src/lib/experimentruncard.svelte` | Updated `onLoadRun` signature |
| `client/src/lib/playpausecta.svelte` | Split-button for "Start new run" |
| `client/src/app.css` | Styles for new UI elements |

---

## Verification
- `cargo check` in `app/`
- `npm run build` in `client/`
- Manual end-to-end: open app → breadcrumb shows "Not set / Not set" → click "Create & load new experiment" → breadcrumb experiment name updates → click "Create & load new run" → breadcrumb run name updates → click "Start training" → Postgres run row status = `ongoing`
