# TypeScript SDK — Spec

## Overview

The Lagrangian TypeScript SDK (`lagrangian-sdk`) is a standalone npm package that provides a fully-typed client for the Lagrangian REST + SSE API. It enables programmatic access to every platform capability: model management, training experiment lifecycle, real-time event streaming, checkpoint retrieval, and simulation control.

## Goals

- Cover every API endpoint with a typed method (no raw fetch calls needed by consumers)
- Expose SSE streams as async iterables with typed event payloads
- Work in browsers (via native fetch) and Node.js 18+ without polyfills
- Publish as both ESM and CommonJS so any consumer toolchain can use it
- Ship TypeScript declaration files so IDEs provide full autocomplete

## Non-Goals

- Authentication / auth token management (the current API has no auth layer)
- Retry / backoff logic (left to the consumer)
- React/Vue/Svelte bindings (plain TypeScript only)
- WebSocket support (the API only uses SSE)

## Target Consumers

- CI/CD pipelines that need to trigger and monitor training runs programmatically
- Research scripts that automate experiment creation and checkpoint retrieval
- The existing Svelte frontend (can adopt the SDK instead of ad-hoc fetch calls)
- External tooling authors building on top of the Lagrangian platform

## Platform Support

| Environment | Minimum version |
|---|---|
| Browsers | Any with native `fetch` + `ReadableStream` (Chrome 76+, Firefox 65+, Safari 14+) |
| Node.js | 18.0+ (ships native `fetch`) |
| TypeScript | 5.0+ |
| Module format | ESM (primary), CommonJS (compatibility) |

---

## Distribution

The SDK lives at `sdk/` in the monorepo and is published separately as `lagrangian-sdk` on npm.

Build output (`sdk/dist/`):
- `index.js` — ESM bundle
- `index.cjs` — CommonJS bundle
- `index.d.ts` — TypeScript declarations
- Source maps for both bundles

---

## Package Layout

```
sdk/
├── package.json       # name: lagrangian-sdk, exports ESM + CJS + types
├── tsconfig.json      # strict, ESNext, moduleResolution: bundler
├── tsup.config.ts     # entry: src/index.ts, format: [esm, cjs], dts: true
└── src/
    ├── index.ts                  # public barrel export
    ├── types.ts                  # typeshare-generated types (see Type Strategy)
    ├── errors.ts                 # LagrangianError
    ├── sse.ts                    # streamSSE<T>() async generator
    ├── client.ts                 # LagrangianClient root class
    └── resources/
        ├── models.ts             # ModelsResource
        ├── experiments.ts        # ExperimentsResource
        ├── checkpoints.ts        # CheckpointsResource
        └── simulations.ts        # SimulationsResource
```

---

## Type Strategy

All shared types come from the Rust backend via `typeshare`. The command is run via `scripts/gen_types.sh`, which outputs to two locations:

- `client/src/common/types.ts` — for the Svelte frontend (existing)
- `sdk/src/types.ts` — for the SDK (new canonical location for the package)

Both files are identical typeshare output. The SDK bundles its own copy so it has no dependency on the `client/` package.

---

## Error Handling

All non-2xx HTTP responses throw a `LagrangianError` with two properties:

| Property | Type | Description |
|---|---|---|
| `status` | `number` | HTTP status code |
| `message` | `string` | Response body text |

Consumers should catch `LagrangianError` and branch on `status` (e.g. 422 for validation failure, 404 for not found).

---

## Client Initialisation

```
new LagrangianClient(config)
```

| Config field | Type | Required | Description |
|---|---|---|---|
| `baseUrl` | `string` | Yes | Server origin, e.g. `http://localhost:8000/api` |
| `headers` | `Record<string, string>` | No | Extra headers added to every request |

The constructor normalises `baseUrl` (strips trailing slash) and exposes four resource namespaces: `models`, `experiments`, `checkpoints`, `simulations`.

---

## Resource API Design

### `client.models`

| Method | Endpoint | Description |
|---|---|---|
| `list()` → `Promise<ModelReference[]>` | GET /models | All saved MJCF models |
| `getDefault()` → `Promise<{ id: string; model: WorldBody }>` | GET /models/default | Default model from env |
| `upload(file: Blob, filename: string)` → `Promise<{ id: string }>` | POST /models/save_model | Upload XML or zip; returns new model ID |

`upload` sends a multipart form with a `file` field. When uploading a directory of assets (XML + STL meshes), callers should zip them first — the server extracts and registers the root XML.

---

### `client.experiments`

| Method | Endpoint | Description |
|---|---|---|
| `list()` → `Promise<TrainingExperiment[]>` | GET /training/experiments | All experiments |
| `create(body?)` → `Promise<TrainingExperiment>` | POST /training/experiments | Create experiment; `body.name` is optional |
| `listRuns(experimentId)` → `Promise<TrainingExperimentRun[]>` | GET /training/experiments/{id}/runs | Runs within an experiment |
| `createRun(experimentId)` → `Promise<TrainingExperimentRun>` | POST /training/experiments/{id}/runs | New run |
| `getRun(experimentId, runId)` → `Promise<TrainingExperimentRun>` | GET /training/experiments/{id}/runs/{runId} | Single run |
| `getRunCheckpoints(experimentId, runId)` → `Promise<CheckpointReference[]>` | GET /training/experiments/{id}/runs/{runId}/checkpoints | Checkpoints for a run |
| `startTraining(experimentId, runId, graph: TrainingGraph)` → `Promise<{ id: string }>` | POST /training/experiments/{id}/runs/{runId}/start | Start training; validates reward fn before returning |
| `stopTraining(experimentId, runId)` → `Promise<{ id: string }>` | POST /training/experiments/{id}/runs/{runId}/stop | Stop in-progress run |
| `streamTrainingUpdates(experimentId, runId)` → `AsyncIterable<ClientMessage>` | GET /training/experiments/{id}/runs/{runId}/updates | SSE stream of training events |

`startTraining` accepts a `TrainingGraph` (the `training_job_controller` wrapper). The server validates and compiles the reward function synchronously before responding 200, so a successful response means training has started.

`streamTrainingUpdates` yields `ClientMessage` tagged-union events. Consumers should handle:
- `{ type: "TrainingUpdatesMessage", content: TrainingUpdatesMessage }` — epoch/update metrics and logs
- `{ type: "StopTrainingMessage", content: StopTrainingMessage }` — training complete or stopped

The async iterable terminates naturally when the SSE stream closes (training ends).

---

### `client.checkpoints`

| Method | Endpoint | Description |
|---|---|---|
| `list()` → `Promise<CheckpointReference[]>` | GET /checkpoints | All checkpoints |
| `get(checkpointId)` → `Promise<CheckpointReference>` | GET /checkpoints/{id} | Single checkpoint |

---

### `client.simulations`

| Method | Endpoint | Description |
|---|---|---|
| `start(params: SimulationParams)` → `Promise<{ id: string }>` | POST /testing/simulations | Start a simulation; returns simulation ID |
| `stop(simulationId)` → `Promise<{ id: string }>` | POST /testing/simulations/{id}/stop | Stop simulation |
| `streamUpdates(simulationId)` → `AsyncIterable<SimulUpdateMessage>` | GET /testing/simulations/{id}/updates | SSE stream of simulation frames |

`streamUpdates` yields `SimulUpdateMessage` objects where each `update` contains `xpos` and `xquat` arrays for each body in the model.

---

## SSE Streaming Design

Both `streamTrainingUpdates` and `streamUpdates` use the same underlying mechanism: a native `fetch` request with `Accept: text/event-stream`, reading the response body as a `ReadableStream<Uint8Array>`. The stream is decoded incrementally, split on SSE event boundaries (`\n\n`), and each `data:` line is JSON-parsed into the typed payload.

This approach works in browsers and Node 18+ without any additional dependencies. The generator terminates when the server closes the connection or the consumer breaks out of the `for await` loop.

---

## Usage Examples (prose)

**Start a training run and stream updates:**
Create a client, upload a model zip, create an experiment, create a run, call `startTraining` with a `TrainingGraph`, then iterate over `streamTrainingUpdates`. Break when a `StopTrainingMessage` is received.

**Poll experiment runs:**
Call `client.experiments.listRuns(expId)` periodically and inspect `run.run_status` to detect completion without SSE.

**Load a checkpoint:**
After training, call `client.experiments.getRunCheckpoints(expId, runId)` then `client.checkpoints.get(checkpointId)` to retrieve checkpoint metadata and the base64-encoded model zip.

---

## Verification

1. `cd sdk && npm install && npm run build` — produces `dist/index.js`, `dist/index.cjs`, `dist/index.d.ts`
2. `npm run typecheck` — zero TypeScript errors
3. Against a running server: create a client, upload a model, create an experiment + run, start training, collect 3 SSE events — mirrors `tests/test_training_trossen_arm.py`
