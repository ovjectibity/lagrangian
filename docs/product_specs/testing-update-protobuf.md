# Testing updates: protobuf binary wire format

Status: implemented. Single combined spec + implementation doc.

## Problem

The testing/simulation update stream was broken in three ways:

1. The app server exploded each `TestUpdateMessage` into one binary frame **per
   `ModelForward`** (`ClientMessage::to_binary_simul_update`), losing the
   `[env][t]` matrix structure.
2. **Camera frames were dropped** — the relay never forwarded `frames`.
3. The binary layout was ad-hoc and undocumented; the SDK hand-rolled a
   `DataView` decoder.

## Decision

Adopt **Protocol Buffers** as the schema-defined wire format for the testing
update, **end-to-end binary** (Python worker → Rust app server → TS SDK). Scope is
**testing updates only**; training updates stay JSON text over WebSocket.

The wire-contract artifact is [`docs/user/specs/testing_updates.proto`](../user/specs/testing_updates.proto).
It is the source of truth for the binary format and mirrors the Rust
`app/src/commondef/testing.rs` types (kept in parity by convention).

## Architecture — Rust is an opaque pass-through

A worker test connection carries **data updates** (protobuf binary frames) and
**control** (JSON text frames, e.g. `StopTesting`). The app server distinguishes
them by WebSocket frame type (`Message::Binary` vs `Message::Text`):

- **Binary** frames are forwarded to SDK subscribers **verbatim** — the server
  never decodes the protobuf and never copies the (heavy) camera bytes.
- **Text** frames are the small JSON control messages, parsed as today.

Consequences:

- **The Rust server takes no protobuf dependency** (no `prost`, no `build.rs`).
  Protobuf codegen lands only in the Python worker (producer) and the TS SDK
  (consumer).
- Zero re-encode / zero camera-byte copies on the server hot path.
- *Future:* a Rust protobuf dependency may be added later when we add **saving
  testing updates & frames to disk** (debugging / replays), where the server
  would need to decode `TestingStreamMessage` to persist it. Out of scope here.

### Binary envelope / type identifier

Binary data frames are wrapped in a top-level `TestingStreamMessage` envelope. Its
`oneof message` arm is the type identifier (`test_update = 1`); future binary
message kinds are added as new arms (2, 3, …). The SDK decodes the envelope and
switches on the set field.

## Schema summary

See the `.proto` for the authoritative definition. Shapes:

- `ModelForward { int64 nbody; repeated double xpos; repeated double xquat }` —
  `nbody` is carried explicitly (redundant: `len(xpos)/3`).
- `EnvUpdates { repeated ModelForward updates }` — one env's time-sequence
  (wrapper, since protobuf has no nested `repeated repeated`).
- `CameraFrames { int64 nenv, ncam, t, h, w; bytes frames }` — dims first, the
  flat `u8` blob last; logical shape `[nenv, t, ncam, h, w, 3]` row-major.
- `TestUpdate { string id; repeated EnvUpdates simulation_state; optional CameraFrames frames }`.
- `TestingStreamMessage { oneof message { TestUpdate test_update = 1 } }`.

## Implementation

### Codegen

- Python: `python -m grpc_tools.protoc` (via `grpcio-tools`) →
  `worker/worker-py/commondef/testing_pb2.py`, imported as
  `from commondef import testing_pb2`. Driven by `scripts/gen_proto.sh`.
- SDK: `protobufjs-cli` (`pbjs` + `pbts`, pure JS — no `protoc`) →
  `sdk/src/proto/testing.js` + `.d.ts`. Driven by the `gen:proto` npm script,
  run before `tsup` in `build`.
- Runtime deps: `protobuf` (worker, in `requirements.txt`), `protobufjs` (SDK).

### Python worker (producer)

- `workflows/testing.py`: `_run_mjx_batched` / `_run_sequential` build a
  `testing_pb2.TestUpdate` directly from arrays — poses as `repeated double`
  (with `nbody = len(xpos)//3`), cameras as one `bytes` via `np.ascontiguousarray(
  frames_u8).tobytes()` plus `nenv/ncam/t/h/w`. Camera bytes never pass through
  the Pydantic `CameraFrames` (whose typeshare model is `List[int]`). The
  `TestUpdate` is wrapped in `TestingStreamMessage` and `.SerializeToString()`
  yields the raw wire bytes.
- `TestingTaskSink.send_test_updates(payload: bytes)` enqueues those bytes;
  `send_stop_testing` stays JSON.
- `main.py` `_sender_loop`: if a queued payload is `bytes`, send a binary WS
  frame; otherwise `json.dumps` a text frame (as before).

### Rust app server (relay)

- `workflows/common_worker_controller.rs` `monitor_testing_until_stop`: new
  `Message::Binary(bytes)` arm forwards bytes via the `on_update` callback; the
  `Message::Text` arm still handles the JSON `StopTesting`.
- `run_test_job`'s callback is `FnMut(Vec<u8>)` (the envelope frame).
- `commondef/workermessages.rs` `TestTaskMessage::TestUpdateBytes(Vec<u8>)`.
- `api/testing.rs`: the relay broadcasts the bytes directly; the per-`ModelForward`
  loop and `ClientMessage::to_binary_simul_update` are deleted. The SDK-facing
  WS already sends `ws::Message::Binary`.
- `commondef/testing.rs` `CameraFrames` fields made `pub`.

### TypeScript SDK (consumer)

- `resources/simulations.ts`: decode each frame with
  `TestingStreamMessage.decode(new Uint8Array(buf))`, take `.test_update`, and map
  to the **exact typeshare `TestUpdateMessage`**:
  - `simulation_state`: `EnvUpdates[] → ModelForward[][]` (`.map(e => e.updates)`,
    dropping the proto-only `nbody`).
  - `frames`: protobuf `CameraFrames` → typeshare `CameraFrames` via
    `Array.from(frames)` and `Number(dim)`.
  - Tradeoff: `Array.from` materializes camera bytes as a `number[]` (heavier in
    JS than `Uint8Array`); chosen for a clean end-user type. The wire stays
    efficient (protobuf bytes).

## Documentation

- Wire contract: [`docs/user/specs/testing_updates.proto`](../user/specs/testing_updates.proto).
- AsyncAPI (JSON): [`docs/user/specs/testing_updates_asyncapi.json`](../user/specs/testing_updates_asyncapi.json)
  documents the `/testing/<id>/updates` WS channel referencing the `.proto`
  (`schemaFormat: application/vnd.google.protobuf`). OpenAPI/utoipa cannot model a
  binary WS stream.

## Out of scope

- Training updates (stay JSON text).
- The deprecated Svelte `client/`.
- On-disk frame storage (separate decision).
