# Phase-2.5: ControlModelInput-Driven State Construction

## Problem

Pi-0 inference on the vx300s arm crashes with a shape mismatch:

```
TypeError: sub got incompatible shapes (55,) vs (8,)
```

Root cause: `testing.py` blindly concatenates all `qpos[:nq] + qvel[:nv]` (55 dims for vx300s),
but the Pi-0 LIBERO checkpoint was trained on a Franka arm whose `norm_state` statistics have
shape `(8,)` (7 arm joints + 1 gripper). The `ControlModelInput` flags that should control which
state components are passed to the model are defined in the schema but never wired into actual
state construction.

## Goals

1. Make `Simulator` the single authority for state construction. Given a `ControlModelInput`,
   it returns labeled component tensors (`dict[str, Array]`). The caller (model class) decides
   ordering and concatenation.

2. Add Warp-backed GPU camera rendering (`mujoco-warp`) so `all_cameras` renders images
   inside `get_model_state` without CPU synchronisation.

3. Wire `ControlModelInput` through both the Python worker (testing + training) and the
   Rust local simulator path, so the spec in the control graph drives state construction end-to-end.

4. Handle the residual Franka/vx300s DOF mismatch gracefully inside Pi-0's normaliser.

## Key Design Decisions

**Labeled dict, not flat array.** `Simulator.get_model_state(input_spec)` returns
`dict[str, Array]` with keys `"cameras"`, `"qdof"`, `"qpos"`, `"qvel"`, `"xpos"`, `"xquat"`
(only for set flags). No concatenation, no ordering — kept entirely in the caller (model class).

**`qdof_state` flag.** Extracts qpos only for actuated joints (via `actuator_trnid` → `jnt_qposadr`).
For vx300s this gives 7 dims; for Franka 8 dims. VLA models expect generalised coordinates,
not Cartesian — `qdof_state` is the right flag for Pi-0.

**Model-owned assembly.** Each model class implements `assemble_state(parts)` with its own
canonical key order. Pi-0 uses `cameras → qdof → qpos → qvel → xpos → xquat`; FFN models
concatenate in dict-insertion order.

**Warp camera rendering.** `mujoco-warp` (NVIDIA Warp backend for MuJoCo) enables `mjx.render()`
to return JAX arrays directly on GPU. Camera rendering is inlined into `get_model_state` and the
`make_state_fn` closure — no separate helper, no CPU sync. `jax.vmap` over camera IDs; when used
in the MJX batched scan path, outer vmap over envs composes naturally.

**Graceful norm truncation in Pi-0.** Rather than requiring the state dim to exactly match the
checkpoint's norm buffers, we normalise `min(state_dim, norm_dim)` dimensions. This means a
7-DOF vx300s state runs correctly against an 8-dim Franka checkpoint with no padding or error.

## Scope

- Python worker: `simulator.py`, `testing.py`, `training.py`, `ppo_trainer.py`, `ppo_model.py`,
  `pretrained/pi0.py`, `pretrained/pi0_jax/model.py`, `requirements.txt`
- Rust local simulator: `app/src/plugin/mod.rs`, `app/src/plugin/mj_simulator.rs`,
  `app/src/workflows/testing.rs`, `app/src/plugin/ppo_trainer.rs`, `app/src/commondef/training.rs`
- Type generation: `scripts/gen_types.sh`
- Test script: `tests/test_testing_remote_pi0_trossen_arm.mjs`

## Implementation

### 1. `worker/worker-py/requirements.txt`

Add `mujoco-warp` (per install docs at https://mujoco.readthedocs.io/en/stable/mjwarp/index.html).
Verify CUDA version compatibility with the Docker base image (`nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04`).

---

### 2. `worker/worker-py/simulator.py`

**Precompute `_actuated_qpos_indices` at init** (both simulators):
For each actuator where `model.actuator_trntype[i] == 0`, look up `model.actuator_trnid[i, 0]`
for the joint index, then `model.jnt_qposadr[joint_idx]` for the qpos address. Include all qpos
components for that joint (hinge/slide=1, ball=4, free=7 via `model.jnt_type`).

**`get_model_state(input_spec)` returns `dict[str, Array]`** — only keys for set flags:
`"qdof"`, `"qpos"`, `"qvel"`, `"xpos"`, `"xquat"`, `"cameras"`. Fallback (no flag set): `{"qpos": ..., "qvel": ...}`.

**Camera rendering — inlined, no separate helper.** When `input_spec.all_cameras` is `True`,
uses `jax.vmap` over camera IDs via `mjx.render()` (no CPU sync):
```python
if input_spec.all_cameras and self._model.ncam > 0:
    frames = jax.vmap(
        lambda cam_id: mjx.render(self._mjx_model, self._mjx_data,
                                  camera=cam_id, width=_CAM_WIDTH, height=_CAM_HEIGHT)
    )(jnp.arange(int(self._model.ncam)))
    result["cameras"] = (frames.astype(jnp.float32) / 255.0).reshape(-1)
```

**`make_state_fn(input_spec)` → `Callable[[mjx_data], dict[str, Array]]`** (`MjxSimulator` only).
Closes over `input_spec` flags, actuated indices, and `mjx_model`. Safe inside `jax.lax.scan` and
`jax.vmap`. `all_cameras` is a Python bool captured at closure time (compile-time branch for XLA).
When the scan body calls `jax.vmap(state_fn)(bd)`, outer vmap batches over envs and inner over cameras.

`MujocoSimulator.get_model_state` uses `mujoco.Renderer` (CPU) for cameras.

---

### 3. `worker/worker-py/testing.py`

Extract `input_spec` in `__init__`:
```python
self._input_spec = (test_params.control_model.content.input
                    if isinstance(test_params.control_model, ControlModelVLA)
                    else _default_input_spec())
```

Sequential path:
```python
parts = self.simulator.get_model_state(self._input_spec)
state = self.model.assemble_state(parts)
action = self.model.get_test_action(state)
```

MJX batched scan path:
```python
state_fn = mjx_sim.make_state_fn(self._input_spec)
assemble_fn = self.model.assemble_state

def scan_body(bd, _):
    parts_batched = jax.vmap(state_fn)(bd)
    states = jax.vmap(assemble_fn)(parts_batched)
    actions = jax.vmap(lambda s: pure_fn(params, s))(states)
    bd_next = jax.vmap(lambda d, a: mjx_sim.forward_dyn_pure(d, a)[0])(bd, actions)
    return bd_next, (bd_next.xpos[:, :nbody], bd_next.xquat[:, :nbody])
```

Module-level `_default_input_spec()`: `ControlModelInput(qpos_state=True, rest False)`.

---

### 4. `worker/worker-py/pretrained/pi0_jax/model.py`

In `infer()` and `infer_chunk()`, normalise only `min(state_dim, norm_dim)` dimensions:
```python
state_dim = state.shape[0]
norm_dim = frozen["norm_state"]["mean"].shape[0]
dim = min(state_dim, norm_dim)
norm_state = self._normalize(
    state[:dim],
    {"mean": frozen["norm_state"]["mean"][:dim], "std": frozen["norm_state"]["std"][:dim]},
)
```

---

### 5. `worker/worker-py/training.py`

Extract `input_spec` from training graph's control model spec (same pattern as testing.py).
Call `get_model_state(input_spec)` and `model.assemble_state(parts)`.
No changes to `ppo_trainer.py` scan body.

---

### 6. Type regeneration + test script

Run `bash scripts/gen_types.sh` to propagate `qdof_state`.

In `tests/test_testing_remote_pi0_trossen_arm.mjs`: set `qdof_state: true`, `qpos_state: false`,
`qvel_state: false`.

---

### 7. `ppo_trainer.py` — extend `TestingModel` ABC

Add `assemble_state` abstract method:
```python
@abstractmethod
def assemble_state(self, parts: dict[str, Array]) -> Array: ...
```

`FFNPPOModel` (`ppo_model.py`):
```python
def assemble_state(self, parts):
    return jnp.concatenate(list(parts.values()))
```

`Pi0Model` (`pretrained/pi0.py`):
```python
_STATE_KEY_ORDER = ("cameras", "qdof", "qpos", "qvel", "xpos", "xquat")

def assemble_state(self, parts):
    return jnp.concatenate([parts[k] for k in self._STATE_KEY_ORDER if k in parts])
```

---

### 8. Rust local Simulator

`Simulator` trait (`plugin/mod.rs`):
```rust
fn get_model_state(&self, input_spec: &ControlModelInput) -> HashMap<String, Tensor>;
```

`MjSimulator` returns a `HashMap<String, Tensor>` keyed by flag: `"qdof"`, `"qpos"`, `"qvel"`,
`"xpos"`, `"xquat"`. No `"cameras"` key (Warp not available in Rust path).

`ControlModelInput` derives `Default` in `training.rs`.

Callers (`testing.rs:67`, `ppo_trainer.rs:181`) extract `input_spec` from the control graph and
assemble a flat tensor in model-specific key order using `Tensor::cat()`.

---

## Verification

1. `bash scripts/gen_types.sh` — no errors, `qdof_state` in generated types
2. `cargo build --manifest-path app/Cargo.toml` — no errors
3. Rebuild + push Docker image: `bash scripts/push_worker_to_ecr.sh`
4. Redeploy: `bash scripts/setup_remote_instance.sh --only-bootstrap <ip>`
5. Run `node tests/test_testing_remote_pi0_trossen_arm.mjs` with `MODEL_PATH=assets/trossen_vx300s/vx300s.xml`
6. Worker logs: state dim = 7 (vx300s actuated joints), no shape error, frames stream back
