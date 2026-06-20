# VLA Fine-Tuning Workflow (Pi-0 + RL)

## Context

Adding RL-based fine-tuning support for pre-trained VLAs (Pi-0 initially) to the Lagrangian platform. The user defines a training graph that points at a pre-trained VLA checkpoint instead of a fresh MLP policy. The remote worker loads the Pi-0 model from a bundled HuggingFace checkpoint, renders camera inputs, runs PPO using the FlowNoise log-prob method (Pi-RL paper §5), and updates only the action head and noise model — the VLM backbone stays frozen.

---

## Files to modify

| File | Change |
|---|---|
| `app/src/commondef/training.rs` | All new type definitions (see below) |
| `app/src/plugin/mod.rs` | Remove `ControlPolicyOutputMode`; import from `commondef::training` |
| `worker/trainer/Dockerfile` | Add git-lfs + clone Pi-0 checkpoint from HF |
| `worker/trainer/trainer-py/ppo_trainer.py` | Remove local `ControlPolicyOutputMode`; add FlowNoise log-prob |
| `worker/trainer/trainer-py/pretrained/pi0.py` | New `Pi0Model` class (loads Pi-0, implements PPOTrainedModel) |
| `worker/trainer/trainer-py/pretrained/__init__.py` | New empty init for the pretrained package |
| `worker/trainer/trainer-py/simulator.py` | Camera rendering for `all_cameras=True` in MJX and MuJoCo paths |
| `worker/trainer/trainer-py/main.py` | Handle `ControlModelPretrainedVLA`; skip checkpointing for Pi-0 |

Run `./scripts/gen_types.sh` after Rust changes to regenerate `client/src/common/types.ts` and `sdk/src/types.ts` (no changes to the script itself needed).

---

## Step 1 — `app/src/commondef/training.rs` type changes

### 1a. Move `ControlPolicyOutputMode` here from `plugin/mod.rs`
Add `#[typeshare]` so it appears in generated TS/Python types:
```
TargetPositions | TargetVelocities | TargetTorques
```

### 1b. Add `ControlModelInput` struct
```
ControlModelInput {
    all_cameras: bool,
    qpos_state: bool,
    qvel_state: bool,
    xpos_state: bool,
    xquat_state: bool,
}
```

### 1c. Add `VLAModel` struct
```
VLAModel {
    input: ControlModelInput,
    output: ControlPolicyOutputMode,
}
```
Constraint (validation, not type-level): when `pi0` is used, `input` must have all flags `true` and `output` must be `TargetPositions`.

### 1d. Add `PretrainedVLA` enum (only one variant for now)
```
PretrainedVLA::Pi0(VLAModel)
```

### 1e. Extend `ControlModel` enum
Add `PretrainedVLA(PretrainedVLA)` variant alongside existing `MLP` and `Checkpoint`.

### 1f. Add `FlowMatchingAction` enum
```
FlowMatchingAction::FlowNoise { noise_model: MLPModel }
```

### 1g. Add `ControlOutputMode` enum
```
ControlOutputMode::ActionMean
ControlOutputMode::FlowMatchingAction(FlowMatchingAction)
```

### 1h. Extend `PPOTrainingMethod`
Add field: `control_output_mode: ControlOutputMode`

Constraint: if `control_model` is `PretrainedVLA(Pi0(_))`, then `control_output_mode` must be `FlowMatchingAction(FlowNoise { .. })`.

---

## Step 2 — `app/src/plugin/mod.rs`

- Delete the `ControlPolicyOutputMode` enum definition (lines 43–48).
- Add `use crate::commondef::training::ControlPolicyOutputMode;` at the top of the file so all downstream uses in `ppo_trainer` (Rust local path) continue to compile.

---

## Step 3 — `worker/trainer/Dockerfile`

Add to the `apt-get install` block: `git`, `git-lfs`

After pip install, add a layer that clones the Pi-0 checkpoint into the default path:
```
RUN git lfs install \
    && git clone https://huggingface.co/lerobot/pi0_libero_base /opt/lagrangian/pi0
```

Set a `PI0_CHECKPOINT_DIR` environment variable in the Dockerfile with default value `/opt/lagrangian/pi0`. The worker code reads this env var at runtime; the container bakes in the default path but it can be overridden via `docker run -e PI0_CHECKPOINT_DIR=...`.

---

## Step 4 — `worker/trainer/trainer-py/ppo_trainer.py`

### 4a. Remove local `ControlPolicyOutputMode`
Delete the local `ControlPolicyOutputMode` Enum class (lines 60–63). Import from `commondef.training` instead (after gen_types.sh is re-run, it will be there).

### 4b. Add FlowNoise log-prob computation (Formula 5, https://arxiv.org/abs/2510.25889)

Pi-0 generates an action via a K-step denoising ODE from noise (τ=1) to clean action (τ=0). Computing `log π(a | obs)` directly requires the ODE's Jacobian — intractable. FlowNoise avoids this by treating each discrete ODE step as a Gaussian transition and summing their closed-form log-probs.

Each step from A_τ → A_{τ+δ} is modelled as:
```
A_{τ+δ} | A_τ, obs  ~  N( Euler_update(A_τ, τ),  diag(σ_φ(A_τ, obs)²) )
```
- `Euler_update` is the deterministic ODE step from Pi-0's pretrained flow head (fixed).
- `σ_φ` is the **noise model** (our learnable MLP) outputting per-dimension standard deviations.

Formula 5 sums these over all K steps:
```
log π(A | obs) = Σ_{k=0}^{K-1}  log N( A_{τ_{k+1}} ; μ_{τ_k},  diag(σ_φ(A_{τ_k}, obs)²) )
```

This sum is computed during rollout (to record `log_prob` for each trajectory step used in PPO) and is also the objective used in the backward pass to update `σ_φ`'s parameters.

### 4c. Extend `PPOTrainedModel` ABC (or leave ABC untouched and handle in subclass)
The existing ABC already has `get_control_policy_output_mode` and `get_pure_policy_fn`. No ABC changes are strictly needed — `Pi0Model` implements the interface.

---

## Step 5 — `worker/trainer/trainer-py/pretrained/pi0.py` — new `Pi0Model`

Create a new directory `worker/trainer/trainer-py/pretrained/` with `__init__.py` and `pi0.py`. Add `Pi0Model` class implementing `PPOTrainedModel`:

- **Constructor**: reads checkpoint path from `os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")`, loads Pi-0 weights, initialises the noise model (MLP with architecture from `MLPModel` spec), sets up optimisers for action head + noise model only (backbone parameters excluded from optimiser).
- **`get_policy_output(state)`**: runs Pi-0 flow matching inference + FlowNoise denoising to produce the action.
- **`get_control_policy_output_mode()`**: returns `ControlPolicyOutputMode.TargetPositions`.
- **`get_pure_policy_fn()`**: returns a JAX-traceable function over action-head + noise-model params only (backbone params are constants, not traced).
- **Backward pass**: only action head and noise model parameter stores are updated; the VLM backbone pytree is frozen (not passed to the optimiser).
- **`get_policy_store` / `get_policy_store_mut`**: return only the trainable params (action head + noise model).

---

## Step 6 — `worker/trainer/trainer-py/simulator.py`

Both `MjxSimulator` and `MujocoSimulator` need a new method (or extended `get_model_state`) to render all attached cameras when `all_cameras=True`.

**Hardcoded default camera spec (to be standardised in the training graph later):**
- Resolution: 244 × 244
- Mode: RGB (3 channels)

- **MuJoCo path**: use `mujoco.Renderer(model, height=244, width=244)` for each named camera in the model; render → (244, 244, 3) uint8 array; stack all cameras along axis 0 → shape (num_cameras, 244, 244, 3). Flatten and concatenate with the state vector before feeding to the policy.
- **MJX path**: same dimensions; use the equivalent MJX renderer call.

The PPO trainer passes `all_cameras` from `ControlModelInput` down to the simulator when building trajectories, and the resulting observation is what gets fed to `Pi0Model.get_policy_output`.

---

## Step 7 — `worker/trainer/trainer-py/main.py`

### 7a. Handle `ControlModelPretrainedVLA` in `_build_model_for_training`
After typeshare regeneration, the Python commondef will expose `ControlModelPretrainedVLA`. Add an `isinstance` branch:
```python
elif isinstance(params.control_model, ControlModelPretrainedVLA):
    # pi0 is the only variant; construct Pi0Model
    vla_spec = params.control_model.content  # Pi0(VLAModel)
    noise_mlp = params.control_output_mode.content.noise_model  # FlowMatchingAction(FlowNoise)
    from pretrained.pi0 import Pi0Model
    return Pi0Model(vla_spec, noise_mlp)
```

### 7b. Skip checkpointing for Pi-0 workloads
In the training loop / checkpoint-save path, add a guard:
```python
if isinstance(runtime.model, Pi0Model):
    pass  # skip checkpoint saving regardless of checkpoint_frequency
```

---

## Step 8 — Regenerate shared types

After all Rust changes:
```
./scripts/gen_types.sh
```
This writes updated types to both `client/src/common/types.ts` and `sdk/src/types.ts`. The Python worker `commondef/training.py` is separately generated by a typeshare-equivalent; ensure it picks up `ControlModelPretrainedVLA`, `ControlOutputMode`, `ControlModelInput`, `ControlPolicyOutputMode`.

---

## Verification

1. `cargo check` in `app/` — Rust types compile; `ControlPolicyOutputMode` resolves from `commondef::training`.
2. `./scripts/gen_types.sh` — no duplicate-name errors; new types appear in both TS output files.
3. `cd sdk && npm run build && npm run typecheck` — SDK still builds cleanly with new types.
4. Dockerfile build — Pi-0 checkpoint cloned successfully into `/opt/lagrangian/pi0`.
5. End-to-end: write a test script (mirrors `test_training_local_trossen_arm.py`) that sends a `TrainingGraph` with `control_model: PretrainedVLA(Pi0(...))`, `control_output_mode: FlowMatchingAction(FlowNoise { noise_model: ... })`, and `acceleration_mode: MLGPUSimulationGPU`, streams SSE updates, and asserts training starts without error.
