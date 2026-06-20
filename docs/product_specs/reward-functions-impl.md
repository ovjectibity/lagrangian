# User-Loaded Reward Functions — Implementation

## Files to create

| File | Purpose |
|---|---|
| `docs/specs/reward-functions.md` | High-level PRD (done) |
| `app/src/api/reward_validation.rs` | Rust-side validation + compilation logic |
| `tests/test_training.py` | End-to-end integration test script |

## Files to modify

| File | Change |
|---|---|
| `app/Cargo.toml` | Add `syn` and `libloading` dependencies |
| `app/src/commondef/training.rs` | Add `RewardLanguage` enum; add `language` field to `RewardSystem` |
| `app/src/api/mod.rs` | Declare `pub mod reward_validation` |
| `app/src/api/training.rs` | Language check + validation gate before training starts; build reward closure for local path |
| `app/src/plugin/ppo_trainer.rs` | Add `user_reward_fn` to `PPOTrainer`; dispatch in `gen_trajectory` |
| `app/src/workflows/training.rs` | Accept `reward_fn` param; forward to `PPOTrainer::new` |
| `worker/trainer/trainer-py/commondef/training.py` | Add `RewardLanguage` enum; add `language` field to `RewardSystem` |
| `worker/trainer/trainer-py/main.py` | exec user reward fn after `ppo_params` extraction; pass to `TrainingWorkflow` |
| `worker/trainer/trainer-py/ppo_trainer.py` | Add `user_reward_fn` param; dispatch in `gen_trajectory_gpu` |
| `worker/trainer/trainer-py/training.py` | Accept + forward `user_reward_fn` to `PPOTrainer` |
| `client/src/common/types.ts` | Regenerate via typeshare |

---

## Step 1 — Cargo deps

`app/Cargo.toml` — add to `[dependencies]`:
```toml
syn = { version = "2", features = ["full"] }
libloading = "0.8"
```

---

## Step 2 — Type changes (Rust)

`app/src/commondef/training.rs` — add before `RewardSystem`:
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[typeshare]
pub enum RewardLanguage {
    Rust,
    Python,
}
```

Add `language` to `RewardSystem`:
```rust
pub struct RewardSystem {
    pub language: RewardLanguage,
    pub reward_function_code: String,
}
```

---

## Step 3 — `reward_validation.rs`

New file `app/src/api/reward_validation.rs`.

### `validate_and_compile_rust_reward(code: &str) -> Result<libloading::Library, String>`

1. **Syntax check**: `syn::parse_str::<syn::File>(code)` — return `Err` on parse failure.
2. **Entry-point check**: iterate `file.items`, find `syn::Item::Fn` where `fn.sig.name == "reward"` at module level. Return `Err` if not found.
3. **Import restriction**: iterate `syn::Item::Use` nodes; for each, extract the leading path segment:
   - If not `std` → `Err("external crate imports are not allowed")`
   - If `std` but second segment is `fs|net|process|env|os` → `Err("import of std::{module} is not allowed")`
4. **Compile**: write to `{tempdir}/reward.rs` the user code + injected C-ABI wrapper below. Run:
   ```
   rustc --crate-type cdylib -o {tempdir}/reward.so {tempdir}/reward.rs
   ```
   as subprocess with 30s timeout. Capture stderr; return as `Err` on non-zero exit or timeout.

   Injected C-ABI wrapper appended to the file:
   ```rust
   #[no_mangle]
   pub extern "C" fn reward_c_abi(
       prev: *const f64, prev_len: usize,
       curr: *const f64, curr_len: usize,
       act:  *const f64, act_len:  usize,
       contacts: *const (usize, usize),
       contacts_len: usize,
   ) -> f64 {
       unsafe {
           reward(
               std::slice::from_raw_parts(prev, prev_len),
               std::slice::from_raw_parts(curr, curr_len),
               std::slice::from_raw_parts(act, act_len),
               std::slice::from_raw_parts(contacts, contacts_len),
           )
       }
   }
   ```

5. **Dummy run**: load `{tempdir}/reward.so` with `libloading::Library::new`, resolve symbol `reward_c_abi`, call with empty-length null-safe slices (pass 0-length pointers). Wrap in a thread with a 30s timeout. Return `Err` on panic or timeout.
6. Return `Ok(library)`.

### `validate_python_reward(code: &str) -> Result<(), String>`

1. **Syntax check**: subprocess `python3 -c "import ast, sys; ast.parse(sys.argv[1])" -- {code}` with 30s timeout.
2. **Entry-point check**: scan lines of `code` for a line matching `^def reward(` (no leading whitespace). Return `Err` if not found.
3. **Import restriction**: subprocess that uses `ast` to walk all `Import`/`ImportFrom` nodes and validates each module against the allowlist: `{jax, numpy, math, functools, itertools, typing, dataclasses}`. Return `Err` with the blocked module name on failure.
4. **Dummy run**: subprocess:
   ```python
   import sys
   _code = sys.argv[1]
   exec(_code, {"__builtins__": __builtins__})
   # then call reward with numpy dummies
   ```
   Full template:
   ```
   python3 -c "
   import sys, numpy as np
   exec(sys.argv[1])
   result = reward(np.zeros(10), np.zeros(10), np.zeros(8), np.zeros((3,3)))
   assert isinstance(float(result), float)
   " -- {code}
   ```
   30s timeout. Return `Err` with captured stderr on failure.

---

## Step 4 — Declare module

`app/src/api/mod.rs` — add:
```rust
pub mod reward_validation;
```

---

## Step 5 — Validation gate in `start_training`

`app/src/api/training.rs`:

Add import: `use crate::commondef::training::RewardLanguage;`

After `let model_path = model_ref.path.clone();`, before `if is_local {`:

```rust
// Language vs acceleration mode check
let reward = &ppo_params.reward_function;
if is_local
    && !matches!(reward.language, RewardLanguage::Rust)
{
    return Err(Status::BadRequest);
}
if !is_local
    && !matches!(reward.language, RewardLanguage::Python)
{
    return Err(Status::BadRequest);
}

// Validate + compile
let user_reward_lib: Option<libloading::Library> =
    if is_local {
        Some(
            crate::api::reward_validation
                ::validate_and_compile_rust_reward(
                    &reward.reward_function_code,
                )
                .map_err(|e| {
                    println!(
                        "Reward validation failed: {e}"
                    );
                    Status::UnprocessableEntity
                })?,
        )
    } else {
        crate::api::reward_validation
            ::validate_python_reward(
                &reward.reward_function_code,
            )
            .map_err(|e| {
                println!(
                    "Reward validation failed: {e}"
                );
                Status::UnprocessableEntity
            })?;
        None
    };
```

Build the user reward closure for the local branch (inside `if is_local {` before the spawn_blocking call):

```rust
use std::sync::Arc as StdArc;
let reward_fn: Option<
    Box<
        dyn Fn(
            &Tensor,
            &Tensor,
            &Tensor,
            Option<&[(usize, usize)]>,
        ) -> f64
            + Send,
    >,
> = user_reward_lib.map(|lib| {
    let lib = StdArc::new(lib);
    let lib_c = lib.clone();
    Box::new(
        move |prev: &Tensor,
              curr: &Tensor,
              act: &Tensor,
              contacts: Option<&[(usize, usize)]>| {
            unsafe {
                type Fn = unsafe extern "C" fn(
                    *const f64,
                    usize,
                    *const f64,
                    usize,
                    *const f64,
                    usize,
                    *const (usize, usize),
                    usize,
                ) -> f64;
                let f: libloading::Symbol<Fn> = lib_c
                    .get(b"reward_c_abi")
                    .expect("reward_c_abi symbol");
                let prev_v: Vec<f64> =
                    Vec::try_from(prev)
                    .unwrap_or_default();
                let curr_v: Vec<f64> =
                    Vec::try_from(curr)
                    .unwrap_or_default();
                let act_v: Vec<f64> =
                    Vec::try_from(act)
                    .unwrap_or_default();
                let cs =
                    contacts.unwrap_or(&[]);
                f(
                    prev_v.as_ptr(),
                    prev_v.len(),
                    curr_v.as_ptr(),
                    curr_v.len(),
                    act_v.as_ptr(),
                    act_v.len(),
                    cs.as_ptr(),
                    cs.len(),
                )
            }
        },
    ) as Box<
        dyn Fn(
            &Tensor,
            &Tensor,
            &Tensor,
            Option<&[(usize, usize)]>,
        ) -> f64
            + Send,
    >
});
```

Pass `reward_fn` into the `spawn_blocking` closure and to `TrainingWorkflow::new`.

---

## Step 6 — `TrainingWorkflow` (Rust)

`app/src/workflows/training.rs`:

Add `tch::Tensor` import if not already present.

Update `TrainingWorkflow::new` signature to accept:
```rust
reward_fn: Option<
    Box<dyn Fn(&Tensor, &Tensor, &Tensor, Option<&[(usize, usize)]>) -> f64 + Send>
>,
```

Pass `reward_fn` to `PPOTrainer::new`.

---

## Step 7 — `PPOTrainer` (Rust)

`app/src/plugin/ppo_trainer.rs`:

Add field to `PPOTrainer`:
```rust
user_reward_fn: Option<
    Box<dyn Fn(&Tensor, &Tensor, &Tensor, Option<&[(usize, usize)]>) -> f64 + Send>
>,
```

Update `PPOTrainer::new` signature and constructor to store it.

In `gen_trajectory`, replace the single `get_reward(...)` call with:
```rust
let reward = match self.user_reward_fn {
    Some(ref f) => f(
        &cstate,
        &cstate_next,
        &sampled_action,
        Some(&contact_pairs),
    ),
    None => self
        .model
        .lock()
        .expect("Model mutex poisoned")
        .get_reward(
            &cstate,
            &cstate_next,
            &sampled_action,
            Some(&contact_pairs),
        ),
};
```

---

## Step 8 — Python type changes

`worker/trainer/trainer-py/commondef/training.py` — add before `RewardSystem`:
```python
class RewardLanguage(str, Enum):
    RUST = "Rust"
    PYTHON = "Python"
```

Update `RewardSystem`:
```python
class RewardSystem(BaseModel):
    language: RewardLanguage
    reward_function_code: str
```

---

## Step 9 — Python worker exec

`worker/trainer/trainer-py/main.py` — add top-level helper (near other `_extract_*` helpers):
```python
def _load_user_reward_fn(code: str):
    import jax.numpy as jnp
    import numpy as np
    import math
    import functools
    import itertools
    ns = {
        "jax": jax,
        "jnp": jnp,
        "numpy": np,
        "np": np,
        "math": math,
        "functools": functools,
        "itertools": itertools,
        "__builtins__": {},
    }
    exec(compile(code, "<reward_fn>", "exec"), ns)
    fn = ns.get("reward")
    if fn is None or not callable(fn):
        raise ValueError(
            "reward function not found in code"
        )
    return fn
```

In the `StartTrainingMessage` block, after `ppo_params = training_controller.training_method.content`, before `TrainingWorkflow` construction:
```python
user_reward_fn = _load_user_reward_fn(
    ppo_params.reward_function.reward_function_code
)
```

Pass `user_reward_fn` to `TrainingWorkflow(...)`.

---

## Step 10 — Python `TrainingWorkflow`

`worker/trainer/trainer-py/training.py` — add `user_reward_fn` param to `TrainingWorkflow.__init__`; store as `self._user_reward_fn`; pass to `PPOTrainer(...)`.

---

## Step 11 — Python `PPOTrainer`

`worker/trainer/trainer-py/ppo_trainer.py`:

Add `user_reward_fn` param to `PPOTrainer.__init__`; store as `self._user_reward_fn`.

In `gen_trajectory_gpu`, replace line 558:
```python
reward_fn = self.model.get_pure_reward_fn()
```
with:
```python
reward_fn = (
    self._user_reward_fn
    if self._user_reward_fn is not None
    else self.model.get_pure_reward_fn()
)
```

---

## Step 12 — typeshare + cargo check

```bash
typeshare app/src/ --lang=typescript \
    --output-file=client/src/common/types.ts

LIBTORCH=/Users/avi/Documents/projects/libtorch \
LIBTORCH_BYPASS_VERSION_CHECK=1 \
cargo check
```

---

## Step 13 — Test script (`tests/test_training.py`)

Standalone Python script. Reads `BASE_URL` from env (default `http://localhost:8000`).

Steps:
1. Upload `assets/trossen_vx300s/vx300s.xml` → get `model_id`
2. Create experiment → `exp_id`
3. Create run → `run_id`
4. Build Rust reward code string: standalone `fn reward(...)` adapting `get_reward_2` logic from `usr/arm.rs` with hardcoded vx300s body indices
5. Start training (local, `AllOnServer`, `RewardLanguage::Rust`) → assert HTTP 200; stream and print first 3 SSE updates
6. Build Python reward code string: top-level `def reward(...)` adapting `pure_reward` from `samples/arm.py` with hardcoded body indices
7. Start remote training (remote, `MLGPUSimulationGPU`, `RewardLanguage::Python`) → assert HTTP 200; stream and print first 3 SSE updates

---

## Verification

1. POST with valid Rust reward fn → HTTP 200
2. POST with Rust fn but remote mode → HTTP 400
3. POST with syntax error in Rust → HTTP 422
4. POST with Rust fn missing `reward` entry point → HTTP 422
5. POST with `use std::fs::File` in Rust → HTTP 422
6. POST with `use serde::Serialize` in Rust → HTTP 422
7. POST with valid Python fn → HTTP 200 (remote)
8. POST with `import os` in Python → HTTP 422
9. `python tests/test_training.py` runs end-to-end against live server
