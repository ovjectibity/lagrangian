# User-Loaded Reward Functions

**Scope:** Server-side (Rust) + Python remote worker. No client changes.

---

## Overview

Users supply their own reward function code as part of the training graph. The server validates, compiles (Rust) or syntax-checks (Python), and dry-runs the function before training begins, rejecting invalid functions with an appropriate HTTP error before any training infrastructure is provisioned.

---

## Language selection

`PPOTrainingMethod.reward_function` is a `RewardSystem` which now carries a `language` field:

- `Rust` — for local (`AllOnServer`) training runs
- `Python` — for remote (GPU worker) training runs

Mismatched language vs acceleration mode → HTTP 400.

---

## Validation requirements

All checks happen synchronously inside the training start API call. If any check fails the request returns HTTP 422 with a descriptive message; training does not start.

### Both languages
1. Entry point must be a **top-level** function named `reward` (not a method or nested function).
2. A **dry run** with dummy inputs must complete within **30 seconds** without error.

### Rust (`AllOnServer` only)
3. Code must be syntactically valid Rust.
4. Imports are restricted: only `std::*` paths are allowed; `std::fs`, `std::net`, `std::process`, `std::env`, `std::os` are explicitly blocked. External crate imports are rejected.
5. Code is compiled to a native dynamic library (`cdylib`) via `rustc`. Compilation failure → HTTP 422.

### Python (remote only)
3. Code must be syntactically valid Python.
4. Imports are restricted to: `jax`, `jax.numpy`, `numpy`, `math`, `functools`, `itertools`, `typing`, `dataclasses`. Any other import → HTTP 422.

---

## Reward function contract

### Rust signature
```rust
pub fn reward(
    prev_cstate: &[f64],
    curr_cstate: &[f64],
    action: &[f64],
    contact_pairs: &[(usize, usize)],
) -> f64
```

- `prev_cstate` / `curr_cstate`: xpos (nbody×3) concatenated with xquat (nbody×4), flattened, from MuJoCo's `xpos` and `xquat` arrays.
- `action`: sampled action vector from the control policy.
- `contact_pairs`: list of (body_i, body_j) index pairs with active contact in the current world state.
- Return: scalar reward as `f64`.

The user may define helper functions and use allowed standard library types. All code must be self-contained in the submitted string.

### Python signature
```python
def reward(
    prev_cstate,   # jnp.Array — xpos (nbody*3) ++ xquat (nbody*4) flattened
    curr_cstate,   # jnp.Array — same format
    action,        # jnp.Array — control policy output
    contact_matrix, # jnp.Array shape [nbody, nbody], float32, 1.0 for active contact
) -> scalar
```

The function must be **JAX-traceable** (compatible with `jit`/`vmap`). Helper functions and allowed imports may appear in the same code block.

---

## Runtime dispatch

Once validated:

- **Local path**: compiled `.so` is loaded at training start; its `reward` symbol is called in place of the hardcoded `get_reward` at each trajectory step.
- **Remote path**: the reward code string is already embedded in `training_graph` sent to the Python worker via `StartTrainingMessage`. The worker `exec`s it into a restricted namespace and replaces the model's default reward function.

The existing hardcoded reward (`get_reward_2` / `arm.py::pure_reward`) remains as a fallback when no user reward function is provided.

---

## Security considerations

- Rust user code is compiled in an isolated temp directory; no access to application source or config.
- Python user code is `exec`d inside a restricted namespace with only the allowed modules pre-populated; the system `__builtins__` are not included (or are restricted).
- The 30-second dry-run timeout prevents infinite loops from blocking the API.
