# Pi-0 Architecture Fix: Joint Sequence + KV Cache

## Context

The current Pi-0 JAX implementation uses a cross-attention design that does not match the paper or the checkpoint weights. The action expert receives the LM's final output as `cross_kv` and computes K/V from it — but the expert's `k_proj` kernel has shape `(1024, 256)` (designed for 1024-dim Expert tokens), not the 2048-dim LM output being passed. This crashes immediately with `ScopeParamShapeError`.

Beyond the crash, the design is architecturally wrong in four additional ways documented below. This spec covers the correct architecture (per the Pi-0 paper §3 and checkpoint weight shapes) and the code changes required.

---

## Current Implementation and Why It Is Wrong

`Pi0Jax.forward()` in `model.py` runs a **two-stage pipeline with cross-attention**:

1. `_run_lm()` runs all 18 LM layers → returns final output `context [B, S, 2048]`
2. `_run_expert()` loops 18 Expert layers, each receiving that same `context` as `cross_kv`. Inside `GemmaAttention`, when `cross_kv` is set, K and V come entirely from the context tensor — Expert tokens never attend to each other.

**Five problems:**

**1 — Dimension mismatch → crash.** Expert `k_proj` kernel `(1024, 256)` receives 2048-dim input → `ScopeParamShapeError`. No cross-attention weights exist in the checkpoint.

**2 — All 18 Expert layers use the same LM final output.** Per-layer LM representations are discarded. The correct design pairs Expert layer `i` with LM layer `i`'s K/V.

**3 — No self-attention among Expert tokens; wrong action mask.** The paper states all-to-all bidirectional attention within the action block. Current code applies a causal upper-triangle mask — wrong.

**4 — τ embedded into state token, blocking caching.** Current: `state_tok = state_proj(state) + t_emb(τ)`. τ changes each denoising step, making state non-cacheable. τ belongs on action tokens, not state.

**5 — Wrong τ formula for action tokens.** `action_time_mlp.in.kernel` shape `(2048, 1024)` = `(2w, w)` was designed to receive `[W_1·a (1024) ‖ φ(τ) (1024)]` concatenated. The current code feeds a 2048-dim sinusoidal of τ alone through this layer (dimensionally accidental) and adds the result to `state_tok`. The paper (§3) specifies:
```
W_3 · swish(W_2 · concat(W_1 · a_{t',τ}, φ(τ)))
```
where φ: ℝ → ℝ^w is a 1024-dim sinusoidal positional encoding of τ.

---

## Correct Architecture

### Joint sequence processing

LM (hidden=2048) and Expert (hidden=1024) run **in lockstep** across 18 layers. At layer `i`:
- LM layer `i` processes `[B, S, 2048]` → produces K_lm_i / V_lm_i in 256-dim space
- Expert layer `i` processes `[B, A, 1024]` → Q_exp attends over `[K_lm_i ‖ K_exp_i]`

Each backbone applies its own k/v_proj to its own tokens:
- LM: `k_proj (2048→256)` → K_lm_i `[B, 1, S, 256]`
- Expert: `k_proj (1024→256)` → K_exp_i `[B, 1, A, 256]`

Both land in 256-dim — directly concatenable. Both use MQA (1 KV head, 8 Q heads, head_dim=256).

### Block-causal mask (paper §3)

```
                    LM tokens [0:S]     state [S]      actions [S+1:S+A]
LM tokens [0:S]      ✓ bidir             ✗              ✗
state [S]            ✓                   ✓              ✗  ← cannot see actions
actions [S+1:S+A]    ✓                   ✓              ✓ bidir (NOT causal within)
```

Expert mask `[A, S+A]`:
```python
mask = jnp.zeros((A, S + A), dtype=jnp.float32)
mask = mask.at[0, S + 1:].set(-1e9)   # state row: cannot see action columns
mask = mask[None, None, :, :]
```

### KV caching

LM tokens never attend to Expert tokens → LM hidden states are independent of the noised action. Cache once per inference call.

- `encode_prefix()`: runs 18 LM layers, returns stacked `(Ks, Vs)` of shape `[18, B, 1, S, 256]`
- Per τ step (10×): runs 18 Expert layers only using the cached `(Ks, Vs)`

State token `state_proj(state)` has no τ — its K/V is constant but NOT separately pre-cached in this change (re-computed each τ step; future optimization).

### τ formula for action tokens

Checkpoint weight shapes confirm the 3-weight MLP:
- W_1 = `action_in_proj` kernel `(32, 1024)` — projects action dim → expert_hidden
- W_2 = `action_time_mlp.in` kernel `(2048, 1024)` — takes `[W_1·a ‖ φ(τ)]` (2w) → w
- W_3 = `action_time_mlp.out` kernel `(1024, 1024)` — final projection

φ(τ) = 1024-dim sinusoidal (half=512 cos frequencies + half sin).

---

## Code Changes

### `worker/worker-py/models/pi0/pi0_jax/gemma.py`

**`GemmaAttention.__call__`**: Replace `cross_kv: Array | None` with `prefix_kv: tuple[Array, Array] | None`.

K/V always come from `x` (self-attention). When `prefix_kv=(K_prefix, V_prefix)` is provided, prepend before dot-product:
```python
q = nn.Dense(q_dim,  use_bias=False, name="q_proj")(x)
k = nn.Dense(kv_dim, use_bias=False, name="k_proj")(x)
v = nn.Dense(kv_dim, use_bias=False, name="v_proj")(x)

q = _apply_rope(reshape_heads(q, num_heads),    freqs)
k = _apply_rope(reshape_heads(k, num_kv_heads), freqs)
v = reshape_heads(v, num_kv_heads)

if prefix_kv is not None:
    K_prefix, V_prefix = prefix_kv          # [B, 1, S, 256]
    k = jnp.concatenate([K_prefix, k], axis=2)
    v = jnp.concatenate([V_prefix, v], axis=2)
```

**`GemmaLayer.__call__`**: rename `cross_kv` → `prefix_kv`, thread to `GemmaAttention`.

**Remove `expert_forward()`** — dead code. Keep `lm_forward()` unchanged.

---

### `worker/worker-py/models/pi0/pi0_jax/model.py`

#### `_rms_norm_apply()` static helper
```python
@staticmethod
def _rms_norm_apply(norm_params: Params, x: Array) -> Array:
    scale = norm_params["scale"]
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + 1e-6) * (1.0 + scale)
```

#### `_run_lm_and_cache()` via `jax.lax.scan`
Stack all 18 layer param dicts into a single pytree, scan over them. Each step returns `(K_roped, V)` as output → accumulated to `(Ks [18,B,1,S,256], Vs [18,B,1,S,256])`.

```python
stacked = jax.tree_util.tree_map(
    lambda *xs: jnp.stack(xs), *[_layer_params(lp) for lp in lm_params["layers"]]
)

def lm_layer(x, layer_p):
    # ... attention + MLP residual blocks ...
    return x, (K_roped, V)

final_x, (Ks, Vs) = jax.lax.scan(lm_layer, x, stacked)
```

#### `encode_prefix()` public method
```python
def encode_prefix(self, frozen, images_b, tokens_b) -> tuple[Array, Array]:
    # SigLIP → mm_proj → embed_tokens → concat → _run_lm_and_cache
    return Ks, Vs   # [18, B, 1, S, 256] each
```

#### `_run_expert_with_cache()` via `jax.lax.scan`
Stack expert params alongside `(Ks, Vs)`; scan 18 layers. Mask is built once and closed over.

```python
def expert_layer(x, inputs):
    layer_p, K_prefix, V_prefix = inputs
    x = GemmaLayer(...).apply({"params": layer_p}, x, freqs,
                               mask=mask, prefix_kv=(K_prefix, V_prefix))
    return x, None

final_x, _ = jax.lax.scan(expert_layer, expert_in, (stacked, Ks, Vs))
```

#### `_action_embed()` — replace `_time_embed()`
```python
def _action_embed(self, trainable, na, t):
    w, half = self._arch["expert_hidden"], self._arch["expert_hidden"] // 2
    freqs   = jnp.exp(-jnp.arange(half, dtype=jnp.float32) * (jnp.log(10000.0) / (half - 1)))
    t_arr   = jnp.array(t, dtype=jnp.float32)
    phi_tau = jnp.concatenate([jnp.cos(t_arr * freqs), jnp.sin(t_arr * freqs)])  # [w]

    W1_a  = _dense_apply(trainable["action_in_proj"]["kernel"],
                         trainable["action_in_proj"].get("bias"), na)   # [B, chunk, w]
    phi_b = jnp.broadcast_to(phi_tau[None, None, :], W1_a.shape)
    h     = jnp.concatenate([W1_a, phi_b], axis=-1)                     # [B, chunk, 2w]

    mlp = trainable["action_time_mlp"]
    h   = jax.nn.silu(_dense_apply(mlp["in"]["kernel"],  mlp["in"].get("bias"),  h))
    return _dense_apply(         mlp["out"]["kernel"], mlp["out"].get("bias"), h)
```

#### `forward()` — new signature
```python
def forward(self, trainable, lm_kv, state_b, noised_action, t):
    state_tok  = _dense_apply(trainable["state_proj"]["kernel"], ..., state_b[:, None, :])
    na         = jnp.broadcast_to(noised_action[:, None, :], (B, chunk_size, max_action_dim))
    action_tok = self._action_embed(trainable, na, t)
    expert_out = self._run_expert_with_cache(trainable["expert"],
                     jnp.concatenate([state_tok, action_tok], axis=1), lm_kv)
    return _dense_apply(trainable["action_out_proj"]["kernel"], ..., expert_out[:, 1:, :])
```

#### `infer()` / `infer_chunk()` — `jax.lax.scan` denoising loop
```python
lm_kv = self.encode_prefix(frozen_sg, images_b, tokens_b)
ts    = 1.0 - jnp.arange(num_inference_steps, dtype=jnp.float32) * dt

def denoise_step(x_b, t):
    vel = self.forward(trainable, lm_kv, state_b, x_b, t)
    return x_b[0] - dt * vel[0, 0, :][None], vel

x_final, all_vels = jax.lax.scan(denoise_step, x_b, ts)
```

---

## Verification

1. Push `gemma.py` + `model.py` to container on EC2 `44.197.202.146`, restart
2. Run `node tests/test_testing_remote_pi0_trossen_arm.mjs`
3. Expected: no `ScopeParamShapeError`, forward pass completes, actions stream back
