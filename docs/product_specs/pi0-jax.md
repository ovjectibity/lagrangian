# Pi-0 Pure JAX Implementation

## Context

Replace the lerobot/PyTorch dependency in the remote worker with a pure JAX/Flax
implementation of Pi-0. lerobot requires Python ≥ 3.12 and pulls in PyTorch (~2 GB).
The `model.safetensors` checkpoint is framework-agnostic — weights load directly
via `safetensors` and the model is implemented in JAX/Flax, already the stack
used for PPO training.

Architecture confirmed from `lerobot/pi0_libero_base` checkpoint header:
- **SigLIP**: 27-layer ViT, hidden 1152, patch 14×14, 256 tokens/image, 16 heads × head_dim 72 (full attention, no GQA)
- **PaliGemma LM backbone** (frozen): 18-layer Gemma 2B, hidden 2048, GQA 8Q/1KV heads, head_dim 256, MLP intermediate 16384
- **Action expert** (trainable): 18-layer Gemma 300M, hidden 1024, same GQA pattern, MLP intermediate 4096
- **Projections**: state/action max dim 32, expert dim 1024, time embed 2048
- **Flow matching**: 10 Euler steps, chunk_size 50, actual action dim 7 (Libero)
- **Text input**: task strings tokenised with PaliGemma tokenizer (`sentencepiece`)

---

## Files

| File | Change |
|---|---|
| `worker/trainer/trainer-py/pretrained/pi0_jax/__init__.py` | New package init |
| `worker/trainer/trainer-py/pretrained/pi0_jax/weights.py` | New: safetensors loader |
| `worker/trainer/trainer-py/pretrained/pi0_jax/siglip.py` | New: SigLIP ViT encoder |
| `worker/trainer/trainer-py/pretrained/pi0_jax/gemma.py` | New: Gemma layer (shared by backbone + expert) |
| `worker/trainer/trainer-py/pretrained/pi0_jax/model.py` | New: Pi0Jax top-level model |
| `worker/trainer/trainer-py/pretrained/pi0.py` | Update: swap PI0Policy → Pi0Jax |
| `worker/trainer/trainer-py/requirements.txt` | Add `safetensors`, `sentencepiece`; remove `torch`, `lerobot` |
| `worker/trainer/Dockerfile` | Simplify HF download (safetensors + config only, no lerobot) |

---

## Step 1 — `weights.py`

Load `model.safetensors` with `safetensors.numpy`. Map flat PyTorch key names to
nested dicts. Transpose linear weights from `[out, in]` (PyTorch) → `[in, out]` (Flax).

**Architecture dims inferred from weight shapes** — no hardcoding, no variant table:
- `input_layernorm.weight: [2048]` → LM hidden=2048
- `gate_proj.weight: [16384, 2048]` → MLP intermediate=16384
- `k_proj.weight: [256, 2048]` → KV total=256 → 1 KV head, head_dim=256
- `position_embedding.weight: [256, 1152]` → 256 patches, SigLIP hidden=1152
- `action_in_proj.weight: [1024, 32]` → action_dim=32, expert hidden=1024
- Count of `layers.{i}` keys → num_layers per sub-model

Returns `frozen_params` (siglip, lm, mm_proj, norms) and `trainable_params`
(expert, action_in_proj, action_out_proj, action_time_mlp, state_proj).

Key mapping examples:
```
model.paligemma_with_expert.paligemma.model.vision_tower
  .vision_model.encoder.layers.{i}.self_attn.q_proj.weight
→ frozen["siglip"]["layers"][i]["self_attn"]["q_proj"]["kernel"]   # transposed

model.paligemma_with_expert.paligemma.model.language_model.model.layers.{i}
  .self_attn.q_proj.weight
→ frozen["lm"]["layers"][i]["self_attn"]["q_proj"]["kernel"]

model.paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight
→ trainable["expert"]["layers"][i]["self_attn"]["q_proj"]["kernel"]

model.action_in_proj.{weight,bias}
→ trainable["action_in_proj"]["kernel"] / ["bias"]

normalize_inputs.buffer_observation_state.{mean,std}
→ frozen["norm_state"]["mean"] / ["std"]
```

---

## Step 2 — `siglip.py`

Flax `SigLIPEncoder(nn.Module)`:
- `patch_embed`: `Conv` [14×14, in=3, out=1152]
- `pos_embed`: learned `Embed` [256, 1152]
- 27 × `SigLIPLayer`: `LayerNorm` → full bidirectional MHSA (16 heads, head_dim 72) → `LayerNorm` → MLP [1152→4304 GELU→1152]
- `post_ln`: final `LayerNorm`

`__call__(images: [B, N_cam, 224, 224, 3]) → [B, N_cam*256, 1152]`

---

## Step 3 — `gemma.py`

Single `GemmaLayer(nn.Module)` parameterised by `(hidden_dim, intermediate_dim,
num_heads, num_kv_heads, head_dim)`. Shared by the 2B backbone and 300M expert.

- **Attention**: GQA; K/V broadcast from `num_kv_heads` to `num_heads` before softmax
- **RoPE**: applied to Q and K; `rope_theta=10000`; frequencies computed from `head_dim`
- **MLP**: GeGLU — `gate_proj * silu(up_proj)` → `down_proj`
- **Norm**: RMSNorm (no bias) before attention and before MLP

Causal masking: **off** for LM backbone (bidirectional over image+text+state tokens);
**on** for action token self-attention in the expert.

---

## Step 4 — `model.py`

`Pi0Jax` (not a Flax module — orchestrates frozen vs trainable params separately):

```python
class Pi0Jax:
    @classmethod
    def from_checkpoint(cls, path: str, n_action_steps: int = 10) -> Pi0Jax:
        # reads config.json for runtime params
        # calls weights.load() to get frozen_params + trainable_params + arch_config

    def get_policy_store(self) -> Params: ...    # trainable_params → PPO updates these
    def get_frozen_store(self) -> Params: ...    # frozen_params → stop_gradient in forward

    def infer(self, trainable, frozen, images, state, task_tokens, rng) -> Array:
        # Euler ODE: num_inference_steps (default 10), τ=1→0
        # Returns action [actual_action_dim] (sliced from max_action_dim=32, unnormalised)

    def forward(self, trainable, frozen, images, state, task_tokens, noised_action, t) -> Array:
        # 1. SigLIP(images) → [N_cam*256, 1152]
        # 2. mm_proj → [N_cam*256, 2048]
        # 3. embed(task_tokens) → [T_tok, 2048]
        # 4. LM backbone (bidirectional, stop_gradient on frozen) over [image+text+state tokens] → context
        # 5. time_mlp(sinusoidal(t)) → [1, 1024]
        # 6. state_proj(state_padded) → [1, 1024]
        # 7. action_in_proj(noised_action_padded) → [chunk_size, 1024]
        # 8. Expert: joint attention over (action_tokens ++ context) → [chunk_size, 1024]
        #    action tokens attend to all context; context tokens attend only to each other
        # 9. action_out_proj → velocity [chunk_size, 32]
        #    32 = max_action_dim; actual DOF (e.g. 7) is zero-padded on input,
        #    sliced on output. Allows one checkpoint to serve multiple robots.
```

**`n_action_steps`**: constructor parameter (default 10). `infer()` executes the
first `n_action_steps` of the `chunk_size=50` prediction per rollout step.
PPO loss is computed over those executed steps only.

---

## Step 5 — Update `pretrained/pi0.py`

Replace `PI0Policy` / lerobot imports with:

```python
from pretrained.pi0_jax import Pi0Jax

self._pi0 = Pi0Jax.from_checkpoint(checkpoint_dir)
self._policy_store = self._pi0.get_policy_store()
self._frozen_store  = self._pi0.get_frozen_store()
```

`get_policy_output(obs)` calls `self._pi0.infer(...)`.
`get_pure_policy_fn()` returns a closure over `forward`, JIT-able and
differentiable w.r.t. `trainable_params` only.

---

## Step 6 — `requirements.txt` and `Dockerfile`

**requirements.txt**: add `safetensors`, `sentencepiece`; remove `torch==2.10.0`, `lerobot`.

**Dockerfile**: replace the full `snapshot_download` layer with a targeted download
of only `model.safetensors` and `config.json` via `huggingface_hub.hf_hub_download`.

---

## Verification

1. `Pi0Jax.from_checkpoint(path)` loads without importing torch or lerobot
2. `infer(...)` returns action of shape `[7]` (Libero DOF) without error
3. Numerical parity check vs lerobot on a fixed (images, state, task) input before removing lerobot from the image
4. `jax.jit(forward)` compiles without error
5. `jax.grad(forward, argnums=0)` returns gradients only for trainable params
6. `test_training_remote_trossen_arm.py` (MLP path) still passes end-to-end

**Future test (tracked separately)**: standalone `Pi0Jax` inference test
(no PPO, no simulator) once VLA simulation workload exists. Verify
`from_checkpoint` + `infer` produces action in expected shape/range.
