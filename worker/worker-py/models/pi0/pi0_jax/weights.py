"""
Load Pi-0 weights from model.safetensors into frozen/trainable Flax pytree dicts.

PyTorch linear weights are [out, in]; Flax expects [in, out] ("kernel").
All transpositions happen here so the rest of the code uses Flax conventions.
Architecture hyperparams are inferred from weight shapes — no hardcoding.
"""
from __future__ import annotations

import gc
import re
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.flax import load_file
import jax
import jax.numpy as jnp
from warnings import deprecated

import logging
_logger = logging.getLogger(__name__)

Params = dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _del_prefix_key_from_map(flat,key_prefix): 
    for key in list(flat.keys()): 
        if key.startswith(key_prefix):
            del flat[key]

@deprecated("Use load_lazy instead to reduce memory duplications")
def load(safetensors_path: str) -> tuple[Params, Params, dict[str, Any]]:
    """Load Pi-0 weights and return (frozen_params, trainable_params, arch_config).

    arch_config holds dims inferred from weight shapes:
        siglip_hidden, siglip_num_layers, siglip_num_heads, siglip_head_dim,
        lm_hidden, lm_num_layers, lm_num_heads, lm_num_kv_heads, lm_head_dim, lm_intermediate,
        expert_hidden, expert_num_layers, expert_num_heads, expert_num_kv_heads,
        expert_head_dim, expert_intermediate,
        action_dim, state_dim, time_embed_dim, num_patches, mm_proj_out,
    """
    # Loading over the GPU by default
    flat = load_file(safetensors_path)

    frozen: Params = {}
    trainable: Params = {}

    # SigLIP vision tower
    frozen["siglip"] = _load_siglip(flat)
    _del_prefix_key_from_map(flat,"model.paligemma_with_expert.paligemma.model.vision_tower.vision_model")
    _del_prefix_key_from_map(flat,"model.paligemma_with_expert.embeddings")
    print(flat.keys())

    # Multi-modal projector
    frozen["mm_proj"] = _load_mm_proj(flat)
    # LM backbone
    frozen["lm"] = _load_lm(flat)
    _del_prefix_key_from_map(flat,"model.paligemma_with_expert.paligemma.model.language_model")
    print(flat.keys())
    # print(flat.keys())
    # del flat["lm"]
    # State/action normalisation buffers
    frozen["norm_state"] = _load_norm(flat, "normalize_inputs.buffer_observation_state")
    frozen["norm_action"] = _load_norm(flat, "normalize_targets.buffer_action")
    frozen["unnorm_action"] = _load_norm(flat, "unnormalize_outputs.buffer_action")

    # Action expert (trainable)
    trainable["expert"] = _load_expert(flat)
    _del_prefix_key_from_map(flat,"model.paligemma_with_expert.gemma_expert.model")
    print(flat.keys())
    # print(flat.keys())
    # Small projections (trainable)
    trainable["state_proj"] = _load_linear(flat, "model.state_proj")
    trainable["action_in_proj"] = _load_linear(flat, "model.action_in_proj")
    trainable["action_out_proj"] = _load_linear(flat, "model.action_out_proj")
    trainable["action_time_mlp"] = {
        "in": _load_linear(flat, "model.action_time_mlp_in"),
        "out": _load_linear(flat, "model.action_time_mlp_out"),
    }

    #Give a log of the devices the loaded Pi-0 Pytree is on
    device_map_frozen = jax.tree.map(lambda node: node.devices(),frozen)
    _logger.info(device_map_frozen)
    device_map_trainable = jax.tree.map(lambda node: node.devices(),trainable)
    _logger.info(device_map_trainable)

    arch = _infer_arch(flat)
    return frozen, trainable, arch


class _LazySafetensors:
    """Read-only mapping over an open safetensors handle that loads on demand.

    ``__getitem__`` pulls a single tensor from the file as a torch tensor, hands
    it to :func:`torch_to_jax` (which converts it and drops + GCs the source),
    and returns the JAX array. Keys come from the file header, so membership and
    iteration are available without reading any tensor data. The handle must
    stay open (i.e. inside its ``with`` block) for the lifetime of this object.
    """

    def __init__(self, handle: Any):
        self._f = handle
        self._keys: set[str] = set(handle.keys())

    def __getitem__(self, key: str) -> jax.Array:
        if key not in self._keys:
            raise KeyError(key)
        return torch_to_jax(self._f.get_tensor(key))

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def keys(self):
        return set(self._keys)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._keys else default

    def __delitem__(self, key: str) -> None:
        # Drops the key from the known set so it is never loaded; no data held.
        self._keys.discard(key)


def load_lazy(safetensors_path: str) -> tuple[Params, Params, dict[str, Any]]:
    """Lazily load Pi-0 weights tensor-by-tensor.

    Same return contract as :func:`load`, but instead of reading every tensor
    into memory up front, each tensor is pulled from the file only when a
    sub-loader asks for it, converted to a JAX array, and the source torch
    tensor is released (see :func:`torch_to_jax`). The sub-loaders are reused
    unchanged via the :class:`_LazySafetensors` mapping.
    """
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        flat = _LazySafetensors(handle)

        frozen: Params = {}
        trainable: Params = {}

        # SigLIP vision tower
        frozen["siglip"] = _load_siglip(flat)
        # _del_prefix_key_from_map(flat, f"{_PG}.paligemma.model.vision_tower.vision_model")
        # _del_prefix_key_from_map(flat, f"{_PG}.embeddings")

        # Multi-modal projector
        frozen["mm_proj"] = _load_mm_proj(flat)
        # LM backbone
        frozen["lm"] = _load_lm(flat)
        # _del_prefix_key_from_map(flat, f"{_PG}.paligemma.model.language_model")

        # State/action normalisation buffers
        frozen["norm_state"] = _load_norm(flat, "normalize_inputs.buffer_observation_state")
        frozen["norm_action"] = _load_norm(flat, "normalize_targets.buffer_action")
        frozen["unnorm_action"] = _load_norm(flat, "unnormalize_outputs.buffer_action")

        # Action expert (trainable)
        trainable["expert"] = _load_expert(flat)
        # _del_prefix_key_from_map(flat, f"{_PG}.gemma_expert.model")

        # Small projections (trainable)
        trainable["state_proj"] = _load_linear(flat, "model.state_proj")
        trainable["action_in_proj"] = _load_linear(flat, "model.action_in_proj")
        trainable["action_out_proj"] = _load_linear(flat, "model.action_out_proj")
        trainable["action_time_mlp"] = {
            "in": _load_linear(flat, "model.action_time_mlp_in"),
            "out": _load_linear(flat, "model.action_time_mlp_out"),
        }

        arch = _infer_arch(flat)

    # Built on CPU tensor-by-tensor; move the finished pytrees to GPU in one pass.
    gpu = jax.devices()[0]
    frozen = jax.device_put(frozen, gpu)
    trainable = jax.device_put(trainable, gpu)

    return frozen, trainable, arch

# ---------------------------------------------------------------------------
# Sub-loaders
# ---------------------------------------------------------------------------

_PG = "model.paligemma_with_expert"


def _load_siglip(flat: dict) -> Params:
    prefix = f"{_PG}.paligemma.model.vision_tower.vision_model"
    emb = f"{prefix}.embeddings"
    out: Params = {}

    # Patch embedding (Conv2d in PyTorch → stored as [out, in, kH, kW])
    # Flax Conv stores kernel as [kH, kW, in, out]
    w = flat[f"{emb}.patch_embedding.weight"]   # [1152, 3, 14, 14]
    out["patch_embed"] = {
        "kernel": np.transpose(w, (2, 3, 1, 0)),  # [14, 14, 3, 1152]
        "bias": flat[f"{emb}.patch_embedding.bias"],
    }
    out["pos_embed"] = {"embedding": flat[f"{emb}.position_embedding.weight"]}

    enc_prefix = f"{prefix}.encoder.layers"
    num_layers = _count_layers(flat, enc_prefix)
    layers = []
    for i in range(num_layers):
        p = f"{enc_prefix}.{i}"
        layers.append({
            "ln1": _load_ln(flat, f"{p}.layer_norm1"),
            "ln2": _load_ln(flat, f"{p}.layer_norm2"),
            "self_attn": {
                "q_proj": _load_linear_bias(flat, f"{p}.self_attn.q_proj"),
                "k_proj": _load_linear_bias(flat, f"{p}.self_attn.k_proj"),
                "v_proj": _load_linear_bias(flat, f"{p}.self_attn.v_proj"),
                "out_proj": _load_linear_bias(flat, f"{p}.self_attn.out_proj"),
            },
            "mlp": {
                "fc1": _load_linear_bias(flat, f"{p}.mlp.fc1"),
                "fc2": _load_linear_bias(flat, f"{p}.mlp.fc2"),
            },
        })
    out["layers"] = _stack_layers(layers)
    out["post_ln"] = _load_ln(flat, f"{prefix}.post_layernorm")
    return out


def _load_mm_proj(flat: dict) -> Params:
    prefix = f"{_PG}.paligemma.model.multi_modal_projector.linear"
    return _load_linear_bias(flat, prefix)


def _load_lm(flat: dict) -> Params:
    prefix = f"{_PG}.paligemma.model.language_model"
    out: Params = {}

    embed_key = f"{prefix}.embed_tokens.weight"
    lm_head_key = f"{_PG}.paligemma.lm_head.weight"

    if embed_key in flat:
        out["embed_tokens"] = {"embedding": flat[embed_key]}
    elif lm_head_key in flat:
        # Weight-tied checkpoint: lm_head and embed_tokens share the same matrix
        out["embed_tokens"] = {"embedding": flat[lm_head_key]}

    out["norm"] = _load_rms(flat, f"{prefix}.norm")

    layers_prefix = f"{prefix}.layers"
    num_layers = _count_layers(flat, layers_prefix)
    out["layers"] = _stack_layers([_load_gemma_layer(flat, f"{layers_prefix}.{i}") for i in range(num_layers)])

    if lm_head_key in flat:
        out["lm_head"] = {"kernel": flat[lm_head_key].T}
    return out


def _load_expert(flat: dict) -> Params:
    prefix = f"{_PG}.gemma_expert.model"
    out: Params = {}
    out["norm"] = _load_rms(flat, f"{prefix}.norm")
    layers_prefix = f"{prefix}.layers"
    num_layers = _count_layers(flat, layers_prefix)
    out["layers"] = _stack_layers([_load_gemma_layer(flat, f"{layers_prefix}.{i}") for i in range(num_layers)])
    return out


def _load_gemma_layer(flat: dict, prefix: str) -> Params:
    return {
        "input_ln": _load_rms(flat, f"{prefix}.input_layernorm"),
        "post_attn_ln": _load_rms(flat, f"{prefix}.post_attention_layernorm"),
        "self_attn": {
            "q_proj": _load_linear(flat, f"{prefix}.self_attn.q_proj"),
            "k_proj": _load_linear(flat, f"{prefix}.self_attn.k_proj"),
            "v_proj": _load_linear(flat, f"{prefix}.self_attn.v_proj"),
            "o_proj": _load_linear(flat, f"{prefix}.self_attn.o_proj"),
        },
        "mlp": {
            "gate_proj": _load_linear(flat, f"{prefix}.mlp.gate_proj"),
            "up_proj": _load_linear(flat, f"{prefix}.mlp.up_proj"),
            "down_proj": _load_linear(flat, f"{prefix}.mlp.down_proj"),
        },
    }


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def torch_to_jax(tensor: Any) -> jax.Array:
    """Convert a PyTorch tensor to a JAX array and release the source tensor.

    Uses DLPack so JAX consumes the tensor's buffer directly — no numpy detour
    (which can't represent bfloat16) and no second allocation. DLPack carries
    the dtype, so bf16 stays bf16. The local reference is then dropped and
    gc.collect() run to reclaim the torch wrapper promptly. The caller must not
    retain its own reference to ``tensor`` if it wants the storage freed.
    """
    arr = jax.dlpack.from_dlpack(tensor.detach())
    del tensor
    gc.collect()
    return arr


def _stack_layers(layers: list[Params]) -> Params:
    """Stack a list of per-layer param dicts into a single pytree with a leading [num_layers, ...] dim on every leaf."""
    def _stack(vals: list) -> Any:
        if isinstance(vals[0], dict):
            return {k: _stack([v[k] for v in vals]) for k in vals[0]}
        return jnp.stack(vals)
    return _stack(layers)


def _load_linear(flat: dict, prefix: str) -> Params:
    return {"kernel": flat[f"{prefix}.weight"].T}


def _load_linear_bias(flat: dict, prefix: str) -> Params:
    return {"kernel": flat[f"{prefix}.weight"].T, "bias": flat[f"{prefix}.bias"]}


def _load_ln(flat: dict, prefix: str) -> Params:
    return {"scale": flat[f"{prefix}.weight"], "bias": flat[f"{prefix}.bias"]}


def _load_rms(flat: dict, prefix: str) -> Params:
    return {"scale": flat[f"{prefix}.weight"]}


def _load_norm(flat: dict, prefix: str) -> Params:
    return {
        "mean": flat.get(f"{prefix}.mean", np.zeros(1, dtype=np.float32)),
        "std": flat.get(f"{prefix}.std", np.ones(1, dtype=np.float32)),
    }


def _count_layers(flat: dict, prefix: str) -> int:
    indices = set()
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    for k in flat:
        m = pattern.match(k)
        if m:
            indices.add(int(m.group(1)))
    return max(indices) + 1 if indices else 0


# ---------------------------------------------------------------------------
# Architecture inference
# ---------------------------------------------------------------------------

def _infer_arch(flat: dict) -> dict[str, Any]:
    arch: dict[str, Any] = {}

    # SigLIP
    pos_key = f"{_PG}.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    if pos_key in flat:
        num_patches, siglip_hidden = flat[pos_key].shape
        arch["siglip_hidden"] = siglip_hidden
        arch["num_patches"] = num_patches
    enc_prefix = f"{_PG}.paligemma.model.vision_tower.vision_model.encoder.layers"
    arch["siglip_num_layers"] = _count_layers(flat, enc_prefix)
    qk = f"{enc_prefix}.0.self_attn.q_proj.weight"
    if qk in flat:
        out_dim, in_dim = flat[qk].shape   # [hidden, hidden] for full attention
        # head_dim inferred as hidden // num_heads; num_heads from q/k match
        kk = f"{enc_prefix}.0.self_attn.k_proj.weight"
        k_out, _ = flat[kk].shape
        # full attention: q_out == k_out == hidden
        siglip_h = in_dim
        # find num_heads from common head_dim 72 or compute
        arch["siglip_num_heads"] = siglip_h // 72 if siglip_h % 72 == 0 else siglip_h // 64
        arch["siglip_head_dim"] = siglip_h // arch["siglip_num_heads"]

    # LM backbone
    lm_prefix = f"{_PG}.paligemma.model.language_model.layers"
    arch["lm_num_layers"] = _count_layers(flat, lm_prefix)
    ln_key = f"{lm_prefix}.0.input_layernorm.weight"
    if ln_key in flat:
        arch["lm_hidden"] = flat[ln_key].shape[0]
    gate_key = f"{lm_prefix}.0.mlp.gate_proj.weight"
    if gate_key in flat:
        arch["lm_intermediate"] = flat[gate_key].shape[0]
    q_key = f"{lm_prefix}.0.self_attn.q_proj.weight"
    k_key = f"{lm_prefix}.0.self_attn.k_proj.weight"
    if q_key in flat and k_key in flat:
        q_out, q_in = flat[q_key].shape
        k_out, _ = flat[k_key].shape
        lm_h = q_in
        lm_head_dim = k_out  # = num_kv_heads * head_dim; since num_kv_heads=1 → head_dim=k_out
        arch["lm_head_dim"] = lm_head_dim
        arch["lm_num_heads"] = q_out // lm_head_dim
        arch["lm_num_kv_heads"] = k_out // lm_head_dim

    # Action expert
    exp_prefix = f"{_PG}.gemma_expert.model.layers"
    arch["expert_num_layers"] = _count_layers(flat, exp_prefix)
    eln_key = f"{exp_prefix}.0.input_layernorm.weight"
    if eln_key in flat:
        arch["expert_hidden"] = flat[eln_key].shape[0]
    egate_key = f"{exp_prefix}.0.mlp.gate_proj.weight"
    if egate_key in flat:
        arch["expert_intermediate"] = flat[egate_key].shape[0]
    eq_key = f"{exp_prefix}.0.self_attn.q_proj.weight"
    ek_key = f"{exp_prefix}.0.self_attn.k_proj.weight"
    if eq_key in flat and ek_key in flat:
        eq_out, _ = flat[eq_key].shape
        ek_out, _ = flat[ek_key].shape
        exp_head_dim = arch.get("lm_head_dim", 256)  # same head_dim as LM
        arch["expert_head_dim"] = exp_head_dim
        arch["expert_num_heads"] = eq_out // exp_head_dim
        arch["expert_num_kv_heads"] = ek_out // exp_head_dim

    # Action / state dims from projection weights
    ain_key = "model.action_in_proj.weight"
    if ain_key in flat:
        expert_h, action_dim = flat[ain_key].shape
        arch["action_dim"] = action_dim
    sp_key = "model.state_proj.weight"
    if sp_key in flat:
        _, state_dim = flat[sp_key].shape
        arch["state_dim"] = state_dim

    # Time embed dim from MLP in weight
    tmlp_key = "model.action_time_mlp_in.weight"
    if tmlp_key in flat:
        _, time_embed_dim = flat[tmlp_key].shape
        arch["time_embed_dim"] = time_embed_dim

    # MM projector output dim
    mm_key = f"{_PG}.paligemma.model.multi_modal_projector.linear.weight"
    if mm_key in flat:
        arch["mm_proj_out"] = flat[mm_key].shape[0]

    return arch
