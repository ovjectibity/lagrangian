"""Gemma decoder layer in pure JAX/Flax.

Shared by both the PaliGemma 2B LM backbone and the Gemma 300M action expert.
Parameterised at construction time so the same module handles both variants.

Key implementation details:
  - RMSNorm (no bias)
  - Grouped-query attention (GQA): K/V broadcast from num_kv_heads to num_heads
  - RoPE applied to Q and K (rope_theta=10000), not to V
  - GeGLU MLP: gate_proj * silu(up_proj) → down_proj
  - No dropout (inference + fine-tuning without it)
"""
from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

Params = dict[str, Any]
Array = jax.Array


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: Array) -> Array:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        ms = jnp.mean(x * x, axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(ms + self.eps)
        return x_norm * (1.0 + scale)   # Gemma uses (1 + scale) * x_norm


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _rope_freqs(head_dim: int, max_len: int, theta: float = 10000.0) -> Array:
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    t = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)                    # [max_len, half]
    freqs = np.concatenate([freqs, freqs], axis=-1)  # [max_len, head_dim]
    return jnp.array(freqs)


def _apply_rope(x: Array, freqs: Array) -> Array:
    """Apply RoPE to x [B, num_heads, T, head_dim]."""
    T = x.shape[2]
    f = freqs[:T]                                    # [T, head_dim]
    cos = jnp.cos(f)[None, None, :, :]
    sin = jnp.sin(f)[None, None, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# GQA attention
# ---------------------------------------------------------------------------

class GemmaAttention(nn.Module):
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @nn.compact
    def __call__(
        self,
        x: Array,
        freqs: Array,
        mask: Array | None = None,
        cross_kv: Array | None = None,
    ) -> Array:
        """
        Args:
            x:        [B, T, hidden]
            freqs:    [max_len, head_dim] RoPE frequencies
            mask:     [B, 1, T, T] additive attention mask (0 or -inf), optional
            cross_kv: if provided, K/V come from this context tensor [B, S, hidden_kv]
                      and RoPE is NOT applied to K (cross-attention mode for expert)
        """
        B, T, _ = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        q = nn.Dense(q_dim, use_bias=False, name="q_proj")(x)   # [B, T, q_dim]
        if cross_kv is not None:
            k = nn.Dense(kv_dim, use_bias=False, name="k_proj")(cross_kv)
            v = nn.Dense(kv_dim, use_bias=False, name="v_proj")(cross_kv)
        else:
            k = nn.Dense(kv_dim, use_bias=False, name="k_proj")(x)
            v = nn.Dense(kv_dim, use_bias=False, name="v_proj")(x)

        def reshape_heads(t: Array, n_heads: int) -> Array:
            B_, S, _ = t.shape
            return t.reshape(B_, S, n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = reshape_heads(q, self.num_heads)           # [B, H, T, D]
        k = reshape_heads(k, self.num_kv_heads)        # [B, Hkv, S, D]
        v = reshape_heads(v, self.num_kv_heads)

        # Apply RoPE to Q always; apply to K only for self-attention
        q = _apply_rope(q, freqs)
        if cross_kv is None:
            k = _apply_rope(k, freqs)

        # Broadcast KV heads to match Q heads
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeats, axis=1)
            v = jnp.repeat(v, repeats, axis=1)

        scale = self.head_dim ** -0.5
        attn = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale  # [B, H, T, S]
        if mask is not None:
            attn = attn + mask
        attn = nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)         # [B, H, T, D]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, q_dim)
        out = nn.Dense(x.shape[-1], use_bias=False, name="o_proj")(out)
        return out


# ---------------------------------------------------------------------------
# GeGLU MLP
# ---------------------------------------------------------------------------

class GemmaMLP(nn.Module):
    intermediate_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        gate = nn.Dense(self.intermediate_dim, use_bias=False, name="gate_proj")(x)
        up   = nn.Dense(self.intermediate_dim, use_bias=False, name="up_proj")(x)
        x = nn.silu(gate) * up
        x = nn.Dense(self.hidden_dim, use_bias=False, name="down_proj")(x)
        return x


# ---------------------------------------------------------------------------
# Full Gemma decoder layer
# ---------------------------------------------------------------------------

class GemmaLayer(nn.Module):
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(
        self,
        x: Array,
        freqs: Array,
        mask: Array | None = None,
        cross_kv: Array | None = None,
    ) -> Array:
        residual = x
        x = RMSNorm(name="input_ln")(x)
        x = GemmaAttention(
            self.num_heads, self.num_kv_heads, self.head_dim,
            name="self_attn",
        )(x, freqs, mask=mask, cross_kv=cross_kv)
        x = residual + x

        residual = x
        x = RMSNorm(name="post_attn_ln")(x)
        x = GemmaMLP(self.intermediate_dim, self.hidden_dim, name="mlp")(x)
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# Causal mask helper
# ---------------------------------------------------------------------------

def causal_mask(T: int, dtype=jnp.float32) -> Array:
    """Upper-triangular mask: [1, 1, T, T], -inf above diagonal."""
    mask = jnp.triu(jnp.full((T, T), -1e9, dtype=dtype), k=1)
    return mask[None, None, :, :]


# ---------------------------------------------------------------------------
# Functional forward helpers
# ---------------------------------------------------------------------------

def lm_forward(
    params: Params,
    token_embeds: Array,
    freqs: Array,
    arch: dict,
) -> Array:
    """Run frozen LM backbone (bidirectional, no causal mask).

    Args:
        params:       frozen["lm"] pytree
        token_embeds: [B, T, lm_hidden] — image + text tokens already projected
        freqs:        RoPE frequencies [max_len, head_dim]
        arch:         arch_config from weights.load()
    Returns:
        context: [B, T, lm_hidden]
    """
    x = token_embeds
    num_layers = arch["lm_num_layers"]
    for i in range(num_layers):
        layer = GemmaLayer(
            num_heads=arch["lm_num_heads"],
            num_kv_heads=arch["lm_num_kv_heads"],
            head_dim=arch["lm_head_dim"],
            intermediate_dim=arch["lm_intermediate"],
            hidden_dim=arch["lm_hidden"],
            name=f"layer_{i}",
        )
        lparams = _layer_params(params["layers"][i])
        x = layer.apply({"params": lparams}, x, freqs, mask=None)

    # Final RMSNorm
    norm = RMSNorm(name="norm")
    x = norm.apply({"params": params["norm"]}, x)
    return x


def expert_forward(
    params: Params,
    action_tokens: Array,
    context: Array,
    freqs: Array,
    arch: dict,
    n_action_tokens: int,
) -> Array:
    """Run trainable action expert with joint attention.

    Action tokens attend to all context tokens (cross-attention style within
    the joint sequence). Context tokens are treated as read-only prefix.

    Args:
        params:          trainable["expert"] pytree
        action_tokens:   [B, A, expert_hidden]  (A = chunk_size + 1 state)
        context:         [B, S, lm_hidden] — LM output (stop_gradient applied by caller)
        freqs:           RoPE frequencies [max_len, head_dim]
        arch:            arch_config from weights.load()
        n_action_tokens: number of action token positions (for causal mask shape)
    Returns:
        [B, A, expert_hidden]
    """
    # Project context from lm_hidden → expert_hidden for cross-attention KV
    # (handled in model.py before calling here; context passed already in expert_hidden)
    x = action_tokens
    A = x.shape[1]
    mask = causal_mask(A)   # action tokens are causally masked within themselves

    num_layers = arch["expert_num_layers"]
    for i in range(num_layers):
        layer = GemmaLayer(
            num_heads=arch["expert_num_heads"],
            num_kv_heads=arch["expert_num_kv_heads"],
            head_dim=arch["expert_head_dim"],
            intermediate_dim=arch["expert_intermediate"],
            hidden_dim=arch["expert_hidden"],
            name=f"layer_{i}",
        )
        lparams = _layer_params(params["layers"][i])
        # Self-attention over action tokens (causal), then cross-attend to context
        x = layer.apply({"params": lparams}, x, freqs, mask=mask, cross_kv=context)

    norm = RMSNorm(name="norm")
    x = norm.apply({"params": params["norm"]}, x)
    return x


# ---------------------------------------------------------------------------
# Param name bridge: weights.py pytree → Flax module names
# ---------------------------------------------------------------------------

def _layer_params(layer: Params) -> Params:
    """Convert weights.py layer dict to Flax GemmaLayer param names."""
    return {
        "input_ln": layer["input_ln"],
        "post_attn_ln": layer["post_attn_ln"],
        "self_attn": {
            "q_proj": layer["self_attn"]["q_proj"],
            "k_proj": layer["self_attn"]["k_proj"],
            "v_proj": layer["self_attn"]["v_proj"],
            "o_proj": layer["self_attn"]["o_proj"],
        },
        "mlp": layer["mlp"],
    }
