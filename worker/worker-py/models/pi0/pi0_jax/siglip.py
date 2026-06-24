"""SigLIP ViT-So400m encoder in pure JAX/Flax.

Architecture (inferred from lerobot/pi0_libero_base checkpoint):
  - 27 transformer encoder layers
  - Hidden dim: 1152, MLP intermediate: 4304
  - Full bidirectional attention (no GQA): 16 heads × head_dim 72
  - Patch size 14×14, input 224×224 → 256 patch tokens per image
  - Learned position embeddings (no CLS token)
  - GELU activation in MLP
"""
from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

Params = dict[str, Any]
Array = jax.Array


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class _SigLIPMLP(nn.Module):
    intermediate_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.intermediate_dim, use_bias=True, name="fc1")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.hidden_dim, use_bias=True, name="fc2")(x)
        return x


class _SigLIPAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        B, T, C = x.shape
        hidden = self.num_heads * self.head_dim

        q = nn.Dense(hidden, use_bias=True, name="q_proj")(x)
        k = nn.Dense(hidden, use_bias=True, name="k_proj")(x)
        v = nn.Dense(hidden, use_bias=True, name="v_proj")(x)

        def split_heads(t: Array) -> Array:
            return t.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = self.head_dim ** -0.5
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, hidden)
        out = nn.Dense(hidden, use_bias=True, name="out_proj")(out)
        return out


class _SigLIPLayer(nn.Module):
    num_heads: int
    head_dim: int
    intermediate_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        residual = x
        x = nn.LayerNorm(name="ln1")(x)
        x = _SigLIPAttention(self.num_heads, self.head_dim, name="self_attn")(x)
        x = residual + x

        residual = x
        x = nn.LayerNorm(name="ln2")(x)
        x = _SigLIPMLP(self.intermediate_dim, self.hidden_dim, name="mlp")(x)
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class SigLIPEncoder(nn.Module):
    """SigLIP ViT image encoder.

    Processes a batch of images (potentially multi-camera) and returns
    patch token embeddings. Params are passed in via apply() and must
    match the weights.py pytree structure directly — no renaming needed.
    """
    num_heads: int
    head_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, params: Params, images: Array) -> Array:
        """
        Args:
            params: frozen["siglip"] pytree from weights.load()
            images: [B, N_cam, H, W, 3] float32 in [0, 1]
        Returns:
            tokens: [B, N_cam * num_patches, hidden_dim]
        """
        B, N_cam, H, W, C = images.shape
        x = images.reshape(B * N_cam, H, W, C)

        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(14, 14),
            strides=(14, 14),
            padding="VALID",
            use_bias=True,
        ).apply({"params": params["patch_embed"]}, x)     # [B*N, ph, pw, hidden]
        ph, pw = x.shape[1], x.shape[2]
        x = x.reshape(B * N_cam, ph * pw, self.hidden_dim)

        x = x + params["pos_embed"]["embedding"]

        layer = _SigLIPLayer(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            intermediate_dim=params["layers"]["mlp"]["fc1"]["kernel"].shape[-1],
            hidden_dim=self.hidden_dim,
        )

        def scan_fn(x, layer_p):
            return layer.apply({"params": layer_p}, x), None

        x, _ = jax.lax.scan(scan_fn, x, params["layers"])

        x = nn.LayerNorm().apply({"params": params["post_ln"]}, x)

        return x.reshape(B, N_cam * ph * pw, self.hidden_dim)


def siglip_forward(params: Params, images: Array, arch: dict) -> Array:
    enc = SigLIPEncoder(
        num_heads=arch["siglip_num_heads"],
        head_dim=arch["siglip_head_dim"],
        hidden_dim=arch["siglip_hidden"],
    )
    return enc.apply({}, params, images)
