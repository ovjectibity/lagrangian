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
    patch token embeddings.
    """
    num_layers: int
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    head_dim: int
    num_patches: int   # 256 for 224×224 @ patch 14

    @nn.compact
    def __call__(self, images: Array) -> Array:
        """
        Args:
            images: [B, N_cam, H, W, 3] float32 in [0, 1]
        Returns:
            tokens: [B, N_cam * num_patches, hidden_dim]
        """
        B, N_cam, H, W, C = images.shape

        # Process all cameras in parallel by merging batch and camera dims
        x = images.reshape(B * N_cam, H, W, C)

        # Patch embedding via strided Conv (no bias in original SigLIP? — bias IS present)
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(14, 14),
            strides=(14, 14),
            padding="VALID",
            use_bias=True,
            name="patch_embed",
        )(x)                                       # [B*N, ph, pw, hidden]
        ph, pw = x.shape[1], x.shape[2]
        x = x.reshape(B * N_cam, ph * pw, self.hidden_dim)   # [B*N, T, hidden]

        # Learned position embeddings
        pos = self.param(
            "pos_embed",
            lambda rng, shape: jnp.zeros(shape),
            (self.num_patches, self.hidden_dim),
        )
        x = x + pos[None, :, :]

        # Transformer encoder layers
        for i in range(self.num_layers):
            x = _SigLIPLayer(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
                name=f"layers_{i}",
            )(x)

        x = nn.LayerNorm(name="post_ln")(x)         # [B*N, T, hidden]

        # Reshape back to separate camera axis, then flatten cameras into tokens
        x = x.reshape(B, N_cam * ph * pw, self.hidden_dim)
        return x


# ---------------------------------------------------------------------------
# Functional forward (for use without Flax module machinery)
# ---------------------------------------------------------------------------

def siglip_forward(params: Params, images: Array, arch: dict) -> Array:
    """Pure-functional SigLIP forward pass.

    Args:
        params: frozen["siglip"] pytree from weights.load()
        images: [B, N_cam, H, W, 3]
        arch:   arch_config from weights.load()
    Returns:
        [B, N_cam * num_patches, siglip_hidden]
    """
    enc = SigLIPEncoder(
        num_layers=arch["siglip_num_layers"],
        hidden_dim=arch["siglip_hidden"],
        intermediate_dim=4304,   # inferred offline from checkpoint; constant for this variant
        num_heads=arch["siglip_num_heads"],
        head_dim=arch["siglip_head_dim"],
        num_patches=arch["num_patches"],
    )
    return enc.apply({"params": _rename_params(params)}, images)


# ---------------------------------------------------------------------------
# Weight name bridge: weights.py pytree → Flax param names
# ---------------------------------------------------------------------------

def _rename_params(p: Params) -> Params:
    """Convert weights.py pytree layout to Flax module param names."""
    out: Params = {
        "patch_embed": {"kernel": p["patch_embed"]["kernel"],
                        "bias": p["patch_embed"]["bias"]},
        "post_ln": {"scale": p["post_ln"]["scale"], "bias": p["post_ln"]["bias"]},
    }
    # pos_embed is stored as a plain array in weights.py; Flax expects it under
    # the param name used in self.param() call — "pos_embed"
    out["pos_embed"] = p["pos_embed"]["embedding"]

    layers_out = {}
    for i, layer in enumerate(p["layers"]):
        layers_out[f"layers_{i}"] = {
            "ln1": {"scale": layer["ln1"]["scale"], "bias": layer["ln1"]["bias"]},
            "ln2": {"scale": layer["ln2"]["scale"], "bias": layer["ln2"]["bias"]},
            "self_attn": {
                "q_proj": {"kernel": layer["self_attn"]["q_proj"]["kernel"],
                            "bias": layer["self_attn"]["q_proj"]["bias"]},
                "k_proj": {"kernel": layer["self_attn"]["k_proj"]["kernel"],
                            "bias": layer["self_attn"]["k_proj"]["bias"]},
                "v_proj": {"kernel": layer["self_attn"]["v_proj"]["kernel"],
                            "bias": layer["self_attn"]["v_proj"]["bias"]},
                "out_proj": {"kernel": layer["self_attn"]["out_proj"]["kernel"],
                              "bias": layer["self_attn"]["out_proj"]["bias"]},
            },
            "mlp": {
                "fc1": {"kernel": layer["mlp"]["fc1"]["kernel"],
                         "bias": layer["mlp"]["fc1"]["bias"]},
                "fc2": {"kernel": layer["mlp"]["fc2"]["kernel"],
                         "bias": layer["mlp"]["fc2"]["bias"]},
            },
        }
    out.update(layers_out)
    return out
