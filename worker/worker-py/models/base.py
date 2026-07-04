"""Shared Flax building blocks reused across model implementations."""
from __future__ import annotations

import flax.linen as nn
import jax

Array = jax.Array


class FlaxMLP(nn.Module):
    """Plain ReLU MLP: num_hidden_layers hidden Dense layers plus a linear head.

    Input dimensionality is fixed at init time by the shape of the sample
    input passed to `module.init`; `module.apply(params, x)` is the pure
    forward fn (batch-friendly over leading axes).
    """

    num_hidden_layers: int
    hidden_layer_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for _ in range(self.num_hidden_layers):
            x = nn.relu(nn.Dense(self.hidden_layer_dim)(x))
        return nn.Dense(self.output_dim)(x)
