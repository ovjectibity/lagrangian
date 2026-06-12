from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ppo_trainer import ControlPolicyOutputMode, PPOTrainedModel

Array = jax.Array
Params = Any


class FFNPPOModel(PPOTrainedModel):
    def __init__(
        self,
        policy: Callable[[Params, Array], Array],
        value: Callable[[Params, Array], Array],
        policy_store: Params,
        value_store: Params,
        nbody: int = 0,
        reward_fn: Callable | None = None,
    ) -> None:
        self.control_output_mode = ControlPolicyOutputMode.TARGETPOSITIONS
        self.policy = policy
        self.value = value
        self.policy_store = policy_store
        self.value_store = value_store
        self.nbody = nbody
        self._reward_fn = reward_fn

    @staticmethod
    def _init_linear_params(key: Array, in_dim: int, out_dim: int) -> dict[str, Array]:
        lim = np.sqrt(6.0 / (in_dim + out_dim))
        w_key, _ = jax.random.split(key)
        w = jax.random.uniform(
            w_key,
            shape=(in_dim, out_dim),
            minval=-lim,
            maxval=lim,
            dtype=jnp.float32,
        )
        b = jnp.zeros((out_dim,), dtype=jnp.float32)
        return {"w": w, "b": b}

    @classmethod
    def init_policy(
        cls,
        input_dimension: int,
        output_dimension: int,
        hidden_layer_dimension: int,
        num_layers: int,
        add_softmax: bool,
        control_policy: bool,
        seed: int = 0,
    ) -> tuple[Callable[[Params, Array], Array], Params]:
        if control_policy and output_dimension % 2 != 0:
            raise ValueError(
                f"Control policy output dim must be even, got {output_dimension}"
            )
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, num_layers + 1)
        layers: list[dict[str, Array]] = []
        layers.append(
            cls._init_linear_params(keys[0], input_dimension, hidden_layer_dimension)
        )
        for i in range(max(0, num_layers - 1)):
            layers.append(
                cls._init_linear_params(keys[i + 1], hidden_layer_dimension, hidden_layer_dimension)
            )
        layers.append(
            cls._init_linear_params(keys[-1], hidden_layer_dimension, output_dimension)
        )
        params = {"layers": layers}

        def policy_fn(policy_params: Params, state: Array) -> Array:
            x = jnp.asarray(state, dtype=jnp.float32)
            for lyr in policy_params["layers"][:-1]:
                x = jnp.matmul(x, lyr["w"]) + lyr["b"]
                x = jax.nn.relu(x)
            out = (
                jnp.matmul(x, policy_params["layers"][-1]["w"])
                + policy_params["layers"][-1]["b"]
            )
            if control_policy:
                action_dim = output_dimension // 2
                mean = jnp.tanh(out[..., :action_dim]) * 0.01
                log_std = jnp.clip(out[..., action_dim:], -5.2, -4.2)
                out = jnp.concatenate([mean, log_std], axis=-1)
            if add_softmax:
                out = jax.nn.softmax(jax.nn.relu(out), axis=-1)
            return out

        return policy_fn, params

    @classmethod
    def init_ppo_trained_model(
        cls,
        policy: Callable[[Params, Array], Array],
        val_estimator: Callable[[Params, Array], Array],
        pol_store: Params,
        val_estimator_store: Params,
        nbody: int = 0,
        reward_fn: Callable | None = None,
    ) -> FFNPPOModel:
        return cls(
            policy=policy,
            value=val_estimator,
            policy_store=pol_store,
            value_store=val_estimator_store,
            nbody=nbody,
            reward_fn=reward_fn,
        )

    def get_reward(
        self,
        cstate: Array,
        cstate_next: Array,
        action: Array,
        contact_pairs: list[tuple[int, int]] | None,
    ) -> float:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into FFNPPOModel")
        contact_matrix = np.zeros((self.nbody, self.nbody), dtype=np.float32)
        if contact_pairs is not None:
            for a, b in contact_pairs:
                if a < self.nbody and b < self.nbody:
                    contact_matrix[a, b] = 1.0
                    contact_matrix[b, a] = 1.0
        return float(self._reward_fn(cstate, cstate_next, action, jnp.asarray(contact_matrix)))

    def get_policy_output(self, state: Array) -> Array:
        return self.policy(self.policy_store, state)

    def get_value_output(self, state: Array) -> float:
        val = self.value(self.value_store, state)
        val_arr = np.asarray(jax.device_get(val), dtype=np.float64).reshape(-1)
        if val_arr.size == 0:
            raise ValueError("Expected non-empty value output")
        return float(val_arr[0])

    def get_policy(self) -> Callable[[Params, Array], Array]:
        return self.policy

    def get_value_func(self) -> Callable[[Params, Array], Array]:
        return self.value

    def get_policy_store(self) -> Params:
        return self.policy_store

    def get_policy_store_mut(self) -> Params:
        return self.policy_store

    def get_value_store(self) -> Params:
        return self.value_store

    def get_value_store_mut(self) -> Params:
        return self.value_store

    def get_control_policy_output_mode(self) -> ControlPolicyOutputMode:
        return self.control_output_mode

    def get_pure_policy_fn(self) -> Callable[[Params, Array], Array]:
        return self.policy

    def get_pure_reward_fn(self) -> Callable[[Array, Array, Array, Array], Array]:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into FFNPPOModel")
        return self._reward_fn
