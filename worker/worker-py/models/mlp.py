from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from commondef.training import ModelInputSpec, ControlPolicyOutputMode
from models.model import PPOTrainedModel, TestingModel

Array = jax.Array
Params = Any


class MLPModel(PPOTrainedModel, TestingModel):
    def __init__(
        self,
        policy: Callable[[Params, Array], Array],
        policy_store: Params,
        control_input_spec: ModelInputSpec,
        value_input_spec: ModelInputSpec,
        reward_input_spec: ModelInputSpec,
        value: Callable[[Params, Array], Array] | None = None,
        value_store: Params | None = None,
        reward_fn: Callable[[dict[str, Array], dict[str, Array]], float] | None = None,
        rng_key: Array | None = None,
    ) -> None:
        if (value is None) != (value_store is None):
            raise ValueError(
                "value and value_store must either both be provided or both be None"
            )
        self.control_output_mode = ControlPolicyOutputMode.TARGETPOSITIONS
        self.policy = policy
        self.value = value
        self.policy_store = policy_store
        self.value_store = value_store
        self._control_input_spec = control_input_spec
        self._value_input_spec = value_input_spec
        self._reward_input_spec = reward_input_spec
        self._reward_fn = reward_fn
        # Held key for eager sampling (get_policy_output); the scan threads fresh
        # per-step keys via policy_input["rng_key"] instead.
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

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
        control_input_spec: ModelInputSpec,
        value_input_spec: ModelInputSpec,
        reward_input_spec: ModelInputSpec,
        reward_fn: Callable[[dict[str, Array], dict[str, Array]], float] | None = None,
        rng_key: Array | None = None,
    ) -> MLPModel:
        return cls(
            policy=policy,
            value=val_estimator,
            policy_store=pol_store,
            value_store=val_estimator_store,
            control_input_spec=control_input_spec,
            value_input_spec=value_input_spec,
            reward_input_spec=reward_input_spec,
            reward_fn=reward_fn,
            rng_key=rng_key,
        )

    # --- Input spec ownership ---

    def get_control_input_spec(self) -> ModelInputSpec:
        return self._control_input_spec

    def get_value_input_spec(self) -> ModelInputSpec:
        return self._value_input_spec

    def get_reward_input_spec(self) -> ModelInputSpec:
        return self._reward_input_spec

    # --- Reward ---

    def get_reward(
        self,
        parts_pre: dict[str, Array],
        parts_post: dict[str, Array],
    ) -> float:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into MLPModel")
        return float(self._reward_fn(parts_pre, parts_post))

    def get_pure_reward_fn(
        self,
    ) -> Callable[[dict[str, Array], dict[str, Array]], float]:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into MLPModel")
        return self._reward_fn

    # --- Policy / value inference ---
    
    def get_policy_output(self, policy_input: dict[str, Array]) -> dict[str,Array]:
        # Eager path: split the held key in place and thread it in via the dict.
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return self.get_policy_fn()(self.policy_store, {**policy_input, "rng_key": subkey})

    # Outputs the output action sample, the process state and the log prob for that action.
    # The per-call sampling key is read from policy_input["rng_key"] (popped before
    # assembling the network input), so the scan can thread a fresh key each step.
    def get_policy_fn(self) -> Callable[[Params, dict[str, Array]], dict[str,Array]]:
        policy = self.policy
        def policy_fn(store, policy_input):
            key = policy_input["rng_key"]
            features = {k: v for k, v in policy_input.items() if k != "rng_key"}
            action_gaussian_params = policy(store, jnp.concatenate(list(features.values()), axis=-1))
            ## -1 to make this function batch friendly
            action_mean, action_log_std = jnp.split(action_gaussian_params,2,axis=-1)
            action_std = jnp.exp(action_log_std)
            eps = jax.random.normal(key, shape=action_mean.shape)
            sampled_action = action_mean + eps * action_std
            # Joint log prob of the sampled action under the diagonal Gaussian:
            # per-dim -0.5*(z^2 + 2*log_std + log(2pi)), summed over action dims.
            normalized = (sampled_action - action_mean) / action_std
            log_two_pi = jnp.log(2.0 * jnp.pi)
            log_prob = jnp.sum(
                -0.5 * (normalized ** 2 + 2.0 * action_log_std + log_two_pi), axis=-1
            )
            # Return the sampled action as well the process state towards the output sample
            return ({
                "action_mean": action_mean,
                "action_log_std_dev": action_log_std,
                "sampled_action_log_prob": log_prob,
                "sampled_action": sampled_action
            })
        return policy_fn

    def get_policy_output_log_prob(self, policy_input: dict[str, Array], policy_output: dict[str,Array]) -> Array: 
        return self.get_policy_output_log_prob_fn()(self.policy_store, policy_input, policy_output)

    def get_policy_output_log_prob_fn(self) -> Callable[[Params, dict[str, Array], dict[str,Array]], Array]: 
        policy = self.policy
        def log_prob_fn(params: Params, input: dict[str, Array], output: dict[str,Array]) -> Array:
            ## -1 to make this function batch friendly
            action_gaussian_params = policy(params, jnp.concatenate(list(input.values()), axis=-1))
            action_mean, action_log_std = jnp.split(action_gaussian_params,2,axis=-1)
            action_std = jnp.exp(action_log_std)
            # Recompute the joint log prob of output["sampled_action"] under the
            # current params (deterministic; no sampling key needed).
            normalized = (output["sampled_action"] - action_mean) / action_std
            log_two_pi = jnp.log(2.0 * jnp.pi)
            log_prob = jnp.sum(
                -0.5 * (normalized ** 2 + 2.0 * action_log_std + log_two_pi), axis=-1
            )
            return log_prob
        return log_prob_fn

    def get_value_output(self, parts: dict[str, Array]) -> float:
        val = self.get_value_fn()(self.get_value_store(), parts)
        val_arr = np.asarray(jax.device_get(val), dtype=np.float64).reshape(-1)
        if val_arr.size == 0:
            raise ValueError("Expected non-empty value output")
        return float(val_arr[0])

    def get_value_fn(self) -> Callable[[Params, dict[str, Array]], Array]:
        if self.value is None:
            raise RuntimeError("No value estimator initialized in MLPModel")
        value = self.value
        return lambda store, parts: value(
            store,
            jnp.concatenate(list(parts.values()), axis=-1),
        )

    def get_policy_store(self) -> Params:
        return self.policy_store

    def get_policy_store_mut(self) -> Params:
        return self.policy_store

    def get_value_store(self) -> Params:
        if self.value_store is None:
            raise RuntimeError("No value estimator store initialized in MLPModel")
        return self.value_store

    def get_value_store_mut(self) -> Params:
        if self.value_store is None:
            raise RuntimeError("No value estimator store initialized in MLPModel")
        return self.value_store

    def get_control_policy_output_mode(self) -> ControlPolicyOutputMode:
        return self.control_output_mode

    # --- TestingModel interface ---
    # get_policy_output / get_policy_fn now return a stats dict; test-time
    # inference wants just the action, so we extract "sampled_action".

    def get_test_action(self, parts: dict[str, Array]) -> Array:
        # Eager: get_policy_output splits the held key and injects it as "rng_key".
        return self.get_policy_output(parts)["sampled_action"]

    def get_test_policy(self) -> Callable[[Params, dict[str, Array]], Array]:
        # Pure fn for vmap/scan batched testing. The caller must supply the
        # per-call sampling key in parts["rng_key"] (as the trainer/testing scan does).
        policy_fn = self.get_policy_fn()
        return lambda store, parts: policy_fn(store, parts)["sampled_action"]

    def get_test_params(self) -> Params:
        return self.policy_store
