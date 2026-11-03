from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from commondef.ioadapter import ControlPolicyInputState
from commondef.training import (
    ControlPolicyInputSpec,
    ControlPolicyOutputSpec,
    InputStateItemSensors,
    InputStateTypes,
)
from models.model import PPOTrainedModel, TestingModel

Array = jax.Array
Params = Any


class MLPModel(PPOTrainedModel, TestingModel):
    def __init__(
        self,
        policy: Callable[[Params, Array], Array],
        policy_store: Params,
        control_input_spec: ControlPolicyInputSpec,
        value_input_spec: ControlPolicyInputSpec,
        reward_input_spec: ControlPolicyInputSpec,
        control_output_spec: ControlPolicyOutputSpec,
        value: Callable[[Params, Array], Array] | None = None,
        value_store: Params | None = None,
        reward_fn: Callable[[ControlPolicyInputState, ControlPolicyInputState], float] | None = None,
        rng_key: Array | None = None,
    ) -> None:
        if (value is None) != (value_store is None):
            raise ValueError(
                "value and value_store must either both be provided or both be None"
            )
        self._control_output_spec = control_output_spec
        self.policy = policy
        self.value = value
        self.policy_store = policy_store
        self.value_store = value_store
        self._control_input_spec = control_input_spec
        self._value_input_spec = value_input_spec
        self._reward_input_spec = reward_input_spec
        self._reward_fn = reward_fn
        # Eager sampling owns this key; pure policy calls receive a key explicitly.
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
        control_input_spec: ControlPolicyInputSpec,
        value_input_spec: ControlPolicyInputSpec,
        reward_input_spec: ControlPolicyInputSpec,
        control_output_spec: ControlPolicyOutputSpec,
        reward_fn: Callable[[ControlPolicyInputState, ControlPolicyInputState], float] | None = None,
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
            control_output_spec=control_output_spec,
            reward_fn=reward_fn,
            rng_key=rng_key,
        )

    # --- Input spec ownership ---

    def get_control_input_spec(self) -> ControlPolicyInputSpec:
        return self._control_input_spec

    def get_value_input_spec(self) -> ControlPolicyInputSpec:
        return self._value_input_spec

    def get_reward_input_spec(self) -> ControlPolicyInputSpec:
        return self._reward_input_spec

    @staticmethod
    def _assemble_input(
        state: ControlPolicyInputState,
        input_spec: ControlPolicyInputSpec,
    ) -> Array:
        """Concatenate state leaves in spec order, retaining leading batch axes.

        Controllable states and contacts already use a trailing feature axis.
        Populated camera leaves have four feature axes (camera, H, W, RGB);
        the state getter's empty camera placeholder remains a zero-width leaf
        until the caller supplies rendered frames.
        """
        features: list[Array] = []
        for requested, actual in zip(
            input_spec.content.states, state.content.states, strict=True
        ):
            if requested.type != actual.type:
                raise ValueError("Input state entries must match input spec order")
            if requested.type == InputStateTypes.INPUT_STATE_CONTROLLABLE:
                value = jnp.asarray(actual.content.state_item)
            else:
                value = jnp.asarray(actual.content)
                if (
                    requested.content == InputStateItemSensors.CAMERAS
                    and value.shape[-1] != 0
                ):
                    if value.ndim < 4:
                        raise ValueError("Camera input must have shape [..., camera, H, W, RGB]")
                    value = value.reshape(
                        value.shape[:-4] + (int(np.prod(value.shape[-4:])),)
                    )
            features.append(value)
        if not features:
            raise ValueError("MLP input spec must request at least one state")
        return jnp.concatenate(features, axis=-1)

    # --- Reward ---

    def get_reward(
        self,
        state_pre: ControlPolicyInputState,
        state_post: ControlPolicyInputState,
    ) -> float:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into MLPModel")
        return float(self._reward_fn(state_pre, state_post))

    def get_pure_reward_fn(
        self,
    ) -> Callable[[ControlPolicyInputState, ControlPolicyInputState], float]:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into MLPModel")
        return self._reward_fn

    # --- Policy / value inference ---
    
    def get_policy_output(
        self, policy_input: ControlPolicyInputState
    ) -> dict[str, Array]:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return self.get_policy_fn()(self.policy_store, policy_input, subkey)

    # Outputs the output action sample, the process state and the log prob for that action.
    def get_policy_fn(
        self,
    ) -> Callable[[Params, ControlPolicyInputState, Array], dict[str, Array]]:
        policy = self.policy
        input_spec = self._control_input_spec

        def policy_fn(
            store: Params, policy_input: ControlPolicyInputState, rng_key: Array
        ) -> dict[str, Array]:
            action_gaussian_params = policy(
                store, MLPModel._assemble_input(policy_input, input_spec)
            )
            ## -1 to make this function batch friendly
            action_mean, action_log_std = jnp.split(action_gaussian_params,2,axis=-1)
            action_std = jnp.exp(action_log_std)
            eps = jax.random.normal(rng_key, shape=action_mean.shape)
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

    def get_policy_output_log_prob(
        self,
        policy_input: ControlPolicyInputState,
        policy_output: dict[str, Array],
    ) -> Array:
        return self.get_policy_output_log_prob_fn()(self.policy_store, policy_input, policy_output)

    def get_policy_output_log_prob_fn(
        self,
    ) -> Callable[[Params, ControlPolicyInputState, dict[str, Array]], Array]:
        policy = self.policy
        input_spec = self._control_input_spec

        def log_prob_fn(
            params: Params,
            policy_input: ControlPolicyInputState,
            output: dict[str, Array],
        ) -> Array:
            ## -1 to make this function batch friendly
            action_gaussian_params = policy(
                params, MLPModel._assemble_input(policy_input, input_spec)
            )
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

    def get_value_output(self, value_input: ControlPolicyInputState) -> float:
        val = self.get_value_fn()(self.get_value_store(), value_input)
        val_arr = np.asarray(jax.device_get(val), dtype=np.float64).reshape(-1)
        if val_arr.size == 0:
            raise ValueError("Expected non-empty value output")
        return float(val_arr[0])

    def get_value_fn(
        self,
    ) -> Callable[[Params, ControlPolicyInputState], Array]:
        if self.value is None:
            raise RuntimeError("No value estimator initialized in MLPModel")
        value = self.value
        input_spec = self._value_input_spec
        return lambda store, state: value(
            store,
            MLPModel._assemble_input(state, input_spec),
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

    def get_control_policy_output_spec(self) -> ControlPolicyOutputSpec:
        return self._control_output_spec

    # --- TestingModel interface ---
    # get_policy_output / get_policy_fn now return a stats dict; test-time
    # inference wants just the action, so we extract "sampled_action".

    def get_test_action(self, policy_input: ControlPolicyInputState) -> Array:
        return self.get_policy_output(policy_input)["sampled_action"]

    def get_test_policy(
        self,
    ) -> Callable[[Params, ControlPolicyInputState, Array], Array]:
        policy_fn = self.get_policy_fn()
        return lambda store, state, rng_key: policy_fn(
            store, state, rng_key
        )["sampled_action"]

    def get_test_params(self) -> Params:
        return self.policy_store
