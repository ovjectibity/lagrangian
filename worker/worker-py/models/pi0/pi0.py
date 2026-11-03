"""Pi-0 VLA model wrapper for RL fine-tuning via PPO.

Loads the Pi-0 pre-trained checkpoint using the pure JAX/Flax Pi0Jax implementation
(no lerobot or PyTorch required). The VLM backbone (SigLIP + PaliGemma LM) is frozen;
the action expert and projection layers are updated by PPO.

Training uses the FlowNoise scheme (Formula 5, arxiv:2510.25889): each Euler
denoising step A_τ → A_{τ+δ} is made stochastic by injecting Gaussian noise with
per-dim std σ_φ(obs, A_τ) from a small noise MLP, so the trajectory log-prob is
a sum of per-step Gaussian log-probs. The policy store is a composite
{"pi0": <action-expert params>, "noise": <noise MLP params>} so PPO updates both.
Test-time inference keeps the deterministic Euler ODE (Pi0Jax.infer, no noise model).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import random

from commondef.ioadapter import PolicyInputState
from commondef.training import (
    PolicyInputSpec,
    PolicyOutputSpec,
    InputStateItemSensors,
    InputStateTypes,
    VLAModel,
)
from commondef.training import MLPModel as MLPModelSpec
from models.base import FlaxMLP
from models.model import PPOTrainedModel, TestingModel
from models.pi0.pi0_jax import Pi0Jax
from models.pi0.pi0_jax.model import _pad_to

logger = logging.getLogger(__name__)

Array = jax.Array
Params = Any

_LOG_TWO_PI = jnp.log(2.0 * jnp.pi)


def _init_mlp(
    mlp_spec: MLPModelSpec,
    rng_key: Array,
) -> tuple[Callable[[Params, Array], Array], Params]:
    module = FlaxMLP(
        num_hidden_layers=mlp_spec.num_hidden_layers,
        hidden_layer_dim=mlp_spec.hidden_layer_dim,
        output_dim=mlp_spec.output_dim,
    )
    params = module.init(
        rng_key,
        jnp.zeros((mlp_spec.input_dim,), dtype=jnp.float32),
    )
    return module.apply, params


def _prep_pi0_inputs(
    pi0: Pi0Jax,
    frozen: Params,
    policy_input: PolicyInputState,
    input_spec: PolicyInputSpec,
    task_tokens: Array,
) -> tuple[tuple[Array, Array], Array]:
    """Single-sample structured state → LM KV cache and normalised state.

    Mirrors the observation preprocessing of Pi0Jax.infer so the training
    denoise loop and the log-prob recompute see identical conditioning.
    Pure/traceable (task tokens are pre-encoded by the caller).
    """
    images, proprio = Pi0Model._split_observation(policy_input, input_spec)
    norm_dim = frozen["norm_state"]["mean"].shape[0]
    dim = min(proprio.shape[0], norm_dim)
    norm_state = pi0._normalize(
        proprio[:dim],
        {"mean": frozen["norm_state"]["mean"][:dim], "std": frozen["norm_state"]["std"][:dim]},
    )
    state_pad = _pad_to(norm_state, pi0._arch.get("state_dim", 32))
    lm_kv = pi0.encode_prefix(frozen, images[None], task_tokens[None])
    return lm_kv, state_pad


def _noise_log_std(
    noise_fn: Callable[[Params, Array], Array],
    noise_store: Params,
    obs: Array,
    a_tau: Array,
) -> Array:
    """FlowNoise σ_φ: (obs, A_τ) → per-dim log std devs, shaped like A_τ.

    obs is the normalised+padded proprio state (the same conditioning the
    action expert sees). log-std clip range matches the MLPModel control
    policy convention.
    """
    flat = jnp.concatenate([obs, a_tau.reshape(-1)])
    log_std = noise_fn(noise_store, flat)
    return jnp.clip(log_std, -5.2, -4.2).reshape(a_tau.shape)


class Pi0Model(PPOTrainedModel, TestingModel):
    """PPO-trainable wrapper for the Pi-0 flow-matching VLA.

    Policy store: {"pi0": trainable action-expert/projection params,
                   "noise": FlowNoise σ_φ MLP params (training only)}.
    Value store:  a small Flax MLP initialised from value_mlp_spec.
    """

    def __init__(
        self,
        vla_spec: VLAModel,
        noise_mlp_spec: MLPModelSpec | None,
        control_input_spec: PolicyInputSpec,
        value_input_spec: PolicyInputSpec,
        reward_input_spec: PolicyInputSpec,
        control_output_spec: PolicyOutputSpec,
        value_mlp_spec: MLPModelSpec | None = None,
        reward_fn: Callable[[PolicyInputState, PolicyInputState], float] | None = None,
        rng_key: Array | None = None,
        n_action_steps: int = 1,
    ) -> None:
        root_rng = rng_key if rng_key is not None else random.PRNGKey(0)
        noise_rng, value_rng, sampling_rng = random.split(root_rng, 3)

        checkpoint_dir = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")
        logger.info("Pi0Model: loading checkpoint from %s", checkpoint_dir)
        self._pi0 = Pi0Jax.from_checkpoint(
            checkpoint_dir,
            n_action_steps=n_action_steps,
        )
        logger.info("Pi0Model: checkpoint loaded")

        self._vla_spec = vla_spec
        self._frozen_store: Params = self._pi0.get_frozen_store()
        self._policy_store: Params = {"pi0": self._pi0.get_policy_store()}

        # FlowNoise σ_φ: only wired for training; the test path uses the
        # deterministic Euler ODE and never touches the noise model.
        if noise_mlp_spec is not None:
            self._noise_fn, noise_store = _init_mlp(noise_mlp_spec, noise_rng)
            self._policy_store["noise"] = noise_store
        else:
            self._noise_fn = None

        self._control_input_spec = control_input_spec
        self._value_input_spec = value_input_spec
        self._reward_input_spec = reward_input_spec
        self._control_output_spec = control_output_spec
        self._reward_fn = reward_fn

        if value_mlp_spec is not None:
            self._value_fn, self._value_store = _init_mlp(
                value_mlp_spec,
                value_rng,
            )
        else:
            self._value_fn = None
            self._value_store = None

        # Eager inference owns this key; pure policy calls receive one explicitly.
        self._rng_key = sampling_rng
        self._last_task: str = ""

    # ------------------------------------------------------------------
    # Input spec ownership
    # ------------------------------------------------------------------

    def get_control_input_spec(self) -> PolicyInputSpec:
        return self._control_input_spec

    def get_value_input_spec(self) -> PolicyInputSpec:
        return self._value_input_spec

    def get_reward_input_spec(self) -> PolicyInputSpec:
        return self._reward_input_spec

    # ------------------------------------------------------------------
    # State assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _split_observation(
        state: PolicyInputState,
        input_spec: PolicyInputSpec,
    ) -> tuple[Array, Array]:
        """Extract camera images and concatenate other state leaves in spec order.

        Rendered camera leaves must use [..., camera, H, W, RGB]. Repeated
        camera requests refer to the same rendered images, used only once.
        The caller must replace the getter's empty camera placeholders before
        invoking a Pi-0 policy. Other leaves retain their trailing feature axis.
        """
        images: Array | None = None
        features: list[Array] = []
        for requested, actual in zip(
            input_spec.content.states, state.content.states, strict=True
        ):
            if requested.type != actual.type:
                raise ValueError("Input state entries must match input spec order")
            if requested.type == InputStateTypes.INPUT_STATE_CONTROLLABLE:
                features.append(jnp.asarray(actual.content.state_item))
            elif requested.content == InputStateItemSensors.CAMERAS:
                camera_frames = jnp.asarray(actual.content)
                if camera_frames.ndim < 4 or camera_frames.shape[-1] != 3:
                    raise ValueError(
                        "Pi-0 requires rendered cameras with shape "
                        "[..., camera, H, W, RGB]"
                    )
                if images is None:
                    images = camera_frames
            else:
                features.append(jnp.asarray(actual.content))
        if images is None:
            raise ValueError("Pi-0 control input spec must request Cameras")
        proprio = (
            jnp.concatenate(features, axis=-1)
            if features
            else jnp.zeros(images.shape[:-4] + (0,), dtype=jnp.float32)
        )
        return images, proprio

    @staticmethod
    def _assemble_value(
        state: PolicyInputState,
        input_spec: PolicyInputSpec,
    ) -> Array:
        """Flatten value features in spec order, preserving leading batch axes."""
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
                        raise ValueError(
                            "Camera input must have shape [..., camera, H, W, RGB]"
                        )
                    feature_size = 1
                    for dimension in value.shape[-4:]:
                        feature_size *= dimension
                    value = value.reshape(value.shape[:-4] + (feature_size,))
            features.append(value)
        if not features:
            raise ValueError("Value input spec must request at least one state")
        return jnp.concatenate(features, axis=-1)

    # ------------------------------------------------------------------
    # PPOTrainedModel interface
    # ------------------------------------------------------------------

    @classmethod
    def init_ppo_trained_model(cls, *args, **kwargs) -> "Pi0Model":
        raise NotImplementedError("Construct Pi0Model directly.")

    def get_policy_output(
        self, policy_input: PolicyInputState
    ) -> dict[str, Array]:
        self._rng_key, subkey = random.split(self._rng_key)
        return self.get_policy_fn()(self._policy_store, policy_input, subkey)

    # Outputs the sampled action chunk, the denoising process state and the
    # log prob for that trajectory. The sampling key is an explicit argument.
    def get_policy_fn(
        self,
    ) -> Callable[[Params, PolicyInputState, Array], dict[str, Array]]:
        if self._noise_fn is None:
            raise RuntimeError(
                "Pi0Model was built without a noise model spec; the FlowNoise "
                "training path is unavailable (test inference uses get_test_policy)."
            )
        pi0 = self._pi0
        frozen = self._frozen_store
        noise_fn = self._noise_fn
        input_spec = self._control_input_spec
        task_tokens = pi0._encode_task(self._last_task)
        n_steps = pi0._num_inference_steps
        chunk = pi0._chunk_size
        max_action_dim = pi0._max_action_dim
        n_action_steps = pi0._n_action_steps
        dt = 1.0 / n_steps

        def _policy(
            policy_store: Params,
            policy_input: PolicyInputState,
            rng_key: Array,
        ) -> dict[str, Array]:
            trainable = policy_store["pi0"]
            noise_store = policy_store["noise"]
            lm_kv, state_pad = _prep_pi0_inputs(
                pi0, frozen, policy_input, input_spec, task_tokens
            )

            init_key, steps_key = random.split(rng_key)
            x0 = random.normal(init_key, (chunk, max_action_dim))
            ts = 1.0 - jnp.arange(n_steps, dtype=jnp.float32) * dt
            step_keys = random.split(steps_key, n_steps)

            def denoise_step(
                x: Array, inp: tuple[Array, Array]
            ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
                t, step_key = inp
                vel = pi0.forward(trainable, lm_kv, state_pad[None], x[None], t)[0]
                mu = x - dt * vel
                log_std = _noise_log_std(noise_fn, noise_store, state_pad, x)
                std = jnp.exp(log_std)
                x_next = mu + std * random.normal(step_key, x.shape)
                normalized = (x_next - mu) / std
                step_log_prob = jnp.sum(
                    -0.5 * (normalized**2 + 2.0 * log_std + _LOG_TWO_PI)
                )
                return x_next, (x, mu, log_std, step_log_prob)

            x_final, (xs, mus, log_stds, step_log_probs) = jax.lax.scan(
                denoise_step, x0, (ts, step_keys)
            )
            # Full denoising trajectory [n_steps + 1, chunk, max_action_dim]:
            # only the realized states are stored — the log-prob recompute
            # re-derives mu/log_std from them under the updated params.
            trajectory = jnp.concatenate([xs, x_final[None]], axis=0)

            actual_dim = frozen["unnorm_action"]["mean"].shape[0]
            actions = pi0._unnormalize(
                x_final[:n_action_steps, :actual_dim], frozen["unnorm_action"]
            )
            # action_mean / action_log_std_dev are the sampling-time per-step
            # Gaussian stats ([n_steps, chunk, max_action_dim]) — diagnostics
            # only (entropy), matching the MLPModel stats-dict convention.
            return {
                "sampled_action": actions,
                "sampled_action_log_prob": jnp.sum(step_log_probs),
                "denoising_trajectory": trajectory,
                "action_mean": mus,
                "action_log_std_dev": log_stds,
            }

        return _policy

    def get_policy_output_log_prob(
        self, policy_input: PolicyInputState, policy_output: dict[str, Array]
    ) -> Array:
        return self.get_policy_output_log_prob_fn()(self._policy_store, policy_input, policy_output)

    def get_policy_output_log_prob_fn(
        self,
    ) -> Callable[[Params, PolicyInputState, dict[str, Array]], Array]:
        if self._noise_fn is None:
            raise RuntimeError(
                "Pi0Model was built without a noise model spec; the FlowNoise "
                "training path is unavailable (test inference uses get_test_policy)."
            )
        pi0 = self._pi0
        frozen = self._frozen_store
        noise_fn = self._noise_fn
        input_spec = self._control_input_spec
        task_tokens = pi0._encode_task(self._last_task)
        n_steps = pi0._num_inference_steps
        dt = 1.0 / n_steps

        def _single(
            policy_store: Params,
            policy_input: PolicyInputState,
            policy_output: dict[str, Array],
        ) -> Array:
            trainable = policy_store["pi0"]
            noise_store = policy_store["noise"]
            lm_kv, state_pad = _prep_pi0_inputs(
                pi0, frozen, policy_input, input_spec, task_tokens
            )

            # Recompute the FlowNoise log prob of the recorded denoising
            # trajectory under the *current* params (Formula 5, arxiv:2510.25889):
            # each recorded A_τ → A_{τ+δ} step is Gaussian around the current
            # Euler prediction with the current σ_φ.
            trajectory = policy_output["denoising_trajectory"]
            ts = 1.0 - jnp.arange(n_steps, dtype=jnp.float32) * dt

            def step(total: Array, inp: tuple[Array, Array, Array]) -> tuple[Array, None]:
                x, x_next, t = inp
                vel = pi0.forward(trainable, lm_kv, state_pad[None], x[None], t)[0]
                mu = x - dt * vel
                log_std = _noise_log_std(noise_fn, noise_store, state_pad, x)
                normalized = (x_next - mu) / jnp.exp(log_std)
                step_log_prob = jnp.sum(
                    -0.5 * (normalized**2 + 2.0 * log_std + _LOG_TWO_PI)
                )
                return total + step_log_prob, None

            total, _ = jax.lax.scan(
                step, jnp.zeros(()), (trajectory[:-1], trajectory[1:], ts)
            )
            return total

        def log_prob_fn(
            policy_store: Params,
            policy_input: PolicyInputState,
            policy_output: dict[str, Array],
        ) -> Array:
            # The trainer supplies [num_trajs, T, ...] (or [T, ...]) leading
            # axes on every leaf; vmap over each extra leading axis.
            # Camera rank is static during tracing: its last four axes are
            # [camera, H, W, RGB], with rollout batch/time axes preceding them.
            images, _ = Pi0Model._split_observation(policy_input, input_spec)
            fn = _single
            for _ in range(images.ndim - 4):
                fn = jax.vmap(fn, in_axes=(None, 0, 0))
            return fn(policy_store, policy_input, policy_output)

        return log_prob_fn

    def get_value_output(self, value_input: PolicyInputState) -> float:
        val = self.get_value_fn()(self.get_value_store(), value_input)
        return float(jnp.squeeze(val))

    def get_value_fn(
        self,
    ) -> Callable[[Params, PolicyInputState], Array]:
        if self._value_fn is None:
            raise RuntimeError("No value estimator initialized in Pi0Model")
        value_fn = self._value_fn
        input_spec = self._value_input_spec
        return lambda store, state: value_fn(
            store, Pi0Model._assemble_value(state, input_spec)
        )

    def get_policy_store(self) -> Params:
        return self._policy_store

    def get_policy_store_mut(self) -> Params:
        return self._policy_store

    def get_value_store(self) -> Params:
        if self._value_store is None:
            raise RuntimeError("No value estimator store initialized in Pi0Model")
        return self._value_store

    def get_value_store_mut(self) -> Params:
        if self._value_store is None:
            raise RuntimeError("No value estimator store initialized in Pi0Model")
        return self._value_store

    def get_control_policy_output_spec(self) -> PolicyOutputSpec:
        return self._control_output_spec

    def get_reward(
        self,
        state_pre: PolicyInputState,
        state_post: PolicyInputState,
    ) -> float:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into Pi0Model")
        return float(self._reward_fn(state_pre, state_post))

    def get_pure_reward_fn(
        self,
    ) -> Callable[[PolicyInputState, PolicyInputState], float]:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into Pi0Model")
        return self._reward_fn

    # ------------------------------------------------------------------
    # TestingModel interface — deterministic Euler ODE (no noise model)
    # ------------------------------------------------------------------

    def get_test_action(self, policy_input: PolicyInputState) -> Array:
        self._rng_key, key = random.split(self._rng_key)
        return self.get_test_policy()(self._policy_store, policy_input, key)

    def get_test_policy(
        self,
    ) -> Callable[[Params, PolicyInputState, Array], Array]:
        pi0 = self._pi0
        frozen = self._frozen_store
        task = self._last_task
        input_spec = self._control_input_spec

        def _test_policy(
            params: Params,
            policy_input: PolicyInputState,
            rng_key: Array,
        ) -> Array:
            images, proprio = Pi0Model._split_observation(policy_input, input_spec)
            return pi0.infer(params["pi0"], frozen, images, proprio, task, rng_key)

        return _test_policy

    def get_test_params(self) -> Params:
        return self._policy_store
