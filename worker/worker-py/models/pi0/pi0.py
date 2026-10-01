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

from commondef.training import ControlPolicyOutputMode, ModelInputSpec, VLAModel
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
    parts: dict[str, Array],
    task_tokens: Array,
) -> tuple[tuple[Array, Array], Array]:
    """Single-sample: parts dict → (LM KV cache, normalised+padded state).

    Mirrors the observation preprocessing of Pi0Jax.infer so the training
    denoise loop and the log-prob recompute see identical conditioning.
    Pure/traceable (task tokens are pre-encoded by the caller).
    """
    images, proprio = Pi0Model._split_observation(parts)
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

    _STATE_KEY_ORDER = ("cameras", "qpos", "qvel", "xpos", "xquat")

    def __init__(
        self,
        vla_spec: VLAModel,
        noise_mlp_spec: MLPModelSpec | None,
        control_input_spec: ModelInputSpec,
        value_input_spec: ModelInputSpec,
        reward_input_spec: ModelInputSpec,
        value_mlp_spec: MLPModelSpec | None = None,
        reward_fn: Callable[[dict[str, Array], dict[str, Array]], float] | None = None,
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
        self._reward_fn = reward_fn

        if value_mlp_spec is not None:
            self._value_fn, self._value_store = _init_mlp(
                value_mlp_spec,
                value_rng,
            )
        else:
            self._value_fn = None
            self._value_store = None

        # Held key for eager sampling (get_policy_output / get_test_action); the
        # scan threads fresh per-step keys via policy_input["rng_key"] instead.
        self._rng_key = sampling_rng
        self._last_task: str = ""

    # ------------------------------------------------------------------
    # Input spec ownership
    # ------------------------------------------------------------------

    def get_control_input_spec(self) -> ModelInputSpec:
        return self._control_input_spec

    def get_value_input_spec(self) -> ModelInputSpec:
        return self._value_input_spec

    def get_reward_input_spec(self) -> ModelInputSpec:
        return self._reward_input_spec

    # ------------------------------------------------------------------
    # State assembly
    # ------------------------------------------------------------------

    @classmethod
    def _split_observation(cls, parts: dict[str, Array]) -> tuple[Array, Array]:
        """Split a labeled parts dict into (camera images, proprio state vector).

        Camera frames come straight through as the image input; the remaining
        proprio parts are concatenated in _STATE_KEY_ORDER.
        """
        images = parts["cameras"]
        proprio = jnp.concatenate(
            [parts[k] for k in cls._STATE_KEY_ORDER if k != "cameras" and k in parts]
        )
        return images, proprio

    @classmethod
    def _assemble_value(cls, parts: dict[str, Array]) -> Array:
        """Concatenate value-input parts into a flat vector for the value MLP.

        cameras (if present) arrive as [..., ncam, H, W, C]; flatten those
        trailing 4 dims so they concatenate with the 1-D proprio parts. Concat
        is on the feature axis (batch-safe over leading [num_trajs, T, ...]).
        """
        return jnp.concatenate([
            parts[k].reshape(*parts[k].shape[:-4], -1) if k == "cameras" else parts[k]
            for k in cls._STATE_KEY_ORDER if k in parts
        ], axis=-1)

    # ------------------------------------------------------------------
    # PPOTrainedModel interface
    # ------------------------------------------------------------------

    @classmethod
    def init_ppo_trained_model(cls, *args, **kwargs) -> "Pi0Model":
        raise NotImplementedError("Construct Pi0Model directly.")

    def get_policy_output(self, policy_input: dict[str, Array]) -> dict[str, Array]:
        # Eager path: split the held key in place and thread it in via the dict.
        self._rng_key, subkey = random.split(self._rng_key)
        return self.get_policy_fn()(self._policy_store, {**policy_input, "rng_key": subkey})

    # Outputs the sampled action chunk, the denoising process state and the
    # log prob for that trajectory. The per-call sampling key is read from
    # policy_input["rng_key"] (popped before assembling the network input),
    # so the scan can thread a fresh key each step.
    def get_policy_fn(self) -> Callable[[Params, dict[str, Array]], dict[str, Array]]:
        if self._noise_fn is None:
            raise RuntimeError(
                "Pi0Model was built without a noise model spec; the FlowNoise "
                "training path is unavailable (test inference uses get_test_policy)."
            )
        pi0 = self._pi0
        frozen = self._frozen_store
        noise_fn = self._noise_fn
        task_tokens = pi0._encode_task(self._last_task)
        n_steps = pi0._num_inference_steps
        chunk = pi0._chunk_size
        max_action_dim = pi0._max_action_dim
        n_action_steps = pi0._n_action_steps
        dt = 1.0 / n_steps

        def _policy(policy_store: Params, policy_input: dict[str, Array]) -> dict[str, Array]:
            key = policy_input["rng_key"]
            parts = {k: v for k, v in policy_input.items() if k != "rng_key"}
            trainable = policy_store["pi0"]
            noise_store = policy_store["noise"]
            lm_kv, state_pad = _prep_pi0_inputs(pi0, frozen, parts, task_tokens)

            init_key, steps_key = random.split(key)
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
        self, policy_input: dict[str, Array], policy_output: dict[str, Array]
    ) -> Array:
        return self.get_policy_output_log_prob_fn()(self._policy_store, policy_input, policy_output)

    def get_policy_output_log_prob_fn(
        self,
    ) -> Callable[[Params, dict[str, Array], dict[str, Array]], Array]:
        if self._noise_fn is None:
            raise RuntimeError(
                "Pi0Model was built without a noise model spec; the FlowNoise "
                "training path is unavailable (test inference uses get_test_policy)."
            )
        pi0 = self._pi0
        frozen = self._frozen_store
        noise_fn = self._noise_fn
        task_tokens = pi0._encode_task(self._last_task)
        n_steps = pi0._num_inference_steps
        dt = 1.0 / n_steps

        def _single(
            policy_store: Params,
            policy_input: dict[str, Array],
            policy_output: dict[str, Array],
        ) -> Array:
            parts = {k: v for k, v in policy_input.items() if k != "rng_key"}
            trainable = policy_store["pi0"]
            noise_store = policy_store["noise"]
            lm_kv, state_pad = _prep_pi0_inputs(pi0, frozen, parts, task_tokens)

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
            policy_input: dict[str, Array],
            policy_output: dict[str, Array],
        ) -> Array:
            # The trainer calls this with [num_trajs, T, ...] (or [T, ...])
            # leading axes on every leaf; vmap over each extra leading axis
            # (cameras is always present for a VLA and is [ncam, H, W, C]
            # unbatched).
            fn = _single
            for _ in range(policy_input["cameras"].ndim - 4):
                fn = jax.vmap(fn, in_axes=(None, 0, 0))
            return fn(policy_store, policy_input, policy_output)

        return log_prob_fn

    def get_value_output(self, parts: dict[str, Array]) -> float:
        flat = self._assemble_value(parts)
        self._ensure_value_store(flat)
        val = self._value_fn(self._value_store, flat)
        return float(jnp.squeeze(val))

    def get_value_fn(self) -> Callable[[Params, dict[str, Array]], Array]:
        # value MLP takes a flat tensor; assemble the parts dict internally.
        return lambda store, parts: self._value_fn(store, self._assemble_value(parts))

    def get_policy_store(self) -> Params:
        return self._policy_store

    def get_policy_store_mut(self) -> Params:
        return self._policy_store

    def get_value_store(self) -> Params:
        return self._value_store

    def get_value_store_mut(self) -> Params:
        return self._value_store

    def get_control_policy_output_mode(self) -> ControlPolicyOutputMode:
        return ControlPolicyOutputMode.TARGETPOSITIONS

    def get_reward(
        self,
        parts_pre: dict[str, Array],
        parts_post: dict[str, Array],
    ) -> float:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into Pi0Model")
        return float(self._reward_fn(parts_pre, parts_post))

    def get_pure_reward_fn(
        self,
    ) -> Callable[[dict[str, Array], dict[str, Array]], float]:
        if self._reward_fn is None:
            raise RuntimeError("No reward function injected into Pi0Model")
        return self._reward_fn

    # ------------------------------------------------------------------
    # TestingModel interface — deterministic Euler ODE (no noise model)
    # ------------------------------------------------------------------

    def get_test_action(self, parts: dict[str, Array]) -> Array:
        self._rng_key, key = random.split(self._rng_key)
        return self.get_test_policy()(self._policy_store, {**parts, "rng_key": key})

    def get_test_policy(self) -> Callable[[Params, dict[str, Array]], Array]:
        pi0 = self._pi0
        frozen = self._frozen_store
        task = self._last_task

        def _test_policy(params: Params, parts: dict[str, Array]) -> Array:
            key = parts["rng_key"]
            feats = {k: v for k, v in parts.items() if k != "rng_key"}
            images, proprio = Pi0Model._split_observation(feats)
            return pi0.infer(params["pi0"], frozen, images, proprio, task, key)

        return _test_policy

    def get_test_params(self) -> Params:
        return self._policy_store

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_value_store(self, state: Array) -> None:
        if self._value_store is None:
            module = FlaxMLP(num_hidden_layers=2, hidden_layer_dim=256, output_dim=1)
            self._value_fn = module.apply
            self._value_store = module.init(random.PRNGKey(99), jnp.zeros_like(state))
