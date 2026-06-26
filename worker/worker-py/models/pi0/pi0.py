"""Pi-0 VLA model wrapper for RL fine-tuning via PPO.

Loads the Pi-0 pre-trained checkpoint using the pure JAX/Flax Pi0Jax implementation
(no lerobot or PyTorch required). The VLM backbone (SigLIP + PaliGemma LM) is frozen;
the action expert and projection layers are updated by PPO.
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
from models.mlp import MLPModel
from models.model import PPOTrainedModel, TestingModel
from models.pi0.pi0_jax import Pi0Jax

logger = logging.getLogger(__name__)

Array = jax.Array
Params = Any


def _init_mlp(mlp_spec: MLPModelSpec, seed: int = 42) -> tuple[Callable, Params]:
    policy_fn, params = MLPModel.init_policy(
        input_dimension=mlp_spec.input_dim,
        output_dimension=mlp_spec.output_dim,
        hidden_layer_dimension=mlp_spec.hidden_layer_dim,
        num_layers=mlp_spec.num_hidden_layers,
        add_softmax=False,
        control_policy=False,
        seed=seed,
    )
    return policy_fn, params

def compute_flow_noise_log_prob(
    noise_model_fn: Callable[[Params, Array, Array], Array],
    noise_model_params: Params,
    obs: Array,
    denoising_steps: list[tuple[Array, Array]],
    euler_steps: list[Array],
) -> Array:
    """Compute log π(A | obs) via FlowNoise (Formula 5, arxiv:2510.25889).

    Each discrete ODE step from A_τ → A_{τ+δ} is modelled as a Gaussian:
        A_{τ+δ} | A_τ, obs  ~  N( euler_step(A_τ, τ),  diag(σ_φ(A_τ, obs)²) )

    The total log-prob is the sum of per-step Gaussian log-probs.

    Args:
        noise_model_fn: pure fn (params, obs, A_tau) -> per-dim log std devs
        noise_model_params: trainable params of the noise model σ_φ
        obs: observation fed to the noise model
        denoising_steps: list of (A_tau, A_tau_next) pairs for each ODE step
        euler_steps: deterministic Euler ODE predictions μ_τ for each step

    Returns:
        Scalar total log-probability of the denoising sequence.
    """
    log_two_pi = jnp.log(2.0 * jnp.pi)
    total_log_prob = jnp.zeros(())
    for (a_tau, a_tau_next), mu in zip(denoising_steps, euler_steps):
        log_std = noise_model_fn(noise_model_params, obs, a_tau)
        std = jnp.exp(log_std)
        normalized = (a_tau_next - mu) / (std + 1e-8)
        step_log_prob = jnp.sum(
            -0.5 * (normalized**2 + 2.0 * log_std + log_two_pi)
        )
        total_log_prob = total_log_prob + step_log_prob
    return total_log_prob

class Pi0Model(PPOTrainedModel, TestingModel):
    """PPO-trainable wrapper for the Pi-0 flow-matching VLA.

    Policy store: trainable_params from Pi0Jax (action expert + projections).
    Value store:  a small MLP initialised from value_mlp_spec.
    """

    _STATE_KEY_ORDER = ("cameras", "qdof", "qpos", "qvel", "xpos", "xquat")

    def __init__(
        self,
        vla_spec: VLAModel,
        noise_mlp_spec: MLPModelSpec,
        control_input_spec: ModelInputSpec,
        value_input_spec: ModelInputSpec,
        reward_input_spec: ModelInputSpec,
        value_mlp_spec: MLPModelSpec | None = None,
        reward_fn: Callable[[dict[str, Array], dict[str, Array]], float] | None = None,
        seed: int = 0,
    ) -> None:
        checkpoint_dir = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")
        logger.info("Pi0Model: loading checkpoint from %s", checkpoint_dir)
        self._pi0 = Pi0Jax.from_checkpoint(checkpoint_dir)
        logger.info("Pi0Model: checkpoint loaded")

        self._vla_spec = vla_spec
        self._policy_store: Params = self._pi0.get_policy_store()
        self._frozen_store: Params = self._pi0.get_frozen_store()

        self._control_input_spec = control_input_spec
        self._value_input_spec = value_input_spec
        self._reward_input_spec = reward_input_spec
        self._reward_fn = reward_fn

        if value_mlp_spec is not None:
            self._value_fn, self._value_store = _init_mlp(value_mlp_spec, seed + 1)
        else:
            self._value_fn = None
            self._value_store = None

        self._last_log_prob: float | None = None
        self._rng_key = random.PRNGKey(seed)
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

    def get_policy_output(self, parts: dict[str, Array]) -> Array:
        """Run Pi-0 inference on a labeled parts dict.

        The camera frames ("cameras") are used as the image input and the
        remaining proprio parts are concatenated (in _STATE_KEY_ORDER) into the
        proprio state vector.
        """
        self._rng_key, key = random.split(self._rng_key)
        task = getattr(self, "_last_task", "")
        images, proprio = self._split_observation(parts)
        actions = self._pi0.infer(
            self._policy_store,
            self._frozen_store,
            images,
            proprio,
            task,
            key,
        )
        return actions[0]

    def get_policy_output_log_prob(self) -> float:
        if self._last_log_prob is None:
            raise RuntimeError("Call get_policy_output before get_policy_output_log_prob.")
        return self._last_log_prob

    def get_value_output(self, parts: dict[str, Array]) -> float:
        flat = self._assemble_value(parts)
        self._ensure_value_store(flat)
        val = self._value_fn(self._value_store, flat)
        return float(jnp.squeeze(val))

    def get_policy(self) -> Callable[[Params, dict[str, Array]], Array]:
        pi0 = self._pi0
        frozen = self._frozen_store
        task = self._last_task
        rng = self._rng_key

        def _policy(policy_store: Params, parts: dict[str, Array]) -> Array:
            imgs, proprio = Pi0Model._split_observation(parts)
            actions = pi0.infer(policy_store, frozen, imgs, proprio, task, rng)
            return actions[0]
        return _policy

    def get_pure_policy_fn(self) -> Callable[[Params, dict[str, Array]], Array]:
        return self.get_policy()

    def get_value_func(self) -> Callable[[Params, dict[str, Array]], Array]:
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
    # TestingModel interface
    # ------------------------------------------------------------------

    def get_test_action(self, parts: dict[str, Array]) -> Array:
        return self.get_policy_output(parts)

    def get_test_policy(self) -> Callable[[Params, dict[str, Array]], Array]:
        return self.get_policy()

    def get_test_params(self) -> Params:
        return self.get_policy_store()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_value_store(self, state: Array) -> None:
        if self._value_store is None:
            obs_dim = int(state.shape[-1])
            spec = MLPModelSpec(
                num_hidden_layers=2,
                hidden_layer_dim=256,
                output_dim=1,
                input_dim=obs_dim,
            )
            self._value_fn, self._value_store = _init_mlp(spec, seed=99)