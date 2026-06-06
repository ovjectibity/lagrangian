"""
Pi-0 VLA model wrapper for RL fine-tuning via PPO + FlowNoise.

Loads the Pi-0 pre-trained checkpoint (bundled via Dockerfile from
HuggingFace lerobot/pi0_libero_base) and adapts it to the PPOTrainedModel
interface. Only the action head and the FlowNoise model are trainable;
the VLM backbone is frozen.

Log-probabilities are computed via the FlowNoise method (Formula 5,
arxiv:2510.25889): the K-step denoising ODE is treated as a discrete-time
MDP where each step's transition is modelled as a Gaussian with mean equal
to the Euler ODE update and variance learned by the noise model.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from commondef.training import ControlPolicyOutputMode, FFNModel, VLAModel
from ppo_trainer import PPOTrainedModel, compute_flow_noise_log_prob
from samples.arm import ArmModel

logger = logging.getLogger(__name__)

Array = jax.Array
Params = Any

# Number of denoising steps used during Pi-0 inference.
_PI0_DENOISING_STEPS = 10


def _init_ffn(ffn_spec: FFNModel, seed: int = 42) -> tuple[Callable, Params]:
    """Initialise a simple FFN matching the FFNModel spec."""
    policy_fn, params = ArmModel.init_policy(
        input_dimension=ffn_spec.input_dim,
        output_dimension=ffn_spec.output_dim,
        hidden_layer_dimension=ffn_spec.hidden_layer_dim,
        num_layers=ffn_spec.num_hidden_layers,
        add_softmax=False,
        control_policy=False,
        seed=seed,
    )
    return policy_fn, params


class Pi0Model(PPOTrainedModel):
    """PPO-trainable wrapper for the Pi-0 flow-matching VLA.

    The VLM backbone is frozen; only the action head (inside Pi-0) and the
    FlowNoise model σ_φ are updated during the PPO backward pass.

    Policy store layout:
        {
            "action_head": <Pi-0 action-head Flax params>,
            "noise_model": <FFN params for σ_φ>,
        }
    """

    def __init__(
        self,
        vla_spec: VLAModel,
        noise_ffn_spec: FFNModel,
        value_ffn_spec: FFNModel | None = None,
        seed: int = 0,
    ) -> None:
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy  # type: ignore[import]

        checkpoint_dir = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")
        logger.info("Pi0Model: loading checkpoint from %s", checkpoint_dir)
        self._pi0: PI0Policy = PI0Policy.from_pretrained(checkpoint_dir)
        logger.info("Pi0Model: checkpoint loaded")

        self._vla_spec = vla_spec

        # Freeze backbone — separate out the action-head params from all Pi-0 params.
        # Pi-0's Flax param tree: {'model': {'action_head': ..., 'vlm': ...}}
        # We keep only action_head in the trainable store; backbone is a constant.
        pi0_params = self._pi0.params  # full Flax pytree
        self._backbone_params: Params = jax.tree_util.tree_map(
            lambda x: x, pi0_params["model"]["vlm"]
        )
        action_head_params: Params = jax.tree_util.tree_map(
            lambda x: x, pi0_params["model"]["action_head"]
        )

        # Noise model σ_φ: (obs, A_τ) → per-dim log std devs.
        # Input dim = obs_dim + action_dim; output dim = action_dim.
        self._noise_model_fn, noise_model_params = _init_ffn(noise_ffn_spec, seed)

        # Trainable policy store = action head + noise model.
        self._policy_store: Params = {
            "action_head": action_head_params,
            "noise_model": noise_model_params,
        }

        # Value function: a fresh FFN (same input dim as obs by default).
        if value_ffn_spec is not None:
            self._value_fn, self._value_store = _init_ffn(value_ffn_spec, seed + 1)
        else:
            # Fallback: small value FFN inferred at first call.
            self._value_fn = None
            self._value_store = None

        self._last_log_prob: float | None = None
        self._rng_key = random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Internal Pi-0 inference helpers
    # ------------------------------------------------------------------

    def _run_pi0_denoising(
        self,
        obs: Array,
        action_head_params: Params,
    ) -> tuple[Array, list[tuple[Array, Array]], list[Array]]:
        """Run K-step flow-matching denoising using Pi-0's action head.

        Returns:
            action: final denoised action (shape [action_dim])
            denoising_steps: list of (A_τ, A_{τ+δ}) pairs (noise → clean direction)
            euler_steps: list of μ_τ (deterministic Euler ODE predictions)
        """
        # Sample initial noise at τ=1.
        self._rng_key, noise_key = random.split(self._rng_key)
        action_dim = self._pi0.config.action_dim
        a = random.normal(noise_key, shape=(action_dim,))

        # Build discrete timestep schedule from τ=1 down to τ=0.
        K = _PI0_DENOISING_STEPS
        timesteps = jnp.linspace(1.0, 0.0, K + 1)

        denoising_steps: list[tuple[Array, Array]] = []
        euler_steps: list[Array] = []

        for k in range(K):
            tau = float(timesteps[k])
            delta = float(timesteps[k] - timesteps[k + 1])

            # Velocity field prediction from Pi-0 action head (frozen backbone
            # provides obs embedding; only action_head_params is differentiated).
            obs_embedding = self._get_obs_embedding(obs)
            v = self._pi0.model.apply(
                {"params": {"action_head": action_head_params}},
                obs_embedding,
                a,
                jnp.array(tau),
                method="predict_velocity",
            )

            # Euler step: μ_τ = A_τ - δ * v_θ(A_τ, τ, obs)
            mu = a - delta * v
            a_next = mu  # deterministic Euler; noise model adds stochasticity

            denoising_steps.append((a, a_next))
            euler_steps.append(mu)
            a = a_next

        return a, denoising_steps, euler_steps

    def _get_obs_embedding(self, obs: Array) -> Array:
        """Run the frozen VLM backbone to get an observation embedding."""
        return self._pi0.model.apply(
            {"params": {"vlm": self._backbone_params}},
            obs,
            method="encode_observation",
        )

    def _ensure_value_store(self, state: Array) -> None:
        """Lazily initialise the value FFN if no spec was given."""
        if self._value_store is None:
            obs_dim = int(state.shape[-1])
            spec = FFNModel(
                num_hidden_layers=2,
                hidden_layer_dim=256,
                output_dim=1,
                input_dim=obs_dim,
            )
            self._value_fn, self._value_store = _init_ffn(spec, seed=99)

    # ------------------------------------------------------------------
    # PPOTrainedModel interface
    # ------------------------------------------------------------------

    @classmethod
    def init_ppo_trained_model(cls, *args, **kwargs) -> Pi0Model:
        raise NotImplementedError(
            "Pi0Model is constructed directly; use Pi0Model(vla_spec, noise_ffn_spec)."
        )

    def get_policy_output(self, state: Array) -> Array:
        """Run Pi-0 denoising inference and record the FlowNoise log-prob."""
        action_head_params = self._policy_store["action_head"]
        noise_model_params = self._policy_store["noise_model"]

        action, denoising_steps, euler_steps = self._run_pi0_denoising(
            state, action_head_params
        )

        # Noise model conditions on (obs, A_τ) → log std per dimension.
        def noise_fn(params: Params, obs: Array, a_tau: Array) -> Array:
            inp = jnp.concatenate([obs, a_tau], axis=-1)
            return self._noise_model_fn(params, inp)

        log_prob = compute_flow_noise_log_prob(
            noise_fn,
            noise_model_params,
            state,
            denoising_steps,
            euler_steps,
        )
        self._last_log_prob = float(log_prob)
        return action

    def get_policy_output_log_prob(self) -> float:
        """Return the FlowNoise log-prob from the last get_policy_output call."""
        if self._last_log_prob is None:
            raise RuntimeError("Call get_policy_output before get_policy_output_log_prob.")
        return self._last_log_prob

    def get_value_output(self, state: Array) -> float:
        self._ensure_value_store(state)
        val = self._value_fn(self._value_store, state)
        return float(jnp.squeeze(val))

    def get_policy(self) -> Callable[[Params, Array], Array]:
        """Return a pure function (policy_store, obs) → action via Pi-0 + FlowNoise."""
        def _policy(policy_store: Params, obs: Array) -> Array:
            action, denoising_steps, euler_steps = self._run_pi0_denoising(
                obs, policy_store["action_head"]
            )
            return action
        return _policy

    def get_pure_policy_fn(self) -> Callable[[Params, Array], Array]:
        return self.get_policy()

    def get_value_func(self) -> Callable[[Params, Array], Array]:
        return self._value_fn  # type: ignore[return-value]

    def get_policy_store(self) -> Params:
        return self._policy_store

    def get_policy_store_mut(self) -> Params:
        return self._policy_store

    def get_value_store(self) -> Params:
        return self._value_store

    def get_value_store_mut(self) -> Params:
        return self._value_store

    def get_control_policy_output_mode(self) -> ControlPolicyOutputMode:
        return ControlPolicyOutputMode.TARGET_POSITIONS

    def get_reward(
        self,
        cstate: Array,
        cstate_next: Array,
        action: Array,
        contact_pairs: list[tuple[int, int]] | None,
    ) -> float:
        # Reward is determined by the user-supplied reward function, not by the model.
        return 0.0

    def get_pure_reward_fn(self) -> Callable[[Array, Array, Array, Array], Array]:
        def _reward(cstate: Array, cstate_next: Array, action: Array, contact: Array) -> Array:
            return jnp.zeros(())
        return _reward
