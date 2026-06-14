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

from commondef.training import ControlPolicyOutputMode, FFNModel, VLAModel
from ppo_model import FFNPPOModel
from ppo_trainer import PPOTrainedModel, TestingModel, compute_flow_noise_log_prob
from pretrained.pi0_jax import Pi0Jax

logger = logging.getLogger(__name__)

Array = jax.Array
Params = Any


def _init_ffn(ffn_spec: FFNModel, seed: int = 42) -> tuple[Callable, Params]:
    policy_fn, params = FFNPPOModel.init_policy(
        input_dimension=ffn_spec.input_dim,
        output_dimension=ffn_spec.output_dim,
        hidden_layer_dimension=ffn_spec.hidden_layer_dim,
        num_layers=ffn_spec.num_hidden_layers,
        add_softmax=False,
        control_policy=False,
        seed=seed,
    )
    return policy_fn, params


class Pi0Model(PPOTrainedModel, TestingModel):
    """PPO-trainable wrapper for the Pi-0 flow-matching VLA.

    Policy store: trainable_params from Pi0Jax (action expert + projections).
    Value store:  a small FFN initialised from value_ffn_spec.
    """

    def __init__(
        self,
        vla_spec: VLAModel,
        noise_ffn_spec: FFNModel,
        value_ffn_spec: FFNModel | None = None,
        seed: int = 0,
    ) -> None:
        checkpoint_dir = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")
        logger.info("Pi0Model: loading checkpoint from %s", checkpoint_dir)
        self._pi0 = Pi0Jax.from_checkpoint(checkpoint_dir)
        logger.info("Pi0Model: checkpoint loaded")

        self._vla_spec = vla_spec
        self._policy_store: Params = self._pi0.get_policy_store()
        self._frozen_store: Params = self._pi0.get_frozen_store()

        if value_ffn_spec is not None:
            self._value_fn, self._value_store = _init_ffn(value_ffn_spec, seed + 1)
        else:
            self._value_fn = None
            self._value_store = None

        self._last_log_prob: float | None = None
        self._rng_key = random.PRNGKey(seed)
        self._last_task: str = ""

    # ------------------------------------------------------------------
    # PPOTrainedModel interface
    # ------------------------------------------------------------------

    @classmethod
    def init_ppo_trained_model(cls, *args, **kwargs) -> "Pi0Model":
        raise NotImplementedError("Construct Pi0Model directly.")

    def get_policy_output(self, state: Array) -> Array:
        """Run Pi-0 inference on the current observation.

        `state` here is the flattened observation vector (images + proprio).
        The Pi0Jax.infer path requires structured inputs; this method expects
        the simulator to provide images and state separately via set_observation().
        Falls back to zero images if not set.
        """
        self._rng_key, key = random.split(self._rng_key)

        # Retrieve last structured observation set by the training loop
        images = getattr(self, "_last_images", None)
        task   = getattr(self, "_last_task", "")

        if images is None:
            logger.warning("Pi0Model: no images set; using zeros for inference")
            cfg = self._pi0._cfg
            res = cfg.get("image_resolution", [224, 224])
            n_cam = len(cfg.get("input_features", {}).get("observation.images", {})) or 1
            images = jnp.zeros((n_cam, res[0], res[1], 3), dtype=jnp.float32)

        actions = self._pi0.infer(
            self._policy_store,
            self._frozen_store,
            images,
            state,
            task,
            key,
        )
        # Return first action step as the control output
        return actions[0]

    def set_observation(self, images: Array, task: str) -> None:
        """Supply structured observation before calling get_policy_output."""
        self._last_images = images
        self._last_task = task

    def get_policy_output_log_prob(self) -> float:
        if self._last_log_prob is None:
            raise RuntimeError("Call get_policy_output before get_policy_output_log_prob.")
        return self._last_log_prob

    def get_value_output(self, state: Array) -> float:
        self._ensure_value_store(state)
        val = self._value_fn(self._value_store, state)
        return float(jnp.squeeze(val))

    def get_policy(self) -> Callable[[Params, Array], Array]:
        pi0 = self._pi0
        frozen = self._frozen_store
        task = self._last_task
        images = getattr(self, "_last_images", jnp.zeros((1, 224, 224, 3)))
        rng = self._rng_key

        def _policy(policy_store: Params, obs: Array) -> Array:
            actions = pi0.infer(policy_store, frozen, images, obs, task, rng)
            return actions[0]
        return _policy

    def get_pure_policy_fn(self) -> Callable[[Params, Array], Array]:
        return self.get_policy()

    def get_value_func(self) -> Callable[[Params, Array], Array]:
        return self._value_fn

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

    def get_reward(self, cstate, cstate_next, action, contact_pairs) -> float:
        return 0.0

    def get_pure_reward_fn(self) -> Callable:
        def _reward(cstate, cstate_next, action, contact):
            return jnp.zeros(())
        return _reward

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # TestingModel interface
    def get_test_action(self, state: Array) -> Array:
        return self.get_policy_output(state)

    def get_pure_test_fn(self) -> Callable[[Params, Array], Array]:
        return self.get_policy()

    def get_test_params(self) -> Params:
        return self.get_policy_store()

    def _ensure_value_store(self, state: Array) -> None:
        if self._value_store is None:
            obs_dim = int(state.shape[-1])
            spec = FFNModel(
                num_hidden_layers=2,
                hidden_layer_dim=256,
                output_dim=1,
                input_dim=obs_dim,
            )
            self._value_fn, self._value_store = _init_ffn(spec, seed=99)
