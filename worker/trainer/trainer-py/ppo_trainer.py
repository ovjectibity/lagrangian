from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
from jax import random

from simulator import Simulator


Array = jax.Array
Params = Any


@dataclass
class Trajectory:
    world_states: list[Array]
    model_outputs: list[Array]
    model_output_samples: list[Array]
    model_output_sample_log_probs: list[float]
    rewards: list[float]
    rtgs: list[float]


class PPOTrainerRuntimeInfo:
    def __init__(
        self,
        mean_policy_loss: list[float],
        mean_value_loss: list[float],
        policy_gradient_norm: list[float],
        value_gradient_norm: list[float],
    ) -> None:
        self._mean_policy_loss = mean_policy_loss
        self._mean_value_loss = mean_value_loss
        self._policy_gradient_norm = policy_gradient_norm
        self._value_gradient_norm = value_gradient_norm

    def mean_policy_loss(self) -> Sequence[float]:
        return self._mean_policy_loss

    def mean_value_loss(self) -> Sequence[float]:
        return self._mean_value_loss

    def policy_gradient_norm(self) -> Sequence[float]:
        return self._policy_gradient_norm

    def value_gradient_norm(self) -> Sequence[float]:
        return self._value_gradient_norm


class ControlPolicyOutputMode(Enum):
    TargetPositions = "TargetPositions"
    TargetVelocities = "TargetVelocities"
    TargetTorques = "TargetTorques"


class PPOTrainedModel(ABC):
    @classmethod
    @abstractmethod
    def init_ppo_trained_model(
        cls,
        policy: Callable[[Params, Array], Array],
        val_estimator: Callable[[Params, Array], Array],
        pol_store: Params,
        val_estimator_store: Params,
    ) -> PPOTrainedModel:
        pass

    @abstractmethod
    def get_reward(
        self,
        cstate: Array,
        cstate_next: Array,
        action: Array,
        contact_pairs: list[tuple[int, int]] | None,
    ) -> float:
        pass

    @abstractmethod
    def get_policy_output(self, state: Array) -> Array:
        pass

    @abstractmethod
    def get_value_output(self, state: Array) -> float:
        pass

    @abstractmethod
    def get_policy(self) -> Callable[[Params, Array], Array]:
        pass

    @abstractmethod
    def get_value_func(self) -> Callable[[Params, Array], Array]:
        pass

    @abstractmethod
    def get_policy_store(self) -> Params:
        pass

    @abstractmethod
    def get_policy_store_mut(self) -> Params:
        pass

    @abstractmethod
    def get_value_store(self) -> Params:
        pass

    @abstractmethod
    def get_value_store_mut(self) -> Params:
        pass

    @abstractmethod
    def get_control_policy_output_mode(self) -> ControlPolicyOutputMode:
        pass


# Backward compatible name requested by prompt text.
PPoTrainedModel = PPOTrainedModel


class PPOTrainer:
    def __init__(
        self,
        simulator: Simulator,
        model: PPOTrainedModel,
        num_trajs: int,
        traj_depth: int,
        num_updates: int,
        epsilon: float,
        learning_rate: float = 1e-3,
        gamma_lambda: tuple[float, float] = (0.99, 0.95),
        seed: int = 0,
    ) -> None:
        self.num_trajs = num_trajs
        self.traj_depth = traj_depth
        self.num_updates = num_updates
        self.epsilon = epsilon
        self.gamma_lambda = gamma_lambda
        self.simulator = simulator
        self.model = model
        self.learning_rate = learning_rate
        self._rng_key = random.PRNGKey(seed)
        self.policy_opt = self._init_adam_state(self.model.get_policy_store())
        self.value_opt = self._init_adam_state(self.model.get_value_store())

    @staticmethod
    def _init_adam_state(params: Params) -> dict[str, Any]:
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return {"m": zeros, "v": zeros, "t": 0}

    def _adam_step(
        self,
        params: Params,
        grads: Params,
        opt_state: dict[str, Any],
    ) -> tuple[Params, dict[str, Any]]:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = opt_state["t"] + 1
        m = jax.tree_util.tree_map(
            lambda m_old, g: beta1 * m_old + (1.0 - beta1) * g,
            opt_state["m"],
            grads,
        )
        v = jax.tree_util.tree_map(
            lambda v_old, g: beta2 * v_old + (1.0 - beta2) * (g * g),
            opt_state["v"],
            grads,
        )
        m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**t), m)
        v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**t), v)
        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - self.learning_rate * mh / (jnp.sqrt(vh) + eps),
            params,
            m_hat,
            v_hat,
        )
        new_state = {"m": m, "v": v, "t": t}
        return new_params, new_state

    def get_action_output_size(self) -> int:
        return self.simulator.nu()

    def gen_trajectory(self) -> Trajectory:
        logger.info("gen_trajectory: start (traj_depth=%s)", self.traj_depth)
        action_output_size = self.get_action_output_size()
        wstates: list[Array] = []
        model_outputs: list[Array] = []
        model_output_samples: list[Array] = []
        rewards: list[float] = []
        self.simulator.reset_state()
        logger.info("gen_trajectory: reset_state done, entering step loop")
        for step in range(self.traj_depth):
            # if step % 10 == 0:
            logger.info("gen_trajectory: step %s/%s", step, self.traj_depth)
            wstate = jnp.asarray(self.simulator.get_model_state())
            cstate = jnp.asarray(self.simulator.get_model_cartesian_state())
            logger.info("gen_trajectory: step %s/%s. Got the wstate & cstate")
            action_gaussian_params = jnp.asarray(self.model.get_policy_output(wstate))
            action_gaussian_mean = action_gaussian_params[:action_output_size]
            action_gaussian_stdev = jnp.exp(action_gaussian_params[action_output_size:])
            self._rng_key, eps_key = random.split(self._rng_key)
            eps = random.normal(eps_key, shape=(action_output_size,))
            sampled_action = action_gaussian_mean + eps * action_gaussian_stdev
            logger.info("gen_trajectory: step %s/%s. Got the policy output")
            output_mode = self.model.get_control_policy_output_mode()
            if output_mode == ControlPolicyOutputMode.TargetTorques:
                controls = sampled_action
            elif output_mode == ControlPolicyOutputMode.TargetPositions:
                controls = self.simulator.modify_target_position(sampled_action)
            elif output_mode == ControlPolicyOutputMode.TargetVelocities:
                controls = self.simulator.get_target_controls_from_target_velocities(
                    sampled_action
                )
            else:
                raise ValueError(f"Unsupported control mode: {output_mode}")
            logger.info("gen_trajectory: step %s/%s. Got the output control. Running forward_dyn now")
            self.simulator.forward_dyn(jnp.asarray(controls))
            logger.info("gen_trajectory: step %s/%s. Ran the forward_dyn. Getting reward now")
            cstate_next = jnp.asarray(self.simulator.get_model_cartesian_state())
            contact_pairs = self.simulator.get_contact_body_index_pairs()
            reward = float(
                self.model.get_reward(
                    cstate,
                    cstate_next,
                    sampled_action,
                    contact_pairs,
                )
            )
            logger.info("gen_trajectory: step %s/%s. Got reward now")
            rewards.append(reward)
            wstates.append(wstate)
            model_outputs.append(action_gaussian_params)
            model_output_samples.append(sampled_action)
        logger.info("gen_trajectory: step loop done, computing terminal value")
        wstate_end = jnp.asarray(self.simulator.get_model_state())
        wstates.append(wstate_end)
        v_terminal = float(self.model.get_value_output(wstates[-1]))
        logger.info("gen_trajectory: computing rtg and log probs")
        rtgs = self.compute_rtg(rewards, v_terminal)
        action_sample_log_probs = self.compute_action_log_probs(
            jnp.stack(model_output_samples),
            jnp.stack(model_outputs),
        )
        logger.info("gen_trajectory: done")
        return Trajectory(
            world_states=wstates,
            model_outputs=model_outputs,
            model_output_samples=model_output_samples,
            model_output_sample_log_probs=[
                float(x) for x in jnp.asarray(action_sample_log_probs)
            ],
            rewards=rewards,
            rtgs=rtgs,
        )

    def gen_trajectories(self, num_trajs: int) -> list[Trajectory]:
        trajs = []
        for i in range(num_trajs):
            logger.info("gen_trajectories: starting traj %s/%s", i + 1, num_trajs)
            traj = self.gen_trajectory()
            logger.info("gen_trajectories: completed traj %s/%s", i + 1, num_trajs)
            trajs.append(traj)
        return trajs

    def compute_gae(self, traj: Trajectory) -> Array:
        advs: list[float] = []
        for i in reversed(range(self.traj_depth)):
            wstate = traj.world_states[i]
            reward = traj.rewards[i]
            val = float(self.model.get_value_output(wstate))
            wstate_next = traj.world_states[i + 1]
            val_next = float(self.model.get_value_output(wstate_next))
            adv = reward + (self.gamma_lambda[0] * val_next) - val
            advs.append(adv)
        advs.reverse()
        gaes: list[Array] = []
        for i in range(self.traj_depth):
            exps = jnp.arange(self.traj_depth - i, dtype=jnp.float32)
            weights = (self.gamma_lambda[0] * self.gamma_lambda[1]) ** exps
            advs_t = jnp.asarray(advs[i:], dtype=jnp.float32)
            gaes.append(jnp.sum(advs_t * weights))
        return jnp.asarray(gaes)

    def compute_rtg(self, rewards: list[float], v_terminal: float) -> list[float]:
        rtgs = [0.0 for _ in range(self.traj_depth)]
        running_rtg = v_terminal
        for i in reversed(range(self.traj_depth)):
            running_rtg = rewards[i] + (self.gamma_lambda[0] * running_rtg)
            rtgs[i] = float(running_rtg)
        return rtgs

    def compute_action_log_probs(
        self,
        model_output_sample: Array,
        model_output: Array,
    ) -> Array:
        action_output_size = self.get_action_output_size()
        action_gaussian_mean = model_output[..., :action_output_size]
        action_gaussian_log_stddev = model_output[..., action_output_size:]
        action_gaussian_stddev = jnp.exp(action_gaussian_log_stddev)
        normalized = (model_output_sample - action_gaussian_mean) / action_gaussian_stddev
        log_two_pi = jnp.log(2.0 * jnp.pi)
        log_probs = -0.5 * (
            normalized**2 + (action_gaussian_log_stddev * 2.0) + log_two_pi
        )
        return jnp.sum(log_probs, axis=-1)

    def get_ppo_clip_objective(
        self,
        trajs: Sequence[Trajectory],
        gaes: Array,
        policy_store: Params | None = None,
    ) -> Array:
        start_log_probs = jnp.stack(
            [jnp.asarray(traj.model_output_sample_log_probs) for traj in trajs]
        )
        wstates = jnp.stack(
            [jnp.stack(traj.world_states[: self.traj_depth]) for traj in trajs]
        )
        action_samples = jnp.stack(
            [jnp.stack(traj.model_output_samples) for traj in trajs]
        )
        if policy_store is None:
            policy_store = self.model.get_policy_store()
        action_gaussian_params = self.model.get_policy()(policy_store, wstates)
        action_log_probs = self.compute_action_log_probs(action_samples, action_gaussian_params)
        ratio = jnp.exp(action_log_probs - start_log_probs)
        clipped = jnp.clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * gaes
        return jnp.mean(jnp.minimum(ratio * gaes, clipped))

    def get_value_function_objective(
        self,
        trajs: Sequence[Trajectory],
        value_store: Params | None = None,
    ) -> Array:
        rtgs = jnp.stack([jnp.asarray(traj.rtgs) for traj in trajs])
        wstates = jnp.stack(
            [jnp.stack(traj.world_states[: self.traj_depth]) for traj in trajs]
        )
        if value_store is None:
            value_store = self.model.get_value_store()
        values = self.model.get_value_func()(value_store, wstates)
        values = jnp.asarray(values)
        if values.ndim > 0 and values.shape[-1] == 1:
            values = jnp.squeeze(values, axis=-1)
        return jnp.mean((values - rtgs) ** 2)

    def get_mean_policy_loss(
        self,
        trajs: Sequence[Trajectory],
        gaes: Array,
        policy_store: Params | None = None,
    ) -> Array:
        return -self.get_ppo_clip_objective(trajs, gaes, policy_store)

    def get_mean_value_loss(
        self,
        trajs: Sequence[Trajectory],
        value_store: Params | None = None,
    ) -> Array:
        return self.get_value_function_objective(trajs, value_store)

    def train_an_epoch(self, trajs: Sequence[Trajectory]) -> PPOTrainerRuntimeInfo:
        gae_v = [self.compute_gae(traj) for traj in trajs]
        gaes = jnp.stack(gae_v)
        gae_mean = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        gaes = (gaes - gae_mean) / (gae_std + 1e-8)
        mean_policy_loss: list[float] = []
        mean_value_loss: list[float] = []
        policy_gradient_norm: list[float] = []
        value_gradient_norm: list[float] = []
        for _ in range(self.num_updates):
            policy_store = self.model.get_policy_store()
            policy_loss_fn = lambda p: self.get_mean_policy_loss(trajs, gaes, p)
            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_store)
            mean_policy_loss.append(float(policy_loss))
            policy_gradient_norm.append(self.get_gradient_norm(policy_grads))
            policy_store, self.policy_opt = self._adam_step(
                policy_store, policy_grads, self.policy_opt
            )
            self.model.get_policy_store_mut().update(policy_store)

            value_store = self.model.get_value_store()
            value_loss_fn = lambda v: self.get_mean_value_loss(trajs, v)
            value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_store)
            mean_value_loss.append(float(value_loss))
            value_gradient_norm.append(self.get_gradient_norm(value_grads))
            value_store, self.value_opt = self._adam_step(
                value_store, value_grads, self.value_opt
            )
            self.model.get_value_store_mut().update(value_store)
        return PPOTrainerRuntimeInfo(
            mean_policy_loss,
            mean_value_loss,
            policy_gradient_norm,
            value_gradient_norm,
        )

    @staticmethod
    def get_gradient_norm(variables: Params) -> float:
        total_norm_sq = 0.0
        for grad in jax.tree_util.tree_leaves(variables):
            total_norm_sq += float(jnp.sum(grad**2))
        return total_norm_sq**0.5

    def get_rollout_mean_entropy(self, trajs: Sequence[Trajectory]) -> float:
        if not trajs:
            return 0.0
        action_output_size = self.get_action_output_size()
        model_outputs = jnp.stack([jnp.stack(traj.model_outputs) for traj in trajs])
        action_gaussian_log_stddev = model_outputs[..., action_output_size:]
        log_two_pi = jnp.log(2.0 * jnp.pi)
        entropy = jnp.mean(
            jnp.sum(action_gaussian_log_stddev + 0.5 * (1.0 + log_two_pi), axis=-1)
        )
        return float(entropy)

    def get_rollout_mean_reward(self, trajs: Sequence[Trajectory]) -> float:
        reward_count = sum(len(traj.rewards) for traj in trajs)
        if reward_count == 0:
            return 0.0
        reward_sum = sum(sum(traj.rewards) for traj in trajs)
        return reward_sum / reward_count
