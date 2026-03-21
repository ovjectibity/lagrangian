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

    @abstractmethod
    def get_pure_policy_fn(self) -> Callable[[Params, Array], Array]:
        """Returns a pure, JAX-traceable policy function: (params, state) -> gaussian_params."""
        pass

    @abstractmethod
    def get_pure_reward_fn(self) -> Callable[[Array, Array, Array, Array], Array]:
        """Returns a pure, JAX-traceable reward function.

        Signature: (cstate, cstate_next, action, contact_matrix) -> scalar reward
        where contact_matrix is a float32 [nbody, nbody] array with 1.0 for active contacts.
        """
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

    def gen_trajectory_step(
        self,
        mjx_data: Any,
        wstate: Array,
        cstate: Array,
        rng_key: Array,
        policy_fn: Callable[[Params, Array], Array],
        policy_store: Params,
        output_mode: ControlPolicyOutputMode,
        reward_fn: Callable[[Array, Array, Array, Array], Array],
        simulator_step_fn: Callable[[Any, Array], tuple[Any, Array, Array, Array]],
    ) -> tuple[Any, Array, Array, Array, Array, Array, Array]:
        """Pure functional single trajectory step.

        Args:
            mjx_data: current mjx simulation state (None for CPU simulator)
            wstate: current world state (joint positions + velocities)
            cstate: current cartesian state (body xpos + xquat)
            rng_key: JAX RNG key for action sampling
            policy_fn: callable (params, state) -> gaussian_params
            policy_store: policy network parameters
            output_mode: how to interpret policy output as controls
            reward_fn: callable (cstate, cstate_next, action, contact_matrix) -> scalar
            simulator_step_fn: callable (mjx_data, controls) -> (mjx_data_next, wstate_next, cstate_next, contact_matrix)

        Returns:
            (mjx_data_next, wstate_next, cstate_next, model_output, model_output_sample, reward, rng_key_next)
        """
        action_output_size = self.get_action_output_size()

        action_gaussian_params = jnp.asarray(policy_fn(policy_store, wstate))
        action_gaussian_mean = action_gaussian_params[:action_output_size]
        action_gaussian_stdev = jnp.exp(action_gaussian_params[action_output_size:])

        rng_key_next, eps_key = random.split(rng_key)
        eps = random.normal(eps_key, shape=(action_output_size,))
        sampled_action = action_gaussian_mean + eps * action_gaussian_stdev

        if output_mode == ControlPolicyOutputMode.TargetTorques:
            controls = sampled_action
        elif output_mode == ControlPolicyOutputMode.TargetPositions:
            controls = sampled_action
        elif output_mode == ControlPolicyOutputMode.TargetVelocities:
            controls = sampled_action
        else:
            raise ValueError(f"Unsupported control mode: {output_mode}")

        mjx_data_next, wstate_next, cstate_next, contact_matrix = simulator_step_fn(mjx_data, controls)

        reward = reward_fn(cstate, cstate_next, sampled_action, contact_matrix)

        return mjx_data_next, wstate_next, cstate_next, action_gaussian_params, sampled_action, reward, rng_key_next

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

    def train_an_epoch(
        self,
        trajs: Sequence[Trajectory],
        post_update_callback: Callable[[int], None] | None = None,
    ) -> PPOTrainerRuntimeInfo:
        logger.info("train_an_epoch: computing GAEs for %s trajs", len(trajs))
        gae_v = [self.compute_gae(traj) for traj in trajs]
        gaes = jnp.stack(gae_v)
        gae_mean = jnp.mean(gaes)
        gae_std = jnp.std(gaes)
        gaes = (gaes - gae_mean) / (gae_std + 1e-8)
        logger.info("train_an_epoch: GAE mean=%.4f std=%.4f", float(gae_mean), float(gae_std))
        mean_policy_loss: list[float] = []
        mean_value_loss: list[float] = []
        policy_gradient_norm: list[float] = []
        value_gradient_norm: list[float] = []
        for update_idx in range(self.num_updates):
            logger.info("train_an_epoch: update %s/%s — computing policy loss", update_idx + 1, self.num_updates)
            policy_store = self.model.get_policy_store()
            policy_loss_fn = lambda p: self.get_mean_policy_loss(trajs, gaes, p)
            policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_store)
            mean_policy_loss.append(float(policy_loss))
            p_grad_norm = self.get_gradient_norm(policy_grads)
            policy_gradient_norm.append(p_grad_norm)
            logger.info("train_an_epoch: update %s/%s — policy_loss=%.6f grad_norm=%.6f", update_idx + 1, self.num_updates, float(policy_loss), p_grad_norm)
            policy_store, self.policy_opt = self._adam_step(
                policy_store, policy_grads, self.policy_opt
            )
            self.model.get_policy_store_mut().update(policy_store)

            logger.info("train_an_epoch: update %s/%s — computing value loss", update_idx + 1, self.num_updates)
            value_store = self.model.get_value_store()
            value_loss_fn = lambda v: self.get_mean_value_loss(trajs, v)
            value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_store)
            mean_value_loss.append(float(value_loss))
            v_grad_norm = self.get_gradient_norm(value_grads)
            value_gradient_norm.append(v_grad_norm)
            logger.info("train_an_epoch: update %s/%s — value_loss=%.6f grad_norm=%.6f", update_idx + 1, self.num_updates, float(value_loss), v_grad_norm)
            value_store, self.value_opt = self._adam_step(
                value_store, value_grads, self.value_opt
            )
            self.model.get_value_store_mut().update(value_store)
            if post_update_callback is not None:
                post_update_callback(update_idx)
        logger.info("train_an_epoch: done")
        return PPOTrainerRuntimeInfo(
            mean_policy_loss,
            mean_value_loss,
            policy_gradient_norm,
            value_gradient_norm,
        )

    @staticmethod
    def get_gradient_norm(variables: Params) -> float:
        # Note: pure version available as module-level pure_gradient_norm()
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

    def gen_trajectory_gpu(
        self,
        mjx_data: Any,
        rng_key: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Generate a single trajectory on GPU using lax.scan.

        The simulator must be an MjxSimulator. Uses pure functions throughout
        so the scan body is JAX-traceable.

        Args:
            mjx_data: initial mjx simulation state (from MjxSimulator after reset)
            rng_key:  JAX RNG key for action sampling

        Returns:
            (wstates, cstate_seq, model_outputs, action_samples, rewards)
            wstates:       [traj_depth+1, state_dim]  — includes terminal state
            cstate_seq:    [traj_depth, cstate_dim]
            model_outputs: [traj_depth, nu*2]
            action_samples:[traj_depth, nu]
            rewards:       [traj_depth]
        """
        logger.debug(
            "gen_trajectory_gpu: start (traj_depth=%s rng_key=%s)",
            self.traj_depth,
            rng_key,
        )
        simulator_step_fn = self.simulator.forward_dyn_pure  # type: ignore[attr-defined]
        policy_fn = self.model.get_pure_policy_fn()
        reward_fn = self.model.get_pure_reward_fn()
        policy_store = self.model.get_policy_store()
        output_mode = self.model.get_control_policy_output_mode()

        # Derive initial wstate and cstate from mjx_data directly (pure, no side effects)
        nq = self.simulator.nq()
        nv = self.simulator.nv()
        nbody = self.simulator.nbody()
        logger.debug(
            "gen_trajectory_gpu: nq=%s nv=%s nbody=%s output_mode=%s",
            nq, nv, nbody, output_mode,
        )

        def initial_states(data: Any) -> tuple[Array, Array]:
            wstate = jnp.concatenate([data.qpos[:nq], data.qvel[:nv]])
            xpos = data.xpos[:nbody].reshape(-1)
            xquat = data.xquat[:nbody].reshape(-1)
            cstate = jnp.concatenate([xpos, xquat])
            return wstate, cstate

        wstate0, cstate0 = initial_states(mjx_data)
        logger.debug(
            "gen_trajectory_gpu: initial wstate shape=%s cstate shape=%s",
            wstate0.shape,
            cstate0.shape,
        )
        logger.debug("gen_trajectory_gpu: launching lax.scan over %s steps", self.traj_depth)

        def scan_body(
            carry: tuple[Any, Array, Array, Array],
            _: None,
        ) -> tuple[tuple[Any, Array, Array, Array], tuple[Array, Array, Array, Array, Array]]:
            mjx_data_c, wstate, cstate, rng = carry
            mjx_data_next, wstate_next, cstate_next, model_output, action_sample, reward, rng_next = (
                self.gen_trajectory_step(
                    mjx_data_c,
                    wstate,
                    cstate,
                    rng,
                    policy_fn,
                    policy_store,
                    output_mode,
                    reward_fn,
                    simulator_step_fn,
                )
            )
            return (mjx_data_next, wstate_next, cstate_next, rng_next), (
                wstate,
                cstate_next,
                model_output,
                action_sample,
                reward,
            )

        (_, wstate_final, _, _), (wstates_t, cstate_seq, model_outputs, action_samples, rewards) = (
            jax.lax.scan(scan_body, (mjx_data, wstate0, cstate0, rng_key), None, length=self.traj_depth)
        )
        logger.debug(
            "gen_trajectory_gpu: lax.scan complete — wstates_t shape=%s rewards shape=%s",
            wstates_t.shape,
            rewards.shape,
        )

        # Append terminal wstate so shape is [traj_depth+1, state_dim]
        wstates = jnp.concatenate([wstates_t, wstate_final[None]], axis=0)
        logger.debug("gen_trajectory_gpu: done (wstates shape=%s)", wstates.shape)

        return wstates, cstate_seq, model_outputs, action_samples, rewards

    def gen_trajectories_gpu(
        self,
        num_trajs: int,
        mjx_data: Any,
        rng_key: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Generate num_trajs trajectories in parallel via vmap on GPU.

        Args:
            num_trajs: number of parallel trajectories
            mjx_data:  initial mjx simulation state (will be replicated across trajs)
            rng_key:   base RNG key; split into num_trajs sub-keys

        Returns:
            (wstates, cstate_seq, model_outputs, action_samples, rewards)
            wstates:       [num_trajs, traj_depth+1, state_dim]
            cstate_seq:    [num_trajs, traj_depth, cstate_dim]
            model_outputs: [num_trajs, traj_depth, nu*2]
            action_samples:[num_trajs, traj_depth, nu]
            rewards:       [num_trajs, traj_depth]
        """
        logger.debug(
            "gen_trajectories_gpu: start (num_trajs=%s traj_depth=%s)",
            num_trajs,
            self.traj_depth,
        )
        keys = jax.random.split(rng_key, num_trajs)
        logger.debug("gen_trajectories_gpu: split %s RNG keys, launching jit(vmap)", num_trajs)

        def single_traj(key: Array) -> tuple[Array, Array, Array, Array, Array]:
            return self.gen_trajectory_gpu(mjx_data, key)

        result = jax.jit(jax.vmap(single_traj))(keys)
        logger.debug(
            "gen_trajectories_gpu: jit(vmap) complete — wstates shape=%s rewards shape=%s",
            result[0].shape,
            result[4].shape,
        )
        return result

    def train_an_epoch_gpu(
        self,
        wstates: Array,
        action_samples: Array,
        rewards: Array,
        model_outputs: Array,
    ) -> PPOTrainerRuntimeInfo:
        """GPU training epoch using pure learning_step.

        Args:
            wstates:       [num_trajs, traj_depth+1, state_dim]
            action_samples:[num_trajs, traj_depth, nu]
            rewards:       [num_trajs, traj_depth]
            model_outputs: [num_trajs, traj_depth, nu*2]

        Returns:
            PPOTrainerRuntimeInfo with per-update losses and gradient norms
        """
        num_trajs_in = wstates.shape[0]
        logger.info(
            "train_an_epoch_gpu: start (num_trajs=%s traj_depth=%s action_size=%s)",
            num_trajs_in,
            self.traj_depth,
            self.get_action_output_size(),
        )
        policy_fn = self.model.get_pure_policy_fn()
        value_fn = self.model.get_value_func()
        action_size = self.get_action_output_size()
        gamma, lam = self.gamma_lambda

        logger.info("train_an_epoch_gpu: computing RTGs for %s trajs", num_trajs_in)

        def compute_rtg_for_traj(traj_rewards: Array, traj_wstates: Array) -> Array:
            v_terminal = jnp.squeeze(value_fn(self.model.get_value_store(), traj_wstates[-1:]), axis=-1)[0]
            return pure_compute_rtg(traj_rewards, gamma, v_terminal)

        rtgs = jax.vmap(compute_rtg_for_traj)(rewards, wstates)   # [num_trajs, traj_depth]
        logger.info("train_an_epoch_gpu: RTGs computed (shape=%s)", rtgs.shape)

        logger.info("train_an_epoch_gpu: computing GAEs for %s trajs", num_trajs_in)

        def compute_gae_for_traj(traj_wstates: Array, traj_rewards: Array) -> Array:
            return pure_compute_gae(traj_wstates, traj_rewards, value_fn, self.model.get_value_store(), gamma, lam)

        gaes_raw = jax.vmap(compute_gae_for_traj)(wstates, rewards)  # [num_trajs, traj_depth]

        gae_mean = jnp.mean(gaes_raw)
        gae_std = jnp.std(gaes_raw)
        gaes = (gaes_raw - gae_mean) / (gae_std + 1e-8)
        logger.info("train_an_epoch_gpu: GAE mean=%.4f std=%.4f (shape=%s)", float(gae_mean), float(gae_std), gaes.shape)

        logger.info("train_an_epoch_gpu: computing start log probs")
        # Compute start log probs from the model outputs collected during rollout
        start_log_probs = jax.vmap(
            lambda mo, sa: pure_action_log_probs(sa, mo, action_size)
        )(model_outputs, action_samples)   # [num_trajs, traj_depth]
        logger.info("train_an_epoch_gpu: start log probs computed (shape=%s)", start_log_probs.shape)

        # wstates for learning uses only the non-terminal steps
        wstates_t = wstates[:, :self.traj_depth, :]  # [num_trajs, traj_depth, state_dim]

        epsilon = self.epsilon
        learning_rate = self.learning_rate

        def update_step(
            carry: tuple[Params, Params, dict, dict],
            _: None,
        ) -> tuple[tuple[Params, Params, dict, dict], tuple[Array, Array, Array, Array]]:
            policy_store_c, value_store_c, policy_opt_c, value_opt_c = carry
            (
                policy_store_next,
                value_store_next,
                policy_opt_next,
                value_opt_next,
                p_loss,
                v_loss,
                p_grad_norm,
                v_grad_norm,
            ) = learning_step(
                policy_store_c,
                value_store_c,
                policy_opt_c,
                value_opt_c,
                policy_fn,
                value_fn,
                wstates_t,
                action_samples,
                start_log_probs,
                gaes,
                rtgs,
                epsilon,
                action_size,
                learning_rate,
            )
            return (policy_store_next, value_store_next, policy_opt_next, value_opt_next), (
                p_loss, v_loss, p_grad_norm, v_grad_norm,
            )

        logger.info(
            "train_an_epoch_gpu: launching jit(lax.scan) over %s update steps", self.num_updates
        )
        init_carry = (
            self.model.get_policy_store(),
            self.model.get_value_store(),
            self.policy_opt,
            self.value_opt,
        )
        (policy_store_final, value_store_final, policy_opt_final, value_opt_final), (
            p_losses, v_losses, p_grad_norms, v_grad_norms,
        ) = jax.jit(lambda c: jax.lax.scan(update_step, c, None, length=self.num_updates))(init_carry)

        self.policy_opt = policy_opt_final
        self.value_opt = value_opt_final
        self.model.get_policy_store_mut().update(policy_store_final)
        self.model.get_value_store_mut().update(value_store_final)

        # Extract per-update scalars for logging and return value (done outside jit)
        mean_policy_loss = [float(p_losses[i]) for i in range(self.num_updates)]
        mean_value_loss = [float(v_losses[i]) for i in range(self.num_updates)]
        policy_gradient_norm = [float(p_grad_norms[i]) for i in range(self.num_updates)]
        value_gradient_norm = [float(v_grad_norms[i]) for i in range(self.num_updates)]

        for update_idx in range(self.num_updates):
            logger.info(
                "train_an_epoch_gpu: update %s/%s — policy_loss=%.6f value_loss=%.6f grad_norm_p=%.6f grad_norm_v=%.6f",
                update_idx + 1, self.num_updates,
                mean_policy_loss[update_idx], mean_value_loss[update_idx],
                policy_gradient_norm[update_idx], value_gradient_norm[update_idx],
            )

        logger.info("train_an_epoch_gpu: done")
        return PPOTrainerRuntimeInfo(
            mean_policy_loss,
            mean_value_loss,
            policy_gradient_norm,
            value_gradient_norm,
        )


# ---------------------------------------------------------------------------
# Pure standalone functions for GPU-accelerated learning
#
# Trajectory arrays convention (all [num_trajs, traj_depth, ...]):
#   wstates:        [num_trajs, traj_depth+1, state_dim]  (includes terminal state)
#   action_samples: [num_trajs, traj_depth, nu]
#   model_outputs:  [num_trajs, traj_depth, nu*2]
#   log_probs:      [num_trajs, traj_depth]
#   rewards:        [num_trajs, traj_depth]
#   rtgs:           [num_trajs, traj_depth]
# ---------------------------------------------------------------------------


def pure_gradient_norm(variables: Params) -> Array:
    leaves = jax.tree_util.tree_leaves(variables)
    sq_sum = jnp.array(0.0)
    for leaf in leaves:
        sq_sum = sq_sum + jnp.sum(leaf ** 2)
    return jnp.sqrt(sq_sum)


def pure_compute_rtg(rewards: Array, gamma: float, v_terminal: Array) -> Array:
    """Compute rewards-to-go for a single trajectory without a Python loop.

    Args:
        rewards:    [T] float array
        gamma:      discount factor
        v_terminal: scalar, value estimate at the terminal state

    Returns:
        rtgs: [T] float array
    """
    def scan_fn(running_rtg: Array, reward: Array) -> tuple[Array, Array]:
        running_rtg = reward + gamma * running_rtg
        return running_rtg, running_rtg

    _, rtgs_reversed = jax.lax.scan(
        scan_fn,
        v_terminal,
        jnp.flip(rewards, axis=0),
    )
    return jnp.flip(rtgs_reversed, axis=0)


def pure_compute_gae(
    wstates: Array,
    rewards: Array,
    value_fn: Callable[[Params, Array], Array],
    value_store: Params,
    gamma: float,
    lam: float,
) -> Array:
    """Compute generalised advantage estimates for a single trajectory without a Python loop.

    Args:
        wstates:     [T+1, state_dim] — includes terminal state at index T
        rewards:     [T] float array
        value_fn:    callable (params, state) -> [1] or scalar
        value_store: value network parameters
        gamma:       discount factor
        lam:         GAE lambda

    Returns:
        gaes: [T] float array
    """
    # Batch value computation over all T+1 states in one call
    all_values = value_fn(value_store, wstates)                  # [T+1, 1] or [T+1]
    all_values = jnp.squeeze(jnp.asarray(all_values), axis=-1)   # [T+1]
    values = all_values[:-1]       # [T]
    values_next = all_values[1:]   # [T]

    deltas = rewards + gamma * values_next - values              # [T]

    def scan_fn(gae_next: Array, delta: Array) -> tuple[Array, Array]:
        gae = delta + gamma * lam * gae_next
        return gae, gae

    _, gaes_reversed = jax.lax.scan(
        scan_fn,
        jnp.array(0.0),
        jnp.flip(deltas, axis=0),
    )
    return jnp.flip(gaes_reversed, axis=0)


def pure_action_log_probs(
    action_samples: Array,
    model_outputs: Array,
    action_size: int,
) -> Array:
    """Compute per-step log probs of sampled actions under the gaussian policy.

    Args:
        action_samples: [..., nu]
        model_outputs:  [..., nu*2] (mean concat log_std)
        action_size:    nu

    Returns:
        log_probs: [...] (summed over action dims)
    """
    mean = model_outputs[..., :action_size]
    log_std = model_outputs[..., action_size:]
    std = jnp.exp(log_std)
    normalized = (action_samples - mean) / std
    log_two_pi = jnp.log(2.0 * jnp.pi)
    log_probs = -0.5 * (normalized ** 2 + log_std * 2.0 + log_two_pi)
    return jnp.sum(log_probs, axis=-1)


def pure_ppo_clip_loss(
    policy_store: Params,
    policy_fn: Callable[[Params, Array], Array],
    wstates: Array,
    action_samples: Array,
    start_log_probs: Array,
    gaes: Array,
    epsilon: float,
    action_size: int,
) -> Array:
    """PPO clipped surrogate loss (negated for gradient ascent via minimisation).

    Args:
        policy_store:    policy network parameters
        policy_fn:       callable (params, [num_trajs, T, state_dim]) -> [num_trajs, T, nu*2]
        wstates:         [num_trajs, T, state_dim]
        action_samples:  [num_trajs, T, nu]
        start_log_probs: [num_trajs, T]
        gaes:            [num_trajs, T] normalised advantages
        epsilon:         PPO clip ratio
        action_size:     nu

    Returns:
        scalar loss (negated clip objective)
    """
    model_outputs = policy_fn(policy_store, wstates)
    log_probs = pure_action_log_probs(action_samples, model_outputs, action_size)
    ratio = jnp.exp(log_probs - start_log_probs)
    clipped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * gaes
    return -jnp.mean(jnp.minimum(ratio * gaes, clipped))


def pure_value_loss(
    value_store: Params,
    value_fn: Callable[[Params, Array], Array],
    wstates: Array,
    rtgs: Array,
) -> Array:
    """MSE value function loss.

    Args:
        value_store: value network parameters
        value_fn:    callable (params, [num_trajs, T, state_dim]) -> [num_trajs, T, 1]
        wstates:     [num_trajs, T, state_dim]
        rtgs:        [num_trajs, T]

    Returns:
        scalar loss
    """
    values = value_fn(value_store, wstates)
    values = jnp.asarray(values)
    if values.ndim > 0 and values.shape[-1] == 1:
        values = jnp.squeeze(values, axis=-1)
    return jnp.mean((values - rtgs) ** 2)


def learning_step(
    policy_store: Params,
    value_store: Params,
    policy_opt: dict,
    value_opt: dict,
    policy_fn: Callable[[Params, Array], Array],
    value_fn: Callable[[Params, Array], Array],
    wstates: Array,
    action_samples: Array,
    start_log_probs: Array,
    gaes: Array,
    rtgs: Array,
    epsilon: float,
    action_size: int,
    learning_rate: float,
) -> tuple[Params, Params, dict, dict, Array, Array, Array, Array]:
    """Single pure parameter update step for policy and value networks.

    Computes gradients via reverse-mode autodiff and applies an Adam step.
    No Python-level side effects — all state is passed in and returned.

    Args:
        policy_store:    current policy parameters
        value_store:     current value parameters
        policy_opt:      Adam state for policy {"m", "v", "t"}
        value_opt:       Adam state for value  {"m", "v", "t"}
        policy_fn:       callable (params, wstates) -> model_outputs
        value_fn:        callable (params, wstates) -> values
        wstates:         [num_trajs, T, state_dim]
        action_samples:  [num_trajs, T, nu]
        start_log_probs: [num_trajs, T]
        gaes:            [num_trajs, T] normalised advantages
        rtgs:            [num_trajs, T]
        epsilon:         PPO clip ratio
        action_size:     nu
        learning_rate:   Adam learning rate

    Returns:
        (policy_store_next, value_store_next, policy_opt_next, value_opt_next,
         policy_loss, value_loss, policy_grad_norm, value_grad_norm)
    """
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8

    def adam_step(
        params: Params,
        grads: Params,
        opt_state: dict,
    ) -> tuple[Params, dict]:
        t = opt_state["t"] + 1
        m = jax.tree_util.tree_map(
            lambda m_old, g: beta1 * m_old + (1.0 - beta1) * g,
            opt_state["m"], grads,
        )
        v = jax.tree_util.tree_map(
            lambda v_old, g: beta2 * v_old + (1.0 - beta2) * (g * g),
            opt_state["v"], grads,
        )
        m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), m)
        v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2 ** t), v)
        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - learning_rate * mh / (jnp.sqrt(vh) + eps_adam),
            params, m_hat, v_hat,
        )
        return new_params, {"m": m, "v": v, "t": t}

    policy_loss_fn = lambda p: pure_ppo_clip_loss(
        p, policy_fn, wstates, action_samples, start_log_probs, gaes, epsilon, action_size,
    )
    policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_store)
    policy_grad_norm = pure_gradient_norm(policy_grads)
    policy_store_next, policy_opt_next = adam_step(policy_store, policy_grads, policy_opt)

    value_loss_fn = lambda v: pure_value_loss(v, value_fn, wstates, rtgs)
    value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_store)
    value_grad_norm = pure_gradient_norm(value_grads)
    value_store_next, value_opt_next = adam_step(value_store, value_grads, value_opt)

    return (
        policy_store_next,
        value_store_next,
        policy_opt_next,
        value_opt_next,
        policy_loss,
        value_loss,
        policy_grad_norm,
        value_grad_norm,
    )
