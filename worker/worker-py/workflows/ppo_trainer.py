from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from simulator.simulator import MjxSimulator, MujocoSimulator, Simulator
from models.model import PPOTrainedModel


Array = jax.Array
Params = Any

# jax.config.update("jax_log_compiles", True)


def _tree_stack(items: Sequence[Any]) -> Any:
    """Stack a list of pytrees (e.g. per-step parts dicts) into one pytree with a
    leading axis, i.e. jnp.stack applied leaf-wise."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *items)

@dataclass
class Trajectory:
    # labeled parts dicts with a leading timestep axis (jnp.stack over time).
    # control_wstates leaves: [traj_depth, ...]      — policy re-evaluation (control spec)
    # value_wstates leaves:   [traj_depth + 1, ...]  — value estimation incl. terminal (value spec)
    control_wstates: dict[str, Array]
    value_wstates: dict[str, Array]
    policy_outputs: dict[str,Array]
    policy_output_samples: list[Array]
    policy_output_sample_log_probs: list[float]
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


class PPOTrainer:
    def __init__(
        self,
        model: PPOTrainedModel,
        num_trajs: int,
        traj_depth: int,
        num_updates: int,
        epsilon: float,
        learning_rate: float = 1e-3,
        gamma_lambda: tuple[float, float] = (0.99, 0.95),
        rng_key: Array | None = None,
        seed: int = 0,
    ) -> None:
        self.num_trajs = num_trajs
        self.traj_depth = traj_depth
        self.num_updates = num_updates
        self.epsilon = epsilon
        self.gamma_lambda = gamma_lambda
        self.model = model
        self.learning_rate = learning_rate
        self._rng_key = rng_key if rng_key is not None else random.PRNGKey(seed)
        self.policy_opt = self._init_adam_state(self.model.get_policy_store())
        self.value_opt = self._init_adam_state(self.model.get_value_store())
        self._jit_gen_trajs: Any = None
        self._jit_train_epoch: Any = None

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

    def gen_trajectory_step(
        self,
        data: Any,
        rng_key: Array,
        simulator_state: dict[str, Any],
    ) -> tuple[
        Any,
        Array,
        dict[str, Array],
        dict[str, Array],
        dict[str, Array],
        Array,
        Array,
        float,
    ]:
        """Generate one CPU trajectory step and return its updated state."""
        simulator_class = simulator_state["simulator_class"]
        if simulator_class is not MujocoSimulator:
            raise ValueError("gen_trajectory_step requires MujocoSimulator state")
        model = simulator_state["mujoco_model"]
        if model is None:
            raise ValueError("MuJoCo data and model are required for CPU trajectories")
        render_ctx = simulator_state["render_ctx"]
        contact_sensor_addresses = simulator_state["contact_sensor_addresses"]
        contact_body_pairs = simulator_state["contact_body_pairs"]

        control_input_spec = self.model.get_control_input_spec()
        value_input_spec = self.model.get_value_input_spec()
        reward_input_spec = self.model.get_reward_input_spec()
        policy_fn = self.model.get_policy_fn()
        policy_store = self.model.get_policy_store()
        output_mode = self.model.get_control_policy_output_mode()

        control_parts = simulator_class.get_model_state_pure(
            control_input_spec,
            data,
            model,
            contact_sensor_addresses,
            contact_body_pairs,
        )
        value_parts = simulator_class.get_model_state_pure(
            value_input_spec,
            data,
            model,
            contact_sensor_addresses,
            contact_body_pairs,
        )
        reward_parts_pre = simulator_class.get_model_state_pure(
            reward_input_spec,
            data,
            model,
            contact_sensor_addresses,
            contact_body_pairs,
        )
        if model.ncam > 0 and any(
            spec.all_cameras
            for spec in (control_input_spec, value_input_spec, reward_input_spec)
        ):
            cameras, data = simulator_class.get_model_render(data, model, render_ctx)
            if control_input_spec.all_cameras:
                control_parts["cameras"] = cameras
            if value_input_spec.all_cameras:
                value_parts["cameras"] = cameras
            if reward_input_spec.all_cameras:
                reward_parts_pre["cameras"] = cameras
        rng_key, step_key = jax.random.split(rng_key)
        pol_output = policy_fn(policy_store, {**control_parts, "rng_key": step_key})
        sampled_action = pol_output["sampled_action"]
        sampled_action_log_prob = pol_output["sampled_action_log_prob"]
        # Chunked policies such as Pi0 return [action_steps, action_dim];
        # this workload currently executes only the first action per step.
        action = sampled_action[0] if sampled_action.ndim == 2 else sampled_action

        controls = simulator_class.get_actuator_input_from_control_input(
            model,
            data,
            np.asarray(action),
            output_mode,
        )
        data = simulator_class.forward_dyn_pure(data, model, controls)
        reward_parts_post = simulator_class.get_model_state_pure(
            reward_input_spec,
            data,
            model,
            contact_sensor_addresses,
            contact_body_pairs,
        )
        if reward_input_spec.all_cameras and model.ncam > 0:
            cameras, data = simulator_class.get_model_render(data, model, render_ctx)
            reward_parts_post["cameras"] = cameras
        reward = float(self.model.get_reward(reward_parts_pre, reward_parts_post))
        return (
            data,
            rng_key,
            control_parts,
            value_parts,
            pol_output,
            action,
            sampled_action_log_prob,
            reward,
        )

    def gen_trajectories(
        self,
        num_trajs: int,
        simulator_state: dict[str, Any],
    ) -> list[Trajectory]:
        simulator_class = simulator_state["simulator_class"]
        if simulator_class is not MujocoSimulator:
            raise ValueError("gen_trajectories requires MujocoSimulator state")
        initial_data = simulator_state["mujoco_data"]
        model = simulator_state["mujoco_model"]
        if initial_data is None or model is None:
            raise ValueError("MuJoCo data and model are required for CPU trajectories")

        split_keys = jax.random.split(self._rng_key, num_trajs + 1)
        self._rng_key = split_keys[0]
        rng_states = list(split_keys[1:])
        data_states = [initial_data] * num_trajs
        control_wstates: list[list[dict[str, Array]]] = [[] for _ in range(num_trajs)]
        value_wstates: list[list[dict[str, Array]]] = [[] for _ in range(num_trajs)]
        policy_outputs: list[list[dict[str, Array]]] = [[] for _ in range(num_trajs)]
        action_samples: list[list[Array]] = [[] for _ in range(num_trajs)]
        action_log_probs: list[list[float]] = [[] for _ in range(num_trajs)]
        rewards: list[list[float]] = [[] for _ in range(num_trajs)]

        logger.info(
            "gen_trajectories: starting time-major rollout "
            "(num_trajs=%s traj_depth=%s)",
            num_trajs,
            self.traj_depth,
        )
        for step in range(self.traj_depth):
            logger.info("gen_trajectories: step %s/%s", step, self.traj_depth)
            for traj_index in range(num_trajs):
                (
                    data_states[traj_index],
                    rng_states[traj_index],
                    control_parts,
                    value_parts,
                    pol_output,
                    action,
                    sampled_action_log_prob,
                    reward,
                ) = self.gen_trajectory_step(
                    data_states[traj_index],
                    rng_states[traj_index],
                    simulator_state,
                )
                control_wstates[traj_index].append(control_parts)
                value_wstates[traj_index].append(value_parts)
                policy_outputs[traj_index].append(pol_output)
                action_samples[traj_index].append(action)
                action_log_probs[traj_index].append(sampled_action_log_prob)
                rewards[traj_index].append(reward)

        value_input_spec = self.model.get_value_input_spec()
        contact_sensor_addresses = simulator_state["contact_sensor_addresses"]
        contact_body_pairs = simulator_state["contact_body_pairs"]
        trajectories: list[Trajectory] = []
        for traj_index in range(num_trajs):
            terminal_value_parts = simulator_class.get_model_state_pure(
                value_input_spec,
                data_states[traj_index],
                model,
                contact_sensor_addresses,
                contact_body_pairs,
            )
            if value_input_spec.all_cameras and model.ncam > 0:
                cameras, data_states[traj_index] = simulator_class.get_model_render(
                    data_states[traj_index], model, simulator_state["render_ctx"]
                )
                terminal_value_parts["cameras"] = cameras
            value_wstates[traj_index].append(terminal_value_parts)
            v_terminal = float(self.model.get_value_output(terminal_value_parts))
            trajectories.append(
                Trajectory(
                    control_wstates=_tree_stack(control_wstates[traj_index]),
                    value_wstates=_tree_stack(value_wstates[traj_index]),
                    policy_outputs=_tree_stack(policy_outputs[traj_index]),
                    policy_output_samples=action_samples[traj_index],
                    policy_output_sample_log_probs=action_log_probs[traj_index],
                    rewards=rewards[traj_index],
                    rtgs=self.compute_rtg(rewards[traj_index], v_terminal),
                )
            )
        logger.info("gen_trajectories: done")
        return trajectories

    def compute_gae(self, traj: Trajectory) -> Array:
        # Single batched value call over the [T+1, ...] value-state dict (no
        # per-step loop); shares the pure GAE implementation with the GPU path.
        rewards = jnp.asarray(traj.rewards, dtype=jnp.float32)
        return pure_compute_gae(
            traj.value_wstates,
            rewards,
            self.model.get_value_fn(),
            self.model.get_value_store(),
            self.gamma_lambda[0],
            self.gamma_lambda[1],
        )

    def compute_rtg(self, rewards: list[float], v_terminal: float) -> list[float]:
        rtgs = [0.0 for _ in range(self.traj_depth)]
        running_rtg = v_terminal
        for i in reversed(range(self.traj_depth)):
            running_rtg = rewards[i] + (self.gamma_lambda[0] * running_rtg)
            rtgs[i] = float(running_rtg)
        return rtgs

    def get_ppo_clip_objective(
        self,
        trajs: Sequence[Trajectory],
        gaes: Array,
        policy_store: Params | None = None,
    ) -> Array:
        start_log_probs = jnp.stack(
            [jnp.asarray(traj.policy_output_sample_log_probs) for traj in trajs]
        )
        # Each traj.control_wstates is already a [T, ...] dict — stack over trajs → [num_trajs, T, ...].
        wstates = _tree_stack([traj.control_wstates for traj in trajs])
        policy_outputs = _tree_stack([traj.policy_outputs for traj in trajs])

        if policy_store is None:
            policy_store = self.model.get_policy_store()
        logger.debug("wstates shape in get_ppo_clip_objective {wstates.shape}")
        ## Use the PURE log-prob fn with the passed policy_store so value_and_grad
        ## differentiates w.r.t. it (the bound get_policy_output_log_prob uses the
        ## model's stored params → zero gradient). Batch-friendly across mlp & vla.
        log_prob_fn = self.model.get_policy_output_log_prob_fn()
        action_log_probs = log_prob_fn(policy_store, wstates, policy_outputs)
        ratio = jnp.exp(action_log_probs - start_log_probs)
        clipped = jnp.clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * gaes
        return jnp.mean(jnp.minimum(ratio * gaes, clipped))

    def get_value_function_objective(
        self,
        trajs: Sequence[Trajectory],
        value_store: Params | None = None,
    ) -> Array:
        rtgs = jnp.stack([jnp.asarray(traj.rtgs) for traj in trajs])
        # value_wstates is a [T+1, ...] dict; strip the terminal (axis 0) then stack over trajs.
        wstates = _tree_stack(
            [jax.tree_util.tree_map(lambda x: x[: self.traj_depth], traj.value_wstates) for traj in trajs]
        )
        if value_store is None:
            value_store = self.model.get_value_store()
        values = self.model.get_value_fn()(value_store, wstates)
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
        # traj.policy_outputs is a time-leading dict; use the per-dim log-std.
        # Diagonal-Gaussian differential entropy = sum_i (log_std_i + 0.5*(1+log 2pi)).
        log_stds = jnp.stack(
            [traj.policy_outputs["action_log_std_dev"] for traj in trajs]
        )  # [num_trajs, T, nu]
        log_two_pi = jnp.log(2.0 * jnp.pi)
        entropy = jnp.mean(
            jnp.sum(log_stds + 0.5 * (1.0 + log_two_pi), axis=-1)
        )
        return float(entropy)

    def get_rollout_mean_reward(self, trajs: Sequence[Trajectory]) -> float:
        reward_count = sum(len(traj.rewards) for traj in trajs)
        if reward_count == 0:
            return 0.0
        reward_sum = sum(sum(traj.rewards) for traj in trajs)
        return reward_sum / reward_count

    def gen_trajectory_step_gpu(
        self,
        mjx_data: Any,
        mjx_model: Any,
        rng_key: Array,
        simulator_class: type[Simulator],
        control_state_fn: Callable[[Any, Any], dict[str, Array]],
        value_state_fn: Callable[[Any, Any], dict[str, Array]],
        reward_state_fn: Callable[[Any, Any], dict[str, Array]],
        control_has_cameras: bool,
        value_has_cameras: bool,
        reward_has_cameras: bool,
        camera_frames: Array | None = None,
        policy_store: Params | None = None,
    ) -> tuple[
        tuple[Any, Array],
        tuple[
            dict[str, Array],
            dict[str, Array],
            dict[str, Array],
            Array,
            dict[str, Array],
        ],
    ]:
        """Generate one JAX-traceable GPU trajectory step."""
        policy_fn = self.model.get_policy_fn()
        if policy_store is None:
            policy_store = self.model.get_policy_store()
        output_mode = self.model.get_control_policy_output_mode()

        rng_key, step_key = jax.random.split(rng_key)
        control_parts = control_state_fn(mjx_data, mjx_model)
        value_parts = value_state_fn(mjx_data, mjx_model)
        reward_parts_pre = reward_state_fn(mjx_data, mjx_model)
        if camera_frames is not None:
            if control_has_cameras:
                control_parts["cameras"] = camera_frames
            if value_has_cameras:
                value_parts["cameras"] = camera_frames
            if reward_has_cameras:
                reward_parts_pre["cameras"] = camera_frames

        # Thread a fresh per-step key in via the dict; control_parts itself is
        # stored without it (log-prob recompute is deterministic).
        policy_output = policy_fn(
            policy_store,
            {**control_parts, "rng_key": step_key},
        )
        sampled_action = policy_output["sampled_action"]
        # Chunked policies such as Pi0 return [action_steps, action_dim];
        # this workload currently executes only the first action per step.
        action = sampled_action[0] if sampled_action.ndim == 2 else sampled_action

        controls = simulator_class.get_actuator_input_from_control_input(
            mjx_model,
            mjx_data,
            action,
            output_mode,
        )
        mjx_data = simulator_class.forward_dyn_pure(
            mjx_data,
            mjx_model,
            controls,
        )
        return (mjx_data, rng_key), (
            control_parts,
            value_parts,
            policy_output,
            action,
            reward_parts_pre,
        )

    def gen_trajectories_gpu(
        self,
        num_trajs: int,
        simulator_state: dict[str, Any],
        rng_key: Array,
    ) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array, Array]:
        """Generate trajectories with a scan over vmapped GPU steps.

        Args:
            num_trajs: number of parallel trajectories
            simulator_state: caller-owned simulator structures and render context
            rng_key:   base RNG key; split into num_trajs sub-keys

        Returns:
            (control_wstates, value_wstates, policy_outputs, action_samples, rewards)
            control_wstates: control parts dict, leaves [num_trajs, traj_depth, ...]
            value_wstates:   value parts dict, leaves [num_trajs, traj_depth+1, ...]  — incl. terminal
            policy_outputs:   [num_trajs, traj_depth, nu*2]
            action_samples:  [num_trajs, traj_depth, nu]
            rewards:         [num_trajs, traj_depth]
        """
        logger.info(
            "gen_trajectories_gpu: start (num_trajs=%s traj_depth=%s)",
            num_trajs,
            self.traj_depth,
        )
        keys = jax.random.split(rng_key, num_trajs)
        policy_store = self.model.get_policy_store()
        logger.info(
            "gen_trajectories_gpu: split %s RNG keys, launching jit(scan(vmap))",
            num_trajs,
        )

        simulator_class = simulator_state["simulator_class"]
        if simulator_class is not MjxSimulator:
            raise ValueError("gen_trajectories_gpu requires MjxSimulator state")
        mjx_data = simulator_state["mjx_data"]
        mjx_model = simulator_state["mjx_model"]
        if mjx_data is None or mjx_model is None:
            raise ValueError("MJX data and model are required for GPU trajectories")
        batched_data = mjx_data

        render_ctx = simulator_state["render_ctx_pytree"]
        contact_sensor_addresses = simulator_state["contact_sensor_addresses"]
        contact_body_pairs = simulator_state["contact_body_pairs"]
        control_input_spec = self.model.get_control_input_spec()
        value_input_spec = self.model.get_value_input_spec()
        reward_input_spec = self.model.get_reward_input_spec()
        reward_fn = self.model.get_pure_reward_fn()
        has_cameras = int(mjx_model.ncam) > 0
        control_has_cameras = has_cameras and control_input_spec.all_cameras
        value_has_cameras = has_cameras and value_input_spec.all_cameras
        reward_has_cameras = has_cameras and reward_input_spec.all_cameras
        needs_step_render = (
            control_has_cameras or value_has_cameras or reward_has_cameras
        )

        def control_state_fn(
            data: Any,
            model: Any,
        ) -> dict[str, Array]:
            return simulator_class.get_model_state_pure(
                control_input_spec,
                data,
                model,
                contact_sensor_addresses,
                contact_body_pairs,
            )

        def value_state_fn(
            data: Any,
            model: Any,
        ) -> dict[str, Array]:
            return simulator_class.get_model_state_pure(
                value_input_spec,
                data,
                model,
                contact_sensor_addresses,
                contact_body_pairs,
            )

        def reward_state_fn(
            data: Any,
            model: Any,
        ) -> dict[str, Array]:
            return simulator_class.get_model_state_pure(
                reward_input_spec,
                data,
                model,
                contact_sensor_addresses,
                contact_body_pairs,
            )

        def all_trajectories(
            ps: Params,
            model: Any,
            data: Any,
            trajectory_keys: Array,
            ctx: Any,
        ) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array, Array]:
            def trajectory_step(
                step_data: Any,
                step_key: Array,
                camera_frames: Array | None,
            ) -> tuple[
                tuple[Any, Array],
                tuple[
                    dict[str, Array],
                    dict[str, Array],
                    dict[str, Array],
                    Array,
                    dict[str, Array],
                ],
            ]:
                return self.gen_trajectory_step_gpu(
                    step_data,
                    model,
                    step_key,
                    simulator_class,
                    control_state_fn,
                    value_state_fn,
                    reward_state_fn,
                    control_has_cameras,
                    value_has_cameras,
                    reward_has_cameras,
                    camera_frames,
                    ps,
                )

            def scan_body(
                carry: tuple[Any, Array],
                _: None,
            ) -> tuple[
                tuple[Any, Array],
                tuple[
                    dict[str, Array],
                    dict[str, Array],
                    dict[str, Array],
                    Array,
                    Array,
                ],
            ]:
                step_data, step_keys = carry
                camera_frames = None
                if needs_step_render:
                    camera_frames, step_data = simulator_class.get_model_render(
                        step_data, model, ctx
                    )
                (next_data, next_keys), (
                    control_parts,
                    value_parts,
                    policy_output,
                    action,
                    reward_parts_pre,
                ) = jax.vmap(
                    trajectory_step,
                    in_axes=(0, 0, 0 if needs_step_render else None),
                )(step_data, step_keys, camera_frames)
                reward_parts_post = jax.vmap(
                    reward_state_fn, in_axes=(0, None)
                )(next_data, model)
                if reward_has_cameras:
                    post_frames, next_data = simulator_class.get_model_render(
                        next_data, model, ctx
                    )
                    reward_parts_post["cameras"] = post_frames
                rewards = jax.vmap(reward_fn)(reward_parts_pre, reward_parts_post)
                return (next_data, next_keys), (
                    control_parts,
                    value_parts,
                    policy_output,
                    action,
                    rewards,
                )

            (data_final, _), step_outputs = jax.lax.scan(
                scan_body,
                (data, trajectory_keys),
                None,
                length=self.traj_depth,
            )
            (
                control_states_t,
                value_states_t,
                policy_outputs_t,
                action_samples_t,
                rewards_t,
            ) = step_outputs

            terminal_value_parts = jax.vmap(
                value_state_fn,
                in_axes=(0, None),
            )(data_final, model)
            if value_has_cameras:
                terminal_frames, data_final = simulator_class.get_model_render(
                    data_final, model, ctx
                )
                terminal_value_parts["cameras"] = terminal_frames
            value_states_t = jax.tree_util.tree_map(
                lambda seq, term: jnp.concatenate([seq, term[None]], axis=0),
                value_states_t,
                terminal_value_parts,
            )

            # lax.scan stacks time first. The training API remains environment
            # first: [num_trajs, traj_depth, ...].
            to_environment_major = lambda tree: jax.tree_util.tree_map(
                lambda value: jnp.swapaxes(value, 0, 1),
                tree,
            )
            return (
                to_environment_major(control_states_t),
                to_environment_major(value_states_t),
                to_environment_major(policy_outputs_t),
                jnp.swapaxes(action_samples_t, 0, 1),
                jnp.swapaxes(rewards_t, 0, 1),
            )

        if self._jit_gen_trajs is None:
            ##DEBUG TRACING & COMPILE:
            # See what JAX will actually compile before running it
            # jaxpr = jax.make_jaxpr(all_trajectories)(
            #     policy_store, mjx_model, batched_data, keys, render_ctx
            # )
            # print(jaxpr)  # full trace as a Jaxpr (JAX's IR before HLO lowering)

            # Go one step further: lower to StableHLO
            t0 = time.time()
            lowered = jax.jit(all_trajectories).lower(
                policy_store,
                mjx_model,
                batched_data,
                keys,
                render_ctx,
            )
            t1 = time.time()
            logger.info("gen_trajectories_gpu: tracing+lowering took %.2fs", t1 - t0)
            # print(lowered.as_text())  # StableHLO text representation

            # Then compile and inspect the compiled artifact
            compiled = lowered.compile()
            t2 = time.time()
            logger.info("gen_trajectories_gpu: XLA compilation took %.2fs", t2 - t1)
            print(compiled.cost_analysis())  # FLOPs, memory estimates per op
            ##DEBUG TRACING & COMPILE complete:

            self._jit_gen_trajs = compiled

        t_exec = time.time()
        result = self._jit_gen_trajs(
            policy_store,
            mjx_model,
            batched_data,
            keys,
            render_ctx,
        )
        jax.block_until_ready(result)
        t_exec_done = time.time()
        logger.info("gen_trajectories_gpu: execution took %.2fs", t_exec_done - t_exec)
        return result

    def train_an_epoch_gpu(
        self,
        control_wstates: dict[str, Array],
        value_wstates: dict[str, Array],
        policy_outputs: dict[str,Array],
        action_samples: Array,
        rewards: Array
    ) -> PPOTrainerRuntimeInfo:
        """GPU training epoch using pure learning_step.

        Args:
            control_wstates: control parts dict, leaves [num_trajs, traj_depth, ...]
            value_wstates:   value parts dict, leaves [num_trajs, traj_depth+1, ...]  — incl. terminal
            policy_outputs: model-produced dictionary whose array leaves have
                leading dimensions [num_trajs, traj_depth, ...]
            action_samples:  [num_trajs, traj_depth, nu]
            rewards:         [num_trajs, traj_depth]

        Returns:
            PPOTrainerRuntimeInfo with per-update losses and gradient norms
        """
        num_trajs_in = rewards.shape[0]
        logger.info(
            "train_an_epoch_gpu: start (num_trajs=%s traj_depth=%s action_size=%s)",
            num_trajs_in,
            self.traj_depth,
            action_samples.shape[-1],
        )
        policy_log_prob_fn = self.model.get_policy_output_log_prob_fn()
        value_fn = self.model.get_value_fn()
        gamma, lam = self.gamma_lambda

        logger.info("train_an_epoch_gpu: computing RTGs for %s trajs", num_trajs_in)

        def compute_rtg_for_traj(traj_rewards: Array, traj_value_wstates: Any) -> Array:
            terminal = jax.tree_util.tree_map(lambda x: x[-1:], traj_value_wstates)
            v_terminal = jnp.squeeze(value_fn(self.model.get_value_store(), terminal), axis=-1)[0]
            return pure_compute_rtg(traj_rewards, gamma, v_terminal)

        rtgs = jax.vmap(compute_rtg_for_traj)(rewards, value_wstates)   # [num_trajs, traj_depth]
        logger.info("train_an_epoch_gpu: RTGs computed (shape=%s)", rtgs.shape)

        logger.info("train_an_epoch_gpu: computing GAEs for %s trajs", num_trajs_in)

        def compute_gae_for_traj(traj_value_wstates: Array, traj_rewards: Array) -> Array:
            return pure_compute_gae(traj_value_wstates, traj_rewards, value_fn, self.model.get_value_store(), gamma, lam)

        gaes_raw = jax.vmap(compute_gae_for_traj)(value_wstates, rewards)  # [num_trajs, traj_depth]

        gae_mean = jnp.mean(gaes_raw)
        gae_std = jnp.std(gaes_raw)
        gaes = (gaes_raw - gae_mean) / (gae_std + 1e-8)
        logger.info("train_an_epoch_gpu: GAE mean=%.4f std=%.4f (shape=%s)", float(gae_mean), float(gae_std), gaes.shape)

        logger.info("train_an_epoch_gpu: computing start log probs")
        start_log_probs = policy_outputs["sampled_action_log_prob"]
        logger.info("train_an_epoch_gpu: start log probs computed (shape=%s)", start_log_probs.shape)

        # control_wstates is already a dict of [num_trajs, traj_depth, ...] — no terminal to strip
        # value_wstates_t strips the terminal (axis 1) from each leaf for per-step value loss
        value_wstates_t = jax.tree_util.tree_map(lambda x: x[:, :self.traj_depth], value_wstates)

        epsilon = self.epsilon
        learning_rate = self.learning_rate
        num_updates = self.num_updates

        def update_all_steps_fn(
            carry: tuple[Params, Params, dict, dict],
            control_wstates_a: Array,
            value_wstates_a: Array,
            policy_output: dict[str,Array],
            start_log_probs_a: Array,
            gaes_a: Array,
            rtgs_a: Array,
        ) -> tuple[tuple[Params, Params, dict, dict], tuple[Array, Array, Array, Array]]:
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
                    policy_log_prob_fn,
                    value_fn,
                    control_wstates_a,
                    value_wstates_a,
                    policy_output,
                    start_log_probs_a,
                    gaes_a,
                    rtgs_a,
                    epsilon,
                    learning_rate,
                )
                return (policy_store_next, value_store_next, policy_opt_next, value_opt_next), (
                    p_loss, v_loss, p_grad_norm, v_grad_norm,
                )
            return jax.lax.scan(update_step, carry, None, length=num_updates)

        logger.info(
            "train_an_epoch_gpu: launching jit(lax.scan) over %s update steps", self.num_updates
        )
        init_carry = (
            self.model.get_policy_store(),
            self.model.get_value_store(),
            self.policy_opt,
            self.value_opt,
        )

        if self._jit_train_epoch is None:
            ##DEBUG TRACING & COMPILE:
            lowered = jax.jit(update_all_steps_fn).lower(
                init_carry, control_wstates, value_wstates_t, policy_outputs, start_log_probs, gaes, rtgs
            )
            compiled = lowered.compile()
            print(compiled.cost_analysis())  # FLOPs, memory estimates per op
            ##DEBUG TRACING & COMPILE complete:
            self._jit_train_epoch = compiled

        (policy_store_final, value_store_final, policy_opt_final, value_opt_final), (
            p_losses, v_losses, p_grad_norms, v_grad_norms,
        ) = self._jit_train_epoch(init_carry, control_wstates, value_wstates_t, policy_outputs, start_log_probs, gaes, rtgs)

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
#   policy_outputs:  [num_trajs, traj_depth, nu*2]
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
    wstates: dict[str, Array],
    rewards: Array,
    value_fn: Callable[[Params, dict[str, Array]], Array],
    value_store: Params,
    gamma: float,
    lam: float,
) -> Array:
    """Compute generalised advantage estimates for a single trajectory without a Python loop.

    Args:
        wstates:     value-state parts dict, leaves [T+1, ...] (incl. terminal at index T)
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

def pure_ppo_clip_loss(
    policy_store: Params,
    policy_fn_log_prob: Callable[[Params, dict[str, Array], dict[str, Array]], Array],
    wstates: dict[str, Array],
    policy_output: dict[str,Array],
    start_log_probs: Array,
    gaes: Array,
    epsilon: float,
) -> Array:
    """PPO clipped surrogate loss (negated for gradient ascent via minimisation).

    Args:
        policy_store:       policy network parameters
        policy_fn_log_prob: pure fn (params, wstates, policy_output) -> per-step
                            joint log-probs [num_trajs, T] of the recorded actions
        wstates:            control parts dict, leaves [num_trajs, T, ...]
        policy_output:      recorded policy stats dict (incl. "sampled_action")
        start_log_probs:    [num_trajs, T] log-probs at sampling time
        gaes:               [num_trajs, T] normalised advantages
        epsilon:            PPO clip ratio

    Returns:
        scalar loss (negated clip objective)
    """
    log_probs = policy_fn_log_prob(policy_store, wstates, policy_output)
    ratio = jnp.exp(log_probs - start_log_probs)
    clipped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * gaes
    return -jnp.mean(jnp.minimum(ratio * gaes, clipped))


def pure_value_loss(
    value_store: Params,
    value_fn: Callable[[Params, dict[str, Array]], Array],
    wstates: dict[str, Array],
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
    policy_fn_log_prob: Callable[[Params, dict[str, Array], dict[str, Array]], Array],
    value_fn: Callable[[Params, dict[str, Array]], Array],
    control_wstates: dict[str, Array],
    value_wstates: dict[str, Array],
    policy_output: dict[str,Array],
    start_log_probs: Array,
    gaes: Array,
    rtgs: Array,
    epsilon: float,
    learning_rate: float,
) -> tuple[Params, Params, dict, dict, Array, Array, Array, Array]:
    """Single pure parameter update step for policy and value networks.

    Computes gradients via reverse-mode autodiff and applies an Adam step.
    No Python-level side effects — all state is passed in and returned.

    Args:
        policy_store:    current policy parameters
        value_store:     current value parameters
        policy_opt:         Adam state for policy {"m", "v", "t"}
        value_opt:          Adam state for value  {"m", "v", "t"}
        policy_fn_log_prob: pure fn (params, control_wstates, policy_output) ->
                            per-step joint log-probs [num_trajs, T]
        value_fn:           callable (params, value_wstates) -> values
        control_wstates:    control parts dict, leaves [num_trajs, T, ...]  — policy loss
        value_wstates:      value parts dict, leaves [num_trajs, T, ...]    — value loss
        policy_output:      recorded policy stats dict (incl. "sampled_action")
        start_log_probs:    [num_trajs, T] log-probs at sampling time
        gaes:               [num_trajs, T] normalised advantages
        rtgs:               [num_trajs, T]
        epsilon:            PPO clip ratio
        learning_rate:      Adam learning rate

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
        p, policy_fn_log_prob, control_wstates, policy_output, start_log_probs, gaes, epsilon
    )
    policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_store)
    policy_grad_norm = pure_gradient_norm(policy_grads)
    policy_store_next, policy_opt_next = adam_step(policy_store, policy_grads, policy_opt)

    value_loss_fn = lambda v: pure_value_loss(v, value_fn, value_wstates, rtgs)
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
