from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from commondef.ioadapter import PolicyInputState
from commondef.records.workload_graph import (
    PolicyInputSpec,
    InputStateItemSensors,
    InputStateTypes,
)
from simulator.simulator import MjxSimulator, MujocoSimulator, Simulator
from simulator.state import SimulatorState
from models.model import PPOTrainedModel


Array = jax.Array
Params = Any


def _tree_stack(items: Sequence[Any]) -> Any:
    """Stack a list of pytrees (e.g. structured input states) into one pytree with a
    leading axis, i.e. jnp.stack applied leaf-wise."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *items)

@dataclass
class Trajectory:
    # Structured input states with a leading timestep axis on every array leaf.
    # control_wstates leaves: [traj_depth, ...]      — policy re-evaluation (control spec)
    # value_wstates leaves:   [traj_depth + 1, ...]  — value estimation incl. terminal (value spec)
    control_wstates: PolicyInputState
    value_wstates: PolicyInputState
    policy_outputs: dict[str,Array]
    # Frozen rollout predictions, including the bootstrap state: [T+1].
    value_outputs: Array
    policy_output_sample_log_probs: Array
    rewards: Array
    rtgs: Array


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
        num_epochs: int,
        epsilon: float,
        learning_rate: float = 1e-3,
        gamma_lambda: tuple[float, float] = (0.99, 0.95),
        rng_key: Array | None = None,
        seed: int = 0,
    ) -> None:
        self.num_trajs = num_trajs
        self.traj_depth = traj_depth
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.gamma_lambda = gamma_lambda
        self.model = model
        # These layouts are static for the model throughout training.
        self._control_camera_indices = self._camera_state_indices(
            model.get_control_input_spec()
        )
        self._value_camera_indices = self._camera_state_indices(
            model.get_value_input_spec()
        )
        self._reward_camera_indices = self._camera_state_indices(
            model.get_reward_input_spec()
        )
        self.learning_rate = learning_rate
        self._rng_key = rng_key if rng_key is not None else random.PRNGKey(seed)
        policy = self.model.get_policy_store()
        self.policy_opt = (
            {name: self._init_adam_state(weights) for name, weights in policy.items()}
            if "pi0" in policy else self._init_adam_state(policy)
        )
        self.value_opt = self._init_adam_state(self.model.get_value_store())
        self._jit_gen_trajs: Any = None
        self._jit_learning_step: Any = None

    @staticmethod
    def _camera_state_indices(input_spec: PolicyInputSpec) -> tuple[int, ...]:
        return tuple(
            index
            for index, state in enumerate(input_spec.content.states)
            if state.type == InputStateTypes.INPUT_STATE_SENSORS
            and state.content == InputStateItemSensors.CAMERAS
        )

    @staticmethod
    def _with_camera_frames(
        state: PolicyInputState,
        camera_frames: Array | np.ndarray,
        camera_indices: tuple[int, ...],
    ) -> PolicyInputState:
        """Replace camera placeholders, preserving spec order and PyTree metadata."""
        if not camera_indices:
            return state
        states = list(state.content.states)
        for index in camera_indices:
            states[index] = replace(states[index], content=camera_frames)
        return replace(state, content=replace(state.content, states=states))

    @staticmethod
    def _init_adam_state(params: Params) -> dict[str, Any]:
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return {"m": zeros, "v": zeros, "t": 0}

    def gen_trajectory_step(
        self,
        data: Any,
        rng_key: Array,
        simulator_state: SimulatorState,
        record_cameras: bool = False,
    ) -> tuple[
        Any,
        Array,
        PolicyInputState,
        PolicyInputState,
        dict[str, Array],
        float,
        np.ndarray | None,
    ]:
        """Generate one CPU trajectory step and return its updated state."""
        simulator_class = simulator_state.simulator_class
        if simulator_class is not MujocoSimulator:
            raise ValueError("gen_trajectory_step requires MujocoSimulator state")
        model = simulator_state.mujoco_model
        if model is None:
            raise ValueError("MuJoCo data and model are required for CPU trajectories")
        render_ctx = simulator_state.render_ctx
        contact_sensor_addresses = simulator_state.contact_sensor_addresses
        contact_body_pairs = simulator_state.contact_body_pairs

        control_input_spec = self.model.get_control_input_spec()
        value_input_spec = self.model.get_value_input_spec()
        reward_input_spec = self.model.get_reward_input_spec()
        policy_fn = self.model.get_policy_fn()
        policy_store = self.model.get_policy_store()
        output_spec = self.model.get_control_policy_output_spec()

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
        if (
            self._control_camera_indices
            or self._value_camera_indices
            or self._reward_camera_indices
        ):
            if render_ctx is None:
                raise ValueError("Camera input specs require a render context")
            cameras, data = simulator_class.get_model_render(data, model, render_ctx)
            control_parts = self._with_camera_frames(
                control_parts, cameras, self._control_camera_indices
            )
            value_parts = self._with_camera_frames(
                value_parts, cameras, self._value_camera_indices
            )
            reward_parts_pre = self._with_camera_frames(
                reward_parts_pre, cameras, self._reward_camera_indices
            )
        rng_key, step_key = jax.random.split(rng_key)
        pol_output = policy_fn(policy_store, control_parts, step_key)
        sampled_action = pol_output["sampled_action"]
        # Chunked policies such as Pi0 return [action_steps, action_dim];
        # this workload currently executes only the first action per step.
        action = sampled_action[0] if sampled_action.ndim == 2 else sampled_action

        controls, valid = simulator_class.get_actuator_input_from_control_input(
            model,
            data,
            np.asarray(action),
            output_spec,
        )
        if not bool(valid):
            raise ValueError("Invalid actuator control mapping in CPU trajectory step")
        data = simulator_class.forward_dyn_pure(data, model, controls, refresh_state=True)
        reward_parts_post = simulator_class.get_model_state_pure(
            reward_input_spec,
            data,
            model,
            contact_sensor_addresses,
            contact_body_pairs,
        )
        cameras = None
        if self._reward_camera_indices or record_cameras:
            cameras, data = simulator_class.get_model_render(data, model, render_ctx)
            reward_parts_post = self._with_camera_frames(
                reward_parts_post, cameras, self._reward_camera_indices
            )
        reward = float(self.model.get_reward(reward_parts_pre, reward_parts_post))
        if not np.isfinite(reward):
            raise ValueError("Training reward must be finite")
        return (
            data,
            rng_key,
            control_parts,
            value_parts,
            pol_output,
            reward,
            cameras,
        )

    def gen_trajectories(
        self,
        num_trajs: int,
        simulator_state: SimulatorState,
        record_batch: Callable[[int, np.ndarray, np.ndarray, np.ndarray | None], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[Trajectory]:
        simulator_class = simulator_state.simulator_class
        if simulator_class is not MujocoSimulator:
            raise ValueError("gen_trajectories requires MujocoSimulator state")
        initial_data = simulator_state.mujoco_data
        model = simulator_state.mujoco_model
        if initial_data is None or model is None:
            raise ValueError("MuJoCo data and model are required for CPU trajectories")

        split_keys = jax.random.split(self._rng_key, num_trajs + 1)
        self._rng_key = split_keys[0]
        rng_states = list(split_keys[1:])
        data_states = [initial_data] * num_trajs
        control_wstates: list[list[PolicyInputState]] = [[] for _ in range(num_trajs)]
        value_wstates: list[list[PolicyInputState]] = [[] for _ in range(num_trajs)]
        policy_outputs: list[list[dict[str, Array]]] = [[] for _ in range(num_trajs)]
        action_log_probs: list[list[float]] = [[] for _ in range(num_trajs)]
        rewards: list[list[float]] = [[] for _ in range(num_trajs)]

        logger.info(
            "gen_trajectories: starting time-major rollout "
            "(num_trajs=%s traj_depth=%s)",
            num_trajs,
            self.traj_depth,
        )
        for step in range(self.traj_depth):
            if should_stop is not None and should_stop():
                return []
            positions, quaternions, frames = [], [], []
            for traj_index in range(num_trajs):
                if should_stop is not None and should_stop():
                    return []
                (
                    data_states[traj_index],
                    rng_states[traj_index],
                    control_parts,
                    value_parts,
                    pol_output,
                    reward,
                    cameras,
                ) = self.gen_trajectory_step(
                    data_states[traj_index],
                    rng_states[traj_index],
                    simulator_state,
                    record_cameras=record_batch is not None and int(model.ncam) > 0,
                )
                control_wstates[traj_index].append(control_parts)
                value_wstates[traj_index].append(value_parts)
                policy_outputs[traj_index].append(pol_output)
                action_log_probs[traj_index].append(float(pol_output["sampled_action_log_prob"]))
                rewards[traj_index].append(reward)
                if record_batch is not None:
                    data = data_states[traj_index]
                    positions.append(np.asarray(data.xpos).copy())
                    quaternions.append(np.asarray(data.xquat).copy())
                    if cameras is not None:
                        frames.append(np.asarray(cameras))
            if record_batch is not None:
                record_batch(
                    step, np.stack(positions)[None], np.stack(quaternions)[None],
                    np.stack(frames)[None] if frames else None,
                )

        value_input_spec = self.model.get_value_input_spec()
        contact_sensor_addresses = simulator_state.contact_sensor_addresses
        contact_body_pairs = simulator_state.contact_body_pairs
        trajectories: list[Trajectory] = []
        for traj_index in range(num_trajs):
            terminal_value_parts = simulator_class.get_model_state_pure(
                value_input_spec,
                data_states[traj_index],
                model,
                contact_sensor_addresses,
                contact_body_pairs,
            )
            if self._value_camera_indices:
                cameras, data_states[traj_index] = simulator_class.get_model_render(
                    data_states[traj_index], model, simulator_state.render_ctx
                )
                terminal_value_parts = self._with_camera_frames(
                    terminal_value_parts, cameras, self._value_camera_indices
                )
            value_wstates[traj_index].append(terminal_value_parts)
            states = _tree_stack(value_wstates[traj_index])
            values = jax.lax.stop_gradient(jnp.asarray(
                self.model.get_value_fn()(self.model.get_value_store(), states)
            ).reshape(self.traj_depth + 1))
            trajectory_rewards = jnp.asarray(rewards[traj_index], dtype=jnp.float32)
            trajectories.append(
                Trajectory(
                    control_wstates=_tree_stack(control_wstates[traj_index]),
                    value_wstates=states,
                    value_outputs=values,
                    policy_outputs=_tree_stack(policy_outputs[traj_index]),
                    policy_output_sample_log_probs=jnp.asarray(action_log_probs[traj_index]),
                    rewards=trajectory_rewards,
                    rtgs=pure_compute_rtg(trajectory_rewards, self.gamma_lambda[0], values[-1]),
                )
            )
        logger.info("gen_trajectories: done")
        return trajectories

    def train_rollout(
        self,
        trajs: Sequence[Trajectory],
        post_epoch_callback: Callable[[int, tuple[float, ...]], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> PPOTrainerRuntimeInfo:
        """One full-batch policy/value update per epoch, reusing frozen rollout data."""
        if len(trajs) != self.num_trajs:
            raise ValueError("Training requires a complete rollout")
        control_states = _tree_stack([traj.control_wstates for traj in trajs])
        value_states = _tree_stack([
            jax.tree_util.tree_map(lambda x: x[:-1], traj.value_wstates)
            for traj in trajs
        ])
        policy_outputs = _tree_stack([traj.policy_outputs for traj in trajs])
        log_probs = jax.lax.stop_gradient(jnp.stack([
            traj.policy_output_sample_log_probs for traj in trajs
        ]))
        rtgs = jax.lax.stop_gradient(jnp.stack([traj.rtgs for traj in trajs]))
        gaes = jnp.stack([
            pure_compute_gae(traj.value_outputs, traj.rewards, *self.gamma_lambda)
            for traj in trajs
        ])
        gaes = jax.lax.stop_gradient((gaes - jnp.mean(gaes)) / (jnp.std(gaes) + 1e-8))
        if self._jit_learning_step is None:
            policy_log_prob_fn = self.model.get_policy_output_log_prob_fn()
            value_fn = self.model.get_value_fn()

            def update(policy, value, policy_opt, value_opt, controls, values, outputs, old_log_probs, advantages, targets):
                return learning_step(
                    policy, value, policy_opt, value_opt, policy_log_prob_fn, value_fn,
                    controls, values, outputs, old_log_probs, advantages, targets,
                    self.epsilon, self.learning_rate,
                )
            self._jit_learning_step = jax.jit(update)

        statistics: list[list[float]] = [[], [], [], []]
        for epoch in range(self.num_epochs):
            if should_stop is not None and should_stop():
                break
            result = self._jit_learning_step(
                self.model.get_policy_store(), self.model.get_value_store(),
                self.policy_opt, self.value_opt, control_states, value_states,
                policy_outputs, log_probs, gaes, rtgs,
            )
            jax.block_until_ready(result)
            policy, value, policy_opt, value_opt, *raw_stats = result
            stats = tuple(float(x) for x in raw_stats)
            # Do not publish/save an update whose numerical result is invalid.
            if not np.isfinite(stats).all():
                raise ValueError(f"Non-finite PPO diagnostics at epoch {epoch}")
            self.model.get_policy_store_mut().update(policy)
            self.model.get_value_store_mut().update(value)
            self.policy_opt, self.value_opt = policy_opt, value_opt
            for history, scalar in zip(statistics, stats):
                history.append(scalar)
            if post_epoch_callback is not None:
                post_epoch_callback(epoch, stats)
        return PPOTrainerRuntimeInfo(*statistics)

    def get_rollout_mean_entropy(self, trajs: Sequence[Trajectory]) -> float | None:
        # VLA denoising-path entropy is deliberately not reported as action entropy.
        if not trajs or "policy_entropy" not in trajs[0].policy_outputs:
            return None
        values = [jnp.ravel(traj.policy_outputs["policy_entropy"]) for traj in trajs]
        return float(jnp.mean(jnp.concatenate(values)))

    def get_rollout_mean_reward(self, trajs: Sequence[Trajectory]) -> float:
        if not trajs:
            return 0.0
        return float(jnp.mean(jnp.stack([jnp.sum(traj.rewards) for traj in trajs])))

    def gen_trajectory_step_gpu(
        self,
        mjx_data: Any,
        mjx_model: Any,
        rng_key: Array,
        simulator_class: type[Simulator],
        control_state_fn: Callable[[Any, Any], PolicyInputState],
        value_state_fn: Callable[[Any, Any], PolicyInputState],
        reward_state_fn: Callable[[Any, Any], PolicyInputState],
        control_has_cameras: bool,
        value_has_cameras: bool,
        reward_has_cameras: bool,
        camera_frames: Array | None = None,
        policy_store: Params | None = None,
    ) -> tuple[
        tuple[Any, Array],
        tuple[
            PolicyInputState,
            PolicyInputState,
            dict[str, Array],
            Array,
            PolicyInputState,
            Array,
        ],
    ]:
        """Generate one JAX-traceable GPU trajectory step."""
        policy_fn = self.model.get_policy_fn()
        if policy_store is None:
            policy_store = self.model.get_policy_store()
        output_spec = self.model.get_control_policy_output_spec()

        rng_key, step_key = jax.random.split(rng_key)
        control_parts = control_state_fn(mjx_data, mjx_model)
        value_parts = value_state_fn(mjx_data, mjx_model)
        reward_parts_pre = reward_state_fn(mjx_data, mjx_model)
        if camera_frames is not None:
            if control_has_cameras:
                control_parts = self._with_camera_frames(
                    control_parts, camera_frames, self._control_camera_indices
                )
            if value_has_cameras:
                value_parts = self._with_camera_frames(
                    value_parts, camera_frames, self._value_camera_indices
                )
            if reward_has_cameras:
                reward_parts_pre = self._with_camera_frames(
                    reward_parts_pre, camera_frames, self._reward_camera_indices
                )

        policy_output = policy_fn(
            policy_store,
            control_parts,
            step_key,
        )
        sampled_action = policy_output["sampled_action"]
        # Chunked policies such as Pi0 return [action_steps, action_dim];
        # this workload currently executes only the first action per step.
        action = sampled_action[0] if sampled_action.ndim == 2 else sampled_action

        controls, valid = simulator_class.get_actuator_input_from_control_input(
            mjx_model,
            mjx_data,
            action,
            output_spec,
        )
        # The host checks validity after the compiled rollout. Never apply
        # invalid/NaN controls while that device computation is completing.
        controls = jnp.where(valid, controls, mjx_data.ctrl)
        mjx_data = simulator_class.forward_dyn_pure(
            mjx_data,
            mjx_model,
            controls,
            refresh_state=True,
        )
        return (mjx_data, rng_key), (
            control_parts,
            value_parts,
            policy_output,
            action,
            reward_parts_pre,
            valid,
        )

    def gen_trajectories_gpu(
        self,
        num_trajs: int,
        simulator_state: SimulatorState,
        rng_key: Array,
        record_batch: Callable[[int, np.ndarray, np.ndarray, np.ndarray | None], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[Trajectory]:
        """Collect MJX rollouts in bounded chunks; publish recordings on the host."""
        simulator_class = simulator_state.simulator_class
        if simulator_class is not MjxSimulator:
            raise ValueError("gen_trajectories_gpu requires MjxSimulator state")
        data, model = simulator_state.mjx_data, simulator_state.mjx_model
        if data is None or model is None:
            raise ValueError("MJX data and model are required for GPU trajectories")
        render_ctx = simulator_state.render_ctx_pytree
        record_cameras = record_batch is not None and int(simulator_state.mujoco_model.ncam) > 0
        control_has_cameras = bool(self._control_camera_indices)
        value_has_cameras = bool(self._value_camera_indices)
        reward_has_cameras = bool(self._reward_camera_indices)
        needs_input_render = control_has_cameras or value_has_cameras or reward_has_cameras
        if (needs_input_render or record_cameras) and render_ctx is None:
            raise ValueError("Camera capture requires a render context")

        def state_fn(spec, state):
            return simulator_class.get_model_state_pure(
                spec, state, model, simulator_state.contact_sensor_addresses,
                simulator_state.contact_body_pairs,
            )

        control_spec = self.model.get_control_input_spec()
        value_spec = self.model.get_value_input_spec()
        reward_spec = self.model.get_reward_input_spec()
        control_state_fn = lambda state, _: state_fn(control_spec, state)
        value_state_fn = lambda state, _: state_fn(value_spec, state)
        reward_state_fn = lambda state, _: state_fn(reward_spec, state)
        reward_fn = self.model.get_pure_reward_fn()

        def collect_chunk(policy, states, keys, ctx, length):
            def step(carry, _):
                step_data, step_keys = carry
                input_frames = None
                if needs_input_render:
                    input_frames, step_data = simulator_class.get_model_render(step_data, model, ctx)
                (next_data, next_keys), (
                    controls, values, outputs, _, reward_pre, valid,
                ) = jax.vmap(
                    lambda d, key, frames: self.gen_trajectory_step_gpu(
                        d, model, key, simulator_class, control_state_fn,
                        value_state_fn, reward_state_fn, control_has_cameras,
                        value_has_cameras, reward_has_cameras, frames, policy,
                    ), in_axes=(0, 0, 0 if needs_input_render else None),
                )(step_data, step_keys, input_frames)
                reward_post = jax.vmap(reward_state_fn, in_axes=(0, None))(next_data, model)
                frames = None
                if reward_has_cameras or record_cameras:
                    frames, next_data = simulator_class.get_model_render(next_data, model, ctx)
                    reward_post = self._with_camera_frames(
                        reward_post, frames, self._reward_camera_indices
                    )
                rewards = jax.vmap(reward_fn)(reward_pre, reward_post)
                return (next_data, next_keys), (
                    controls, values, outputs, rewards, valid,
                    next_data.xpos, next_data.xquat, frames if record_cameras else None,
                )
            return jax.lax.scan(step, (states, keys), None, length=length)

        if self._jit_gen_trajs is None:
            self._jit_gen_trajs = jax.jit(collect_chunk, static_argnums=(4,))
        keys = jax.random.split(rng_key, num_trajs)
        chunks = []
        for start in range(0, self.traj_depth, 10):
            if should_stop is not None and should_stop():
                return []
            length = min(10, self.traj_depth - start)
            (data, keys), outputs = self._jit_gen_trajs(
                self.model.get_policy_store(), data, keys, render_ctx, length,
            )
            jax.block_until_ready(outputs)
            controls, values, policy_outputs, rewards, valid, positions, quaternions, frames = outputs
            if not np.isfinite(np.asarray(rewards)).all():
                raise ValueError("Training rewards must be finite")
            invalid = np.argwhere(~np.asarray(valid))
            if len(invalid):
                timestep, trajectory = invalid[0]
                raise ValueError(
                    f"Invalid actuator control mapping: trajectory_index={trajectory} "
                    f"timestep={start + timestep}"
                )
            # PNG encoding, S3 writes and outgoing records happen only after the GPU chunk.
            if record_batch is not None:
                record_batch(
                    start, np.asarray(positions), np.asarray(quaternions),
                    np.asarray(frames) if frames is not None else None,
                )
            chunks.append((controls, values, policy_outputs, rewards))

        controls, values, policy_outputs, rewards = jax.tree_util.tree_map(
            lambda *parts: jnp.concatenate(parts, axis=0), *chunks,
        )
        terminal = jax.vmap(value_state_fn, in_axes=(0, None))(data, model)
        if value_has_cameras:
            frames, data = simulator_class.get_model_render(data, model, render_ctx)
            terminal = self._with_camera_frames(terminal, frames, self._value_camera_indices)
        values = jax.tree_util.tree_map(
            lambda sequence, last: jnp.concatenate([sequence, last[None]], axis=0),
            values, terminal,
        )
        controls, values, policy_outputs, rewards = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (controls, values, policy_outputs, rewards),
        )
        # Evaluate every rollout state once, before any optimizer update.
        value_outputs = jax.lax.stop_gradient(jnp.asarray(
            self.model.get_value_fn()(self.model.get_value_store(), values)
        ).reshape(num_trajs, self.traj_depth + 1))
        rtgs = jax.vmap(lambda r, v: pure_compute_rtg(r, self.gamma_lambda[0], v[-1]))(
            rewards, value_outputs,
        )
        return [Trajectory(
            control_wstates=jax.tree_util.tree_map(lambda x: x[i], controls),
            value_wstates=jax.tree_util.tree_map(lambda x: x[i], values),
            value_outputs=value_outputs[i],
            policy_outputs=jax.tree_util.tree_map(lambda x: x[i], policy_outputs),
            policy_output_sample_log_probs=policy_outputs["sampled_action_log_prob"][i],
            rewards=rewards[i], rtgs=rtgs[i],
        ) for i in range(num_trajs)]


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
    all_values: Array,
    rewards: Array,
    gamma: float,
    lam: float,
) -> Array:
    """GAE from frozen rollout values [T+1] and rewards [T]."""
    values = all_values[:-1]       # [T]
    values_next = all_values[1:]   # [T]

    deltas = rewards + gamma * values_next - values              # [T]

    def scan_fn(gae_next: Array, delta: Array) -> tuple[Array, Array]:
        gae = delta + gamma * lam * gae_next
        return gae, gae

    _, gaes_reversed = jax.lax.scan(
        scan_fn,
        jnp.array(0.0, dtype=deltas.dtype),
        jnp.flip(deltas, axis=0),
    )
    return jnp.flip(gaes_reversed, axis=0)

def pure_ppo_clip_loss(
    policy_store: Params,
    policy_fn_log_prob: Callable[[Params, PolicyInputState, dict[str, Array]], Array],
    wstates: PolicyInputState,
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
        wstates:            structured control state, leaves [num_trajs, T, ...]
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
    value_fn: Callable[[Params, PolicyInputState], Array],
    wstates: PolicyInputState,
    rtgs: Array,
) -> Array:
    """MSE value function loss.

    Args:
        value_store: value network parameters
        value_fn:    callable (params, structured state) -> [num_trajs, T, 1]
        wstates:     structured value state, leaves [num_trajs, T, ...]
        rtgs:        [num_trajs, T]

    Returns:
        scalar loss
    """
    values = value_fn(value_store, wstates)
    values = jnp.asarray(values)
    if values.ndim == rtgs.ndim + 1 and values.shape[-1] == 1:
        values = jnp.squeeze(values, axis=-1)
    if values.shape != rtgs.shape:
        raise ValueError("Value predictions must provide one scalar per rollout timestep")
    return jnp.mean((values - rtgs) ** 2)


def pure_adam_step(
    params: Params,
    grads: Params,
    opt_state: dict,
    learning_rate: float,
) -> tuple[Params, dict]:
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    if "t" not in opt_state:
        results = {
            name: pure_adam_step(params[name], grads[name], state, learning_rate)
            for name, state in opt_state.items()
        }
        return (
            {name: result[0] for name, result in results.items()},
            {name: result[1] for name, result in results.items()},
        )
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


def learning_step(
    policy_store: Params,
    value_store: Params,
    policy_opt: dict,
    value_opt: dict,
    policy_fn_log_prob: Callable[[Params, PolicyInputState, dict[str, Array]], Array],
    value_fn: Callable[[Params, PolicyInputState], Array],
    control_wstates: PolicyInputState,
    value_wstates: PolicyInputState,
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
        policy_opt:         Adam state for policy; VLA components each have {"m", "v", "t"}
        value_opt:          Adam state for value  {"m", "v", "t"}
        policy_fn_log_prob: pure fn (params, control_wstates, policy_output) ->
                            per-step joint log-probs [num_trajs, T]
        value_fn:           callable (params, value_wstates) -> values
        control_wstates:    structured control state, leaves [num_trajs, T, ...]
        value_wstates:      structured value state, leaves [num_trajs, T, ...]
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
    policy_loss_fn = lambda p: pure_ppo_clip_loss(
        p, policy_fn_log_prob, control_wstates, policy_output, start_log_probs, gaes, epsilon
    )
    policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_store)
    policy_grad_norm = pure_gradient_norm(policy_grads)
    policy_store_next, policy_opt_next = pure_adam_step(policy_store, policy_grads, policy_opt, learning_rate)

    value_loss_fn = lambda v: pure_value_loss(v, value_fn, value_wstates, rtgs)
    value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_store)
    value_grad_norm = pure_gradient_norm(value_grads)
    value_store_next, value_opt_next = pure_adam_step(value_store, value_grads, value_opt, learning_rate)

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
