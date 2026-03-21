from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ppo_trainer import ControlPolicyOutputMode, PPOTrainedModel

Array = jax.Array
Params = Any

class ArmModel(PPOTrainedModel):
    def __init__(
        self,
        policy: Callable[[Params, Array], Array],
        value: Callable[[Params, Array], Array],
        policy_store: Params,
        value_store: Params,
        nbody: int = 0,
    ) -> None:
        self.control_output_mode = ControlPolicyOutputMode.TargetPositions
        self.policy = policy
        self.value = value
        self.policy_store = policy_store
        self.value_store = value_store
        self.nbody = nbody
        self.bq = 7
        self.target_cube_body_index = 3
        self.start_center_xy = (0.35, 0.22)
        self.target_center_xy = (0.35, -0.22)
        self.start_radius = 0.09
        self.target_radius = 0.04
        self.plane_z = 0.15

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
            cls._init_linear_params(
                keys[0],
                input_dimension,
                hidden_layer_dimension,
            )
        )
        for i in range(max(0, num_layers - 1)):
            layers.append(
                cls._init_linear_params(
                    keys[i + 1],
                    hidden_layer_dimension,
                    hidden_layer_dimension,
                )
            )
        layers.append(
            cls._init_linear_params(
                keys[-1],
                hidden_layer_dimension,
                output_dimension,
            )
        )
        params = {"layers": layers}

        def policy_fn(policy_params: Params, state: Array) -> Array:
            x = jnp.asarray(state, dtype=jnp.float32)
            for lyr in policy_params["layers"][:-1]:
                x = jnp.matmul(x, lyr["w"]) + lyr["b"]
                x = jax.nn.relu(x)
            out = jnp.matmul(x, policy_params["layers"][-1]["w"]) + policy_params["layers"][-1]["b"]

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
        nbody: int = 0,
    ) -> ArmModel:
        return cls(
            policy=policy,
            value=val_estimator,
            policy_store=pol_store,
            value_store=val_estimator_store,
            nbody=nbody,
        )

    def forward_policy_impl(self, state: Array) -> Array:
        return self.policy(self.policy_store, state)

    def get_policy_impl(self) -> Callable[[Params, Array], Array]:
        return self.policy

    def get_policy_store_impl(self) -> Params:
        return self.policy_store

    def get_policy_store_mut_impl(self) -> Params:
        return self.policy_store

    @staticmethod
    def _to_1d_float_array(state: Array) -> np.ndarray:
        vals = np.asarray(jax.device_get(state), dtype=np.float64).reshape(-1)
        if vals.size % 7 != 0:
            raise ValueError(
                f"Expected cartesian state len to be 7*nbody, got {vals.size}"
            )
        return vals

    def get_body_pos(
        self,
        state: Array,
        body_index: int,
    ) -> tuple[float, float, float]:
        state_vals = self._to_1d_float_array(state)
        nbody = state_vals.size // 7
        if body_index >= nbody:
            raise ValueError(f"Body index {body_index} out of bounds for nbody {nbody}")
        pos_offset = body_index * 3
        return (
            float(state_vals[pos_offset]),
            float(state_vals[pos_offset + 1]),
            float(state_vals[pos_offset + 2]),
        )

    def get_body_quat(
        self,
        state: Array,
        body_index: int,
    ) -> tuple[float, float, float, float]:
        state_vals = self._to_1d_float_array(state)
        nbody = state_vals.size // 7
        if body_index >= nbody:
            raise ValueError(f"Body index {body_index} out of bounds for nbody {nbody}")
        quat_offset = nbody * 3 + body_index * 4
        return (
            float(state_vals[quat_offset]),
            float(state_vals[quat_offset + 1]),
            float(state_vals[quat_offset + 2]),
            float(state_vals[quat_offset + 3]),
        )

    def get_reward_2(
        self,
        cstate: Array,
        cstate_next: Array,
        contact_pairs: list[tuple[int, int]] | None,
    ) -> float:
        left_box_body_index = 1
        right_box_body_index = 2
        gripper_body_index = 10
        gripper_prop_body_index = 11
        left_finger_body_index = 12
        right_finger_body_index = 13
        left_box_start = (0.35, 0.22, 0.0)
        right_box_start = (0.35, -0.22, 0.0)
        w_prog = 10.0
        w_gripper_above = 2.0
        w_gripper_below = 4.0
        w_gripper_down = 1.5
        w_box_move = 20.0
        w_cube_drop = 20.0
        w_grip_contact = 20.0
        w_cube_contact_clean = 70.0
        cube_drop_z_threshold = 0.10
        gripper_xy_reward_radius = 0.035

        next_cx, next_cy, next_cz = self.get_body_pos(cstate_next, self.target_cube_body_index)
        next_cube_dx = next_cx - self.target_center_xy[0]
        next_cube_dy = next_cy - self.target_center_xy[1]
        next_cube_dz = next_cz - 0.025
        next_cube_dist = (next_cube_dx * next_cube_dx + next_cube_dy * next_cube_dy) ** 0.5

        prev_cx, prev_cy, prev_cz = self.get_body_pos(cstate, self.target_cube_body_index)
        prev_cube_dx = prev_cx - self.target_center_xy[0]
        prev_cube_dy = prev_cy - self.target_center_xy[1]
        prev_cube_dz = prev_cz - 0.025
        prev_cube_dist = (prev_cube_dx * prev_cube_dx + prev_cube_dy * prev_cube_dy) ** 0.5
        r_progress = (prev_cube_dist - next_cube_dist) * w_prog
        _ = (next_cube_dz, prev_cube_dz, r_progress)

        gripper_next_x, gripper_next_y, gripper_next_z = self.get_body_pos(
            cstate_next, gripper_body_index
        )
        gripper_xy_dist = (
            (gripper_next_x - next_cx) ** 2 + (gripper_next_y - next_cy) ** 2
        ) ** 0.5
        gripper_xy = -w_gripper_above * gripper_xy_dist

        z_clearance = 0.03
        z_sat_k = 4.0
        z_gap = gripper_next_z - next_cz
        z_above = max(z_gap, 0.0)
        if z_gap < 0.0 and (abs(z_gap) - z_clearance) > 0.0:
            z_below = abs(z_gap) - z_clearance
        else:
            z_below = 0.0
        r_above_z = w_gripper_above * np.exp(-z_sat_k * z_above)
        p_below_z = -w_gripper_below * z_below
        if gripper_xy_dist <= gripper_xy_reward_radius:
            r_gripper_above = p_below_z + r_above_z
        else:
            r_gripper_above = p_below_z

        gripper_qw, gripper_qx, gripper_qy, gripper_qz = self.get_body_quat(
            cstate_next, gripper_body_index
        )
        gripper_x_world_z = 2.0 * (gripper_qx * gripper_qz - gripper_qw * gripper_qy)
        down_alignment = float(np.clip(-gripper_x_world_z, 0.0, 1.0))
        r_gripper_down = w_gripper_down * down_alignment

        if contact_pairs is not None:
            left_contact = False
            right_contact = False
            prop_contact = False
            has_disallowed_cube_contact = False
            for body_a, body_b in contact_pairs:
                cube_other: int | None
                if body_a == self.target_cube_body_index:
                    cube_other = body_b
                elif body_b == self.target_cube_body_index:
                    cube_other = body_a
                else:
                    cube_other = None
                if cube_other is None:
                    continue
                if cube_other == left_finger_body_index:
                    left_contact = True
                elif cube_other == right_finger_body_index:
                    right_contact = True
                elif cube_other == gripper_prop_body_index:
                    prop_contact = True
                else:
                    has_disallowed_cube_contact = True
            contact_count = int(left_contact) + int(right_contact) + int(prop_contact)
            has_allowed_contact = contact_count > 0
            r_grip_contact = w_grip_contact * contact_count
            if has_allowed_contact and not has_disallowed_cube_contact:
                r_cube_contact_clean = w_cube_contact_clean
            else:
                r_cube_contact_clean = 0.0
        else:
            r_grip_contact = 0.0
            r_cube_contact_clean = 0.0

        left_x, left_y, left_z = self.get_body_pos(cstate_next, left_box_body_index)
        right_x, right_y, right_z = self.get_body_pos(cstate_next, right_box_body_index)
        _ = (left_z, right_z)
        left_disp = ((left_x - left_box_start[0]) ** 2 + (left_y - left_box_start[1]) ** 2) ** 0.5
        right_disp = ((right_x - right_box_start[0]) ** 2 + (right_y - right_box_start[1]) ** 2) ** 0.5
        r_box_moved = -w_box_move * (left_disp + right_disp)

        if next_cz < cube_drop_z_threshold:
            p_cube_drop = -w_cube_drop
        else:
            p_cube_drop = 0.0

        return (
            r_gripper_above
            + r_gripper_down
            + gripper_xy
            + r_box_moved
            + p_cube_drop
            + r_grip_contact
            + r_cube_contact_clean
        )

    def get_reward(
        self,
        cstate: Array,
        cstate_next: Array,
        action: Array,
        contact_pairs: list[tuple[int, int]] | None,
    ) -> float:
        _ = action
        return self.get_reward_2(cstate, cstate_next, contact_pairs)

    def get_policy_output(self, state: Array) -> Array:
        return self.forward_policy_impl(state)

    def get_value_output(self, state: Array) -> float:
        val = self.value(self.value_store, state)
        val_arr = np.asarray(jax.device_get(val), dtype=np.float64).reshape(-1)
        if val_arr.size == 0:
            raise ValueError("Expected non-empty value output")
        return float(val_arr[0])

    def get_policy(self) -> Callable[[Params, Array], Array]:
        return self.get_policy_impl()

    def get_value_func(self) -> Callable[[Params, Array], Array]:
        return self.value

    def get_policy_store(self) -> Params:
        return self.get_policy_store_impl()

    def get_policy_store_mut(self) -> Params:
        return self.get_policy_store_mut_impl()

    def get_value_store(self) -> Params:
        return self.value_store

    def get_value_store_mut(self) -> Params:
        return self.value_store

    def get_control_policy_output_mode(self) -> ControlPolicyOutputMode:
        return self.control_output_mode

    def get_pure_policy_fn(self) -> Callable[[Params, Array], Array]:
        return self.policy

    def get_pure_reward_fn(self) -> Callable[[Array, Array, Array, Array], Array]:
        target_cube_body_index = self.target_cube_body_index
        left_finger_body_index = 12
        right_finger_body_index = 13
        gripper_prop_body_index = 11
        gripper_body_index = 10
        left_box_body_index = 1
        right_box_body_index = 2
        target_center_xy = self.target_center_xy
        left_box_start = (0.35, 0.22, 0.0)
        right_box_start = (0.35, -0.22, 0.0)
        w_prog = 10.0
        w_gripper_above = 2.0
        w_gripper_below = 4.0
        w_gripper_down = 1.5
        w_box_move = 20.0
        w_cube_drop = 20.0
        w_grip_contact = 20.0
        w_cube_contact_clean = 70.0
        cube_drop_z_threshold = 0.10
        gripper_xy_reward_radius = 0.035
        z_clearance = 0.03
        z_sat_k = 4.0

        def _get_body_pos_jax(cstate: Array, body_index: int, nbody: int) -> tuple[Array, Array, Array]:
            x = cstate[body_index * 3]
            y = cstate[body_index * 3 + 1]
            z = cstate[body_index * 3 + 2]
            return x, y, z

        def _get_body_quat_jax(cstate: Array, body_index: int, nbody: int) -> tuple[Array, Array, Array, Array]:
            quat_offset = nbody * 3 + body_index * 4
            return cstate[quat_offset], cstate[quat_offset + 1], cstate[quat_offset + 2], cstate[quat_offset + 3]

        def pure_reward(cstate: Array, cstate_next: Array, action: Array, contact_matrix: Array) -> Array:
            nbody = contact_matrix.shape[0]

            next_cx, next_cy, next_cz = _get_body_pos_jax(cstate_next, target_cube_body_index, nbody)
            next_cube_dx = next_cx - target_center_xy[0]
            next_cube_dy = next_cy - target_center_xy[1]
            next_cube_dist = jnp.sqrt(next_cube_dx ** 2 + next_cube_dy ** 2)

            prev_cx, prev_cy, _ = _get_body_pos_jax(cstate, target_cube_body_index, nbody)
            prev_cube_dx = prev_cx - target_center_xy[0]
            prev_cube_dy = prev_cy - target_center_xy[1]
            prev_cube_dist = jnp.sqrt(prev_cube_dx ** 2 + prev_cube_dy ** 2)
            _ = (prev_cube_dist - next_cube_dist) * w_prog

            gripper_next_x, gripper_next_y, gripper_next_z = _get_body_pos_jax(cstate_next, gripper_body_index, nbody)
            gripper_xy_dist = jnp.sqrt((gripper_next_x - next_cx) ** 2 + (gripper_next_y - next_cy) ** 2)
            gripper_xy = -w_gripper_above * gripper_xy_dist

            z_gap = gripper_next_z - next_cz
            z_above = jnp.maximum(z_gap, 0.0)
            z_below = jnp.maximum(jnp.abs(z_gap) - z_clearance, 0.0) * jnp.where(z_gap < 0.0, 1.0, 0.0)
            r_above_z = w_gripper_above * jnp.exp(-z_sat_k * z_above)
            p_below_z = -w_gripper_below * z_below
            r_gripper_above = jnp.where(
                gripper_xy_dist <= gripper_xy_reward_radius,
                p_below_z + r_above_z,
                p_below_z,
            )

            qw, qx, qy, qz = _get_body_quat_jax(cstate_next, gripper_body_index, nbody)
            gripper_x_world_z = 2.0 * (qx * qz - qw * qy)
            down_alignment = jnp.clip(-gripper_x_world_z, 0.0, 1.0)
            r_gripper_down = w_gripper_down * down_alignment

            left_contact = contact_matrix[target_cube_body_index, left_finger_body_index]
            right_contact = contact_matrix[target_cube_body_index, right_finger_body_index]
            prop_contact = contact_matrix[target_cube_body_index, gripper_prop_body_index]
            # sum of all cube contacts minus allowed ones — if nonzero, cube has disallowed contacts
            cube_row = contact_matrix[target_cube_body_index]
            allowed_mask = jnp.zeros_like(cube_row).at[left_finger_body_index].set(1.0).at[right_finger_body_index].set(1.0).at[gripper_prop_body_index].set(1.0)
            disallowed_contacts = jnp.sum(cube_row * (1.0 - allowed_mask))
            has_disallowed = disallowed_contacts > 0.0
            contact_count = left_contact + right_contact + prop_contact
            r_grip_contact = w_grip_contact * contact_count
            r_cube_contact_clean = jnp.where(
                (contact_count > 0.0) & ~has_disallowed,
                w_cube_contact_clean,
                0.0,
            )

            left_x, left_y, _ = _get_body_pos_jax(cstate_next, left_box_body_index, nbody)
            right_x, right_y, _ = _get_body_pos_jax(cstate_next, right_box_body_index, nbody)
            left_disp = jnp.sqrt((left_x - left_box_start[0]) ** 2 + (left_y - left_box_start[1]) ** 2)
            right_disp = jnp.sqrt((right_x - right_box_start[0]) ** 2 + (right_y - right_box_start[1]) ** 2)
            r_box_moved = -w_box_move * (left_disp + right_disp)

            p_cube_drop = jnp.where(next_cz < cube_drop_z_threshold, -w_cube_drop, 0.0)

            return (
                r_gripper_above
                + r_gripper_down
                + gripper_xy
                + r_box_moved
                + p_cube_drop
                + r_grip_contact
                + r_cube_contact_clean
            )

        return pure_reward
