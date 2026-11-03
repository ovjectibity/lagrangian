"""MJX operational-space helpers for robot forces and actuator controls.

The named end effector and robot base are bodies. Commands and wrenches use
world coordinates at the end-effector body origin. The model/data must describe
the same forwarded state, and body names remain static for each compiled model.
The robot base is assumed fixed; environment DOFs are held fixed in the reduced
inertia calculation. Robot descendants include the gripper, whose actual
Jacobian columns are retained. The end-effector helper returns robot DOF
forces; the actuator helper maps those forces to robot-only controls.
"""

import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src import support as mjx_support
import numpy as np

def _body_id(model: mjx.Model, name: str) -> int:
    body_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Unknown MuJoCo body: {name}")
    return body_id


def _robot_body_mask(model: mjx.Model, robot_base_name: str) -> np.ndarray:
    """Identify the named robot body and all of its descendants at trace time."""
    base_id = _body_id(model, robot_base_name)
    parent_ids = np.asarray(model.body_parentid)
    in_robot = np.zeros(model.nbody, dtype=bool)
    in_robot[base_id] = True
    # MuJoCo numbers parents before their child bodies, so this pass propagates
    # subtree membership without inspecting dynamic JAX values.
    for body_id in range(model.nbody):
        parent_id = int(parent_ids[body_id])
        if body_id != base_id and parent_id != body_id:
            in_robot[body_id] = in_robot[parent_id]
    return in_robot


def _robot_dof_indices(model: mjx.Model, robot_base_name: str) -> jnp.ndarray:
    """Select DOFs in the named robot subtree, including gripper joints."""
    in_robot = _robot_body_mask(model, robot_base_name)

    indices = np.flatnonzero(in_robot[np.asarray(model.dof_bodyid)])
    if indices.size == 0:
        raise ValueError(f"Robot base body has no DOFs: {robot_base_name}")
    # Model topology is static during JAX tracing, so this output length is
    # constant for every invocation with this model.
    return jnp.asarray(indices, dtype=jnp.int32)


def _partial_dof_indices(
    model: mjx.Model,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robot-local and model-global DOF IDs below ``rebase_body_name``.

    Both body names are resolved from model topology, hence the selected
    coordinate count and ordering are static for a given compiled model.  The
    rebased body must be within the robot base subtree; its descendant joints
    (rather than only joints attached directly to that body) define the
    partial coordinate task.
    """
    in_robot = _robot_body_mask(model, robot_base_name)
    in_rebase = _robot_body_mask(model, rebase_body_name)
    rebase_id = _body_id(model, rebase_body_name)
    if not in_robot[rebase_id]:
        raise ValueError(
            f"Rebase body {rebase_body_name} is outside robot base {robot_base_name}"
        )

    dof_bodies = np.asarray(model.dof_bodyid)
    robot_dofs = np.flatnonzero(in_robot[dof_bodies])
    partial_dofs = np.flatnonzero(in_rebase[dof_bodies] & in_robot[dof_bodies])
    if partial_dofs.size == 0:
        raise ValueError(
            f"Rebase body {rebase_body_name} has no descendant joint DOFs"
        )

    # ``robot_dofs`` is sorted in global DOF order, so searchsorted yields
    # stable robot-local coordinates for every selected descendant DOF.
    return np.searchsorted(robot_dofs, partial_dofs), partial_dofs


def _partial_joint_ids(
    model: mjx.Model,
    robot_base_name: str,
    rebase_body_name: str,
) -> np.ndarray:
    """Return joint IDs attached to the rebased body subtree in model order."""
    in_robot = _robot_body_mask(model, robot_base_name)
    in_rebase = _robot_body_mask(model, rebase_body_name)
    rebase_id = _body_id(model, rebase_body_name)
    if not in_robot[rebase_id]:
        raise ValueError(
            f"Rebase body {rebase_body_name} is outside robot base {robot_base_name}"
        )
    joint_bodies = np.asarray(model.jnt_bodyid)
    return np.flatnonzero(in_rebase[joint_bodies] & in_robot[joint_bodies])


def _robot_actuator_indices(model: mjx.Model, robot_base_name: str) -> np.ndarray:
    """Select robot actuators and reject unsupported transmissions."""
    in_robot = _robot_body_mask(model, robot_base_name)
    trn_types = np.asarray(model.actuator_trntype)
    trn_ids = np.asarray(model.actuator_trnid)
    joint_bodies = np.asarray(model.jnt_bodyid)
    site_bodies = np.asarray(model.site_bodyid)
    geom_bodies = np.asarray(model.geom_bodyid)
    selected = []

    def site_in_robot(site_id: int) -> bool:
        return site_id >= 0 and bool(in_robot[site_bodies[site_id]])

    for actuator_id, trn_type in enumerate(trn_types):
        trn_type = int(trn_type)
        target_id, reference_id = map(int, trn_ids[actuator_id])
        if trn_type in (
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        ):
            belongs_to_robot = bool(in_robot[joint_bodies[target_id]])
        elif trn_type == int(mujoco.mjtTrn.mjTRN_SITE):
            belongs_to_robot = site_in_robot(target_id) or site_in_robot(
                reference_id
            )
        elif trn_type == int(mujoco.mjtTrn.mjTRN_SLIDERCRANK):
            belongs_to_robot = site_in_robot(target_id) or site_in_robot(
                reference_id
            )
        elif trn_type == int(mujoco.mjtTrn.mjTRN_BODY):
            belongs_to_robot = bool(in_robot[target_id])
        elif trn_type == int(mujoco.mjtTrn.mjTRN_TENDON):
            first = int(model.tendon_adr[target_id])
            count = int(model.tendon_num[target_id])
            belongs_to_robot = False
            for wrap_id in range(first, first + count):
                wrap_type = int(model.wrap_type[wrap_id])
                object_id = int(model.wrap_objid[wrap_id])
                if wrap_type == int(mujoco.mjtWrap.mjWRAP_JOINT):
                    body_id = int(joint_bodies[object_id])
                elif wrap_type == int(mujoco.mjtWrap.mjWRAP_SITE):
                    body_id = int(site_bodies[object_id])
                elif wrap_type in (
                    int(mujoco.mjtWrap.mjWRAP_SPHERE),
                    int(mujoco.mjtWrap.mjWRAP_CYLINDER),
                ):
                    body_id = int(geom_bodies[object_id])
                else:
                    continue  # Pulley elements have no attached body.
                belongs_to_robot |= bool(in_robot[body_id])
        else:
            raise ValueError(f"Unsupported actuator transmission: {trn_type}")

        if belongs_to_robot:
            if trn_type not in (
                int(mujoco.mjtTrn.mjTRN_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            ):
                raise ValueError(
                    f"Robot actuator {actuator_id} has unsupported transmission "
                    f"{mujoco.mjtTrn(trn_type).name}"
                )
            selected.append(actuator_id)

    if not selected:
        raise ValueError(f"Robot has no supported actuators: {robot_base_name}")
    return np.asarray(selected, dtype=np.int32)


def get_cartesian_body_jacobian(
    model: mjx.Model,
    data: mjx.Data,
    target_body: str,
    robot_base_name: str,
) -> jnp.ndarray:
    """Return a body's world-frame origin Jacobian, shape ``(6, robot_nv)``."""
    body_id = _body_id(model, target_body)
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    # MJX returns (nv, 3) linear and angular blocks. Keep linear rows first
    # throughout the task-space helpers; retain the actual gripper columns.
    jacp, jacr = mjx.jac(model, data, data.xpos[body_id], jnp.asarray(body_id))
    return jnp.concatenate((jacp[robot_dofs].T, jacr[robot_dofs].T), axis=0)


def get_cartesian_body_jacobian_derivative(
    model: mjx.Model,
    data: mjx.Data,
    target_body: str,
    robot_base_name: str,
) -> jnp.ndarray:
    """Return the world-frame time derivative of the body Jacobian."""
    body_id = _body_id(model, target_body)
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    # MJX 3.11 exposes jac_dot in its support module, but not its public API.
    # It uses the forwarded cdof_dot/cvel fields; no CPU MuJoCo call is made.
    jacp_dot, jacr_dot = mjx_support.jac_dot(
        model, data, data.xpos[body_id], jnp.asarray(body_id)
    )
    return jnp.concatenate(
        (jacp_dot[robot_dofs].T, jacr_dot[robot_dofs].T), axis=0
    )


def get_generalised_inertia(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
) -> jnp.ndarray:
    """Return the robot block of the full-model inertia matrix."""
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    # Selecting coordinates does not remove the gripper bodies or their mass.
    # Environment DOF accelerations are assumed fixed in this reduced model.
    return mjx.full_m(model, data)[jnp.ix_(robot_dofs, robot_dofs)]


def get_inverse_generalised_inertia(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
) -> jnp.ndarray:
    return jnp.linalg.inv(get_generalised_inertia(model, data, robot_base_name))


def get_generalised_bias_forces(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
) -> jnp.ndarray:
    return data.qfrc_bias[_robot_dof_indices(model, robot_base_name)]


def get_generalised_forces_from_task_space_config_error(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
    pos_config_error: jnp.ndarray,
    vel_config_error: jnp.ndarray,
    kd: jnp.ndarray | float,
    kp: jnp.ndarray | float,
    task_space_jacobian: jnp.ndarray,
    task_space_jacobian_derivative: jnp.ndarray,
) -> jnp.ndarray:
    """Map task configuration PD errors to robot-base generalized forces.

    ``pos_config_error`` is already expressed in the task's velocity tangent
    space:
    for example, an EEF pose uses three position and three axis-angle values,
    while a partial joint task uses one value per selected DOF.  Consequently,
    this helper does not interpret absolute targets, deltas, or quaternions.

    ``vel_config_error`` is the desired-minus-current task velocity in the
    same coordinate order.  This makes non-zero task velocity commands
    explicit rather than assuming a zero velocity target internally.

    The supplied Jacobian maps robot-base generalized velocity to task
    velocity.  Its derivative must use the same coordinate order.  Constraint
    forces are intentionally omitted, and passive forces are compensated, in
    line with the existing OSC helpers.
    """
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    robot_nv = robot_dofs.shape[0]
    if pos_config_error.ndim != 1 or vel_config_error.ndim != 1:
        raise ValueError(
            "pos_config_error and vel_config_error must be one-dimensional"
        )
    task_nv = pos_config_error.shape[0]
    if vel_config_error.shape != (task_nv,):
        raise ValueError(
            "vel_config_error must have shape "
            f"{(task_nv,)}, got {vel_config_error.shape}"
        )
    expected_shape = (task_nv, robot_nv)
    if task_space_jacobian.shape != expected_shape:
        raise ValueError(
            "task_space_jacobian must have shape "
            f"{expected_shape}, got {task_space_jacobian.shape}"
        )
    if task_space_jacobian_derivative.shape != expected_shape:
        raise ValueError(
            "task_space_jacobian_derivative must have shape "
            f"{expected_shape}, got {task_space_jacobian_derivative.shape}"
        )

    kp = jnp.asarray(kp)
    kd = jnp.asarray(kd)
    if kp.shape not in ((), (task_nv,)) or kd.shape not in ((), (task_nv,)):
        raise ValueError(
            "kp and kd must be scalars or have one value per task coordinate"
        )

    robot_qvel = data.qvel[robot_dofs]
    desired_acceleration = kp * pos_config_error + kd * vel_config_error
    inverse_inertia = get_inverse_generalised_inertia(
        model, data, robot_base_name
    )
    task_inertia = jnp.linalg.pinv(
        task_space_jacobian
        @ inverse_inertia
        @ task_space_jacobian.T
    )
    bias_minus_passive = (
        get_generalised_bias_forces(model, data, robot_base_name)
        - data.qfrc_passive[robot_dofs]
    )
    task_force = task_inertia @ (
        desired_acceleration
        - task_space_jacobian_derivative @ robot_qvel
        + task_space_jacobian @ inverse_inertia @ bias_minus_passive
    )
    return task_space_jacobian.T @ task_force


def get_actuator_control_input_from_generalised_forces(
    model: mjx.Model,
    data: mjx.Data,
    tau: jnp.ndarray,
    robot_base_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map robot generalized forces to robot actuator controls.

    Returns controls in model actuator order and a scalar validity flag. The
    flag is false if the minimum-norm force solution misses tau, its gain is
    zero, or its force/control exceeds an actuator limit. A false flag does not
    rule out a different bounded solution when actuators are redundant. Static
    actuator configurations outside the supported subset raise ValueError at
    trace time.
    """
    actuator_ids_static = _robot_actuator_indices(model, robot_base_name)
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    actuator_ids = jnp.asarray(actuator_ids_static, dtype=jnp.int32)
    if data.impl.name == "JAX":
        moments = data._impl.actuator_moment[actuator_ids_static]
    elif data.impl.name == "WARP":
        # Warp stores moment rows sparsely. Supported joint transmissions have
        # at most six entries, including free-joint transmissions.
        if model.nJmom == 0:
            raise ValueError("Robot actuators have no transmission moments")
        offsets = jnp.arange(6, dtype=jnp.int32)
        row_start = data._impl.moment_rowadr[actuator_ids_static]
        row_count = data._impl.moment_rownnz[actuator_ids_static]
        addresses = jnp.clip(
            row_start[:, None] + offsets[None, :], 0, model.nJmom - 1
        )
        valid = offsets[None, :] < row_count[:, None]
        columns = jnp.where(valid, data._impl.moment_colind[addresses], 0)
        values = jnp.where(valid, data._impl.actuator_moment[addresses], 0)
        moments = jnp.zeros(
            (actuator_ids_static.size, model.nv), dtype=data.qvel.dtype
        )
        moments = moments.at[
            jnp.arange(actuator_ids_static.size)[:, None], columns
        ].add(values)
    else:
        raise ValueError(f"Unsupported MJX implementation: {data.impl}")
    moments = moments[:, robot_dofs]
    if tau.shape != (moments.shape[1],):
        raise ValueError(
            "tau must have one entry per robot DOF: "
            f"expected {(moments.shape[1],)}, got {tau.shape}"
        )

    nactuator = model.actuator_trntype.shape[0]
    if model.nu != nactuator or data.actuator_length.shape[0] != nactuator:
        raise ValueError("Only single-input, single-output actuators are supported")
    if np.any(
        np.asarray(model.actuator_dyntype)[actuator_ids_static]
        != int(mujoco.mjtDyn.mjDYN_NONE)
    ):
        raise ValueError("Robot actuators with activation dynamics are unsupported")

    gain_types = np.asarray(model.actuator_gaintype)[actuator_ids_static]
    bias_types = np.asarray(model.actuator_biastype)[actuator_ids_static]
    if np.any(
        ~np.isin(
            gain_types,
            (int(mujoco.mjtGain.mjGAIN_FIXED), int(mujoco.mjtGain.mjGAIN_AFFINE)),
        )
    ) or np.any(
        ~np.isin(
            bias_types,
            (int(mujoco.mjtBias.mjBIAS_NONE), int(mujoco.mjtBias.mjBIAS_AFFINE)),
        )
    ):
        raise ValueError("Robot actuator gain or bias type is unsupported")

    # Solve M.T @ F_act = tau. The pseudoinverse selects the minimum-norm
    # actuator forces if multiple actuators act on the same robot DOFs.
    actuator_forces = jnp.linalg.pinv(moments.T) @ tau
    force_error = jnp.linalg.norm(moments.T @ actuator_forces - tau)
    force_valid = force_error <= 1e-4 * (1.0 + jnp.linalg.norm(tau))

    length = data.actuator_length[actuator_ids]
    velocity = data._impl.actuator_velocity[actuator_ids]
    gain_params = model.actuator_gainprm[actuator_ids]
    bias_params = model.actuator_biasprm[actuator_ids]
    gain = jnp.where(
        jnp.asarray(gain_types == int(mujoco.mjtGain.mjGAIN_FIXED)),
        gain_params[:, 0],
        gain_params[:, 0]
        + gain_params[:, 1] * length
        + gain_params[:, 2] * velocity,
    )
    bias = jnp.where(
        jnp.asarray(bias_types == int(mujoco.mjtBias.mjBIAS_NONE)),
        0.0,
        bias_params[:, 0]
        + bias_params[:, 1] * length
        + bias_params[:, 2] * velocity,
    )
    gain_valid = jnp.all(jnp.abs(gain) > 1e-8)
    safe_gain = jnp.where(jnp.abs(gain) > 1e-8, gain, 1.0)
    controls = (actuator_forces - bias) / safe_gain

    control_limited = jnp.asarray(model.actuator_ctrllimited)[actuator_ids]
    force_limited = jnp.asarray(model.actuator_forcelimited)[actuator_ids]
    control_range = model.actuator_ctrlrange[actuator_ids]
    force_range = model.actuator_forcerange[actuator_ids]
    controls_valid = jnp.all(
        ~control_limited
        | (
            (controls >= control_range[:, 0] - 1e-6)
            & (controls <= control_range[:, 1] + 1e-6)
        )
    )
    limits_valid = jnp.all(
        ~force_limited
        | (
            (actuator_forces >= force_range[:, 0] - 1e-6)
            & (actuator_forces <= force_range[:, 1] + 1e-6)
        )
    )
    finite = jnp.all(jnp.isfinite(controls)) & jnp.all(
        jnp.isfinite(actuator_forces)
    )
    valid = force_valid & gain_valid & controls_valid & limits_valid & finite
    return controls, valid


def get_partial_dof_jacobian(
    model: mjx.Model,
    robot_base_name: str,
    rebase_body_name: str,
) -> jnp.ndarray:
    """Return ``S`` mapping robot-base velocity to rebased descendant velocity.

    If ``qdot_r`` contains all velocity coordinates below ``robot_base_name``
    and ``qdot_p`` contains just the coordinates of descendant joints below
    ``rebase_body_name``, then ``qdot_p = S @ qdot_r``.  This is a constant
    coordinate-selection Jacobian, not a Cartesian body Jacobian.
    """
    partial_local_dofs, robot_global_dofs = _partial_dof_indices(
        model, robot_base_name, rebase_body_name
    )
    robot_nv = robot_global_dofs.size
    return jnp.asarray(
        np.eye(robot_nv, dtype=np.float32)[partial_local_dofs], dtype=jnp.float32
    )


def get_eef_gripper_task_space_jacobian(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
    eef_body: str,
    gripper_body: str,
) -> jnp.ndarray:
    """Return the stacked EEF-pose and gripper-coordinate task Jacobian.

    The first six rows map robot-base generalized velocity to the EEF body's
    world-frame linear and angular velocity.  The remaining rows map it to
    the generalized velocities of all descendant joints below
    ``gripper_body``.  Both blocks therefore share the same robot-base DOF
    columns and can be used as one equal-priority task-space Jacobian.
    """
    eef_jacobian = get_cartesian_body_jacobian(
        model, data, eef_body, robot_base_name
    )
    gripper_jacobian = get_partial_dof_jacobian(
        model, robot_base_name, gripper_body
    ).astype(eef_jacobian.dtype)
    return jnp.concatenate((eef_jacobian, gripper_jacobian), axis=0)


def get_eef_gripper_task_space_jacobian_derivative(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
    eef_body: str,
    gripper_body: str,
) -> jnp.ndarray:
    """Return the derivative of the stacked EEF-pose/gripper task Jacobian.

    The Cartesian body Jacobian changes with the robot configuration.  The
    gripper block is the fixed coordinate-selection matrix ``S``, so its time
    derivative is identically zero.
    """
    eef_jacobian_derivative = get_cartesian_body_jacobian_derivative(
        model, data, eef_body, robot_base_name
    )
    gripper_jacobian = get_partial_dof_jacobian(
        model, robot_base_name, gripper_body
    )
    gripper_jacobian_derivative = jnp.zeros(
        gripper_jacobian.shape,
        dtype=eef_jacobian_derivative.dtype,
    )
    return jnp.concatenate(
        (eef_jacobian_derivative, gripper_jacobian_derivative), axis=0
    )


def get_actuator_control_input_from_eef_gripper_pos_commands(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
    eef_body: str,
    gripper_body: str,
    eef_pose_delta: jnp.ndarray,
    gripper_qpos_delta: jnp.ndarray,
    kd: jnp.ndarray | float,
    kp: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map EEF-pose and gripper-qpos deltas to robot actuator controls.

    Assumptions:

    * ``eef_pose_delta`` is ordered as ``[delta_x, delta_y, delta_z,
      axis_angle_x, axis_angle_y, axis_angle_z]``.  The translation is a
      world-frame delta.  The final three values define a true relative
      world-frame rotation, represented as axis-angle—not component-wise
      quaternion or orientation differences.
    * ``gripper_qpos_delta`` has one tangent-space entry per descendant joint
      DOF below ``gripper_body``.  Ball/free-joint rotations therefore use
      three tangent coordinates rather than quaternion components.
    * ``kp`` and ``kd`` are scalar or one value per combined task coordinate:
      six EEF coordinates followed by the gripper subtree's tangent DOFs.
    * Desired task velocity and acceleration are zero.  Constraint-force
      compensation remains intentionally excluded by the shared OSC helper.

    The EEF and gripper are treated as one equal-priority task.  One combined
    generalized force is therefore converted to controls only once, avoiding
    separate actuator inversions for the two commands.
    """
    if eef_pose_delta.shape != (6,):
        raise ValueError(
            "eef_pose_delta must have shape (6,) "
            "[world translation delta, relative world axis-angle]"
        )

    relative_axis_angle = eef_pose_delta[3:]
    relative_angle = jnp.linalg.norm(relative_axis_angle)
    # For the zero-angle case, sin(angle / 2) / angle has the finite limit
    # 1/2. This produces a proper relative quaternion in JIT-compatible JAX.
    quaternion_scale = jnp.where(
        relative_angle > 1e-8,
        jnp.sin(relative_angle / 2.0) / relative_angle,
        0.5,
    )
    relative_quaternion = jnp.concatenate(
        (
            jnp.cos(relative_angle / 2.0)[None],
            relative_axis_angle * quaternion_scale,
        )
    )
    rotation_axis, rotation_angle = mjx_math.quat_to_axis_angle(
        relative_quaternion
    )
    eef_pos_config_error = jnp.concatenate(
        (
            eef_pose_delta[:3],
            rotation_axis * rotation_angle,
        )
    )

    selection = get_partial_dof_jacobian(
        model, robot_base_name, gripper_body
    )
    partial_nv = selection.shape[0]
    if gripper_qpos_delta.shape != (partial_nv,):
        raise ValueError(
            "gripper_qpos_delta must have shape "
            f"{(partial_nv,)}, got {gripper_qpos_delta.shape}"
        )

    task_space_jacobian = get_eef_gripper_task_space_jacobian(
        model, data, robot_base_name, eef_body, gripper_body
    )
    task_space_jacobian_derivative = (
        get_eef_gripper_task_space_jacobian_derivative(
            model, data, robot_base_name, eef_body, gripper_body
        )
    )
    pos_config_error = jnp.concatenate(
        (eef_pos_config_error, gripper_qpos_delta)
    )
    vel_config_error = -task_space_jacobian @ data.qvel[
        _robot_dof_indices(model, robot_base_name)
    ]
    tau = get_generalised_forces_from_task_space_config_error(
        model,
        data,
        robot_base_name,
        pos_config_error,
        vel_config_error,
        kd,
        kp,
        task_space_jacobian,
        task_space_jacobian_derivative,
    )
    return get_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )


def get_actuator_control_input_from_partial_qpos(
    model: mjx.Model,
    data: mjx.Data,
    partial_qpos: jnp.ndarray,
    is_delta: bool,
    kd: jnp.ndarray | float,
    kp: jnp.ndarray | float,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map a descendant-joint qpos target to robot actuator controls.

    Absolute targets use the descendant subtree's qpos coordinates; delta
    targets use its tangent-space DOFs.  The selected DOFs form the task
    Jacobian ``S`` and use a zero target velocity.
    """
    selection = get_partial_dof_jacobian(
        model, robot_base_name, rebase_body_name
    ).astype(data.qvel.dtype)
    partial_nv = selection.shape[0]
    if is_delta:
        if partial_qpos.shape != (partial_nv,):
            raise ValueError(
                "partial qpos delta must have shape "
                f"{(partial_nv,)}, got {partial_qpos.shape}"
            )
        pos_config_error = partial_qpos
    else:
        joint_ids = _partial_joint_ids(model, robot_base_name, rebase_body_name)
        joint_types = np.asarray(model.jnt_type)
        qpos_addresses = np.asarray(model.jnt_qposadr)
        joint_widths = [
            7 if joint_types[joint_id] == int(mujoco.mjtJoint.mjJNT_FREE)
            else 4 if joint_types[joint_id] == int(mujoco.mjtJoint.mjJNT_BALL)
            else 1
            for joint_id in joint_ids
        ]
        partial_nq = sum(joint_widths)
        if partial_qpos.shape != (partial_nq,):
            raise ValueError(
                "absolute partial qpos must have shape "
                f"{(partial_nq,)}, got {partial_qpos.shape}"
            )

        errors = []
        target_address = 0
        for joint_id, width in zip(joint_ids, joint_widths):
            current_address = int(qpos_addresses[joint_id])
            target = partial_qpos[target_address : target_address + width]
            current = data.qpos[current_address : current_address + width]
            if width == 7:
                error = jnp.concatenate(
                    (
                        target[:3] - current[:3],
                        mjx_math.quat_sub(target[3:], current[3:]),
                    )
                )
            elif width == 4:
                error = mjx_math.quat_sub(target, current)
            else:
                error = target - current
            errors.append(error)
            target_address += width
        pos_config_error = jnp.concatenate(errors)

    tau = get_generalised_forces_from_task_space_config_error(
        model,
        data,
        robot_base_name,
        pos_config_error,
        -selection @ data.qvel[_robot_dof_indices(model, robot_base_name)],
        kd,
        kp,
        selection,
        jnp.zeros(selection.shape, dtype=selection.dtype),
    )
    return get_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )


def get_actuator_control_input_from_partial_qvel(
    model: mjx.Model,
    data: mjx.Data,
    partial_qvel: jnp.ndarray,
    is_delta: bool,
    kd: jnp.ndarray | float,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map a descendant-joint velocity target or delta to actuator controls.

    Absolute targets and deltas both have one entry per descendant DOF.  The
    partial task is dynamically lifted through the full robot-base inertia;
    this retains arm/gripper inertial coupling while only commanding the
    rebased descendant coordinates.
    """
    selection = get_partial_dof_jacobian(
        model, robot_base_name, rebase_body_name
    ).astype(data.qvel.dtype)
    partial_nv = selection.shape[0]
    if partial_qvel.shape != (partial_nv,):
        raise ValueError(
            "partial qvel must have shape "
            f"{(partial_nv,)}, got {partial_qvel.shape}"
        )
    current_velocity = selection @ data.qvel[
        _robot_dof_indices(model, robot_base_name)
    ]
    velocity_error = (
        partial_qvel if is_delta else partial_qvel - current_velocity
    )
    tau = get_generalised_forces_from_task_space_config_error(
        model,
        data,
        robot_base_name,
        jnp.zeros((partial_nv,), dtype=data.qvel.dtype),
        velocity_error,
        kd,
        0.0,
        selection,
        jnp.zeros(selection.shape, dtype=selection.dtype),
    )
    return get_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )


def get_actuator_control_input_from_partial_tau(
    model: mjx.Model,
    data: mjx.Data,
    partial_tau: jnp.ndarray,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map descendant-coordinate generalized forces to actuator controls.

    ``partial_tau`` is a covector in the selected descendant coordinates.
    Its robot-base representation is therefore ``S.T @ partial_tau``.
    """
    selection = get_partial_dof_jacobian(
        model, robot_base_name, rebase_body_name
    ).astype(data.qvel.dtype)
    partial_nv = selection.shape[0]
    if partial_tau.shape != (partial_nv,):
        raise ValueError(
            "partial tau must have shape "
            f"{(partial_nv,)}, got {partial_tau.shape}"
        )
    return get_actuator_control_input_from_generalised_forces(
        model, data, selection.T @ partial_tau, robot_base_name
    )


# CPU MuJoCo helpers deliberately have their own NumPy implementation rather
# than dispatching from the MJX helpers above.  Keeping the backends separate
# preserves the latter's JIT-friendly JAX-only call tree.
def _mujoco_body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Unknown MuJoCo body: {name}")
    return body_id


def _mujoco_robot_body_mask(
    model: mujoco.MjModel, robot_base_name: str
) -> np.ndarray:
    """Return a static mask for a body's full descendant subtree."""
    base_id = _mujoco_body_id(model, robot_base_name)
    in_robot = np.zeros(model.nbody, dtype=bool)
    in_robot[base_id] = True
    for body_id, parent_id in enumerate(np.asarray(model.body_parentid)):
        if body_id != base_id and body_id != parent_id:
            in_robot[body_id] = in_robot[parent_id]
    return in_robot


def _mujoco_robot_dof_indices(
    model: mujoco.MjModel, robot_base_name: str
) -> np.ndarray:
    in_robot = _mujoco_robot_body_mask(model, robot_base_name)
    dofs = np.flatnonzero(in_robot[np.asarray(model.dof_bodyid)])
    if dofs.size == 0:
        raise ValueError(f"Robot base body has no DOFs: {robot_base_name}")
    return dofs.astype(np.int32, copy=False)


def _mujoco_partial_dof_indices(
    model: mujoco.MjModel,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    in_robot = _mujoco_robot_body_mask(model, robot_base_name)
    in_rebase = _mujoco_robot_body_mask(model, rebase_body_name)
    rebase_id = _mujoco_body_id(model, rebase_body_name)
    if not in_robot[rebase_id]:
        raise ValueError(
            f"Rebase body {rebase_body_name} is outside robot base {robot_base_name}"
        )
    dof_bodies = np.asarray(model.dof_bodyid)
    robot_dofs = np.flatnonzero(in_robot[dof_bodies])
    partial_dofs = np.flatnonzero(in_rebase[dof_bodies] & in_robot[dof_bodies])
    if partial_dofs.size == 0:
        raise ValueError(
            f"Rebase body {rebase_body_name} has no descendant joint DOFs"
        )
    return np.searchsorted(robot_dofs, partial_dofs), partial_dofs


def _mujoco_partial_joint_ids(
    model: mujoco.MjModel,
    robot_base_name: str,
    rebase_body_name: str,
) -> np.ndarray:
    in_robot = _mujoco_robot_body_mask(model, robot_base_name)
    in_rebase = _mujoco_robot_body_mask(model, rebase_body_name)
    rebase_id = _mujoco_body_id(model, rebase_body_name)
    if not in_robot[rebase_id]:
        raise ValueError(
            f"Rebase body {rebase_body_name} is outside robot base {robot_base_name}"
        )
    joint_bodies = np.asarray(model.jnt_bodyid)
    return np.flatnonzero(in_rebase[joint_bodies] & in_robot[joint_bodies])


def _mujoco_robot_actuator_indices(
    model: mujoco.MjModel, robot_base_name: str
) -> np.ndarray:
    """Return supported joint-transmission actuators within the robot tree."""
    in_robot = _mujoco_robot_body_mask(model, robot_base_name)
    trn_ids = np.asarray(model.actuator_trnid)
    joint_bodies = np.asarray(model.jnt_bodyid)
    site_bodies = np.asarray(model.site_bodyid)
    geom_bodies = np.asarray(model.geom_bodyid)
    selected: list[int] = []
    for actuator_id, trn_type in enumerate(np.asarray(model.actuator_trntype)):
        trn_type = int(trn_type)
        target_id, reference_id = map(int, trn_ids[actuator_id])
        if trn_type in (
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        ) and target_id >= 0:
            belongs_to_robot = bool(in_robot[joint_bodies[target_id]])
        elif trn_type in (
            int(mujoco.mjtTrn.mjTRN_SITE),
            int(mujoco.mjtTrn.mjTRN_SLIDERCRANK),
        ):
            belongs_to_robot = (
                target_id >= 0
                and bool(in_robot[site_bodies[target_id]])
            ) or (
                reference_id >= 0
                and bool(in_robot[site_bodies[reference_id]])
            )
        elif trn_type == int(mujoco.mjtTrn.mjTRN_BODY):
            belongs_to_robot = target_id >= 0 and bool(in_robot[target_id])
        elif trn_type == int(mujoco.mjtTrn.mjTRN_TENDON):
            first = int(model.tendon_adr[target_id])
            count = int(model.tendon_num[target_id])
            belongs_to_robot = False
            for wrap_id in range(first, first + count):
                wrap_type = int(model.wrap_type[wrap_id])
                object_id = int(model.wrap_objid[wrap_id])
                if wrap_type == int(mujoco.mjtWrap.mjWRAP_JOINT):
                    body_id = int(joint_bodies[object_id])
                elif wrap_type == int(mujoco.mjtWrap.mjWRAP_SITE):
                    body_id = int(site_bodies[object_id])
                elif wrap_type in (
                    int(mujoco.mjtWrap.mjWRAP_SPHERE),
                    int(mujoco.mjtWrap.mjWRAP_CYLINDER),
                ):
                    body_id = int(geom_bodies[object_id])
                else:
                    continue
                belongs_to_robot |= bool(in_robot[body_id])
        else:
            raise ValueError(f"Unsupported actuator transmission: {trn_type}")

        if belongs_to_robot:
            if trn_type not in (
                int(mujoco.mjtTrn.mjTRN_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            ):
                raise ValueError(
                    f"Robot actuator {actuator_id} has unsupported transmission"
                )
            selected.append(actuator_id)
    if not selected:
        raise ValueError(f"Robot has no supported actuators: {robot_base_name}")
    return np.asarray(selected, dtype=np.int32)


def get_mujoco_cartesian_body_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_body: str,
    robot_base_name: str,
) -> np.ndarray:
    """Return world origin Jacobian, shaped ``(6, robot_nv)``."""
    body_id = _mujoco_body_id(model, target_body)
    jacp = np.zeros((3, model.nv), dtype=data.qvel.dtype)
    jacr = np.zeros((3, model.nv), dtype=data.qvel.dtype)
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    return np.concatenate((jacp[:, robot_dofs], jacr[:, robot_dofs]), axis=0)


def get_mujoco_cartesian_body_jacobian_derivative(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_body: str,
    robot_base_name: str,
) -> np.ndarray:
    """Return the forwarded world-frame body-Jacobian derivative."""
    body_id = _mujoco_body_id(model, target_body)
    jacp_dot = np.zeros((3, model.nv), dtype=data.qvel.dtype)
    jacr_dot = np.zeros((3, model.nv), dtype=data.qvel.dtype)
    mujoco.mj_jacDot(
        model, data, jacp_dot, jacr_dot, data.xpos[body_id], body_id
    )
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    return np.concatenate(
        (jacp_dot[:, robot_dofs], jacr_dot[:, robot_dofs]), axis=0
    )


def get_mujoco_generalised_inertia(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_base_name: str,
) -> np.ndarray:
    full_inertia = np.empty((model.nv, model.nv), dtype=data.qvel.dtype)
    mujoco.mj_fullM(model, data, full_inertia)
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    return full_inertia[np.ix_(robot_dofs, robot_dofs)]


def get_mujoco_inverse_generalised_inertia(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_base_name: str,
) -> np.ndarray:
    return np.linalg.inv(
        get_mujoco_generalised_inertia(model, data, robot_base_name)
    )


def get_mujoco_generalised_bias_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_base_name: str,
) -> np.ndarray:
    return data.qfrc_bias[_mujoco_robot_dof_indices(model, robot_base_name)]


def get_mujoco_generalised_forces_from_task_space_config_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_base_name: str,
    pos_config_error: np.ndarray,
    vel_config_error: np.ndarray,
    kd: np.ndarray | float,
    kp: np.ndarray | float,
    task_space_jacobian: np.ndarray,
    task_space_jacobian_derivative: np.ndarray,
) -> np.ndarray:
    """CPU NumPy equivalent of the MJX task-space PD force helper."""
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    robot_nv = robot_dofs.size
    task_nv = pos_config_error.shape[0]
    if pos_config_error.ndim != 1 or vel_config_error.shape != (task_nv,):
        raise ValueError("Task configuration errors must be matching vectors")
    if task_space_jacobian.shape != (task_nv, robot_nv):
        raise ValueError("task_space_jacobian has an invalid shape")
    if task_space_jacobian_derivative.shape != (task_nv, robot_nv):
        raise ValueError("task_space_jacobian_derivative has an invalid shape")
    kp = np.asarray(kp)
    kd = np.asarray(kd)
    if kp.shape not in ((), (task_nv,)) or kd.shape not in ((), (task_nv,)):
        raise ValueError("kp and kd must be scalars or one value per task DOF")
    inverse_inertia = get_mujoco_inverse_generalised_inertia(
        model, data, robot_base_name
    )
    task_inertia = np.linalg.pinv(
        task_space_jacobian @ inverse_inertia @ task_space_jacobian.T
    )
    bias_minus_passive = (
        get_mujoco_generalised_bias_forces(model, data, robot_base_name)
        - data.qfrc_passive[robot_dofs]
    )
    task_force = task_inertia @ (
        kp * pos_config_error
        + kd * vel_config_error
        - task_space_jacobian_derivative @ data.qvel[robot_dofs]
        + task_space_jacobian @ inverse_inertia @ bias_minus_passive
    )
    return task_space_jacobian.T @ task_force


def get_mujoco_actuator_control_input_from_generalised_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau: np.ndarray,
    robot_base_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map robot generalized forces to robot actuator controls with NumPy."""
    actuator_ids = _mujoco_robot_actuator_indices(model, robot_base_name)
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    moments = data.actuator_moment[actuator_ids][:, robot_dofs]
    if tau.shape != (robot_dofs.size,):
        raise ValueError(f"tau must have shape {(robot_dofs.size,)}, got {tau.shape}")

    if hasattr(model, "actuator_ctrlnum") and np.any(
        np.asarray(model.actuator_ctrlnum)[actuator_ids] != 1
    ):
        raise ValueError("Only single-input, single-output actuators are supported")
    if hasattr(model, "actuator_outnum") and np.any(
        np.asarray(model.actuator_outnum)[actuator_ids] != 1
    ):
        raise ValueError("Only single-input, single-output actuators are supported")
    if np.any(
        np.asarray(model.actuator_dyntype)[actuator_ids]
        != int(mujoco.mjtDyn.mjDYN_NONE)
    ):
        raise ValueError("Robot actuators with activation dynamics are unsupported")
    gain_types = np.asarray(model.actuator_gaintype)[actuator_ids]
    bias_types = np.asarray(model.actuator_biastype)[actuator_ids]
    if np.any(
        ~np.isin(
            gain_types,
            (int(mujoco.mjtGain.mjGAIN_FIXED), int(mujoco.mjtGain.mjGAIN_AFFINE)),
        )
    ) or np.any(
        ~np.isin(
            bias_types,
            (int(mujoco.mjtBias.mjBIAS_NONE), int(mujoco.mjtBias.mjBIAS_AFFINE)),
        )
    ):
        raise ValueError("Robot actuator gain or bias type is unsupported")

    actuator_forces = np.linalg.pinv(moments.T) @ tau
    force_error = np.linalg.norm(moments.T @ actuator_forces - tau)
    force_valid = force_error <= 1e-4 * (1.0 + np.linalg.norm(tau))
    length = data.actuator_length[actuator_ids]
    velocity = data.actuator_velocity[actuator_ids]
    gain_params = model.actuator_gainprm[actuator_ids]
    bias_params = model.actuator_biasprm[actuator_ids]
    gain = np.where(
        gain_types == int(mujoco.mjtGain.mjGAIN_FIXED),
        gain_params[:, 0],
        gain_params[:, 0] + gain_params[:, 1] * length + gain_params[:, 2] * velocity,
    )
    bias = np.where(
        bias_types == int(mujoco.mjtBias.mjBIAS_NONE),
        0.0,
        bias_params[:, 0] + bias_params[:, 1] * length + bias_params[:, 2] * velocity,
    )
    gain_valid = np.all(np.abs(gain) > 1e-8)
    controls = (actuator_forces - bias) / np.where(
        np.abs(gain) > 1e-8, gain, 1.0
    )
    control_limited = np.asarray(model.actuator_ctrllimited)[actuator_ids]
    force_limited = np.asarray(model.actuator_forcelimited)[actuator_ids]
    control_range = model.actuator_ctrlrange[actuator_ids]
    force_range = model.actuator_forcerange[actuator_ids]
    controls_valid = np.all(
        ~control_limited
        | ((controls >= control_range[:, 0] - 1e-6) & (controls <= control_range[:, 1] + 1e-6))
    )
    limits_valid = np.all(
        ~force_limited
        | ((actuator_forces >= force_range[:, 0] - 1e-6) & (actuator_forces <= force_range[:, 1] + 1e-6))
    )
    valid = np.asarray(
        force_valid
        and gain_valid
        and controls_valid
        and limits_valid
        and np.all(np.isfinite(controls))
        and np.all(np.isfinite(actuator_forces))
    )
    return controls, valid


def get_mujoco_partial_dof_jacobian(
    model: mujoco.MjModel,
    robot_base_name: str,
    rebase_body_name: str,
) -> np.ndarray:
    partial_local_dofs, robot_global_dofs = _mujoco_partial_dof_indices(
        model, robot_base_name, rebase_body_name
    )
    return np.eye(robot_global_dofs.size, dtype=np.float64)[partial_local_dofs]


def get_mujoco_eef_gripper_task_space_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_base_name: str,
    eef_body: str,
    gripper_body: str,
) -> np.ndarray:
    return np.concatenate(
        (
            get_mujoco_cartesian_body_jacobian(
                model, data, eef_body, robot_base_name
            ),
            get_mujoco_partial_dof_jacobian(
                model, robot_base_name, gripper_body
            ),
        ),
        axis=0,
    )


def get_mujoco_eef_gripper_task_space_jacobian_derivative(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_base_name: str,
    eef_body: str,
    gripper_body: str,
) -> np.ndarray:
    eef_derivative = get_mujoco_cartesian_body_jacobian_derivative(
        model, data, eef_body, robot_base_name
    )
    gripper_selection = get_mujoco_partial_dof_jacobian(
        model, robot_base_name, gripper_body
    )
    return np.concatenate(
        (eef_derivative, np.zeros_like(gripper_selection)), axis=0
    )


def get_mujoco_actuator_control_input_from_eef_gripper_pos_commands(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_base_name: str,
    eef_body: str,
    gripper_body: str,
    eef_pose_delta: np.ndarray,
    gripper_qpos_delta: np.ndarray,
    kd: np.ndarray | float,
    kp: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU equivalent of the equal-priority EEF/gripper OSC adapter."""
    if eef_pose_delta.shape != (6,):
        raise ValueError("eef_pose_delta must have shape (6,)")
    selection = get_mujoco_partial_dof_jacobian(
        model, robot_base_name, gripper_body
    )
    if gripper_qpos_delta.shape != (selection.shape[0],):
        raise ValueError(
            "gripper_qpos_delta must have one entry per gripper descendant DOF"
        )
    jacobian = get_mujoco_eef_gripper_task_space_jacobian(
        model, data, robot_base_name, eef_body, gripper_body
    )
    jacobian_derivative = get_mujoco_eef_gripper_task_space_jacobian_derivative(
        model, data, robot_base_name, eef_body, gripper_body
    )
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    tau = get_mujoco_generalised_forces_from_task_space_config_error(
        model,
        data,
        robot_base_name,
        np.concatenate((eef_pose_delta, gripper_qpos_delta)),
        -jacobian @ data.qvel[robot_dofs],
        kd,
        kp,
        jacobian,
        jacobian_derivative,
    )
    return get_mujoco_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )


def get_mujoco_actuator_control_input_from_partial_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    partial_qpos: np.ndarray,
    is_delta: bool,
    kd: np.ndarray | float,
    kp: np.ndarray | float,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    selection = get_mujoco_partial_dof_jacobian(
        model, robot_base_name, rebase_body_name
    )
    _, partial_dofs = _mujoco_partial_dof_indices(
        model, robot_base_name, rebase_body_name
    )
    if is_delta:
        if partial_qpos.shape != (selection.shape[0],):
            raise ValueError("partial qpos delta has an invalid shape")
        pos_error = partial_qpos
    else:
        joint_ids = _mujoco_partial_joint_ids(
            model, robot_base_name, rebase_body_name
        )
        target_qpos = data.qpos.copy()
        target_offset = 0
        for joint_id in joint_ids:
            width = 7 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else (
                4 if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL else 1
            )
            if target_offset + width > partial_qpos.size:
                raise ValueError("absolute partial qpos has an invalid shape")
            address = int(model.jnt_qposadr[joint_id])
            target_qpos[address : address + width] = partial_qpos[
                target_offset : target_offset + width
            ]
            target_offset += width
        if target_offset != partial_qpos.size:
            raise ValueError("absolute partial qpos has an invalid shape")
        full_error = np.empty((model.nv,), dtype=data.qvel.dtype)
        mujoco.mj_differentiatePos(model, full_error, 1.0, data.qpos, target_qpos)
        pos_error = full_error[partial_dofs]
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    tau = get_mujoco_generalised_forces_from_task_space_config_error(
        model,
        data,
        robot_base_name,
        pos_error,
        -selection @ data.qvel[robot_dofs],
        kd,
        kp,
        selection,
        np.zeros_like(selection),
    )
    return get_mujoco_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )


def get_mujoco_actuator_control_input_from_partial_qvel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    partial_qvel: np.ndarray,
    is_delta: bool,
    kd: np.ndarray | float,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    selection = get_mujoco_partial_dof_jacobian(
        model, robot_base_name, rebase_body_name
    )
    if partial_qvel.shape != (selection.shape[0],):
        raise ValueError("partial qvel has an invalid shape")
    robot_dofs = _mujoco_robot_dof_indices(model, robot_base_name)
    current_velocity = selection @ data.qvel[robot_dofs]
    velocity_error = partial_qvel if is_delta else partial_qvel - current_velocity
    tau = get_mujoco_generalised_forces_from_task_space_config_error(
        model,
        data,
        robot_base_name,
        np.zeros((selection.shape[0],), dtype=data.qvel.dtype),
        velocity_error,
        kd,
        0.0,
        selection,
        np.zeros_like(selection),
    )
    return get_mujoco_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )


def get_mujoco_actuator_control_input_from_partial_tau(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    partial_tau: np.ndarray,
    robot_base_name: str,
    rebase_body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    selection = get_mujoco_partial_dof_jacobian(
        model, robot_base_name, rebase_body_name
    )
    if partial_tau.shape != (selection.shape[0],):
        raise ValueError("partial tau has an invalid shape")
    return get_mujoco_actuator_control_input_from_generalised_forces(
        model, data, selection.T @ partial_tau, robot_base_name
    )
