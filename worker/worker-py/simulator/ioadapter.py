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


def get_jacobian(
    model: mjx.Model,
    data: mjx.Data,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    """Return the world-frame body-origin Jacobian, shape (6, robot_nv)."""
    body_id = _body_id(model, eef_name)
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    # MJX returns (nv, 3) linear and angular blocks. Keep linear rows first
    # throughout the task-space helpers; retain the actual gripper columns.
    jacp, jacr = mjx.jac(model, data, data.xpos[body_id], jnp.asarray(body_id))
    return jnp.concatenate((jacp[robot_dofs].T, jacr[robot_dofs].T), axis=0)


def get_jacobian_transpose(
    model: mjx.Model,
    data: mjx.Data,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    return get_jacobian(model, data, eef_name, robot_base_name).T


def get_jacobian_derivative(
    model: mjx.Model,
    data: mjx.Data,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    """Return the world-frame time derivative of the body Jacobian."""
    body_id = _body_id(model, eef_name)
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


def get_generalised_constraint_forces(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
) -> jnp.ndarray:
    """Read robot constraint forces; the OSC force law does not compensate them."""
    return data.qfrc_constraint[_robot_dof_indices(model, robot_base_name)]


def get_task_space_inverse_inertia(
    model: mjx.Model,
    data: mjx.Data,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    jacobian = get_jacobian(model, data, eef_name, robot_base_name)
    inverse_inertia = get_inverse_generalised_inertia(model, data, robot_base_name)
    return jacobian @ inverse_inertia @ jacobian.T


def get_task_space_inertia(
    model: mjx.Model,
    data: mjx.Data,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    # The pseudoinverse also gives a finite result at singular task poses;
    # exact acceleration tracking requires a full-rank task Jacobian.
    return jnp.linalg.pinv(
        get_task_space_inverse_inertia(model, data, eef_name, robot_base_name)
    )


def compute_task_space_force(
    model: mjx.Model,
    data: mjx.Data,
    desired_acceleration: jnp.ndarray,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    """Compute equation (24)'s nominal world-frame task-space wrench."""
    if desired_acceleration.shape != (6,):
        raise ValueError("desired_acceleration must have shape (6,)")

    robot_dofs = _robot_dof_indices(model, robot_base_name)
    jacobian = get_jacobian(model, data, eef_name, robot_base_name)
    jacobian_dot = get_jacobian_derivative(model, data, eef_name, robot_base_name)
    inverse_inertia = get_inverse_generalised_inertia(model, data, robot_base_name)
    # This is the same calculation as get_task_space_inertia, using the J and
    # H inverse already needed below rather than computing them a second time.
    task_inertia = jnp.linalg.pinv(jacobian @ inverse_inertia @ jacobian.T)
    bias = get_generalised_bias_forces(model, data, robot_base_name)
    passive = data.qfrc_passive[robot_dofs]

    # Desired feed-forward acceleration is supplied by the caller. This law
    # compensates current-state passive forces, which MuJoCo stores separately
    # from bias forces. Constraint forces are omitted; null-space force is zero.
    # H*qacc + C = J.T*F + P, hence the compensation term uses C - P.
    return task_inertia @ (
        desired_acceleration
        - jacobian_dot @ data.qvel[robot_dofs]
        + jacobian @ inverse_inertia @ (bias - passive)
    )


def get_generalised_forces_from_task_space_force(
    model: mjx.Model,
    data: mjx.Data,
    task_space_force: jnp.ndarray,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    if task_space_force.shape != (6,):
        raise ValueError("task_space_force must have shape (6,)")
    return get_jacobian_transpose(
        model, data, eef_name, robot_base_name
    ) @ task_space_force


def get_generalised_forces_from_eef_deltas(
    model: mjx.Model,
    data: mjx.Data,
    xpos_delta: jnp.ndarray,
    xquat_delta: jnp.ndarray,
    kd: jnp.ndarray | float,
    ke: jnp.ndarray | float,
    eef_name: str,
    robot_base_name: str,
) -> jnp.ndarray:
    """Map scaled world-frame pose deltas to robot generalized forces."""
    if xpos_delta.shape != (3,) or xquat_delta.shape != (4,):
        raise ValueError("xpos_delta and xquat_delta must have shapes (3,) and (4,)")
    ke = jnp.asarray(ke)
    kd = jnp.asarray(kd)
    if ke.shape not in ((), (6,)) or kd.shape not in ((), (6,)):
        raise ValueError("ke and kd must be scalars or have shape (6,)")

    body_id = _body_id(model, eef_name)
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    base_id = _body_id(model, robot_base_name)
    ancestor = body_id
    while ancestor != base_id and ancestor != 0:
        ancestor = int(model.body_parentid[ancestor])
    if ancestor != base_id:
        raise ValueError("End-effector body is not part of the robot")

    current_pos = data.xpos[body_id]
    current_quat = data.xquat[body_id]
    # Deltas have already been unnormalized/scaled. The quaternion is a unit
    # relative rotation in world coordinates, so target = delta * current.
    target_pos = current_pos + xpos_delta
    target_quat = mjx_math.quat_mul(xquat_delta, current_quat)
    rotation_error = mjx_math.quat_mul(target_quat, mjx_math.quat_inv(current_quat))
    axis, angle = mjx_math.quat_to_axis_angle(rotation_error)
    pose_error = jnp.concatenate((target_pos - current_pos, axis * angle))

    # Desired velocity and acceleration are both zero. ke is Kp, kd is Kd;
    # each gain is a scalar or a six-element vector [linear, angular].
    jacobian = get_jacobian(model, data, eef_name, robot_base_name)
    current_velocity = jacobian @ data.qvel[robot_dofs]
    desired_acceleration = ke * pose_error - kd * current_velocity

    task_space_force = compute_task_space_force(
        model, data, desired_acceleration, eef_name, robot_base_name
    )
    # This returns generalized forces for the robot DOFs, including gripper
    # joints; mapping to actuator control inputs is intentionally deferred.
    return get_generalised_forces_from_task_space_force(
        model, data, task_space_force, eef_name, robot_base_name
    )


def get_robot_actuator_moment_arms(
    model: mjx.Model,
    data: mjx.Data,
    robot_base_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return robot actuator moments and actuator IDs in model order.

    Rows correspond to the returned actuator IDs; columns correspond to the
    robot DOFs returned by ``_robot_dof_indices``. The state must already have
    passed through MJX forward dynamics so its transmission data is current.
    """
    actuator_ids = _robot_actuator_indices(model, robot_base_name)
    robot_dofs = _robot_dof_indices(model, robot_base_name)

    if data.impl.name == "JAX":
        moments = data._impl.actuator_moment[actuator_ids]
    elif data.impl.name == "WARP":
        # Warp stores each moment row sparsely. Supported joint transmissions
        # have at most six entries, including the free-joint transmissions.
        if model.nJmom == 0:
            raise ValueError("Robot actuators have no transmission moments")
        offsets = jnp.arange(6, dtype=jnp.int32)
        row_start = data._impl.moment_rowadr[actuator_ids]
        row_count = data._impl.moment_rownnz[actuator_ids]
        addresses = jnp.clip(
            row_start[:, None] + offsets[None, :], 0, model.nJmom - 1
        )
        valid = offsets[None, :] < row_count[:, None]
        columns = jnp.where(valid, data._impl.moment_colind[addresses], 0)
        values = jnp.where(valid, data._impl.actuator_moment[addresses], 0)
        moments = jnp.zeros((actuator_ids.size, model.nv), dtype=data.qvel.dtype)
        moments = moments.at[
            jnp.arange(actuator_ids.size)[:, None], columns
        ].add(values)
    else:
        raise ValueError(f"Unsupported MJX implementation: {data.impl}")

    return moments[:, robot_dofs], jnp.asarray(actuator_ids, dtype=jnp.int32)


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
    moments, actuator_ids = get_robot_actuator_moment_arms(
        model, data, robot_base_name
    )
    if tau.shape != (moments.shape[1],):
        raise ValueError(
            "tau must have one entry per robot DOF: "
            f"expected {(moments.shape[1],)}, got {tau.shape}"
        )

    actuator_ids_static = _robot_actuator_indices(model, robot_base_name)
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


def get_actuator_control_input_from_eef_deltas(
    model: mjx.Model,
    data: mjx.Data,
    xpos_delta: jnp.ndarray,
    xquat_delta: jnp.ndarray,
    kd: jnp.ndarray | float,
    ke: jnp.ndarray | float,
    eef_name: str,
    robot_base_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return robot actuator controls and their validity for EEF pose deltas."""
    tau = get_generalised_forces_from_eef_deltas(
        model,
        data,
        xpos_delta,
        xquat_delta,
        kd,
        ke,
        eef_name,
        robot_base_name,
    )
    return get_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )


def get_actuator_control_input_from_qpos(
    model: mjx.Model,
    data: mjx.Data,
    qpos: jnp.ndarray,
    is_delta: bool,
    kd: jnp.ndarray | float,
    kp: jnp.ndarray | float,
    robot_base_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map a robot position target or tangent-space delta to controls.

    Absolute targets have one entry per robot qpos coordinate and contain
    unit quaternions. Deltas have one entry per robot DOF, so ball/free-joint
    rotations use three tangent coordinates rather than four quaternion
    components. The current robot velocity is the zero-velocity target for
    PD control. ``is_delta`` is static at JAX trace time because the input
    lengths can differ when quaternion joints are present.
    """
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    robot_nv = robot_dofs.shape[0]
    kd = jnp.asarray(kd)
    kp = jnp.asarray(kp)
    if kd.shape not in ((), (robot_nv,)) or kp.shape not in ((), (robot_nv,)):
        raise ValueError("kp and kd must be scalars or have one value per robot DOF")

    if is_delta:
        if qpos.shape != (robot_nv,):
            raise ValueError(
                f"qpos delta must have shape {(robot_nv,)}, got {qpos.shape}"
            )
        position_error = qpos
    else:
        in_robot = _robot_body_mask(model, robot_base_name)
        joint_ids = np.flatnonzero(in_robot[np.asarray(model.jnt_bodyid)])
        joint_types = np.asarray(model.jnt_type)
        qpos_addresses = np.asarray(model.jnt_qposadr)
        joint_widths = [
            7 if joint_types[joint_id] == int(mujoco.mjtJoint.mjJNT_FREE)
            else 4 if joint_types[joint_id] == int(mujoco.mjtJoint.mjJNT_BALL)
            else 1
            for joint_id in joint_ids
        ]
        robot_nq = sum(joint_widths)
        if qpos.shape != (robot_nq,):
            raise ValueError(
                f"absolute qpos must have shape {(robot_nq,)}, got {qpos.shape}"
            )

        errors = []
        target_address = 0
        for joint_id, width in zip(joint_ids, joint_widths):
            current_address = int(qpos_addresses[joint_id])
            target = qpos[target_address : target_address + width]
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
        position_error = jnp.concatenate(errors)

    desired_acceleration = kp * position_error - kd * data.qvel[robot_dofs]
    # Match the EEF helper's reduced robot dynamics: hold environment DOF
    # accelerations fixed and omit current constraint forces.
    tau = (
        get_generalised_inertia(model, data, robot_base_name)
        @ desired_acceleration
        + get_generalised_bias_forces(model, data, robot_base_name)
        - data.qfrc_passive[robot_dofs]
    )
    return get_actuator_control_input_from_tau(
        model, data, tau, robot_base_name
    )


def get_actuator_control_input_from_qvel(
    model: mjx.Model,
    data: mjx.Data,
    qvel: jnp.ndarray,
    is_delta: bool,
    kd: jnp.ndarray | float,
    robot_base_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map an absolute robot velocity target or velocity delta to controls."""
    robot_dofs = _robot_dof_indices(model, robot_base_name)
    robot_nv = robot_dofs.shape[0]
    if qvel.shape != (robot_nv,):
        raise ValueError(f"qvel must have shape {(robot_nv,)}, got {qvel.shape}")
    kd = jnp.asarray(kd)
    if kd.shape not in ((), (robot_nv,)):
        raise ValueError("kd must be a scalar or have one value per robot DOF")

    velocity_error = qvel if is_delta else qvel - data.qvel[robot_dofs]
    desired_acceleration = kd * velocity_error
    # Match the EEF helper's reduced robot dynamics and passive-force term.
    tau = (
        get_generalised_inertia(model, data, robot_base_name)
        @ desired_acceleration
        + get_generalised_bias_forces(model, data, robot_base_name)
        - data.qfrc_passive[robot_dofs]
    )
    return get_actuator_control_input_from_tau(
        model, data, tau, robot_base_name
    )


def get_actuator_control_input_from_tau(
    model: mjx.Model,
    data: mjx.Data,
    tau: jnp.ndarray,
    robot_base_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map robot DOF forces to robot actuator controls and a validity flag."""
    return get_actuator_control_input_from_generalised_forces(
        model, data, tau, robot_base_name
    )
