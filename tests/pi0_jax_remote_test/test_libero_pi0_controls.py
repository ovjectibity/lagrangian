#!/usr/bin/env python3
"""Trace LIBERO observations -> Pi-0 -> actuator mapping -> MJX dynamics.

Run with run_remote_test.sh test_libero_pi0_controls.py <remote-ip>.
Direct invocation supports --help, --steps, --task, and --apply-invalid-controls.
LIBERO_DATASET, LIBERO_TASK_INDEX, NUM_TIMESTEPS, and PI0_CHECKPOINT_DIR are
read inside the worker container. The task prompt defaults to the empty string,
matching the current testing workflow. Actions are passed through unscaled.
At step zero, additional probes isolate each action coordinate at the same
state, inspect the controller's task inertia, and compare command magnitudes.
These probes do not change the controls or state used by the rollout.

Invalid mappings retain previous controls by default, as in TestingWorkflow.
--apply-invalid-controls instead feeds finite mapped controls into dynamics;
MuJoCo's actuator limits still apply. Either mode exits nonzero after completing
the requested steps if any mapping was invalid. Non-finite values fail immediately.
"""

import argparse
from dataclasses import replace
import logging
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from commondef.records.workload_graph import (
    PolicyInputSpecStandardInputSpec,
    PolicyOutputSpecEEFOutputSpec,
)
from datasets.libero.libero import get_task_mjcfs
from models.pi0.pi0 import Pi0Model
from models.pi0.pi0_jax import Pi0Jax
from simulator.ioadapter import (
    _robot_dof_indices,
    get_eef_gripper_task_space_jacobian,
    get_generalised_inertia,
    get_actuator_control_input_from_eef_gripper_pos_commands,
    get_actuator_control_input_from_generalised_forces,
    get_mujoco_actuator_control_input_from_generalised_forces,
)
from simulator.simulator import MjxSimulator


logger = logging.getLogger(__name__)


def check_control_clipping():
    """Exercise bounds, signed gains, bias, and invalid solves without Pi-0."""
    mjcf = '''
        <mujoco><size njmax="128"/><worldbody><body name="robot">
          <joint name="joint"/><joint name="passive" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body></worldbody><actuator>
          <general joint="joint" gainprm="1" biastype="affine"
            ctrllimited="true" ctrlrange="-10 10"
            forcelimited="true" forcerange="-5 5"/>
        </actuator></mujoco>
    '''
    model = mujoco.MjModel.from_xml_string(mjcf)
    # label, gain, bias, control limits, force limits, tau, expected control/valid
    cases = [
        ("within_bounds", 2, 1, (-10, 10), (-5, 5), (3, 0), 1, True),
        ("positive_gain", 2, 1, (-10, 10), (-5, 5), (20, 0), 2, True),
        ("negative_gain_upper_force", -2, 1, (-10, 10), (-5, 5), (20, 0), -2, True),
        ("negative_gain_lower_force", -2, 1, (-10, 10), (-5, 5), (-20, 0), 3, True),
        ("control_tighter", 2, 1, (0, 1), (-100, 100), (20, 0), 1, True),
        ("position_actuator_bias", 1000, -10, (0, 0.04), (-20, 20), (100, 0), 0.03, True),
        ("panda_force_boundary", 1000, 0, (0, 0.04), (-20, 20), (26, 0), 0.02, True),
        ("force_only", 2, 1, None, (-5, 5), (20, 0), 2, True),
        ("control_only", 2, 1, (0, 1), None, (20, 0), 1, True),
        ("unlimited", 2, 1, None, None, (20, 0), 9.5, True),
        ("incompatible_bounds", 1, 0, (0, 1), (5, 6), (3, 0), None, False),
        ("zero_gain", 0, 0, (-10, 10), (-5, 5), (3, 0), None, False),
        ("unrepresentable_force", 1, 0, (-10, 10), (-5, 5), (0, 1), None, False),
        ("nonfinite_request", 1, 0, (-10, 10), (-5, 5), (np.inf, 0), None, False),
    ]
    for label, gain, bias, control_range, force_range, tau, expected, expected_valid in cases:
        model.actuator_gainprm[0, 0] = gain
        model.actuator_biasprm[0, 0] = bias
        model.actuator_ctrllimited[0] = control_range is not None
        model.actuator_forcelimited[0] = force_range is not None
        model.actuator_ctrlrange[0] = control_range or (-10, 10)
        model.actuator_forcerange[0] = force_range or (-5, 5)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # The CPU helper currently expects dense moments; newer MuJoCo stores
        # them sparsely. Adapt only this test fixture, without changing that API.
        moments = np.asarray(data.actuator_moment)
        if moments.ndim == 1:
            dense = np.zeros((model.nu, model.nv))
            for row in range(model.nu):
                start = data.moment_rowadr[row]
                end = start + data.moment_rownnz[row]
                dense[row, data.moment_colind[start:end]] = moments[start:end]
            moments = dense
        cpu_data = SimpleNamespace(
            actuator_moment=moments, actuator_length=data.actuator_length,
            actuator_velocity=data.actuator_velocity,
        )
        mjx_model = mjx.put_model(model, impl="jax")
        mjx_data = mjx.put_data(model, data, impl="jax")
        with np.errstate(invalid="ignore"):
            cpu_result = get_mujoco_actuator_control_input_from_generalised_forces(
                model, cpu_data, np.asarray(tau), "robot"
            )
        gpu_result = jax.jit(lambda state, force:
            get_actuator_control_input_from_generalised_forces(mjx_model, state, force, "robot")
        )(mjx_data, jnp.asarray(tau, dtype=jnp.float32))
        for backend, (controls, valid) in (("NumPy", cpu_result), ("MJX/JIT", gpu_result)):
            controls, valid = np.asarray(controls), bool(np.asarray(valid))
            if valid != expected_valid:
                raise AssertionError(f"{label}/{backend}: valid={valid}, expected {expected_valid}")
            if expected_valid:
                np.testing.assert_allclose(controls, [expected], rtol=1e-5, atol=1e-6)
                force = gain * controls[0] + bias
                if control_range is not None:
                    assert control_range[0] - 1e-6 <= controls[0] <= control_range[1] + 1e-6
                if force_range is not None:
                    assert force_range[0] - 1e-5 <= force <= force_range[1] + 1e-5
            logger.info("CLIPPING CHECK %s/%s: controls=%s valid=%s PASS", label, backend, controls.tolist(), valid)
    logger.info("PASS: %d clipping cases on NumPy and MJX/JIT", len(cases))
    for nworld in (None, 1):
        data, _, _, cpu_model = MjxSimulator.init(
            mjcf=mjcf, nworld=nworld, naconmax=1, naccdmax=1,
        )
        jax.block_until_ready(data)
        if data._impl.njmax != cpu_model.njmax:
            raise AssertionError(
                f"Constraint capacity mismatch: allocated={data._impl.njmax}, "
                f"configured={cpu_model.njmax}; sync simulator.py to the worker"
            )
        logger.info("PASS: configured njmax=%d honored with nworld=%s", data._impl.njmax, nworld)


def log_array(label, value, *, values=True):
    array = np.asarray(jax.block_until_ready(value))
    finite = np.isfinite(array)
    logger.info(
        "%s: shape=%s dtype=%s finite=%s min=%s max=%s mean=%s",
        label, array.shape, array.dtype, bool(finite.all()),
        array.min() if array.size else None,
        array.max() if array.size else None,
        array.mean() if array.size else None,
    )
    if values:
        logger.info("%s values=%s", label, array.tolist())
    if not finite.all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def log_state(label, data, eef_id):
    logger.info("%s: simulation_time=%s", label, np.asarray(data.time).tolist())
    for field in ("qpos", "qvel", "qacc", "ctrl", "qfrc_bias", "qfrc_passive"):
        log_array(f"{label}.{field}", getattr(data, field))
    log_array(f"{label}.eef_xpos", data.xpos[:, eef_id])
    log_array(f"{label}.eef_xquat", data.xquat[:, eef_id])


def log_actuator_limits(cpu_model, data, controls):
    """Report requested force before MuJoCo applies control/force saturation."""
    length = np.asarray(data.actuator_length)[0]
    velocity = np.asarray(data._impl.actuator_velocity)[0]
    gains = cpu_model.actuator_gainprm
    biases = cpu_model.actuator_biasprm
    gain = np.where(
        cpu_model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED,
        gains[:, 0], gains[:, 0] + gains[:, 1] * length + gains[:, 2] * velocity,
    )
    bias = np.where(
        cpu_model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE,
        0.0, biases[:, 0] + biases[:, 1] * length + biases[:, 2] * velocity,
    )
    requested_force = gain * controls[0] + bias
    limits_ok = True
    for index in range(cpu_model.nu):
        name = mujoco.mj_id2name(cpu_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        control_low, control_high = cpu_model.actuator_ctrlrange[index]
        force_low, force_high = cpu_model.actuator_forcerange[index]
        control_limited = bool(cpu_model.actuator_ctrllimited[index])
        force_limited = bool(cpu_model.actuator_forcelimited[index])
        # Mapping uses float32; recomputing force here with CPU float64 model
        # parameters can differ by a few ULPs at a saturated boundary.
        control_tolerance = 1e-6 * (1 + max(abs(control_low), abs(control_high)))
        force_tolerance = 1e-6 * (1 + max(abs(force_low), abs(force_high)))
        control_ok = not control_limited or (
            control_low - control_tolerance <= controls[0, index] <= control_high + control_tolerance
        )
        force_ok = not force_limited or (
            force_low - force_tolerance <= requested_force[index] <= force_high + force_tolerance
        )
        lower, upper = (control_low, control_high) if control_limited else (-np.inf, np.inf)
        if force_limited and abs(gain[index]) > 1e-8:
            force_bounds = sorted(((force_low - bias[index]) / gain[index], (force_high - bias[index]) / gain[index]))
            lower, upper = max(lower, force_bounds[0]), min(upper, force_bounds[1])
        limits_ok = limits_ok and control_ok and force_ok
        logger.info(
            "actuator[%d]=%s control=%g range=[%g,%g] limited=%s ok=%s; "
            "gain=%g bias=%g requested_force=%g range=[%g,%g] limited=%s ok=%s",
            index, name, controls[0, index], control_low, control_high,
            control_limited, control_ok, gain[index], bias[index],
            requested_force[index], force_low, force_high, force_limited, force_ok,
        )
        logger.info("actuator[%d] feasible_control_interval=[%.10g,%.10g] feasible=%s", index, lower, upper, lower <= upper)
    return limits_ok


def explain_mapping(cpu_model, model, data, action, controls, map_controls, map_requested, output_spec):
    """Compare commands at identical state; never advance counterfactual dynamics."""
    base = output_spec.content.base_body_name
    eef = output_spec.content.eef_state.rebase_body
    gripper = output_spec.content.gripper.rebase_body
    logger.info("EXPLAIN: all probes use the same step-zero state and raw policy action")
    logger.info(
        "Controller equations: e=action (rotation converted through quaternion); "
        "a_des=25*e-10*J*qvel; Lambda=pinv(J*inv(M)*J.T); "
        "tau=J.T*Lambda*(a_des-Jdot*qvel+J*inv(M)*(bias-passive)); "
        "actuator_force=pinv(moment.T)*tau; control=(actuator_force-bias)/gain"
    )
    geometry = jax.jit(jax.vmap(lambda state: (
        get_eef_gripper_task_space_jacobian(model, state, base, eef, gripper),
        get_generalised_inertia(model, state, base),
        state.qvel[_robot_dof_indices(model, base)],
    )))
    jacobian, inertia, robot_velocity = jax.block_until_ready(geometry(data))
    jacobian, inertia = np.asarray(jacobian)[0], np.asarray(inertia)[0]
    # Use JAX's pinv here as in the production controller, including its cutoff.
    inverse_task_inertia = jnp.asarray(jacobian) @ jnp.linalg.inv(jnp.asarray(inertia)) @ jnp.asarray(jacobian).T
    task_inertia = np.asarray(jnp.linalg.pinv(inverse_task_inertia))
    singular_values = np.linalg.svd(np.asarray(inverse_task_inertia), compute_uv=False)
    cutoff = 10 * max(inverse_task_inertia.shape) * np.finfo(singular_values.dtype).eps * singular_values[0]
    logger.info(
        "EXPLAIN geometry: task_inverse_inertia_singular_values=%s "
        "pinv_cutoff=%g retained_rank=%d/%d robot_inertia_condition=%g",
        singular_values.tolist(), cutoff, np.count_nonzero(singular_values > cutoff),
        len(singular_values), np.linalg.cond(inertia),
    )
    log_array("EXPLAIN task_inertia_diagonal", np.diag(task_inertia))
    log_array("EXPLAIN proportional_acceleration=25*raw_action", 25 * action)
    log_array("EXPLAIN task_velocity=J*qvel", jacobian @ np.asarray(robot_velocity)[0])

    def probe(label, candidate):
        mapped, valid = jax.block_until_ready(map_controls(data, jnp.asarray(candidate)))
        requested, _ = jax.block_until_ready(map_requested(data, jnp.asarray(candidate)))
        mapped = np.asarray(mapped)
        logger.info(
            "EXPLAIN probe=%s action=%s requested_controls=%s clipped_controls=%s valid=%s",
            label, np.asarray(candidate).tolist(), np.asarray(requested).tolist(), mapped.tolist(), np.asarray(valid).tolist(),
        )
        return np.asarray(requested)

    raw = np.asarray(action)
    baseline = probe("zero_action_bias_and_velocity_baseline", np.zeros_like(raw))
    contributions = []
    labels = ("translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z", "gripper")
    if raw.shape != (1, len(labels)):
        raise ValueError(f"Explanatory probes require one 7D Panda action, got {raw.shape}")
    for index, label in enumerate(labels):
        candidate = np.zeros_like(raw)
        candidate[:, index] = raw[:, index]
        contributions.append(probe(f"only_{label}", candidate) - baseline)
    reconstructed = baseline + np.sum(contributions, axis=0)
    logger.info(
        "EXPLAIN additive reconstruction: max_control_error=%g; "
        "each contribution excludes the zero-action baseline",
        np.max(np.abs(reconstructed - controls)),
    )
    for index in range(cpu_model.nu):
        name = mujoco.mj_id2name(cpu_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        logger.info(
            "EXPLAIN actuator=%s raw_control=%g baseline=%g contributions=%s",
            name, controls[0, index], baseline[0, index],
            {label: float(value[0, index]) for label, value in zip(labels, contributions)},
        )
    for scale in (0.5, 0.25, 0.1, 0.05):
        probe(f"all_coordinates_times_{scale}", raw * scale)
    pose_scaled = raw.copy()
    pose_scaled[:, :3] *= 0.05
    pose_scaled[:, 3:6] *= 0.5
    probe("hypothesis_pose_scaled_gripper_unchanged", pose_scaled)
    pose_scaled[:, 6:] = 0
    probe("hypothesis_pose_scaled_gripper_zero", pose_scaled)
    logger.info("EXPLAIN: pose factors 0.05/0.5 are a controller-convention hypothesis; no gripper conversion is assumed")

    gain_probe = jax.jit(jax.vmap(
        lambda state, command, kp: get_actuator_control_input_from_eef_gripper_pos_commands(
            model, state, base, eef, gripper, command[:6], command[6:], 10.0, kp
        ), in_axes=(0, 0, None),
    ))
    for kp in (0.0, 5.0, 10.0, 25.0):
        mapped, valid = jax.block_until_ready(gain_probe(data, action, kp))
        logger.info("EXPLAIN gain_probe kp=%g kd=10 controls=%s valid=%s", kp, np.asarray(mapped).tolist(), np.asarray(valid).tolist())
    gripper_id = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, gripper)
    if gripper_id >= 0 and cpu_model.body_jntnum[gripper_id] == 1:
        joint_id = int(cpu_model.body_jntadr[gripper_id])
        qpos = float(np.asarray(data.qpos)[0, cpu_model.jnt_qposadr[joint_id]])
        low, high = cpu_model.jnt_range[joint_id]
        logger.info(
            "EXPLAIN gripper: current_qpos=%g requested_delta=%g implied_target=%g "
            "joint_range=[%g,%g] armature=%g commanded_delta/travel=%g",
            qpos, raw[0, 6], qpos + raw[0, 6], low, high,
            cpu_model.dof_armature[cpu_model.jnt_dofadr[joint_id]], raw[0, 6] / (high - low),
        )
    logger.info("EXPLAIN complete; rollout will continue with its original raw-action mapping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.environ.get("LIBERO_DATASET", "libero_spatial"))
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("LIBERO_TASK_INDEX", "0")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("NUM_TIMESTEPS", "3")))
    parser.add_argument("--task", default="", help="Pi-0 language prompt; empty matches TestingWorkflow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apply-invalid-controls", action="store_true")
    parser.add_argument("--check-clipping-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    check_control_clipping()
    if args.check_clipping_only:
        return
    checkpoint = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")
    base_body = os.environ.get("ROBOT_BASE_BODY", "robot0_base")
    eef_body = os.environ.get("EEF_BODY", "gripper0_right_eef")
    gripper_body = os.environ.get("GRIPPER_BODY", "gripper0_right_leftfinger")
    logger.info("Configuration: %s checkpoint=%s devices=%s", vars(args), checkpoint, jax.devices())
    logger.info("Bodies: base=%s eef=%s gripper=%s", base_body, eef_body, gripper_body)
    logger.info("Raw Pi-0 actions: no additional scaling, clipping, or frame conversion")

    input_spec = PolicyInputSpecStandardInputSpec.model_validate({
        "content": {
            "base_body_name": base_body,
            "states": [
                {"type": "InputStateControllable", "content": {
                    "rebase_body_tree": None, "excluded_body_trees": [],
                    "state_item": {"type": "QPos"},
                }},
                {"type": "InputStateSensors", "content": "Cameras"},
            ],
        },
    })
    output_spec = PolicyOutputSpecEEFOutputSpec.model_validate({
        "content": {
            "base_body_name": base_body,
            "eef_state": {
                "is_delta": True, "rebase_body": eef_body,
                "state_item": {"type": "XPose", "content": {
                    "frame_spec": "WorldFrame", "orientation_spec": "AxisAngle",
                }},
            },
            "gripper": {
                "is_delta": True, "rebase_body": gripper_body,
                "state_item": {"type": "QPos"},
            },
        },
    })
    logger.info("Input spec: %s", input_spec.model_dump_json())
    logger.info("Output spec: %s", output_spec.model_dump_json())

    started = time.perf_counter()
    logger.info("START LIBERO MJCF helper: dataset=%s task_index=%d", args.dataset, args.task_index)
    (mjcf,) = get_task_mjcfs([args.task_index], args.dataset)
    logger.info("DONE LIBERO MJCF: characters=%d elapsed=%.3fs", len(mjcf), time.perf_counter() - started)

    started = time.perf_counter()
    logger.info("START Pi-0 checkpoint load: %s", checkpoint)
    pi0 = Pi0Jax.from_checkpoint(checkpoint, n_action_steps=1)
    trainable, frozen = pi0.get_policy_store(), pi0.get_frozen_store()
    jax.block_until_ready((trainable, frozen))
    logger.info("DONE checkpoint load: elapsed=%.3fs", time.perf_counter() - started)
    for name in ("norm_state", "unnorm_action"):
        for field in ("mean", "std"):
            log_array(f"checkpoint.{name}.{field}", frozen[name][field])

    started = time.perf_counter()
    logger.info("START MJX/Warp initialization: nworld=1; existing default state and capacities")
    data, model, cpu_data, cpu_model = MjxSimulator.init(mjcf=mjcf, nworld=1)
    jax.block_until_ready(data)
    logger.info(
        "DONE simulator: elapsed=%.3fs nq=%d nv=%d nu=%d nbody=%d ngeom=%d "
        "ncam=%d neq=%d timestep=%g initial_cpu_contacts=%d njmax=%d",
        time.perf_counter() - started, cpu_model.nq, cpu_model.nv, cpu_model.nu,
        cpu_model.nbody, cpu_model.ngeom, cpu_model.ncam, cpu_model.neq,
        cpu_model.opt.timestep, cpu_data.ncon, cpu_model.njmax,
    )
    logger.info("MJX allocated constraint capacity: njmax=%s", data._impl.njmax)
    if cpu_model.ncam == 0:
        raise ValueError("LIBERO model has no cameras for Pi-0 inference")
    eef_id = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, eef_body)
    if eef_id < 0:
        raise ValueError(f"Unknown EEF body: {eef_body}")
    log_state("initial_state", data, eef_id)

    infer = jax.jit(jax.vmap(
        lambda params, buffers, images, state, key: pi0.infer(
            params, buffers, images, state, args.task, key
        ), in_axes=(None, None, 0, 0, 0),
    ))
    observe = jax.jit(jax.vmap(
        lambda state: MjxSimulator.get_model_state_pure(input_spec, state, model)
    ))
    map_controls = jax.jit(jax.vmap(
        lambda state, action: MjxSimulator.get_actuator_input_from_control_input(
            model, state, action, output_spec
        )
    ))
    # Disable limits only for diagnostics of the original request. Dynamics
    # and the actual mapping always use the original model with its limits.
    unlimited_model = model.replace(
        actuator_ctrllimited=np.zeros_like(model.actuator_ctrllimited),
        actuator_forcelimited=np.zeros_like(model.actuator_forcelimited),
    )
    map_requested = jax.jit(jax.vmap(
        lambda state, action: MjxSimulator.get_actuator_input_from_control_input(
            unlimited_model, state, action, output_spec
        )
    ))
    forward = jax.jit(jax.vmap(
        lambda state, controls: MjxSimulator.forward_dyn_pure(state, model, controls)
    ))
    _, rng = jax.random.split(jax.random.PRNGKey(args.seed))
    invalid_steps = []
    logger.info("START render context: one world; first use of each JIT stage includes compilation")
    context = MjxSimulator.get_render_context(cpu_model, nworld=1)
    try:
        render_context = context.pytree()
        for step in range(args.steps):
            step_started = time.perf_counter()
            logger.info("========== STEP %d/%d ==========", step, args.steps - 1)
            log_state("before", data, eef_id)

            started = time.perf_counter()
            logger.info("START observation extraction and camera rendering")
            observation = observe(data)
            frames, data = MjxSimulator.get_model_render(data, model, render_context)
            log_array("camera_frames [world,camera,height,width,rgb]", frames, values=False)
            for camera in range(cpu_model.ncam):
                name = mujoco.mj_id2name(cpu_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                log_array(f"camera[{camera}]={name}", frames[:, camera], values=False)
            states = list(observation.content.states)
            states[1] = replace(states[1], content=frames)
            observation = replace(observation, content=replace(observation.content, states=states))
            images, proprio = Pi0Model._split_observation(observation, input_spec)
            log_array("policy.proprio", proprio)
            logger.info(
                "Pi-0 consumes first %d proprio coordinates; elapsed=%.3fs",
                min(proprio.shape[-1], frozen["norm_state"]["mean"].shape[0]),
                time.perf_counter() - started,
            )

            started = time.perf_counter()
            logger.info("START Pi-0 inference: prompt=%r", args.task)
            rng, step_key = jax.random.split(rng)
            keys = jax.random.split(step_key, 1)
            log_array("policy.rng_keys", keys)
            actions = infer(trainable, frozen, images, proprio, keys)
            log_array("policy.action_chunk", actions)
            action = actions[:, 0]
            log_array("policy.selected_first_action", action)
            logger.info("DONE inference: elapsed=%.3fs", time.perf_counter() - started)

            started = time.perf_counter()
            logger.info("START actuator mapping: world pose delta + gripper QPos delta; kp=25 kd=10")
            controls, valid = map_controls(data, action)
            requested, requested_valid = map_requested(data, action)
            requested_np = log_array("mapping.requested_controls_before_clipping", requested)
            controls_np = log_array("mapping.controls", controls)
            logger.info("mapping.clipped_actuator_indices=%s", np.flatnonzero(
                ~np.isclose(requested_np[0], controls_np[0], rtol=1e-5, atol=1e-6)
            ).tolist())
            valid_np = np.asarray(jax.block_until_ready(valid))
            logger.info("mapping.valid=%s elapsed=%.3fs", valid_np.tolist(), time.perf_counter() - started)
            logger.info("mapping.unconstrained_solve_valid=%s", np.asarray(requested_valid).tolist())
            limits_ok = log_actuator_limits(cpu_model, data, controls_np)
            if valid_np.all() and not limits_ok:
                raise AssertionError("Mapping reported valid controls outside actuator limits")
            if step == 0:
                explain_mapping(cpu_model, model, data, action, requested_np, map_controls, map_requested, output_spec)
            if not valid_np.all():
                invalid_steps.append(step)
                logger.error("Invalid actuator mapping: task_index=%d env_index=0 timestep=%d", args.task_index, step)
            applied = controls if args.apply_invalid_controls else jnp.where(valid[:, None], controls, data.ctrl)
            log_array("dynamics.applied_controls", applied)
            logger.info(
                "Control selection: %s",
                "mapped controls" if valid_np.all() or args.apply_invalid_controls
                else "previous controls retained because mapping was invalid",
            )

            started = time.perf_counter()
            logger.info("START forward dynamics: one MuJoCo timestep")
            previous_qpos = data.qpos
            data = forward(data, applied)
            jax.block_until_ready(data)
            logger.info("DONE forward dynamics: elapsed=%.3fs", time.perf_counter() - started)
            log_state("after", data, eef_id)
            log_array("dynamics.qpos_change", data.qpos - previous_qpos)
            logger.info("DONE step=%d elapsed=%.3fs", step, time.perf_counter() - step_started)
    finally:
        close = getattr(context, "close", None)
        if callable(close):
            close()
        logger.info("Render context released")

    if invalid_steps:
        raise AssertionError(f"Invalid actuator mappings at steps {invalid_steps}; see logs above")
    logger.info("PASS: %d observation/policy/mapping/dynamics steps", args.steps)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("FAIL: LIBERO Pi-0 controls diagnostic")
        raise
