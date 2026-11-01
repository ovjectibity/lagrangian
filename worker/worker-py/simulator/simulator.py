from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import numpy as np

from commondef.training import (
    ControlPolicyInputSpec,
    ControlPolicyInputSpecTypes,
    InputStateItemControllableTypes,
    InputStateItemSensors,
    InputStateTypes,
    IOStateXFrame,
    IOStateXOrientationType,
)
from commondef.ioadapter import (
    ControlPolicyInputState,
    ControlPolicyInputStateStandardInputSpec,
    InputStateControllableData,
    InputStateDataInputStateControllable,
    InputStateDataInputStateSensors,
    StandardInputData,
)

try:
    import mujoco
    from mujoco import mjx
    import warp
except ModuleNotFoundError:
    mujoco = None
    mjx = None
    warp = None

# Camera render resolution (hardcoded; to be standardised in training graph later).
_CAM_HEIGHT = 224
_CAM_WIDTH = 224

class Simulator(ABC):
    @staticmethod
    def _get_contact_pair_sensor_metadata(
        model: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed contact-pair sensor addresses and their body IDs.

        A contact-pair sensor is an incoming MJCF ``<contact>`` sensor with
        ``body1`` and ``body2`` criteria, ``num=\"1\"``, and
        ``data=\"found\"``.  The MuJoCo compiler stores the contact data
        selection and slot count in ``sensor_intprm``.  Restricting the
        helper to this layout gives every selected sensor one scalar in
        ``data.sensordata`` and one fixed ``(body1, body2)`` output slot.

        This runs once while setting up a simulator; it must not be called
        from a JAX-traced state-extraction function.
        """
        if mujoco is None:
            raise ModuleNotFoundError(
                "mujoco is not installed. Install dependencies with `uv sync`."
            )

        nsensor = int(model.nsensor)
        if nsensor == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 2), dtype=np.int32),
            )

        sensor_type = np.asarray(model.sensor_type).reshape(nsensor)
        sensor_objtype = np.asarray(model.sensor_objtype).reshape(nsensor)
        sensor_reftype = np.asarray(model.sensor_reftype).reshape(nsensor)
        sensor_objid = np.asarray(model.sensor_objid).reshape(nsensor)
        sensor_refid = np.asarray(model.sensor_refid).reshape(nsensor)
        sensor_dim = np.asarray(model.sensor_dim).reshape(nsensor)
        sensor_adr = np.asarray(model.sensor_adr).reshape(nsensor)
        sensor_intprm = np.asarray(model.sensor_intprm).reshape(nsensor, -1)
        if sensor_intprm.shape[1] < 3:
            raise ValueError("MuJoCo contact sensor parameters are incomplete")

        is_body_pair = (
            (sensor_type == int(mujoco.mjtSensor.mjSENS_CONTACT))
            & (sensor_objtype == int(mujoco.mjtObj.mjOBJ_BODY))
            & (sensor_reftype == int(mujoco.mjtObj.mjOBJ_BODY))
            # ``sensor_intprm[:, 0]`` is the contact-data bitfield.  The
            # first bit is ``found`` and must be the only selected field.
            & (sensor_intprm[:, 0] == 1)
            # ``sensor_intprm[:, 2]`` is the MJCF ``num`` attribute.
            & (sensor_intprm[:, 2] == 1)
            & (sensor_dim == 1)
        )
        sensor_ids = np.flatnonzero(is_body_pair)
        body_pairs = np.column_stack(
            (sensor_objid[sensor_ids], sensor_refid[sensor_ids])
        ).astype(np.int32, copy=False)

        if np.any(body_pairs < 0) or np.any(body_pairs >= int(model.nbody)):
            raise ValueError("Contact-pair sensor references an invalid body")

        return (
            np.asarray(sensor_adr[sensor_ids], dtype=np.int32),
            np.asarray(body_pairs, dtype=np.int32),
        )

    @staticmethod
    def _get_contact_pair_sensor_state(
        data: Any,
        contact_sensor_addresses: jnp.ndarray | np.ndarray,
        contact_body_pairs: jnp.ndarray | np.ndarray,
        array_module: Any,
    ) -> jnp.ndarray | np.ndarray:
        """Return configured body pairs whose scalar ``found`` sensor is set."""
        sensor_addresses = array_module.asarray(
            contact_sensor_addresses,
            dtype=array_module.int32,
        )
        body_pairs = array_module.asarray(
            contact_body_pairs,
            dtype=array_module.int32,
        )
        if sensor_addresses.ndim != 1:
            raise ValueError("contact_sensor_addresses must be one-dimensional")
        if body_pairs.ndim != 2 or body_pairs.shape != (sensor_addresses.shape[0], 2):
            raise ValueError(
                "contact_body_pairs must have shape (len(contact_sensor_addresses), 2)"
            )

        found = data.sensordata[..., sensor_addresses] > 0
        pairs = array_module.where(found[..., None], body_pairs, -1)
        return pairs.astype(data.qpos.dtype).reshape(found.shape[:-1] + (-1,))

    @staticmethod
    def _selected_body_indices(
        model: Any,
        base_name: str | None,
        rebase_name: str | None,
        excluded_names: tuple[str, ...],
        name_to_id: Any,
    ) -> np.ndarray:
        """Select one rebased body subtree, then remove excluded subtrees."""
        parents = np.asarray(model.body_parentid)

        def subtree(name: str) -> np.ndarray:
            root_id = int(name_to_id(model, name))
            if root_id < 0:
                raise ValueError(f"Unknown MuJoCo body: {name}")
            mask = np.zeros(int(model.nbody), dtype=bool)
            mask[root_id] = True
            for body_id in range(root_id + 1, int(model.nbody)):
                mask[body_id] = mask[int(parents[body_id])]
            return mask

        base = (
            subtree(base_name)
            if base_name is not None
            else np.ones(int(model.nbody), dtype=bool)
        )
        selected = base
        if rebase_name is not None:
            rebase_id = int(name_to_id(model, rebase_name))
            if rebase_id < 0:
                raise ValueError(f"Unknown MuJoCo body: {rebase_name}")
            if not base[rebase_id]:
                raise ValueError(
                    f"Rebase body {rebase_name} is outside base body tree"
                )
            selected = base & subtree(rebase_name)
        for name in excluded_names:
            selected &= ~subtree(name)
        return np.flatnonzero(selected)

    @staticmethod
    def _get_model_state(
        input_spec: ControlPolicyInputSpec,
        data: Any,
        model: Any,
        contact_sensor_addresses: jnp.ndarray | np.ndarray | None,
        contact_body_pairs: jnp.ndarray | np.ndarray | None,
        array_module: Any,
        name_to_id: Any,
    ) -> ControlPolicyInputState:
        """Extract spec-ordered state arrays without rendering or changing Data."""

        if input_spec.type != ControlPolicyInputSpecTypes.STANDARD_INPUT_SPEC:
            raise ValueError(f"Unsupported ControlPolicyInputSpec: {input_spec.type}")

        content = input_spec.content

        def extract_controllable(state: Any) -> InputStateControllableData:
            rebase = state.rebase_body_tree
            excluded = tuple(state.excluded_body_trees)
            item = state.state_item
            bodies = Simulator._selected_body_indices(
                model,
                content.base_body_name,
                rebase,
                excluded,
                name_to_id,
            )

            if item.type == InputStateItemControllableTypes.Q_POS:
                joint_ids = np.flatnonzero(
                    np.isin(np.asarray(model.jnt_bodyid), bodies)
                )
                addresses = np.asarray(model.jnt_qposadr)
                qpos_ids = (
                    np.concatenate(
                        [
                            np.arange(
                                addresses[joint_id],
                                addresses[joint_id + 1]
                                if joint_id + 1 < int(model.njnt)
                                else int(model.nq),
                            )
                            for joint_id in joint_ids
                        ]
                    )
                    if joint_ids.size
                    else np.empty(0, dtype=np.int32)
                )
                value = data.qpos[..., qpos_ids]
            elif item.type in (
                InputStateItemControllableTypes.Q_VEL,
                InputStateItemControllableTypes.GENERALISED_FORCES,
            ):
                dof_ids = np.flatnonzero(
                    np.isin(np.asarray(model.dof_bodyid), bodies)
                )
                source = (
                    data.qvel
                    if item.type == InputStateItemControllableTypes.Q_VEL
                    else data.qfrc_applied
                )
                value = source[..., dof_ids]
            elif item.type == InputStateItemControllableTypes.X_POSE:
                pose_spec = item.content
                if pose_spec.frame_spec != IOStateXFrame.WORLDFRAME:
                    raise ValueError(
                        f"Unsupported XPose frame: {pose_spec.frame_spec}"
                    )
                positions = data.xpos[..., bodies, :]
                orientations = data.xquat[..., bodies, :]
                if pose_spec.orientation_spec == IOStateXOrientationType.AXISANGLE:
                    # q and -q represent the same orientation; make the
                    # returned angle no greater than pi before conversion.
                    orientations = array_module.where(
                        orientations[..., :1] < 0,
                        -orientations,
                        orientations,
                    )
                    vector = orientations[..., 1:]
                    magnitude = array_module.linalg.norm(vector, axis=-1)
                    epsilon = array_module.finfo(orientations.dtype).eps
                    axis = vector / array_module.maximum(
                        magnitude[..., None], epsilon
                    )
                    axis = array_module.where(
                        magnitude[..., None] > epsilon,
                        axis,
                        array_module.asarray(
                            [1.0, 0.0, 0.0], dtype=orientations.dtype
                        ),
                    )
                    angle = 2 * array_module.arctan2(
                        magnitude, orientations[..., 0]
                    )
                    orientations = array_module.concatenate(
                        (axis, angle[..., None]), axis=-1
                    )
                elif pose_spec.orientation_spec != IOStateXOrientationType.QUATERNION:
                    raise ValueError(
                        "Unsupported XPose orientation: "
                        f"{pose_spec.orientation_spec}"
                    )
                pose = array_module.concatenate((positions, orientations), axis=-1)
                value = pose.reshape(pose.shape[:-2] + (-1,))
            else:
                raise ValueError(f"Unsupported input state item: {item.type}")

            if array_module is np:
                value = value.copy()
            return InputStateControllableData(
                rebase_body_tree=rebase,
                excluded_body_trees=excluded,
                state_item=value,
            )

        states = []
        for state in content.states:
            if state.type == InputStateTypes.INPUT_STATE_CONTROLLABLE:
                states.append(
                    InputStateDataInputStateControllable(
                        type=state.type,
                        content=extract_controllable(state.content),
                    )
                )
            elif state.type == InputStateTypes.INPUT_STATE_SENSORS:
                if state.content == InputStateItemSensors.CAMERAS:
                    # Rendering remains separate because a vmapped renderer
                    # requires a correspondingly batched MJX data object.
                    value = array_module.empty((0,), dtype=data.qpos.dtype)
                elif state.content == InputStateItemSensors.CONTACTPAIRS:
                    if contact_sensor_addresses is None or contact_body_pairs is None:
                        raise ValueError(
                            "ContactPairs requires contact_sensor_addresses and "
                            "contact_body_pairs"
                        )
                    value = Simulator._get_contact_pair_sensor_state(
                        data,
                        contact_sensor_addresses,
                        contact_body_pairs,
                        array_module,
                    )
                else:
                    raise ValueError(f"Unsupported sensor state item: {state.content}")
                if array_module is np:
                    value = value.copy()
                states.append(
                    InputStateDataInputStateSensors(
                        type=state.type,
                        content=value,
                    )
                )
            else:
                raise ValueError(f"Unsupported input state: {state.type}")

        return ControlPolicyInputStateStandardInputSpec(
            type=input_spec.type,
            content=StandardInputData(
                base_body_name=content.base_body_name,
                states=states,
            ),
        )

    @staticmethod
    @abstractmethod
    def forward_dyn_pure(
        data: Any,
        model: Any,
        controls: jnp.ndarray | np.ndarray,
    ) -> Any:
        """Advance a copied physics state without mutating simulator-owned state."""
        pass

    @staticmethod
    @abstractmethod
    def inverse_dyn_pure(
        data: Any,
        model: Any,
        acc: jnp.ndarray | np.ndarray,
    ) -> tuple[jnp.ndarray | np.ndarray, Any]:
        pass

    @staticmethod
    @abstractmethod
    def get_model_state_pure(
        input_spec: ControlPolicyInputSpec,
        data: Any,
        model: Any,
        contact_sensor_addresses: jnp.ndarray | np.ndarray | None = None,
        contact_body_pairs: jnp.ndarray | np.ndarray | None = None,
    ) -> ControlPolicyInputState:
        """Extract input-spec-shaped state without rendering."""
        pass

    @staticmethod
    @abstractmethod
    def get_model_render(
        data: Any,
        model: Any,
        render_ctx: Any,
    ) -> tuple[jnp.ndarray | np.ndarray, Any]:
        """Render all model cameras and return RGB frames and data.

        Carry the returned data forward: MJX versions that return a post-render
        execution token include it in that data object.
        """
        pass

    @staticmethod
    @abstractmethod
    def init(
        model_path: str | None = None,
        mjcf: str | None = None,
    ) -> tuple[Any, ...]:
        """Build backend-native state from one MJCF source.

        ``mjcf`` is an XML string; ``model_path`` is a path to an MJCF file.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_render_context(model: Any, nworld: int = 1) -> Any:
        """Build a caller-owned backend render context for ``model``.

        MJX callers pass the CPU ``MjModel`` returned by ``init`` because
        rendering resources are created from the MuJoCo model.
        """
        pass

    @staticmethod
    @abstractmethod
    def nq(data: Any) -> int:
        pass

    @staticmethod
    @abstractmethod
    def nu(data: Any) -> int:
        pass

    @staticmethod
    @abstractmethod
    def nv(data: Any) -> int:
        pass

    @staticmethod
    @abstractmethod
    def nbody(data: Any) -> int:
        pass

    @staticmethod
    @abstractmethod
    def get_actuator_input_from_control_input(
        model: Any,
        data: Any,
        control_input: jnp.ndarray | np.ndarray,
        control_policy_output_mode: ControlPolicyOutputMode,
    ) -> jnp.ndarray | np.ndarray:
        """Map an nv-sized policy output onto supported actuator inputs."""
        pass

class MjxSimulator(Simulator):
    @staticmethod
    def init(
        model_path: str | None = None,
        mjcf: str | None = None,
        nworld: int | None = None,
        naconmax: int | None = None,
        naccdmax: int | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Build MJX and C MuJoCo state from an MJCF source.

        Returns ``(mjx_data, mjx_model, mj_data, mj_model)``. When ``nworld``
        is supplied, the MJX data is batched over that many worlds. Contact
        capacities default to ``ngeom * (ngeom - 1) // 2`` when omitted.
        """
        if mujoco is None or mjx is None:
            raise ModuleNotFoundError(
                "mujoco/mjx is not installed. Install dependencies with `uv sync`."
            )
        if (model_path is None) == (mjcf is None):
            raise ValueError("Provide exactly one of model_path or mjcf")
        if nworld is not None and nworld < 1:
            raise ValueError("nworld must be at least 1")

        if mjcf is not None:
            logger.info("MjxSimulator: loading model from XML string")
            cpu_model = mujoco.MjModel.from_xml_string(mjcf)
        else:
            logger.info("MjxSimulator: loading model from path: %s", model_path)
            cpu_model = mujoco.MjModel.from_xml_path(model_path)
        logger.info(
            "MjxSimulator: model loaded (nq=%s nu=%s nv=%s ngeom=%s "
            "nbody=%s nconmax=%s npair=%s)",
            cpu_model.nq,
            cpu_model.nu,
            cpu_model.nv,
            cpu_model.ngeom,
            cpu_model.nbody,
            cpu_model.nconmax,
            cpu_model.npair,
        )
        logger.info(
            "MjxSimulator: model texture coordinates "
            "(mesh_texcoord=%s mesh_texcoordadr=%s mesh_texcoordnum=%s "
            "mesh_facetexcoord=%s)",
            cpu_model.mesh_texcoord,
            cpu_model.mesh_texcoordadr,
            cpu_model.mesh_texcoordnum,
            cpu_model.mesh_facetexcoord,
        )

        default_contact_capacity = cpu_model.ngeom * (cpu_model.ngeom - 1) // 2
        if naconmax is None:
            naconmax = default_contact_capacity
        if naccdmax is None:
            naccdmax = default_contact_capacity

        logger.info("MjxSimulator: creating MjData")
        cpu_data = mujoco.MjData(cpu_model)
        logger.info("MjxSimulator: running mj_forward")
        mujoco.mj_forward(cpu_model, cpu_data)
        logger.info(
            "MjxSimulator: after mj_forward — ncon=%s "
            "(allocated contact buffer size=%s)",
            cpu_data.ncon,
            cpu_data.contact.dist.shape[0],
        )

        logger.info("MjxSimulator: calling mjx.put_model")
        mjx_model = mjx.put_model(cpu_model, impl="warp")
        logger.info(
            "MjxSimulator: calling mjx.make_data "
            "(naconmax=%s naccdmax=%s)",
            naconmax,
            naccdmax,
        )
        if nworld is None:
            mjx_data = mjx.make_data(
                cpu_model,
                impl="warp",
                naconmax=naconmax,
                naccdmax=naccdmax,
            )
            mjx_data = mjx.forward(mjx_model, mjx_data)
            _ = mjx.step(mjx_model, mjx_data)
        else:
            mjx_data = jax.vmap(
                lambda _: mjx.make_data(
                    cpu_model,
                    impl="warp",
                    naconmax=naconmax,
                    naccdmax=naccdmax,
                )
            )(jnp.arange(nworld))
            mjx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(
                mjx_model, mjx_data
            )
            _ = jax.vmap(mjx.step, in_axes=(None, 0))(mjx_model, mjx_data)
        # logger.info("MjxSimulator: init complete")

        # logger.info("MjxSimulator: tracing and benchmarking mjx.step")
        # bench_step = jax.jit(lambda data: mjx.step(mjx_model, data))
        # start = time.perf_counter()
        # bench_lowered = bench_step.lower(mjx_data)
        # lowered_at = time.perf_counter()
        # logger.info(
        #     "MjxSimulator: benchmark — lowering took %.2fs",
        #     lowered_at - start,
        # )
        # bench_compiled = bench_lowered.compile()
        # compiled_at = time.perf_counter()
        # logger.info(
        #     "MjxSimulator: benchmark — XLA compilation took %.2fs",
        #     compiled_at - lowered_at,
        # )
        # bench_result = bench_compiled(mjx_data)
        # jax.block_until_ready(bench_result)
        # first_execution_at = time.perf_counter()
        # logger.info(
        #     "MjxSimulator: benchmark — first execution took %.2fs",
        #     first_execution_at - compiled_at,
        # )
        # for _ in range(10):
        #     bench_result = bench_compiled(bench_result)
        # jax.block_until_ready(bench_result)
        # finished_at = time.perf_counter()
        # logger.info(
        #     "MjxSimulator: benchmark — 10 steps took %.2fs (%.4fs/step)",
        #     finished_at - first_execution_at,
        #     (finished_at - first_execution_at) / 10,
        # )

        return mjx_data, mjx_model, cpu_data, cpu_model

    @staticmethod
    def get_render_context(model: mujoco.MjModel, nworld: int = 1) -> Any:
        """Builds a fresh Warp render context sized for `nworld` worlds.

        Stateless — the Simulator does not own or cache this. Building it is
        expensive (mesh BVHs, Warp buffer allocation), so the caller should
        build it once (sized for its actual batch — e.g. num_parallel_envs)
        and reuse it for the whole rollout.

        Lifecycle: the returned RenderContext must be kept alive by the
        caller for as long as it's used — its __del__ frees the underlying
        buffers, and RenderContextPytree (from .pytree()) holds only a lookup
        key, not a reference back to this object. Call .pytree() to get the
        JAX-traceable/vmappable handle to pass into mjx.render/get_rgb.
        """
        if mujoco is None or mjx is None:
            raise ModuleNotFoundError(
                "mujoco/mjx is not installed. Install dependencies with `uv sync`."
            )
        if nworld < 1:
            raise ValueError("nworld must be at least 1")

        # STLs do not contain mesh texture coordinates, but textures can still
        # be enabled for them and cause the render megakernel to crash.
        use_textures = len(model.mesh_texcoord) > 0
        return mjx.create_render_context(
            model,
            nworld=nworld,
            cam_res=(_CAM_WIDTH, _CAM_HEIGHT),
            use_textures=use_textures,
        )

    @staticmethod
    def nq(data: Any) -> int:
        return int(data.qpos.shape[0])

    @staticmethod
    def nu(data: Any) -> int:
        return int(data.ctrl.shape[0])

    @staticmethod
    def nv(data: Any) -> int:
        return int(data.qvel.shape[0])

    @staticmethod
    def nbody(data: Any) -> int:
        return int(data.xpos.shape[0])

    @staticmethod
    def forward_dyn_pure(data: Any, model: Any, controls: jnp.ndarray) -> Any:
        """Return the next immutable MJX state without mutating ``data``."""
        data_with_controls = data.replace(ctrl=controls.reshape(data.ctrl.shape))
        return mjx.step(model, data_with_controls)

    @staticmethod
    def get_actuator_input_from_control_input(
        model: Any,
        data: Any,
        control_input: jnp.ndarray,
        control_policy_output_mode: ControlPolicyOutputMode,
    ) -> jnp.ndarray:
        """Map an nv-sized policy output onto supported MJX controls.

        ``control_policy_output_mode`` must be static when this function is
        transformed with ``jax.jit``. Unsupported mappings retain the
        corresponding values from ``data.ctrl``.
        """
        if control_input.ndim != 1:
            raise ValueError("control_input must be one-dimensional")
        if control_input.shape[0] != data.qvel.shape[0]:
            raise ValueError(
                "control_input length must equal model.nv: "
                f"expected {data.qvel.shape[0]}, got {control_input.shape[0]}"
            )

        mode_label = control_policy_output_mode.value
        logger.warning(
            "Tracing MJX control mapping for %s; unsupported mappings retain "
            "their existing data.ctrl values",
            mode_label,
        )

        nu = data.ctrl.shape[0]
        nv = data.qvel.shape[0]
        nactuator = model.actuator_trnid.shape[0]
        njnt = model.jnt_dofadr.shape[0]
        noutput = data.actuator_length.shape[0]
        if nu == 0 or nactuator == 0 or njnt == 0 or noutput == 0:
            if logger.isEnabledFor(logging.DEBUG) and nv > 0:
                jax.debug.print(
                    f"WARNING: MJX {mode_label} mapping ignored all "
                    "{count} policy "
                    "values because the model has no supported actuators",
                    count=nv,
                )
            return data.ctrl

        actuator_indices = jnp.arange(nactuator, dtype=jnp.int32)
        control_addresses = (
            model.actuator_ctrladr
            if hasattr(model, "actuator_ctrladr")
            else actuator_indices
        )
        control_counts = (
            model.actuator_ctrlnum
            if hasattr(model, "actuator_ctrlnum")
            else jnp.ones((nactuator,), dtype=jnp.int32)
        )
        output_addresses = (
            model.actuator_outadr
            if hasattr(model, "actuator_outadr")
            else actuator_indices
        )
        output_counts = (
            model.actuator_outnum
            if hasattr(model, "actuator_outnum")
            else jnp.ones((nactuator,), dtype=jnp.int32)
        )

        joint_ids = model.actuator_trnid[:, 0]
        safe_joint_ids = jnp.clip(joint_ids, 0, njnt - 1)
        joint_types = model.jnt_type[safe_joint_ids]
        dof_addresses = model.jnt_dofadr[safe_joint_ids]
        direct_joint = (
            (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
            | (
                model.actuator_trntype
                == int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)
            )
        )
        scalar_joint = (
            (joint_types == int(mujoco.mjtJoint.mjJNT_HINGE))
            | (joint_types == int(mujoco.mjtJoint.mjJNT_SLIDE))
        )
        valid_mapping = (
            direct_joint
            & (joint_ids >= 0)
            & (joint_ids < njnt)
            & scalar_joint
            & (control_counts == 1)
            & (output_counts == 1)
            & (control_addresses >= 0)
            & (control_addresses < nu)
            & (output_addresses >= 0)
            & (output_addresses < data.actuator_length.shape[0])
            & (dof_addresses >= 0)
            & (dof_addresses < nv)
        )

        gain_params = model.actuator_gainprm
        bias_params = model.actuator_biasprm
        fixed_gain = model.actuator_gaintype == int(
            mujoco.mjtGain.mjGAIN_FIXED
        )
        affine_bias = model.actuator_biastype == int(
            mujoco.mjtBias.mjBIAS_AFFINE
        )
        no_dynamics = model.actuator_dyntype == int(
            mujoco.mjtDyn.mjDYN_NONE
        )
        position_actuator = (
            fixed_gain
            & affine_bias
            & (
                model.actuator_dyntype
                != int(mujoco.mjtDyn.mjDYN_INTEGRATOR)
            )
            & jnp.isclose(bias_params[:, 0], 0.0)
            & ~jnp.isclose(bias_params[:, 1], 0.0)
            & jnp.isclose(gain_params[:, 0], -bias_params[:, 1])
        )
        velocity_actuator = (
            fixed_gain
            & affine_bias
            & no_dynamics
            & jnp.isclose(bias_params[:, 0], 0.0)
            & jnp.isclose(bias_params[:, 1], 0.0)
            & ~jnp.isclose(bias_params[:, 2], 0.0)
            & jnp.isclose(gain_params[:, 0], -bias_params[:, 2])
        )
        motor_actuator = (
            fixed_gain
            & (model.actuator_biastype == int(mujoco.mjtBias.mjBIAS_NONE))
            & no_dynamics
        )

        safe_dof_addresses = jnp.clip(dof_addresses, 0, nv - 1)
        safe_control_addresses = jnp.clip(control_addresses, 0, nu - 1)
        safe_output_addresses = jnp.clip(
            output_addresses,
            0,
            data.actuator_length.shape[0] - 1,
        )
        policy_values = control_input[safe_dof_addresses]

        if control_policy_output_mode == ControlPolicyOutputMode.TARGETPOSITIONSDIFF:
            supported = valid_mapping & position_actuator
            mapped_values = (
                data.actuator_length[safe_output_addresses] + policy_values
            )
        elif control_policy_output_mode == ControlPolicyOutputMode.TARGETPOSITIONS:
            supported = valid_mapping & position_actuator
            mapped_values = policy_values
        elif control_policy_output_mode == ControlPolicyOutputMode.TARGETVELOCITIES:
            supported = valid_mapping & velocity_actuator
            mapped_values = policy_values
        elif control_policy_output_mode == ControlPolicyOutputMode.TARGETTORQUES:
            supported = valid_mapping & motor_actuator
            mapped_values = policy_values
        else:
            raise ValueError(
                f"Unsupported control policy output mode: {control_policy_output_mode}"
            )

        limited = model.actuator_ctrllimited[safe_control_addresses]
        control_ranges = model.actuator_ctrlrange[safe_control_addresses]
        mapped_values = jnp.where(
            limited,
            jnp.clip(mapped_values, control_ranges[:, 0], control_ranges[:, 1]),
            mapped_values,
        )

        control_slots = jnp.arange(nu, dtype=jnp.int32)
        actuator_to_control = supported[:, None] & (
            control_addresses[:, None] == control_slots[None, :]
        )
        mapped_controls = jnp.sum(
            jnp.where(actuator_to_control, mapped_values[:, None], 0),
            axis=0,
        )
        has_mapping = jnp.any(actuator_to_control, axis=0)
        actuator_input = jnp.where(has_mapping, mapped_controls, data.ctrl)

        if logger.isEnabledFor(logging.DEBUG):
            dof_slots = jnp.arange(nv, dtype=jnp.int32)
            mapped_dofs = jnp.any(
                supported[:, None]
                & (dof_addresses[:, None] == dof_slots[None, :]),
                axis=0,
            )
            unsupported_actuators = nactuator - jnp.count_nonzero(supported)
            unsupported_policy_values = nv - jnp.count_nonzero(mapped_dofs)

            def log_unsupported(_: None) -> jnp.ndarray:
                jax.debug.print(
                    f"WARNING: MJX {mode_label} mapping retained existing "
                    "controls "
                    "for {actuator_count} unsupported actuators and ignored "
                    "{input_count} policy values",
                    actuator_count=unsupported_actuators,
                    input_count=unsupported_policy_values,
                )
                return jnp.asarray(0, dtype=jnp.int32)

            jax.lax.cond(
                (unsupported_actuators > 0) | (unsupported_policy_values > 0),
                log_unsupported,
                lambda _: jnp.asarray(0, dtype=jnp.int32),
                operand=None,
            )

        return actuator_input

    @staticmethod
    def inverse_dyn_pure(
        data: Any,
        model: Any,
        acc: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Any]:
        """Return inverse forces and updated data without mutating MJX state."""
        data_with_acc = data.replace(qacc=acc.reshape(data.qacc.shape))
        inverse_data = mjx.inverse(model, data_with_acc)
        return inverse_data.qfrc_inverse, inverse_data

    @staticmethod
    def get_model_state_pure(
        input_spec: ControlPolicyInputSpec,
        data: Any,
        model: Any,
        contact_sensor_addresses: jnp.ndarray | None = None,
        contact_body_pairs: jnp.ndarray | None = None,
    ) -> ControlPolicyInputState:
        """Extract ordered MJX state leaves."""
        return Simulator._get_model_state(
            input_spec,
            data,
            model,
            contact_sensor_addresses,
            contact_body_pairs,
            jnp,
            lambda m, name: mjx.name2id(m, mujoco.mjtObj.mjOBJ_BODY, name),
        )

    @staticmethod
    def get_model_render(
        data: Any,
        model: Any,
        render_ctx: Any,
    ) -> tuple[jnp.ndarray, Any]:
        """Return RGB frames [world, camera, H, W, 3] and MJX data."""
        if model.ncam == 0:
            raise ValueError("Cannot render a model with no cameras")
        render_output = mjx.render(model, data, render_ctx)
        if len(render_output) == 3:
            rgb_packed, _, updated_data = render_output
        else:
            # MJX 3.11.0 returns only (rgb, depth); newer versions also
            # return Data carrying the post-render execution token.
            rgb_packed, _ = render_output
            updated_data = data
        frames = jnp.stack(
            [
                mjx.get_rgb(render_ctx, cam_id, rgb_packed)
                for cam_id in range(int(model.ncam))
            ],
            axis=1,
        )
        return frames, updated_data

class MujocoSimulator(Simulator):
    @staticmethod
    def init(
        model_path: str | None = None,
        mjcf: str | None = None,
    ) -> tuple[Any, Any]:
        """Build forwarded MuJoCo ``(data, model)`` from an MJCF source."""
        if mujoco is None:
            raise ModuleNotFoundError(
                "mujoco is not installed. Install dependencies with `uv sync`."
            )
        if (model_path is None) == (mjcf is None):
            raise ValueError("Provide exactly one of model_path or mjcf")

        if mjcf is not None:
            logger.info("MujocoSimulator: loading model from XML string")
            model = mujoco.MjModel.from_xml_string(mjcf)
        else:
            logger.info("MujocoSimulator: loading model from path: %s", model_path)
            model = mujoco.MjModel.from_xml_path(model_path)
        logger.info(
            "MujocoSimulator: model loaded (nq=%s nu=%s nv=%s)",
            model.nq,
            model.nu,
            model.nv,
        )
        logger.info("MujocoSimulator: creating MjData")
        data = mujoco.MjData(model)
        logger.info("MujocoSimulator: running mj_forward")
        mujoco.mj_forward(model, data)
        logger.info("MujocoSimulator: init complete")
        return data, model

    @staticmethod
    def nq(data: Any) -> int:
        return int(data.qpos.shape[0])

    @staticmethod
    def nu(data: Any) -> int:
        return int(data.ctrl.shape[0])

    @staticmethod
    def nv(data: Any) -> int:
        return int(data.qvel.shape[0])

    @staticmethod
    def nbody(data: Any) -> int:
        return int(data.xpos.shape[0])

    @staticmethod
    def forward_dyn_pure(data: Any, model: Any, controls: np.ndarray) -> Any:
        """Return a stepped copy of MuJoCo data without mutating ``data``."""
        data_next = copy.copy(data)
        data_next.ctrl[:] = controls.reshape(data_next.ctrl.shape)
        mujoco.mj_step(model, data_next)
        return data_next

    @staticmethod
    def inverse_dyn_pure(
        data: Any,
        model: Any,
        acc: np.ndarray,
    ) -> tuple[np.ndarray, Any]:
        """Return inverse forces and updated data without mutating MuJoCo state."""
        inverse_data = copy.copy(data)
        inverse_data.qacc[:] = acc.reshape(inverse_data.qacc.shape)
        mujoco.mj_inverse(model, inverse_data)
        forces = inverse_data.qfrc_inverse.copy()
        return forces, inverse_data

    @staticmethod
    def get_model_state_pure(
        input_spec: ControlPolicyInputSpec,
        data: Any,
        model: Any,
        contact_sensor_addresses: np.ndarray | None = None,
        contact_body_pairs: np.ndarray | None = None,
    ) -> ControlPolicyInputState:
        """Extract ordered MuJoCo state leaves."""
        return Simulator._get_model_state(
            input_spec,
            data,
            model,
            contact_sensor_addresses,
            contact_body_pairs,
            np,
            lambda m, name: mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, name
            ),
        )

    @staticmethod
    def get_model_render(
        data: Any,
        model: Any,
        render_ctx: mujoco.Renderer,
    ) -> tuple[np.ndarray, Any]:
        """Render all cameras and return RGB frames and unchanged MuJoCo data."""
        if model.ncam == 0:
            raise ValueError("Cannot render a model with no cameras")
        frames: list[np.ndarray] = []
        for cam_id in range(int(model.ncam)):
            render_ctx.update_scene(data, camera=cam_id)
            frames.append(render_ctx.render().astype(np.float32) / 255.0)
        return np.stack(frames, axis=0), data

    @staticmethod
    def get_render_context(
        model: mujoco.MjModel,
        nworld: int = 1,
    ) -> mujoco.Renderer:
        """Builds a fresh mujoco.Renderer for this model.

        Stateless — the Simulator does not own or cache this. Construction is
        non-trivial (OpenGL/EGL context), so the caller should build one once
        (e.g. before a per-timestep loop) and reuse/close it themselves,
        rather than reconstructing it on every get_model_state call.
        """
        if mujoco is None:
            raise ModuleNotFoundError(
                "mujoco is not installed. Install dependencies with `uv sync`."
            )
        if nworld != 1:
            raise ValueError("MuJoCo rendering supports only nworld=1")
        return mujoco.Renderer(model, height=_CAM_HEIGHT, width=_CAM_WIDTH)

    @staticmethod
    def get_actuator_input_from_control_input(
        model: Any,
        data: Any,
        control_input: np.ndarray,
        control_policy_output_mode: ControlPolicyOutputMode,
    ) -> np.ndarray:
        """Map an nv-sized policy output onto supported MuJoCo controls."""
        if control_input.ndim != 1:
            raise ValueError("control_input must be one-dimensional")
        if control_input.shape[0] != data.qvel.shape[0]:
            raise ValueError(
                "control_input length must equal model.nv: "
                f"expected {data.qvel.shape[0]}, got {control_input.shape[0]}"
            )

        nu = data.ctrl.shape[0]
        nv = data.qvel.shape[0]
        nactuator = model.actuator_trnid.shape[0]
        njnt = model.jnt_dofadr.shape[0]
        noutput = data.actuator_length.shape[0]
        if nu == 0 or nactuator == 0 or njnt == 0 or noutput == 0:
            if nv > 0:
                logger.warning(
                    "MuJoCo %s mapping ignored all %s policy values because "
                    "the model has no supported actuators",
                    control_policy_output_mode.value,
                    nv,
                )
            return data.ctrl.copy()

        actuator_indices = np.arange(nactuator, dtype=np.int32)
        control_addresses = (
            model.actuator_ctrladr
            if hasattr(model, "actuator_ctrladr")
            else actuator_indices
        )
        control_counts = (
            model.actuator_ctrlnum
            if hasattr(model, "actuator_ctrlnum")
            else np.ones((nactuator,), dtype=np.int32)
        )
        output_addresses = (
            model.actuator_outadr
            if hasattr(model, "actuator_outadr")
            else actuator_indices
        )
        output_counts = (
            model.actuator_outnum
            if hasattr(model, "actuator_outnum")
            else np.ones((nactuator,), dtype=np.int32)
        )

        joint_ids = model.actuator_trnid[:, 0]
        safe_joint_ids = np.clip(joint_ids, 0, njnt - 1)
        joint_types = model.jnt_type[safe_joint_ids]
        dof_addresses = model.jnt_dofadr[safe_joint_ids]
        direct_joint = (
            (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
            | (
                model.actuator_trntype
                == int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)
            )
        )
        scalar_joint = (
            (joint_types == int(mujoco.mjtJoint.mjJNT_HINGE))
            | (joint_types == int(mujoco.mjtJoint.mjJNT_SLIDE))
        )
        valid_mapping = (
            direct_joint
            & (joint_ids >= 0)
            & (joint_ids < njnt)
            & scalar_joint
            & (control_counts == 1)
            & (output_counts == 1)
            & (control_addresses >= 0)
            & (control_addresses < nu)
            & (output_addresses >= 0)
            & (output_addresses < noutput)
            & (dof_addresses >= 0)
            & (dof_addresses < nv)
        )

        gain_params = model.actuator_gainprm
        bias_params = model.actuator_biasprm
        fixed_gain = model.actuator_gaintype == int(
            mujoco.mjtGain.mjGAIN_FIXED
        )
        affine_bias = model.actuator_biastype == int(
            mujoco.mjtBias.mjBIAS_AFFINE
        )
        no_dynamics = model.actuator_dyntype == int(
            mujoco.mjtDyn.mjDYN_NONE
        )
        position_actuator = (
            fixed_gain
            & affine_bias
            & (
                model.actuator_dyntype
                != int(mujoco.mjtDyn.mjDYN_INTEGRATOR)
            )
            & np.isclose(bias_params[:, 0], 0.0)
            & ~np.isclose(bias_params[:, 1], 0.0)
            & np.isclose(gain_params[:, 0], -bias_params[:, 1])
        )
        velocity_actuator = (
            fixed_gain
            & affine_bias
            & no_dynamics
            & np.isclose(bias_params[:, 0], 0.0)
            & np.isclose(bias_params[:, 1], 0.0)
            & ~np.isclose(bias_params[:, 2], 0.0)
            & np.isclose(gain_params[:, 0], -bias_params[:, 2])
        )
        motor_actuator = (
            fixed_gain
            & (model.actuator_biastype == int(mujoco.mjtBias.mjBIAS_NONE))
            & no_dynamics
        )

        safe_dof_addresses = np.clip(dof_addresses, 0, nv - 1)
        safe_control_addresses = np.clip(control_addresses, 0, nu - 1)
        safe_output_addresses = np.clip(output_addresses, 0, noutput - 1)
        policy_values = control_input[safe_dof_addresses]

        if control_policy_output_mode == ControlPolicyOutputMode.TARGETPOSITIONSDIFF:
            supported = valid_mapping & position_actuator
            mapped_values = (
                data.actuator_length[safe_output_addresses] + policy_values
            )
        elif control_policy_output_mode == ControlPolicyOutputMode.TARGETPOSITIONS:
            supported = valid_mapping & position_actuator
            mapped_values = policy_values
        elif control_policy_output_mode == ControlPolicyOutputMode.TARGETVELOCITIES:
            supported = valid_mapping & velocity_actuator
            mapped_values = policy_values
        elif control_policy_output_mode == ControlPolicyOutputMode.TARGETTORQUES:
            supported = valid_mapping & motor_actuator
            mapped_values = policy_values
        else:
            raise ValueError(
                f"Unsupported control policy output mode: {control_policy_output_mode}"
            )

        limited = model.actuator_ctrllimited[safe_control_addresses]
        control_ranges = model.actuator_ctrlrange[safe_control_addresses]
        mapped_values = np.where(
            limited,
            np.clip(mapped_values, control_ranges[:, 0], control_ranges[:, 1]),
            mapped_values,
        )

        actuator_input = data.ctrl.copy()
        supported_indices = np.flatnonzero(supported)
        actuator_input[control_addresses[supported_indices]] = mapped_values[
            supported_indices
        ]

        mapped_dofs = np.zeros((nv,), dtype=bool)
        mapped_dofs[dof_addresses[supported_indices]] = True
        unsupported_actuators = nactuator - supported_indices.shape[0]
        unsupported_policy_values = nv - np.count_nonzero(mapped_dofs)
        if unsupported_actuators > 0 or unsupported_policy_values > 0:
            logger.warning(
                "MuJoCo %s mapping retained existing controls for %s "
                "unsupported actuators and ignored %s policy values",
                control_policy_output_mode.value,
                unsupported_actuators,
                unsupported_policy_values,
            )

        return actuator_input
