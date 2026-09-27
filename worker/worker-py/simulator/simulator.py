from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import numpy as np

from commondef.training import ControlPolicyOutputMode, ModelInputSpec

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
    def _get_position_actuator_indices(model: Any) -> np.ndarray:
        """Return compiled actuator indices for scalar position servos.

        MuJoCo compiles actuator shortcuts into generic actuator parameters.
        A position servo has fixed gain, affine bias with a position-feedback
        term, and non-integrator dynamics. Integrated-velocity servos share
        the position-feedback term but use integrator dynamics and are
        intentionally excluded.

        This helper runs once against the CPU ``MjModel`` during setup. Each
        selected actuator must have exactly one control and one force output.
        """
        if mujoco is None:
            raise ModuleNotFoundError(
                "mujoco is not installed. Install dependencies with `uv sync`."
            )

        nactuator = int(getattr(model, "nactuator", model.nu))
        if nactuator == 0:
            return np.zeros((0,), dtype=np.int32)

        gain_type = np.asarray(model.actuator_gaintype).reshape(nactuator)
        bias_type = np.asarray(model.actuator_biastype).reshape(nactuator)
        dyn_type = np.asarray(model.actuator_dyntype).reshape(nactuator)
        gain_params = np.asarray(model.actuator_gainprm).reshape(nactuator, -1)
        bias_params = np.asarray(model.actuator_biasprm).reshape(nactuator, -1)
        if gain_params.shape[1] < 1 or bias_params.shape[1] < 2:
            raise ValueError("MuJoCo actuator gain or bias parameters are incomplete")

        is_position_actuator = (
            (gain_type == int(mujoco.mjtGain.mjGAIN_FIXED))
            & (bias_type == int(mujoco.mjtBias.mjBIAS_AFFINE))
            & (dyn_type != int(mujoco.mjtDyn.mjDYN_INTEGRATOR))
            & np.isclose(bias_params[:, 0], 0.0)
            & ~np.isclose(bias_params[:, 1], 0.0)
            & np.isclose(gain_params[:, 0], -bias_params[:, 1])
        )
        actuator_indices = np.flatnonzero(is_position_actuator).astype(
            np.int32,
            copy=False,
        )

        control_counts = np.asarray(
            getattr(
                model,
                "actuator_ctrlnum",
                np.ones((nactuator,), dtype=np.int32),
            )
        ).reshape(nactuator)
        output_counts = np.asarray(
            getattr(
                model,
                "actuator_outnum",
                np.ones((nactuator,), dtype=np.int32),
            )
        ).reshape(nactuator)
        invalid_control_count = actuator_indices[
            control_counts[actuator_indices] != 1
        ]
        if invalid_control_count.size:
            raise ValueError(
                "Position actuators must each have exactly one control input; "
                f"invalid actuator indices: {invalid_control_count.tolist()}"
            )
        invalid_output_count = actuator_indices[
            output_counts[actuator_indices] != 1
        ]
        if invalid_output_count.size:
            raise ValueError(
                "Position actuators must each have exactly one force output; "
                f"invalid actuator indices: {invalid_output_count.tolist()}"
            )

        return actuator_indices

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

        found = data.sensordata[sensor_addresses] > 0
        return array_module.where(found[:, None], body_pairs, -1).astype(
            data.qpos.dtype
        ).reshape(-1)

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
        input_spec: ModelInputSpec,
        data: Any,
        model: Any,
        render_ctx: Any = None,
        actuated_qpos_indices: jnp.ndarray | np.ndarray | None = None,
        contact_sensor_addresses: jnp.ndarray | np.ndarray | None = None,
        contact_body_pairs: jnp.ndarray | np.ndarray | None = None,
    ) -> dict[str, jnp.ndarray | np.ndarray]:
        """render_ctx: caller-managed camera render handle, required only when
        input_spec.all_cameras is set. See Simulator._get_render_context /
        Simulator._get_renderer (impl-specific) for how to build one.

        Contact pairs are returned only when ``input_spec.contact_pairs`` and
        both contact metadata arrays are supplied. The arrays are generated
        once by ``_get_contact_pair_sensor_metadata`` for the incoming model.
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
    def modify_target_position(
        self,
        target_pos_diff: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray | np.ndarray:
        pass

    # @abstractmethod
    # def get_target_controls_from_target_positions(
    #     self,
    #     target_pos_diff: jnp.ndarray | np.ndarray,
    # ) -> jnp.ndarray | np.ndarray:
    #     pass

    @staticmethod
    @abstractmethod
    def get_target_act_length_from_target_diff(
        model: Any,
        data: Any,
        target_diff: jnp.ndarray | np.ndarray,
        position_actuator_indices: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray | np.ndarray:
        """Return position-actuator length targets from actuator-length deltas."""
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

    @staticmethod
    @abstractmethod
    def get_target_controls_from_target_velocities(
        data: Any,
        model: Any,
        target_velocities: jnp.ndarray | np.ndarray,
    ) -> jnp.ndarray | np.ndarray:
        pass

    # @staticmethod
    # def _compute_actuated_qpos_indices(model: Any) -> jnp.ndarray:
    #     """Return qpos indices belonging to each actuated joint.

    #     For hinge/slide joints: 1 component. Ball: 4. Free: 7.
    #     The result has model-dependent length, so compute it once from the
    #     untraced MuJoCo model and pass it to ``get_model_state_pure``.
    #     """
    #     nu = int(model.nu)
    #     nq = int(model.nq)
    #     njnt = int(model.njnt)
    #     indices: list[int] = []
    #     actuator_trnid = np.asarray(model.actuator_trnid).reshape(nu, 2)
    #     jnt_qposadr = np.asarray(model.jnt_qposadr)
    #     jnt_type = np.asarray(model.jnt_type)
    #     for actuator_idx in range(nu):
    #         joint_id = int(actuator_trnid[actuator_idx, 0])
    #         if not 0 <= joint_id < njnt:
    #             raise ValueError(
    #                 f"Actuator {actuator_idx} is not attached to a valid joint"
    #             )
    #         qpos_adr = int(jnt_qposadr[joint_id])
    #         jtype = int(jnt_type[joint_id])
    #         # mjtJoint: 0=free(7), 1=ball(4), 2=slide(1), 3=hinge(1)
    #         if jtype == 0:
    #             n = 7
    #         elif jtype == 1:
    #             n = 4
    #         else:
    #             n = 1
    #         if qpos_adr < 0 or qpos_adr + n > nq:
    #             raise ValueError(
    #                 f"Actuated joint {joint_id} has an invalid qpos address"
    #             )
    #         indices.extend(range(qpos_adr, qpos_adr + n))
    #     return jnp.asarray(indices, dtype=jnp.int32)



class MjxSimulator(Simulator):
    @staticmethod
    def init(
        model_path: str | None = None,
        mjcf: str | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Build MJX and C MuJoCo state from an MJCF source.

        Returns ``(mjx_data, mjx_model, mj_data, mj_model)``.
        """
        if mujoco is None or mjx is None:
            raise ModuleNotFoundError(
                "mujoco/mjx is not installed. Install dependencies with `uv sync`."
            )
        if (model_path is None) == (mjcf is None):
            raise ValueError("Provide exactly one of model_path or mjcf")

        cpu_model = (
            mujoco.MjModel.from_xml_string(mjcf)
            if mjcf is not None
            else mujoco.MjModel.from_xml_path(model_path)
        )
        cpu_data = mujoco.MjData(cpu_model)
        mujoco.mj_forward(cpu_model, cpu_data)
        mjx_model = mjx.put_model(cpu_model, impl="warp")
        mjx_data = mjx.make_data(cpu_model, impl="warp")
        mjx_data = mjx.forward(mjx_model, mjx_data)
        return mjx_data, mjx_model, cpu_data, cpu_model

    # def __init__(
    #     self,
    #     model_path: str | None = None,
    #     mjcf: str | None = None,
    # ) -> None:
    #     if mujoco is None or mjx is None:
    #         raise ModuleNotFoundError(
    #             "mujoco/mjx is not installed. Install dependencies with `uv sync`."
    #         )
    #     if (model_path is None and mjcf is None) or (model_path is not None and mjcf is not None):
    #         raise ValueError("Provide exactly one of model_path or mjcf")
    #     if mjcf is not None:
    #         logger.info("MjxSimulator: loading model from XML string")
    #         self._model = mujoco.MjModel.from_xml_string(mjcf)
    #     else:
    #         logger.info("MjxSimulator: loading model from path: %s", model_path)
    #         self._model = mujoco.MjModel.from_xml_path(model_path)
    #     logger.info(
    #         "MjxSimulator: model loaded (nq=%s nu=%s nv=%s ngeom=%s nbody=%s nconmax=%s npair=%s)",
    #         self._model.nq, self._model.nu, self._model.nv,
    #         self._model.ngeom, self._model.nbody, self._model.nconmax, self._model.npair,
    #     )
    #     # logger.info("Tracing mjModel's texture coordinates %s",
    #     #             self._model.mesh_texcoord,
    #     #             self._model.mesh_texcoordadr,
    #     #             self._model.mesh_texcoordnum,
    #     #             self._model.mesh_facetexcoord)

    #     ## Fix for the render megakernel crash issue, 
    #     # STLs don't have mesh texture coordinates but 
    #     # still get textures enabled (causing the crash) if not 
    #     # handled as follows
    #     if len(self._model.mesh_texcoord) == 0: 
    #         self._use_textures = False
    #     else: 
    #         self._use_textures = True

    #     logger.info("MjxSimulator: creating MjData")
    #     self._data = mujoco.MjData(self._model)
    #     logger.info("MjxSimulator: running mj_forward")
    #     mujoco.mj_forward(self._model, self._data)
    #     logger.info(
    #         "MjxSimulator: after mj_forward — ncon=%s (allocated contact buffer size=%s)",
    #         self._data.ncon, self._data.contact.dist.shape[0],
    #     )
    #     logger.info("MjxSimulator: calling mjx.put_model")
    #     # print(dir(mujoco.mjx.warp.types.GraphMode))
    #     # graph_mode = mujoco.mjx.warp.types.GraphMode.NONE
    #     self._mjx_model = mjx.put_model(
    #         self._model,
    #         impl="warp",
    #         # graph_mode=graph_mode,
    #     )
    #     # self._mjx_model = self._mjx_model.replace(
    #     #     opt=self._mjx_model.opt.replace(
    #     #         _impl=self._mjx_model.opt._impl.replace(
    #     #             graph_mode=graph_mode,
    #     #         ),
    #     #     ),
    #     # )
    #     # logger.info(
    #     #     "MjxSimulator: requested graph mode: %r (%s)",
    #     #     graph_mode,
    #     #     int(graph_mode),
    #     # )
    #     # logger.info(
    #     #     "MjxSimulator: stored graph mode: %r (%s)",
    #     #     self._mjx_model.opt._impl.graph_mode,
    #     #     int(self._mjx_model.opt._impl.graph_mode),
    #     # )
    #     logger.info("MjxSimulator: calling mjx.put_data")
    #     # self._mjx_data = mjx.put_data(self._model, self._data, impl="warp")
    #     self._mjx_data = mjx.make_data(self._model, impl="warp")
    #     mjx.forward(self._mjx_model,self._mjx_data)
    #     logger.info("MjxSimulator: init complete")

    #     # Benchmark: isolate mjx.step timing
    #     # logger.info("MjxSimulator: running mjx.step benchmark...")
    #     # _bench_step = jax.jit(lambda d: mjx.step(self._mjx_model, d))
    #     # t0 = time.time()
    #     # _bench_lowered = _bench_step.lower(self._mjx_data)
    #     # t1 = time.time()
    #     # logger.info("MjxSimulator: bench — lowering took %.2fs", t1 - t0)
    #     # _bench_compiled = _bench_lowered.compile()
    #     # t2 = time.time()
    #     # logger.info("MjxSimulator: bench — XLA compile took %.2fs", t2 - t1)
    #     # _bench_result = _bench_compiled(self._mjx_data)
    #     # jax.block_until_ready(_bench_result)
    #     # t3 = time.time()
    #     # logger.info("MjxSimulator: bench — first execution took %.2fs", t3 - t2)
    #     # for _ in range(10):
    #     #     _bench_result = _bench_compiled(_bench_result)
    #     # jax.block_until_ready(_bench_result)
    #     # t4 = time.time()
    #     # logger.info("MjxSimulator: bench — 10 steps took %.2fs (%.4fs/step)", t4 - t3, (t4 - t3) / 10)

    #     self.target_position: list[float] = self._get_actuated_qpos_targets()

    def _get_render_context(self, nworld: int) -> Any:
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
        ctx = mjx.create_render_context(
            self._model, nworld=nworld,
            use_textures=self._use_textures) #, cam_res=(_CAM_WIDTH, _CAM_HEIGHT)
        # self._mjx_data = mjx.refit_bvh(self._mjx_model,self._mjx_data,ctx.pytree())
        return ctx

    def _sync_cpu_from_mjx(self) -> None:
        self._data = mjx.get_data(self._model, self._mjx_data)

    def _sync_mjx_from_cpu(self) -> None:
        self._mjx_data = mjx.put_data(self._model, self._data, impl="warp")
        # self._mjx_data = mjx.make_data(self._model, impl="warp")

    def qpos(self) -> list[float]:
        return self._mjx_data.qpos[: MjxSimulator.nq(self._mjx_data)].tolist()

    def qvel(self) -> list[float]:
        return self._mjx_data.qvel[: MjxSimulator.nv(self._mjx_data)].tolist()

    def _is_torque_actuator(self) -> bool:
        nu = MjxSimulator.nu(self._data)
        if nu == 0:
            return True
        first_biastype = int(self._mjx_model.actuator_biastype[0])
        return first_biastype == int(mujoco.mjtBias.mjBIAS_NONE)

    def _get_actuator_joint_indices(self) -> list[tuple[int, int]]:
        nu = MjxSimulator.nu(self._data)
        nq = MjxSimulator.nq(self._data)
        nv = MjxSimulator.nv(self._data)
        actuator_trnid = self._mjx_model.actuator_trnid.reshape(nu, 2)
        jnt_qposadr = self._mjx_model.jnt_qposadr
        jnt_dofadr = self._mjx_model.jnt_dofadr
        indices: list[tuple[int, int]] = []
        for actuator_idx in range(nu):
            joint_id = int(actuator_trnid[actuator_idx, 0])
            if joint_id < 0:
                raise ValueError(
                    f"Actuator {actuator_idx} is not attached to a valid joint (joint_id={joint_id})"
                )
            qpos_adr = int(jnt_qposadr[joint_id])
            dof_adr = int(jnt_dofadr[joint_id])
            if qpos_adr < 0 or qpos_adr >= nq:
                raise ValueError(
                    f"Invalid qpos address {qpos_adr} for actuator {actuator_idx} (nq={nq})"
                )
            if dof_adr < 0 or dof_adr >= nv:
                raise ValueError(
                    f"Invalid dof address {dof_adr} for actuator {actuator_idx} (nv={nv})"
                )
            indices.append((qpos_adr, dof_adr))
        return indices

    def _clamp_controls_to_actuator_range(self, controls: list[float]) -> list[float]:
        nu = MjxSimulator.nu(self._data)
        if len(controls) != nu:
            raise ValueError(f"Control dimension mismatch. Expected {nu}, got {len(controls)}")
        ctrl_limited = self._mjx_model.actuator_ctrllimited
        ctrl_range = self._mjx_model.actuator_ctrlrange.reshape(nu, 2)
        clamped: list[float] = []
        for i, val in enumerate(controls):
            is_limited = bool(ctrl_limited[i]) if ctrl_limited.size else True
            if not is_limited:
                clamped.append(float(val))
                continue
            min_v = float(ctrl_range[i, 0])
            max_v = float(ctrl_range[i, 1])
            clamped.append(float(jnp.clip(val, min_v, max_v)))
        return clamped

    def _get_actuated_qpos_targets(self) -> list[float]:
        nu = MjxSimulator.nu(self._data)
        qpos = self.qpos()
        actuator_joint_indices = self._get_actuator_joint_indices()
        if len(actuator_joint_indices) != nu:
            raise ValueError(
                f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
            )
        targets = [qpos[qpos_idx] for qpos_idx, _ in actuator_joint_indices]
        return self._clamp_controls_to_actuator_range(targets)

    def _array_to_list(
        self,
        x: jnp.ndarray,
        expected_size: int,
        name: str,
    ) -> list[float]:
        vals = x.reshape(-1).tolist()
        if len(vals) != expected_size:
            raise ValueError(
                f"{name} dimension mismatch. Expected {expected_size}, got {len(vals)}"
            )
        return vals

    def _extract_mesh_data_as_obj(self, mesh_id: int) -> str | None:
        if mesh_id < 0:
            return None
        vert_adr = int(np.asarray(self._model.mesh_vertadr)[mesh_id])
        vert_num = int(np.asarray(self._model.mesh_vertnum)[mesh_id])
        face_adr = int(np.asarray(self._model.mesh_faceadr)[mesh_id])
        face_num = int(np.asarray(self._model.mesh_facenum)[mesh_id])
        verts = np.asarray(self._model.mesh_vert)[vert_adr : vert_adr + vert_num]
        faces = np.asarray(self._model.mesh_face)[face_adr : face_adr + face_num]
        obj_rows: list[str] = []
        for x, y, z in verts:
            obj_rows.append(f"v {x} {y} {z}")
        for a, b, c in faces:
            obj_rows.append(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}")
        return "\n".join(obj_rows)

    def _geom_type_name(self, geom_type: int) -> str | None:
        geom_map = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "Plane",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "Sphere",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "Cylinder",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "Capsule",
            int(mujoco.mjtGeom.mjGEOM_BOX): "Box",
            int(mujoco.mjtGeom.mjGEOM_MESH): "Mesh",
        }
        return geom_map.get(int(geom_type))

    def _build_body_recursive(
        self,
        body_id: int,
        geoms_by_body: dict[int, list[dict[str, Any]]],
        children_by_body: dict[int, list[int]],
    ) -> dict[str, Any]:
        children = [
            self._build_body_recursive(child_id, geoms_by_body, children_by_body)
            for child_id in children_by_body.get(body_id, [])
        ]
        body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return {
            "name": body_name,
            "body_id": body_id,
            "pos": np.asarray(self._data.xpos[body_id], dtype=np.float32).tolist(),
            "quat": np.asarray(self._data.xquat[body_id], dtype=np.float32).tolist(),
            "euler": None,
            "bodies": children if children else None,
            "geoms": geoms_by_body.get(body_id),
        }

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

    def forward_dyn(self, controls: jnp.ndarray) -> None:
        print("Controls array is: ",controls)
        print("Controls dtype & shape is: ",controls.dtype,controls.shape)
        self._mjx_data = self._mjx_data.replace(
            ctrl=controls.reshape(MjxSimulator.nu(self._mjx_data))
        )
        self._mjx_data = mjx.step(self._mjx_model, self._mjx_data)
        # self._mjx_data = mjx.refit_bvh(self._mjx_model, self._mjx_data, render_ctx.pytree())

    @staticmethod
    def forward_dyn_pure(data: Any, model: Any, controls: jnp.ndarray) -> Any:
        """Return the next immutable MJX state without mutating ``data``."""
        data_with_controls = data.replace(ctrl=controls.reshape(data.ctrl.shape))
        return mjx.step(model, data_with_controls)

    @staticmethod
    def get_target_act_length_from_target_diff(
        model: Any,
        data: Any,
        target_diff: jnp.ndarray,
        position_actuator_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Add actuator-length deltas to MJX position-servo lengths."""
        if target_diff.ndim != 1:
            raise ValueError("target_diff must be one-dimensional")
        if position_actuator_indices.ndim != 1:
            raise ValueError("position_actuator_indices must be one-dimensional")
        if target_diff.shape[0] != position_actuator_indices.shape[0]:
            raise ValueError(
                "target_diff length must equal the number of position "
                "actuators: "
                f"expected {position_actuator_indices.shape[0]}, "
                f"got {target_diff.shape[0]}"
            )
        if target_diff.shape[0] == 0:
            return target_diff

        control_indices = (
            model.actuator_ctrladr[position_actuator_indices]
            if hasattr(model, "actuator_ctrladr")
            else position_actuator_indices
        )
        output_indices = (
            model.actuator_outadr[position_actuator_indices]
            if hasattr(model, "actuator_outadr")
            else position_actuator_indices
        )
        target_lengths = data.actuator_length[output_indices] + target_diff
        return jnp.where(
            model.actuator_ctrllimited[control_indices],
            jnp.clip(
                target_lengths,
                model.actuator_ctrlrange[control_indices, 0],
                model.actuator_ctrlrange[control_indices, 1],
            ),
            target_lengths,
        )

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
    def get_target_controls_from_target_velocities(
        data: Any,
        model: Any,
        target_velocities: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert target joint velocities to controls from MJX state.

        This preserves the existing 0.1-second target-velocity horizon and
        assumes each actuator is transmitted directly through a joint.
        """
        controls = target_velocities.reshape(data.ctrl.shape)
        if controls.shape[0] == 0:
            return controls

        actuator_joint_ids = model.actuator_trnid[:, 0]
        qpos_indices = model.jnt_qposadr[actuator_joint_ids]
        dof_indices = model.jnt_dofadr[actuator_joint_ids]
        is_torque_actuator = (
            model.actuator_biastype[0]
            == int(mujoco.mjtBias.mjBIAS_NONE)
        )

        def position_controls(_: None) -> jnp.ndarray:
            return data.qpos[qpos_indices] + controls * 0.1

        def torque_controls(_: None) -> jnp.ndarray:
            desired_qacc = jnp.zeros_like(data.qacc).at[dof_indices].set(
                (controls - data.qvel[dof_indices]) / 0.1
            )
            inverse_forces, _ = MjxSimulator.inverse_dyn_pure(
                data,
                model,
                desired_qacc,
            )
            return inverse_forces[dof_indices]

        return jax.lax.cond(
            is_torque_actuator,
            torque_controls,
            position_controls,
            operand=None,
        )

    def inverse_dyn(self, acc: jnp.ndarray) -> jnp.ndarray:
        forces, self._mjx_data = MjxSimulator.inverse_dyn_pure(
            self._mjx_data,
            self._mjx_model,
            acc,
        )
        return forces

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
        input_spec: ModelInputSpec,
        data: Any,
        model: Any,
        render_ctx: Any = None,
        actuated_qpos_indices: jnp.ndarray | None = None,
        contact_sensor_addresses: jnp.ndarray | None = None,
        contact_body_pairs: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Extract model inputs from caller-owned MJX data."""
        result: dict[str, jnp.ndarray] = {}

        if input_spec.qdof_state:
            if actuated_qpos_indices is None:
                raise ValueError(
                    "qdof_state requires precomputed actuated_qpos_indices"
                )
            result["qdof"] = data.qpos[actuated_qpos_indices]
        if input_spec.qpos_state:
            result["qpos"] = data.qpos[: model.nq]
        if input_spec.qvel_state:
            result["qvel"] = data.qvel[: model.nv]
        if input_spec.xpos_state:
            result["xpos"] = data.xpos[: model.nbody].reshape(-1)
        if input_spec.xquat_state:
            result["xquat"] = data.xquat[: model.nbody].reshape(-1)
        if input_spec.all_cameras and model.ncam > 0:
            if render_ctx is None:
                raise ValueError(
                    "all_cameras requested but no MJX render context was provided"
                )
            rgb_packed, _ = mjx.render(model, data, render_ctx)
            result["cameras"] = jnp.stack(
                [
                    mjx.get_rgb(render_ctx, cam_id, rgb_packed)
                    for cam_id in range(int(model.ncam))
                ],
                axis=0,
            ).reshape(int(model.ncam), _CAM_HEIGHT, _CAM_WIDTH, 3)
        if (
            input_spec.contact_pairs
            and contact_sensor_addresses is not None
            and contact_body_pairs is not None
        ):
            result["contact_pairs"] = Simulator._get_contact_pair_sensor_state(
                data,
                contact_sensor_addresses,
                contact_body_pairs,
                jnp,
            )
        if not result:
            result["qpos"] = data.qpos[: model.nq]
            result["qvel"] = data.qvel[: model.nv]

        return result

    # def get_model_cartesian_state(self) -> jnp.ndarray:
    #     self._sync_cpu_from_mjx()
    #     xpos = np.asarray(self._data.xpos[: self._model.nbody], dtype=np.float64).reshape(-1)
    #     xquat = np.asarray(self._data.xquat[: self._model.nbody], dtype=np.float64).reshape(-1)
    #     return jnp.asarray(np.concatenate([xpos, xquat], axis=0))

    # def get_world_body(self) -> Any | None:
    #     self._sync_cpu_from_mjx()
    #     geoms_by_body: dict[int, list[dict[str, Any]]] = defaultdict(list)
    #     for geom_id in range(int(self._model.ngeom)):
    #         geom_type_val = int(np.asarray(self._model.geom_type)[geom_id])
    #         geom_body_id = int(np.asarray(self._model.geom_bodyid)[geom_id])
    #         geom_type = self._geom_type_name(geom_type_val)
    #         geom_dataid = int(np.asarray(self._model.geom_dataid)[geom_id])
    #         mesh_obj = self._extract_mesh_data_as_obj(geom_dataid) if geom_type == "Mesh" else None
    #         geom_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    #         geom = {
    #             "name": geom_name,
    #             "pos": np.asarray(self._model.geom_pos[geom_id], dtype=np.float32).tolist(),
    #             "quat": np.asarray(self._model.geom_quat[geom_id], dtype=np.float32).tolist(),
    #             "euler": None,
    #             "geom_t": geom_type,
    #             "size": np.asarray(self._model.geom_size[geom_id], dtype=np.float32).tolist(),
    #             "mesh_obj": mesh_obj,
    #             "group": int(np.asarray(self._model.geom_group)[geom_id]),
    #         }
    #         geoms_by_body[geom_body_id].append(geom)

    #     children_by_body: dict[int, list[int]] = defaultdict(list)
    #     body_parentid = np.asarray(self._model.body_parentid)
    #     for body_id in range(1, int(self._model.nbody)):
    #         parent_id = int(body_parentid[body_id])
    #         children_by_body[parent_id].append(body_id)

    #     world_root = self._build_body_recursive(0, geoms_by_body, children_by_body)
    #     return {"bodies": [world_root]}

    # def get_simul_update(self) -> tuple[list[float], list[float]]:
    #     self._sync_cpu_from_mjx()
    #     xpos = np.asarray(self._data.xpos[: self._model.nbody], dtype=np.float64).reshape(-1)
    #     xquat = np.asarray(self._data.xquat[: self._model.nbody], dtype=np.float64).reshape(-1)
    #     return xpos.tolist(), xquat.tolist()

    # def get_contact_body_index_pairs(self) -> list[tuple[int, int]]:
    #     self._sync_cpu_from_mjx()
    #     ncon = int(self._data.ncon)
    #     geom_bodyid = np.asarray(self._model.geom_bodyid)
    #     contact_pairs: list[tuple[int, int]] = []
    #     for i in range(ncon):
    #         contact = self._data.contact[i]
    #         geom1 = int(contact.geom1)
    #         geom2 = int(contact.geom2)
    #         if geom1 < 0 or geom2 < 0:
    #             continue
    #         body1 = int(geom_bodyid[geom1])
    #         body2 = int(geom_bodyid[geom2])
    #         if body1 < 0 or body2 < 0:
    #             continue
    #         contact_pairs.append((body1, body2))
    #     return contact_pairs

    def modify_target_position(self, target_pos_diff: jnp.ndarray) -> jnp.ndarray:
        nu = MjxSimulator.nu(self._data)
        target_diff_vals = self._array_to_list(target_pos_diff, nu, "Target position diff")
        if len(self.target_position) != nu:
            self.target_position = self._get_actuated_qpos_targets()
        updated_targets = [target + diff for target, diff in zip(self.target_position, target_diff_vals)]
        self.target_position = self._clamp_controls_to_actuator_range(updated_targets)
        return jnp.asarray(self.target_position)

    # def get_target_controls_from_target_positions(
    #     self,
    #     target_pos_diff: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     nu = self.nu()
    #     nv = self.nv()
    #     target_diff_vals = self._array_to_list(target_pos_diff, nu, "Target position diff")
    #     actuator_joint_indices = self._get_actuator_joint_indices()
    #     if len(actuator_joint_indices) != nu:
    #         raise ValueError(
    #             f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
    #         )
    #     if not self._is_torque_actuator():
    #         qpos = self.qpos()
    #         ctrl_vals = [
    #             qpos[qpos_idx] + target_diff_vals[i]
    #             for i, (qpos_idx, _) in enumerate(actuator_joint_indices)
    #         ]
    #         ctrl_vals = self._clamp_controls_to_actuator_range(ctrl_vals)
    #         return jnp.asarray(ctrl_vals)

    #     t = 0.1
    #     qvel = self.qvel()
    #     if len(qvel) != nv:
    #         raise ValueError(f"Velocity dimension mismatch. Expected {nv}, got {len(qvel)}")
    #     qacc_des = np.zeros((nv,), dtype=np.float64)
    #     for i, (_, dof_idx) in enumerate(actuator_joint_indices):
    #         dq = target_diff_vals[i]
    #         v = qvel[dof_idx]
    #         qacc_des[dof_idx] = ((dq / t) - v) / t
    #     qfrc_vals = np.asarray(self.inverse_dyn(jnp.asarray(qacc_des)), dtype=np.float64).tolist()
    #     ctrl_vals = [qfrc_vals[dof_idx] for _, dof_idx in actuator_joint_indices]
    #     ctrl_vals = self._clamp_controls_to_actuator_range(ctrl_vals)
    #     return jnp.asarray(ctrl_vals)

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

        model = (
            mujoco.MjModel.from_xml_string(mjcf)
            if mjcf is not None
            else mujoco.MjModel.from_xml_path(model_path)
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return data, model

    # def __init__(
    #     self,
    #     model_path: str | None = None,
    #     mjcf: str | None = None,
    # ) -> None:
    #     if mujoco is None:
    #         raise ModuleNotFoundError(
    #             "mujoco is not installed. Install dependencies with `uv sync`."
    #         )
    #     if (model_path is None and mjcf is None) or (model_path is not None and mjcf is not None):
    #         raise ValueError("Provide exactly one of model_path or mjcf")
    #     if mjcf is not None:
    #         logger.info("MujocoSimulator: loading model from XML string")
    #         self._model = mujoco.MjModel.from_xml_string(mjcf)
    #     else:
    #         logger.info("MujocoSimulator: loading model from path: %s", model_path)
    #         self._model = mujoco.MjModel.from_xml_path(model_path)
    #     logger.info("MujocoSimulator: model loaded (nq=%s nu=%s nv=%s)", self._model.nq, self._model.nu, self._model.nv)
    #     self._data = mujoco.MjData(self._model)
    #     mujoco.mj_forward(self._model, self._data)
    #     logger.info("MujocoSimulator: init complete")
    #     self.target_position: list[float] = self._get_actuated_qpos_targets()

    def _is_torque_actuator(self) -> bool:
        nu = MujocoSimulator.nu(self._data)
        if nu == 0:
            return True
        first_biastype = int(np.asarray(self._model.actuator_biastype)[0])
        return first_biastype == int(mujoco.mjtBias.mjBIAS_NONE)

    def _get_actuator_joint_indices(self) -> list[tuple[int, int]]:
        nu = MujocoSimulator.nu(self._data)
        nq = MujocoSimulator.nq(self._data)
        nv = MujocoSimulator.nv(self._data)
        actuator_trnid = np.asarray(self._model.actuator_trnid).reshape(nu, 2)
        jnt_qposadr = np.asarray(self._model.jnt_qposadr)
        jnt_dofadr = np.asarray(self._model.jnt_dofadr)
        indices: list[tuple[int, int]] = []
        for actuator_idx in range(nu):
            joint_id = int(actuator_trnid[actuator_idx, 0])
            if joint_id < 0:
                raise ValueError(
                    f"Actuator {actuator_idx} is not attached to a valid joint (joint_id={joint_id})"
                )
            qpos_adr = int(jnt_qposadr[joint_id])
            dof_adr = int(jnt_dofadr[joint_id])
            if qpos_adr < 0 or qpos_adr >= nq:
                raise ValueError(
                    f"Invalid qpos address {qpos_adr} for actuator {actuator_idx} (nq={nq})"
                )
            if dof_adr < 0 or dof_adr >= nv:
                raise ValueError(
                    f"Invalid dof address {dof_adr} for actuator {actuator_idx} (nv={nv})"
                )
            indices.append((qpos_adr, dof_adr))
        return indices

    def _clamp_controls_to_actuator_range(self, controls: list[float]) -> list[float]:
        nu = MujocoSimulator.nu(self._data)
        if len(controls) != nu:
            raise ValueError(f"Control dimension mismatch. Expected {nu}, got {len(controls)}")
        ctrl_limited = np.asarray(self._model.actuator_ctrllimited)
        ctrl_range = np.asarray(self._model.actuator_ctrlrange, dtype=np.float64).reshape(nu, 2)
        clamped: list[float] = []
        for i, val in enumerate(controls):
            is_limited = bool(ctrl_limited[i]) if ctrl_limited.size else True
            if not is_limited:
                clamped.append(float(val))
                continue
            min_v = float(ctrl_range[i, 0])
            max_v = float(ctrl_range[i, 1])
            clamped.append(float(np.clip(val, min_v, max_v)))
        return clamped

    def _get_actuated_qpos_targets(self) -> list[float]:
        nu = MujocoSimulator.nu(self._data)
        qpos = self.qpos()
        actuator_joint_indices = self._get_actuator_joint_indices()
        if len(actuator_joint_indices) != nu:
            raise ValueError(
                f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
            )
        targets = [qpos[qpos_idx] for qpos_idx, _ in actuator_joint_indices]
        return self._clamp_controls_to_actuator_range(targets)

    def _array_to_list(
        self,
        x: np.ndarray,
        expected_size: int,
        name: str,
    ) -> list[float]:
        vals = x.reshape(-1).tolist()
        if len(vals) != expected_size:
            raise ValueError(
                f"{name} dimension mismatch. Expected {expected_size}, got {len(vals)}"
            )
        return vals

    def _extract_mesh_data_as_obj(self, mesh_id: int) -> str | None:
        if mesh_id < 0:
            return None
        vert_adr = int(np.asarray(self._model.mesh_vertadr)[mesh_id])
        vert_num = int(np.asarray(self._model.mesh_vertnum)[mesh_id])
        face_adr = int(np.asarray(self._model.mesh_faceadr)[mesh_id])
        face_num = int(np.asarray(self._model.mesh_facenum)[mesh_id])
        verts = np.asarray(self._model.mesh_vert)[vert_adr : vert_adr + vert_num]
        faces = np.asarray(self._model.mesh_face)[face_adr : face_adr + face_num]
        obj_rows: list[str] = []
        for x, y, z in verts:
            obj_rows.append(f"v {x} {y} {z}")
        for a, b, c in faces:
            obj_rows.append(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}")
        return "\n".join(obj_rows)

    def _geom_type_name(self, geom_type: int) -> str | None:
        geom_map = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "Plane",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "Sphere",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "Cylinder",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "Capsule",
            int(mujoco.mjtGeom.mjGEOM_BOX): "Box",
            int(mujoco.mjtGeom.mjGEOM_MESH): "Mesh",
        }
        return geom_map.get(int(geom_type))

    def _build_body_recursive(
        self,
        body_id: int,
        geoms_by_body: dict[int, list[dict[str, Any]]],
        children_by_body: dict[int, list[int]],
    ) -> dict[str, Any]:
        children = [
            self._build_body_recursive(child_id, geoms_by_body, children_by_body)
            for child_id in children_by_body.get(body_id, [])
        ]
        body_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return {
            "name": body_name,
            "body_id": body_id,
            "pos": np.asarray(self._data.xpos[body_id], dtype=np.float32).tolist(),
            "quat": np.asarray(self._data.xquat[body_id], dtype=np.float32).tolist(),
            "euler": None,
            "bodies": children if children else None,
            "geoms": geoms_by_body.get(body_id),
        }

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

    def forward_dyn(self, controls: np.ndarray) -> None:
        ctrl_vals = controls.reshape(-1)
        nu = MujocoSimulator.nu(self._data)
        if len(ctrl_vals) != nu:
            raise ValueError(f"Control dimension mismatch. Expected {nu}, got {len(ctrl_vals)}")
        self._data.ctrl[:nu] = ctrl_vals
        mujoco.mj_step(self._model, self._data)

    @staticmethod
    def forward_dyn_pure(data: Any, model: Any, controls: np.ndarray) -> Any:
        """Return a stepped copy of MuJoCo data without mutating ``data``."""
        data_next = copy.copy(data)
        data_next.ctrl[:] = controls.reshape(data_next.ctrl.shape)
        mujoco.mj_step(model, data_next)
        return data_next

    def inverse_dyn(self, acc: np.ndarray) -> np.ndarray:
        nv = MujocoSimulator.nv(self._data)
        self._data.qacc[:nv] = acc.reshape(nv)
        mujoco.mj_inverse(self._model, self._data)
        return self._data.qfrc_inverse[:nv].copy()

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
        input_spec: ModelInputSpec,
        data: Any,
        model: Any,
        render_ctx: mujoco.Renderer | None = None,
        actuated_qpos_indices: np.ndarray | None = None,
        contact_sensor_addresses: np.ndarray | None = None,
        contact_body_pairs: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract model inputs from caller-owned MuJoCo data."""
        result: dict[str, np.ndarray] = {}

        if input_spec.qdof_state:
            if actuated_qpos_indices is None:
                raise ValueError(
                    "qdof_state requires precomputed actuated_qpos_indices"
                )
            result["qdof"] = data.qpos[actuated_qpos_indices].copy()
        if input_spec.qpos_state:
            result["qpos"] = data.qpos[: model.nq].copy()
        if input_spec.qvel_state:
            result["qvel"] = data.qvel[: model.nv].copy()
        if input_spec.xpos_state:
            result["xpos"] = data.xpos[: model.nbody].reshape(-1).copy()
        if input_spec.xquat_state:
            result["xquat"] = data.xquat[: model.nbody].reshape(-1).copy()
        if input_spec.all_cameras and model.ncam > 0:
            if render_ctx is None:
                raise ValueError(
                    "all_cameras requested but no MuJoCo renderer was provided"
                )
            frames: list[np.ndarray] = []
            for cam_id in range(int(model.ncam)):
                render_ctx.update_scene(data, camera=cam_id)
                frames.append(render_ctx.render().astype(np.float32) / 255.0)
            result["cameras"] = np.stack(frames, axis=0)
        if (
            input_spec.contact_pairs
            and contact_sensor_addresses is not None
            and contact_body_pairs is not None
        ):
            result["contact_pairs"] = Simulator._get_contact_pair_sensor_state(
                data,
                contact_sensor_addresses,
                contact_body_pairs,
                np,
            )
        if not result:
            result["qpos"] = data.qpos[: model.nq].copy()
            result["qvel"] = data.qvel[: model.nv].copy()

        return result

    def qpos(self) -> list[float]:
        return np.asarray(
            self._data.qpos[: MujocoSimulator.nq(self._data)], dtype=np.float64
        ).tolist()

    def qvel(self) -> list[float]:
        return np.asarray(
            self._data.qvel[: MujocoSimulator.nv(self._data)], dtype=np.float64
        ).tolist()

    def _get_renderer(self) -> mujoco.Renderer:
        """Builds a fresh mujoco.Renderer for this model.

        Stateless — the Simulator does not own or cache this. Construction is
        non-trivial (OpenGL/EGL context), so the caller should build one once
        (e.g. before a per-timestep loop) and reuse/close it themselves,
        rather than reconstructing it on every get_model_state call.
        """
        return mujoco.Renderer(self._model, height=_CAM_HEIGHT, width=_CAM_WIDTH)

    # def get_model_cartesian_state(self) -> np.ndarray:
    #     xpos = np.asarray(self._data.xpos[:self._model.nbody], dtype=np.float64).reshape(-1)
    #     xquat = np.asarray(self._data.xquat[:self._model.nbody], dtype=np.float64).reshape(-1)
    #     return jnp.asarray(np.concatenate([xpos, xquat], axis=0))

    # def get_world_body(self) -> Any | None:
    #     geoms_by_body: dict[int, list[dict[str, Any]]] = defaultdict(list)
    #     for geom_id in range(int(self._model.ngeom)):
    #         geom_type_val = int(np.asarray(self._model.geom_type)[geom_id])
    #         geom_body_id = int(np.asarray(self._model.geom_bodyid)[geom_id])
    #         geom_type = self._geom_type_name(geom_type_val)
    #         geom_dataid = int(np.asarray(self._model.geom_dataid)[geom_id])
    #         mesh_obj = self._extract_mesh_data_as_obj(geom_dataid) if geom_type == "Mesh" else None
    #         geom_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    #         geom = {
    #             "name": geom_name,
    #             "pos": np.asarray(self._model.geom_pos[geom_id], dtype=np.float32).tolist(),
    #             "quat": np.asarray(self._model.geom_quat[geom_id], dtype=np.float32).tolist(),
    #             "euler": None,
    #             "geom_t": geom_type,
    #             "size": np.asarray(self._model.geom_size[geom_id], dtype=np.float32).tolist(),
    #             "mesh_obj": mesh_obj,
    #             "group": int(np.asarray(self._model.geom_group)[geom_id]),
    #         }
    #         geoms_by_body[geom_body_id].append(geom)

    #     children_by_body: dict[int, list[int]] = defaultdict(list)
    #     body_parentid = np.asarray(self._model.body_parentid)
    #     for body_id in range(1, int(self._model.nbody)):
    #         parent_id = int(body_parentid[body_id])
    #         children_by_body[parent_id].append(body_id)

    #     world_root = self._build_body_recursive(0, geoms_by_body, children_by_body)
    #     return {"bodies": [world_root]}

    # def get_simul_update(self) -> tuple[list[float], list[float]]:
    #     xpos = np.asarray(self._data.xpos[:self._model.nbody], dtype=np.float64).reshape(-1)
    #     xquat = np.asarray(self._data.xquat[:self._model.nbody], dtype=np.float64).reshape(-1)
    #     return xpos.tolist(), xquat.tolist()

    # def get_contact_body_index_pairs(self) -> list[tuple[int, int]]:
    #     ncon = int(self._data.ncon)
    #     geom_bodyid = np.asarray(self._model.geom_bodyid)
    #     contact_pairs: list[tuple[int, int]] = []
    #     for i in range(ncon):
    #         contact = self._data.contact[i]
    #         geom1 = int(contact.geom1)
    #         geom2 = int(contact.geom2)
    #         if geom1 < 0 or geom2 < 0:
    #             continue
    #         body1 = int(geom_bodyid[geom1])
    #         body2 = int(geom_bodyid[geom2])
    #         if body1 < 0 or body2 < 0:
    #             continue
    #         contact_pairs.append((body1, body2))
    #     return contact_pairs

    def modify_target_position(
        self,
        target_pos_diff: np.ndarray,
    ) -> np.ndarray:
        nu = MujocoSimulator.nu(self._data)
        target_diff_vals = self._array_to_list(target_pos_diff, nu, "Target position diff")
        if len(self.target_position) != nu:
            self.target_position = self._get_actuated_qpos_targets()
        updated_targets = [target + diff for target, diff in zip(self.target_position, target_diff_vals)]
        self.target_position = self._clamp_controls_to_actuator_range(updated_targets)
        return np.asarray(self.target_position)

    def get_target_controls_from_target_positions(
        self,
        target_pos_diff: np.ndarray,
    ) -> np.ndarray:
        nu = MujocoSimulator.nu(self._data)
        nv = MujocoSimulator.nv(self._data)
        target_diff_vals = self._array_to_list(target_pos_diff, nu, "Target position diff")
        actuator_joint_indices = self._get_actuator_joint_indices()
        if len(actuator_joint_indices) != nu:
            raise ValueError(
                f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
            )
        if not self._is_torque_actuator():
            qpos = self.qpos()
            ctrl_vals = [
                qpos[qpos_idx] + target_diff_vals[i]
                for i, (qpos_idx, _) in enumerate(actuator_joint_indices)
            ]
            ctrl_vals = self._clamp_controls_to_actuator_range(ctrl_vals)
            return np.asarray(ctrl_vals)

        t = 0.1
        qvel = self.qvel()
        if len(qvel) != nv:
            raise ValueError(f"Velocity dimension mismatch. Expected {nv}, got {len(qvel)}")
        qacc_des = np.zeros((nv,), dtype=np.float64)
        for i, (_, dof_idx) in enumerate(actuator_joint_indices):
            dq = target_diff_vals[i]
            v = qvel[dof_idx]
            qacc_des[dof_idx] = ((dq / t) - v) / t
        qfrc_vals = self.inverse_dyn(qacc_des).tolist()
        ctrl_vals = [qfrc_vals[dof_idx] for _, dof_idx in actuator_joint_indices]
        ctrl_vals = self._clamp_controls_to_actuator_range(ctrl_vals)
        return np.asarray(ctrl_vals)

    @staticmethod
    def get_target_act_length_from_target_diff(
        model: Any,
        data: Any,
        target_diff: np.ndarray,
        position_actuator_indices: np.ndarray,
    ) -> np.ndarray:
        """Add actuator-length deltas to MuJoCo position-servo lengths."""
        if target_diff.ndim != 1:
            raise ValueError("target_diff must be one-dimensional")
        if position_actuator_indices.ndim != 1:
            raise ValueError("position_actuator_indices must be one-dimensional")
        if target_diff.shape[0] != position_actuator_indices.shape[0]:
            raise ValueError(
                "target_diff length must equal the number of position "
                "actuators: "
                f"expected {position_actuator_indices.shape[0]}, "
                f"got {target_diff.shape[0]}"
            )
        if target_diff.shape[0] == 0:
            return target_diff

        control_indices = (
            model.actuator_ctrladr[position_actuator_indices]
            if hasattr(model, "actuator_ctrladr")
            else position_actuator_indices
        )
        output_indices = (
            model.actuator_outadr[position_actuator_indices]
            if hasattr(model, "actuator_outadr")
            else position_actuator_indices
        )
        target_lengths = data.actuator_length[output_indices] + target_diff
        return np.where(
            model.actuator_ctrllimited[control_indices],
            np.clip(
                target_lengths,
                model.actuator_ctrlrange[control_indices, 0],
                model.actuator_ctrlrange[control_indices, 1],
            ),
            target_lengths,
        )

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

    @staticmethod
    def get_target_controls_from_target_velocities(
        data: Any,
        model: Any,
        target_velocities: np.ndarray,
    ) -> np.ndarray:
        """Convert target joint velocities to controls from MuJoCo state.

        This preserves the existing 0.1-second target-velocity horizon and
        assumes each actuator is transmitted directly through a joint.
        """
        controls = target_velocities.reshape(data.ctrl.shape)
        if controls.shape[0] == 0:
            return controls

        actuator_joint_ids = model.actuator_trnid[:, 0]
        qpos_indices = model.jnt_qposadr[actuator_joint_ids]
        dof_indices = model.jnt_dofadr[actuator_joint_ids]
        is_torque_actuator = (
            int(model.actuator_biastype[0])
            == int(mujoco.mjtBias.mjBIAS_NONE)
        )

        if not is_torque_actuator:
            return data.qpos[qpos_indices] + controls * 0.1

        desired_qacc = np.zeros_like(data.qacc)
        desired_qacc[dof_indices] = (
            controls - data.qvel[dof_indices]
        ) / 0.1
        inverse_forces, _ = MujocoSimulator.inverse_dyn_pure(
            data,
            model,
            desired_qacc,
        )
        return inverse_forces[dof_indices]
