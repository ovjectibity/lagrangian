from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
import numpy as np

try:
    import mujoco
    from mujoco import mjx
except ModuleNotFoundError:
    mujoco = None
    mjx = None


Array = jax.Array


class Simulator(ABC):
    @abstractmethod
    def reset_state(self) -> None:
        pass

    @abstractmethod
    def nq(self) -> int:
        pass

    @abstractmethod
    def nu(self) -> int:
        pass

    @abstractmethod
    def nv(self) -> int:
        pass

    @abstractmethod
    def nbody(self) -> int:
        pass

    @abstractmethod
    def forward_dyn(self, controls: Array) -> None:
        pass

    @abstractmethod
    def inverse_dyn(self, acc: Array) -> Array:
        pass

    @abstractmethod
    def get_model_state(self, input_spec: Any | None = None) -> dict[str, Array]:
        pass

    @abstractmethod
    def get_model_cartesian_state(self) -> Array:
        pass

    @abstractmethod
    def get_world_body(self) -> Any | None:
        pass

    @abstractmethod
    def get_simul_update(self) -> tuple[list[float], list[float]]:
        pass

    @abstractmethod
    def get_contact_body_index_pairs(self) -> list[tuple[int, int]]:
        pass

    @abstractmethod
    def modify_target_position(self, target_pos_diff: Array) -> Array:
        pass

    @abstractmethod
    def get_target_controls_from_target_positions(
        self,
        target_pos_diff: Array,
    ) -> Array:
        pass

    @abstractmethod
    def get_target_controls_from_target_velocities(
        self,
        target_vel_diff: Array,
    ) -> Array:
        pass

    def make_state_fn(self, input_spec: Any) -> Callable[[Any], dict[str, Array]]:
        """Return a pure (mjx_data → dict[str, Array]) closure for use in lax.scan / vmap.

        Only MjxSimulator implements this. MujocoSimulator raises NotImplementedError.
        """
        raise NotImplementedError("make_state_fn is only supported for MjxSimulator")

    def _compute_actuated_qpos_indices(self) -> list[int]:
        """Return qpos indices belonging to each actuated joint.

        For hinge/slide joints: 1 component. Ball: 4. Free: 7.
        Used for qdof_state extraction (generalised coords of actuated joints only).
        """
        nu = self.nu()
        indices: list[int] = []
        actuator_trnid = np.asarray(self._model.actuator_trnid).reshape(nu, 2)  # type: ignore[attr-defined]
        jnt_qposadr = np.asarray(self._model.jnt_qposadr)  # type: ignore[attr-defined]
        jnt_type = np.asarray(self._model.jnt_type)  # type: ignore[attr-defined]
        for actuator_idx in range(nu):
            joint_id = int(actuator_trnid[actuator_idx, 0])
            qpos_adr = int(jnt_qposadr[joint_id])
            jtype = int(jnt_type[joint_id])
            # mjtJoint: 0=free(7), 1=ball(4), 2=slide(1), 3=hinge(1)
            if jtype == 0:
                n = 7
            elif jtype == 1:
                n = 4
            else:
                n = 1
            indices.extend(range(qpos_adr, qpos_adr + n))
        return indices


# Camera render resolution (hardcoded; to be standardised in training graph later).
_CAM_HEIGHT = 244
_CAM_WIDTH = 244


class MjxSimulator(Simulator):
    def __init__(
        self,
        model_path: str | None = None,
        mjcf: str | None = None,
    ) -> None:
        if mujoco is None or mjx is None:
            raise ModuleNotFoundError(
                "mujoco/mjx is not installed. Install dependencies from requirements.txt."
            )
        if (model_path is None and mjcf is None) or (model_path is not None and mjcf is not None):
            raise ValueError("Provide exactly one of model_path or mjcf")
        if mjcf is not None:
            logger.info("MjxSimulator: loading model from XML string")
            self._model = mujoco.MjModel.from_xml_string(mjcf)
        else:
            logger.info("MjxSimulator: loading model from path: %s", model_path)
            self._model = mujoco.MjModel.from_xml_path(model_path)
        logger.info(
            "MjxSimulator: model loaded (nq=%s nu=%s nv=%s ngeom=%s nbody=%s nconmax=%s npair=%s)",
            self._model.nq, self._model.nu, self._model.nv,
            self._model.ngeom, self._model.nbody, self._model.nconmax, self._model.npair,
        )
        logger.info("MjxSimulator: creating MjData")
        self._data = mujoco.MjData(self._model)
        logger.info("MjxSimulator: running mj_forward")
        mujoco.mj_forward(self._model, self._data)
        logger.info(
            "MjxSimulator: after mj_forward — ncon=%s (allocated contact buffer size=%s)",
            self._data.ncon, self._data.contact.dist.shape[0],
        )
        logger.info("MjxSimulator: calling mjx.put_model")
        self._mjx_model = mjx.put_model(self._model)
        logger.info("MjxSimulator: calling mjx.put_data")
        self._mjx_data = mjx.put_data(self._model, self._data)
        logger.info("MjxSimulator: init complete")

        # Benchmark: isolate mjx.step timing
        logger.info("MjxSimulator: running mjx.step benchmark...")
        _bench_step = jax.jit(lambda d: mjx.step(self._mjx_model, d))
        t0 = time.time()
        _bench_lowered = _bench_step.lower(self._mjx_data)
        t1 = time.time()
        logger.info("MjxSimulator: bench — lowering took %.2fs", t1 - t0)
        _bench_compiled = _bench_lowered.compile()
        t2 = time.time()
        logger.info("MjxSimulator: bench — XLA compile took %.2fs", t2 - t1)
        _bench_result = _bench_compiled(self._mjx_data)
        jax.block_until_ready(_bench_result)
        t3 = time.time()
        logger.info("MjxSimulator: bench — first execution took %.2fs", t3 - t2)
        for _ in range(10):
            _bench_result = _bench_compiled(_bench_result)
        jax.block_until_ready(_bench_result)
        t4 = time.time()
        logger.info("MjxSimulator: bench — 10 steps took %.2fs (%.4fs/step)", t4 - t3, (t4 - t3) / 10)
        self.target_position: list[float] = self._get_actuated_qpos_targets()
        self._actuated_qpos_idx: list[int] = self._compute_actuated_qpos_indices()
        self._actuated_qpos_idx_jnp: Array = jnp.array(self._actuated_qpos_idx, dtype=jnp.int32)

    def _sync_cpu_from_mjx(self) -> None:
        self._data = mjx.get_data(self._model, self._mjx_data)

    def _sync_mjx_from_cpu(self) -> None:
        self._mjx_data = mjx.put_data(self._model, self._data)

    def qpos(self) -> list[float]:
        self._sync_cpu_from_mjx()
        return np.asarray(self._data.qpos[: self.nq()], dtype=np.float64).tolist()

    def qvel(self) -> list[float]:
        self._sync_cpu_from_mjx()
        return np.asarray(self._data.qvel[: self.nv()], dtype=np.float64).tolist()

    def _is_torque_actuator(self) -> bool:
        nu = self.nu()
        if nu == 0:
            return True
        first_biastype = int(np.asarray(self._model.actuator_biastype)[0])
        return first_biastype == int(mujoco.mjtBias.mjBIAS_NONE)

    def _get_actuator_joint_indices(self) -> list[tuple[int, int]]:
        nu = self.nu()
        nq = self.nq()
        nv = self.nv()
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
        nu = self.nu()
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
        nu = self.nu()
        qpos = self.qpos()
        actuator_joint_indices = self._get_actuator_joint_indices()
        if len(actuator_joint_indices) != nu:
            raise ValueError(
                f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
            )
        targets = [qpos[qpos_idx] for qpos_idx, _ in actuator_joint_indices]
        return self._clamp_controls_to_actuator_range(targets)

    def _array_to_list(self, x: Array, expected_size: int, name: str) -> list[float]:
        vals = np.asarray(jax.device_get(x), dtype=np.float64).reshape(-1).tolist()
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

    def reset_state(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._sync_mjx_from_cpu()
        self.target_position = self._get_actuated_qpos_targets()

    def nq(self) -> int:
        return int(self._model.nq)

    def nu(self) -> int:
        return int(self._model.nu)

    def nv(self) -> int:
        return int(self._model.nv)

    def nbody(self) -> int:
        return int(self._model.nbody)

    def forward_dyn(self, controls: Array) -> None:
        print("Controls array is: ",controls)
        print("Controls dtype & shape is: ",controls.dtype,controls.shape)
        self._mjx_data = self._mjx_data.replace(ctrl=controls.reshape(self.nu()))
        self._mjx_data = mjx.step(self._mjx_model, self._mjx_data)

    def forward_dyn_pure(
        self,
        mjx_data: Any,
        controls: Array,
    ) -> tuple[Any, Array, Array, Array]:
        """Pure functional step: takes mjx_data + controls, returns (mjx_data_next, wstate, cstate, contact_matrix).

        contact_matrix is a float32 [nbody, nbody] array with 1.0 for active contact pairs.
        Does not mutate self._mjx_data.
        """
        nq = self.nq()
        nu = self.nu()
        nv = self.nv()
        nbody = self.nbody()
        geom_bodyid = jnp.asarray(self._model.geom_bodyid)

        mjx_data_next = mjx.step(self._mjx_model, mjx_data.replace(ctrl=controls.reshape(nu)))

        wstate = jnp.concatenate([
            mjx_data_next.qpos[:nq],
            mjx_data_next.qvel[:nv],
        ])
        xpos = mjx_data_next.xpos[:nbody].reshape(-1)
        xquat = mjx_data_next.xquat[:nbody].reshape(-1)
        cstate = jnp.concatenate([xpos, xquat])

        geom1 = mjx_data_next.contact.geom1
        geom2 = mjx_data_next.contact.geom2
        body1 = geom_bodyid[geom1]
        body2 = geom_bodyid[geom2]
        contact_matrix = jnp.zeros((nbody, nbody), dtype=jnp.float32)
        contact_matrix = contact_matrix.at[body1, body2].set(1.0)
        contact_matrix = contact_matrix.at[body2, body1].set(1.0)

        return mjx_data_next, wstate, cstate, contact_matrix

    def inverse_dyn(self, acc: Array) -> Array:
        self._sync_cpu_from_mjx()
        self._data.qacc[: self.nv()] = np.asarray(jax.device_get(acc), dtype=np.float64).reshape(self.nv())
        mujoco.mj_inverse(self._model, self._data)
        return jnp.asarray(np.asarray(self._data.qfrc_inverse[: self.nv()], dtype=np.float64))

    def get_model_state(self, input_spec: Any | None = None) -> dict[str, Array]:
        self._sync_cpu_from_mjx()
        result: dict[str, Array] = {}
        if input_spec is None:
            result["qpos"] = jnp.asarray(np.asarray(self._data.qpos[: self.nq()], dtype=np.float64))
            result["qvel"] = jnp.asarray(np.asarray(self._data.qvel[: self.nv()], dtype=np.float64))
            return result
        if input_spec.qdof_state:
            qpos_all = np.asarray(self._data.qpos, dtype=np.float64)
            result["qdof"] = jnp.asarray(qpos_all[self._actuated_qpos_idx])
        if input_spec.qpos_state:
            result["qpos"] = jnp.asarray(np.asarray(self._data.qpos[: self.nq()], dtype=np.float64))
        if input_spec.qvel_state:
            result["qvel"] = jnp.asarray(np.asarray(self._data.qvel[: self.nv()], dtype=np.float64))
        if input_spec.xpos_state:
            result["xpos"] = jnp.asarray(
                np.asarray(self._data.xpos[: self.nbody()], dtype=np.float64).reshape(-1)
            )
        if input_spec.xquat_state:
            result["xquat"] = jnp.asarray(
                np.asarray(self._data.xquat[: self.nbody()], dtype=np.float64).reshape(-1)
            )
        if input_spec.all_cameras and self._model.ncam > 0:
            ncam = int(self._model.ncam)
            frames = jax.vmap(
                lambda cam_id: mjx.render(
                    self._mjx_model, self._mjx_data,
                    camera=cam_id, width=_CAM_WIDTH, height=_CAM_HEIGHT,
                )
            )(jnp.arange(ncam))
            result["cameras"] = (frames.astype(jnp.float32) / 255.0).reshape(-1)
        if not result:
            result["qpos"] = jnp.asarray(np.asarray(self._data.qpos[: self.nq()], dtype=np.float64))
            result["qvel"] = jnp.asarray(np.asarray(self._data.qvel[: self.nv()], dtype=np.float64))
        return result

    def make_state_fn(self, input_spec: Any) -> Callable[[Any], dict[str, Array]]:
        actuated_idx = self._actuated_qpos_idx_jnp
        nq = self.nq()
        nv = self.nv()
        nbody = self.nbody()
        mjx_model = self._mjx_model
        ncam = int(self._model.ncam)
        all_cameras = bool(input_spec.all_cameras)
        qdof_state = bool(input_spec.qdof_state)
        qpos_state = bool(input_spec.qpos_state)
        qvel_state = bool(input_spec.qvel_state)
        xpos_state = bool(input_spec.xpos_state)
        xquat_state = bool(input_spec.xquat_state)

        def state_fn(data: Any) -> dict[str, Array]:
            result: dict[str, Array] = {}
            if qdof_state:
                result["qdof"] = data.qpos[actuated_idx]
            if qpos_state:
                result["qpos"] = data.qpos[:nq]
            if qvel_state:
                result["qvel"] = data.qvel[:nv]
            if xpos_state:
                result["xpos"] = data.xpos[:nbody].reshape(-1)
            if xquat_state:
                result["xquat"] = data.xquat[:nbody].reshape(-1)
            if all_cameras and ncam > 0:
                frames = jax.vmap(
                    lambda cam_id: mjx.render(
                        mjx_model, data,
                        camera=cam_id, width=_CAM_WIDTH, height=_CAM_HEIGHT,
                    )
                )(jnp.arange(ncam))
                result["cameras"] = (frames.astype(jnp.float32) / 255.0).reshape(-1)
            if not result:
                result["qpos"] = data.qpos[:nq]
                result["qvel"] = data.qvel[:nv]
            return result

        return state_fn

    def get_model_cartesian_state(self) -> Array:
        self._sync_cpu_from_mjx()
        xpos = np.asarray(self._data.xpos[: self._model.nbody], dtype=np.float64).reshape(-1)
        xquat = np.asarray(self._data.xquat[: self._model.nbody], dtype=np.float64).reshape(-1)
        return jnp.asarray(np.concatenate([xpos, xquat], axis=0))

    def get_world_body(self) -> Any | None:
        self._sync_cpu_from_mjx()
        geoms_by_body: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for geom_id in range(int(self._model.ngeom)):
            geom_type_val = int(np.asarray(self._model.geom_type)[geom_id])
            geom_body_id = int(np.asarray(self._model.geom_bodyid)[geom_id])
            geom_type = self._geom_type_name(geom_type_val)
            geom_dataid = int(np.asarray(self._model.geom_dataid)[geom_id])
            mesh_obj = self._extract_mesh_data_as_obj(geom_dataid) if geom_type == "Mesh" else None
            geom_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom = {
                "name": geom_name,
                "pos": np.asarray(self._model.geom_pos[geom_id], dtype=np.float32).tolist(),
                "quat": np.asarray(self._model.geom_quat[geom_id], dtype=np.float32).tolist(),
                "euler": None,
                "geom_t": geom_type,
                "size": np.asarray(self._model.geom_size[geom_id], dtype=np.float32).tolist(),
                "mesh_obj": mesh_obj,
                "group": int(np.asarray(self._model.geom_group)[geom_id]),
            }
            geoms_by_body[geom_body_id].append(geom)

        children_by_body: dict[int, list[int]] = defaultdict(list)
        body_parentid = np.asarray(self._model.body_parentid)
        for body_id in range(1, int(self._model.nbody)):
            parent_id = int(body_parentid[body_id])
            children_by_body[parent_id].append(body_id)

        world_root = self._build_body_recursive(0, geoms_by_body, children_by_body)
        return {"bodies": [world_root]}

    def get_simul_update(self) -> tuple[list[float], list[float]]:
        self._sync_cpu_from_mjx()
        xpos = np.asarray(self._data.xpos[: self._model.nbody], dtype=np.float64).reshape(-1)
        xquat = np.asarray(self._data.xquat[: self._model.nbody], dtype=np.float64).reshape(-1)
        return xpos.tolist(), xquat.tolist()

    def get_contact_body_index_pairs(self) -> list[tuple[int, int]]:
        self._sync_cpu_from_mjx()
        ncon = int(self._data.ncon)
        geom_bodyid = np.asarray(self._model.geom_bodyid)
        contact_pairs: list[tuple[int, int]] = []
        for i in range(ncon):
            contact = self._data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 < 0 or geom2 < 0:
                continue
            body1 = int(geom_bodyid[geom1])
            body2 = int(geom_bodyid[geom2])
            if body1 < 0 or body2 < 0:
                continue
            contact_pairs.append((body1, body2))
        return contact_pairs

    def modify_target_position(self, target_pos_diff: Array) -> Array:
        target_diff_vals = self._array_to_list(target_pos_diff, self.nu(), "Target position diff")
        if len(self.target_position) != self.nu():
            self.target_position = self._get_actuated_qpos_targets()
        updated_targets = [target + diff for target, diff in zip(self.target_position, target_diff_vals)]
        self.target_position = self._clamp_controls_to_actuator_range(updated_targets)
        return jnp.asarray(self.target_position)

    def get_target_controls_from_target_positions(self, target_pos_diff: Array) -> Array:
        nu = self.nu()
        nv = self.nv()
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
            return jnp.asarray(ctrl_vals)

        t = 0.1
        qvel = self.qvel()
        if len(qvel) != nv:
            raise ValueError(f"Velocity dimension mismatch. Expected {nv}, got {len(qvel)}")
        qacc_des = np.zeros((nv,), dtype=np.float64)
        for i, (_, dof_idx) in enumerate(actuator_joint_indices):
            dq = target_diff_vals[i]
            v = qvel[dof_idx]
            qacc_des[dof_idx] = ((dq / t) - v) / t
        qfrc_vals = np.asarray(self.inverse_dyn(jnp.asarray(qacc_des)), dtype=np.float64).tolist()
        ctrl_vals = [qfrc_vals[dof_idx] for _, dof_idx in actuator_joint_indices]
        ctrl_vals = self._clamp_controls_to_actuator_range(ctrl_vals)
        return jnp.asarray(ctrl_vals)

    def get_target_controls_from_target_velocities(self, target_vel_diff: Array) -> Array:
        nu = self.nu()
        nv = self.nv()
        target_vel_vals = self._array_to_list(target_vel_diff, nu, "Target velocity")
        actuator_joint_indices = self._get_actuator_joint_indices()
        if len(actuator_joint_indices) != nu:
            raise ValueError(
                f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
            )
        if not self._is_torque_actuator():
            qpos = self.qpos()
            t = 0.1
            ctrl_vals = [
                qpos[qpos_idx] + target_vel_vals[i] * t
                for i, (qpos_idx, _) in enumerate(actuator_joint_indices)
            ]
            return jnp.asarray(ctrl_vals)

        qvel = self.qvel()
        if len(qvel) != nv:
            raise ValueError(f"Velocity dimension mismatch. Expected {nv}, got {len(qvel)}")
        t = 0.1
        qacc_des = np.zeros((nv,), dtype=np.float64)
        for i, (_, dof_idx) in enumerate(actuator_joint_indices):
            target_v = target_vel_vals[i]
            v_curr = qvel[dof_idx]
            qacc_des[dof_idx] = (target_v - v_curr) / t
        qfrc_vals = np.asarray(self.inverse_dyn(jnp.asarray(qacc_des)), dtype=np.float64).tolist()
        ctrl_vals = [qfrc_vals[dof_idx] for _, dof_idx in actuator_joint_indices]
        return jnp.asarray(ctrl_vals)


class MujocoSimulator(Simulator):
    def __init__(
        self,
        model_path: str | None = None,
        mjcf: str | None = None,
    ) -> None:
        if mujoco is None:
            raise ModuleNotFoundError(
                "mujoco is not installed. Install dependencies from requirements.txt."
            )
        if (model_path is None and mjcf is None) or (model_path is not None and mjcf is not None):
            raise ValueError("Provide exactly one of model_path or mjcf")
        if mjcf is not None:
            logger.info("MujocoSimulator: loading model from XML string")
            self._model = mujoco.MjModel.from_xml_string(mjcf)
        else:
            logger.info("MujocoSimulator: loading model from path: %s", model_path)
            self._model = mujoco.MjModel.from_xml_path(model_path)
        logger.info("MujocoSimulator: model loaded (nq=%s nu=%s nv=%s)", self._model.nq, self._model.nu, self._model.nv)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        logger.info("MujocoSimulator: init complete")
        self.target_position: list[float] = self._get_actuated_qpos_targets()
        self._actuated_qpos_idx: list[int] = self._compute_actuated_qpos_indices()

    def _is_torque_actuator(self) -> bool:
        nu = self.nu()
        if nu == 0:
            return True
        first_biastype = int(np.asarray(self._model.actuator_biastype)[0])
        return first_biastype == int(mujoco.mjtBias.mjBIAS_NONE)

    def _get_actuator_joint_indices(self) -> list[tuple[int, int]]:
        nu = self.nu()
        nq = self.nq()
        nv = self.nv()
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
        nu = self.nu()
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
        nu = self.nu()
        qpos = self.qpos()
        actuator_joint_indices = self._get_actuator_joint_indices()
        if len(actuator_joint_indices) != nu:
            raise ValueError(
                f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
            )
        targets = [qpos[qpos_idx] for qpos_idx, _ in actuator_joint_indices]
        return self._clamp_controls_to_actuator_range(targets)

    def _array_to_list(self, x: Array, expected_size: int, name: str) -> list[float]:
        vals = np.asarray(jax.device_get(x), dtype=np.float64).reshape(-1).tolist()
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

    def reset_state(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self.target_position = self._get_actuated_qpos_targets()

    def nq(self) -> int:
        return int(self._model.nq)

    def nu(self) -> int:
        return int(self._model.nu)

    def nv(self) -> int:
        return int(self._model.nv)

    def nbody(self) -> int:
        return int(self._model.nbody)

    def forward_dyn(self, controls: Array) -> None:
        ctrl_vals = np.asarray(jax.device_get(controls), dtype=np.float64).reshape(-1).tolist()
        if len(ctrl_vals) != self.nu():
            raise ValueError(f"Control dimension mismatch. Expected {self.nu()}, got {len(ctrl_vals)}")
        self._data.ctrl[:self.nu()] = ctrl_vals
        mujoco.mj_step(self._model, self._data)

    def inverse_dyn(self, acc: Array) -> Array:
        self._data.qacc[:self.nv()] = np.asarray(jax.device_get(acc), dtype=np.float64).reshape(self.nv())
        mujoco.mj_inverse(self._model, self._data)
        return jnp.asarray(np.asarray(self._data.qfrc_inverse[:self.nv()], dtype=np.float64))

    def qpos(self) -> list[float]:
        return np.asarray(self._data.qpos[:self.nq()], dtype=np.float64).tolist()

    def qvel(self) -> list[float]:
        return np.asarray(self._data.qvel[:self.nv()], dtype=np.float64).tolist()

    def get_model_state(self, input_spec: Any | None = None) -> dict[str, Array]:
        result: dict[str, Array] = {}
        if input_spec is None:
            result["qpos"] = jnp.asarray(np.asarray(self._data.qpos[: self.nq()], dtype=np.float64))
            result["qvel"] = jnp.asarray(np.asarray(self._data.qvel[: self.nv()], dtype=np.float64))
            return result
        if input_spec.qdof_state:
            qpos_all = np.asarray(self._data.qpos, dtype=np.float64)
            result["qdof"] = jnp.asarray(qpos_all[self._actuated_qpos_idx])
        if input_spec.qpos_state:
            result["qpos"] = jnp.asarray(np.asarray(self._data.qpos[: self.nq()], dtype=np.float64))
        if input_spec.qvel_state:
            result["qvel"] = jnp.asarray(np.asarray(self._data.qvel[: self.nv()], dtype=np.float64))
        if input_spec.xpos_state:
            result["xpos"] = jnp.asarray(
                np.asarray(self._data.xpos[: self.nbody()], dtype=np.float64).reshape(-1)
            )
        if input_spec.xquat_state:
            result["xquat"] = jnp.asarray(
                np.asarray(self._data.xquat[: self.nbody()], dtype=np.float64).reshape(-1)
            )
        if input_spec.all_cameras and self._model.ncam > 0:
            ncam = int(self._model.ncam)
            frames: list[np.ndarray] = []
            with mujoco.Renderer(self._model, height=_CAM_HEIGHT, width=_CAM_WIDTH) as renderer:
                for cam_id in range(ncam):
                    renderer.update_scene(self._data, camera=cam_id)
                    frames.append(renderer.render().astype(np.float32) / 255.0)
            result["cameras"] = jnp.asarray(
                np.concatenate([f.reshape(-1) for f in frames], axis=0)
                if frames else np.zeros((_CAM_HEIGHT * _CAM_WIDTH * 3,), dtype=np.float32)
            )
        if not result:
            result["qpos"] = jnp.asarray(np.asarray(self._data.qpos[: self.nq()], dtype=np.float64))
            result["qvel"] = jnp.asarray(np.asarray(self._data.qvel[: self.nv()], dtype=np.float64))
        return result

    def get_model_cartesian_state(self) -> Array:
        xpos = np.asarray(self._data.xpos[:self._model.nbody], dtype=np.float64).reshape(-1)
        xquat = np.asarray(self._data.xquat[:self._model.nbody], dtype=np.float64).reshape(-1)
        return jnp.asarray(np.concatenate([xpos, xquat], axis=0))

    def get_world_body(self) -> Any | None:
        geoms_by_body: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for geom_id in range(int(self._model.ngeom)):
            geom_type_val = int(np.asarray(self._model.geom_type)[geom_id])
            geom_body_id = int(np.asarray(self._model.geom_bodyid)[geom_id])
            geom_type = self._geom_type_name(geom_type_val)
            geom_dataid = int(np.asarray(self._model.geom_dataid)[geom_id])
            mesh_obj = self._extract_mesh_data_as_obj(geom_dataid) if geom_type == "Mesh" else None
            geom_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom = {
                "name": geom_name,
                "pos": np.asarray(self._model.geom_pos[geom_id], dtype=np.float32).tolist(),
                "quat": np.asarray(self._model.geom_quat[geom_id], dtype=np.float32).tolist(),
                "euler": None,
                "geom_t": geom_type,
                "size": np.asarray(self._model.geom_size[geom_id], dtype=np.float32).tolist(),
                "mesh_obj": mesh_obj,
                "group": int(np.asarray(self._model.geom_group)[geom_id]),
            }
            geoms_by_body[geom_body_id].append(geom)

        children_by_body: dict[int, list[int]] = defaultdict(list)
        body_parentid = np.asarray(self._model.body_parentid)
        for body_id in range(1, int(self._model.nbody)):
            parent_id = int(body_parentid[body_id])
            children_by_body[parent_id].append(body_id)

        world_root = self._build_body_recursive(0, geoms_by_body, children_by_body)
        return {"bodies": [world_root]}

    def get_simul_update(self) -> tuple[list[float], list[float]]:
        xpos = np.asarray(self._data.xpos[:self._model.nbody], dtype=np.float64).reshape(-1)
        xquat = np.asarray(self._data.xquat[:self._model.nbody], dtype=np.float64).reshape(-1)
        return xpos.tolist(), xquat.tolist()

    def get_contact_body_index_pairs(self) -> list[tuple[int, int]]:
        ncon = int(self._data.ncon)
        geom_bodyid = np.asarray(self._model.geom_bodyid)
        contact_pairs: list[tuple[int, int]] = []
        for i in range(ncon):
            contact = self._data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 < 0 or geom2 < 0:
                continue
            body1 = int(geom_bodyid[geom1])
            body2 = int(geom_bodyid[geom2])
            if body1 < 0 or body2 < 0:
                continue
            contact_pairs.append((body1, body2))
        return contact_pairs

    def modify_target_position(self, target_pos_diff: Array) -> Array:
        target_diff_vals = self._array_to_list(target_pos_diff, self.nu(), "Target position diff")
        if len(self.target_position) != self.nu():
            self.target_position = self._get_actuated_qpos_targets()
        updated_targets = [target + diff for target, diff in zip(self.target_position, target_diff_vals)]
        self.target_position = self._clamp_controls_to_actuator_range(updated_targets)
        return jnp.asarray(self.target_position)

    def get_target_controls_from_target_positions(self, target_pos_diff: Array) -> Array:
        nu = self.nu()
        nv = self.nv()
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
            return jnp.asarray(ctrl_vals)

        t = 0.1
        qvel = self.qvel()
        if len(qvel) != nv:
            raise ValueError(f"Velocity dimension mismatch. Expected {nv}, got {len(qvel)}")
        qacc_des = np.zeros((nv,), dtype=np.float64)
        for i, (_, dof_idx) in enumerate(actuator_joint_indices):
            dq = target_diff_vals[i]
            v = qvel[dof_idx]
            qacc_des[dof_idx] = ((dq / t) - v) / t
        qfrc_vals = np.asarray(self.inverse_dyn(jnp.asarray(qacc_des)), dtype=np.float64).tolist()
        ctrl_vals = [qfrc_vals[dof_idx] for _, dof_idx in actuator_joint_indices]
        ctrl_vals = self._clamp_controls_to_actuator_range(ctrl_vals)
        return jnp.asarray(ctrl_vals)

    def get_target_controls_from_target_velocities(self, target_vel_diff: Array) -> Array:
        nu = self.nu()
        nv = self.nv()
        target_vel_vals = self._array_to_list(target_vel_diff, nu, "Target velocity")
        actuator_joint_indices = self._get_actuator_joint_indices()
        if len(actuator_joint_indices) != nu:
            raise ValueError(
                f"Actuator index mapping mismatch. Expected {nu}, got {len(actuator_joint_indices)}"
            )
        if not self._is_torque_actuator():
            qpos = self.qpos()
            t = 0.1
            ctrl_vals = [
                qpos[qpos_idx] + target_vel_vals[i] * t
                for i, (qpos_idx, _) in enumerate(actuator_joint_indices)
            ]
            return jnp.asarray(ctrl_vals)

        qvel = self.qvel()
        if len(qvel) != nv:
            raise ValueError(f"Velocity dimension mismatch. Expected {nv}, got {len(qvel)}")
        t = 0.1
        qacc_des = np.zeros((nv,), dtype=np.float64)
        for i, (_, dof_idx) in enumerate(actuator_joint_indices):
            target_v = target_vel_vals[i]
            v_curr = qvel[dof_idx]
            qacc_des[dof_idx] = (target_v - v_curr) / t
        qfrc_vals = np.asarray(self.inverse_dyn(jnp.asarray(qacc_des)), dtype=np.float64).tolist()
        ctrl_vals = [qfrc_vals[dof_idx] for _, dof_idx in actuator_joint_indices]
        return jnp.asarray(ctrl_vals)
