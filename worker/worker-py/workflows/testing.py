from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from commondef import testing_updates_pb2 as pb
from commondef.remoteworkermessages import (
    RemoteWorkerClientMessageStopTestingMessage,
    StopTestMessage,
    TrainingAccelerationMode,
)
from commondef.testing import (
    DatasetLibero,
    LiberoDatasetType,
    TestJobController,
    TestMethodCustom,
    TestMethodDataset,
)
from commondef.training import ControlModelVLA, ModelInputSpec
from datasets.libero.libero import get_task_mjcfs
from models.mlp import MLPModel
from models.model import TestingModel
from models.pi0.pi0 import Pi0Model
from simulator.simulator import MjxSimulator, MujocoSimulator


logger = logging.getLogger(__name__)

Array = jax.Array

UPDATE_CHUNK_SIZE = 10


def _build_stream_bytes(
    test_id: str,
    task_sim_states: list[tuple[int, list[list[tuple[Any, Any]]]]],
    task_frames_u8: list[tuple[int, Any]] | None,
) -> bytes:
    """Build a `TestingStreamMessage` from task-major pose and frame updates.

    `task_sim_states` has shape [task][env][t], with each leaf containing an
    `(xpos, xquat)` pair. `task_frames_u8`, if present, contains one
    `(task_index, frames)` entry per task with camera output, where `frames`
    has shape [env, t, camera, height, width, channel]. Protobuf's `bytes`
    field receives each contiguous raw u8 buffer.
    """
    update = pb.TestUpdate(
        id=test_id,
        simulation_state=[
            pb.TaskUpdates(
                task_index=task_index,
                environments=[
                    pb.EnvUpdates(updates=[
                        pb.ModelForward(
                            nbody=len(xpos) // 3,
                            xpos=xpos,
                            xquat=xquat,
                        )
                        for xpos, xquat in env_updates
                    ])
                    for env_updates in task_sim_state
                ],
            )
            for task_index, task_sim_state in task_sim_states
        ],
    )
    if task_frames_u8 is not None:
        state_task_indices = {task_index for task_index, _ in task_sim_states}
        frame_task_indices: set[int] = set()
        for task_index, frames_u8 in task_frames_u8:
            if task_index not in state_task_indices:
                raise ValueError(
                    "Camera frames must reference a task present in simulation state"
                )
            if task_index in frame_task_indices:
                raise ValueError("Camera frames may contain only one entry per task")
            if len(frames_u8.shape) != 6:
                raise ValueError(
                    "Camera frames must have shape "
                    "[env, t, camera, height, width, channel]"
                )
            frame_task_indices.add(task_index)
            nenv, t, ncam, h, w = (int(d) for d in frames_u8.shape[:5])
            update.frames.add(
                task_index=task_index,
                nenv=nenv,
                t=t,
                ncam=ncam,
                h=h,
                w=w,
                frames=np.ascontiguousarray(frames_u8).tobytes(),
            )
    return pb.TestingStreamMessage(test_update=update).SerializeToString()


class TestingTaskSink(ABC):
    @abstractmethod
    def send_test_updates(self, payload: bytes) -> None:
        pass

    @abstractmethod
    def send_stop_testing(self, message: StopTestMessage) -> None:
        pass


class CallbackTestingTaskSink(TestingTaskSink):
    def __init__(self, send_message: Callable[[dict[str, Any] | bytes], None]) -> None:
        self._send_message = send_message

    def send_test_updates(self, payload: bytes) -> None:
        self._send_message(payload)

    def send_stop_testing(self, message: StopTestMessage) -> None:
        wrapped = RemoteWorkerClientMessageStopTestingMessage(content=message)
        self._send_message(wrapped.model_dump(mode="json", by_alias=True))


class TestingWorkflow:
    def __init__(
        self,
        test_id: str,
        test_controller: TestJobController,
        model_path: str | None = None,
        rng_key: jax.Array | None = None,
    ) -> None:
        self.id = test_id
        self.test_controller = test_controller
        self.simulator_states: dict[int | None, dict[str, Any]] = {}

        try:
            test_method = test_controller.test_method
            task_mjcfs: dict[int, str] = {}
            if isinstance(test_method, TestMethodCustom):
                if model_path is None:
                    raise ValueError("Custom testing requires a simulation model path")
                params = test_method.content.test_params
                control_model = test_method.content.control_model
                self.num_envs_per_task = int(params.num_parallel_envs)
                self.num_timesteps = int(params.num_timesteps)
            elif isinstance(test_method, TestMethodDataset):
                dataset = test_method.content.dataset
                if not isinstance(dataset, DatasetLibero):
                    raise ValueError("Unsupported testing dataset")
                params = dataset.content
                control_model = test_method.content.control_model
                self.num_envs_per_task = int(params.num_envs_per_task)
                self.num_timesteps = int(params.num_timesteps)

                task_indices = list(params.tasks_index)
                if len(task_indices) != len(set(task_indices)):
                    raise ValueError("LIBERO task indices must be unique")

                dataset_names = {
                    LiberoDatasetType.LIBERO10: "libero_10",
                    LiberoDatasetType.LIBERO90: "libero_90",
                    LiberoDatasetType.LIBEROGOAL: "libero_goal",
                    LiberoDatasetType.LIBEROOBJECT: "libero_object",
                    LiberoDatasetType.LIBEROSPATIAL: "libero_spatial",
                }
                dataset_name = dataset_names[params.libero_dataset_type]
                mjcfs = get_task_mjcfs(task_indices, dataset_name)
                task_mjcfs = dict(zip(task_indices, mjcfs, strict=True))
            else:
                raise ValueError("Unsupported testing method")
            if self.num_envs_per_task < 1:
                raise ValueError("num_envs_per_task must be at least 1")

            root_rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
            model_rng, self._rng_key = jax.random.split(root_rng)
            self.model = self._build_model_for_testing(
                control_model,
                rng_key=model_rng,
            )

            control_spec = self.model.get_control_input_spec()
            mode = self.test_controller.acceleration_mode

            def initialize_simulator_state(
                *,
                source_model_path: str | None = None,
                mjcf: str | None = None,
            ) -> dict[str, Any]:
                if mode == TrainingAccelerationMode.MLGPUSIMULATIONGPU:
                    (
                        mjx_data,
                        mjx_model,
                        mujoco_data,
                        mujoco_model,
                    ) = MjxSimulator.init(
                        model_path=source_model_path,
                        mjcf=mjcf,
                    )
                    contact_sensor_addresses, contact_body_pairs = (
                        MjxSimulator._get_contact_pair_sensor_metadata(mujoco_model)
                    )
                    return {
                        "simulator_class": MjxSimulator,
                        "mjx_data": mjx_data,
                        "mjx_model": mjx_model,
                        "mujoco_data": mujoco_data,
                        "mujoco_model": mujoco_model,
                        "render_ctx": None,
                        "render_ctx_pytree": None,
                        "contact_sensor_addresses": jnp.asarray(
                            contact_sensor_addresses,
                            dtype=jnp.int32,
                        ),
                        "contact_body_pairs": jnp.asarray(
                            contact_body_pairs,
                            dtype=jnp.int32,
                        ),
                    }
                if mode in (
                    TrainingAccelerationMode.MLCPUSIMULATIONCPU,
                    TrainingAccelerationMode.MLGPUSIMULATIONCPU,
                ):
                    mujoco_data, mujoco_model = MujocoSimulator.init(
                        model_path=source_model_path,
                        mjcf=mjcf,
                    )
                    contact_sensor_addresses, contact_body_pairs = (
                        MujocoSimulator._get_contact_pair_sensor_metadata(mujoco_model)
                    )
                    return {
                        "simulator_class": MujocoSimulator,
                        "mjx_data": None,
                        "mjx_model": None,
                        "mujoco_data": mujoco_data,
                        "mujoco_model": mujoco_model,
                        "render_ctx": None,
                        "render_ctx_pytree": None,
                        "contact_sensor_addresses": contact_sensor_addresses,
                        "contact_body_pairs": contact_body_pairs,
                    }
                raise ValueError(
                    f"Unsupported local testing acceleration mode: {mode}"
                )

            def initialize_render_context(
                simulator_state: dict[str, Any],
                nworld: int,
            ) -> None:
                simulator_class = simulator_state["simulator_class"]
                mujoco_model = simulator_state["mujoco_model"]
                if simulator_class is MjxSimulator:
                    render_ctx = MjxSimulator.get_render_context(
                        mujoco_model,
                        nworld=nworld,
                    )
                    simulator_state["render_ctx"] = render_ctx
                    simulator_state["render_ctx_pytree"] = render_ctx.pytree()
                else:
                    simulator_state["render_ctx"] = (
                        MujocoSimulator.get_render_context(mujoco_model)
                    )

            if isinstance(test_method, TestMethodCustom):
                simulator_state = initialize_simulator_state(
                    source_model_path=model_path,
                )
                self.simulator_states[None] = simulator_state
                if control_spec.all_cameras:
                    initialize_render_context(
                        simulator_state,
                        nworld=self.num_envs_per_task,
                    )
            else:
                for task_index, mjcf in task_mjcfs.items():
                    simulator_state = initialize_simulator_state(mjcf=mjcf)
                    self.simulator_states[task_index] = simulator_state
                    initialize_render_context(
                        simulator_state,
                        nworld=self.num_envs_per_task,
                    )
                    logger.info(
                        "Initialized LIBERO task simulator: id=%s task_index=%s "
                        "num_envs=%s acceleration=%s",
                        self.id,
                        task_index,
                        self.num_envs_per_task,
                        mode,
                    )

            logger.info(
                "TestingWorkflow initialized: id=%s tasks=%s "
                "num_timesteps=%s num_envs_per_task=%s acceleration=%s",
                self.id,
                list(self.simulator_states),
                self.num_timesteps,
                self.num_envs_per_task,
                test_controller.acceleration_mode,
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _build_model_for_testing(
        control_model: Any,
        rng_key: jax.Array | None = None,
    ) -> TestingModel:
        null_spec = ModelInputSpec(
            all_cameras=False,
            qpos_state=False,
            qvel_state=False,
            xpos_state=False,
            xquat_state=False,
            contact_pairs=False,
        )
        if isinstance(control_model, ControlModelVLA):
            vla_spec = control_model.content
            return Pi0Model(
                vla_spec,
                noise_mlp_spec=None,
                control_input_spec=vla_spec.input_spec,
                value_input_spec=null_spec,
                reward_input_spec=null_spec,
                rng_key=rng_key,
                n_action_steps=1,
            )

        control_mlp = control_model.content
        policy, policy_store = MLPModel.init_policy(
            input_dimension=int(control_mlp.input_dim),
            output_dimension=int(control_mlp.output_dim),
            hidden_layer_dimension=int(control_mlp.hidden_layer_dim),
            num_layers=int(control_mlp.num_hidden_layers),
            add_softmax=False,
            control_policy=True,
        )
        return MLPModel(
            policy=policy,
            policy_store=policy_store,
            control_input_spec=control_mlp.input_spec,
            value_input_spec=null_spec,
            reward_input_spec=null_spec,
            reward_fn=None,
            rng_key=rng_key,
        )

    def close(self) -> None:
        """Release caller-owned rendering resources, if any."""
        for task_index, simulator_state in self.simulator_states.items():
            render_ctx = simulator_state.get("render_ctx")
            simulator_state["render_ctx"] = None
            simulator_state["render_ctx_pytree"] = None
            if render_ctx is None:
                continue

            close = getattr(render_ctx, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception(
                        "Failed to close testing render context: "
                        "test_id=%s task_index=%s",
                        self.id,
                        task_index,
                    )
                    continue
            logger.info(
                "Testing render context released: test_id=%s task_index=%s",
                self.id,
                task_index,
            )

    def run_test_blocking(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        try:
            self._run_test_blocking(tx, should_stop)
        finally:
            # Cleanup runs in the same worker thread as testing, so an
            # asyncio cancellation cannot close a renderer still in use.
            self.close()

    def _run_test_blocking(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        logger.info(
            "Test run started: id=%s tasks=%s num_timesteps=%s "
            "num_envs_per_task=%s",
            self.id,
            list(self.simulator_states),
            self.num_timesteps,
            self.num_envs_per_task,
        )

        simulator_classes = {
            simulator_state["simulator_class"]
            for simulator_state in self.simulator_states.values()
        }
        if simulator_classes == {MjxSimulator}:
            self._run_mjx_batched(
                tx,
                should_stop,
                self.num_timesteps,
                self.num_envs_per_task,
            )
        elif simulator_classes == {MujocoSimulator}:
            self._run_sequential(
                tx,
                should_stop,
                self.num_timesteps,
                self.num_envs_per_task,
            )
        else:
            raise ValueError(
                f"Unsupported testing simulator classes: {simulator_classes}"
            )

        logger.info("Test run finished, sending StopTestingMessage: id=%s", self.id)
        tx.send_stop_testing(StopTestMessage(id=self.id))

    def _run_mjx_batched(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None,
        num_timesteps: int,
        num_envs: int,
    ) -> None:
        test_policy_fn = self.model.get_test_policy()
        params = self.model.get_test_params()
        output_mode = self.model.get_control_policy_output_mode()
        control_spec = self.model.get_control_input_spec()

        for task_index, simulator_state in self.simulator_states.items():
            if should_stop is not None and should_stop():
                logger.info(
                    "Test run stop requested before task: id=%s task_index=%s",
                    self.id,
                    task_index,
                )
                break

            simulator_class = simulator_state["simulator_class"]
            if simulator_class is not MjxSimulator:
                raise ValueError("MJX batched testing requires MjxSimulator state")
            mjx_data = simulator_state["mjx_data"]
            mjx_model = simulator_state["mjx_model"]
            if mjx_data is None or mjx_model is None:
                raise ValueError("MJX data and model are required for batched testing")

            logger.info(
                "MJX test task started: id=%s task_index=%s",
                self.id,
                task_index,
            )
            nbody = int(mjx_model.nbody)
            batched_data = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * num_envs),
                mjx_data,
            )
            render_ctx = simulator_state["render_ctx_pytree"]
            contact_sensor_addresses = simulator_state["contact_sensor_addresses"]
            contact_body_pairs = simulator_state["contact_body_pairs"]
            has_cameras = control_spec.all_cameras and int(mjx_model.ncam) > 0

            def state_fn(data: Any) -> dict[str, Array]:
                return simulator_class.get_model_state_pure(
                    control_spec,
                    data,
                    mjx_model,
                    contact_sensor_addresses,
                    contact_body_pairs,
                )

            def scan_body(carry: Any, _: Any) -> tuple[Any, tuple[Any, Any, Any]]:
                bd, rng = carry
                rng, step_key = jax.random.split(rng)
                env_keys = jax.random.split(step_key, num_envs)
                parts_batched = jax.vmap(state_fn)(bd)
                if has_cameras:
                    camera_frames, bd = simulator_class.get_model_render(
                        bd, mjx_model, render_ctx
                    )
                    parts_batched["cameras"] = camera_frames
                else:
                    camera_frames = jnp.empty(
                        (num_envs, 0, 0, 0, 3), dtype=jnp.float32
                    )
                actions = jax.vmap(
                    lambda p, k: test_policy_fn(params, {**p, "rng_key": k})
                )(parts_batched, env_keys)
                # Batched chunked policies such as Pi0 return
                # [num_envs, action_steps, action_dim]; execute only the first
                # action for each environment during this simulator step.
                if actions.ndim == 3:
                    actions = actions[:, 0]

                def step(data: Any, action: Array) -> Any:
                    controls = (
                        simulator_class.get_actuator_input_from_control_input(
                            mjx_model,
                            data,
                            action,
                            output_mode,
                        )
                    )
                    return simulator_class.forward_dyn_pure(
                        data,
                        mjx_model,
                        controls,
                    )

                bd_next = jax.vmap(step)(bd, actions)
                return (bd_next, rng), (
                    camera_frames,
                    bd_next.xpos[:, :nbody],
                    bd_next.xquat[:, :nbody],
                )

            remaining = num_timesteps
            while remaining > 0:
                if should_stop is not None and should_stop():
                    logger.info(
                        "Test run stop requested: id=%s task_index=%s remaining=%s",
                        self.id,
                        task_index,
                        remaining,
                    )
                    break
                chunk = min(UPDATE_CHUNK_SIZE, remaining)
                (
                    (batched_data, self._rng_key),
                    (camera_frames, chunk_xpos, chunk_xquat),
                ) = jax.lax.scan(
                    scan_body,
                    (batched_data, self._rng_key),
                    None,
                    length=chunk,
                )
                xpos_np = chunk_xpos.reshape(chunk, num_envs, -1).tolist()
                xquat_np = chunk_xquat.reshape(chunk, num_envs, -1).tolist()
                sim_state = [
                    [(xpos_np[t][i], xquat_np[t][i]) for t in range(chunk)]
                    for i in range(num_envs)
                ]
                frames_u8 = None
                if has_cameras:
                    frames_u8 = np.asarray(
                        (camera_frames * 255.0).round().astype(jnp.uint8)
                    ).transpose(1, 0, 2, 3, 4, 5)
                tx.send_test_updates(
                    _build_stream_bytes(
                        self.id,
                        [(0 if task_index is None else task_index, sim_state)],
                        None if frames_u8 is None else [
                            (0 if task_index is None else task_index, frames_u8)
                        ],
                    )
                )
                remaining -= chunk

    def _run_sequential(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None,
        num_timesteps: int,
        num_envs: int,
    ) -> None:
        control_spec = self.model.get_control_input_spec()
        output_mode = self.model.get_control_policy_output_mode()

        for task_index, simulator_state in self.simulator_states.items():
            if should_stop is not None and should_stop():
                logger.info(
                    "Test run stop requested before task: id=%s task_index=%s",
                    self.id,
                    task_index,
                )
                break

            simulator_class = simulator_state["simulator_class"]
            if simulator_class is not MujocoSimulator:
                raise ValueError("Sequential testing requires MujocoSimulator state")
            initial_data = simulator_state["mujoco_data"]
            model = simulator_state["mujoco_model"]
            if initial_data is None or model is None:
                raise ValueError(
                    "MuJoCo data and model are required for sequential testing"
                )

            render_ctx = simulator_state["render_ctx"]
            contact_sensor_addresses = simulator_state["contact_sensor_addresses"]
            contact_body_pairs = simulator_state["contact_body_pairs"]
            logger.info(
                "Sequential test task started: id=%s task_index=%s",
                self.id,
                task_index,
            )

            for env_index in range(num_envs):
                if should_stop is not None and should_stop():
                    logger.info(
                        "Test run stop requested: id=%s task_index=%s env_index=%s",
                        self.id,
                        task_index,
                        env_index,
                    )
                    break

                # Every environment starts from the task's unmodified data
                # template. The renderer is reused sequentially and update_scene
                # replaces its previous scene on every render.
                data = copy.copy(initial_data)
                for timestep in range(num_timesteps):
                    if should_stop is not None and should_stop():
                        logger.info(
                            "Test run stop requested: id=%s task_index=%s "
                            "env_index=%s timestep=%s",
                            self.id,
                            task_index,
                            env_index,
                            timestep,
                        )
                        break
                    logger.info(
                        "Running sequential test: task_index=%s env_index=%s "
                        "timestep=%s",
                        task_index,
                        env_index,
                        timestep,
                    )
                    parts = simulator_class.get_model_state_pure(
                        control_spec,
                        data,
                        model,
                        contact_sensor_addresses,
                        contact_body_pairs,
                    )
                    cameras = None
                    if control_spec.all_cameras and model.ncam > 0:
                        cameras, data = simulator_class.get_model_render(
                            data, model, render_ctx
                        )
                        parts["cameras"] = cameras
                    action = self.model.get_test_action(parts)
                    # Chunked policies such as Pi0 return
                    # [action_steps, action_dim]; this workload currently
                    # executes only the first action per step.
                    if action.ndim == 2:
                        action = action[0]
                    controls = simulator_class.get_actuator_input_from_control_input(
                        model,
                        data,
                        np.asarray(action),
                        output_mode,
                    )
                    data = simulator_class.forward_dyn_pure(data, model, controls)
                    xpos = data.xpos[: model.nbody].reshape(-1).tolist()
                    xquat = data.xquat[: model.nbody].reshape(-1).tolist()
                    frames_u8 = (
                        np.asarray((cameras * 255.0).round().astype(np.uint8))[
                            None, None
                        ]
                        if cameras is not None
                        else None
                    )
                    sim_state = [[(xpos, xquat)]]
                    tx.send_test_updates(
                        _build_stream_bytes(
                            self.id,
                            [(0 if task_index is None else task_index, sim_state)],
                            None if frames_u8 is None else [
                                (0 if task_index is None else task_index, frames_u8)
                            ],
                        )
                    )
