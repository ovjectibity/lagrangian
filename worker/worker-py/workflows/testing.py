from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4
import logging
import os
import zipfile
from typing import Any, Callable
from pathlib import Path

from commondef.records.assets import (
    CheckpointModelType, CheckpointRef, SimulationModelRef,
    SimulationModelType, SimulationModelFileType, TrajectoryCameraFrameRef,
)

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from commondef.luts import USER_METRICS_LUT, WORKLOAD_STRUCTURE_LUT
from commondef.records.workload import (
    WorkloadUserMetrics, WorkloadSimulationModelTrajectoryState,
    WorkloadLogs, LogType, LogStream,
)
from commondef.remoteworker.remoteworkermessages import (
    WorkloadUpdates, RemoteWorkerClientMessageTestUpdatesMessage,
)
from commondef.ioadapter import PolicyInputState
from commondef.records.workload_graph import (
    TrainingAccelerationMode,
    DatasetLibero,
    LiberoDatasetType,
    TestJobController,
    TestMethodCustom,
    TestMethodDataset,
    ControlModel,
    PolicyInputSpecStandardInputSpec,
    InputStateItemSensors,
    InputStateTypes,
    MLModelMLP,
    MLModelVLA,
    StandardInputSpec,
)
from datasets.libero.libero import get_task_mjcfs, save_task_model
from models.mlp import MLPModel
from models.model import TestingModel, load_checkpoint, load_reward_function
from models.pi0.pi0 import Pi0Model
from simulator.simulator import Simulator, MjxSimulator, MujocoSimulator
from utils.asset_storage import upload_file
from simulator.state import SimulatorState


logger = logging.getLogger(__name__)

Array = jax.Array

UPDATE_CHUNK_SIZE = 10


class TestingTaskSink(ABC):
    @abstractmethod
    def send_test_updates(self, payload: WorkloadUpdates) -> None:
        pass

class CallbackTestingTaskSink(TestingTaskSink):
    def __init__(self, send_message: Callable[[dict[str, Any]], None]) -> None:
        self._send_message = send_message

    def send_test_updates(self, payload: WorkloadUpdates) -> None:
        message = RemoteWorkerClientMessageTestUpdatesMessage(content=payload)
        self._send_message(message.model_dump(mode="json"))


class TestingWorkflow:
    def __init__(
        self,
        workload_id: str,
        project_id: str,
        test_controller: TestJobController,
        model_paths: dict[str, str],
        simulation_models: list[SimulationModelRef],
        rng_key: jax.Array | None = None,
        checkpoints: dict[str, tuple[CheckpointRef, Path]] | None = None,
        s3_bucket: str | None = None,
    ) -> None:
        if not workload_id or workload_id in {".", ".."} or any(c in workload_id for c in ("/", "\\", "\x00")):
            raise ValueError("Workload ID must be a single path component")
        self.id = workload_id
        self._asset_root = Path("/tmp/workload_assets") / workload_id
        self._s3_bucket = s3_bucket or os.environ.get("S3_BUCKET")
        self._model_refs: dict[int | None, SimulationModelRef] = {}
        self._generated_models: dict[int | None, Path] = {}
        self.test_controller = test_controller
        self.simulator_states: dict[int | None, SimulatorState] = {}

        try:
            test_method = test_controller.test_method
            custom = isinstance(test_method, TestMethodCustom)
            method = "custom" if custom else "dataset"
            structures = [
                row for row in WORKLOAD_STRUCTURE_LUT.values()
                if row["workload_type"] == "testing" and row["method_applicability"] == method
            ]
            if len(structures) != 1:
                raise ValueError(f"Expected one testing structure LUT for {method}")
            self._structure_id = structures[0]["workload_structure_id"]
            self._metric_definitions = {
                row["name"]: row for row in USER_METRICS_LUT.values()
                if row["workload_structure_id"] == self._structure_id
            }
            reward = test_method.content.reward_function if custom else None
            self._reward_input_spec = reward.input_spec if reward is not None else None
            self._reward_fn = load_reward_function(reward) if reward is not None else None
            self._reward_camera_indices = tuple(
                index for index, state in enumerate(reward.input_spec.content.states)
                if state.type == InputStateTypes.INPUT_STATE_SENSORS
                and state.content == InputStateItemSensors.CAMERAS
            ) if reward is not None else ()
            task_paths: dict[int | None, str] = {}
            if isinstance(test_method, TestMethodCustom):
                model_id = test_method.content.simulation_model_id
                matching = [ref for ref in simulation_models if ref.simulation_model_id == model_id]
                model_path = model_paths.get(model_id)
                if len(matching) != 1 or model_path is None:
                    raise ValueError("Custom testing requires its simulation model reference and local path")
                self._model_refs[None] = matching[0]
                task_paths[None] = model_path
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
                if not task_indices:
                    raise ValueError("Dataset testing requires at least one task")
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
                self._validate_asset_bucket()
                mjcfs = get_task_mjcfs(task_indices, dataset_name)
                for task_index, mjcf in zip(task_indices, mjcfs, strict=True):
                    model_id = str(uuid4())
                    directory = self._asset_root / "models" / model_id
                    filename = save_task_model(mjcf, directory)
                    archive = directory / f"m_{model_id}.zip"
                    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                        for path in sorted(directory.rglob("*")):
                            if path.is_file() and path != archive:
                                bundle.write(path, path.relative_to(directory).as_posix())
                    now = datetime.now(timezone.utc)
                    self._model_refs[task_index] = SimulationModelRef(
                        project_id=project_id, simulation_model_id=model_id, created_at=now, updated_at=now,
                        name=f"{dataset_name} task {task_index}",
                        uri_path=self._asset_uri(f"models/{model_id}/m_{model_id}.zip"),
                        model_type=SimulationModelType.MJCF,
                        file_type=SimulationModelFileType.ZIP, base_file_name=filename,
                    )
                    self._generated_models[task_index] = archive
                    task_paths[task_index] = str(directory / filename)
            else:
                raise ValueError("Unsupported testing method")
            if self.num_envs_per_task < 1:
                raise ValueError("num_envs_per_task must be at least 1")
            if self.num_timesteps < 1:
                raise ValueError("num_timesteps must be at least 1")

            root_rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
            model_rng, self._rng_key = jax.random.split(root_rng)
            self.model = self._build_model_for_testing(
                control_model,
                rng_key=model_rng,
            )

            load_checkpoint(
                self.model, control_model.model_type, control_model.model,
                CheckpointModelType.POLICY, checkpoints or {},
            )

            self._entropy_fn = self.model.get_test_entropy_fn()
            control_spec = self.model.get_control_input_spec()
            # The spec and this layout stay fixed throughout the rollout.
            self._camera_state_indices: tuple[int, ...] = tuple(
                index
                for index, state in enumerate(control_spec.content.states)
                if state.type == InputStateTypes.INPUT_STATE_SENSORS
                and state.content == InputStateItemSensors.CAMERAS
            )
            mode = self.test_controller.acceleration_mode

            for task_index, path in task_paths.items():
                mjx_data = mjx_model = None
                if mode == TrainingAccelerationMode.MLGPUSIMULATIONGPU:
                    simulator_class = MjxSimulator
                    mjx_data, mjx_model, mujoco_data, mujoco_model = MjxSimulator.init(
                        model_path=path, nworld=self.num_envs_per_task,
                    )
                elif mode in (
                    TrainingAccelerationMode.MLCPUSIMULATIONCPU,
                    TrainingAccelerationMode.MLGPUSIMULATIONCPU,
                ):
                    simulator_class = MujocoSimulator
                    mujoco_data, mujoco_model = MujocoSimulator.init(model_path=path)
                else:
                    raise ValueError(f"Unsupported testing acceleration mode: {mode}")

                addresses, body_pairs = simulator_class._get_contact_pair_sensor_metadata(mujoco_model)
                if simulator_class is MjxSimulator:
                    addresses = jnp.asarray(addresses, dtype=jnp.int32)
                    body_pairs = jnp.asarray(body_pairs, dtype=jnp.int32)
                state = SimulatorState(
                    simulator_class=simulator_class,
                    mujoco_data=mujoco_data, mujoco_model=mujoco_model,
                    mjx_data=mjx_data, mjx_model=mjx_model,
                    contact_sensor_addresses=addresses, contact_body_pairs=body_pairs,
                )
                self.simulator_states[task_index] = state
                if int(mujoco_model.ncam) == 0:
                    if self._camera_state_indices or self._reward_camera_indices:
                        raise ValueError("Input specs request cameras but the simulation model has none")
                    continue
                if custom:
                    self._validate_asset_bucket()
                if simulator_class is MjxSimulator:
                    state.render_ctx = MjxSimulator.get_render_context(
                        mujoco_model, nworld=self.num_envs_per_task,
                    )
                    state.render_ctx_pytree = state.render_ctx.pytree()
                else:
                    state.render_ctx = MujocoSimulator.get_render_context(mujoco_model)

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
        control_model: ControlModel,
        rng_key: jax.Array | None = None,
    ) -> TestingModel:
        null_spec = PolicyInputSpecStandardInputSpec(
            content=StandardInputSpec(base_body_name=None, states=[]),
        )
        if isinstance(control_model.model, MLModelVLA):
            vla_spec = control_model.model
            return Pi0Model(
                vla_spec,
                noise_mlp_spec=None,
                control_input_spec=control_model.input_spec,
                value_input_spec=null_spec,
                reward_input_spec=null_spec,
                control_output_spec=control_model.output_spec,
                rng_key=rng_key,
                n_action_steps=1,
            )

        if not isinstance(control_model.model, MLModelMLP):
            raise ValueError("Unsupported testing ML model")
        control_mlp = control_model.model.content
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
            control_input_spec=control_model.input_spec,
            value_input_spec=null_spec,
            reward_input_spec=null_spec,
            control_output_spec=control_model.output_spec,
            reward_fn=None,
            rng_key=rng_key,
        )

    def _with_camera_frames(
        self,
        state: PolicyInputState,
        camera_frames: Array | np.ndarray,
        indices: tuple[int, ...] | None = None,
    ) -> PolicyInputState:
        """Replace every camera placeholder with the same rendered frame array.

        Runtime states are frozen PyTrees; replace their containers while
        retaining all static metadata and the order requested by the spec.
        Rendering happens once per timestep, outside the environment vmap.
        """
        states = list(state.content.states)
        for index in (self._camera_state_indices if indices is None else indices):
            states[index] = replace(states[index], content=camera_frames)
        return replace(state, content=replace(state.content, states=states))

    def _validate_asset_bucket(self) -> None:
        if not self._s3_bucket or "/" in self._s3_bucket:
            raise ValueError("Configure S3_BUCKET with a bucket name for worker-generated assets")

    def _asset_uri(self, key: str) -> str:
        prefix = "worker/workloads" if key.startswith("cameras/") else "worker"
        return f"s3://{self._s3_bucket}/{prefix}/{self.id}/{key}"

    def _send_update(
        self, tx: TestingTaskSink, *, logs: list[WorkloadLogs] | None = None,
        metrics: list[WorkloadUserMetrics] | None = None,
        camera_frames: list[TrajectoryCameraFrameRef] | None = None,
        model_states: list[WorkloadSimulationModelTrajectoryState] | None = None,
        simulation_models: list[SimulationModelRef] | None = None,
    ) -> None:
        tx.send_test_updates(WorkloadUpdates(
            workload_id=self.id, logs=logs or [], metrics=metrics or [], checkpoints=[],
            camera_frames=camera_frames or [], model_states=model_states or [],
            simulation_models=simulation_models or [],
        ))

    def _log(self, tx: TestingTaskSink, text: str) -> None:
        now = datetime.now(timezone.utc)
        self._send_update(tx, logs=[WorkloadLogs(
            workload_id=self.id, created_at=now, log_timestamp=now,
            log_id=str(uuid4()), log_line_text=text, user_facing=True,
            log_type=LogType.INFO, stream=LogStream.WORKER,
        )])

    def _start_task(self, tx: TestingTaskSink, task_index: int | None) -> None:
        archive = self._generated_models.get(task_index)
        if archive is not None:
            ref = self._model_refs[task_index]
            upload_file(archive, ref.uri_path)
            # Publish the model before any records reference its generated ID.
            self._send_update(tx, simulation_models=[ref])
        self._log(tx, f"Starting testing task {0 if task_index is None else task_index}")

    def _metric_record(
        self, name: str, coordinates: list[list[int]], values: list[float],
    ) -> WorkloadUserMetrics:
        definition = self._metric_definitions[name]
        structure = WORKLOAD_STRUCTURE_LUT[definition["workload_structure_id"]]
        dimensions = structure["number_of_state_indices"] - len(
            set(definition["aggregation_indices"]) | set(definition["na_indices"])
        )
        if len(coordinates) != len(values) or any(len(c) != dimensions for c in coordinates):
            raise ValueError(f"Metric {name} does not match its LUT coordinates")
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {name} contains non-finite values")
        return WorkloadUserMetrics(
            metrics_entry_id=str(uuid4()), created_at=datetime.now(timezone.utc),
            workload_id=self.id, metric_id=definition["metric_id"],
            workload_structure_state_values=coordinates, metric_values=values,
        )

    def _send_metric(
        self, tx: TestingTaskSink, name: str,
        coordinates: list[list[int]], values: list[float],
    ) -> None:
        if values:
            self._send_update(tx, metrics=[self._metric_record(name, coordinates, values)])

    def _publish_batch(
        self, tx: TestingTaskSink, task_index: int | None, start_timestep: int,
        environments: list[int], positions: np.ndarray, quaternions: np.ndarray,
        frames: np.ndarray | None, rewards: np.ndarray | None,
    ) -> None:
        """Publish post-step snapshots, uploading every PNG before its reference.

        Pose arrays are [time, environment, body, component]. Frames, when
        present, are [time, environment, camera, height, width, RGB]. This runs
        in the workflow thread, outside JIT and outside the asyncio event loop.
        """
        model = self.simulator_states[task_index].mujoco_model
        ref = self._model_refs[task_index]
        task_id = 0 if task_index is None else task_index
        steps, envs = positions.shape[:2]
        if envs != len(environments) or positions.shape != (steps, envs, int(model.nbody), 3):
            raise ValueError("Pose batch does not match the model or environment count")
        if quaternions.shape != (steps, envs, int(model.nbody), 4):
            raise ValueError("Quaternion batch does not match the pose batch")
        if frames is not None:
            if frames.ndim != 6 or frames.shape[:3] != (steps, envs, int(model.ncam)) or frames.shape[-1] != 3:
                raise ValueError("Camera frame batch does not match the model or pose batch")
            if not np.isfinite(frames).all():
                raise ValueError("Camera frames must contain only finite values")
            if frames.dtype != np.uint8:
                frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
            directory = self._asset_root / "cameras"
            directory.mkdir(parents=True, exist_ok=True)
        elif int(model.ncam) > 0:
            raise ValueError("Post-step camera frames are missing")
        coordinates = [
            [task_id, environment, start_timestep + t]
            for t in range(steps) for environment in environments
        ]
        metrics = []
        if rewards is not None:
            if rewards.shape != (steps, envs):
                raise ValueError("Reward batch does not match the pose batch")
            metrics = [self._metric_record("Reward", coordinates, rewards.reshape(-1).tolist())]
        states = []
        cameras = []
        now = datetime.now(timezone.utc)
        for t in range(steps):
            for env_slot, environment in enumerate(environments):
                coordinate = [task_id, environment, start_timestep + t]
                states.append(WorkloadSimulationModelTrajectoryState(
                    created_at=now, sim_model_state_entry_id=str(uuid4()),
                    workload_id=self.id, workload_structure_id=self._structure_id,
                    workload_structure_state=coordinate,
                    simulation_model_id=ref.simulation_model_id,
                    state=Simulator.get_world_body(model, positions[t, env_slot], quaternions[t, env_slot]),
                ))
                if frames is None:
                    continue
                for camera_id, frame in enumerate(frames[t, env_slot]):
                    frame_id = str(uuid4())
                    filename = f"cf_{frame_id}.png"
                    path = directory / filename
                    Image.fromarray(frame).save(path, format="PNG")
                    uri = self._asset_uri(f"cameras/{filename}")
                    upload_file(path, uri)
                    cameras.append(TrajectoryCameraFrameRef(
                        camera_frame_id=frame_id, workload_id=self.id,
                        workload_structure_id=self._structure_id,
                        workload_structure_state=coordinate,
                        camera_id=str(camera_id), model_camera_name=model.camera(camera_id).name or "",
                        uri_path=uri, simulation_model_id=ref.simulation_model_id,
                    ))
        self._send_update(tx, metrics=metrics, camera_frames=cameras, model_states=states)

    def close(self) -> None:
        """Release caller-owned rendering resources, if any."""
        for task_index, simulator_state in self.simulator_states.items():
            render_ctx = simulator_state.render_ctx
            simulator_state.render_ctx = None
            simulator_state.render_ctx_pytree = None
            if render_ctx is None:
                continue

            close = getattr(render_ctx, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.exception(
                        "Failed to close testing render context: "
                        "workload_id=%s task_index=%s",
                        self.id,
                        task_index,
                    )
                    continue
            logger.info(
                "Testing render context released: workload_id=%s task_index=%s",
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

        self._log(tx, "Testing started")
        simulator_classes = {
            simulator_state.simulator_class
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

        self._log(tx, "Testing stopped" if should_stop is not None and should_stop() else "Testing completed")
        logger.info("Test run finished: id=%s", self.id)

    def _run_mjx_batched(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None,
        num_timesteps: int,
        num_envs: int,
    ) -> None:
        test_policy_fn = self.model.get_test_policy()
        params = self.model.get_test_params()
        output_spec = self.model.get_control_policy_output_spec()
        control_spec = self.model.get_control_input_spec()

        for task_index, simulator_state in self.simulator_states.items():
            if should_stop is not None and should_stop():
                logger.info(
                    "Test run stop requested before task: id=%s task_index=%s",
                    self.id,
                    task_index,
                )
                break

            self._start_task(tx, task_index)
            simulator_class = simulator_state.simulator_class
            if simulator_class is not MjxSimulator:
                raise ValueError("MJX batched testing requires MjxSimulator state")
            mjx_data = simulator_state.mjx_data
            mjx_model = simulator_state.mjx_model
            if mjx_data is None or mjx_model is None:
                raise ValueError("MJX data and model are required for batched testing")

            logger.info(
                "MJX test task started: id=%s task_index=%s",
                self.id,
                task_index,
            )
            nbody = int(mjx_model.nbody)
            batched_data = mjx_data
            render_ctx = simulator_state.render_ctx_pytree
            contact_sensor_addresses = simulator_state.contact_sensor_addresses
            contact_body_pairs = simulator_state.contact_body_pairs
            needs_input_cameras = bool(self._camera_state_indices or self._reward_camera_indices)
            has_cameras = int(simulator_state.mujoco_model.ncam) > 0
            task_id = 0 if task_index is None else task_index
            reward_totals = np.zeros(num_envs, dtype=np.float64)
            entropy_total = 0.0

            def state_fn(data: Any) -> PolicyInputState:
                return simulator_class.get_model_state_pure(
                    control_spec,
                    data,
                    mjx_model,
                    contact_sensor_addresses,
                    contact_body_pairs,
                )

            def reward_state_fn(data: Any) -> PolicyInputState:
                return simulator_class.get_model_state_pure(
                    self._reward_input_spec, data, mjx_model,
                    contact_sensor_addresses, contact_body_pairs,
                )

            def scan_body(
                carry: tuple[Any, Array], _: None
            ) -> tuple[tuple[Any, Array], tuple[Array, ...]]:
                bd, rng = carry
                rng, step_key = jax.random.split(rng)
                env_keys = jax.random.split(step_key, num_envs)
                input_state: PolicyInputState = jax.vmap(state_fn)(bd)
                if needs_input_cameras:
                    camera_frames, bd = simulator_class.get_model_render(
                        bd, mjx_model, render_ctx
                    )
                    input_state = self._with_camera_frames(input_state, camera_frames)
                else:
                    camera_frames = jnp.empty(
                        (num_envs, 0, 0, 0, 3), dtype=jnp.float32
                    )
                reward_pre = None
                if self._reward_fn is not None:
                    reward_pre = jax.vmap(reward_state_fn)(bd)
                    if self._reward_camera_indices:
                        reward_pre = self._with_camera_frames(
                            reward_pre, camera_frames, self._reward_camera_indices,
                        )
                entropy = (
                    jax.vmap(self._entropy_fn, in_axes=(None, 0))(params, input_state)
                    if self._entropy_fn is not None else jnp.empty((num_envs, 0))
                )
                actions = jax.vmap(
                    test_policy_fn, in_axes=(None, 0, 0)
                )(params, input_state, env_keys)
                # Batched chunked policies such as Pi0 return
                # [num_envs, action_steps, action_dim]; execute only the first
                # action for each environment during this simulator step.
                if actions.ndim == 3:
                    actions = actions[:, 0]

                def step(data: Any, action: Array) -> tuple[Any, Array]:
                    controls, valid = (
                        simulator_class.get_actuator_input_from_control_input(
                            mjx_model,
                            data,
                            action,
                            output_spec,
                        )
                    )
                    # A dynamic validity flag cannot raise a Python error in
                    # scan. Retain existing controls for an invalid solve and
                    # report it at the chunk boundary before publishing updates.
                    controls = jnp.where(valid, controls, data.ctrl)
                    return (
                        simulator_class.forward_dyn_pure(data, mjx_model, controls, refresh_state=True),
                        valid,
                    )

                bd_next, controls_valid = jax.vmap(step)(bd, actions)
                output_frames = jnp.empty((num_envs, 0, 0, 0, 3), dtype=jnp.float32)
                if has_cameras:
                    output_frames, bd_next = simulator_class.get_model_render(
                        bd_next, mjx_model, render_ctx,
                    )
                rewards = jnp.empty((num_envs, 0))
                if self._reward_fn is not None:
                    reward_post = jax.vmap(reward_state_fn)(bd_next)
                    if self._reward_camera_indices:
                        reward_post = self._with_camera_frames(
                            reward_post, output_frames, self._reward_camera_indices,
                        )
                    rewards = jax.vmap(self._reward_fn)(reward_pre, reward_post)
                return (bd_next, rng), (
                    output_frames,
                    bd_next.xpos[:, :nbody],
                    bd_next.xquat[:, :nbody],
                    controls_valid, rewards, entropy,
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
                    (camera_frames, chunk_xpos, chunk_xquat, controls_valid, rewards, entropies),
                ) = jax.lax.scan(
                    scan_body,
                    (batched_data, self._rng_key),
                    None,
                    length=chunk,
                )
                valid_np = np.asarray(controls_valid)
                if not valid_np.all():
                    chunk_step, env_index = np.argwhere(~valid_np)[0]
                    timestep = num_timesteps - remaining + int(chunk_step)
                    raise ValueError(
                        "Invalid actuator control mapping: "
                        f"task_index={task_index} env_index={int(env_index)} "
                        f"timestep={timestep}"
                    )
                start_timestep = num_timesteps - remaining
                reward_values = np.asarray(rewards) if self._reward_fn is not None else None
                if self._entropy_fn is not None:
                    entropy_values = np.asarray(entropies)
                    if not np.isfinite(entropy_values).all():
                        raise ValueError("Policy entropy contains non-finite values")
                    entropy_total += float(entropy_values.sum())
                self._publish_batch(
                    tx, task_index, start_timestep, list(range(num_envs)),
                    np.asarray(chunk_xpos), np.asarray(chunk_xquat),
                    np.asarray(camera_frames) if has_cameras else None, reward_values,
                )
                if reward_values is not None:
                    reward_totals += reward_values.sum(axis=0)
                remaining -= chunk
            if remaining == 0:
                if self._reward_fn is not None:
                    self._send_metric(
                        tx, "Mean trajectory reward",
                        [[task_id, env] for env in range(num_envs)],
                        (reward_totals / num_timesteps).tolist(),
                    )
                if self._entropy_fn is not None:
                    self._send_metric(
                        tx, "Mean policy entropy", [[task_id]],
                        [entropy_total / (num_envs * num_timesteps)],
                    )

    def _run_sequential(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None,
        num_timesteps: int,
        num_envs: int,
    ) -> None:
        control_spec = self.model.get_control_input_spec()
        output_spec = self.model.get_control_policy_output_spec()
        test_policy_fn = self.model.get_test_policy()
        params = self.model.get_test_params()

        for task_index, simulator_state in self.simulator_states.items():
            if should_stop is not None and should_stop():
                logger.info(
                    "Test run stop requested before task: id=%s task_index=%s",
                    self.id,
                    task_index,
                )
                break

            self._start_task(tx, task_index)
            simulator_class = simulator_state.simulator_class
            if simulator_class is not MujocoSimulator:
                raise ValueError("Sequential testing requires MujocoSimulator state")
            initial_data = simulator_state.mujoco_data
            model = simulator_state.mujoco_model
            if initial_data is None or model is None:
                raise ValueError(
                    "MuJoCo data and model are required for sequential testing"
                )

            render_ctx = simulator_state.render_ctx
            contact_sensor_addresses = simulator_state.contact_sensor_addresses
            contact_body_pairs = simulator_state.contact_body_pairs
            logger.info(
                "Sequential test task started: id=%s task_index=%s",
                self.id,
                task_index,
            )

            task_id = 0 if task_index is None else task_index
            entropy_total = 0.0
            entropy_count = 0
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
                reward_total = 0.0
                completed = 0
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
                    input_state: PolicyInputState = (
                        simulator_class.get_model_state_pure(
                            control_spec,
                            data,
                            model,
                            contact_sensor_addresses,
                            contact_body_pairs,
                        )
                    )
                    cameras = None
                    if self._camera_state_indices:
                        cameras, data = simulator_class.get_model_render(
                            data, model, render_ctx
                        )
                        input_state = self._with_camera_frames(input_state, cameras)
                    reward_pre = None
                    if self._reward_fn is not None:
                        reward_pre = simulator_class.get_model_state_pure(
                            self._reward_input_spec, data, model,
                            contact_sensor_addresses, contact_body_pairs,
                        )
                        if self._reward_camera_indices:
                            # Rendering may be needed only by the reward input.
                            if cameras is None:
                                cameras, data = simulator_class.get_model_render(data, model, render_ctx)
                            reward_pre = self._with_camera_frames(
                                reward_pre, cameras, self._reward_camera_indices,
                            )
                        reward_pre = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), reward_pre)
                    entropy = (
                        float(np.asarray(self._entropy_fn(params, input_state)).reshape(()))
                        if self._entropy_fn is not None else None
                    )
                    if entropy is not None and not np.isfinite(entropy):
                        raise ValueError("Policy entropy contains non-finite values")
                    self._rng_key, step_key = jax.random.split(self._rng_key)
                    action = test_policy_fn(params, input_state, step_key)
                    # Chunked policies such as Pi0 return
                    # [action_steps, action_dim]; this workload currently
                    # executes only the first action per step.
                    if action.ndim == 2:
                        action = action[0]
                    controls, valid = (
                        simulator_class.get_actuator_input_from_control_input(
                            model,
                            data,
                            np.asarray(action),
                            output_spec,
                        )
                    )
                    if not bool(valid):
                        raise ValueError(
                            "Invalid actuator control mapping: "
                            f"task_index={task_index} env_index={env_index} "
                            f"timestep={timestep}"
                        )
                    data = simulator_class.forward_dyn_pure(data, model, controls, refresh_state=True)
                    output_frames = None
                    if int(model.ncam) > 0:
                        output_frames, data = simulator_class.get_model_render(data, model, render_ctx)
                    reward = None
                    if self._reward_fn is not None:
                        reward_post = simulator_class.get_model_state_pure(
                            self._reward_input_spec, data, model,
                            contact_sensor_addresses, contact_body_pairs,
                        )
                        if self._reward_camera_indices:
                            reward_post = self._with_camera_frames(
                                reward_post, output_frames, self._reward_camera_indices,
                            )
                        reward = float(np.asarray(self._reward_fn(reward_pre, reward_post)).reshape(()))
                        reward_total += reward
                    if entropy is not None:
                        entropy_total += entropy
                        entropy_count += 1
                    completed += 1
                    self._publish_batch(
                        tx, task_index, timestep, [env_index],
                        np.asarray(data.xpos[: model.nbody])[None, None],
                        np.asarray(data.xquat[: model.nbody])[None, None],
                        np.asarray(output_frames)[None, None] if output_frames is not None else None,
                        np.array([[reward]]) if reward is not None else None,
                    )

                if completed == num_timesteps and self._reward_fn is not None:
                    self._send_metric(
                        tx, "Mean trajectory reward", [[task_id, env_index]],
                        [reward_total / num_timesteps],
                    )
            if self._entropy_fn is not None and entropy_count == num_envs * num_timesteps:
                self._send_metric(
                    tx, "Mean policy entropy", [[task_id]],
                    [entropy_total / entropy_count],
                )
