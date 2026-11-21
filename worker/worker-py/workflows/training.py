from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
import zipfile
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from PIL import Image

from commondef.ioadapter import PolicyInputState
from commondef.luts import USER_METRICS_LUT, WORKLOAD_STRUCTURE_LUT
from commondef.records.assets import CheckpointModelType, CheckpointRef, TrajectoryCameraFrameRef
from commondef.records.workload import (
    WorkloadLogs, WorkloadUserMetrics, WorkloadSimulationModelTrajectoryState,
    LogType, LogStream,
)
from commondef.remoteworker.remoteworkermessages import (
    WorkloadUpdates, RemoteWorkerClientMessageTrainingUpdatesMessage,
)
from commondef.records.workload_graph import (
    CheckpointFrequencyTypes, ControlOutputModeFlowMatchingAction,
    InputStateItemSensors, InputStateTypes, MLModelMLP, MLModelVLA,
    PPOTrainingMethod, TrainingAccelerationMode, TrainingJobController,
)
from workflows.ppo_trainer import PPOTrainer
from models.mlp import MLPModel
from models.model import PPOTrainedModel, load_checkpoint, load_reward_function
from models.pi0.pi0 import Pi0Model
from simulator.simulator import Simulator, MjxSimulator, MujocoSimulator
from utils.asset_storage import upload_file
from simulator.state import SimulatorState

logger = logging.getLogger(__name__)


class TrainingTaskSink(ABC):
    @abstractmethod
    def send_training_updates(self, message: WorkloadUpdates) -> None:
        pass

class CallbackTrainingTaskSink(TrainingTaskSink):
    """
    Adapter for websocket integration:
    pass a callable that sends the serialized payload to your server.
    """

    def __init__(self, send_message: Callable[[dict[str, Any]], None]) -> None:
        self._send_message = send_message

    def send_training_updates(self, message: WorkloadUpdates) -> None:
        payload = RemoteWorkerClientMessageTrainingUpdatesMessage(content=message)
        self._send_message(payload.model_dump(mode="json"))


class TrainingWorkflow:
    def __init__(
        self,
        training_id: str,
        training_controller: TrainingJobController,
        model_path: str,
        checkpoints: dict[str, tuple[CheckpointRef, Path]] | None = None,
        rng_key: jax.Array | None = None,
        s3_bucket: str | None = None,
    ) -> None:
        if not training_id or training_id in {".", ".."} or any(c in training_id for c in ("/", "\\", "\x00")):
            raise ValueError("Workload ID must be a single path component")
        self.id = training_id
        self._asset_root = Path("/tmp/workload_assets") / training_id
        self._s3_bucket = s3_bucket or os.environ.get("S3_BUCKET")
        self._last_saved_coordinate: list[int] | None = None
        self.training_controller = training_controller
        self.params: PPOTrainingMethod = training_controller.training_method.content
        self.simulator_state: SimulatorState | None = None

        try:
            if not self._s3_bucket or self._s3_bucket.strip() != self._s3_bucket or any(
                c in self._s3_bucket for c in ("/", "\\", "?", "#", "\x00")
            ):
                raise ValueError("S3_BUCKET must be a bucket name")
            if min(self.params.num_trajs, self.params.num_steps_per_traj, self.params.num_rollouts, self.params.num_epochs) <= 0:
                raise ValueError("Training rollout, epoch and trajectory counts must be positive")
            if not np.isfinite([self.params.epsilon, self.params.gamma, self.params.lambda_]).all():
                raise ValueError("PPO coefficients must be finite")
            if self.params.epsilon <= 0 or not 0 <= self.params.gamma <= 1 or not 0 <= self.params.lambda_ <= 1:
                raise ValueError("Invalid PPO clipping, discount or GAE coefficient")
            frequency = training_controller.checkpoint_frequency
            if frequency.type != CheckpointFrequencyTypes.LAST_ONLY and frequency.content <= 0:
                raise ValueError("Checkpoint frequency must be positive")
            structures = {
                tuple(row["workload_state_index_names"]): row["workload_structure_id"]
                for row in WORKLOAD_STRUCTURE_LUT.values()
                if row["workload_type"] == "training" and row["method_applicability"] == "ppo"
            }
            self._rollout_structure_id = structures[("rollout", "rollout_trajectory", "rollout_trajectory_timestep")]
            self._update_structure_id = structures[("rollout", "epoch", "minibatch", "trajectory", "trajectory_timestep")]
            self._metric_definitions = {
                row["name"]: row for row in USER_METRICS_LUT.values()
                if row["workload_structure_id"] in (self._rollout_structure_id, self._update_structure_id)
            }
            root_rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
            model_rng, workflow_rng = jax.random.split(root_rng)
            self._rng_key, trainer_key = jax.random.split(workflow_rng)

            reward_fn = load_reward_function(self.params.reward_function)
            self.model = self._build_model_for_training(
                self.params,
                reward_fn,
                rng_key=model_rng,
            )
            checkpoint_refs = checkpoints or {}
            policy_optimizer = load_checkpoint(
                self.model, self.params.control_model.model_type,
                self.params.control_model.model, CheckpointModelType.POLICY, checkpoint_refs, restore_optimizer=True,
            )
            value_optimizer = load_checkpoint(
                self.model, self.params.value_model.model_type,
                self.params.value_model.model, CheckpointModelType.VALUE, checkpoint_refs, restore_optimizer=True,
            )
            noise_optimizer = None
            if isinstance(self.params.control_output_mode, ControlOutputModeFlowMatchingAction):
                noise = self.params.control_output_mode.content.content
                if not isinstance(self.params.control_model.model, MLModelVLA):
                    raise ValueError("Flow-noise checkpoints require a VLA policy model")
                noise_optimizer = load_checkpoint(
                    self.model, noise.model_type, MLModelMLP(content=noise.noise_model),
                    CheckpointModelType.FLOWNOISE, checkpoint_refs, restore_optimizer=True,
                )

            mode = self.training_controller.acceleration_mode
            needs_render_context = any(
                state.type == InputStateTypes.INPUT_STATE_SENSORS
                and state.content == InputStateItemSensors.CAMERAS
                for input_spec in (
                    self.model.get_control_input_spec(),
                    self.model.get_value_input_spec(),
                    self.model.get_reward_input_spec(),
                )
                for state in input_spec.content.states
            )
            if mode in (
                TrainingAccelerationMode.MLCPUSIMULATIONCPU,
                TrainingAccelerationMode.MLGPUSIMULATIONCPU,
            ):
                mujoco_data, mujoco_model = MujocoSimulator.init(model_path=model_path)
                contact_sensor_addresses, contact_body_pairs = (
                    MujocoSimulator._get_contact_pair_sensor_metadata(mujoco_model)
                )
                self.simulator_state = SimulatorState(
                    simulator_class=MujocoSimulator,
                    mujoco_data=mujoco_data,
                    mujoco_model=mujoco_model,
                    contact_sensor_addresses=contact_sensor_addresses,
                    contact_body_pairs=contact_body_pairs,
                )
            elif mode == TrainingAccelerationMode.MLGPUSIMULATIONGPU:
                (
                    mjx_data,
                    mjx_model,
                    mujoco_data,
                    mujoco_model,
                ) = MjxSimulator.init(
                    model_path=model_path,
                    nworld=self.params.num_trajs,
                )
                contact_sensor_addresses, contact_body_pairs = (
                    MjxSimulator._get_contact_pair_sensor_metadata(mujoco_model)
                )
                self.simulator_state = SimulatorState(
                    simulator_class=MjxSimulator,
                    mjx_data=mjx_data,
                    mjx_model=mjx_model,
                    mujoco_data=mujoco_data,
                    mujoco_model=mujoco_model,
                    contact_sensor_addresses=jnp.asarray(
                        contact_sensor_addresses,
                        dtype=jnp.int32,
                    ),
                    contact_body_pairs=jnp.asarray(
                        contact_body_pairs,
                        dtype=jnp.int32,
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported local training acceleration mode: {mode}"
                )

            if needs_render_context and mujoco_model.ncam == 0:
                raise ValueError("Training input specs request Cameras, but the model has no cameras")
            if mujoco_model.ncam > 0:
                simulator_class = self.simulator_state.simulator_class
                render_ctx = simulator_class.get_render_context(
                    mujoco_model,
                    nworld=(
                        self.params.num_trajs
                        if simulator_class is MjxSimulator
                        else 1
                    ),
                )
                self.simulator_state.render_ctx = render_ctx
                if simulator_class is MjxSimulator:
                    self.simulator_state.render_ctx_pytree = render_ctx.pytree()

            self.trainer = PPOTrainer(
                model=self.model,
                num_trajs=self.params.num_trajs,
                traj_depth=self.params.num_steps_per_traj,
                num_epochs=self.params.num_epochs,
                epsilon=self.params.epsilon,
                gamma_lambda=(self.params.gamma, self.params.lambda_),
                rng_key=trainer_key,
            )
            if policy_optimizer is not None:
                if isinstance(self.params.control_model.model, MLModelVLA):
                    self.trainer.policy_opt["pi0"] = policy_optimizer
                else:
                    self.trainer.policy_opt = policy_optimizer
            if noise_optimizer is not None:
                self.trainer.policy_opt["noise"] = noise_optimizer
            if value_optimizer is not None:
                self.trainer.value_opt = value_optimizer
            logger.info(
                "TrainingWorkflow initialized: id=%s rollouts=%s epochs=%s acceleration=%s",
                self.id, self.params.num_rollouts, self.params.num_epochs,
                self.training_controller.acceleration_mode,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release caller-owned rendering resources, if any."""
        if self.simulator_state is None:
            return
        render_ctx = self.simulator_state.render_ctx
        self.simulator_state.render_ctx = None
        self.simulator_state.render_ctx_pytree = None
        if render_ctx is None:
            return

        close = getattr(render_ctx, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception(
                    "Failed to close training render context: training_id=%s",
                    self.id,
                )
                return
        logger.info("Training render context released: training_id=%s", self.id)

    @staticmethod
    def _build_model_for_training(
        params: PPOTrainingMethod,
        reward_fn: Callable[[PolicyInputState, PolicyInputState], float],
        rng_key: jax.Array | None = None,
    ) -> PPOTrainedModel:
        control_model = params.control_model
        if not isinstance(params.value_model.model, MLModelMLP):
            raise ValueError("Only MLP value models are supported for training")
        value_mlp = params.value_model.model.content
        if value_mlp.output_dim != 1:
            raise ValueError("The value model must output one scalar per state")

        if isinstance(control_model.model, MLModelVLA):
            if not isinstance(
                params.control_output_mode,
                ControlOutputModeFlowMatchingAction,
            ):
                raise ValueError(
                    "VLA control model requires "
                    "control_output_mode=FlowMatchingAction(FlowNoise)"
                )
            vla_spec = control_model.model
            noise_mlp_spec = params.control_output_mode.content.content.noise_model
            return Pi0Model(
                vla_spec,
                noise_mlp_spec,
                control_input_spec=control_model.input_spec,
                value_input_spec=params.value_model.input_spec,
                reward_input_spec=params.reward_function.input_spec,
                control_output_spec=control_model.output_spec,
                value_mlp_spec=value_mlp,
                reward_fn=reward_fn,
                rng_key=rng_key,
                n_action_steps=1,
            )

        if not isinstance(control_model.model, MLModelMLP):
            raise ValueError("Unsupported training ML model")
        control_mlp = control_model.model.content
        policy, policy_store = MLPModel.init_policy(
            input_dimension=int(control_mlp.input_dim),
            output_dimension=int(control_mlp.output_dim),
            hidden_layer_dimension=int(control_mlp.hidden_layer_dim),
            num_layers=int(control_mlp.num_hidden_layers),
            add_softmax=False,
            control_policy=True,
        )
        value, value_store = MLPModel.init_policy(
            input_dimension=int(value_mlp.input_dim),
            output_dimension=int(value_mlp.output_dim),
            hidden_layer_dimension=int(value_mlp.hidden_layer_dim),
            num_layers=int(value_mlp.num_hidden_layers),
            add_softmax=False,
            control_policy=False,
        )
        return MLPModel.init_ppo_trained_model(
            policy=policy,
            val_estimator=value,
            pol_store=policy_store,
            val_estimator_store=value_store,
            control_input_spec=control_model.input_spec,
            value_input_spec=params.value_model.input_spec,
            reward_input_spec=params.reward_function.input_spec,
            control_output_spec=control_model.output_spec,
            reward_fn=reward_fn,
            rng_key=rng_key,
        )

    def _send_update(self, tx: TrainingTaskSink, **fields: Any) -> None:
        payload = dict(logs=[], metrics=[], checkpoints=[], camera_frames=[], model_states=[], simulation_models=[])
        payload.update(fields)
        tx.send_training_updates(WorkloadUpdates(workload_id=self.id, **payload))

    def _log(self, tx: TrainingTaskSink, text: str) -> None:
        now = datetime.now(timezone.utc)
        self._send_update(tx, logs=[WorkloadLogs(
            workload_id=self.id, created_at=now, log_timestamp=now,
            log_id=str(uuid4()), log_line_text=text, user_facing=True,
            log_type=LogType.INFO, stream=LogStream.WORKER,
        )])

    def _metric(self, name: str, coordinate: list[int], value: float) -> WorkloadUserMetrics:
        definition = self._metric_definitions[name]
        if not np.isfinite(value):
            raise ValueError(f"Non-finite training metric: {name}")
        structure = WORKLOAD_STRUCTURE_LUT[definition["workload_structure_id"]]
        excluded = set(definition["aggregation_indices"]) | set(definition["na_indices"])
        dimensions = structure["number_of_state_indices"] - len(excluded)
        if len(coordinate) != dimensions:
            raise ValueError(f"Metric coordinates do not match the LUT: {name}")
        return WorkloadUserMetrics(
            metrics_entry_id=str(uuid4()), created_at=datetime.now(timezone.utc),
            workload_id=self.id, metric_id=definition["metric_id"],
            workload_structure_state_values=[coordinate], metric_values=[value],
        )

    def _asset_uri(self, key: str) -> str:
        return f"s3://{self._s3_bucket}/worker/{self.id}/{key}"

    def _save_checkpoints(self, tx: TrainingTaskSink, coordinate: list[int]) -> None:
        if coordinate == self._last_saved_coordinate:
            return
        policy = self.model.get_policy_store()
        vla = isinstance(self.params.control_model.model, MLModelVLA)
        components = [
            (CheckpointModelType.POLICY, self.params.control_model.model,
             policy["pi0"] if vla else policy,
             self.trainer.policy_opt["pi0"] if vla else self.trainer.policy_opt),
            (CheckpointModelType.VALUE, self.params.value_model.model,
             self.model.get_value_store(), self.trainer.value_opt),
        ]
        if vla:
            noise = self.params.control_output_mode.content.content.noise_model
            components.append((CheckpointModelType.FLOWNOISE, MLModelMLP(content=noise),
                               policy["noise"], self.trainer.policy_opt["noise"]))
        refs = []
        with ocp.PyTreeCheckpointer() as checkpointer:
            for role, spec, weights, optimizer in components:
                checkpoint_id = str(uuid4())
                directory = self._asset_root / "checkpoints" / checkpoint_id
                directory.mkdir(parents=True)
                checkpointer.save(str(directory / "weights"), weights)
                checkpointer.save(str(directory / "optimizer"), optimizer)
                filename = f"cp_{checkpoint_id}.zip"
                path = directory / filename
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                    for root in (directory / "weights", directory / "optimizer"):
                        for file in sorted(root.rglob("*")):
                            if file.is_file():
                                archive.write(file, file.relative_to(directory))
                uri = self._asset_uri(f"checkpoints/{checkpoint_id}/{filename}")
                upload_file(path, uri)
                now = datetime.now(timezone.utc)
                refs.append(CheckpointRef(
                    checkpoint_id=checkpoint_id, created_at=now, updated_at=now,
                    name=f"{role.value} rollout {coordinate[0]} epoch {coordinate[1]}",
                    uri_path=uri, workload_id=self.id,
                    workload_structure_id=self._update_structure_id,
                    workload_structure_state=coordinate,
                    # Checkpoints describe an update over all rollout samples.
                    na_indices=[3, 4],
                    model_type=role, model_spec=spec,
                ))
        # Publish the set only after every component was saved and uploaded.
        self._send_update(tx, checkpoints=refs)
        self._last_saved_coordinate = coordinate.copy()

    def _publish_rollout_batch(
        self, tx: TrainingTaskSink, rollout: int, start: int,
        positions: np.ndarray, quaternions: np.ndarray, frames: np.ndarray | None,
    ) -> None:
        model = self.simulator_state.mujoco_model
        if positions.ndim != 4 or positions.shape[1:] != (self.params.num_trajs, int(model.nbody), 3):
            raise ValueError("Training poses do not match the model or trajectory count")
        steps, environments = positions.shape[:2]
        if quaternions.shape != (steps, environments, int(model.nbody), 4):
            raise ValueError("Training quaternions do not match the pose batch")
        if frames is not None:
            if frames.ndim != 6 or frames.shape[:3] != (steps, environments, int(model.ncam)) or frames.shape[-1] != 3:
                raise ValueError("Training cameras do not match the pose batch")
            if not np.isfinite(frames).all():
                raise ValueError("Training camera frames must be finite")
            if frames.dtype != np.uint8:
                frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
        elif int(model.ncam):
            raise ValueError("Training camera frames are missing")
        states, camera_refs = [], []
        directory = self._asset_root / "cameras"
        if frames is not None:
            directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        for timestep in range(steps):
            for environment in range(environments):
                coordinate = [rollout, environment, start + timestep]
                states.append(WorkloadSimulationModelTrajectoryState(
                    created_at=now, sim_model_state_entry_id=str(uuid4()),
                    workload_id=self.id, workload_structure_id=self._rollout_structure_id,
                    workload_structure_state=coordinate,
                    simulation_model_id=self.params.simulation_model_id,
                    state=Simulator.get_world_body(model, positions[timestep, environment], quaternions[timestep, environment]),
                ))
                if frames is None:
                    continue
                for camera, frame in enumerate(frames[timestep, environment]):
                    frame_id = str(uuid4())
                    filename = f"cf_{frame_id}.png"
                    path = directory / filename
                    Image.fromarray(frame).save(path, format="PNG")
                    uri = self._asset_uri(f"cameras/{filename}")
                    upload_file(path, uri)
                    camera_refs.append(TrajectoryCameraFrameRef(
                        camera_frame_id=frame_id, workload_id=self.id,
                        workload_structure_id=self._rollout_structure_id,
                        workload_structure_state=coordinate, camera_id=str(camera),
                        model_camera_name=model.camera(camera).name or "", uri_path=uri,
                        simulation_model_id=self.params.simulation_model_id,
                    ))
        self._send_update(tx, camera_frames=camera_refs, model_states=states)

    def run_training_blocking(
        self, tx: TrainingTaskSink, should_stop: Callable[[], bool] | None = None,
    ) -> None:
        try:
            self._run_training_blocking(tx, should_stop)
        finally:
            self.close()

    def _run_training_blocking(
        self, tx: TrainingTaskSink, should_stop: Callable[[], bool] | None = None,
    ) -> None:
        if self.simulator_state is None:
            raise RuntimeError("Training simulator state is not initialized")
        stopped = lambda: should_stop is not None and should_stop()
        frequency = self.training_controller.checkpoint_frequency
        use_gpu = self.simulator_state.simulator_class is MjxSimulator
        self._log(tx, "Training started")
        last_coordinate = None
        completed = False
        for rollout in range(self.params.num_rollouts):
            if stopped():
                break
            self._log(tx, f"Collecting rollout {rollout}")
            record = lambda start, positions, quaternions, frames: self._publish_rollout_batch(
                tx, rollout, start, positions, quaternions, frames,
            )
            if use_gpu:
                self._rng_key, rollout_rng = jax.random.split(self._rng_key)
                trajectories = self.trainer.gen_trajectories_gpu(
                    self.params.num_trajs, self.simulator_state, rollout_rng, record, should_stop,
                )
            else:
                trajectories = self.trainer.gen_trajectories(
                    self.params.num_trajs, self.simulator_state, record, should_stop,
                )
            if not trajectories or stopped():
                break
            metrics = [self._metric(
                "Mean rollout reward", [rollout], self.trainer.get_rollout_mean_reward(trajectories),
            )]
            entropy = self.trainer.get_rollout_mean_entropy(trajectories)
            if entropy is not None:
                metrics.append(self._metric("Mean rollout policy entropy", [rollout], entropy))
            self._send_update(tx, metrics=metrics)

            def post_epoch(epoch: int, stats: tuple[float, ...]) -> None:
                nonlocal last_coordinate
                last_coordinate = [rollout, epoch, 0]
                policy_loss, value_loss, policy_norm, value_norm = stats
                self._send_update(tx, metrics=[
                    self._metric("PPO clip objective", last_coordinate, -policy_loss),
                    self._metric("Mean value loss", last_coordinate, value_loss),
                    self._metric("Policy gradient norm", last_coordinate, policy_norm),
                    self._metric("Value gradient norm", last_coordinate, value_norm),
                ])
                # One minibatch/update per epoch makes these two cadences identical.
                ordinal = rollout * self.params.num_epochs + epoch + 1
                if frequency.type in (
                    CheckpointFrequencyTypes.POST_EPOCH, CheckpointFrequencyTypes.POST_UPDATE,
                ) and ordinal % frequency.content == 0:
                    self._save_checkpoints(tx, last_coordinate)

            diagnostics = self.trainer.train_rollout(trajectories, post_epoch, should_stop)
            if len(diagnostics.mean_policy_loss()) != self.params.num_epochs:
                break
            if frequency.type == CheckpointFrequencyTypes.POST_ROLLOUT and (rollout + 1) % frequency.content == 0:
                self._save_checkpoints(tx, last_coordinate)
            completed = rollout == self.params.num_rollouts - 1
        if completed and not stopped() and frequency.type == CheckpointFrequencyTypes.LAST_ONLY and last_coordinate is not None:
            self._save_checkpoints(tx, last_coordinate)
        self._log(tx, "Training stopped" if stopped() else "Training completed")
