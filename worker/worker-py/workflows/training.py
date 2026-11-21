from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import io
import logging
import os
import zipfile
from typing import Any, Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from commondef.ioadapter import PolicyInputState
from commondef.records.assets import CheckpointModelType, CheckpointRef
from commondef.records.workload_graph import (
    CheckpointFrequencyTypes, ControlOutputModeFlowMatchingAction,
    InputStateItemSensors, InputStateTypes, MLModelMLP, MLModelVLA,
    PPOTrainingMethod, TrainingAccelerationMode, TrainingJobController,
)
from workflows.ppo_trainer import PPOTrainer
from models.mlp import MLPModel
from models.model import PPOTrainedModel, load_checkpoint, load_reward_function
from models.pi0.pi0 import Pi0Model
from simulator.simulator import MjxSimulator, MujocoSimulator
from simulator.state import SimulatorState

logger = logging.getLogger(__name__)


class TrainingTaskSink(ABC):
    @abstractmethod
    def send_training_updates(self, message: dict[str, Any]) -> None:
        pass

class CallbackTrainingTaskSink(TrainingTaskSink):
    """
    Adapter for websocket integration:
    pass a callable that sends the serialized payload to your server.
    """

    def __init__(self, send_message: Callable[[dict[str, Any]], None]) -> None:
        self._send_message = send_message

    def send_training_updates(self, message: dict[str, Any]) -> None:
        # Outgoing workflow updates retain their existing wire format for now.
        self._send_message({"type": "TrainingUpdatesMessage", "content": message})


class TrainingWorkflow:
    def __init__(
        self,
        training_id: str,
        training_controller: TrainingJobController,
        model_path: str,
        checkpoints: dict[str, tuple[CheckpointRef, Path]] | None = None,
        rng_key: jax.Array | None = None,
    ) -> None:
        self.id = training_id
        self.training_controller = training_controller
        self.params: PPOTrainingMethod = training_controller.training_method.content
        self.simulator_state: SimulatorState | None = None

        try:
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

            if needs_render_context:
                if mujoco_model.ncam == 0:
                    raise ValueError(
                        "Training input specs request Cameras, but the model has no cameras"
                    )
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
                num_updates=self.params.num_updates,
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
            self._ckpt_dir = f"/tmp/checkpoints/{training_id}"
            os.makedirs(self._ckpt_dir, exist_ok=True)
            self._checkpointer = ocp.PyTreeCheckpointer()
            logger.info(
                "TrainingWorkflow initialized: id=%s epochs=%s epsilon=%s "
                "acceleration=%s",
                self.id,
                self.params.num_epochs,
                self.params.epsilon,
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

    @staticmethod
    def log_and_send_blocking(
        tx: TrainingTaskSink,
        training_id: str,
        msg: str,
    ) -> None:
        logger.info("training_id=%s %s", training_id, msg)
        tx.send_training_updates({
            "id": training_id,
            "updates": [{
                "training_run_state": {
                    "type": "PPOTrainingRunState",
                    "content": {"epoch_index": 0, "update_index": 0},
                },
                "logs": [msg], "metrics": [], "checkpoints": [],
            }],
        })

    def _save_and_send_checkpoint(
        self,
        tx: TrainingTaskSink,
        label: str,
        epoch_index: int,
        update_index: int,
    ) -> None:
        if type(self.model).__name__ == "Pi0Model":
            logger.info(
                "Checkpoint skipped for Pi-0 workload: training_id=%s label=%s",
                self.id,
                label,
            )
            return
        base = os.path.join(self._ckpt_dir, label)
        self._checkpointer.save(os.path.join(base, "policy"),     self.model.get_policy_store())
        self._checkpointer.save(os.path.join(base, "value"),      self.model.get_value_store())
        self._checkpointer.save(os.path.join(base, "policy_opt"), self.trainer.policy_opt)
        self._checkpointer.save(os.path.join(base, "value_opt"),  self.trainer.value_opt)
        logger.info("Checkpoint saved: training_id=%s label=%s", self.id, label)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _, filenames in os.walk(base):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    zf.write(fpath, arcname=os.path.relpath(fpath, base))
        model_zip_base64 = base64.b64encode(buf.getvalue()).decode()

        tx.send_training_updates({
            "id": self.id,
            "updates": [{
                "training_run_state": {
                    "type": "PPOTrainingRunState",
                    "content": {"epoch_index": epoch_index, "update_index": update_index},
                },
                "logs": [], "metrics": [],
                "checkpoints": [{"model_zip_base64": model_zip_base64}],
            }],
        })
        logger.info("Checkpoint sent: training_id=%s epoch=%s update=%s", self.id, epoch_index, update_index)

    def run_training_blocking(
        self,
        tx: TrainingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        try:
            self._run_training_blocking(tx, should_stop)
        finally:
            # Cleanup runs in the same worker thread as training, so an
            # asyncio cancellation cannot close a renderer still in use.
            self.close()

    def _run_training_blocking(
        self,
        tx: TrainingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        simulator_state = self.simulator_state
        if simulator_state is None:
            raise RuntimeError("Training simulator state is not initialized")
        use_gpu = (
            self.training_controller.acceleration_mode
            == TrainingAccelerationMode.MLGPUSIMULATIONGPU
        )
        logger.info("Training run started: id=%s use_gpu=%s", self.id, use_gpu)

        last_epoch_run = -1
        for i in range(self.params.num_epochs):
            if should_stop is not None and should_stop():
                logger.info("Training run stop requested: id=%s epoch=%s", self.id, i)
                break
            last_epoch_run = i
            self.log_and_send_blocking(
                tx,
                self.id,
                f"Running the training epoch {i}",
            )

            freq = self.training_controller.checkpoint_frequency

            if use_gpu:
                logger.info("Getting trajectories for epoch=%s",i)
                self._rng_key, epoch_rng = jax.random.split(self._rng_key)
                control_wstates, value_wstates, model_outputs, action_samples, rewards = (
                    self.trainer.gen_trajectories_gpu(
                        self.params.num_trajs,
                        simulator_state,
                        epoch_rng,
                    )
                )
                logger.info("Got trajectories for epoch=%s. Training the epoch now",i)
                runtime_info = self.trainer.train_an_epoch_gpu(
                    control_wstates, value_wstates, model_outputs, action_samples, rewards
                )
                logger.info("Trained the epoch=%s. ",i)
                mean_entropy = (
                    float(jnp.mean(model_outputs["policy_entropy"]))
                    if "policy_entropy" in model_outputs else None
                )
                mean_reward = float(jnp.mean(jnp.sum(rewards, axis=-1)))
                # PostUpdate on GPU: save after the scan completes if num_updates is a multiple of n
                if (
                    freq.type == CheckpointFrequencyTypes.POST_UPDATE
                    and self.params.num_updates % freq.content == 0
                ):
                    self._save_and_send_checkpoint(
                        tx,
                        label=f"epoch_{i}_updates",
                        epoch_index=i,
                        update_index=self.params.num_updates - 1,
                    )
            else:
                def _post_update_cb(update_idx: int) -> None:
                    if (update_idx + 1) % freq.content == 0:
                        self._save_and_send_checkpoint(
                            tx,
                            label=f"epoch_{i}_update_{update_idx}",
                            epoch_index=i,
                            update_index=update_idx,
                        )

                post_update_callback = (
                    _post_update_cb
                    if freq.type == CheckpointFrequencyTypes.POST_UPDATE
                    else None
                )
                trajs = self.trainer.gen_trajectories(
                    self.params.num_trajs,
                    simulator_state,
                )
                runtime_info = self.trainer.train_an_epoch(trajs, post_update_callback)
                mean_entropy = self.trainer.get_rollout_mean_entropy(trajs)
                mean_reward = self.trainer.get_rollout_mean_reward(trajs)

            if (
                freq.type == CheckpointFrequencyTypes.POST_EPOCH
                and (i + 1) % freq.content == 0
            ):
                self._save_and_send_checkpoint(
                    tx,
                    label=f"epoch_{i}",
                    epoch_index=i,
                    update_index=self.params.num_updates - 1,
                )
            logger.info(
                "Epoch completed: id=%s epoch=%s mean_reward=%.6f mean_entropy=%s",
                self.id,
                i,
                mean_reward,
                mean_entropy,
            )

            # Preserve the legacy outgoing metrics until the output-contract pass.
            num_updates = len(runtime_info.mean_policy_loss())
            update_steps = [float(i * num_updates + u) for u in range(num_updates)]
            metrics = [
                {"type": name, "content": {"x_axis": xs, "y_axis": ys}}
                for name, xs, ys in (
                    ("MeanReward", [float(i)], [mean_reward]),
                    ("MeanPolicyLoss", update_steps, list(runtime_info.mean_policy_loss())),
                    ("ValueLoss", update_steps, list(runtime_info.mean_value_loss())),
                    ("PolicyGradientNorm", update_steps, list(runtime_info.policy_gradient_norm())),
                    ("ValueGradientNorm", update_steps, list(runtime_info.value_gradient_norm())),
                )
            ]
            if mean_entropy is not None:
                metrics.insert(0, {
                    "type": "MeanEntropy",
                    "content": {"x_axis": [float(i)], "y_axis": [mean_entropy]},
                })
            tx.send_training_updates({
                "id": self.id,
                "updates": [{
                    "training_run_state": {
                        "type": "PPOTrainingRunState",
                        "content": {"epoch_index": i, "update_index": num_updates - 1},
                    },
                    "logs": [], "metrics": metrics, "checkpoints": [],
                }],
            })

        if freq.type == CheckpointFrequencyTypes.LAST_ONLY and last_epoch_run >= 0:
            self._save_and_send_checkpoint(
                tx,
                label="last",
                epoch_index=last_epoch_run,
                update_index=self.params.num_updates - 1,
            )

        logger.info("Training run finished: id=%s", self.id)
