from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import functools
import io
import logging
import math
import os
import zipfile
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from commondef.remoteworkermessages import (
    Checkpoint,
    StopTrainingMessage,
    RemoteWorkerClientMessageStopTrainingMessage,
    RemoteWorkerClientMessageTrainingUpdatesMessage,
    TrainingUpdate,
    TrainingUpdatesMessage,
)
from commondef.training import (
    CheckpointFrequencyTypes,
    ControlModelVLA,
    ControlOutputModeFlowMatchingAction,
    MetricsData,
    PPOTrainingMethod,
    PPOTrainingRunState,
    TrainingAccelerationMode,
    TrainingJobController,
    TrainingMetricsMeanEntropy,
    TrainingMetricsMeanPolicyLoss,
    TrainingMetricsMeanReward,
    TrainingMetricsPolicyGradientNorm,
    TrainingMetricsValueGradientNorm,
    TrainingMetricsValueLoss,
    TrainingRunStatePPOTrainingRunState,
)
from workflows.ppo_trainer import PPOTrainer
from models.mlp import MLPModel
from models.model import PPOTrainedModel
from models.pi0.pi0 import Pi0Model
from simulator.simulator import MjxSimulator, MujocoSimulator

logger = logging.getLogger(__name__)


class TrainingTaskSink(ABC):
    @abstractmethod
    def send_training_updates(self, message: TrainingUpdatesMessage) -> None:
        pass

    @abstractmethod
    def send_stop_training(self, message: StopTrainingMessage) -> None:
        pass


class CallbackTrainingTaskSink(TrainingTaskSink):
    """
    Adapter for websocket integration:
    pass a callable that sends the serialized payload to your server.
    """

    def __init__(self, send_message: Callable[[dict[str, Any]], None]) -> None:
        self._send_message = send_message

    def send_training_updates(self, message: TrainingUpdatesMessage) -> None:
        wrapped = RemoteWorkerClientMessageTrainingUpdatesMessage(content=message)
        self._send_message(wrapped.model_dump(mode="json", by_alias=True))

    def send_stop_training(self, message: StopTrainingMessage) -> None:
        wrapped = RemoteWorkerClientMessageStopTrainingMessage(content=message)
        self._send_message(wrapped.model_dump(mode="json", by_alias=True))


class TrainingWorkflow:
    def __init__(
        self,
        training_id: str,
        training_controller: TrainingJobController,
        model_path: str,
        checkpoint_paths: list[str] | None = None,
        rng_key: jax.Array | None = None,
    ) -> None:
        self.id = training_id
        self.training_controller = training_controller
        self.params: PPOTrainingMethod = training_controller.training_method.content

        root_rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        model_rng, workflow_rng = jax.random.split(root_rng)
        self._rng_key, trainer_key = jax.random.split(workflow_rng)

        reward_fn = self._load_user_reward_fn(
            self.params.reward_function.reward_function_code
        )
        self.model = self._build_model_for_training(
            self.params,
            reward_fn,
            rng_key=model_rng,
        )
        for checkpoint_path in checkpoint_paths or []:
            try:
                self._load_checkpoint_into_model(self.model, checkpoint_path)
            except Exception:
                logger.exception(
                    "Checkpoint load failed: training_id=%s checkpoint_path=%s",
                    self.id,
                    checkpoint_path,
                )

        mode = self.training_controller.acceleration_mode
        needs_render_context = any(
            input_spec.all_cameras
            for input_spec in (
                self.model.get_control_input_spec(),
                self.model.get_value_input_spec(),
                self.model.get_reward_input_spec(),
            )
        )
        if mode in (
            TrainingAccelerationMode.MLCPUSIMULATIONCPU,
            TrainingAccelerationMode.MLGPUSIMULATIONCPU,
        ):
            mujoco_data, mujoco_model = MujocoSimulator.init(model_path=model_path)
            contact_sensor_addresses, contact_body_pairs = (
                MujocoSimulator._get_contact_pair_sensor_metadata(mujoco_model)
            )
            render_ctx = (
                MujocoSimulator.get_render_context(mujoco_model)
                if needs_render_context
                else None
            )
            self.simulator_state: dict[str, Any] = {
                "simulator_class": MujocoSimulator,
                "mjx_data": None,
                "mjx_model": None,
                "mujoco_data": mujoco_data,
                "mujoco_model": mujoco_model,
                "render_ctx": render_ctx,
                "render_ctx_pytree": None,
                "contact_sensor_addresses": contact_sensor_addresses,
                "contact_body_pairs": contact_body_pairs,
            }
        elif mode == TrainingAccelerationMode.MLGPUSIMULATIONGPU:
            (
                mjx_data,
                mjx_model,
                mujoco_data,
                mujoco_model,
            ) = MjxSimulator.init(model_path=model_path)
            contact_sensor_addresses, contact_body_pairs = (
                MjxSimulator._get_contact_pair_sensor_metadata(mujoco_model)
            )
            render_ctx = (
                MjxSimulator.get_render_context(
                    mujoco_model,
                    nworld=self.params.num_trajs,
                )
                if needs_render_context
                else None
            )
            self.simulator_state = {
                "simulator_class": MjxSimulator,
                "mjx_data": mjx_data,
                "mjx_model": mjx_model,
                "mujoco_data": mujoco_data,
                "mujoco_model": mujoco_model,
                "render_ctx": render_ctx,
                "render_ctx_pytree": (
                    render_ctx.pytree() if render_ctx is not None else None
                ),
                "contact_sensor_addresses": jnp.asarray(
                    contact_sensor_addresses,
                    dtype=jnp.int32,
                ),
                "contact_body_pairs": jnp.asarray(
                    contact_body_pairs,
                    dtype=jnp.int32,
                ),
            }
        else:
            raise ValueError(f"Unsupported local training acceleration mode: {mode}")

        self.trainer = PPOTrainer(
            model=self.model,
            num_trajs=self.params.num_trajs,
            traj_depth=self.params.num_steps_per_traj,
            num_updates=self.params.num_updates,
            epsilon=self.params.epsilon,
            gamma_lambda=(self.params.gamma, self.params.lambda_),
            rng_key=trainer_key,
        )
        self._ckpt_dir = f"/tmp/checkpoints/{training_id}"
        os.makedirs(self._ckpt_dir, exist_ok=True)
        self._checkpointer = ocp.PyTreeCheckpointer()
        logger.info(
            "TrainingWorkflow initialized: id=%s epochs=%s epsilon=%s acceleration=%s",
            self.id,
            self.params.num_epochs,
            self.params.epsilon,
            self.training_controller.acceleration_mode,
        )

    @staticmethod
    def _load_user_reward_fn(code: str) -> Any:
        namespace: dict[str, Any] = {
            "jax": jax,
            "jnp": jnp,
            "numpy": np,
            "np": np,
            "math": math,
            "functools": functools,
            "__builtins__": {"__import__": __import__},
        }
        exec(compile(code, "<reward_fn>", "exec"), namespace)
        reward_fn = namespace.get("reward")
        if reward_fn is None or not callable(reward_fn):
            raise ValueError("reward function not found in code")
        return reward_fn

    @staticmethod
    def _build_model_for_training(
        params: PPOTrainingMethod,
        reward_fn: Any,
        rng_key: Any = None,
    ) -> PPOTrainedModel:
        if isinstance(params.control_model, ControlModelVLA):
            if not isinstance(
                params.control_output_mode,
                ControlOutputModeFlowMatchingAction,
            ):
                raise ValueError(
                    "VLA control model requires "
                    "control_output_mode=FlowMatchingAction(FlowNoise)"
                )
            vla_spec = params.control_model.content
            noise_mlp_spec = params.control_output_mode.content.content.noise_model
            return Pi0Model(
                vla_spec,
                noise_mlp_spec,
                control_input_spec=vla_spec.input_spec,
                value_input_spec=params.value_model.content.input_spec,
                reward_input_spec=params.reward_function.input_spec,
                value_mlp_spec=params.value_model.content,
                reward_fn=reward_fn,
                rng_key=rng_key,
            )

        control_mlp = params.control_model.content
        value_mlp = params.value_model.content
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
            control_input_spec=control_mlp.input_spec,
            value_input_spec=value_mlp.input_spec,
            reward_input_spec=params.reward_function.input_spec,
            reward_fn=reward_fn,
            rng_key=rng_key,
        )

    @staticmethod
    def _load_checkpoint_into_model(
        model: PPOTrainedModel,
        checkpoint_path: str,
    ) -> None:
        checkpointer = ocp.PyTreeCheckpointer()
        model.policy_store = checkpointer.restore(
            os.path.join(checkpoint_path, "policy"),
            item=model.get_policy_store(),
        )
        model.value_store = checkpointer.restore(
            os.path.join(checkpoint_path, "value"),
            item=model.get_value_store(),
        )
        logger.info("Checkpoint loaded: checkpoint_path=%s", checkpoint_path)

    @staticmethod
    def log_and_send_blocking(
        tx: TrainingTaskSink,
        training_id: str,
        msg: str,
    ) -> None:
        logger.info("training_id=%s %s", training_id, msg)
        tx.send_training_updates(
            TrainingUpdatesMessage(
                id=training_id,
                updates=[
                    TrainingUpdate(
                        training_run_state=TrainingRunStatePPOTrainingRunState(
                            content=PPOTrainingRunState(epoch_index=0, update_index=0)
                        ),
                        logs=[msg],
                        metrics=[],
                        checkpoints=[],
                    )
                ],
            )
        )

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

        checkpoint = Checkpoint(model_zip_base64=model_zip_base64)
        tx.send_training_updates(
            TrainingUpdatesMessage(
                id=self.id,
                updates=[
                    TrainingUpdate(
                        training_run_state=TrainingRunStatePPOTrainingRunState(
                            content=PPOTrainingRunState(
                                epoch_index=epoch_index,
                                update_index=update_index,
                            )
                        ),
                        logs=[],
                        metrics=[],
                        checkpoints=[checkpoint],
                    )
                ],
            )
        )
        logger.info("Checkpoint sent: training_id=%s epoch=%s update=%s", self.id, epoch_index, update_index)

    def run_training_blocking(
        self,
        tx: TrainingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
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
                        self.simulator_state,
                        epoch_rng,
                    )
                )
                logger.info("Got trajectories for epoch=%s. Training the epoch now",i)
                runtime_info = self.trainer.train_an_epoch_gpu(
                    control_wstates, value_wstates, model_outputs, action_samples, rewards
                )
                logger.info("Trained the epoch=%s. ",i)
                mean_entropy = 0.0  # entropy not computed in GPU path yet
                mean_reward = float(jax.numpy.mean(rewards))
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
                    self.simulator_state,
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
                "Epoch completed: id=%s epoch=%s mean_reward=%.6f mean_entropy=%.6f",
                self.id,
                i,
                mean_reward,
                mean_entropy,
            )

            mean_entropy_m = TrainingMetricsMeanEntropy(
                content=MetricsData(x_axis=[float(i)], y_axis=[mean_entropy])
            )
            mean_reward_m = TrainingMetricsMeanReward(
                content=MetricsData(x_axis=[float(i)], y_axis=[mean_reward])
            )

            num_updates = len(runtime_info.mean_policy_loss())
            base_update_step = i * num_updates
            update_steps = [float(base_update_step + u) for u in range(num_updates)]

            mean_policy_loss_m = TrainingMetricsMeanPolicyLoss(
                content=MetricsData(
                    x_axis=update_steps.copy(),
                    y_axis=list(runtime_info.mean_policy_loss()),
                )
            )
            mean_value_loss_m = TrainingMetricsValueLoss(
                content=MetricsData(
                    x_axis=update_steps.copy(),
                    y_axis=list(runtime_info.mean_value_loss()),
                )
            )
            policy_grad_norm_m = TrainingMetricsPolicyGradientNorm(
                content=MetricsData(
                    x_axis=update_steps.copy(),
                    y_axis=list(runtime_info.policy_gradient_norm()),
                )
            )
            value_grad_norm_m = TrainingMetricsValueGradientNorm(
                content=MetricsData(
                    x_axis=update_steps,
                    y_axis=list(runtime_info.value_gradient_norm()),
                )
            )

            tx.send_training_updates(
                TrainingUpdatesMessage(
                    id=self.id,
                    updates=[
                        TrainingUpdate(
                            training_run_state=TrainingRunStatePPOTrainingRunState(
                                content=PPOTrainingRunState(
                                    epoch_index=i,
                                    update_index=num_updates - 1,
                                )
                            ),
                            logs=[],
                            metrics=[
                                mean_entropy_m,
                                mean_reward_m,
                                mean_policy_loss_m,
                                mean_value_loss_m,
                                policy_grad_norm_m,
                                value_grad_norm_m,
                            ],
                            checkpoints=[],
                        )
                    ],
                )
            )

        if freq.type == CheckpointFrequencyTypes.LAST_ONLY and last_epoch_run >= 0:
            self._save_and_send_checkpoint(
                tx,
                label="last",
                epoch_index=last_epoch_run,
                update_index=self.params.num_updates - 1,
            )

        logger.info("Training run finished, sending StopTrainingMessage: id=%s", self.id)
        tx.send_stop_training(StopTrainingMessage(id=self.id))
