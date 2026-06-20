from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import io
import logging
import os
import zipfile
from typing import Any, Callable

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
from workflows.ppo_trainer import PPOTrainer, PPOTrainedModel
from simulator.simulator import Simulator

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
        simulator: Simulator,
        model: PPOTrainedModel,
    ) -> None:
        self.id = training_id
        self.simulator = simulator
        self.model = model
        self.training_controller = training_controller
        self.params: PPOTrainingMethod = training_controller.training_method.content
        self.trainer = PPOTrainer(
            simulator=simulator,
            model=model,
            num_trajs=self.params.num_trajs,
            traj_depth=self.params.num_steps_per_traj,
            num_updates=self.params.num_updates,
            epsilon=self.params.epsilon,
            gamma_lambda=(self.params.gamma, self.params.lambda_),
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

    def _is_gpu_training(self) -> bool:
        mode = self.training_controller.acceleration_mode
        return mode in (
            TrainingAccelerationMode.MLGPUSIMULATIONCPU,
            TrainingAccelerationMode.MLGPUSIMULATIONGPU,
        )

    def run_training_blocking(
        self,
        tx: TrainingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        use_gpu = self._is_gpu_training()
        logger.info("Training run started: id=%s use_gpu=%s", self.id, use_gpu)

        if use_gpu:
            import jax
            from simulator import MjxSimulator
            mjx_sim: MjxSimulator = self.simulator  # type: ignore[assignment]
            mjx_sim.reset_state()
            base_mjx_data = mjx_sim._mjx_data
            base_rng = jax.random.PRNGKey(0)

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
                base_rng, epoch_rng = jax.random.split(base_rng)
                control_wstates, value_wstates, model_outputs, action_samples, rewards = (
                    self.trainer.gen_trajectories_gpu(
                        self.params.num_trajs,
                        base_mjx_data,
                        epoch_rng,
                    )
                )
                logger.info("Got trajectories for epoch=%s. Training the epoch now",i)
                runtime_info = self.trainer.train_an_epoch_gpu(
                    control_wstates, value_wstates, action_samples, rewards, model_outputs
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
                trajs = self.trainer.gen_trajectories(self.params.num_trajs)
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
