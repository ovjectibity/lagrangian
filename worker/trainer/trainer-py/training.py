from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from commondef.remoteworkermessages import (
    StopTrainingMessage,
    TrainingRemoteWorkerClientMessageStopTrainingMessage,
    TrainingRemoteWorkerClientMessageTrainingUpdatesMessage,
    TrainingUpdatesMessage,
)
from commondef.training import (
    MetricsData,
    PPOTrainingMethod,
    TrainingMetricsMeanEntropy,
    TrainingMetricsMeanPolicyLoss,
    TrainingMetricsMeanReward,
    TrainingMetricsPolicyGradientNorm,
    TrainingMetricsValueGradientNorm,
    TrainingMetricsValueLoss,
)
from ppo_trainer import PPOTrainer, PPOTrainedModel
from simulator import Simulator


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
        wrapped = TrainingRemoteWorkerClientMessageTrainingUpdatesMessage(content=message)
        self._send_message(wrapped.model_dump(mode="json", by_alias=True))

    def send_stop_training(self, message: StopTrainingMessage) -> None:
        wrapped = TrainingRemoteWorkerClientMessageStopTrainingMessage(content=message)
        self._send_message(wrapped.model_dump(mode="json", by_alias=True))


class TrainingWorkflow:
    def __init__(
        self,
        training_id: str,
        params: PPOTrainingMethod,
        simulator: Simulator,
        model: PPOTrainedModel,
    ) -> None:
        self.id = training_id
        self.simulator = simulator
        self.model = model
        self.params = params
        # Mirrors app/src/workflows/training.rs values.
        self.trainer = PPOTrainer(
            simulator=simulator,
            model=model,
            num_trajs=100,
            traj_depth=10000,
            num_updates=10,
            epsilon=params.epsilon,
        )

    @staticmethod
    def log_and_send_blocking(
        tx: TrainingTaskSink,
        training_id: str,
        msg: str,
    ) -> None:
        print(msg)
        tx.send_training_updates(
            TrainingUpdatesMessage(
                id=training_id,
                metrics=[],
                logs=[msg],
            )
        )

    def run_training_blocking(
        self,
        tx: TrainingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        for i in range(self.params.num_epochs):
            if should_stop is not None and should_stop():
                break
            self.log_and_send_blocking(
                tx,
                self.id,
                f"Running the training epoch {i}",
            )
            trajs = self.trainer.gen_trajectories(10)
            runtime_info = self.trainer.train_an_epoch(trajs)
            mean_entropy = self.trainer.get_rollout_mean_entropy(trajs)
            mean_reward = self.trainer.get_rollout_mean_reward(trajs)

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
                    metrics=[
                        mean_entropy_m,
                        mean_reward_m,
                        mean_policy_loss_m,
                        mean_value_loss_m,
                        policy_grad_norm_m,
                        value_grad_norm_m,
                    ],
                    logs=[],
                )
            )

        tx.send_stop_training(StopTrainingMessage(id=self.id))
