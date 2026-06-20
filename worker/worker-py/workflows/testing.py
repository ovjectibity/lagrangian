from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp

from commondef.remoteworkermessages import (
    ControlModelInput,
    ControlModelVLA,
    ModelForward,
    RemoteWorkerClientMessageStopTestingMessage,
    RemoteWorkerClientMessageTestUpdatesMessage,
    StopTestMessage,
    TestUpdateMessage,
    TrainingAccelerationMode,
)
from commondef.testing import TestJobController
from workflows.ppo_trainer import TestingModel
from simulator.simulator import MjxSimulator, Simulator


def _default_input_spec() -> ControlModelInput:
    return ControlModelInput(
        all_cameras=False,
        qpos_state=True,
        qvel_state=False,
        xpos_state=False,
        xquat_state=False,
        qdof_state=False,
    )

logger = logging.getLogger(__name__)

Array = jax.Array
Params = Any

UPDATE_CHUNK_SIZE = 10


class TestingTaskSink(ABC):
    @abstractmethod
    def send_test_updates(self, message: TestUpdateMessage) -> None:
        pass

    @abstractmethod
    def send_stop_testing(self, message: StopTestMessage) -> None:
        pass


class CallbackTestingTaskSink(TestingTaskSink):
    def __init__(self, send_message: Callable[[dict[str, Any]], None]) -> None:
        self._send_message = send_message

    def send_test_updates(self, message: TestUpdateMessage) -> None:
        wrapped = RemoteWorkerClientMessageTestUpdatesMessage(content=message)
        self._send_message(wrapped.model_dump(mode="json", by_alias=True))

    def send_stop_testing(self, message: StopTestMessage) -> None:
        wrapped = RemoteWorkerClientMessageStopTestingMessage(content=message)
        self._send_message(wrapped.model_dump(mode="json", by_alias=True))


class TestingWorkflow:
    def __init__(
        self,
        test_id: str,
        test_controller: TestJobController,
        simulator: Simulator,
        model: TestingModel,
    ) -> None:
        self.id = test_id
        self.simulator = simulator
        self.model = model
        self.test_controller = test_controller
        self.params = test_controller.test_params
        if isinstance(test_controller.test_params.control_model, ControlModelVLA):
            self._input_spec = test_controller.test_params.control_model.content.input
        else:
            self._input_spec = _default_input_spec()
        logger.info(
            "TestingWorkflow initialized: id=%s num_timesteps=%s num_parallel_envs=%s acceleration=%s",
            self.id,
            self.params.num_timesteps,
            self.params.num_parallel_envs,
            test_controller.acceleration_mode,
        )

    def _is_mjx(self) -> bool:
        return self.test_controller.acceleration_mode != TrainingAccelerationMode.MLCPUSIMULATIONCPU

    def run_test_blocking(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        use_mjx = self._is_mjx()
        num_timesteps = int(self.params.num_timesteps)
        num_parallel_envs = int(self.params.num_parallel_envs)
        logger.info(
            "Test run started: id=%s use_mjx=%s num_timesteps=%s num_parallel_envs=%s",
            self.id,
            use_mjx,
            num_timesteps,
            num_parallel_envs,
        )

        self.simulator.reset_state()

        if use_mjx and num_parallel_envs > 1 and isinstance(self.simulator, MjxSimulator):
            self._run_mjx_batched(tx, should_stop, num_timesteps, num_parallel_envs)
        else:
            self._run_sequential(tx, should_stop, num_timesteps)

        logger.info("Test run finished, sending StopTestingMessage: id=%s", self.id)
        tx.send_stop_testing(StopTestMessage(id=self.id))

    def _run_mjx_batched(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None,
        num_timesteps: int,
        num_parallel_envs: int,
    ) -> None:
        mjx_sim: MjxSimulator = self.simulator  # type: ignore[assignment]
        nbody = mjx_sim.nbody()
        pure_fn = self.model.get_pure_test_fn()
        params = self.model.get_test_params()

        batched_data = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * num_parallel_envs),
            mjx_sim._mjx_data,
        )

        state_fn = mjx_sim.make_state_fn(self._input_spec)
        assemble_fn = self.model.assemble_state

        def scan_body(bd: Any, _: Any) -> tuple[Any, tuple[Any, Any]]:
            parts_batched = jax.vmap(state_fn)(bd)
            states = jax.vmap(assemble_fn)(parts_batched)
            actions = jax.vmap(lambda s: pure_fn(params, s))(states)
            bd_next = jax.vmap(lambda d, a: mjx_sim.forward_dyn_pure(d, a)[0])(bd, actions)
            return bd_next, (bd_next.xpos[:, :nbody], bd_next.xquat[:, :nbody])

        remaining = num_timesteps
        while remaining > 0:
            if should_stop is not None and should_stop():
                logger.info("Test run stop requested: id=%s remaining=%s", self.id, remaining)
                break
            chunk = min(UPDATE_CHUNK_SIZE, remaining)
            batched_data, (chunk_xpos, chunk_xquat) = jax.lax.scan(
                scan_body, batched_data, None, length=chunk
            )
            # chunk_xpos: [chunk, num_parallel_envs, nbody, 3]
            xpos_np = chunk_xpos.reshape(chunk, num_parallel_envs, -1).tolist()
            xquat_np = chunk_xquat.reshape(chunk, num_parallel_envs, -1).tolist()
            updates = [
                [ModelForward(xpos=xpos_np[t][i], xquat=xquat_np[t][i])
                 for i in range(num_parallel_envs)]
                for t in range(chunk)
            ]
            tx.send_test_updates(TestUpdateMessage(id=self.id, updates=updates))
            remaining -= chunk

    def _run_sequential(
        self,
        tx: TestingTaskSink,
        should_stop: Callable[[], bool] | None,
        num_timesteps: int,
    ) -> None:
        for _ in range(num_timesteps):
            if should_stop is not None and should_stop():
                logger.info("Test run stop requested: id=%s", self.id)
                break
            parts = self.simulator.get_model_state(self._input_spec)
            state = self.model.assemble_state(parts)
            action = self.model.get_test_action(state)
            self.simulator.forward_dyn(action)
            xpos, xquat = self.simulator.get_simul_update()
            tx.send_test_updates(TestUpdateMessage(
                id=self.id,
                updates=[[ModelForward(xpos=xpos, xquat=xquat)]],
            ))
