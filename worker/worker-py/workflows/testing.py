from __future__ import annotations

from abc import ABC, abstractmethod
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
from commondef.testing import TestJobController
from models.model import TestingModel
from simulator.simulator import MjxSimulator, Simulator


logger = logging.getLogger(__name__)

Array = jax.Array
Params = Any

UPDATE_CHUNK_SIZE = 10


def _build_stream_bytes(
    test_id: str,
    sim_state: list[list[tuple[Any, Any]]],
    frames_u8: Any | None,
) -> bytes:
    """Build a `TestingStreamMessage` from an env-major [env][t] matrix of
    (xpos, xquat) plus optional camera frames, and let protobuf serialize it.

    `frames_u8`, if present, is a uint8 array of shape [nenv, t, ncam, h, w, c]
    (protobuf's `bytes` field needs raw bytes, hence the `.tobytes()`).
    See docs/user/specs/testing_updates.proto.
    """
    update = pb.TestUpdate(
        id=test_id,
        simulation_state=[
            pb.EnvUpdates(updates=[
                pb.ModelForward(nbody=len(xpos) // 3, xpos=xpos, xquat=xquat)
                for xpos, xquat in env_updates
            ])
            for env_updates in sim_state
        ],
    )
    if frames_u8 is not None:
        nenv, t, ncam, h, w = (int(d) for d in frames_u8.shape[:5])
        update.frames.CopyFrom(pb.CameraFrames(
            nenv=nenv, t=t, ncam=ncam, h=h, w=w,
            frames=np.ascontiguousarray(frames_u8).tobytes(),
        ))
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
        simulator: Simulator,
        model: TestingModel,
        rng_key: Any | None = None,
    ) -> None:
        self.id = test_id
        self.simulator = simulator
        self.model = model
        self.test_controller = test_controller
        self.params = test_controller.test_params
        # Single seed threaded from main.py; the batched scan splits fresh
        # per-step, per-env keys from it. Sequential inference uses the model's
        # own held key (via get_test_action).
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
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
        pure_fn = self.model.get_test_policy()
        params = self.model.get_test_params()

        batched_data = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * num_parallel_envs),
            mjx_sim._mjx_data,
        )

        state_fn = mjx_sim.make_state_fn(self.model.get_control_input_spec())

        def scan_body(carry: Any, _: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            bd, rng = carry
            rng, step_key = jax.random.split(rng)
            env_keys = jax.random.split(step_key, num_parallel_envs)
            parts_batched = jax.vmap(state_fn)(bd)
            # Pass the labeled parts dict straight to the policy (it splits
            # cameras from proprio internally); thread a fresh per-env sampling
            # key in via "rng_key".
            actions = jax.vmap(
                lambda p, k: pure_fn(params, {**p, "rng_key": k})
            )(parts_batched, env_keys)
            bd_next = jax.vmap(lambda d, a: mjx_sim.forward_dyn_pure(d, a)[0])(bd, actions)
            return (bd_next, rng), (parts_batched["cameras"], bd_next.xpos[:, :nbody], bd_next.xquat[:, :nbody])

        remaining = num_timesteps
        while remaining > 0:
            if should_stop is not None and should_stop():
                logger.info("Test run stop requested: id=%s remaining=%s", self.id, remaining)
                break
            logger.info("Running the first loop of batched rollout for _run_mjx_batched in testing.py %s", remaining)
            chunk = min(UPDATE_CHUNK_SIZE, remaining)
            (batched_data, self._rng_key), (camera_frames, chunk_xpos, chunk_xquat) = jax.lax.scan(
                scan_body, (batched_data, self._rng_key), None, length=chunk
            )
            # chunk_xpos: [chunk, num_parallel_envs, nbody, 3]
            xpos_np = chunk_xpos.reshape(chunk, num_parallel_envs, -1).tolist()
            xquat_np = chunk_xquat.reshape(chunk, num_parallel_envs, -1).tolist()
            # env-major [env][timestep] of (xpos, xquat) to match the frames layout
            sim_state = [
                [(xpos_np[t][i], xquat_np[t][i]) for t in range(chunk)]
                for i in range(num_parallel_envs)
            ]
            # camera_frames: [chunk, num_parallel_envs, ncam, H, W, C] float32 in
            # [0, 1]; frames field is u8, so denormalise and reorder to env-major
            # [nenv, t, ncam, H, W, C].
            frames_u8 = np.asarray(
                (camera_frames * 255.0).round().astype(jnp.uint8)
            ).transpose(1, 0, 2, 3, 4, 5)
            tx.send_test_updates(_build_stream_bytes(self.id, sim_state, frames_u8))
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
            parts = self.simulator.get_model_state(self.model.get_control_input_spec())
            # Pass the labeled parts dict; get_policy_output splits cameras from proprio.
            action = self.model.get_test_action(parts)
            self.simulator.forward_dyn(action)
            xpos, xquat = self.simulator.get_simul_update()
            # cameras: [ncam, H, W, C] float32 in [0, 1] when requested; frames
            # field is u8, so denormalise and add the leading [nenv=1, t=1] dims.
            cameras = parts.get("cameras")
            frames_u8 = (
                np.asarray((cameras * 255.0).round().astype(jnp.uint8))[None, None]
                if cameras is not None else None
            )
            sim_state = [[(xpos, xquat)]]  # single env, single timestep
            tx.send_test_updates(_build_stream_bytes(self.id, sim_state, frames_u8))
