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
from commondef.testing import TestJobController, TestMethodCustom
from commondef.training import ControlModelVLA, ModelInputSpec
from models.mlp import MLPModel
from models.model import TestingModel
from models.pi0.pi0 import Pi0Model
from simulator.simulator import MjxSimulator, MujocoSimulator


logger = logging.getLogger(__name__)

Array = jax.Array

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
        model_path: str,
        rng_key: jax.Array | None = None,
    ) -> None:
        self.id = test_id
        self.test_controller = test_controller
        self.simulator_state: dict[str, Any] = {}

        try:
            if not isinstance(test_controller.test_method, TestMethodCustom):
                raise ValueError(
                    "TestingWorkflow currently supports only custom test methods"
                )
            test_method = test_controller.test_method.content
            self.params = test_method.test_params

            root_rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
            model_rng, self._rng_key = jax.random.split(root_rng)
            self.model = self._build_model_for_testing(
                test_method.control_model,
                rng_key=model_rng,
            )

            control_spec = self.model.get_control_input_spec()
            mode = self.test_controller.acceleration_mode
            if mode == TrainingAccelerationMode.MLGPUSIMULATIONGPU:
                (
                    mjx_data,
                    mjx_model,
                    mujoco_data,
                    mujoco_model,
                ) = MjxSimulator.init(model_path=model_path)
                contact_sensor_addresses, contact_body_pairs = (
                    MjxSimulator._get_contact_pair_sensor_metadata(mujoco_model)
                )
                self.simulator_state = {
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
                if control_spec.all_cameras:
                    render_ctx = MjxSimulator.get_render_context(
                        mujoco_model,
                        nworld=self.params.num_parallel_envs,
                    )
                    self.simulator_state["render_ctx"] = render_ctx
                    self.simulator_state["render_ctx_pytree"] = render_ctx.pytree()
            elif mode in (
                TrainingAccelerationMode.MLCPUSIMULATIONCPU,
                TrainingAccelerationMode.MLGPUSIMULATIONCPU,
            ):
                mujoco_data, mujoco_model = MujocoSimulator.init(
                    model_path=model_path
                )
                contact_sensor_addresses, contact_body_pairs = (
                    MujocoSimulator._get_contact_pair_sensor_metadata(mujoco_model)
                )
                self.simulator_state = {
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
                if control_spec.all_cameras:
                    self.simulator_state["render_ctx"] = (
                        MujocoSimulator.get_render_context(mujoco_model)
                    )
            else:
                raise ValueError(
                    f"Unsupported local testing acceleration mode: {mode}"
                )

            logger.info(
                "TestingWorkflow initialized: id=%s num_timesteps=%s "
                "num_parallel_envs=%s acceleration=%s",
                self.id,
                self.params.num_timesteps,
                self.params.num_parallel_envs,
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
        render_ctx = self.simulator_state.get("render_ctx")
        self.simulator_state["render_ctx"] = None
        self.simulator_state["render_ctx_pytree"] = None
        if render_ctx is None:
            return

        close = getattr(render_ctx, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception(
                    "Failed to close testing render context: test_id=%s",
                    self.id,
                )
                return
        logger.info("Testing render context released: test_id=%s", self.id)

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
        use_mjx = (
            self.test_controller.acceleration_mode
            == TrainingAccelerationMode.MLGPUSIMULATIONGPU
        )
        num_timesteps = int(self.params.num_timesteps)
        num_parallel_envs = int(self.params.num_parallel_envs)
        logger.info(
            "Test run started: id=%s use_mjx=%s num_timesteps=%s num_parallel_envs=%s",
            self.id,
            use_mjx,
            num_timesteps,
            num_parallel_envs,
        )

        if use_mjx:
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
        simulator_class = self.simulator_state["simulator_class"]
        if simulator_class is not MjxSimulator:
            raise ValueError("MJX batched testing requires MjxSimulator state")
        mjx_data = self.simulator_state["mjx_data"]
        mjx_model = self.simulator_state["mjx_model"]
        if mjx_data is None or mjx_model is None:
            raise ValueError("MJX data and model are required for batched testing")

        nbody = int(mjx_model.nbody)
        test_policy_fn = self.model.get_test_policy()
        params = self.model.get_test_params()
        output_mode = self.model.get_control_policy_output_mode()

        batched_data = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * num_parallel_envs),
            mjx_data,
        )

        control_spec = self.model.get_control_input_spec()
        render_ctx = self.simulator_state["render_ctx_pytree"]
        contact_sensor_addresses = self.simulator_state["contact_sensor_addresses"]
        contact_body_pairs = self.simulator_state["contact_body_pairs"]
        has_cameras = control_spec.all_cameras and int(mjx_model.ncam) > 0

        def state_fn(data: Any) -> dict[str, Array]:
            return simulator_class.get_model_state_pure(
                control_spec,
                data,
                mjx_model,
                render_ctx,
                contact_sensor_addresses,
                contact_body_pairs,
            )

        def scan_body(carry: Any, _: Any) -> tuple[Any, tuple[Any, Any, Any]]:
            bd, rng = carry
            rng, step_key = jax.random.split(rng)
            env_keys = jax.random.split(step_key, num_parallel_envs)
            parts_batched = jax.vmap(state_fn)(bd)
            # Pass the labeled parts dict straight to the policy (it splits
            # cameras from proprio internally); thread a fresh per-env sampling
            # key in via "rng_key".
            actions = jax.vmap(
                lambda p, k: test_policy_fn(params, {**p, "rng_key": k})
            )(parts_batched, env_keys)
            # Batched chunked policies such as Pi0 return
            # [num_envs, action_steps, action_dim]; execute only the first
            # action for each environment during this simulator step.
            if actions.ndim == 3:
                actions = actions[:, 0]

            def step(data: Any, action: Array) -> Any:
                controls = simulator_class.get_actuator_input_from_control_input(
                    mjx_model,
                    data,
                    action,
                    output_mode,
                )
                return simulator_class.forward_dyn_pure(data, mjx_model, controls)

            bd_next = jax.vmap(step)(bd, actions)
            camera_frames = (
                parts_batched["cameras"]
                if has_cameras
                else jnp.empty((num_parallel_envs, 0, 0, 0, 3), dtype=jnp.float32)
            )
            return (bd_next, rng), (
                camera_frames,
                bd_next.xpos[:, :nbody],
                bd_next.xquat[:, :nbody],
            )

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
            frames_u8 = None
            if has_cameras:
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
        simulator_class = self.simulator_state["simulator_class"]
        if simulator_class is not MujocoSimulator:
            raise ValueError("Sequential testing requires MujocoSimulator state")
        data = self.simulator_state["mujoco_data"]
        model = self.simulator_state["mujoco_model"]
        if data is None or model is None:
            raise ValueError("MuJoCo data and model are required for sequential testing")

        control_spec = self.model.get_control_input_spec()
        render_ctx = self.simulator_state["render_ctx"]
        contact_sensor_addresses = self.simulator_state["contact_sensor_addresses"]
        contact_body_pairs = self.simulator_state["contact_body_pairs"]
        output_mode = self.model.get_control_policy_output_mode()

        for i in range(num_timesteps):
            if should_stop is not None and should_stop():
                logger.info("Test run stop requested: id=%s", self.id)
                break
            logger.info("Running the sequential test loop at timestep: %s", i)
            parts = simulator_class.get_model_state_pure(
                control_spec,
                data,
                model,
                render_ctx,
                contact_sensor_addresses,
                contact_body_pairs,
            )
            action = self.model.get_test_action(parts)
            # Chunked policies such as Pi0 return [action_steps, action_dim];
            # this workload currently executes only the first action per step.
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
            cameras = parts.get("cameras")
            frames_u8 = (
                np.asarray((cameras * 255.0).round().astype(np.uint8))[None, None]
                if cameras is not None
                else None
            )
            sim_state = [[(xpos, xquat)]]
            tx.send_test_updates(_build_stream_bytes(self.id, sim_state, frames_u8))

        self.simulator_state["mujoco_data"] = data
