from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
from pathlib import Path
from commondef.records.assets import CheckpointModelType, CheckpointRef
from commondef.records.workload_graph import (
    PolicyInputSpec, PolicyOutputSpec, ModelType, ModelTypeNew,
    CheckpointTypePretrained, CheckpointTypeCheckpointID, Pretrained,
    MLModel, MLModelVLA,
)
from commondef.ioadapter import PolicyInputState
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

Array = jax.Array
Params = Any


def load_checkpoint(
    model: Any,
    request: ModelType,
    model_spec: MLModel,
    model_type: CheckpointModelType,
    checkpoints: dict[str, tuple[CheckpointRef, Path]],
    *,
    restore_optimizer: bool = False,
) -> dict[str, Any] | None:
    """Restore requested per-model weights, optionally including Adam state.

    Training requests optimizer state; inference only loads the weights.
    Pretrained Pi-0 continues to use PI0_CHECKPOINT_DIR. A requested checkpoint
    failure is fatal rather than silently falling back to initialized weights.
    """
    if isinstance(request, ModelTypeNew):
        return None
    reference = request.content
    if isinstance(reference, CheckpointTypePretrained):
        if (
            reference.content != Pretrained.PI0
            or not isinstance(model_spec, MLModelVLA)
            or model_type != CheckpointModelType.POLICY
        ):
            raise ValueError("Pretrained Pi-0 is only supported for VLA policy models")
        return None
    if not isinstance(reference, CheckpointTypeCheckpointID):
        raise ValueError("Unsupported checkpoint request")
    checkpoint_id = reference.content
    try:
        if checkpoint_id not in checkpoints:
            raise ValueError("Checkpoint reference was not supplied")
        ref, directory = checkpoints[checkpoint_id]
        if ref.checkpoint_id != checkpoint_id or ref.model_type != model_type:
            raise ValueError("Checkpoint ID or model role does not match the requested model")
        if ref.model_spec.model_dump(mode="json") != model_spec.model_dump(mode="json"):
            raise ValueError("Checkpoint model specification does not match the requested model")

        if model_type == CheckpointModelType.VALUE:
            target = model.get_value_store_mut()
        elif model_type == CheckpointModelType.FLOWNOISE:
            target = model.get_policy_store_mut()["noise"]
        elif isinstance(model_spec, MLModelVLA):
            target = model.get_policy_store_mut()["pi0"]
        else:
            target = model.get_policy_store_mut()

        if not all((directory / name).is_dir() for name in ("weights", "optimizer")):
            raise ValueError("Checkpoint requires weights/ and optimizer/ directories")
        optimizer = None
        with ocp.PyTreeCheckpointer() as checkpointer:
            weights = checkpointer.restore(
                str(directory / "weights"), item=target,
                restore_args=ocp.checkpoint_utils.construct_restore_args(target),
            )
            pairs = [(weights, target)]
            if restore_optimizer:
                zeros = jax.tree_util.tree_map(jnp.zeros_like, target)
                optimizer_template = {"m": zeros, "v": zeros, "t": 0}
                optimizer = checkpointer.restore(
                    str(directory / "optimizer"), item=optimizer_template,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(optimizer_template),
                )
                pairs.append((optimizer, optimizer_template))
        for restored, expected in pairs:
            if jax.tree_util.tree_structure(restored) != jax.tree_util.tree_structure(expected):
                raise ValueError("Checkpoint tree structure does not match the model")
            for actual_leaf, expected_leaf in zip(
                jax.tree_util.tree_leaves(restored), jax.tree_util.tree_leaves(expected),
            ):
                if np.shape(actual_leaf) != np.shape(expected_leaf):
                    raise ValueError("Checkpoint tensor shape does not match the model")
        if optimizer is not None:
            step = np.asarray(optimizer["t"])
            if step.dtype.kind not in "iu" or int(step) < 0:
                raise ValueError("Checkpoint optimizer step must be a nonnegative integer")
        target.clear()
        target.update(weights)
        return optimizer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {model_type.value} checkpoint {checkpoint_id!r}: {exc}"
        ) from exc


class TestingModel(ABC):
    """Minimal interface for test-time inference, independent of PPO concepts."""

    @abstractmethod
    def get_control_input_spec(self) -> PolicyInputSpec:
        """Input spec the simulator uses for control-policy inference."""
        ...

    @abstractmethod
    def get_test_action(self, policy_input: PolicyInputState) -> Array:
        """Run eager single-step inference using a structured simulator state."""
        ...

    @abstractmethod
    def get_test_policy(
        self,
    ) -> Callable[[Params, PolicyInputState, Array], Array]:
        """Return pure ``(params, state, rng_key) -> action`` inference."""
        ...

    @abstractmethod
    def get_test_params(self) -> Params:
        """Params passed as first arg to get_pure_test_fn."""
        ...

    @abstractmethod
    def get_control_policy_output_spec(self) -> PolicyOutputSpec:
        """Static interpretation of the policy's flat control output."""
        ...


class PPOTrainedModel(ABC):
    @classmethod
    @abstractmethod
    def init_ppo_trained_model(
        cls,
        policy: Callable[[Params, Array], Array],
        val_estimator: Callable[[Params, Array], Array],
        pol_store: Params,
        val_estimator_store: Params,
    ) -> PPOTrainedModel:
        pass

    # --- Input spec ownership (mirrors Rust PPOTrainedModel trait) ---

    @abstractmethod
    def get_control_input_spec(self) -> PolicyInputSpec:
        pass

    @abstractmethod
    def get_value_input_spec(self) -> PolicyInputSpec:
        pass

    @abstractmethod
    def get_reward_input_spec(self) -> PolicyInputSpec:
        pass

    # --- Reward (pre/post structured state, matching the Rust/C ABI contract) ---

    @abstractmethod
    def get_reward(
        self,
        state_pre: PolicyInputState,
        state_post: PolicyInputState,
    ) -> float:
        pass

    @abstractmethod
    def get_pure_reward_fn(
        self,
    ) -> Callable[[PolicyInputState, PolicyInputState], float]:
        pass

    # --- Policy / value inference ---
    # --- The following methods move the action sampling & action 
    # --- sample log prob calculation logic to the model instead of 
    # --- Gaussian output assumption in ppo_trainer.py 
    # --- For flow matching VLAs like Pi-0 this might also 
    # --- require some additional state such as that generated during the unrolling process 

    @abstractmethod
    def get_policy_output(
        self,
        policy_input: PolicyInputState,
    ) -> dict[str, Array]:
        """Return an eager policy sample using the model-owned RNG state."""
        pass

    @abstractmethod
    def get_policy_fn(
        self,
    ) -> Callable[[Params, PolicyInputState, Array], dict[str, Array]]:
        """Return pure ``(params, state, rng_key) -> sampled policy output``."""
        pass
    
    # --- The get_policy_output_log_prob function recomputes what it would take to generate the output sample
    # --- given the input & the sample process state with the current model & not with the model 
    # --- which generated that output process ----------------

    @abstractmethod
    def get_policy_output_log_prob(
        self,
        policy_input: PolicyInputState,
        policy_output: dict[str, Array],
    ) -> Array:
        pass

    @abstractmethod
    def get_policy_output_log_prob_fn(
        self,
    ) -> Callable[[Params, PolicyInputState, dict[str, Array]], Array]:
        pass

    @abstractmethod
    def get_value_output(self, value_input: PolicyInputState) -> float:
        pass

    @abstractmethod
    def get_value_fn(
        self,
    ) -> Callable[[Params, PolicyInputState], Array]:
        pass

    @abstractmethod
    def get_policy_store(self) -> Params:
        pass

    @abstractmethod
    def get_policy_store_mut(self) -> Params:
        pass

    @abstractmethod
    def get_value_store(self) -> Params:
        pass

    @abstractmethod
    def get_value_store_mut(self) -> Params:
        pass

    @abstractmethod
    def get_control_policy_output_spec(self) -> PolicyOutputSpec:
        pass
