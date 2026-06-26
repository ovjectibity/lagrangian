from abc import ABC, abstractmethod
from typing import Any, Callable
from commondef.training import ModelInputSpec, ControlPolicyOutputMode
import jax

Array = jax.Array
Params = Any

class TestingModel(ABC):
    """Minimal interface for test-time inference, independent of PPO concepts."""

    @abstractmethod
    def get_control_input_spec(self) -> ModelInputSpec:
        """Input spec the simulator uses to build the state dict for control inference."""
        ...

    @abstractmethod
    def assemble_control_state(self, parts: dict[str, Array]) -> Array:
        """Assemble labeled state parts dict into a flat tensor for control inference.

        parts keys: "cameras", "qdof", "qpos", "qvel", "xpos", "xquat" (only flags set).
        Must be a pure JAX function (safe to jax.vmap over batched pytree dicts).
        """
        ...

    @abstractmethod
    def get_test_action(self, state: Array | dict[str, Array]) -> Array:
        """Single-step inference: state (labeled parts dict or flat tensor) → control action."""
        ...

    @abstractmethod
    def get_test_policy(self) -> Callable[[Params, Array | dict[str,Array]], Array]:
        """Pure (params, state) -> action for vmap-able MJX batched testing."""
        ...

    @abstractmethod
    def get_test_params(self) -> Params:
        """Params passed as first arg to get_pure_test_fn."""
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
    def get_control_input_spec(self) -> ModelInputSpec:
        pass

    @abstractmethod
    def get_value_input_spec(self) -> ModelInputSpec:
        pass

    @abstractmethod
    def get_reward_input_spec(self) -> ModelInputSpec:
        pass

    # --- State assembly (model owns ordering/concatenation) ---

    @abstractmethod
    def assemble_control_state(self, parts: dict[str, Array]) -> Array:
        """Assemble labeled state parts dict into a flat tensor for the control policy."""
        pass

    @abstractmethod
    def assemble_value_state(self, parts: dict[str, Array]) -> Array:
        """Assemble labeled state parts dict into a flat tensor for the value function."""
        pass

    # --- Reward (pre/post state dicts, matching Rust and the C ABI reward interface) ---

    @abstractmethod
    def get_reward(
        self,
        parts_pre: dict[str, Array],
        parts_post: dict[str, Array],
    ) -> float:
        pass

    @abstractmethod
    def get_pure_reward_fn(self) -> Callable[[dict[str, Array], dict[str, Array]], float]:
        pass

    # --- Policy / value inference ---

    @abstractmethod
    def get_policy_output(self, state: Array) -> Array:
        pass

    @abstractmethod
    def get_value_output(self, state: Array) -> float:
        pass

    @abstractmethod
    def get_policy(self) -> Callable[[Params, Array], Array]:
        pass

    @abstractmethod
    def get_value_func(self) -> Callable[[Params, Array], Array]:
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
    def get_control_policy_output_mode(self) -> ControlPolicyOutputMode:
        pass

    @abstractmethod
    def get_pure_policy_fn(self) -> Callable[[Params, Array], Array]:
        pass