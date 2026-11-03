from abc import ABC, abstractmethod
from typing import Any, Callable
from commondef.training import ControlPolicyInputSpec, ControlPolicyOutputSpec
from commondef.ioadapter import ControlPolicyInputState
import jax

Array = jax.Array
Params = Any

class TestingModel(ABC):
    """Minimal interface for test-time inference, independent of PPO concepts."""

    @abstractmethod
    def get_control_input_spec(self) -> ControlPolicyInputSpec:
        """Input spec the simulator uses for control-policy inference."""
        ...

    @abstractmethod
    def get_test_action(self, policy_input: ControlPolicyInputState) -> Array:
        """Run eager single-step inference using a structured simulator state."""
        ...

    @abstractmethod
    def get_test_policy(
        self,
    ) -> Callable[[Params, ControlPolicyInputState, Array], Array]:
        """Return pure ``(params, state, rng_key) -> action`` inference."""
        ...

    @abstractmethod
    def get_test_params(self) -> Params:
        """Params passed as first arg to get_pure_test_fn."""
        ...

    @abstractmethod
    def get_control_policy_output_spec(self) -> ControlPolicyOutputSpec:
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
    def get_control_input_spec(self) -> ControlPolicyInputSpec:
        pass

    @abstractmethod
    def get_value_input_spec(self) -> ControlPolicyInputSpec:
        pass

    @abstractmethod
    def get_reward_input_spec(self) -> ControlPolicyInputSpec:
        pass

    # --- Reward (pre/post structured state, matching the Rust/C ABI contract) ---

    @abstractmethod
    def get_reward(
        self,
        state_pre: ControlPolicyInputState,
        state_post: ControlPolicyInputState,
    ) -> float:
        pass

    @abstractmethod
    def get_pure_reward_fn(
        self,
    ) -> Callable[[ControlPolicyInputState, ControlPolicyInputState], float]:
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
        policy_input: ControlPolicyInputState,
    ) -> dict[str, Array]:
        """Return an eager policy sample using the model-owned RNG state."""
        pass

    @abstractmethod
    def get_policy_fn(
        self,
    ) -> Callable[[Params, ControlPolicyInputState, Array], dict[str, Array]]:
        """Return pure ``(params, state, rng_key) -> sampled policy output``."""
        pass
    
    # --- The get_policy_output_log_prob function recomputes what it would take to generate the output sample
    # --- given the input & the sample process state with the current model & not with the model 
    # --- which generated that output process ----------------

    @abstractmethod
    def get_policy_output_log_prob(
        self,
        policy_input: ControlPolicyInputState,
        policy_output: dict[str, Array],
    ) -> Array:
        pass

    @abstractmethod
    def get_policy_output_log_prob_fn(
        self,
    ) -> Callable[[Params, ControlPolicyInputState, dict[str, Array]], Array]:
        pass

    @abstractmethod
    def get_value_output(self, value_input: ControlPolicyInputState) -> float:
        pass

    @abstractmethod
    def get_value_fn(
        self,
    ) -> Callable[[Params, ControlPolicyInputState], Array]:
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
    def get_control_policy_output_spec(self) -> ControlPolicyOutputSpec:
        pass
