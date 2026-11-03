from abc import ABC, abstractmethod
from typing import Any, Callable
from commondef.training import PolicyInputSpec, PolicyOutputSpec
from commondef.ioadapter import PolicyInputState
import jax

Array = jax.Array
Params = Any

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
