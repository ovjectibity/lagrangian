"""JAX PyTree runtime states corresponding to control-policy input specs."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from commondef.training import (
    ControlPolicyInputSpecTypes,
    ControlPolicyOutputSpecTypes,
    InputStateTypes,
)


# Runtime state mirrors ControlPolicyInputSpec exactly. The arrays extracted
# from the simulator replace the spec's state-item leaves. Body names and
# tagged-union discriminants remain static PyTree metadata so JAX can compile
# one stable tree shape for a given input specification.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state_item",),
    meta_fields=("rebase_body_tree", "excluded_body_trees"),
)
@dataclass(frozen=True)
class InputStateControllableData:
    rebase_body_tree: str | None
    excluded_body_trees: tuple[str, ...]
    state_item: jnp.ndarray | np.ndarray


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("content",),
    meta_fields=("type",),
)
@dataclass(frozen=True)
class InputStateDataInputStateControllable:
    type: Literal[InputStateTypes.INPUT_STATE_CONTROLLABLE]
    content: InputStateControllableData


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("content",),
    meta_fields=("type",),
)
@dataclass(frozen=True)
class InputStateDataInputStateSensors:
    type: Literal[InputStateTypes.INPUT_STATE_SENSORS]
    content: jnp.ndarray | np.ndarray


InputStateData = (
    InputStateDataInputStateControllable | InputStateDataInputStateSensors
)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("states",),
    meta_fields=("base_body_name",),
)
@dataclass(frozen=True)
class StandardInputData:
    base_body_name: str | None
    states: list[InputStateData]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("content",),
    meta_fields=("type",),
)
@dataclass(frozen=True)
class ControlPolicyInputStateStandardInputSpec:
    type: Literal[ControlPolicyInputSpecTypes.STANDARD_INPUT_SPEC]
    content: StandardInputData


ControlPolicyInputState = ControlPolicyInputStateStandardInputSpec


# Runtime mirrors for control-policy output specs. These keep command metadata
# static and replace each requested state item with its policy-produced array.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state_item",),
    meta_fields=("is_delta", "rebase_body_tree", "excluded_body_trees"),
)
@dataclass(frozen=True)
class OutputStateTreeData:
    is_delta: bool
    rebase_body_tree: str | None
    excluded_body_trees: tuple[str, ...]
    state_item: jnp.ndarray | np.ndarray


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state_item",),
    meta_fields=("is_delta", "body_name"),
)
@dataclass(frozen=True)
class OutputStateSingularData:
    is_delta: bool
    body_name: str
    state_item: jnp.ndarray | np.ndarray


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state",),
    meta_fields=("base_body_name",),
)
@dataclass(frozen=True)
class ProprioceptiveOutputData:
    base_body_name: str | None
    state: OutputStateTreeData


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("eef_state", "gripper"),
    meta_fields=("base_body_name",),
)
@dataclass(frozen=True)
class EEFOutputData:
    base_body_name: str | None
    eef_state: OutputStateSingularData
    gripper: OutputStateTreeData


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("content",),
    meta_fields=("type",),
)
@dataclass(frozen=True)
class ControlPolicyOutputStateProprioceptiveOutputSpec:
    type: Literal[ControlPolicyOutputSpecTypes.PROPRIOCEPTIVE_OUTPUT_SPEC]
    content: ProprioceptiveOutputData


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("content",),
    meta_fields=("type",),
)
@dataclass(frozen=True)
class ControlPolicyOutputStateEEFOutputSpec:
    type: Literal[ControlPolicyOutputSpecTypes.EEF_OUTPUT_SPEC]
    content: EEFOutputData


ControlPolicyOutputState = (
    ControlPolicyOutputStateProprioceptiveOutputSpec
    | ControlPolicyOutputStateEEFOutputSpec
)
