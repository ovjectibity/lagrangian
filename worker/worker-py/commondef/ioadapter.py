"""JAX PyTree runtime states corresponding to control-policy input specs."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from commondef.training import ControlPolicyInputSpecTypes


# Only state_item is dynamic. Names, flags, and variant tags are static JAX
# metadata; body-name lists become tuples so the metadata is hashable.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state_item",),
    meta_fields=("is_delta", "included_body_names", "excluded_body_names"),
)
@dataclass(frozen=True)
class IOStateData:
    is_delta: bool
    included_body_names: tuple[str, ...]
    excluded_body_names: tuple[str, ...]
    state_item: jnp.ndarray | np.ndarray


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("states",),
    meta_fields=("robot_base_body_name",),
)
@dataclass(frozen=True)
class OrdinaryIOStateDataConfiguration:
    robot_base_body_name: str | None
    states: list[IOStateData]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("eef_state", "qpos_gripper_state"),
    meta_fields=("robot_base_body_name", "gripper_body_name", "end_effector_body_name"),
)
@dataclass(frozen=True)
class EEFOnlyIOStateDataConfiguration:
    robot_base_body_name: str
    gripper_body_name: str
    end_effector_body_name: str
    eef_state: IOStateData
    qpos_gripper_state: IOStateData


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("content",),
    meta_fields=("type",),
)
@dataclass(frozen=True)
class ControlPolicyInputStateOrdinaryConfiguration:
    type: Literal[ControlPolicyInputSpecTypes.ORDINARY_CONFIGURATION]
    content: OrdinaryIOStateDataConfiguration


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("content",),
    meta_fields=("type",),
)
@dataclass(frozen=True)
class ControlPolicyInputStateEEFOnlyConfiguration:
    type: Literal[ControlPolicyInputSpecTypes.EEF_ONLY_CONFIGURATION]
    content: EEFOnlyIOStateDataConfiguration


ControlPolicyInputState = (
    ControlPolicyInputStateOrdinaryConfiguration
    | ControlPolicyInputStateEEFOnlyConfiguration
)
