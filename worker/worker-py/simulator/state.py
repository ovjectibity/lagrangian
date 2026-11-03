"""Caller-owned physics state and rendering resources for one simulation task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import mujoco
from mujoco import mjx
import numpy as np

from simulator.simulator import Simulator


@dataclass
class SimulatorState:
    """Hold backend state, contact metadata, and optional render resources.

    The workflow owns this container and the lifetime of its render context.
    MJX tasks retain CPU model/data alongside their MJX model/data because
    renderer creation and contact metadata extraction use the CPU model.
    """

    simulator_class: type[Simulator]
    mujoco_data: mujoco.MjData
    mujoco_model: mujoco.MjModel
    contact_sensor_addresses: jax.Array | np.ndarray
    contact_body_pairs: jax.Array | np.ndarray
    mjx_data: mjx.Data | None = None
    mjx_model: mjx.Model | None = None
    render_ctx: Any | None = None
    render_ctx_pytree: Any | None = None
