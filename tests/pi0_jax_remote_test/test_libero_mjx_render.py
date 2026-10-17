#!/usr/bin/env python3
"""Render one stepped LIBERO task with the worker's MJX-Warp simulator code.

Run through run_remote_test.sh; PNGs are saved in its artifact directory.
LIBERO_DATASET and LIBERO_TASK_INDEX select the task (libero_spatial, 0 by default).
"""

import os
from pathlib import Path

import jax
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image

from datasets.libero.libero import get_task_mjcfs
from simulator.simulator import MjxSimulator


def main() -> None:
    dataset = os.environ.get("LIBERO_DATASET", "libero_spatial")
    task_index = int(os.environ.get("LIBERO_TASK_INDEX", "0"))
    (mjcf,) = get_task_mjcfs([task_index], dataset)
    data, model, cpu_data, cpu_model = MjxSimulator.init(
        mjcf=mjcf,
        nworld=1,
        naconmax=50000,
        naccdmax=50000,
    )

    print(f"task={dataset}[{task_index}] ncam={cpu_model.ncam}", flush=True)
    print(
        f"model.ngeom={cpu_model.ngeom} model.nbody={cpu_model.nbody} "
        f"model.nconmax={cpu_model.nconmax} "
        f"initial_cpu_contacts={cpu_data.ncon}",
        flush=True,
    )
    print(f"model.cam_resolution={np.asarray(cpu_model.cam_resolution).tolist()}", flush=True)
    if cpu_model.ncam == 0:
        raise ValueError("The LIBERO task has no cameras to render")

    data = jax.vmap(mjx.step, in_axes=(None, 0))(model, data)
    jax.block_until_ready(data)
    print("completed one MJX-Warp step", flush=True)

    for label, resolution in (("default", None), ("224x224", (224, 224))):
        if resolution is None:
            context = MjxSimulator.get_render_context(cpu_model, nworld=1)
        else:
            context = mjx.create_render_context(
                cpu_model,
                nworld=1,
                cam_res=resolution,
                use_textures=len(cpu_model.mesh_texcoord) > 0,
            )
        try:
            frames, data = MjxSimulator.get_model_render(
                data, model, context.pytree()
            )
            frames = np.asarray(jax.block_until_ready(frames))
            height, width = frames.shape[2:4]
            patches = (height // 14) * (width // 14)
            print(
                f"{label}: frames.shape={frames.shape}, "
                f"SigLIP patches/camera={patches} (checkpoint expects 256)",
                flush=True,
            )
            for camera_id in range(cpu_model.ncam):
                camera_name = mujoco.mj_id2name(
                    cpu_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
                )
                output_path = Path.cwd() / (
                    f"{dataset}_task{task_index}_{label}_camera{camera_id}.png"
                )
                pixels = np.clip(frames[0, camera_id] * 255, 0, 255).astype(
                    np.uint8
                )
                Image.fromarray(pixels, mode="RGB").save(output_path)
                print(
                    f"  camera[{camera_id}]={camera_name!r} saved {output_path}",
                    flush=True,
                )
        finally:
            close = getattr(context, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()
