#!/usr/bin/env python3
"""Test: load Pi-0 checkpoint and run a full inference pass with dummy inputs."""
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import jax
import jax.numpy as jnp
from models.pi0.pi0_jax import Pi0Jax
import random

CHECKPOINT_DIR = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")

# Set to True to skip the full denoising scan and only test the LM backbone.
TEST_LM_ONLY = os.environ.get("TEST_LM_ONLY", "1") == "1"


def _cpu_ram_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6   # kB → GB
    except Exception:
        pass
    return 0.0


def _gpu_gb() -> tuple[float, float]:
    try:
        s = jax.devices()[0].memory_stats()
        return s["bytes_in_use"] / 1e9, s["bytes_limit"] / 1e9
    except Exception:
        return 0.0, 0.0


def _log_stats(label: str) -> None:
    used_gb, limit_gb = _gpu_gb()
    cpu_gb = _cpu_ram_gb()
    jax.debug.print("JAX DEBUG: [{label}]  GPU {used_gb} / {limit_gb} GB   CPU RSS {cpu_gb} GB",label=label,used_gb=used_gb,limit_gb=limit_gb,cpu_gb=cpu_gb)
    logger.info("[%s]  GPU %.2f / %.2f GB   CPU RSS %.2f GB", label, used_gb, limit_gb, cpu_gb)


def main():
    _log_stats("before load")

    logger.info("Loading checkpoint from %s ...", CHECKPOINT_DIR)
    pi0 = Pi0Jax.from_checkpoint(CHECKPOINT_DIR)
    _log_stats("after load")

    H, W = pi0._image_resolution
    images = jnp.zeros((1, H, W, 3), dtype=jnp.float32)   # [N_cam=1, H, W, 3]
    task   = "pick up the block"
    # state  = jnp.zeros(pi0._arch["state_dim"], dtype=jnp.float32)
    # rng    = jax.random.PRNGKey(0)

    # ------------------------------------------------------------------ #
    # Part 1: test just the LM backbone (encode_prefix)                   #
    # ------------------------------------------------------------------ #
    if TEST_LM_ONLY:
        pi0._frozen["lm"]["layers"] = jax.tree_util.tree_map(
            lambda a: a[:1], pi0._frozen["lm"]["layers"]
        )
        logger.info("Truncated LM to 1 layer for memory hypothesis test")
        _log_stats("after lm layer truncation")

    # Verify pool can satisfy a 144 KiB allocation before we invoke any scan.
    # probe = jnp.zeros((288, 256), dtype=jnp.bfloat16)   # exactly 144 KiB
    # probe.block_until_ready()
    # _log_stats("after 144 KiB probe alloc")
    # del probe

    logger.info("--- encode_prefix only ---")
    images_b = images[None]
    tokens_b = pi0._encode_task(task)[None]

    jit_encode_prefix = jax.jit(pi0.encode_prefix)
    compiled_step = jit_encode_prefix.lower(pi0._frozen, images_b, tokens_b).compile()
    compiled_stats = compiled_step.memory_analysis()
    print(compiled_stats)

    # Ks, Vs = jit_encode_prefix(pi0._frozen, images_b, tokens_b)
    # # Ks, Vs = pi0.encode_prefix(pi0._frozen, images_b, tokens_b)
    # Ks = jax.block_until_ready(Ks)
    # Vs = jax.block_until_ready(Vs)
    # _log_stats("after encode_prefix")
    # logger.info("Ks shape: %s  Vs shape: %s", Ks.shape, Vs.shape)

    # if TEST_LM_ONLY:
    #     logger.info("TEST_LM_ONLY=1 — stopping here")
    #     logger.info("PASS (LM only)")
    #     return
    
    # jax.profiler.save_device_memory_profile(f"memory_abc.prof")

    # ------------------------------------------------------------------ #
    # Part 2: full denoising scan                                         #
    # JIT-compile the scan body (task closed over as constant)            #
    # ------------------------------------------------------------------ #
    # logger.info("--- full denoising (JIT) ---")

    # jit_infer = jax.jit(
    #     lambda trainable, frozen, images, state, rng:
    #         pi0.infer(trainable, frozen, images, state, task, rng)
    # )

    # logger.info("Running infer — images %s, state %s, task: %r", images.shape, state.shape, task)
    # actions = jit_infer(pi0._trainable, pi0._frozen, images, state, rng)
    # actions = jax.block_until_ready(actions)

    # _log_stats("after infer")
    # logger.info("actions shape: %s", actions.shape)
    # logger.info("actions:\n%s", actions)

    # assert actions.ndim == 2, f"expected 2D actions, got shape {actions.shape}"
    # assert actions.shape[0] == pi0._n_action_steps, \
    #     f"expected {pi0._n_action_steps} action steps, got {actions.shape[0]}"

    # logger.info("PASS")


if __name__ == "__main__":
    main()
