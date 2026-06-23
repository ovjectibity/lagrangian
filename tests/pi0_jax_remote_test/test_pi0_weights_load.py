#!/usr/bin/env python3
"""Test: load Pi-0 checkpoint weights and verify stacked layer structure.

Runs inside the worker container. Checks:
  - load() completes without OOM
  - layers are stacked pytrees (leading [num_layers, ...] dim), not Python lists
  - GPU memory before and after load
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import jax
import models.pi0.pi0_jax.weights as wt

CHECKPOINT_DIR = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")
SAFETENSORS_PATH = os.path.join(CHECKPOINT_DIR, "model.safetensors")


def _gpu_gb() -> tuple[float, float]:
    try:
        s = jax.devices()[0].memory_stats()
        return s["bytes_in_use"] / 1e9, s["bytes_limit"] / 1e9
    except Exception:
        return 0.0, 0.0


def main():
    used, limit = _gpu_gb()
    logger.info("GPU before load: %.2f / %.2f GB", used, limit)

    logger.info("Loading from %s ...", SAFETENSORS_PATH)
    frozen, trainable, arch = wt.load(SAFETENSORS_PATH)

    used, limit = _gpu_gb()
    logger.info("GPU after load:  %.2f / %.2f GB", used, limit)
    logger.info("arch: %s", arch)

    # lm layers
    lm_layers = frozen["lm"]["layers"]
    assert isinstance(lm_layers, dict), f"lm layers should be a stacked dict, got {type(lm_layers)}"
    q_shape = lm_layers["self_attn"]["q_proj"]["kernel"].shape
    logger.info("lm  q_proj kernel shape: %s  (expect [%d, ...])", q_shape, arch["lm_num_layers"])
    assert q_shape[0] == arch["lm_num_layers"]

    # expert layers
    exp_layers = trainable["expert"]["layers"]
    assert isinstance(exp_layers, dict), f"expert layers should be a stacked dict, got {type(exp_layers)}"
    eq_shape = exp_layers["self_attn"]["q_proj"]["kernel"].shape
    logger.info("exp q_proj kernel shape: %s  (expect [%d, ...])", eq_shape, arch["expert_num_layers"])
    assert eq_shape[0] == arch["expert_num_layers"]

    # siglip layers
    sig_layers = frozen["siglip"]["layers"]
    assert isinstance(sig_layers, dict), f"siglip layers should be a stacked dict, got {type(sig_layers)}"
    sq_shape = sig_layers["self_attn"]["q_proj"]["kernel"].shape
    logger.info("sig q_proj kernel shape: %s  (expect [%d, ...])", sq_shape, arch["siglip_num_layers"])
    assert sq_shape[0] == arch["siglip_num_layers"]

    logger.info("PASS")


if __name__ == "__main__":
    main()
