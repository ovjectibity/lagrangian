"""Pi0Jax — pure JAX/Flax Pi-0 VLA model.

Replaces the lerobot/PyTorch PI0Policy with a framework-agnostic implementation
that loads directly from model.safetensors using the safetensors library.

Frozen parameters (SigLIP, LM backbone, norms) are kept separate from trainable
parameters (action expert, projection layers) so PPO gradients only flow through
the trainable portion.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece as spm
from huggingface_hub import hf_hub_download

from . import weights as wt
from .siglip import siglip_forward
from .gemma import (
    RMSNorm,
    GemmaLayer,
    _apply_rope,
    _rope_freqs,
)
import logging
_logger = logging.getLogger(__name__)

Params = dict[str, Any]
Array = jax.Array

_MAX_SEQ_LEN = 2048   # RoPE cache length


# ---------------------------------------------------------------------------
# Pi0Jax
# ---------------------------------------------------------------------------

class Pi0Jax:
    """Pure JAX Pi-0 model.

    Orchestrates frozen and trainable parameter sets. The `forward` method is
    fully JAX-traceable and differentiable w.r.t. `trainable_params`.
    """

    def __init__(
        self,
        frozen_params: Params,
        trainable_params: Params,
        arch: dict,
        config: dict,
        n_action_steps: int = 10,
    ) -> None:
        self._frozen = frozen_params
        self._trainable = trainable_params
        self._arch = arch
        self._cfg = config
        self._n_action_steps = n_action_steps

        self._chunk_size: int = config.get("chunk_size", 50)
        self._max_action_dim: int = arch.get("action_dim", 32)
        self._num_inference_steps: int = config.get("num_inference_steps", 10)
        self._image_resolution: tuple[int, int] = tuple(config.get("image_resolution", [224, 224]))
        self._tok_max_len: int = config.get("tokenizer_max_length", 48)

        # Pre-compute RoPE frequencies (shared by LM and expert)
        lm_hd = arch.get("lm_head_dim", 256)
        self._lm_freqs: Array = jnp.array(_rope_freqs(lm_hd, _MAX_SEQ_LEN))
        exp_hd = arch.get("expert_head_dim", 256)
        self._exp_freqs: Array = jnp.array(_rope_freqs(exp_hd, _MAX_SEQ_LEN))

        self._tokenizer = None   # lazy-loaded on first text encode

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        n_action_steps: int = 10,
    ) -> "Pi0Jax":
        """Load from a Pi-0 checkpoint directory.

        Expects `model.safetensors` and `config.json` in `path`.
        """
        cfg_path = os.path.join(path, "config.json")
        with open(cfg_path) as f:
            config = json.load(f)

        safetensors_path = os.path.join(path, "model.safetensors")
        frozen, trainable, arch = wt.load_lazy(safetensors_path)

        # safetensors.numpy upcasts bfloat16 checkpoint weights to float32 (numpy has no
        # native bfloat16 type), doubling GPU memory from ~7GB to ~14GB. Explicitly cast
        # back to bfloat16 so the weights fit comfortably on a 16GB T4.
        # frozen    = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.bfloat16), frozen)
        # trainable = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.bfloat16), trainable)

        _log_gpu_memory("after weight load")
        return cls(frozen, trainable, arch, config, n_action_steps=n_action_steps)

    # ------------------------------------------------------------------
    # PPOTrainedModel interface helpers
    # ------------------------------------------------------------------

    def get_policy_store(self) -> Params:
        return self._trainable

    def get_frozen_store(self) -> Params:
        return self._frozen

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _encode_task(self, task: str) -> Array:
        """Tokenise a task string → int32 token ids [tok_max_len]."""
        if self._tokenizer is None:
            checkpoint_dir = os.environ.get("PI0_CHECKPOINT_DIR", "/opt/lagrangian/pi0")
            tok_path = os.path.join(checkpoint_dir, "tokenizer.model")
            if not os.path.exists(tok_path):
                tok_path = hf_hub_download(
                    "google/paligemma-3b-pt-224",
                    "tokenizer.model",
                    local_dir=checkpoint_dir,
                    token=os.environ.get("HF_TOKEN"),
                )
            sp = spm.SentencePieceProcessor()
            sp.Load(tok_path)
            self._tokenizer = sp

        ids = self._tokenizer.EncodeAsIds(task)
        # Pad / truncate to tok_max_len
        if len(ids) >= self._tok_max_len:
            ids = ids[: self._tok_max_len]
        else:
            ids = ids + [0] * (self._tok_max_len - len(ids))
        return jnp.array(ids, dtype=jnp.int32)   # [tok_max_len]

    # ------------------------------------------------------------------
    # Forward pass (differentiable w.r.t. trainable)
    # ------------------------------------------------------------------
    
    def forward(
        self,
        trainable: Params,
        lm_kv: tuple[Array, Array],
        state_b: Array,
        noised_action: Array,
        t: float | Array,
    ) -> Array:
        """Predict flow-matching velocity for one denoising step.

        Args:
            trainable:     trainable parameter pytree (gradients flow here)
            lm_kv:         (Ks, Vs) stacked LM KV cache [18, B, 1, S, 256] each,
                           pre-computed by encode_prefix()
            state_b:       [B, state_dim] proprioceptive state (already normalised)
            noised_action: [B, chunk_size, max_action_dim] noised action chunk
            t:             scalar timestep in [0, 1]
        Returns:
            velocity: [B, chunk_size, max_action_dim]
        """
        state_tok = _dense_apply(
            trainable["state_proj"]["kernel"],
            trainable["state_proj"].get("bias"),
            state_b[:, None, :],
        )  # [B, 1, expert_hidden]

        action_tok = self._action_embed(trainable, noised_action, t)  # [B, chunk, expert_hidden]

        expert_in  = jnp.concatenate([state_tok, action_tok], axis=1)
        expert_out = self._run_expert_with_cache(trainable["expert"], expert_in, lm_kv)

        action_out = expert_out[:, 1:, :]  # [B, chunk, expert_hidden]
        return _dense_apply(
            trainable["action_out_proj"]["kernel"],
            trainable["action_out_proj"].get("bias"),
            action_out,
        )

    # ------------------------------------------------------------------
    # Inference (Euler ODE)
    # ------------------------------------------------------------------

    def infer(
        self,
        trainable: Params,
        frozen: Params,
        images: Array,
        state: Array,
        task: str,
        rng: Array,
    ) -> Array:
        """Run Euler ODE denoising and return the first n_action_steps actions.

        Args:
            images:  [N_cam, H, W, 3] (single sample, no batch dim)
            state:   [state_dim]
            task:    task description string
            rng:     JAX PRNG key
        Returns:
            actions: [n_action_steps, actual_action_dim]
        """
        state_dim = state.shape[0]
        norm_dim = frozen["norm_state"]["mean"].shape[0]
        dim = min(state_dim, norm_dim)
        norm_state = self._normalize(
            state[:dim],
            {"mean": frozen["norm_state"]["mean"][:dim], "std": frozen["norm_state"]["std"][:dim]},
        )
        state_pad   = _pad_to(norm_state, self._arch.get("state_dim", 32))
        task_tokens = self._encode_task(task)

        images_b = images[None]
        state_b  = state_pad[None]
        tokens_b = task_tokens[None]

        lm_kv = self.encode_prefix(frozen, images_b, tokens_b)

        x_b = jax.random.normal(rng, (1, self._chunk_size, self._max_action_dim))

        dt = 1.0 / self._num_inference_steps
        ts = 1.0 - jnp.arange(self._num_inference_steps, dtype=jnp.float32) * dt

        def denoise_step(x_b, t):
            vel = self.forward(trainable, lm_kv, state_b, x_b, t)
            return x_b - dt * vel, None

        x_final, _ = jax.lax.scan(denoise_step, x_b, ts)
        # x_final: [1, chunk_size, max_action_dim]

        actual_dim = frozen["unnorm_action"]["mean"].shape[0]
        chunk = x_final[0, :self._n_action_steps, :actual_dim]   # [n_action_steps, actual_dim]
        return self._unnormalize(chunk, frozen["unnorm_action"])

    # ------------------------------------------------------------------
    # Pure policy function (for PPO trainer)
    # ------------------------------------------------------------------

    def get_pure_forward(self) -> Callable:
        """Return a pure function (trainable, frozen, images, state, tokens, action, t) → vel.

        JIT-able and differentiable w.r.t. trainable (argnums=0).
        """
        return self.forward

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rms_norm_apply(norm_params: Params, x: Array) -> Array:
        scale = norm_params["scale"]
        ms = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(ms + 1e-6) * (1.0 + scale)

    def _run_lm_and_cache(
        self, lm_params: Params, x: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        arch         = self._arch
        num_heads    = arch["lm_num_heads"]
        num_kv_heads = arch["lm_num_kv_heads"]
        head_dim     = arch["lm_head_dim"]
        q_dim        = num_heads * head_dim
        freqs        = self._lm_freqs

        def lm_layer(x, layer_p):
            residual = x
            normed   = self._rms_norm_apply(layer_p["input_ln"], x)
            B, S, _  = normed.shape

            Q = _dense_apply(layer_p["self_attn"]["q_proj"]["kernel"], None, normed)
            K = _dense_apply(layer_p["self_attn"]["k_proj"]["kernel"], None, normed)
            V = _dense_apply(layer_p["self_attn"]["v_proj"]["kernel"], None, normed)

            Q = Q.reshape(B, S, num_heads,    head_dim).transpose(0, 2, 1, 3)
            K = K.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            V = V.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            Q       = _apply_rope(Q, freqs)
            K_roped = _apply_rope(K, freqs)

            K_full = jnp.repeat(K_roped, num_heads // num_kv_heads, axis=1)
            V_full = jnp.repeat(V,       num_heads // num_kv_heads, axis=1)
            scale  = head_dim ** -0.5
            attn   = jnp.einsum("bhtd,bhsd->bhts", Q, K_full) * scale
            attn   = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
            out    = jnp.einsum("bhts,bhsd->bhtd", attn, V_full)
            out    = out.transpose(0, 2, 1, 3).reshape(B, S, q_dim)
            out    = _dense_apply(layer_p["self_attn"]["o_proj"]["kernel"], None, out)
            x      = residual + out

            residual = x
            normed2  = self._rms_norm_apply(layer_p["post_attn_ln"], x)
            gate = _dense_apply(layer_p["mlp"]["gate_proj"]["kernel"], None, normed2)
            up   = _dense_apply(layer_p["mlp"]["up_proj"]["kernel"],   None, normed2)
            x    = residual + _dense_apply(
                layer_p["mlp"]["down_proj"]["kernel"], None, jax.nn.silu(gate) * up
            )
            return x, (K_roped, V)

        final_x, (Ks, Vs) = jax.lax.scan(lm_layer, x, lm_params["layers"])
        final_x = self._rms_norm_apply(lm_params["norm"], final_x)
        return final_x, (Ks, Vs)

    def encode_prefix(
        self, frozen: Params, images_b: Array, tokens_b: Array
    ) -> tuple[Array, Array]:
        frozen_sg = jax.lax.stop_gradient(frozen)
        _log_gpu_memory("encode_prefix: start")
        vis  = siglip_forward(frozen_sg["siglip"], images_b, self._arch)
        vis  = _dense_apply(frozen_sg["mm_proj"]["kernel"], frozen_sg["mm_proj"]["bias"], vis)
        _log_gpu_memory("encode_prefix: after siglip")
        txt  = frozen_sg["lm"]["embed_tokens"]["embedding"][tokens_b]
        lm_x = jnp.concatenate([vis, txt], axis=1)
        _, (Ks, Vs) = self._run_lm_and_cache(frozen_sg["lm"], lm_x)
        _log_gpu_memory("encode_prefix: after lm")
        return Ks, Vs

    def _run_expert_with_cache(
        self, expert_params: Params, expert_in: Array, lm_kv: tuple[Array, Array]
    ) -> Array:
        Ks, Vs = lm_kv
        S = Ks.shape[3]
        A = expert_in.shape[1]

        mask = jnp.zeros((A, S + A), dtype=jnp.float32)
        mask = mask.at[0, S + 1:].set(-1e9)
        mask = mask[None, None, :, :]

        arch = self._arch

        def expert_layer(x, inputs):
            layer_p, K_prefix, V_prefix = inputs
            layer = GemmaLayer(
                num_heads=arch["expert_num_heads"],
                num_kv_heads=arch["expert_num_kv_heads"],
                head_dim=arch["expert_head_dim"],
                intermediate_dim=arch["expert_intermediate"],
                hidden_dim=arch["expert_hidden"],
                name="exp_layer",
            )
            x = layer.apply(
                {"params": layer_p}, x, self._exp_freqs,
                mask=mask, prefix_kv=(K_prefix, V_prefix),
            )
            return x, None

        final_x, _ = jax.lax.scan(expert_layer, expert_in, (expert_params["layers"], Ks, Vs))
        norm = RMSNorm(name="exp_norm")
        return norm.apply({"params": expert_params["norm"]}, final_x)

    def _action_embed(self, trainable: Params, na: Array, t: float | Array) -> Array:
        """W_3 · swish(W_2 · concat(W_1·a, φ(τ))) per paper §3."""
        w    = self._arch["expert_hidden"]
        half = w // 2
        freqs = jnp.exp(
            -jnp.arange(half, dtype=jnp.float32) * (jnp.log(10000.0) / (half - 1))
        )
        t_arr   = jnp.array(t, dtype=jnp.float32)
        phi_tau = jnp.concatenate([jnp.cos(t_arr * freqs), jnp.sin(t_arr * freqs)])  # [w]

        W1_a = _dense_apply(
            trainable["action_in_proj"]["kernel"],
            trainable["action_in_proj"].get("bias"),
            na,
        )  # [B, chunk, w]
        phi_b = jnp.broadcast_to(phi_tau[None, None, :], W1_a.shape)
        h = jnp.concatenate([W1_a, phi_b], axis=-1)  # [B, chunk, 2w]

        mlp = trainable["action_time_mlp"]
        h = jax.nn.silu(_dense_apply(mlp["in"]["kernel"],  mlp["in"].get("bias"),  h))
        return _dense_apply(         mlp["out"]["kernel"], mlp["out"].get("bias"), h)

    @staticmethod
    def _normalize(x: Array, norm: Params) -> Array:
        return (x - norm["mean"]) / (norm["std"] + 1e-8)

    @staticmethod
    def _unnormalize(x: Array, norm: Params) -> Array:
        return x * norm["std"] + norm["mean"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _dense_apply(kernel: Array, bias: Array | None, x: Array) -> Array:
    """Matrix multiply with optional bias: x @ kernel [+ bias]."""
    out = jnp.einsum("...i,ij->...j", x, kernel)
    if bias is not None:
        out = out + bias
    return out


def _pad_to(x: Array, target_dim: int) -> Array:
    pad = target_dim - x.shape[-1]
    if pad <= 0:
        return x[..., :target_dim]
    return jnp.pad(x, [(0, pad)])


def _cpu_ram_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6   # kB → GB
    except Exception:
        pass
    return 0.0


def _log_gpu_memory(label: str) -> None:
    try:
        stats = jax.devices()[0].memory_stats()
        used_gb  = stats["bytes_in_use"] / 1e9
        limit_gb = stats["bytes_limit"] / 1e9
        cpu_gb   = _cpu_ram_gb()
        jax.debug.print("JAX DEBUG: [{label}]  GPU {used_gb} / {limit_gb} GB   CPU RSS {cpu_gb} GB",label,used_gb,limit_gb,cpu_gb)
        _logger.info("[%s]  GPU %.2f / %.2f GB   CPU RSS %.2f GB", label, used_gb, limit_gb, cpu_gb)
    except Exception:
        pass
