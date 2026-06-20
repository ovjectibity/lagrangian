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

from . import weights as wt
from .siglip import siglip_forward, _rename_params as _siglip_rename
from .gemma import (
    RMSNorm,
    GemmaLayer,
    _rope_freqs,
    _layer_params,
    causal_mask,
)

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
        frozen, trainable, arch = wt.load(safetensors_path)

        # Convert numpy arrays to jax arrays
        frozen = jax.tree_util.tree_map(jnp.array, frozen)
        trainable = jax.tree_util.tree_map(jnp.array, trainable)

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
            import sentencepiece as spm
            tok_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tokenizer.model",
            )
            if not os.path.exists(tok_path):
                # Download from HuggingFace on first call
                from huggingface_hub import hf_hub_download
                tok_path = hf_hub_download(
                    "google/paligemma-3b-pt-224",
                    "tokenizer.model",
                    local_dir=os.path.dirname(tok_path),
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
        frozen: Params,
        images: Array,
        state: Array,
        task_tokens: Array,
        noised_action: Array,
        t: float | Array,
    ) -> Array:
        """Predict flow-matching velocity for one denoising step.

        Args:
            trainable:     trainable parameter pytree (gradients flow here)
            frozen:        frozen parameter pytree (stop_gradient applied internally)
            images:        [B, N_cam, H, W, 3] float32 in [0, 1]
            state:         [B, state_dim] proprioceptive state (already normalised)
            task_tokens:   [B, T_tok] int32 token ids
            noised_action: [B, max_action_dim] noised action (already normalised, zero-padded)
            t:             scalar timestep in [0, 1]
        Returns:
            velocity: [B, chunk_size, max_action_dim]
        """
        frozen_sg = jax.lax.stop_gradient(frozen)

        # 1. SigLIP vision encoder → [B, N_cam*num_patches, siglip_hidden]
        visual_tokens = siglip_forward(
            frozen_sg["siglip"], images, self._arch
        )

        # 2. Multi-modal projector: [B, S_vis, siglip_hidden] → [B, S_vis, lm_hidden]
        visual_tokens = _dense_apply(
            frozen_sg["mm_proj"]["kernel"],
            frozen_sg["mm_proj"]["bias"],
            visual_tokens,
        )

        # 3. Text token embeddings: [B, T_tok, lm_hidden]
        embed_table = frozen_sg["lm"]["embed_tokens"]["embedding"]  # [vocab, lm_hidden]
        text_tokens = embed_table[task_tokens]                       # [B, T_tok, lm_hidden]

        # 4. Concatenate visual + text tokens → context input [B, S, lm_hidden]
        context_input = jnp.concatenate([visual_tokens, text_tokens], axis=1)

        # 5. LM backbone (frozen, bidirectional)
        context = self._run_lm(frozen_sg["lm"], context_input)      # [B, S, lm_hidden]

        # 6. Time embedding: sinusoidal → MLP → [B, 1, expert_hidden]
        t_emb = self._time_embed(trainable, t)                       # [B, 1, expert_hidden]

        # 7. State projection: [B, state_dim] → [B, 1, expert_hidden]
        state_tok = _dense_apply(
            trainable["state_proj"]["kernel"],
            trainable["state_proj"].get("bias"),
            state[:, None, :],   # [B, 1, state_dim]
        )
        state_tok = state_tok + t_emb   # inject timestep into state token

        # 8. Action projection: [B, chunk_size, max_action_dim] → [B, chunk_size, expert_hidden]
        # Expand noised_action to chunk dimension
        B = images.shape[0]
        na = jnp.broadcast_to(
            noised_action[:, None, :],
            (B, self._chunk_size, self._max_action_dim),
        )
        action_tok = _dense_apply(
            trainable["action_in_proj"]["kernel"],
            trainable["action_in_proj"].get("bias"),
            na,
        )

        # 9. Expert input: [state_tok] ++ action_tokens  [B, 1+chunk, expert_hidden]
        expert_in = jnp.concatenate([state_tok, action_tok], axis=1)

        # 11. Action expert [B, 1+chunk, expert_hidden]
        # context is [B, S, lm_hidden]; k_proj/v_proj inside each GemmaLayer handle the projection
        expert_out = self._run_expert(trainable["expert"], expert_in, context)

        # Drop state token, keep action tokens
        action_out = expert_out[:, 1:, :]   # [B, chunk_size, expert_hidden]

        # 12. Output projection → velocity [B, chunk_size, max_action_dim]
        velocity = _dense_apply(
            trainable["action_out_proj"]["kernel"],
            trainable["action_out_proj"].get("bias"),
            action_out,
        )
        return velocity

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
        # Normalise state — truncate to min(state_dim, norm_dim) to handle robot DOF mismatch
        state_dim = state.shape[0]
        norm_dim = frozen["norm_state"]["mean"].shape[0]
        dim = min(state_dim, norm_dim)
        norm_state = self._normalize(
            state[:dim],
            {"mean": frozen["norm_state"]["mean"][:dim], "std": frozen["norm_state"]["std"][:dim]},
        )
        # Pad state to max_state_dim
        state_pad = _pad_to(norm_state, self._arch.get("state_dim", 32))

        # Tokenise task
        task_tokens = self._encode_task(task)   # [T_tok]

        # Add batch dim
        images_b = images[None]          # [1, N_cam, H, W, 3]
        state_b  = state_pad[None]       # [1, max_state_dim]
        tokens_b = task_tokens[None]     # [1, T_tok]

        # Sample initial noise x_1 ~ N(0, I)  [max_action_dim]
        x = jax.random.normal(rng, (self._max_action_dim,))
        x_b = x[None]   # [1, max_action_dim]

        # Euler ODE: τ from 1 → 0
        dt = 1.0 / self._num_inference_steps
        for step in range(self._num_inference_steps):
            t = 1.0 - step * dt
            vel = self.forward(trainable, frozen, images_b, state_b, tokens_b, x_b, t)
            # vel: [1, chunk_size, max_action_dim] — take first chunk position
            vel0 = vel[0, 0, :]   # [max_action_dim]
            x = x - dt * vel0
            x_b = x[None]

        # Unnormalise and slice to actual action dim
        action = self._unnormalize(x, frozen["unnorm_action"])
        actual_dim = self._cfg.get("output_features", {}).get("action", {}).get("shape", [self._max_action_dim])[0]
        action = action[:actual_dim]

        # Return first n_action_steps by broadcasting (Pi-0 predicts full chunk;
        # for single-step rollout we return n_action_steps copies)
        # Full chunk variant: run forward once and take chunk[:n_action_steps]
        return jnp.broadcast_to(action[None], (self._n_action_steps, actual_dim))

    def infer_chunk(
        self,
        trainable: Params,
        frozen: Params,
        images: Array,
        state: Array,
        task: str,
        rng: Array,
    ) -> Array:
        """Run Euler ODE and return full chunk [chunk_size, actual_action_dim]."""
        state_dim = state.shape[0]
        norm_dim = frozen["norm_state"]["mean"].shape[0]
        dim = min(state_dim, norm_dim)
        norm_state = self._normalize(
            state[:dim],
            {"mean": frozen["norm_state"]["mean"][:dim], "std": frozen["norm_state"]["std"][:dim]},
        )
        state_pad  = _pad_to(norm_state, self._arch.get("state_dim", 32))
        task_tokens = self._encode_task(task)

        images_b = images[None]
        state_b  = state_pad[None]
        tokens_b = task_tokens[None]

        x = jax.random.normal(rng, (self._max_action_dim,))
        x_b = x[None]

        dt = 1.0 / self._num_inference_steps
        chunk_vel = None
        for step in range(self._num_inference_steps):
            t = 1.0 - step * dt
            vel = self.forward(trainable, frozen, images_b, state_b, tokens_b, x_b, t)
            chunk_vel = vel[0]   # [chunk_size, max_action_dim] — last step velocity
            vel0 = vel[0, 0, :]
            x = x - dt * vel0
            x_b = x[None]

        # Use the final velocity chunk as the action prediction
        # (standard Pi-0 uses the denoised x; we use the denoised per-step actions)
        actual_dim = self._cfg.get("output_features", {}).get("action", {}).get("shape", [self._max_action_dim])[0]
        chunk = chunk_vel[:self._n_action_steps, :actual_dim]
        return chunk

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

    def _run_lm(self, lm_params: Params, x: Array) -> Array:
        arch = self._arch
        for i, layer_p in enumerate(lm_params["layers"]):
            lp = _layer_params(layer_p)
            layer = GemmaLayer(
                num_heads=arch["lm_num_heads"],
                num_kv_heads=arch["lm_num_kv_heads"],
                head_dim=arch["lm_head_dim"],
                intermediate_dim=arch["lm_intermediate"],
                hidden_dim=arch["lm_hidden"],
                name=f"lm_layer_{i}",
            )
            x = layer.apply({"params": lp}, x, self._lm_freqs, mask=None)
        norm = RMSNorm(name="lm_norm")
        x = norm.apply({"params": lm_params["norm"]}, x)
        return x

    def _run_expert(self, expert_params: Params, x: Array, context: Array) -> Array:
        arch = self._arch
        A = x.shape[1]
        mask = causal_mask(A)
        for i, layer_p in enumerate(expert_params["layers"]):
            lp = _layer_params(layer_p)
            layer = GemmaLayer(
                num_heads=arch["expert_num_heads"],
                num_kv_heads=arch["expert_num_kv_heads"],
                head_dim=arch["expert_head_dim"],
                intermediate_dim=arch["expert_intermediate"],
                hidden_dim=arch["expert_hidden"],
                name=f"exp_layer_{i}",
            )
            x = layer.apply({"params": lp}, x, self._exp_freqs, mask=mask, cross_kv=context)
        norm = RMSNorm(name="exp_norm")
        x = norm.apply({"params": expert_params["norm"]}, x)
        return x

    def _time_embed(self, trainable: Params, t: float | Array) -> Array:
        """Sinusoidal time embedding → MLP → [B, 1, expert_hidden]."""
        time_dim = self._arch.get("time_embed_dim", 2048)
        half = time_dim // 2
        freqs = jnp.exp(
            -jnp.arange(half, dtype=jnp.float32) * (jnp.log(10000.0) / (half - 1))
        )
        t_arr = jnp.array(t, dtype=jnp.float32)
        args = t_arr * freqs
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)  # [time_dim]
        emb = emb[None, None, :]   # [1, 1, time_dim]

        mlp = trainable["action_time_mlp"]
        emb = _dense_apply(mlp["in"]["kernel"], mlp["in"].get("bias"), emb)
        emb = jax.nn.silu(emb)
        emb = _dense_apply(mlp["out"]["kernel"], mlp["out"].get("bias"), emb)
        return emb   # [1, 1, expert_hidden]

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
