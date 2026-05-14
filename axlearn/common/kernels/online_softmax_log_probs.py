# Copyright © 2026 Apple Inc.
"""Pallas kernel for fused streaming log-probs on TPU.

Computes per-position log-normalizer and top-k logits without materializing the full
``[batch, seq_len, vocab_size]`` logits tensor in HBM.  The matmul, online log-sum-exp,
and running top-k merge are fused in VMEM so that only a ``[tile_s, tile_v]`` tile of
logits exists at any time (~1 MB), eliminating the multi-ten-GB HBM allocation required by
the full matmul approach.

Batch and sequence dimensions are folded into a single "tokens" dimension so the kernel
uses a simple 2D grid with ``(tile_s, D)`` and ``(tile_v, D)`` BlockSpecs.

Grid: ``(total_tokens // tile_s, V_padded // tile_v)`` with dimension semantics
``("parallel", "arbitrary")``.  Token tiles run independently; vocabulary chunks
stream through each program sequentially.

This module is sharding-agnostic — all sharding orchestration (shard_map,
cross-device reductions) is handled by the caller.
"""

import functools
from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from axlearn.common.utils import Tensor, get_tpu_dot_precision

NEG_INF = -1e15


# ---------------------------------------------------------------------------
# Kernel data structures
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass
class KernelOutputs:
    """HBM output refs written on the last vocab chunk."""

    max_ref: Tensor  # [tile_s, 1]
    sum_exp_ref: Tensor  # [tile_s, 1]
    topk_vals_ref: Tensor  # [tile_s, top_k]
    topk_idx_ref: Tensor  # [tile_s, top_k]


@dataclass(frozen=True)
class KernelConfig:
    """Compile-time constants for the kernel."""

    vocab_size: int
    tile_v: int
    top_k: int
    dot_dtype: jnp.dtype = jnp.float32


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def _online_softmax_log_probs_kernel(
    x_ref,  # [tile_s, D]
    weight_ref,  # [tile_v, D]
    outputs: KernelOutputs,
    # VMEM scratch (PrefetchScalarGridSpec requires a flat sequence for scratch).
    running_max,  # [tile_s, 1]  f32
    running_sum_exp,  # [tile_s, 1]  f32
    topk_vals_scratch,  # [tile_s, top_k]  f32
    topk_idx_scratch,  # [tile_s, top_k]  i32
    *,
    cfg: KernelConfig,
):
    v_idx = pl.program_id(1)

    # ---- init on first vocab chunk ----
    @pl.when(v_idx == 0)
    def _init():
        running_max[...] = jnp.full_like(running_max, NEG_INF)
        running_sum_exp[...] = jnp.zeros_like(running_sum_exp)
        if cfg.top_k > 0:
            topk_vals_scratch[...] = jnp.full_like(topk_vals_scratch, NEG_INF)
            topk_idx_scratch[...] = jnp.full_like(topk_idx_scratch, -1)

    # ---- matmul in VMEM via MXU ----
    x_tile = x_ref[...]  # [tile_s, D]
    w_tile = weight_ref[...]  # [tile_v, D]
    chunk_logits = pl.dot(
        x_tile.astype(cfg.dot_dtype),
        w_tile.astype(cfg.dot_dtype).T,
        precision=get_tpu_dot_precision(cfg.dot_dtype),
    )

    # ---- mask invalid vocab entries ----
    chunk_start = v_idx * cfg.tile_v
    col_ids = chunk_start + lax.broadcasted_iota(jnp.int32, chunk_logits.shape, 1)
    chunk_logits = jnp.where(col_ids < cfg.vocab_size, chunk_logits, NEG_INF)

    # ---- online log-sum-exp ----
    m_prev = running_max[...]  # [tile_s, 1]
    l_prev = running_sum_exp[...]  # [tile_s, 1]

    m_curr = chunk_logits.max(axis=-1, keepdims=True)  # [tile_s, 1]
    m_next = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp(m_prev - m_next)
    l_next = correction * l_prev + jnp.sum(jnp.exp(chunk_logits - m_next), axis=-1, keepdims=True)
    running_max[...] = m_next
    running_sum_exp[...] = l_next

    # ---- running top-k (only when top_k > 0) ----
    if cfg.top_k > 0:
        # Extract chunk top-k via K passes of argmax-and-mask.
        # For small K (<=32), this is faster than sort-based extraction because
        # max/argmax are single-instruction reductions on TPU vector units.
        # Use jnp.where with column masks instead of .at[].set() (scatter not
        # supported in Pallas TPU lowering).
        chunk_remaining = chunk_logits  # [tile_s, tile_v]
        tile_s_val = x_tile.shape[0]
        chunk_tk_vals = jnp.full((tile_s_val, cfg.top_k), NEG_INF, dtype=jnp.float32)
        chunk_tk_idx = jnp.full((tile_s_val, cfg.top_k), -1, dtype=jnp.int32)

        for k_i in range(cfg.top_k):
            best_val = chunk_remaining.max(axis=-1)
            best_local = chunk_remaining.argmax(axis=-1)
            col_mask = lax.broadcasted_iota(jnp.int32, chunk_tk_vals.shape, 1) == k_i
            chunk_tk_vals = jnp.where(col_mask, best_val[:, jnp.newaxis], chunk_tk_vals)
            chunk_tk_idx = jnp.where(
                col_mask, (best_local + chunk_start)[:, jnp.newaxis], chunk_tk_idx
            )
            sel_mask = (
                lax.broadcasted_iota(jnp.int32, chunk_remaining.shape, 1)
                == best_local[:, jnp.newaxis]
            )
            chunk_remaining = jnp.where(sel_mask, NEG_INF, chunk_remaining)

        # Merge with running top-k: concat [tile_s, 2K] -> pick top K.
        prev_vals = topk_vals_scratch[...]
        prev_idx = topk_idx_scratch[...]

        merged_vals = jnp.concatenate([prev_vals, chunk_tk_vals], axis=-1)
        merged_idx = jnp.concatenate([prev_idx, chunk_tk_idx], axis=-1)

        new_vals = jnp.full((tile_s_val, cfg.top_k), NEG_INF, dtype=jnp.float32)
        new_idx = jnp.full((tile_s_val, cfg.top_k), -1, dtype=jnp.int32)

        for k_i in range(cfg.top_k):
            best_val = merged_vals.max(axis=-1)
            best_pos = merged_vals.argmax(axis=-1)
            pos_mask = (
                lax.broadcasted_iota(jnp.int32, merged_idx.shape, 1) == best_pos[:, jnp.newaxis]
            )
            best_global_idx = jnp.sum(
                jnp.where(pos_mask, merged_idx, jnp.zeros_like(merged_idx)), axis=-1
            )
            col_mask = lax.broadcasted_iota(jnp.int32, new_vals.shape, 1) == k_i
            new_vals = jnp.where(col_mask, best_val[:, jnp.newaxis], new_vals)
            new_idx = jnp.where(col_mask, best_global_idx[:, jnp.newaxis], new_idx)
            sel_mask = (
                lax.broadcasted_iota(jnp.int32, merged_vals.shape, 1) == best_pos[:, jnp.newaxis]
            )
            merged_vals = jnp.where(sel_mask, NEG_INF, merged_vals)

        topk_vals_scratch[...] = new_vals
        topk_idx_scratch[...] = new_idx

    # ---- write outputs only on the last vocab chunk ----
    @pl.when(v_idx == pl.num_programs(1) - 1)
    def _finalize():
        outputs.max_ref[...] = running_max[...]
        outputs.sum_exp_ref[...] = running_sum_exp[...]
        if cfg.top_k > 0:
            outputs.topk_vals_ref[...] = topk_vals_scratch[...]
            outputs.topk_idx_ref[...] = topk_idx_scratch[...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to_tile_boundaries(
    x: Tensor, weight: Tensor, *, tile_s: int, tile_v: int
) -> tuple[Tensor, Tensor, int, int]:
    """Pads x and weight to tile boundaries.

    Returns:
        (padded_x, padded_weight, padded_S, num_vocab_chunks)
    """
    S = x.shape[1]  # pylint: disable=invalid-name
    V = weight.shape[0]  # pylint: disable=invalid-name
    num_vocab_chunks = (V + tile_v - 1) // tile_v
    padded_V = num_vocab_chunks * tile_v  # pylint: disable=invalid-name
    padded_S = ((S + tile_s - 1) // tile_s) * tile_s  # pylint: disable=invalid-name

    if padded_V > V:
        weight = jnp.pad(weight, ((0, padded_V - V), (0, 0)))
    if padded_S > S:
        x = jnp.pad(x, ((0, 0), (0, padded_S - S), (0, 0)))

    return x, weight, padded_S, num_vocab_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def online_softmax_log_probs_pallas(
    x: Tensor,
    weight: Tensor,
    *,
    top_k: int = 0,
    tile_s: int = 128,
    tile_v: int = 1024,
    interpret: bool = False,
) -> Union[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """Fused streaming log-normalizer and top-k via Pallas (TPU).

    This kernel supports only forward. Backprop will be added latter

    This function is **sharding-agnostic** — it operates on the arrays it
    receives without any ``shard_map`` or cross-device communication.  For
    sharded deployments, the caller should wrap this in ``shard_map`` and
    handle cross-device reductions externally.

    The kernel computes only the log-normalizer (and optionally top-k logits).
    Target log-prob computation (``target_logit - log_normalizer``) is the
    caller's responsibility, enabling the caller to handle target_logit
    extraction with sharded or full weight as appropriate.

    Args:
        x: Hidden states ``[B, S, D]`` (bf16 or f32).
        weight: Embedding weight ``[V, D]`` (bf16 or f32).
        top_k: If > 0, also return top-k logits and indices per position.
        tile_s: Sequence tile size.
        tile_v: Vocabulary tile size.
        interpret: If True, run in Pallas interpret mode (for CPU testing).

    Returns:
        - When ``top_k == 0``: ``log_normalizer [B, S]``.
        - When ``top_k > 0``: ``(log_normalizer, topk_logits, topk_indices)``
          where ``topk_logits`` are raw logits (not log-probs).
    """
    B, S, D = x.shape  # pylint: disable=invalid-name
    V = weight.shape[0]  # pylint: disable=invalid-name

    x, weight, padded_S, num_vocab_chunks = _pad_to_tile_boundaries(  # pylint: disable=invalid-name
        x, weight, tile_s=tile_s, tile_v=tile_v
    )

    # --- fold batch into seq ---
    total_tokens = B * padded_S
    num_token_blocks = total_tokens // tile_s
    x_flat = x.reshape(total_tokens, D)

    k_out = max(top_k, 1)
    grid = (num_token_blocks, num_vocab_chunks)

    # --- build and run pallas_call ---
    x_spec = pl.BlockSpec((tile_s, D), lambda t, v: (t, 0))
    w_spec = pl.BlockSpec((tile_v, D), lambda t, v: (v, 0))
    out_1_spec = pl.BlockSpec((tile_s, 1), lambda t, v: (t, 0))
    out_k_spec = pl.BlockSpec((tile_s, k_out), lambda t, v: (t, 0))

    kfn = functools.partial(
        _online_softmax_log_probs_kernel,
        cfg=KernelConfig(vocab_size=V, tile_v=tile_v, top_k=top_k, dot_dtype=x.dtype),
    )

    result = pl.pallas_call(
        kfn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[x_spec, w_spec],
            out_specs=KernelOutputs(
                max_ref=out_1_spec,
                sum_exp_ref=out_1_spec,
                topk_vals_ref=out_k_spec,
                topk_idx_ref=out_k_spec,
            ),
            scratch_shapes=[
                pltpu.VMEM((tile_s, 1), jnp.float32),
                pltpu.VMEM((tile_s, 1), jnp.float32),
                pltpu.VMEM((tile_s, k_out), jnp.float32),
                pltpu.VMEM((tile_s, k_out), jnp.int32),
            ],
            grid=grid,
        ),
        out_shape=KernelOutputs(
            max_ref=jax.ShapeDtypeStruct((total_tokens, 1), jnp.float32),
            sum_exp_ref=jax.ShapeDtypeStruct((total_tokens, 1), jnp.float32),
            topk_vals_ref=jax.ShapeDtypeStruct((total_tokens, k_out), jnp.float32),
            topk_idx_ref=jax.ShapeDtypeStruct((total_tokens, k_out), jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(x_flat, weight)

    # --- combine partial max + sum_exp into log-normalizer ---
    log_normalizer_flat = (result.max_ref + jnp.log(result.sum_exp_ref)).squeeze(-1)

    # --- unflatten and trim ---
    log_normalizer = log_normalizer_flat.reshape(B, padded_S)[:, :S]

    if top_k > 0:
        topk_logits = result.topk_vals_ref[:, :top_k].reshape(B, padded_S, top_k)[:, :S, :]
        topk_indices = result.topk_idx_ref[:, :top_k].reshape(B, padded_S, top_k)[:, :S, :]
        return log_normalizer, topk_logits, topk_indices

    return log_normalizer
