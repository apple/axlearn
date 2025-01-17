# Copyright © 2024 Apple Inc.

""" Pallas kernels for Mamba2

High-level idea: this kernel implements a two-level chunking algorithm to
balance memory consumption and running speed. Intuitively, we store chunk-level
hidden states to avoid recomputation, and subchunk-level states are recomputed based
on the chunk-level states.


Notations:
    nb: number of chunks
    ns: number of subchunks
    bl: subchunk size
    dkn: number of tiles in the dk dim
    dvn: number of tiles in the dv dim
    dk: state_dim (corresponds to dim of qk heads)
    dv: head_dim (corresponds to dim of v heads)

q/k/v is used as it's more intuitive than b/c/x of SSD in the orginal implementation,
see section 7.2 https://arxiv.org/pdf/2405.21060. Accordingly, dk/dv is used instead
of state_dim/head_dim.  This notation is also used in linear attention models.
However, state_dim/head_dim is used in the model file to be consistent with Mamba1
and the original implementation.

"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import lax
from jax._src.lax.control_flow import for_loop
from jax.experimental import pallas as pl

from axlearn.common.utils import Tensor


def _matmul_fp32(lhs: Tensor, rhs: Tensor) -> Tensor:
    """A wrapper around jax.lax.dot to conduct float32 matmul"""
    return jax.lax.dot(lhs, rhs, precision="float32", preferred_element_type=jnp.float32)


@jax.custom_vjp
def _ssd(q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, h0: Tensor) -> Tensor:
    """A differentiable function that computes the output of SSD.

    Args:
        q: [bs, num_heads, seq_len, dk]
        k: [bs, num_heads, seq_len, dk]
        v: [bs, num_heads, seq_len, dv]
        log_alpha: [bs, num_heads, seq_len]
        h0: [bs, num_heads, dk, dv]

    Returns:
        o: [bs, num_heads, seq_len, dv]
    """
    (
        o,
        _,
    ) = _ssd_forward(q, k, v, log_alpha, h0)
    return o


def _ssd_forward_kernel(
    q_ref: Tensor,
    k_ref: Tensor,
    v_ref: Tensor,
    cum_log_alpha_ref: Tensor,
    initial_state_ref: Tensor,
    gamma_ref: Tensor,
    mutable_ch_ref: Tensor,
    mutable_final_state_ref: Tensor,
    mutable_o_ref: Tensor,
):
    """Forward kernel for SSD.

    Args:
        q_ref: tensor reference of shape [ns, bl, singleton_dim]
        k_ref: tensor reference of shape [ns, bl, singleton_dim]
        v_ref: tensor reference of shape [ns, bl, singleton_dim]
        cum_log_alpha_ref: tensor reference of shape [ns, bl]
        initial_state_ref: tensor reference of shape [singleton_dim, singleton_dim]
        gamma_ref: tensor reference of shape [ns, bl, singleton_dim]

    Output via mutable tensors:
        mutable_ch_ref: tensor reference of shape [ns, singleton_dim, singleton_dim]
        mutable_final_state_ref: tensor reference of shape [singleton_dim, singleton_dim]
        mutable_o_ref: tensor reference of shape [ns, bl, singleton_dim]

    Note on intial_state and final_state:
        * initial_state is at seq-level and not updated during the forward pass
        * final_state is used to pass chunk-level states across different chunks
            - it will be initialized to initial_state at the beginning of each chunk
            - it will be updated after processing each chunk
            - in the end, it will return as the seq-level final state
    """
    subchunk_dim, subchunk_size = cum_log_alpha_ref.shape[0], cum_log_alpha_ref.shape[1]
    casual_mask = jnp.tril(jnp.ones((subchunk_size, subchunk_size)), k=0)

    # In our grid definition, axis 4 is the chunk index.
    @pl.when(pl.program_id(axis=4) == 0)
    def init_carry():
        mutable_final_state_ref[:, :] = initial_state_ref[:, :]

    def _ssd_forward_chunk_loop_body(t: int, h_carry_ref: Tensor):
        subchunk_idx = t
        prev_state = h_carry_ref[:, :]

        q_block = q_ref[subchunk_idx, :].astype(jnp.float32)
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)

        # Notation mapping wrt. the paper: lambda -> Lambda, gamma -> gamma, beta -> Gamma.
        lambda_block = cum_log_alpha_ref[subchunk_idx, :]
        gamma_block = gamma_ref[subchunk_idx]

        lambda_block = jnp.expand_dims(lambda_block, axis=-1)  # [bl, 1]
        beta_block = (
            jnp.expand_dims(gamma_block, axis=0) - lambda_block
        )  # [bl, singleton_dim] after broadcasting
        ssd_mask_block = lambda_block - jnp.transpose(lambda_block, [1, 0])
        ssd_mask_block = ssd_mask_block * casual_mask

        lambda_block = jnp.exp(lambda_block)
        beta_block = jnp.exp(beta_block)
        gamma_block = jnp.exp(gamma_block)
        ssd_mask_block = jnp.exp(ssd_mask_block)

        q_tilde_block = q_block * lambda_block
        k_tilde_block = k_block * beta_block

        o_block_inter = _matmul_fp32(q_tilde_block, prev_state)
        intra_att = _matmul_fp32(q_block, k_block.T)
        attn_mask = casual_mask * ssd_mask_block
        o_block_intra = _matmul_fp32((intra_att * attn_mask), v_block)
        o_block = o_block_inter + o_block_intra

        cur_state = prev_state * jnp.expand_dims(gamma_block, axis=-1) + _matmul_fp32(
            k_tilde_block.T, v_block
        )  # [d_k, d_v]
        h_carry_ref[:, :] = cur_state
        mutable_o_ref[subchunk_idx, :] = o_block.astype(mutable_o_ref.dtype)

    # Obtain final state from previous chunk.
    h_carry = mutable_final_state_ref[:, :]
    mutable_ch_ref[:, :] = mutable_final_state_ref[:, :]
    final_state = for_loop.for_loop(
        subchunk_dim,
        _ssd_forward_chunk_loop_body,
        h_carry,
    )
    mutable_final_state_ref[:, :] = final_state


@jax.jit
def _ssd_forward(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, initial_state: Tensor
) -> Tuple:
    """Forward pass for SSD.

    Args:
        q, k: [bs, num_heads, seq_len, dk]
        v: [bs, num_heads, seq_len, dv]
        log_alpha: [bs, num_heads, seq_len]
        initial_state: [singleton_dim, singleton_dim]

    Returns:
        o: [bs, num_heads, seq_len, dv]
        residuals: Tuple of tensors to be used in the backward
    """
    bs, num_qk_heads, seq_len, k_head_dim = q.shape
    _, num_v_heads, _, v_head_dim = v.shape
    # TODO (bailin-wang): the following defaults works best for v5p, but they may not be optimal
    # for others tpu types. We may need to expose them as arguments in the future.
    singleton_dim = 128
    chunk_size, subchunk_size = 512, 64
    acc_dtype, orig_dtype = jnp.float32, q.dtype

    assert seq_len % chunk_size == 0 and chunk_size % subchunk_size == 0

    assert num_v_heads % num_qk_heads == 0
    num_heads = num_v_heads
    num_head_per_group = num_v_heads // num_qk_heads

    assert k_head_dim % singleton_dim == 0
    assert v_head_dim % singleton_dim == 0
    num_k_tiles = k_head_dim // singleton_dim
    num_v_tiles = v_head_dim // singleton_dim

    # Add two extra dims for chunk-wise computation.
    chunk_dim = seq_len // chunk_size
    subchunk_dim = chunk_size // subchunk_size

    grid = (bs, num_heads, num_k_tiles, num_v_tiles, chunk_dim)

    # q/k/v tensors are kept in bf16 and converted later to fp32 in VMEM.
    log_alpha = log_alpha.astype(jnp.float32)
    initial_state = initial_state.astype(jnp.float32)

    # None is effectively 1, but the dim will be squeezed out.
    qk_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    qk_spec = pl.BlockSpec(
        lambda b, h, k, v, m: (b, lax.div(h, num_head_per_group), m, 0, k), qk_tiling
    )
    v_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    v_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, m, 0, v), v_tiling)

    alpha_tiling = (None, None, None, subchunk_dim, subchunk_size)
    alpha_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, m, 0, 0), alpha_tiling)

    # Initial hidden states.
    is_tiling = (None, None, singleton_dim, singleton_dim)
    is_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, k, v), is_tiling)

    # Chunk-wise states (not subchunk-wise states).
    ch_tiling = (None, None, None, singleton_dim, singleton_dim)
    ch_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, m, k, v), ch_tiling)

    # Chunk-wise final states help pass states from the previous chunk to the next.
    fs_spec = is_spec

    ch_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, chunk_dim, k_head_dim, v_head_dim), dtype=acc_dtype
    )
    fs_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32
    )

    # Pre-compute the cumulative sum of log_alpha.
    log_alpha = rearrange(
        log_alpha, "b h (nb ns bl) -> b h nb ns bl", nb=chunk_dim, ns=subchunk_dim
    )
    cum_log_alpha = jnp.cumsum(log_alpha, axis=-1)

    q = rearrange(q, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    k = rearrange(k, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    v = rearrange(v, "b h (nb bl) dv -> b h nb bl dv", bl=subchunk_size)

    # Pallas kernels operate on tiles of size at least [8, 128].
    gamma = cum_log_alpha[:, :, :, :, subchunk_size - 1 :]  # [b, h, nb, ns, 1]
    gamma_expanded = jnp.repeat(gamma, singleton_dim, axis=-1)  # [b, h, nb, ns, singleton_dim]
    gamma_tiling = (None, None, None, subchunk_dim, singleton_dim)
    gamma_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, m, 0, 0), gamma_tiling)

    o_tiling = (None, None, None, subchunk_dim, subchunk_size, singleton_dim)
    o_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, k, m, 0, v), o_tiling)
    o_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, num_k_tiles, chunk_dim * subchunk_dim, subchunk_size, v_head_dim),
        dtype=orig_dtype,
    )

    chunk_states, final_state, o = pl.pallas_call(
        _ssd_forward_kernel,
        in_specs=(qk_spec, qk_spec, v_spec, alpha_spec, is_spec, gamma_spec),
        out_specs=(ch_spec, fs_spec, o_spec),
        out_shape=(ch_shape, fs_shape, o_shape),
        grid=grid,
        compiler_params=dict(
            mosaic=dict(
                dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary")
            )
        ),
    )(q, k, v, cum_log_alpha, initial_state, gamma_expanded)

    o = jnp.sum(o, axis=2)  # sum over dkn dim
    o = rearrange(o, "b h nb bl dv -> b h (nb bl) dv")

    # Input tensors q/k/v stored in the residual list for backward pass are reshaped, and
    # cum_log_alpha and gamma are upcasted to float32.
    final_state = final_state.astype(orig_dtype)
    return o, (q, k, v, cum_log_alpha, gamma_expanded, chunk_states, final_state)


def _ssd_backward_kernel(
    q_ref: Tensor,
    k_ref: Tensor,
    v_ref: Tensor,
    cum_log_alpha_ref: Tensor,
    gamma_ref: Tensor,
    ch_ref: Tensor,
    mutable_do_ref: Tensor,
    mutable_dq_ref: Tensor,
    mutable_dk_ref: Tensor,
    mutable_dv_ref: Tensor,
    mutable_dh_carry_ref: Tensor,
):
    """Backward kernel for SSD.

    Args:
        q_ref: tensor reference of shape [ns, bl, singleton_dim]
        k_ref: tensor reference of shape [ns, bl, singleton_dim]
        v_ref: tensor reference of shape [ns, bl, singleton_dim]
        cum_log_alpha_ref: tensor reference of shape [ns, bl]
        gamma_ref: tensor reference of shape [ns, bl, singleton_dim]
        ch_ref: tensor reference of shape [ns, singleton_dim, singleton_dim]

    Output via mutable tensors:
        mutable_do_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dq_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dk_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dv_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dh_carry_ref: tensor reference of shape [ns, singleton_dim, singleton_dim]

    Note: similar to final_state in the forward pass, dh_carry is used to pass gradients wrt.
    hidden states across different chunks. It will be initalized to zero at the last chunk.
    The final gradient wrt. hidden states will be returned as the gradient wrt. initial_state.
    """
    subchunk_dim, subchunk_size = cum_log_alpha_ref.shape[0], cum_log_alpha_ref.shape[1]
    causal_mask = jnp.tril(jnp.ones((subchunk_size, subchunk_size)), k=0).astype(jnp.float32)

    # In our grid definition, axis 4 is the chunk index.
    @pl.when(pl.program_id(axis=4) == 0)
    def init_carry():
        mutable_dh_carry_ref[:, :] = jnp.zeros_like(mutable_dh_carry_ref, dtype=jnp.float32)

    def _ssd_backward_dq_chunk_loop_body(t: int, h_carry_ref: Tensor):
        subchunk_idx = t
        h_block = h_carry_ref[:, :]  # final states from previous chunk
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)
        do_block = mutable_do_ref[subchunk_idx, :].astype(jnp.float32)

        lambda_block = cum_log_alpha_ref[subchunk_idx, :]
        gamma_block = gamma_ref[subchunk_idx]

        lambda_block = jnp.expand_dims(lambda_block, axis=-1)  # [nb, 1]
        beta_block = gamma_block - lambda_block  # [nb, d_k]
        ssd_mask_block = lambda_block - jnp.transpose(lambda_block, [1, 0])
        ssd_mask_block = ssd_mask_block * causal_mask

        lambda_block = jnp.exp(lambda_block)
        beta_block = jnp.exp(beta_block)
        gamma_block = jnp.exp(gamma_block)
        ssd_mask_block = jnp.exp(ssd_mask_block)

        k_tilde_block = k_block * beta_block

        attn_mask = causal_mask * ssd_mask_block
        d_intra_att = _matmul_fp32(do_block, v_block.T) * attn_mask

        dq_tilde_block = _matmul_fp32(do_block, h_block.T)
        dq_block_1 = dq_tilde_block * lambda_block
        dq_block_2 = _matmul_fp32(d_intra_att, k_block)
        dq_block = dq_block_1 + dq_block_2
        mutable_dq_ref[subchunk_idx, :] = dq_block

        next_h_block = h_block * jnp.expand_dims(gamma_block, axis=-1) + _matmul_fp32(
            k_tilde_block.T, v_block
        )
        h_carry_ref[:, :] = next_h_block

    def _ssd_backward_dkv_chunk_loop_body(t: int, dh_carry_ref: Tensor):
        subchunk_idx = t
        dh_block = dh_carry_ref[:, :]
        q_block = q_ref[subchunk_idx, :].astype(jnp.float32)
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)
        do_block = mutable_do_ref[subchunk_idx, :].astype(jnp.float32)
        causal_mask = jnp.tril(jnp.ones((subchunk_size, subchunk_size)), k=0).astype(jnp.float32)

        lambda_block = cum_log_alpha_ref[subchunk_idx, :]
        gamma_block = gamma_ref[subchunk_idx]

        lambda_block = jnp.expand_dims(lambda_block, axis=-1)  # [nb, 1]
        beta_block = gamma_block - lambda_block  # [nb, d_k]
        ssd_mask_block = lambda_block - jnp.transpose(lambda_block, [1, 0])
        ssd_mask_block = ssd_mask_block * causal_mask

        lambda_block = jnp.exp(lambda_block)
        beta_block = jnp.exp(beta_block)
        gamma_block = jnp.exp(gamma_block)
        ssd_mask_block = jnp.exp(ssd_mask_block)

        q_tilde_block = q_block * lambda_block
        k_tilde_block = k_block * beta_block

        intra_att = _matmul_fp32(q_block, k_block.T)
        attn_mask = causal_mask * ssd_mask_block
        d_intra_att = _matmul_fp32(do_block, v_block.T) * attn_mask

        dk_block_1 = _matmul_fp32(d_intra_att.T, q_block)
        dk_tilde_block = _matmul_fp32(v_block, dh_block.T)
        dk_block_2 = dk_tilde_block * beta_block
        dk_block = dk_block_1 + dk_block_2
        mutable_dk_ref[subchunk_idx, :] = dk_block

        dv_block_1 = _matmul_fp32((intra_att * attn_mask).T, do_block)
        dv_block_2 = _matmul_fp32(k_tilde_block, dh_block)
        dv_block = dv_block_1 + dv_block_2
        mutable_dv_ref[subchunk_idx, :] = dv_block

        prev_dh_block = dh_block * jnp.expand_dims(gamma_block, axis=-1) + _matmul_fp32(
            q_tilde_block.T, do_block
        )
        dh_carry_ref[:, :] = prev_dh_block

    h_carry = ch_ref[:, :]
    _ = for_loop.for_loop(subchunk_dim, _ssd_backward_dq_chunk_loop_body, h_carry)

    dh_carry = mutable_dh_carry_ref[:, :]
    dinitial_state = for_loop.for_loop(
        subchunk_dim, _ssd_backward_dkv_chunk_loop_body, dh_carry, reverse=True
    )
    mutable_dh_carry_ref[:, :] = dinitial_state


@jax.jit
def _ssd_backward(residuals: Tuple, do: Tensor) -> Tuple:
    """Backward pass for SSD.

    Args:
        residuals: Tuple of tensors returned from the forward pass
        do: [bs, num_heads, seq_len, dv]

    Returns:
        dq: [bs, num_heads, seq_len, dk]
        dk: [bs, num_heads, seq_len, dk]
        dv: [bs, num_heads, seq_len, dv]
        dlog_alpha: [bs, num_heads, seq_len]
        dinitial_state: [bs, num_heads, dk, dv]
    """
    q, k, v, cum_log_alpha, gamma_expanded, chunk_states, final_state = residuals

    # `final_state` preserves the original dtype (e.g., bfloat16).
    orig_dtype = final_state.dtype

    singleton_dim = 128
    bs, num_heads, chunk_dim, subchunk_dim, subchunk_size = cum_log_alpha.shape
    k_dim, v_dim = q.shape[-1], v.shape[-1]
    num_k_tiles, num_v_tiles = k_dim // singleton_dim, v_dim // singleton_dim
    num_qk_heads = q.shape[1]
    num_head_per_group = num_heads // num_qk_heads

    grid = (bs, num_heads, num_k_tiles, num_v_tiles, chunk_dim)

    qk_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    qk_spec = pl.BlockSpec(
        lambda b, h, k, v, m: (b, lax.div(h, num_head_per_group), chunk_dim - 1 - m, 0, k),
        qk_tiling,
    )
    v_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    v_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, v), v_tiling)

    alpha_tiling = (None, None, None, subchunk_dim, subchunk_size)
    alpha_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, 0), alpha_tiling)
    gamma_tiling = (None, None, None, subchunk_dim, singleton_dim)
    gamma_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, 0), gamma_tiling)

    ch_tiling = (None, None, None, singleton_dim, singleton_dim)
    ch_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, k, v), ch_tiling)

    do_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    do_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, chunk_dim - 1 - m, 0, v), do_tiling)

    dqk_tiling = (None, None, None, None, subchunk_dim, subchunk_size, singleton_dim)
    dqk_spec = pl.BlockSpec(
        lambda b, h, k, v, m: (
            b,
            lax.div(h, num_head_per_group),
            lax.rem(h, num_head_per_group),
            v,
            chunk_dim - 1 - m,
            0,
            k,
        ),
        dqk_tiling,
    )
    dqk_shape = jax.ShapeDtypeStruct(
        shape=(
            bs,
            num_qk_heads,
            num_head_per_group,
            num_v_tiles,
            chunk_dim * subchunk_dim,
            subchunk_size,
            k_dim,
        ),
        dtype=jnp.float32,
    )

    dv_tiling = (None, None, None, subchunk_dim, subchunk_size, singleton_dim)
    dv_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, k, chunk_dim - 1 - m, 0, v), dv_tiling)
    dv_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, num_k_tiles, chunk_dim * subchunk_dim, subchunk_size, v_dim),
        dtype=jnp.float32,
    )

    dh_carry_tiling = (None, None, singleton_dim, singleton_dim)
    dh_carry_spec = pl.BlockSpec(lambda b, h, k, v, m: (b, h, k, v), dh_carry_tiling)
    dh_carry_shape = jax.ShapeDtypeStruct(shape=(bs, num_heads, k_dim, v_dim), dtype=jnp.float32)

    do = rearrange(do, "b h (nb bl) dv -> b h nb bl dv", bl=subchunk_size)

    dq, dk, dv, dinitial_state = pl.pallas_call(
        _ssd_backward_kernel,
        in_specs=(qk_spec, qk_spec, v_spec, alpha_spec, gamma_spec, ch_spec, do_spec),
        out_specs=(dqk_spec, dqk_spec, dv_spec, dh_carry_spec),
        out_shape=(dqk_shape, dqk_shape, dv_shape, dh_carry_shape),
        grid=grid,
        compiler_params=dict(
            mosaic=dict(
                dimension_semantics=("parallel", "parallel", "parallel", "parallel", "arbitrary")
            )
        ),
    )(q, k, v, cum_log_alpha, gamma_expanded, chunk_states, do)

    # Sum over dvn dim.
    dq = jnp.sum(dq, axis=3)
    dk = jnp.sum(dk, axis=3)
    dq = rearrange(dq, "b ng nhg nb bl dk -> b ng nhg (nb bl) dk")
    dk = rearrange(dk, "b ng nhg nb bl dk -> b ng nhg (nb bl) dk")

    # Compute dlog_alpha via `q * dq - k * dk`.
    dq_ = rearrange(dq, "b ng nhg l dk -> b (ng nhg) l dk")
    dk_ = rearrange(dk, "b ng nhg l dk -> b (ng nhg) l dk")

    q_ = repeat(q, "b ng nb bl dk -> b (ng nhg) nb bl dk", nhg=num_head_per_group)
    k_ = repeat(k, "b ng nb bl dk -> b (ng nhg) nb bl dk", nhg=num_head_per_group)
    q_ = rearrange(q_, "b h nb bl dk -> b h (nb bl) dk")
    k_ = rearrange(k_, "b h nb bl dk -> b h (nb bl) dk")

    dlog_alpha_ = jnp.sum(dq_ * q_ - dk_ * k_, axis=-1)
    dlog_alpha = lax.cumsum(dlog_alpha_, axis=2, reverse=True)

    # Sum over dkn dim.
    dv = jnp.sum(dv, axis=2)
    dv = rearrange(dv, "b h nb bl dv -> b h (nb bl) dv")

    # Sum over nhg dim
    dq = jnp.sum(dq, axis=2)
    dk = jnp.sum(dk, axis=2)
    # `dlog_alpha` is always in float32, `dv` is also in float32.
    dq, dk = dq.astype(orig_dtype), dk.astype(orig_dtype)

    dinitial_state = dinitial_state.astype(orig_dtype)
    return dq, dk, dv, dlog_alpha, dinitial_state


_ssd.defvjp(_ssd_forward, _ssd_backward)


@jax.jit
@jax.named_call  # `named_call` ensures the name is used in tracing, which is useful for profiling.
def ssd(q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, h0: Optional[Tensor] = None) -> Tensor:
    """Differentiable function that computes the output of SSD.

    Args:
        q: [batch_size, num_groups, seq_len, dk]
        k: [batch_size, num_groups, seq_len, dk]
        v: [batch_size, num_groups, seq_len, dv]
        log_alpha: [batch_size, num_heads, seq_len]
        h0: [batch_size, num_heads, dk, dv]

    Returns:
        output: [batch_size, num_heads, seq_len, dv]

    The notion of groups is similar to the group in multi-group attention (or more preciesly
    multi-value attention) -- one group of q/k corresponds to multiple v heads.
    """

    bs, ng, _, dk = q.shape
    bs, nh, _, dv = v.shape
    assert nh % ng == 0
    assert v.dtype == jnp.float32
    assert log_alpha.dtype == jnp.float32

    if h0 is None:
        h0 = jnp.zeros((bs, nh, dk, dv), dtype=jnp.float32)

    output = _ssd(q, k, v, log_alpha, h0)
    return output


def ssd_linear_scan(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, h0: Union[Tensor, None] = None
) -> Tensor:
    """LinearScan based reference implementations for testing SSD kernels.

    Args:
        q, k: [batch_size, num_groups, seq_len, dk]
        k: [batch_size, num_groups, seqlen, dk]
        v: [batch_size, num_heads, seq_len, dv]
        log_alpha: [batch_size, num_heads, seq_len]
        h0: [batch_size, num_heads, dk, dv] or None

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
        hidden_state: [batch_size, num_heads, dk, dv]
    """
    bs, ng, _, dk = q.shape
    bs, nh, _, dv = v.shape
    assert nh % ng == 0

    # The linearscan kernel assumes that nh == ng, so we need to repeat q/k.
    num_head_per_group = nh // ng
    q = repeat(q, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)
    k = repeat(k, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)

    # ITt's more convenient for vmap to have internal states of size [dv, dk]
    if h0 is None:
        h0 = jnp.zeros((bs, nh, dv, dk), dtype=jnp.float32)
    else:
        # to be consistent with pallas api, h0 is in dk x dv as input
        h0 = rearrange(h0, "b h dk dv -> b h dv dk")

    # All inputs are upcasted to float32, making this function a good reference funciton to
    # test pallas kernel's numerical precision in the case of bf16 inputs.
    dtype = q.dtype
    if dtype == jnp.bfloat16:
        q, k, v, h0 = map(lambda x: x.astype(jnp.float32), (q, k, v, h0))

    def scan_body_fn(h_prev, current_inputs):
        acc_dtype = h_prev.dtype
        q_t, k_t, v_t, log_a_t = current_inputs
        a_t = jnp.exp(log_a_t).astype(acc_dtype)
        h_next = a_t * h_prev + jnp.einsum("i,j->ij", v_t, k_t, preferred_element_type=jnp.float32)
        o_t = jnp.einsum("ij,j->i", h_next, q_t, preferred_element_type=jnp.float32)
        return h_next, o_t.astype(q_t.dtype)

    def single_head_scan(q_head, k_head, v_head, alpha_head, h0_head):
        return jax.lax.scan(scan_body_fn, h0_head, (q_head, k_head, v_head, alpha_head))

    multi_head_scan = jax.vmap(single_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0))
    batched_scan = jax.vmap(multi_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0))

    # Note: if dk > 128 (e.g., 256), somehow jax jvp would fail; a work-around
    # is to add another dim to ensure that minor dk is always 128.
    q = rearrange(q, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    k = rearrange(k, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    h0 = rearrange(h0, "b h dv (dkn dks) -> dkn b h dv dks", dks=128)

    batched_scan = jax.vmap(batched_scan, in_axes=(0, 0, None, None, 0), out_axes=(0, 0))
    final_state, output = batched_scan(q, k, v, log_alpha, h0)
    final_state = rearrange(final_state, "dkn b h dv dks -> b h dv (dkn dks)")
    output = jnp.sum(output, axis=0)

    final_state = rearrange(final_state, "b h dv dk -> b h dk dv")

    if dtype == jnp.bfloat16:
        output = output.astype(jnp.bfloat16)
        final_state = final_state.astype(jnp.bfloat16)

    return output, final_state


def ssd_linear_scan_w_hidden_states(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, h0: Union[Tensor, None] = None
) -> Tensor:
    """LinearScan based reference implementations for testing SSD kernels.

    This version additionally returns the hidden states of all tokens.

    Args:
        q: [batch_size, num_groups, seqlen, dk]
        k: [batch_size, num_groups, seqlen, dk]
        v: [batch_size, num_heads, seqlen, dv]
        log_alpha: [batch_size, num_heads, seqlen]
        h0: [batch_size, num_heads, dk, dv] or None

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
        hidden_states: [batch_size, num_heads, seq_len, dk, dv]
    """
    bs, ng, _, dk = q.shape
    bs, nh, _, dv = v.shape
    assert nh % ng == 0

    num_head_per_group = nh // ng
    q = repeat(q, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)
    k = repeat(k, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)

    if h0 is None:
        h0 = jnp.zeros((bs, nh, dv, dk), dtype=jnp.float32)
    else:
        # to be consistent with pallas api, h0 is in dk x dv as input
        h0 = rearrange(h0, "b h dk dv -> b h dv dk")

    dtype = q.dtype
    if dtype == jnp.bfloat16:
        q, k, v, h0 = map(lambda x: x.astype(jnp.float32), (q, k, v, h0))

    def scan_body_fn(h_prev, current_inputs):
        acc_dtype = h_prev.dtype
        k_t, v_t, log_a_t = current_inputs
        a_t = jnp.exp(log_a_t).astype(acc_dtype)
        h_next = a_t * h_prev + jnp.einsum("i,j->ij", v_t, k_t)
        return h_next, h_next

    def single_head_scan(k_head, v_head, alpha_head, h0_head):
        return jax.lax.scan(scan_body_fn, h0_head, (k_head, v_head, alpha_head))

    multi_head_scan = jax.vmap(single_head_scan, in_axes=(0, 0, 0, 0), out_axes=(0, 0))
    batched_scan = jax.vmap(multi_head_scan, in_axes=(0, 0, 0, 0), out_axes=(0, 0))

    k = rearrange(k, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    h0 = rearrange(h0, "b h dv (dkn dks) -> dkn b h dv dks", dks=128)

    batched_scan = jax.vmap(batched_scan, in_axes=(0, None, None, 0), out_axes=(0, 0))
    final_state, hidden_states = batched_scan(k, v, log_alpha, h0)
    assert final_state is not None

    hidden_states = rearrange(hidden_states, "dkn b h l dv dks -> b h l (dkn dks) dv")
    output = jnp.einsum(
        "b h l s, b h l s d -> b h l d", q, hidden_states, preferred_element_type=jnp.float32
    )

    if dtype == jnp.bfloat16:
        output = output.astype(jnp.bfloat16)
    return output, hidden_states


def ssd_linear_scan_w_timestep(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, timestep: Tensor, h0=None
) -> Tensor:
    """LinearScan that takes timestep as input and masks useless k/v based on timestep.

    This function is used during inference where decoding might start from different timesteps.

    Args:
        q: [batch_size, num_groups, seqlen, dk]
        k: [batch_size, num_groups, seqlen, dk]
        v: [batch_size, num_heads, seqlen, dv]
        log_alpha: [batch_size, num_heads, seqlen]
        h0: [batch_size, num_heads, dk, dv] or None
        timestep: [batch_size, seqlen] or None

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
        hidden_states: [batch_size, num_heads, dk, dv]

    """
    bs, ng, l, dk = q.shape
    bs, nh, l, dv = v.shape
    assert nh % ng == 0

    num_head_per_group = nh // ng
    q = repeat(q, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)
    k = repeat(k, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)

    timestep_mask = jnp.arange(l)[None, :] >= timestep[:, None]
    k = jnp.where(timestep_mask[:, None, :, None], 0.0, k)
    v = jnp.where(timestep_mask[:, None, :, None], 0.0, v)
    log_alpha = jnp.where(timestep_mask[:, None, :], 0.0, log_alpha)

    if h0 is None:
        h0 = jnp.zeros((bs, nh, dv, dk), dtype=jnp.float32)
    else:
        # to be consistent with pallas api, h0 is in dk x dv as input
        h0 = rearrange(h0, "b h dk dv -> b h dv dk")

    dtype = q.dtype
    if dtype == jnp.bfloat16:
        q, k, v, h0 = map(lambda x: x.astype(jnp.float32), (q, k, v, h0))

    def scan_body_fn(h_prev, current_inputs):
        acc_dtype = h_prev.dtype
        q_t, k_t, v_t, log_a_t = current_inputs
        a_t = jnp.exp(log_a_t).astype(acc_dtype)
        h_next = a_t * h_prev + jnp.einsum("i,j->ij", v_t, k_t, preferred_element_type=jnp.float32)
        o_t = jnp.einsum("ij,j->i", h_next, q_t, preferred_element_type=jnp.float32)
        return h_next, o_t.astype(q_t.dtype)

    def single_head_scan(q_head, k_head, v_head, alpha_head, h0_head):
        return jax.lax.scan(scan_body_fn, h0_head, (q_head, k_head, v_head, alpha_head))

    multi_head_scan = jax.vmap(single_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0))
    batched_scan = jax.vmap(multi_head_scan, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0))

    q = rearrange(q, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    k = rearrange(k, "b h l (dkn dks) -> dkn b h l dks", dks=128)
    h0 = rearrange(h0, "b h dv (dkn dks) -> dkn b h dv dks", dks=128)

    batched_scan = jax.vmap(batched_scan, in_axes=(0, 0, None, None, 0), out_axes=(0, 0))
    final_state, output = batched_scan(q, k, v, log_alpha, h0)
    final_state = rearrange(final_state, "dkn b h dv dks -> b h dv (dkn dks)")
    output = jnp.sum(output, axis=0)

    final_state = rearrange(final_state, "b h dv dk -> b h dk dv")

    if dtype == jnp.bfloat16:
        output = output.astype(jnp.bfloat16)

    return output, final_state
