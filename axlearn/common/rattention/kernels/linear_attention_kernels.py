# Copyright © 2025 Apple Inc.

""" Pallas kernels for Linear Attention (LA) specialized for sliding window attention.

A specialized feature map from the following reference is used to support sliding window attention.
The chunking strategy is similar to the one used in ssm_kernels/ssd_kernels.py.

Notations:
    nb: number of chunks
    ns: number of subchunks
    bl: subchunk size
    dk: head dim of q/k
    dv: head_dim of v

References:
    Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
        (https://arxiv.org/abs/2006.16236)

    TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer
        (https://arxiv.org/abs/2307.14995)

"""

from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jax import lax
from jax._src.lax.control_flow import for_loop
from jax.experimental import pallas as pl

from axlearn.common.rattention.kernels.utils import FeatureMap, get_feature_map
from axlearn.common.utils import Tensor


def _matmul_fp32(lhs: Tensor, rhs: Tensor) -> Tensor:
    """A wrapper around jax.lax.dot to conduct float32 matmul"""
    return jax.lax.dot(lhs, rhs, precision="float32", preferred_element_type=jnp.float32)


def right_shift_and_zero_pad(x: Tensor, shift_size: int, axis: int = 1):
    """
    Right shift the tensor and pad zeros on the left side, e.g., [2, 5, 3] with shift_size=2
    -> [0, 0, 2].

    Args:
        x: input tensor of shape [batch_size, num_heads, seq_len, dk].
        shift_size: the number of elements to shift.
        axis: the axis to shift.

    Returns:
        y: shifted tensor of the same shape as x.
    """

    if x.shape[axis] <= shift_size:
        return jnp.zeros_like(x)

    padding_shape = list(x.shape)
    padding_shape[axis] = shift_size
    zeros = jnp.zeros(padding_shape, dtype=x.dtype)

    # Create slicing indices for the main tensor
    slice_indices = []
    for i, _ in enumerate(x.shape):
        if i == axis:
            slice_indices.append(slice(0, x.shape[i] - shift_size))
        else:
            slice_indices.append(slice(None))
    slice_indices = tuple(slice_indices)
    sliced = x[slice_indices]

    # Concatenate zeros and sliced tensor
    return jnp.concatenate([zeros, sliced], axis=axis)


@partial(
    jax.custom_vjp,
    nondiff_argnums=(
        4,
        5,
    ),
)
def _linear_attention(
    q: Tensor, k: Tensor, v: Tensor, h0: Tensor, feat_map: FeatureMap, chunk_size: int
) -> Tensor:
    """A differentiable function that computes the output of linear attention.

    Args:
        q: [batch_size, num_heads, seq_len, dk]
        k: [batch_size, num_kv_heads, seq_len, dk]
        v: [batch_size, num_kv_heads, seq_len, dv]
        h0: [batch_size, num_heads, dk, dv]
        feat_map: an instance of FeatureMap
        chunk_size: int, size of each chunk

    Returns:
        o: [batch_size, num_heads, seq_len, dv]
    """
    (
        o,
        _,
    ) = _linear_attention_forward(q, k, v, h0, feat_map, chunk_size)
    return o


def _linear_attention_forward_kernel(
    q_ref: Tensor,
    k_ref: Tensor,
    v_ref: Tensor,
    initial_state_ref: Tensor,
    mutable_ch_ref: Tensor,
    mutable_final_state_ref: Tensor,
    mutable_o_ref: Tensor,
    *,
    feat_map: FeatureMap,
):
    """Forward kernel for LA.

    Args:
        q_ref: tensor reference of shape [ns, bl, singleton_dim]
        k_ref: tensor reference of shape [ns, bl, singleton_dim]
        v_ref: tensor reference of shape [ns, bl, singleton_dim]
        initial_state_ref: tensor reference of shape [singleton_dim, singleton_dim]

    Output via mutable tensors:
        mutable_ch_ref: tensor reference of shape [ns, singleton_dim, singleton_dim]
        mutable_final_state_ref: tensor reference of shape [singleton_dim, singleton_dim]
        mutable_o_ref: tensor reference of shape [ns, bl, singleton_dim]

    Note on initial_state and final_state:
        * initial_state is at seq-level and not updated during the forward pass
        * mutable_final_state is used to pass chunk-level states across different chunks
            - it will be initialized to initial_state at the beginning of each chunk
            - it will be updated after processing each chunk
            - in the end, it will return as the seq-level final state
    """
    subchunk_dim, subchunk_size = q_ref.shape[0], q_ref.shape[1]
    casual_mask = jnp.tril(jnp.ones((subchunk_size, subchunk_size)), k=0)
    feat_fn = get_feature_map(feat_map)

    @pl.when(pl.program_id(axis=2) == 0)
    def init_carry():
        mutable_final_state_ref[:, :] = initial_state_ref[:, :]

    def _la_forward_chunk_loop_body(t: int, h_carry_ref: Tensor):
        subchunk_idx = t
        h_block = h_carry_ref[:, :]

        q_block = q_ref[subchunk_idx, :].astype(jnp.float32)
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)

        q_block_ = feat_fn.fwd(q_block)
        k_block_ = feat_fn.fwd(k_block)

        o_block_inter = _matmul_fp32(q_block_, h_block)
        intra_att = _matmul_fp32(q_block_, k_block_.T)
        o_block_intra = _matmul_fp32((intra_att * casual_mask), v_block)
        o_block = o_block_inter + o_block_intra

        next_h_block = h_block + _matmul_fp32(k_block_.T, v_block)  # [d_k, d_v]
        mutable_o_ref[subchunk_idx, :] = o_block.astype(mutable_o_ref.dtype)

        h_carry_ref[:, :] = next_h_block

    # `ch_ref` stores the final state from previous chunk.
    mutable_ch_ref[:, :] = mutable_final_state_ref[:, :]

    # Obtain final state from previous chunk.
    h_carry = mutable_final_state_ref[:, :]
    final_state = for_loop.for_loop(
        subchunk_dim,
        _la_forward_chunk_loop_body,
        h_carry,
    )
    mutable_final_state_ref[:, :] = final_state


@partial(
    jax.jit,
    static_argnames=(
        "feat_map",
        "chunk_size",
    ),
)
def _linear_attention_forward(
    q: Tensor, k: Tensor, v: Tensor, initial_state: Tensor, feat_map: FeatureMap, chunk_size: int
) -> tuple:
    """Forward pass for linear attention.

    Args:
        q: [batch_size, num_heads, seq_len, dk]
        k, v: [batch_size, num_kv_heads, seq_len, dv]
        initial_state: [batch_size, num_heads, dk, dv]
        chunk_size: int, size of each chunk

    Returns:
        o: [batch_size, num_heads, seq_len, dv]
        residuals: tuple of tensors to be used in the backward
    """
    bs, num_heads, seq_len, k_head_dim = q.shape
    _, num_kv_heads, _, v_head_dim = v.shape
    singleton_dim = 128
    # TODO: (bailin-wang) make subchunk_size configurable.
    subchunk_size = 128
    acc_dtype, orig_dtype = jnp.float32, q.dtype

    nh_per_kv_head = num_heads // num_kv_heads
    assert seq_len % chunk_size == 0 and chunk_size % subchunk_size == 0

    assert k_head_dim == singleton_dim
    assert v_head_dim == singleton_dim

    # Add two extra dims for chunk-wise computation.
    chunk_dim = seq_len // chunk_size
    subchunk_dim = chunk_size // subchunk_size

    grid = (bs, num_heads, chunk_dim)

    # q/k/v tensors are kept in bf16 and converted later to fp32 in VMEM.
    initial_state = initial_state.astype(jnp.float32)

    # None is effectively 1, but the dim will be squeezed out.
    q_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    q_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (b, h, m, 0, 0),
        block_shape=q_tiling,
    )
    kv_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    kv_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (b, lax.div(h, nh_per_kv_head), m, 0, 0), block_shape=kv_tiling
    )

    # Initial hidden states.
    is_tiling = (None, None, 2 * singleton_dim, singleton_dim)
    is_spec = pl.BlockSpec(index_map=lambda b, h, m: (b, h, 0, 0), block_shape=is_tiling)

    # Chunk-wise states (not subchunk-wise states).
    ch_tiling = (None, None, None, 2 * singleton_dim, singleton_dim)
    ch_spec = pl.BlockSpec(index_map=lambda b, h, m: (b, h, m, 0, 0), block_shape=ch_tiling)

    # Chunk-wise final states help pass states from the previous chunk to the next.
    fs_spec = is_spec

    ch_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, chunk_dim, 2 * k_head_dim, v_head_dim), dtype=acc_dtype
    )
    fs_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, 2 * k_head_dim, v_head_dim), dtype=jnp.float32
    )

    q = rearrange(q, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    k = rearrange(k, "b h (nb bl) dk -> b h nb bl dk", bl=subchunk_size)
    v = rearrange(v, "b h (nb bl) dv -> b h nb bl dv", bl=subchunk_size)

    o_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    o_spec = pl.BlockSpec(index_map=lambda b, h, m: (b, h, m, 0, 0), block_shape=o_tiling)
    o_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, chunk_dim * subchunk_dim, subchunk_size, v_head_dim),
        dtype=orig_dtype,
    )

    la_forward_kernel = partial(_linear_attention_forward_kernel, feat_map=feat_map)
    chunk_states, final_state, o = pl.pallas_call(
        la_forward_kernel,
        in_specs=(q_spec, kv_spec, kv_spec, is_spec),
        out_specs=(ch_spec, fs_spec, o_spec),
        out_shape=(ch_shape, fs_shape, o_shape),
        grid=grid,
        interpret=False,
        compiler_params=dict(
            mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))
        ),
    )(q, k, v, initial_state)

    o = rearrange(o, "b h nb bl dv -> b h (nb bl) dv")
    return o, (q, k, v, chunk_states, final_state)


def _linear_attention_backward_kernel(
    q_ref: Tensor,
    k_ref: Tensor,
    v_ref: Tensor,
    ch_ref: Tensor,
    mutable_do_ref: Tensor,
    mutable_dq_ref: Tensor,
    mutable_dk_ref: Tensor,
    mutable_dv_ref: Tensor,
    mutable_dh_carry_ref: Tensor,
    *,
    feat_map: FeatureMap,
):
    """Backward kernel for linear attention.

    Args:
        q_ref: tensor reference of shape [ns, bl, singleton_dim]
        k_ref: tensor reference of shape [ns, bl, singleton_dim]
        v_ref: tensor reference of shape [ns, bl, singleton_dim]
        ch_ref: tensor reference of shape [ns, singleton_dim, singleton_dim]

    Output via mutable tensors:
        mutable_do_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dq_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dk_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dv_ref: tensor reference of shape [ns, bl, singleton_dim]
        mutable_dh_carry_ref: tensor reference of shape [ns, singleton_dim, singleton_dim]

    Note: similar to final_state in the forward pass, dh_carry is used to pass gradients wrt.
    hidden states across different chunks. It will be initialized to zero at the last chunk.
    The final gradient wrt. hidden states will be returned as the gradient wrt. initial_state.
    """
    subchunk_dim, subchunk_size = q_ref.shape[0], q_ref.shape[1]
    causal_mask = jnp.tril(jnp.ones((subchunk_size, subchunk_size)), k=0).astype(jnp.float32)
    feat_fn = get_feature_map(feat_map)

    @pl.when(pl.program_id(axis=2) == 0)
    def init_carry():
        mutable_dh_carry_ref[:, :] = jnp.zeros_like(mutable_dh_carry_ref, dtype=jnp.float32)

    def _la_backward_dq_chunk_loop_body(t: int, h_carry_ref: Tensor):
        subchunk_idx = t
        h_block = h_carry_ref[:, :]  # final states from previous chunk
        q_block = q_ref[subchunk_idx, :].astype(jnp.float32)
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)
        q_block_ = feat_fn.fwd(q_block)
        k_block_ = feat_fn.fwd(k_block)
        do_block = mutable_do_ref[subchunk_idx, :].astype(jnp.float32)

        d_intra_att = _matmul_fp32(do_block, v_block.T) * causal_mask

        dq_block_1 = _matmul_fp32(do_block, h_block.T)
        dq_block_2 = _matmul_fp32(d_intra_att, k_block_)
        dq_block_ = dq_block_1 + dq_block_2
        dq_block_ = feat_fn.bwd(q_block_, dq_block_)
        mutable_dq_ref[subchunk_idx, :] = dq_block_.astype(mutable_dq_ref.dtype)

        next_h_block = h_block + _matmul_fp32(k_block_.T, v_block)
        h_carry_ref[:, :] = next_h_block

    def _la_backward_dkv_chunk_loop_body(t: int, carry_ref: Tensor):
        subchunk_idx = t
        (_, dh_carry_ref) = carry_ref
        dh_block = dh_carry_ref[:, :]

        q_block = q_ref[subchunk_idx, :].astype(jnp.float32)
        k_block = k_ref[subchunk_idx, :].astype(jnp.float32)
        v_block = v_ref[subchunk_idx, :].astype(jnp.float32)
        q_block_ = feat_fn.fwd(q_block)
        k_block_ = feat_fn.fwd(k_block)
        do_block = mutable_do_ref[subchunk_idx, :].astype(jnp.float32)
        causal_mask = jnp.tril(jnp.ones((subchunk_size, subchunk_size)), k=0).astype(jnp.float32)

        intra_att = _matmul_fp32(q_block_, k_block_.T)
        d_intra_att = _matmul_fp32(do_block, v_block.T) * causal_mask

        dk_block_1 = _matmul_fp32(d_intra_att.T, q_block_)
        dk_block_2 = _matmul_fp32(v_block, dh_block.T)
        dk_block_ = dk_block_1 + dk_block_2
        dk_block = feat_fn.bwd(k_block_, dk_block_)
        mutable_dk_ref[subchunk_idx, :] = dk_block.astype(mutable_dk_ref.dtype)

        dv_block_1 = _matmul_fp32((intra_att * causal_mask).T, do_block)
        dv_block_2 = _matmul_fp32(k_block_, dh_block)
        dv_block = dv_block_1 + dv_block_2
        mutable_dv_ref[subchunk_idx, :] = dv_block.astype(mutable_dv_ref.dtype)

        prev_dh_block = dh_block + _matmul_fp32(q_block_.T, do_block)
        dh_carry_ref[:, :] = prev_dh_block

    h_carry = ch_ref[:, :]
    h_tilde = for_loop.for_loop(subchunk_dim, _la_backward_dq_chunk_loop_body, h_carry)

    h_carry = h_tilde[:, :]
    dh_carry = mutable_dh_carry_ref[:, :]
    _, dinitial_state = for_loop.for_loop(
        subchunk_dim, _la_backward_dkv_chunk_loop_body, (h_carry, dh_carry), reverse=True
    )
    mutable_dh_carry_ref[:, :] = dinitial_state


def _la_backward(feat_map: FeatureMap, chunk_size: int, residuals: tuple, do: Tensor) -> tuple:
    """Backward pass for LA.

    Args:
        chunk_size: int, size of each chunk
        residuals: tuple of tensors returned from the forward pass
        do: [batch_size, num_heads, seq_len, dv]

    Returns:
        dq: [batch_size, num_heads, seq_len, dk]
        dk: [batch_size, num_kv_heads, seq_len, dk]
        dv: [batch_size, num_kv_heads, seq_len, dv]
        dinitial_state: [batch_size, num_heads, dk, dv]
    """
    q, k, v, chunk_states, final_state = residuals
    del final_state  # final_state is not used in the backward pass
    orig_dtype = v.dtype

    singleton_dim = 128
    num_heads = q.shape[1]
    bs, num_kv_heads, _, subchunk_size, _ = k.shape
    nh_per_kv_head = num_heads // num_kv_heads
    subchunk_dim = chunk_size // subchunk_size
    chunk_dim = k.shape[2] // subchunk_dim
    dk, dv = q.shape[-1], v.shape[-1]

    grid = (bs, num_heads, chunk_dim)

    q_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    q_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (
            b,
            h,
            chunk_dim - 1 - m,
            0,
            0,
        ),
        block_shape=q_tiling,
    )
    kv_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    kv_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (b, lax.div(h, nh_per_kv_head), chunk_dim - 1 - m, 0, 0),
        block_shape=kv_tiling,
    )

    ch_tiling = (None, None, None, 2 * singleton_dim, singleton_dim)
    ch_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (b, h, chunk_dim - 1 - m, 0, 0), block_shape=ch_tiling
    )

    do_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    do_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (b, h, chunk_dim - 1 - m, 0, 0), block_shape=do_tiling
    )

    dq_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    dq_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (
            b,
            h,
            chunk_dim - 1 - m,
            0,
            0,
        ),
        block_shape=dq_tiling,
    )
    dq_shape = jax.ShapeDtypeStruct(
        shape=(
            bs,
            num_heads,
            chunk_dim * subchunk_dim,
            subchunk_size,
            dk,
        ),
        dtype=orig_dtype,
    )

    dkv_tiling = (None, None, subchunk_dim, subchunk_size, singleton_dim)
    dkv_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (b, h, chunk_dim - 1 - m, 0, 0), block_shape=dkv_tiling
    )
    dkv_shape = jax.ShapeDtypeStruct(
        shape=(bs, num_heads, chunk_dim * subchunk_dim, subchunk_size, dv),
        dtype=orig_dtype,
    )

    dh_carry_tiling = (None, None, 2 * singleton_dim, singleton_dim)
    dh_carry_spec = pl.BlockSpec(
        index_map=lambda b, h, m: (b, h, 0, 0), block_shape=dh_carry_tiling
    )
    dh_carry_shape = jax.ShapeDtypeStruct(shape=(bs, num_heads, 2 * dk, dv), dtype=jnp.float32)

    do = rearrange(do, "b h (nb bl) dv -> b h nb bl dv", bl=subchunk_size)

    la_backward_kernel = partial(_linear_attention_backward_kernel, feat_map=feat_map)
    dq, dk, dv, dinitial_state = pl.pallas_call(
        la_backward_kernel,
        in_specs=(q_spec, kv_spec, kv_spec, ch_spec, do_spec),
        out_specs=(dq_spec, dkv_spec, dkv_spec, dh_carry_spec),
        out_shape=(dq_shape, dkv_shape, dkv_shape, dh_carry_shape),
        grid=grid,
        interpret=False,
        compiler_params=dict(
            mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))
        ),
    )(q, k, v, chunk_states, do)

    dq = rearrange(dq, "b nh nb bl dk -> b nh (nb bl) dk")
    dk = rearrange(dk, "b (nkvh nhp) nb bl dk -> b nkvh nhp (nb bl) dk", nhp=nh_per_kv_head)
    dk = dk.sum(axis=2)
    dv = rearrange(dv, "b (nkvh nhp) nb bl dv -> b nkvh nhp (nb bl) dv", nhp=nh_per_kv_head)
    dv = dv.sum(axis=2)

    return dq, dk, dv, dinitial_state


_linear_attention.defvjp(_linear_attention_forward, _la_backward)


@jax.named_call  # `named_call` ensures the name is used in tracing, which is useful for profiling.
@partial(
    jax.jit,
    static_argnames=(
        "window_size",
        "feat_map",
        "chunk_size",
    ),
)
def residual_linear_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    h0: Tensor,
    *,
    window_size: int,
    feat_map: FeatureMap,
    chunk_size: int,
) -> Tensor:
    """Differentiable function that computes the output of linear attention.

    Args:
        q: [batch_size, num_heads, seq_len, dk]
        k: [batch_size, num_kv_heads, seq_len, dk]
        v: [batch_size, num_kv_heads, seq_len, dv]
        h0: [batch_size, num_heads, dk, dv]
        window_size: int, size of the sliding window
        feat_map: an instance of FeatureMap
        chunk_size: int, size of each chunk

    Returns:
        output: [batch_size, num_heads, seq_len, dv]
    """

    bs, nh, _, _ = q.shape
    bs, nkvh, _, _ = v.shape
    assert nh % nkvh == 0

    # Using softmax feature map, we can shift k/v as the input, which is equivalent to
    # to shift the feature map of k/v.
    k = right_shift_and_zero_pad(k, window_size + 1, axis=2)
    v = right_shift_and_zero_pad(v, window_size + 1, axis=2)

    if h0.shape[0] == 1:
        h0 = repeat(h0, "l h dk dv -> (l b) h dk dv", b=bs)
    h0 = h0.astype(jnp.float32)

    output = _linear_attention(q, k, v, h0, feat_map, chunk_size)
    return output


def _inner_linear_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    h0: Tensor,
) -> tuple[Tensor, Tensor]:
    """Inner function for linear scan attention.
    Args:
        q: [batch_size, num_heads, seq_len, dk]
        k: [batch_size, num_kv_heads, seq_len, dk]
        v: [batch_size, num_kv_heads, seq_len, dv]
        h0: [batch_size, num_heads, dk, dv]
    Returns:
        output: [batch_size, num_heads, seq_len, dv]
        final_state: [batch_size, num_heads, dk, dv]
    """
    orig_dtype = q.dtype

    _, nh, _, _ = q.shape
    _, nkvh, _, _ = v.shape
    assert nh % nkvh == 0
    q = rearrange(q, "b (ng gs) l dk -> b ng gs l dk", ng=nkvh)
    h0 = rearrange(h0, "b (ng gs) dk dv -> b ng gs dv dk", ng=nkvh)

    def single_head_scan(q_head, k_head, v_head, h0_head):
        def scan_body_fn(h_prev, current_inputs):
            q_t, k_t, v_t = current_inputs
            h_next = h_prev + jnp.einsum("i,j->ij", v_t, k_t, preferred_element_type=jnp.float32)
            o_t = jnp.einsum("ij,j->i", h_next, q_t, preferred_element_type=jnp.float32)
            return h_next, o_t.astype(orig_dtype)

        return jax.lax.scan(scan_body_fn, h0_head, (q_head, k_head, v_head))

    # vmap to group size
    single_group_scan = jax.vmap(single_head_scan, in_axes=(0, None, None, 0), out_axes=(0, 0))

    # vmap to multi-head
    multi_head_scan = jax.vmap(single_group_scan, in_axes=(0, 0, 0, 0), out_axes=(0, 0))

    # vmap to batch_dim
    batched_scan = jax.vmap(multi_head_scan, in_axes=(0, 0, 0, 0), out_axes=(0, 0))

    final_state, output = batched_scan(q, k, v, h0)

    final_state = rearrange(final_state, "b ng gs dv dk -> b (ng gs) dk dv")
    output = rearrange(output, "b ng gs l dv -> b (ng gs) l dv")

    # Ideally, final_state should always be in float32. But inference engines (e.g., softserve)
    # usally requires the cached state to be in bfloat16, the same as the dtype for kv cache.
    if orig_dtype == jnp.bfloat16:
        final_state = final_state.astype(jnp.bfloat16)
    return output, final_state


@partial(jax.jit, static_argnames=("window_size", "feat_map", "chunk_size"))
def residual_linear_attention_linear_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    h0: Tensor,
    *,
    window_size: int,
    feat_map: FeatureMap,
    chunk_size: int,
) -> Tensor:
    """LinearScan based reference implementations.

    Args:
        q: [batch_size, num_heads, seq_len, dk]
        k: [batch_size, num_kv_heads, seq_len, dk]
        v: [batch_size, num_kv_heads, seq_len, dv]
        h0: [batch_size, num_heads, dk, dv]
        window_size: int, size of the sliding window
        feat_map: an instance of FeatureMap
        chunk_size: int, size of each chunk

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
    """
    del chunk_size
    # Apply feature function to q/k.
    feat_fn = get_feature_map(feat_map)
    q = feat_fn.fwd(q)
    k = feat_fn.fwd(k)

    k = right_shift_and_zero_pad(k, window_size + 1, axis=2)
    v = right_shift_and_zero_pad(v, window_size + 1, axis=2)
    h0 = h0.astype(jnp.float32)

    return _inner_linear_scan(q, k, v, h0)


@partial(jax.jit, static_argnames=("window_size", "feat_map", "chunk_size"))
def residual_linear_attention_w_timestep(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    h0: Tensor,
    timestep: Tensor,
    *,
    window_size: int,
    feat_map: FeatureMap,
    chunk_size: int,
) -> Tensor:
    """LinearScan that takes timestep as input and masks useless k/v based on timestep.

    This function is used during inference where decoding might start from different timesteps.

    Args:
        q: [batch_size, num_heads, seq_len, dk]
        k: [batch_size, num_kv_heads, seq_len, dk]
        v: [batch_size, num_kv_heads, seq_len, dv]
        h0: [batch_size, num_heads, dk, dv]
        timestep: [batch_size]
        window_size: int, size of the sliding window
        feat_map: an instance of FeatureMap
        chunk_size: int, size of each chunk

     Returns:
        output: [batch_size, num_heads, seq_len, dv]
        hidden_state: [batch_size, num_heads, dk, dv]
    """
    del chunk_size

    # Apply kernel function to q/k.
    feat_fn = get_feature_map(feat_map)
    q = feat_fn.fwd(q)
    k = feat_fn.fwd(k)
    l = q.shape[2]

    k = right_shift_and_zero_pad(k, window_size + 1, axis=2)
    v = right_shift_and_zero_pad(v, window_size + 1, axis=2)
    h0 = h0.astype(jnp.float32)

    # Mask out k/v based on timestep.
    timestep_mask = jnp.arange(l)[None, :] >= timestep[:, None]
    k = jnp.where(timestep_mask[:, None, :, None], 0.0, k)
    v = jnp.where(timestep_mask[:, None, :, None], 0.0, v)

    return _inner_linear_scan(q, k, v, h0)
