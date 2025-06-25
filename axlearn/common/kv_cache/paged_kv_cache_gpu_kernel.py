# Copyright © 2025 Apple Inc.
"""Kernels for efficient insertion of new k/v_proj into paged kv cache.

This kernel is a temporary workaround of occasional performance problems with
`scatter_update_pages` on GPU.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas.triton import TritonCompilerParams

from axlearn.common.utils import Tensor


def _scatter_pages_kernel(
    page_indices: Tensor,  # (pages_per_batch,)
    key_positions: Tensor,  # scalar
    kv_pages: Tensor,  # (num_pages, page_size, head_dim)
    kv_proj: Tensor,  # (head_dim,)
    out_kv_pages: Tensor,  # (num_pages, page_size, head_dim)
):
    page_size = kv_pages.shape[-2]
    pos = key_positions[...]
    cond = (0 <= pos) & (pos < page_size * page_indices.shape[0])
    pos = jnp.where(cond, pos, 0)
    page_idx, offset = jnp.divmod(pos, page_size)
    page_idx = page_indices[page_idx]

    @pl.when((page_idx != 0) & cond)
    def true_fn():
        out_kv_pages[page_idx, offset, ...] = kv_proj[...]


def gpu_scatter_update_pages_shmap_fn(
    kv_pages: Tensor,
    kv_proj: Tensor,
    page_indices: Tensor,
    key_positions: Tensor,
) -> Tensor:
    batch_size, pages_per_batch = page_indices.shape
    num_heads, num_pages, page_size, head_dim = kv_pages.shape
    kv_proj = kv_proj.squeeze(2)

    pages_spec = pl.BlockSpec((None, num_pages, page_size, head_dim), lambda i, j: (i, 0, 0, 0))
    return pl.pallas_call(
        _scatter_pages_kernel,
        in_specs=[
            pl.BlockSpec((None, pages_per_batch), lambda i, j: (j, 0)),  # page_indices
            pl.BlockSpec((None,), lambda i, j: (j,)),  # key_positions
            pages_spec,
            pl.BlockSpec((None, None, head_dim), lambda i, j: (i, j, 0)),
        ],
        out_specs=pages_spec,
        grid=(num_heads, batch_size),
        out_shape=jax.ShapeDtypeStruct(kv_pages.shape, kv_pages.dtype),
        compiler_params=TritonCompilerParams(num_warps=max(head_dim // 32, 1), num_stages=1),
        interpret=jax.default_backend() == "cpu",
        input_output_aliases={2: 0},  # Output is aliased with input `kv_pages`.
    )(page_indices, key_positions, kv_pages, kv_proj)
