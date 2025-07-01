# Copyright © 2025 Apple Inc.
"""Kernels for efficient insertion of new k/v_proj into paged kv cache.

This kernel is a temporary workaround of significant performance problems with
`scatter_update_pages` on TPU. This kernel is quite slow: it has 10x latency compared to k/v
projection at the same batch size. However, it's faster than the non paged kv cache update. It's
sufficiently fast to unblock us.

TODO(hanzhi-zhou): Optimize this kernel.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from axlearn.common.utils import Tensor


def _scatter_pages_kernel(
    page_indices: Tensor,
    key_positions: Tensor,
    kv_pages: Tensor,  # (min(8, page_size), head_dim)
    kv_proj: Tensor,  # (1, head_dim)
    out_kv_pages: Tensor,  # (min(8, page_size), head_dim)
    *,
    page_size: int,
):
    batch = pl.program_id(1)
    pos = key_positions[batch]
    cond = (0 <= pos) & (pos < page_size * page_indices.shape[1])
    pos = jnp.where(cond, pos, 0)
    page_idx, offset = jnp.divmod(pos, page_size)

    cond = (page_indices[batch, page_idx] != 0) & cond

    @pl.when(cond)
    def compute():
        offset_in_page = offset % 8
        # Note: The following code is equivalent to
        # out_kv_pages[offset_in_page, ...] = kv_proj[...]
        # However, Pallas TPU only supports operating on tiles that are multiple of (8, 128), so it
        # will give the following error: "Mosaic failed to compile TPU kernel: cannot statically
        # prove that index in dimension 2 is a multiple of 8". We workaround this by applying
        # masked updates that are a multiple of (8, 128).
        # There seems to be a bug related in-kernel data repeat:
        # data = pltpu.repeat(kv_proj[...], out_kv_pages.shape[0], 0)
        data = kv_proj[...]
        mask = jax.lax.broadcasted_iota(jnp.int32, out_kv_pages.shape, 0) == offset_in_page
        # This where statement will fail: "INTERNAL: Mosaic failed to compile TPU kernel:
        # Not implemented: mask relayout with non-32 bitwidth in vector layout."
        # out_kv_pages[...] = jnp.where(mask, data, kv_pages[...])
        out_kv_pages[...] = data * mask + kv_pages[...] * (1 - mask)

    @pl.when(~cond)
    def pass_through():
        out_kv_pages[...] = kv_pages[...]


def tpu_scatter_update_pages_shmap_fn(
    kv_pages: Tensor,
    kv_proj: Tensor,
    page_indices: Tensor,
    key_positions: Tensor,
) -> Tensor:
    batch_size = page_indices.shape[0]
    num_heads, _, page_size, head_dim = kv_pages.shape
    page_block_size = min(8, page_size)
    kv_proj = jnp.tile(kv_proj, (1, 1, page_block_size, 1))

    def block_idx_map(i, batch, page_indices, key_positions):
        pos = key_positions[batch]
        pos = jnp.where((0 <= pos) & (pos < page_size * page_indices.shape[1]), pos, 0)
        page_idx, offset_in_page = jnp.divmod(pos, page_size)
        # If pos is out-of-bound, we'll fetch the first page for that batch, but we won't update it
        # in the kernel.
        return (i, page_indices[batch, page_idx], offset_in_page // 8, 0)

    if page_size > 8:
        assert page_size % 8 == 0
    # Note: The minimum tile size for the last two dim is the smaller of the last two dim and
    # (8, 128). Since we're only updating one entry in each page, we load the minimum size for
    # each block.
    pages_spec = pl.BlockSpec((None, None, page_block_size, head_dim), block_idx_map)
    return pl.pallas_call(
        partial(_scatter_pages_kernel, page_size=page_size),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pages_spec,
                pl.BlockSpec(
                    (None, None, page_block_size, head_dim), lambda i, j, p, o: (i, j, 0, 0)
                ),
            ],
            out_specs=pages_spec,
            grid=(num_heads, batch_size),
        ),
        out_shape=jax.ShapeDtypeStruct(kv_pages.shape, kv_pages.dtype),
        compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=jax.default_backend() == "cpu",
        input_output_aliases={2: 0},  # Output is aliased with input `kv_pages`.
    )(page_indices, key_positions, kv_pages, kv_proj)
