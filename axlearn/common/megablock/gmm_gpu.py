# Copyright © 2025 Apple Inc.

"""Grouped matrix multiplication kernel for GPU written in Pallas.
Metadata calculation is ported from make_group_metadata function without logic change:
https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/megablox/gmm.py#L79

Calculation from a group offset and aggregate with existing results is not supported yet.

"""

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas.triton import CompilerParams as TritonCompilerParams

from axlearn.common.utils import Tensor

partial = functools.partial
GroupMetadata = Any

NUM_STAGES = 4
NUM_WARPS = 2

DEFAULT_TILING = (32, 32, 64)


def is_gpu() -> bool:
    return jax.default_backend() == "gpu"


def _assert_is_supported_dtype(dtype: jnp.dtype) -> None:
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.float16):
        raise ValueError(f"Expected bfloat16, float16 or float32 array but got {dtype}.")


def _select_input_dtype(lhs: Tensor, rhs: Tensor) -> jnp.dtype:
    """A type to which both input should be adapted to before dot product."""
    # In case of mixed input precision, we need to convert bf16 argument to fp32 beforehand.
    if is_gpu() and lhs.dtype == jnp.bfloat16 and rhs.dtype == jnp.bfloat16:
        return jnp.bfloat16
    elif is_gpu() and lhs.dtype == jnp.float16 and rhs.dtype == jnp.float16:
        return jnp.float16
    else:
        return jnp.float32


def get_gpu_dot_precision(dtype) -> jax.lax.DotAlgorithmPreset:
    """Get the suitable DotAlgorithmPreset for the given dtype."""
    if dtype == jnp.float32:
        return jax.lax.DotAlgorithmPreset.TF32_TF32_F32
    if dtype == jnp.float16:
        return jax.lax.DotAlgorithmPreset.F16_F16_F32
    if dtype == jnp.bfloat16:
        return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    raise ValueError(f"Unsupported dtype {dtype}")


def _validate_args(
    *,
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    expected_rhs_dims: int = 3,
) -> tuple[Tensor, Tensor, jnp.dtype]:
    """Validates the arguments for the gmm function."""
    # Validate 'lhs'.
    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")
    _assert_is_supported_dtype(lhs.dtype)

    # Validate 'rhs'.
    if rhs.ndim != expected_rhs_dims:
        raise ValueError(
            f"Expected {expected_rhs_dims}-tensor for 'rhs' but got" f" {rhs.ndim}-tensor."
        )
    _assert_is_supported_dtype(rhs.dtype)

    # Validate 'group_sizes'.
    if group_sizes.dtype != jnp.int32:
        raise ValueError(f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")

    return lhs, group_sizes, _select_input_dtype(lhs, rhs)


def _calculate_num_tiles(x: int, tx: int) -> int:
    tiles, rem = divmod(x, tx)
    if rem:
        raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
    return tiles


def make_group_metadata(
    *,
    group_sizes: Tensor,
    m: int,
    tm: int,
    start_group: Tensor,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
    """Create the metadata needed for grouped matmul computation.

        Groups may not align perfectly with tiles, so we adjust them by
        round down and round up to ensure full tile coverage for each group,
        in order to have all groups fit perfectly into tm-sized tiles.

        Example of how metadata got calculated:
        for tm = 4, m = 16, num_groups=4, group_sizes = jnp.array([3, 4, 3, 6])
        the compute steps as following:
        group_ends = [3, 7, 10, 16]
        group_offsets = [0, 3, 7, 10, 16] (In MOE, group_offsets can present how many tokens to be
        processed by a expert before handle
        the rest of tokens to the next expert)
        rounded_group_ends = [ 4,  8, 12, 16]
        group_starts = [ 0, 3, 7, 10]  (shifting group_ends to the left)
        rounded_group_starts = [ 0, 0, 4, 8]
        rounded_group_sizes = [4, 8, 8, 8]
        group_tiles = [1, 2, 2, 2]

        get which group owns each tile along m-dimension:
        where num_groups=4, jnp.arange(num_groups) = [0, 1, 2, 3], tiles_m=m/tm = 16/4 = 4
        group_ids = [0, 1, 1, 2, 2, 3, 3]
        each value represent the tile id along m-dimension
        m_tile_ids = [0, 0, 1, 1, 2, 2, 3]

    Args:
      group_sizes: A 1d Tensor with shape [num_groups] and jnp.int32 dtype.
      m: The number of rows in lhs.
      tm: The m-dimension tile size being used.
      start_group: The group in group sizes to start computing from. This is
        particularly useful for when rhs num_groups is sharded.
      num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
        combination with group_offset.
      visit_empty_groups: If True, do not squeeze tiles for empty groups out of
        the metadata. This is necessary for tgmm, where we at least need to zero
        the output for each group.

    Returns:
      tuple of:
        group_offsets: A 1d Tensor with shape [num_groups+1] and jnp.int32
          dtype. group_offsets[i] indicates the row at which group [i] starts in
          the lhs matrix and group_offsets[i-1] = m.
        group_ids: A 1d Tensor with shape [m_tiles + num_groups] and
          jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
          work on.
        m_tile_ids: A 1d Tensor with shape [m_tiles + num_groups] and
          jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
          will work on.
        group_tiles: A 1d Tensor with shape [num_groups] and jnp.int32
          dtype. group_tiles indicates how many tiles each group calculate with.
      num_tiles: The number of m-dimension tiles to execute.
    """
    num_groups = group_sizes.shape[0]
    end_group = start_group + num_nonzero_groups - 1

    # Calculate the offset of each group, starting at zero. This metadata is
    # similar to row offsets in a CSR matrix. The following properties hold:
    #
    # group_offsets.shape = [num_groups + 1]
    # group_offsets[0] = 0
    # group_offsets[num_groups] = m
    #
    # The row at which group 'i' starts is group_offsets[i].

    # Example A:
    # num_groups = 5, m = 8,
    # group_sizes = group_sizes = Array([2, 2, 2, 1, 1], dtype=int32)
    # group_ends = Array([2, 4, 6, 7, 8], dtype=int32)
    group_ends = jnp.cumsum(group_sizes)
    # group_offsets = Array([0, 2, 4, 6, 7, 8], dtype=int32)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    # Assign a group id to each grid index.
    #
    # If a group starts somewhere other than the start of a tile or ends somewhere
    # other than the end of a tile we need to compute that full tile. Calculate
    # the number of tiles for each group by rounding their end up to the nearest
    # 'tm' and their start down to the nearest 'tm'.

    # Example A continue:
    # for m = 8, tm = 2, group_ends = Array([2, 4, 6, 7, 8], dtype=int32)
    # rounded_group_ends = Array([2, 4, 6, 8, 8], dtype=int32)
    # group_starts = Array([0, 2, 4, 6, 7], dtype=int32)
    # rounded_group_starts = Array([0, 2, 4, 6, 6], dtype=int32)
    # rounded_group_sizes = Array([2, 2, 2, 2, 2], dtype=int32)
    # group_tiles = Array([1, 1, 1, 1, 1], dtype=int32)

    # (1) Round the group_ends up to the nearest multiple of 'tm'.
    #
    # NOTE: This does not change group_offsets[num_groups], which is m
    # (because we enforce m is divisible by tm).
    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)

    # (2) Round the group_starts down to the nearest multiple of 'tm'.
    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm

    # (3) Calculate the number of rows in each group.
    #
    # NOTE: Handle zero-sized groups as a special case. If the start for a
    # zero-sized group is not divisible by 'tm' its start will be rounded down and
    # its end will be rounded up such that its size will become 1 tile here.
    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

    # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
    #
    # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
    # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
    # initial partial tiles if it's first row does not occur in the first row of a
    # tile. The '0-th' group never has a partial tile because it always starts at
    # the 0-th row.
    #
    # If no group has a partial tile, the total number of tiles is equal to
    # 'm // tm'. If every group has a partial except the 0-th group, the total
    # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
    #
    # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
    #
    # Where tiles_m = m // tm.
    #
    # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
    # (1) and (2) so this division is exact.
    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        # Insert one tile for empty groups.
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    # Create the group ids for each grid index based on the tile counts for each
    # group.
    #
    # NOTE: This repeat(...) will pad group_ids with the final group id if
    # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
    # such that we only execute the necessary number of tiles.
    tiles_m = _calculate_num_tiles(m, tm)
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Assign an m-dimension tile id to each grid index.
    #
    # NOTE: Output tiles can only be re-visited consecutively. The following
    # procedure guarantees that m-dimension tile indices respect this.

    # (1) Calculate how many times each m-dimension tile will be visited.
    #
    # Each tile is guaranteed to be visited once by the group that owns the tile.
    # The remaining possible visits occur when a group starts inside of a tile at
    # a position other than the first row. We can calculate which m-dimension tile
    # each group starts in by floor-dividing its offset with `tm` and then count
    # tile visits with a histogram.
    #
    # To avoid double counting tile visits from the group that owns the tile,
    # filter these out by assigning their tile id to `tile_m` (one beyond the max)
    # such that they're ignored by the subsequent histogram. Also filter out any
    # group which is empty.
    #
    # Example B:
    # for group_offsets=jnp.array([0, 3, 7, 10, 16])
    # [0, 3, 7, 10] % 4 -> [0, 3, 3, 2] -> [True, False, False, False]
    partial_tile_mask = jnp.logical_or((group_offsets[:-1] % tm) == 0, group_sizes == 0)

    # Explicitly enable tiles for zero sized groups, if specified. This covers
    # zero sized groups that start on a tile-aligned row and those that do not.
    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

    # Example B continue:
    # group_offsets[:-1] // tm = Array([0, 0, 1, 2], dtype=int32)
    # partial_tile_ids = Array([4, 0, 1, 2], dtype=int32)
    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m, group_offsets[:-1] // tm)

    # Example B continue:
    # tile_visits = Array([2., 2., 2., 1.], dtype=float32)
    tile_visits = jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1

    # Create the m-dimension tile ids for each grid index based on the visit
    # counts for each tile.
    # Example B continue:
    # m_tile_ids = Array([0, 0, 1, 1, 2, 2, 3], dtype=int32)
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    # Account for sharding.
    #
    # Find the start of the groups owned by our shard and shift the group_ids and
    # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
    #
    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    # Calculate the number of tiles we need to compute for our shard.
    #
    # Remove tile visits that belong to a group not in our shard.
    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)

    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids, group_tiles), num_tiles


def _zero_uninitialized_memory(
    out: Tensor,
    *,
    start_group: Tensor,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> Tensor:
    """Zero out uninitialized memory from output."""
    group_offsets = group_metadata[0]
    group_start = group_offsets[start_group]
    group_end = group_offsets[start_group + num_nonzero_groups]
    valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
    valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
    return jnp.where(valid_mask[:, None], out, 0)


def _generate_group_mask_cond(
    m_i: int, tm: int, group_mask_start: int, group_mask_end: int
) -> Tensor:
    """Create mask condition for grouped matrix multiplication between matrix A and B.
    Args:
      m_i: tile index along m dimension.
      tm: tile size along m dimension.
      group_mask_start: the index of m dimension, where the matrix multiplication starts between A
                        and B.
      group_mask_end: the index of m dimension, where the matrix multiplication ends between A
                      and B.

    Returns:
      A Tensor with shape True/False condition.
    """
    lower_mask = (jnp.arange(tm) > group_mask_start - m_i * tm - 1) & (jnp.arange(tm) > -1)
    higher_mask = (jnp.arange(tm) < tm) & (jnp.arange(tm) < group_mask_end - m_i * tm)
    return lower_mask & higher_mask


def _get_tiling(tiling: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate and return the appropriate tiling dimensions for GPU.

    Args:
        tiling: config or default tiling

    Returns: tuple of tiling along m, k, n dimensions

    """
    tm, tk, tn = tiling
    if max(tm, tk, tn) > 64:
        tm, tk, tn = DEFAULT_TILING
    return tm, tk, tn


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
    ],
)
def gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: Optional[tuple[int, int, int]] = DEFAULT_TILING,
    group_offset: Optional[Tensor] = None,
    transpose_rhs: Optional[bool] = False,
    interpret: Optional[bool] = False,
) -> Tensor:
    """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

    Args:
      lhs: A 2d Tensor with shape [m, k].
      rhs: A 3d Tensor with shape [num_groups, k, n].
      group_sizes: A 1d Tensor with shape [num_groups] and jnp.int32 dtype.
      preferred_element_type: jnp.dtype, the element type for the output matrix.
      tiling: 3-tuple of ints. The m, k and n-dimension tile sizes, where m, k and n are the
      dimensions of lhs and rhs.
      group_offset: The group in group sizes to start computing from. This is
        particularly useful for when rhs num_groups is sharded.
      transpose_rhs: True if the rhs needs to be transposed.
      interpret: Whether or not to run the kernel in interpret mode, helpful for
        testing and debugging.

    Returns:
      A 2d Tensor with shape [m, n].
    """
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        raise NotImplementedError("Specified group_offset is not supported on GPU yet.")

    num_current_groups = rhs.shape[0]
    num_total_groups = group_sizes.shape[0]
    lhs, group_sizes, _ = _validate_args(lhs=lhs, rhs=rhs, group_sizes=group_sizes)

    if transpose_rhs:
        rhs = jnp.swapaxes(rhs, 1, 2)
    # Gather shape information. Here rhs is already swaped if transpose_rhs is True
    m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])

    tm, tk, tn = _get_tiling(tiling)
    # Note: Because we don't support irregular tiles yet, (tk, tn) should always apply to
    # (rhs.shape[1],rhs.shape[2]), where rhs is the padded input with shape [num_groups, k, n]
    # of gmm function. This maybe different from TPU gmm tiling when transpose_rhs is Ture:
    # https://github.com/jax-ml/jax/blob/49aad1b97fa937583ff21df194fae4cd50be20eb/jax/experimental
    # /pallas/ops/tpu/megablox/gmm.py#L380
    if transpose_rhs:
        tk, tn = tn, tk
    tiles_n = _calculate_num_tiles(n, tn)
    tiles_k = _calculate_num_tiles(k, tk)

    # Create the metadata we need for computation.
    group_metadata, _ = make_group_metadata(
        # pylint: disable=unbalanced-tuple-unpacking
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )
    group_offsets, group_ids, m_tile_ids, _ = group_metadata

    def matmul_kernel(
        lhs: Tensor,
        rhs: Tensor,
        group_offsets_ref: Tensor,
        group_ids_ref: Tensor,
        m_tile_ids_ref: Tensor,
        o_ref: Tensor,
    ):
        """Kernel function to compute lhs @ rhs with mask.

          We need to apply mask because different rhs tile may apply to the same lhs tile.
          for example:
          ---- m_i = 0
          row_0   --
          row_1     group 0 (row 0, 1, 2)
          row_2
          row_3   -- (row 3 is group_mask_start for group 1, and group_mask_end for group 0)
          ---- m_i = 1
          row_4     group 1 (row 3, 4, 5, 6)
          row_5
          row_6
          row_7   --
          ---- m_i = 2
          row_8     group 2 (row 7, 8, 9)
          row_9
          row_10  --
          row_11
          ---- m_i = 3
          row_12
          row_13    group 2 (row 10, 11, 12, 13, 14, 15)
          row_14
          row_15
          ---- m_i = 4
          row_16 --
          ...

        Args:
          lhs: A 2d Tensor with shape [m, k].
          rhs: A 3d Tensor with shape [num_groups, k, n].
          group_offsets_ref: A 1d Tensor with shape [num_groups] and jnp.int32 dtype.
                              eg, from above example, group_offsets_ref = [0, 3, 7, 10, 16]
          group_ids_ref: A 1d Tensor with the group id to be used for each grid calculation.
                              eg, from above example, group_ids_ref = [0, 1, 1, 2, 2, 3, 3]
          m_tile_ids_ref: A 1d Tensor with the tile id along m dimension to be used for each grid
                          calculation.
                              eg, from above example, m_tile_ids_ref = [0, 0, 1, 1, 2, 2, 3]
          o_ref: the kernel calculation result being returned as ref.

        Returns:
          o_ref: the kernel calculation result.
        """

        grid_id = pl.program_id(0)
        o = jnp.zeros([tm, tn], dtype=jnp.float32)

        # note: m_i != tile_m_i(a.k.a grid_id)
        m_i = pl.load(m_tile_ids_ref, pl.dslice(grid_id, 1))

        group_id = pl.load(group_ids_ref, pl.dslice(grid_id, 1))

        group_mask_start = pl.load(group_offsets_ref, pl.dslice(group_id, 1))
        group_mask_end = pl.load(group_offsets_ref, pl.dslice(group_id + 1, 1))

        cond = _generate_group_mask_cond(m_i, tm, group_mask_start, group_mask_end)
        group_mask = (cond)[:, None]

        tm_slice = pl.ds(m_i * tm, tm)
        tn_slice = slice(None)

        def body(k_i, o):
            tk_slice = pl.ds(k_i * tk, tk)
            # (tm, tk)
            lhs_load = pl.load(lhs, (tm_slice, tk_slice), mask=group_mask, other=0.0)
            # (tk, tn)
            rhs_load = pl.load(rhs, (group_id, tk_slice, tn_slice))
            rhs_load_2d = rhs_load.squeeze(0)
            o += pl.dot(
                lhs_load, rhs_load_2d, precision=get_gpu_dot_precision(preferred_element_type)
            )
            return o

        o = lax.fori_loop(0, tiles_k, body, o)

        # Results from multiple grids could write to the same slice,
        # but no racing condition is expected with mask
        pl.store(o_ref, (tm_slice, tn_slice), o.astype(o_ref.dtype), mask=group_mask)

    def call_gmm(lhs: Tensor, rhs: Tensor, g_offsets: Tensor, g_ids: Tensor, m_t_ids: Tensor):
        return pl.pallas_call(
            matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
            in_specs=[
                pl.BlockSpec((m, k), lambda tile_m_i, n_i: (0, 0)),
                pl.BlockSpec((len(group_offsets) - 1, k, tn), lambda tile_m_i, n_i: (0, 0, n_i)),
                pl.BlockSpec((len(g_offsets),), lambda tile_m_i, n_i: (0)),
                pl.BlockSpec((len(g_ids),), lambda tile_m_i, n_i: (0)),
                pl.BlockSpec((len(m_t_ids),), lambda tile_m_i, n_i: (0)),
            ],
            out_specs=pl.BlockSpec((m, tn), lambda tile_m_i, n_i: (0, n_i)),
            grid=(len(group_ids), tiles_n),
            interpret=interpret,
            compiler_params=TritonCompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        )(lhs, rhs, g_offsets, g_ids, m_t_ids)

    out = call_gmm(
        lhs, rhs, jnp.asarray(group_offsets), jnp.asarray(group_ids), jnp.asarray(m_tile_ids)
    )

    if num_current_groups < num_total_groups:
        out = _zero_uninitialized_memory(
            out,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            group_metadata=group_metadata,
        )
    return out


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "num_actual_groups",
        "interpret",
    ],
)
def tgmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: Optional[tuple[int, int, int]] = DEFAULT_TILING,
    group_offset: Optional[Tensor] = None,
    num_actual_groups: Optional[int] = None,
    interpret: bool = False,
) -> Tensor:
    """Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :].

    Args:
      lhs: A 2d Tensor with shape [k, m].
      rhs: A 2d Tensor with shape [m, n].
      group_sizes: A 1d Tensor with shape [num_groups] and jnp.int32 dtype.
      preferred_element_type: jnp.dtype, the element type for the output matrix.
      tiling: 3-tuple of ints. The m, k and n-dimension tile sizes.
      group_offset: The group in group sizes to start computing from. This is
        particularly useful for when rhs num_groups is sharded.
      num_actual_groups: For when num_groups is sharded and we should only compute
        the groups that are local, starting from group_offset.
      interpret: Whether or not to run the kernel in interpret mode, helpful for
        testing and debugging.

    Returns:
      A  3d Tensor with shape [num_groups, k, n].
    """
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        raise NotImplementedError("Specified group_offset is not supported on GPU yet.")

    # Gather shape information.
    k, m, n = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
    num_groups = group_sizes.shape[0]
    num_actual_groups = num_actual_groups if num_actual_groups is not None else num_groups

    tm, tk, tn = _get_tiling(tiling)
    tiles_k = _calculate_num_tiles(k, tk)
    tiles_n = _calculate_num_tiles(n, tn)

    # Create the metadata we need for computation.
    group_metadata, _ = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=num_actual_groups,
        visit_empty_groups=True,
    )

    # - The values in m_tile_id represent the m dim tile id been selected for each group,
    # one tile m could be used by multiple groups, values are sorted by tile id
    # - The values in group_ids represent the group id to be applied for each tile m,
    # sorted by group id.
    # - The values in group_changing_index maps to the index of group_ids and m_tile_id, which
    # represent when the group id switch to a new one in group_ids
    # eg:
    # m_tile_ids = [0, 0, 1, 1, 2, 2, 3]
    # group_ids = [0, 1, 1, 2, 2, 3, 3]
    # group_changing_index = [0, 1, 3, 5, 7]
    group_offsets, _, m_tile_ids, group_tiles = group_metadata
    group_changing_index = jnp.concatenate(
        [jnp.cumsum(group_tiles) - group_tiles, jnp.cumsum(group_tiles)[-1:]]
    )

    def kernel(
        lhs: Tensor,
        rhs: Tensor,
        group_offsets_ref: Tensor,
        m_tile_ids_ref: Tensor,
        group_changing_index_ref: Tensor,
        o_ref: Tensor,
    ):
        """Kernel function to compute the lhs @ rhs with mask.

        Args:
          lhs: A 2d Tensor with shape [k, m].
          rhs: A 2d Tensor with shape [m, n].
          group_offsets_ref: A 1d Tensor with shape [num_groups] and jnp.int32 dtype.
                              eg, from above example, group_offsets_ref = [0, 3, 7, 10, 16]
          m_tile_ids_ref: A 1d Tensor with the tile id along m dimension to be used for each grid
                          calculation.
                          eg, from above example, m_tile_ids_ref = [0, 0, 1, 1, 2, 2, 3]
          group_changing_index_ref: A 1d Tensor holding the index of m_tile_ids_ref when the group
                                    id got switched.
                                    eg, from above example, group_ids_ref = [0, 1, 3, 5, 7]
          o_ref: the kernel calculation result being returned as ref
        Returns:
          o_ref: the kernel calculation result.
        """
        group_id = pl.program_id(0)

        # Get the tiles index range along m dim to be used for gmm calculation
        group_start_idx = pl.load(group_changing_index_ref, pl.dslice(group_id, 1))
        group_end_idx = pl.load(group_changing_index_ref, pl.dslice(group_id + 1, 1))

        # group offset for corresponding tiles along the m dim
        group_mask_start = pl.load(group_offsets_ref, pl.dslice(group_id, 1))
        group_mask_end = pl.load(group_offsets_ref, pl.dslice(group_id + 1, 1))

        tk_slice = pl.ds(0, tk)
        tn_slice = pl.ds(0, tn)

        o = jnp.zeros([tk, tn], dtype=jnp.float32)

        # Each loop calculate (tk, tm) @ (tm, tn)
        def loop_tiles_m(i, o):
            m_i = pl.load(m_tile_ids_ref, pl.dslice(i, 1))

            # Same condition being used in gmm forward pass:
            cond = _generate_group_mask_cond(m_i, tm, group_mask_start, group_mask_end)
            # mask for out grad, which is in (tm, tn) shape and share the same mask as the gmm
            # forward pass lhs inputs, apply to rows
            rhs_mask = cond[:, None]
            # mask for transposed lhs, which is in (tk, tm) shape, apply to columns
            lhs_mask = cond[None, :]

            tm_slice = pl.ds(m_i * tm, tm)

            # (tk, tm)
            lhs_load = pl.load(lhs, (tk_slice, tm_slice), mask=lhs_mask, other=0.0)
            # (tm, tn)
            rhs_load = pl.load(rhs, (tm_slice, tn_slice), mask=rhs_mask, other=0.0)

            o += pl.dot(lhs_load, rhs_load, precision=get_gpu_dot_precision(preferred_element_type))

            return o

        # Accumulate the gradient from different tiles m for the same group.
        # The unrelated tiles along m dimension are filtered out by group_start_idx and
        # group_end_id, in order to save the compute resource.
        o = lax.fori_loop(jnp.sum(group_start_idx), jnp.sum(group_end_idx), loop_tiles_m, o)

        o = o[None, :]
        pl.store(o_ref, (pl.ds(group_id, 1), slice(None), slice(None)), o.astype(o_ref.dtype))

    def call_tgmm(lhs: Tensor, rhs: Tensor, g_offsets, m_t_ids, g_changing_idx):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((num_actual_groups, k, n), preferred_element_type),
            in_specs=[
                pl.BlockSpec((tk, m), lambda _, k_i, n_i: (k_i, 0)),
                pl.BlockSpec((m, tn), lambda _, k_i, n_i: (0, n_i)),
                pl.BlockSpec((len(g_offsets),), lambda _, k_i, n_i: (0)),
                pl.BlockSpec((len(m_t_ids),), lambda _, k_i, n_i: (0)),
                pl.BlockSpec((len(g_changing_idx),), lambda _, k_i, n_i: (0)),
            ],
            out_specs=pl.BlockSpec(
                (num_actual_groups, tk, tn), lambda g_idx, k_i, n_i: (0, k_i, n_i)
            ),
            grid=(num_actual_groups, tiles_k, tiles_n),
            interpret=interpret,
            compiler_params=TritonCompilerParams(num_warps=NUM_WARPS, num_stages=NUM_STAGES),
        )(lhs, rhs, g_offsets, m_t_ids, g_changing_idx)

    out = call_tgmm(
        lhs,
        rhs,
        jnp.asarray(group_offsets),
        jnp.asarray(m_tile_ids),
        jnp.asarray(group_changing_index),
    )

    return out
