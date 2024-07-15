# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-deepmind/recurrentgemma
# Copyright 2024 The recurrentgemma authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Pallas kernels for use with Mamba models."""
import functools
from typing import NamedTuple, Tuple

import jax
from jax import numpy as jnp
from jax._src.lax.control_flow import for_loop
from jax.experimental import pallas as pl

from axlearn.common.utils import Tensor

# MATMUL_PREC can be set to jax.lax.Precision("float32") for greater accuracy.
MATMUL_PREC = None


class MambaArgumentBlockSpecs(NamedTuple):
    """A NamedTuple for storing the pl.BlockSpecs associated with arguments to the Mamba
    Pallas kernels."""

    # A BlockSpec for arguments with shape [batch_size, seq_len, inner_dim].
    x_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [state_dim, inner_dim].
    a_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [batch_size, seq_len, state_dim].
    b_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [1, inner_dim].
    d_spec: pl.BlockSpec
    # A BlockSpec for arguments with shape [batch_size, state_dim, inner_dim].
    carry_spec: pl.BlockSpec


def _parameter_blockspecs(
    state_dim: int, seq_tile_size: int, dim_tile_size: int
) -> MambaArgumentBlockSpecs:
    """Returns `pl.BlockSpec`s for the parameters of the forward and backward Mamba kernels.

    Args:
        state_dim: The Mamba state_dim.
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        An instance of MambaArgumentBlockSpecs.
    """

    # Note: `None` is equivalent to 1, but the dimension is squeezed away in the kernel.
    x_tiling = (None, seq_tile_size, dim_tile_size)
    x_spec = pl.BlockSpec(lambda b, d, s: (b, s, d), x_tiling)

    # Note: we do not tile over state_dim.
    a_tiling = (state_dim, dim_tile_size)
    a_spec = pl.BlockSpec(lambda b, d, s: (0, d), a_tiling)

    b_tiling = (None, seq_tile_size, state_dim)
    b_spec = pl.BlockSpec(lambda b, d, s: (b, s, 0), b_tiling)

    d_tiling = (1, dim_tile_size)
    d_spec = pl.BlockSpec(lambda b, d, s: (0, d), d_tiling)

    carry_tiling = (None, state_dim, dim_tile_size)
    carry_spec = pl.BlockSpec(lambda b, d, s: (b, 0, 0), carry_tiling)

    return MambaArgumentBlockSpecs(
        x_spec=x_spec,
        a_spec=a_spec,
        b_spec=b_spec,
        d_spec=d_spec,
        carry_spec=carry_spec,
    )


def _in_kernel_dtype() -> jnp.dtype:
    """Returns the dtype to use within kernel computations."""
    return jnp.float32


# pylint: disable=invalid-name


def _update_carry(
    t: int,
    mutable_carry_ref: Tensor,
    *,
    dtype: jnp.dtype,
    x_ref: Tensor,
    a_ref: Tensor,
    b_ref: Tensor,
    delta_ref: Tensor,
):
    """Computes h_{t+1} = a_bar_t * h_t + b_bar_t * x_t, where:
            a_bar_t = exp(delta_t * a), b_bar_t = delta_t * b_t.

    Args:
        t: the current time-step.
        mutable_carry_ref: A [state_dim, dim_tile_size] Tensor reference to h_t,
            modified by the function.
        x_ref: A [seq_len, dim_tile_size] Tensor reference.
        a_ref: A [state_dim, dim_tile_size] Tensor reference.
        b_ref: A [seq_len, state_dim] Tensor reference.
        delta_ref: A [seq_len, dim_tile_size] Tensor reference.
    """
    delta = jnp.expand_dims(delta_ref[t].astype(dtype), axis=0)  # [1, inner_dim]
    a = a_ref[:].astype(dtype)  # [state_dim, inner_dim]
    a_bar = jnp.exp(delta * a)  # [state_sim, inner_dim]
    b = jnp.expand_dims(b_ref[t].astype(dtype), axis=-1)  # [state_dim, 1]
    b_bar = delta * b * jnp.expand_dims(x_ref[t].astype(dtype), axis=0)
    # carry_ref holds h_t.
    a_bar *= mutable_carry_ref[:]
    # Put h_{t+1} in the a_bar register.
    a_bar += b_bar
    # carry_ref now holds h_{t+1}.
    mutable_carry_ref[:] = a_bar


def _forward_kernel_boundary_hs(
    x_ref: Tensor,
    a_ref: Tensor,
    b_ref: Tensor,
    delta_ref: Tensor,
    mutable_carry_ref: Tensor,
    mutable_boundary_hs_ref: Tensor,
):
    """Computes the Mamba forward states, but not outputs.

    Args:
        x_ref: A [seq_len, dim_tile_size] Tensor reference.
        a_ref: A [state_dim, dim_tile_size] Tensor reference.
        b_ref: A [seq_len, state_dim] Tensor reference.
        delta_ref: A [seq_len, dim_tile_size] Tensor reference.
        mutable_carry_ref: A [state_dim, dim_tile_size] Tensor reference to h_t,
            modified by the function.
        mutable_boundary_hs_ref: A [state_dim, dim_tile_size] Tensor reference,
            modified by the function.
    """
    dtype = _in_kernel_dtype()
    seq_len = x_ref.shape[0]

    # Zero h_carry_ref, which holds h_t, only when in the first tile along the sequence dimension.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_carry():
        mutable_carry_ref[:] = jnp.zeros_like(mutable_carry_ref)

    h_carry = mutable_carry_ref[:]
    h_T = for_loop.for_loop(  # fills h_ref
        seq_len,
        functools.partial(
            _update_carry,
            dtype=dtype,
            x_ref=x_ref,
            a_ref=a_ref,
            b_ref=b_ref,
            delta_ref=delta_ref,
        ),
        h_carry,
    )
    # Store carry as a reference.
    mutable_carry_ref[:] = h_T
    # Store final state in boundary_hs_ref, so we can use it later for gradients.
    mutable_boundary_hs_ref[:] = h_T


def _backward_kernel(
    dy_ref: Tensor,
    x_ref: Tensor,
    a_ref: Tensor,
    b_ref: Tensor,
    c_ref: Tensor,
    delta_ref: Tensor,
    d_ref: Tensor,
    boundary_hs_ref: Tensor,
    mutable_dx_ref: Tensor,
    mutable_da_ref: Tensor,
    mutable_db_ref: Tensor,
    mutable_dc_ref: Tensor,
    mutable_ddelta_ref: Tensor,
    mutable_dd_ref: Tensor,
    mutable_dcarry_ref: Tensor,
    mutable_forward_hs_ref: Tensor,
):
    """Computes the Mamba backward pass.

    Args:
        dy_ref: A [seq_len, dim_tile_size] Tensor reference.
        x_ref: A [seq_len, dim_tile_size] Tensor reference.
        a_ref: A [state_dim, dim_tile_size] Tensor reference.
        b_ref: A [seq_len, state_dim] Tensor reference.
        c_ref: A [seq_len, state_dim] Tensor reference.
        delta_ref: A [seq_len, dim_tile_size] Tensor reference.
        d_ref: A [1, dim_tile_size] Tensor reference.
        boundary_hs_ref: a [state_dim, dim_tile_size] Tensor reference.
        mutable_dx_ref: A [seq_len, dim_tile_size] Tensor reference, modified by the function.
        mutable_da_ref: A [state_dim, dim_tile_size] Tensor reference, modified by the function.
        mutable_db_ref: A [seq_len, state_dim] Tensor reference, modified by the function.
        mutable_dc_ref: A [seq_len, state_dim] Tensor reference, modified by the function.
        mutable_ddelta_ref: A [seq_len, dim_tile_size] Tensor reference, modified by the function.
        mutable_dd_ref: A [1, dim_tile_size] Tensor reference, modified by the function.
        mutable_dcarry_ref: A [state_dim, dim_tile_size] Tensor reference, modified by the function.
        mutable_forward_hs_ref: A [seq_len, state_dim, dim_tile_size] Tensor reference,
            modified by the function.
    """
    seq_len = dy_ref.shape[0]
    dtype = _in_kernel_dtype()

    # If it's the final backward block, which corresponds to the first forward block,
    # then the boundary state is 0.
    @pl.when(pl.program_id(axis=2) == (pl.num_programs(axis=2) - 1))
    def zero_boundary():
        boundary_hs_ref[:] = jnp.zeros_like(boundary_hs_ref)

    def _forward_kernel_save_hs_body(t: int, mutable_carry_ref: Tensor):
        """Computes h_{t+1} = a_bar_t * h_t + b_bar_t * x_t and saves it in
        `mutable_forward_hs_ref`."""
        _update_carry(
            t,
            mutable_carry_ref,
            dtype=dtype,
            x_ref=x_ref,
            a_ref=a_ref,
            b_ref=b_ref,
            delta_ref=delta_ref,
        )
        mutable_forward_hs_ref[t] = mutable_carry_ref[:]

    # Compute all the forward states in the block and store in forward_hs_ref.
    # TODO(swiseman): investigate computing dc as part of this loop (h/t bailin-wang);
    # currently we run into scoped vmem errors.
    h_carry = boundary_hs_ref[:]

    _ = for_loop.for_loop(
        seq_len,
        _forward_kernel_save_hs_body,
        h_carry,
    )

    a = a_ref[:].astype(dtype)
    d = jnp.squeeze(d_ref[:], axis=0).astype(dtype)
    ones = jnp.ones((a_ref.shape[1], 1))[:]

    def _backward_kernel_body(i: int, mutable_carry_ref: Tensor):
        """Computes gradient contributions wrt parameters from time-step t, and
        partially computes d/dh_{t-1}."""
        t = seq_len - 1 - i
        dh_t = mutable_carry_ref[:].astype(dtype)
        delta = jnp.expand_dims(delta_ref[t].astype(dtype), axis=0)  # [1, inner]
        a_bar = jnp.exp(delta * a)
        dL_da_bar = dh_t * mutable_forward_hs_ref[t - 1].astype(dtype) * a_bar  # [state, inner]
        mutable_da_ref[:] += dL_da_bar * delta

        x = jnp.expand_dims(x_ref[t].astype(dtype), axis=0)  # [1, inner]
        b = jnp.expand_dims(b_ref[t].astype(dtype), axis=-1)  # [state, 1]
        ddelta = jnp.sum(dL_da_bar * a, axis=0) + jnp.sum(dh_t * b * x, axis=0)
        mutable_ddelta_ref[t] = ddelta.astype(mutable_ddelta_ref.dtype)

        dh_t_delta = dh_t * delta
        # Use an all-ones vector to sum dh_t_delta * x over its final dimension, since
        # Pallas won't let us sum over axis=1.
        mutable_db_ref[t] = jax.lax.dot_general(
            (dh_t_delta * x),
            ones,
            (
                ((1,), (0,)),
                ((), ()),
            ),
            preferred_element_type=jnp.float32,
            precision=MATMUL_PREC,
        )
        dy = dy_ref[t]
        mutable_dd_ref[:] += x.astype(dtype) * dy
        mutable_dx_ref[t] = (jnp.sum(dh_t_delta * b, axis=0) + dy * d).astype(mutable_dx_ref.dtype)
        mutable_dc_ref[t] = jax.lax.dot_general(  # equivalent to h_t @ dy: [state, 1]
            mutable_forward_hs_ref[t],  # [state, inner]
            jnp.expand_dims(dy, axis=-1),  # [inner, 1]
            (
                ((1,), (0,)),
                ((), ()),
            ),
            preferred_element_type=jnp.float32,
            precision=MATMUL_PREC,
        )
        # Compute d/dh_{t-1} and store it.
        dh_prev = jnp.expand_dims(dy_ref[t - 1], axis=0) * jnp.expand_dims(c_ref[t - 1], axis=-1)
        dh_prev += dh_t * a_bar
        mutable_carry_ref[:] = dh_prev

    # Initialize da and dd to zero if this is the first backward tile.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_da_dd():
        mutable_da_ref[:] = jnp.zeros_like(mutable_da_ref)
        mutable_dd_ref[:] = jnp.zeros_like(mutable_dd_ref)

    # Finish computing d/dh_T. If it's the final timestep, dcarry_ref gets 0s. Otherwise it contains
    # d/dh_T as computed in the subsequent tile, minus the contribution from the current step.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_dcarry():
        mutable_dcarry_ref[:] = jnp.zeros_like(mutable_dcarry_ref)

    dcarry = mutable_dcarry_ref[:] + (
        jnp.expand_dims(dy_ref[seq_len - 1].astype(dtype), axis=0)
        * jnp.expand_dims(c_ref[seq_len - 1].astype(dtype), axis=-1)
    )

    # Compute param grads for steps T thru 2 and d/dh_t for steps T-1 thru 1.
    dh_1 = for_loop.for_loop(
        seq_len - 1,
        _backward_kernel_body,
        dcarry,
    )
    # Compute param grads for step 1. Note: difficult to refactor this without adding
    # control-flow, which may slow things down.
    delta1 = jnp.expand_dims(delta_ref[0].astype(dtype), axis=0)
    a_bar1 = jnp.exp(delta1 * a)
    dL_da1_bar = dh_1 * boundary_hs_ref[:].astype(dtype) * a_bar1
    mutable_da_ref[:] += dL_da1_bar * delta1

    x1 = jnp.expand_dims(x_ref[0].astype(dtype), axis=0)
    b1 = jnp.expand_dims(b_ref[0].astype(dtype), axis=-1)
    mutable_ddelta_ref[0] = (
        jnp.sum(dL_da1_bar * a, axis=0) + jnp.sum(dh_1 * b1 * x1, axis=0)
    ).astype(mutable_ddelta_ref.dtype)
    # Compute d/db1 and d/dx1
    dh_1_delta = dh_1 * delta1
    mutable_db_ref[0] = jax.lax.dot_general(
        (dh_1_delta * x1),
        ones,
        (
            ((1,), (0,)),
            ((), ()),
        ),
        preferred_element_type=jnp.float32,
        precision=MATMUL_PREC,
    )
    dy1 = dy_ref[0]
    mutable_dd_ref[:] += dy1 * x1.astype(dtype)
    mutable_dx_ref[0] = (jnp.sum(dh_1_delta * b1, axis=0) + dy1 * d).astype(mutable_dx_ref.dtype)
    mutable_dc_ref[0] = jax.lax.dot_general(
        mutable_forward_hs_ref[0],
        jnp.expand_dims(dy1, axis=-1),
        (
            ((1,), (0,)),
            ((), ()),
        ),
        preferred_element_type=jnp.float32,
        precision=MATMUL_PREC,
    )
    # Compute part of d/d_h0 (i.e., grad wrt final step of previous tile). If necessary,
    # the contribution from previous timestep will be added above.
    dh_0 = dh_1 * a_bar1
    mutable_dcarry_ref[:] = dh_0


def _loop_backward_pallas(
    dy: Tensor,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    d: Tensor,
    seq_tile_size: int,
    dim_tile_size: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute the Mamba backward pass using a Pallas kernel.

    Args:
        dy: [batch_size, seq_len, inner_dim]
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seq_len, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        A 6-tuple of gradients wrt x, a, b, c, delta, and d (resp.) each having the same shape
        as the original argument.
    """
    # Calculating each time-step's contribution to the Mamba parameter gradients requires
    # access to forward states from both the current and previous time-steps. We want to avoid
    # saving all forward states in HBM, and so we use the following strategy:
    # - Recompute all forward states before the backward pass in order to obtain "boundary" states.
    #     - "Boundary" states correspond to the last time-step of a block.
    # - Save boundary states to HBM.
    # - Pass the boundary state from the i-th block to the i+1-th block, so each block has the state
    #   from before its first time-step.
    # - Recompute the states within a block from the boundary state, and update gradients.
    #
    # A faster and simpler solution would involve simply reversing the Mamba forward pass, from
    # the last time-step. But this seems to be difficult to accomplish in a numerically stable way.

    state_dim = b.shape[2]
    x_dtype = x.dtype
    # All parameters except `a` need to be in float32 in order for Pallas not to complain.
    dy, x, b, c, delta, d = [arg.astype(jnp.float32) for arg in [dy, x, b, c, delta, d]]
    batch_tile_size = 1
    x_spec, a_spec, b_spec, d_spec, carry_spec = _parameter_blockspecs(
        state_dim,
        seq_tile_size,
        dim_tile_size,
    )
    hcarry_shape = (x.shape[0], a.shape[0], dim_tile_size)
    carry_dtype = _in_kernel_dtype()
    # Create a batch_size x inner_dim x seq_len grid, which allows us to carry state
    # between sequence blocks.
    grid = (
        x.shape[0] // batch_tile_size,
        x.shape[2] // dim_tile_size,
        x.shape[1] // seq_tile_size,
    )
    # Prepare to save the seq_len / seq_tile_size boundary states in HBM.
    seq_grid_size = x.shape[1] // seq_tile_size
    boundary_hs_shape = (
        x.shape[0],
        seq_grid_size + 1,
    ) + a.shape  # [batch_size, seq_len / seq_tile_size + 1, state_dim, inner_dim]
    boundary_hs_tiling = (None, None, state_dim, dim_tile_size)
    # In the forward pass, i-th block writes to i+1-th block for use in the backward pass.
    bdry_spec = pl.BlockSpec(lambda b, d, s: (b, s + 1, 0, d), boundary_hs_tiling)
    bw_bdry_spec = pl.BlockSpec(
        lambda b, d, s: (b, seq_grid_size - 1 - s, 0, d), boundary_hs_tiling
    )
    # Write boundary states to hbm.
    _, boundary_hs = pl.pallas_call(
        _forward_kernel_boundary_hs,
        grid=grid,
        in_specs=[x_spec, a_spec, b_spec, x_spec],
        out_shape=[
            jax.ShapeDtypeStruct(hcarry_shape, carry_dtype),
            jax.ShapeDtypeStruct(boundary_hs_shape, carry_dtype),
        ],
        out_specs=[carry_spec, bdry_spec],
        compiler_params=dict(
            mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))
        ),
    )(x, a, b, delta)

    # Create BlockSpecs for gradient tensors with different shapes from their forward
    # counterparts.
    # We will not have contiguous writes along the batch_size axis, so we cannot
    # accumulate da or dd gradients over examples. We therefore add a batch dimension.
    da_shape = (x.shape[0],) + a.shape
    da_tiling = (None, state_dim, dim_tile_size)
    da_spec = pl.BlockSpec(lambda b, d, s: (b, 0, d), da_tiling)

    dd_shape = (x.shape[0],) + d.shape
    dd_tiling = (None, 1, dim_tile_size)
    dd_spec = pl.BlockSpec(lambda b, d, s: (b, 0, d), dd_tiling)

    # Similarly, db and dc gradients cannot accumulate over the inner_dim blocks,
    # since they are not accessed contiguously. We therefore add a seq_tile_size
    # dimension.
    db_shape = (grid[1],) + b.shape + (1,)
    db_tiling = (None, None, seq_tile_size, state_dim, 1)
    db_spec = pl.BlockSpec(lambda b, d, s: (d, b, s, 0, 0), db_tiling)

    # Reverse BlockSpecs along the sequence axis, since gradients are computed from
    # last time-step to first.
    def _reversed_blockspec(spec):
        return pl.BlockSpec(
            lambda b, d, s: spec.index_map(b, d, seq_grid_size - 1 - s), spec.block_shape
        )

    bw_x_spec = _reversed_blockspec(x_spec)
    bw_b_spec = _reversed_blockspec(b_spec)
    db_spec = _reversed_blockspec(db_spec)

    # Allocate scratch space to recompute forward states within a block. Ideally
    # this could be done directly in VMEM, but it does not appear to be possible.
    forward_hs_shape = (batch_tile_size, seq_tile_size, state_dim, dim_tile_size)
    forward_hs_tiling = (None, seq_tile_size, state_dim, dim_tile_size)
    forward_hs_spec = pl.BlockSpec(lambda b, d, s: (0, 0, 0, 0), forward_hs_tiling)

    dx, da, db, dc, ddelta, dd, _, _ = pl.pallas_call(
        _backward_kernel,
        grid=grid,
        in_specs=[
            bw_x_spec,
            bw_x_spec,
            a_spec,
            bw_b_spec,
            bw_b_spec,
            bw_x_spec,
            d_spec,
            bw_bdry_spec,
        ],
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, carry_dtype),
            jax.ShapeDtypeStruct(da_shape, carry_dtype),
            jax.ShapeDtypeStruct(db_shape, carry_dtype),
            jax.ShapeDtypeStruct(db_shape, carry_dtype),
            jax.ShapeDtypeStruct(delta.shape, carry_dtype),
            jax.ShapeDtypeStruct(dd_shape, carry_dtype),
            jax.ShapeDtypeStruct(hcarry_shape, carry_dtype),
            jax.ShapeDtypeStruct(forward_hs_shape, carry_dtype),
        ],
        out_specs=[
            bw_x_spec,
            da_spec,
            db_spec,
            db_spec,
            bw_x_spec,
            dd_spec,
            carry_spec,
            forward_hs_spec,
        ],
        compiler_params=dict(
            mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary")),
        ),
    )(
        dy,
        x,
        a,
        b,
        c,
        delta,
        d,
        boundary_hs,
    )
    dx = dx.astype(x_dtype)
    # Accumulate gradients over any additional dimensions we've added,
    # and cast to the right type.
    da = jnp.sum(da, axis=0).astype(x_dtype)
    db = jnp.squeeze(jnp.sum(db, axis=0), axis=-1).astype(x_dtype)
    dc = jnp.squeeze(jnp.sum(dc, axis=0), axis=-1).astype(x_dtype)
    ddelta = ddelta.astype(x_dtype)
    dd = jnp.sum(dd, axis=0).astype(x_dtype)
    return dx, da, db, dc, ddelta, dd


def _forward_kernel(
    x_ref: Tensor,
    a_ref: Tensor,
    b_ref: Tensor,
    c_ref: Tensor,
    delta_ref: Tensor,
    d_ref: Tensor,
    mutable_y_ref: Tensor,
    mutable_carry_ref: Tensor,
):
    """Computes the Mamba forward pass.

    Args:
        x_ref: A [seq_len, dim_tile_size] Tensor reference.
        a_ref: A [state_dim, dim_tile_size] Tensor reference.
        b_ref: A [seq_len, state_dim] Tensor reference.
        c_ref: A [seq_len, state_dim] Tensor reference.
        delta_ref: A [seq_len, dim_tile_size] Tensor reference.
        d_ref: A [1, dim_tile_size] Tensor reference.
        mutable_y_ref: A [seq_len, dim_tile_size] Tensor reference, modified by the function.
        mutable_carry_ref: A [state_dim, dim_tile_size] Tensor reference to ht,
            modified by the function.
    """
    dtype = _in_kernel_dtype()

    def _forward_kernel_body(t: int, mutable_carry_ref: Tensor):
        """Computes h_{t+1}, stores it in mutable_carry_ref, computes output,
        and stores it in y_ref."""
        # Compute h_{t+1} = h_t * a_bar_t + b_bar_x_t, where:
        # a_bar_t = exp(delta_t * a), and b_bar_x_t = delta_t * b_t * x_t.
        # The calculation of the carry below can be replaced with `update_carry`, but this
        # decreases speed by about 5%, presumably because `a_bar` cannot then stay in a register.
        delta = jnp.expand_dims(delta_ref[t].astype(dtype), axis=0)  # [1, inner_dim]
        a = a_ref[:].astype(dtype)  # [state_dim, inner_dim]
        a_bar = jnp.exp(delta * a)  # [state_dim, inner_dim]
        x = jnp.expand_dims(x_ref[t].astype(dtype), axis=0)  # [1, inner_dim]
        b_bar = delta * jnp.expand_dims(b_ref[t].astype(dtype), axis=-1) * x
        a_bar *= mutable_carry_ref[:]
        # Put h_{t+1} in the a_bar register.
        a_bar += b_bar
        # Store h_{t+1} in mutable_carry_ref.
        mutable_carry_ref[:] = a_bar
        ct = jnp.expand_dims(c_ref[t].astype(dtype), axis=0)  # [1, state_dim]
        yt = jax.lax.dot_general(  # Equivalent to ct @ a_bar: [1, inner_dim]
            ct,
            a_bar,
            (
                ((1,), (0,)),
                ((), ()),
            ),
            preferred_element_type=jnp.float32,
            precision=MATMUL_PREC,
        )
        x *= d_ref[:].astype(dtype)
        yt += x
        mutable_y_ref[t] = jnp.squeeze(yt, axis=0).astype(mutable_y_ref.dtype)

    seq_len = x_ref.shape[0]

    # Initialize h_carry_ref to 0 iff it's the first time-step. Otherwise do nothing.
    @pl.when(pl.program_id(axis=2) == 0)
    def zero_carry():
        mutable_carry_ref[:] = jnp.zeros_like(mutable_carry_ref)

    # Loop variables need to be "concrete," not references.
    h_carry = mutable_carry_ref[:]
    # Fill y_ref inside the loop.
    h_T = for_loop.for_loop(
        seq_len,
        _forward_kernel_body,
        h_carry,
    )
    # Store the carry as a reference, for possible use later.
    mutable_carry_ref[:] = h_T


def _loop_forward_pallas(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    d: Tensor,
    seq_tile_size: int,
    dim_tile_size: int,
) -> Tensor:
    """Compute the Mamba scan using a Pallas kernel.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seq_len, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        A [batch_size, seq_len, inner_dim] Tensor representing the output of Mamba's forward scan.
    """
    state_dim = b.shape[2]
    x_dtype = x.dtype
    carry_dtype = _in_kernel_dtype()

    batch_tile_size = 1  # We always consider batch examples individually.
    # Create a batch_size x inner_dim x seq_len grid, which allows us to carry state
    # between sequence blocks.
    grid = (
        x.shape[0] // batch_tile_size,
        x.shape[2] // dim_tile_size,
        x.shape[1] // seq_tile_size,
    )
    x_spec, a_spec, b_spec, d_spec, carry_spec = _parameter_blockspecs(
        state_dim, seq_tile_size, dim_tile_size
    )
    # Create the output shape for the carry. We just need its last dimension to be
    # the size of the block, since we will run through the entire sequence before
    # moving on to the next inner_dim-block.
    hcarry_shape = (x.shape[0], a.shape[0], dim_tile_size)

    outputs = pl.pallas_call(
        _forward_kernel,
        grid=grid,
        in_specs=[x_spec, a_spec, b_spec, b_spec, x_spec, d_spec],
        out_shape=[
            # Note: using bfloat16 as y's return type results in each y_t being 0
            # for all t s.t. t % seq_tile_size == 1.
            jax.ShapeDtypeStruct(x.shape, carry_dtype),  # Output 1: y.
            jax.ShapeDtypeStruct(hcarry_shape, carry_dtype),  # Output 2: hcarry.
        ],
        out_specs=[x_spec, carry_spec],
        compiler_params=dict(
            mosaic=dict(
                # Below we tell the compiler which dimensions can be parallelized.
                # All but the sequence dimension are embarassingly parallel.
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            )
        ),
    )(
        x.astype(jnp.float32),
        a,
        b.astype(jnp.float32),
        c.astype(jnp.float32),
        delta.astype(jnp.float32),
        d,
    )
    y = outputs[0].astype(x_dtype)
    return y


@functools.partial(jax.custom_vjp, nondiff_argnums=[6, 7])
def _mamba_scan(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    d: Tensor,
    seq_tile_size: int,
    dim_tile_size: int,
) -> Tensor:
    """A differentiable function mapping the Mamba recurrence's arguments to an output Tensor.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seq_len, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns: A [batch_size, seq_len, inner_dim] Tensor representing the Mamba scan's output.
    """
    return _loop_forward_pallas(x, a, b, c, delta, d, seq_tile_size, dim_tile_size)


def _mamba_scan_fwd(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    d: Tensor,
    seq_tile_size: int,
    dim_tile_size: int,
) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Forward function for `mamba_scan`."""
    y = _loop_forward_pallas(x, a, b, c, delta, d, seq_tile_size, dim_tile_size)
    return y, (x, a, b, c, delta, d)


def _mamba_scan_bwd(
    seq_tile_size: int,
    dim_tile_size: int,
    res: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    dy: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute gradients with respect to `_mamba_scan_fwd`'s arguments."""
    x, a, b, c, delta, d = res  # The tensors we saved from the forward pass.
    dx, da, db, dc, ddelta, dd = _loop_backward_pallas(
        dy, x, a, b, c, delta, d, seq_tile_size, dim_tile_size
    )
    # Return gradients in the same order as arguments are passed in.
    return dx, da, db, dc, ddelta, dd


_mamba_scan.defvjp(_mamba_scan_fwd, _mamba_scan_bwd)


def _pad_to_multiple(x: Tensor, *, divisor: int, axis: int) -> Tensor:
    """Pads the variable `x` to have size along `axis` divisible by `divisor`."""
    if x.shape[axis] % divisor == 0:
        return x
    n = divisor - x.shape[axis] % divisor
    pad_shape = list(x.shape)
    pad_shape[axis] = n
    zeros = jnp.zeros(pad_shape, dtype=x.dtype)
    return jnp.concatenate([x, zeros], axis=axis)


def compute_mamba_scan(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    d: Tensor,
    *,
    seq_tile_size: int,
    dim_tile_size: int,
) -> Tensor:
    """Computes a Mamba scan given inputs. This function is the external interface
    to using Pallas kernels for Mamba scans.

    Note: we will be shard_mapping a closure over this function, and so
    `x`, `a`, `b`, `c`, `delta`, `d` cannot be specified with kwargs.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seq_len, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles into which the sequence dimension will be divided.
        dim_tile_size: The size of the tiles into which the "inner" dimension will be divided.

    Returns:
        A [batch_size, seqlen, inner_dim] Tensor representing the Mamba scan's output.
    """
    assert (
        seq_tile_size > 0 and seq_tile_size % 8 == 0
    ), "`seq_tile_size` must be a positive multiple of 8."
    assert (
        dim_tile_size > 0 and dim_tile_size % 128 == 0
    ), "`dim_tile_size` must be a positive multiple of 128."
    _, seqlen, inner = x.shape

    x, b, c, delta = [
        _pad_to_multiple(arg, divisor=seq_tile_size, axis=1) for arg in [x, b, c, delta]
    ]
    x, delta = [_pad_to_multiple(arg, divisor=dim_tile_size, axis=2) for arg in [x, delta]]
    a, d = [_pad_to_multiple(arg, divisor=dim_tile_size, axis=1) for arg in [a, d]]
    y = _mamba_scan(x, a, b, c, delta, d, seq_tile_size, dim_tile_size)
    # Remove zero-padding if any.
    return y[:, :seqlen, :inner]
