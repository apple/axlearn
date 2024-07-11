# Copyright © 2024 Apple Inc.

"""Tests Mamba Pallas kernels."""
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from axlearn.common.ssm import MambaMixerLayer
from axlearn.common.ssm_kernels import mamba_kernels
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import Tensor

if jax.default_backend() != "tpu":
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


# Use higher precision matmuls for testing.
mamba_kernels.MATMUL_PREC = jax.lax.Precision("float32")


def _mamba_inputs(
    *,
    batch_size: int,
    seq_len: int,
    inner_dim: int,
    dtype: jnp.dtype,
    prng_key: Tensor,
    state_dim: int = 16,
) -> Tuple[Tensor, MambaMixerLayer.SSMParameters]:
    """Computes random Mamba inputs.

    Args:
        batch_size: The batch size of the inputs.
        seq_len: The sequence length of the inputs.
        inner_dim: The Mamba layer's 'inner' dimension.
        dtype: The desired dtype of the parameters.
        prng_key: A key for initializing random parameters.
        state_dim: The state dimension of the Mamba layer.

    Returns:
        A tensor of shape [batch_size, seq_len, inner_dim] representing an input.
        An instance of MambaMixerLayer.SSMParameters.
    """
    k1, k2, k3, k4, k5, k6 = jax.random.split(prng_key, 6)
    x = jax.random.normal(k1, (batch_size, seq_len, inner_dim), dtype=dtype) * 0.1
    a = jax.random.normal(k2, (state_dim, inner_dim), dtype=dtype) * 0.1
    b = jax.random.normal(k3, (batch_size, seq_len, state_dim), dtype=dtype) * 0.1
    c = jax.random.normal(k4, (batch_size, seq_len, state_dim), dtype=dtype) * 0.1
    delta = jax.random.normal(k5, (batch_size, seq_len, inner_dim), dtype=dtype) * 0.1
    d = jax.random.normal(k6, (1, inner_dim), dtype=dtype) * 0.1
    return x, MambaMixerLayer.SSMParameters(a=a, b=b, c=c, delta=delta, d=d)


def loop_forward_jax(
    *,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    d: Tensor,
) -> Tensor:
    """Pure jax implementation of the Mamba forward loop, with a linear scan.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seqlen, inner_dim]
        d: [1, inner_dim]

    Returns:
       A tensor of shape [batch_size, seq_len, inner_dim], representing the output of the
       Mamba recurrence.
    """

    def forward_kernel_body_jax(
        carry_ref,
        t,
        x_ref,
        a_ref,
        b_ref,
        c_ref,
        delta_ref,
        d_ref,
    ):
        # new state = carry * a_bar + b_bar_x
        delta = jnp.expand_dims(delta_ref[:, t], axis=1)  # [b, 1, inner]
        a = jnp.expand_dims(a_ref[:], axis=0)  # [1, state, inner]
        a_bar = jnp.exp(delta * a)  # [b, state, inner]
        b = jnp.expand_dims(b_ref[:, t], axis=-1)  # [b, state, 1]
        b_bar = delta * b * jnp.expand_dims(x_ref[:, t], axis=1)  # [b, state, inner]

        a_bar *= carry_ref[:]
        a_bar += b_bar  # holds the new state
        yt = jnp.squeeze(jnp.expand_dims(c_ref[:, t], axis=1) @ a_bar, axis=1)
        yt += x_ref[:, t] * d_ref
        yt = yt.astype(x.dtype)
        return a_bar, yt

    h_carry_ref = jnp.zeros(x.shape[:1] + a.shape)

    _, ys = jax.lax.scan(
        functools.partial(
            forward_kernel_body_jax,
            x_ref=x,
            a_ref=a,
            b_ref=b,
            c_ref=c,
            delta_ref=delta,
            d_ref=d,
        ),
        h_carry_ref,
        xs=jnp.arange(x.shape[1]),
    )
    return jnp.swapaxes(ys, 0, 1)


def jax_loss(x: Tensor, a: Tensor, b: Tensor, c: Tensor, delta: Tensor, d: Tensor) -> Tensor:
    """A simple squared loss for testing the backward computation.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [inner_dim, state_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seqlen, inner_dim]
        d: [inner_dim]

    Returns:
        A scalar loss.
    """
    y = loop_forward_jax(x=x, a=a, b=b, c=c, delta=delta, d=d)
    return jnp.sum(y * y)


jax_grad = jax.jit(jax.grad(jax_loss, argnums=(0, 1, 2, 3, 4, 5)))


@functools.partial(jax.jit, static_argnums=[6, 7])
def pallas_loss(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    delta: Tensor,
    d: Tensor,
    seq_tile_size: int,
    dim_tile_size: int,
) -> Tensor:
    """A simple squared loss for testing the backward computation.

    Args:
        x: [batch_size, seq_len, inner_dim]
        a: [state_dim, inner_dim]
        b: [batch_size, seq_len, state_dim]
        c: [batch_size, seq_len, state_dim]
        delta: [batch_size, seqlen, inner_dim]
        d: [1, inner_dim]
        seq_tile_size: The size of the tiles along the sequence dimension.
        dim_tile_size: The size of the tiles along the 'inner' dimension.

    Returns:
        A scalar loss.
    """
    # pylint: disable=protected-access
    y = mamba_kernels._mamba_scan(
        x, a, b, c, delta, d, seq_tile_size=seq_tile_size, dim_tile_size=dim_tile_size
    )
    return jnp.sum(y * y)


pallas_grad = jax.jit(jax.grad(pallas_loss, argnums=(0, 1, 2, 3, 4, 5)), static_argnums=[6, 7])


class MambaPallasKernelTest(TestCase):
    """Tests Mamba Pallas kernels."""

    @parameterized.product(
        batch_size=[1, 4],
        seq_tile_size=[8, 16],
        dim_tile_size=[128, 256],
        dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_forward(
        self, batch_size: int, seq_tile_size: int, dim_tile_size: int, dtype: jnp.dtype
    ):
        seq_len = 64
        inner_dim = 1024
        x, (a, b, c, delta, d) = _mamba_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            inner_dim=inner_dim,
            dtype=dtype,
            prng_key=jax.random.PRNGKey(0),
        )
        y_jax = loop_forward_jax(x=x, a=a, b=b, c=c, delta=delta, d=d)
        # pylint: disable=protected-access
        y_pallas = mamba_kernels._mamba_scan(x, a, b, c, delta, d, seq_tile_size, dim_tile_size)
        self.assertEqual(y_pallas.dtype, dtype)
        atol = 1e-6 if dtype == jnp.float32 else 5e-5
        assert_allclose(y_jax, y_pallas, atol=atol)

    @parameterized.product(
        batch_size=[1, 4],
        seq_tile_size=[8, 16],
        dim_tile_size=[128, 256],
        dtype=[jnp.float32, jnp.bfloat16],
    )
    def test_backward(
        self, batch_size: int, seq_tile_size: int, dim_tile_size: int, dtype: jnp.dtype
    ):
        seq_len = 64
        inner_dim = 1024
        x, (a, b, c, delta, d) = _mamba_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            inner_dim=inner_dim,
            dtype=dtype,
            prng_key=jax.random.PRNGKey(1),
        )
        d_args_jax = jax_grad(x, a, b, c, delta, d)
        d_args_pallas = pallas_grad(x, a, b, c, delta, d, seq_tile_size, dim_tile_size)
        self.assertTrue(all(g.dtype == dtype for g in d_args_pallas))
        atol = 1e-6 if dtype == jnp.float32 else 5e-2
        self.assertNestedAllClose(d_args_jax, d_args_pallas, atol=atol)
