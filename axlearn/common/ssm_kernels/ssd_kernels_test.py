# Copyright © 2024 Apple Inc.

"""Tests SSD Pallas kernels."""
from typing import Union

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from absl.testing import parameterized
from einops import rearrange, repeat
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from torch.nn import functional as F

from axlearn.common.ssm_kernels.ssd_kernels import _ssd_backward, _ssd_forward, ssd, ssd_linear_scan
from axlearn.common.test_utils import TestCase, assert_allclose

if jax.default_backend() != "tpu":
    pytest.skip(reason="Incompatible hardware", allow_module_level=True)


def _ssd_reference(q, k, v, log_alpha, h0):
    """Reference implementation of SSD for comparison.

    Args:
        q/k: [batch_size, num_heads, seq_len, dk]
        v: [batch_size, num_heads, seq_len, dv]
        log_alpha: [batch_size, num_heads, seq_len]
        h0: [batch_size, num_heads, dk, dv]

    Returns:
        o: [batch_size, num_heads, seq_len, dv]
    """
    return ssd_linear_scan(q, k, v, log_alpha, h0)[0]


def _ssd_naive_reference(q, k, v, log_alpha, h0=None):
    """For-loop reference implementation of SSD.

    Note that this implementation somehow have worse
    numerical stability than the vmap version above.

    Args:
        q/k: [batch_size, num_heads, seq_len, dk]
        v: [batch_size, num_heads, seq_len, dv]
        log_alpha: [batch_size, num_heads, seq_len]
        h0: [batch_size, num_heads, dk, dv]

    Returns:
        o: [batch_size, num_heads, seq_len, dv]
        h: [batch_size, num_heads, dk, dv]
    """
    bs, ng, l, dk = q.shape
    _, _, _, dv = v.shape

    bs, ng, l, dk = q.shape
    bs, nh, l, dv = v.shape
    assert nh % ng == 0

    num_head_per_group = nh // ng
    q = repeat(q, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)
    k = repeat(k, "b ng l dk -> b (ng nhg) l dk", nhg=num_head_per_group)

    if h0 is None:
        h0 = jnp.zeros((bs, nh, dk, dv), dtype=jnp.float32)

    o_list = []
    h = h0
    for t in range(l):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        alpha_t = jnp.exp(log_alpha[:, :, t, None, None])

        h = alpha_t * h + jnp.einsum(
            "...i,...j->...ij", k_t, v_t, preferred_element_type=jnp.float32
        )
        o_t = jnp.einsum("...ij,...i->...j", h, q_t, preferred_element_type=jnp.float32)
        o_list.append(o_t)
    o = jnp.stack(o_list, axis=2)
    return o, h


# disable some pylint checks to allow copied code to pass checks

# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=unused-variable


def segsum(x):
    """More stable segment sum calculation. Helper function copied from
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py.
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_chunk_tri(X, A, B, C, chunk_size=16, initial_states=None):
    """Reference implementation of SSD with chunked computation, copied from
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py.

    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)

    X, A, B, C corresponds to V, \alpha, K, Q in linear attention
    (H_t = \alpha H_{t-1)+ K_t^\top V_t, O_t = Q_t S_t).
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % chunk_size == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=chunk_size) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


@jax.jit
def _ssd_reference_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    h0: Union[jax.Array, None],
    do: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    o, vjp = jax.vjp(_ssd_reference, q, k, v, alpha, h0)
    return o, vjp(do)


def _generate_ssd_inputs(shape, dtype, seed, paramn="gla", zero_h0=True):
    """
    Args:
        shape: [bs, ng, nh, l, dk, dv]
        dtype: float32, bfloat16
        seed: random seed
        paramn: "mamba" or "gla"
        zero_h0: whether to generate zero initial hidden state

    Returns:
        q, k, v, log_alpha, h0, do
    """
    bs, ng, nh, l, dk, dv = shape
    rng = jax.random.PRNGKey(seed)
    q_key, k_key, v_key, alpha_key, h_key, dh_key = jax.random.split(rng, 6)

    if paramn == "mamba":
        q = jax.random.uniform(q_key, (bs, ng, l, dk), dtype=dtype)
        k = jax.random.uniform(k_key, (bs, ng, l, dk), dtype=dtype)
        v = jax.random.uniform(v_key, (bs, nh, l, dv), dtype=jnp.float32)

        log_alpha = -jnp.exp(jax.random.uniform(alpha_key, (bs, nh, l), dtype=jnp.float32))
        dt = jax.random.normal(alpha_key, (bs, nh, l), dtype=jnp.float32)
        dt = jnp.log(1.0 + jnp.exp(dt - 4))

        log_alpha = dt * log_alpha
        v = v * dt[..., None]
    elif paramn == "gla":
        q = jax.random.normal(q_key, (bs, ng, l, dk), dtype=dtype)
        k = jax.random.normal(k_key, (bs, ng, l, dk), dtype=dtype)
        v = jax.random.normal(v_key, (bs, nh, l, dv), dtype=dtype)

        # shortconv (skipped) and non-linear activation
        q = jnn.silu(q)
        k = jnn.silu(k)
        v = jnn.silu(v)

        # l2 norm (help reduces the range of dq/dk -> better precision for bfloat16)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)

        log_alpha = (
            jnn.log_sigmoid(jax.random.normal(alpha_key, (bs, nh, l), dtype=jnp.float32)) / 16.0
        )
    else:
        raise ValueError(f"Unsupported param: {paramn}")

    if zero_h0:
        h0 = jnp.zeros((bs, nh, dk, dv), dtype=jnp.float32)
    else:
        h0 = jax.random.normal(h_key, (bs, nh, dk, dv), dtype=jnp.float32)

    do = jax.random.normal(dh_key, (bs, nh, l, dv), dtype=dtype)

    # log_alpha is always in float32
    log_alpha = log_alpha.astype(jnp.float32)
    return q, k, v, log_alpha, h0, do


class SSDPallasKernelTest(TestCase):
    @parameterized.product(
        batch_size=[2, 4],
        num_heads=[4, 8],
        seq_len=[1024, 2048],
        dk=[128, 256],
        dv=[128, 256],
        seed=[0, 1],
    )
    def test_ssd_forward(
        self, batch_size: int, num_heads: int, seq_len: int, dk: int, dv: int, seed: int
    ) -> None:
        """Test SSD forward pass against Tri's torch reference implementation."""
        # Set the device to CPU
        device = "cpu"

        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Generate random input data
        x = np.random.rand(batch_size, seq_len, num_heads, dk).astype(np.float32)
        dt = np.random.rand(batch_size, seq_len, num_heads).astype(np.float32)
        dt = np.log(1.0 + np.exp(dt - 4))
        A = -np.exp(np.random.rand(batch_size, seq_len, num_heads).astype(np.float32))
        B = np.random.rand(batch_size, seq_len, num_heads, dv).astype(np.float32)
        C = np.random.rand(batch_size, seq_len, num_heads, dv).astype(np.float32)

        # Compute intermediate variables
        x_bar = x * dt[..., None]
        A_bar = A * dt

        # Convert numpy arrays to torch tensors
        x_torch = torch.tensor(x, dtype=torch.float32)
        dt_torch = torch.tensor(dt, dtype=torch.float32)
        A_torch = torch.tensor(A, dtype=torch.float32)
        B_torch = torch.tensor(B, dtype=torch.float32)
        C_torch = torch.tensor(C, dtype=torch.float32)
        x_bar_torch = torch.tensor(x_bar, dtype=torch.float32)
        A_bar_torch = torch.tensor(A_bar, dtype=torch.float32)

        # Compute the torch reference output
        y_torch, _ = ssd_chunk_tri(x_bar_torch, A_bar_torch, B_torch, C_torch)

        # Convert numpy arrays to jax arrays
        x_jax = jnp.array(x, dtype=jnp.float32)
        dt_jax = jnp.array(dt, dtype=jnp.float32)
        A_jax = jnp.array(A, dtype=jnp.float32)
        B_jax = jnp.array(B, dtype=jnp.float32)
        C_jax = jnp.array(C, dtype=jnp.float32)
        x_bar_jax = jnp.array(x_bar, dtype=jnp.float32)
        A_bar_jax = jnp.array(A_bar, dtype=jnp.float32)

        # Reshape jax arrays for comparison
        x_jax = rearrange(x_jax, "b t h d -> b h t d")
        dt_jax = rearrange(dt_jax, "b t h -> b h t")
        A_jax = rearrange(A_jax, "b t h -> b h t")
        B_jax = rearrange(B_jax, "b t h n -> b h t n")
        C_jax = rearrange(C_jax, "b t h n -> b h t n")
        x_bar_jax = rearrange(x_bar_jax, "b t h d -> b h t d")
        A_bar_jax = rearrange(A_bar_jax, "b t h -> b h t")

        # Compute the jax output
        y_jax = ssd(C_jax, B_jax, x_bar_jax, A_bar_jax, h0=None)
        y_jax = rearrange(y_jax, "b h t d -> b t h d")

        assert_allclose(y_torch.numpy(), np.asarray(y_jax), atol=1e-3, rtol=1e-3)

    @parameterized.product(
        batch_size=[2, 4],
        num_heads=[4, 8],
        seq_len=[1024, 2048],
        dk=[128, 256],
        dv=[128, 256],
        dtype=["float32", "bfloat16"],
        seed=[0, 1],
    )
    def test_forward_and_backward(self, batch_size, num_heads, seq_len, dk, dv, dtype, seed):
        try:
            self.ssd_forward_and_backward(batch_size, num_heads, seq_len, dk, dv, dtype, seed)
        except Exception as e:
            # breakpoint()  # uncomment for debugging failed conditions
            raise e

    def ssd_forward_and_backward(self, batch_size, num_heads, seq_len, dk, dv, dtype, seed):
        num_groups = num_heads
        shape = (batch_size, num_groups, num_heads, seq_len, dk, dv)
        q, k, v, log_alpha, h0, do = _generate_ssd_inputs(shape, dtype, seed)
        if dtype == "float32":
            tol = 1e-3
        elif dtype == "bfloat16":
            tol = 1e-2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        o_pallas, residuals = _ssd_forward(q, k, v, log_alpha, h0)
        final_state_pallas = residuals[-1]
        o_ref, final_state_ref = ssd_linear_scan(q, k, v, log_alpha, h0)

        assert_allclose(o_pallas, o_ref, atol=tol, rtol=tol)
        assert_allclose(final_state_pallas, final_state_ref, atol=tol, rtol=tol)

        dq_pallas, dk_pallas, dv_pallas, dlog_alpha_pallas, dh0_pallas = _ssd_backward(
            residuals, do
        )
        _, ssd_reference_grad_ = jax.vjp(_ssd_reference, q, k, v, log_alpha, h0)
        dq_ref, dk_ref, dv_ref, dlog_alpha_ref, dh0_ref = ssd_reference_grad_(do)

        assert_allclose(dq_pallas, dq_ref, atol=tol, rtol=tol)
        assert_allclose(dk_pallas, dk_ref, atol=tol, rtol=tol)
        assert_allclose(dv_pallas, dv_ref, atol=tol, rtol=tol)
        assert_allclose(dlog_alpha_pallas, dlog_alpha_ref, atol=tol, rtol=tol)
        assert_allclose(dh0_pallas, dh0_ref, atol=tol, rtol=tol)


class ShardSSDPallasKernelTest(TestCase):
    # this test only works for four devices
    @pytest.mark.skipif(jax.device_count() != 4, reason="Requires 4 devices")
    def test_sharded_ssd_wo_sp(self):
        batch, ngroups, nheads, seqlen, k_head_dim, v_head_dim = 8, 4, 4, 1024, 256, 128
        dtype = "float32"
        q, k, v, log_alpha, _, _ = _generate_ssd_inputs(
            (batch, ngroups, nheads, seqlen, k_head_dim, v_head_dim), dtype, 0
        )

        o_ref, _ = ssd_linear_scan(q, k, v, log_alpha)

        devices = mesh_utils.create_device_mesh((2, 1, 1, 1, 2))
        mesh = Mesh(devices, axis_names=("data", "expert", "fsdp", "seq", "model"))

        def get_sharded_ssd(mesh):
            """
            Note: current version assumes that h0 is None, for which you don't
            need to provide partition spec.
            """
            sharded_ssd = shard_map(
                ssd,
                mesh=mesh,
                in_specs=(
                    PartitionSpec(("data", "expert", "fsdp"), ("seq", "model"), None, None),
                    PartitionSpec(
                        ("data", "expert", "fsdp"),
                        ("seq", "model"),
                        None,
                        None,
                    ),
                    PartitionSpec(("data", "expert", "fsdp"), ("seq", "model"), None, None),
                    PartitionSpec(("data", "expert", "fsdp"), ("seq", "model"), None),
                ),
                out_specs=PartitionSpec(("data", "expert", "fsdp"), "model", "seq", None),
                check_rep=False,
            )
            return sharded_ssd

        sharded_ssd = get_sharded_ssd(mesh)
        o_pallas = sharded_ssd(q, k, v, log_alpha)

        assert_allclose(o_pallas, o_ref, atol=1e-3, rtol=1e-3)
