# Copyright © 2024 Apple Inc.

"""Tests SSD Pallas kernels."""
from typing import Union

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

from axlearn.common.ein_ops import rearrange, repeat
from axlearn.common.golden import load_golden
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
        batch_size=[2],
        num_heads=[4, 8],
        seq_len=[128],
        dk=[64],
        dv=[64],
        seed=[0],
    )
    def test_ssd_forward(
        self, batch_size: int, num_heads: int, seq_len: int, dk: int, dv: int, seed: int
    ) -> None:
        """Test SSD forward pass against pre-computed reference output."""
        case_name = f"bs{batch_size}_nh{num_heads}_sl{seq_len}_dk{dk}_dv{dv}_s{seed}"
        golden = load_golden(
            "axlearn.common.ssm_kernels.ssd_kernels_test",
            f"test_ssd_forward_{case_name}",
        )
        x_bar = jnp.array(golden["inputs"]["x_bar"], dtype=jnp.float32)
        A_bar = jnp.array(golden["inputs"]["A_bar"], dtype=jnp.float32)
        B = jnp.array(golden["inputs"]["B"], dtype=jnp.float32)
        C = jnp.array(golden["inputs"]["C"], dtype=jnp.float32)

        x_bar_jax = rearrange(x_bar, "b t h d -> b h t d")
        A_bar_jax = rearrange(A_bar, "b t h -> b h t")
        B_jax = rearrange(B, "b t h n -> b h t n")
        C_jax = rearrange(C, "b t h n -> b h t n")

        y_jax = ssd(C_jax, B_jax, x_bar_jax, A_bar_jax, h0=None)
        y_jax = rearrange(y_jax, "b h t d -> b t h d")

        assert_allclose(golden["outputs"]["y_torch"], np.asarray(y_jax), atol=1e-3, rtol=1e-3)

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
            sharded_ssd = jax.shard_map(
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
                check_vma=False,
            )
            return sharded_ssd

        sharded_ssd = get_sharded_ssd(mesh)
        # pylint: disable-next=too-many-function-args
        o_pallas = sharded_ssd(q, k, v, log_alpha)

        assert_allclose(o_pallas, o_ref, atol=1e-3, rtol=1e-3)
