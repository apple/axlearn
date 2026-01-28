# Copyright © 2025 Apple Inc.

"""Tests grouped matrix multiplication kernel for GPU.

GPU device kind been tested:
H100
"""

# pytype: disable=pyi-error
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax import lax

from axlearn.common.megablock.ops import gmm as gmm_lib
from axlearn.common.test_utils import TestCase, Tolerance
from axlearn.common.utils import Tensor


def tolerances_map(dtype: jnp.dtype):
    tolerance_map = {
        1.0: Tolerance(rtol=1e-5, atol=2e-1),
        0.99: Tolerance(rtol=1e-5, atol=2e-2),
        0.97: Tolerance(rtol=1e-5, atol=1e-2),
        0.95: Tolerance(rtol=1e-5, atol=2e-3),
        0.9: Tolerance(rtol=1e-5, atol=1e-3),
        0.8: Tolerance(rtol=1e-5, atol=1e-4),
    }
    if dtype == jnp.float16:
        tolerance_map = {
            1.0: Tolerance(rtol=1e-5, atol=4e-1),
            0.99: Tolerance(rtol=1e-5, atol=1e-1),
            0.95: Tolerance(rtol=1e-5, atol=2e-1),
            0.9: Tolerance(rtol=1e-5, atol=4e-1),
            0.85: Tolerance(rtol=1e-5, atol=5e-2),
            0.75: Tolerance(rtol=1e-5, atol=5e-3),
        }
    return tolerance_map


def generate_group_sizes(num_groups: int, maxval: int) -> Tensor:
    # Generate random integers between 1 and maxval, and then normalize to ensure the sum is maxval.
    random_sizes = jax.random.randint(jax.random.PRNGKey(2), (num_groups,), minval=1, maxval=maxval)

    total_sum = random_sizes.sum()
    scaling_factor = maxval / total_sum

    # Scale the random numbers to ensure their sum is exactly maxval
    group_sizes = jnp.floor(random_sizes * scaling_factor).astype(jnp.int32)

    # If there is any discrepancy due to floor operation, adjust the last element
    discrepancy = maxval - group_sizes.sum()
    group_sizes = group_sizes.at[-1].add(discrepancy)

    return group_sizes


def generate_perfect_group_sizes(num_groups: int, maxval: int, block_size: int) -> Tensor:
    """Generate group sizes which perfectly match the tile size, so that one tile only processed
    by one group."""
    random_sizes = (
        jax.random.randint(
            jax.random.PRNGKey(2), (num_groups,), minval=1, maxval=(maxval // block_size) + 1
        )
        * block_size
    )  # Scale the values by block_size to ensure divisibility

    total_sum = random_sizes.sum()
    scaling_factor = maxval / total_sum  # Scaling factor to make the sum equal to maxval

    group_sizes = jnp.floor(random_sizes * scaling_factor).astype(jnp.int32)

    group_sizes = (group_sizes // block_size) * block_size

    # If there is any discrepancy due to floor operation, adjust the last element
    discrepancy = maxval - group_sizes.sum()
    group_sizes = group_sizes.at[-1].add(discrepancy)

    group_sizes = (group_sizes // block_size) * block_size

    return group_sizes


def generate_random_data(
    m: int, k: int, n: int, num_groups: int, test_dtype: jnp.dtype, transpose: bool
) -> Tuple[Tensor, Tensor, Tensor]:
    lhs = jax.random.normal(jax.random.PRNGKey(0), (m, k), dtype=test_dtype)
    if transpose:
        rhs = jax.random.normal(jax.random.PRNGKey(1), (num_groups, n, k), dtype=test_dtype)
    else:
        rhs = jax.random.normal(jax.random.PRNGKey(1), (num_groups, k, n), dtype=test_dtype)
    group_sizes = generate_group_sizes(num_groups=num_groups, maxval=m)

    return lhs, rhs, group_sizes


# Refer to the gmm reference function for the TPU gmm unit tests:
# https://github.com/jax-ml/jax/blob/main/tests/pallas/tpu_gmm_test.py#L125
def reference_gmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> Tensor:
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
        result = jax.lax.dot(
            lhs[start : start + size, :],
            rhs[i, :, :],
            preferred_element_type=preferred_element_type,
        )
        out.append(result)
        # pylint: disable=unnecessary-list-index-lookup
        start += group_sizes[i]
    return jnp.concatenate(out, axis=0)


def matmul_with_preferred_dtype(a: Tensor, b: Tensor, preferred_element_type: jnp.dtype) -> Tensor:
    """Performs matmul, potentially casting inputs for accumulation."""
    a_casted = a.astype(preferred_element_type)
    b_casted = b.astype(preferred_element_type)
    return jnp.matmul(a_casted, b_casted)


def gmm_lax_scan(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> Tensor:
    """
    Computes Grouped Matrix Multiplication using lax.scan, this function is used for gmm
    benchmark. (We need this function over reference_gmm because JAX does not support dynamically
    sized arrays within JIT compiled functions.)

    Args:
        lhs: Left-hand side tensor with shape (M, K).
        rhs: Right-hand side tensor with shape (NumGroups, K, N).
        group_sizes: Tensor containing the size of each group along the M dimension.
        preferred_element_type: Preferred dtype for accumulation/output.

    Returns:
        Output tensor with shape (M, N).
    """
    m = lhs.shape[0]
    # k = lhs.shape[1]
    # num_groups = rhs.shape[0]
    # n = rhs.shape[2]

    if not isinstance(group_sizes, jax.Array) or not jnp.issubdtype(group_sizes.dtype, jnp.integer):
        raise TypeError(
            f"group_sizes must be a JAX array with integer dtype, got {type(group_sizes)}"
        )
    if lhs.shape[1] != rhs.shape[1]:
        raise ValueError(f"Incompatible K dimensions: lhs {lhs.shape}, rhs {rhs.shape}")
    if rhs.shape[0] != group_sizes.shape[0]:
        raise ValueError(
            f"Incompatible NumGroups dimension: rhs {rhs.shape}, group_sizes {group_sizes.shape}"
        )

    def compute_group_contribution(
        carry_start_index: int, group_data: Tuple[int, Tensor]
    ) -> Tuple[int, Tensor]:
        """
        Scan body: Computes the contribution of one group to the final output.

        Args:
            carry_start_index: The starting row index for the current group in lhs/output.
            group_data: A tuple (group_size, rhs_slice) for the current group.
                        rhs_slice has shape (K, N).

        Returns:
            A tuple (next_start_index, masked_contribution).
            masked_contribution has shape (M, N), zero everywhere except for the
            rows corresponding to this group.
        """
        group_size, rhs_slice = group_data
        start_index = carry_start_index

        # Calculate the contribution for this group
        full_contribution = matmul_with_preferred_dtype(lhs, rhs_slice, preferred_element_type)

        # Create a mask for the rows belonging to the current group
        row_indices = jnp.arange(m)  # Shape (M,)
        mask = (row_indices >= start_index) & (row_indices < start_index + group_size)  # Shape (M,)

        # Apply the mask. Broadcast mask from (M,) to (M, 1) to select rows.
        # Result is zero where mask is False.
        masked_contribution = jnp.where(
            mask[:, None],  # (M, 1)
            full_contribution,  # (M, N)
            jnp.zeros_like(full_contribution),  # (M, N)
        )

        # Calculate the start index for the *next* group
        next_start_index = start_index + group_size
        return next_start_index, masked_contribution

    init_carry = 0

    # iterate through groups
    scan_xs = (group_sizes, rhs)

    _, all_masked_contributions = lax.scan(compute_group_contribution, init_carry, scan_xs)

    final_out = jnp.sum(all_masked_contributions, axis=0)  # Shape (M, N)

    return final_out.astype(preferred_element_type)


# (m, k, n, tm, tk, tn, num_groups)
GROUPED_MATMUL_TESTS = [
    (64, 128, 32, 16, 32, 16, 2),
    (512, 128, 128, 16, 32, 16, 16),
    (128, 128, 128, 16, 16, 32, 20),
    (512, 512, 2048, 32, 16, 32, 40),
    (1024, 512, 2048, 32, 16, 64, 40),
    (4096, 512, 2048, 64, 16, 64, 80),
    #     (128, 16, 64, 128, 128, 128, 4),  # Partial tile are not supported yet.
]


class GmmTest(TestCase):
    """Tests GPU and TPU decoding."""

    # Verify gmm_lib with both forward and backward passes
    @parameterized.product(
        [
            dict(zip(["m", "k", "n", "tm", "tk", "tn", "num_groups"], args))
            for args in GROUPED_MATMUL_TESTS
        ],
        out_dtype=[jnp.float32, jnp.float16],
        transpose_rhs=[True, False],
        interpret=[False],
    )
    # pylint: disable-next=too-many-positional-arguments
    def test_gmm_lib(
        self,
        m: int,
        k: int,
        n: int,
        tm: int,
        tk: int,
        tn: int,
        num_groups: int,
        out_dtype: jnp.dtype,
        transpose_rhs: bool,
        interpret: bool = False,
    ):
        if not jax.default_backend() == "gpu":
            self.skipTest("Incompatible hardware")

        lhs, rhs, group_sizes = generate_random_data(m, k, n, num_groups, out_dtype, transpose_rhs)

        out, vjpfun = jax.vjp(
            partial(
                gmm_lib,
                preferred_element_type=out_dtype,
                transpose_rhs=transpose_rhs,
                interpret=interpret,
                tiling=(tm, tk, tn),
            ),
            lhs,
            rhs,
            group_sizes,
        )

        def reference_fn(lhs, rhs, group_sizes, preferred_element_type):
            return reference_gmm(
                lhs, rhs, group_sizes, preferred_element_type=preferred_element_type
            )

        expected_out, reference_vjpfun = jax.vjp(
            partial(reference_fn, preferred_element_type=out_dtype),
            lhs,
            rhs.swapaxes(1, 2) if transpose_rhs else rhs,
            group_sizes,
        )

        self.assertEqual(out.dtype, out_dtype)
        self.assertEqual(expected_out.dtype, out_dtype)

        cotangent = jax.random.normal(jax.random.PRNGKey(3), (m, n), dtype=out_dtype)
        grad_lhs, grad_rhs, *_ = vjpfun(cotangent)
        expected_grad_lhs, expected_grad_rhs, *_ = reference_vjpfun(cotangent)

        self.assertAllCloseWithOutliers(
            out,
            expected_out,
            tolerance_map=tolerances_map(out_dtype),
        )
        self.assertAllCloseWithOutliers(
            grad_lhs,
            expected_grad_lhs,
            tolerance_map=tolerances_map(out_dtype),
        )
        self.assertAllCloseWithOutliers(
            grad_rhs,
            expected_grad_rhs.swapaxes(1, 2) if transpose_rhs else expected_grad_rhs,
            tolerance_map=tolerances_map(out_dtype),
        )

    @parameterized.product(
        [
            dict(zip(["m", "k", "n", "tm", "tk", "tn", "num_groups"], args))
            for args in GROUPED_MATMUL_TESTS
        ],
        out_dtype=[jnp.float32],
        transpose_rhs=[False],
        interpret=[False],
    )
    # pylint: disable-next=too-many-positional-arguments
    def test_gmm_benchmark_fn(
        self,
        m: int,
        k: int,
        n: int,
        tm: int,
        tk: int,
        tn: int,
        num_groups: int,
        out_dtype: jnp.dtype,
        transpose_rhs: bool,
        interpret: bool = False,
    ):
        if not jax.default_backend() == "gpu":
            self.skipTest("Incompatible hardware")

        lhs, rhs, group_sizes = generate_random_data(m, k, n, num_groups, out_dtype, transpose_rhs)

        out, vjpfun = jax.vjp(
            partial(
                gmm_lib,
                preferred_element_type=out_dtype,
                transpose_rhs=transpose_rhs,
                interpret=interpret,
                tiling=(tm, tk, tn),
            ),
            lhs,
            rhs.swapaxes(1, 2) if transpose_rhs else rhs,
            group_sizes,
        )

        def reference_fn(lhs, rhs, group_sizes, preferred_element_type):
            rhs = rhs.swapaxes(1, 2) if transpose_rhs else rhs
            return gmm_lax_scan(
                lhs, rhs, group_sizes, preferred_element_type=preferred_element_type
            )

        expected_out, reference_vjpfun = jax.vjp(
            partial(reference_fn, preferred_element_type=out_dtype),
            lhs,
            rhs.swapaxes(1, 2) if transpose_rhs else rhs,
            group_sizes,
        )
        np.testing.assert_equal(out.dtype, out_dtype)
        np.testing.assert_equal(expected_out.dtype, out_dtype)

        atol, rtol = 3e-1, 1e-5
        self.assertNestedAllClose(out, expected_out, atol=atol, rtol=rtol)

        cotangent = jax.random.normal(jax.random.PRNGKey(3), (m, n))
        grad_lhs, grad_rhs, *_ = vjpfun(cotangent)
        expected_grad_lhs, expected_grad_rhs, *_ = reference_vjpfun(cotangent)

        self.assertNestedAllClose(grad_lhs, expected_grad_lhs, atol=atol, rtol=rtol)
        self.assertNestedAllClose(grad_rhs, expected_grad_rhs, atol=atol, rtol=rtol)
