# Copyright © 2023 Apple Inc.

"""Tests for _optimization_barrier.py."""

import functools
import math
from typing import Callable

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from axlearn.common import ops, test_utils
from axlearn.common.test_utils import TestCase


class OptimizationBarrierTest(TestCase):
    """Tests ops.optimization_barrier"""

    def test_forward_optimization_barrier(self):
        """Test that constant folding happens without a barrier and does
        not happen with a barrier by inspecting the optimized HLO for the
        presence / absence of the original constants and folded constant.
        """
        a = "7"
        b = "13"
        product = str(int(a) * int(b))

        @functools.partial(jax.jit, static_argnames="wrapper")
        def f(wrapper: Callable):
            x = int(a)
            x *= wrapper(int(b))
            return x

        compile_options = dict(
            xla_cpu_enable_fast_math=True,
            xla_cpu_fast_math_honor_nans=False,
            xla_cpu_fast_math_honor_infs=False,
            xla_cpu_fast_math_honor_functions=False,
            xla_cpu_fast_math_honor_division=False,
        )
        # Test that constant folding does happen without barrier.
        hlo = f.lower(lambda x: x).compile(compile_options).as_text()
        hlo = test_utils.clean_hlo(hlo)
        self.assertNotIn(a, hlo)  # Original constants are not in HLO.
        self.assertNotIn(b, hlo)
        self.assertIn(product, hlo)  # Folded constant is in HLO.
        # Test that constant folding does not happen with barrier.
        hlo = f.lower(ops.forward_optimization_barrier).compile(compile_options).as_text()
        hlo = test_utils.clean_hlo(hlo)
        self.assertIn(a, hlo)
        self.assertIn(b, hlo)
        self.assertNotIn(product, hlo)

    def test_forward_optimization_barrier_grad(self):
        """Tests that `forward_optimization_barrier` does not wrap new computations generated
        during the backward pass during autodifferentiation.
        """
        a = jnp.array(2**8, dtype=jnp.float16)
        x = jnp.array(2**-8, dtype=jnp.float16)

        @jax.jit
        @jax.value_and_grad
        def f_without_barrier(x):
            return a * (a * x**2)

        @jax.jit
        @jax.value_and_grad
        def f_with_barrier(x):
            return a * ops.forward_optimization_barrier(a * x**2)

        # Test without barrier
        result = f_without_barrier(x)
        chex.assert_trees_all_equal(result, (math.inf, math.inf))
        # Test with barrier
        result = f_with_barrier(x)
        chex.assert_trees_all_equal(result, (1, math.inf))

    def test_forward_optimization_barrier_vmap(self):
        """Tests that `forward_optimization_barrier` works with vmap."""
        a = jnp.array(2**8, dtype=jnp.float16)
        x = jnp.array(2**-8, dtype=jnp.float16)
        x = x[None]

        @jax.jit
        @jax.vmap
        def f_with_barrier(x):
            print(x.shape)
            return a * ops.forward_optimization_barrier(a * x) ** 2

        result = f_with_barrier(x)
        chex.assert_trees_all_equal(result.squeeze(), 256)


if __name__ == "__main__":
    absltest.main()
