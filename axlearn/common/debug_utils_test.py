"""Tests for debug_utils.py"""
from typing import Callable

import jax
from absl.testing import parameterized
from jax.experimental import checkify

from axlearn.common import struct, test_utils
from axlearn.common.debug_utils import (
    checkify_and_rerun_on_nonfinite,
    checkify_pjit,
    checking_leaks_pjit,
)
from axlearn.common.utils import Tensor


class TestPjitWrappers(test_utils.TestCase):
    """Tests for functions that produce `JitFn` wrappers."""

    def setUp(self):
        super().setUp()
        self.enter_context(jax.checking_leaks())

    @parameterized.parameters(checkify_and_rerun_on_nonfinite, checkify_pjit)
    def test_float_wrappers(self, make_pjit: Callable):
        """Tests `checkify_and_rerun_on_nonfinite` and `checkify_pjit`."""
        wrapped_pjit = make_pjit(errors=checkify.float_checks)

        @wrapped_pjit
        def fn(x, y):
            return x / y

        print(fn(8, 2))
        self.assertNestedAllClose(fn(8, 2), 4.0, atol=0, rtol=1e-6)
        with self.assertRaisesRegex(checkify.JaxRuntimeError, "division by zero"):
            fn(6, 0)

    def test_checking_leaks_pjit(self):
        class Static(struct.PyTreeNode):
            val: Callable = struct.field(pytree_node=False)

        @checking_leaks_pjit()
        def fn(x: Tensor) -> Static:
            return Static(lambda: x)

        with self.assertRaisesRegex(Exception, "Leaked trace"):
            fn(6)

        @checking_leaks_pjit()
        def fn2(x: Tensor) -> Tensor:
            return x

        self.assertEqual(fn2(5), 5)
