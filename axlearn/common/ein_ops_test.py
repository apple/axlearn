# Copyright © 2025 Apple Inc.

"""Tests einops."""
import contextlib

import einops
import jax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common.ein_ops import _parse_axes, rearrange, repeat
from axlearn.common.test_utils import TestCase


class EinopsTest(TestCase):
    @parameterized.parameters(
        ["a", ("a",)],
        [" a", ("a",)],
        ["a ", ("a",)],
        [" a ", ("a",)],
        ["_", ("_",)],
        ["1", ("1",)],
        [" 1", ("1",)],
        ["1 ", ("1",)],
        [" 1 ", ("1",)],
        ["3", ValueError],
        ["1a", ValueError],
        ["-", ValueError],
        ["a-", ValueError],
        ["_a", ("_a",)],
        ["_3", ("_3",)],
        ["a1", ("a1",)],
        ["a3_", ("a3_",)],
        ["  a3_  ", ("a3_",)],
        ["a a", ValueError],
        ["a b a", ValueError],
        ["b t (g k) h", ("b", "t", ("g", "k"), "h")],
        ["b t (g 1 k) h", ValueError],
        ["x (_y z9) (a0 b1)", ("x", ("_y", "z9"), ("a0", "b1"))],
        ["  x  ( _y z9 )  ( a0 b1 ) ", ("x", ("_y", "z9"), ("a0", "b1"))],
        ["as_df _as221dfs ____asdsad324", ("as_df", "_as221dfs", "____asdsad324")],
        ["x y ) z", ValueError],
        ["x y ( ) z", ValueError],
        ["x (y) z", ValueError],
        ["(x (y z)", ValueError],
        ["(x (y z))", ValueError],
        ["(x (y z) k)", ValueError],
    )
    def test_parse_axes(self, pattern, expected):
        if expected == ValueError:
            ctx = self.assertRaises(ValueError)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            parsed = _parse_axes(pattern)
            self.assertEqual(parsed, expected)

    @parameterized.product(
        pattern=[
            "a b -> a",
            "a -> a b",
            "a (b -> a b",
            "a (b) -> a b",
            "a (b 1) -> a b",
            "a b -> (a 1) b",
            "a b -> (a (b))",
            "a b -> (a (b c))",
            "a b -> a 1",
            "a b -> a c",
            "a a b -> a b",
            "a b -> a b b",
        ]
    )
    def test_rearrange_invalid_pattern(self, pattern):
        a, b = 2, 4
        x = jnp.arange(a * b).reshape((a, b))
        ctx = self.assertRaises(ValueError)
        with ctx:
            rearrange(x, pattern)

    @parameterized.product(pattern=["a b->a b", "a b  ->  a b", "a b->(a b)", "a b -> ( a b )"])
    def test_rearrange_varying_spaces(self, pattern):
        a, b = 2, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(
        pattern=[
            "foo b -> foo b",
            "foo boo -> foo boo",
            "foo boo -> (foo boo)",
            "foo boo -> ( foo boo )",
        ]
    )
    def test_rearrange_long_names(self, pattern):
        a, b = 2, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(pattern=["a b -> b a"])
    def test_rearrange_empty_inputs(self, pattern):
        a, b = 0, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(
        pattern=["a b -> 1 a b", "a b -> a 1 b", "a b -> a b 1", "a b -> a 1 b 1"]
    )
    def test_rearrange_expand_dims(self, pattern):
        a, b = 2, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(pattern=["a 1 b 1 -> a 1 b", "a 1 b 1 -> a b 1", "a 1 b 1 -> a b"])
    def test_rearrange_squeeze(self, pattern):
        a, b = 2, 4
        x = jnp.arange(a * b).reshape((a, 1, b, 1))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(pattern=["a b c -> b c a", "a b c -> b a c", "a b c -> c b a"])
    def test_rearrange_transpose(self, pattern):
        a, b, c = 2, 3, 4
        x = jnp.arange(a * b * c).reshape((a, b, c))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(
        pattern=[
            "b t k g h -> b t (k g) h",
            "b t k g h -> b t (k g h)",
            "b t k g h -> (b t) k (g h)",
            "b t k g h -> (b t k) (g h)",
        ]
    )
    def test_rearrange_out_paren(self, pattern):
        b, t, k, g, h = 2, 3, 4, 5, 6
        x = jnp.arange(b * t * k * g * h).reshape((b, t, k, g, h))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(
        pattern=[
            "b t (k g) h -> b t k g h",
            "b t (k g h) -> b t k g h",
            "(b t) k (g h) -> b t k g h",
            "(b t k) (g h) -> b t k g h",
        ]
    )
    def test_rearrange_in_paren(self, pattern):
        b, t, k, g, h = 2, 3, 4, 5, 6
        x = jnp.arange(b * t * k * g * h).reshape((b, t, k, g, h))
        x = einops.rearrange(x, "b t k g h ->" + pattern.split("->")[0])
        self.assertEqual(
            rearrange(x, pattern, b=b, t=t, k=k, g=g, h=h).tolist(),
            einops.rearrange(x, pattern, b=b, t=t, k=k, g=g, h=h).tolist(),
        )

    @parameterized.product(
        pattern=[
            "b (t c) -> b t c",
            "b (c t) -> b t c",
            "(b c t) -> b t c",
        ]
    )
    def test_rearrange_in_paren_partial(self, pattern):
        b, t, c = 2, 3, 4
        x = jnp.arange(b * t * c).reshape((b, t, c))
        x = einops.rearrange(x, "b t c ->" + pattern.split("->")[0])
        self.assertEqual(
            rearrange(x, pattern, b=b, t=t).tolist(),
            einops.rearrange(x, pattern, b=b, t=t).tolist(),
        )

    @parameterized.product(
        pattern=[
            "b t k g h -> b t (g k) h",
            "b t k g h -> t b (g h k)",
            "b t k g h -> k (h g) (t b)",
            "b t k g h -> (g t k) (h b)",
        ]
    )
    def test_rearrange_out_paren_transpose(self, pattern):
        b, t, k, g, h = 2, 3, 4, 5, 6
        x = jnp.arange(b * t * k * g * h).reshape((b, t, k, g, h))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(
        pattern=[
            "b t (g k) h -> b t k g h",
            "(k g h) t b -> b t k g h",
            "k (h g) (t b) -> b t k g h",
            "(t g k) (h b) -> b t k g h",
        ]
    )
    def test_rearrange_in_paren_transpose(self, pattern):
        b, t, k, g, h = 2, 3, 4, 5, 6
        x = jnp.arange(b * t * k * g * h).reshape((b, t, k, g, h))
        x = einops.rearrange(x, "b t k g h ->" + pattern.split("->")[0])
        self.assertEqual(
            rearrange(x, pattern, b=b, t=t, k=k, g=g, h=h).tolist(),
            einops.rearrange(x, pattern, b=b, t=t, k=k, g=g, h=h).tolist(),
        )

    @parameterized.product(pattern=["a b c -> b c a", "a b c -> (b a) c", "a b c -> c 1 (b a)"])
    def test_rearrange_jit(self, pattern):
        a, b, c = 2, 3, 4
        x = jnp.arange(a * b * c).reshape((a, b, c))

        @jax.jit
        def test():
            return rearrange(x, pattern)

        self.assertEqual(test().tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(
        pattern=[
            "a b -> a (b 1)",
            "a b -> a (b b)",
            "a b -> a (b 1 k)",
            "a b -> a ((b 1 k))",
            "a b -> a (b k",
            "a b -> b (a k)",
            "a b -> b a",
            "a b -> (a k) (k b)",
        ]
    )
    def test_repeat_invalid_pattern(self, pattern):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))
        ctx = self.assertRaises(ValueError)
        with ctx:
            repeat(x, pattern, k=k)

    @parameterized.product(
        pattern=["a b -> a b k", "a b -> a k b", "a b -> a (b k)", "a b -> a (k b)"]
    )
    def test_repeat(self, pattern):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(repeat(x, pattern, k=k).tolist(), einops.repeat(x, pattern, k=k).tolist())

    @parameterized.product(
        pattern=[" a b ->a b  k ", "a b -> a (b k)", "a b->a (b k)", "a b  ->  a ( b k )"]
    )
    def test_repeat_varying_spaces(self, pattern):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(repeat(x, pattern, k=k).tolist(), einops.repeat(x, pattern, k=k).tolist())

    @parameterized.product(
        pattern=["a b -> a (b koo)", "foo b -> foo (b koo)", "foo boo -> foo (boo koo)"]
    )
    def test_repeat_long_names(self, pattern):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(
            repeat(x, pattern, koo=k).tolist(), einops.repeat(x, pattern, koo=k).tolist()
        )

    @parameterized.product(pattern=["a b -> (a k) (b l)", "a b -> (k a) (l b)"])
    def test_repeat_multiple(self, pattern):
        a, b, k, l = 2, 3, 4, 5
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(
            repeat(x, pattern, k=k, l=l).tolist(), einops.repeat(x, pattern, k=k, l=l).tolist()
        )

    @parameterized.product(pattern=["a b -> a b k", "a b -> a (k b)", "a b -> (k a) b"])
    def test_repeat_jit(self, pattern):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))

        @jax.jit
        def test():
            return repeat(x, pattern, k=k)

        self.assertEqual(test().tolist(), einops.repeat(x, pattern, k=k).tolist())


if __name__ == "__main__":
    absltest.main()
