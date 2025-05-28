# Copyright © 2025 Apple Inc.

"""Tests einops."""
import contextlib

import einops
import jax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common.ein_ops import (
    _get_input_reshape,
    _parse_axes,
    _parse_pattern,
    rearrange,
    repeat,
)
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
        ["...", ("...",)],
        [" ...", ("...",)],
        ["... ", ("...",)],
        [" ... ", ("...",)],
        ["(...)", (("...",),)],
        ["((...))", ValueError(r"Unexpected characters in pattern: '\(\)'")],
        ["3", ValueError("Unexpected characters in pattern: '3'")],
        ["1a", ValueError("Unexpected characters in pattern: '1'")],
        ["-", ValueError("Unexpected characters in pattern: '-'")],
        ["a-", ValueError("Unexpected characters in pattern: '-'")],
        ["_a", ("_a",)],
        ["_3", ("_3",)],
        ["a1", ("a1",)],
        ["a3_", ("a3_",)],
        ["  a3_  ", ("a3_",)],
        ["a ...", ("a", "...")],
        ["... a", ("...", "a")],
        ["a a", ValueError(r"Duplicated axis name: 'a' in axes='a a'\.")],
        ["a b a", ValueError(r"Duplicated axis name: 'a' in axes='a b a'\.")],
        ["b t (g k) h", ("b", "t", ("g", "k"), "h")],
        ["b t (g ... k) h", ("b", "t", ("g", "...", "k"), "h")],
        [
            "b t (g 1 k) h",
            ValueError(
                r"Invalid axis name: '1'\. Must match Python _IDENTIFIER='\[_a-z\]\[_a-z0-9\]\*'\."
            ),
        ],
        ["x (_y z9) (a0 b1)", ("x", ("_y", "z9"), ("a0", "b1"))],
        ["  x  ( _y z9 )  ( a0 b1 ) ", ("x", ("_y", "z9"), ("a0", "b1"))],
        ["as_df _as221dfs ____asdsad324", ("as_df", "_as221dfs", "____asdsad324")],
        ["x y ) z", ValueError(r"Unexpected characters in pattern: '\)'")],
        ["x y ( ) z", ValueError(r"Group '\( \)' must contain at least two axes.")],
        ["x (y) z", ValueError(r"Group '\(y\)' must contain at least two axes.")],
        ["(x (y z)", ValueError(r"Unexpected characters in pattern: '\('")],
        ["(x (y z))", ValueError(r"Unexpected characters in pattern: '\( \)'")],
        ["(x (y z) k)", ValueError(r"Unexpected characters in pattern: '\(  \)'")],
    )
    def test_parse_axes(self, pattern, expected):
        if isinstance(expected, ValueError):
            ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            parsed = _parse_axes(pattern)
            self.assertEqual(parsed, expected)

    @parameterized.parameters(
        ["a b", (2, 3), ValueError(r"pattern='a b' doesn't have ->\.")],
        ["a b - a b", (2, 3), ValueError(r"pattern='a b - a b' doesn't have ->\.")],
        ["a b -> a b", (2, 3), (("a", "b"), ("a", "b"))],
        ["a b->a b", (2, 3), (("a", "b"), ("a", "b"))],
        ["  a b  ->  a b  ", (2, 3), (("a", "b"), ("a", "b"))],
        [
            "... b c -> ... c b",
            (1, 2, 3),
            (("_ELLIPSIS_0", "b", "c"), ("_ELLIPSIS_0", "c", "b")),
        ],
        [
            "... b c -> ... c b",
            (1, 2, 3, 4),
            (("_ELLIPSIS_0", "_ELLIPSIS_1", "b", "c"), ("_ELLIPSIS_0", "_ELLIPSIS_1", "c", "b")),
        ],
        [
            "b ... c -> c ... b",
            (1, 2, 3, 4),
            (("b", "_ELLIPSIS_0", "_ELLIPSIS_1", "c"), ("c", "_ELLIPSIS_0", "_ELLIPSIS_1", "b")),
        ],
        [
            "b c ... -> c b ...",
            (1, 2, 3, 4),
            (("b", "c", "_ELLIPSIS_0", "_ELLIPSIS_1"), ("c", "b", "_ELLIPSIS_0", "_ELLIPSIS_1")),
        ],
        [
            "b c ... -> ... c b",
            (1, 2, 3, 4),
            (("b", "c", "_ELLIPSIS_0", "_ELLIPSIS_1"), ("_ELLIPSIS_0", "_ELLIPSIS_1", "c", "b")),
        ],
        [
            "b c ... -> a c b d",
            (1, 2, 3, 4),
            ValueError(r"lhs and rhs contain '\.\.\.' asymmetrically\."),
        ],
        [
            "a b c d -> a b ...",
            (1, 2, 3, 4),
            ValueError(r"lhs and rhs contain '\.\.\.' asymmetrically\."),
        ],
        [
            "b (c ...) -> b c ...",
            (1, 2, 3),
            ValueError(r"Only rhs is allowed to have \.\.\. inside a group\."),
        ],
        [
            "b c ... -> b (c ...)",
            (1, 2, 3, 4),
            (
                ("b", "c", "_ELLIPSIS_0", "_ELLIPSIS_1"),
                ("b", ("c", "_ELLIPSIS_0", "_ELLIPSIS_1")),
            ),
        ],
        [
            "b c ... -> b c (...)",
            (1, 2, 3, 4),
            (
                ("b", "c", "_ELLIPSIS_0", "_ELLIPSIS_1"),
                ("b", "c", ("_ELLIPSIS_0", "_ELLIPSIS_1")),
            ),
        ],
        [
            "... b c -> ... c b",
            (1, 2, 3, 4, 5),
            (
                ("_ELLIPSIS_0", "_ELLIPSIS_1", "_ELLIPSIS_2", "b", "c"),
                ("_ELLIPSIS_0", "_ELLIPSIS_1", "_ELLIPSIS_2", "c", "b"),
            ),
        ],
        [
            "b t (g k) h -> b (g t) h k",
            (1, 2, 4, 3),
            (("b", "t", ("g", "k"), "h"), ("b", ("g", "t"), "h", "k")),
        ],
        [
            "... (g k) h -> ... g (h k)",
            (1, 2, 4, 3),
            (
                ("_ELLIPSIS_0", "_ELLIPSIS_1", ("g", "k"), "h"),
                ("_ELLIPSIS_0", "_ELLIPSIS_1", "g", ("h", "k")),
            ),
        ],
        ["foo b -> foo b", (1, 2), (("foo", "b"), ("foo", "b"))],
        ["foo boo -> foo boo", (1, 2), (("foo", "boo"), ("foo", "boo"))],
        ["foo boo -> (foo boo)", (1, 2), (("foo", "boo"), (("foo", "boo"),))],
        ["foo boo -> ( foo boo )", (1, 2), (("foo", "boo"), (("foo", "boo"),))],
    )
    def test_parse_pattern(self, pattern, in_shape, expected):
        if isinstance(expected, ValueError):
            ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            parsed = _parse_pattern(pattern=pattern, in_shape=in_shape)
            self.assertEqual(parsed, expected)

    @parameterized.parameters(
        ["a b c", (2, 3, 4), dict(), dict(a=2, b=3, c=4)],
        ["  a   b  c  ", (2, 3, 4), dict(), dict(a=2, b=3, c=4)],
        ["ace base__ case_123", (2, 3, 4), dict(), dict(ace=2, base__=3, case_123=4)],
        ["1 a 1 b 1 c 1", (1, 2, 1, 3, 1, 4, 1), dict(), dict(a=2, b=3, c=4)],
        ["(a b) c", (4, 5), dict(a=2), dict(a=2, b=2, c=5)],
        ["(a b) c", (4, 5), dict(b=2), dict(a=2, b=2, c=5)],
        ["(a b) c", (4, 5), dict(a=2, b=2), dict(a=2, b=2, c=5)],
        ["(a b) c", (4, 5), dict(a=2, b=2, c=5), dict(a=2, b=2, c=5)],
        [
            "(a b) c",
            (4, 5),
            dict(),
            ValueError(r"Multiple unknown axes \(a, b\) in a group are not allowed\."),
        ],
        [
            "(a b) c",
            (4, 5),
            dict(a=2, b=3),
            ValueError(r"Incompatible shape reshape: \(4, 5\) -> \[2, 3, 5\]"),
        ],
        [
            "(a b) c",
            (4, 5),
            dict(a=2, b=2, c=4),
            ValueError(r"Conflicting axis size for c: from tensor 5, from user 4\."),
        ],
        ["(a b c)", (12,), dict(a=2, b=2, c=3), dict(a=2, b=2, c=3)],
        ["(a b c)", (12,), dict(b=2, c=3), dict(a=2, b=2, c=3)],
        ["(a b c)", (12,), dict(a=2, b=2), dict(a=2, b=2, c=3)],
        ["(a b c)", (12,), dict(a=2, c=3), dict(a=2, b=2, c=3)],
        [
            "(a b c)",
            (12,),
            dict(a=2),
            ValueError(r"Multiple unknown axes \(b, c\) in a group are not allowed\."),
        ],
        [
            "(a b c)",
            (12,),
            dict(c=3),
            ValueError(r"Multiple unknown axes \(a, b\) in a group are not allowed\."),
        ],
        [
            "(a b c)",
            (12,),
            dict(),
            ValueError(r"Multiple unknown axes \(a, b\) in a group are not allowed\."),
        ],
        [
            "(a b c)",
            (12, 3),
            dict(a=2, b=2, c=3),
            ValueError(r"Incompatible shape reshape: \(12, 3\) -> \[2, 2, 3\]"),
        ],
    )
    def test_get_input_reshape(self, axes, shape, axes_lengths, expected):
        lhs_axes = _parse_axes(axes)
        if isinstance(expected, ValueError):
            ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            dim_size_map = _get_input_reshape(
                shape=shape, lhs_axes=lhs_axes, axes_lengths=axes_lengths
            )
            self.assertEqual(dim_size_map, expected)

    @parameterized.parameters(
        ("a b -> a", ValueError("Mismatch between LHS axes and RHS axes: {'a'} vs .*")),
        ("a -> a b", ValueError(r"Incompatible shape reshape: \(2, 4\) -> \[2\]")),
        ("a (b -> a b", ValueError(r"Unexpected characters in pattern: '\('")),
        ("a (b) -> a b", ValueError(r"Group '\(b\)' must contain at least two axes\.")),
        (
            "a (b 1) -> a b",
            ValueError(
                r"Invalid axis name: '1'\. Must match Python _IDENTIFIER='\[_a-z\]\[_a-z0-9\]\*'\."
            ),
        ),
        (
            "a b -> (a 1) b",
            ValueError(
                r"Invalid axis name: '1'\. Must match Python _IDENTIFIER='\[_a-z\]\[_a-z0-9\]\*'\."
            ),
        ),
        ("a b -> (a (b))", ValueError(r"Unexpected characters in pattern: '\( \)'")),
        ("a b -> (a (b c))", ValueError(r"Unexpected characters in pattern: '\( \)'")),
        ("a b -> a 1", ValueError("Mismatch between LHS axes and RHS axes: {'a'} vs .*")),
        ("a b -> a c", ValueError(r"Missing axis c in input\.")),
        ("a a b -> a b", ValueError(r"Duplicated axis name: 'a' in axes='a a b'\.")),
        ("a b -> a b b", ValueError(r"Duplicated axis name: 'b' in axes='a b b'\.")),
        ("a ... -> a b ...", ValueError(r"Missing axis b in input\.")),
        ("a ... -> a (b ...)", ValueError(r"Missing axis b in input\.")),
    )
    def test_rearrange_invalid_pattern(self, pattern, expected):
        a, b = 2, 4
        x = jnp.arange(a * b).reshape((a, b))
        ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        with ctx:
            rearrange(x, pattern)

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
            "a b ... -> b a ...",
            "a b ... -> b ... a",
            "a ... b -> ... b a",
            "... a b -> b a ...",
        ]
    )
    def test_rearrange_transpose_ellipsis(self, pattern):
        a, b, c, d = 2, 3, 4, 5
        x = jnp.arange(a * b * c * d).reshape((a, b, c, d))
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

    @parameterized.product(
        pattern=[
            "... k g h -> ... (g k) h",
            "k ... g h -> ... (g h k)",
            "k g ... h -> k (h g) ...",
            "k g h ... -> (g k) h (...)",
            "k g h ... -> (g k) (... h)",
        ]
    )
    def test_rearrange_out_paren_ellipse(self, pattern):
        b, t, k, g, h = 2, 3, 4, 5, 6
        x = jnp.arange(b * t * k * g * h).reshape((b, t, k, g, h))
        self.assertEqual(rearrange(x, pattern).tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.product(
        pattern=[
            "... (g k) h -> ... k g h",
            "(k g h) ... -> ... k g h",
            "k (h g) ... -> k ... g h",
            "k ... (g h) -> ... k g h",
        ]
    )
    def test_rearrange_in_paren_ellipse(self, pattern):
        b, t, k, g, h = 2, 3, 4, 5, 6
        x = jnp.arange(b * t * k * g * h).reshape((b, t, k, g, h))
        x = einops.rearrange(x, "... k g h ->" + pattern.split("->")[0])
        self.assertEqual(
            rearrange(x, pattern, k=k, g=g, h=h).tolist(),
            einops.rearrange(x, pattern, k=k, g=g, h=h).tolist(),
        )

    @parameterized.product(pattern=["a b c -> b c a", "a b c -> (b a) c", "a b c -> c 1 (b a)"])
    def test_rearrange_jit(self, pattern):
        a, b, c = 2, 3, 4
        x = jnp.arange(a * b * c).reshape((a, b, c))

        @jax.jit
        def test():
            return rearrange(x, pattern)

        self.assertEqual(test().tolist(), einops.rearrange(x, pattern).tolist())

    @parameterized.parameters(
        (
            "a b -> a (b 1)",
            ValueError(
                r"Invalid axis name: '1'\. Must match Python _IDENTIFIER='\[_a-z\]\[_a-z0-9\]\*'\."
            ),
        ),
        ("a b -> a (b b)", ValueError(r"Duplicated axis name: 'b' in axes='a \(b b\)'\.")),
        ("a b -> (a k)", ValueError("lhs axes {.*} must be same to rhs axes {.*}.")),
        (
            "a b -> a (b 1 k)",
            ValueError(
                r"Invalid axis name: '1'\. Must match Python _IDENTIFIER='\[_a-z\]\[_a-z0-9\]\*'\."
            ),
        ),
        ("a b -> a ((b 1 k))", ValueError(r"Unexpected characters in pattern: '\(\)'")),
        ("a b -> a (b k", ValueError(r"Unexpected characters in pattern: '\('")),
        ("a b -> b (a k)", ValueError(r"repeat doesn't allow reordering existing axes\.")),
        ("a b -> b a", ValueError(r"repeat doesn't allow reordering existing axes\.")),
        (
            "a b -> (a k) (k b)",
            ValueError(r"Duplicated axis name: 'k' in axes='\(a k\) \(k b\)'\."),
        ),
        ("a ... -> ... a", ValueError(r"repeat doesn't allow reordering existing axes\.")),
        (
            "a ... -> (a k) (k ...)",
            ValueError(r"Duplicated axis name: 'k' in axes='\(a k\) \(k \.\.\.\)'\."),
        ),
    )
    def test_repeat_invalid_pattern(self, pattern, expected):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))
        ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        with ctx:
            repeat(x, pattern, k=k)

    @parameterized.parameters(
        ["a b ... -> a b", ValueError(r"lhs and rhs contain '\.\.\.' asymmetrically\.")],
        ["a b -> a b ...", ValueError(r"lhs and rhs contain '\.\.\.' asymmetrically\.")],
    )
    def test_repeat_invalid_pattern_advanced(self, pattern, expected):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))
        ctx = self.assertRaisesRegex(ValueError, expected.args[0])
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
        pattern=["a b -> a b k 1", "a b -> a k 1 b", "a b -> a 1 (b k)", "a b -> 1 a (k b)"]
    )
    def test_repeat_expand_dims(self, pattern):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, b))
        self.assertEqual(repeat(x, pattern, k=k).tolist(), einops.repeat(x, pattern, k=k).tolist())

    @parameterized.product(
        pattern=["a 1 b 1 -> a b k", "a 1 b 1 -> a k b", "a 1 b 1 -> a (b k)", "a 1 b 1 -> a (k b)"]
    )
    def test_repeat_squeeze(self, pattern):
        a, b, k = 2, 3, 4
        x = jnp.arange(a * b).reshape((a, 1, b, 1))
        self.assertEqual(repeat(x, pattern, k=k).tolist(), einops.repeat(x, pattern, k=k).tolist())

    @parameterized.product(
        pattern=[
            "... c -> ... c k",
            "... c -> ... (c k)",
            "c ... -> c ... k",
            "c ... -> c (... k)",
            "c ... -> c (...) k",
        ]
    )
    def test_repeat_ellipsis(self, pattern):
        a, b, c, k = 2, 3, 4, 5
        x = jnp.arange(a * b * c).reshape((a, b, c))
        self.assertEqual(repeat(x, pattern, k=k).tolist(), einops.repeat(x, pattern, k=k).tolist())

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
