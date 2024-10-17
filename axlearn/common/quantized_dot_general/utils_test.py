# Copyright © 2024 Apple Inc.

"""Tests utils."""
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from aqt.jax.v2.config import DotGeneral
from aqt.jax.v2.numerics import int_numerics, no_numerics

from axlearn.common.quantized_dot_general.utils import (
    is_einsum_swapped_operands,
    lhs_activation_aqt_config,
    rhs_activation_aqt_config,
)
from axlearn.common.test_utils import TestCase


class TestUtils(TestCase):
    """Tests Utils."""

    def test_lhs_activation_aqt_config(self):
        # Make sure key configs are as expected
        cfg: DotGeneral = lhs_activation_aqt_config()
        # Int 8 for forward and dlhs, bf16 for drhs
        # Check if cfg.fwd.lhs is of the expected type or has the expected attribute
        self.assertTrue(
            hasattr(cfg.fwd.lhs, "numerics")
            and isinstance(cfg.fwd.lhs.numerics, int_numerics.IntSymmetric)
        )
        self.assertTrue(isinstance(cfg.fwd.rhs.numerics, int_numerics.IntSymmetric))
        self.assertTrue(isinstance(cfg.dlhs.lhs.numerics, int_numerics.IntSymmetric))
        self.assertTrue(isinstance(cfg.dlhs.rhs.numerics, int_numerics.IntSymmetric))
        self.assertTrue(isinstance(cfg.drhs.lhs.numerics, no_numerics.NoNumerics))
        self.assertTrue(isinstance(cfg.drhs.rhs.numerics, no_numerics.NoNumerics))
        self.assertEqual(cfg.fwd.lhs.numerics.bits, 8)
        self.assertEqual(cfg.fwd.rhs.numerics.bits, 8)
        self.assertEqual(cfg.dlhs.lhs.numerics.bits, 8)
        self.assertEqual(cfg.dlhs.rhs.numerics.bits, 8)
        # Stochastic rounding for dlhs / drhs lhs
        self.assertIsNotNone(cfg.dlhs.lhs.numerics.noise_fn)
        self.assertIsNotNone(cfg.drhs.lhs.numerics.noise_fn)
        # No stochastic rounding for dlhs / drhs rhs
        self.assertIsNone(cfg.dlhs.rhs.numerics.noise_fn)
        self.assertIsNone(cfg.drhs.rhs.numerics.noise_fn)

    def test_rhs_activation_aqt_config(self):
        # Make sure key configs are as expected
        cfg: DotGeneral = rhs_activation_aqt_config()
        # Int 8 for forward and drhs, bf16 for dlhs
        # Check if cfg.fwd.lhs is of the expected type or has the expected attribute
        self.assertTrue(
            hasattr(cfg.fwd.lhs, "numerics")
            and isinstance(cfg.fwd.lhs.numerics, int_numerics.IntSymmetric)
        )
        self.assertTrue(isinstance(cfg.fwd.rhs.numerics, int_numerics.IntSymmetric))
        self.assertTrue(isinstance(cfg.dlhs.lhs.numerics, no_numerics.NoNumerics))
        self.assertTrue(isinstance(cfg.dlhs.rhs.numerics, no_numerics.NoNumerics))
        self.assertTrue(isinstance(cfg.drhs.lhs.numerics, int_numerics.IntSymmetric))
        self.assertTrue(isinstance(cfg.drhs.rhs.numerics, int_numerics.IntSymmetric))
        self.assertEqual(cfg.fwd.lhs.numerics.bits, 8)
        self.assertEqual(cfg.fwd.rhs.numerics.bits, 8)
        self.assertEqual(cfg.drhs.lhs.numerics.bits, 8)
        self.assertEqual(cfg.drhs.rhs.numerics.bits, 8)
        # No Stochastic rounding for dlhs / drhs lhs
        self.assertIsNone(cfg.dlhs.lhs.numerics.noise_fn)
        self.assertIsNone(cfg.drhs.lhs.numerics.noise_fn)
        # Stochastic rounding for dlhs / drhs rhs
        self.assertIsNotNone(cfg.dlhs.rhs.numerics.noise_fn)
        self.assertIsNotNone(cfg.drhs.rhs.numerics.noise_fn)

    @parameterized.product(b=[2, 16], d=[4, 32], h=[8, 64])
    def test_is_einsum_swapped_operands(self, b, d, h):
        # In the most basic case, einsum should retain original operand order
        self.assertFalse(
            is_einsum_swapped_operands(
                "bd,dh->bh",
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    [b, d],
                    dtype=jnp.bfloat16,
                ),
                jax.random.normal(
                    jax.random.PRNGKey(1),
                    [d, h],
                    dtype=jnp.bfloat16,
                ),
            )
        )
        # einsum would attempt to reduce the number of transpose,
        # which might end up swapping operands
        self.assertTrue(
            is_einsum_swapped_operands(
                "dh,bd->bh",
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    [d, h],
                    dtype=jnp.bfloat16,
                ),
                jax.random.normal(
                    jax.random.PRNGKey(1),
                    [b, d],
                    dtype=jnp.bfloat16,
                ),
            )
        )
