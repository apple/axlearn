# Copyright © 2024 Apple Inc.

"""Tests utils."""
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from aqt.jax.v2.config import CalibrationMode, DequantMode, DotGeneral, Tensor

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
        # Check for expected attributes
        self.assertTrue(isinstance(cfg.fwd.lhs, Tensor))
        self.assertTrue(isinstance(cfg.fwd.rhs, Tensor))
        self.assertTrue(isinstance(cfg.dlhs.lhs, Tensor))
        self.assertTrue(isinstance(cfg.dlhs.rhs, Tensor))
        self.assertTrue(isinstance(cfg.drhs.lhs, Tensor))
        self.assertTrue(isinstance(cfg.drhs.rhs, Tensor))
        # Check for other attributes based on actual structure
        # Example: Check if use_fwd_quant is set correctly
        self.assertEqual(cfg.fwd.lhs.use_fwd_quant, False)
        self.assertEqual(cfg.fwd.rhs.use_fwd_quant, False)
        # Check dequant_mode
        self.assertEqual(cfg.fwd.lhs.dequant_mode, DequantMode.OUTPUT)
        self.assertEqual(cfg.fwd.rhs.dequant_mode, DequantMode.OUTPUT)
        # Check calibration_mode
        self.assertEqual(cfg.fwd.lhs.calibration_mode, CalibrationMode.CONTRACTING_AXIS)
        self.assertEqual(cfg.fwd.rhs.calibration_mode, CalibrationMode.CONTRACTING_AXIS)
        # Check if the accumulator dtype is set correctly
        # Adjust the following lines based on the actual structure of cfg
        self.assertEqual(cfg.fwd.dg_accumulator_dtype, jnp.int32)
        self.assertEqual(cfg.dlhs.dg_accumulator_dtype, jnp.int32)
        self.assertIsNone(cfg.drhs.dg_accumulator_dtype)

    def test_rhs_activation_aqt_config(self):
        # Make sure key configs are as expected
        cfg: DotGeneral = rhs_activation_aqt_config()
        # Check for expected attributes
        self.assertTrue(isinstance(cfg.fwd.lhs, Tensor))
        self.assertTrue(isinstance(cfg.fwd.rhs, Tensor))
        self.assertTrue(isinstance(cfg.dlhs.lhs, Tensor))
        self.assertTrue(isinstance(cfg.dlhs.rhs, Tensor))
        self.assertTrue(isinstance(cfg.drhs.lhs, Tensor))
        self.assertTrue(isinstance(cfg.drhs.rhs, Tensor))
        # Check for other attributes based on actual structure
        self.assertEqual(cfg.fwd.lhs.use_fwd_quant, False)
        self.assertEqual(cfg.fwd.rhs.use_fwd_quant, False)
        # Check dequant_mode
        self.assertEqual(cfg.fwd.lhs.dequant_mode, DequantMode.OUTPUT)
        self.assertEqual(cfg.fwd.rhs.dequant_mode, DequantMode.OUTPUT)
        # Check calibration_mode
        self.assertEqual(cfg.fwd.lhs.calibration_mode, CalibrationMode.CONTRACTING_AXIS)
        self.assertEqual(cfg.fwd.rhs.calibration_mode, CalibrationMode.CONTRACTING_AXIS)
        # Check if the accumulator dtype is set correctly
        # Adjust the following lines based on the actual structure of cfg
        self.assertEqual(cfg.fwd.dg_accumulator_dtype, jnp.int32)
        self.assertIsNone(cfg.dlhs.dg_accumulator_dtype)
        self.assertEqual(cfg.drhs.dg_accumulator_dtype, jnp.int32)

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
