# Copyright © 2024 Apple Inc.

"""Tests QuantizedDotGeneral."""

import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from axlearn.common.decoder import Decoder
from axlearn.common.module import functional as F
from axlearn.common.quantized_dot_general.activation_clipping import HardActivationClippingLayer
from axlearn.common.quantized_dot_general.layers import (
    DotGeneralQuantizationType,
    QuantizedDotGeneral,
    set_quantized_dot_general_recursively,
)
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape


class TestQuantizedDotGeneral(TestCase):
    """Tests QuantizedDotGeneral layer."""

    # TODO(jiarui): Assert output for INT8 once TPU tests are available in CI
    @parameterized.product(
        b=[2, 16],
        d=[4, 32],
        h=[8, 64],
        quantization_type_and_assert_output=[
            (None, True),  # Test bf16, ensure parity on output
            (
                DotGeneralQuantizationType.INT_8,
                False,
            ),  # Test INT8, ignore output parity since this is executing on CPU instead of TPU
        ],
    )
    def test_einsum_maybe_quantized(self, b, d, h, quantization_type_and_assert_output):
        quantization_type, assert_output = quantization_type_and_assert_output
        # When config is None, maybe_quantized_einsum should reduce to einsum
        with Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "fsdp")):
            quantized_dot_general_cfg = QuantizedDotGeneral.default_config().set(
                quantization_type=quantization_type
            )
            quantized_dot_general_layer = quantized_dot_general_cfg.set(
                name="quantized_dot_general_layer"
            ).instantiate(parent=None)
            params = quantized_dot_general_layer.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(123)
            )
            inputs = [
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
            ]
            output, _ = F(
                quantized_dot_general_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=dict(zip(("subscripts", "activation", "kernel"), inputs)),
                is_training=True,
                method="einsum_maybe_quantized",
            )
            reference = jnp.einsum(*inputs)
            if assert_output:
                self.assertNestedAllClose(output, reference)

    def test_set_quantized_dot_general_recursively(self):
        cfg = Decoder.default_config()
        self.assertIsNone(cfg.transformer.layer.feed_forward.linear1.quantized_dot_general)
        clipping_config = HardActivationClippingLayer.default_config().set(
            clipping_max_abs=2, clipping_summary=True
        )
        set_quantized_dot_general_recursively(
            cfg,
            quantized_dot_general=QuantizedDotGeneral.default_config().set(
                quantization_type=DotGeneralQuantizationType.INT_8,
                activation_clipping=clipping_config,
            ),
        )
        quantized_dot_general_config = (
            cfg.transformer.layer.feed_forward.linear1.quantized_dot_general
        )
        self.assertIsNotNone(quantized_dot_general_config)
        self.assertEqual(
            DotGeneralQuantizationType.INT_8,
            quantized_dot_general_config.quantization_type,
        )
        self.assertIsNotNone(
            clipping_config,
            quantized_dot_general_config.activation_clipping,
        )

    @parameterized.product(b=[8], d=[16], h=[32])
    def test_einsum_maybe_quantized_fp8(self, b, d, h):
        if jax.default_backend() != "gpu":
            self.skipTest("Need H100 GPUs for fp8 compute.")
        if not is_supported_mesh_shape((8, 1)):
            return
        with Mesh(mesh_utils.create_device_mesh((8, 1)), ("data", "fsdp")):
            q_dot_cfg: QuantizedDotGeneral.Config = QuantizedDotGeneral.default_config()
            q_dot_cfg.quantization_type = DotGeneralQuantizationType.FP_8
            q_dot_cfg.name = "quantized_dot_general_layer"
            q_dot_cfg.fp8_amax_history_length = 0
            quantized_dot_general_layer: QuantizedDotGeneral = q_dot_cfg.instantiate(parent=None)
            params = quantized_dot_general_layer.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(123)
            )
            inputs = [
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
            ]

            output, _ = F(
                quantized_dot_general_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=dict(zip(("subscripts", "activation", "kernel"), inputs)),
                is_training=True,
                method="einsum_maybe_quantized",
            )
            reference = jnp.einsum(*inputs)
            self.assertNestedAllClose(output, reference, atol=0.25, rtol=0.2)
