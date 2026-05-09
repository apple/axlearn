# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# naver/splade:
# Copyright (c) 2021-present NAVER Corp.
# Licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

"""Tests Splade modules."""

import jax
import jax.random
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common.golden import load_golden
from axlearn.common.layers import Linear
from axlearn.common.module import functional as F
from axlearn.common.splade import SpladePooling
from axlearn.common.test_utils import TestCase, assert_allclose


class SpladePoolingTest(TestCase):
    @parameterized.parameters("max", "sum")
    def test_splade_pooling(self, mode):
        dim = 32
        vocab_size = 64

        splade_pooling_layer_cfg = SpladePooling.default_config().set(
            name="splade_pooling_layer",
            input_dim=dim,
            output_dim=vocab_size,
            splade_mode=mode,
        )
        splade_pooling_layer_cfg.vocab_mapping.inner_head = Linear.default_config().set(
            bias=False, input_dim=dim, output_dim=vocab_size
        )
        splade_layer = splade_pooling_layer_cfg.instantiate(parent=None)

        # Test without paddings.
        golden_no_pad = load_golden(
            "axlearn.common.splade_test", f"test_splade_pooling_{mode}_no_padding"
        )
        tokens_no_pad = jnp.asarray(golden_no_pad["inputs"]["tokens"])
        layer_output_no_pad, _ = F(
            splade_layer,
            inputs=dict(tokens=tokens_no_pad, paddings=None),
            state=golden_no_pad["params"],
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_output_no_pad, golden_no_pad["outputs"]["values"])

        # Test with paddings.
        golden_pad = load_golden(
            "axlearn.common.splade_test", f"test_splade_pooling_{mode}_with_padding"
        )
        tokens_pad = jnp.asarray(golden_pad["inputs"]["tokens"])
        # Golden stores torch-style mask (1=valid, 0=padded).
        # AXLearn expects paddings where 1=padded, so invert.
        axlearn_paddings = 1.0 - golden_pad["inputs"]["paddings"]
        layer_output_pad, _ = F(
            splade_layer,
            inputs=dict(tokens=tokens_pad, paddings=jnp.asarray(axlearn_paddings)),
            state=golden_pad["params"],
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_output_pad, golden_pad["outputs"]["values"])


if __name__ == "__main__":
    absltest.main()
