# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/detectron2:
# Copyright 2019-2020, detectron2 contributors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests window attention."""
# pylint: disable=no-self-use,too-many-lines,too-many-public-methods,invalid-name
import jax.random
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.golden import load_golden
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.vision.window_attention import (
    window_partition_with_num_windows,
    window_partition_with_window_size,
    window_unpartition_with_num_windows,
    window_unpartition_with_window_size,
)

_MODULE_NAME = "axlearn.vision.window_attention_test"


class WindowAttentionTest(TestCase):
    """Tests window attention utils."""

    @parameterized.parameters(
        [
            dict(
                inputs_shape=[2, 100, 100, 3],
                window_size=14,
                golden_key="test_window_partition_b2_h100_w100_c3_ws14",
            ),
            dict(
                inputs_shape=[2, 100, 100, 3],
                window_size=7,
                golden_key="test_window_partition_b2_h100_w100_c3_ws7",
            ),
            dict(
                inputs_shape=[2, 224, 224, 8],
                window_size=14,
                golden_key="test_window_partition_b2_h224_w224_c8_ws14",
            ),
            dict(
                inputs_shape=[4, 7, 7, 8],
                window_size=10,
                golden_key="test_window_partition_b4_h7_w7_c8_ws10",
            ),
            dict(
                inputs_shape=[4, 10, 10, 4],
                window_size=10,
                golden_key="test_window_partition_b4_h10_w10_c4_ws10",
            ),
            dict(
                inputs_shape=[2, 1, 1, 4],
                window_size=2,
                golden_key="test_window_partition_b2_h1_w1_c4_ws2",
            ),
            dict(
                inputs_shape=[2, 1, 1, 4],
                window_size=1,
                golden_key="test_window_partition_b2_h1_w1_c4_ws1",
            ),
            dict(
                inputs_shape=[4, 8, 15, 4],
                window_size=10,
                golden_key="test_window_partition_b4_h8_w15_c4_ws10",
            ),
        ]
    )
    def test_window_partition_with_window_size(
        self,
        inputs_shape,  # pylint: disable=unused-argument
        window_size,
        golden_key,
    ):
        golden = load_golden(_MODULE_NAME, golden_key)
        inputs = golden["inputs"]["x"]
        ref_output_windows = golden["outputs"]["windows"]
        ref_resized_height = int(golden["outputs"]["resized_height"])
        ref_resized_width = int(golden["outputs"]["resized_width"])

        output_windows, (resized_height, resized_width) = window_partition_with_window_size(
            inputs, window_size
        )

        # Tests window output shape
        self.assertEqual(output_windows.shape, ref_output_windows.shape)
        self.assertEqual((resized_height, resized_width), (ref_resized_height, ref_resized_width))
        # Tests window value close
        assert_allclose(output_windows, ref_output_windows)

    @parameterized.parameters(
        [
            dict(
                inputs_shape=[128, 14, 14, 3],
                window_size=14,
                resized_hw=(112, 112),
                original_hw=(100, 100),
                golden_key="test_window_unpartition_b128_h14_w14_c3_ws14_rh112_rw112_oh100_ow100",
            ),
            dict(
                inputs_shape=[450, 7, 7, 3],
                window_size=7,
                resized_hw=(105, 105),
                original_hw=(100, 100),
                golden_key="test_window_unpartition_b450_h7_w7_c3_ws7_rh105_rw105_oh100_ow100",
            ),
            dict(
                inputs_shape=[512, 14, 14, 8],
                window_size=14,
                resized_hw=(224, 224),
                original_hw=(224, 224),
                golden_key="test_window_unpartition_b512_h14_w14_c8_ws14_rh224_rw224_oh224_ow224",
            ),
            dict(
                inputs_shape=[4, 10, 10, 8],
                window_size=10,
                resized_hw=(10, 10),
                original_hw=(7, 7),
                golden_key="test_window_unpartition_b4_h10_w10_c8_ws10_rh10_rw10_oh7_ow7",
            ),
            dict(
                inputs_shape=[4, 10, 10, 4],
                window_size=10,
                resized_hw=(10, 10),
                original_hw=(10, 10),
                golden_key="test_window_unpartition_b4_h10_w10_c4_ws10_rh10_rw10_oh10_ow10",
            ),
            dict(
                inputs_shape=[2, 2, 2, 4],
                window_size=2,
                resized_hw=(2, 2),
                original_hw=(1, 1),
                golden_key="test_window_unpartition_b2_h2_w2_c4_ws2_rh2_rw2_oh1_ow1",
            ),
            dict(
                inputs_shape=[2, 1, 1, 4],
                window_size=1,
                resized_hw=(1, 1),
                original_hw=(1, 1),
                golden_key="test_window_unpartition_b2_h1_w1_c4_ws1_rh1_rw1_oh1_ow1",
            ),
            dict(
                inputs_shape=[8, 10, 10, 4],
                window_size=10,
                resized_hw=(10, 20),
                original_hw=(8, 15),
                golden_key="test_window_unpartition_b8_h10_w10_c4_ws10_rh10_rw20_oh8_ow15",
            ),
        ]
    )
    def test_window_unpartition_with_window_size(
        self,
        inputs_shape,  # pylint: disable=unused-argument
        window_size,
        resized_hw,
        original_hw,
        golden_key,
    ):
        golden = load_golden(_MODULE_NAME, golden_key)
        inputs = golden["inputs"]["windows"]
        ref_outputs = golden["outputs"]["ref"]

        outputs = window_unpartition_with_window_size(inputs, window_size, resized_hw, original_hw)

        # Tests target output shape
        self.assertEqual(outputs.shape, ref_outputs.shape)
        # Tests window value close
        assert_allclose(outputs, ref_outputs)

    @parameterized.parameters(
        [
            dict(
                inputs_shape=[2, 112, 112, 3],
                num_windows=16,
            ),  # (512, 7, 7, 3), (112 112)
            dict(
                inputs_shape=[2, 112, 112, 3],
                num_windows=16,
            ),  # (512, 7, 7, 3), (112 112)
            dict(
                inputs_shape=[2, 112, 112, 3],
                num_windows=8,
            ),  # (128, 14, 14, 3), (112, 112)
            dict(
                inputs_shape=[2, 112, 112, 3],
                num_windows=4,
            ),  # (32, 28, 28, 3), (112, 112)
            dict(
                inputs_shape=[2, 112, 112, 3],
                num_windows=1,
            ),  # (2, 112, 112, 3), (112, 112)
            dict(
                inputs_shape=[2, 10, 10, 3],
                num_windows=3,
            ),  # (18, 3, 3, 3), (9, 9)
            dict(
                inputs_shape=[2, 112, 10, 3],
                num_windows=3,
            ),  # (18, 37, 3, 3), (111, 9)
            dict(
                inputs_shape=[2, 10, 10, 3],
                num_windows=10,
            ),  # (200, 1, 1, 3), (10, 10)
        ]
    )
    def test_window_partition_unpartition_with_num_windows(
        self,
        inputs_shape,
        num_windows,
    ):
        batch, height, width, channels = inputs_shape
        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [batch, height, width, channels])

        # Window partition specified by fixed numbers of windows
        output_windows, (resized_height, resized_width) = window_partition_with_num_windows(
            inputs, num_windows
        )

        # Window unpartition specified by fixed numbers of windows
        outputs = window_unpartition_with_num_windows(
            output_windows,
            num_windows,
            (resized_height, resized_width),
            (height, width),
        )
        # Tests input and output shape, should be equal
        self.assertEqual(inputs.shape, outputs.shape)
        assert_allclose(
            inputs[:, :resized_height, :resized_width, :],
            outputs[:, :resized_height, :resized_width, :],
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
