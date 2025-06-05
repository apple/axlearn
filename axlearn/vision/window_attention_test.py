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
import tensorflow as tf
import torch.nn.functional as F
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.vision.window_attention import (
    window_partition_with_num_windows,
    window_partition_with_window_size,
    window_unpartition_with_num_windows,
    window_unpartition_with_window_size,
)


def window_partition_torch(x, window_size):
    """Partition into non-overlapping windows with padding if needed.

    Reference:
    https://github.com/facebookresearch/detectron2/blob/d1f8accbc92c7c7e1c08e37d3ec9f6d1fc83d235/detectron2/modeling/backbone/utils.py#L16-L38

    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition_torch(windows, window_size, pad_hw, hw):
    """Window unpartition into original sequences and removing padding.

    Reference:
    https://github.com/facebookresearch/detectron2/blob/d1f8accbc92c7c7e1c08e37d3ec9f6d1fc83d235/detectron2/modeling/backbone/utils.py#L40-L60

    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class WindowAttentionTest(TestCase, tf.test.TestCase):
    """Tests window attention utils."""

    @parameterized.parameters(
        [
            dict(inputs_shape=[2, 100, 100, 3], window_size=14),  # (128, 14, 14, 3), (112, 112)
            dict(inputs_shape=[2, 100, 100, 3], window_size=7),  # (450, 7, 7, 3), (105, 105)
            dict(inputs_shape=[2, 224, 224, 8], window_size=14),  # (512, 14, 14, 8), (224, 224)
            dict(inputs_shape=[4, 7, 7, 8], window_size=10),  # (4, 10, 10, 8), (10, 10)
            dict(inputs_shape=[4, 10, 10, 4], window_size=10),  # (4, 10, 10, 4), (10, 10)
            dict(inputs_shape=[2, 1, 1, 4], window_size=2),  # (2, 2, 2, 4), (2, 2)
            dict(inputs_shape=[2, 1, 1, 4], window_size=1),  # (2, 1, 1, 4), (1, 1)
            dict(inputs_shape=[4, 8, 15, 4], window_size=10),  # (8, 10, 10, 4), (10, 20)
        ]
    )
    def test_window_partition_with_window_size(
        self,
        inputs_shape,
        window_size,
    ):
        batch, height, width, channels = inputs_shape

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [batch, height, width, channels])

        output_windows, (resized_height, resized_width) = window_partition_with_window_size(
            inputs, window_size
        )

        # Compute torch ref outputs.
        torch_inputs = as_torch_tensor(inputs)
        ref_output_windows, (ref_resized_height, ref_resized_width) = window_partition_torch(
            torch_inputs, window_size
        )
        ref_output_windows = ref_output_windows.detach().numpy()

        # Tests window output shape
        self.assertAllEqual(output_windows.shape, ref_output_windows.shape)
        self.assertAllEqual(
            (resized_height, resized_width), (ref_resized_height, ref_resized_width)
        )
        # Tests window value close
        assert_allclose(output_windows, ref_output_windows)

    @parameterized.parameters(
        [
            dict(
                inputs_shape=[128, 14, 14, 3],
                window_size=14,
                resized_hw=(112, 112),
                original_hw=(100, 100),
            ),
            dict(
                inputs_shape=[450, 7, 7, 3],
                window_size=7,
                resized_hw=(105, 105),
                original_hw=(100, 100),
            ),
            dict(
                inputs_shape=[512, 14, 14, 8],
                window_size=14,
                resized_hw=(224, 224),
                original_hw=(224, 224),
            ),
            dict(
                inputs_shape=[4, 10, 10, 8], window_size=10, resized_hw=(10, 10), original_hw=(7, 7)
            ),
            dict(
                inputs_shape=[4, 10, 10, 4],
                window_size=10,
                resized_hw=(10, 10),
                original_hw=(10, 10),
            ),
            dict(inputs_shape=[2, 2, 2, 4], window_size=2, resized_hw=(2, 2), original_hw=(1, 1)),
            dict(inputs_shape=[2, 1, 1, 4], window_size=1, resized_hw=(1, 1), original_hw=(1, 1)),
            dict(
                inputs_shape=[8, 10, 10, 4],
                window_size=10,
                resized_hw=(10, 20),
                original_hw=(8, 15),
            ),
        ]
    )
    def test_window_unpartition_with_window_size(
        self,
        inputs_shape,
        window_size,
        resized_hw,
        original_hw,
    ):
        batch, height, width, channels = inputs_shape

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [batch, height, width, channels])

        outputs = window_unpartition_with_window_size(inputs, window_size, resized_hw, original_hw)

        # Compute torch ref outputs.
        torch_inputs = as_torch_tensor(inputs)
        ref_outputs = window_unpartition_torch(torch_inputs, window_size, resized_hw, original_hw)
        ref_outputs = ref_outputs.detach().numpy()

        # Tests target output shape
        self.assertAllEqual(outputs.shape, ref_outputs.shape)
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
        self.assertAllEqual(inputs.shape, outputs.shape)
        assert_allclose(
            inputs[:, :resized_height, :resized_width, :],
            outputs[:, :resized_height, :resized_width, :],
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
