# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/detectron2:
# Copyright 2019-2020, detectron2 contributors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""An AXLearn implementation of ViTDet transformer window attention.

ViTDet References:
https://github.com/facebookresearch/detectron2/blob/d1f8accbc92c7c7e1c08e37d3ec9f6d1fc83d235/detectron2/modeling/backbone/utils.py#L16-L60
"""

from jax import numpy as jnp

from axlearn.common.utils import Tensor


def window_partition_with_window_size(
    # pylint: disable-next=redefined-builtin
    input: Tensor,
    window_size: int,
) -> tuple[Tensor, tuple[int, int]]:
    """Partitions input tensor into non-overlapping windows with given window size (padding).
    Args:
        input: The input Tensor with shape (batch, height, width, dim).
        window_size: The input window size used to split 2D features along spatial,
            following ViTDet, must be > 0.
    Returns:
        windows: Output Tensor after partition with shape
            (batch * num_windows, window_size, window_size, dim).
        (resized_height, resized_width): output padded height and width before partition.
    """
    batch, height, width, channels = input.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height > 0 or pad_width > 0:
        input = jnp.pad(input, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
    resized_height, resized_width = height + pad_height, width + pad_width

    input = jnp.reshape(
        input,
        (
            batch,
            resized_height // window_size,
            window_size,
            resized_width // window_size,
            window_size,
            channels,
        ),
    )
    windows = jnp.reshape(
        jnp.transpose(input, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, channels)
    )
    return windows, (resized_height, resized_width)


def window_partition_with_num_windows(
    # pylint: disable-next=redefined-builtin
    input: Tensor,
    num_windows: int,
) -> tuple[Tensor, tuple[int, int]]:
    """Partitions input tensor into non-overlapping windows with given number of windows (cropping).
    Args:
        input: The input Tensor with shape (batch, height, width, dim).
        num_windows: The number of windows is specified to split 2D features along spatial,
            following UViT, must be > 0.
    Returns:
        windows: Output Tensor after partition with shape
            (batch * num_windows, window_size, window_size, dim).
        (resized_height, resized_width): output cropped height and width before partition.
    """
    batch, height, width, channels = input.shape
    crop_height = height % num_windows
    crop_width = width % num_windows
    resized_height, resized_width = height - crop_height, width - crop_width
    if resized_height > 0 or resized_width > 0:
        input = input[:, :resized_height, :resized_width, :]
    # safeguard here to prevent resized_height // num_windows
    # or resized_width // num_windows < 1
    input = jnp.reshape(
        input,
        (
            batch,
            num_windows,
            max(resized_height // num_windows, 1),
            num_windows,
            max(resized_width // num_windows, 1),
            channels,
        ),
    )
    windows = jnp.reshape(
        jnp.transpose(input, (0, 1, 3, 2, 4, 5)),
        (-1, resized_height // num_windows, resized_width // num_windows, channels),
    )
    return windows, (resized_height, resized_width)


def window_unpartition_with_window_size(
    windows: Tensor,
    window_size: int,
    resized_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> Tensor:
    """Unpartition window tensor into original sequences and remove padding given window size.
    Args:
        windows: The input Tensor with shape [batch * num_windows, window_size, window_size, dim).
        window_size: The input window size used to split 2D features along spatial,
            following ViTDet, must be > 0.
        resized_hw: Padded height and width.
        original_hw: Original height and width before padding.
    Returns:
        x: Output unpartitioned Tensor with shape (batch, height, width, dim).
    """
    resized_height, resized_width = resized_hw
    height, width = original_hw
    batch = windows.shape[0] // (resized_height * resized_width // window_size // window_size)
    x = jnp.reshape(
        windows,
        (
            batch,
            resized_height // window_size,
            resized_width // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = jnp.reshape(
        jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (batch, resized_height, resized_width, -1)
    )

    if resized_height > height or resized_width > width:
        x = x[:, :height, :width, :]
    return x


def window_unpartition_with_num_windows(
    windows: Tensor,
    num_windows: int,
    resized_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> Tensor:
    """Unpartition window tensor into original sequences and remove padding given number of windows.
    Args:
        windows: The input Tensor with shape [batch * num_windows, window_size, window_size, dim).
        num_windows: The number of windows is specified to split 2D features along spatial,
            following UViT, must be > 0.
        resized_hw: Cropped height and width.
        original_hw: Original height and width before cropping.
    Returns:
        x: Output unpartitioned Tensor with shape (batch, height, width, dim).
    """
    resized_height, resized_width = resized_hw
    height, width = original_hw
    batch = windows.shape[0] // (num_windows**2)
    x = jnp.reshape(
        windows,
        (
            batch,
            num_windows,
            num_windows,
            resized_height // num_windows,
            resized_width // num_windows,
            -1,
        ),
    )
    x = jnp.reshape(
        jnp.transpose(x, (0, 1, 3, 2, 4, 5)), (batch, resized_height, resized_width, -1)
    )
    if height > resized_height or width > resized_width:
        x = jnp.pad(x, ((0, 0), (0, height % num_windows), (0, width % num_windows), (0, 0)))
    return x
