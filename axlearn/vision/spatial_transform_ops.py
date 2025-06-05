# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2022 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Spatial transform ops.

This module contains differentiable resampling ops in JAX.
Reference TensorFlow implementation:
https://github.com/tensorflow/models/blob/master/research/object_detection/utils/spatial_transform_ops.py
"""
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from axlearn.common.utils import Tensor


def _coordinate_vector(start: Tensor, end: Tensor, size: int, align_endpoints: bool) -> Tensor:
    """Generates uniformly spaced 1D coordinates.

    Args:
        start: A float tensor of shape [batch, num_boxes, 1] indicating start values.
        end: A float tensor of shape [batch, num_boxes, 1] indicating end values.
        size: Number of points in coordinate vector.
        align_endpoints: Whether to align first and last points exactly to endpoints.

    Returns:
        A 3D float tensor of shape [batch, num_boxes, size] containing grid coordinates.
    """
    length = end - start
    if align_endpoints:
        relative_grid_spacing = jnp.linspace(0.0, 1.0, size)
        offset = jnp.where(size > 1, 0, length / 2)
    else:
        relative_grid_spacing = jnp.linspace(0.0, 1.0, size + 1)[:-1]
        offset = length / (2 * size)
    relative_grid_spacing = jnp.reshape(relative_grid_spacing, (1, 1, size))
    absolute_coordinates = start + offset + relative_grid_spacing * length
    return absolute_coordinates


@dataclass
class BoxGrid:
    """Box Grid coordinates.

    y: A float tensor of shape [batch, num_boxes, size_y] containing y coordinates for grid points.
    x: A float tensor of shape [batch, num_boxes, size_x] containing x coordinates for grid points.
    """

    y: Tensor
    x: Tensor


def box_grid_coordinates(
    boxes: Tensor, *, size_y: int, size_x: int, align_corners: bool
) -> BoxGrid:
    """Generates coordinate vectors for a `size x size` grid in boxes.

    Each box is divided uniformly into a grid of size x size rectangular cells. This function
    returns coordinate vectors describing the center of each cell.

    If `align_corners` is true, grid points are uniformly spread such that the corner points on the
    grid exactly overlap corners of the boxes. Note that output coordinates are expressed in the
    same coordinate frame as input boxes.

    Args:
        boxes: A float tensor of shape [batch, num_boxes, 4] containing boxes of the form [ymin,
            xmin, ymax, xmax].
        size_y: Size of the grid in y axis.
        size_x: Size of the grid in x axis.
        align_corners: Whether to align the corner grid points exactly with box corners.

    Returns:
        A BoxGrid object containing box grid coordinates.
    """
    ymin, xmin, ymax, xmax = jnp.split(boxes, indices_or_sections=4, axis=-1)
    box_grid_y = _coordinate_vector(ymin, ymax, size_y, align_corners)
    box_grid_x = _coordinate_vector(xmin, xmax, size_x, align_corners)
    return BoxGrid(y=box_grid_y, x=box_grid_x)


@dataclass
class FeatureGrid:
    """Feature Grid coordinates.

    y0: An int32 tensor of shape [batch, num_boxes, size] containing y coordinate vector for the top
        neighbors.
    x0: An int32 tensor of shape [batch, num_boxes, size] containing x coordinate vector for the
        left neighbors.
    y1: An int32 tensor of shape [batch, num_boxes, size] containing y coordinate vector for the
        bottom neighbors.
    x1: An int32 tensor of shape [batch, num_boxes, size] containing x coordinate vector for the
        right neighbors.
    """

    y0: Tensor
    x0: Tensor
    y1: Tensor
    x1: Tensor


def feature_grid_coordinates(box_grid: BoxGrid, true_feature_shapes: Tensor) -> FeatureGrid:
    """Returns feature grid coordinates for bi-linear interpolation.

    Box grid is specified in absolute coordinate system with origin at left top (0, 0). The returned
    coordinate vectors contain 0-based feature point indices. This function snaps each point in the
    box grid to nearest 4 points on the feature map. Points that fall outside of the unpadded
    feature maps are clipped to the boundary.

    In this function we also follow the convention of treating feature pixels as point objects with
    no spatial extent.

    Args:
        box_grid: A BoxGrid object containing box grid coordinates.
        true_feature_shapes: A [batch, num_boxes, 2] tensor indicating height and width of the
            unpaded feature maps to which the boxes are mapped.

    Returns:
        A FeatureGrid object containing feature grid coordinates.
    """
    feature_grid_y0 = jnp.maximum(jnp.floor(box_grid.y), 0)
    feature_grid_x0 = jnp.maximum(jnp.floor(box_grid.x), 0)
    feature_grid_y0 = jnp.minimum(feature_grid_y0, true_feature_shapes[..., 0:1] - 1).astype(
        jnp.int32
    )
    feature_grid_x0 = jnp.minimum(feature_grid_x0, true_feature_shapes[..., 1:2] - 1).astype(
        jnp.int32
    )
    feature_grid_y1 = jnp.minimum(feature_grid_y0 + 1, true_feature_shapes[..., 0:1] - 1).astype(
        jnp.int32
    )
    feature_grid_x1 = jnp.minimum(feature_grid_x0 + 1, true_feature_shapes[..., 1:2] - 1).astype(
        jnp.int32
    )

    return FeatureGrid(
        y0=feature_grid_y0, x0=feature_grid_x0, y1=feature_grid_y1, x1=feature_grid_x1
    )


def get_box_levels(
    boxes: Tensor,
    *,
    min_level: int,
    max_level: int,
    unit_scale_level: int,
    pretraining_image_size: int,
) -> Tensor:
    """Returns FPN feature level for each box to resample features at the correct scale.

    See section 4.2 of https://arxiv.org/pdf/1612.03144.pdf for details.

    Args:
        boxes: A float tensor of shape [batch, num_boxes, 4] containing boxes of the form [ymin,
            xmin, ymax, xmax] in normalized coordinates.
        min_level: An integer indicating the smallest valid level for assignment.
        max_level: An integer indicating the largest valid level for assignment.
        unit_scale_level: Index of the feature map which most closely matches the resolution of the
            pretrained model.
        pretraining_image_size: Image size used for pre training. This is used to map the boxes to
            feature maps at the correct scale.

    Returns:
        An int32 tensor of shape [batch_size, num_boxes] containing feature level indices.
    """
    assert (
        min_level <= unit_scale_level <= max_level
    ), f"`unit_scale_index` must be in [{min_level}, {max_level}]. Found {unit_scale_level}."
    box_height_width = jnp.maximum(boxes[:, :, 2:4] - boxes[:, :, 0:2], 0)
    areas_sqrt = jnp.sqrt(jnp.prod(box_height_width, axis=2))
    log_of_scaled_area = jnp.where(
        areas_sqrt / pretraining_image_size >= 0.0,
        jnp.log2(areas_sqrt / pretraining_image_size),
        0.0,
    )
    levels = (jnp.floor(log_of_scaled_area) + unit_scale_level).astype(np.int32)
    levels = jnp.minimum(max_level, jnp.maximum(min_level, levels))
    return levels


@dataclass
class PaddedFeatures:
    """Container to hold padded features.

    features: A 5D float tensor of shape [batch, num_levels, max_height, max_width, channels]
        containing stacked features.
    true_shapes: A 2D int32 tensor of shape [num_levels, 2] containing height and width of the
        feature maps before padding.
    """

    features: Tensor
    true_shapes: Tensor


def pad_to_max_size(features: Sequence[Tensor]) -> PaddedFeatures:
    """Zero pads features to max height and max width and stacks them up.

    Args:
        features: A list of num_levels 4D float tensors of shape [batch, height_i, width_i,
            channels] containing feature maps.

    Returns:
        A PaddedFeatures object holding stacked padded features.
    """
    true_shapes = np.stack(list(map(lambda x: np.array(x.shape[1:3]), features)))
    max_height, max_width = np.max(true_shapes, axis=0)

    def pad_tensor(tensor):
        _, height, width, _ = tensor.shape
        paddings = ((0, 0), (0, max_height - height), (0, max_width - width), (0, 0))
        return jnp.pad(tensor, pad_width=paddings, mode="constant", constant_values=0)

    padded_features = jnp.stack(list(map(pad_tensor, features)), axis=1)
    return PaddedFeatures(features=padded_features, true_shapes=true_shapes)


def ravel_indices(
    *,
    feature_grid_y: Tensor,
    feature_grid_x: Tensor,
    num_levels: int,
    height: int,
    width: int,
    box_levels: Tensor,
) -> Tensor:
    R"""Returns grid indices in a flattened feature map of shape [-1, channels].

     The returned 1-D array can be used to gather feature grid points from a feature map that has
     been flattened from [batch, num_levels, max_height, max_width, channels] to [batch * num_levels
     * max_height * max_width, channels].

    Note: This helper method only converts multidimensional indexes into 1 dimensional index. It
    makes the RoIAlign function much more readable. Actual flattening of features happens in
    RoIAlign. Now as to why we flatten the features, it is an efficient way to resample features
    from multiple Feature Pyramid Network Layers simultaneously.

     Args:
         feature_grid_y: An int32 tensor of shape [batch, num_boxes, size_y] containing y coordinate
            vector.
         feature_grid_x: An int32 tensor of shape [batch, num_boxes, size_x] containing x coordinate
            vector.
         num_levels: Number of feature levels.
         height: Padded height of feature maps.
         width: Padded width of feature maps.
         box_levels: An int32 tensor of shape [batch, num_boxes] indicating feature level assigned
            to each box.

     Returns:
         indices: A 1D int32 tensor containing feature point indices in a flattened feature grid of
            shape [batch * num_levels * max_height * max_width, channels]. A 4 dimensional index
            [n_{1}, n_{2}, n_{3}, n_{4}] in the features tensor [batch (N_{1}), num_levels (N_{2}),
            max_height (N_{3}), max_width (N_{4}), channels (N_{5})] gets mapped to the linear index
            `\sum_{i=1}^{4}\left ( \prod_{j=1}^{i-1}N_{j} \right )n_{i}`.
    """
    batch_size, num_boxes = feature_grid_y.shape[0:2]
    size_y = feature_grid_y.shape[2]
    size_x = feature_grid_x.shape[2]
    height_dim_offset = width
    level_dim_offset = height * height_dim_offset
    batch_dim_offset = num_levels * level_dim_offset

    batch_dim_indices = jnp.reshape(
        jnp.arange(batch_size) * batch_dim_offset, [batch_size, 1, 1, 1]
    ) * jnp.ones([1, num_boxes, size_y, size_x], dtype=np.int32)
    box_level_indices = jnp.reshape(
        box_levels * level_dim_offset, [batch_size, num_boxes, 1, 1]
    ) * jnp.ones([1, 1, size_y, size_x], dtype=np.int32)
    height_indices = jnp.reshape(
        feature_grid_y * height_dim_offset, [batch_size, num_boxes, size_y, 1]
    ) * jnp.ones([1, 1, 1, size_x], dtype=np.int32)
    width_indices = jnp.reshape(feature_grid_x, [batch_size, num_boxes, 1, size_x]) * jnp.ones(
        [1, 1, size_y, 1], dtype=np.int32
    )
    indices = batch_dim_indices + box_level_indices + height_indices + width_indices
    flattened_indices = jnp.reshape(indices, [-1])
    return flattened_indices


def valid_coordinates(
    *, feature_grid_y: Tensor, feature_grid_x: Tensor, true_feature_shapes: Tensor
) -> Tensor:
    """Computes a indicator vector for valid indices.

    Computes an indicator vector which is true for points on feature map and false for points off
    feature map.

    Args:
        feature_grid_y: An int32 tensor of shape [batch, num_boxes, size_y] containing y coordinate
            vector.
        feature_grid_x: An int32 tensor of shape [batch, num_boxes, size_x] containing x coordinate
            vector.
        true_feature_shapes: A int32 tensor of shape [batch, num_boxes, 2] containing valid height
            and width of feature maps. Feature maps are assumed to be aligned to the left top
            corner.

    Returns:
        indices: A 1D bool tensor containing flattened valid feature indicator.
    """
    height = true_feature_shapes[:, :, 0:1].astype(feature_grid_y.dtype)
    width = true_feature_shapes[:, :, 1:2].astype(feature_grid_x.dtype)
    valid_indicator = ((feature_grid_y >= 0) & (feature_grid_y < height))[:, :, :, jnp.newaxis] & (
        (feature_grid_x >= 0) & (feature_grid_x < width)
    )[:, :, jnp.newaxis, :]
    return jnp.reshape(valid_indicator, [-1])


def gather_valid_indices(*, tensor: Tensor, indices: Tensor, padding_value: float = 0.0) -> Tensor:
    """Gather values for valid indices.

    Args:
        tensor: A multidimensional tensor to gather valid values from.
        indices: A 1-D int32 tensor containing indices along axis 0 of `tensor`. Invalid indices
            must be marked with -1.
        padding_value: Value to return for invalid indices.

    Returns:
        A tensor sliced based on indices. For indices that are equal to -1, returns rows of
        padding value.
    """
    padded_tensor = jnp.concatenate(
        [padding_value * jnp.ones((1,) + tensor.shape[1:], dtype=tensor.dtype), tensor],
        axis=0,
    )
    return padded_tensor[indices + 1]


def _feature_bilinear_interpolation(
    *,
    box_grid: BoxGrid,
    feature_grid: FeatureGrid,
    features_per_box: Tensor,
    num_samples_per_cell_y: int,
    num_samples_per_cell_x: int,
    output_size: tuple[int, int],
) -> Tensor:
    """Resamples features at box grid coordinates with bilinear interpolation.

    Args:
        box_grid: A BoxGrid object with box grid coordinates.
        feature_grid: A FeatureGrid object with feature grid coordinates.
        features_per_box: A float tensor of shape [batch_size, num_boxes, size_y, size_x] with per
            box features.
        num_samples_per_cell_y: Number of grid points to sample along y axis in each cell.
        num_samples_per_cell_x: Number of grid points to sample along x axis in each cell.
        output_size: An list of two integers [size_y, size_x] indicating the output feature size for
            each box.
    Returns:
        A [batch_size, num_boxes, output_size[0], output_size[1], num_filters] float tensor with
        resampled box features.
    """
    batch_size, num_boxes, size_y, size_x, num_filters = features_per_box.shape
    # Cast tensors into dtype of features.
    box_grid_y = box_grid.y.astype(features_per_box.dtype)
    box_grid_x = box_grid.x.astype(features_per_box.dtype)
    feature_grid_y0 = feature_grid.y0.astype(features_per_box.dtype)
    feature_grid_x0 = feature_grid.x0.astype(features_per_box.dtype)

    # RoI Align operation is a bilinear interpolation of four neighboring feature points f0, f1,
    # f2, and f3 onto point y, x given by
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    #
    # Unrolling the matrix multiplies gives us:
    # f(y, x) = (hy * hx) f00 + (hy * lx) f01 + (ly * hx) f10 + (lx * ly) f11
    # f(y, x) = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    #
    # This can be computed by applying point wise multiplication and sum_pool in a 2x2 window.
    ly = box_grid_y - feature_grid_y0
    lx = box_grid_x - feature_grid_x0
    hy = 1.0 - ly
    hx = 1.0 - lx

    kernel_y = jnp.reshape(jnp.stack([hy, ly], axis=3), [batch_size, num_boxes, size_y, 1])
    kernel_x = jnp.reshape(jnp.stack([hx, lx], axis=3), [batch_size, num_boxes, 1, size_x])

    # Denominator enables us to take the average of samples in each bin with a sum pool operation.
    interpolation_kernel = kernel_y * kernel_x / (num_samples_per_cell_y * num_samples_per_cell_x)

    # Interpolate the gathered features with computed interpolation kernels.
    features_per_box *= jnp.expand_dims(interpolation_kernel, axis=4)
    features_per_box = jnp.reshape(
        features_per_box, [batch_size * num_boxes, size_y, size_x, num_filters]
    )

    # This combines the two pooling operations - sum_pool to perform bilinear interpolation and
    # avg_pool to pool the values in each bin.
    features_per_box = jax.lax.reduce_window(
        features_per_box,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, num_samples_per_cell_y * 2, num_samples_per_cell_x * 2, 1),
        window_strides=(1, num_samples_per_cell_y * 2, num_samples_per_cell_x * 2, 1),
        padding="VALID",
    )

    features_per_box = jnp.reshape(
        features_per_box, [batch_size, num_boxes, output_size[0], output_size[1], num_filters]
    )

    return features_per_box


def roi_align(
    *,
    features: Sequence[Tensor],
    boxes: Tensor,
    box_levels: Tensor,
    output_size: tuple[int, int],
    num_samples_per_cell_y: int = 1,
    num_samples_per_cell_x: int = 1,
    align_corners: bool = False,
) -> Tensor:
    """Applies RoI Align and returns feature for boxes.

    Given multiple features maps indexed by different levels, and a set of boxes where each box is
    mapped to a certain level, this function selectively crops and resizes boxes from the
    corresponding feature maps.

    We follow the RoI Align technique in https://arxiv.org/pdf/1703.06870.pdf figure 3.
    Specifically, each box is subdivided uniformly into a grid consisting of output_size[0] x
    output_size[1] rectangular cells. Within each cell we select `num_points` points uniformly
    and compute feature values using bi-linear interpolation. Finally, we average pool the
    interpolated values in each cell to obtain a [output_size[0], output_size[1], channels] feature.

    If `align_corners` is true, sampling points are uniformly spread such that corner points
    exactly overlap corners of the boxes. In this function we also follow the convention of
    treating feature pixels as point objects with no spatial extent.

    Args:
        features: A list of 4D float tensors of shape [batch_size, height_i, width_i, channels]
            containing features. Note that each feature map must have the same batch_size and
            channels. These are typically multilevel features from a feature pyramid network.
        boxes: A 3D float tensor of shape [batch_size, num_boxes, 4] containing boxes of the form
            [ymin, xmin, ymax, xmax] in image coordinates.
        box_levels: An int32 tensor of shape [batch_size, num_boxes] representing the feature level
            index for each box.
        output_size: A list of two integers [size_y, size_x] indicating the output feature size for
            each box.
        num_samples_per_cell_y: Number of grid points to sample along y axis in each cell.
        num_samples_per_cell_x: Number of grid points to sample along x axis in each cell.
        align_corners: Whether to align the corner grid points exactly with box corners.

    Returns:
        A 5D float tensor of shape [batch_size, num_boxes, output_size[0], output_size[1], channels]
        representing the cropped features.
    """
    padded_features = pad_to_max_size(features)
    (
        batch_size,
        num_levels,
        max_feature_height,
        max_feature_width,
        num_filters,
    ) = padded_features.features.shape
    num_boxes = boxes.shape[1]

    true_feature_shapes = padded_features.true_shapes.astype(boxes.dtype)
    # [batch_size, num_boxes, 2]
    true_feature_shapes = jnp.take_along_axis(
        true_feature_shapes[None, ...], box_levels[..., None], axis=1
    )

    size_y = output_size[0] * num_samples_per_cell_y
    size_x = output_size[1] * num_samples_per_cell_x

    # box_grid.{y, x} are of the shape [batch, num_boxes, size_{y,x}].
    box_grid = box_grid_coordinates(
        boxes, size_y=size_y, size_x=size_x, align_corners=align_corners
    )
    # feature_grid.{y0, x0, y1, x1} are of the shape [batch, num_boxes, size]
    feature_grid = feature_grid_coordinates(box_grid, true_feature_shapes)
    feature_grid_y = jnp.reshape(
        jnp.stack([feature_grid.y0, feature_grid.y1], axis=3), [batch_size, num_boxes, -1]
    )
    feature_grid_x = jnp.reshape(
        jnp.stack([feature_grid.x0, feature_grid.x1], axis=3), [batch_size, num_boxes, -1]
    )
    # 1D tensor of shape [batch_size * num_boxes * size_y * size_x]
    feature_coordinates = ravel_indices(
        feature_grid_y=feature_grid_y,
        feature_grid_x=feature_grid_x,
        num_levels=num_levels,
        height=max_feature_height,
        width=max_feature_width,
        box_levels=box_levels,
    )
    valid_indices = valid_coordinates(
        feature_grid_y=feature_grid_y,
        feature_grid_x=feature_grid_x,
        true_feature_shapes=true_feature_shapes,
    )
    feature_coordinates = jnp.where(
        valid_indices, feature_coordinates, -1 * jnp.ones_like(feature_coordinates)
    )
    flattened_features = jnp.reshape(padded_features.features, [-1, num_filters])
    flattened_feature_values = gather_valid_indices(
        tensor=flattened_features, indices=feature_coordinates, padding_value=0.0
    )

    # For each point in output grid defined by [size_y, size_x] we have 4 nearby points from the
    # feature map. Hence, the `*2` factor.
    features_per_box = jnp.reshape(
        flattened_feature_values, [batch_size, num_boxes, size_y * 2, size_x * 2, num_filters]
    )
    return _feature_bilinear_interpolation(
        box_grid=box_grid,
        feature_grid=feature_grid,
        features_per_box=features_per_box,
        num_samples_per_cell_y=num_samples_per_cell_y,
        num_samples_per_cell_x=num_samples_per_cell_x,
        output_size=output_size,
    )
