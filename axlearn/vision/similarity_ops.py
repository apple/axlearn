# Copyright © 2023 Apple Inc.

"""Box similarity metrics."""

from typing import Optional

import jax.numpy as jnp

from axlearn.common.utils import Tensor


def areas(boxes: Tensor) -> Tensor:
    """Computes area.

    Args:
        boxes: An N-d float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].

    Returns:
        A float tensor of shape [..., N] containing areas.
    """
    y_min, x_min, y_max, x_max = jnp.split(boxes, indices_or_sections=4, axis=-1)
    return jnp.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def pairwise_intersection_areas(*, boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Computes pairwise intersection.

    Args:
         boxes_a: An N-d float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         boxes_b: An N-d float tensor of shape [..., M, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].

    Returns:
        A float Tensor of shape [..., N, M] with pairwise intersections areas.
    """
    ymin_a, xmin_a, ymax_a, xmax_a = jnp.split(boxes_a, indices_or_sections=4, axis=-1)
    ymin_b, xmin_b, ymax_b, xmax_b = jnp.split(boxes_b, indices_or_sections=4, axis=-1)

    all_pairs_min_ymax = jnp.minimum(ymax_a[..., None, 0], ymax_b[..., None, :, 0])
    all_pairs_max_ymin = jnp.maximum(ymin_a[..., None, 0], ymin_b[..., None, :, 0])
    intersect_heights = jnp.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = jnp.minimum(xmax_a[..., None, 0], xmax_b[..., None, :, 0])
    all_pairs_max_xmin = jnp.maximum(xmin_a[..., None, 0], xmin_b[..., None, :, 0])
    intersect_widths = jnp.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

    return intersect_heights * intersect_widths


def pairwise_iou(
    *,
    boxes_a: Tensor,
    boxes_b: Tensor,
    paddings_a: Optional[Tensor] = None,
    paddings_b: Optional[Tensor] = None,
    fill_value: Optional[float] = -1.0,
) -> Tensor:
    """Computes pairwise intersection over union (IoU).

    Args:
         boxes_a: A float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         boxes_b: A float tensor of shape [..., M, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         paddings_a: A bool tensor of shape [..., N] indicating paddings in `boxes_a`.
         paddings_b: A bool tensor of shape [..., M] indicating paddings in `boxes_b`.
         fill_value: A float value to fill pair comparisons involving paddings.

    Returns:
        A float Tensor of shape [..., N, M] with pairwise IoU.
    """
    intersections = pairwise_intersection_areas(boxes_a=boxes_a, boxes_b=boxes_b)
    unions = areas(boxes_a)[..., None] + areas(boxes_b)[..., None, :] - intersections
    p_iou = jnp.where(intersections > 0.0, intersections / unions, 0.0)

    if paddings_a is None:
        paddings_a = jnp.zeros(boxes_a.shape[:-1], dtype=bool)
    if paddings_b is None:
        paddings_b = jnp.zeros(boxes_b.shape[:-1], dtype=bool)
    fill_loc = paddings_a[..., None] | paddings_b[..., None, :]

    return jnp.where(fill_loc, fill_value, p_iou)


def pairwise_ioa(
    *,
    boxes_a: Tensor,
    boxes_b: Tensor,
    paddings_a: Optional[Tensor] = None,
    paddings_b: Optional[Tensor] = None,
    fill_value: Optional[float] = -1.0,
) -> Tensor:
    """Computes pairwise intersection over area (IoA) of boxes_a.

    Args:
         boxes_a: A float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         boxes_b: A float tensor of shape [..., M, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         paddings_a: A bool tensor of shape [..., N] indicating paddings in `boxes_a`.
         paddings_b: A bool tensor of shape [..., M] indicating paddings in `boxes_b`.
         fill_value: A float value to fill pair comparisons involving paddings.

    Returns:
        A float Tensor of shape [..., N, M] with pairwise IoA.
    """
    intersections = pairwise_intersection_areas(boxes_a=boxes_a, boxes_b=boxes_b)
    area_a = areas(boxes_a)[..., None]
    # Set to zero where intersections=0 to avoid divide by zero error.
    p_iou = jnp.where(intersections > 0.0, intersections / area_a, 0.0)

    if paddings_a is None:
        paddings_a = jnp.zeros(boxes_a.shape[:-1], dtype=bool)
    if paddings_b is None:
        paddings_b = jnp.zeros(boxes_b.shape[:-1], dtype=bool)
    fill_loc = paddings_a[..., None] | paddings_b[..., None, :]

    return jnp.where(fill_loc, fill_value, p_iou)


def elementwise_intersection_areas(*, boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Computes elementwise intersection.

    Args:
         boxes_a: An float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         boxes_b: An float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].

    Returns:
        A float Tensor of shape [..., N] with elementwise intersections areas.
    """
    ymin_a, xmin_a, ymax_a, xmax_a = jnp.split(boxes_a, indices_or_sections=4, axis=-1)
    ymin_b, xmin_b, ymax_b, xmax_b = jnp.split(boxes_b, indices_or_sections=4, axis=-1)

    min_ymax = jnp.minimum(ymax_a[..., 0], ymax_b[..., 0])
    max_ymin = jnp.maximum(ymin_a[..., 0], ymin_b[..., 0])
    intersect_heights = jnp.maximum(0.0, min_ymax - max_ymin)

    min_xmax = jnp.minimum(xmax_a[..., 0], xmax_b[..., 0])
    max_xmin = jnp.maximum(xmin_a[..., 0], xmin_b[..., 0])
    intersect_widths = jnp.maximum(0.0, min_xmax - max_xmin)

    return intersect_heights * intersect_widths


def elementwise_iou(
    *,
    boxes_a: Tensor,
    boxes_b: Tensor,
    paddings_a: Optional[Tensor] = None,
    paddings_b: Optional[Tensor] = None,
    fill_value: Optional[float] = -1.0,
) -> Tensor:
    """Computes elementwise intersection over union (IoU).

    Args:
         boxes_a: A float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         boxes_b: A float tensor of shape [..., N, 4] with boxes in corner representation
            [ymin, xmin, ymax, xmax].
         paddings_a: A bool tensor of shape [..., N] indicating paddings in `boxes_a`.
         paddings_b: A bool tensor of shape [..., N] indicating paddings in `boxes_b`.
         fill_value: A float value to fill pair comparisons involving paddings.

    Returns:
        A float Tensor of shape [..., N] with elementwise IoU.
    """
    intersections = elementwise_intersection_areas(boxes_a=boxes_a, boxes_b=boxes_b)
    unions = areas(boxes_a) + areas(boxes_b) - intersections
    e_iou = jnp.where(intersections > 0.0, intersections / unions, 0.0)

    if paddings_a is None:
        paddings_a = jnp.zeros(boxes_a.shape[:-1], dtype=bool)
    if paddings_b is None:
        paddings_b = jnp.zeros(boxes_b.shape[:-1], dtype=bool)
    fill_loc = paddings_a | paddings_b

    return jnp.where(fill_loc, fill_value, e_iou)
