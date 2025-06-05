# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License")

# baiivision/EVA:
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI).
# Licensed under The MIT License.
#
# facebookresearch/dinov2:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC-BY-NC 4.0 license.

"""Tests loss functions."""
# pylint: disable=too-many-lines
import math
import re
from collections.abc import Sequence
from typing import Optional

import chex
import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import optax
import pytest
import tensorflow as tf
import torch
from absl.testing import parameterized
from torch import nn
from torchvision.ops import generalized_box_iou_loss

from axlearn.common.loss import (
    ReductionMethod,
    _reduce_loss,
    asymmetric_contrastive_loss_from_features,
    asymmetric_contrastive_loss_from_logits,
    bilinear_mean_squared_error,
    binary_cross_entropy,
    categorical_hinge_loss,
    contrastive_logits,
    cross_entropy,
    flops_loss,
    focal_loss,
    giou_loss,
    huber_loss,
    kl_divergence,
    koleo_loss,
    l1_loss,
    large_margin_cosine_loss,
    mean_squared_error,
    negative_cosine_similarity_loss,
    ranking_pairwise_loss,
    symmetric_contrastive_loss_from_features,
    symmetric_contrastive_loss_from_logits,
)
from axlearn.common.metrics import WeightedScalar
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Tensor

# fix seed to make tests with random input values deterministic
np.random.seed(123)


def test_cross_entropy():
    logits = jnp.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    targets = jnp.asarray([0, 1, 2])
    assert jnp.allclose(cross_entropy(logits, targets)[0], 0.0)


def test_accuracy():
    pad_token_id = 0
    logits = jnp.asarray(
        [
            # Predicted labels: 1, 2, 0.
            [[0.0, 100.0, 0.0], [0.0, 0.0, 100.0], [100.0, 0.0, 0.0]],
            # Predicted labels: 1, 0, 0.
            [[0.0, 100.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        ]
    )
    targets = jnp.asarray(
        [
            [1, 2, pad_token_id],
            [2, pad_token_id, pad_token_id],
        ]
    )
    live_targets = (targets != pad_token_id).astype(jnp.float32)
    _, loss_dict = cross_entropy(logits, targets, live_targets=live_targets)
    assert jnp.allclose(loss_dict["accuracy"], 2 / 3)


def test_binary_cross_entropy():
    logits = jnp.asarray([[100.0, 5.0, 0.0], [0.0, 100.0, 25.0], [0.0, 2000.0, 100.0]])
    targets = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    live_targets = jnp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    tf_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tf_cross_entropy = tf_bce(targets, logits)
    assert jnp.allclose(
        binary_cross_entropy(logits, target_labels=targets, live_targets=live_targets)[0],
        tf_cross_entropy.numpy(),
    )

    logits = jnp.asarray([[100.0, 5.0, 0.0], [5.0, 100.0, 25.0], [0.0, 2000.0, 100.0]])
    targets = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    live_targets = jnp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    tf_cross_entropy = tf_bce(targets, logits)
    assert jnp.allclose(
        binary_cross_entropy(logits, target_labels=targets, live_targets=live_targets)[0],
        tf_cross_entropy.numpy(),
    )


def test_binary_cross_entropy_zero_loss():
    logits = jnp.asarray(
        [[100.0, -100.0, -100.0], [-100.0, 100.0, -100.0], [-100.0, -100.0, 100.0]]
    )
    targets = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    live_targets = jnp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    tf_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tf_cross_entropy = tf_bce(targets, logits)
    assert jnp.allclose(
        binary_cross_entropy(logits, target_labels=targets, live_targets=live_targets)[0],
        tf_cross_entropy.numpy(),
    )


def test_cross_entropy_with_label_smoothing():
    logits = jnp.asarray([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 100.0, 0.0]])
    targets = jnp.asarray([0, 1, 2])
    # After label smoothing, the target label for 1st example is [0.7, 0.1, 0.1, 0.1].
    # Each example should get loss = 0.1 * -100 * -1 * 3 = 30.
    assert jnp.allclose(cross_entropy(logits, targets, label_smoothing=0.4)[0], 30.0)


def test_cross_entropy_with_less_than_0_label_smoothing_exception():
    logits = jnp.asarray([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 100.0, 0.0]])
    targets = jnp.asarray([0, 1, 2])
    with pytest.raises(
        ValueError, match=re.escape("label_smoothing should be in the range [0, 1].")
    ):
        cross_entropy(logits, targets, label_smoothing=-0.1)


def test_cross_entropy_with_greater_than_1_label_smoothing_exception():
    logits = jnp.asarray([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 100.0, 0.0]])
    targets = jnp.asarray([0, 1, 2])
    with pytest.raises(
        ValueError, match=re.escape("label_smoothing should be in the range [0, 1].")
    ):
        cross_entropy(logits, targets, label_smoothing=1.1)


def test_cross_entropy_with_0_num_classes_exception():
    logits = jnp.asarray([[], [], []])
    targets = jnp.asarray([0, 1, 2])
    with pytest.raises(ValueError, match=re.escape("num classes should be at least 1.")):
        cross_entropy(logits, targets, label_smoothing=0.1)


def test_cross_entropy_with_less_than_0_z_loss_scale_exception():
    logits = jnp.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    targets = jnp.asarray([0, 1, 2])
    with pytest.raises(ValueError, match=re.escape("z_loss_scale should not be negative.")):
        cross_entropy(logits, targets, z_loss_scale=-1)


def _standard_cross_entropy(
    logits: Tensor,
    target_labels: Tensor,
    live_targets: Tensor = None,
    z_loss_scale: float = 0.0,
    soft_target_labels: Tensor = None,
) -> Tensor:
    """Reference cross entropy implementation."""
    num_classes = logits.shape[-1]
    cross_entropy_loss = -(
        jax.nn.log_softmax(logits)
        * (
            jax.nn.one_hot(target_labels, num_classes)
            if soft_target_labels is None
            else soft_target_labels
        )
    ).sum(axis=-1)

    if live_targets is None:
        live_targets = jnp.logical_and(0 <= target_labels, target_labels < num_classes)
    # Z-loss.
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_z = jnp.squeeze(logits_sum, axis=-1)
    total_z_loss = jax.lax.square(log_z)

    per_example_loss = cross_entropy_loss + total_z_loss * z_loss_scale

    live_targets = live_targets.astype(per_example_loss.dtype)
    return (per_example_loss * live_targets).sum() / jnp.maximum(live_targets.sum(), 1)


def test_cross_entropy_z_loss():
    logits = jnp.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    targets = jnp.asarray([0, 1, 2])
    assert jnp.allclose(cross_entropy(logits, targets, z_loss_scale=1)[0], 100**2)

    # Test the gradient.
    grad_z_0 = jax.grad(cross_entropy, has_aux=True)(logits, targets, z_loss_scale=0)[0]
    grad_ref = jax.grad(_standard_cross_entropy)(logits, targets)
    grad_z_1 = jax.grad(cross_entropy, has_aux=True)(logits, targets, z_loss_scale=1)[0]
    assert jnp.allclose(grad_z_0, grad_ref)
    assert not jnp.allclose(grad_z_1, grad_ref)


def test_cross_entropy_soft_target_gradient():
    logits = jnp.asarray([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    targets = jnp.asarray([0, 1, 2])
    soft_target_labels = jnp.asarray([[0.8, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.7]])

    grad = jax.grad(cross_entropy, has_aux=True)(
        logits, targets, soft_target_labels=soft_target_labels
    )[0]
    grad_ref = jax.grad(_standard_cross_entropy)(
        logits, targets, soft_target_labels=soft_target_labels
    )
    assert jnp.allclose(grad, grad_ref)


def test_cross_entropy_soft_target_with_z_loss_gradient():
    logits = jnp.asarray([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    targets = jnp.asarray([0, 1, 2])
    soft_target_labels = jnp.asarray([[0.8, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.7]])

    grad = jax.grad(cross_entropy, has_aux=True)(
        logits, targets, z_loss_scale=1.0, soft_target_labels=soft_target_labels
    )[0]
    grad_ref = jax.grad(_standard_cross_entropy)(
        logits, targets, z_loss_scale=1.0, soft_target_labels=soft_target_labels
    )
    assert jnp.allclose(grad, grad_ref)


def test_cross_entropy_live_targets():
    logits = jnp.asarray([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    targets = jnp.asarray([0, 1, 2])
    live_targets = jnp.asarray([1, 0, 0])
    assert cross_entropy(logits, targets)[0] > 0.0
    assert jnp.allclose(cross_entropy(logits, targets, live_targets=live_targets)[0], 0.0)


# TODO(jbiloki): Convert tf style masking to have attribute _keras_mask
# to allow internal tf logic to mask requires keras >= 2.11.0.
def test_binary_cross_entropy_live_targets():
    logits = jnp.asarray([100.0, -100.0, -100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    targets = jnp.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    live_targets = jnp.asarray([1, 1, 1, 0, 0, 0, 1, 1, 0])
    tf_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tf_cross_entropy = tf_bce(targets[live_targets > 0], logits[live_targets > 0])
    assert jnp.allclose(
        binary_cross_entropy(logits, target_labels=targets, live_targets=live_targets)[0],
        tf_cross_entropy.numpy(),
    )

    logits = jnp.asarray([100.0, -100.0, -100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    targets = jnp.asarray([10.0, -1.0, -5.0, 0.0, 1.0, 0.0, 3.0, 1.0, 1.0])
    live_targets = jnp.logical_and(0 <= targets, targets < 2)
    tf_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tf_cross_entropy = tf_bce(targets[live_targets], logits[live_targets])
    assert jnp.allclose(
        binary_cross_entropy(logits, target_labels=targets)[0], tf_cross_entropy.numpy()
    )


def test_binary_cross_entropy_single():
    logits = jnp.asarray([[100.0], [-100], [100.0]])
    targets = jnp.asarray([[1.0], [0.0], [1.0]])
    live_targets = jnp.asarray([[1], [1], [1]])
    tf_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tf_cross_entropy = tf_bce(targets, logits)
    assert jnp.allclose(
        binary_cross_entropy(logits, target_labels=targets, live_targets=live_targets)[0],
        tf_cross_entropy.numpy(),
    )


def test_cross_entropy_valid_targets():
    logits = jnp.asarray([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    targets = jnp.asarray([0, -1, 100])

    assert jnp.allclose(cross_entropy(logits, targets)[0], 0.0)


def test_cross_entropy_soft_target_labels():
    logits = jnp.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    targets = jnp.asarray([-1, 0, 1])
    soft_labels = jnp.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )
    # When soft labels are provided, targets are only used for masking.
    # For just the unmasked examples, we have:
    # -((0.1 * -100 + 0.1 * -100) + (0.1 * -100 + 0.8 * -100)) / 2 = 55.
    assert jnp.allclose(cross_entropy(logits, targets, soft_target_labels=soft_labels)[0], 55.0)

    with pytest.raises(ValueError, match="smoothing should be set to 0"):
        cross_entropy(logits, targets, soft_target_labels=soft_labels, label_smoothing=0.1)


@pytest.mark.parametrize(
    "preds,targets,expected,weights",
    [
        ([], [], WeightedScalar(0, 0), None),  # No targets.
        (
            [0.0, 1.0, 2.0],
            [jnp.nan, jnp.nan, jnp.nan],
            WeightedScalar(0, 0),
            [False, False, False],
        ),  # No targets.
        ([jnp.inf], [0], WeightedScalar(jnp.inf, 1), None),  # inf - 0 is inf.
        ([jnp.inf], [jnp.inf], WeightedScalar(jnp.nan, 1), None),  # inf - inf is nan.
        (
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            WeightedScalar(1.0, 5),
            None,
        ),  # Basic case.
        (
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [2.0, jnp.nan, 2.0, 2.0, jnp.nan],
            WeightedScalar(5.0 / 3.0, 3),
            [True, False, True, True, False],
        ),  # Basic case.
        (
            [7, 5, 11],
            [3, 2, 13],
            WeightedScalar(12, 3),
            [2, 0, 1],
        ),  # Test non-boolean weights.
    ],
)
def test_mean_squared_error(
    preds: Sequence[float],
    targets: Sequence[float],
    expected: WeightedScalar,
    weights: Optional[Sequence[float]],
):
    preds = jnp.asarray(preds)
    targets = jnp.asarray(targets)
    if weights is not None:
        weights = jnp.asarray(weights)

    # Makes sure compat with jit.
    actual = jax.jit(mean_squared_error)(preds, targets, weights)
    if not (jnp.isnan(actual.mean) and jnp.isnan(expected.mean)):
        chex.assert_trees_all_close(actual.mean, expected.mean)
    chex.assert_trees_all_close(actual.weight, expected.weight)


def test_l1_loss():
    actual = l1_loss(jnp.array([1, 2, 3]), jnp.array([5, 7, 13]))
    expected = WeightedScalar((4 + 5 + 10) / 3, 3)
    chex.assert_trees_all_close(actual.mean, expected.mean)


def test_bilinear_mean_squared_error():
    pred = jnp.arange(16).reshape(4, 4)[None]
    target = (jnp.arange(16) ** 2).reshape(4, 4)[None]
    actual = bilinear_mean_squared_error(pred, target, shape=(1, 2, 2))
    diff = pred - target
    groups = jnp.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    expected = jnp.mean(diff.flatten()[groups].mean(axis=1) ** 2)
    chex.assert_trees_all_close(actual.mean, expected)


def _ref_asymmetric_contrastive_loss_from_features(
    x, y, temperature: float = 1.0, soft_labels: Tensor = None
):
    return _standard_cross_entropy(
        jnp.einsum("i d, j d -> i j", x, y) / temperature,
        jnp.arange(x.shape[0]),
        soft_target_labels=soft_labels,
    )


def _ref_asymmetric_contrastive_loss_from_logits(
    logits, temperature: float = 1.0, soft_labels: Tensor = None
):
    return _standard_cross_entropy(
        logits / temperature,
        jnp.arange(logits.shape[0]),
        soft_target_labels=soft_labels,
    )


def test_asymmetric_contrastive_loss_from_logits():
    logits = jnp.asarray([[10.0, -10.0], [-10.0, 10.0]])
    loss = asymmetric_contrastive_loss_from_logits(logits)
    assert jnp.equal(loss, _ref_asymmetric_contrastive_loss_from_logits(logits))


def test_asymmetric_contrastive_loss_from_logits_with_paddings():
    logits = jnp.asarray([[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]])
    logits_with_paddings = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    loss = asymmetric_contrastive_loss_from_logits(
        logits_with_paddings, key_paddings=jnp.asarray([0, 0, 1, 0])
    )
    assert jnp.allclose(loss, _ref_asymmetric_contrastive_loss_from_logits(logits))


def test_asymmetric_contrastive_loss_from_logits_with_soft_labels():
    logits = jnp.asarray([[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]])
    logits_with_paddings = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    soft_labels = jnp.asarray([[0.1, 0.2, 0.4], [0.4, 0.5, 0.7]])
    soft_labels_with_paddings = jnp.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]])
    loss = asymmetric_contrastive_loss_from_logits(
        logits_with_paddings,
        key_paddings=jnp.asarray([0, 0, 1, 0]),
        soft_labels=soft_labels_with_paddings,
    )
    assert jnp.allclose(
        loss,
        _ref_asymmetric_contrastive_loss_from_logits(
            logits,
            soft_labels=soft_labels,
        ),
    )


def test_asymmetric_contrastive_loss_from_features():
    queries = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    positive_keys = jnp.asarray([[-2.0, -2.0], [-2.0, 1.0]])
    negative_keys = jnp.asarray([[3.0, 4.0], [5.0, 6.0]])
    loss = asymmetric_contrastive_loss_from_features(
        queries, positive_keys, negative_keys=negative_keys
    )
    assert jnp.equal(
        loss,
        _ref_asymmetric_contrastive_loss_from_features(
            queries, jnp.vstack([positive_keys, negative_keys])
        ),
    )


def test_asymmetric_contrastive_loss_from_features_with_paddings():
    queries = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    positive_keys = jnp.asarray([[-2.0, -2.0], [-2.0, 1.0]])
    negative_keys = jnp.asarray([[3.0, 4.0], [5.0, 6.0]])
    negative_keys_with_paddings = jnp.asarray([[3.0, 4.0], [-1.0, -1.0], [5.0, 6.0]])
    loss = asymmetric_contrastive_loss_from_features(
        queries,
        positive_keys,
        negative_keys=negative_keys_with_paddings,
        negative_key_paddings=jnp.asarray([0, 1, 0]),
    )
    assert jnp.allclose(
        loss,
        _ref_asymmetric_contrastive_loss_from_features(
            queries, jnp.vstack([positive_keys, negative_keys])
        ),
    )


def test_asymmetric_contrastive_loss_from_features_with_soft_labels():
    queries = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    positive_keys = jnp.asarray([[-2.0, -2.0], [-2.0, 1.0]])
    negative_keys = jnp.asarray([[3.0, 4.0], [5.0, 6.0]])
    negative_keys_with_paddings = jnp.asarray([[3.0, 4.0], [-1.0, -1.0], [5.0, 6.0]])
    positive_key_soft_labels = jnp.asarray([[0.4, 0.2], [0.1, 0.2]])
    negative_key_soft_labels = jnp.asarray([[0.5, 0.4], [0.3, 0.2]])
    negative_key_soft_labels_with_paddings = jnp.asarray([[0.5, 0.2, 0.4], [0.3, 0.7, 0.2]])
    loss = asymmetric_contrastive_loss_from_features(
        queries,
        positive_keys,
        negative_keys=negative_keys_with_paddings,
        negative_key_paddings=jnp.asarray([0, 1, 0]),
        soft_labels=jnp.hstack([positive_key_soft_labels, negative_key_soft_labels_with_paddings]),
    )
    assert jnp.allclose(
        loss,
        _ref_asymmetric_contrastive_loss_from_features(
            queries,
            jnp.vstack([positive_keys, negative_keys]),
            soft_labels=jnp.hstack([positive_key_soft_labels, negative_key_soft_labels]),
        ),
    )


def test_symmetric_contrastive_loss_from_features():
    x = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    y = jnp.asarray([[-2.0, -2.0], [-2.0, 1.0]])
    x_negatives = y_negatives = jnp.asarray([[3.0, 4.0], [5.0, 6.0]])
    loss = symmetric_contrastive_loss_from_features(
        x, y, x_negatives=x_negatives, y_negatives=y_negatives
    )
    x_with_negatives = jnp.vstack([x, x_negatives])
    y_with_negatives = jnp.vstack([y, y_negatives])
    ref_loss = (
        _ref_asymmetric_contrastive_loss_from_features(x, y_with_negatives)
        + _ref_asymmetric_contrastive_loss_from_features(y, x_with_negatives)
    ) / 2.0
    assert jnp.equal(loss, ref_loss)


def test_symmetric_contrastive_loss_from_logits_with_paddings():
    x_y_logits = jnp.asarray([[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]])
    x_y_logits_with_paddings = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])

    y_x_logits = jnp.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    y_x_logits_with_paddings = jnp.asarray([[1.0, 2.0, 3.0, -1.0], [-1.0, -2.0, -3.0, -1.0]])

    loss = symmetric_contrastive_loss_from_logits(
        x_y_logits_with_paddings,
        y_x_logits_with_paddings,
        y_as_key_paddings=jnp.asarray([0, 0, 1, 0]),
        x_as_key_paddings=jnp.asarray([0, 0, 0, 1]),
    )

    ref_loss = (
        _ref_asymmetric_contrastive_loss_from_logits(x_y_logits)
        + _ref_asymmetric_contrastive_loss_from_logits(y_x_logits)
    ) / 2.0
    assert jnp.allclose(loss, ref_loss)


def test_symmetric_contrastive_loss_from_logits_with_soft_labels():
    x_y_logits = jnp.asarray([[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]])
    x_y_logits_with_paddings = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])

    y_x_logits = jnp.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    y_x_logits_with_paddings = jnp.asarray([[1.0, 2.0, 3.0, -1.0], [-1.0, -2.0, -3.0, -1.0]])

    y_as_key_soft_labels = jnp.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    y_as_key_soft_labels_with_paddings = jnp.asarray([[0.1, 0.2, 99, 0.3], [0.3, 0.2, 99, 0.1]])
    x_as_key_soft_labels = jnp.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    x_as_key_soft_labels_with_paddings = jnp.asarray([[0.1, 0.2, 0.3, 7], [0.3, 0.2, 0.1, 7]])
    loss = symmetric_contrastive_loss_from_logits(
        x_y_logits_with_paddings,
        y_x_logits_with_paddings,
        y_as_key_paddings=jnp.asarray([0, 0, 1, 0]),
        x_as_key_paddings=jnp.asarray([0, 0, 0, 1]),
        y_as_key_soft_labels=y_as_key_soft_labels_with_paddings,
        x_as_key_soft_labels=x_as_key_soft_labels_with_paddings,
    )

    ref_loss = (
        _ref_asymmetric_contrastive_loss_from_logits(x_y_logits, soft_labels=y_as_key_soft_labels)
        + _ref_asymmetric_contrastive_loss_from_logits(y_x_logits, soft_labels=x_as_key_soft_labels)
    ) / 2.0
    assert jnp.allclose(loss, ref_loss)


def test_symmetric_contrastive_loss_from_features_with_paddings():
    x = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    y = jnp.asarray([[-2.0, -2.0], [-2.0, 1.0]])
    x_negatives = y_negatives = jnp.asarray([[3.0, 4.0], [5.0, 6.0]])
    x_negatives_with_paddings = jnp.asarray([[-1.0, -1.0], [3.0, 4.0], [5.0, 6.0]])
    y_negatives_with_paddings = jnp.asarray([[3.0, 4.0], [5.0, 6.0], [-1.0, -1.0]])
    loss = symmetric_contrastive_loss_from_features(
        x,
        y,
        x_negatives=x_negatives_with_paddings,
        y_negatives=y_negatives_with_paddings,
        x_negative_paddings=jnp.asarray([1, 0, 0]),
        y_negative_paddings=jnp.asarray([0, 0, 1]),
    )
    x_with_negatives = jnp.vstack([x, x_negatives])
    y_with_negatives = jnp.vstack([y, y_negatives])
    ref_loss = (
        _ref_asymmetric_contrastive_loss_from_features(x, y_with_negatives)
        + _ref_asymmetric_contrastive_loss_from_features(y, x_with_negatives)
    ) / 2.0
    assert jnp.allclose(loss, ref_loss)


def test_symmetric_contrastive_loss_from_features_with_soft_labels():
    x = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    y = jnp.asarray([[-2.0, -2.0], [-2.0, 1.0]])
    x_negatives = y_negatives = jnp.asarray([[3.0, 4.0], [5.0, 6.0]])
    x_negatives_with_paddings = jnp.asarray([[-1.0, -1.0], [3.0, 4.0], [5.0, 6.0]])
    y_negatives_with_paddings = jnp.asarray([[3.0, 4.0], [5.0, 6.0], [-1.0, -1.0]])
    y_as_key_soft_labels = jnp.asarray([[0.4, 0.3], [0.2, 0.1]])
    x_as_key_soft_labels = jnp.asarray([[0.1, 0.3], [0.4, 0.1]])
    x_negatives_soft_labels = jnp.asarray([[0.3, 0.1], [0.1, 0.5]])
    x_negatives_soft_labels_with_paddings = jnp.asarray([[0.4, 0.3, 0.1], [0.2, 0.1, 0.5]])
    y_negatives_soft_labels = jnp.asarray([[0.4, 0.3], [0.2, 0.1]])
    y_negatives_soft_labels_with_paddings = jnp.asarray([[0.4, 0.3, 0.1], [0.2, 0.1, 0.5]])

    loss = symmetric_contrastive_loss_from_features(
        x,
        y,
        x_negatives=x_negatives_with_paddings,
        y_negatives=y_negatives_with_paddings,
        x_negative_paddings=jnp.asarray([1, 0, 0]),
        y_negative_paddings=jnp.asarray([0, 0, 1]),
        y_as_key_soft_labels=jnp.hstack(
            [y_as_key_soft_labels, y_negatives_soft_labels_with_paddings]
        ),
        x_as_key_soft_labels=jnp.hstack(
            [x_as_key_soft_labels, x_negatives_soft_labels_with_paddings]
        ),
    )
    x_with_negatives = jnp.vstack([x, x_negatives])
    x_y_soft_labels = jnp.hstack([y_as_key_soft_labels, y_negatives_soft_labels])
    y_x_soft_labels = jnp.hstack([x_as_key_soft_labels, x_negatives_soft_labels])
    y_with_negatives = jnp.vstack([y, y_negatives])
    ref_loss = (
        _ref_asymmetric_contrastive_loss_from_features(
            x, y_with_negatives, soft_labels=x_y_soft_labels
        )
        + _ref_asymmetric_contrastive_loss_from_features(
            y, x_with_negatives, soft_labels=y_x_soft_labels
        )
    ) / 2.0
    assert jnp.allclose(loss, ref_loss)


def test_contrastive_logits_error():
    with pytest.raises(ValueError):
        x = jnp.asarray(np.random.rand(3, 10))
        y = jnp.asarray(np.random.rand(4, 8))
        contrastive_logits(x, y)

    # TODO(adesai22): Reactivate check for constrastive loss when temperature <= 0.
    # Currently, this is not supported in jit so we do not raise an error.
    # with pytest.raises(ValueError):
    #     x = y = jnp.asarray(np.random.rand(3, 10))
    #     contrastive(x, y, 0)

    # with pytest.raises(ValueError):
    #     x = y = jnp.asarray(np.random.rand(3, 10))
    #     contrastive(x, y, -1)


def test_categorical_hinge_loss():
    batch_size = 8
    num_samples = 128
    num_classes = 20
    logits = np.random.uniform(0, 1, size=[batch_size, num_samples, num_classes]).astype(np.float32)
    targets = np.random.randint(0, num_classes, size=[batch_size, num_samples]).astype(np.int32)
    targets = jax.nn.one_hot(targets, num_classes)

    loss = categorical_hinge_loss(logits, targets)

    # Compare with the hinge_loss from Keras implementation.
    # Note keras arg order is different from axlearn.
    ref_loss = tf.keras.losses.categorical_hinge(targets, logits)

    assert jnp.allclose(loss, ref_loss.numpy())


@pytest.mark.parametrize("sample_weight_dim", [0, 1, 3])
def test_focal_loss(sample_weight_dim):
    batch_size = 8
    num_classes = 91
    num_samples = 1000
    logits = np.random.uniform(-1, 1, size=[batch_size, num_samples, num_classes]).astype(
        np.float32
    )
    targets = np.random.randint(0, num_classes, size=[batch_size, num_samples]).astype(np.int32)
    targets = jax.nn.one_hot(targets, num_classes=num_classes)

    if sample_weight_dim == 3:
        sample_weight = (
            np.random.uniform(-1, 1, size=[batch_size, num_samples, 1]).astype(np.float32) >= 0.0
        )
        sample_weight = sample_weight / sample_weight.sum()
    elif sample_weight_dim == 1:
        sample_weight = np.random.uniform(-1, 1, size=[]).astype(np.float32)
    else:
        sample_weight = None

    loss = focal_loss(logits, targets, alpha=0.25, gamma=1.5, sample_weight=sample_weight)

    class FocalLoss(tf.keras.losses.Loss):
        """TF implementation of Focal loss to test against.

        Reference:
        https://github.com/tensorflow/models/blob/master/official/vision/losses/focal_loss.py
        """

        def __init__(self, alpha, gamma, reduction=tf.keras.losses.Reduction.AUTO, name=None):
            self._alpha = alpha
            self._gamma = gamma
            super().__init__(reduction=reduction, name=name)

        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            positive_label_mask = tf.equal(y_true, 1.0)
            cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true, logits=y_pred
            )
            probs = tf.sigmoid(y_pred)
            probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
            modulator = tf.pow(1.0 - probs_gt, self._gamma)
            loss = modulator * cross_entropy_loss
            weighted_loss = tf.where(
                positive_label_mask, self._alpha * loss, (1.0 - self._alpha) * loss
            )
            return weighted_loss

    ref_fn = FocalLoss(alpha=0.25, gamma=1.5, reduction=tf.keras.losses.Reduction.SUM)
    ref_loss = ref_fn(y_true=targets, y_pred=logits, sample_weight=sample_weight)
    assert jnp.allclose(loss, ref_loss.numpy())


@pytest.mark.parametrize(
    "reduce_axis, sample_weight_dim, reduction",
    [
        (None, 0, ReductionMethod.SUM),
        (None, 0, ReductionMethod.MEAN),
        (-1, 0, ReductionMethod.SUM),
        (-1, 0, ReductionMethod.MEAN),
        (-1, 1, ReductionMethod.SUM),
        (-1, 1, ReductionMethod.MEAN),
        (-1, 3, ReductionMethod.SUM),
        (-1, 3, ReductionMethod.MEAN),
    ],
)
def test_huber_loss(reduce_axis, sample_weight_dim, reduction):
    batch_size = 8
    num_samples = 1000
    logits = np.random.uniform(-1, 1, size=[batch_size, num_samples, 4]).astype(np.float32)
    targets = np.random.uniform(-1, 1, size=[batch_size, num_samples, 4]).astype(np.float32)

    if sample_weight_dim == 3:
        sample_weight = (
            np.random.uniform(-1, 1, size=[batch_size, num_samples]).astype(np.float32) >= 0.0
        )
        sample_weight = sample_weight / sample_weight.sum()
    elif sample_weight_dim == 1:
        sample_weight = np.random.uniform(-1, 1, size=[]).astype(np.float32)
    else:
        sample_weight = None

    loss = huber_loss(
        predictions=logits,
        targets=targets,
        delta=1.0,
        reduce_axis=reduce_axis,
        sample_weight=sample_weight,
        reduction=reduction,
    )

    # Reference from tf.keras.losses.Huber, performing reduce_mean on the last dim by default.
    if reduce_axis == -1:
        ref_fn = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.SUM)
        ref_loss = ref_fn(targets, logits, sample_weight=sample_weight)
        denominator = 1
        if reduction == ReductionMethod.MEAN:
            if sample_weight is not None:
                denominator = sample_weight.sum()
            else:
                denominator = batch_size * num_samples
        assert jnp.allclose(loss, ref_loss.numpy() / denominator)


def reference_flops_loss_calculation(embeddings, sparsity_threshold=0.0):
    # Ref: https://github.com/biswajitsc/sparse-embed/blob/master/model.py#L134
    abs_embeddings = tf.abs(embeddings)
    average_sparsity_count = (
        tf.reduce_sum(tf.cast(abs_embeddings <= sparsity_threshold, tf.float32))
        / embeddings.shape[0]
    )
    l1_norm_col = tf.reduce_mean(abs_embeddings, axis=0)
    mean_flops_sur = tf.reduce_sum(l1_norm_col * l1_norm_col)
    return mean_flops_sur, average_sparsity_count


@pytest.mark.parametrize("embedding_shape", [(8, 1, 1024), (8, 1024), (8, 2, 1024)])
def test_flops_loss(embedding_shape):
    sparsity_threshold = 0.2
    embeddings = np.random.uniform(-1, 1, size=embedding_shape).astype(np.float32)
    ref_loss, ref_average_sparsity_count = reference_flops_loss_calculation(
        embeddings, sparsity_threshold=sparsity_threshold
    )
    if embedding_shape == (8, 2, 1024):
        with pytest.raises(AssertionError, match=re.escape("Invalid embeddings shape!")):
            flops_loss(embeddings=embeddings)
    else:
        axlearn_loss, axlearn_average_sparsity_count = flops_loss(
            embeddings=embeddings, sparsity_threshold=sparsity_threshold
        )
        assert jnp.allclose(axlearn_loss, np.array(ref_loss))
        assert jnp.allclose(axlearn_average_sparsity_count, np.array(ref_average_sparsity_count))


def test_flops_loss_with_paddings():
    # Shape: [2, 2].
    logits_2d = jnp.asarray([[-1, 2], [-2, 2]])
    # Shape: [2, 1, 2].
    logits_3d = jnp.asarray([[[-1, 2]], [[-2, 2]]])
    paddings = jnp.asarray([0, 1])
    sparsity_threshold = 1
    loss_2d, average_sparsity_count_2d = flops_loss(
        embeddings=logits_2d, paddings=paddings, sparsity_threshold=sparsity_threshold
    )
    loss_3d, average_sparsity_count_3d = flops_loss(
        embeddings=logits_3d, paddings=paddings, sparsity_threshold=sparsity_threshold
    )
    expected_loss = (1 / 1) ** 2 + (2 / 1) ** 2
    expected_average_sparsity_count = 1 / 1
    assert loss_2d == expected_loss
    assert loss_3d == expected_loss
    assert average_sparsity_count_2d == expected_average_sparsity_count
    assert average_sparsity_count_3d == expected_average_sparsity_count


def test_large_margin_loss_1_0_equals_cross_entropy():
    logits = jnp.array([[-1.0, 0, 1], [1, -1, 2]])
    labels = jnp.array([2, 0])
    ce_loss, _ = cross_entropy(logits, labels)
    loss_value, _ = large_margin_cosine_loss(logits, labels=labels)
    np.testing.assert_array_equal(ce_loss, loss_value)


def test_large_margin_loss_larger_margin_increases_loss():
    logits = jnp.array([[-1.0, 0, 1], [1, -1, 2]])
    soft_labels = jnp.array([[0.0, 0, 1], [1, 0, 0]])
    loss1_value, _ = large_margin_cosine_loss(logits=logits, soft_labels=soft_labels, margin=0.1)
    loss2_value, _ = large_margin_cosine_loss(logits=logits, soft_labels=soft_labels, margin=0.2)
    np.testing.assert_array_less(loss1_value, loss2_value)


def test_large_margin_loss_larger_scale_increases_mistake_loss():
    logits = jnp.array([[-1.0, 0, 1], [1, -1, 2]])
    soft_labels = jnp.array([[0.0, 0, 1], [1, 0, 0]])
    loss1_value, _ = large_margin_cosine_loss(logits=logits, soft_labels=soft_labels, alpha=2)
    loss2_value, _ = large_margin_cosine_loss(logits=logits, soft_labels=soft_labels, alpha=3)
    np.testing.assert_array_less(loss1_value, loss2_value)


def test_large_margin_loss_returns_accuracy_of_valid():
    logits = jnp.array([[-1.0, 0, 1], [1, -1, 2], [1, -1, 2]])
    soft_labels = jnp.array([[0.0, 0, 1], [1, 0, 0], [0, 0, 0]])
    _, outputs = large_margin_cosine_loss(
        logits=logits, soft_labels=soft_labels, margin=0.5, alpha=2
    )
    np.testing.assert_array_equal(0.5, outputs["accuracy"])
    np.testing.assert_array_equal(2, outputs["num_examples"])


@pytest.mark.parametrize(
    "reduction, eps",
    [
        (ReductionMethod.NONE, 1e-7),
        (ReductionMethod.SUM, 1e-7),
        (ReductionMethod.MEAN, 1e-7),
        (ReductionMethod.SUM, 1e-3),
    ],
)
def test_giou_loss_random_input(reduction: ReductionMethod, eps: float):
    batch_size = 8
    num_samples = 1000
    predictions = np.random.uniform(0, 1, size=[batch_size, num_samples, 4]).astype(np.float32)
    predictions[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2]
    predictions[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3]
    targets = np.random.uniform(0, 1, size=[batch_size, num_samples, 4]).astype(np.float32)
    targets[:, :, 2] = targets[:, :, 0] + targets[:, :, 2]
    targets[:, :, 3] = targets[:, :, 1] + targets[:, :, 3]
    loss = giou_loss(
        predictions=predictions,
        targets=targets,
        reduction=reduction,
        eps=eps,
    )
    # jax uses y1, x1, y2, x2, whereas torch expects x1, y1, x2, y2
    predictions_torch = torch.Tensor(predictions[:, :, [1, 0, 3, 2]])
    targets_torch = torch.Tensor(targets[:, :, [1, 0, 3, 2]])
    ref_loss = generalized_box_iou_loss(
        predictions_torch, targets_torch, reduction=reduction, eps=eps
    )
    assert jnp.allclose(loss, ref_loss.numpy())


def test_giou_loss():
    predictions = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    targets = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],  # perfect overlap
            [2.0, 2.0, 3.0, 3.0],  # no overlap
            [10.0, 10.0, 11.0, 11.0],  # no overlap2
            [1.5, 1.0, 2.0, 2.0],  # some overlap
            [1.5, 1.5, 3.0, 3.0],  # some overlap2
        ]
    )
    loss = giou_loss(
        predictions=predictions,
        targets=targets,
        reduction=ReductionMethod.NONE,
        eps=1e-9,
    )
    expected_loss = np.array(
        [
            0.0,
            1.5,
            1 + (10**2 - 2) / 10**2,
            0.5,
            1 - 0.5**2 / (1.5**2 + 1 - 0.5**2) + (4 - (1.5**2 + 1 - 0.5**2)) / 4,
        ]
    )
    assert jnp.allclose(loss, expected_loss)

    # test with sample weights
    sample_weight = np.array([100.0, 1.0, 0.0, 10.0, 5.0])
    loss = giou_loss(
        predictions=predictions,
        targets=targets,
        sample_weight=sample_weight,
        reduction=ReductionMethod.NONE,
        eps=1e-9,
    )
    expected_loss = np.array(
        [
            0.0,
            1.5,
            1 + (10**2 - 2) / 10**2,
            0.5,
            1 - 0.5**2 / (1.5**2 + 1 - 0.5**2) + (4 - (1.5**2 + 1 - 0.5**2)) / 4,
        ]
    )
    assert jnp.allclose(loss, expected_loss * sample_weight)

    # test with sample weights and reduction
    sample_weight = np.array([100.0, 1.0, 0.0, 10.0, 5.0])
    loss = giou_loss(
        predictions=predictions,
        targets=targets,
        sample_weight=sample_weight,
        reduction=ReductionMethod.MEAN,
        eps=1e-9,
    )
    expected_loss = np.array(
        [
            0.0,
            1.5,
            1 + (10**2 - 2) / 10**2,
            0.5,
            1 - 0.5**2 / (1.5**2 + 1 - 0.5**2) + (4 - (1.5**2 + 1 - 0.5**2)) / 4,
        ]
    )
    assert jnp.allclose(loss, jnp.sum(expected_loss * sample_weight) / jnp.sum(sample_weight))


def test_cosine_similarity_without_mask():
    predictions = np.random.uniform(-1, 1, size=[2, 4, 16]).astype(np.float32)
    targets = np.random.uniform(-1, 1, size=[2, 4, 16]).astype(np.float32)
    _, aux = negative_cosine_similarity_loss(predictions=predictions, targets=targets)
    ref = optax.cosine_similarity(predictions, targets)
    assert jnp.allclose(aux["cosine_similarity"], ref, atol=1e-06)


def test_cosine_similarity_with_mask():
    predictions = np.random.rand(2, 4, 16)
    targets = np.random.rand(2, 4, 16)
    live_targets = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
    loss, _ = negative_cosine_similarity_loss(
        predictions=predictions, targets=targets, live_targets=live_targets
    )
    masked_predictions = predictions[:, 1:3, :]
    masked_targets = targets[:, 1:3, :]
    ref = optax.cosine_similarity(masked_predictions, masked_targets)
    ref_loss = -jnp.mean(ref)
    assert jnp.allclose(loss, ref_loss)


def test_negative_cosine_similarity_loss_against_torch():
    predictions = np.random.rand(2, 4, 16)
    targets = np.random.rand(2, 4, 16)
    live_targets = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
    loss, _ = negative_cosine_similarity_loss(
        predictions=predictions, targets=targets, live_targets=live_targets
    )
    # Ref torch implementation from:
    # https://github.com/baaivision/EVA/blob/86cf99c50612b11bad39bfcf17899c410a7030d4/eva/engine_for_pretraining.py#L39-L42
    masked_predictions = predictions[:, 1:3, :]
    masked_targets = targets[:, 1:3, :]
    ref_loss_fn = nn.CosineSimilarity(dim=-1)
    ref = -ref_loss_fn(
        torch.from_numpy(masked_predictions).float(), torch.from_numpy(masked_targets).float()
    ).mean()
    assert jnp.allclose(loss, ref.numpy())


def test_kl_divergence():
    # pylint: disable=protected-access
    # pytype: disable=module-attr
    predictions = np.random.rand(8, 1000)
    log_predictions = np.log(predictions)
    targets = np.random.rand(8, 1000)

    # Test probability targets against optax.
    _, aux = kl_divergence(log_predictions, targets)
    ref_loss = optax._src.loss.kl_divergence(log_predictions, targets)
    assert jnp.allclose(aux["per_example_loss"], ref_loss)

    # Test log-space targets against optax.
    log_targets = jnp.log(targets)
    loss, aux = kl_divergence(log_predictions, log_targets, is_log_targets=True)
    ref_loss = optax._src.loss.kl_divergence_with_log_targets(log_predictions, log_targets)
    assert jnp.allclose(aux["per_example_loss"], ref_loss)

    # Test against TF.
    tf_ref_loss = tf.keras.losses.KLDivergence()(targets, predictions)
    assert jnp.allclose(loss, tf_ref_loss.numpy())

    # Test against Torch.
    torch_ref_loss = nn.KLDivLoss(reduction="batchmean")(
        torch.from_numpy(log_predictions), torch.from_numpy(targets)
    )
    assert jnp.allclose(loss, torch_ref_loss.numpy())
    # pylint: enable=protected-access
    # pytype: enable=module-attr


def reference_koleo_loss(embeddings, eps=1e-8):
    # Reference from:
    # https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    embeddings = torch.nn.functional.normalize(embeddings, eps=eps, p=2, dim=-1)
    dots = torch.mm(embeddings, embeddings.t())
    n = embeddings.shape[0]
    dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, dim=1)  # noqa: E741  # pylint: disable=invalid-name
    pdist = nn.PairwiseDistance(2, eps=eps)
    distances = pdist(embeddings, embeddings[I])  # BxD, BxD -> B
    return -torch.log(distances + eps).mean().numpy()


def test_koleo_loss_wo_norm():
    # Input without normalization
    embeddings = np.random.rand(10, 128)
    jax_loss = koleo_loss(embeddings, eps=1e-8)
    assert jnp.allclose(jax_loss, reference_koleo_loss(torch.Tensor(embeddings), eps=1e-8))


def test_koleo_loss_w_norm():
    # Input with normalization
    embeddings = np.random.rand(10, 128)
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    jax_loss = koleo_loss(jnp.array(norm_embeddings), eps=1e-8, normalize_embedding=False)
    assert jnp.allclose(jax_loss, reference_koleo_loss(torch.Tensor(norm_embeddings), eps=1e-8))


def test_koleo_loss_zero_dist():
    embeddings = np.random.rand(10, 128)
    # Input without normalization and has zero distance between embeddings.
    two_embeddings = np.concatenate((embeddings, embeddings))
    jax_loss = koleo_loss(two_embeddings, eps=1e-8)
    assert jnp.allclose(jax_loss, reference_koleo_loss(torch.Tensor(two_embeddings), eps=1e-8))


def _softplus(x):
    return math.log(1 + math.exp(x))


class PairwiseLossFunctionsTest(TestCase):
    def test_single_pair(self):
        loss, weight = ranking_pairwise_loss(
            logits=jnp.array([[3, 0]], dtype=jnp.float32),
            ranks=jnp.array([[1, 2]], dtype=jnp.float32),
            loss_scale=jnp.array([1], dtype=jnp.float32),
        )
        self.assertAlmostEqual(weight.item(), 1)
        self.assertAlmostEqual(loss.item(), _softplus(-3))

    def test_multiple_pairs(self):
        loss, weight = ranking_pairwise_loss(
            logits=jnp.array([[3, 0, 1]], dtype=jnp.float32),
            ranks=jnp.array([[1, 2, 3]], dtype=jnp.float32),
            loss_scale=jnp.array([1], dtype=jnp.float32),
        )
        self.assertAlmostEqual(weight.item(), 3)
        self.assertAlmostEqual(loss.item(), sum(_softplus(-x) for x in (3 - 0, 3 - 1, 0 - 1)) / 3)

    def test_2positive_1negative(self):
        loss, weight = ranking_pairwise_loss(
            logits=jnp.array([[3, 0, 1]], dtype=jnp.float32),
            ranks=jnp.array([[1, 2, 0]], dtype=jnp.float32),
            loss_scale=jnp.array([1], dtype=jnp.float32),
        )
        self.assertAlmostEqual(weight.item(), 1)
        self.assertAlmostEqual(loss.item(), _softplus(-3))

    def test_pairwise_ranking_loss_3positive_2negative(self):
        loss, weight = ranking_pairwise_loss(
            logits=jnp.array([[3, 4, 0, 1, 2]], dtype=jnp.float32),
            ranks=jnp.array([[1, 2, 3, 0, 0]], dtype=jnp.float32),
            loss_scale=jnp.array([1], dtype=jnp.float32),
        )
        # 3 pairs between ranked docs. No loss between negatives.
        self.assertAlmostEqual(weight.item(), 3)
        self.assertAlmostEqual(loss.item(), sum(_softplus(-x) for x in (3 - 4, 3 - 0, 4 - 0)) / 3)

    def test_ignore_padding(self):
        loss, weight = ranking_pairwise_loss(
            logits=jnp.array([[3, 0, 1, 5]], dtype=jnp.float32),
            ranks=jnp.array([[1, 2, 0, -100]], dtype=jnp.float32),
            loss_scale=jnp.array([1], dtype=jnp.int32),
        )
        self.assertAlmostEqual(weight.item(), 1)
        self.assertAlmostEqual(loss.item(), _softplus(-3))


class ReduceLossTest(parameterized.TestCase):
    @parameterized.product(
        reduction=(ReductionMethod.NONE, ReductionMethod.MEAN, ReductionMethod.SUM),
        sample_weights=(np.array([[1.0], [5.0]]), None),
    )
    # pylint: disable-next=no-self-use
    def test_reduce_loss(self, reduction, sample_weights):
        expected_reduced_loss = {
            (ReductionMethod.NONE, False): np.array([[1.0, 3.0, 10.0], [10.0, 15.0, 0.0]]),
            (ReductionMethod.MEAN, False): 6.5,
            (ReductionMethod.SUM, False): 39.0,
            (ReductionMethod.NONE, True): np.array([[1.0, 3.0, 10.0], [2.0, 3.0, 0.0]]),
            (ReductionMethod.MEAN, True): 3.1666667,
            (ReductionMethod.SUM, True): 19.0,
        }
        loss = np.array([[1.0, 3.0, 10.0], [2.0, 3.0, 0.0]])
        reduced_loss = _reduce_loss(loss=loss, sample_weight=sample_weights, reduction=reduction)
        assert jnp.allclose(
            expected_reduced_loss[(reduction, sample_weights is None)], reduced_loss
        )

    @parameterized.parameters(
        (np.array([[0.0], [0.0]]), 0.0),
        (np.array([[-2e-8], [1e-8]]), 0.01),
        (np.array([[1e-8], [1e-8]]), 0.02),
    )
    def test_small_sample_weights(self, sample_weights, expected_loss):
        loss = np.array([[1.0], [1.0]])
        reduced_loss = _reduce_loss(
            loss=loss, sample_weight=sample_weights, reduction=ReductionMethod.MEAN
        )
        self.assertAlmostEqual(reduced_loss, expected_loss)
