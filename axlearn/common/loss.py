# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/t5x:
# Copyright 2022 The T5X Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# keras-team/keras:
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# tensorflow/tpu:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# deepmind/optax:
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# facebookresearch/fvcore:
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Loss functions."""
# pylint: disable=too-many-lines
import enum
from typing import Optional, Union

import jax
import jax.numpy as jnp
import optax

from axlearn.common.metrics import WeightedScalar
from axlearn.common.normalize import l2_normalize
from axlearn.common.utils import Tensor

NEG_INF = -1e15


class ReductionMethod(str, enum.Enum):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


def _reduce_loss(
    *, loss: Tensor, reduction: ReductionMethod, sample_weight: Optional[Tensor], eps: float = 1e-6
):
    """Reduces loss tensor.

    Args:
        loss: A [...] float tensor of arbitrary shape.
        reduction: The reduction method.
        sample_weight: A [...] float tensor, must be broadcastable to same shape as loss.
        eps: A small number to prevent division by zero.

    Returns:
        A float tensor:
            - with equal shape to `loss` if ReductionMethod.NONE
            - with scalar value else.
    """
    # Scale loss by per sample weight.
    if sample_weight is not None:
        loss = loss * sample_weight

    if reduction == ReductionMethod.NONE:
        reduced_loss = loss
    elif reduction == ReductionMethod.SUM:
        reduced_loss = jnp.sum(loss)
    elif reduction == ReductionMethod.MEAN:
        if sample_weight is not None:
            denominator = jnp.sum(sample_weight)
            sign = jax.lax.cond(denominator < 0, lambda: -1.0, lambda: 1.0)
            denominator = sign * jnp.maximum(jnp.abs(denominator), eps)
            reduced_loss = jnp.sum(loss) / denominator
        else:
            reduced_loss = jnp.mean(loss)
    return reduced_loss


def cross_entropy(
    logits: Tensor,
    target_labels: Tensor,
    *,
    live_targets: Optional[Tensor] = None,
    z_loss_scale: float = 0.0,
    label_smoothing: float = 0.0,
    soft_target_labels: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the cross entropy loss between logits and target_labels.

    Computes a stabilized-gradient version of:
        -jnp.sum(targets * nn.log_softmax(logits), axis=-1) * live_targets
        / jnp.sum(live_targets, axis=-1)
    where targets is a categorical one-hot float array based on target_labels.

    This function extends the T5X implementation by supporting masked labels.

    Ref: https://github.com/google-research/t5x/blob/90d74f/t5x/losses.py#L26

    Args:
        logits: A float Tensor of shape [..., num_classes].
        target_labels: An int Tensor of shape [...].
            The per-example loss will be 0 if the corresponding target is masked, or out-of-class
            (i.e. target_labels[i] < 0 or target_labels[i] >= num_classes).
        live_targets: Indicates which examples should contribute to the loss.
            A bool or 0/1 Tensor broadcastable to `target_labels`. 1 indicates positions that
            contribute to the loss. If None, infer from 0 <= target_labels < num_classes.
        z_loss_scale: Coefficient for auxiliary z-loss loss term.
        label_smoothing: The factor to control label smoothing.
        soft_target_labels: Optional labels that are already smoothed/in one-hot form. If provided,
            target_labels will only be used for inferring the live targets during loss calculation.

    Returns:
        (loss, all_losses), where
        loss is a scalar tensor for the cross entropy loss;
        all_losses is a dictionary containing:
            * "total_loss": a scalar representing the overall
                loss = cross_entropy_loss + z_loss_scale * z_loss.
            * "cross_entropy_loss": the cross_entropy_loss.
            * "z_loss": the unscaled z_loss.
            * "per_target_loss": the loss per target, of the same shape as `target_labels`.

    Raises:
        ValueError: If z_loss_scale is negative.
        ValueError: If cfg.label_smoothing is > 0 when soft_labels is provided.
    """
    if logits.dtype in (jnp.bfloat16, jnp.float16):
        logits = logits.astype(jnp.float32)
    if z_loss_scale < 0:
        raise ValueError("z_loss_scale should not be negative.")
    num_classes = logits.shape[-1]
    if soft_target_labels is not None:
        if label_smoothing != 0:
            raise ValueError("Label smoothing should be set to 0 if soft labels are used.")
        targets = soft_target_labels
    else:
        targets = _one_hot_with_label_smoothing(
            target_labels, num_classes, label_smoothing=label_smoothing
        )
    per_target_loss, per_target_cross_entropy_loss, per_target_z_loss = _stable_cross_entropy(
        logits, targets, z_loss_scale
    )
    if live_targets is None:
        live_targets = jnp.logical_and(0 <= target_labels, target_labels < num_classes)
    live_targets = live_targets.astype(per_target_loss.dtype)
    denominator = jnp.maximum(live_targets.sum(), 1)
    cross_entropy_loss = (per_target_cross_entropy_loss * live_targets).sum() / denominator
    z_loss = (per_target_z_loss * live_targets).sum() / denominator
    loss = (per_target_loss * live_targets).sum() / denominator
    predicted_labels = jnp.argmax(logits, axis=-1)
    accuracy = (jnp.equal(predicted_labels, target_labels) * live_targets).sum() / denominator
    return loss, {
        "total_loss": loss,
        "z_loss": z_loss,
        "cross_entropy_loss": cross_entropy_loss,
        "per_target_loss": per_target_loss,
        "accuracy": accuracy,
    }


def binary_cross_entropy(
    logits: Tensor,
    *,
    target_labels: Tensor,
    live_targets: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the binary cross entropy loss between logits and targets.

    Computes a stabilized-gradient version of:
        -jnp.sum(targets * jnp.log(logits) + (1-targets) * jnp.log(1 - logits), axis=-1)
        * live_targets / jnp.sum(live_targets, axis=-1)
    where targets are a one-hot float array.

    Args:
        logits: A float Tensor of shape [batch_size, d0, ..., dN].
        target_labels: An 0/1 int Tensor of the same shape as `logits`.
            The per-example loss will be 0 if the corresponding target is masked.
        live_targets: Indicates which examples should contribute to the loss.
            A bool or 0/1 Tensor broadcastable to `target_labels`. 1 indicates positions that
            contribute to the loss. If None, infer from 0 <= target_labels < 2.

    Returns:
        (loss, all_losses), where
        loss is a scalar tensor for the binary cross entropy loss;
        all_losses is a dictionary containing:
            * "binary_cross_entropy_loss": the binary_cross_entropy_loss.
            * "per_target_loss": the loss per target, of the same shape as `target_labels`.
    """
    if logits.dtype in (jnp.bfloat16, jnp.float16):
        logits = logits.astype(jnp.float32)
    per_target_cross_entropy_loss = sigmoid_cross_entropy_with_logits(logits, target_labels)
    if live_targets is None:
        live_targets = jnp.logical_and(0 <= target_labels, target_labels < 2)
    live_targets = live_targets.astype(per_target_cross_entropy_loss.dtype)
    binary_cross_entropy_loss = (per_target_cross_entropy_loss * live_targets).sum() / jnp.maximum(
        live_targets.sum(), 1
    )
    return binary_cross_entropy_loss, {
        "binary_cross_entropy_loss": binary_cross_entropy_loss,
        "per_target_loss": per_target_cross_entropy_loss,
    }


def _one_hot_with_label_smoothing(
    labels: Tensor, num_classes: int, label_smoothing: float = 0.0
) -> Tensor:
    if num_classes < 1:
        raise ValueError("num classes should be at least 1.")
    if label_smoothing < 0 or label_smoothing > 1:
        raise ValueError("label_smoothing should be in the range [0, 1].")
    labels_one_hot = jax.nn.one_hot(labels, num_classes)
    labels_one_hot = labels_one_hot * (1 - label_smoothing) + label_smoothing / num_classes
    return labels_one_hot


@jax.custom_vjp
def _stable_cross_entropy(logits: Tensor, targets: Tensor, z_loss_scale: float) -> Tensor:
    """Computes cross entropy loss with stable custom gradient.

    This is a copy of x-entropy loss from the T5X codebase.
    Ref: https://github.com/google-research/t5x/blob/90d74/t5x/losses.py#L26

    Computes a stabilized-gradient version of:
      -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
    If z_loss_scale > 0, then an auxiliary loss equal to z_loss_scale*log(z)^2
    will be added to the cross entropy loss (z = softmax normalization constant).
    The two uses of z_loss_scale are:
    1. To keep the logits from drifting too far from zero, which can cause
       unacceptable roundoff errors in bfloat16.
    2. To encourage the logits to be normalized log-probabilities.

    Args:
      logits: [..., num_classes] float array.
      targets: categorical one-hot targets [..., num_classes] float
        array.
      z_loss_scale: coefficient for auxiliary z-loss loss term.

    Returns:
      tuple with the total loss and the z_loss, both
      float arrays with shape [batch, length].

    TODO(@bwzhang): Factorize the following as a log_softmax_with_z_loss function.
    """
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    cross_entropy_loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxiliary z-loss term.
    log_z = jnp.squeeze(logits_sum, axis=-1)
    total_z_loss = jax.lax.square(log_z)
    loss = cross_entropy_loss + total_z_loss * z_loss_scale
    return loss, cross_entropy_loss, total_z_loss


def _stable_cross_entropy_fwd(logits: Tensor, targets: Tensor, z_loss_scale: float = 0.0):
    """Forward-mode of `cross_entropy_with_logits`.

    This is a copy of x-entropy loss from the T5X codebase.
    Ref: https://github.com/google-research/t5x/blob/90d74/t5x/losses.py#L60
    """
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    cross_entropy_loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxiliary z-loss term.
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    total_z_loss = jax.lax.square(log_z)
    loss = cross_entropy_loss + total_z_loss * z_loss_scale
    return (loss, cross_entropy_loss, total_z_loss), (
        logits,
        targets,
        z_loss_scale,
        exp_shifted,
        sum_exp,
        log_softmax,
        log_z,
    )


def _stable_cross_entropy_bwd(
    res: tuple,
    g: tuple[Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Backward-mode of `cross_entropy_with_logits`.

    This is a copy of x-entropy loss from the T5X codebase.
    Ref: https://github.com/google-research/t5x/blob/90d74/t5x/losses.py#L81
    """
    g = g[0]  # Ignore cross_entropy_loss and z_loss component as that is only used for logging.
    logits, targets, z_loss_scale, exp_shifted, sum_exp, log_softmax, log_z = res
    # z-loss term adds the (2 * z_loss_scale * log_z) factor.
    deriv = (
        jnp.expand_dims(jnp.sum(targets, axis=-1) + 2 * z_loss_scale * log_z, -1)
        * exp_shifted
        / sum_exp
        - targets
    )
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
    return (
        jnp.asarray(g_logits, logits.dtype),
        jnp.asarray(g_targets, targets.dtype),
        jnp.array(0.0),
    )  # sets z-loss coeff gradient to 0


# jax.custom_vjp can be used to define the custom derivative rules for
# JAX-transformable python functions. One use case is to improve the numerical
# stability of differentiation. Here we use the custom_vjp to improve the stability
# of cross_entropy loss.
# Ref: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
_stable_cross_entropy.defvjp(_stable_cross_entropy_fwd, _stable_cross_entropy_bwd)


def _weighted_mean(
    arr: Tensor,
    *,
    sample_weight: Optional[Tensor] = None,
    eps: float = 1e-7,
) -> WeightedScalar:
    """Computes the weighted average of arr without dividing by 0.

    Args:
        arr: A float Tensor of any shape.
        sample_weight: A float tensor to weight each sample by. Its shape must be a prefix of the
                       shape of `arr`.
        eps: If the total weight used to compute the mean is below eps, it is increased to eps.

    Returns:
        A WeightedScalar with the result.
    """
    if sample_weight is None:
        sample_weight = jnp.array(1.0)
    while sample_weight.ndim < arr.ndim:
        sample_weight = sample_weight[..., None]
    sample_weight = jnp.broadcast_to(sample_weight, arr.shape)
    total_weight = sample_weight.sum()
    loss = jnp.sum(sample_weight * arr) / jnp.maximum(total_weight, eps)
    return WeightedScalar(loss, total_weight)


def mean_squared_error(
    preds: Tensor,
    targets: Tensor,
    sample_weight: Optional[Tensor] = None,
    eps: float = 1e-7,
) -> WeightedScalar:
    """Computes mean squared error loss.

    Args:
        preds: A float Tensor of any shape.
        targets: A float Tensor of same shape as `preds`.
        sample_weight: A float tensor to weight each sample by. Its shape must be a prefix of the
               shape of `targets`.
        eps: If the total weight used to compute the mean is below eps, it is increased to eps.

    Returns:
        A WeightedScalar consisting of the loss and the number of examples that contributed to the
        loss. If there are no targets, a WeightedScalar with 0 loss and 0 weight is returned.
    """
    diff = (preds - targets) ** 2
    return _weighted_mean(diff, sample_weight=sample_weight, eps=eps)


def bilinear_mean_squared_error(
    preds: Tensor,
    targets: Tensor,
    *,
    shape: tuple[int, ...],
    sample_weight: Optional[Tensor] = None,
    eps: float = 1e-7,
) -> WeightedScalar:
    """Computes the mean squared error loss after bilinear downsampling to shape `shape`.

    Args:
        preds: A float Tensor of any shape.
        targets: A float Tensor of same shape as `preds`.
        shape: The shape preds and targets should be resized to using bilinear resampling
               prior to computing the MSE.
        sample_weight: A float tensor to weight each sample by. Its shape must be a prefix of the
               shape of `targets`.
        eps: If the total weight used to compute the mean is below eps, it is increased to eps.

    Returns:
        A WeightedScalar consisting of the loss and the number of examples that contributed to the
        loss. If there are no targets, a WeightedScalar with 0 loss and 0 weight is returned.
    """
    src_shape = jnp.broadcast_shapes(preds.shape, targets.shape)
    for dim, new_dim in jax.util.safe_zip(src_shape, shape):
        if not (dim % new_dim == 0 or new_dim % dim == 0):
            raise NotImplementedError(
                f"The dimensions in shape and (preds-targets).shape must be "
                f"whole multiples of one another, {src_shape} vs. {shape}"
            )
    diff = preds - targets
    diff = jax.image.resize(diff, shape=shape, method="bilinear", antialias=False)
    diff = diff**2
    return _weighted_mean(diff, sample_weight=sample_weight, eps=eps)


def l1_loss(
    preds: Tensor,
    targets: Tensor,
    sample_weight: Optional[Tensor] = None,
    eps: float = 1e-7,
) -> WeightedScalar:
    """Computes mean l1 loss.

    Args:
        preds: A float Tensor of any shape.
        targets: A float Tensor of same shape as `preds`.
        sample_weight: A float tensor to weight each sample by. Its shape must be a prefix of the
               shape of `targets`.
        eps: If the total weight used to compute the mean is below eps, it is increased to eps.

    Returns:
        A WeightedScalar consisting of the loss and the number of examples that contributed to the
        loss. If there are no targets, a WeightedScalar with 0 loss and 0 weight is returned.
    """
    diff = jnp.abs(preds - targets)
    return _weighted_mean(diff, sample_weight=sample_weight, eps=eps)


def contrastive_logits(x: Tensor, y: Tensor) -> Tensor:
    """Computes contrastive logits between two tensors.

    Args:
        x: A float Tensor of shape [num_features_in_x, feature_size].
        y: A float Tensor of shape [num_features_in_y, feature_size].

    Returns:
        The logits (similarity matrix) of shape [num_features_in_x, num_features_in_y].

    Raises:
        ValueError: If x or y have invalid shapes.
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    if x.shape[1] != y.shape[1]:
        raise ValueError("`x[1]` and `y[1]` must have the same feature size!")
    logits = jnp.einsum("i d, j d -> i j", x, y)
    return logits


# TODO(jlei2, bowen-zhang9): Support query_paddings.
def asymmetric_contrastive_loss_from_logits(
    logits: Tensor,
    *,
    key_paddings: Tensor = None,
    temperature: Union[Tensor, float] = 1.0,
    soft_labels: Optional[Tensor] = None,
) -> Tensor:
    """Asymmetric contrastive loss from logits.

    When soft_labels is None, minimizing the loss will maximize the gap between logits[i, i] and
    logits[i, j] where i != j. The target labels of cross-entropy look like:
    ```
    1 0 0
    0 1 0
    0 0 1
    ```

    When soft_labels is not None, the cross-entropy between softmax-ed logits and soft_labels will
    be minimized. soft_labels could look like:
    ```
    0.4 0.0 0.0
    0.0 0.3 0.0
    0.0 0.0 0.9
    ```
    and even
    ```
    0.6 0.2 0.2
    0.0 0.3 0.0
    0.1 0.0 0.9
    ```
    where each key may have an associated soft label.

    Args:
        logits: A float Tensor of shape [num_queries, num_keys] where num_queries <= num_keys.
        key_paddings: A 0/1 Tensor of shape [num_keys,] where 0 means valid key and 1 means
            padded ones. The first num_queries elements are expected to be valid positive keys.
        temperature: A positive scalar float to be divided from logits. Default is 1.0.
        soft_labels: Optional soft labels already in shape of [num_queries, num_keys] where values
            are expected to be in range [0, 1]. Cross-entropy contrastive loss will use as labels
            when not None.

    Returns:
        A scalar of asymmetric contrastive loss. The softmax cross-entropy loss will only be
        calculated for each query among the non-padded candidate keys.
    """
    logits_with_temperature = logits / temperature

    num_keys = logits_with_temperature.shape[1]
    if key_paddings is None:
        key_paddings = jnp.zeros(num_keys)

    masked_logits = logits_with_temperature + key_paddings * NEG_INF

    if soft_labels is not None:
        assert soft_labels.shape == masked_logits.shape, (
            f"soft_labels has a shape of {soft_labels.shape} while logits has a shape of "
            f"{masked_logits.shape}!"
        )
        soft_labels = soft_labels * (1 - key_paddings)

    loss = cross_entropy(
        masked_logits, jnp.arange(logits_with_temperature.shape[0]), soft_target_labels=soft_labels
    )[0]
    return loss


def asymmetric_contrastive_loss_from_features(
    queries: Tensor,
    positive_keys: Tensor,
    *,
    negative_keys: Tensor = None,
    negative_key_paddings: Tensor = None,
    temperature: Union[Tensor, float] = 1.0,
    soft_labels: Optional[Tensor] = None,
):
    """Asymmetric contrastive loss from features.

    Args:
        queries: A float Tensor of shape [num_queries, feature_size].
        positive_keys: A float Tensor of shape [num_positive_keys, feature_size]
            where num_queries == num_positive_keys.
        negative_keys: An optional float Tensor of shape [num_negative_keys, feature_size].
        negative_key_paddings: A 0/1 Tensor of shape [num_negative_keys,] where 0 means valid key
            and 1 means padded ones.
        temperature: A positive scalar float to be divided from logits. Default is 1.0.
        soft_labels: Optional soft labels already in shape of
            [num_queries, num_positive_keys] or
            [num_queries, num_positive_keys + num_negative_keys] when negative_keys exists.
            Values are expected to be in range [0, 1]. Cross-entropy contrastive loss will use as
            labels when not None.

    Returns:
        A scalar of asymmetric contrastive loss. The softmax cross-entropy loss will only be
        calculated for each query among the non-padded candidate keys.
    """
    assert len(positive_keys.shape) == 2
    assert queries.shape[0] == positive_keys.shape[0]

    positive_key_paddings = jnp.zeros(positive_keys.shape[0])

    if negative_keys is not None:
        assert len(negative_keys.shape) == 2
        assert positive_keys.shape[1] == negative_keys.shape[1]
        keys = jnp.vstack([positive_keys, negative_keys])

        if negative_key_paddings is None:
            negative_key_paddings = jnp.zeros(negative_keys.shape[0])
        key_paddings = jnp.hstack([positive_key_paddings, negative_key_paddings])
    else:
        keys = positive_keys
        key_paddings = positive_key_paddings

    logits = contrastive_logits(queries, keys)
    loss = asymmetric_contrastive_loss_from_logits(
        logits,
        key_paddings=key_paddings,
        temperature=temperature,
        soft_labels=soft_labels,
    )
    return loss


def symmetric_contrastive_loss_from_logits(  # pylint: disable=missing-param-doc
    x_y_logits: Tensor,
    y_x_logits: Tensor,
    *,
    y_as_key_paddings: Tensor = None,
    x_as_key_paddings: Tensor = None,
    temperature: Union[float, Tensor] = 1.0,
    y_as_key_soft_labels: Optional[Tensor] = None,
    x_as_key_soft_labels: Optional[Tensor] = None,
):
    """Symmetric contrastive loss from logits.

    Args:
        x_y_logits: A float Tensor of shape [num_positive_features_in_x, num_features_in_y].
        y_x_logits: A float Tensor of shape [num_positive_features_in_y, num_features_in_x].
            In most cases, num_positive_features_in_x == num_positive_features_in_y.
        y_as_key_paddings: A 0/1 Tensor of shape [num_features_in_y,] where 0 means valid key
            and 1 means padded ones.
        x_as_key_paddings: A 0/1 Tensor of shape [num_features_in_x,] where 0 means valid key
            and 1 means padded ones.
        temperature: A positive scalar float to be divided from logits. Default is 1.0.
        y_as_key_soft_labels, x_as_key_soft_labels: Optional soft labels already in shape of
            [num_positive_features_in_x, num_features_in_y] (for y_as_key_soft_labels)
            [num_positive_features_in_y, num_features_in_x] (for x_as_key_soft_labels). Values are
            expected to be in range [0, 1]. Cross-entropy contrastive loss will use as labels when
            not None.

    Returns:
        A scalar of symmetric contrastive loss. The symmetric contrastive loss is the average of
        asymmetric contrastive loss from x_y_logits and y_x_logits.
    """
    loss = (
        asymmetric_contrastive_loss_from_logits(
            x_y_logits,
            temperature=temperature,
            key_paddings=y_as_key_paddings,
            soft_labels=y_as_key_soft_labels,
        )
        + asymmetric_contrastive_loss_from_logits(
            y_x_logits,
            temperature=temperature,
            key_paddings=x_as_key_paddings,
            soft_labels=x_as_key_soft_labels,
        )
    ) / 2.0
    return loss


def symmetric_contrastive_loss_from_features(
    x: Tensor,
    y: Tensor,
    *,
    x_negatives: Tensor = None,
    y_negatives: Tensor = None,
    x_negative_paddings: Tensor = None,
    y_negative_paddings: Tensor = None,
    temperature: Union[Tensor, float] = 1.0,
    y_as_key_soft_labels: Optional[Tensor] = None,
    x_as_key_soft_labels: Optional[Tensor] = None,
):
    """Symmetric contrastive loss from features.

    Args:
        x: A float Tensor of shape [num_x_positives, feature_size].
        y: A float Tensor of shape [num_y_positives, feature_size]
            where num_x_positives == num_y_positives.
        x_negatives: An optional float Tensor of shape [num_x_negatives, feature_size].
        y_negatives: An optional float Tensor of shape [num_y_negatives, feature_size].
        x_negative_paddings: A 0/1 Tensor of shape [num_x_negatives,] where 0 means valid key
            and 1 means padded ones.
        y_negative_paddings: A 0/1 Tensor of shape [num_y_negatives,] where 0 means valid key
            and 1 means padded ones.
        temperature: A positive scalar float to be divided from logits. Default is 1.0.
        y_as_key_soft_labels: Optional soft labels already in shape of
            [num_x_positives, num_y_positives + num_y_negatives] where
            num_y_negatives=0 when there is no y_negatives. Values are
            expected to be in range [0, 1].
        x_as_key_soft_labels: Optional soft labels already in shape of
            [num_y_positives, num_x_positives + num_x_negatives] where
            num_x_negatives=0 when there is no x_negatives. Values are
            expected to be in range [0, 1].

    Returns:
        A scalar of symmetric contrastive loss. The symmetric contrastive loss is the average of
        asymmetric contrastive loss from x to y and non-padded y_negatives (when not None), and
        that from y to x and non-padded x_negatives (when not None).
    """
    assert x.shape[0] == y.shape[0]
    # [num_x_positives, num_y_positives]
    x_y_sim = contrastive_logits(x, y)

    y_paddings = jnp.zeros(y.shape[0])
    x_paddings = jnp.zeros(x.shape[0])

    if y_negatives is not None:
        # [num_x_positives, num_y_negatives]
        x_y_negatives_sim = contrastive_logits(x, y_negatives)
        assert x_y_sim.shape[0] == x_y_negatives_sim.shape[0]
        # [num_x_positives, num_y_positives + num_y_negatives]
        x_y_logits = jnp.hstack([x_y_sim, x_y_negatives_sim])

        if y_negative_paddings is None:
            y_negative_paddings = jnp.zeros(y_negatives.shape[0])
        y_as_key_paddings = jnp.hstack([y_paddings, y_negative_paddings])
    else:
        # [num_x_positives, num_y_positives]
        x_y_logits = x_y_sim
        y_as_key_paddings = y_paddings

    y_x_sim = x_y_sim.T

    if x_negatives is not None:
        # [num_y_positives, num_x_negatives]
        y_x_negatives_sim = contrastive_logits(y, x_negatives)
        assert y_x_sim.shape[0] == y_x_negatives_sim.shape[0]
        # [num_y_positives, num_x_positives + num_x_negatives]
        y_x_logits = jnp.hstack([y_x_sim, y_x_negatives_sim])

        if x_negative_paddings is None:
            x_negative_paddings = jnp.zeros(x_negatives.shape[0])
        x_as_key_paddings = jnp.hstack([x_paddings, x_negative_paddings])
    else:
        # [num_y_positives, num_x_positives]
        y_x_logits = y_x_sim
        x_as_key_paddings = x_paddings

    loss = symmetric_contrastive_loss_from_logits(
        x_y_logits,
        y_x_logits,
        y_as_key_paddings=y_as_key_paddings,
        x_as_key_paddings=x_as_key_paddings,
        temperature=temperature,
        y_as_key_soft_labels=y_as_key_soft_labels,
        x_as_key_soft_labels=x_as_key_soft_labels,
    )
    return loss


def sigmoid_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """Implementation based on `tf.nn.sigmoid_cross_entropy_with_logits`.

    Reference: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits.
    """
    return jnp.maximum(logits, 0) - targets * logits + jnp.log(1 + jnp.exp(-jnp.abs(logits)))


def categorical_hinge_loss(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """Computes the categorical hinge loss between `y_true` & `logits`.

    `loss = maximum(neg - pos + 1, 0)`
    where `neg=maximum((1-targets)*logits) and pos=sum(targets*logits)`

    Args:
        logits: A float tensor of size [..., num_classes].
        targets: A float tensor of size [..., num_classes] represent one-hot labels.
            targets must be of the same size of logits.

    Returns:
        A tensor of the  shape [...] which accumulates the hinge loss over all categories.

    Reference:
        https://github.com/keras-team/keras/blob/v2.11.0/keras/losses.py#L1833-L1865
    """

    pos = jnp.sum(targets * logits, axis=-1)
    neg = jnp.amax((1.0 - targets) * logits, axis=-1)
    loss = jnp.maximum(0.0, neg - pos + 1.0)

    return loss


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 1.5,
    sample_weight: Optional[Tensor] = None,
) -> Tensor:
    """Compute the focal loss between `logits` and the golden `targets` values.

    Focal loss = -(1-pt)^gamma * log(pt),
    where pt is the probability of being classified to the true class.

    Reference:
    TF1: https://github.com/tensorflow/models/blob/master/official/vision/losses/focal_loss.py
    TF2: https://github.com/tensorflow/tpu/blob/master/models/official/detection/modeling/losses.py

    Args:
        logits: A float tensor of size [..., num_classes].
        targets: A float tensor of size [..., num_classes] represent one-hot labels.
            All 0 if it's a padding target.
        alpha: A float scalar multiplying alpha to the loss for positive examples
            and (1-alpha) to the loss for negative examples.
        gamma: A float scalar modulating loss for hard and easy examples.
        sample_weight: A float scalar or tensor of size [...] normalizes per sample loss.

    Returns:
        A float32 tensor representing the total loss.
    """
    logits = logits.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    is_positive_label = jnp.equal(targets, 1.0)

    cross_entropy_loss = sigmoid_cross_entropy_with_logits(logits, targets)

    probs = jax.nn.sigmoid(logits)
    probs_gt = jnp.where(is_positive_label, probs, 1.0 - probs)
    modulator = jnp.power(1.0 - probs_gt, gamma)

    loss = modulator * cross_entropy_loss
    weighted_loss = jnp.where(is_positive_label, alpha * loss, (1.0 - alpha) * loss)
    # Zero out losses on padding targets.
    # TODO(xianzhi): disable padding as it breaks focal loss. Will enable it later.
    # weighted_loss *= targets.sum(axis=-1, keepdims=True) > 0

    if sample_weight is not None:
        weighted_loss = weighted_loss * sample_weight

    return jnp.sum(weighted_loss)


def huber_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    delta: float = 1.0,
    sample_weight: Optional[Tensor] = None,
    reduce_axis: Optional[int] = None,
    reduction: ReductionMethod = ReductionMethod.NONE,
) -> Tensor:
    """Extend the optax.huber_loss with more functionalities.

    Args:
        predictions: A float tensor of shape [...] representing the predictions.
        targets: A float tensor of shape [...] representing the targets.
        delta: the bounds for the huber loss transformation, defaults at 1.
        sample_weight: A float tensor of shape [...] normalizes per sample loss.
        reduce_axis: if not None, take reduce mean of the loss along the given axis.
        reduction: The reduction method on the final loss.

    Returns:
        A float32 tensor representing the total loss.
    """
    # Optax huber loss: https://github.com/deepmind/optax/blob/master/optax/_src/loss.py#L60-L89
    loss = optax.huber_loss(predictions, targets, delta=delta)
    # Reduce mean along the given axis. The reduce axis can't be the sample dimension.
    # The main use case is to reduce the last axis, in order to match tf.keras.losses.Huber:
    # https://github.com/keras-team/keras/blob/b80dd12da9c0bc3f569eca3455e77762cf2ee8ef/keras/losses.py#L1882-L1889
    if reduce_axis is not None:
        loss = jnp.mean(loss, axis=reduce_axis)
    loss = _reduce_loss(loss=loss, reduction=reduction, sample_weight=sample_weight)
    return loss


def flops_loss(
    *,
    embeddings: Tensor,
    paddings: Optional[Tensor] = None,
    sparsity_threshold: float = 0.0,
) -> Tensor:
    """The FLOPs loss in 'Minimizing FLOPs to learn efficient sparse representations' ICLR2020.

    a_j = mean(|a_{ij}|, axis=0)
    L =  sum(a_j^2)

    Ref: https://openreview.net/pdf?id=SygpC6Ntvr The paragraph below Eq.3.

    Args:
        embeddings: A float tensor of shape [batch_size, vocab_size] or [batch_size, 1, vocab_size].
        paddings: A 0/1 tensor of shape [batch_size] where 1 means padding and 0 means valid
            example.
        sparsity_threshold: Embedding elements that are no greater than this threshold will be
            counted sparse.

    Returns:
        A tuple of (loss, average_sparsity_count):
        - loss: A float32 tensor representing the FLOPs loss.
        - average_sparsity_count: Average number of elements that are no greater than
            sparsity_threshold per example.
    """
    if len(embeddings.shape) == 3:
        assert embeddings.shape[1] == 1, "Invalid embeddings shape!"
        embeddings = jnp.squeeze(embeddings, axis=1)

    if paddings is None:
        paddings = jnp.zeros(embeddings.shape[0])

    is_valid = 1 - jnp.expand_dims(paddings, 1)
    is_negligible = jnp.abs(embeddings) <= sparsity_threshold
    average_sparsity_count = jnp.sum(
        jnp.sum(is_negligible, axis=-1, keepdims=True) * is_valid
    ) / jnp.sum(is_valid)
    per_dim_loss = jnp.sum(jnp.abs(embeddings) * is_valid, axis=0) / jnp.sum(is_valid)
    loss = jnp.sum(per_dim_loss**2)

    return loss, average_sparsity_count


def large_margin_cosine_loss(
    logits: Tensor,
    *,
    labels: Optional[Tensor] = None,
    soft_labels: Optional[Tensor] = None,
    alpha: float = 1.0,
    margin: float = 0.0,
) -> Tensor:
    """Loss based on https://arxiv.org/pdf/1801.09414.pdf.

    Assumes that `logits` are cosine distances between examples and class embeddings.

    Args:
        logits: a float Tensor of shape [..., num_classes].
        labels: an int Tensor of shape [...].
            Targets should contain the ground truth token ids in the range [0, num_classes).
            Out-of-class targets are ignored in the loss calculation.
        soft_labels: optional soft labels generated from data augmentation which should be used
            instead of one-hot labels.
        alpha: inverse temperature parameter.
        margin: margin for better separation between classes.

    Returns:
        A float Tensor represents the loss.

    Raises:
        ValueError: if neither labels nor soft_labels were provided.
    """
    if labels is None and soft_labels is None:
        raise ValueError("Neither labels nor soft_labels provided!")

    num_classes = logits.shape[-1]
    labels_onehot = (
        jax.nn.one_hot(labels, num_classes, dtype=logits.dtype)
        if soft_labels is None
        else soft_labels
    )

    original_logits = logits
    logits = alpha * (logits - margin * labels_onehot)

    per_example_loss = jnp.sum(-1 * labels_onehot * jax.nn.log_softmax(logits), axis=-1)
    num_examples = (labels_onehot.sum(axis=-1) > 0).sum()
    denominator = jnp.maximum(1, num_examples)
    loss = per_example_loss.sum() / denominator

    predictions = jnp.argmax(original_logits, axis=-1)
    gt_labels = jnp.argmax(labels_onehot, axis=-1) if labels is None else labels
    accuracy = jnp.equal(predictions, gt_labels).sum() / denominator

    return loss, {"accuracy": accuracy, "num_examples": num_examples}


def giou_loss(
    *,
    predictions: Tensor,
    targets: Tensor,
    reduction: ReductionMethod = ReductionMethod.NONE,
    sample_weight: Optional[Tensor] = None,
    eps: float = 1e-7,
) -> Tensor:
    """Generalized Intersection over Union loss.

    Reference: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/giou_loss.py#L6
    Paper: https://arxiv.org/abs/1902.09630

    Args:
        predictions: A float tensor of shape [..., 4] representing the box predictions.
            Box format is y1, x1, y2, x2.
        targets: A float tensor of shape [..., 4] representing the box targets.
            Box format is y1, x1, y2, x2.
        reduction: The reduction method on the final loss.
        sample_weight: A float tensor of shape [...] normalizes per sample loss.
        eps: Small number to prevent division by zero.

    Returns:
        A float32 tensor representing the loss.
    """

    def _valid_boxes(y1: Tensor, x1: Tensor, y2: Tensor, x2: Tensor) -> Tensor:
        return jnp.logical_and((y2 >= y1), (x2 >= x1))

    def _compute_area(y1: Tensor, x1: Tensor, y2: Tensor, x2: Tensor) -> Tensor:
        return (y2 - y1) * (x2 - x1)

    pred_y1, pred_x1, pred_y2, pred_x2 = jnp.moveaxis(predictions, -1, 0)
    targ_y1, targ_x1, targ_y2, targ_x2 = jnp.moveaxis(targets, -1, 0)

    # Intersection box
    intersect_y1 = jnp.maximum(pred_y1, targ_y1)
    intersect_x1 = jnp.maximum(pred_x1, targ_x1)
    intersect_y2 = jnp.minimum(pred_y2, targ_y2)
    intersect_x2 = jnp.minimum(pred_x2, targ_x2)

    # Smallest enclosing box
    enclosing_y1 = jnp.minimum(pred_y1, targ_y1)
    enclosing_x1 = jnp.minimum(pred_x1, targ_x1)
    enclosing_y2 = jnp.maximum(pred_y2, targ_y2)
    enclosing_x2 = jnp.maximum(pred_x2, targ_x2)

    intersection = _compute_area(intersect_y1, intersect_x1, intersect_y2, intersect_x2)
    intersection = intersection * _valid_boxes(
        intersect_y1, intersect_x1, intersect_y2, intersect_x2
    )

    pred_area = _compute_area(pred_y1, pred_x1, pred_y2, pred_x2)
    target_area = _compute_area(targ_y1, targ_x1, targ_y2, targ_x2)

    union = pred_area + target_area - intersection
    iou = intersection / (union + eps)

    enclosing_area = _compute_area(enclosing_y1, enclosing_x1, enclosing_y2, enclosing_x2)
    giou = iou - ((enclosing_area - union) / (enclosing_area + eps))

    loss = 1 - giou

    loss = _reduce_loss(loss=loss, reduction=reduction, sample_weight=sample_weight)
    return loss


def negative_cosine_similarity_loss(
    predictions: Tensor,
    targets: Tensor,
    *,
    normalize_embedding: bool = True,
    eps: float = 1e-8,
    live_targets: Optional[Tensor] = None,
    reduction: ReductionMethod = ReductionMethod.MEAN,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the negative cross similarity loss between predictions and targets.

    Args:
        predictions: A float Tensor of shape [..., dim].
        targets: A float Tensor of shape [..., dim].
        normalize_embedding: If True, apply normalization to embeddings.
        eps: minimum norm for terms in the denominator of the cosine similarity.
        live_targets: A bool or 0/1 Tensor of shape [...] indicates the valid positions for
            computing loss. 1 indicates positions that contribute to the loss.
        reduction: The reduction method.

    Returns:
        (loss, aux), where
        - loss is a scalar tensor representing the mean negative cosine similarity loss;
        - aux is a dictionary containing auxiliary outputs:
            * "cosine_similarity": A float Tensor of shape [...], representing cosine similarity.
            * "elementwise_similarity": A float Tensor of shape [..., dim], representing the
                elementwise cosine similarity.
    """
    # Compute l2 norm of targets and predictions.
    if normalize_embedding:
        targets = l2_normalize(targets, eps=eps)
        predictions = l2_normalize(predictions, eps=eps)
    # Compute the elementwise similarity.
    elementwise_similarity = targets * predictions
    # Compute cosine_similarity.
    cosine_similarity = jnp.sum(elementwise_similarity, axis=-1)
    loss = -1 * _reduce_loss(
        loss=cosine_similarity, reduction=reduction, sample_weight=live_targets
    )
    return loss, {
        "cosine_similarity": cosine_similarity,
        "elementwise_similarity": elementwise_similarity,
    }


def kl_divergence(
    log_predictions: Tensor,
    targets: Tensor,
    *,
    is_log_targets: bool = False,
) -> Tensor:
    """Computes the Kullback-Leibler divergence (relative entropy) loss.

    Measures the information gain achieved if target probability distribution
    would be used instead of predicted probability distribution.

    References: https://github.com/deepmind/optax/blob/master/optax/_src/loss.py#L524-L549

    Args:
        log_predictions: Probabilities of predicted distribution with shape [..., dim].
            Expected to be in the log-space to avoid underflow.
        targets: Probabilities of target distribution with shape [..., dim].
            Expected to be strictly positive.
        is_log_targets: Indicating the targets is in the log-space or not.

    Returns:
        (loss, aux), where
        - loss is a scalar Tensor representing the KL divergence loss;
        - aux is a dictionary containing auxiliary outputs:
            * per_example_loss: Per example KL divergence of predicted distribution from
                target distribution with shape [...].
    """
    if not is_log_targets:
        # We assume `predictions` is in the log-space and `targets` is probabilities.
        loss = targets * (jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions)
    else:
        # We assume `targets` and `predictions` are both in the log-space.
        loss = jnp.exp(targets) * (targets - log_predictions)
    per_example_loss = jnp.sum(loss, axis=-1)
    # TODO(xianzhi): support more reduction methods if needed.
    loss = jnp.mean(per_example_loss)
    return loss, {"per_example_loss": per_example_loss}


def koleo_loss(
    embeddings: Tensor,
    *,
    normalize_embedding: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Compute KoLeo loss for the given embedding.

    The loss derives from the Kozachenko-Leonenko differential entropy estimator, and encourages a
    uniform span of the features within a batch.

    Given a set of n vectors (x1, . . . , xn), it is defined as -1/n sum^n_{i=1} log(d_{n,i})
    where d_{n,i} = min_{j~=i} ||xi-xj|| is the minimum distance between xi and any other point
    within the batch.

    Ref: https://arxiv.org/abs/1806.03198

    Args:
        embeddings: A float tensor of shape [batch_size, dimension].
        normalize_embedding: If True, apply normalization to embeddings first.
        eps: eps for l2 norm and log. Defaults to 1e-8.

    Returns:
        a scalar tensor representing the loss.
    """
    if normalize_embedding:
        embeddings = l2_normalize(embeddings, eps=eps)
    # Compute the inner product between each embedding.
    dots = jnp.matmul(embeddings, embeddings.T)
    # Fill in the diagonal with -1.
    n = embeddings.shape[0]
    i, j = jnp.diag_indices(n)
    dots = dots.at[..., i, j].set(-1)
    # Find the one with max inner product, i.e. the one with min distance.
    max_indices = jnp.argmax(dots, axis=-1)
    # Compute L2 distance.
    # Alternatively, we can use:
    #   jnp.sqrt(jnp.sum((embeddings[max_indices] - embeddings) ** 2, axis=-1) + eps ** 2)
    # To keep the same implementation as torch, we keep the eps inside.
    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html
    distances = jnp.sqrt(jnp.sum((embeddings[max_indices] - embeddings + eps) ** 2, axis=-1))
    return -jnp.mean(jnp.log(distances + eps))


def pairwise_loss(
    *, logits: Tensor, pair_weights: Tensor, loss_scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Computes the mean pairwise loss from logits.

    Args:
        logits: A Tensor of shape [batch_size, num_docs_per_query].
        pair_weights: A Tensor of shape [batch_size, num_docs_per_query, num_docs_per_query], where
          pair_weights[b, i, j] > 0 if doc i should have a higher score than doc j,
          0 if pair (i, j) should be ignored for the loss.
        loss_scale: A Tensor of shape [batch_size,], showing the loss scale for each sample.

    Returns:
        (mean_loss, total_weight): the mean pairwise ranking loss
            among specified pairs and total pair weight.
    """
    # [batch_size, num_docs_per_query, num_docs_per_query].
    deltas = logits[:, :, None] - logits[:, None, :]
    # [batch_size, num_docs_per_query, num_docs_per_query].
    pairwise_loss_values = optax.sigmoid_binary_cross_entropy(
        logits=deltas, labels=jnp.greater(pair_weights, 0).astype(logits.dtype)
    )
    total_weight = pair_weights.sum()
    mean_loss = (pairwise_loss_values * pair_weights * loss_scale[:, None, None]).sum() / (
        1e-8 + total_weight
    )
    return mean_loss, total_weight


def ranking_pairwise_loss(
    *, logits: Tensor, ranks: Tensor, loss_scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Computes pairwise loss among ranked docs (ranks > 0).

    For every pair of docs (a, b) with 0 < rank_a < rank_b,
    i.e., (a, b) represent a pair of graded docs for the given query where doc a is ranked
    ahead of doc b, we compute pairwise loss with a binary classification loss
    where `logit = logit_a - logit_b`:

    ```
    optax.sigmoid_binary_cross_entropy(logit_a - logit_b, labels=1)
    ```

    Args:
        logits: A Tensor of shape [batch_size, num_docs_per_query].
        ranks: An int Tensor of shape [batch_size, num_docs_per_query], where positive values
            represent the ranking among docs (rank=1 means the most relevant doc), and 0 values
            represent negative docs.
            The implementation ignores any ranks that are not positive so in reality
            negative docs and paddings are treated the same way for this loss.
        loss_scale: A Tensor of shape [batch_size,], showing the loss scale for each sample.

    Returns:
        (mean_loss, num_pairs): the mean pairwise ranking loss among ranked doc pairs.
    """
    ranked = jnp.greater(ranks, 0)
    # ranked_pair[q, i, j] == True iff both i and j are ranked for query q and
    # ranks[q, i] < ranks[q, j] (doc i is more relevant than doc j.)
    ranked_pairs = (ranked[:, :, None] & jnp.less(ranks[:, :, None], ranks[:, None, :])).astype(
        logits.dtype
    )
    return pairwise_loss(logits=logits, pair_weights=ranked_pairs, loss_scale=loss_scale)
