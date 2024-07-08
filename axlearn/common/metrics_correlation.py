# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# scikit-learn/scikit-learn:
# Copyright (c) 2007-2023 The scikit-learn developers. All rights reserved.
# Licensed under BSD 3 clause.
#
# scipy/scipy:
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers. All rights reserved.
# Licensed under BSD 3 clause.

"""Correlation metrics like Pearson's r and Spearman's rho."""
from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.metrics_classification import confusion_matrix
from axlearn.common.utils import Tensor


def matthews_corrcoef(
    y_true: Tensor, y_pred: Tensor, *, weight: Optional[Tensor] = None, eps: float = 1e-8
) -> Tensor:
    """Computes Matthews correlation coefficient (MCC).

    References:
    https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d6dd034403370fea552b21a6776bef18/sklearn/metrics/_classification.py#L834
    https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case

    Args:
        y_true: Tensor of shape [batch_size] and values [0, num_classes) representing ground truth.
        y_pred: Tensor of shape [batch_size] and values [0, num_classes) representing predictions.
        weight: Optional Tensor of shape [batch_size] representing sample weights. If None, weights
            all samples equally.
        eps: Epsilon for numerical stability.

    Returns:
        A scalar Tensor in the range [-1,1] representing the MCC.
    """
    batch_size = y_true.shape[0]
    # Normalize targets and preds to [0, num_unique_classes-1].
    # https://github.com/scikit-learn/scikit-learn/blob/e2dd39194d613eb0f011450cc41831cc429c67c9/sklearn/metrics/_classification.py#L916-L919
    # Note that jit requires a static size, so we fill to batch_size with jnp.nan -- the subsequent
    # lookup will ignore those values, so long as the input itself contain all numbers.
    uniques = jnp.unique(
        jnp.hstack([y_true, y_pred]).reshape(-1).astype(jnp.float32),
        size=batch_size,
        fill_value=jnp.nan,
    )
    y_true = jnp.searchsorted(uniques, y_true)
    y_pred = jnp.searchsorted(uniques, y_pred)
    # Compute metrics.
    mat = confusion_matrix(y_true, y_pred, num_classes=batch_size, weight=weight)
    # Number of times each class truly occurred.
    t_sum = mat.sum(axis=1, dtype=jnp.float32)
    # Number of times each class was predicted.
    p_sum = mat.sum(axis=0, dtype=jnp.float32)
    # Number of samples correctly predicted.
    num_correct = jnp.trace(mat, dtype=jnp.float32)
    # Total number of samples.
    num_samples = p_sum.sum()
    cov_ytyp = num_correct * num_samples - jnp.dot(t_sum, p_sum)
    cov_ypyp = num_samples**2 - jnp.dot(p_sum, p_sum)
    cov_ytyt = num_samples**2 - jnp.dot(t_sum, t_sum)
    return cov_ytyp * jax.lax.rsqrt(cov_ytyt * cov_ypyp + eps)


def pearson_corrcoef(
    x: Tensor, y: Tensor, *, eps: float = 1e-8, weight: Optional[Tensor] = None
) -> Tensor:
    """Pearson's correlation coefficient (PCC).

    Note: This differs from jnp.corrcoef slightly, in that we compute a weighted PCC. The weights
    can e.g. be set to a 1/0 mask to handle ignored targets, where 0 will ignore the corresponding
    inputs in the computation.

    References:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient

    Args:
        x: A Tensor of shape [batch_size].
        y: A Tensor of shape [batch_size].
        weight: Optional Tensor of shape [batch_size] representing sample weights. If None, weights
            all samples equally.
        eps: Epsilon for numerical stability.

    Returns:
        A scalar Tensor in the range [-1, 1] representing the PCC.

    Raises:
        ValueError: If input shapes are invalid.
    """
    if weight is None:
        weight = jnp.ones_like(x)
    if not x.shape == y.shape == weight.shape:
        raise ValueError("Input shapes should be equal.")
    weight = weight / jnp.maximum(weight.sum(axis=-1), 1)

    # Centered x, y.
    x = x - (weight * x).sum(axis=-1)
    y = y - (weight * y).sum(axis=-1)

    # Compute weighted (co)variances.
    cov_xy = (weight * x * y).sum(axis=-1)
    var_x = (weight * x * x).sum(axis=-1)
    var_y = (weight * y * y).sum(axis=-1)

    # Compute weighted correlation.
    return cov_xy * jax.lax.rsqrt(var_x * var_y + eps)


def _rankdata(x: Tensor) -> Tensor:
    """Jax implementation of scipy.stats.rankdata using 'average' method.

    Assigns ranks to data, dealing with ties by averaging the rank of tied elements.

    References:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
    https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_stats_py.py#L9065-L9171

    Args:
        x: A Tensor of shape [batch_size].

    Returns:
        An Tensor of shape x.shape, containing rank scores.
    """
    sorter = jnp.argsort(x)

    inv = jnp.empty(sorter.size, dtype=jnp.int32)
    inv = inv.at[sorter].set(jnp.arange(sorter.size, dtype=jnp.int32))

    x = x[sorter]
    obs = jnp.r_[True, x[1:] != x[:-1]]
    dense = obs.cumsum()[inv]

    # Cumulative counts of each unique value.
    count = jnp.r_[jnp.nonzero(obs, size=obs.size, fill_value=obs.size)[0], obs.size]

    # Average method.
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman_corrcoef(x: Tensor, y: Tensor, *, eps: float = 1e-8, mask: Optional[Tensor] = None):
    """Spearman's correlation coefficient.

    This implementation supports binary masking. We treat masked indices as ignored,
    meaning we calculate the ranks as if those indices do not exist in the data.

    References:
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_stats_py.py#L4732-L4922


    Args:
        x: A Tensor of shape [batch_size].
        y: A Tensor of shape [batch_size].
        mask: Optional Tensor of shape [batch_size] that is boolean. If mask is None,
            no indices are masked.
        eps: Epsilon for numerical stability.

    Returns:
        A scalar Tensor in the range [-1, 1] representing the Spearman's corrcoef.

    Raises:
        ValueError: If input shapes are invalid.
    """
    if x.shape != y.shape:
        raise ValueError(f"Input shapes should be equal. Got x: {x.shape} and y: {y.shape}.")

    if mask is not None:
        if x.shape != mask.shape:
            raise ValueError(
                f"Input shapes should be equal. Got x: {x.shape} and mask: {mask.shape}."
            )

        # Replace masked elements with -inf so they will be ranked lowest
        x = jnp.where(mask != 0, x, -jnp.inf)
        y = jnp.where(mask != 0, y, -jnp.inf)

    ranked_x = _rankdata(x)
    ranked_y = _rankdata(y)

    return pearson_corrcoef(ranked_x, ranked_y, eps=eps, weight=mask)
