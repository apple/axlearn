# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# scikit-learn/scikit-learn:
# Copyright (c) 2007-2023 The scikit-learn developers. All rights reserved.
# Licensed under BSD 3 clause.

"""Classification metrics."""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from jax.experimental.sparse import BCOO
from jax.scipy.integrate import trapezoid

from axlearn.common.metrics import WeightedScalar
from axlearn.common.utils import Tensor


def confusion_matrix(
    y_true: Tensor, y_pred: Tensor, *, num_classes: int, weight: Optional[Tensor] = None
) -> Tensor:
    """Computes confusion matrix.

    References:
    https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d6dd034403370fea552b21a6776bef18/sklearn/metrics/_classification.py#L222

    Args:
        y_true: Tensor of shape [batch_size] and values [0, num_classes) representing ground truth.
        y_pred: Tensor of shape [batch_size] and values [0, num_classes) representing predictions.
        num_classes: Number of classes.
        weight: Optional Tensor of shape [batch_size] representing sample weights. If None, weights
            all samples equally.

    Returns:
        A Tensor C of shape [num_classes, num_classes] and values [0, batch_size] representing the
        un-normalized confusion matrix. C[i,j] represents the number of (weighted) samples with
        true label i and predicted label j.

    Raises:
        ValueError: If input shapes are invalid.
    """
    if weight is None:
        weight = jnp.ones_like(y_true)
    if not y_true.ndim == y_pred.ndim == weight.ndim == 1:
        raise ValueError("Inputs should all be rank 1.")
    if not y_true.shape == y_pred.shape == weight.shape:
        raise ValueError("Input shapes should be equal.")
    indices = jnp.stack([y_true, y_pred], axis=-1)
    return BCOO((weight, indices), shape=(num_classes, num_classes)).todense()


def precision_recall_f_score(
    y_true: Tensor,
    y_pred: Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1e-8,
    weight: Optional[Tensor] = None,
) -> dict[str, WeightedScalar]:
    """Computes precision, recall, and F-beta score for binary classification.

    References:
    https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d6dd034403370fea552b21a6776bef18/sklearn/metrics/_classification.py#L1389
    https://en.wikipedia.org/wiki/F-score

    Args:
        y_true: Tensor of shape [batch_size] and values [0,1] representing ground truth.
        y_pred: Tensor of shape [batch_size] and values [0,1] representing predictions.
        beta: Recall is considered beta times as important as precision. By default, beta=1, which
            computes the familiar F1 score.
        eps: Epsilon for numerical stability.
        weight: Optional Tensor of shape [batch_size] representing sample weights. If None, weights
            all samples equally.

    Returns:
        A dict with keys "precision", "recall", and "f_score". Each is a scalar Tensor with values
        in [0,1].

    Raises:
        ValueError: If beta is not positive.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    mat = confusion_matrix(y_true, y_pred, num_classes=2, weight=weight)
    tp = mat[1, 1]
    tp_fp = tp + mat[0, 1]  # TP + FP = number of positives predicted
    tp_fn = tp + mat[1, 0]  # TP + FN = number of positives
    beta2 = beta**2
    precision = tp / jnp.maximum(tp_fp, 1)
    recall = tp / jnp.maximum(tp_fn, 1)
    return dict(
        precision=WeightedScalar(precision, tp_fp),
        recall=WeightedScalar(recall, tp_fn),
        f_score=(1 + beta2) * precision * recall / (beta2 * precision + recall + eps),
    )


def f_score(
    y_true: Tensor,
    y_pred: Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1e-8,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Computes F-beta score for the positive class in binary classification.

    Please refer to `precision_recall_fscore`.
    """
    p_r_f = precision_recall_f_score(y_true, y_pred, beta=beta, eps=eps, weight=weight)
    return p_r_f["f_score"]


def binary_clf_curve(
    y_true: Tensor,
    y_score: Tensor,
    *,
    weight: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Calculate true and false positives per binary classification threshold.

    Please remember to assign padding samples weight 0 so that it will be ignored
    during calculation. y_scores with weight 0 will be masked as inf.

    References:
    https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/metrics/_ranking.py#L712

    Args:
        y_true: Int tensor of shape [num_samples] and value [0, 1]
            representing ground truth.
        y_score: Float tensor of shape [num_samples] and value (-inf, inf)
            representing model predicted scores.
        weight: Optional tensor of shape [num_samples] and values [0, inf) representing sample
            weights. If None, weights all samples equally.

    Returns:
        A dict with keys "fps", "tps", and "thresholds".
            Each is a scalar Tensor with values of shape [num_samples].
            fps, tps have values in [0, inf) and thresholds have values in (-inf, inf).
            The order is based on descending order of thresholds for unmasked examples. The
            jax.numpy.finfo(jnp.float32).max value in thresholds should be ignored.
            Element i in fps/tps are the tps/fps of predictions with score >= thresholds[i].
    """
    # Sort scores and corresponding truth unmasked values descending.
    if weight is not None:
        desc_y_pred_indices = jnp.argsort(y_score * (weight != 0))[::-1]
    else:
        desc_y_pred_indices = jnp.argsort(y_score)[::-1]
    thresholds = y_score[desc_y_pred_indices]
    y_true = y_true[desc_y_pred_indices]
    if weight is not None:
        weight = weight[desc_y_pred_indices]
        thresholds = jnp.where(weight.astype(bool), thresholds, jnp.finfo(jnp.float32).max)
    else:
        weight = jnp.ones_like(thresholds)

    tps = jnp.cumsum(y_true * weight)
    fps = jnp.cumsum((1 - y_true) * weight)

    # Handle cases when we have multiple predictions with the same y_score.
    # Different from sklearn implementation, we keep those indices with duplicated y_score due
    # to the need of maintaining a static shape to use jit.
    # Duplicated y_scores share the same fp, tp, achieved by assigning the rightmost tp, fp
    # value to all elements in the duplicate sequence.
    # Append 1 at the end so that y_pred_diff's shape matches tps/fps' shape, also to make sure
    # we include the last element.
    y_pred_diff = jnp.r_[jnp.diff(thresholds), 1]
    tps = jnp.where(y_pred_diff, tps, jnp.iinfo(jnp.int32).max)
    # Fill up padding values from the right, because the rightmost values of a duplicated
    # sequence contains the correct (maximum) tp, fp.
    tps = jax.lax.cummin(tps, reverse=True)
    fps = jnp.where(y_pred_diff, fps, jnp.iinfo(jnp.int32).max)
    fps = jax.lax.cummin(fps, reverse=True)

    return dict(fps=fps, tps=tps, thresholds=thresholds)


def precision_recall_curve(
    y_true: Tensor, y_score: Tensor, *, weight: Optional[Tensor] = None
) -> dict[str, Tensor]:
    """Compute precision-recall pairs for different probability thresholds.

    y_scores with weight 0 will be masked as inf and ignored during calculation.


    Note: this implementation is restricted to the binary classification task.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives.
    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold. This ensures that the graph starts on the
    y axis.
    Different from sklearn, we are not appending an extra 1 to precision nor
    an extra 0 to recall.

    References:
    https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/metrics/_ranking.py#L797

    Args:
        y_true : Int tensor of shape (n_samples,) and values [0,1]
            representing ground truth.
        y_score : Float tensor of shape (n_samples,) and values (-inf,inf)
            representing model predicted scores.
        weight : Optional tensor of shape (n_samples,) and values [0, inf),
            representing sample weights. If None, weights all samples equally.

    Returns:
        A dict with keys "precisions", "recalls", and "thresholds".
            Each is a scalar Tensor with values of shape [num_samples].
            precisions, recalls have values in [0, 1] and thresholds have values in (-inf, inf).
            The order is based on ascending order of thresholds.
            Element i in precision/recall are the precision/recall of predictions with
            score >= thresholds[i].
    """
    output = binary_clf_curve(y_true=y_true, y_score=y_score, weight=weight)
    fps, tps, thresholds = output["fps"], output["tps"], output["thresholds"]

    ps = tps + fps
    # Scale by ps which contains weight.
    precision = jnp.where(ps != 0, tps / ps, 0)
    # Recall will be 0, if total number of tps is 0. This is different from sklearn.
    recall = jnp.where(tps[-1] != 0, tps / tps[-1], 0)

    # Reverse the outputs so recall is decreasing.
    sl = slice(None, None, -1)
    return dict(precisions=precision[sl], recalls=recall[sl], thresholds=thresholds[sl])


def binary_classification_roc_auc_score(
    y_true: Tensor,
    y_score: Tensor,
    sample_weight: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) for binary
    classification model.

    Reference:
    https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_ranking.py#L365

    Args:
        y_true: Int tensor of shape [num_samples] where each value is 0 or 1 representing target
            label.
        y_score: Float tensor of shape [num_samples] representing model predicted scores.
        sample_weight: Optional tensor of shape [num_samples] representing sample weights.
            If None, weights all samples equally.

    Returns:
        (score, valid_inputs), where
        - score: A scalar Tensor with value in [0, 1] representing the AUC (ROC) score.
        - valid_input: False if AUC score can't be computed from input args, for example, when
          there is only one label; otherwise True.

    Raises:
        ValueError: if y_true has any value neither 0 nor 1.
    """
    has_binary_label = ((y_true == 0) | (y_true == 1)).all()
    checkify.check(has_binary_label, "y_true can only be 0 or 1!")

    if sample_weight is not None:
        valid_y_true = jnp.where(sample_weight > 0, True, False)
        has_multiple_labels = (jnp.sum(y_true, where=valid_y_true) > 0) & (
            jnp.sum(1 - y_true, where=valid_y_true) > 0
        )
    else:
        has_multiple_labels = (0 < y_true.sum()) & (y_true.sum() < y_true.shape[0])

    valid_input = has_multiple_labels & has_binary_label
    score = jax.lax.cond(
        valid_input,
        lambda: _compute_area_under_the_curve(y_true, y_score, sample_weight),
        # Return roc auc as 0.0 when there is only one label.
        lambda: 0.0,
    )
    return score, valid_input


def _compute_area_under_the_curve(
    y_true: Tensor,
    y_score: Tensor,
    sample_weight: Optional[Tensor] = None,
) -> Tensor:
    """Helper function to compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    It computes ROC AUC score for cases when both positive and negative labels exist in input
    samples. 'Args' and 'Returns' are the same with function binary_classification_roc_auc_score.
    """
    x, y = roc_curve(y_true, y_score, sample_weight=sample_weight)
    area = trapezoid(y, x)
    return area


def roc_curve(
    y_true: Tensor,
    y_score: Tensor,
    sample_weight: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Computes Receiver Operating Characteristic (ROC).

    References:
    https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_ranking.py#L892
    https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/metrics/_ranking.py#L703

    Args:
        y_true: Int tensor of shape [num_samples] where each value is 0 or 1 representing target
        y_score: Float tensor of shape [num_samples] representing model predicted scores.
        sample_weight: Optional Tensor of shape [num_samples] representing sample weights.
            If None, weights all samples equally.

    Returns:
        fpr: Tensor of shape [num_samples + 1] and values [0,1] representing false positive rate.
        tpr: Tensor of shape [num_samples + 1] and values [0,1] representing true positive rate.
    """
    output = binary_clf_curve(y_true=y_true, y_score=y_score, weight=sample_weight)
    tps = output["tps"]
    fps = output["fps"]

    tps = jnp.r_[0, tps]
    fps = jnp.r_[0, fps]

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return fpr, tpr


def brier_score(labels: Tensor, logits: Tensor) -> Tensor:
    """Compute Brier score for a probabilistic prediction.

    Given a probability vector p over possible outcomes, it is the mean squared error of the
    prediction computed as: sum((p[i] - l[i])**2), where l[k] is 1 if k is the actual
    outcome (label) and 0 otherwise.

    Args:
        labels: Tensor of shape [batch_size] and values [0, num_classes) representing ground truth.
        logits: Tensor of shape [batch_size, num_classes] representing the prediction logits.

    Returns:
        Brier score of shape [batch_size].
    """
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    oh_labels = jax.nn.one_hot(labels, logits.shape[-1])
    return jnp.sum(jnp.square(probs - oh_labels), axis=-1)
