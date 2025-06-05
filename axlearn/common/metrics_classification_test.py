# Copyright © 2023 Apple Inc.

"""Tests classification metrics."""

# pylint: disable=no-self-use
import logging

import evaluate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax import nn
from jax.experimental import checkify
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import fbeta_score as sklearn_fbeta_score
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.metrics._ranking import _binary_clf_curve as sklearn_binary_clf_curve

from axlearn.common.metrics_classification import (
    binary_classification_roc_auc_score,
    binary_clf_curve,
    brier_score,
    confusion_matrix,
    f_score,
    precision_recall_curve,
    precision_recall_f_score,
    roc_curve,
)
from axlearn.common.module import NestedTensor
from axlearn.common.test_utils import TestCase, TestWithTemporaryCWD, assert_allclose
from axlearn.common.utils import Tensor

IGNORE_TARGET_LABEL = -1


class TestMetrics(TestWithTemporaryCWD):
    """Tests metrics."""

    def test_confusion_matrix(self):
        batch_size = 100
        num_classes = 30

        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)
        weight = jax.random.uniform(jax.random.PRNGKey(111), [batch_size], jnp.float32, 0, 10)

        # Test equivalence with sklearn.
        expected = sklearn_confusion_matrix(
            label, pred, labels=np.arange(num_classes), sample_weight=weight
        )
        actual = jax.jit(confusion_matrix, static_argnames=("num_classes",))(
            label, pred, num_classes=num_classes, weight=weight
        )
        assert_allclose(expected, actual)
        assert_allclose(
            confusion_matrix(label, pred, num_classes=num_classes, weight=jnp.zeros_like(pred)), 0
        )

        # Test that repeats are summed.
        pred = jnp.zeros(batch_size, dtype=jnp.int32)
        label = jnp.zeros(batch_size, dtype=jnp.int32)
        self.assertEqual(jnp.squeeze(confusion_matrix(label, pred, num_classes=1)), batch_size)

    def test_confusion_matrix_validation(self):
        with self.assertRaisesRegex(ValueError, "rank 1"):
            confusion_matrix(jnp.ones((1, 2)), jnp.ones((1, 2)), num_classes=3)
        with self.assertRaisesRegex(ValueError, "shapes should be equal"):
            confusion_matrix(jnp.ones(5), jnp.ones(3), num_classes=3)

    @parameterized.parameters(0.5, 1, 2)
    def test_f_score(self, beta: float):
        batch_size = 100
        num_classes = 2

        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)
        weight = jax.random.uniform(jax.random.PRNGKey(111), [batch_size], jnp.float32, 0, 10)

        jit_f_score = jax.jit(f_score, static_argnames=("beta", "eps"))

        # Test equivalence with sklearn.
        expected = sklearn_fbeta_score(label, pred, beta=beta, sample_weight=weight)
        actual = jit_f_score(label, pred, beta=beta, weight=weight)
        assert_allclose(expected, actual)

    @pytest.mark.skip(reason="Intended to be run manually as it requires `evaluate.load`.")
    def test_f_score_hf(self):
        batch_size = 100
        num_classes = 2

        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)

        jit_f_score = jax.jit(f_score, static_argnames=("beta", "eps"))

        # Test equivalence with hf (which computes f1).
        actual = jit_f_score(pred, label, beta=1)
        metric = evaluate.load("glue", config_name="mrpc")
        expected = metric.compute(predictions=pred, references=label)["f1"]
        assert_allclose(expected, actual)

    def test_f_score_validation(self):
        with self.assertRaisesRegex(ValueError, "beta must be positive"):
            f_score(jnp.ones(5), jnp.ones(5), beta=0)

    @parameterized.parameters(0.5, 1, 2)
    def test_precision_recall_f_score(self, beta: float):
        batch_size = 100
        num_classes = 2

        pred = jax.random.randint(jax.random.PRNGKey(123), [batch_size], 0, num_classes)
        label = jax.random.randint(jax.random.PRNGKey(321), [batch_size], 0, num_classes)
        weight = jax.random.randint(jax.random.PRNGKey(111), [batch_size], 0, 3)

        jit_prf_score = jax.jit(precision_recall_f_score, static_argnames=("beta", "eps"))
        actual = jit_prf_score(label, pred, beta=beta, weight=weight)

        # Test equivalence with sklearn.
        expected = sklearn_fbeta_score(label, pred, beta=beta, sample_weight=weight)
        assert_allclose(expected, actual["f_score"])

        # Check weights.
        tp = (weight * (pred & label)).sum()
        self.assertEqual(actual["recall"].mean * actual["recall"].weight, tp)
        self.assertEqual(actual["precision"].mean * actual["precision"].weight, tp)

    PR_CURVE_TEST_CASES = [
        {
            # Test case where almost all samples have non-zero floating point weights.
            "y_score": jax.random.uniform(jax.random.PRNGKey(123), [100], jnp.float32, 0, 1),
            "y_true": jax.random.randint(jax.random.PRNGKey(321), [100], 0, 2),
            "weights": jax.random.uniform(jax.random.PRNGKey(111), [100], jnp.float32, 0, 10),
            "batch_size": 100,
        },
        {
            # Test case where some samples have 0 weights due to padding or 0-weight samples.
            "y_score": jax.random.uniform(jax.random.PRNGKey(123), [10], jnp.float32, 0, 1),
            "y_true": jax.random.randint(jax.random.PRNGKey(321), [10], 0, 2),
            "weights": jnp.ones(10),
            "batch_size": 15,
        },
        {
            # Test case where samples have tied thresholds.
            "y_score": jax.random.choice(
                jax.random.PRNGKey(111), jnp.array([0.5, 0.3, 0.2]), shape=[10]
            ),
            "y_true": jax.random.randint(jax.random.PRNGKey(321), [10], 0, 2),
            "weights": jnp.ones(10),
            "batch_size": 10,
        },
        {
            # Test case where some mistakes have been made.
            "y_score": jnp.array([1, 0]),
            "y_true": jnp.ones(2),
            "weights": jnp.ones(2),
            "batch_size": 2,
        },
        {
            # Test case where we have perfect predictions.
            "y_score": jnp.array([1, 0]),
            "y_true": jnp.array([1, 0]),
            "weights": jnp.ones(2),
            "batch_size": 2,
        },
        {
            # Test case where we have a complete failure.
            "y_score": jnp.array([1, 0]),
            "y_true": jnp.array([0, 1]),
            "weights": jnp.ones(2),
            "batch_size": 2,
        },
        {
            # Test case where we have a complete failure and all true positives.
            "y_score": jnp.array([0.25, 0.75]),
            "y_true": jnp.array([1, 1]),
            "weights": jnp.ones(2),
            "batch_size": 2,
        },
        {
            # Test case where we have float weights that sum up less than 1 with 1 positive.
            "y_score": jnp.array([0.25, 0.75]),
            "y_true": jnp.array([1, 1]),
            "weights": jnp.array([0.5, 0.0]),
            "batch_size": 2,
        },
        {
            # Test case where we have float weights that sum up less than 1 with 1 pos and 1 neg.
            "y_score": jnp.array([0.25, 0.5, 0.75]),
            "y_true": jnp.array([1, 0, 1]),
            "weights": jnp.array([0.5, 0.25, 0.0]),
            "batch_size": 3,
        },
        {
            # Test case where we have only one example.
            "y_score": jnp.array([0.25]),
            "y_true": jnp.array([1]),
            "weights": jnp.ones(1),
            "batch_size": 1,
        },
        {
            # Test case where some examples are masked out.
            "y_score": jnp.repeat(
                jnp.array(
                    [
                        [9.9655354e-01, 3.7714792e-04, 3.7637529e-01],
                        [1.7892958e-01, 3.9876872e-01, 9.9247831e-01],
                    ]
                ),
                5,
                axis=-1,
            ).reshape(-1),
            "y_true": jnp.array(
                [
                    [[0, 1, 1, 0, 1], [1, 0, 0, 1, 0], [0, 0, 1, 0, 0]],
                    [[1, 1, 0, 0, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]],
                ]
            )
            .reshape(-1)
            .astype(jnp.float32),
            "weights": jnp.array(
                [
                    [
                        [True, True, True, True, True],
                        [True, False, True, True, True],
                        [True, True, True, False, True],
                    ],
                    [
                        [True, True, True, True, False],
                        [True, True, True, True, True],
                        [False, False, False, False, False],
                    ],
                ]
            )
            .reshape(-1)
            .astype(jnp.float32),
            "batch_size": 30,
        },
        {
            # Test case where we have non-integer y_score and all true negatives.
            "y_score": jnp.array([0.25, 0.75]),
            "y_true": jnp.array([0, 0]),
            "weights": jnp.ones(2),
            "batch_size": 2,
        },
        {
            # Test case for the edge case where all samples are true negatives.
            "y_score": jax.random.uniform(jax.random.PRNGKey(123), [5], jnp.float32, 0, 1),
            "y_true": jnp.zeros(5),
            "weights": jnp.ones(5),
            "batch_size": 5,
        },
    ]

    @parameterized.parameters(PR_CURVE_TEST_CASES)
    def test_binary_clf_curve(
        self, y_score: NestedTensor, y_true: NestedTensor, weights: NestedTensor, batch_size: int
    ):
        if y_score.shape[0] < batch_size:
            preds_padded = -1 * jnp.ones(shape=batch_size)
            labels_padded = jnp.zeros(shape=batch_size)
            weights_padded = jnp.zeros(shape=batch_size)
            is_padding = jax.random.choice(
                jax.random.PRNGKey(111),
                jnp.array([False, True]),
                shape=[batch_size],
                p=jnp.array(
                    [(batch_size - y_score.shape[0]) / batch_size, y_score.shape[0] / batch_size]
                ),
            )
            non_padding_idx = jnp.nonzero(is_padding, size=y_score.shape[0])
            y_score = preds_padded.at[non_padding_idx].set(y_score)
            y_true = labels_padded.at[non_padding_idx].set(y_true)
            weights = weights_padded.at[non_padding_idx].set(weights)

        # Test under jit
        jit_f = jax.jit(binary_clf_curve)
        checked_jit_f = checkify.checkify(jit_f, errors=checkify.user_checks)
        _, output = checked_jit_f(y_true=y_true, y_score=y_score, weight=weights)
        fps, tps, thresholds = output["fps"], output["tps"], output["thresholds"]
        ref_fps, ref_tps, ref_thresholds = sklearn_binary_clf_curve(
            y_true=y_true, y_score=y_score, sample_weight=weights
        )

        # Our version of binary_clf_curve won't be able to remove thresholds that have 0 weight
        # from output due to static shape requirement. Adding 1 at the end so that we pick up the
        # last element.
        unique_mask = jnp.abs(jnp.r_[jnp.diff(thresholds), 1])
        valid_sample_mask = jnp.logical_and(thresholds != -1, unique_mask)
        valid_sample_idx = jnp.nonzero(valid_sample_mask, size=ref_fps.shape[0])

        assert_allclose(fps[valid_sample_idx], ref_fps)
        assert_allclose(tps[valid_sample_idx], ref_tps)
        assert_allclose(thresholds[valid_sample_idx], ref_thresholds)

    @parameterized.parameters(
        {
            # Test to ensure we fill the duplicated thresholds with correct tps, fps
            "y_score": jnp.array([0.4, 0.8, 0.4, 0.6, 0.1, 0.1]),
            "y_true": jnp.array([0, 1, 0, 1, 1, 0]),
            "weights": jnp.ones(6),
            "ref_tps": jnp.array([1, 2, 2, 2, 3, 3]),
            "ref_fps": jnp.array([0, 0, 2, 2, 3, 3]),
            "ref_thresholds": jnp.array([0.8, 0.6, 0.4, 0.4, 0.1, 0.1]),
        }
    )
    def test_binary_clf_curve_custom(
        self,
        y_score: NestedTensor,
        y_true: NestedTensor,
        weights: NestedTensor,
        ref_tps: NestedTensor,
        ref_fps: NestedTensor,
        ref_thresholds: NestedTensor,
    ):
        jit_f = jax.jit(binary_clf_curve)
        checked_jit_f = checkify.checkify(jit_f, errors=checkify.user_checks)
        _, output = checked_jit_f(y_true=y_true, y_score=y_score, weight=weights)
        fps, tps, thresholds = output["fps"], output["tps"], output["thresholds"]
        assert_allclose(fps, ref_fps)
        assert_allclose(tps, ref_tps)
        assert_allclose(thresholds, ref_thresholds)

    # We are not testing on the case where all samples are true negatives in this function
    # due to different behaviors from sklearn.
    @parameterized.parameters(PR_CURVE_TEST_CASES[:-2])
    def test_precision_recall_curve(
        self, y_score: NestedTensor, y_true: NestedTensor, weights: NestedTensor, batch_size: int
    ):
        if y_score.shape[0] < batch_size:
            preds_padded = -1 * jnp.ones(shape=batch_size)
            labels_padded = jnp.zeros(shape=batch_size)
            weights_padded = jnp.zeros(shape=batch_size)
            is_padding = jax.random.choice(
                jax.random.PRNGKey(111),
                jnp.array([False, True]),
                shape=[batch_size],
                p=jnp.array(
                    [(batch_size - y_score.shape[0]) / batch_size, y_score.shape[0] / batch_size]
                ),
            )
            non_padding_idx = jnp.where(is_padding, size=y_score.shape[0])
            y_score = preds_padded.at[non_padding_idx].set(y_score)
            y_true = labels_padded.at[non_padding_idx].set(y_true)
            weights = weights_padded.at[non_padding_idx].set(weights)

        jit_f = jax.jit(precision_recall_curve)
        checked_jit_f = checkify.checkify(jit_f, errors=checkify.user_checks)
        _, output = checked_jit_f(y_true=y_true, y_score=y_score, weight=weights)
        precisions, recalls, thresholds = (
            output["precisions"],
            output["recalls"],
            output["thresholds"],
        )

        ref_precisions, ref_recalls, ref_thresholds = sklearn_precision_recall_curve(
            y_true=y_true,
            probas_pred=y_score,
            sample_weight=weights,
        )

        # Since we keep duplicated threshold, we need to select only the unique elements to compare
        # with sklearn. Adding 1 at the beginning so that we pick up the first element.
        unique_mask = jnp.abs(jnp.r_[1, jnp.diff(thresholds)])
        valid_sample_mask = jnp.logical_and(thresholds != jnp.finfo(jnp.float32).max, unique_mask)
        valid_sample_idx = jnp.nonzero(valid_sample_mask, size=ref_thresholds.shape[0])[0]

        # Adding index -1 to the end of index since we are not including the extra element
        # during previous masking.
        assert_allclose(precisions[valid_sample_idx], ref_precisions[:-1])
        assert_allclose(recalls[valid_sample_idx], ref_recalls[:-1])
        assert_allclose(thresholds[valid_sample_idx], ref_thresholds)

    @parameterized.parameters(PR_CURVE_TEST_CASES[-2:])
    def test_precision_recall_curve_no_tps(
        self,
        y_score: NestedTensor,
        y_true: NestedTensor,
        weights: NestedTensor,
        batch_size: int,  # pylint: disable=unused-argument
    ):
        jit_f = jax.jit(precision_recall_curve)
        checked_jit_f = checkify.checkify(jit_f, errors=checkify.user_checks)
        _, output = checked_jit_f(y_true=y_true, y_score=y_score, weight=weights)
        precisions, recalls, thresholds = (
            output["precisions"],
            output["recalls"],
            output["thresholds"],
        )

        # Our precision recall curve will return all 0s for cases where there is no tp.
        # For recall, sklearn has all 1s expect the last one which is 0.
        # For precision, sklearn has all 0s except the last one which is 1.

        ref_precisions = jnp.zeros(shape=[len(precisions)])
        ref_recalls = jnp.zeros(shape=[len(recalls)])
        ref_thresholds = jnp.sort(thresholds)

        assert_allclose(precisions, ref_precisions)
        assert_allclose(recalls, ref_recalls)
        assert_allclose(thresholds, ref_thresholds)

    def get_random_input_binary_classification(
        self, num_samples: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        pred = jax.random.uniform(jax.random.PRNGKey(123), [num_samples], minval=0, maxval=1)
        label = jax.random.randint(jax.random.PRNGKey(321), [num_samples], 0, 2)
        sample_weight = jax.random.uniform(
            jax.random.PRNGKey(111), [num_samples], minval=0, maxval=1
        )
        return pred, label, sample_weight

    def test_binary_classification_auc_same_prediction(self):
        label = jnp.array([1, 1, 0, 1, 0])
        pred = jnp.array([0, 0.5, 0.5, 0.5, 1])
        sample_weight = jax.random.uniform(jax.random.PRNGKey(111), [5], minval=0, maxval=1)
        actual, _ = binary_classification_roc_auc_score(label, pred, sample_weight=sample_weight)
        expected = sklearn_roc_auc_score(label, pred, sample_weight=sample_weight)
        assert_allclose(expected, actual)

    def test_binary_classification_auc(self):
        pred, label, sample_weight = self.get_random_input_binary_classification(64)
        expected = sklearn_roc_auc_score(label, pred, sample_weight=sample_weight)
        actual, valid_input = binary_classification_roc_auc_score(
            label, pred, sample_weight=sample_weight
        )
        assert_allclose(expected, actual)
        self.assertEqual(valid_input, True)

        # Test under jit
        jit_f = jax.jit(binary_classification_roc_auc_score)
        checked_jit_f = checkify.checkify(jit_f, errors=checkify.user_checks)
        _, (actual, _) = checked_jit_f(label, pred, sample_weight=sample_weight)
        assert_allclose(expected, actual)

    def test_binary_classification_auc_one_input_label(self):
        label = jnp.array([0, 0, 0, 0])
        pred = jnp.array([0, 0.5, 0.5, 1])
        sample_weight = jax.random.uniform(jax.random.PRNGKey(111), [4], minval=0, maxval=1)
        actual, valid_input = binary_classification_roc_auc_score(
            label, pred, sample_weight=sample_weight
        )
        assert_allclose(0.0, actual)
        self.assertEqual(valid_input, False)

    def test_binary_classification_auc_zero_weight(self):
        label = jnp.array([1, 0, 1, 0])
        pred = jnp.array([0, 0.5, 0.5, 0.5])
        sample_weight = jnp.array([1, 0, 1, 0])
        actual, valid_input = binary_classification_roc_auc_score(
            label, pred, sample_weight=sample_weight
        )
        assert_allclose(actual, 0.0)
        self.assertEqual(valid_input, False)

        sample_weight = jnp.array([1, 1, 1, 0])
        actual, _ = binary_classification_roc_auc_score(label, pred, sample_weight=sample_weight)
        expected = sklearn_roc_auc_score(label, pred, sample_weight=sample_weight)
        assert_allclose(actual, expected)

        label = jnp.array([1, 0, 1, 0])
        pred = jnp.array([0, 1, 0, 1])
        sample_weight = jnp.array([1, 1, 1, 1])
        actual, _ = binary_classification_roc_auc_score(label, pred, sample_weight=sample_weight)
        expected = sklearn_roc_auc_score(label, pred, sample_weight=sample_weight)
        assert_allclose(actual, expected)

    def test_binary_classification_non_binary_label(self):
        label = jnp.array([-1, 0])
        pred = jnp.array([0, 0.5])
        sample_weight = jnp.array([1, 1])
        jit_f = jax.jit(binary_classification_roc_auc_score)
        checked_jit_f = checkify.checkify(jit_f, errors=checkify.user_checks)
        errors, (actual, valid_input) = checked_jit_f(label, pred, sample_weight=sample_weight)
        logging.info(errors.get())
        assert_allclose(0.0, actual)
        self.assertEqual(valid_input, False)

    def test_roc_curve(self):
        pred, label, sample_weight = self.get_random_input_binary_classification(64)
        fpr_actual, tpr_actual = roc_curve(label, pred, sample_weight)
        fpr_expected, tpr_expected, _ = sklearn_roc_curve(label, pred, sample_weight=sample_weight)
        assert_allclose(fpr_actual, fpr_expected)
        assert_allclose(tpr_actual, tpr_expected)

    @parameterized.parameters(
        {
            "labels": jnp.array([2]),
            "logits": jnp.array([1, 2, 3]),
            "expected": 0.1800611607097189,
        },
        {
            "labels": jnp.array([2, 0]),
            "logits": jnp.array([[1, 2, 3], [1, -1e10, -1e10]]),
            "expected": [0.1800611607097189, 0],
        },
    )
    def test_brier_score(self, labels, logits, expected):
        actual = brier_score(labels=labels, logits=logits)
        assert_allclose(actual, expected)


class MultiLabelMetricCalculatorTest(TestCase):
    def test_multi_label(self):
        """This function adds tests for multilabel and considers
        scikit-learn the ground truth.
        The tests cover:
            1. No errors
            2. Recall errors
            3. Precision errors
            4. Batched inputs with precision/recall errors
        """
        num_classes = 10

        # Test no mistakes.
        targets = jnp.array([[1, 5, 9, 2, -1]])
        logits = jnp.array([[-1, 1, 1, -1, -1, 1, -1, -1, -1, 1]], dtype=jnp.float32)
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        targets = jnp.sum(jax.nn.one_hot(targets, num_classes), axis=1).astype(jnp.int32)
        mask = jnp.any(live_targets, axis=-1).reshape(-1, 1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(
            sklearn_recall_score(targets, preds, average="micro"), summaries["recall"].mean
        )
        self.assertEqual(
            sklearn_precision_score(targets, preds, average="micro"),
            summaries["precision"].mean,
        )

        # Test Recall mistake.
        targets = jnp.array([[1, 5, -1, -1, -1]])
        logits = jnp.array([[-1, 1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=jnp.float32)
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        targets = jnp.sum(jax.nn.one_hot(targets, num_classes), axis=1).astype(jnp.int32)
        mask = jnp.any(live_targets, axis=-1).reshape(-1, 1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(
            sklearn_recall_score(targets, preds, average="micro"), summaries["recall"].mean
        )
        self.assertEqual(
            sklearn_precision_score(targets, preds, average="micro"),
            summaries["precision"].mean,
        )

        # Test precision mistake.
        targets = jnp.array([[1, 5, 9, 2, -1]])
        logits = jnp.array([[-1, 1, 1, -1, 1, 1, -1, -1, -1, 1]], dtype=jnp.float32)
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        targets = jnp.sum(jax.nn.one_hot(targets, num_classes), axis=1).astype(jnp.int32)
        mask = jnp.any(live_targets, axis=-1).reshape(-1, 1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(
            sklearn_recall_score(targets, preds, average="micro"), summaries["recall"].mean
        )
        self.assertEqual(
            sklearn_precision_score(targets, preds, average="micro"),
            summaries["precision"].mean,
        )

        # Test batch.
        targets = jnp.array([[1, 5, 9, 2, -1], [1, 5, -1, -1, -1]])
        logits = jnp.array(
            [[-1, 1, 1, -1, 1, 1, -1, -1, -1, 1], [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1]],
            dtype=jnp.float32,
        )
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        targets = jnp.sum(jax.nn.one_hot(targets, num_classes), axis=1).astype(jnp.int32)
        mask = jnp.any(live_targets, axis=-1).reshape(-1, 1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(
            sklearn_recall_score(targets, preds, average="micro"), summaries["recall"].mean
        )
        self.assertEqual(
            sklearn_precision_score(targets, preds, average="micro"),
            summaries["precision"].mean,
        )

        # Live Test
        targets = jnp.array([[1, -1], [-1, -1], [0, -1], [0, -1]])
        logits = jnp.array(
            [
                [0.006889927200973034, -0.008306934498250484],
                [0.0027843592688441277, -0.0099781583994627],
                [0.006896782200783491, -0.00829133577644825],
                [0.00032793590798974037, -0.002820595633238554],
            ],
            dtype=jnp.float32,
        )
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        targets = jnp.sum(jax.nn.one_hot(targets, 2), axis=1).astype(jnp.int32)
        mask = jnp.any(live_targets, axis=-1).reshape(-1, 1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        mask, _ = np.where(mask)
        targets = targets[mask]
        preds = preds[mask]
        self.assertEqual(
            sklearn_recall_score(targets, preds, average="micro"), summaries["recall"].mean
        )
        self.assertEqual(
            sklearn_precision_score(targets, preds, average="micro"),
            summaries["precision"].mean,
        )

    def test_binary_label(self):
        """This function adds tests for binary and considers
        sklearn the ground truth.
        The tests cover:
            1. No errors
            2. No positives
            2. Recall errors
            3. Precision errors
        """
        # Test no mistakes.
        targets = jnp.array([[1], [0]])
        logits = jnp.array([[1], [-1]], dtype=jnp.float32)
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        mask = jnp.any(live_targets, axis=-1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(sklearn_recall_score(targets, preds), summaries["recall"].mean)
        self.assertEqual(
            sklearn_precision_score(targets, preds),
            summaries["precision"].mean,
        )

        # Test mistake with no positives.
        targets = jnp.array([[0], [0]])
        logits = jnp.array([[1], [-1]], dtype=jnp.float32)
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        mask = jnp.any(live_targets, axis=-1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(sklearn_recall_score(targets, preds), summaries["recall"].mean)
        self.assertEqual(
            sklearn_precision_score(targets, preds),
            summaries["precision"].mean,
        )

        # Test mistake with recall.
        targets = jnp.array([[1], [1]])
        logits = jnp.array([[1], [-1]], dtype=jnp.float32)
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        mask = jnp.any(live_targets, axis=-1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(sklearn_recall_score(targets, preds), summaries["recall"].mean)
        self.assertEqual(
            sklearn_precision_score(targets, preds),
            summaries["precision"].mean,
        )

        # Test mistake with precision.
        targets = jnp.array([[1], [0]])
        logits = jnp.array([[1], [1]], dtype=jnp.float32)
        live_targets = (targets != IGNORE_TARGET_LABEL).astype(logits.dtype)
        mask = jnp.any(live_targets, axis=-1)
        preds = jnp.where(jnp.greater(nn.sigmoid(logits), 0.5), 1, 0)
        summaries = precision_recall_f_score(
            y_true=(mask * targets).reshape(-1),
            y_pred=(mask * preds).reshape(-1),
        )
        self.assertEqual(sklearn_recall_score(targets, preds), summaries["recall"].mean)
        self.assertEqual(
            sklearn_precision_score(targets, preds),
            summaries["precision"].mean,
        )


if __name__ == "__main__":
    absltest.main()
