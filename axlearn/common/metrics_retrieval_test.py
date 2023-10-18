# Copyright © 2023 Apple Inc.

"""Tests retrieval metrics."""
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from sklearn.metrics import ndcg_score

from axlearn.common.loss import contrastive_logits
from axlearn.common.metrics_retrieval import (
    average_precision_at_k,
    average_rank,
    calculate_accuracy_metrics,
    calculate_mean_average_precision_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    top_k_accuracy,
)
from axlearn.common.test_utils import TestCase


class TopKAccuracyTest(TestCase):
    def setUp(self):
        super().setUp()
        # Similar code in evaler_zero_shot_classification_test.
        # pylint: disable=duplicate-code
        x = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        targets = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0.5, 1],
        ]
        # pylint: enable=duplicate-code

        self.x = jnp.asarray(x)
        self.targets = jnp.asarray(targets)
        self.sim = contrastive_logits(self.x, self.targets)

    @parameterized.parameters(
        [
            ([[0], [1], [2]], [1], [[1, 1, 1]]),
            ([[3], [4], [5]], [1], [[0, 0, 0]]),
            ([[0, 3], [-1, 4], [-1, 5]], [1], [[1, 0, 0]]),
            ([[3], [4], [5]], [2], [[1, 1, 1]]),
            ([[1], [1], [1]], [2], [[0, 1, 0]]),
            ([[1, 0], [1, -1], [1, -1]], [2], [[1, 1, 0]]),
            ([[3], [4], [5]], [1, 2], [[0, 0, 0], [1, 1, 1]]),
            ([[3, -1], [4, -1], [5, -1]], [1, 2], [[0, 0, 0], [1, 1, 1]]),
        ]
    )
    def test_output_value(self, gt_targets, top_ks, expected):
        out = top_k_accuracy(self.sim, gt_targets=jnp.asarray(gt_targets), top_ks=top_ks)
        np.testing.assert_equal(np.asarray(out), expected)

        relevance_labels = jnp.sum(jax.nn.one_hot(gt_targets, self.sim.shape[1]), 1)
        out = top_k_accuracy(
            self.sim, gt_targets=None, relevance_labels=relevance_labels, top_ks=top_ks
        )
        np.testing.assert_equal(np.asarray(out), expected)

    def test_counts(self):
        out = top_k_accuracy(
            self.sim, gt_targets=jnp.asarray([[1], [4], [2]]), top_ks=[1, 2, 3], return_counts=True
        )
        np.testing.assert_equal(np.asarray(out), [[0, 0, 1], [0, 1, 1], [1, 1, 1]])

        out = top_k_accuracy(
            self.sim,
            gt_targets=jnp.asarray([[1, 3], [4, 5], [2, -1]]),
            top_ks=[1, 2, 3],
            return_counts=True,
        )
        np.testing.assert_equal(np.asarray(out), [[0, 0, 1], [1, 1, 1], [2, 2, 1]])

    def test_similarity_bias(self):
        similarity_bias = np.zeros(self.targets.shape[0])
        similarity_bias[0] = -100000

        out = top_k_accuracy(
            self.sim,
            gt_targets=jnp.asarray([[0], [1], [2]]),
            top_ks=[1],
            similarity_bias=similarity_bias,
        )
        np.testing.assert_equal(np.asarray(out), [[0, 1, 1]])

        out = top_k_accuracy(
            self.sim,
            gt_targets=jnp.asarray([[0], [1], [2]]),
            top_ks=[2],
            similarity_bias=similarity_bias,
        )
        np.testing.assert_equal(np.asarray(out), [[0, 1, 1]])

    def test_similarity_bias_2d(self):
        similarity_bias = np.zeros_like(self.sim)
        similarity_bias[2] = -100000
        similarity_bias[:, 3] = -100000

        out = top_k_accuracy(
            self.sim,
            gt_targets=jnp.asarray([[0, -1, -1], [2, 4, 5], [-1, -1, -1]]),
            top_ks=[1, 2, 3],
            similarity_bias=similarity_bias,
        )
        np.testing.assert_equal(np.asarray(out), [[1, 0, 0], [1, 1, 0], [1, 1, 0]])

        out = top_k_accuracy(
            self.sim,
            gt_targets=jnp.asarray([[0, -1, -1], [2, 4, 5], [-1, -1, -1]]),
            top_ks=[1, 2, 3],
            similarity_bias=similarity_bias,
            return_counts=True,
        )
        np.testing.assert_equal(np.asarray(out), [[1, 0, 0], [1, 1, 0], [1, 2, 0]])


def test_average_precision_at_k():
    sim = jnp.asarray([[1, 5, 6, 7, 2, 4, 3], [4, 3, 2, 5, 1, 7, 6], [1, 1, 1, 1, 1, 1, 1]])
    relevance_labels = jnp.asarray(
        [[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    top_ks_for_map = [1, 2, 3, -1]
    maps = average_precision_at_k(sim, relevance_labels, top_ks_for_map)
    expected_maps = {
        1: [(1 / 1) / 1, 1 / 1, 0.0],
        2: [(1 / 1 + 2 / 2) / 2, (1 / 1 + 0) / 2, 0.0],
        3: [(1 / 1 + 2 / 2 + 0) / 3, (1 / 1 + 0 + 0) / 3, 0.0],
        -1: [
            (1 / 1 + 2 / 2 + 0 + 3 / 4 + 4 / 5 + 0 + 5 / 7) / 5,
            (1 / 1 + 0 + 0 + 0 + 2 / 5 + 3 / 6 + 4 / 7) / 4,
            0.0,
        ],
    }

    for k in top_ks_for_map:
        assert jnp.allclose(maps[k], jnp.asarray(expected_maps[k]))

    top_ks_for_map = [1, 2, 3]
    maps = average_precision_at_k(sim, relevance_labels, top_ks_for_map)
    for k in top_ks_for_map:
        assert jnp.allclose(maps[k], jnp.asarray(expected_maps[k]))

    with pytest.raises(AssertionError):
        top_ks_for_map = [1, 2, 999]
        average_precision_at_k(sim, relevance_labels, top_ks_for_map)


class NDCGTest(TestCase):
    @parameterized.parameters(
        {
            "scores": [1, 5, 6, 7, 2, 4, 3],
            "relevance_labels": [2, 0, 1, 5, 0, 7, 8],
        },
        {
            "scores": [4, 3, 2, 5, 1, 7, 6],
            "relevance_labels": [0, 4, 2, 0, 5, 3, 0],
        },
        {
            "scores": [1, 1, 1, 1, 1, 1, 1],
            "relevance_labels": [0, 0, 0, 0, 0, 0, 0],
        },
        {
            "scores": [1, 1, 1, 1, 1, 1, 1],
            "relevance_labels": [0, 0, 0, 0, 2, 2, 3],
        },
    )
    def test_ndcg_at_k(self, scores: List[float], relevance_labels: List[float]):
        scores = jnp.array([scores])
        relevance_labels = jnp.array([relevance_labels])
        top_ks_for_ndcg = [1, 2, 3, -1]
        ndcgs = ndcg_at_k(scores=scores, relevance_labels=relevance_labels, top_ks=top_ks_for_ndcg)

        for k in top_ks_for_ndcg:
            if k == -1:
                self.assertAlmostEqual(
                    ndcg_score(y_true=relevance_labels, y_score=scores, k=7),
                    ndcgs[k].item(),
                    places=6,
                )
            else:
                self.assertAlmostEqual(
                    ndcg_score(y_true=relevance_labels, y_score=scores, k=k),
                    ndcgs[k].item(),
                    places=6,
                )


def test_average_rank():
    scores = jnp.asarray([[1, 5, 6, 7, 2, 4, 3], [4, 3, 2, 5, 1, 7, 6], [1, 1, 1, 1, 1, 1, 1]])
    relevance_labels = jnp.asarray(
        [[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0]]
    )
    query_padding = jnp.asarray([[0], [0], [1]])
    avg_rank = average_rank(
        scores=scores, relevance_labels=relevance_labels, query_padding=query_padding
    )["avg_rank"]
    expected_rank = (1 + 5) / 2
    assert expected_rank == avg_rank


def test_average_rank_with_no_relevant_item_query():
    scores = jnp.asarray(
        [[1, 5, 6, 7, 2, 4, 3], [4, 3, 2, 5, 1, 7, 6], [1, 1, 1, 1, 1, 1, 1], [4, 3, 2, 5, 1, 7, 6]]
    )
    relevance_labels = jnp.asarray(
        [[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    query_padding = jnp.asarray([[0], [0], [1], [0]])
    avg_rank = average_rank(
        scores=scores, relevance_labels=relevance_labels, query_padding=query_padding
    )["avg_rank"]
    expected_rank = (1 + 5) / 2
    assert expected_rank == avg_rank


def test_mrr():
    scores = jnp.asarray([[1, 5, 6, 7, 2, 4, 3], [4, 3, 2, 5, 1, 7, 6], [1, 1, 1, 1, 1, 1, 1]])
    relevance_labels = jnp.asarray(
        [[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0]]
    )
    query_padding = jnp.asarray([[0], [0], [1]])
    mrr = mean_reciprocal_rank(
        scores=scores, relevance_labels=relevance_labels, query_padding=query_padding
    )["mrr"]
    expected_mrr = (1 / 1 + 1 / 5) / 2
    assert expected_mrr == mrr


def test_mrr_with_no_relevant_item_query():
    scores = jnp.asarray(
        [[1, 5, 6, 7, 2, 4, 3], [4, 3, 2, 5, 1, 7, 6], [1, 1, 1, 1, 1, 1, 1], [4, 3, 2, 5, 1, 7, 6]]
    )
    relevance_labels = jnp.asarray(
        [[1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    )
    query_padding = jnp.asarray([[0], [0], [1], [0]])
    mrr = mean_reciprocal_rank(
        scores=scores, relevance_labels=relevance_labels, query_padding=query_padding
    )["mrr"]
    expected_mrr = (1 / 1 + 1 / 5) / 2
    assert expected_mrr == mrr


class CalculateMeanMetricTest(TestCase):
    def test_calculate_mean_average_precision(self):
        metrics = calculate_mean_average_precision_metrics(
            top_ks=[1, 2],
            scores=jnp.array([[0.5, 1.0, 0.0], [0.5, 0.0, 1.0], [0.5, 0.0, 1.0], [0.5, 0.0, 1.0]]),
            relevance_labels=jnp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
            query_padding=jnp.array([False, False, False, True]),
            categories=jnp.array([0, 1, 0, 0]),
            categories_names=("cat", "dog", "unknown"),
        )
        expected_metrics = {
            "MAP@1": (1.0 + 0.0 + 0.0) / 3,
            "MAP@1_cat": (1.0 + 0.0) / 2,
            "MAP@1_dog": 0.0,
            "MAP@1_avg_category": ((1.0 + 0.0) / 2 + 0.0) / 2,
            "MAP@2": (0.5 + 0.5 + 0.5) / 3,
            "MAP@2_cat": (0.5 + 0.5) / 2,
            "MAP@2_dog": 0.5,
            "MAP@2_avg_category": ((0.5 + 0.5) / 2 + 0.5) / 2,
        }
        self.assertNestedAllClose(expected_metrics, metrics)

    def test_calculate_accuracy(self):
        metrics = calculate_accuracy_metrics(
            top_ks=[1, 2],
            scores=jnp.array([[0.5, 1.0, 0.0], [0.5, 0.0, 1.0], [0.5, 0.0, 1.0], [0.5, 0.0, 1.0]]),
            relevance_labels=jnp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
            query_padding=jnp.array([False, False, False, True]),
            categories=jnp.array([0, 1, 0, 0]),
            categories_names=("cat", "dog"),
        )
        expected_metrics = {
            "accuracy@1": (1.0 + 0.0 + 0.0) / 3,
            "accuracy@1_cat": (1.0 + 0.0) / 2,
            "accuracy@1_dog": 0.0,
            "accuracy@1_avg_category": ((1.0 + 0.0) / 2 + 0.0) / 2,
            "accuracy@2": (1.0 + 1.0 + 1.0) / 3,
            "accuracy@2_cat": (1.0 + 1.0) / 2,
            "accuracy@2_dog": 1.0,
            "accuracy@2_avg_category": ((1.0 + 1.0) / 2 + 1.0) / 2,
        }
        self.assertNestedAllClose(expected_metrics, metrics)

    def test_raises_on_missing_names(self):
        with self.assertRaisesRegex(ValueError, ".*categories_names"):
            calculate_mean_average_precision_metrics(
                top_ks=[1, 2],
                scores=jnp.array([[0.5, 1.0, 0.0], [0.5, 0.0, 1.0], [0.5, 0.0, 1.0]]),
                relevance_labels=jnp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                categories=jnp.array([0, 1, 0]),
                query_padding=jnp.array([False, False, False]),
            )
