# Copyright © 2024 Apple Inc.
"""Tests for metrics.py."""
import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

# pylint: disable=no-self-use
from axlearn.common import metrics, summary, test_utils


class TestMetricAccumulator(test_utils.TestCase):
    """Tests metrics."""

    def test_metric_accumulator(self):
        """Tests MetricAccumulator and the `accumulate()` methods of `WeightedScalar` and
        `Summary`.
        """
        acc = metrics.MetricAccumulator.default_config().instantiate()
        summaries = [
            dict(
                image=summary.ImageSummary(jnp.ones((3, 4, 5))),
                loss=metrics.WeightedScalar(2, 5),
                junk=object(),
            ),
            dict(
                image=summary.ImageSummary(10 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedScalar(5, 10),
                junk=object(),
            ),
            dict(
                image=summary.ImageSummary(7 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedScalar(100, 0),
                junk=object(),
            ),
        ]

        summaries_copy = jax.tree_util.tree_map(lambda x: x, summaries)
        for s in summaries_copy:
            acc.update(s)
        result = acc.summaries()
        expected = dict(
            image=summary.ImageSummary(jnp.ones((3, 4, 5))),
            loss=metrics.WeightedScalar(4, 15),
            junk=None,
        )

        chex.assert_trees_all_equal_structs(result, expected)
        result = jax.tree_util.tree_leaves(result)
        expected = jax.tree_util.tree_leaves(expected)
        chex.assert_trees_all_close(result, expected)


if __name__ == "__main__":
    absltest.main()
