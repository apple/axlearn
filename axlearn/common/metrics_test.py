# Copyright © 2024 Apple Inc.
"""Tests for metrics.py."""
import jax
import jax.numpy as jnp
from absl.testing import absltest

# pylint: disable=no-self-use
from axlearn.common import metrics, summary, test_utils


class TestMetricAccumulator(test_utils.TestCase):
    """Tests metrics."""

    def test_with_image_summary(self):
        """Regression test that image summaries logged in a MetricAccumulator do not have value
        replaced with an empty tuple.
        """
        acc = metrics.MetricAccumulator.default_config().instantiate()
        summaries = dict(image=summary.ImageSummary(jnp.ones((3, 4, 5))), garbage=object())
        summaries_copy = jax.tree_util.tree_map(lambda x: x, summaries)
        acc.update(summaries_copy)
        result = acc.summaries()
        expected = summaries
        expected["garbage"] = tuple()
        self.assertEqual(result, expected)
        self.assertIsInstance(result["image"].value(), jax.Array)


if __name__ == "__main__":
    absltest.main()
