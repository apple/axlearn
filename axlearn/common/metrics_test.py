# Copyright © 2024 Apple Inc.

"""Tests for metrics.py."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

from axlearn.common import metrics, summary, test_utils, utils
from axlearn.common.module import Summable


class TestMetricAccumulator(test_utils.TestCase):
    """Tests metrics."""

    # pylint: disable-next=no-self-use
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

        summaries_copy = jax.tree.map(lambda x: x, summaries)
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

    def test_flatten_unflatten_metric_accumulator(self):
        acc = metrics.MetricAccumulator.default_config().instantiate()
        summaries = [
            dict(
                image=summary.ImageSummary(jnp.ones((3, 4, 5))),
                loss=metrics.WeightedScalar(2, 5),
            ),
            dict(
                image=summary.ImageSummary(10 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedScalar(5, 10),
            ),
            dict(
                image=summary.ImageSummary(7 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedScalar(100, 0),
            ),
        ]
        summaries_copy = jax.tree.map(lambda x: x, summaries)
        for s in summaries_copy:
            acc.update(s)

        flat, tree = jax.tree_util.tree_flatten(acc)
        unflattened = jax.tree_util.tree_unflatten(tree, flat)
        expected = jax.tree_util.tree_leaves(acc.summaries())
        result = jax.tree_util.tree_leaves(unflattened.summaries())
        chex.assert_trees_all_close(result, expected)

    @parameterized.parameters(
        # Test a case with total weight=0.
        dict(weight=[0.0, 0.0], expected=metrics.WeightedScalar(0.0, 0.0)),
        # Test a case with total weight<0.
        dict(weight=[0.0, -1.0], expected=metrics.WeightedScalar(0.0, -1.0)),
        # Test cases with total weight>0.
        dict(weight=[0.0, 1.0], expected=metrics.WeightedScalar(1.0, 1.0)),
        dict(weight=[1, 0.1], expected=metrics.WeightedScalar(1.0, 1.1)),
        # Test cases with jax arrays.
        dict(
            weight=[jnp.array(1), jnp.array(0.1)],
            expected=metrics.WeightedScalar(jnp.array(1.0), jnp.array(1.1)),
        ),
        dict(
            weight=[jnp.array(1), jnp.array(0.1)],
            expected=metrics.WeightedScalar(
                jnp.array(1.0, dtype=jnp.bfloat16), jnp.array(1.1, dtype=jnp.bfloat16)
            ),
            dtype=jnp.bfloat16,
        ),
        # Test cases with integer weights.
        dict(weight=[1, 1], expected=metrics.WeightedScalar(1.0, 2)),
        dict(weight=[0, 0], expected=metrics.WeightedScalar(0.0, 0)),
    )
    def test_weighted_scalar(
        self,
        weight: list[float],
        expected: metrics.WeightedScalar,
        dtype: Optional[jnp.dtype] = None,
    ):
        if dtype is not None:
            mean = utils.cast_floats([jnp.array(1.0), jnp.array(1.0)], dtype)
            weight = utils.cast_floats(weight, dtype)
        else:
            mean = [1.0, 1.0]

        def add(weight):
            a = metrics.WeightedScalar(mean=mean[0], weight=weight[0])
            b = metrics.WeightedScalar(mean=mean[1], weight=weight[1])
            return a + b

        # Test with and without jit.
        self.assertNestedAllClose(expected, add(weight))
        self.assertNestedAllClose(expected, jax.jit(add)(weight))

        # Test isinstance check.
        self.assertIsInstance(metrics.WeightedScalar(1.0, 1.0), Summable)
