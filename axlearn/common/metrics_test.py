# Copyright © 2024 Apple Inc.

"""Tests for metrics.py."""

import contextlib
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common import metrics, summary, test_utils, utils
from axlearn.common.metrics import MaxSummary, MinSummary, SumSummary
from axlearn.common.module import Summable


class TestMetricAccumulator(test_utils.TestCase):
    """Tests metrics."""

    # pylint: disable-next=no-self-use
    def test_metric_accumulator(self):
        """Tests MetricAccumulator and the `accumulate()` methods of `WeightedSummary` and
        `Summary`.
        """
        acc = metrics.MetricAccumulator.default_config().instantiate()
        summaries = [
            dict(
                image=summary.ImageSummary(jnp.ones((3, 4, 5))),
                loss=metrics.WeightedSummary(2, 5),
                junk=object(),
            ),
            dict(
                image=summary.ImageSummary(10 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedSummary(5, 10),
                junk=object(),
            ),
            dict(
                image=summary.ImageSummary(7 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedSummary(100, 0),
                junk=object(),
            ),
        ]

        summaries_copy = jax.tree.map(lambda x: x, summaries)
        for s in summaries_copy:
            acc.update(s)
        result = acc.summaries()
        expected = dict(
            image=summary.ImageSummary(jnp.ones((3, 4, 5))),
            loss=metrics.WeightedSummary(4, 15),
            junk=None,
        )

        chex.assert_trees_all_equal_structs(result, expected)
        result = jax.tree_util.tree_leaves(result)
        expected = jax.tree_util.tree_leaves(expected)
        chex.assert_trees_all_close(result, expected)

    @parameterized.parameters(
        dict(cls=metrics.MinSummary, expected=-10),
        dict(cls=metrics.MaxSummary, expected=10),
    )
    def test_metric_min_max_accumulator(self, cls, expected):
        acc = metrics.MetricAccumulator.default_config().instantiate()
        summaries = [
            dict(foo=cls(jnp.array(5))),
            dict(foo=cls(jnp.array(-10))),
            dict(foo=cls(jnp.array(10))),
        ]

        summaries_copy = jax.tree.map(lambda x: x, summaries)
        for s in summaries_copy:
            acc.update(s)
        result = acc.summaries()
        expected = dict(foo=cls(jnp.array(expected)))

        chex.assert_trees_all_equal_structs(result, expected)
        result = jax.tree_util.tree_leaves(result)
        expected = jax.tree_util.tree_leaves(expected)
        self.assertEqual(result, expected)

    def test_flatten_unflatten_metric_accumulator(self):
        acc = metrics.MetricAccumulator.default_config().instantiate()
        summaries = [
            dict(
                image=summary.ImageSummary(jnp.ones((3, 4, 5))),
                loss=metrics.WeightedSummary(2, 5),
            ),
            dict(
                image=summary.ImageSummary(10 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedSummary(5, 10),
            ),
            dict(
                image=summary.ImageSummary(7 * jnp.ones((3, 4, 5))),
                loss=metrics.WeightedSummary(100, 0),
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
        dict(weight=[0.0, 0.0], expected=metrics.WeightedSummary(0.0, 0.0)),
        # Test a case with total weight<0.
        dict(weight=[0.0, -1.0], expected=metrics.WeightedSummary(0.0, -1.0)),
        # Test cases with total weight>0.
        dict(weight=[0.0, 1.0], expected=metrics.WeightedSummary(1.0, 1.0)),
        dict(weight=[1, 0.1], expected=metrics.WeightedSummary(1.0, 1.1)),
        # Test cases with jax arrays.
        dict(
            weight=[jnp.array(1), jnp.array(0.1)],
            expected=metrics.WeightedSummary(jnp.array(1.0), jnp.array(1.1)),
        ),
        dict(
            weight=[jnp.array(1), jnp.array(0.1)],
            expected=metrics.WeightedSummary(
                jnp.array(1.0, dtype=jnp.bfloat16), jnp.array(1.1, dtype=jnp.bfloat16)
            ),
            dtype=jnp.bfloat16,
        ),
        # Test cases with integer weights.
        dict(weight=[1, 1], expected=metrics.WeightedSummary(1.0, 2)),
        dict(weight=[0, 0], expected=metrics.WeightedSummary(0.0, 0)),
    )
    def test_weighted_scalar(
        self,
        weight: list[float],
        expected: metrics.WeightedSummary,
        dtype: Optional[jnp.dtype] = None,
    ):
        if dtype is not None:
            mean = utils.cast_floats([jnp.array(1.0), jnp.array(1.0)], dtype)
            weight = utils.cast_floats(weight, dtype)
        else:
            mean = [1.0, 1.0]

        def add(weight):
            a = metrics.WeightedSummary(mean=mean[0], weight=weight[0])
            b = metrics.WeightedSummary(mean=mean[1], weight=weight[1])
            return a + b

        # Test with and without jit.
        self.assertNestedAllClose(expected, add(weight))
        self.assertNestedAllClose(expected, jax.jit(add)(weight))

        # Test isinstance check.
        self.assertIsInstance(metrics.WeightedSummary(1.0, 1.0), Summable)


class MinSummaryTest(test_utils.TestCase):
    @parameterized.parameters(
        (jnp.array(1), jnp.array(10), jnp.array(1)),
        (jnp.array([1, 2]), jnp.array([10, -5]), jnp.array([1, -5])),
        (jnp.array([[1, 2], [3, 4]]), jnp.array([[10, -5], [0, 8]]), jnp.array([[1, -5], [0, 4]])),
        (1, None, ValueError("MinSummary value must be a Tensor, but got <class 'int'>.")),
    )
    def test_min_summary(self, value, other_value, expected):
        min_summary = MinSummary(value)  # pytype: disable=wrong-arg-count
        if isinstance(expected, ValueError):
            ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            min_summary.validate()
            if not isinstance(expected, ValueError):
                new_summary = min_summary.accumulate(
                    MinSummary(other_value)  # pytype: disable=wrong-arg-count
                )
                chex.assert_trees_all_close(new_summary.value(), expected)


class MaxSummaryTest(test_utils.TestCase):
    @parameterized.parameters(
        (jnp.array(1), jnp.array(-10), jnp.array(1)),
        (jnp.array([1, 2]), jnp.array([10, -5]), jnp.array([10, 2])),
        (jnp.array([[1, 2], [3, 4]]), jnp.array([[10, -5], [0, 8]]), jnp.array([[10, 2], [3, 8]])),
        (1, None, ValueError("MaxSummary value must be a Tensor, but got <class 'int'>.")),
    )
    def test_max_summary(self, value, other_value, expected):
        max_summary = MaxSummary(value)  # pytype: disable=wrong-arg-count
        if isinstance(expected, ValueError):
            ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            max_summary.validate()
            if not isinstance(expected, ValueError):
                new_summary = max_summary.accumulate(
                    MaxSummary(other_value)  # pytype: disable=wrong-arg-count
                )
                chex.assert_trees_all_close(new_summary.value(), expected)


class SumSummaryTest(test_utils.TestCase):
    @parameterized.parameters(
        (jnp.array(1), jnp.array(2), jnp.array(3)),
        (jnp.array([1, 2]), jnp.array([10, -5]), jnp.array([11, -3])),
        (
            jnp.array([[1, 2], [3, 4]]),
            jnp.array([[10, -5], [0, 8]]),
            jnp.array([[11, -3], [3, 12]]),
        ),
        (1, None, ValueError("SumSummary value must be a Tensor, but got <class 'int'>.")),
    )
    def test_sum_summary(self, value, other_value, expected):
        sum_summary = SumSummary(value)  # pytype: disable=wrong-arg-count
        if isinstance(expected, ValueError):
            ctx = self.assertRaisesRegex(ValueError, expected.args[0])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            sum_summary.validate()
            if not isinstance(expected, ValueError):
                new_summary = sum_summary.accumulate(
                    SumSummary(other_value)  # pytype: disable=wrong-arg-count
                )
                chex.assert_trees_all_close(new_summary.value(), expected)


class ShapeMismatchTest(test_utils.TestCase):
    """Tests shape mismatch errors during accumulation."""

    @parameterized.parameters(
        dict(cls=MinSummary, shape1=(2, 3), shape2=(2, 4)),
        dict(cls=MaxSummary, shape1=(2, 3), shape2=(3, 3)),
        dict(cls=SumSummary, shape1=(5,), shape2=(6,)),
    )
    def test_shape_mismatch(self, cls, shape1, shape2):
        summary1 = cls(jnp.ones(shape1))
        summary2 = cls(jnp.ones(shape2))
        with self.assertRaisesRegex(ValueError, "Shape mismatch:"):
            summary1.accumulate(summary2)


if __name__ == "__main__":
    absltest.main()
