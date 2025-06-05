# Copyright © 2023 Apple Inc.

"""Metrics."""

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from absl import logging

from axlearn.common.config import Configurable
from axlearn.common.summary import Summary
from axlearn.common.utils import NestedTensor, Tensor


class WeightedScalarValue(Summary):
    """A weighted scalar value represents a mean value and a weight."""

    mean: Tensor
    weight: Tensor

    def value(self) -> Optional[Union[NestedTensor, int, float]]:
        return self.mean


class WeightedScalar(WeightedScalarValue):
    """A weighted scalar represents a weighted Summable value.

    Weight should be a scalar and is assumed to be non-negative.
    A weight of zero corresponds to zero mean.
    """

    def __add__(self, other: "WeightedScalar") -> "WeightedScalar":
        # TODO(markblee): Handle possible overflows.
        weight = self.weight + other.weight
        # Use the "double-where" trick to avoid division by 0.
        # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
        # The only case where weight<=0 is if both weights are 0, since they are non-negative.
        mean = jnp.where(
            weight > 0,
            (self.mean * self.weight + other.mean * other.weight)
            / jnp.where(weight > 0, weight, 1),
            0.0,
        )
        return WeightedScalar(mean, weight)

    def accumulate(self, other: Summary) -> Summary:
        if not isinstance(other, WeightedScalar):
            raise TypeError(f"Expected WeightedScalar, got {type(other)}.")
        return self + other


class MetricAccumulator(Configurable):
    """A MetricAccumulator is used during evaluation to accumulate metrics across batches."""

    def __init__(self, cfg: Configurable.Config):
        super().__init__(cfg)
        self._summaries: dict[str, Any] = {}

    def update(self, model_outputs: dict[str, Any]):
        logging.debug(
            "MetricAccumulator.update: current=%s update=%s",
            self._summaries,
            model_outputs,
        )
        model_outputs = self._tree_map(
            lambda x: x if isinstance(x, Summary) else None, model_outputs
        )

        if not self._summaries:
            # Save all summaries from first nonempty batch.
            self._summaries = model_outputs
        else:
            # Accumulate summaries from batches after the first.
            self._summaries = self._tree_map(
                lambda x, y: x.accumulate(y), self._summaries, model_outputs
            )
        logging.debug("MetricAccumulator.update: merged=%s", self._summaries)

    def summaries(self) -> dict[str, Any]:
        return self._summaries

    @staticmethod
    def _tree_map(*args, **kwargs):
        is_leaf = lambda x: isinstance(x, Summary)
        return jax.tree.map(*args, **kwargs, is_leaf=is_leaf)


def _metric_accumulator_flatten(v: MetricAccumulator) -> tuple[tuple, tuple]:
    """Specifies a flattening recipe for `MetricAccumulator`."""
    summaries = v.summaries()
    sorted_items = sorted(summaries.items(), key=lambda x: x[0])
    if not sorted_items:
        return ((), ())
    summaries_keys, summaries_values = zip(*sorted_items)
    return (summaries_values, summaries_keys)


def _metric_accumulator_unflatten(
    summaries_keys: tuple, summaries_values: tuple
) -> MetricAccumulator:
    """Specifies an unflattening recipe for `MetricAccumulator`."""
    accumulator = MetricAccumulator.default_config().instantiate()
    summaries = dict(zip(summaries_keys, summaries_values))
    accumulator.update(summaries)
    return accumulator


jax.tree_util.register_pytree_node(
    MetricAccumulator,
    _metric_accumulator_flatten,
    _metric_accumulator_unflatten,
)
