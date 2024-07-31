# Copyright © 2023 Apple Inc.

"""Metrics."""
import typing
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from absl import logging

from axlearn.common.config import Configurable
from axlearn.common.module import Summable
from axlearn.common.summary import Summary
from axlearn.common.utils import NestedTensor, Tensor


class WeightedScalarValue(Summary):
    """A weighted scalar value represents a mean value and a weight."""

    mean: Tensor
    weight: Tensor

    def value(self) -> Optional[Union[NestedTensor, int, float]]:
        return self.mean


@typing.runtime_checkable  # Needed for isinstance checks to work.
class WeightedScalar(WeightedScalarValue, Summable):
    """A weighted scalar represents a weighted Summable value."""

    def __add__(self, other: "WeightedScalar") -> "WeightedScalar":
        weight = self.weight + other.weight
        mean = jnp.where(
            weight > 0, (self.mean * self.weight + other.mean * other.weight) / weight, 0.0
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
        self._summaries: Dict[str, Any] = {}

    def update(self, model_outputs: Dict[str, Any]):
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

    def summaries(self) -> Dict[str, Any]:
        return self._summaries

    @staticmethod
    def _tree_map(*args, **kwargs):
        is_leaf = lambda x: isinstance(x, Summary)
        return jax.tree_util.tree_map(*args, **kwargs, is_leaf=is_leaf)


def _metric_accumulator_flatten(v: MetricAccumulator) -> Tuple[Tuple, Tuple]:
    """Specifies a flattening recipe for `MetricAccumulator`."""
    summaries = v.summaries()
    sorted_items = sorted(summaries.items(), key=lambda x: x[0])
    if not sorted_items:
        return ((), ())
    summaries_keys, summaries_values = zip(*sorted_items)
    return (summaries_values, summaries_keys)


def _metric_accumulator_unflatten(
    summaries_keys: Tuple, summaries_values: Tuple
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
