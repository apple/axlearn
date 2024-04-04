# Copyright © 2023 Apple Inc.

"""Metrics."""
import typing
from typing import Any, Dict, Optional, Union

import jax
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
        if weight > 0:
            mean = (self.mean * self.weight + other.mean * other.weight) / weight
        else:
            mean = 0.0
        return WeightedScalar(mean, weight)


class MetricAccumulator(Configurable):
    """A MetricAccumulator is used during evaluation to accumulate metrics across batches."""

    def __init__(self, cfg: Configurable.Config):
        super().__init__(cfg)
        self._scalars = {}

    def update(self, model_outputs: Dict[str, Any]):
        logging.debug(
            "MetricAccumulator.update: current=%s update=%s",
            self._scalars,
            model_outputs,
        )
        scalars = self._tree_map(lambda x: x if isinstance(x, Summary) else tuple(), model_outputs)
        if not self._scalars:
            self._scalars = scalars
        else:
            self._scalars = self._tree_map(lambda x, y: x + y, self._scalars, scalars)
        logging.debug("MetricAccumulator.update: merged=%s", self._scalars)

    def summaries(self) -> Dict[str, Any]:
        return self._scalars

    @staticmethod
    def _tree_map(*args, **kwargs):
        is_leaf = lambda x: isinstance(x, Summary)
        return jax.tree_util.tree_map(*args, **kwargs, is_leaf=is_leaf)
