# Copyright © 2023 Apple Inc.

"""The optimizer API.

The API largely follows that of optax, but with a few changes to support partition and
factorization, specifically:

1. The optimizer contains an additional "partition" function, which computes the partition specs
   for optimizer states for model parallelism.

2. init and update take params with OptParam, instead of jnp.ndarray, as leaf nodes. OptParam
   contains the jnp.ndarray value along with auxiliary information about the parameter, including:
   - FactorizationSpec: which dimensions can be factorized. Note that we cannot derive factorization
     dims from only tensor shapes, since we do not want to factorize along the stacking axis for
     parameters of Repeat and Pipeline layers.
   - weight_decay_scale: control the weight decay rate.
"""
import dataclasses
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import optax
import typing_extensions

from axlearn.common.base_layer import FactorizationSpec, NestedParameterSpec
from axlearn.common.utils import Tensor, TensorSpec


@dataclasses.dataclass
class OptParam:
    """A parameter to be optimized by an optimizer."""

    value: Tensor
    factorization_spec: Optional[FactorizationSpec]
    weight_decay_scale: Optional[float]

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape


# NestedOptParam = Union[OptParam, Dict[str, "NestedOptParam"]]
NestedOptParam = Union[OptParam, Dict[str, Any]]

# Similar to optax.TransformInitFn, but with NestedOptParam as inputs so that factorization specs
# are available.
TransformInitFn = Callable[[NestedOptParam], optax.OptState]


class TransformUpdateFn(typing_extensions.Protocol):
    """Similar to optax.TransformUpdateFn, but with two differences:

    (1) params is required;
    (2) params is of type NestedOptParam and therefore contains factorization spec.
    """

    def __call__(
        self, updates: optax.Updates, state: optax.OptState, params: NestedOptParam
    ) -> Tuple[optax.Updates, optax.OptState]:
        ...


# Specification of an optimizer state array.
OptStateSpec = TensorSpec
NestedOptStateSpec = Union[OptStateSpec, Dict, Sequence]
TransformPartitionSpecFn = Callable[[NestedParameterSpec], NestedOptStateSpec]


class PartitionedGradientTransformation(NamedTuple):
    """An optax-style optimizer with a function to partition the inputs across devices.

    For new optimizers, using `UpdateTransformation` is preferred instead because it supports
    more types of optimizers and allows better reuse of functionality across different optimizers.

    Despite this, there are no plans to stop supporting this class.
    """

    init: TransformInitFn
    update: TransformUpdateFn
    partition: TransformPartitionSpecFn
