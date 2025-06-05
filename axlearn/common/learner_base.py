# Copyright © 2024 Apple Inc.
"""Interfaces for learners and modules used inside of them."""

from __future__ import annotations

from typing import Any

from axlearn.common.base_layer import ParameterSpec
from axlearn.common.module import Module
from axlearn.common.optimizer_base import OptParam
from axlearn.common.utils import Nested


class LearnerModule(Module):
    """Any stateful module used inside a `BaseLearner`, including the learner itself.

    E.g., an `Ema` module that could be used to compute an EMA in all the places we need to compute
    an EMA in optimizers.
    """

    def create_state_partition_specs(self, model_param_specs: Nested[ParameterSpec]) -> Any:
        """Creates learner state partition_specs.

        The return type is a pytree with `TensorSpec`s as leaves.
        Must have the same tree structure returned by `init()`.
        """
        raise NotImplementedError(type(self))

    def init(self, model_params: Nested[OptParam]) -> Any:
        """Initializes learner state.

        The return type is a pytree with `Tensor`s as leaves.
        Must have the same tree structure returned by `create_state_partition_specs()`.
        """
        raise NotImplementedError(type(self))
