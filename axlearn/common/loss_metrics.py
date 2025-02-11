# Copyright © 2025 Apple Inc.

"""Layers for computing training time metrics."""

from axlearn.common.base_layer import BaseLayer
from axlearn.common.utils import Nested, Tensor


class BaseLossMetrics(BaseLayer):
    """A module for computing training time metrics.

    See `causal_lm.Model` for an example usage.
    """

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        predict_outputs: Nested[Tensor],
        module_outputs: Nested[Tensor],
    ) -> tuple[Tensor, Nested[Tensor]]:
        """Computes metrics from inputs and predictions.

        Args:
            input_batch: A mapping from input keys to Tensors.
            predict_outputs: Model predictions for computing metrics.
            module_outputs: Outputs from the model's invocation context.

        Returns:
            A tuple (loss, metrics).
        """
        raise NotImplementedError(type(self))
