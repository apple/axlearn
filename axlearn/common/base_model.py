# Copyright © 2023 Apple Inc.

"""Base model definition."""

from typing import Optional

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.metrics import BaseLossMetrics
from axlearn.common.module import Nested, NestedTensor, Tensor


class BaseModel(BaseLayer):
    """The base class of a model."""

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        """Computes loss and auxiliary outputs with the given inputs.

        Args:
            input_batch: a NestedTensor representing an input batch.

        Returns:
            (loss, aux), where `loss` is a scalar Tensor representing the model loss and `aux`
            is a NestedTensor containing model-specific auxiliary outputs.
        """
        raise NotImplementedError(type(self))


class PredictModel(BaseModel):
    """A model that implements a predict method."""

    @config_class
    class Config(BaseModel.Config):
        """Configures PredictModel."""

        metrics: Optional[BaseLossMetrics.Config] = None

    def predict(self, input_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Computes predictions with the given inputs.

        Args:
            input_batch: A nested Tensor representing an input batch, containing Tensors with a
                leading dimension of `batch_size`.

        Returns:
            A nested Tensor containing Tensors with a leading dimension of `batch_size`.
        """
        raise NotImplementedError(type(self))
