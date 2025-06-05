# Copyright © 2023 Apple Inc.

"""Base model definition."""

from axlearn.common.base_layer import BaseLayer
from axlearn.common.module import NestedTensor, Tensor


class BaseModel(BaseLayer):
    """The base class of a model.

    Some subclasses also implement a `predict` method:

    def predict(self, input_batch: NestedTensor, **kwargs) -> NestedTensor:
        Computes predictions with the given inputs.

        Args:
            input_batch: a NestedTensor representing an input batch, containing Tensors with a
                leading dimension of `batch_size`.

        Returns:
            A NestedTensor containing Tensors with a leading dimension of `batch_size`.
    """

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        """Computes loss and auxiliary outputs with the given inputs.

        Args:
            input_batch: a NestedTensor representing an input batch.

        Returns:
            (loss, aux), where `loss` is a scalar Tensor representing the model loss and `aux`
            is a NestedTensor containing model-specific auxiliary outputs.
        """
        raise NotImplementedError(type(self))
