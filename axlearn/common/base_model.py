# Copyright © 2023 Apple Inc.

"""Base model definition for trainable end-to-end models in AXLearn.

This module provides BaseModel, which defines the interface for complete models that
compute training objectives (losses). This is distinct from BaseLayer, which is used
for individual neural network components.

Key Distinction - When to use BaseModel vs BaseLayer:

Use BaseModel when creating:
    - Complete, trainable models that compute a loss
    - End-to-end architectures that define training objectives
    - Models that will be directly passed to a Trainer
    Examples: LanguageModel, ImageClassifier, ObjectDetector, DiffusionModel

Use BaseLayer when creating:
    - Individual neural network components or layers
    - Building blocks that transform tensors
    - Reusable modules that will be composed into models
    Examples: Linear, Attention, TransformerBlock, Conv2D, LayerNorm, FFN

The critical difference is in the forward() method:
    - BaseLayer subclasses: forward() returns transformed tensors (flexible signature)
    - BaseModel subclasses: forward() MUST return (loss, aux_outputs) for training

Example:
    # A transformer block is a BaseLayer - it transforms tensors
    class TransformerBlock(BaseLayer):
        def forward(self, x: Tensor) -> Tensor:
            x = self.attention(x)
            return self.feed_forward(x)  # Returns tensor

    # A language model is a BaseModel - it computes loss for training
    class LanguageModel(BaseModel):
        def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
            logits = self.transformer(input_batch["input_ids"])
            loss = cross_entropy(logits, input_batch["labels"])
            return loss, {"logits": logits}  # Returns (loss, aux)
"""

from axlearn.common.base_layer import BaseLayer
from axlearn.common.module import NestedTensor, Tensor


class BaseModel(BaseLayer):
    """Base class for trainable models that compute losses for optimization.

    BaseModel defines the interface between models and training loops. While BaseLayer
    provides infrastructure for neural network components, BaseModel specifically defines
    the contract for complete models that can be trained.

    When to extend BaseModel:
        Your class should extend BaseModel if it:
        1. Represents a complete, end-to-end trainable model
        2. Computes a loss that will be optimized during training
        3. Will be passed directly to a Trainer

    When NOT to extend BaseModel (use BaseLayer instead):
        - Individual layers or components (Linear, Attention, Conv2D)
        - Composite building blocks (TransformerBlock, ResNetBlock)
        - Any module that transforms tensors but doesn't compute loss

    Contract:
        Subclasses MUST implement forward() with the exact signature:
            forward(input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]

        Where:
        - input_batch: Dict-like structure with model inputs (e.g., "input_ids", "labels")
        - Returns: (loss, aux_outputs)
            - loss: Scalar tensor to be minimized during training
            - aux_outputs: Dict of additional outputs (predictions, metrics, intermediates)

    Optional interface:
        Subclasses may also implement predict() for inference without computing loss:
            predict(input_batch: NestedTensor, **kwargs) -> NestedTensor

        This is useful for evaluation, serving, or generation tasks where you only
        need model outputs without the training loss.

    Example:
        ```python
        class ImageClassifier(BaseModel):
            def __init__(self, cfg: Config, *, parent: Optional[Module]):
                super().__init__(cfg, parent=parent)
                # Compose layers to build the model
                self._add_child("backbone", ResNet.default_config())
                self._add_child("head", Linear.default_config())

            def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
                # Extract inputs
                images = input_batch["images"]
                labels = input_batch["labels"]

                # Forward pass through layers
                features = self.backbone(images)
                logits = self.head(features)

                # Compute loss for training
                loss = cross_entropy(logits, labels)
                accuracy = (logits.argmax(-1) == labels).mean()

                # Return loss and auxiliary outputs
                return loss, {"logits": logits, "accuracy": accuracy}

            def predict(self, input_batch: NestedTensor) -> NestedTensor:
                # For inference, compute outputs without loss
                images = input_batch["images"]
                features = self.backbone(images)
                logits = self.head(features)
                return {"predictions": logits.argmax(-1), "logits": logits}
        ```

    The distinction between BaseModel and BaseLayer is fundamental to AXLearn's architecture:
    models define WHAT to optimize (the loss), while layers define HOW to transform data."""

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        """Computes loss and auxiliary outputs with the given inputs.

        Args:
            input_batch: a NestedTensor representing an input batch.

        Returns:
            (loss, aux), where `loss` is a scalar Tensor representing the model loss and `aux`
            is a NestedTensor containing model-specific auxiliary outputs.
        """
        raise NotImplementedError(type(self))
