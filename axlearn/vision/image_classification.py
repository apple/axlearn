# Copyright © 2023 Apple Inc.

"""Image classification models."""

from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import ClassificationMetric, Dropout, Linear
from axlearn.common.module import Module, Tensor
from axlearn.common.utils import NestedTensor


class ImageClassificationModel(BaseModel):
    """An image classification model."""

    @config_class
    class Config(BaseLayer.Config):
        backbone: Required[InstantiableConfig] = REQUIRED
        num_classes: Required[int] = REQUIRED
        classifier: InstantiableConfig = Linear.default_config().set(
            bias=True,
            param_partition_spec=("model", None),
        )
        dropout: Dropout.Config = Dropout.default_config()
        metric: InstantiableConfig = ClassificationMetric.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("backbone", cfg.backbone)
        self._add_child("dropout", cfg.dropout)
        self._add_child(
            "classifier",
            cfg.classifier.clone(
                input_dim=self.backbone.endpoints_dims["embedding"],
                output_dim=cfg.num_classes,
            ),
        )
        self._add_child("metric", cfg.metric.clone(num_classes=cfg.num_classes))

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32
        return cfg

    def predict(self, input_batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute prediction on an input batch.

        Args:
            input_batch: A dict containing key "image" with value of shape (batch, height, width,
                channels).

        Returns:
            A Dict of {str: Tensor} containing intermediate features and classification logits.
        """
        x = input_batch["image"]
        # The backbone is assumed to output the embedding for classification.
        endpoints = self.backbone(x)
        x = self.dropout(endpoints["embedding"])
        x = self.classifier(x)
        endpoints["logits"] = x
        return endpoints

    def forward(self, input_batch: dict[str, Tensor]) -> tuple[Tensor, NestedTensor]:
        # [batch].
        outputs = self.predict(input_batch)
        # Compute metrics.
        loss = self.metric(
            outputs["logits"],
            labels=input_batch.get("label"),
            soft_labels=input_batch.get("soft_label"),
        )
        return loss, outputs
