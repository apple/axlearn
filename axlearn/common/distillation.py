# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A collection of distillation methods."""
from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import InstantiableConfig
from axlearn.common.layers import Linear
from axlearn.common.loss import kl_divergence, negative_cosine_similarity_loss
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import (
    REQUIRED,
    Module,
    NestedTensor,
    Required,
    Tensor,
    child_context,
    config_class,
)


class BaseDistillationModel(BaseModel):
    """A base distillation model."""

    @config_class
    class Config(BaseLayer.Config):
        teacher: Required[BaseLayer.Config] = REQUIRED
        student: Required[BaseLayer.Config] = REQUIRED

    def forward(self, input_batch: dict[str, Tensor]) -> tuple[Tensor, NestedTensor]:
        raise NotImplementedError("Not implemented forward method for BaseDistillationModel")


class NegativeCosineSimilarityMetric(BaseLayer):
    """Computes the negative cosine similarity distillation loss."""

    @config_class
    class Config(BaseLayer.Config):
        # Name of the embedding layer for student.
        student_embedding: str = "embedding"
        # Name of the embedding layer for teacher. If not set, will use student_embedding.
        teacher_embedding: Optional[str] = None
        normalize_embedding: bool = True

    def forward(
        self,
        *,
        predictions: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> Tensor:
        """Computes distillation metrics, e.g. loss, accuracy.

        Args:
            predictions: a float Tensor of shape [..., dim] representing the student embeddings.
            targets: a float Tensor of shape [..., dim] representing the teacher embeddings.
            kwargs: additional keyword arguments.

        Returns:
            A float Tensor represents the final negative cosine similarity loss.
        """
        del kwargs
        cfg = self.config
        # Compute negative cosine similarity between student and teacher embeddings.
        loss, _ = negative_cosine_similarity_loss(
            predictions=predictions[cfg.student_embedding],
            targets=targets[cfg.teacher_embedding or cfg.student_embedding],
            normalize_embedding=cfg.normalize_embedding,
        )
        return loss


class KLDivergenceMetric(BaseLayer):
    """Computes the KL-divergence distillation loss and accuracy."""

    @config_class
    class Config(BaseLayer.Config):
        is_log_targets: bool = False
        temperature: float = 1.0

    def forward(
        self,
        *,
        predictions: Tensor,
        targets: Tensor,
        label: Tensor,
        **kwargs,
    ) -> Tensor:
        """Computes KL-divergence metrics, e.g. loss, accuracy.

        Args:
            predictions: a float Tensor of shape [..., num_classes] representing the
                prediction probabilities from the student model.
            targets: a float Tensor of shape [..., num_classes] representing the
                prediction probabilities from the teacher model.
            label: an int Tensor of shape [...].
                Targets should contain the ground truth token ids in the range [0, num_classes).
                Out-of-class targets are ignored in the loss calculation.
            kwargs: additional keyword arguments.

        Returns:
            A float Tensor represents the final KL-divergence loss.

        Raises:
           ValueError: If temperature is not positive.
        """
        del kwargs
        cfg = self.config
        temp = cfg.temperature
        if temp <= 0:
            raise ValueError(f"Distillation temperature has to be positive, got {temp}.")

        targets = jax.nn.softmax(targets["logits"] / temp)
        predictions = jax.nn.log_softmax(predictions["logits"] / temp)

        # Compute KL divergence between log_predictions and targets.
        loss, _ = kl_divergence(
            log_predictions=predictions,
            targets=jnp.log(targets) if cfg.is_log_targets else targets,
            is_log_targets=cfg.is_log_targets,
        )
        # Compute top-1 accuracy for student and teacher models.
        is_valid_example = label >= 0
        num_examples = is_valid_example.sum()
        denominator = jnp.maximum(1, num_examples)

        predictions = jnp.argmax(predictions, axis=-1)
        accuracy = jnp.equal(predictions, label).sum() / denominator
        self.add_summary("accuracy", WeightedScalar(accuracy, num_examples))

        predictions_t = jnp.argmax(targets, axis=-1)
        accuracy_t = jnp.equal(predictions_t, label).sum() / denominator
        self.add_summary("accuracy_teacher", WeightedScalar(accuracy_t, num_examples))

        return loss


class DistillationModel(BaseDistillationModel):
    """A standard distillation model with a frozen teacher and a student.

    Reference:
    https://arxiv.org/abs/1503.02531
    https://arxiv.org/abs/2106.05237
    """

    @config_class
    class Config(BaseDistillationModel.Config):
        metric: InstantiableConfig = KLDivergenceMetric.default_config()

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("teacher", cfg.teacher)
        self._add_child("student", cfg.student)
        self._add_child("metric", cfg.metric)

    def forward(self, input_batch: dict[str, Tensor]) -> tuple[Tensor, NestedTensor]:
        # We freeze teacher weights in optimizer.
        with child_context("teacher", module=self.teacher, is_training=False):
            teacher_outputs = self.teacher.predict(input_batch)
        student_outputs = self.student.predict(input_batch)
        loss = self.metric(
            predictions=student_outputs,
            targets=teacher_outputs,
            **input_batch,
        )
        return loss, {"student_outputs": student_outputs, "teacher_outputs": teacher_outputs}


class DeiTDistillationMetric(BaseLayer):
    """Computes the DeiT distillation loss and accuracy.

    The training loss contains two terms:
        1) A regular cross-entropy loss term computed from the student's classification logits
            and the true labels;
        2) A distillation loss term computed from the student's distillation logits and the
            pseudo-labels generated from the teacher's classification logits.
    At test time, the prediction is made from the softmax of the sum of both classification and
        distillation logits.

    Reference:
    https://github.com/facebookresearch/deit/blob/ae4dba9b453b9e18faa781edbc13039aaeca9b68/losses.py
    """

    @config_class
    class Config(BaseLayer.Config):
        num_classes: Required[int] = REQUIRED
        # Weight of the distillation loss term in the final loss. Must be in [0, 1].
        alpha: float = 0.5

    def forward(
        self,
        *,
        logits_teacher: Tensor,
        logits_classification: Tensor,
        logits_distillation: Tensor,
        labels: Tensor,
        soft_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes classification metrics, e.g. loss, accuracy.

        Args:
            logits_teacher: a float Tensor of shape [..., num_classes] representing the
                classification logits from the teacher model.
            logits_classification: a float Tensor of shape [..., num_classes] representing the
                logits from the classification token of the student model.
            logits_distillation: a float Tensor of shape [..., num_classes] representing the
                logits from the distillation token of the student model.
            labels: an int Tensor of shape [...].
                Targets should contain the ground truth token ids in the range [0, num_classes).
                Out-of-class targets are ignored in the loss calculation.
            soft_labels: a float Tensor of shape [..., num_classes] representing the soft labels
                generated from data augmentation. If not None, it is already in one-hot form and
                has been smoothed.

        Returns:
            A float Tensor represents the final loss. The final loss is a weighted sum of a
                classification cross-entropy loss and a distillation loss:
                loss = (1 - alpha) * loss_classification + alpha * loss_distillation.
        """
        cfg = self.config
        is_valid_example = labels >= 0
        num_examples = is_valid_example.sum()
        denominator = jnp.maximum(1, num_examples)

        # Compute the cross-entropy loss with true labels.
        if soft_labels is not None:
            labels_onehot = soft_labels
        else:
            labels_onehot = jax.nn.one_hot(
                labels, cfg.num_classes, dtype=logits_classification.dtype
            )
        per_example_loss = jnp.sum(
            -1 * labels_onehot * jax.nn.log_softmax(logits_classification), axis=-1
        )
        loss_classification = per_example_loss.sum() / denominator

        # Compute the hard-distillation loss with labels generated from the teacher's predictions.
        pseudo_labels = jnp.argmax(logits_teacher, axis=1)
        pseudo_labels_onehot = jax.nn.one_hot(
            pseudo_labels, cfg.num_classes, dtype=logits_classification.dtype
        )
        # Apply the mask of the classification loss to the distillation loss.
        per_example_loss = (
            jnp.sum(-1 * pseudo_labels_onehot * jax.nn.log_softmax(logits_distillation), axis=-1)
            * is_valid_example
        )
        loss_distillation = per_example_loss.sum() / denominator

        # The final loss is a weighted sum of the two loss terms.
        loss = loss_classification * (1 - cfg.alpha) + loss_distillation * cfg.alpha

        # The predictions are made from the sum of the classification logits and the distillation
        # logits.
        predictions = jnp.argmax(logits_classification + logits_distillation, axis=-1)
        accuracy = jnp.equal(predictions, labels).sum() / denominator
        self.add_summary("loss", WeightedScalar(loss, num_examples))
        self.add_summary("accuracy", WeightedScalar(accuracy, num_examples))

        # Log accuracy for the teacher model.
        predictions_t = jnp.argmax(logits_teacher, axis=-1)
        accuracy_t = jnp.equal(predictions_t, labels).sum() / denominator
        self.add_summary("accuracy_teacher", WeightedScalar(accuracy_t, num_examples))
        return loss


class DeiTDistillationModel(BaseDistillationModel):
    """A DeiT distillation model (https://arxiv.org/pdf/2012.12877.pdf)."""

    @config_class
    class Config(BaseDistillationModel.Config):
        distillation_classifier: InstantiableConfig = Linear.default_config().set(
            bias=True, param_partition_spec=("model", None)
        )
        metric: InstantiableConfig = DeiTDistillationMetric.default_config()
        num_classes: int = 1000

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("teacher", cfg.teacher.clone(num_classes=cfg.num_classes))
        self._add_child("student", cfg.student.clone(num_classes=cfg.num_classes))
        self._add_child(
            "distillation_classifier",
            cfg.distillation_classifier.clone(
                input_dim=self.student.backbone.endpoints_dims["embedding"],
                output_dim=cfg.num_classes,
            ),
        )
        self._add_child("metric", cfg.metric.clone(num_classes=cfg.num_classes))

    def forward(self, input_batch: dict[str, Tensor]) -> tuple[Tensor, NestedTensor]:
        with child_context("teacher", module=self.teacher, is_training=False):
            predictions_t = self.teacher.predict(input_batch)

        predictions_s = self.student.predict(input_batch)

        x = jnp.squeeze(predictions_s["distillation_features"], axis=1)
        logits_kd = self.distillation_classifier(x)

        # Compute metrics.
        loss = self.metric(
            logits_teacher=predictions_t["logits"],
            logits_classification=predictions_s["logits"],
            logits_distillation=logits_kd,
            labels=input_batch.get("label"),
            soft_labels=input_batch.get("soft_label"),
        )
        return loss, dict(
            logits=predictions_s["logits"] + logits_kd,
            logits_student=predictions_s["logits"],
            logits_teacher=predictions_t["logits"],
        )
