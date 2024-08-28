# Copyright © 2023 Apple Inc.

"""HuggingFace sequence classification wrappers."""
from typing import Any, Optional

import jax.numpy as jnp
import optax
from flax.training.common_utils import onehot
from transformers.models.albert.modeling_flax_albert import FlaxAlbertForSequenceClassification
from transformers.models.bert.modeling_flax_bert import FlaxBertForSequenceClassification
from transformers.models.roberta.modeling_flax_roberta import (
    FlaxRobertaForSequenceClassification,
    create_position_ids_from_input_ids,
)

from axlearn.common.config import config_class
from axlearn.common.module import NestedTensor
from axlearn.common.utils import Tensor
from axlearn.huggingface.hf_module import HfModuleWrapper


class HfSequenceClassificationWrapper(HfModuleWrapper):
    """A wrapper for an HF sequence classification model so that it can be used with AXLearn.

    It should not be directly used. Instead, use its subclasses.

    Remember to include "classifier" in pretrained_keys_to_skip if num_label in hf_config is
    different from that in pre-trained model.
    """

    @config_class
    class Config(HfModuleWrapper.Config):
        """Configures HfSequenceClassificationWrapper."""

        # Indicate the type of loss to use.
        # Currently, there are two options:
        #     - "softmax" for multi-class classification.
        #     - "bce" (binary cross entropy) for multi-label classification.
        loss: Optional[str] = "softmax"

    def forward(
        self,
        input_batch: NestedTensor,
        **kwargs,
    ) -> tuple[Tensor, NestedTensor]:
        """Runs prediction with targets to compute the loss.
        Currently, cross-entropy loss is used.

        Args:
            input_batch: a dict with the following entries:
                input_ids: an int Tensor of shape [batch_size, seq_len]
                    representing indices of input sequence tokens in the vocabulary.
                    Values should be in the range [0, vocab_size].
                token_type_ids: an optional int Tensor of shape [batch_size, seq_len]
                    indicating the first and second portions of the input.
                    Values should be either 0 (first sentence token) or 1 (second sentence token).
                target_labels: an int Tensor of shape [batch_size]
                    indicating the ground-truth labels to predict.
                    Non-padding values should be in the range [0, num_labels-1].
                    In the case of multi-label, an int Tensor of shape [batch_size, num_labels]
                    Non-padding values should be 0 or 1.
                    If any value for the example is NEGATIVE,
                    it means the example is a padding example.
            kwargs: optional auxiliary keyword args, not used for this model.

        Returns:
            (loss, aux_outputs), where loss is a scalar Tensor
            and aux_outputs is a FlaxSequenceClassifierOutput containing:
                logits: a Tensor of shape [batch_size, num_labels].

        Raises:
            NotImplementedError: If kwargs is provided.
        """
        if len(kwargs) != 0:
            raise NotImplementedError(f"Don't know how to configure kwargs for {type(self)}")

        # TODO(@zhucheng_tu): Use constants for field names like "input_ids" and "target_labels".
        input_ids: Tensor = input_batch["input_ids"]
        target_labels: Tensor = input_batch["target_labels"]

        if len(input_batch["target_labels"].shape) > 1:
            is_valid_input: Tensor = jnp.all(
                jnp.greater_equal(input_batch["target_labels"], 0), axis=1, keepdims=True
            )
        else:
            is_valid_input: Tensor = jnp.greater_equal(input_batch["target_labels"], 0)
        num_inputs = is_valid_input.sum()

        hf_output = self.predict(
            input_batch=dict(
                input_ids=input_ids,
                token_type_ids=input_batch.get("token_type_ids"),
            )
        )

        if self.config.loss == "softmax":
            loss = (
                optax.softmax_cross_entropy(
                    hf_output["logits"],
                    onehot(target_labels, num_classes=self._hf_config.num_labels),
                )
                * is_valid_input
            ).sum() / jnp.maximum(1, num_inputs)
        elif self.config.loss == "bce":
            loss = (
                optax.sigmoid_binary_cross_entropy(hf_output["logits"], target_labels)
                * is_valid_input
            ).sum() / (jnp.maximum(1, num_inputs) * target_labels.shape[-1])
        else:
            raise NotImplementedError(f"Invalid loss type: {self.config.loss}.")

        return loss, hf_output


class HfBertForSequenceClassificationWrapper(HfSequenceClassificationWrapper):
    """A wrapper for an HF BERT for sequence classification module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config().set(hf_model_type=FlaxBertForSequenceClassification)
        return cfg

    def _dummy_input_kwargs(self) -> dict[str, Optional[Tensor]]:
        """Returns a dictionary of kwargs to pass to linen.Module.init."""
        return {**super()._dummy_input_kwargs(), "head_mask": None}

    def _forward_kwargs(self, input_batch: dict[str, Tensor]) -> dict[str, Any]:
        """Returns a dictionary of kwargs for HF module's forward __call__."""
        return {**super()._forward_kwargs(input_batch), "head_mask": None}


class HfAlbertForSequenceClassificationWrapper(HfSequenceClassificationWrapper):
    """A wrapper for an HF ALBERT for sequence classification module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config().set(hf_model_type=FlaxAlbertForSequenceClassification)
        return cfg


class HfRobertaForSequenceClassificationWrapper(HfSequenceClassificationWrapper):
    """A wrapper for an HF RoBERTa for sequence classification module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config().set(hf_model_type=FlaxRobertaForSequenceClassification)
        return cfg

    def _dummy_input_kwargs(self) -> dict[str, Optional[Tensor]]:
        """Returns a dictionary of kwargs to pass to linen.Module.init."""
        return {**super()._dummy_input_kwargs(), "head_mask": None}

    def _forward_kwargs(self, input_batch: dict[str, Tensor]) -> dict[str, Any]:
        """Returns a dictionary of kwargs for HF module's forward __call__."""
        kwargs = super()._forward_kwargs(input_batch)
        input_ids: Tensor = input_batch["input_ids"]
        kwargs.update(
            position_ids=create_position_ids_from_input_ids(
                input_ids, self._hf_config.pad_token_id
            ),
            head_mask=None,
        )
        return kwargs
