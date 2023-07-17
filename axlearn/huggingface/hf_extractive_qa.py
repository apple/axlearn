# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# huggingface/transformers:
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License").

"""AXLearn Wrapper for Hugging face Pretrained Models for Extractive Question Answering."""

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import optax
from jax.nn import one_hot
from transformers.models.bert.modeling_flax_bert import FlaxBertForQuestionAnswering
from transformers.models.roberta.modeling_flax_roberta import (
    FlaxRobertaForQuestionAnswering,
    create_position_ids_from_input_ids,
)
from transformers.models.xlm_roberta.modeling_flax_xlm_roberta import (
    FlaxXLMRobertaForQuestionAnswering,
)

from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import NestedTensor
from axlearn.common.utils import Tensor
from axlearn.huggingface.hf_module import HfModuleWrapper


class _HfExtractiveQuestionAnsweringWrapper(HfModuleWrapper):
    """A wrapper for an HF extractive question answering model so that it can be used with AXLearn.

    It should not be directly used. Instead, use its subclasses.

    Reference:
    https://github.com/huggingface/transformers/blob/02b176c4ce14340d26d42825523f406959c6c202/src/transformers/models/roberta/modeling_flax_roberta.py#L1309
    """

    def _forward_kwargs(self, input_batch: Dict[str, Tensor]) -> Dict[str, Any]:
        """Returns a dictionary of kwargs for HF module's forward __call__."""
        return {**super()._forward_kwargs(input_batch), "output_hidden_states": True}

    def forward(  # pylint: disable=duplicate-code
        self,
        input_batch: NestedTensor,
        **kwargs,
    ) -> Tuple[Tensor, NestedTensor]:
        """Runs question answering prediction with targets to compute the loss.
        Currently, cross-entropy loss is use for
        start position and end position.

        Args:
            input_batch: a dict with the following entries:
                input_ids: an int Tensor of shape [batch_size, seq_len]
                    representing indices of input sequence tokens in the vocabulary.
                    Values should be in the range [0, vocab_size).
                token_type_ids: an optional int Tensor of shape [batch_size, seq_len]
                    indicating the first and second portions of the input.
                    Values should be either 0 (first sentence token) or 1 (second sentence token)
                    and 0 for pad tokens, if any.
                start_positions: an int Tensor of shape [batch_size]
                    indicating the span start token position (inclusive).
                    Start positions greater than (seq_len - 1) will be clipped to (seq_len - 1).
                    Padding examples have start_position=-1 and are not taken into account
                    for computing the loss.
                end_positions: an int Tensor of shape [batch_size]
                    indicating the span end token position (inclusive).
                    End positions greater than (seq_len - 1) will be clipped to (seq_len - 1).
                    Padding examples have end_position=-1 and are not taken into account
                    for computing the loss.

                    * Note: If start_position == end_position, then answer is
                        a single token. If contains_answer == False, then
                        start_position = end_position = 0.
            kwargs: optional auxiliary keyword args.

        Returns:
            (loss, aux_outputs), where loss is a scalar Tensor
            and hf_outputs is a dictionary containing:
                start_logits: a Tensor of shape [batch_size, seq_len].
                end_logits: a Tensor of shape [batch_size, seq_len].
                hidden_state: a tuple of Tensors (one for each layer)
                    of shape [batch_size, seq_len, hidden_size].

        Raises:
            NotImplementedError: If kwargs is provided but unsupported.
        """
        if len(kwargs) != 0:
            raise NotImplementedError(f"Don't know how to configure kwargs for {type(self)}")

        input_ids: Tensor = input_batch["input_ids"]
        start_positions: Tensor = input_batch["start_positions"]
        end_positions: Tensor = input_batch["end_positions"]

        hf_output = self.predict(
            input_batch=dict(
                input_ids=input_ids,
                token_type_ids=input_batch.get("token_type_ids"),
            )
        )
        start_logits = hf_output["start_logits"]
        end_logits = hf_output["end_logits"]

        loss = None
        if start_positions is not None and end_positions is not None:
            # If the start/end positions are outside of model inputs, we ignore these terms.
            seq_len = start_logits.shape[1]
            start_positions = jnp.clip(start_positions, a_max=seq_len - 1)
            end_positions = jnp.clip(end_positions, a_max=seq_len - 1)

            is_valid_input = jnp.logical_and(start_positions >= 0, end_positions >= 0)
            num_inputs = is_valid_input.sum()

            start_loss = (
                optax.softmax_cross_entropy(
                    start_logits, one_hot(start_positions, num_classes=seq_len)
                )
                * is_valid_input
            ).sum() / jnp.maximum(1, num_inputs)
            end_loss = (
                optax.softmax_cross_entropy(end_logits, one_hot(end_positions, num_classes=seq_len))
                * is_valid_input
            ).sum() / jnp.maximum(1, num_inputs)
            loss = (start_loss + end_loss) / 2
            self.add_summary("start_loss", WeightedScalar(start_loss, num_inputs))
            self.add_summary("end_loss", WeightedScalar(end_loss, num_inputs))

        return loss, hf_output


class HfBertForExtractiveQuestionAnsweringWrapper(_HfExtractiveQuestionAnsweringWrapper):
    """A wrapper for an HF BERT for extractive question answering module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config().set(hf_model_type=FlaxBertForQuestionAnswering)
        return cfg  # pylint: disable=duplicate-code

    def _dummy_input_kwargs(self) -> Dict[str, Optional[Tensor]]:
        """Returns a dictionary of kwargs to pass to linen.Module.init."""
        return {**super()._dummy_input_kwargs(), "head_mask": None}

    def _forward_kwargs(self, input_batch: Dict[str, Tensor]) -> Dict[str, Any]:
        """Returns a dictionary of kwargs for HF module's forward __call__."""
        return {**super()._forward_kwargs(input_batch), "head_mask": None}


class HfRobertaForExtractiveQuestionAnsweringWrapper(_HfExtractiveQuestionAnsweringWrapper):
    """A wrapper for an HF RoBERTa for extractive question answering module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config().set(hf_model_type=FlaxRobertaForQuestionAnswering)
        return cfg

    # pylint: disable=duplicate-code
    def _dummy_input_kwargs(self) -> Dict[str, Optional[Tensor]]:
        """Returns a dictionary of kwargs to pass to linen.Module.init."""
        return {**super()._dummy_input_kwargs(), "head_mask": None}

    def _forward_kwargs(self, input_batch: Dict[str, Tensor]) -> Dict[str, Any]:
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


class HfXLMRobertaForExtractiveQuestionAnsweringWrapper(  # pylint: disable=too-many-ancestors
    HfRobertaForExtractiveQuestionAnsweringWrapper
):
    """A wrapper for an HF XLM-RoBERTa for extractive question answering module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config().set(hf_model_type=FlaxXLMRobertaForQuestionAnswering)
        return cfg
