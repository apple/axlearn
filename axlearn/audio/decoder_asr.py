# Copyright © 2023 Apple Inc.
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""ASR decoder layers."""

from typing import Callable, Optional, Union, cast

import chex
import jax
import jax.numpy as jnp
import optax

from axlearn.audio.aligner import ctc_aligner
from axlearn.common import decoding, flax_struct
from axlearn.common.attention import TransformerAttentionLayer
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, ConfigOr, Required, config_class, maybe_instantiate
from axlearn.common.decoder import Decoder
from axlearn.common.decoding import (
    NEG_INF,
    PrefixMerger,
    StopOnSubsequence,
    add_decoding_dim,
    beam_search_decode,
    compute_merge_matrix_by_prefix_ids,
    flatten_decoding_dim,
    infer_initial_time_step,
    sample_decode,
    unflatten_decoding_dim,
)
from axlearn.common.layers import Embedding, Linear
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.loss import cross_entropy
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module, child_context
from axlearn.common.rnn import BaseRNNCell, LSTMCell
from axlearn.common.transducer import Transducer, log_probs_from_blank_and_tokens
from axlearn.common.utils import Nested, Tensor, safe_not, vectorized_tree_map


def _is_valid_ctc_seq(
    *, paddings: Tensor, target_labels: Tensor, target_paddings: Tensor
) -> Tensor:
    """Returns whether each input sequence passes validity check.

    Note that `optax.ctc_loss` returns -logeps (default to 1e5) if the
    input length is smaller than the label length plus number of
    consecutive duplications, because we need a blank label to transition
    between the same labels. When this condition is not met, it should be
    considered as an invalid sequence and the loss should be ignored.

    A validity check is passed if for an example when:
        input.length >= labels.length + num(consecutive dup label tokens)

    Args:
        paddings: A 0/1 tensor of shape [batch_size, num_frames], indicating whether
            an input frame is a padding.
        target_labels: A int32 tensor of shape [batch_size, num_frames].
        target_paddings: A 0/1 tensor of shape [batch_size, num_frames], indicating
            whether a label is a padding. Note that at the moment, `target_paddings`
            must be left-justified, i.e., it must starts with 0 and followed by 1, and
            not transition back to 0.
            TODO(yqw): support generic target_paddings.

    Returns:
        A float tensor of [batch_size, ] indicating if each (input, label) pair is valid,
        with a value of 1.0 indicating valid and 0.0 otherwise.
    """
    # [batch_size, ]
    label_lengths = jnp.sum(1.0 - target_paddings, axis=-1)
    # [batch_size, ]
    input_lengths = jnp.sum(1.0 - paddings, axis=-1)
    # [batch_size, num_frames - 1]
    dups = (1.0 - target_paddings[:, 1:]) * (target_labels[:, :-1] == target_labels[:, 1:])
    # [batch_size, ]
    num_consecutive_dups = jnp.sum(dups, axis=-1)
    # [batch_size, ]
    is_valid = (label_lengths + num_consecutive_dups) <= input_lengths
    return is_valid


class CommonPrefixMerger(PrefixMerger):
    """Merges equivalent lower-ranked beams into higher-ranked ones.

    Beams are compared after removing repeats and blanks.

    See Section 3.1 of https://dl.acm.org/doi/10.1145/1143844.1143891.
    """

    def __init__(self, blank_id: int, pad_id: int = -1, remove_repeats: bool = True):
        self._blank_id = blank_id
        self._pad_id = pad_id
        self._remove_repeats = remove_repeats

    def init_state(self, *, tokens: Tensor) -> Nested[Tensor]:
        """Initializes the prefix merger state from the initial prefix `tokens`.

        If the initial prefix is non-empty, we produce state equivalent to initializing from an
        empty prefix and invoking `update` token-by-token until the end of the initial prefix.
        """
        outputs = _map_label_sequences(
            tokens,
            remove_repeats=self._remove_repeats,
            blank_id=self._blank_id,
            pad_id=self._pad_id,
        )
        # Compute last tokens.
        last_token = jnp.take_along_axis(outputs["sequences"], outputs["lengths"] - 1, axis=-1)
        return dict(
            sequences=outputs["sequences"],
            last_token=jnp.squeeze(last_token, axis=2),
            lengths=jnp.squeeze(outputs["lengths"], axis=2),
        )

    def compute(self, state: Nested[Tensor]) -> Tensor:
        """Computes a merge matrix by comparing prefixes."""
        return compute_merge_matrix_by_prefix_ids(state["sequences"])

    def update(self, *, tokens: Tensor, state: Nested[Tensor]) -> Nested[Tensor]:
        """Updates prefix merger state given the next candidate token."""

        def _update_seq(token: Tensor, seq_state: Nested[Tensor]) -> Nested[Tensor]:
            skip = jnp.logical_or(token == seq_state["last_token"], token == self._blank_id)
            return dict(
                sequences=jnp.where(
                    skip,
                    seq_state["sequences"],
                    jax.lax.dynamic_update_index_in_dim(
                        seq_state["sequences"], token, seq_state["lengths"], axis=0
                    ),
                ),
                last_token=token,
                lengths=seq_state["lengths"] + jnp.where(skip, 0, 1),
            )

        # vmap over both `batch_size` and `num_decodes`.
        return jax.vmap(jax.vmap(_update_seq))(tokens, state)


class DecodeOutputs(flax_struct.PyTreeNode):
    """Output of decoding."""

    # Raw decode output sequences. May contain blank and/or repeated tokens.
    # An int Tensor of shape [batch_size, num_decodes, max_decode_len].
    raw_sequences: Tensor
    # Post-processed sequences, e.g. after removing blanks or repeated tokens.
    # An int Tensor of shape [batch_size, num_decodes, max_decode_len].
    sequences: Tensor
    # Paddings of the post-processed sequences.
    # A 0/1 Tensor of shape [batch_size, num_decodes, max_decode_len].
    paddings: Tensor
    # Scores corresponding to sequences above (log probabilities).
    # A float Tensor of shape [batch_size, num_decodes].
    scores: Tensor


def _compute_target_paddings(target_labels: Tensor, *, vocab_size: int) -> Tensor:
    """Infers paddings from out-of-range targets.

    Args:
        target_labels: An int Tensor of any shape; commonly of shape [batch_size, seq_len].

    Returns:
        A 0/1 Tensor of same shape as `target_labels`.
    """
    return jnp.logical_or(vocab_size <= target_labels, target_labels < 0)


class BaseASRDecoderModel(BaseModel):
    """Base ASR decoder model interface."""

    @config_class
    class Config(BaseModel.Config):
        """Configures BaseASRDecoderModel."""

        # Dimensionality of inputs from acoustic encoder.
        input_dim: Required[int] = REQUIRED
        # The vocab size.
        vocab_size: Required[int] = REQUIRED

    def predict(self, input_batch: Nested[Tensor]) -> Tensor:
        """Computes logits.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.

        Returns:
            A dict containing logits. The shape of logits depend on the decoder.
        """
        raise NotImplementedError(type(self))

    def forward(self, input_batch: Nested[Tensor]) -> tuple[Tensor, Nested[Tensor]]:
        """Computes decoder loss.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim] of encoder outputs.
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                target_labels: An int Tensor of shape [batch_size, num_labels].
                target: A dictionary with input_ids as key, and an int Tensor of shape
                    [batch_size, num_labels] as value.

            For both target_labels and target["input_ids"], values should be in the range
                [0, vocab_size). Out-of-range values are excluded from the loss calculation
                (e.g., paddings and EOS can be represented this way).

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar loss value.
                aux_outputs: A dict containing:
                    per_example_loss: A float Tensor of shape [batch_size].
                    per_example_weight: A float Tensor of shape [batch_size].
        """
        raise NotImplementedError(type(self))

    def beam_search_decode(
        self, input_batch: Nested[Tensor], num_decodes: int, **kwargs
    ) -> DecodeOutputs:
        """Beam search decoding.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
            num_decodes: Beam size.
            kwargs: Additional kwargs. See corresponding subclass for details.

        Returns:
            Beam search decode outputs. See `DecodeOutputs` for details.
        """
        raise NotImplementedError(type(self))

    def _input_stats_summaries(
        self, input_batch: Nested[Tensor], *, target_paddings: Tensor, is_valid_example: Tensor
    ) -> dict[str, Union[WeightedScalar, Tensor]]:
        """Computes input lengths stats.

        Args:
            input_batch: See forward method signature.
            target_paddings: See _compute_target_paddings method return value.
            is_valid_example: A 0/1 Tensor of shape [batch_size], 1 if the example is
                a valid input to the loss computation.

        Returns:
            A dictionary of input stats summaries.
        """
        valid_frames = (1.0 - input_batch["paddings"]) * is_valid_example[:, None]
        valid_labels = (1.0 - target_paddings) * is_valid_example[:, None]

        total_source_lengths = jnp.sum(valid_frames)
        total_target_lengths = jnp.sum(valid_labels)
        total_num_examples = jnp.maximum(is_valid_example.sum(), 1.0)
        total_num_frames = jnp.maximum(jnp.size(input_batch["paddings"]), 1)
        input_stats = {
            "input_stats/average_target_length": WeightedScalar(
                total_target_lengths / total_num_examples, total_num_examples
            ),
            "input_stats/average_source_length": WeightedScalar(
                total_source_lengths / total_num_examples, total_num_examples
            ),
            "input_stats/frame_packing_efficiency": WeightedScalar(
                total_source_lengths / total_num_frames, total_num_frames
            ),
        }
        return input_stats


class CTCDecoderModel(BaseASRDecoderModel):
    """CTC decoder model.

    CTC maps continuous sequences (e.g. speech embeddings) to "labelings", sequences over a finite
    vocab (with size `vocab_size`). The vocab does not have to contain EOS.
    Output sequences should be no longer than input sequences, and may possibly be shorter (e.g.
    after removing repeated tokens and/or "blanks", represented by `blank_id`).

    Reference:
    https://dl.acm.org/doi/10.1145/1143844.1143891
    """

    @config_class
    class Config(BaseASRDecoderModel.Config):
        """Configures CTCDecoderModel."""

        # Layer to map hidden state to vocab logits.
        lm_head: BaseLayer.Config = Linear.default_config()
        # Blank token ID.
        blank_id: int = 0

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "lm_head", cfg.lm_head.set(input_dim=cfg.input_dim, output_dim=cfg.vocab_size)
        )

    def predict(self, input_batch: Nested[Tensor]) -> Tensor:
        """Computes logits.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.

        Returns:
            Logits of shape [batch_size, num_frames, vocab_size]. Logits corresponding to padding
            frames will be 0's. Note that the returned logits are not proper log probabilities, i.e.
            we have not subtracted the log-partition function.
        """
        inputs = input_batch["inputs"]
        paddings: Tensor = input_batch["paddings"]
        logits = self.lm_head(inputs)
        return logits * safe_not(paddings)[..., None]

    def _loss_summaries(
        self,
        *,
        total_ctc_loss: Tensor,
        per_example_weight: Tensor,
        paddings: Tensor,
        target_paddings: Tensor,
    ) -> dict[str, Union[WeightedScalar, Tensor]]:
        valid_frame_mask = (1.0 - paddings) * per_example_weight[:, None]
        valid_label_mask = (1.0 - target_paddings) * per_example_weight[:, None]

        num_valid_frames = jnp.maximum(jnp.sum(valid_frame_mask), 1.0)
        num_valid_labels = jnp.maximum(jnp.sum(valid_label_mask), 1.0)
        per_frame_loss = total_ctc_loss / num_valid_frames
        per_label_loss = total_ctc_loss / num_valid_labels
        batch_size = jnp.maximum(per_example_weight.shape[0], 1.0)

        ret_dict = {}
        # 1. loss/example_weight
        ret_dict["loss/example_weight"] = WeightedScalar(jnp.mean(per_example_weight), batch_size)
        # 2. loss/ctc_loss
        ret_dict["loss/ctc_loss"] = WeightedScalar(
            total_ctc_loss / jnp.maximum(per_example_weight.sum(), 1),
            jnp.maximum(per_example_weight.sum(), 1),
        )
        # 3. loss/invalid_seq_percent, per_frame_ctc_loss, per_label_ctc_loss
        invalid_example_percent = 1.0 - jnp.sum(per_example_weight) / batch_size
        ret_dict["loss/invalid_seq_percent"] = invalid_example_percent
        ret_dict["loss/per_frame_ctc_loss"] = WeightedScalar(per_frame_loss, num_valid_frames)
        ret_dict["loss/per_label_ctc_loss"] = WeightedScalar(per_label_loss, num_valid_labels)

        return ret_dict

    def forward(
        self,
        input_batch: Nested[Tensor],
    ) -> tuple[Tensor, Nested[Tensor]]:
        """Computes CTC loss.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                target_labels: An int Tensor of shape [batch_size, num_labels].
                    Values should be in the range [0, vocab_size). We assume there are no BOS
                    tokens, and that sequences are not truncated. Out-of-range values are excluded
                    from the loss calculation (e.g., paddings and EOS can be represented this way).

        Returns:
            A tuple (loss, aux_outputs):
                loss: A scalar loss value.
                aux_outputs: A dict containing:
                    per_example_loss: A float Tensor of shape [batch_size].
                    per_example_weight: A float Tensor of shape [batch_size].
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        target_labels: Tensor = input_batch["target_labels"]
        target_paddings: Tensor = _compute_target_paddings(target_labels, vocab_size=cfg.vocab_size)

        # Compute CTC loss.
        logits = self.predict(input_batch)
        per_example_loss = optax.ctc_loss(
            logits=logits,
            logit_paddings=paddings,
            labels=target_labels,
            label_paddings=target_paddings,
            blank_id=cfg.blank_id,
        )

        # Drop examples with targets longer than inputs.
        per_example_weight = _is_valid_ctc_seq(
            paddings=paddings, target_labels=target_labels, target_paddings=target_paddings
        )
        per_example_weight = per_example_weight.astype(per_example_loss.dtype)

        # Compute weighted loss.
        loss = jnp.sum(per_example_loss * per_example_weight) / jnp.maximum(
            per_example_weight.sum(), 1
        )
        aux_outputs = dict(per_example_weight=per_example_weight, per_example_loss=per_example_loss)
        # Add summaries.
        summary = self._input_stats_summaries(
            input_batch=input_batch,
            target_paddings=target_paddings,
            is_valid_example=per_example_weight,
        )
        summary.update(
            self._loss_summaries(
                total_ctc_loss=jnp.sum(per_example_loss * per_example_weight),
                per_example_weight=per_example_weight,
                paddings=paddings,
                target_paddings=target_paddings,
            )
        )
        for name, value in summary.items():
            self.add_summary(name, value)
        return loss, aux_outputs

    def _tokens_to_scores(
        self,
        input_batch: Nested[Tensor],
        *,
        num_decodes: int,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
    ) -> Callable[[Tensor, Nested[Tensor]], tuple[Tensor, Nested[Tensor]]]:
        """Returns a function that maps current token IDs and model state to next logits and updated
        state, to be used with decoding (see e.g. `beam_search_decode` or `sample_decode`).
        """
        paddings = input_batch["paddings"]
        logits_modifier = maybe_instantiate(logits_modifier)

        # [batch_size, num_frames, vocab_size].
        logits = self.predict(input_batch)
        if logits.dtype in (jnp.bfloat16, jnp.float16):
            # Cast for log softmax.
            logits = logits.astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits)
        # Mask out log probs at padding frames.
        log_probs += paddings[..., None] * NEG_INF
        # Extend log_probs by 1 step, so we can always decode up to `num_frames` non-EOS tokens.
        # [batch_size, num_frames + 1, vocab_size].
        log_probs = jnp.pad(log_probs, ((0, 0), (0, 1), (0, 0)), constant_values=NEG_INF)
        # Add a dummy EOS token:
        # eos_log_probs[b, t, :] = 0 if paddings_extended[b, t] else NEG_INF.
        paddings_extended = jnp.pad(paddings, ((0, 0), (0, 1)), constant_values=1)
        eos_log_probs = safe_not(paddings_extended)[:, :, None] * NEG_INF
        # [batch_size, num_frames + 1, vocab_size + 1].
        log_probs = jnp.concatenate([log_probs, eos_log_probs], axis=-1)
        # Apply logits modifier after (e.g. if applying top-k, don't factor in padding scores).
        if logits_modifier:
            log_probs = logits_modifier(log_probs)

        def tokens_to_scores(
            token_ids: Tensor, state: Nested[Tensor]
        ) -> tuple[Tensor, Nested[Tensor]]:
            # CTC assumes conditional independence between frames.
            del token_ids
            time_step = state["time_step"]
            # [batch_size, vocab_size].
            log_probs_t = log_probs[:, time_step, :]
            # [batch_size * num_decodes, vocab_size].
            log_probs_t = flatten_decoding_dim(
                add_decoding_dim(log_probs_t, num_decodes=num_decodes),
            )
            state["time_step"] = time_step + 1
            return log_probs_t, state

        return tokens_to_scores

    def beam_search_decode(
        self,
        input_batch: Nested[Tensor],
        num_decodes: int = 1,
        prefix_merger: Optional[PrefixMerger] = None,
    ) -> DecodeOutputs:
        """CTC beam search decoding with optional prefix merging.

        The output hypotheses will have blanks and repeats removed (via `_map_label_sequences`).

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
            num_decodes: Beam size.
            prefix_merger: An optional PrefixMerger to apply during decoding.

        Returns:
            DecodeOutputs, containing:
                raw_sequences: An int Tensor of shape [batch_size, num_decodes, num_frames].
                sequences: An int Tensor of shape [batch_size, num_decodes, num_frames].
                paddings: A 0/1 Tensor of shape [batch_size, num_decodes, num_frames].
                scores: A Tensor of shape [batch_size, num_decodes].

        Raises:
            ValueError: If max_decode_len is not None.
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        # Add 1 so we can drop EOS while ensuring decodes can be up to `num_frames`.
        max_decode_len = paddings.shape[-1] + 1
        beam_search_outputs = beam_search_decode(
            inputs=jnp.zeros_like(paddings),
            time_step=jnp.zeros(paddings.shape[0], dtype=jnp.int32),
            cache={"time_step": jnp.array(0)},
            tokens_to_scores=self._tokens_to_scores(input_batch, num_decodes=num_decodes),
            num_decodes=num_decodes,
            eos_id=cfg.vocab_size,  # Dummy EOS token.
            max_decode_len=max_decode_len,
            prefix_merger=prefix_merger,
        )
        return self._postprocess_outputs(
            sequences=beam_search_outputs.sequences,
            paddings=paddings,
            scores=beam_search_outputs.scores,
        )

    def sample_decode(
        self,
        input_batch: Nested[Tensor],
        *,
        num_decodes: int = 1,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
    ) -> DecodeOutputs:
        """CTC sample decoding.

        The output hypotheses will have blanks and repeats removed (via `_map_label_sequences`).
        To perform greedy decoding, provide `top_k_logits(1)` as the logits modifier.

        Args:
            input_batch: See `beam_search_decode`.
            num_decodes: See `beam_search_decode`.
            logits_modifier: An optional logits modifier to apply prior to softmax.
                If None, do not modify the logits.

        Returns:
            See `beam_search_decode`.
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        # Add 1 so we can drop EOS while ensuring decodes can be up to `num_frames`.
        max_decode_len = paddings.shape[-1] + 1
        sample_decode_outputs = sample_decode(
            inputs=jnp.zeros_like(paddings),
            time_step=jnp.zeros(paddings.shape[0], dtype=jnp.int32),
            cache={"time_step": jnp.array(0)},
            tokens_to_scores=self._tokens_to_scores(
                input_batch, num_decodes=num_decodes, logits_modifier=logits_modifier
            ),
            num_decodes=num_decodes,
            prng_key=self.prng_key,
            max_decode_len=max_decode_len,
            stop_decoding_condition=StopOnSubsequence([[cfg.vocab_size]]),  # Dummy EOS token.
        )
        return self._postprocess_outputs(
            sequences=sample_decode_outputs.sequences,
            paddings=paddings,
            scores=sample_decode_outputs.token_scores,
        )

    def greedy_decode(self, input_batch: Nested[Tensor]) -> DecodeOutputs:
        """CTC greedy decoding.

        The output hypotheses will have blanks and repeats removed (via `_map_label_sequences`).

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.

        Returns:
            DecodeOutputs, containing:
                raw_sequences: An int Tensor of shape [batch_size, 1, num_frames].
                sequences: An int Tensor of shape [batch_size, 1, num_frames].
                paddings: A 0/1 Tensor of shape [batch_size, 1, num_frames].
                scores: A Tensor of shape [batch_size, 1].
        """
        cfg: CTCDecoderModel.Config = self.config
        paddings: Tensor = input_batch["paddings"]
        # [batch_size, num_frames, vocab_size].
        logits = self.predict(input_batch)
        # [batch, 1, num_frames].
        sequences = jnp.argmax(logits, axis=-1)[:, None, :]
        # Remove repeats and blanks.
        # We make the assumption that the trailing padding positions have 0 as the argmax index.
        outputs = _map_label_sequences(
            inputs=sequences, remove_repeats=True, blank_id=cfg.blank_id, pad_id=0
        )

        # [batch_size, num_frames, vocab_size].
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_probs += paddings[..., None] * NEG_INF
        # [batch, num_frames, 1].
        scores = jnp.take_along_axis(log_probs, sequences[:, 0, :, None], axis=-1)
        # [batch, 1].
        scores = jnp.sum(jnp.squeeze(scores, axis=-1) * safe_not(paddings), axis=1, keepdims=True)

        return DecodeOutputs(
            raw_sequences=sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=scores,
        )

    def _postprocess_outputs(self, *, sequences: Tensor, paddings: Tensor, scores: Tensor):
        cfg: CTCDecoderModel.Config = self.config
        live_mask = safe_not(paddings)[:, None, :]
        # Drop dummy decode position and mask outputs corresponding to padding frames.
        sequences = sequences[..., :-1] * live_mask
        # If given per-token scores, sum non-padding scores along sequence dim.
        if scores.ndim == 3:
            scores = (scores[..., :-1] * live_mask).sum(axis=-1)
        # Remove repeats and blanks.
        outputs = _map_label_sequences(
            inputs=sequences, remove_repeats=True, blank_id=cfg.blank_id, pad_id=0
        )
        return DecodeOutputs(
            raw_sequences=sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=scores,
        )

    def align(self, input_batch: Nested[Tensor]) -> Nested[Tensor]:
        """Given an input_batch that contains both audio and labels, outputs text-audio alignment.
        Args:
            input_batch: See `CTCDecoderModel`'s forward interface. `input_batch` should contain:
                * inputs: A Tensor of shape [batch_size, num_frames, dim].
                * paddings: A 0/1 Tensor of shape [batch_size, num_frames].
                * target_labels: A Tensor of shape [batch_size, label_length].
                    target_labels < 0 means this is a padding position.
        Returns:
            A NestedTensor, converted from `ctc_aligner.AlignmentOutput` object
        """
        logits = self.predict(input_batch)
        log_posterior = jax.nn.log_softmax(logits, axis=-1)
        log_pos_paddings = cast(Tensor, input_batch["paddings"])
        labels = cast(Tensor, input_batch["target_labels"])
        label_paddings = jnp.where(labels >= 0, 0, 1)

        alignment_output = ctc_aligner.ctc_forced_alignment(
            log_pos=log_posterior,
            log_pos_paddings=log_pos_paddings,
            labels=labels,
            label_paddings=label_paddings,
            blank_id=self.config.blank_id,
        )
        return alignment_output.asdict()


def _map_label_sequences(
    inputs: Tensor, *, remove_repeats: bool, blank_id: int = 0, pad_id: int = 0
) -> Nested[Tensor]:
    """Removes blanks, paddings, and repeats from the input sequences, as used in CTC or RNN-T.

    Note that unless pad_id is the same as blank_id, pad_id should not be in the range of the vocab.
    We use pad_id to infer padding positions in the inputs.

    Args:
        inputs: An int Tensor of shape [..., max_decode_len] containing decoded sequences.
        remove_repeats: A boolean indicating whether we remove repeats or not. It is True for CTC,
            False for RNN-T.
        blank_id: Token ID corresponding to blanks.
        pad_id: Token ID corresponding to paddings.

    Returns:
        A dict containing:
            sequences: A Tensor of shape [..., max_decode_len] containing label sequences.
            paddings: A 0/1 Tensor of shape [..., max_decode_len]. 1's represent paddings.
            lengths: A Tensor of shape [..., 1] containing the length of each sequence.
    """
    max_decode_len = inputs.shape[-1]
    # Identify points at which the token is a valid label token to keep.
    # `indicators` has shape [batch_size, num_decodes, max_decode_len], and has a value of 1
    # in positions corresponding to inputs we intend to keep,
    # i.e., the token is not blank or padding.
    indicators = (inputs != blank_id) & (inputs != pad_id)
    if remove_repeats:
        # Identify points at which curr != prev.
        # I.e., the token is not blank or padding, and is different from the previous token.
        y = jnp.concatenate([jnp.full(inputs.shape[:-1] + (1,), pad_id), inputs], axis=-1)
        indicators = (y[..., 1:] != y[..., :-1]) & indicators

    # Compute lengths of final sequences. [..., 1].
    lens = jnp.sum(indicators, axis=-1, keepdims=True, dtype=inputs.dtype)

    # Compute sequences by left-justifying the tokens-to-keep. Under jit, we use a dispatch matrix
    # of shape [batch_size, num_decodes, max_decode_len, max_decode_len].
    # dispatch[..., from, to] == 1 means inputs[:, :, from] is put at sequences[:, :, to].
    # dispatch[..., i, :] == 0 means we drop token i in the inputs.
    # [batch_size, num_decodes, max_decode_len, max_decode_len].
    dispatch = jax.nn.one_hot(
        jnp.cumsum(indicators, axis=-1) * indicators - 1, max_decode_len, dtype=inputs.dtype
    )
    sequences = jnp.einsum("...nm,...n->...m", dispatch, inputs)
    paddings = jnp.arange(max_decode_len) >= lens
    if pad_id != 0:
        sequences = jnp.where(paddings, pad_id, sequences)
    return dict(sequences=sequences, paddings=paddings, lengths=lens)


class RNNPredictionNetwork(BaseLayer):
    """RNN prediction network internal language model."""

    @config_class
    class Config(BaseLayer.Config):
        """Configs RNNPredictionNetwork."""

        # Vocab size.
        vocab_size: Required[int] = REQUIRED
        # The embedding dim.
        emb_dim: Required[int] = REQUIRED
        # The output dim.
        output_dim: Required[int] = REQUIRED

        # Embedding lookup layer.
        embedding: Embedding.Config = Embedding.default_config()
        # RNN cell of the internal LM. Defaults to a 1 layer LSTM.
        rnn_cell: BaseRNNCell.Config = LSTMCell.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "embedding", cfg.embedding.set(num_embeddings=cfg.vocab_size, dim=cfg.emb_dim)
        )
        self._add_child("rnn", cfg.rnn_cell.set(input_dim=cfg.emb_dim, output_dim=cfg.output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        """Computes prediction network output from the inputs.

        Args:
            inputs: An int Tensor of shape [batch_size, num_labels]. Valid tokens are in the range
                [0, vocab_size). Out-of-range token ids are clamped to the bounds of the array.
                See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
                #out-of-bounds-indexing.

        Returns:
            A Tensor of shape [batch_size, num_labels, output_dim].
        """
        time_major_outputs = self.rnn(
            time_major_inputs=jnp.transpose(self.embedding(x=inputs), [1, 0, 2])
        )
        return jnp.transpose(time_major_outputs, [1, 0, 2])

    def init_states(self, *, batch_size: int) -> Nested[Tensor]:
        """Returns the prediction network initial step states, to be used by `extend_step`."""
        return self.rnn.init_states(batch_size=batch_size)

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        data: Tensor,
    ) -> tuple[Nested[Tensor], Tensor]:
        """Computes prediction network outputs and RNN state updates for one step.

        Args:
            cached_states: A NestedTensor returned by `init_states` or `extend_step`.
            data: An int Tensor of shape [batch_size, num_labels].

        Returns:
            (updated_cached_states, outputs), where `outputs` is a Tensor of shape
                [batch_size, output_dim].
        """
        return self.rnn.extend_step(data=self.embedding(x=data), cached_states=cached_states)


class TransducerDecoderModel(BaseASRDecoderModel):
    """Transducer decoder.

    It is often referred as rnn-transducer or rnnt in the literature.
    """

    @config_class
    class Config(BaseASRDecoderModel.Config):
        """Configures TransducerDecoderModel."""

        # The lm dim.
        lm_dim: Required[int] = REQUIRED
        # The joint network dim.
        joint_dim: Required[int] = REQUIRED

        # Blank token ID.
        blank_id: int = 0
        bos_id: int = 1
        eos_id: int = 2

        # Prediction network internal language model.
        prediction_network: RNNPredictionNetwork.Config = RNNPredictionNetwork.default_config()
        # Joint network that combines acoustic model and language model features.
        # AM projection.
        am_proj: Linear.Config = Linear.default_config()
        # LM projection.
        lm_proj: Linear.Config = Linear.default_config()
        # Transducer that maps the hidden state to vocab logits.
        transducer: Transducer.Config = Transducer.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.eos_id == cfg.blank_id:
            raise ValueError(
                "eos_id and blank_id should be different for the transducer model, "
                f"but got eos_id = blank_id = {cfg.blank_id}."
            )
        self.vlog(
            3,
            (
                f"am_dim={cfg.input_dim}, lm_dim={cfg.lm_dim}, joint_dim={cfg.joint_dim}, "
                f"vocab_size={cfg.vocab_size}."
            ),
        )
        # In most common cases, am_data and lm_data are summed together after the projection, thus
        # we only keep one bias in the two projections.
        self._add_child(
            "am_proj", cfg.am_proj.set(input_dim=cfg.input_dim, output_dim=cfg.joint_dim, bias=True)
        )
        self._add_child(
            "lm_proj", cfg.lm_proj.set(input_dim=cfg.lm_dim, output_dim=cfg.joint_dim, bias=False)
        )
        self._add_child(
            "prediction_network",
            cfg.prediction_network.set(
                vocab_size=cfg.vocab_size,
                output_dim=cfg.lm_dim,
            ),
        )
        transducer_cfg = cfg.transducer.set(input_dim=cfg.joint_dim, vocab_size=cfg.vocab_size)
        transducer_cfg.logits_to_log_probs.blank_id = cfg.blank_id
        self._add_child("transducer", transducer_cfg)

    def forward(self, input_batch: Nested[Tensor]) -> tuple[Tensor, Nested[Tensor]]:
        """Computes the transducer loss.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                target_labels: An int Tensor of shape [batch_size, num_labels]. Prediction target
                    of the transducer decoder.
                target: A dictionary with input_ids as key, and an int Tensor of shape
                    [batch_size, num_labels] as value. Prediction inputs to the transducer decoder.

            For both target_labels and target["input_ids"], values should be in the range
            [0, vocab_size). target_labels does not contain BOS and valid label tokens are
            followed by a EOS token. input_ids starts with a BOS token. Sequences are not
            truncated. Out-of-range values are excluded from the loss calculation.

        Returns:
            A tuple (loss, per_example):
                loss: A scalar of the transducer loss.
                per_example: A dict containing transducer decoder outputs of the following keys:
                    weight: A tensor of shape [batch_size], the aggregation weight of the
                        per-example loss.
                    loss: A tensor of shape [batch_size] representing per-example loss.
        """
        cfg: TransducerDecoderModel.Config = self.config

        # [batch, src_max_len, joint_dim].
        am_data = self.am_proj(input_batch["inputs"])
        am_paddings: Tensor = input_batch["paddings"]
        chex.assert_type(am_paddings, jnp.bool)
        target_labels: Tensor = input_batch["target_labels"]
        target_paddings: Tensor = _compute_target_paddings(target_labels, vocab_size=cfg.vocab_size)

        # [batch, tgt_max_len, joint_dim].
        lm_data = self.lm_proj(self.prediction_network(inputs=input_batch["target"]["input_ids"]))

        _, per_example = self.transducer(
            am_data=am_data,
            am_paddings=am_paddings,
            lm_data=lm_data,
            lm_paddings=target_paddings,
            target_labels=target_labels,
        )
        per_example_loss, per_example_weight = (
            per_example["loss"],
            per_example["is_valid_example"],
        )
        per_example_weight = per_example_weight.astype(per_example_loss.dtype)

        # Compute weighted loss.
        loss = jnp.sum(per_example_loss * per_example_weight) / jnp.maximum(
            per_example_weight.sum(), 1
        )
        aux_outputs = dict(per_example_weight=per_example_weight, per_example_loss=per_example_loss)

        # Add input summaries.
        input_summary = self._input_stats_summaries(
            input_batch=input_batch,
            target_paddings=target_paddings,
            is_valid_example=per_example_weight,
        )
        for name, value in input_summary.items():
            self.add_summary(name, value)

        self.add_summary(
            "loss/example_weight",
            WeightedScalar(jnp.mean(per_example_weight), per_example_weight.shape[0]),
        )
        self.add_summary(
            "loss/rnnt_loss",
            WeightedScalar(loss, jnp.maximum(1, per_example_weight.sum())),
        )
        return loss, aux_outputs

    def _tokens_to_scores(
        self,
        input_batch: Nested[Tensor],
        *,
        num_decodes: int,
        max_decode_len: int,
    ) -> Callable[[Tensor, Nested[Tensor]], tuple[Tensor, Nested[Tensor]]]:
        """Returns a function that maps current token IDs and model state to next logits and updated
            state, to be used with decoding, see `beam_search_decode`.

        The signature is [batch*beam, vocab], {} = tokens_to_scores([batch*beam, 1], {}).
            state_cache contains keys:
            - am_step: the am frame index.
            - lm_states: the prediction network rnn states.
            - lm_data: the projected rnn prediction network outputs.
            - decode_step: number of decode steps.
        """
        cfg = self.config
        vocab_size = cfg.vocab_size
        blank_id, eos_id = cfg.blank_id, cfg.eos_id
        # [batch].
        src_len = jnp.sum(1 - input_batch["paddings"], axis=-1)
        # [batch, src_max_len, joint_dim].
        am_data = self.am_proj(input_batch["inputs"])
        batch_size, src_max_len = input_batch["paddings"].shape  # pytype: disable=attribute-error

        def tokens_to_scores(
            token_ids: Tensor, state_cache: Nested[Tensor]
        ) -> tuple[Tensor, Nested[Tensor]]:
            # [batch*beam, 1].
            is_blank = token_ids == blank_id

            # 1. Computes am_data at current step.
            # [batch*beam].
            am_step_at_t_flatten = state_cache["am_step"] + jnp.squeeze(is_blank, axis=1)
            # [batch, beam].
            am_step_at_t = unflatten_decoding_dim(
                am_step_at_t_flatten, batch_size=batch_size, num_decodes=num_decodes
            )
            # [batch, beam, src_max_len].
            am_indices_at_t = jax.nn.one_hot(am_step_at_t, src_max_len, dtype=am_data.dtype)

            # Slice am_t. am_data_at_t[b, k, :] = am_data[b, am_step_at_t[b, k], :].
            # [batch, beam, joint_dim].
            am_data_at_t = jnp.einsum("bso,bks->bko", am_data, am_indices_at_t)
            # [batch*beam, 1, joint_dim]. Flatten and add back the sequence dimension.
            am_data_at_t = flatten_decoding_dim(am_data_at_t)[:, None, :]

            # 2. Computes lm_data at current step.
            with child_context("prediction_network_decode", module=self.prediction_network):
                # [batch*beam, ...], [batch*beam, joint_dim].
                new_lm_states, new_preproj_lm_data = self.prediction_network.extend_step(
                    data=jnp.squeeze(token_ids, axis=-1),
                    cached_states=state_cache["lm_states"],
                )
            new_lm_data = self.lm_proj(new_preproj_lm_data)
            # lm_data = state_cache["lm_data"] if is_blank else new_lm_data.
            # [batch*beam, 1, joint_dim].
            lm_data_at_t = (
                state_cache["lm_data"] * is_blank[:, :, None]
                + (new_lm_data * (1 - is_blank))[:, None, :]
            )

            # updated_lm_states = state_cache["lm_states"] if is_blank else new_lm_states.
            # [batch*beam, ...].
            lm_states_at_t = vectorized_tree_map(
                lambda x1, x2: x1 * is_blank + x2 * (1 - is_blank),
                state_cache["lm_states"],
                new_lm_states,
            )
            pred = self.transducer.predict(am_data=am_data_at_t, lm_data=lm_data_at_t)

            # [batch*beam, 1, 1, vocab].
            log_probs = log_probs_from_blank_and_tokens(
                log_prob_blank=pred["log_prob_blank"],  # [batch*beam, 1, 1].
                log_prob_tokens=pred["log_prob_tokens"],  # [batch*beam, 1, 1, vocab].
                blank_id=blank_id,
            )
            # [batch*beam, vocab].
            log_probs = jnp.squeeze(log_probs, axis=(1, 2))

            # Force eos when all speech frames are consumed or at the last step.
            # [batch*beam, 1].
            force_eos = jnp.logical_or(
                # all frames are consumed.
                flatten_decoding_dim(am_step_at_t >= src_len[:, None]),
                # reaches last step
                state_cache["decode_step"] == max_decode_len - 1,
            )[:, None]

            # [1, vocab].
            eos_id_onehot = jax.nn.one_hot(eos_id, vocab_size, dtype=jnp.int32)[None, :]
            # log_probs[b, eos] = 0 if force_eos[b] else NEG_INF.
            # log_probs is of shape [batch*beam, vocab].
            log_probs *= 1 - eos_id_onehot
            log_probs += (1 - force_eos) * eos_id_onehot * NEG_INF
            # log_probs[b, non_eos] = NEG_INF if force_eos[b].
            log_probs += force_eos * (1 - eos_id_onehot) * NEG_INF

            new_cache = dict(
                am_step=am_step_at_t_flatten,
                lm_data=lm_data_at_t,
                lm_states=lm_states_at_t,
                decode_step=state_cache["decode_step"] + 1,
            )
            return log_probs, new_cache

        return tokens_to_scores

    def beam_search_decode(
        self,
        input_batch: Nested[Tensor],
        num_decodes: int,
        max_decode_len: int,
        prefix_merger: Optional[PrefixMerger] = None,
    ) -> DecodeOutputs:
        """Transducer label-synchronous search.

        Each hypothesis in the beam has the same length of tokens, including
            both blank and label tokens.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim] from encoder outputs.
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
            num_decodes: Beam size.
            max_decode_len: maximum number of decode steps to run beam search.
                Decoding terminates if an eos token is not emitted after max_decode_steps
                steps. This value can depend on the tokenization.
            prefix_merger: An optional PrefixMerger to apply during decoding.

        Returns:
            DecodeOutputs, containing
                raw_sequences: An int Tensor of shape [batch_size, num_decodes, max_decode_len].
                sequences: An int Tensor of shape [batch_size, num_decodes, max_decode_len].
                paddings: A 0/1 Tensor of shape [batch_size, num_decodes, max_decode_len].
                scores: A Tensor of shape [batch_size, num_decodes].

        Raises:
            ValueError: If max_decode_len <= src_max_len.
        """
        paddings: Tensor = input_batch["paddings"]
        batch_size, src_max_len = paddings.shape
        if max_decode_len <= src_max_len:
            raise ValueError(f"{max_decode_len=} is smaller than {src_max_len=}.")

        cfg = self.config
        blank_id, eos_id, bos_id = cfg.blank_id, cfg.eos_id, cfg.bos_id

        # Starts decoding with [BOS] token.
        inputs = jnp.zeros((batch_size, max_decode_len), dtype=jnp.int32)
        inputs = inputs.at[:, 0].set(bos_id)

        init_states = {
            "am_step": jnp.zeros(batch_size),
            "lm_states": self.prediction_network.init_states(batch_size=batch_size),
            "lm_data": jnp.zeros((batch_size, 1, self.config.joint_dim)),
            "decode_step": jnp.array(0),
        }

        beam_search_outputs = beam_search_decode(
            inputs=inputs,
            time_step=infer_initial_time_step(inputs, pad_id=0),
            cache=init_states,
            tokens_to_scores=self._tokens_to_scores(
                input_batch, num_decodes=num_decodes, max_decode_len=max_decode_len
            ),
            eos_id=eos_id,
            num_decodes=num_decodes,
            max_decode_len=max_decode_len,
            prefix_merger=prefix_merger,
        )
        # We drop eos token in the decoded sequence.
        decode_paddings = jnp.logical_or(
            jnp.cumsum(beam_search_outputs.sequences == eos_id, axis=-1),
            # Return all paddings for invalid sequences.
            (beam_search_outputs.scores == NEG_INF)[..., None],
        )
        # Drop dummy decode position and mask outputs corresponding to padding.
        sequences = beam_search_outputs.sequences * (1 - decode_paddings)
        # Remove blanks.
        outputs = _map_label_sequences(
            inputs=sequences,
            remove_repeats=False,
            blank_id=blank_id,
            pad_id=0,
        )
        return DecodeOutputs(
            raw_sequences=beam_search_outputs.sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=beam_search_outputs.scores,
        )


def _paddings_from_decode_sequence(*, sequences: Tensor, scores: Tensor, eos_id: int) -> Tensor:
    """Computes paddings from decoded outputs.

    Args:
        sequences: An integer Tensor of shape [batch_size, num_decodes, max_decode_len].
        scores: A float Tensor of shape  [batch_size, num_decodes, 1 or max_decode_len]. Positions
            corresponding to scores <= NEG_INF will be treated as paddings.
        eos_id: An integer of eos token id.

    Returns:
        A 0/1 Tensor of shape [batch_size, num_decodes, max_decode_len].
    """
    # [batch_size, num_decodes, max_decode_len].
    # Note: jax.lax.cummax doesn't support axis=-1.
    eos_indicator = (sequences == eos_id).astype(sequences.dtype)
    paddings_exclude_eos = jax.lax.cummax(eos_indicator, axis=sequences.ndim - 1)
    paddings = jnp.pad(paddings_exclude_eos, ((0, 0), (0, 0), (1, 0)), constant_values=0)[:, :, :-1]
    # Return all paddings for invalid tokens/sequences.
    return jnp.logical_or(paddings, scores <= NEG_INF)


class LASDecoderModel(BaseASRDecoderModel):
    """Listen Attend and Spell decoder.

    https://arxiv.org/abs/1508.01211
    """

    @config_class
    class Config(BaseASRDecoderModel.Config):
        """Configures LASDecoderModel.

        Attributes:
            decoder: A config that instantiates the autoregressive Decoder.
            z_loss_scale: Coefficient for auxiliary z-loss loss term.
            label_smoothing: The factor to control label smoothing in cross entropy.
        """

        decoder: Decoder.Config = Decoder.default_config()
        z_loss_scale: float = 0.0
        label_smoothing: float = 0.0

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        transformer_cfg = cfg.decoder.transformer.layer
        transformer_cfg.cross_attention = TransformerAttentionLayer.default_config()
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        cfg.decoder.set(vocab_size=cfg.vocab_size)
        cfg.decoder.transformer.layer.cross_attention.source_dim = cfg.input_dim
        self._add_child("decoder", cfg.decoder)

    def forward(self, input_batch: Nested[Tensor]) -> tuple[Tensor, Nested[Tensor]]:
        """Computes the cross entropy loss.

        See BaseASRDecoderModel.forward docstring for details.
        """
        cfg: LASDecoderModel.Config = self.config
        target_labels: Tensor = input_batch["target_labels"]
        target_paddings: Tensor = _compute_target_paddings(target_labels, vocab_size=cfg.vocab_size)

        # Cross-attention logit biases: [batch_size, 1, 1, source_len].
        cross_attention_logit_biases = self._compute_attention_logit_biases(
            paddings=input_batch["paddings"]
        )
        predict_outputs = self.decoder(
            input_batch=input_batch["target"],
            cross_attention_data=input_batch["inputs"],
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        # Filter out empty source sequences.
        # [batch_size].
        is_valid_example = (1 - input_batch["paddings"]).sum(axis=-1) > 0
        live_targets = (1 - target_paddings) * is_valid_example[:, None]
        loss, loss_dict = cross_entropy(
            logits=predict_outputs["logits"],  # [batch_size, target_len, num_classes].
            target_labels=target_labels,  # [batch_size, target_len].
            live_targets=live_targets,
            z_loss_scale=cfg.z_loss_scale,
            label_smoothing=cfg.label_smoothing,
        )

        # Add input summaries.
        input_summary = self._input_stats_summaries(
            input_batch=input_batch,
            target_paddings=target_paddings,
            is_valid_example=is_valid_example,
        )
        for name, value in input_summary.items():
            self.add_summary(name, value)
        # [batch_size, target_len].
        per_token_loss = loss_dict["per_target_loss"] * live_targets
        per_example_loss = jnp.sum(per_token_loss, axis=-1)
        # Number of valid tokens per example.
        per_example_weight = jnp.sum(live_targets, axis=-1)
        num_targets = per_example_weight.sum()
        self.add_summary("loss", WeightedScalar(loss, num_targets))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_targets))
        self.add_summary("token_accuracy", WeightedScalar(loss_dict["accuracy"], num_targets))
        return loss, dict(per_example_loss=per_example_loss, per_example_weight=per_example_weight)

    def _compute_attention_logit_biases(
        self,
        paddings: Tensor,
    ) -> Tensor:
        """Computes cross-attention logit biases for LAS decoder.

        Args:
            paddings: A 0/1 Tensor of shape [batch_size, source_len].

        Returns:
            Attention logit biases of shape [batch_size, num_heads, target_len, source_len].
        """
        # Compute padding logit biases: [batch_size, num_heads=1, target_len=1, source_len].
        cross_attention_biases = (paddings == 1) * NEG_INF
        cross_attention_biases = cross_attention_biases[:, None, None, :]
        return cross_attention_biases

    def _validate_decode_batch(self, input_batch: Nested[Tensor]):
        """Raises ValueError if `prefix` is not present in `input_batch`."""
        if "prefix" not in input_batch:
            raise ValueError("Input batch is expected to contain `prefix`.")

    def beam_search_decode(
        self,
        input_batch: Nested[Tensor],
        num_decodes: int,
        max_decode_len: int,
        brevity_penalty: Optional[Callable[[jnp.array, Tensor], jnp.array]] = None,
    ) -> DecodeOutputs:
        """LAS beam search decode.

        Args:
            input_batch: A dict containing:
                inputs: A Tensor of shape [batch_size, num_frames, dim].
                paddings: A 0/1 Tensor of shape [batch_size, num_frames]. 1's represent paddings.
                prefix: An int Tensor of shape [batch_size, prefix_length]. The prefix for
                    each example in the batch should begin with the [BOS] token.
            num_decodes: The number of decoded sequences to return. These are the number of
                hypotheses per batch example.
            max_decode_len: The maximum sequence length of decoded sequences, including prefix.
            brevity_penalty: Brevity penalty function for length normalization during beam search.
                Defaults to None, which is to use the function ((5 + length) / 6)^\alpha without
                length normalization, i.e. alpha=0. See Eq. 14 https://arxiv.org/abs/1609.08144.
                https://github.com/tensorflow/lingvo/blob/05a076b0783a8bbf4a770095966c472bb37bbf65/lingvo/core/beam_search_helper.py#L101-L107.

        Returns:
            DecodeOutputs, containing:
                raw_sequences: An int Tensor of shape [batch_size, num_decodes, max_decode_len],
                    raw sequences returned by beam search.
                sequences: An int Tensor of shape [batch_size, num_decodes, max_decode_len].
                    raw_sequences with padding and eos replaced with 0.
                paddings: A 0/1 Tensor of shape [batch_size, num_decodes, max_decode_len].
                scores: A Tensor of shape [batch_size, num_decodes].

        Raises:
             ValueError: If `prefix` is not present in `input_batch`.
        """
        dec_cfg = self.config.decoder
        self._validate_decode_batch(input_batch)

        # Cross-attention logit biases: [batch_size, num_heads=1, target_len=1, source_len].
        cross_attention_logit_biases = self._compute_attention_logit_biases(
            paddings=input_batch["paddings"]
        )

        with child_context("beam_search_decode", module=self.decoder):
            beam_search_outputs: decoding.BeamSearchOutputs = self.decoder.beam_search_decode(
                input_batch=input_batch,
                max_sequence_length=max_decode_len,
                num_decodes=num_decodes,
                cross_attention_data=input_batch["inputs"],
                cross_attention_logit_biases=cross_attention_logit_biases,
                brevity_penalty=brevity_penalty,
            )
            paddings = _paddings_from_decode_sequence(
                sequences=beam_search_outputs.sequences,
                scores=beam_search_outputs.scores[..., None],
                eos_id=dec_cfg.eos_token_id,
            )
            # Drop dummy decode position and mask outputs corresponding to padding.
            sequences = beam_search_outputs.sequences * safe_not(paddings)
        return DecodeOutputs(
            raw_sequences=beam_search_outputs.sequences,
            sequences=sequences,
            paddings=paddings,
            scores=beam_search_outputs.scores,
        )

    def sample_decode(
        self,
        input_batch: dict[str, Tensor],
        num_decodes: int,
        max_decode_len: int,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
    ) -> DecodeOutputs:
        """LAS sample decoding.

        Args:
            input_batch: See `beam_search_decode`.
            num_decodes: See `beam_search_decode`.
            max_decode_len: See `beam_search_decode`.
            logits_modifier: An optional logits modifier to apply prior to softmax.
                If None, do not modify the logits.

        Returns:
            See `beam_search_decode`.
        """
        cfg: LASDecoderModel.Config = self.config
        eos_id = cfg.decoder.eos_token_id
        self._validate_decode_batch(input_batch)

        cross_attention_logit_biases = self._compute_attention_logit_biases(
            paddings=input_batch["paddings"]
        )

        with child_context("sample_decode", module=self.decoder):
            sample_decode_outputs: decoding.SampleOutputs = self.decoder.sample_decode(
                input_batch=input_batch,
                max_sequence_length=max_decode_len,
                num_decodes=num_decodes,
                cross_attention_data=input_batch["inputs"],
                cross_attention_logit_biases=cross_attention_logit_biases,
                logits_modifier=logits_modifier,
                stop_decoding_condition=StopOnSubsequence([[eos_id]]),
            )
            paddings = _paddings_from_decode_sequence(
                sequences=sample_decode_outputs.sequences,
                scores=sample_decode_outputs.token_scores,
                eos_id=eos_id,
            )

            # Drop dummy decode position and mask outputs corresponding to padding.
            sequences = sample_decode_outputs.sequences * safe_not(paddings)
            # [batch_size, num_decodes].
            scores = jnp.sum(sample_decode_outputs.token_scores * safe_not(paddings), axis=-1)
        return DecodeOutputs(
            raw_sequences=sample_decode_outputs.sequences,
            sequences=sequences,
            paddings=paddings,
            scores=scores,
        )
