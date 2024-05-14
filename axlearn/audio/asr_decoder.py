# Copyright © 2023 Apple Inc.

"""ASR decoder layers."""

from typing import Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax

from axlearn.common import struct
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, ConfigOr, Required, config_class, maybe_instantiate
from axlearn.common.decoding import (
    NEG_INF,
    PrefixMerger,
    StopOnSubsequence,
    add_decoding_dim,
    beam_search_decode,
    compute_merge_matrix_by_prefix_ids,
    flatten_decoding_dim,
    sample_decode,
)
from axlearn.common.layers import Embedding, Linear
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module
from axlearn.common.rnn import BaseRNNCell, LSTMCell
from axlearn.common.utils import Nested, Tensor


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


class CTCPrefixMerger(PrefixMerger):
    """Merges equivalent lower-ranked beams into higher-ranked ones.

    Beams are compared after removing repeats and blanks following CTC.

    See Section 3.1 of https://dl.acm.org/doi/10.1145/1143844.1143891.
    """

    def __init__(self, blank_id: int):
        self._blank_id = blank_id

    def init_state(self, *, tokens: Tensor) -> Nested[Tensor]:
        """Initializes the prefix merger state from the initial prefix `tokens`.

        If the initial prefix is non-empty, we produce state equivalent to initializing from an
        empty prefix and invoking `update` token-by-token until the end of the initial prefix.
        """
        outputs = _map_label_sequences(
            tokens, remove_repeats=True, blank_id=self._blank_id, pad_id=-1
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


class DecodeOutputs(struct.PyTreeNode):
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


class BaseASRDecoderModel(BaseModel):
    """ASR decoder model base."""

    @config_class
    class Config(BaseModel.Config):
        """Configures BaseASRDecoderModel."""

        # Dimensionality of inputs.
        input_dim: Required[int] = REQUIRED
        # The vocab size.
        vocab_size: Required[int] = REQUIRED

    def forward(
        self,
        input_batch: Nested[Tensor],
    ) -> Tuple[Tensor, Nested[Tensor]]:
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

    def _compute_target_paddings(
        self,
        input_batch: Nested[Tensor],
    ) -> Tensor:
        """Computes target paddings and other input statistics.

        Args:
            input_batch: See forward method signature.

        Returns:
            target_paddings: A 0/1 Tensor of shape [batch_size, num_labels].
        """
        cfg = self.config
        target_labels: Tensor = input_batch["target_labels"]
        # Infer target_paddings from out-of-range labels.
        target_paddings = jnp.logical_or(cfg.vocab_size <= target_labels, target_labels < 0)
        return target_paddings

    def _input_stats_summaries(
        self, input_batch: Nested[Tensor]
    ) -> Dict[str, Union[WeightedScalar, Tensor]]:
        target_labels: Tensor = input_batch["target_labels"]
        target_paddings = self._compute_target_paddings(target_labels)
        batch_size = jnp.maximum(target_labels.shape[0], 1)
        num_source_elements = jnp.maximum(input_batch["paddings"].size, 1)  # type: ignore
        target_lengths = jnp.sum(1 - target_paddings, axis=-1)
        source_lengths = jnp.sum(1 - input_batch["paddings"], axis=-1)
        # pytype: disable=attribute-error
        ret_dict = {
            "input_stats/average_target_length": WeightedScalar(
                jnp.mean(target_lengths), batch_size
            ),
            "input_stats/average_source_length": WeightedScalar(
                jnp.mean(source_lengths), batch_size
            ),
            "input_stats/frame_packing_effiency": WeightedScalar(
                jnp.sum(source_lengths) / num_source_elements, num_source_elements
            ),
        }
        # pytype: enable=attribute-error
        return ret_dict


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
        paddings = input_batch["paddings"]
        logits = self.lm_head(inputs)
        return logits * (1 - paddings[..., None])

    def _input_stats_summaries(
        self, input_batch: Nested[Tensor], per_example_weight: Tensor
    ) -> Dict[str, Union[WeightedScalar, Tensor]]:
        paddings = input_batch["paddings"]
        target_paddings = self._compute_target_paddings(input_batch)
        valid_frame_mask = (1.0 - paddings) * per_example_weight[:, None]
        valid_label_mask = (1.0 - target_paddings) * per_example_weight[:, None]
        num_valid_frames = jnp.sum(valid_frame_mask)
        num_valid_labels = jnp.sum(valid_label_mask)
        num_valid_examples = jnp.maximum(per_example_weight.sum(), 1.0)
        # pytype: disable=attribute-error
        num_total_frames = jnp.maximum(input_batch["paddings"].size, 1)
        ret_dict = {
            "input_stats/average_target_length": WeightedScalar(
                num_valid_labels / num_valid_examples, num_valid_examples
            ),
            "input_stats/average_source_length": WeightedScalar(
                num_valid_frames / num_valid_examples, num_valid_examples
            ),
            "input_stats/frame_packing_effiency": WeightedScalar(
                num_valid_frames / num_total_frames, num_total_frames
            ),
        }
        # pytype: enable=attribute-error
        return ret_dict

    def _loss_summaries(
        self,
        *,
        total_ctc_loss: Tensor,
        per_example_weight: Tensor,
        paddings: Tensor,
        target_paddings: Tensor,
    ) -> Dict[str, Union[WeightedScalar, Tensor]]:
        valid_frame_mask = (1.0 - paddings) * per_example_weight[:, None]
        valid_label_mask = (1.0 - target_paddings) * per_example_weight[:, None]

        num_valid_frames = jnp.sum(valid_frame_mask)
        num_valid_labels = jnp.sum(valid_label_mask)
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
    ) -> Tuple[Tensor, Nested[Tensor]]:
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
        target_paddings: Tensor = self._compute_target_paddings(input_batch)
        paddings: Tensor = input_batch["paddings"]
        target_labels: Tensor = input_batch["target_labels"]

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
            input_batch=input_batch, per_example_weight=per_example_weight
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
    ) -> Callable[[Tensor, Nested[Tensor]], Tuple[Tensor, Nested[Tensor]]]:
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
        eos_log_probs = (1 - paddings_extended[:, :, None]) * NEG_INF
        # [batch_size, num_frames + 1, vocab_size + 1].
        log_probs = jnp.concatenate([log_probs, eos_log_probs], axis=-1)
        # Apply logits modifier after (e.g. if applying top-k, don't factor in padding scores).
        if logits_modifier:
            log_probs = logits_modifier(log_probs)

        def tokens_to_scores(
            token_ids: Tensor, state: Nested[Tensor]
        ) -> Tuple[Tensor, Nested[Tensor]]:
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
            time_step=jnp.zeros(paddings.shape[0], dtype=paddings.dtype),
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
            time_step=jnp.zeros(paddings.shape[0], dtype=paddings.dtype),
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
        scores = jnp.sum(jnp.squeeze(scores, axis=-1) * (1 - paddings), axis=1, keepdims=True)

        return DecodeOutputs(
            raw_sequences=sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=scores,
        )

    def _postprocess_outputs(self, *, sequences: Tensor, paddings: Tensor, scores: Tensor):
        cfg: CTCDecoderModel.Config = self.config
        live_mask = 1 - paddings[:, None, :]
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
    paddings = (jnp.arange(max_decode_len) >= lens).astype(inputs.dtype)
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
    ) -> Tuple[Nested[Tensor], Tensor]:
        """Computes prediction network outputs and RNN state updates for one step.

        Args:
            cached_states: A NestedTensor returned by `init_states` or `extend_step`.
            data: An int Tensor of shape [batch_size, num_labels].

        Returns:
            (updated_cached_states, outputs), where `outputs` is a Tensor of shape
                [batch_size, output_dim].
        """
        return self.rnn.extend_step(inputs=self.embedding(x=data), cached_states=cached_states)
