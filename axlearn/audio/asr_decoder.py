# Copyright © 2023 Apple Inc.

"""ASR decoder layers."""

from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import optax

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
from axlearn.common.layers import Linear
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor


def _is_valid_ctc_seq(
    *, paddings: Tensor, target_labels: Tensor, target_paddings: Tensor
) -> Tensor:
    """Returns for per example sequence if it passes validity check.

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
        whether a label is a padding.

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
        outputs = _map_label_sequences(tokens, blank_id=self._blank_id, pad_id=-1)
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


@chex.dataclass
class DecodeOutputs:
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


class CTCDecoderModel(BaseModel):
    """CTC decoder model.

    CTC maps continuous sequences (e.g. speech embeddings) to "labelings", sequences over a finite
    vocab (with size `vocab_size`). The vocab does not have to contain EOS.
    Output sequences should be no longer than input sequences, and may possibly be shorter (e.g.
    after removing repeated tokens and/or "blanks", represented by `blank_token_id`).

    Reference:
    https://dl.acm.org/doi/10.1145/1143844.1143891
    """

    @config_class
    class Config(BaseModel.Config):
        """Configures CTCDecoderModel."""

        # Dimensionality of inputs.
        dim: Required[int] = REQUIRED
        # The vocab size.
        vocab_size: Required[int] = REQUIRED
        # Layer to map hidden state to vocab logits.
        lm_head: BaseLayer.Config = Linear.default_config()
        # Blank token ID.
        blank_token_id: int = 0

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("lm_head", cfg.lm_head.set(input_dim=cfg.dim, output_dim=cfg.vocab_size))

    def predict(self, input_batch: Nested[Tensor]) -> Nested[Tensor]:
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
        paddings: Tensor = input_batch["paddings"]
        target_labels: Tensor = input_batch["target_labels"]

        # Infer target_paddings from out-of-range labels.
        target_paddings = jnp.logical_or(cfg.vocab_size <= target_labels, target_labels < 0)

        # Compute CTC loss.
        logits = self.predict(input_batch)
        per_example_loss = optax.ctc_loss(
            logits=logits,
            logit_paddings=paddings,
            labels=target_labels,
            label_paddings=target_paddings,
            blank_id=cfg.blank_token_id,
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
        paddings = input_batch["paddings"]
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
    ):
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
        paddings = input_batch["paddings"]
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

    def _postprocess_outputs(self, *, sequences: Tensor, paddings: Tensor, scores: Tensor):
        cfg: CTCDecoderModel.Config = self.config
        live_mask = 1 - paddings[:, None, :]
        # Drop dummy decode position and mask outputs corresponding to padding frames.
        sequences = sequences[..., :-1] * live_mask
        # If given per-token scores, sum non-padding scores along sequence dim.
        if scores.ndim == 3:
            scores = (scores[..., :-1] * live_mask).sum(axis=-1)
        # Remove repeats and blanks.
        outputs = _map_label_sequences(inputs=sequences, blank_id=cfg.blank_token_id, pad_id=0)
        return DecodeOutputs(
            raw_sequences=sequences,
            sequences=outputs["sequences"],
            paddings=outputs["paddings"],
            scores=scores,
        )


def _map_label_sequences(inputs: Tensor, *, blank_id: int = 0, pad_id: int = 0) -> Nested[Tensor]:
    """Removes blanks, paddings, and repeats from the input sequences, as seen in CTC.

    Args:
        inputs: An int Tensor of shape [..., max_decode_len] containing decoded sequences.
        blank_id: Token ID corresponding to blanks.
        pad_id: Token ID corresponding to paddings.

    Returns:
        A dict containing:
            sequences: A Tensor of shape [..., max_decode_len] containing label sequences.
            paddings: A 0/1 Tensor of shape [..., max_decode_len]. 1's represent paddings.
            lengths: A Tensor of shape [..., 1] containing the length of each sequence.
    """
    max_decode_len = inputs.shape[-1]

    # Identify points at which curr != prev, excluding blanks and paddings.
    # `indicators` will have shape [batch_size, num_decodes, max_decode_len], and have a value
    # of 1 in positions corresponding to inputs we intend to keep (i.e., the token is not blank
    # or padding, and is different from the previous token).
    y = jnp.concatenate([jnp.full(inputs.shape[:-1] + (1,), pad_id), inputs], axis=-1)
    indicators = (y[..., 1:] != y[..., :-1]) & (inputs != blank_id) & (inputs != pad_id)

    # Compute lengths of final sequences. [..., 1].
    lens = jnp.sum(indicators, axis=-1, keepdims=True, dtype=inputs.dtype)

    # Compute sequences by left-justifying the tokens-to-keep. Under jit, we use a dispatch matrix
    # of shape [batch_size, num_decodes, max_decode_len, max_decode_len]. dispatch[..., i, j] == 1
    # means token i goes to position j. dispatch[..., i, :] == 0 means we drop token i.
    # [batch_size, num_decodes, max_decode_len, max_decode_len].
    dispatch = jax.nn.one_hot(
        jnp.cumsum(indicators, axis=-1) * indicators - 1, max_decode_len, dtype=inputs.dtype
    )
    sequences = jnp.einsum("...nm,...n->...m", dispatch, inputs)
    paddings = (jnp.arange(max_decode_len) >= lens).astype(inputs.dtype)
    if pad_id != 0:
        sequences = jnp.where(paddings, pad_id, sequences)
    return dict(sequences=sequences, paddings=paddings, lengths=lens)
