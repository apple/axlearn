# Copyright © 2023 Apple Inc.

"""Decoder layers."""

import contextlib
from typing import Callable, Optional, Protocol, Union

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common import logit_modifiers
from axlearn.common.attention import (
    AttentionLogitBiasLayer,
    BaseStackedTransformerLayer,
    CausalAttentionLogitBiasLayer,
    ForwardMode,
    StackedTransformerLayer,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    Configurable,
    InstantiableConfig,
    Required,
    config_class,
    maybe_instantiate,
)
from axlearn.common.decoding import (
    BeamSearchOutputs,
    BrevityPenaltyFn,
    SampleOutputs,
    StopDecodingCondition,
    StopOnSubsequence,
    beam_search_decode,
    infer_initial_time_step,
    sample_decode,
)
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import Dropout, LayerNorm, set_dropout_rate_recursively
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.module import (
    Module,
    Tensor,
    child_context,
    current_context,
    new_output_collection,
)
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    TensorSpec,
    validate_contains_paths,
    with_sharding_constraint,
)


# TODO(markblee): Remove this when we have a better solution at the decoding loop level.
@contextlib.contextmanager
def _temporary_output_collection():
    """Overrides the output collection without introducing a child context.

    We avoid introducing a child context to ensure that RedirectionLayer still works.
    """
    ctx = current_context()
    old_output_collection = ctx.output_collection  # pytype: disable=attribute-error
    tmp_output_collection = new_output_collection()
    ctx.output_collection = tmp_output_collection  # pytype: disable=attribute-error
    yield
    ctx.output_collection = old_output_collection
    # Discard summaries e.g. from `extend_step`.
    del tmp_output_collection


def _segment_ids_from_causal_input_ids(input_ids: Tensor, *, pad_token_id: int) -> Tensor:
    """Computes segment_ids from inputs.

    All tokens are treated as part of the same segment, except for trailing padding tokens.
    This matches the behavior of assuming positions are arange(seq_len) when positions are None.

    Example:
        input_ids: [
            [1,0,2,3,0,0],
            [1,2,3,4,0,5],
            [0,0,0,0,0,0],
        ]
        segment_ids: [
            [1,1,1,1,0,0],
            [1,1,1,1,1,1],
            [0,0,0,0,0,0],
        ]

    Args:
        input_ids: A Tensor of shape [..., seq_len].
        pad_token_id: An integer.

    Returns:
        A Tensor of shape [..., seq_len] with values in [0,1].
    """
    non_pad_indicator = (input_ids != pad_token_id).astype(input_ids.dtype)
    # Note: jax.lax.cummax doesn't support axis=-1.
    return jax.lax.cummax(non_pad_indicator, axis=input_ids.ndim - 1, reverse=True)


def _scores_from_logits(
    logits: Tensor, logits_modifier: Optional[LogitsToLogitsFn] = None
) -> Tensor:
    """Produces decoding scores from logits and optional logit modifier."""
    if logits.dtype in (jnp.bfloat16, jnp.float16):
        # Cast for log softmax.
        logits = logits.astype(jnp.float32)

    log_probs = jax.nn.log_softmax(logits)
    if logits_modifier is not None:
        log_probs = logits_modifier(log_probs)

    return log_probs


# NOTE: We use a Protocol instead of defining a base layer so that decoder implementations can
# inherit from other base classes without resorting to multiple inheritance.
class BaseDecoder(Protocol):
    """Defines the interface that a Decoder must follow for compatibility with decoding."""

    @property
    def prng_key(self) -> Tensor:
        """A PRNG key for sampling."""

    # TODO(markblee): Remove this in favor of prefill_states.
    def init_states(self, *, batch_size: int, max_sequence_length: int) -> Nested[Tensor]:
        """Initializes cache for autoregressive cached decoding.

        Args:
            batch_size: The batch size of the target to be decoded.
            max_sequence_length: The sequence length of the target to be decoded.

        Returns:
            The cache as a Nested Tensor with key and value initialized.
        """

    def prefill_states(
        self, *, time_step: Tensor, input_batch: Nested[Tensor], **kwargs
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """Initializes cache for autoregressive cached decoding.

        TODO(markblee): Rename to init_states once we add support for decoding at non-zero time
        step.

        Args:
            time_step: A Tensor of shape [batch]. Each value is an index into the length dimension
                indicating where decoding will start from. If `time_step` exceeds `target_length`,
                reads consume the last token in the sequence, and writes are no-ops.
            input_batch: A nested Tensor. See corresponding implementation for details.
            kwargs: Additional kwargs for prefilling.

        Returns:
            A Nested Tensor, which can be used as `cached_states` for the initial call of
            `extend_step()`.
            A Nested Tensor representing outputs for the given inputs. `output['logits']` will have
            shape [batch, vocab_size].
        """

    def extend_step(
        self, *, cached_states: Nested[Tensor], input_ids: Tensor, **kwargs
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """Computes incremental outputs during autoregressive decoding.

        Args:
            cached_states: A Nested Tensor returned by `prefill_states()` or a previous invocation
                of `extend_step()`.
            input_ids: An int Tensor of shape [batch, target_step_length], where
                `target_step_length` is 1.
            kwargs: Additional kwargs for incremental decoding.

        Returns:
            (updated_cached_states, output), where:
            `updated_cached_states` represents the new cached states incorporating `input_ids`;
            `output` represents the output for the given input data. `output['logits']` will
            have shape [batch, vocab_size].
        """


class DecodingLayer(Configurable):
    """Base decoding API."""

    @config_class
    class Config(Configurable.Config):
        """Configures DecodingLayer."""

        pad_token_id: Required[int] = REQUIRED
        eos_token_id: Required[int] = REQUIRED

    def __init__(self, cfg, *, decoder: BaseDecoder):
        super().__init__(cfg)
        self._decoder = decoder

    def beam_search_decode(
        self,
        *,
        input_batch: Nested[Tensor],
        max_sequence_length: int,
        num_decodes: int,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        brevity_penalty: Optional[BrevityPenaltyFn] = None,
    ) -> BeamSearchOutputs:
        """Perform beam search decoding.

        Args:
            input_batch: A dict containing:
                prefix: The prefix to use for prompting of shape [batch, max_prefix_length].
                    The prefix for each example in the batch should begin with a prompt token (e.g.
                    BOS).
                    The prefix will be padded with `cfg.pad_token_id` to `max_sequence_length`, thus
                    it is expected that `max_prefix_length <= max_sequence_length`.
            max_sequence_length: The maximum sequence length of tokens to generate.
            num_decodes: The number of decoded sequences to return. These are the number of
                hypotheses per batch example.
            cross_attention_data: A float Tensor of shape [batch_size, source_len, hidden_dim].
            cross_attention_logit_biases: A Tensor of shape [batch_size, target_len, source_len].
                A -inf represents a disconnected position pair.
                `target_len` should be broadcastable to `max_sequence_length`.
            brevity_penalty: Brevity penalty function for length normalization during beam search.

        Returns:
            The beam search outputs.

        Raises:
            ValueError: If pad_token_id is non-zero.
        """
        validate_contains_paths(input_batch, paths=["prefix"])
        prefix = input_batch["prefix"]

        cfg: DecodingLayer.Config = self.config
        tokens_to_scores_fn = self._tokens_to_scores(
            num_decodes=num_decodes,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            logits_modifier=None,
        )
        input_ids = self._pad(
            prefix, max_sequence_length=max_sequence_length, pad_id=cfg.pad_token_id
        )
        time_step = infer_initial_time_step(prefix, pad_id=cfg.pad_token_id)
        prefill_batch = {**input_batch}
        prefill_batch["input_ids"] = input_ids
        init_states, _ = self._decoder.prefill_states(
            time_step=time_step,
            input_batch=prefill_batch,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        return beam_search_decode(
            inputs=input_ids,
            time_step=time_step,
            cache=init_states,
            tokens_to_scores=tokens_to_scores_fn,
            eos_id=cfg.eos_token_id,
            num_decodes=num_decodes,
            brevity_penalty=brevity_penalty,
            pad_id=cfg.pad_token_id,
        )

    def sample_decode(
        self,
        *,
        input_batch: Nested[Tensor],
        max_sequence_length: int,
        num_decodes: int,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        logits_modifier: Optional[ConfigOr[LogitsToLogitsFn]] = None,
        stop_decoding_condition: Optional[StopDecodingCondition] = None,
    ) -> SampleOutputs:
        """Perform sample-based decoding.

        Args:
            input_batch: A dict containing:
                prefix: The prefix to use for prompting of shape [batch, max_prefix_length].
                    The prefix for each example in the batch should begin with a prompt token (e.g.
                    BOS).
                    The prefix will be padded with `cfg.pad_token_id` to `max_sequence_length`, thus
                    it is expected that `max_prefix_length <= max_sequence_length`.
            max_sequence_length: The maximum sequence length of tokens to generate.
            num_decodes: The number of decoded sequences to return.
                These are the number of hypotheses per batch example.
            cross_attention_data: A float Tensor of shape [batch_size, source_len, hidden_dim].
            cross_attention_logit_biases: A Tensor of shape [batch_size, target_len, source_len].
                A -inf represents a disconnected position pair.
                `target_len` should be broadcastable to `max_sequence_length`.
            logits_modifier: Function used to adjust the raw next-token logit distribution values,
                to e.g. implement top-k/top-p/etc sampling (see `logit_modifiers`).
                If None, do not modify the logits.
            stop_decoding_condition: StopDecodingCondition callable indicating if generation should
                stop. If None, stop on EOS.

        Returns:
            The sample decoding outputs.
        """
        validate_contains_paths(input_batch, paths=["prefix"])
        prefix = input_batch["prefix"]

        cfg: DecodingLayer.Config = self.config
        logits_modifier = maybe_instantiate(logits_modifier)
        tokens_to_scores_fn = self._tokens_to_scores(
            num_decodes=num_decodes,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            logits_modifier=logits_modifier,
        )
        input_ids = self._pad(
            prefix, max_sequence_length=max_sequence_length, pad_id=cfg.pad_token_id
        )
        time_step = infer_initial_time_step(prefix, pad_id=cfg.pad_token_id)
        prefill_batch = {**input_batch}
        prefill_batch["input_ids"] = input_ids
        init_states, init_outputs = self._decoder.prefill_states(
            time_step=time_step,
            input_batch=prefill_batch,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        init_scores = _scores_from_logits(init_outputs["logits"], logits_modifier=logits_modifier)
        # Extract scores corresponding to prefix tokens. Since each sequence in input_ids starts
        # with the [BOS] token, shift them so they line up with the scores of the output tokens.
        score_indices = jnp.roll(input_ids[:, :, None], shift=-1)
        # [batch_size, seq_len, vocab_size] --> [batch_size, seq_len].
        init_scores = jnp.squeeze(jnp.take_along_axis(init_scores, score_indices, axis=-1), axis=-1)
        return sample_decode(
            inputs=input_ids,
            time_step=time_step,
            cache=init_states,
            tokens_to_scores=tokens_to_scores_fn,
            stop_decoding_condition=(
                stop_decoding_condition or StopOnSubsequence([[cfg.eos_token_id]])
            ),
            num_decodes=num_decodes,
            prng_key=self._decoder.prng_key,
            pad_id=cfg.pad_token_id,
            input_token_scores=init_scores,
        )

    def _tokens_to_scores(
        self,
        *,
        num_decodes: int,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        logits_modifier: Optional[LogitsToLogitsFn] = None,
    ) -> Callable[[Tensor, NestedTensor], tuple[Tensor, NestedTensor]]:
        """Build a fn mapping current token IDs and model state to next logits and updated state."""

        # TODO(markblee): Move cross attention data handling to Decoder.
        if cross_attention_data is not None:
            # Shape [batch*num_decodes, source_len, hidden_dim].
            cross_attention_data = jnp.repeat(cross_attention_data, num_decodes, axis=0)
        if cross_attention_logit_biases is not None:
            # Shape [batch*num_decodes, target_len, source_len].
            cross_attention_logit_biases = jnp.repeat(
                cross_attention_logit_biases, num_decodes, axis=0
            )
            # Expand to [batch*num_decodes, num_heads=1, target_len, source_len].
            if cross_attention_logit_biases.ndim == 3:
                cross_attention_logit_biases = cross_attention_logit_biases[:, None, ...]

        def tokens_to_scores(token_ids: Tensor, cache: NestedTensor) -> tuple[Tensor, NestedTensor]:
            """Maps current token IDs and model state to next logits and updated state.

            Args:
                token_ids: An int Tensor of shape [batch*num_decodes, 1].
                cache: A NestedTensor of cached states.

            Returns:
                (log_probs, updated_cache), where log_probs has shape
                [token_ids.shape[0], vocab_size] and represents log probabilities of the next
                tokens; and updated_cache is the updated cache.
            """
            time_step = cache["time_step"]
            assert time_step.ndim == 1

            # Select attention biases corresponding to the current time steps.
            # We alias the nonlocal variable.
            cross_attention_biases = cross_attention_logit_biases
            if cross_attention_biases is not None:
                # Note: the target_len dimension can be 1 during decoding.
                # When indexing, we clip the indices instead of producing NaNs.
                # TODO(markblee): Consider removing `take_along_axis` entirely if we restrict
                # target_len to always be 1 during decoding.
                # [batch*num_decodes, num_heads, 1, source_len].
                cross_attention_biases = jnp.take_along_axis(
                    cross_attention_biases, time_step[:, None, None, None], mode="clip", axis=2
                )

            # Use a temporary output collection to avoid tracer leaks during extend_step.
            with _temporary_output_collection():
                updated_state, outputs = self._decoder.extend_step(
                    cached_states=cache,
                    input_ids=token_ids,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_biases,
                )

            logits = outputs["logits"]
            log_probs = _scores_from_logits(logits[:, -1, :], logits_modifier=logits_modifier)
            return log_probs, updated_state

        return tokens_to_scores

    @staticmethod
    def _pad(prefix: Tensor, *, max_sequence_length: int, pad_id: int) -> Tensor:
        """Accept token IDs input tensor and pad if necessary to max_sequence_length."""
        return jnp.concatenate(
            [
                prefix,
                jnp.full(
                    (prefix.shape[0], max_sequence_length - prefix.shape[1]),
                    pad_id,
                    dtype=prefix.dtype,
                ),
            ],
            axis=1,
        )


# TODO(gyin): Add unittest for Decoder forward
class Decoder(BaseLayer):
    """Construct a decoder transformer to output hidden states and logits based on lm head."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures Decoder."""

        # attention_mask can be None if the attention layer supports the causal mode, e.g.,
        # FlashAttention with `causal=True`.
        attention_mask: Optional[
            AttentionLogitBiasLayer.Config
        ] = CausalAttentionLogitBiasLayer.default_config()
        vocab_size: Required[int] = REQUIRED  # Size of vocabulary.
        # Dimensionality of embeddings and inputs to each transformer layer.
        dim: Required[int] = REQUIRED
        # Dropout rate applied throughout model, except for child Dropout configs with rate set
        # explicitly.
        dropout_rate: float = 0.0
        # Vector from input ids table.
        emb: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config()
        # Transformer model trunk.
        transformer: BaseStackedTransformerLayer.Config = StackedTransformerLayer.default_config()
        # Layer norm applied to transformer output.
        output_norm: Optional[InstantiableConfig] = LayerNorm.default_config()
        # Optional dropout rate for the transformer output.
        # If output_dropout.rate is None, it will default to cfg.dropout_rate
        output_dropout: Dropout.Config = Dropout.default_config()
        # Optional LmHead layer maps the hidden state to vocab logits (if None use emb.token_emb).
        lm_head: Optional[InstantiableConfig] = None
        pad_token_id: int = 0  # Int ID of the inputs to be masked for self-attention.
        eos_token_id: int = 1  # Int ID of the end of sequence token id.
        # Specifies how to partition the output logits of shape [batch, max_seq_len, vocab_size].
        logits_partition_spec: tuple[Union[Optional[str], tuple[Optional[str]]], ...] = (
            "data",
            None,
            "model",
        )
        # The logit modifier to apply. If None, does not modify logits.
        output_logits_modifier: Optional[ConfigOr[logit_modifiers.LogitsToLogitsFn]] = None
        # The decoding implementation.
        decoding: DecodingLayer.Config = DecodingLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        set_dropout_rate_recursively(cfg, dropout_rate=cfg.dropout_rate, set_only_if_none=True)

        if cfg.attention_mask is not None:
            self._add_child("attention_mask", cfg.attention_mask)
        self._add_child("emb", cfg.emb.set(dim=cfg.dim, vocab_size=cfg.vocab_size))
        self._add_child("transformer", cfg.transformer.set(input_dim=cfg.dim))
        if cfg.output_norm is not None:
            self._add_child("output_norm", cfg.output_norm.set(input_dim=cfg.dim))
        self._add_child("output_dropout", cfg.output_dropout)
        if cfg.lm_head is not None:  # pytype: disable=attribute-error
            self._add_child(
                "lm_head", cfg.lm_head.set(vocab_size=cfg.vocab_size, embedding_dim=cfg.dim)
            )
        self._output_logits_modifier = maybe_instantiate(cfg.output_logits_modifier)
        self._decoding: DecodingLayer = cfg.decoding.set(
            pad_token_id=cfg.pad_token_id, eos_token_id=cfg.eos_token_id
        ).instantiate(decoder=self)

    def _forward_for_mode(
        self,
        *,
        mode: ForwardMode,
        input_batch: Nested[Tensor],
        self_attention_logit_biases: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        cached_states: Optional[NestedTensor] = None,
    ) -> tuple[Optional[NestedTensor], Tensor]:
        validate_contains_paths(input_batch, paths=["input_ids"])
        input_segment_ids = input_batch.get("input_segment_ids", None)
        positions = input_batch.get("positions", None)

        emb_batch = {**input_batch}
        emb_batch["inputs"] = emb_batch["input_ids"]
        x = self.emb(input_batch=emb_batch)

        if mode == ForwardMode.FORWARD:
            transformer_state, x = (
                None,
                self.transformer(
                    x,
                    self_attention_logit_biases=self_attention_logit_biases,
                    target_segment_ids=input_segment_ids,
                    target_positions=positions,
                    cross_attention_data=cross_attention_data,
                    cross_attention_logit_biases=cross_attention_logit_biases,
                ),
            )
        elif mode == ForwardMode.INIT_STATES:
            assert cached_states is not None
            if input_segment_ids is not None:
                raise ValueError("input_segment_ids is not supported in INIT_STATES.")
            transformer_state, x = self.transformer.init_states(
                time_step=cached_states["transformer_state"],
                data=x,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
            )
        elif mode == ForwardMode.EXTEND_STEP:
            assert cached_states is not None
            if input_segment_ids is not None:
                raise ValueError("input_segment_ids is not supported in EXTEND_STEP.")
            transformer_state, x = self.transformer.extend_step(
                cached_states=cached_states["transformer_state"],
                data=x,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=cross_attention_data,
                cross_attention_logit_biases=cross_attention_logit_biases,
            )
        else:
            raise ValueError(f"Unrecognized mode {mode}.")
        x = x.data
        self._add_tensor_stats("outputs", x)

        if "output_norm" in self.children:
            x = self.output_norm(x)
            self._add_tensor_stats("norm_outputs", x)
        x = self.output_dropout(x)
        if "lm_head" in self.children:
            logits = self.lm_head(x)
        else:
            # Reuse the token embedding.
            with child_context("emb_attend", module=self.emb):
                logits = self.emb.attend(x)
        if self._output_logits_modifier is not None:
            logits = self._output_logits_modifier(logits)
        logits = with_sharding_constraint(logits, PartitionSpec(*self.config.logits_partition_spec))
        # TODO(markblee): Rename to just "transformer". "transformer_state" is a bit redundant.
        return dict(transformer_state=transformer_state), dict(logits=logits, hidden_states=x)

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_logit_biases: Optional[Tensor] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Computes decoder hidden states and logits from input ids and cross attention hidden
        states.

        Args:
            input_batch: A dict containing:
                * input_ids: An int Tensor of shape [batch_size, target_len].
                    Values should be in the range [0, vocab_size).
                * input_segment_ids: An optional Tensor of same shape as `input_ids` with values in
                    [0, num_segments]. Tokens are only allowed to attend to other tokens within the
                    same segment. input_segment_ids == 0 represents paddings. If None, inferred from
                    input_ids != pad_token_id.
                * token_type_ids: An optional int Tensor of shape [batch_size, target_len].
                    Values should be in the range [0, type_vocab_size).
                * positions: An optional int Tensor of shape [batch_size, target_len].
                    If None, assumed to be jnp.arange(target_len) for each sequence.
            cross_attention_data: A float Tensor of shape [batch_size, source_len, hidden_dim].
            cross_attention_logit_biases: A Tensor of shape [batch_size, target_len, source_len].
                A -inf represents a disconnected position pair.

        Returns:
            A dict containing:
                hidden_states: A float Tensor of shape [batch_size, target_len, hidden_dim].
                logits: A float Tensor of shape [batch_size, target_len, num_classes], where
                    num_classes depends on the configured lm_head.
        """
        validate_contains_paths(input_batch, paths=["input_ids"])
        input_ids = input_batch["input_ids"]
        input_segment_ids = input_batch.get("input_segment_ids", None)
        positions = input_batch.get("positions", None)

        _, output = self._forward_for_mode(
            mode=ForwardMode.FORWARD,
            input_batch=input_batch,
            # [batch_size, num_heads, seq_len, seq_len].
            self_attention_logit_biases=self.compute_attention_logit_biases(
                input_ids, segment_ids=input_segment_ids, positions=positions
            ),
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
            cached_states=None,
            **kwargs,
        )
        return output

    def init_states(self, *, batch_size: int, max_sequence_length: int) -> NestedTensor:
        """See `BaseDecoder.init_states` for details."""
        cfg: Decoder.Config = self.config
        init_state, _ = self.transformer.init_states(
            time_step=None,
            data=TensorSpec([batch_size, max_sequence_length, cfg.dim], dtype=jnp.float32),
        )
        return dict(
            transformer_state=init_state,
            input_ids=jnp.full(
                (batch_size, max_sequence_length), cfg.pad_token_id, dtype=jnp.int32
            ),
            time_step=jnp.zeros(batch_size, dtype=jnp.int32),
        )

    def prefill_states(
        self,
        *,
        time_step: Tensor,
        input_batch: Nested[Tensor],
        **kwargs,
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """See `BaseDecoder.prefill_states` for details.

        Args:
            time_step: A Tensor of shape [batch_size]. See `BaseDecoder.prefill_states` for details.
            input_batch: See `forward` for details.
            kwargs: See `forward` for details.

        Returns:
            See `BaseDecoder.prefill_states` for details.
        """
        validate_contains_paths(input_batch, paths=["input_ids"])
        input_ids = input_batch["input_ids"]
        input_segment_ids = input_batch.get("input_segment_ids", None)
        positions = input_batch.get("positions", None)

        states, outputs = self._forward_for_mode(
            mode=ForwardMode.INIT_STATES,
            cached_states=dict(transformer_state=time_step),
            input_batch=input_batch,
            # TODO(markblee): Consider supporting packed inputs for more efficient prefilling.
            self_attention_logit_biases=self.compute_attention_logit_biases(
                input_ids, segment_ids=input_segment_ids, positions=positions
            ),
            **kwargs,
        )
        self.add_module_output("prefill_hidden_states", outputs["hidden_states"])
        states = dict(time_step=time_step, input_ids=input_ids, **states)
        return states, outputs

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        input_ids: Tensor,
        **kwargs,
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        """See `BaseDecoder.extend_step` for details."""
        time_step: Tensor = cached_states["time_step"]
        assert time_step.ndim == 1

        # Update cached input_ids via "scatter via one-hot broadcast" trick.
        # Note: in the cases where `time_step` exceeds `target_len`, the update becomes a no-op.
        # --> [B, T].
        cached_inputs: Tensor = cached_states["input_ids"]
        target_len = cached_inputs.shape[-1]
        oh_indices = jax.nn.one_hot(time_step, target_len, dtype=input_ids.dtype)
        updated_inputs = cached_inputs * (1 - oh_indices) + input_ids * oh_indices

        # Compute self-attention-mask logit biases. [B, N, T, T].
        self_attention_biases = self.compute_attention_logit_biases(
            updated_inputs,
            segment_ids=jnp.ones_like(updated_inputs),
            positions=jnp.arange(target_len)[None, :],
        )
        # Select logit biases corresponding to time step. [B, N, 1, T].
        # Note: if `time_step` exceeds `target_len`, e.g. in the case where one decode starts at a
        # later index than another, clip the indices instead of producing NaNs.
        # TODO(markblee): Update attention masks to support explicit positions, so we can skip this.
        if self_attention_biases is not None:
            self_attention_biases = jnp.take_along_axis(
                self_attention_biases,
                time_step[:, None, None, None],
                axis=2,
                mode="clip",
            )

        input_segment_ids = kwargs.pop("input_segment_ids", None)
        token_type_ids = kwargs.pop("token_type_ids", None)
        positions = kwargs.pop("positions", jnp.expand_dims(time_step, 1))

        updated_states, outputs = self._forward_for_mode(
            mode=ForwardMode.EXTEND_STEP,
            input_batch=dict(
                input_ids=input_ids,
                input_segment_ids=input_segment_ids,
                token_type_ids=token_type_ids,
                positions=positions,
            ),
            self_attention_logit_biases=self_attention_biases,
            cached_states=cached_states,
            **kwargs,
        )
        updated_states = dict(
            input_ids=updated_inputs,
            # There are some non-greedy DFS/BFS and sliding attention algorithms that
            # recursively search through potentials.
            # They backtrace to some anchor time step after exploring for t steps.
            # This requires tracking time_step separately from the attention time_step.
            time_step=cached_states["time_step"] + 1,
            **updated_states,
        )
        return updated_states, outputs

    def beam_search_decode(
        self,
        *,
        input_batch: Nested[Tensor],
        max_sequence_length: int,
        num_decodes: int,
        **kwargs,
    ):
        """See configured `decoding` implementation for details."""
        return self._decoding.beam_search_decode(
            input_batch=input_batch,
            max_sequence_length=max_sequence_length,
            num_decodes=num_decodes,
            **kwargs,
        )

    def sample_decode(
        self,
        *,
        input_batch: Nested[Tensor],
        max_sequence_length: int,
        num_decodes: int,
        **kwargs,
    ):
        """See configured `decoding` implementation for details."""
        return self._decoding.sample_decode(
            input_batch=input_batch,
            max_sequence_length=max_sequence_length,
            num_decodes=num_decodes,
            **kwargs,
        )

    def compute_attention_logit_biases(
        self,
        input_ids: Tensor,
        *,
        segment_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Produces self-attention logit biases.

        Args:
            input_ids: A Tensor of shape [batch_size, seq_len].
            segment_ids: An optional integer Tensor of shape [batch_size, seq_len] with values in
                [0, num_segments). Tokens are only allowed to attend to other tokens within the same
                segment. segment_ids == 0 represents paddings. If provided, positions should also be
                provided.
            positions: An optional Tensor of same shape as `input_ids` with values in [0, seq_len).
                This can be used to produce biases for packed inputs. If None, assumed to be
                jnp.arange(target_len) for each sequence. If provided, segment_ids should also be
                provided.

        Returns:
            Attention logit biases of shape [batch_size, num_heads, seq_len, seq_len],
            or None if cfg.attention_mask is None.

        """
        if "attention_mask" not in self.children:
            return None
        cfg = self.config
        if segment_ids is None:
            segment_ids = _segment_ids_from_causal_input_ids(
                input_ids, pad_token_id=cfg.pad_token_id
            )
        if positions is None:
            positions = jnp.arange(input_ids.shape[-1])[None, :]  # [batch=1, seq_len].
        return self.attention_mask(segment_ids=segment_ids, positions=positions)


class LmHead(BaseLayer):
    """LM head layer for decoder to compute logits."""

    @config_class
    class Config(BaseLayer.Config):
        vocab_size: Required[int] = REQUIRED  # Size of the LM vocabulary.
        # TODO(markblee): Rename this to input_dim.
        embedding_dim: Required[int] = REQUIRED  # Dimensionality of vocabulary embedding table.

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        # Some similarity with Embedding.
        # pylint: disable=duplicate-code
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=(cfg.vocab_size, cfg.embedding_dim),
                mesh_axes=cfg.param_partition_spec,
            )
        )
        # pylint: enable=duplicate-code

    def forward(self, x: Tensor) -> Tensor:
        """Computes logits with token embedding.

        Args:
            x: An int Tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            A float Tensor of shape [batch_size, seq_len, vocab_size].
        """
        return jnp.einsum("bsh,vh->bsv", x, self.parameters["weight"])
