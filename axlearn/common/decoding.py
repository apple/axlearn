# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/t5x:
# Copyright 2023 The T5X Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# huggingface/transformers:
# Copyright 2020 The HuggingFace Team Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Beam search for inference from a trained model.

Reference: https://github.com/google-research/t5x/blob/main/t5x/decoding.py

The key differences from the reference impl are:
    1) Gets rid of the cache_* functions that handled the cache PyTree.
       This is now handled by the common _gather_* functions through the use of NestedTensor.
    2) Cache offset (stacked states of scan-based layers) is supported through vectorized_tree_map.
"""
from collections.abc import Sequence

# pylint: disable=too-many-lines
from typing import Callable, Literal, NamedTuple, Optional, Protocol, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from axlearn.common import flax_struct
from axlearn.common.utils import NestedTensor, Tensor, vectorized_tree_map

# Constants
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = np.array(-1.0e7)


# Utility functions:


def add_decoding_dim(x: Tensor, num_decodes: int) -> Tensor:
    """Creates num_decodes as second dimension in non-scalar array x and tiles into it."""
    if x.ndim > 0:
        x = jnp.expand_dims(x, axis=1)
        tile_dims = [1] * x.ndim
        tile_dims[1] = num_decodes
        return jnp.tile(x, tile_dims)

    # Scalar values get converted to tensor on jit.
    # However, we do not want to handle num_decodes dimension for scalars.
    return x


def flatten_decoding_dim(x: Tensor) -> Tensor:
    """Flattens the first two dimensions of a non-scalar array."""
    if x.ndim > 0:
        return x.reshape((-1,) + x.shape[2:])

    # Scalar values get converted to tensor on jit.
    # However, we do not want to handle num_decodes dimension for scalars.
    return x


def unflatten_decoding_dim(x: Tensor, batch_size: int, num_decodes: int) -> Tensor:
    """Unflattens the first, flat batch*decoding dimension of a non-scalar array."""
    if x.ndim > 0:
        assert batch_size * num_decodes == x.shape[0], x.shape
        return x.reshape((batch_size, num_decodes) + x.shape[1:])

    # Scalar values get converted to tensor on jit.
    # However, we do not want to handle num_decodes dimension for scalars.
    return x


def _gather_beams(
    nested: NestedTensor,
    beam_indices: Tensor,
    batch_size: int,
    old_beam_size: int,
    new_beam_size: int,
    one_hot: bool = True,
) -> Tensor:
    """Gathers the beam slices indexed by beam_indices into new beam array.

    Args:
        nested: NestedTensor or scalars (the latter ignored).
        beam_indices: Array of beam_indices.
        batch_size: Size of batch.
        old_beam_size: Size of _old_ beam dimension.
        new_beam_size: Size of _new_ beam dimension.
        one_hot: Whether to perform gathers by one-hot contraction or directly.

    Returns:
        New beam arrays
        [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...].

    TODO(sneha): The method does a likely O(n) copy to track history,
          could use a tree / single parent pointer instead.
          Check with timeit after.
    """
    if one_hot:
        # Gather via one-hot contraction, needed for SPMD partitioning.
        oh_beam_indices = jax.nn.one_hot(beam_indices, old_beam_size, dtype=jnp.int32)

        def gather_fn(x):
            return jnp.einsum("bno,bo...->bn...", oh_beam_indices, x).astype(x.dtype)

    else:
        # True gather via fancy indexing.
        batch_indices = jnp.reshape(
            jnp.arange(batch_size * new_beam_size) // new_beam_size, (batch_size, new_beam_size)
        )

        def gather_fn(x):
            return x[batch_indices, beam_indices]

    return vectorized_tree_map(lambda x: gather_fn(x) if x.ndim > 1 else x, nested)


def _top_k_two_stage(x: Tensor, k: int):
    """Wrapper around lax.top_k with low-batch optimization.
    It *increases* batch and *decreases* vocab.

    Args:
        x: Shape f32[batch, num_samples].
        k: Indicates how many top values to return.

    Returns:
        Largest k values and indices with shape (f32[batch, k], s32[batch, k]).
    """
    batch, num_samples = x.shape
    num_lanes = 128
    if isinstance(batch, int) and batch <= 8 and num_samples > 8 * num_lanes * k:
        # At small batch, when num_samples is sufficiently large, optimize
        # execution on TPU by doing TopK in two stages. Reshaping 'x' to fill
        # lanes reduces tensor padding in TopK call.
        if num_samples % num_lanes != 0:
            # Pad input tensor to multiples of num_lanes.
            num_samples_rounded_up = num_samples + (num_lanes - num_samples % num_lanes)
            x = jnp.pad(
                x,
                ((0, 0), (0, num_samples_rounded_up - num_samples)),
                mode="constant",
                constant_values=np.NINF,
            )  # why not NEG_INF?
            num_samples = num_samples_rounded_up
        # Reshape input tensor to fill lanes.
        num_samples_sublanes = int(num_samples / num_lanes)
        x_reshaped = jnp.reshape(x, (batch * num_lanes, num_samples_sublanes))
        # First stage top_k.
        vals, indices = jax.lax.top_k(x_reshaped, k)
        indices = jnp.reshape(indices, (batch, num_lanes, k))
        index_offsets = jnp.reshape(num_samples_sublanes * jnp.arange(num_lanes), (1, num_lanes, 1))
        indices = jnp.reshape(jnp.add(index_offsets, indices), (batch, num_lanes * k))
        vals = jnp.reshape(vals, (batch, num_lanes * k))
        # Second stage top_k.
        vals_s2, indices_s2 = jax.lax.top_k(vals, k)
        indices_s2 = jnp.take_along_axis(indices, indices_s2, axis=1)
        return vals_s2, indices_s2
    else:
        # Use default TopK implementation.
        return jax.lax.top_k(x, k)


def _gather_topk_beams(
    nested: NestedTensor, score_or_log_prob: Tensor, batch_size: int, new_beam_size: int
) -> Tensor:
    """Gathers the top-k beam slices given by score_or_log_prob array.

    Args:
        nested: NestedTensor or scalars (the latter ignored).
        score_or_log_prob: [batch_size, old_beam_size] array of values to sort by
            for top-k selection of beam slices.
        batch_size: Size of batch.
        new_beam_size: Size of _new_ top-k selected beam dimension.

    Returns:
        New beam arrays containing top k new_beam_size slices
        [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...].
    """
    _, topk_indices = jax.lax.top_k(score_or_log_prob, k=new_beam_size)
    return _gather_beams(
        nested, topk_indices, batch_size, score_or_log_prob.shape[1], new_beam_size
    )


class BrevityPenaltyFn(Protocol):
    def __call__(self, *, length: Tensor, raw_scores: Tensor) -> Tensor:
        """Compute the brevity penalty based on the length of a decoding and its raw scores.

        Args:
            length: A tensor of shape broadcastable to [batch_size, beam_size].
            raw_scores: Raw beam search scores.

        Returns:
            Normalized scores of same shape as `raw_scores`.
        """


def brevity_penalty_fn(
    *,
    alpha: float = 0.0,
    bp_type: Literal["t5", "hf"] = "t5",
) -> BrevityPenaltyFn:
    """Brevity penalty function to do length normalization during beam search.

    If alpha is zero, we do not apply length normaliation.

    T5 reference:
    https://github.com/google-research/t5x/blob/main/t5x/decoding.py#L656

    HF reference:
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation_beam_search.py#L872

    Args:
        alpha: Brevity penalty scaling parameter.
        bp_type: The way to compute the bevity penalty.

    Returns:
        A penalty function for brevity.
    """

    def fn(*, length: Tensor, raw_scores: Tensor) -> Tensor:
        if bp_type == "t5":
            bp = jnp.power(((5.0 + length) / 6.0), alpha)
        elif bp_type == "hf":
            bp = jnp.power(length, alpha)
        else:
            raise NotImplementedError(
                f"{bp_type} brevity penalty is not yet supported. Supported types are [t5, hf]."
            )
        return raw_scores / bp

    return fn


def compute_merge_matrix_by_prefix_ids(prefix_ids: Tensor) -> Tensor:
    """Computes a merge matrix by comparing prefixes to merge equivalent prefixes.

    Args:
        prefix_ids: An int tensor of shape [batch_size, num_decodes, ...], prefix_ids[b, i]
            is equivalent to prefix_ids[b, j] iff prefix_ids[b, i, ...] == prefix_ids[b, j, ...].

    Returns:
        A merge matrix of shape [batch_size, num_decodes, num_decodes]. Please see PrefixMerger
        class comments on "merge matrix".
    """
    prefix_equivalence = prefix_ids[:, :, None, ...] == prefix_ids[:, None, :, ...]
    if prefix_equivalence.ndim > 3:
        prefix_equivalence = jnp.prod(prefix_equivalence, axis=range(3, prefix_equivalence.ndim))
    # prefix_equivalence_mask[b, i, j] = 1 iff there exists k < i s.t.
    # prefix_equivalence[b, k, j] == 1.
    prefix_equivalence_mask = (
        jnp.pad(prefix_equivalence.cumsum(axis=1)[:, :-1, :], ((0, 0), (1, 0), (0, 0))) > 0
    )
    return prefix_equivalence * (1 - prefix_equivalence_mask)


def _merge_prefixes(merge_matrix: Tensor, *, log_probs: Tensor) -> Tensor:
    """Merges prefixes according to `merge_matrix`.

    Args:
        merge_matrix: See PrefixMerger class comments on "merge matrix".
        log_probs: A Tensor of shape [batch_size, num_decodes].

    Returns:
        updated_log_probs, where updated_log_probs[b, to] = logsumexp(
            log_probs[b, from] for all from s.t. merge_matrix[b, to, from] is True
        )
        Specifically, if a prefix is merged into another prefix, the updated log prob will be
        NEG_INF.
    """
    merge_matrix = merge_matrix.astype(log_probs.dtype)
    # transfer_log_probs[b, to, from] =
    #   log_probs[b, from] if merge_matrix[b, to, from] = 1
    #   NEG_INF otherwise.
    transfer_log_probs = merge_matrix * log_probs[:, None, :] + (1 - merge_matrix) * jnp.full_like(
        merge_matrix, NEG_INF
    )
    return jax.nn.logsumexp(transfer_log_probs, axis=-1)


class PrefixMerger:
    """A PrefixMerger computes merge matrix and merges prefixes accordingly.

    A "merge matrix" is a 0/1 tensor of shape [batch_size, num_decodes, num_decodes], where
    merge[b, to, from] = 1 iff we should merge prefix[b, from] into prefix[b, to].

    It satisfies the following properties:
    (1) sum(merge[b, :, from]) == 1, i.e., each prefix is merged into exactly one other prefix.
    (2) If merge[b, to, from] = 1, then merge[b, to, to] = 1, i.e., the destination prefix should
        not merge into another prefix.
    (3) merge[b, to, from] = 1 only if to <= from, i.e.,
        we only merge lower-ranked prefixes into higher-ranked ones.
    """

    def init_state(self, *, tokens: Tensor) -> NestedTensor:
        """Initializes prefix merger state.

        Args:
            tokens: The initial live sequences, of shape [batch_size, num_decodes, max_decode_len].
                When prefilling the decoding cache, this consists of the decoding prefixes padded to
                `max_decode_len`.

        Returns:
            The initial state.
        """
        raise NotImplementedError(type(self))

    def compute(self, state: NestedTensor) -> Tensor:
        """Computes the merge matrix.

        Args:
            state: As returned by `init_state` or `update`. Each tensor must be either a scalar or
                has leading dimensions [batch_size, num_decodes].

        Returns:
            A merge matrix. See class comments.
        """
        raise NotImplementedError(type(self))

    def update(self, *, tokens: Tensor, state: NestedTensor) -> NestedTensor:
        """Updates state.

        Args:
            tokens: An int tensor of shape [batch_size, num_decodes], representing the next token
                to be added to each prefix.
            state: As returned by `init_state` or `merge`. Each tensor must be either a scalar or
                has leading dimensions [batch_size, num_decodes].

        Returns:
            updated_state, which should have the same structure and shapes as `state`.
        """
        raise NotImplementedError(type(self))


class _BeamState(NamedTuple):
    """Holds beam search state data."""

    # The position of the decoding loop in the length dimension.
    cur_index: Tensor  # scalar int32: current decoded length index.
    # The active sequence log probabilities and finished sequence scores.
    live_scores: Tensor  # float32: [batch_size, beam_size].
    finished_scores: Tensor  # float32: [batch_size, beam_size].
    # The current active-beam-searching and finished sequences.
    live_seqs: Tensor  # int32: [batch_size, beam_size, max_decode_len].
    finished_seqs: Tensor  # int32: [batch_size, beam_size, max_decode_len].
    # The current state of the autoregressive decoding caches.
    cache: NestedTensor
    # The prefix merger state.
    prefix_merger: NestedTensor


class BeamSearchOutputs(flax_struct.PyTreeNode):
    """Output values after performing beam search decoding."""

    # Sequences that end with eos_id.
    sequences: Tensor  # int32: [batch_size, num_decodes, max_decode_len].
    # Scores corresponding to sequences above (log probabilities)
    # with length normalization.
    scores: Tensor  # float32: [batch_size, num_decodes].

    # Sequences that have reached max_decode_len without eos_id.
    live_sequences: Tensor  # int32: [batch_size, num_decodes, max_decode_len].
    # Scores corresponding to live_sequences above (log probabilities).
    # No length normalization.
    live_scores: Tensor  # float32: [batch_size, num_decodes].


def _beam_init(
    *,
    inputs: Tensor,
    time_step: Tensor,
    beam_size: int,
    max_decode_len: int,
    cache: NestedTensor,
    pad_id: int,
    prefix_merger: Optional[PrefixMerger] = None,
) -> _BeamState:
    """Initializes the beam search state data structure.

    Args:
        inputs: An int tensor of shape [batch_size, length] where length <= max_decode_length.
        time_step: Initial time steps for decoding of shape [batch_size].
        batch_size: Size of batch.
        beam_size: Number of hypotheses per beam (aka num_decodes).
        max_decode_len: The maximum length of the sequence to be generated.
        cache: State of the decoder model.
        pad_id: Token ID associated with padded input.
        prefix_merger: Optional prefix merger.

    Returns:
        Fully initialized _BeamState.

    Raises:
        ValueError: If inputs has an invalid shape.
    """
    if inputs.shape[0] != time_step.shape[0]:
        raise ValueError(
            f"Expected inputs.shape[0] ({inputs.shape[0]}) "
            f"== time_step.shape[0] ({time_step.shape[0]})."
        )
    if inputs.shape[1] > max_decode_len:
        raise ValueError(
            f"Expected inputs.shape[1] ({inputs.shape[1]}) <= max_decode_len ({max_decode_len})."
        )
    batch_size = inputs.shape[0]
    live_seqs = jnp.full((batch_size, beam_size, max_decode_len), pad_id, dtype=jnp.int32)
    # Inputs are the prefix we will use for teacher forcing.
    live_seqs = live_seqs.at[:, :, : inputs.shape[1]].set(inputs[:, None, :])

    prefix_merger_state = None
    if prefix_merger is not None:
        prefix_merger_state = prefix_merger.init_state(tokens=live_seqs)

    return _BeamState(
        cur_index=time_step,
        # Handle first time step by masking out scores of all but the top hypothesis in the beam.
        live_scores=jnp.tile(jnp.array([0.0] + [NEG_INF] * (beam_size - 1)), [batch_size, 1]),
        finished_scores=jnp.ones((batch_size, beam_size)) * NEG_INF,
        live_seqs=live_seqs,
        finished_seqs=jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32),
        # Expand cache to num_decodes size.
        cache=vectorized_tree_map(lambda x: add_decoding_dim(x, beam_size), cache),
        prefix_merger=prefix_merger_state,
    )


# pylint: disable-next=too-many-statements
def beam_search_decode(
    *,
    inputs: Tensor,
    time_step: Tensor,
    cache: NestedTensor,
    tokens_to_scores: Callable[[Tensor, NestedTensor], tuple[Tensor, NestedTensor]],
    eos_id: int,
    num_decodes: int,
    max_decode_len: Optional[int] = None,
    loop: Literal["lax", "python"] = "lax",
    brevity_penalty: Optional[BrevityPenaltyFn] = None,
    pad_id: int = 0,
    prefix_merger: Optional[PrefixMerger] = None,
) -> BeamSearchOutputs:
    """Performs beam search decoding.

    Args:
        inputs: An int32 Tensor of shape [batch_size, length] containing a sequence of tokens.
        time_step: Initial time steps for decoding of shape [batch_size].
            Time steps are indices into the sequence dimension, and range from [0, length).
            If time_step[i] == 0, it means there is no prefix (we assume inputs[i, 0] == BOS whether
            there's prefix or not).
            Otherwise, inputs[i, 1:time_step[i] + 1] represents the input prefix for sequence i.
            Tokens within this input prefix are not modified. This simulates teacher forcing on the
            prefix positions.
            See also `infer_initial_time_step` for details on how time step is initialized from the
            input.
        cache: State of the decoder model. Each tensor in `cache` must be either a scalar or
            has shape [batch_size, ...]. If `time_step[i] > 0`, the assumption is that the `cache`
            has been prefilled for the prefix positions.
        tokens_to_scores: Fast autoregressive decoder function taking single token slices and cache
            and returning next-token scores and updated cache.
            Note: the callable is expected to return log_probs. Higher the better.
            [batch*beam, vocab], updated_cache = tokens_to_scores([batch*beam, 1], flat_cache).
            Each tensor in `flat_cache` and `updated_cache` is either a scalar or has batch*beam
            as the leading dim.
        eos_id: ID of end-of-sentence token for target vocabulary.
        num_decodes: Number of decoded sequences to be returned. This is equivalent
            to the number of beams used in the beam search.
            Warning: setting num_decodes=1 is not equivalent to greedy search, because our beam
            search algorithm does not require a sequence ending with EOS to have a score in the
            top k among all partial hypotheses in the beam. Instead, an EOS is appended to every
            partial hypothesis in the beam, and in the end the top-k sequences with EOS are
            returned. To match greedy decoding, you can use `sample_decode` with `top_k_logits(k=1)`
            as the logit modifier.
        max_decode_len: An optional maximum length of decoded sequence. If
            None, it uses `inputs.shape[1]` as `max_decode_len`.
        loop: "lax" or "python". The latter should only be used for debugging.
        brevity_penalty: Brevity penalty function to add length normalization in the beam search.
        pad_id: Token ID associated with padded input.
        prefix_merger: An optional PrefixMerger instance. If not None, used to merge prefixes that
            are semantically equivalent. `prefix_merger.update` will be invoked with the same
            sequence of tokens as `tokens_to_scores`, i.e., including the BOS tokens and
            other tokens from `inputs`, but not the EOS tokens.

    Returns:
        BeamSearchOutputs, containing the top-scoring finished and live sequences and their
        corresponding scores (in the descending order). We only apply brevity penalty to the
        finished sequences. Live scores here do not include brevity penalty.

    Raises:
        NotImplementedError: If an unsupported loop is provided.
    """
    # If brevity_penalty is set as None, we explicitly use the default function
    # without length normalization.
    if brevity_penalty is None:
        brevity_penalty = brevity_penalty_fn(alpha=0.0)

    beam_size = num_decodes
    batch_size = inputs.shape[0]
    if max_decode_len is None:
        max_decode_len = inputs.shape[1]
    max_decode_len += 1

    # Initialize beam search state.
    beam_search_init_state = _beam_init(
        inputs=inputs,
        time_step=time_step,
        beam_size=beam_size,
        max_decode_len=max_decode_len,
        cache=cache,
        prefix_merger=prefix_merger,
        pad_id=pad_id,
    )

    def beam_search_loop_cond_fn(state: _BeamState) -> bool:
        """Beam search loop termination condition."""
        # Have we reached max decoding length?
        # Because we mutate the "i+1" position, we stop one token before the end.
        not_at_end = state.cur_index < max_decode_len - 1

        # Is no further progress in the beam search possible?
        # Get the best possible scores from alive sequences. [batch_size, 1].
        best_live_scores = brevity_penalty(
            length=jnp.array(max_decode_len),
            raw_scores=jnp.max(state.live_scores, axis=1, keepdims=True),
        )

        # Get the worst scores from finished sequences. [batch_size, 1].
        worst_finished_scores = jnp.min(state.finished_scores, axis=1, keepdims=True)
        # If no best possible live score is better than current worst finished
        # scores, the search cannot improve the finished set further.
        search_terminated = worst_finished_scores > best_live_scores

        # If we're not at the max decode length, and the search hasn't terminated,
        # continue looping.
        return jnp.any(not_at_end & ~search_terminated)

    def beam_search_loop_body_fn(state: _BeamState) -> _BeamState:
        """Beam search loop state update function."""
        # [batch].
        cur_index = state.cur_index
        next_index = cur_index + 1

        # Collect the current position slice along length to feed the fast autoregressive decoder
        # model. Flatten the beam dimension into batch dimension for feeding into the model.
        # --> [batch * beam, 1]
        flat_ids = flatten_decoding_dim(
            jnp.take_along_axis(state.live_seqs, cur_index[:, None, None], axis=2)
        )
        # Flatten beam dimension into batch to be compatible with model.
        # {[batch, beam, ...], ...} --> {[batch * beam, ...], ...}.
        flat_cache = vectorized_tree_map(flatten_decoding_dim, state.cache)

        # Call fast-decoder model on current tokens to get next-position logits.
        # --> [batch * beam, vocab].
        new_flat_log_probs, new_flat_cache = tokens_to_scores(flat_ids, flat_cache)

        prefix_merger_state = state.prefix_merger
        live_scores = state.live_scores
        if prefix_merger:
            prefix_merger_state = prefix_merger.update(
                tokens=jnp.reshape(flat_ids, [batch_size, num_decodes]),
                state=prefix_merger_state,
            )
            merge_matrix = prefix_merger.compute(prefix_merger_state)
            live_scores = _merge_prefixes(merge_matrix=merge_matrix, log_probs=live_scores)

        # Unflatten beam dimension.
        # [batch * beam, vocab] --> [batch, beam, vocab].
        candidate_log_probs = unflatten_decoding_dim(new_flat_log_probs, batch_size, beam_size)
        # Unflatten beam dimension in decoding cache.
        # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}.
        new_cache = vectorized_tree_map(
            lambda x: unflatten_decoding_dim(x, batch_size, beam_size), new_flat_cache
        )

        # Add new logprobs to existing prefix logprobs.
        # --> [batch, beam, vocab].
        log_probs = candidate_log_probs + jnp.expand_dims(live_scores, axis=2)

        # We'll need the vocab size, gather it from the log probability dimension.
        vocab_size = log_probs.shape[-1]

        # The sequence log_probs of those ending with eos_id at the current time step.
        # [batch, beam].
        cur_finishing_scores = log_probs[:, :, eos_id]
        # The sequence log_probs [batch, beam, vocab], where sequences ending with eos_id have
        # log prob of NEG_INF, so that we will only pick tokens for "live" sequences.
        log_probs += NEG_INF * jax.nn.one_hot(eos_id, vocab_size, dtype=log_probs.dtype)

        # Flatten beam and vocab dimensions.
        flat_log_probs = log_probs.reshape((batch_size, beam_size * vocab_size))

        # Gather the top K scores from _all_ beams.
        # --> [batch, beam], [batch, beam].
        topk_log_probs, topk_indices = _top_k_two_stage(flat_log_probs, k=beam_size)

        # Recover token ID by modulo division.
        topk_ids = topk_indices % vocab_size
        # Recover the beam index by floor division.
        topk_beam_indices = topk_indices // vocab_size

        # Force decode `inputs` into topk_ids up until PAD. When `inputs` is all PADs this is a
        # no-op.
        next_input_token = jnp.take_along_axis(
            inputs.astype(jnp.int32), next_index[:, None], axis=1, mode="clip"
        )

        # [batch, 1].
        out_of_prompt = next_input_token == pad_id

        # When forcing prompts, update log probabilities to `0` for the top of the beam and NEG_INF
        # for the rest, effectively keeping only one beam alive.
        # --> [batch, beams].
        inside_prompt_log_probs = jnp.concatenate(
            [
                jnp.zeros((batch_size, 1), dtype=topk_log_probs.dtype),
                jnp.full_like(topk_log_probs[:, : beam_size - 1], NEG_INF),
            ],
            axis=1,
        )
        top_alive_log_probs = (
            topk_log_probs * out_of_prompt + inside_prompt_log_probs * ~out_of_prompt
        )
        # Apply the brevity penalty.
        cur_finishing_scores = brevity_penalty(
            length=next_index[:, None],
            raw_scores=cur_finishing_scores,
        )
        # Set cur_finishing_scores[i] to be NEG_INF if it is still within the given prompt,
        # so that it will not be picked when we select top_finished_scores.
        cur_finishing_scores = cur_finishing_scores * out_of_prompt + NEG_INF * ~out_of_prompt

        # Gather K top beams.
        # --> [batch, beam, length].
        topk_alive_seq = _gather_beams(
            state.live_seqs,
            topk_beam_indices,
            batch_size,
            beam_size,
            beam_size,
        )

        # Compute next token ID. Expand for broadcasting.
        # --> [batch, beam, 1].
        topk_ids = jnp.expand_dims(
            topk_ids * out_of_prompt + next_input_token * ~out_of_prompt,
            axis=2,
        )
        # Note: for indices in `next_index` that exceed `max_decode_len-1`, one-hot will zero-out
        # the update.
        oh_indices = jax.nn.one_hot(
            next_index[:, None], topk_alive_seq.shape[-1], dtype=topk_alive_seq.dtype
        )
        # Update sequences for the top-k new sequences.
        # --> [batch, beam, length].
        topk_alive_seq = topk_alive_seq * (1 - oh_indices) + topk_ids * oh_indices

        # Gather the top k beam-associated caches.
        # --> {[batch, beams, ...], ...}.
        top_alive_cache = _gather_beams(
            new_cache, topk_beam_indices, batch_size, beam_size, beam_size
        )
        prefix_merger_state = _gather_beams(
            prefix_merger_state, topk_beam_indices, batch_size, beam_size, beam_size
        )

        # Combine sequences, scores, and flags along the beam dimension and compare
        # new finished sequence scores to existing finished scores and select the
        # best from the new set of beams.
        cur_finishing_seq = state.live_seqs * (1 - oh_indices) + eos_id * oh_indices
        finished_seqs = jnp.concatenate(  # --> [batch, 2*beams, length].
            [state.finished_seqs, cur_finishing_seq], axis=1
        )
        finished_scores = jnp.concatenate(  # --> [batch, 2*beams].
            [state.finished_scores, cur_finishing_scores], axis=1
        )
        # --> [batch, beams, length], [batch, beams].
        top_finished_seq, top_finished_scores = _gather_topk_beams(
            [finished_seqs, finished_scores], finished_scores, batch_size, beam_size
        )

        # When decoding starts at different indices, some sequences can reach the end first.
        # In these cases, we use this mask to "freeze" beams corresponding to those batch indices.
        out_of_sequence = next_index >= max_decode_len

        def mask_out_of_sequence(old: Tensor, new: Tensor):
            assert old.ndim == new.ndim
            mask = out_of_sequence
            while mask.ndim < new.ndim:
                mask = mask[..., None]
            return old * mask + new * ~mask

        return _BeamState(
            cur_index=state.cur_index + 1,
            live_scores=mask_out_of_sequence(state.live_scores, top_alive_log_probs),
            finished_scores=mask_out_of_sequence(state.finished_scores, top_finished_scores),
            live_seqs=mask_out_of_sequence(state.live_seqs, topk_alive_seq),
            finished_seqs=mask_out_of_sequence(state.finished_seqs, top_finished_seq),
            # For simplicity and efficiency, we don't mask updates to the cache.
            # Ultimately this should have no impact as the sequences and scores are masked.
            cache=top_alive_cache,
            prefix_merger=prefix_merger_state,
        )

    # Run while loop and get final beam search state.
    if loop == "lax":
        final_state = lax.while_loop(
            beam_search_loop_cond_fn, beam_search_loop_body_fn, beam_search_init_state
        )
    elif loop == "python":
        state = beam_search_init_state
        while beam_search_loop_cond_fn(state):
            state = beam_search_loop_body_fn(state)
        final_state = state
    else:
        raise NotImplementedError(loop)

    return BeamSearchOutputs(
        # Drop the first dummy 0 token.
        sequences=final_state.finished_seqs[:, :, 1:],
        scores=final_state.finished_scores,
        live_sequences=final_state.live_seqs[:, :, 1:],
        live_scores=final_state.live_scores,
    )


class DecodingState(NamedTuple):
    """Holds sample decoding state data."""

    # The position of the decoding loop in the length dimension.
    cur_index: Tensor  # scalar int32: current decoded length index.
    # The active sequences.
    sequences: Tensor  # int32: [batch_size, num_decodes, max_decode_len].
    # The sequence token log probabilities.
    token_scores: Tensor  # float32: [batch_size, num_decodes, max_decode_len].
    # Whether a stop decoding condition has been reached.
    stop_decoding: Tensor  # bool: [batch_size, num_decodes].
    # The current state of the autoregressive decoding caches.
    cache: NestedTensor
    # Random generator state.
    prng_key: Tensor


def _decode_init(
    *,
    inputs: Tensor,
    time_step: Tensor,
    num_decodes: int,
    max_decode_len: int,
    cache: NestedTensor,
    prng_key: Tensor,
    pad_id: int,
    token_scores: Optional[Tensor] = None,
) -> DecodingState:
    """Initializes the sample decode state data structure.

    Args:
        inputs: An int32 tensor of shape [batch_size, length] where length <= max_decode_len.
        time_step: Initial time steps for decoding of shape [batch_size].
        num_decodes: Number of sequences to decode per batch example.
        max_decode_len: The maximum length of the sequence to be generated (including dummy prompt
            token).
        cache: State of the decoder model.
        prng_key: The initial JAX random key state.
        pad_id: Token ID associated with padded input.
        token_scores: Optional initial scores of shape [batch_size, length] where
            length < max_decode_len, e.g. as produced by prefilling. Note that length should be
            strictly less than max_decode_len, as we exclude the scores for the dummy prompt token.
            Defaults to all zeros.

    Returns:
        Fully initialized DecodingState.

    Raises:
        ValueError: If inputs has an invalid shape.
    """
    if inputs.shape[0] != time_step.shape[0]:
        raise ValueError(
            f"Expected inputs.shape[0] ({inputs.shape[0]}) "
            f"== time_step.shape[0] ({time_step.shape[0]})."
        )
    if inputs.shape[1] > max_decode_len:
        raise ValueError(
            f"Expected inputs.shape[1] ({inputs.shape[1]}) <= max_decode_len ({max_decode_len})."
        )
    batch_size = inputs.shape[0]
    sequences = jnp.full((batch_size, num_decodes, max_decode_len), pad_id, dtype=jnp.int32)
    # Inputs are the prefix we will use for teacher forcing.
    sequences = sequences.at[:, :, : inputs.shape[1]].set(inputs[:, None, :])

    init_scores = jnp.zeros((batch_size, num_decodes, max_decode_len), dtype=jnp.float32)
    if token_scores is not None:
        if token_scores.shape[1] >= max_decode_len:
            raise ValueError(
                f"Expected token_scores.shape[1] ({token_scores.shape[1]}) < {max_decode_len}"
            )
        # Note: scores at index 0 are for the dummy prompt token, which will be dropped.
        init_scores = init_scores.at[:, :, 1 : 1 + token_scores.shape[1]].set(
            token_scores[:, None, :]
        )

    return DecodingState(
        cur_index=time_step,
        token_scores=init_scores,
        sequences=sequences,
        stop_decoding=jnp.zeros((batch_size, num_decodes), bool),
        # Expand cache to num_decodes size.
        cache=vectorized_tree_map(lambda x: add_decoding_dim(x, num_decodes=num_decodes), cache),
        prng_key=prng_key,
    )


class SampleOutputs(flax_struct.PyTreeNode):
    """Output values after performing sample decoding."""

    # Sequences which may or may not end with eos_id.
    sequences: Tensor  # int32: [batch_size, num_decodes, max_decode_len].
    # Scores corresponding to sequences above.
    token_scores: Tensor  # float32: [batch_size, num_decodes, max_decode_len].


class StopDecodingCondition(Protocol):
    """Callable which, given index, sequences and prompt mask, returns
    a bool tensor indicating if a sequence should stop decoding."""

    def __call__(self, *, index: Tensor, sequences: Tensor, out_of_prompt: Tensor) -> Tensor:
        """Given the current index, tensor sequences (batch x decodes x length) tensor, and
        a boolean mask indicating if we out out of the prompt, return a (batch x decodes) boolean
        tensor indicating if the given sequences have reached some stop condition.

        Args:
            index: Scalar or Tensor of shape [batch], giving the index of the most recently decoded
                token.
            sequences: Decoded tokens. Values past index are not defined. Shape
                float[batch, decodes, max_length].
            out_of_prompt: Prompt mask. Useful to not terminate if we see a sequence in the prompt.
                Shape bool[batch, decodes].

        Returns:
            bool[batch, decodes] boolean tensor indicating if decoding should stop.
        """


class StopOnSubsequence:
    """Early stopping on suffix-matches."""

    def __init__(
        self, stopping_seqs: Union[int, Sequence[int], Sequence[Sequence[int]]], pad_value=-1
    ):
        """Stops decoding when a sequence suffix-matches one of `stopping_seqs`.

        Note: we prefix pad targets by -1; if this is a meaningful token, override by setting
            pad_value.

        Args:
            stopping_seqs: List of lists of ids that mean we should stop decoding.
            pad_value: Safe value to use for padding.

        Raises:
            ValueError: if stopping_seqs is an empty list.
        """
        self.pad_value = pad_value

        if isinstance(stopping_seqs, int):
            stopping_seqs = [[stopping_seqs]]
        if isinstance(stopping_seqs, list) and stopping_seqs:
            if isinstance(stopping_seqs[0], int):
                stopping_seqs = [stopping_seqs]

        if any(len(seq) == 0 for seq in stopping_seqs):
            indices = np.argwhere([len(seq) == 0 for seq in stopping_seqs]).flatten().tolist()
            raise ValueError(
                "Zero length stopping seqs are not supported. "
                f"Zero length seqs at indices {indices}."
            )
        self.longest = max(len(el) for el in stopping_seqs)
        self.targets = jnp.stack(
            [
                jnp.pad(jnp.array(el), (self.longest - len(el), 0), constant_values=pad_value)
                for el in stopping_seqs
            ]
        )

    def __call__(self, *, index: Tensor, sequences: Tensor, out_of_prompt: Tensor) -> Tensor:
        sequences = jnp.pad(
            sequences, [(0, 0), (0, 0), (self.longest - 1, 0)], constant_values=self.pad_value
        )
        index = jnp.reshape(index, (-1, 1, 1))

        # `index` can take different values across the batch. We slice `self.longest`-length
        # sequences starting from each index.
        # [batch, num_decodes=1, length + longest - 1].
        index = index + jnp.arange(self.longest)
        # TODO(markblee): Compare against dispatch matrix + einsum to understand the performance
        # tradeoff between number of updates vs dispatch matrix size.
        # [batch, num_decodes, longest].
        sequences = jnp.take_along_axis(sequences, index, axis=-1)

        token_match = (self.targets[None, None, :, :] == sequences[:, :, None, :]) | (
            self.targets == self.pad_value
        )
        return token_match.all(-1).any(-1) & out_of_prompt


def sample_decode(
    *,
    inputs: Tensor,
    time_step: Tensor,
    cache: NestedTensor,
    tokens_to_scores: Callable[[Tensor, NestedTensor], tuple[Tensor, NestedTensor]],
    stop_decoding_condition: StopDecodingCondition,
    num_decodes: int,
    prng_key: Tensor,
    max_decode_len: Optional[int] = None,
    loop: Literal["lax", "python"] = "lax",
    pad_id: int = 0,
    input_token_scores: Optional[Tensor] = None,
) -> SampleOutputs:
    """Performs sampling decoding.

    Args:
        inputs: An int32 Tensor of shape [batch_size, length] containing a sequence of tokens.
            Please refer to `beam_search_decode` for more information on `inputs`.
        time_step: Initial time steps for decoding of shape [batch_size].
            Please refer to `beam_search_decode` for more information on `time_step`.
        cache: State of the decoder model.
        tokens_to_scores: Fast autoregressive decoder function taking single token
            slices and cache and returning next-token scores and updated cache.
            [batch*num_decodes, vocab], {} = tokens_to_scores([batch*num_decodes, 1], {}).
            NestedTensor usually has batch*num_decodes as the leading dim.
            The scores represents logits for sampling the next token, i.e.,
            the sampling probabilities will be softmax(scores, axis=-1).
            The caller can implement temperature-based, top-k, or top-p sampling as part
            of `tokens_to_scores`.
        stop_decoding_condition: StopDecodingCondition instance which given index, current sequences
            and prompt mask returns a bool tensor indicating if a sequence is complete.
        num_decodes: Number of decoded sequences to be returned for each input sequence.
        prng_key: The random key.
        max_decode_len: An optional maximum length of decoded sequence. If
            None, it uses `inputs.shape[1]` as `max_decode_len`.
        loop: "lax" or "python". The latter should only be used for debugging.
        pad_id: Token ID associated with padded input.
        input_token_scores: Optional initial scores of shape [batch_size, length] where
            length < max_decode_len, e.g. as produced by prefilling. Note that length should be
            strictly less than max_decode_len, as we exclude the scores for the dummy prompt token.
            In other words, input_token_scores[i, j - 1] represents the score for inputs[i, j],
            since the token at inputs[i, 0] does not have a score. Defaults to all zeros.

    Returns:
        SampleOutputs, containing the finished or live sequences and their corresponding scores.

    Raises:
        NotImplementedError: If an unsupported loop is provided.
    """
    batch_size = inputs.shape[0]
    if max_decode_len is None:
        max_decode_len = inputs.shape[1]
    # Account for conditioning input token.
    max_decode_len += 1

    # Initialize state.
    sample_decode_init_state = _decode_init(
        inputs=inputs,
        time_step=time_step,
        num_decodes=num_decodes,
        max_decode_len=max_decode_len,
        cache=cache,
        prng_key=prng_key,
        pad_id=pad_id,
        token_scores=input_token_scores,
    )

    def sample_decode_loop_cond_fn(state: DecodingState) -> bool:
        """Sample decode loop termination condition."""
        # TODO(markblee): More generally, can we wrap all conditions into stop_decoding_condition?
        not_at_end = state.cur_index < (max_decode_len - 1)  # [batch].
        terminate_early = jnp.all(state.stop_decoding, axis=1)  # [batch].
        return jnp.any(not_at_end & (~terminate_early))

    def sample_decode_loop_body_fn(state: DecodingState) -> DecodingState:
        """Sample decode loop state update function."""
        # [batch].
        cur_index = state.cur_index

        # Flatten the num_decodes dimension for the cache and the input IDs for this step.
        # [batch * num_decodes, 1].
        flat_ids = flatten_decoding_dim(
            jnp.take_along_axis(state.sequences, cur_index[:, None, None], axis=2)
        )
        # {[batch, num_decodes, ...], ...} --> {[batch * num_decodes, ...], ...}.
        flat_cache = vectorized_tree_map(flatten_decoding_dim, state.cache)

        # Call model on current tokens to get next-position logits and then unflatten.
        new_flat_log_probs, updated_flat_cache = tokens_to_scores(flat_ids, flat_cache)
        # [batch * num_decodes, vocab] --> [batch, num_decodes, vocab].
        candidate_log_probs = unflatten_decoding_dim(new_flat_log_probs, batch_size, num_decodes)
        # {[batch * num_decodes, ...], ...} --> {[batch, num_decodes, ...], ...}.
        updated_cache = vectorized_tree_map(
            lambda x: unflatten_decoding_dim(x, batch_size, num_decodes), updated_flat_cache
        )

        # Sample next token IDs according to logits.
        prng_key, updated_prng_key = jax.random.split(state.prng_key)
        # [batch, num_decodes].
        sampled_next_token = jax.random.categorical(prng_key, logits=candidate_log_probs, axis=2)

        # We allow next_index to exceed `max_decode_len-1`:
        # - When reading from next_index, mode="clip" will effectively read `max_decode_len-1`;
        # - When writing to next_index, one-hot will cause the write to become a no-op.
        # [batch].
        next_index = cur_index + 1

        # Collect next input token.
        # [batch, num_decodes].
        next_input_token = jnp.squeeze(
            jnp.take_along_axis(state.sequences, next_index[:, None, None], axis=2, mode="clip"),
            axis=2,
        )
        out_of_prompt = next_input_token == pad_id

        # Compute the overall next token based on whether we are inside the prompt or not.
        next_token = sampled_next_token * out_of_prompt + next_input_token * ~out_of_prompt
        # [batch, num_decodes].
        next_token_log_prob = jnp.sum(
            candidate_log_probs * jax.nn.one_hot(next_token, candidate_log_probs.shape[-1]),
            axis=-1,
        )

        # If end sequence tokens already emitted, adjust next token to pad_id and update log_prob.
        next_token = (
            next_token * ~state.stop_decoding
            + jnp.full_like(next_token, pad_id) * state.stop_decoding
        )
        # [batch, num_decodes].
        next_token_log_prob = (
            next_token_log_prob * ~state.stop_decoding
            + jnp.zeros_like(next_token_log_prob) * state.stop_decoding
        )

        # Update score and sequence trackers. For indices in `next_index` that exceed
        # `max_decode_len-1`, one-hot will zero-out the update.
        # [batch, num_decodes=1, length].
        oh_indices = jax.nn.one_hot(
            next_index[:, None], state.sequences.shape[-1], dtype=state.sequences.dtype
        )
        # [batch, num_decodes, length].
        updated_sequences = (
            state.sequences * (1 - oh_indices) + jnp.expand_dims(next_token, 2) * oh_indices
        )
        updated_token_scores = (
            state.token_scores * (1 - oh_indices)
            + jnp.expand_dims(next_token_log_prob, 2) * oh_indices
        )
        updated_stop_decoding = state.stop_decoding | stop_decoding_condition(
            index=jnp.minimum(next_index, max_decode_len - 1),
            sequences=updated_sequences,
            out_of_prompt=out_of_prompt,
        )
        return DecodingState(
            cur_index=next_index,
            token_scores=updated_token_scores,
            sequences=updated_sequences,
            stop_decoding=updated_stop_decoding,
            cache=updated_cache,
            prng_key=updated_prng_key,
        )

    # Run while loop and get final sample decoding state.
    if loop == "lax":
        final_state = lax.while_loop(
            sample_decode_loop_cond_fn, sample_decode_loop_body_fn, sample_decode_init_state
        )
    elif loop == "python":
        state = sample_decode_init_state
        while sample_decode_loop_cond_fn(state):
            state = sample_decode_loop_body_fn(state)
        final_state = state
    else:
        raise NotImplementedError(loop)

    return SampleOutputs(
        # Drop first conditioning input token.
        sequences=final_state.sequences[:, :, 1:],
        token_scores=final_state.token_scores[:, :, 1:],
    )


def infer_initial_time_step(prefix: Tensor, *, pad_id: int) -> Tensor:
    """Computes initial time step based on prefix.

    We infer these from the last non-pad token in the prefix (i.e., the prefix itself can have
    pad tokens). If the prefix consists of all pad tokens, we start at index 0.

    Args:
        prefix: Initial decoding inputs of shape [batch_size, ..., prefix_length].
        pad_id: Token ID corresponding to padding.

    Returns:
        Initial time steps of shape [batch_size, ...] with values in [0, prefix_length).
    """
    return ((prefix != pad_id) * jnp.arange(prefix.shape[-1])).max(axis=-1)
