# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/fairseq:
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.

"""Input processing for masked language modeling.

References:
https://github.com/pytorch/fairseq/blob/7e758841da9e05cb21826a60d30a563a9e189d1d/fairseq/tasks/masked_lm.py#L113
"""

import enum
import functools
from typing import Callable, Optional

import seqio
import tensorflow as tf

from axlearn.common import input_tf_data
from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.input_text import infer_bos_id, tokenize
from axlearn.common.input_tf_data import rekey, shuffle
from axlearn.common.utils import Tensor


def _ids_to_word_starts(inputs: tf.Tensor, vocab: seqio.SentencePieceVocabulary) -> tf.Tensor:
    """Computes start of word indexes.

    Note: The function expects morphun tokenized input.

    Args:
        inputs: int Tensor [seq_len] of piece ids.
        vocab: sentence piece vocabulary object.

    Returns:
        int Tensor containing start word indices, shape <= [seq_len].
    """
    tokens = vocab.tf_tokenizer.id_to_string(inputs)  # Default encoding is UTF-8.
    # Compute word starts by looking for one of ▁, [, or <.
    # pylint: disable-next=anomalous-backslash-in-string
    word_starts = tf.where(tf.strings.regex_full_match(tokens, b"^(\xe2\x96\x81|\\[|<).*$"))
    # Flatten the result, ensuring the result still has rank 1.
    return tf.reshape(word_starts, [-1])


class MLMAction(enum.Enum):
    """Each token is assigned an `action` for MLM:

    0: Do nothing, do not include in loss calculation.
    1: Mask this token, include in loss calculation.
    2: Swap this token, include in loss calculation.
    3: Keep this token, include in loss calculation.
    """

    DO_NOTHING = 0
    MASK = 1
    # Note: this means we replace a token with a random one.
    SWAP = 2
    KEEP = 3


def _validate_mlm_actions(
    *,
    select_token_prob: float,
    mask_selected_prob: float,
    swap_selected_prob: float,
):
    if not 0 <= select_token_prob <= 1:
        raise ValueError("Expected select_token_prob to be in [0,1].")
    if not 0 <= mask_selected_prob <= 1:
        raise ValueError("Expected mask_selected_prob to be in [0,1].")
    if not 0 <= swap_selected_prob <= 1:
        raise ValueError("Expected swap_selected_prob to be in [0,1].")
    if mask_selected_prob + swap_selected_prob > 1:
        raise ValueError("Expected mask_selected_prob + swap_selected_prob <= 1.")


def iid_mlm_actions(
    *,
    select_token_prob: float = 0.15,
    mask_selected_prob: float = 0.8,
    swap_selected_prob: float = 0.1,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Applies masking by treating each token independently.

    Args:
        select_token_prob: Probability of selecting a token for MLM.
        mask_selected_prob: Probability of masking, given a token has been selected.
        swap_selected_prob: Probability of swapping with a random token, given a token has been
            selected.

    Returns:
        A function that takes a seq_len and outputs actions of shape [seq_len].
    """
    _validate_mlm_actions(
        select_token_prob=select_token_prob,
        mask_selected_prob=mask_selected_prob,
        swap_selected_prob=swap_selected_prob,
    )
    do_nothing_prob = 1.0 - select_token_prob
    mask_prob = select_token_prob * mask_selected_prob
    swap_prob = select_token_prob * swap_selected_prob
    keep_prob = select_token_prob * (1.0 - mask_selected_prob - swap_selected_prob)

    action_probs = [0] * len(MLMAction)
    action_probs[MLMAction.DO_NOTHING.value] = do_nothing_prob
    action_probs[MLMAction.MASK.value] = mask_prob
    action_probs[MLMAction.SWAP.value] = swap_prob
    action_probs[MLMAction.KEEP.value] = keep_prob
    assert len(MLMAction) == 4, "Not all actions were assigned a probability."

    # tf.random.categorical takes unnormalized log probabilities (logits).
    return lambda seq_len: tf.random.categorical(tf.math.log([action_probs]), seq_len)[0]


def roberta_mlm_actions(
    *,
    select_token_prob: float = 0.15,
    mask_selected_prob: float = 0.8,
    swap_selected_prob: float = 0.1,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Applies masking as seen in fairseq RoBERTa, by selecting a fixed number of tokens to act on.
    The defaults match the fairseq defaults.

    Reference:
    https://github.com/facebookresearch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/fairseq/data/mask_tokens_dataset.py#L102

    Args:
        select_token_prob: Probability of selecting a token for MLM.
        mask_selected_prob: Probability of masking, given a token has been selected.
        swap_selected_prob: Probability of swapping with a random token, given a token has been
            selected.

    Returns:
        A function that takes a seq_len and outputs actions of shape [seq_len].
    """
    _validate_mlm_actions(
        select_token_prob=select_token_prob,
        mask_selected_prob=mask_selected_prob,
        swap_selected_prob=swap_selected_prob,
    )
    swap_or_keep_selected_prob = 1 - mask_selected_prob  # Always >= swap_selected_prob >= 0.
    keep_selected_prob = swap_or_keep_selected_prob - swap_selected_prob  # Always >= 0.

    def generate_actions(seq_len: tf.Tensor):
        # Decide how many elements to select. Following fairseq, we add random noise in the range
        # [0,1) for probabilistic rounding.
        num_selected = tf.cast(
            select_token_prob * tf.cast(seq_len, tf.float32) + tf.random.uniform([]), tf.int32
        )

        # Sample uniformly without replacement using Gumbel-max trick.
        # See: https://github.com/tensorflow/tensorflow/issues/9260
        _, indices = tf.nn.top_k(tf.random.uniform([seq_len]), k=num_selected)
        # Note: this is subtly different from using `tf.nn.in_top_k`, which includes all candidates
        # straddling the top-k boundary, instead of returning exactly k.
        mask = tf.reduce_sum(tf.one_hot(indices, seq_len), axis=0)
        actions = tf.cast(mask * MLMAction.MASK.value, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.bool)

        if swap_or_keep_selected_prob > 0:
            decisions = tf.random.uniform([seq_len])

            # Among the ones we masked, choose some fraction to either swap or keep.
            keep = tf.logical_and(mask, decisions < keep_selected_prob)
            swap = tf.logical_and(
                mask,
                tf.logical_and(
                    keep_selected_prob <= decisions, decisions < swap_or_keep_selected_prob
                ),
            )

            # "Unmask" the ones we want to keep.
            actions = tf.where(keep, x=MLMAction.KEEP.value, y=actions)

            # Swap some tokens.
            actions = tf.where(swap, x=MLMAction.SWAP.value, y=actions)

        return actions

    return generate_actions


def roberta_mlm_actions_combinatorial_ngram(
    *,
    mask_prob: float = 1.0,
    swap_prob: float = 0.0,
    n: int = 1,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Applies roberta_mlm_actions style masking to all ngrams, one at a time.

    Args:
        mask_prob: Probability of masking.
        swap_prob: Probability of swapping with a random token.
        n: n in ngram.

    Returns:
        A function that takes a scalar seq_len and
        outputs actions of shape [seq_len - n + 1, seq_len].

    Raises:
        ValueError: If probabilities don't add up.
    """
    keep_prob = 1 - (mask_prob + swap_prob)
    swap_or_keep_prob = 1 - mask_prob  # Always >= swap_prob >= 0.
    if not mask_prob + swap_prob + keep_prob == 1:
        raise ValueError("Expected sum of probabilities to be equal to 1.")

    # seq_len: int32 scalar tensor.
    def generate_actions(seq_len: tf.Tensor):
        # Compute combinatorial ngram indices to mask.
        # [seq_len - n + 1 , 1].
        start_indices = tf.range(seq_len - n + 1)[:, None]
        # [1, seq_len].
        indices = tf.range(seq_len)[None, :]
        # [seq_len - n + 1, seq_len].
        mask = tf.logical_and(indices >= start_indices, indices < start_indices + n)

        # Generate combinatorial ngram actions.
        actions = tf.cast(mask, dtype=tf.int32) * MLMAction.MASK.value
        num_of_examples = tf.shape(mask)[0]

        if swap_or_keep_prob > 0:
            # We have only one ngram per example to decide about.
            decisions = tf.random.uniform([num_of_examples, 1])

            # Among the ones we masked, choose some fraction to either swap or keep.
            keep = tf.logical_and(mask, decisions < keep_prob)
            swap = tf.logical_and(
                mask,
                tf.logical_and(keep_prob <= decisions, decisions < swap_or_keep_prob),
            )

            # "Unmask" the ones we want to keep.
            actions = tf.where(keep, x=MLMAction.KEEP.value, y=actions)

            # Swap some tokens.
            actions = tf.where(swap, x=MLMAction.SWAP.value, y=actions)

        return actions

    return generate_actions


def apply_mlm_mask(
    *,
    whole_word_mask: bool = False,
    input_key: str = "input_ids",
    target_key: str = "target_labels",
    ignore_input_ids: list[int],
    ignore_target_id: int,
    mask_id: int,
    vocab_cfg: InstantiableConfig,
    actions_cfg: InstantiableConfig = config_for_function(iid_mlm_actions),
    is_translation_style: bool = False,
) -> input_tf_data.DatasetToDatasetFn:
    """Applies dynamic MLM masking to the inputs, meaning each epoch generates a different mask.

    The strategy for selecting `action`s is configurable via `actions_cfg`. By default, we apply
    an IID masking strategy.

    Note: this should not assume eager execution for compatibility with tfds mapper.
    Note: it's possible to produce examples where nothing is masked (i.e. all targets are ignored).
        This is mitigated by filtering out short sequences or packing sequences together.

    Args:
        whole_word_mask: If True, apply_mlm_mask works at word level instead of token level.
        input_key: The name of the key in each example to be subjected to MLM.
        target_key: The name of the key in each example which will have the ground truth of MLM.
        ignore_input_ids: List of ids we want to force apply do_nothing_prob to. eg: pad token ID.
        ignore_target_id: Ignore target ID.
        mask_id: Mask token ID.
        vocab_cfg: Config to instantiate the seqio vocab.
        actions_cfg: Strategy for assigning actions.
        is_translation_style: If True, targets are not ignored for MLMAction.DO_NOTHING actions.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing key `input_key`
        with int32 Tensors of shape [seq_len] as values, and each output example is a dict with
        `input_key` and `target_key` as keys and int32 tensors of shape [seq_len] as values.

    Raises:
        ValueError: If mask token ID is invalid.
    """
    ignore_input_ids = tf.constant(ignore_input_ids, shape=(len(ignore_input_ids), 1))
    vocab: seqio.SentencePieceVocabulary = vocab_cfg.instantiate()
    actions_fn: Callable[[tf.Tensor], tf.Tensor] = actions_cfg.instantiate()

    # Validate that the vocab has a `mask_id`.
    unk_id = vocab.unk_id
    if mask_id == unk_id:
        raise ValueError(f"Mask token ID ({mask_id}) should be different from unk ID ({unk_id}).")

    # Validate that `mask_id` is not an ignored ID.
    if mask_id in ignore_input_ids:
        raise ValueError(f"Mask token ID ({mask_id}) should not be ignored ({ignore_input_ids}).")

    # `mask_id` is always prepended to the input sequence (see notes for tf.concat below).
    # Here, we validate that the `mask_id` constitutes a start-of-word.
    if tf.size(_ids_to_word_starts([mask_id], vocab)) != 1:
        raise ValueError(f"Mask token ID ({mask_id}) should constitute a start-of-word.")

    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        inputs = example[input_key]

        # Each input should be [seq_len].
        tf.assert_equal(tf.rank(inputs), 1)

        # Prepend a dummy start-of-word token (currently `mask_id`), which we discard at the end.
        # In the whole-word-masking case, we construct a RaggedTensor where each row groups all
        # tokens corresponding to a whole word. Prepending a dummy start-of-word ensures that we
        # always have a "start of word" index at 0, which is necessary for constructing valid
        # RaggedTensors. There are several cases:
        # 1. Input already starts with a start-of-word. Masking behavior is unchanged, as the added
        #    dummy token is masked independently and then discarded.
        # 2. Input does not start with, but contains, a start-of-word. The leading segment is then
        #    treated as a word. This is acceptable given that the segment is likely truncated from
        #    an actual start-of-word, e.g. from the previous document.
        # 3. Input does not contain a start-of-word at all. To prevent masking the entire sequence,
        #    this falls back to masking tokens independently (see tf.cond).
        # In the non-whole-word-masking case, we also mask tokens independently (see tf.cond).
        # pylint: disable-next=no-value-for-parameter,unexpected-keyword-arg
        inputs = tf.concat([[mask_id], inputs], axis=0)
        flat_seq_len = tf.shape(inputs)[0]

        # Get begin indices of words and segment inputs.
        start_word_idxs = tf.constant([], dtype=tf.int64)
        if whole_word_mask:
            start_word_idxs = _ids_to_word_starts(inputs, vocab)

        # If whole word masking is disabled, or if the input contains no words,
        # fallback to masking each token separately.
        start_word_idxs = tf.cond(
            tf.size(start_word_idxs) > 1,
            true_fn=lambda: start_word_idxs,
            false_fn=lambda: tf.range(flat_seq_len, dtype=tf.int64),
        )

        inputs = tf.RaggedTensor.from_row_starts(inputs, start_word_idxs)
        targets = inputs
        seq_len = tf.shape(start_word_idxs)[0]  # Number of words (or tokens).

        # [seq_len].
        actions = actions_fn(seq_len)

        # Ignored input ids should have no action. tf.where takes from x where mask is True, and
        # from y otherwise.
        actions = tf.where(
            tf.reduce_any(
                tf.RaggedTensor.from_row_starts(
                    tf.reduce_any(inputs.flat_values == ignore_input_ids, axis=0), start_word_idxs
                ),
                axis=1,
            ),
            x=tf.constant(MLMAction.DO_NOTHING.value, dtype=actions.dtype),
            y=actions,
        )
        actions = actions[:, None]  # For ragged input.

        # Assign mask tokens.
        inputs = tf.where(
            actions == MLMAction.MASK.value, x=tf.constant(mask_id, dtype=inputs.dtype), y=inputs
        )

        # Swap tokens with random tokens. Note: maxval is exclusive.
        # TODO(markblee): fairseq also supports frequency weighted random replacement, where
        # tokens are selected based on frequency in the vocab. This is not used by default.
        # https://github.com/facebookresearch/fairseq/blob/e83bd93cd32ec5d58a473ca9aac0585b135ae0fd/fairseq/data/mask_tokens_dataset.py#L36-L37
        swap_positions = actions == MLMAction.SWAP.value
        swap_tokens = tf.random.uniform(
            [flat_seq_len],
            minval=0,
            maxval=vocab.vocab_size,
            dtype=inputs.dtype,
        )
        swap_tokens = tf.RaggedTensor.from_row_starts(swap_tokens, start_word_idxs)

        inputs = tf.where(swap_positions, x=swap_tokens, y=inputs)

        if not is_translation_style:
            # For tokens with no action, ignore their corresponding targets.
            targets = tf.where(
                actions == MLMAction.DO_NOTHING.value,
                x=tf.constant(ignore_target_id, dtype=targets.dtype),
                y=targets,
            )

        # Drop the dummy token.
        example[input_key] = inputs.flat_values[1:]
        example[target_key] = targets.flat_values[1:]
        return example

    return input_tf_data.preserve_element_spec(
        seqio.map_over_dataset(process_example_fn), key_map={target_key: input_key}
    )


def apply_mlm_mask_combinatorial_ngram(
    *,
    whole_word_mask: bool = False,
    input_key: str = "input_ids",
    target_key: str = "target_labels",
    ignore_input_ids: list[int],
    ignore_target_id: int,
    mask_id: int,
    vocab_cfg: InstantiableConfig,
    actions_cfg: InstantiableConfig = config_for_function(roberta_mlm_actions_combinatorial_ngram),
    is_translation_style: bool = False,
    n: int = 1,
) -> input_tf_data.DatasetToDatasetFn:
    """Applies MLM masking to all ngrams in the input, one at a time.

    The strategy for selecting `action`s is configurable via `actions_cfg`. By default, we apply
    the fairseq style masking strategy.

    Note: this should not assume eager execution for compatibility with tfds mapper.
    Note: it's possible to produce examples where nothing is masked (i.e. all targets are ignored).
        This is mitigated by filtering out short sequences or packing sequences together.

    Args:
        whole_word_mask: If True, apply_mlm_mask_combinatorial_ngram works
            at word level instead of token level.
        input_key: The name of the key in each example to be subjected to MLM.
        target_key: The name of the key in each example which will have the ground truth of MLM.
        ignore_input_ids: List of ids we want to force apply do_nothing_prob to. eg: pad token ID.
        ignore_target_id: Ignore target ID.
        mask_id: Mask token ID.
        vocab_cfg: Config to instantiate the seqio vocab.
        actions_cfg: Strategy for assigning actions.
        is_translation_style: If True, targets are not ignored for MLMAction.DO_NOTHING actions.
        n: n in ngram.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing key `input_key`
        with int32 Tensors of shape [seq_len] as values, and output is num_of_examples long list of
        example dicts with `input_key` and `target_key` as keys and
        int32 tensors of shape [seq_len] as values.
        num_of_examples is determined by (seq_len - n + 1).

    Raises:
        ValueError: If mask token ID is invalid.
        InvalidArgumentError: If n > seq_len.
    """
    ignore_input_ids = tf.constant(ignore_input_ids, shape=(len(ignore_input_ids), 1))
    vocab: seqio.SentencePieceVocabulary = vocab_cfg.instantiate()
    actions_fn: Callable[[tf.Tensor], tf.Tensor] = actions_cfg.set(n=n).instantiate()

    # Validate that the vocab has a `mask_id`.
    unk_id = vocab.unk_id
    if mask_id == unk_id:
        raise ValueError(f"Mask token ID ({mask_id}) should be different from unk ID ({unk_id}).")

    # Validate that `mask_id` is not an ignored ID.
    if mask_id in ignore_input_ids:
        raise ValueError(f"Mask token ID ({mask_id}) should not be ignored ({ignore_input_ids}).")

    # `mask_id` is always prepended to the input sequence (see notes for tf.concat below).
    # Here, we validate that the `mask_id` constitutes a start-of-word.
    if tf.size(_ids_to_word_starts([mask_id], vocab)) != 1:
        raise ValueError(f"Mask token ID ({mask_id}) should constitute a start-of-word.")

    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        inputs = example[input_key]

        # Each input should be [seq_len].
        tf.assert_equal(tf.rank(inputs), 1)

        flat_seq_len = tf.shape(inputs)[0]

        # Get begin indices of words and segment inputs.
        start_word_idxs = tf.constant([], dtype=tf.int64)
        if whole_word_mask:
            start_word_idxs = _ids_to_word_starts(inputs, vocab)

        # If whole word masking is disabled, or if the input contains no words,
        # fallback to masking each token separately.
        start_word_idxs = tf.cond(
            tf.size(start_word_idxs) > 1,
            true_fn=lambda: start_word_idxs,
            false_fn=lambda: tf.range(flat_seq_len, dtype=tf.int64),
        )

        # We don't support whole word mask if input doesnt start with whole word.
        tf.assert_equal(tf.cast(start_word_idxs[0], tf.int32), 0)

        inputs = tf.RaggedTensor.from_row_starts(inputs, start_word_idxs)
        targets = inputs
        seq_len = tf.shape(start_word_idxs)[0]  # Number of words (or tokens).

        # [num_of_examples, seq_len]. num_of_examples = seq_len - n + 1.
        tf.debugging.assert_greater_equal(seq_len, n, "n (ngram) needs to be <= seq_len")
        actions = actions_fn(seq_len)
        num_of_examples = tf.shape(actions)[0]

        # Ignored input ids should have no action. tf.where takes from x where mask is True, and
        # from y otherwise.
        actions = tf.where(
            tf.reduce_any(
                tf.RaggedTensor.from_row_starts(
                    tf.reduce_any(inputs.flat_values == ignore_input_ids, axis=0), start_word_idxs
                ),
                axis=1,
            ),
            x=tf.constant(MLMAction.DO_NOTHING.value, dtype=actions.dtype),
            y=actions,
        )
        actions = actions[:, :, None]  # For ragged input.

        # Assign mask tokens.
        inputs = tf.where(
            actions == MLMAction.MASK.value, x=tf.constant(mask_id, dtype=inputs.dtype), y=inputs
        )

        # Swap tokens with random tokens. Note: maxval is exclusive.
        # TODO(markblee): fairseq also supports frequency weighted random replacement, where
        # tokens are selected based on frequency in the vocab. This is not used by default.
        # https://github.com/facebookresearch/fairseq/blob/e83bd93cd32ec5d58a473ca9aac0585b135ae0fd/fairseq/data/mask_tokens_dataset.py#L36-L37
        swap_positions = actions == MLMAction.SWAP.value
        # We swap every token in a selected ngram with a different one (mostly).
        flat_output_examples_len = flat_seq_len * num_of_examples
        swap_tokens = tf.random.uniform(
            [flat_output_examples_len],
            minval=0,
            maxval=vocab.vocab_size,
            dtype=inputs.dtype,
        )
        # Rearrange swap candidates to align with inputs.
        swap_tokens = tf.RaggedTensor.from_nested_row_splits(swap_tokens, inputs.nested_row_splits)
        # Broadcast swap_positions to match inputs and swap_tokens.
        swap_positions = tf.RaggedTensor.from_uniform_row_length(
            tf.reshape(swap_positions, (seq_len * num_of_examples, 1)), seq_len, num_of_examples
        )
        # Convert ragged splits to same type.
        inputs = inputs.with_row_splits_dtype(tf.dtypes.int32)
        swap_tokens = swap_tokens.with_row_splits_dtype(tf.dtypes.int32)
        # Assign swap tokens.
        inputs = tf.where(swap_positions, x=swap_tokens, y=inputs)

        if not is_translation_style:
            # For tokens with no action, ignore their corresponding targets.
            x = tf.constant(ignore_target_id, dtype=targets.dtype)
        else:
            x = targets
        targets = tf.where(
            actions == MLMAction.DO_NOTHING.value,
            x=x,
            y=targets,
        )

        example[input_key] = tf.reshape(inputs.flat_values, (num_of_examples, -1))
        example[target_key] = tf.reshape(targets.flat_values, (num_of_examples, -1))
        return example

    return lambda ds: ds.map(process_example_fn, num_parallel_calls=tf.data.AUTOTUNE).unbatch()


def text_to_mlm_input(
    *,
    is_training: bool,
    sentence_piece_vocab: InstantiableConfig,
    normalization: InstantiableConfig,
    max_len: int,
    # pylint: disable-next=redefined-outer-name
    apply_mlm_mask: InstantiableConfig,
    shuffle_buffer_size: int,
    mask_token: str = "[MASK]",
    ignore_target_id: int = 0,
    truncation: Optional[InstantiableConfig] = None,
    bos_token: Optional[str] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Input to support MLM.

    Implements RoBERTa "full-sentences" input processing:
    - Each input is packed with full sentences sampled contiguously from one or more documents
    - An input sequence may contain sentences from multiple documents
    - Sentences sampled from different documents have an extra separator (EOS) token in between them
    - Each packed sequence has a single BOS token prepended (even if we have multiple sentences)
    - Masking can occur at any position in the sequence, including BOS and EOS

    We assume the input follows the format:
    - Each line contains one or more full sentences from the same document.
    - An empty line should be used to indicate end of document.
    - This means that the input should either not be shuffled, or be shuffled at the document level.

    Note: the Huggingface implementation differs in that separator (EOS) tokens are exempted from
    masking via the `special_tokens_mask` produced by the tokenizer.

    Args:
        is_training: whether the input is used for training.
        sentence_piece_vocab: sentence piece vocab.
        normalization: text normalization mapper.
        max_len: the maximum number of tokens per sequence.
        apply_mlm_mask: mlm masking mapper.
        shuffle_buffer_size: if <= 0, no shuffling is applied. Otherwise, shuffles with this buffer
            size.
        mask_token: the mask token.
        ignore_target_id: the ID used for ignored targets.
        truncation: truncation strategy to use for long sequences. If None, drops long sequences.
        bos_token: optional BOS token. If None, infers from vocab (see `infer_bos_id`).

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing key "text" with
        string values and each output example is a dict with "input_ids" and "target_labels" as keys
        and int32 tensors of shape [max_len] as values. Keys of output examples are aligned with
        bert.BertModel.forward.
    """
    del is_training
    vocab = sentence_piece_vocab.instantiate()

    # Compute mlm token ids.
    pad_id = vocab.pad_id
    eos_id = vocab.eos_id
    unk_id = vocab.unk_id
    mask_id = vocab.tokenizer.piece_to_id(mask_token)
    bos_id = vocab.tokenizer.piece_to_id(bos_token) if bos_token else infer_bos_id(vocab)

    # TODO(markblee): support non-zero padding
    assert pad_id == ignore_target_id == 0, "SeqIO pads with zeros"
    token_ids = [pad_id, eos_id, mask_id, unk_id]
    assert len(set(token_ids)) == len(token_ids), f"Token IDs must be unique, got {token_ids}"

    # Subtract 1 from max_len so we have room for BOS.
    max_len_without_bos = max_len - 1

    def filter_long_sequences(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda x: tf.shape(x["input_ids"])[0] <= max_len_without_bos)

    @seqio.map_over_dataset
    def append_bos(example: dict[str, Tensor]) -> dict[str, Tensor]:
        # BOS is only added to the very first sentence of the input, not subsequent ones.
        # Note: this also drops other keys from example.
        # pylint: disable-next=no-value-for-parameter,unexpected-keyword-arg
        return {"input_ids": tf.concat([[bos_id], example["input_ids"]], axis=0)}

    return input_tf_data.chain(
        # Normalize inputs like lower case, trim etc.
        normalization,
        rekey({"input_ids": "text"}),
        # Convert string to ids.
        tokenize(
            output_features={
                "input_ids": seqio.Feature(
                    vocab,
                    add_eos=True,
                    dtype=tf.int32,
                ),
            },
            with_eos=True,
        ),
        # Filter long documents pre-packing if we don't want to truncate.
        truncation.set(max_len=max_len_without_bos) if truncation else filter_long_sequences,
        functools.partial(
            seqio.trim_and_pack_dataset,
            feature_lengths={"input_ids": max_len_without_bos},
        ),
        append_bos,
        # Note: following fairseq, all tokens can be masked, including BOS and EOS.
        apply_mlm_mask.set(
            ignore_input_ids=[pad_id],
            ignore_target_id=ignore_target_id,
            mask_id=mask_id,
            vocab_cfg=sentence_piece_vocab,
        ),
        # Shuffle after packing.
        shuffle(shuffle_buffer_size=shuffle_buffer_size),
    )
