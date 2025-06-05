# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/google-research:
# Copyright 2022 The Google Research Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google-research/text-to-text-transfer-transformer:
# Copyright 2021 The T5 Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Input processing as seen in T5.

References:
https://github.com/google-research/google-research/blob/77e1f14f3f7af7dc91dcdba7402ebc46c55ac2a6/primer/t5_tasks.py#L20
"""
import functools

import seqio
import tensorflow as tf

from axlearn.common import input_tf_data
from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.input_lm import ModelType, make_autoregressive_inputs


# TODO(markblee): Move to input_text.random_chunking processor to support different chunking modes.
def select_random_chunk(
    *, chunk_size: int, input_key: str = "source_ids"
) -> input_tf_data.DatasetToDatasetFn:
    """Token-preprocessor to extract one span of at most `chunk_size` tokens.

    If the token sequence is longer than `chunk_size`, we split the sequences into multiple segments
    with length `chunk_size` and randomly return one of them. Otherwise, we return the full
    sequence.

    Notes:
    - This function only retains `input_key` and drops all other input keys.
    - This is generally followed by `reduce_concat_tokens`.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/006196f21db316698bda30862f327bbdf195fc2b/t5/data/preprocessors.py#L1942

    Args:
        chunk_size: The maximum size of each chunk.
        input_key: The feature to apply chunking to.

    Returns:
        A DatasetToDatasetFn where each input example should be a dict containing key `input_key`
        with tensors as values and each output example is a dict containing key `input_key` with
        tensors of maximum length `chunk_size` as values. Other keys are dropped.

    Examples:
        chunk_size: 4
        inputs: {input_key: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        outputs: {input_key: [1, 2, 3, 4]}
            or   {input_key: [5, 6, 7, 8]}
            or   {input_key: [9]}
    """

    @seqio.map_over_dataset
    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        tokens = example[input_key]
        n_tokens = tf.shape(tokens)[0]
        num_segments = tf.cast(
            tf.math.ceil(tf.cast(n_tokens, tf.float32) / tf.cast(chunk_size, tf.float32)),
            tf.int32,
        )
        start = chunk_size * tf.random.uniform([], maxval=num_segments, dtype=tf.int32)
        end = tf.minimum(start + chunk_size, n_tokens)
        chunk = tokens[start:end]
        return {input_key: chunk}

    def process_dataset_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
        # Filter empty examples.
        dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[input_key]), 0))
        return process_example_fn(dataset)

    return process_dataset_fn


def reduce_concat_tokens(
    *, batch_size: int, input_key: str = "source_ids"
) -> input_tf_data.DatasetToDatasetFn:
    """Concatenate every N (batch_size) sequences into a long one.

    If we want to generate examples of exactly the right length (to avoid wasting space on padding),
    then we use this function, followed by `split_tokens`.

    Notes:
    - This function only retains `input_key` and drops all other input keys.
    - This function assumes that `input_key` does not already contain padding tokens.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/006196f21db316698bda30862f327bbdf195fc2b/t5/data/preprocessors.py#L2063

    Args:
        input_key: Which feature to use from the dataset.
        batch_size: The number of sequences to concatenate into one.

    Returns:
        A DatasetToDatasetFn where each input example should be a dict containing keys `input_key`
        with tensors as values and each output example is a dict with `input_key` as keys and long
        tensors as values.

    Examples:
        batch_size: 2
        inputs:
            [
                {input_key: [1, 2]},
                {input_key: [3, 4, 5, 6]},
                {input_key: [7]},
                {input_key: [8, 9]}
            ]
        after padded_batch:
            [
                {input_key: [[1, 2, 0, 0], [3, 4, 5, 6]]},
                {input_key: [[7, 0], [8, 9]]
            ]
        outputs:
            [
                {input_key: [1, 2, 3, 4, 5, 6]},
                {input_key: [7, 8, 9]}
            ]
    """

    @seqio.map_over_dataset
    def process_example_fn(x: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        tokens = tf.reshape(x[input_key], [-1])
        # Gather all non-padding tokens.
        # Note that this assumes no padding tokens in the sequence prior to padded_batch.
        tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
        return {input_key: tokens}

    def process_dataset_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.map(
            lambda x: {input_key: x[input_key]}, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.padded_batch(batch_size, padded_shapes={input_key: [-1]})
        return process_example_fn(dataset)

    return process_dataset_fn


def random_spans_helper(
    *,
    desired_source_length: int,
    noise_density: float,
    mean_noise_span_length: float,
    source_sentinel_length: int = 1,
    target_sentinel_length: int = 1,
) -> tuple[int, int]:
    """Computes input lengths to avoid padding in `random_spans_noise_mask`.

    During span corruption, some tokens in the input are replaced by sentinel tokens. This function
    tells us how long the original input must be, in order for the noised encoder inputs to have
    length `desired_source_length`. It also returns the length of the resulting targets.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/006196f21db316698bda30862f327bbdf195fc2b/t5/data/preprocessors.py#L2483

    Args:
        desired_source_length: Desired length of encoder inputs.
        noise_density: The percentage of tokens that will be masked.
        mean_noise_span_length: Average length of masked spans.
        source_sentinel_length: Length of each sentinel in the noised source.
        target_sentinel_length: Length of each sentinel in the noised target.

    Returns:
        A tuple of (input_length, target_length). `input_length` represents the number of tokens in
        the original input. `target_length` represents the length of target after noising.
    """

    def _noised_source_target_lengths(input_length: int) -> tuple[int, int]:
        """Calculates the length of noised source and target."""
        num_noise_tokens = int(round(float(input_length) * noise_density))
        num_nonnoise_tokens = input_length - num_noise_tokens
        num_noise_spans = int(round(float(num_noise_tokens) / mean_noise_span_length))
        # Include EOS.
        source_length = num_nonnoise_tokens + num_noise_spans * source_sentinel_length + 1
        target_length = num_noise_tokens + num_noise_spans * target_sentinel_length + 1
        return source_length, target_length

    input_length = desired_source_length - 1
    while _noised_source_target_lengths(input_length + 1)[0] <= desired_source_length:
        input_length += 1
    source_length, target_length = _noised_source_target_lengths(input_length)

    # Minor hack to get the target length to be equal to source length which is more likely to have
    # been set to a nice round number.
    if noise_density == 0.5 and target_length > source_length:
        input_length -= 1
        target_length -= 1
    return input_length, target_length


def split_tokens(
    *, max_tokens_per_segment: int, input_key: str = "source_ids"
) -> input_tf_data.DatasetToDatasetFn:
    """Split examples into multiple examples each.

    The intended use case is to break up long examples for use in unsupervised transfer-learning.
    `max_tokens_per_segment` can be computed from `random_spans_helper`.

    This function is generally preceded by `reduce_concat_tokens`.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/006196f21db316698bda30862f327bbdf195fc2b/t5/data/preprocessors.py#L2280

    Args:
        max_tokens_per_segment: The maximum number of tokens in each segment. Only the final
            segment may be shorter.
        input_key: Name of the feature to be split.

    Returns:
        A DatasetToDatasetFn where each input example should be a dict containing keys
        "input_ids" with tensors as values and each output example is a dict with
        "input_ids" as keys and values are tensors with maximum length equal to `max_len`.

    Examples:
        max_tokens_per_segment: 3
        inputs: [
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}
        ]
        outputs: [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5, 6]},
            {"input_ids": [7, 8]},
        ]
    """

    @seqio.map_over_dataset
    def split_tokens_example(
        x: dict[str, tf.Tensor]
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        """Split one token sequence into multiple sequences."""
        tokens = x[input_key]
        n_tokens = tf.shape(tokens)[0]
        length = max_tokens_per_segment

        # Pad to a multiple of length, then use tf.reshape to split up the tokens
        # into num_segments segments each of the given length.
        num_segments = tf.cast(
            tf.math.ceil(tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)), tf.int32
        )
        padding = num_segments * length - n_tokens
        feature_keys_to_split = [input_key]
        orig_lengths = {}
        outputs = {}

        for k in feature_keys_to_split:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_equal(
                        n_tokens,
                        tf.shape(x[k])[0],
                        message=(
                            f"Additional feature {k} is not the same size as "
                            f"{input_key} along axis 0 in split_tokens()."
                        ),
                    )
                ]
            ):
                shape = tf.shape(x[k])[1:]
                padded = tf.pad(
                    x[k],
                    tf.concat([[[0, padding]], tf.zeros([len(shape), 2], dtype=tf.int32)], 0),
                )
                orig_lengths[k] = tf.concat(
                    [tf.repeat(length, num_segments - 1), [length - padding]], 0
                )
                outputs[k] = tf.reshape(padded, tf.concat([[-1, length], shape], 0))
        return outputs, orig_lengths

    def strip_padding(
        inputs: dict[str, tf.Tensor], orig_lengths: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        output = {}
        for k, v in inputs.items():
            output[k] = v[: orig_lengths[k]]
        return output

    def process_dataset_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
        # Filter empty examples.
        dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[input_key]), 0))
        dataset = split_tokens_example(dataset)
        dataset = dataset.unbatch()
        dataset = dataset.map(strip_padding, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    return process_dataset_fn


def _random_segmentation(seq_len: int, num_segments: int) -> tf.Tensor:
    """Partitions a sequence of items randomly into non-empty segments.

    This is used by `random_spans_noise_mask`.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/006196f21db316698bda30862f327bbdf195fc2b/t5/data/preprocessors.py#L2741

    Args:
        seq_len: Number of tokens to segment.
        num_segments: An integer scalar in [1, seq_len].

    Returns:
        A Tensor with shape [num_segments] containing positive integers that add
        up to seq_len.

    Examples:
        seq_len: 9
        num_segments: 3
        outputs: [3, 3, 3]
            or   [2, 5, 2]
    """
    first_in_segment = tf.pad(
        tf.random.shuffle(tf.cast(tf.range(seq_len - 1) < num_segments - 1, tf.int32)),
        [[1, 0]],
    )
    segment_id = tf.cumsum(first_in_segment, axis=0)
    segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
    return segment_length


def random_spans_noise_mask(
    *,
    seq_len: int,
    noise_density: float,
    mean_noise_span_length: float,
) -> tf.Tensor:
    """Noise mask consisting of random spans of noise tokens.

    The number of noise tokens and the number of noise spans and non-noise spans are determined
    deterministically as follows:

        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)

    Spans alternate between non-noise and noise, beginning with non-noise. Subject to the above
    restrictions, all masks are equally likely.

    Note: When `num_noise_tokens` is less than `mean_noise_span_length`, mask one span with length
    equal to `num_noise_tokens`.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/421f9c3239991f46a0596514c838f00515f16296/t5/data/preprocessors.py#L2903
    https://arxiv.org/pdf/1910.10683.pdf Page 21, Table 3, Row 7.

    Args:
        seq_len: Length of input sequence.
        noise_density: Approximate density of output mask.
        mean_noise_span_length: Average length of masked spans.

    Returns:
        A bool Tensor with shape [seq_len], where True indicates positions to be masked.

    Examples:
        noise_density: 0.4
        seq_len: 10
        mean_noise_span_length: 2.0
        outputs: [False, True, True, False, False, False, True, True, False, False]
            or   [False, True, False, False, False, False, False, True, True, True]

        When `num_noise_tokens` is less than `mean_noise_span_length`:
        noise_density: 0.2
        seq_len: 10
        mean_noise_span_length: 3.0
        outputs: [False, False, False, False, False, False, False, False, True, True]

        When `num_noise_tokens` is not divisible by `mean_noise_span_length`:
        noise_density: 0.3
        seq_len: 10
        mean_noise_span_length: 2.0
        outputs: [False, False, False, False, True, True, False, False, False, True]
    """
    if noise_density == 0:
        return tf.zeros(seq_len, tf.bool)

    def to_int(x):
        return tf.cast(x, tf.int32)

    def to_float(x):
        return tf.cast(x, tf.float32)

    orig_length = seq_len
    # Increase length to avoid degeneracy.
    seq_len = tf.maximum(seq_len, 2)

    num_noise_tokens = to_int(tf.round(to_float(seq_len) * noise_density))
    # Avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), seq_len - 1)
    num_nonnoise_tokens = seq_len - num_noise_tokens

    num_noise_spans = to_int(tf.round(to_float(num_noise_tokens) / mean_noise_span_length))
    # Avoid degeneracy by ensuring positive number of noise spans.
    num_noise_spans = tf.maximum(num_noise_spans, 1)

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = tf.reshape(
        tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = tf.cumsum(interleaved_span_lengths, axis=0)[:-1]
    span_starts_indicator = tf.math.unsorted_segment_sum(
        tf.ones_like(span_starts), span_starts, seq_len
    )
    span_num = tf.cumsum(span_starts_indicator, axis=0)
    is_noise = tf.equal(span_num % 2, 1)
    return is_noise[:orig_length]


def _noise_span_to_unique_sentinel(
    *, input_ids: tf.Tensor, noise_mask: tf.Tensor, vocab: seqio.Vocabulary
) -> tf.Tensor:
    """Replaces each run of consecutive noise tokens with a different sentinel.

    The idea here is to be able to align the dropped spans in the inputs with the markers in the
    targets.

    We want to generate training examples like:
    "We hold X to be Y that" -> "X these truths Y self evident Z"

    Sentinels are assigned in decreasing order within the sequence starting at `vocab.size - 1`.
    That is, we appropriate the last tokens in the vocabulary for additional use as sentinels.
    Note that one can add extra IDs to the vocab for this purpose.

    Reference:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/006196f21db316698bda30862f327bbdf195fc2b/t5/data/preprocessors.py#L2869

    Args:
        input_ids: Input token IDs.
        noise_mask: A boolean Tensor with same shape as `input_ids`.
        vocab: A seqio vocab.

    Returns:
        A Tensor with the same dtype as `input_ids`.
    """
    prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])
    first_noise_tokens = tf.logical_and(noise_mask, tf.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

    # Note: vocab.vocab_size includes extra_ids.
    sentinel_id = vocab.vocab_size
    sentinel = sentinel_id - tf.cumsum(tf.cast(first_noise_tokens, input_ids.dtype), axis=0)

    input_ids = tf.where(first_noise_tokens, sentinel, input_ids)
    return tf.boolean_mask(input_ids, tf.logical_not(subsequent_noise_tokens))


def _nonnoise_span_to_unique_sentinel(
    *, input_ids: tf.Tensor, noise_mask: tf.Tensor, vocab: seqio.Vocabulary
) -> tf.Tensor:
    """Refer to `noise_span_to_unique_sentinel`.

    Reference:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/006196f21db316698bda30862f327bbdf195fc2b/t5/data/preprocessors.py#L2910
    """
    return _noise_span_to_unique_sentinel(
        input_ids=input_ids, noise_mask=tf.logical_not(noise_mask), vocab=vocab
    )


def apply_t5_mask(
    *,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    source_key: str = "source_ids",
    target_key: str = "target_labels",
    vocab_cfg: InstantiableConfig,
) -> input_tf_data.DatasetToDatasetFn:
    """Applies masking to the T5 inputs. Any masked span is replaced by a unique sentinel
    token. The mask for the target is exactly the inverse of the mask for the source.

    Args:
        noise_density: Approximate density of output mask.
        mean_noise_span_length: Average length of masked spans.
        source_key: Name of the key corresponding to the input of encoder.
        target_key: Name of the key corresponding to the ground truth of decoder.
        vocab_cfg: Config to instantiate the seqio vocab.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing key "source_ids"
        with int32 Tensors as values, and each output example is a dict with "source_ids", and
        "target_labels" as keys and int32 tensors as values.
    """
    vocab = vocab_cfg.instantiate()

    @seqio.map_over_dataset
    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        source_ids = example[source_key]
        seq_len = tf.shape(source_ids)[0]
        noise_mask = random_spans_noise_mask(
            noise_density=noise_density,
            seq_len=seq_len,
            mean_noise_span_length=mean_noise_span_length,
        )
        example[source_key] = _noise_span_to_unique_sentinel(
            input_ids=source_ids, noise_mask=noise_mask, vocab=vocab
        )
        example[target_key] = _nonnoise_span_to_unique_sentinel(
            input_ids=source_ids, noise_mask=noise_mask, vocab=vocab
        )
        return example

    return process_example_fn


def map_prefix_to_value(is_training: bool, value: int = 0) -> input_tf_data.DatasetToDatasetFn:
    """Remaps "prefix" key to `value`. Set to 0 to match T5X behavior.

    Note: This must be done after packing, since seqio has bugs when packing with 0's.
    """

    @seqio.map_over_dataset
    def process_eval_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        example["prefix"] = tf.constant([value])
        return example

    @seqio.map_over_dataset
    def process_train_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        # Handle packed inputs by replacing at segment starts.
        if "positions" in example["target"]:
            target_ids = example["target"]["input_ids"]
            target_positions = example["target"]["positions"]
            example["target"]["input_ids"] = tf.where(target_positions == 0, value, target_ids)

        # Handle padded inputs by replacing at index 0.
        else:
            example["target"]["input_ids"] = tf.concat(
                [[value], example["target"]["input_ids"][1:]], 0
            )
        return example

    return process_train_fn if is_training else process_eval_fn


def make_t5_autoregressive_inputs(
    *,
    model_type: ModelType = ModelType.ENCODER_DECODER,
    vocab_cfg: InstantiableConfig,
    max_source_length: int,
    max_target_length: int,
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    max_chunk_size: int = 65536,
    num_sequences_to_concat: int = 128,
) -> input_tf_data.DatasetToDatasetFn:
    """Generates autoregressive inputs via the T5 span corruption objective.

    Reference:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/9860243bbd8a93fb284edbe9d9abfc1de40b7bc8/t5/data/preprocessors.py#L1914

    Notes:
    - This processor is intended to be used with `input_lm.text2text_lm_input(with_eos=False, ...)`
        (see corresponding docstring for details).
    - The denoising mapper is expected to follow the signature of `input_mlm.apply_mlm_mask` (see
        corresponding docstring for details). It consumes "source_ids" as inputs and produces
        "source_ids" (with some tokens masked) and "target_labels" as outputs.

    Args:
        model_type: Must be ENCODER_DECODER.
        vocab_cfg: A config that instantiates to a seqio vocab.
        max_source_length: The maximum number of tokens in encoder input.
        max_target_length: The maximum number of tokens in decoder input.
        noise_density: Approximate density of output mask.
        mean_noise_span_length: Average length of masked spans.
        max_chunk_size: Max size of each input document.
        num_sequences_to_concat: Number of consecutive sequences to concat.

    Returns:
        A DatasetToDatasetFn which consumes "source_ids" (without BOS or EOS) as input, and emits
        "source_ids" (with some tokens replaced with sentinels, trimmed to max_source_length, with
        EOS but not BOS) and "target_labels" (trimmed to max_target_length with EOS) by applying a
        span corruption mapper.

    Raises:
        NotImplementedError: If model_type is unsupported.
        ValueError: If the noised targets exceed `max_target_length`.
    """
    if model_type != ModelType.ENCODER_DECODER:
        raise NotImplementedError("Expected ENCODER_DECODER model_type")
    vocab = vocab_cfg.instantiate()
    source_length, target_length = random_spans_helper(
        desired_source_length=max_source_length,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
    )
    # Sanity check that the given configs don't require targets to exceed max_target_length.
    if target_length > max_target_length:
        raise ValueError(
            f"Expected target length for span corruption ({target_length}) "
            f"exceeds max target length ({max_target_length})."
        )
    return input_tf_data.chain(
        # For each sequence longer than `chunk_size`, select a random span of `chunk_size` tokens.
        # Otherwise, keep the entire sequence.
        config_for_function(select_random_chunk).set(chunk_size=max_chunk_size),
        # Pack every `batch_size` sequences into a long one.
        config_for_function(reduce_concat_tokens).set(batch_size=num_sequences_to_concat),
        # Split the sequences so that the maximum length of encoded inputs is `source_length`.
        config_for_function(split_tokens).set(max_tokens_per_segment=source_length),
        # Apply noise mask.
        config_for_function(apply_t5_mask).set(
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            vocab_cfg=vocab_cfg,
        ),
        # Trim to max length before appending EOS, ensuring that EOS is always present.
        functools.partial(
            seqio.preprocessors.append_eos_after_trim,
            output_features={
                "source_ids": seqio.Feature(vocab, add_eos=True, dtype=tf.int32),
                "target_labels": seqio.Feature(vocab, add_eos=True, dtype=tf.int32),
            },
            sequence_length={"source_ids": max_source_length, "target_labels": max_target_length},
        ),
        # The above processing drops input keys besides "source_ids", so we add "prefix" afterwards.
        input_tf_data.add_static_fields({"prefix": tf.constant([vocab.eos_id])}),
        # Generate "target_ids" from "prefix" and "target_labels".
        config_for_function(make_autoregressive_inputs),
    )
