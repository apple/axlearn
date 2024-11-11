# Copyright © 2023 Apple Inc.

"""Input processing for ASR."""

from typing import Any, Optional

import seqio
import tensorflow as tf

from axlearn.common import input_text, input_tf_data
from axlearn.common.config import InstantiableConfig
from axlearn.common.utils import Nested


def speech_input(
    *,
    max_len: int,
    input_key: str = "speech",
    normalize_by_scale: Optional[float] = None,
    truncate: bool = False,
) -> input_tf_data.DatasetToDatasetFn:
    """Speech-only input processor for converting audio into fixed length sequences.

    1. If `normalize_by_scale` is not None, normalize inputs keyed at `input_key`.
    2. Filter out empty examples and examples that would otherwise be truncated if
        truncate is False.
    3. Apply padding and truncation. This also injects `paddings`, where padding positions are
        indicated by 1's.

    Args:
        max_len: Maximum sequence length. Examples exceeding `max_len` will be filtered if truncate
            is False, truncated to max_len if truncate is True.
        input_key: Key in each example corresponding to the audio inputs.
        normalize_by_scale: If not None, normalize the speech by dividing by this scale.
        truncate: Whether to allow speech to be truncated.

    Returns:
        A DatasetToDatasetFn.
        * Each input example is a dict containing `input_key` with float Tensor values (and possibly
            other keys) of shape [audio_len];
        * Each output example contains "inputs" corresponding to `input_key` with values possibly
            padded, truncated, and (optionally) normalized, as well as "paddings", a 0/1 int Tensor
            with 1's at padding positions. All other keys besides `input_key` will be kept
            unchanged.
    """
    if max_len < 1:
        raise ValueError("Max length must be at least 1.")

    @seqio.map_over_dataset
    def segments_to_paddings(example: dict[str, tf.Tensor]):
        inputs = example.pop(input_key)
        lengths = tf.shape(inputs)[-1]
        inputs = input_tf_data.trim_and_pad_tensor(inputs, max_len=max_len)
        example["inputs"] = tf.cast(inputs, tf.float32)
        example["paddings"] = tf.cast(tf.range(max_len) >= lengths, tf.int32)
        return example

    processors = []
    if normalize_by_scale:
        processors.append(_normalize_by_scale(input_key=input_key, scale=normalize_by_scale))
    # Filter inputs that are too short, and inputs that would otherwise be truncated
    # if truncate=False.
    filter_max_len = None if truncate else max_len
    processors.append(_filter_by_length(input_key=input_key, min_len=1, max_len=filter_max_len))
    return input_tf_data.chain(*processors, segments_to_paddings)


def text_input(
    *,
    max_len: int,
    vocab: InstantiableConfig[seqio.Vocabulary],
    input_key: str = "text",
    truncate: bool,
    min_len: int = 1,
    eos_id: Optional[int] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Processor for text targets.

    1. Form "target_labels" by tokenizing `input_key` without EOS appended.
    2. Drop examples where length of "target_labels" is strictly less than `min_len`.
    3. If `truncate=False`, drop examples whose lengths including EOS would be greater than
        `max_len`.
    4. Append EOS and apply padding (and if `truncate=True`, apply truncation). Note that EOS is
        appended prior to truncation. We pad with -1, an out-of-vocab token ID.

    If autoregressive "input_ids" are desired (e.g. for LAS-style inputs), chain with
    `make_autoregressive_inputs`.

    Args:
        max_len: Maximum sequence length after padding and (optionally) truncation. See above for
            details on how examples exceeding `max_len` are treated.
        vocab: Vocabulary for tokenization.
        input_key: Key in each example corresponding to the text inputs.
        truncate: Whether to allow text to be truncated.
        min_len: Minimum length of "target_labels". For instance, `min_len=1` means examples with
            empty "target_labels" are dropped.
        eos_id: EOS token ID. If None, infers from `vocab.eos_id`.

    Returns:
        A DatasetToDatasetFn.
        * Each input example is a dict containing `input_key` with text values (and possibly
            other keys);
        * Each output example contains "target_labels" corresponding to tokenized `input_key` with
            values possibly padded or truncated. "target_labels" may contain a trailing EOS token,
            if it has not been truncated. All other keys will be kept unchanged.
    """
    if max_len < 1:
        raise ValueError("Max length must be at least 1.")

    spm_vocab = vocab.instantiate()
    if eos_id is None:
        eos_id = spm_vocab.eos_id
    pad_id = -1

    @seqio.map_over_dataset
    def trim_and_pad_with_eos(example: Nested[tf.Tensor]) -> Nested[tf.Tensor]:
        labels = tf.concat([example["target_labels"], [eos_id]], -1)
        labels = input_tf_data.trim_and_pad_tensor(labels, max_len=max_len, pad_id=pad_id)
        example["target_labels"] = labels
        return example

    processors = [
        input_tf_data.rekey({"target_labels": input_key}, retain_original_inputs=True),
        input_text.tokenize(
            {"target_labels": seqio.Feature(spm_vocab)},
            copy_pretokenized=False,
        ),
        _filter_by_length(input_key="target_labels", min_len=min_len),
    ]
    if not truncate:
        processors.append(_filter_by_length(input_key="target_labels", max_len=max_len - 1))

    return input_tf_data.chain(*processors, trim_and_pad_with_eos)


def make_autoregressive_inputs(
    *,
    vocab: InstantiableConfig[seqio.Vocabulary],
    bos_id: int,
) -> input_tf_data.DatasetToDatasetFn:
    """Forms "input_ids" by right-shifting "target_labels" and prepending BOS.

    Args:
        vocab: Vocabulary for tokenization.
        bos_id: BOS token ID.

    Returns:
        A processor that applies processing described above.
    """
    spm_vocab = vocab.instantiate()
    bos_id = spm_vocab.vocab_size if bos_id is None else bos_id

    @seqio.map_over_dataset
    def process_example_fn(example: Nested[tf.Tensor]) -> Nested[tf.Tensor]:
        example["input_ids"] = tf.concat([[bos_id], example["target_labels"][:-1]], 0)
        return example

    return process_example_fn


def _filter_by_length(
    *,
    input_key: str,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Filters examples where shape[-1] not in [min_len, max_len]."""

    def process_example_fn(example: Nested[tf.Tensor]) -> bool:
        shape = tf.shape(example[input_key])[-1]
        lower_bound = min_len is None or min_len <= shape
        upper_bound = max_len is None or shape <= max_len
        return lower_bound and upper_bound

    return lambda ds: ds.filter(process_example_fn)


def _normalize_by_scale(*, input_key: str, scale: float) -> input_tf_data.DatasetToDatasetFn:
    """Normalizes inputs by the provided scale."""
    if scale <= 0:
        raise ValueError(f"scale must be strictly positive, got {scale}.")

    def process_example_fn(example: Nested[tf.Tensor]) -> Nested[tf.Tensor]:
        example[input_key] = tf.cast(example[input_key], tf.float32) / scale
        return example

    return seqio.map_over_dataset(process_example_fn)


def pad_example_fn(element_spec: Nested[Any]) -> Nested[Any]:
    """Returns padding ASR examples.

    Each padding example contains:
    * A "source/paddings" key consisting of an all 1's Tensor;
    * "target/input_ids" and "target_labels" containing all -1s.
    """
    example = input_tf_data.default_pad_example_fn(element_spec)
    # Set source paddings to 1s.
    example["source"]["paddings"] = tf.ones_like(example["source"]["paddings"], dtype=tf.int32)
    # Set text tokens to -1s.
    example["target"]["input_ids"] = -1 * tf.ones_like(
        example["target"]["input_ids"], dtype=tf.int32
    )
    example["target_labels"] = -1 * tf.ones_like(example["target_labels"], dtype=tf.int32)
    return example
