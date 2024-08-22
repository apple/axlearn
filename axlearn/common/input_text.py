# Copyright © 2023 Apple Inc.
#
# huggingface/transformers:
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# facebookresearch/fairseq:
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Licensed under the MIT license.

"""Generic text input processing utilities.

Note that while tfds is used, Hugging Face datasets can be loaded using `huggingface:` prefix.
See: https://www.tensorflow.org/datasets/community_catalog/huggingface
"""
import functools
import random
from collections.abc import Sequence
from typing import Callable, Optional, Union

import nltk
import numpy as np
import numpy.typing
import seqio
import tensorflow as tf
import tensorflow_text as tf_text
from seqio import map_over_dataset

# pylint: disable-next=no-name-in-module
from tensorflow.python.ops import string_ops
from tensorflow_text.python.ops.bert_tokenizer import AccentPreservingBasicTokenizer

from axlearn.common import input_tf_data
from axlearn.common.config import ConfigOr, maybe_instantiate
from axlearn.common.input_tf_data import DatasetToDatasetFn
from axlearn.common.utils import Tensor

INPUT_IDS = "input_ids"
TOKEN_TYPE_IDS = "token_type_ids"
TARGET_LABELS = "target_labels"


def perplexity(targets: Sequence[str], scores: Sequence[int]) -> dict[str, seqio.metrics.Scalar]:
    """Computes perplexity metric.

    Reference: https://github.com/google/seqio#score-metrics

    Args:
        targets: postprocessed targets
        scores: log-likelihood scores for each example according to the model
    """
    del targets
    return {"perplexity": seqio.metrics.Scalar(np.exp(np.mean(scores)))}


def preprocess_chunks(chunk_size: int) -> input_tf_data.DatasetToDatasetFn:
    """Preprocess documents by splitting them into chunks of size <= chunk_size,
    where size is determined by UTF-8 characters.

    TODO(markblee): support different types of chunking modes.

    Each original example is assumed to be a full document with a "text" field.
    No padding is applied to the resulting chunks.

    Args:
        chunk_size: size of chunks in UTF-8 characters.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing key "text" with
        string values, and each output example is a dict with key "text" mapping to each chunk.
    """

    def split_chunks_by_character(text: tf.Tensor) -> Sequence[str]:
        text = bytes.decode(text.numpy(), "utf-8")
        chunks = []
        for k in range(0, len(text), chunk_size):
            chunks.append(text[k : k + chunk_size])
        # Return a np.array so tf doesn't flatten into one string.
        return np.array(chunks)

    def process_dataset_fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(
            lambda x: {"text": tf.py_function(split_chunks_by_character, [x["text"]], tf.string)},
            num_parallel_calls=tf.data.AUTOTUNE,
        ).unbatch()

    return process_dataset_fn


def strip_accents(
    fields: Sequence[str], normalization_form: str = "NFKD"
) -> input_tf_data.DatasetToDatasetFn:
    """Strips accents from fields.

    Args:
        fields: Fields to strip accents from.
        normalization_form: One of "NFD" or "NFKD".

    Returns:
        A DatasetToDatasetFn that removes accents from the fields.
    """

    assert normalization_form in ("NFD", "NFKD")

    def example_fn(example: dict[str, str]) -> dict[str, str]:
        for field in fields:
            normalized_utf8 = tf_text.normalize_utf8(
                example[field], normalization_form=normalization_form
            )
            # Reference:
            # https://github.com/tensorflow/text/blob/4887f2034678bbac662040eeff6ec22d69724361/tensorflow_text/python/ops/bert_tokenizer.py#LL151C55-L151C69.
            # pylint: disable-next=line-too-long
            # \p{Mn} is "a character intended to be combined with another character without taking up extra space".
            accent_removed = string_ops.regex_replace(normalized_utf8, r"\p{Mn}", "")
            example[field] = accent_removed
        return example

    return seqio.map_over_dataset(example_fn)


def tokenize(
    output_features: ConfigOr[seqio.preprocessors.OutputFeaturesType],
    copy_pretokenized: bool = True,
    with_eos: bool = False,
) -> input_tf_data.DatasetToDatasetFn:
    """Tokenizes output_features using SeqIO.

    Args:
        output_features: A mapping from field name to seqio.preprocessors.OutputFeaturesType.
            Their vocabulary attribute will be used to tokenize the specified feature.
            Alternatively, a config which instantiates to one.
        copy_pretokenized: Whether to pass through copies of original features
            with "_pretokenized" suffix added to the key.
        with_eos: Whether to append EOS to the end of the sequence.

    Returns:
        A DatasetToDatasetFn which tokenizes the fields specified in output_features.
    """
    output_features = maybe_instantiate(output_features)
    return functools.partial(
        seqio.preprocessors.tokenize,
        output_features=output_features,
        copy_pretokenized=copy_pretokenized,
        with_eos=with_eos,
    )


def add_token_type_ids(
    input_key: Union[str, Sequence[str]],
    output_key: str = TOKEN_TYPE_IDS,
) -> input_tf_data.DatasetToDatasetFn:
    """Add token type IDs to each example of one or multiple sequences.

    This processor is usually used after `tokenize`.
    The last dimension should contain the input IDs.
    Note sequences should either be 1D `tf.Tensor`s or `tf.RaggedTensor`s containing the input IDs
    for single examples, or 2D `tf.Tensor`s or `tf.RaggedTensor`s containing the input IDs for
    multiple examples, where the first dim is the batch dimension and second dim is the input IDs.

    Assume final concatenated sequences always start and end with BOS/EOS, and sub-sequences
    are separated by a single separator token.
    Tokens belonging to the i-th sequence (0-based) will have token type ID of `i`.
    The BOS token will have token type ID of 0, and EOS will have token type ID of `i`.
    The token type ID of the separator will be the same as that of the preceding sequence,
    matching the behavior of Hugging Face.

    Ref:
    https://github.com/huggingface/transformers/blob/6f79d264422245d88c7a34032c1a8254a0c65752/src/transformers/models/bert/tokenization_bert.py#L320-L347

    Args:
        input_key: Field name(s) of one or multiple tokenized sequences to be added
            token type IDs.
        output_key: Name of the field that contains token type IDs to be outputted.

    Returns:
        A DatasetToDatasetFn creating token type IDs in the `output_key`.
    """
    if isinstance(input_key, str):
        input_key = [input_key]

    def example_fn(
        example: dict[str, Union[tf.Tensor, tf.RaggedTensor]]
    ) -> dict[str, Union[tf.Tensor, tf.RaggedTensor]]:
        token_type_ids = []
        for i, key in enumerate(input_key):
            t = example[key]
            # In either branch, the total number of tokens with token type ID i is
            #   int(i == 0) + seq_len + 1.
            # For i == 0, the first +1 is to account for the BOS token, the last +1 is to account
            # for SEP if there are multiple sequences or EOS if there is only one sequence.
            # For 1 <= i < len(input_key) - 1, the +1 is to account for the SEP token.
            # For i == len(input_key) - 1, the +1 is to account for the EOS token.
            if isinstance(t, tf.RaggedTensor):
                # seq_len is a 1D tensor indicating the length of each sequence.
                seq_len = t.row_lengths()
                shape_out = int(i == 0) + seq_len + 1
                total_num_elements = tf.math.reduce_sum(shape_out)
                out = tf.RaggedTensor.from_row_lengths(tf.fill((total_num_elements,), i), shape_out)
            else:
                shape = tf.shape(t)
                # seq_len is a scalar indicating the length of each sequence.
                seq_len = shape[-1]
                # pylint: disable-next=no-value-for-parameter,unexpected-keyword-arg
                shape_out = tf.concat([shape[:-1], [int(i == 0) + seq_len + 1]], axis=-1)
                out = tf.fill(shape_out, i)
            token_type_ids.append(out)

        # pylint: disable-next=no-value-for-parameter,unexpected-keyword-arg
        example[output_key] = tf.concat(token_type_ids, axis=-1)
        return example

    return seqio.map_over_dataset(example_fn)


def tokenize_example(
    text: tf.Tensor,
    sp_vocab: seqio.vocabularies.SentencePieceVocabulary,
    replace_newlines_with: str,
) -> tf.Tensor:
    """Tokenizes text for a single example using sentence piece vocabulary.

    Does not apply padding or special tokens (BOS/EOS).

    Args:
        text: Input text as a string tensor.
        sp_vocab: Sentencepiece vocabulary used to tokenize.
        replace_newlines_with: Value to replace "\n" in input text with before tokenizing.

    Returns:
        A tensor with int32 token ids.
    """
    text = tf.strings.regex_replace(text, "\n", replace_newlines_with)
    return sp_vocab.encode_tf(text)


def num_bytes(
    ids: tf.Tensor,
    sp_vocab: seqio.SentencePieceVocabulary,
    newlines_replaced_with: str = "\n",
) -> tf.Tensor:
    """Compute the number of bytes contained in token IDs, treating EOS as 1 byte.

    Args:
        ids: The token ids. Can be a 1D or 2D tensor.
        sp_vocab: A seqio wrapped sentence piece tokenizer.
        newlines_replaced_with: The string value that newlines have been replaced with.

    Returns:
        The number of bytes contained in ids. If `ids` is 1D, the return value will be a scalar
        tensor. If `ids` have shape [batch_size, :], the return value will have shape [batch_size].
    """
    # Revert newlines before calculating bytes.
    # pylint: disable-next=protected-access
    text = tf.strings.regex_replace(sp_vocab._decode_tf(ids), newlines_replaced_with, "\n")
    token_bytes = tf.strings.length(text, unit="BYTE")
    # Add the number of EOS tokens and expand to have a leading dimension.
    tf_eos_id = tf.constant(sp_vocab.eos_id, dtype=tf.int32)
    return token_bytes + tf.reduce_sum(tf.cast(ids == tf_eos_id, dtype=tf.int32), axis=-1)


# We pad upcast byte arrays using this value.
STRINGS_BYTE_ARRAY_PAD_VALUE = tf.int32.max


def pack_strings(
    strings: tf.Tensor,
    max_packed_byte_array_len: int,
) -> tf.Tensor:
    """Packs tensor of strings into a padded upcast to int32 utf-8 byte block
        to allow for JAX all-process communication collectives.

    We pad with STRINGS_BYTE_ARRAY_PAD_VALUE so as not to clash with a byte.

    Args:
        strings: Input string tensor.
        max_packed_byte_array_len: The maximum number of bytes a given string could occupy.

    Returns:
        A packed int32 tensor representing utf-8 bytes padded to max_packed_byte_array_len,
            of shape [strings.shape[0], max_packed_byte_array_len].
    """
    # Get ragged tensor of int32 utf-8 bytes.
    ragged_packed_strings = tf.strings.unicode_decode(
        strings, input_encoding="UTF-8", errors="strict"
    )
    string_byte_lengths = ragged_packed_strings.row_lengths()
    tf_max_packed_byte_array_len = tf.constant(max_packed_byte_array_len, dtype=tf.int64)
    # Check that max_packed_byte_array_len is sufficient.
    tf.debugging.assert_less_equal(tf.reduce_max(string_byte_lengths), tf_max_packed_byte_array_len)
    packed_strings = ragged_packed_strings.to_tensor(default_value=STRINGS_BYTE_ARRAY_PAD_VALUE)
    # pylint: disable-next=unexpected-keyword-arg,no-value-for-parameter
    padded_packed_strings = tf.concat(
        (
            packed_strings,
            tf.fill(
                (tf.shape(strings)[0], max_packed_byte_array_len - tf.shape(packed_strings)[1]),
                STRINGS_BYTE_ARRAY_PAD_VALUE,
            ),
        ),
        axis=1,
    )
    return padded_packed_strings


def unpack_strings(strings_byte_array: Tensor) -> list[str]:
    """Python logic to extract strings from a utf-8 byte array upcast to int32
        and padded with STRINGS_BYTE_ARRAY_PAD_VALUE.

    Args:
        strings_byte_array: [num_strings, max_packed_byte_array_len] int32 array.

    Returns:
        List of strings.
    """
    strings = []
    for byte_string in strings_byte_array:
        # Exclude padded bytes and encode back to utf-8.
        byte_string = byte_string[byte_string != STRINGS_BYTE_ARRAY_PAD_VALUE]
        string = tf.strings.unicode_encode(byte_string, output_encoding="UTF-8", errors="strict")
        strings.append(string.numpy().decode("utf-8"))
    return strings


def infer_bos_id(vocab: seqio.SentencePieceVocabulary) -> int:
    """Infer BOS for vocabs that don't have one.

    Args:
        vocab: the seqio vocabulary, which may or may not have a BOS token.

    Returns:
        The BOS token ID.

    Raises:
        ValueError: if no viable BOS token ID can be inferred.
    """
    # Use EOS as prompt if no BOS is available.
    bos_id = vocab.eos_id if vocab.tokenizer.bos_id() == -1 else vocab.tokenizer.bos_id()
    if bos_id == -1:
        raise ValueError(f"Cannot infer viable bos_id from seqio vocabulary: {vocab}")
    return bos_id


def roberta_normalize(
    cased: bool = True,
    input_key: Union[str, Sequence[str]] = "text",
) -> input_tf_data.DatasetToDatasetFn:
    """Normalize inputs following RoBERTa.

    In fairseq, text is for the most part passed directly to a BPE encoder,
    although we support lowercasing since some in-house trained tokenizers
    assume lowercase.

    Reference:
    https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/examples/roberta/README.pretraining.md
    https://github.com/pytorch/fairseq/blob/83e615d66905b8ca7483122a37da1a85f13f4b8e/examples/roberta/multiprocessing_bpe_encoder.py#L108
    https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/models/roberta/hub_interface.py#L35

    Args:
        cased: if False, lowercase the inputs.
        input_key: keys corresponding to input text fields to normalize.

    Returns:
        A DatasetToDatasetFn where each input example should be a dict containing key "text" with
        string values, and each output example is a dict with the same keys, but with normalization
        applied to the string value corresponding to "text".
    """
    if isinstance(input_key, str):
        input_key = [input_key]

    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        for key in input_key:
            text = example[key]
            if not cased:
                text = tf.strings.lower(text)
            text = tf.strings.strip(text)
            example[key] = text
        return example

    return seqio.map_over_dataset(process_example_fn)


def bert_normalize_mapper(
    cased: bool = True,
    input_key: Union[str, Sequence[str]] = "text",
    reduce_axis: Optional[int] = None,
) -> Callable[[dict[str, Union[str, tf.Tensor]]], dict[str, tf.Tensor]]:
    """Constructs a mapper for `bert_normalize`.

    This is kept as a separate function so we can use it in non-tf-dataset mappers,
    such as Hugging Face data pipelines.

    Args:
        cased: if False, lowercase the inputs.
        input_key: keys corresponding to input text fields to normalize.
        reduce_axis: Axis to join normalized tokens.
            If None, joins across all axes, i.e. the output is a scalar string.

    Returns:
        A mapper that applies normalization to the input example.
    """
    basic_tokenizer = AccentPreservingBasicTokenizer(lower_case=not cased, normalization_form=None)
    if isinstance(input_key, str):
        input_key = [input_key]

    def process_example_fn(example: dict[str, Union[str, tf.Tensor]]) -> dict[str, tf.Tensor]:
        for key in input_key:
            text = example[key]  # Note: TF uses utf-8 encoding by default.
            # Replace 0x0, 0xFFFD, and non-spacing marks, following original BERT implementation.
            # References:
            # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L291
            # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L226
            # pylint: disable-next=anomalous-backslash-in-string
            text = tf.strings.regex_replace(text, "[\\0\\x{FFFD}\\p{Mn}]+", "")
            tokens = basic_tokenizer.tokenize(text)
            if reduce_axis is None:
                tokens = tokens.flat_values
            text = tf.strings.reduce_join(tokens, separator=" ", axis=reduce_axis)
            if reduce_axis is None:
                text = tf.ensure_shape(
                    text, []
                )  # Enforce rank so we can tokenize in a subsequent step.
            example[key] = text
        return example

    return process_example_fn


def bert_normalize(
    cased: bool = True,
    input_key: Union[str, Sequence[str]] = "text",
    reduce_axis: Optional[int] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Normalize inputs following BERT.

    TODO(markblee): The `tensorflow_text` implementation seems to differ slightly from original
    bert/huggingface implementations in the case where we strip accents, but it's not a use-case
    at the moment, so the option is omitted.

    Args:
        cased: if False, lowercase the inputs.
        input_key: keys corresponding to input text fields to normalize.
            Other fields are dropped.
        reduce_axis: Axis to join normalized tokens.
            If None, joins across all axes, i.e. the output is a scalar string.

    Returns:
        A DatasetToDatasetFn where each input example should be a dict containing key "text" with
        string values, and each output example is a dict with the same keys, but with normalization
        applied to the string value corresponding to "text".
    """
    return seqio.map_over_dataset(
        bert_normalize_mapper(cased=cased, input_key=input_key, reduce_axis=reduce_axis)
    )


def split_sentences(
    input_key: str = "text",
    max_sentences_per_example: Optional[int] = None,
    passthrough_keys: Optional[list[str]] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Splits the input documents into sentences, and appends an empty sentence at document
    boundaries.

    Args:
        input_key: Input to split.
        max_sentences_per_example: Optionally limit the number of sentences from each example.
            Sampling is done without replacement, keeping original ordering of sentences.
        passthrough_keys: Keys to pass-through (replicate).

    Returns:
        A DatasetToDatasetFn where each input example should be a dict containing key "text" with
        string values representing full documents, and where each output example is a dict
        containing key "text" with string values representing full sentences.
    """
    nltk.download("punkt")
    passthrough_keys = passthrough_keys or []

    def sentence_tokenize(text: tf.Tensor) -> numpy.typing.ArrayLike:
        # List of strings. [num_sentences+1]
        sentences = nltk.tokenize.sent_tokenize(bytes.decode(text.numpy()).lower())
        # Sample a subset of sentences, excluding the appended document boundary.
        if max_sentences_per_example is not None and len(sentences) > max_sentences_per_example > 0:
            idxs = random.sample(range(len(sentences)), max_sentences_per_example)
            sentences = [sentences[i] for i in sorted(idxs)]  # Sort to keep original order.
        # Append an empty sentence to indicate document boundary.
        sentences.append("")
        # Return a np.array so tf doesn't flatten into one string.
        return np.array(sentences), len(sentences)

    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        sentences, shape = tf.py_function(
            sentence_tokenize, [example[input_key]], (tf.string, tf.int32)
        )
        output = {input_key: tf.ensure_shape(sentences, [None])}
        for key in passthrough_keys:
            output[key] = tf.tile([example[key]], [shape])
        return output

    return lambda ds: ds.map(process_example_fn, num_parallel_calls=tf.data.AUTOTUNE).unbatch()


def random_chunking(max_len: int, input_key: str = INPUT_IDS) -> input_tf_data.DatasetToDatasetFn:
    """Picks a random chunk of a 1D Tensor.

    We always take a chunk of `min(size, max_len)`, where `size` is the length of the input and
    `max_len` is the maximum chunk size (where the actual chunk size can be smaller if the input
    itself is shorter than `max_len`).

    Args:
        max_len: maximum size of the chunk.
        input_key: the name of the key in each example to apply chunking to.

    Returns:
        An DatasetToDatasetFn where each input example should be a dict containing key `input_key`
        corresponding to token IDs, and each output is a dict containing the same keys, but with
        `input_key` mapping to the randomly-selected chunk of token IDs.
    """

    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        input_ids = example[input_key]
        size = tf.size(input_ids)
        max_offset = tf.maximum(0, size - max_len)
        offset = tf.random.uniform([], minval=0, maxval=max_offset + 1, dtype=tf.int32)
        example[input_key] = input_ids[offset : offset + max_len]
        return example

    return seqio.map_over_dataset(process_example_fn)


def join_string_features(
    features: Sequence[str], key: str, separator: str = "</s>"
) -> DatasetToDatasetFn:
    """Builds a sequence from the given string features where features are separated by separator.

    Args:
        features: a list of field names to build the sequence from.
        key: the key to store the final joined sequence under.
        separator: a string to insert between features.

    Returns:
        A DatasetToDatasetFn that takes each input example and generates a string from the given
        features and separator and adds to the example as example[key].
    """

    def example_fn(example: dict[str, Tensor]) -> dict[str, Tensor]:
        example[key] = tf.strings.join(
            [example[feature] for feature in features], separator=separator
        )
        return example

    return map_over_dataset(example_fn)
