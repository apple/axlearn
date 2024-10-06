# Copyright © 2023 Apple Inc.

# pylint: disable=too-many-lines
"""Input processing for language modeling."""

import enum
import functools
from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal, Optional

import seqio
import tensorflow as tf
from absl import logging

from axlearn.common import input_tf_data
from axlearn.common.config import InstantiableConfig, config_for_function, maybe_set_config
from axlearn.common.input_text import num_bytes, tokenize, tokenize_example
from axlearn.common.input_tf_data import rekey

# Value of "target_labels" which will be ignored in seq2seq processing.
SEQ2SEQ_IGNORE_TARGET_LABEL = -1


class InputDataType(Enum):
    """Represents input data types for decoder-only language model training.

    Values:
        LM: Text contained in a single field. Standard format for decoder-only
            language model training.
        SEQ2SEQ_MASK: Seq2seq data with input text and output text contained in separate fields.
            Mask (drop) the loss on the input tokens during training.
        SEQ2SEQ_NO_MASK: Seq2seq data with input text and output text contained in separate fields.
            Calculate the loss on both the input and output tokens during training.
    """

    LM = "lm"
    SEQ2SEQ_MASK = "seq2seq_mask"
    SEQ2SEQ_NO_MASK = "seq2seq_no_mask"


class PackingMethodType(Enum):
    """Represents methods of packing multiple documents into a single sequence.

    Values:
        EOS_DELIM_NO_MASK: Documents within a sequence are delimited with an EOS token. No
            self-attention masks between documents.
        EOS_DELIM_MASK: Documents within a sequence are delimited with an EOS token. Inject
            `input_segment_ids` and `input_positions` for self-attention masks in a causal LM.
    """

    EOS_DELIM_NO_MASK = "eos_delim_no_mask"
    EOS_DELIM_MASK = "eos_delim_mask"


def text_to_lm_training_input(
    vocab_cfg: InstantiableConfig,
    max_len: int,
    replace_newlines_with: str = "\n",
    window_size: int = 128,
    max_padding_fraction: float = 1,
    shuffle_buffer_size: int = 1024,
    is_training: bool = True,
    token_adjuster_cfg: Optional[InstantiableConfig] = None,
    packing_method: Optional[PackingMethodType] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Returns a function that generates training inputs for language models from raw text.

    Strategy:
        1. Repeat the input dataset if is_training to avoid truncation.
        2. Tokenize and concat a window of input examples into a single token stream,
            starting & ending with an EOS separator and including EOS between each document.
        3. Apply token_adjuster, if supplied.
        4. Either trim tokens from the end, or pad, so that final concat stream divides max length.
            If we have an example which would require < max_len * max_padding_fraction padding,
            keep & pad, else drop that example.
        5. Pack into a batch by reading in max length chunks.
            Each max length chunk may span >= 1 document.
        6. Unbatch to yield a single example of max length at a time.
        7. Apply shuffle with provided buffer size so that we read a randomized
            slice out of the data. This stops us from e.g. reading consecutive
            slices out of the same book at training time.

    Args:
        vocab_cfg: Config to instantiate the seqio vocab.
        max_len: the maximum number of tokens per sequence.
        replace_newlines_with: replace newlines with this value.
        window_size: the number of examples to pack before chunking into max_len sequences.
        max_padding_fraction: the maximum fraction of a batch example that we are willing to pad.
            E.g. if this is 0.5 then we will pad an example with >= 0.5 * max_len viable tokens,
            else drop it entirely.
        shuffle_buffer_size: The size of the shuffle buffer used to aggregate individual
            examples before sampling. N.B. must be sufficiently large to cover many documents
            sliced up, otherwise a single document at a time may dominate data feeding.
        is_training: whether the input is used for training (should be True).
        token_adjuster_cfg: pre-batching modifier for the token stream, used for transforms
            like fill in the middle (fitm).
        packing_method: method of packing multiple documents into a single sequence. If None,
            defaults to EOS_DELIM_NO_MASK.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing key "text" with
        string values and each output example is a dict with:
            * "input_ids" as int32 tensor with shape [max_len],
            * "target_labels" as int32 tensor with shape [max_len],
            * "target_num_bytes" as int32 scalar describing the number of bytes in "target_labels",
            * "input_segment_ids" as an optional int32 tensor with shape [max_len], present if
                packing_method is EOS_DELIM_MASK.
            * "input_positions" as an optional int32 tensor with shape [max_len], present if
                packing_method is EOS_DELIM_MASK.
    """
    if not is_training:
        logging.warning("is_training was %s, did you mean to use this processor?", is_training)
    vocab = vocab_cfg.instantiate()
    if token_adjuster_cfg is not None:
        token_adjuster = maybe_set_config(token_adjuster_cfg, vocab_cfg=vocab_cfg).instantiate()
    else:
        token_adjuster = None

    assert 0 <= max_padding_fraction <= 1.0

    def batch(ids: tf.Tensor) -> tf.Tensor:
        """Pads or truncates ids so as to divide max length, then group into temporary batch."""
        len_ids = tf.shape(ids)[0]
        remainder = len_ids % tf.constant(max_len)
        tf_pad_id = tf.constant(vocab.pad_id, dtype=tf.int32)
        # If the remainder isn't long enough to satisfy max_padding_fraction for a given example,
        # drop it, else pad to fill to max_len.
        new_ids = tf.cond(
            remainder > tf.constant(int(max_len * (1 - max_padding_fraction)), dtype=tf.int32),
            lambda: tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                (ids, tf.broadcast_to(tf_pad_id, shape=(tf.constant(max_len) - remainder,))),
                axis=0,
            ),
            lambda: ids[: len_ids - remainder],
        )
        return tf.reshape(new_ids, shape=(-1, max_len))

    def process_batched(inputs: dict[str, Any]) -> dict[str, Any]:
        """Chunks each jagged input window into a batch of equal-length training examples."""
        # A ragged tensor with dim 0 of size `window_size`.
        tokens = tokenize_example(
            inputs["text"],
            sp_vocab=vocab,
            replace_newlines_with=replace_newlines_with,
        )
        # Append EOS to every sequence.
        eos_id = tf.constant(vocab.eos_id, dtype=tf.int32)
        # pylint: disable-next=unexpected-keyword-arg,no-value-for-parameter
        tokens_with_eos = tf.concat([tokens, tf.fill([window_size, 1], eos_id)], axis=-1)
        # Concatenate all the sequences.
        # TODO(tom_gunter): Understand whether we should limit max tokens contributed by an eg.
        concat_tokens = tokens_with_eos.merge_dims(0, 1)

        # apply token_adjuster
        if token_adjuster is not None:
            concat_tokens = token_adjuster(concat_tokens, max_len=max_len)

        # Convert packed text -> batched examples of length `max_len`.
        batched_input_ids = batch(tf.roll(concat_tokens, 1, axis=0))

        batched_target_labels = batch(concat_tokens)
        target_num_bytes = num_bytes(
            batched_target_labels, sp_vocab=vocab, newlines_replaced_with=replace_newlines_with
        )

        result = dict(
            input_ids=batched_input_ids,
            target_labels=batched_target_labels,
            target_num_bytes=target_num_bytes,
        )

        if packing_method == PackingMethodType.EOS_DELIM_MASK:
            segment_starts = tf.cast(tf.equal(batched_input_ids, vocab.eos_id), tf.int32)
            segment_ids = tf.cumsum(segment_starts, axis=1)
            segment_ids = segment_ids - segment_ids[:, :1] + 1
            segment_ids = tf.where(tf.equal(batched_input_ids, vocab.pad_id), 0, segment_ids)

            assert segment_ids.shape[1] == max_len

            def cummax_on_last_axis(x):
                """tf.scan implementation of cummax."""
                assert len(x.shape) == 2
                # transpose to [seq, batch] for tf.scan, which iterates on dimension 0.
                x = tf.transpose(x)
                x = tf.scan(tf.maximum, x, initializer=tf.reduce_min(x, axis=0))
                x = tf.transpose(x)
                return x

            # segment_start_offsets[i] = i if it's the first position of a segment, otherwise 0.
            segment_start_offsets = segment_starts * tf.range(max_len)
            # segment_start_offsets[i] = j if j is the first position of a segment containing
            # position i.
            segment_start_offsets = cummax_on_last_axis(segment_start_offsets)
            # Within-segment positions.
            positions = tf.range(max_len) - segment_start_offsets
            # Mask out padded positions.
            positions = tf.where(tf.equal(batched_input_ids, vocab.pad_id), 0, positions)

            result["input_segment_ids"] = segment_ids
            result["input_positions"] = positions

        return result

    def process(ds: tf.data.Dataset) -> tf.data.Dataset:
        """Applies processing strategy to dataset."""
        if is_training:
            # Repeat the input dataset.
            ds = ds.repeat()

        # Ensure input text has a shape. Shape information can be lost if ds is created using a
        # `tfds.decode.Decoder` (e.g., if `decoders` is not None in `input_tf_data.tfds_dataset`).
        ds = ds.map(
            lambda inputs: {
                k: tf.ensure_shape(v, []) if k == "text" else v for k, v in inputs.items()
            },
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Group into sets of `window_size` documents and map document groups into batched token
        # stream.
        ds = ds.batch(window_size, drop_remainder=True).map(
            process_batched, num_parallel_calls=tf.data.AUTOTUNE
        )
        # Unbatch to yield one training example per iteration.
        ds = ds.unbatch()
        # Shuffle so that read order is not dominated by document order.
        if shuffle_buffer_size > 0:
            ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        return ds

    return process


def text_to_lm_eval_input(
    vocab_cfg: InstantiableConfig,
    max_len: int,
    replace_newlines_with: str = "\n",
    stride: Optional[int] = None,
    is_training: bool = False,
    index_key: Optional[str] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Returns a function that generates eval inputs for language models from raw text.

    The strategy:
        1. For each input example perform a strided slice.
        2. Pad the final slice to divide max length and pack into a temporary batch.
        3. Drop examples with 0 target bytes.
        4. Unbatch to yield a single example at a time.

    N.B For a given document, the first strided slice always contains all target labels.
    Further slices will mask all but the trailing stride of target labels by replacing with pad ID.
    This is so that we do not simply drop shorter than stride documents from evaluation,
    however differs from the common approach (popularized by the Compressive Transformer
    <https://arxiv.org/abs/1911.05507> on PG19), in which all documents are
    concatenated together and a rolling strided slice is used to produce batch elements.

    N.B. The most rigorous (and favorable to the LM) way to eval is to take a rolling slice
    with a stride of 1 and only score the model on the final token prediction.
    That way every predicted token has context of max length.
    In practice people usually do a rolling slice with a stride of 512 tokens,
    so that we score 512 tokens at once and each of those 512 has at least
    max_len - 512 context to rely on. For the first slice of a document we score all
    the tokens, even though the first token only has a context of 1 to rely on.

    Args:
        vocab_cfg: Config to instantiate the seqio vocab.
        max_len: the maximum number of tokens per sequence.
        replace_newlines_with: replace newlines with this value.
        stride: the stride to use when slicing a tokenized document into examples.
            If None, defaults to max length as stride.
        is_training: whether the input is used for training (should be False).
        index_key: if not None, key associated with uint32 source example index value.
            Useful for e.g. later grouping strided rolls by source example.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing key "text" with
        string values and each output example is a dict with:
            * "input_ids" as int32 tensor with shape [max_len],
            * "target_labels" as int32 tensor with shape [max_len] & repeated stride values
                masked with the vocabulary pad ID---so we don't count overlapping strides multiple
                times---see example below.
            * "target_num_bytes" as int32 scalar describing the number of bytes in "target_labels".
                EOS and the newline value are treated as 1 byte each.
            " <index_key> as uint32 representing passthrough index associated with each output.

        E.g for max_len=5, stride=2:
        text: "The cat sat on the mat."
        raw pieces: [EOS, '▁The', '▁cat', '▁sat', '▁on', '▁the', '▁mat', EOS]
        raw tokens: [  1,    388,   4421,   2064,   339,    268,   1077,   1]
        input_ids: [
            [   1,  388, 4421, 2064,  339],
            [4421, 2064,  339,  268, 1077],
        ]
        target_labels: [
            [ 388, 4421, 2064,  339,  268],
            [   0,    0,    0, 1077,    1],
        ]
        target_num_bytes: [18, 4]
    """
    if is_training:
        logging.warning("is_training was %s, did you mean to use this processor?", is_training)
    del is_training

    vocab = vocab_cfg.instantiate()
    stride: int = (
        max_len if stride is None else stride
    )  # We default to max length as the stride value.
    assert 0 < stride <= max_len

    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    def strided_slice(t: tf.Tensor, pad_stride: bool) -> tf.Tensor:
        """Returns possibly overlapping slices from the input tensor as a batch."""
        len_t = tf.shape(t)[0]
        # For the first slice we never pad the stride, so that short examples are included.
        output = tf.compat.v1.placeholder_with_default(t[:max_len], shape=(None,))
        for ix in tf.range(stride, len_t - max_len + 1, stride):
            t_slice = t[ix : ix + max_len]
            if pad_stride:
                # Mask out the strided segment with the pad ID.
                mask = tf.broadcast_to(
                    tf.constant([vocab.pad_id], dtype=tf.int32), (max_len - stride,)
                )
                t_slice = tf.concat(
                    (mask, t_slice[max_len - stride :]),
                    axis=0,
                )
            output = tf.concat((output, t_slice), axis=0)
        return tf.reshape(output, shape=(-1, max_len))

    def process_document(inputs: dict[str, Any]) -> dict[str, Any]:
        """Tokenizes & pads each document yielding a strided slice over the result."""
        # Tokenize.
        tf_eos_id = tf.constant([vocab.eos_id], dtype=tf.int32)
        tokens = tokenize_example(
            inputs["text"], sp_vocab=vocab, replace_newlines_with=replace_newlines_with
        )
        input_ids = tf.concat((tf_eos_id, tokens), axis=0)
        target_labels = tf.concat((tokens, tf_eos_id), axis=0)
        # Build padding.
        remainder = tf.shape(input_ids)[0] % max_len
        padding = tf.broadcast_to(
            tf.constant([vocab.pad_id], dtype=tf.int32), shape=(max_len - remainder,)
        )
        # Group.
        target_labels = strided_slice(tf.concat((target_labels, padding), axis=0), pad_stride=True)
        input_ids = strided_slice(tf.concat((input_ids, padding), axis=0), pad_stride=False)
        # Get target ids bytes and mask.
        target_num_bytes = num_bytes(
            target_labels, sp_vocab=vocab, newlines_replaced_with=replace_newlines_with
        )
        outputs = dict(
            input_ids=input_ids,
            target_labels=target_labels,
            target_num_bytes=target_num_bytes,
        )
        if index_key is not None:
            index = inputs[index_key]
            # We require a uint32 index as enabling 64 bit types on TPU causes all types to default
            # to 64 bit when possible.
            tf.debugging.assert_integer(index)
            tf.debugging.assert_less_equal(
                index, tf.constant(tf.dtypes.uint32.max, dtype=tf.uint32)
            )
            tf.debugging.assert_greater_equal(
                index, tf.constant(tf.dtypes.uint32.min, dtype=tf.uint32)
            )
            outputs[index_key] = tf.experimental.numpy.full_like(
                target_num_bytes, index, dtype=tf.uint32
            )

        # Drop examples that have 0 target bytes.
        return {k: tf.boolean_mask(v, target_num_bytes > 0, axis=0) for k, v in outputs.items()}

    def process(ds: tf.data.Dataset) -> tf.data.Dataset:
        """Applies processing strategy to dataset."""
        # Map each document into batched set of eval examples.
        ds = ds.map(process_document, num_parallel_calls=tf.data.AUTOTUNE)
        # Unbatch to yield one eval example per iteration.
        return ds.unbatch()

    # pylint: enable=unexpected-keyword-arg,no-value-for-parameter
    return process


class ModelType(str, enum.Enum):
    """Represents a type of language modeling.

    Values:
    - ENCODER_DECODER: Encoder decoder model type like T5/Bart/NMT.
    - DECODER_ONLY: Decoder only model type like GPT.
    """

    ENCODER_DECODER = "ENCODER_DECODER"
    DECODER_ONLY = "DECODER_ONLY"


def joint_truncation_for_seq2seq_lm_input(
    max_sequence_length: int,
) -> input_tf_data.DatasetToDatasetFn:
    """Processes an example by truncating the example["prefix"] + example["input_ids"] +
        example["target_labels"] from the left until len(example["prefix"] +
        example["input_ids"] + example["target_labels"]) - 1 <= max_sequence_length.

    Used before feeding into make_autoregressive_inputs for decoder_only models. -1 because
        the final input_ids after make_autoregressive_inputs would be
        concat(example["prefix"], example["input_ids"], example["target_labels"][:-1]),
        where the last token of example["target_labels"] is not included.

    Useful for few-shot learning setting where the input can be long and should be truncated
        from the left. We require that max_sequence_length to be greater than or equal to
        target_labels to ensure no truncation happens for target.

    Args:
        max_sequence_length: maximum sequence length of an example for truncation.

    Returns:
        A DatasetToDatasetFn that truncates the example according to the process described above.

    Raises:
        AssertionError: If max_sequence_length is less than the length of example["target_labels"].
    """

    @seqio.utils.map_over_dataset
    def process_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        target_labels = example["target_labels"]
        prefix = example["prefix"]  # This should match the "prefix" key used in decoding.
        input_ids = example["input_ids"]

        tf.debugging.assert_rank(target_labels, 1)
        tf.debugging.assert_rank(prefix, 1)
        tf.debugging.assert_rank(input_ids, 1)

        orig_target_labels_len = tf.size(target_labels)
        orig_input_ids_len = tf.size(input_ids)
        orig_prefix_len = tf.size(prefix)

        # Requires max_sequence_length to be greater than or equal to target_labels.
        tf.debugging.assert_less_equal(
            orig_target_labels_len,
            max_sequence_length,
            "max_sequence_length must be greater than target_labels length!",
        )
        total_tokens_to_truncate = tf.maximum(
            orig_input_ids_len + orig_target_labels_len + orig_prefix_len - 1 - max_sequence_length,
            0,
        )
        # Truncation starts from prefix.
        prefix_tokens_to_truncate = tf.minimum(total_tokens_to_truncate, orig_prefix_len)
        # Truncation continues on input_ids.
        input_ids_tokens_to_truncate = tf.minimum(
            total_tokens_to_truncate - prefix_tokens_to_truncate, orig_input_ids_len
        )
        # Truncate prefix from the left.
        prefix = prefix[prefix_tokens_to_truncate:]
        # Truncate input_ids from the left.
        input_ids = input_ids[input_ids_tokens_to_truncate:]
        # Under the above assertion, no target_labels should be truncated.
        return dict(
            prefix=prefix,
            target_labels=target_labels,
            input_ids=input_ids,
            prefix_tokens_to_truncate=prefix_tokens_to_truncate,
            input_ids_tokens_to_truncate=input_ids_tokens_to_truncate,
        )

    return process_fn


def make_autoregressive_inputs(
    model_type: ModelType = ModelType.ENCODER_DECODER,
    input_data_type: InputDataType = InputDataType.SEQ2SEQ_MASK,
    passthrough_keys: Optional[Sequence[str]] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Prepares autoregressive inputs.

    Notes:
    - This processor is intended to be used with `input_lm.text2text_lm_input(with_eos=True, ...)`
    (see corresponding docstring for details).

    Processing:
        Decoder-only: Concatenate "input_ids" and "target_labels" as "input_ids".
            If input_data_type == InputDataType.SEQ2SEQ_MASK, prepend "input_ids" as
                IGNORE_TARGET_LABEL in target_labels so they don't contribute to the loss.
            If input_data_type == InputDataType.SEQ2SEQ_NO_MASK, do not replace "input_ids"
                with IGNORE_TARGET_LABEL in target_labels so they contribute to the loss.
                However, we still replace "prefix" with IGNORE_TARGET_LABEL in target_labels
                so they don't contribute to the loss.
            "target_labels" is assumed to have EOS appended.

            Example:
                pad=0, bos=1, eos=2, ignore_target_label=-1

                if input_data_type == InputDataType.SEQ2SEQ_MASK:
                    Before: input_ids=[100, 101], target_labels=[102, 2], prefix=[1]
                    After: input_ids=[1, 100, 101, 102], target_labels=[-1, -1, 102, 2]

                    Before: input_ids=[100, 101], target_labels=[102, 2], prefix=[1, 3]
                    After: input_ids=[1, 3, 100, 101, 102], target_labels=[-1, -1, -1 102, 2]

                if input_data_type == InputDataType.SEQ2SEQ_NO_MASK:
                    Before: input_ids=[100, 101], target_labels=[102, 2], prefix=[1]
                    After: input_ids=[1, 100, 101, 102], target_labels=[100, 101, 102, 2]

                    Before: input_ids=[100, 101], target_labels=[102, 2], prefix=[1, 3]
                    After: input_ids=[1, 3, 100, 101, 102], target_labels=[-1, 100, 101, 102, 2]

            Specifically, outputs will be:
                "input_ids": An int32 Tensor with prefix prepended, and EOS dropped.
                "target_labels": An int32 Tensor with EOS retained.

        Encoder-Decoder (causal lm): Right shift "target_labels" and prepend "prefix" to form
            "target_ids" (decoder input ids). All decoded prefix tokens are ignored from the loss.
            "source_ids" and "target_labels" is assumed to have EOS appended.

            Example:
                pad=0, bos=1, eos=2, ignore_target_label=-1

                Before: source_ids=[100, 101, 2], target_labels=[102, 2], prefix=[1]
                After: source_ids=[100, 101, 2], target_ids=[1, 102], target_labels=[102, 2]

                Before: source_ids=[100, 101, 2], target_labels=[102, 2], prefix=[1, 3]
                After: source_ids=[100, 101, 2], target_ids=[1, 3, 102], target_labels=[-1, 102, 2]

            Specifically, outputs will be:
                "source_ids": An int32 Tensor with EOS retained.
                "target_ids": An int32 Tensor with prefix prepended, and EOS dropped.
                "target_labels": An int32 Tensor with EOS retained.

    Args:
        model_type: The type of model arch. Defaults to ENCODER_DECODER.
        input_data_type: Input data types for decoder-only language model training.
            Must be SEQ2SEQ_NO_MASK or SEQ2SEQ_MASK if model_type is ModelType.DECODER_ONLY.
        passthrough_keys: Keys to pass-through (i.e., copy from the input example).
            Note that this means the keys 'prefix', 'source_ids', 'target_ids', 'input_ids',
            and 'target_labels' will not be overwritten if provided here.

    Returns:
        A DatasetToDatasetFn, where processing depends on `model_type` described above. See also
        above for details on prefix/EOS handling.

    Raises:
        NotImplementedError: If input_data_type is not SEQ2SEQ_NO_MASK or SEQ2SEQ_MASK when
            model_type is ModelType.DECODER_ONLY.
    """
    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg

    passthrough_keys = passthrough_keys or []

    @seqio.utils.map_over_dataset
    def prepare_encoder_decoder_inputs(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        source_ids = example["source_ids"]
        target_labels = example["target_labels"]

        # This should match the "prefix" key used in decoding.
        # Casting in case prefix is empty and of float32.
        prefix = tf.cast(example["prefix"], target_labels.dtype)

        # Concat prefix and target_labels without EOS.
        target_ids = tf.concat([prefix, target_labels[:-1]], axis=0)
        prefix_padding = SEQ2SEQ_IGNORE_TARGET_LABEL * tf.ones_like(prefix)[:-1]
        target_labels = tf.concat([prefix_padding, target_labels], axis=0)

        new_example = dict(
            prefix=prefix,
            source_ids=source_ids,
            target_ids=target_ids,
            target_labels=target_labels,
        )
        for key in passthrough_keys:
            if key in example:
                new_example[key] = example[key]
        return new_example

    @seqio.utils.map_over_dataset
    def prepare_decoder_only_inputs(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        target_labels = example["target_labels"]
        # Casting in case prefix is empty and of float32.
        prefix = tf.cast(example["prefix"], target_labels.dtype)
        tf.debugging.assert_rank(prefix, 1)

        source_ids = tf.concat([prefix, example["input_ids"]], axis=0)
        input_ids = tf.concat([source_ids, target_labels[:-1]], axis=0)
        if input_data_type == InputDataType.SEQ2SEQ_MASK:
            source_padding = SEQ2SEQ_IGNORE_TARGET_LABEL * tf.ones_like(source_ids)[:-1]
            target_labels = tf.concat([source_padding, target_labels], axis=0)
        elif input_data_type == InputDataType.SEQ2SEQ_NO_MASK:
            # Ignore the prefix tokens.
            prefix_padding = SEQ2SEQ_IGNORE_TARGET_LABEL * tf.ones_like(prefix)
            source_ids_with_prefix_padding = tf.concat(
                [prefix_padding, example["input_ids"]], axis=0
            )
            target_labels = tf.concat([source_ids_with_prefix_padding[1:], target_labels], axis=0)
        else:
            raise NotImplementedError(
                f"Unsupported InputDataType: {input_data_type}. "
                f"Must be SEQ2SEQ_MASK or SEQ2SEQ_NO_MASK"
            )

        new_example = dict(
            prefix=prefix,
            input_ids=input_ids,
            target_labels=target_labels,
        )
        for key in passthrough_keys:
            if key in example:
                new_example[key] = example[key]
        return new_example

    # pylint: enable=no-value-for-parameter,unexpected-keyword-arg

    return (
        prepare_decoder_only_inputs
        if model_type == ModelType.DECODER_ONLY
        else prepare_encoder_decoder_inputs
    )


@seqio.map_over_dataset
def map_targets_out_of_class(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Maps 0-padding on "target_labels" to out-of-class labels.
    Note that seqio always pads with 0's, so we must do this after packing/padding.
    """
    example["target_labels"] = tf.where(
        example["target_labels"] == 0, SEQ2SEQ_IGNORE_TARGET_LABEL, example["target_labels"]
    )
    return example


def _trim_and_pack_with_segments(
    feature_lengths: dict[str, int]
) -> input_tf_data.DatasetToDatasetFn:
    """Trim and pack inputs, injecting `*_segment_ids` and `*_positions`.

    This works around a bug in seqio that produces incorrect packings when sequences contain
    non-trailing 0's, i.e., 0's that appear prior to a non-zero token.
    In particular, we first remap 0's to some non-zero proxy, pack, and then recover
    the original 0's.
    Note that all 0's are treated equally, even trailing 0's. This produces slightly different
    behavior than `seqio.trim_and_pack_dataset`, which packs over trailing 0's.

    Args:
        feature_lengths: See `seqio.trim_and_pack_dataset`.

    Returns:
        The same output as `seqio.trim_and_pack_dataset`, except 0's are treated as non-zero tokens.
    """
    # This can be any non-zero token, even one that exists in the original input.
    proxy_id = 1

    @seqio.map_over_dataset
    def remap_intermediate_zeros(example: dict[str, tf.Tensor]):
        for key in feature_lengths:
            ids = example[key]
            remap_mask = ids == 0
            example[key] = tf.where(remap_mask, proxy_id, ids)
            # Ensure the mask itself has no 0's.
            example[f"{key}_remap_mask"] = tf.where(remap_mask, 1, -1)
        return example

    @seqio.map_over_dataset
    def restore_intermediate_zeros(example: dict[str, tf.Tensor]):
        for key in feature_lengths:
            # Remove unused keys.
            for suffix in ["segment_ids", "positions"]:
                example.pop(f"{key}_remap_mask_{suffix}")

            # Recover the original 0's.
            remap_mask = example.pop(f"{key}_remap_mask")
            example[key] = tf.where(remap_mask == 1, 0, example[key])
        return example

    return input_tf_data.chain(
        remap_intermediate_zeros,
        functools.partial(
            seqio.trim_and_pack_dataset,
            feature_lengths=dict(
                **{f"{k}_remap_mask": v for k, v in feature_lengths.items()}, **feature_lengths
            ),
        ),
        restore_intermediate_zeros,
    )


def _trim_and_pad_with_segments(
    feature_lengths: dict[str, int]
) -> input_tf_data.DatasetToDatasetFn:
    """Trim and pad inputs, injecting `*_segment_ids`, `*_positions`.

    Args:
        feature_lengths: See `seqio.trim_and_pad_dataset`.

    Returns:
        The same output as `seqio.trim_and_pad_dataset`, except in addition we return the following
        fields for each feature:
        - `{input_key}_segment_ids`: will be 1's for the first `input_key.shape[-1]` tokens, and 0's
            for padding;
        - `{input_key}_positions`: will be `range(input_key.shape[-1])`, and 0's for padding.
    """

    # Make a copy to avoid mutating input.
    feature_lengths_with_new_keys = dict(**feature_lengths)
    for key, length in feature_lengths.items():
        for new_key in [f"{key}_segment_ids", f"{key}_positions"]:
            assert new_key not in feature_lengths_with_new_keys
            feature_lengths_with_new_keys[new_key] = length

    @seqio.map_over_dataset
    def add_segments_positions(example: dict[str, tf.Tensor]):
        for key in feature_lengths:
            example[f"{key}_segment_ids"] = tf.ones_like(example[key])
            example[f"{key}_positions"] = tf.range(tf.shape(example[key])[-1])
        return example

    return input_tf_data.chain(
        add_segments_positions,
        functools.partial(
            seqio.trim_and_pad_dataset, feature_lengths=feature_lengths_with_new_keys
        ),
    )


# pylint: disable-next=too-many-branches
def text2text_lm_input(
    *,
    is_training: bool,
    model_type: ModelType = ModelType.ENCODER_DECODER,
    target_sentence_piece_vocab: InstantiableConfig,
    source_sentence_piece_vocab: Optional[InstantiableConfig] = None,
    max_target_length: int,
    max_source_length: Optional[int] = None,
    source_key: str = "source_text",
    target_key: str = "target_text",
    processor_cfg: Optional[InstantiableConfig] = None,
    with_eos: bool = True,
    packing_mode: Literal["pack", "pad", "none"] = "pad",
) -> input_tf_data.DatasetToDatasetFn:
    """Generates inputs for language models from text to text inputs.

    This function accepts any general `processor_cfg` that conforms to the following spec:
    - If `model_type` is ENCODER_DECODER, `processor_cfg` should consume examples with tokenized
        "source_ids" and/or "target_labels", and emit "source_ids", "target_ids", and
        "target_labels".
        "source_ids" (and "target_labels", if present) will be tokenized and appended with EOS (if
        with_eos=True) prior to feeding into `processor_cfg`.
    - If `model_type` is DECODER, `processor_cfg` should consume examples with tokenized
        "input_ids" and/or "target_labels", and emit "input_ids", "target_labels".
        "input_ids" and "target_labels" will be tokenized from source_key and target_key,
        respectively, if present. Prior to feeding into `processor_cfg`, "target_labels" will be
        appended with EOS (if with_eos=True); "input_ids" will NOT be appended with EOS regardless
        of with_eos. This is to be consistent with LLM pertaining where EOS is used to
        delimitate document boundaries.

    Note that input examples are not constrained to having both "source_ids" and "target_labels" (in
    the ENCODER_DECODER case), or both "input_ids" and "target_labels" (in the DECODER_ONLY case).
    For example, for an ENCODER_DECODER model, an input pipeline may provide only "source_ids". This
    is fine as long as the `processor_cfg` is still able to emit the necessary outputs "source_ids",
    "target_ids" and "target_labels".

    Note also that the processor does not need to trim the emitted ids; this function will ensure
    that all inputs conform to the `max_target_length` and `max_source_length`.

    Args:
        is_training: Whether the input is used for training.
        model_type: Language model type. Defaults to ENCODER_DECODER.
        target_sentence_piece_vocab: Config used to instantiate seqio.SentencePieceVocabulary.
        source_sentence_piece_vocab: Config used to instantiate seqio.SentencePieceVocabulary. If
            None, use target vocab.
        max_target_length: The maximum number of target tokens per sequence.
        max_source_length: The maximum number of source tokens per sequence. If None, use
            max_target_length.
        source_key: The feature name for the source field. Will be renamed to "input_ids" for
            decoder only or "source_ids" for others.
        target_key: The feature name for the target field. Will be renamed to "target_labels".
        processor_cfg: The processor to use for generating autoregressive inputs. If None, uses
            `make_autoregressive_inputs`. See details above.
        with_eos: Whether to append EOS to "source_ids", "input_ids" and "target_labels" during
            tokenization (prior to applying `processor_cfg`).
        packing_mode: Whether to pack or pad sequences. If "none", no packing or padding is applied.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict containing `source_key` and
        `target_key`, and where each output example is processed according to the `processor_cfg`,
        and then truncated to the given `max_{source,target}_length`.

        Please refer to the corresponding `processor_cfg` docstring for details on BOS/EOS handling.

        Note: Truncation to `max_{source,target}_length` always happens last -- this means that a
        truncated sequence will not end with EOS.

    Raises:
        ValueError: If max_source_length or source_sentence_piece_vocab are provided and model_type
            is DECODER_ONLY.
        NotImplementedError: Packing for DECODER_ONLY.
    """
    del is_training
    if model_type == ModelType.DECODER_ONLY:
        if not (max_source_length is None and source_sentence_piece_vocab is None):
            raise ValueError(
                "max_source_length and source_sentence_piece_vocab should be None for DECODER_ONLY"
            )

    # Instantiate vocab(s).
    target_vocab = target_sentence_piece_vocab.instantiate()
    source_vocab = target_vocab
    if source_sentence_piece_vocab is not None:
        source_vocab = source_sentence_piece_vocab.instantiate()

    if target_vocab.pad_id != 0 or source_vocab.pad_id != 0:
        raise ValueError("seqio pads with 0's.")

    # Handle rekeys.
    source_id_key = "input_ids" if model_type == ModelType.DECODER_ONLY else "source_ids"
    rekey_map = {source_id_key: source_key, "target_labels": target_key}  # Rename these keys.

    # Handle feature lengths.
    feature_lengths = {
        source_id_key: max_source_length or max_target_length,
        "target_labels": max_target_length,
    }
    if model_type != ModelType.DECODER_ONLY:
        feature_lengths["target_ids"] = max_target_length

    # Instantiate main processor.
    processor_cfg = processor_cfg or config_for_function(make_autoregressive_inputs)
    processor_cfg = maybe_set_config(
        processor_cfg,
        model_type=model_type,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    process_inputs_fn = processor_cfg.instantiate()

    # Retain all original feature keys.
    pack_or_pad_rekey = {k: k for k in feature_lengths}
    for k in feature_lengths:
        # Drop "target_labels_{segment_ids,positions}", as they are redundant.
        if not k.endswith("ids"):
            continue
        # Instead of the more verbose "source_ids_segment_ids", we rename to
        # "source_segment_ids". Likewise for "source_ids_positions" to "source_positions".
        for suffix in ["segment_ids", "positions"]:
            pack_or_pad_rekey[k.replace("_ids", f"_{suffix}")] = f"{k}_{suffix}"

    # Instantiate post-processor. Padding can be used e.g. for eval.
    if packing_mode == "pack":
        pack_or_pad_fn = input_tf_data.chain(
            _trim_and_pack_with_segments(feature_lengths=feature_lengths),
            rekey(pack_or_pad_rekey, retain_original_inputs=False),
        )
    elif packing_mode == "pad":
        pack_or_pad_fn = input_tf_data.chain(
            _trim_and_pad_with_segments(feature_lengths=feature_lengths),
            # For trim_and_pad, retain all original keys except for the rekeyed ones.
            rekey(pack_or_pad_rekey, retain_original_inputs=True),
            input_tf_data.remove_fields(
                [x for k in feature_lengths for x in [f"{k}_segment_ids", f"{k}_positions"]],
            ),
        )
    elif packing_mode == "none":
        pack_or_pad_fn = input_tf_data.identity()
    else:
        raise ValueError(f"Unrecognized packing mode: {packing_mode}")

    # EncoderDecoderModel expects encoder inputs under "source" and decoder inputs under "target".
    if model_type == ModelType.ENCODER_DECODER:
        encoder_decoder_rekey = {}
        for key in ["source", "target"]:
            encoder_decoder_rekey.update(
                {
                    f"{key}/input_ids": f"{key}_ids",
                    f"{key}/input_segment_ids": f"{key}_segment_ids",
                    f"{key}/positions": f"{key}_positions",
                }
            )
        make_encoder_decoder_inputs = input_tf_data.chain(
            # Retain original keys except those that were rekeyed.
            rekey(
                encoder_decoder_rekey,
                default_value=None,
                separator="/",
                retain_original_inputs=True,
            ),
            input_tf_data.remove_fields(list(set(encoder_decoder_rekey.values()))),
        )
    else:
        make_encoder_decoder_inputs = input_tf_data.identity()

    # TODO(gyin): Avoid double tokenization for decoder only.
    return input_tf_data.chain(
        rekey(rekey_map, retain_original_inputs=True),
        # Tokenize features (if available). Features specified here which are not in the input
        # example are skipped.
        tokenize(
            output_features={
                "target_labels": seqio.Feature(
                    target_vocab,
                    add_eos=True,
                    dtype=tf.int32,
                ),
                source_id_key: seqio.Feature(
                    source_vocab,
                    add_eos=model_type == ModelType.ENCODER_DECODER,
                    dtype=tf.int32,
                ),
            },
            copy_pretokenized=False,
            with_eos=with_eos,
        ),
        process_inputs_fn,
        pack_or_pad_fn,
        # Ensures that padding introduced at "target_labels" during pad/pack are out-of-class.
        map_targets_out_of_class,
        make_encoder_decoder_inputs,
    )


def lm_text_preprocessor(
    *,
    vocab_cfg: InstantiableConfig,
    max_sequence_length: int,
    replace_newlines_with: str,
    shuffle_buffer_size: int = 8192,
    max_padding_fraction: float = 1.0,
    window_size: int = 128,
    additional_preprocessors: Optional[Sequence[InstantiableConfig]] = None,
    token_adjuster_cfg: Optional[InstantiableConfig] = None,
    packing_method: Optional[PackingMethodType] = None,
    is_training: Optional[bool] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Produces processors for lm data for LM training."""

    def processors_for_default_lm() -> Sequence[InstantiableConfig]:
        processors = list(additional_preprocessors) if additional_preprocessors else []
        processors.append(
            # At the moment our decontaminated datasets have either
            # text as "clean_text", OR text as "text".
            # TODO(tom_gunter): Remove once uniform versions are created with text as "text"
            # (in process).
            config_for_function(input_tf_data.rekey).set(
                # Will retain "text" if it exists, else map "clean_text" to "text".
                key_map={"text": "clean_text"},
                default_value=None,
                retain_original_inputs=True,
            )
        )
        # Build training examples out of "text".
        processors.append(
            config_for_function(text_to_lm_training_input).set(
                vocab_cfg=vocab_cfg,
                max_len=max_sequence_length,
                replace_newlines_with=replace_newlines_with,
                max_padding_fraction=max_padding_fraction,
                window_size=window_size,
                shuffle_buffer_size=shuffle_buffer_size,
                token_adjuster_cfg=token_adjuster_cfg,
                packing_method=packing_method,
                is_training=is_training if is_training is not None else True,
            )
        )
        return processors

    final_processors = processors_for_default_lm()
    return input_tf_data.chain(*final_processors)


def augment_text_from_inputs_targets_pretokenized(
    *,
    replace_newlines_with: str,
    source_key: str,
    target_key: str,
    eos_id: int,
    prompt: str = "",
    join_with: str = "\n\n",
) -> input_tf_data.DatasetToDatasetFn:
    """Generate inputs_pretokenized, targets_pretokenized and prefix.

    Args:
        replace_newlines_with: Text to replace newlines with.
        source_key: The feature name for the source field.
        target_key: The feature name for the target field.
        eos_id: Eos ID that is used as the default prefix.
        prompt: The prefix to prepend to the full example "text".
        join_with: The string to insert between the source_key and target_key.

    Returns:
        A DatasetToDatasetFn, with the possible new keys inputs_pretokenized,
        targets_pretokenized and prefix.
    """

    def fn(inputs: dict[str, Any]) -> dict[str, Any]:
        inputs["inputs_pretokenized"] = tf.strings.regex_replace(
            prompt + inputs[source_key] + join_with, "\n", replace_newlines_with
        )
        inputs["targets_pretokenized"] = tf.strings.regex_replace(
            inputs[target_key], "\n", replace_newlines_with
        )
        inputs["prefix"] = inputs.get("prefix", [eos_id])
        return inputs

    return seqio.map_over_dataset(fn)


def lm_from_seq2seq_text_preprocessor(
    *,
    is_training: bool,
    vocab_cfg: InstantiableConfig,
    max_sequence_length: int,
    replace_newlines_with: str,
    input_data_type: InputDataType = InputDataType.SEQ2SEQ_MASK,
    source_key: str = "inputs_pretokenized",
    target_key: str = "targets_pretokenized",
    prompt: str = "",
    join_with: str = "\n\n",
    autoregressive_processor_cfg: Optional[InstantiableConfig] = None,
    packing_mode: Literal["pack", "pad", "none"] = "pad",
) -> input_tf_data.DatasetToDatasetFn:
    """Produces processors for lm or seq2seq data for LM training.

    if input_data_type is SEQ2SEQ_NO_MASK, tokenize
        - prompt + inputs[source_key] + join_with, and
        - inputs[target_key]
        separately and concatenate the token ids for input_ids and target_labels.
    If input_data_type is SEQ2SEQ_MASK, same as SEQ2SEQ_NO_MASK but mask the loss on the input
        tokens - only calculate loss on the target tokens.

    Args:
        is_training: whether the input is used for training (should be True).
        vocab_cfg: Config to instantiate the seqio vocab.
        max_sequence_length: maximum sequence length of an example.
        replace_newlines_with: Text to replace newlines with.
        input_data_type: Input data types for decoder-only language model training.
        source_key: The feature name for the source field.
        target_key: The feature name for the target field.
        prompt: The prefix to prepend to the full example "text".
        join_with: The string to insert between the source_key and target_key.
        autoregressive_processor_cfg: Optional autoregressive processor. See
            `input_lm.text2text_lm_input` for details.
        packing_mode: Whether to pack or pad sequences. If "none", no packing or padding is applied.

    Returns:
        A DatasetToDatasetFn, where each input example should be a dict
         containing:
            * the keys source_key and target_key with string values (when
                input_data_type is SEQ2SEQ_MASK or SEQ2SEQ_NO_MASK).
         Each output example is a dict with:
            * "input_ids" as int32 tensor with shape [max_len],
            * "target_labels" as int32 tensor with shape [max_len],
    """
    vocab = vocab_cfg.instantiate()
    eos_id = vocab.eos_id

    if autoregressive_processor_cfg is None:
        autoregressive_processor_cfg = config_for_function(make_autoregressive_inputs).set(
            model_type=ModelType.DECODER_ONLY,
            input_data_type=input_data_type,
        )

    final_processors = [
        augment_text_from_inputs_targets_pretokenized(
            replace_newlines_with=replace_newlines_with,
            prompt=prompt,
            join_with=join_with,
            eos_id=eos_id,
            source_key=source_key,
            target_key=target_key,
        ),
        config_for_function(text2text_lm_input).set(
            is_training=is_training,
            model_type=ModelType.DECODER_ONLY,
            target_sentence_piece_vocab=vocab_cfg,
            max_target_length=max_sequence_length,
            source_key="inputs_pretokenized",
            target_key="targets_pretokenized",
            processor_cfg=autoregressive_processor_cfg,
            packing_mode=packing_mode,
        ),
    ]

    return input_tf_data.chain(*final_processors)


def lm_decoding_prefix_processor(
    *,
    vocab_cfg: InstantiableConfig,
    max_sequence_length: int,
) -> input_tf_data.DatasetToDatasetFn:
    """Prepares the prefix for decoder-only model decoding & evaluation.

    Args:
        vocab_cfg: Config to instantiate the seqio vocab.
        max_sequence_length: Maximum sequence length (input + generated tokens).

    Returns:
        A DatasetToDatasetFn to generate decoding prefixes from supervised input examples.

        Each input example contains `input_ids` and `target_labels`. For example,
            input_ids:      [1, 100, 101, 102, 110, 111]
            target_labels:  [-1, -1,  -1, 110, 111,   1]
        where `target_labels` represents ground truth decoding outputs.

        Each output example will contain a prefix of `input_ids`,
        replacing the tokens from `target_labels` with `vocab.pad_id`. For example,
            prefix:         [1, 100, 101, 102,   0,   0]

    Raises:
        ValueError if that there is gap in between ignore tokens
            (i.e. the -1's are not contiguous) in target_labels after prepending
            the ignore token.

    Currently, the processor can only support the inputs with below constraints:
        * Each example should only contain 1 source-target pair.
        * Target labels should mask source part to be ignored.

    Example steps:
        1. features from input example:
        eos=1, ignore_target_label=-1
        original source ids = [1, 100, 101, 102]

        input_ids:      [1, 100, 101, 102, 110, 111]
        target_labels:  [-1, -1,  -1, 110, 111,   1]

        2. Compute target_ids by prepend -1 that corresponds start token (eos=1)
        and truncate last token (eos=1) in target labels:

        target_ids:     [-1, -1,  -1,  -1, 110, 111]

        3. Select input ids where the target ids are ignored
        in same positions and pad others:

        prefix:         [1, 100, 101, 102,   0,   0]


    More features may be supported in the future.
    """
    vocab = vocab_cfg.instantiate()

    @seqio.map_over_dataset
    def prepare_prefix(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        input_ids = example["input_ids"]
        target_labels = example["target_labels"]

        # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        target_ids = tf.concat([[SEQ2SEQ_IGNORE_TARGET_LABEL], target_labels[:-1]], axis=0)
        # pylint: enable=no-value-for-parameter,unexpected-keyword-arg

        is_ignored = tf.cast(target_ids == SEQ2SEQ_IGNORE_TARGET_LABEL, tf.int64)
        # Find the starting and ending index of 1s
        start = tf.argmax(is_ignored)
        end = (
            tf.cast(tf.size(is_ignored), start.dtype)
            - tf.argmax(tf.reverse(is_ignored, axis=[0]))
            - 1
        )
        # Assert there is no gap in between ignore tokens.
        tf.debugging.assert_equal(tf.reduce_sum(is_ignored[start : end + 1]), end - start + 1)

        example["prefix"] = tf.where(
            target_ids == SEQ2SEQ_IGNORE_TARGET_LABEL,
            input_ids,
            vocab.pad_id,
        )
        return example

    def pad_prefix():
        return functools.partial(
            seqio.trim_and_pad_dataset, feature_lengths={"prefix": max_sequence_length}
        )

    return input_tf_data.chain(
        prepare_prefix,
        # TODO(taiyi_kuo): Remove this once we have a prefix shape handling method.
        # We need to pad prefix for the below purposes:
        # 1. Pad to the same length for batch processing.
        # 2. Control max decode length by prefix.shape[-1].
        config_for_function(pad_prefix),
        is_training=False,
    )
