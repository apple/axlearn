# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/text-to-text-transfer-transformer:
# Copyright 2022 The T5 Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# huggingface/transformers:
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Input processing for GLUE benchmark.

Reference:
https://github.com/huggingface/transformers/blob/692e61e91a0b83f5b847902ed619b7c74c0a5dda/examples/pytorch/text-classification/run_glue.py
https://www.tensorflow.org/datasets/catalog/glue
"""

import functools
from collections.abc import Sequence
from typing import Optional, Union

import seqio
import tensorflow as tf

from axlearn.common import input_tf_data
from axlearn.common.config import InstantiableConfig
from axlearn.common.input_lm import SEQ2SEQ_IGNORE_TARGET_LABEL
from axlearn.common.input_text import infer_bos_id, tokenize
from axlearn.common.input_tf_data import DatasetToDatasetFn, rekey


def preprocess_copa() -> input_tf_data.DatasetToDatasetFn:
    """Preprocess inputs for COPA.

    In particular, given an example:

        {
            "premise": "I packed up my belongings.",
            "question": "cause",
            "choice1": "I was hunting for a new apartment.",
            "choice2": "I was moving out of my apartment.",
            "label": 1,  # Choice 2.
        }

    We preprocess into two examples:

        {
            "premise": "I packed up my belongings. What's the CAUSE for this?",
            "choice": "I was hunting for a new apartment.",
            "label": 1,
        },
        {
            "premise": "I packed up my belongings. What's the CAUSE for this?",
            "choice": "I was moving out of my apartment.",
            "label": 1,
        }

    The processor thus produces examples in batches of 2. A downstream processor can then unbatch or
    concat these examples. See also `postprocess_copa`.

    Returns:
        A DatasetToDatasetFn preprocessing the dataset for COPA.
    """

    def process_example_fn(example: dict[str, tf.Tensor]):
        # Reference: Table 6 https://arxiv.org/pdf/1905.00537.pdf
        if example["question"] == "cause":
            question = "What's the CAUSE for this?"
        else:
            question = "What happened as a RESULT?"

        # TODO(markblee,xiang-kong): Investigate whether question (or separator) are required.
        premise = example["premise"] + " " + question
        return {
            "premise": [premise, premise],
            "choice": [example["choice1"], example["choice2"]],
            "label": [example["label"], example["label"]],
        }

    return seqio.map_over_dataset(process_example_fn)


def postprocess_copa():
    """Post-process for COPA by grouping consecutive examples.

    For COPA, examples are unbatched during processing, but need to be grouped again prior to
    feeding to the model. The assumption is that consecutive pairs correspond to the same original
    example.

    For example:
        Input:
        {
            "input_ids": [1, 100, 101, 1, 1, 102, 103, 1],
            "target_labels": 1,
        },
        {
            "input_ids": [1, 200, 201, 1, 1, 202, 203, 1],
            "target_labels": 1,
        }
        Output:
        {
            "input_ids": [
                [1, 100, 101, 1, 1, 102, 103, 1],
                [1, 200, 201, 1, 1, 202, 203, 1]
            ],
            "target_labels": 1,
        }

    Returns:
        A processor that consumes examples with "input_ids" of shape [max_len], scalar
        "target_labels", and produces examples with "input_ids" of shape [2, max_len], and
        "target_labels" of shape [2].
    """

    def reduce_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        target_label = example["target_labels"][0]
        # Sanity check that all "target_labels" match.
        for label in example["target_labels"][1:]:
            tf.debugging.assert_equal(target_label, label, message="Batch has invalid targets.")
        return {
            "input_ids": example["input_ids"],
            "target_labels": target_label,
        }

    def process_dataset_fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.batch(2)
        ds = ds.map(reduce_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    return process_dataset_fn


def add_special_tokens_for_roberta(
    vocab_cfg: InstantiableConfig,
    input_key: Union[str, Sequence[str]],
    output_key: str = "input_ids",
    bos_token: Optional[str] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Add special tokens to one or multiple sequences, in the style of RoBERTa.

    Specifically, sequences always start and end with BOS/EOS, and multiple
    sequences are separated by double EOS.

    Examples:
        <s> only one input </s>
        <s> first input </s></s> second input </s>
        <s> first input </s></s> second input </s></s> third input </s>

    Args:
        vocab_cfg: Vocab config.
        input_key: One or multiple input sequences.
        output_key: Key corresponding to the concatenated fields.
        bos_token: Optional BOS token. If None, infers from vocab (see `infer_bos_id`).

    Returns:
        A DatasetToDatasetFn mapping the concatenated sequence to key `output_key`.

    Raises:
        ValueError: If no `input_keys` provided.
    """
    vocab = vocab_cfg.instantiate()
    bos_id = vocab.tokenizer.piece_to_id(bos_token) if bos_token else infer_bos_id(vocab)
    eos_id = vocab.eos_id
    if isinstance(input_key, str):
        input_key = [input_key]

    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Builds inputs from multiple sequences in the style of RoBERTa.

        Args:
            example: Input example dictionary for one or multiple input sequences.

        Returns:
            The same example, with concatenated inputs in key `output_key`.
            All other keys remain unchanged.
        """
        parts = [[bos_id], example[input_key[0]], [eos_id]]
        for key in input_key[1:]:
            parts.extend([[eos_id], example[key], [eos_id]])
        # pylint: disable-next=no-value-for-parameter,unexpected-keyword-arg
        example[output_key] = tf.concat(parts, axis=0)
        return example

    return seqio.map_over_dataset(process_example_fn)


def preprocess_record() -> DatasetToDatasetFn:
    """Handles input examples of ReCoRD task.
    Each example contains a passage, a query, several entities and answers.
    We will split each input example into several examples according entities.

    As an example, if the input looks like:
    {
        "passage": "I am a passage."
        "query": "I am a query."
        "entities: ["candidate1", candidate2"]
        "answers: ["candidate1"]
    }
    the final input examples will be:
    [
        {
            "passage": "I am a passage."
            "query": "I am a query."
            "entity: "candidate1"
            "label: 1 # `candidate1` is in `answers` list.
        },
        {
            "passage": "I am a passage."
            "query": "I am a query."
            "entity: "candidate2"
            "label: 0 # `candidate2` is NOT in `answers` list.
        },
    ]
    """

    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        split_examples = {}
        num_entities = tf.size(example["entities"])

        def expand_dim(x: tf.Tensor) -> tf.Tensor:
            # Expands the input tensor according to the number of entities.
            expand_ratio = tf.math.maximum(num_entities, 1)
            # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            return tf.broadcast_to(x, shape=tf.concat([[expand_ratio], tf.shape(x)], axis=0))

        split_examples["passage"] = expand_dim(example["passage"])
        split_examples["query"] = expand_dim(example["query"])
        split_examples["entity"] = example["entities"]
        # Assigns true to Entities in the `answers`.
        label = tf.math.equal(example["entities"][:, None], example["answers"][None, :])
        label = tf.reduce_any(label, axis=1)
        split_examples["label"] = tf.cast(label, tf.int64)
        return split_examples

    return seqio.map_over_dataset(process_example_fn)


def multi_sequence_truncation(max_len: int, input_key: Union[str, Sequence[str]]):
    """Handles truncation of one or multiple sequences.

    We first set up the same length threshold for each sequences, i.e., (max_len//num(seqs)),
    then we compute the total number of tokens under thresholds and find out sequences exceeding
    this threshold. Finally, we put the extra number tokens to these long sequences evenly.
    The strategy is currently to always truncate from the right.

    TODO(markblee): Make the truncation implementation configurable, e.g:
    https://huggingface.co/docs/transformers/pad_truncation

    Args:
        max_len: Maximum sequence length.
        input_key: Field(s) to truncate.

    Returns:
        An DatasetToDatasetFn where each input example should be a dict containing the provided
        fields corresponding to token IDs, and each output example contains the same fields, but
        corresponding to truncated IDs.
    """
    if isinstance(input_key, str):
        input_key = [input_key]

    threshold = max_len // len(input_key)
    threshold = tf.constant(threshold)

    # TODO(xiang): explore better truncation strategy.
    def process_example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        sizes = tf.convert_to_tensor([tf.size(example[key]) for key in input_key])
        thresholds = tf.ones_like(sizes) * threshold
        # The total number of tokens under thresholds.
        total = tf.reduce_sum(tf.minimum(thresholds, sizes))
        # Whether sequences are exceeding the thresholds.
        truncated = tf.cast(tf.greater(sizes, thresholds), tf.int32)
        # Divide the number of capacity left among truncated sequences.
        threshold_inc = (max_len - total) // tf.maximum(1, tf.reduce_sum(truncated))
        # Add threshold_inc to the threshold of each truncated sequence.
        thresholds += truncated * threshold_inc
        remainder = (max_len - total) % tf.maximum(1, tf.reduce_sum(truncated))
        # Obtain indices of top-`remainder` longest sequence
        sorted_indices = tf.argsort(sizes, direction="DESCENDING")[:remainder]
        # Give one more threshold budget for longer sequences from `remainder`.
        thresholds = tf.tensor_scatter_nd_add(
            thresholds,
            tf.expand_dims(sorted_indices, axis=1),
            tf.ones_like(sorted_indices),
        )
        # Compute final sizes of each sequence.
        sizes = tf.minimum(thresholds, sizes)
        # Truncate from right.
        for i, key in enumerate(input_key):
            example[key] = example[key][: sizes[i]]

        return example

    return seqio.map_over_dataset(process_example_fn)


def add_prefix_concat_sequence_pair(
    *,
    dataset_name: str,
    label_names: Sequence[str],
    input_key: Union[str, tuple[str, str]],
    source_key: str = "source_text",
    target_key: str = "target_text",
) -> input_tf_data.DatasetToDatasetFn:
    """Concatenate sequence pairs into a single sequence and prepend `dataset_name` to `source_key`.

    Notes:
    - For classification task, `target_key` maps to the label name corresponding to the class ID.
    - For regression task, `target_key` maps to the string form of the float label.

    Adapted from:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/50a797f3386d3985c2e387cc20626a6ac1483336/t5/data/preprocessors.py#L734-L812
    https://github.com/google-research/text-to-text-transfer-transformer/blob/50a797f3386d3985c2e387cc20626a6ac1483336/t5/data/preprocessors.py#L816-L855

    Args:
        dataset_name: The name of the GLUE sub-dataset.
        label_names: A sequence of names of label. ["none"] for regression task.
        input_key: Input text keys.
        source_key: String of key for the concatenated sequence which is prepended with
            `dataset_name` in output dataset.
        target_key: String of key for the label name in output dataset.

    Returns:
        A DatasetToDatasetFn where each input example has "label", "idx", and `input_key`(s). Each
        output example is a dict with keys `source_key`, `target_key`, and "idx", where `source_key`
        has been concatenated and prepended with `dataset_name`.

    Raises:
        NotImplementedError: Current version does not support WSC and ReCoRD tasks in SuperGLUE.

    Example:
        For QQP, which is a classification task, here is an example with fake data:
        dataset_name: "qqp"
        label_names: ["not_duplicate", "duplicate"]
        inputs:
            {
                "question1": "Hello",
                "question2": "World",
                "label": 0,
                "idx": 3,
            }
        outputs:
            {
                "source_text": "qqp question1: Hello question2: World",
                "target_text": "not_duplicate",
                "idx": 3,
            }

        For STSB, which is a regression task, here is an example with fake data:
        dataset_name: "stsb"
        label_names: ["none"]
        inputs:
            {
                "sentence1": "Hello",
                "sentence2": "World",
                "label": 1.8,
                "idx": 3,
            }
        outputs:
            {
                "source_text": "stsb question1: Hello question2: World",
                "target_text": "1.8",
                "idx": 3,
            }
    """

    def _join_input_keys(example: dict[str, Union[int, str]]) -> tf.Tensor:
        """Concatenate dataset name, fields in the `input_keys` and their corresponding values.
        Args:
            example: A dictionary to represent an input example.

        Returns:
            A tensor to indicate the input feature concatenation.
        """
        input_keys = input_key
        if isinstance(input_key, str):
            input_keys = [input_key]
        input_keys = sorted(input_keys)
        # Add dataset name at the start.
        strs_to_join = [dataset_name]
        for key in input_keys:
            strs_to_join.append(f"{key}:")
            strs_to_join.append(example[key])
        return tf.strings.join(strs_to_join, separator=" ")

    @seqio.map_over_dataset
    def glue_fn(example: dict[str, Union[int, str]]) -> dict[str, Union[int, str, tf.Tensor]]:
        """Support general tasks in GLUE/SuperGLUE."""
        label_name = tf.cond(
            # When no label is provided (label == -1), use "<unk>".
            # This should only happen for examples in the test set.
            tf.equal(example["label"], -1),
            lambda: tf.constant("<unk>"),
            # Otherwise grab the label text from label_names.
            lambda: tf.gather(label_names, tf.cast(example["label"], tf.int32)),
        )
        source_ids = _join_input_keys(example)
        return {
            source_key: source_ids,
            target_key: label_name,
            "idx": example["idx"],
            "prefix": example["prefix"],
        }

    @seqio.map_over_dataset
    def stsb_fn(example: dict[str, Union[int, str]]) -> dict[str, Union[int, str]]:
        """STSB preprocessor."""
        label_name = tf.as_string(tf.round(example["label"] * 5) / 5, precision=1)
        source_ids = _join_input_keys(example)
        return {
            source_key: source_ids,
            target_key: label_name,
            "idx": example["idx"],
            "prefix": example["prefix"],
        }

    @seqio.map_over_dataset
    def wsc_fn(example: dict[str, Union[int, str]]) -> dict[str, Union[int, str]]:
        """WSC preprocessor.

        https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py#L859
        WSC includes a sentence along with 2 'spans': the first denoting a noun and
        the other a pronoun. The 'label' specifies whether or not the pronoun is
        referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
        around the pronoun.

        For example, a typical example from WSC might look like
        {
            'text': 'We show an example here .',
            'span1_text': 'example',
            'span1_index': 3,
            'span2_text': 'We',
            'span2_index': 0,
            'label': 0
        }

        This example would be transformed to
        {
            'inputs': 'wsc text: # We # show an * example * here .',
            'targets': 'False'
        }

        Args:
            example: A dictionary to represent an input example.

        Returns:
            A preprocessed example in which target spans are surrounded by special tokens.
        """

        def _mark_span(text, span_str, span_idx, mark):
            """Add `mark` around the text span specified by the `span_str` and `span_idx`
            Args:
                text: A string Tensor.
                span_str: The string text of the target span.
                span_idx: The index of the target span in the input sequence.
                mark: The symbol to be added around the target span.

            Returns:
                A string tensor containing special symbol `mark` around the target span.
            """
            pattern_template = r"^((?:\S+\s){N})(W)"
            # Fill in the span index.
            pattern = tf.strings.regex_replace(pattern_template, "N", tf.as_string(span_idx))
            # Fill in the span string.
            pattern = tf.strings.regex_replace(pattern, "W", span_str)
            # Add the special symbol `mark` around the target span.
            # pylint: disable=consider-using-f-string
            return tf.strings.regex_replace(text, pattern, r"\1{0} \2 {0}".format(mark))

        text = example["text"]
        text = _mark_span(text, example["span1_text"], example["span1_index"], "*")
        # Compensate for 2 added "words" added in previous step.
        span2_index = example["span2_index"] + 2 * tf.cast(
            example["span1_index"] < example["span2_index"], tf.int32
        )
        text = _mark_span(text, example["span2_text"], span2_index, "#")

        # Add benchmark name at the start
        strs_to_join = ["wsc", "text:", text]
        label_name = tf.cond(
            # When no label is provided (label == -1), use "<unk>"
            tf.equal(example["label"], -1),
            lambda: tf.constant("<unk>"),
            # Otherwise use False/True.
            lambda: tf.gather(["False", "True"], example["label"]),
        )

        source_ids = tf.strings.join(strs_to_join, separator=" ")
        return {
            source_key: source_ids,
            target_key: label_name,
            "idx": example["idx"],
            "prefix": example["prefix"],
        }

    def record_fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        """ReCoRD preprocessor.

        https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py#L917
        ReCoRD contains a context passage, a query containing a '@placeholder' string,
        and a set of candidates for the placeholder. There are also a list of answers,
        any of which would be considered correct.

        For example,
        {
            'passsage': 'This is the passage.',
            'query': 'A @placeholder is a dog.',
            'entities': ['Leopard', 'Chihuahua', 'Husky'],
            'answers': ['Chihuahua', 'Husky'],
        }
        After processing, this example will become:
        {
            'inputs': 'record query: A @placeholder is a dog. entities: Leopard, '
                        'Chihuahua, Husky passage: This is the passage.',
            'targets': 'Chihuahua',
        }
        and
        {
            'inputs': 'record query: A @placeholder is a dog. entities: Leopard, '
                        'Chihuahua, Husky passage: This is the passage.',
            'targets': 'Husky',
        }

        Args:
            ds: a ReCoRD tf.data.Dataset to process.

        Returns:
            A tf.data.Dataset contained processed examples.
        """

        def _split_answers(example):
            """Obtain one example per answer."""
            example_clone = example.copy()
            num_answers = tf.size(example_clone["answers"])

            def expand_dim(x: tf.Tensor) -> tf.Tensor:
                # Expand the input tensor according to the number of entities.
                expand_ratio = tf.math.maximum(num_answers, 1)
                # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
                return tf.broadcast_to(x, shape=tf.concat([[expand_ratio], tf.shape(x)], axis=0))

            for k, v in example.items():
                if k != "idx":
                    example_clone[k] = expand_dim(v)
                example_clone["targets"] = tf.cond(
                    tf.greater(num_answers, 0),
                    lambda: example["answers"],
                    lambda: tf.constant(["<unk>"]),
                )
                example_clone["idx"] = {
                    "passage": expand_dim(example["idx"]["passage"]),
                    "query": expand_dim(example["idx"]["query"]),
                }

            return example_clone

        def _process_example_fn(example: dict[str, Union[int, str]]) -> dict[str, Union[int, str]]:
            passage = example["passage"]
            # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py#L978-L981
            passage = tf.strings.regex_replace(passage, r"(\.|\?|\!|\"|\')\n@highlight\n", r"\1 ")
            passage = tf.strings.regex_replace(passage, r"\n@highlight\n", ". ")

            strs_to_join = [
                "record query:",
                example["query"],
                "entities:",
                tf.strings.reduce_join(example["entities"], separator=", "),
                "passage:",
                passage,
            ]
            source_ids = tf.strings.join(strs_to_join, separator=" ")

            return {
                source_key: source_ids,
                target_key: example["targets"],
                "idx": example["idx"],
                "prefix": example["prefix"],
            }

        ds = ds.map(_split_answers)
        ds = ds.unbatch()
        return ds.map(_process_example_fn)

    if dataset_name == "stsb":
        return stsb_fn
    elif dataset_name == "wsc":
        return wsc_fn
    elif dataset_name == "record":
        return record_fn
    else:
        return glue_fn


# TODO(markblee): Consider removing normalization and handling that at the source.
def text_to_glue_input(
    *,
    is_training: bool,
    input_key: Union[str, Sequence[str]],
    max_len: int,
    vocab_cfg: InstantiableConfig,
    truncation: InstantiableConfig,
    concatenation: InstantiableConfig,
    normalization: Optional[InstantiableConfig] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """Preprocesses GLUE input.

    The processor is intended to work with one or multiple sequence inputs.
    Input mappers should consume an `input_key` field containing either a single string or a tuple
    of multiple strings.

    Note: The caller must set `truncation.max_len`, which depends on the style of
    `concatenation` (e.g. how many special tokens are added).

    Args:
        is_training: Whether the input is used for training.
        input_key: Input field(s) containing text.
        max_len: Maximum sequence length.
        vocab_cfg: Vocab config.
        truncation: Truncation config. Should support multiple sequence truncation.
        concatenation: Concatenation config. Should add special tokens like BOS, EOS, and
            in the multiple sequences case concatenate with the appropriate separators.
        normalization: Optional normalization config. If not None, applied prior to tokenization.

    Returns:
        A processor that consumes examples containing the appropriate input fields for the given
        GLUE task (see `glue_input_fields`), which produces output examples containing key
        "input_ids" corresponding to token IDs of shape [max_seq_len] and "target_labels"
        corresponding to scalar labels.

    Raises:
        ValueError: If input_key has invalid type.
    """
    del is_training
    vocab = vocab_cfg.instantiate()

    if isinstance(input_key, str):
        input_key = [input_key]

    if len(input_key) < 1:
        raise ValueError("Expected at least one input_key")
    processors = []
    if normalization:
        if hasattr(normalization, "input_key"):
            normalization.set(input_key=input_key)
        processors.append(normalization)

    processors.extend(
        [
            # Tokenize all input fields separately, without EOS.
            tokenize(
                output_features={
                    key: seqio.Feature(
                        vocab,
                        add_eos=False,
                        dtype=tf.int32,
                    )
                    for key in input_key
                },
                copy_pretokenized=False,
            ),
            # Truncate before applying special tokens.
            truncation.set(input_key=input_key),
            # Apply special tokens and concatenate fields into one input sequence.
            concatenation.set(
                vocab_cfg=vocab_cfg,
                input_key=input_key,
                output_key="input_ids",
            ),
            # Rekey the label field, and drop other fields besides input_ids.
            rekey({"target_labels": "label", "input_ids": "input_ids"}),
            functools.partial(
                seqio.trim_and_pad_dataset,
                feature_lengths={"input_ids": max_len},
            ),
        ]
    )

    return input_tf_data.chain(*processors)


# TODO(markblee): T5 pretraining to support arbitrary "prefix".
def make_glue_autoregressive_inputs(
    *, vocab_cfg: InstantiableConfig, max_source_length: int, max_target_length: int
) -> input_tf_data.DatasetToDatasetFn:
    """Preprocesses GLUE inputs for seq2seq, e.g. as seen in T5.

    See also `input_lm.text2text_lm_input`.

    Notes:
    - We trim "source_ids" and "target_labels" to `max_{source,target}_len`-1 prior to appending
        EOS, so they always end in EOS (even if truncation happens).
    - We prepend "prefix" to "target_ids", similar to pretraining.

    Args:
        vocab_cfg: A config that instantiates to a vocab.
        max_source_length: Maximum source sequence length.
        max_target_length: Maximum target sequence length.

    Returns:
        A processor that consumes examples with "source_ids" and "target_labels" as keys, both with
        EOS appended; and outputs examples with keys "source_ids", "target_labels" and "target_ids".

    Examples:
        eos_id: 1
        max_len: 4
        inputs:
            {
                "prefix": [100],
                "source_ids": [10, 11, 12, 1],
                "target_labels": [20, 21, 22, 23, 1],
                "idx": 3,
            }
        outputs:
            {
                "prefix": [100],
                "source_ids": [100, 10, 11, 1],
                "target_labels": [20, 21, 22, 1],
                "target_ids": [100, 20, 21, 22],
                "idx": 3,
            }
    """
    vocab = vocab_cfg.instantiate()
    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg

    def prepare_encoder_decoder_inputs(example: dict[str, tf.Tensor]):
        prefix = example["prefix"]

        # Drop the EOS added by `input_lm.text2text_lm_input`.
        source_ids = example["source_ids"][:-1]
        target_labels = example["target_labels"][:-1]

        # Truncate to max length prior to appending EOS.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/9860243bbd8a93fb284edbe9d9abfc1de40b7bc8/t5/data/tasks.py#L154
        source_ids = tf.concat([source_ids[: max_source_length - 1], [vocab.eos_id]], axis=0)
        # Generate "target_ids" from "target_labels" in an autoregressive manner.
        target_ids = tf.concat([prefix, target_labels], axis=0)
        # Make sure that prefix is ignored in "target_labels".
        prefix_padding = SEQ2SEQ_IGNORE_TARGET_LABEL * tf.ones_like(prefix)[:-1]
        target_labels = tf.concat([prefix_padding, target_labels], axis=0)
        # Truncate targets to max length prior to appending EOS.
        target_labels = tf.concat([target_labels[: max_target_length - 1], [vocab.eos_id]], axis=0)
        # Trim target_ids. This is not strictly necessary, since `input_lm.text2text_lm_input` will
        # trim anyway, but we do it to be consistent with the other outputs.
        target_ids = target_ids[:max_target_length]

        return dict(
            idx=example["idx"],
            prefix=prefix,
            source_ids=source_ids,
            target_ids=target_ids,
            target_labels=target_labels,
        )

    # pylint: enable=no-value-for-parameter,unexpected-keyword-arg
    return seqio.map_over_dataset(prepare_encoder_decoder_inputs)
