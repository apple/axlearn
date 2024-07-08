# Copyright © 2023 Apple Inc.

"""Input processing for reading comprehension.

This module implements functions necessary to prepare reading comprehension datasets for training
and evaluation.
"""
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import tensorflow as tf
from seqio import map_over_dataset
from transformers import PreTrainedTokenizer

from axlearn.common.config import REQUIRED, Configurable, Required, config_class, maybe_instantiate
from axlearn.common.input_tf_data import BuildDatasetFn, DatasetToDatasetFn


class TokenizerForReadingComprehension(ABC):
    """Abstract reading comprehension tokenizer class.

    This abstract class provides a generic interface which a tokenizer can implement such that the
    reading comprehension functions do not need to rely on the minutiae of different model or
    library specific tokenizers.
    """

    @abstractmethod
    def subtokenize(self, text: str) -> List[str]:
        """Subtokenizes text.

        Takes text and produces a list of subtokens. The subtokenizer should not add special
        characters (CLS, PAD, EOS, etc.).

        Args:
            text: A string with the input text.

        Returns:
            A list of subtokens created from the input text.
        """

    @abstractmethod
    def encode(self, subtokens: List[str]) -> List[int]:
        """Encodes subtokens into input IDs.

        Takes a list of subtokens (from the subtokenize() function) and converts each element into
        their input ID. The encoder should not add special characters (CLS, PAD, EOS, etc.).

        Args:
            subtokens: A list of subtokens from the subtokenize() function.

        Returns:
            A list of input IDs corresponding to each element in the input subtokens list.
        """

    def __call__(self, text: str) -> List[int]:
        """Tokenizes text into input IDs.

        Same as calling encode(subtokenize(text)).

        Args:
            text: A string with the input text.

        Returns:
            A list of input IDs created from the input text.
        """
        return self.encode(self.subtokenize(text))

    @property
    @abstractmethod
    def num_sep_between_pair(self) -> int:
        """When the tokenizer is given a pair of inputs, how many SEP tokens to add between them."""

    @property
    @abstractmethod
    def cls_token_id(self) -> int:
        """The token ID of the CLS token."""

    @property
    @abstractmethod
    def sep_token_id(self) -> int:
        """The token ID of the SEP token."""

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """The token ID of the EOS token."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """The token ID of the PAD token."""


class HFTokenizerForReadingComprehension(TokenizerForReadingComprehension, Configurable):
    """Generic Hugging Face tokenizer wrapper for reading comprehension."""

    _tokenizer: PreTrainedTokenizer
    _num_sep_between_pair: int

    @config_class
    class Config(Configurable.Config):
        # The huggingface tokenizer.
        tokenizer: Required[PreTrainedTokenizer] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self._tokenizer = cfg.tokenizer.instantiate()
        self._num_sep_between_pair = (
            self._tokenizer.num_special_tokens_to_add(pair=True)
            - self._tokenizer.num_special_tokens_to_add()
        )

    def subtokenize(self, text: str) -> List[str]:
        """Subtokenizes text.

        Args:
            text: A string with the input text.

        Returns:
            A list of subtokens created from the input text.
        """
        return self._tokenizer.tokenize(text)

    def encode(self, subtokens: List[str]) -> List[int]:
        """Encodes subtokens into input IDs.

        Args:
            subtokens: A list of subtokens from the subtokenize() function.

        Returns:
            A list of input IDs corresponding to each element in the input subtokens list.
        """
        return self._tokenizer.encode(subtokens, add_special_tokens=False)

    @property
    def num_sep_between_pair(self):
        """When the tokenizer is given a pair of inputs, how many SEP tokens to add between them."""
        return self._num_sep_between_pair

    @property
    def cls_token_id(self):
        """The token ID of the CLS token."""
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        """The token ID of the SEP token."""
        return self._tokenizer.sep_token_id

    @property
    def eos_token_id(self):
        """The token ID of the EOS token."""
        # Some tokenizers do not specify an EOS token, but instead just use the SEP token (e.g.
        # BERT)
        return (
            self._tokenizer.eos_token_id
            if self._tokenizer.eos_token_id is not None
            else self._tokenizer.sep_token_id
        )

    @property
    def pad_token_id(self):
        """The token ID of the PAD token."""
        return self._tokenizer.pad_token_id


def build_example_dataset_fn(
    examples: List[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]],
) -> BuildDatasetFn:
    """Returns a BuildDatasetFn that generates reading comprehension examples from dictionaries.

    Args:
        - examples: A list of dictionaries that are examples which each contain:
            - "context": The passage string on which to perform reading comprehension.
            - "question": The question string.
            - "answers": A list of dicts, each describing an answer for the
                question from the context, where each dict contains a "text" key
                with a string value indicating the answer span and an "answer_start"
                key with an int value indicating the index of the start character.

    Returns:
        A BuildDatasetFn that builds a dataset where each output example is a dict containing:
        - "context": The passage string of shape [1].
        - "question": The question string of shape [1].
        - "answer_texts": A string tensor of shape [num_answers] whose values are the answers texts.
        - "answer_start_positions": An int tensor of shape [num_answers] whose values are
          in range [0, len(answer_texts[i])-1] for the corresponding answer i in answer_texts
          indicating the starting string index of the answer in the context.
    """

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for ex in examples:
                context = tf.convert_to_tensor([ex["context"]])
                question = tf.convert_to_tensor([ex["question"]])
                texts = tf.convert_to_tensor([ans["text"] for ans in ex["answers"]])
                start_positions = tf.convert_to_tensor(
                    [ans["answer_start"] for ans in ex["answers"]]
                )

                yield dict(
                    context=context,
                    question=question,
                    answer_texts=texts,
                    answer_start_positions=start_positions,
                )

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                "context": tf.TensorSpec(shape=(1,), dtype=tf.string),
                "question": tf.TensorSpec(shape=(1,), dtype=tf.string),
                "answer_texts": tf.TensorSpec(shape=(None,), dtype=tf.string),
                "answer_start_positions": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
        )

    return ds_fn


def parse_examples_jsonl() -> DatasetToDatasetFn:
    """Returns a DatasetToDatasetFn that generates reading comprehension examples from JSON lines.

    Returns:
        A DatasetToDatasetFn where:

        Each input example is a JSON line containing:
        - "context": The passage string on which to perform reading comprehension.
        - "question": The question string.
        - "answers": A list of dicts, each describing an answer for the question from the context,
            where each dict contains a "text" key with a string value indicating the answer span
            and an "answer_start" key with an int value indicating the index of the start character.

        Each output example is a dict containing:
        - "context": The passage string of shape [1].
        - "question": The question string of shape [1].
        - "answer_texts": A string tensor of shape [num_answers] whose values are the answers texts.
        - "answer_start_positions": An int tensor of shape [num_answers] whose values are
          in range [0, len(answer_texts[i])-1] for the corresponding answer i in answer_texts
          indicating the starting string index of the answer in the context.
    """

    def parse_line(input_line: tf.Tensor) -> Dict[str, tf.Tensor]:
        dtypes = OrderedDict(
            context=tf.string,
            question=tf.string,
            answer_texts=tf.string,
            answer_start_positions=tf.int32,
        )

        def parse_json(s):
            example = json.loads(s.numpy())
            field_values = [
                tf.convert_to_tensor([example["context"]], dtype=tf.string),
                tf.convert_to_tensor([example["question"]], dtype=tf.string),
                tf.convert_to_tensor([ans["text"] for ans in example["answers"]], dtype=tf.string),
                tf.convert_to_tensor(
                    [ans["answer_start"] for ans in example["answers"]], dtype=tf.int32
                ),
            ]
            return field_values

        parsed = tf.py_function(
            parse_json,
            inp=[input_line],
            Tout=[*dtypes.values()],
        )

        context, question, texts, start_positions = parsed
        context.set_shape((1,))
        question.set_shape((1,))
        texts.set_shape((None,))
        start_positions.set_shape((None,))

        return dict(
            context=context,
            question=question,
            answer_texts=texts,
            answer_start_positions=start_positions,
        )

    return map_over_dataset(parse_line)


def examples_to_jsonl_dataset(
    examples: List[Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]],
) -> BuildDatasetFn:
    """Returns a BuildDatasetFn that generates JSON lines from example dictionaries.

    Args:
        - examples: A list of dictionaries that are examples which each contain:
            - "context": The passage string on which to perform reading comprehension.
            - "question": The question string.
            - "answers": A list of dicts, each describing an answer for the
                question from the context, where each dict contains a "text" key
                with a string value indicating the answer span and an "answer_start"
                key with an int value indicating the index of the start character.

    Returns:
        A BuildDatasetFn that builds a dataset whose lines are JSON lines representing
        each example from the input examples.
    """

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for ex in examples:
                yield tf.convert_to_tensor(json.dumps(ex))

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature=tf.TensorSpec(shape=[], dtype=tf.string),
        )

    return ds_fn


def convert_example_to_features(
    tokenizer: TokenizerForReadingComprehension,
    max_length: int,
    doc_stride: int,
    is_training: bool,
    max_query_length: int = 64,
) -> DatasetToDatasetFn:
    """Returns a DatasetToDatasetFn that converts a reading comprehension example to input features.

    This generic function prepares a reading comprehension example into multiple input features by
    tokenizing and chunking the example based off of the TokenizerForReadingComprehension, max
    input length, and a given document stride. For documents that are longer than the max_length,
    the document is "chunked" into multiple model inputs such that there is a doc_stride amount of
    token overlap between the chunks of the context.

    Args:
        tokenizer: The TokenizerForReadingComprehension to tokenize and encode the input.
        max_length: The maximum model input length.
        doc_stride: For sequences longer than max_length, the amount of overlap between chunks.
        is_training: A flag of whether the tokenizer is used for training.
        max_query_length: The maximum length of the query in tokens.

    Returns:
        A DatasetToDatasetFn where:

        Each input example is a dictionary containing:
        - "context": The passage string on which to perform reading comprehension.
        - "question": The question string.
        - "answer_texts": A list of size (1,) which is the string value indicating the answer.
        - "answer_start_positions": A list of size (1,) which is an int value indicating the index
            of the start character of the answer in the context.

        Each output example is a dictionary containing:
        - "input_ids": An int32 tensor of size (max_length,) containing the padded model input as
            token IDs, complete with special tokens.
        - "token_type_ids": An int32 tensor of size (max_length,) containing padded segment token
            indices with 0 to indicate first portion of the input and 1 for the second portion.
        - "start_positions": An int32 tensor of size (1,) containing the position in input_ids which
            the answer span begins.
        - "end_positions": An int32 tensor of size (1,) containing the position in input_ids where
            the answer span ends.
    """
    model_input_names = ["input_ids", "token_type_ids", "start_positions", "end_positions"]

    # Necessary when the tokenizer is specified as an InstantiableConfig.
    tokenizer = maybe_instantiate(tokenizer)

    def convert_example_to_features_fn(
        question: tf.Tensor,
        context: tf.Tensor,
        answer_texts: tf.Tensor,
        answer_start_positions: tf.Tensor,
    ) -> List[tf.Tensor]:
        question = question.numpy().item().decode("utf-8")
        context = context.numpy().item().decode("utf-8")
        answer_texts = [s.decode("utf-8") for s in answer_texts.numpy().tolist()]
        answer_start_positions = answer_start_positions.numpy().tolist()

        # Split the context into whitespace-delimited tokens, and preserve a record of the mapping
        # from each original character index to their respective the document tokens.
        doc_tokens, char_to_doc_token_offset = _get_doc_tokens_and_char_offsets(context=context)

        # Use the subtokenize of the tokenizer to subtokenize each whitespace-delimited token into
        # their respective subtokens. Additionally computes the subtoken indices the answer span by
        # mapping from the original answer character indices to their subtoken indices.
        (
            doc_subtokens,
            answer_subtoken_start,
            answer_subtoken_end,
        ) = _get_subtokens_and_answer_subtoken_indices(
            tokenizer=tokenizer,
            doc_tokens=doc_tokens,
            char_to_doc_token_offset=char_to_doc_token_offset,
            answer_start_positions=answer_start_positions,
            answer_texts=answer_texts,
            is_training=is_training,
        )

        # Split the query into its own subtokens.
        query_subtokens = tokenizer.subtokenize(question)[:max_query_length]

        # Compute the maximum chunk of the context we can fit into a single model input. The
        # non-context part of the model input is composed of the query subtokens and the special
        # tokens added to the model (CLS, EOS, SEP), so we take the maximum model length and
        # subtract the size of the non-context part of the model input.
        max_doc_subtokens_per_chunk = (
            max_length - len(query_subtokens) - (2 + tokenizer.num_sep_between_pair)
        )

        doc_subtoken_chunks = _get_doc_subtoken_chunks(
            doc_subtokens=doc_subtokens,
            max_doc_subtokens_per_chunk=max_doc_subtokens_per_chunk,
            doc_stride=doc_stride,
        )

        # Each chunk will have the same pre-context input, which is CLS token, query tokens, then
        # the SEP tokens.
        pre_chunk_tokens_ids = (
            [tokenizer.cls_token_id]
            + tokenizer.encode(query_subtokens)
            + ([tokenizer.sep_token_id] * tokenizer.num_sep_between_pair)
        )

        features = _featurize_chunks(
            tokenizer=tokenizer,
            model_input_names=model_input_names,
            pre_chunk_tokens_ids=pre_chunk_tokens_ids,
            doc_subtoken_chunks=doc_subtoken_chunks,
            max_doc_subtokens_per_chunk=max_doc_subtokens_per_chunk,
            answer_subtoken_start=answer_subtoken_start,
            answer_subtoken_end=answer_subtoken_end,
            doc_stride=doc_stride,
            max_length=max_length,
            is_training=is_training,
        )

        return [tf.convert_to_tensor(features[k]) for k in model_input_names]

    def prepare_features(input_example: Dict[str, tf.Tensor]) -> Dict[str, List[tf.Tensor]]:
        tokenized = tf.py_function(
            func=convert_example_to_features_fn,
            inp=(
                input_example["question"],
                input_example["context"],
                input_example["answer_texts"],
                input_example["answer_start_positions"],
            ),
            Tout=[tf.int32 for _ in model_input_names],
        )

        return {k: tokenized[i] for i, k in enumerate(model_input_names)}

    class ConvertExampleToFeaturesFn(DatasetToDatasetFn):
        def __call__(self, ds: tf.data.Dataset, **kwargs) -> tf.data.Dataset:
            ds = map_over_dataset(prepare_features)(ds)
            return ds.unbatch()

    return ConvertExampleToFeaturesFn()


def _get_doc_tokens_and_char_offsets(context: str) -> Tuple[List[str], List[int]]:
    """Splits the context into whitespace-delimited tokens and returns character offsets

    Helper function to split the context into whitespace-delimited tokens, and return both the
    document tokens and the mapping from each original character index to their respective the
    document tokens.

    Args:
        context: The input context.

    Returns:
        A list of whitespace-delimited tokens from the context as well as a list mapping each
        original context character to a whitespace-delimited token.
    """
    doc_tokens = []
    char_to_doc_token_offset = []
    prev_is_whitespace = True
    for c in context:
        if c.isspace():
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_doc_token_offset.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_doc_token_offset


def _get_subtokens_and_answer_subtoken_indices(
    tokenizer: TokenizerForReadingComprehension,
    doc_tokens: List[str],
    char_to_doc_token_offset: List[int],
    answer_start_positions: List[int],
    answer_texts: List[str],
    is_training: bool,
) -> Tuple[List[str], int, int]:
    """Subtokenize and return the answer subtoken indices

    Use the subtokenize of the tokenizer to subtokenize each whitespace-delimited token into
    their respective subtokens. Additionally computes the subtoken indices the answer span by
    mapping from the original answer character indices to their subtoken indices.

    Args:
        tokenizer: The tokenizer to use for subtokenization.
        doc_tokens: The whitespace-delimited tokens in the context.
        char_to_doc_token_offset: A mapping from each original character index to their respective
            the document tokens.
        answer_start_positions: A list containing an int value indicating the index of the start
            character of the answer in the context.
        answer_texts: A list containing a string value indicating the answer text in the context.
        is_training: Indicate if these are training examples.

    Returns:
        Returns the document subtokens, the answer start subtoken index, and the answer end subtoken
        index.
    """

    doc_subtokens = []
    doc_token_to_subtoken_index = []
    doc_token_to_subtoken_count = []
    for token in doc_tokens:
        doc_token_to_subtoken_index.append(len(doc_subtokens))

        # To let tokenizers handle prefix_space where relevant (e.g. RoBERTa) we add a space to
        # the beginning of each token.
        subtokens = tokenizer.subtokenize(" " + token)

        doc_token_to_subtoken_count.append(len(subtokens))
        for subtoken in subtokens:
            doc_subtokens.append(subtoken)

    if is_training and len(answer_start_positions) > 0:
        # Compute the subtoken indices the answer span by mapping from the original answer
        # character indices to their context token indices, then to their subtoken indices.
        answer_token_start = char_to_doc_token_offset[answer_start_positions[0]]
        answer_token_end = char_to_doc_token_offset[
            answer_start_positions[0] + len(answer_texts[0]) - 1  # -1 to point at last character.
        ]
        answer_subtoken_start = doc_token_to_subtoken_index[answer_token_start]
        answer_subtoken_end = (
            doc_token_to_subtoken_index[answer_token_end]
            + doc_token_to_subtoken_count[answer_token_end]
            - 1  # -1 to point at last subtoken.
        )
    else:
        # Indicate an impossible answer start and end, as a start index of -1 is impossible.
        answer_subtoken_start = -1
        answer_subtoken_end = -1

    return doc_subtokens, answer_subtoken_start, answer_subtoken_end


def _get_doc_subtoken_chunks(
    doc_subtokens: List[str], max_doc_subtokens_per_chunk: int, doc_stride: int
) -> List[List[int]]:
    """Produce chunks of the context.

    Produce chunks of the context based off of the max_doc_subtokens_per_chunk and the
    doc_stride. For contexts that larger than max_doc_subtokens_per_chunk are produced by
    using a sliding window approach, with a window size of max_doc_subtokens_per_chunk and a
    stride of doc_stride.

    Args:
        doc_subtokens: The whitespace-delimited tokens in the context.
        max_doc_subtokens_per_chunk: The maximum subtokens per chunk.
        doc_stride: The amount to move the sliding window to create the next chunk.

    Returns:
        A list of context chunks.
    """

    doc_subtoken_chunks = []
    curr_doc_subtoken_chunk_offset = 0
    is_window_done_sliding = False
    while not is_window_done_sliding:
        # Create a chunk by grabbing a window of size max_doc_subtokens_per_chunk.
        doc_subtoken_chunks.append(
            doc_subtokens[
                curr_doc_subtoken_chunk_offset : curr_doc_subtoken_chunk_offset
                + max_doc_subtokens_per_chunk
            ]
        )

        # Test if the chunk we just created goes past the end of the context length. If so, we
        # are done sliding the window and creating chunks.
        is_window_done_sliding = (
            curr_doc_subtoken_chunk_offset + max_doc_subtokens_per_chunk
        ) >= len(doc_subtokens)

        # Move the start of the window by doc_stride.
        curr_doc_subtoken_chunk_offset = len(doc_subtoken_chunks) * doc_stride

    return doc_subtoken_chunks


def _featurize_chunks(
    tokenizer: TokenizerForReadingComprehension,
    model_input_names: List[str],
    pre_chunk_tokens_ids: List[int],
    doc_subtoken_chunks: List[List[str]],
    max_doc_subtokens_per_chunk: int,
    answer_subtoken_start: int,
    answer_subtoken_end: int,
    doc_stride: int,
    max_length: int,
    is_training: bool,
) -> Dict[str, List]:
    """Creates the model input features for each chunk.

    Args:
        tokenizer: THe tokenizer to encode subtokens into token IDs.
        model_input_names: The names of model input features.
        pre_chunk_tokens_ids: All the token IDs that precede the context chunk token IDs.
        doc_subtoken_chunks: All the context chunks.
        max_doc_subtokens_per_chunk: The maximum subtokens per chunk.
        answer_subtoken_start: The subtoken index in the context where the answer starts.
        answer_subtoken_end: The subtoken index in the context where the answer ends.
        doc_stride: The amount to move the sliding window to create the next chunk.
        max_length: The maximum model input length.
        is_training: Indicate if these are training examples.

    Returns:
        A dictionary of features, where each entry contains a list of that feature for every chunk.
    """

    features = {k: [] for k in model_input_names}
    for chunk_index, doc_subtoken_chunk in enumerate(doc_subtoken_chunks):
        # To fill the model input to max length, we calculate the length of the pre-padded input
        # to determine padding length.
        padding_length = max_length - (len(pre_chunk_tokens_ids) + len(doc_subtoken_chunk) + 1)

        # After the pre-context parts of the input, the remaining parts are the context chunk,
        # EOS token, then padding tokens to hit max length.
        features["input_ids"].append(
            pre_chunk_tokens_ids
            + tokenizer.encode(doc_subtoken_chunk)
            + [tokenizer.eos_token_id]
            + ([tokenizer.pad_token_id] * padding_length)
        )
        # The token_type_ids are 1 for all the tokens relating to the context, and 0 otherwise.
        features["token_type_ids"].append(
            [0] * len(pre_chunk_tokens_ids)
            + [1] * len(doc_subtoken_chunk)
            + [1]
            + ([0] * padding_length)
        )

        if is_training:
            # Determine the subtoken start and end of the chunk in the original, unchunked
            # context subtokens.
            chunk_start = chunk_index * (max_doc_subtokens_per_chunk - doc_stride)
            chunk_end = chunk_start + len(doc_subtoken_chunk)

            # Determine if the answer span occurs in the current chunk.
            if answer_subtoken_start >= chunk_start and answer_subtoken_end <= chunk_end:
                # If so, we will calculate the answer span for this particular model input with
                # this particular chunk, offsetting by this chunk's start and the pre-context
                # input.
                features["start_positions"].append(
                    answer_subtoken_start - chunk_start + len(pre_chunk_tokens_ids)
                )
                features["end_positions"].append(
                    answer_subtoken_end - chunk_start + len(pre_chunk_tokens_ids)
                )
            else:
                # If this answer span does not contain the answer span, we'll use the
                # cls_token_id as an indicator of this.
                features["start_positions"].append(tokenizer.cls_token_id)
                features["end_positions"].append(tokenizer.cls_token_id)
        else:
            # When not training, use placeholder start and end answer positions.
            features["start_positions"].append(0)
            features["end_positions"].append(0)

    return features
