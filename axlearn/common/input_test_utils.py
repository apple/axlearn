# Copyright © 2023 Apple Inc.

"""Common test utilities for input processing tests."""

import os
from collections.abc import Sequence

import tensorflow as tf

from axlearn.common import input_tf_data

tokenizers_dir = os.path.join(os.path.dirname(__file__), "../data/tokenizers")
SENTENCEPIECE_DIR = os.path.join(tokenizers_dir, "sentencepiece")
BPE_DIR = os.path.join(tokenizers_dir, "bpe")
t5_sentence_piece_vocab_file = os.path.join(SENTENCEPIECE_DIR, "t5-base")
opt_vocab_file = os.path.join(BPE_DIR, "opt.model")
roberta_base_vocab_file = os.path.join(BPE_DIR, "roberta-base-vocab.json")
roberta_base_merges_file = os.path.join(BPE_DIR, "roberta-base-merges.txt")


def count_batches(dataset, max_batches=100):
    count = 0
    for _ in dataset:
        if count >= max_batches:
            return -1
        count += 1
    return count


def make_ds_fn(
    is_training: bool, texts: list[str], repeat: int = 100
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for _ in range(repeat):
                for index, text in enumerate(texts):
                    yield {"text": text, "index": index}

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                "text": tf.TensorSpec(shape=(), dtype=tf.string),
                "index": tf.TensorSpec(shape=(), dtype=tf.uint32),
            },
        )

    return ds_fn


def make_ragged_ds_fn(
    is_training: bool, texts: list[dict], repeat: int = 100
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for _ in range(repeat):
                for index, item in enumerate(texts):
                    yield {"text": tf.ragged.constant(item["text"]), "index": index}

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                "text": tf.RaggedTensorSpec(shape=([None, None]), dtype=tf.string),
                "index": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
        )

    return ds_fn


def make_seq2seq_ds_fn(
    is_training: bool,
    sources: list[str],
    targets: list[str],
    repeat: int = 100,
    source_key: str = "source",
    target_key: str = "target",
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def ds_fn() -> tf.data.Dataset:
        def data_gen():
            for _ in range(repeat):
                for index, (source, target) in enumerate(zip(sources, targets)):
                    yield {source_key: source, target_key: target, "index": index}

        return tf.data.Dataset.from_generator(
            data_gen,
            output_signature={
                source_key: tf.TensorSpec(shape=(), dtype=tf.string),
                target_key: tf.TensorSpec(shape=(), dtype=tf.string),
                "index": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
        )

    return ds_fn


def extract_text(example: dict[str, tf.Tensor], input_key: str = "text") -> str:
    return bytes.decode(example[input_key].numpy(), "utf-8")


def extract_text_ragged(example: dict[str, tf.Tensor], input_key: str = "text") -> list[list[str]]:
    result = []
    for item in example[input_key].numpy():
        sub_result = []
        for sub_item in item:
            sub_result.append(bytes.decode(sub_item, "utf-8"))
        result.append(sub_result)
    return result


def assert_oneof(test_case: tf.test.TestCase, actual: tf.Tensor, candidates: Sequence[tf.Tensor]):
    if not isinstance(test_case, tf.test.TestCase):
        raise ValueError("test_case should be an instance of tf.test.TestCase")
    for candidate in candidates:
        try:
            test_case.assertAllEqual(actual, candidate)
            return
        except AssertionError:
            pass
    raise AssertionError(f"Expected {actual} to be equal to one of {candidates}")
