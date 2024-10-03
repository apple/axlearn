# Copyright © 2024 Apple Inc.

"""Tests language modeling inputs."""

import os

import grain.python as grain
import numpy as np
import pytest
import seqio
from absl.testing import parameterized

from axlearn.common.input_fake import fake_grain_source
from axlearn.common.input_grain import Input, Tensor, maybe_to_iter_dataset, prefetch_dataset
from axlearn.common.input_grain_lm import (
    _make_autoregressive_inputs,
    _trim_or_pad_and_batch,
    text_to_lm_training_input,
)
from axlearn.common.input_grain_text import with_regex_mapping
from axlearn.common.input_grain_text_test import t5_sentence_piece_vocab_file
from axlearn.common.test_utils import TestCase


class MakeAutoregressveInputsTest(TestCase):
    """Tests `make_autoregressive_inputs`."""

    @parameterized.parameters(
        # Test a case without windowing.
        dict(
            target_labels=[
                np.array([10, 11, 12, 1, 21, 22, 23, 24, 1, 25]),
            ],
            expected=[
                dict(
                    target_labels=np.array([10, 11, 12, 1, 21, 22, 23, 24, 1, 25]),
                    input_ids=np.array([25, 10, 11, 12, 1, 21, 22, 23, 24, 1]),
                ),
            ],
            max_len=10,
        ),
        # No windowing, with truncating.
        dict(
            target_labels=[
                np.array([10, 11, 12, 1, 21, 22, 23, 24, 1, 25]),
            ],
            expected=[
                dict(
                    input_ids=np.array([25, 10, 11, 12, 1]),
                    target_labels=np.array([10, 11, 12, 1, 21]),
                ),
                dict(
                    input_ids=np.array([21, 22, 23, 24, 1]),
                    target_labels=np.array([22, 23, 24, 1, 25]),
                ),
            ],
            max_len=5,
        ),
        # Windowing across ragged with padding.
        dict(
            target_labels=[
                np.array([10, 11, 12, 1, 21]),
                np.array([100, 1, 102, 103, 104, 105]),
            ],
            expected=[
                dict(
                    target_labels=np.array([10, 11, 12, 1, 21, 100, 1, 102]),
                    input_ids=np.array([105, 10, 11, 12, 1, 21, 100, 1]),
                ),
                dict(
                    target_labels=np.array([103, 104, 105, -1, -1, -1, -1, -1]),
                    input_ids=np.array([102, 103, 104, -1, -1, -1, -1, -1]),
                ),
            ],
            max_len=8,
            window_size=2,  # Divides number of inputs evenly.
        ),
        # Windowing across ragged with padding.
        dict(
            target_labels=[
                np.array([10, 11, 12, 1, 21]),
                np.array([100, 1, 102, 103, 104, 105]),
            ],
            expected=[
                dict(
                    target_labels=np.array([10, 11, 12, 1, 21, 100]),
                    input_ids=np.array([105, 10, 11, 12, 1, 21]),
                ),
                dict(
                    target_labels=np.array([1, 102, 103, 104, 105, -1]),
                    input_ids=np.array([100, 1, 102, 103, 104, -1]),
                ),
            ],
            max_len=6,
            window_size=3,  # Does not divide number of inputs evenly.
        ),
    )
    def test_make_autoregressive_inputs(
        self, target_labels: list, expected: list, max_len: int, window_size: int = 1
    ):
        split_fn = _trim_or_pad_and_batch
        ds = fake_grain_source([{"target_labels": x} for x in target_labels])
        ds = _make_autoregressive_inputs(
            ds, max_len=max_len, split_fn=split_fn, window_size=window_size
        )
        actual = list(ds)
        self.assertNestedEqual(expected, actual)


class TrimOrPadAndBatchTest(TestCase):
    """Tests `trim_and_pad_batch`."""

    @parameterized.parameters(
        dict(
            ids=np.ones([1], dtype=np.int32),
            expected=np.array([[1, -1, -1]], dtype=np.int32),
            max_len=3,
            max_padding_fraction=1.0,  # Always pad.
        ),
        dict(
            ids=np.arange(1, 5),
            expected=np.array([[1, 2, 3]]),
            max_len=3,
            max_padding_fraction=0.5,  # Drop last.
        ),
        dict(
            ids=np.arange(1, 6),
            expected=np.array([[1, 2, 3], [4, 5, -1]]),
            max_len=3,
            max_padding_fraction=0.5,  # Keep last.
        ),
        dict(
            ids=np.arange(1, 6),
            expected=np.array([[1, 2, 3]]),
            max_len=3,
            max_padding_fraction=0.0,  # Never pad.
        ),
    )
    def test_trim_or_pad_and_batch(
        self, ids: Tensor, expected: Tensor, max_len: int, max_padding_fraction: float
    ):
        self.assertNestedEqual(
            expected,
            _trim_or_pad_and_batch(ids, max_len=max_len, max_padding_fraction=max_padding_fraction),
        )

    def test_rank(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            _trim_or_pad_and_batch(np.ones([2, 2]), max_len=1)


class LmTrainingInputTest(TestCase):
    """Tests `text_to_lm_training_input`."""

    @parameterized.parameters("Lorem ipsum dolor sit amet,", " consectetur adipiscing elit\n")
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_training_lm_processor_single_example(self, text: str):
        max_len = 32
        vocab = seqio.SentencePieceVocabulary(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )
        examples = [{"text": text, "index": i} for i in range(10)]
        ds = fake_grain_source(examples)
        ds = text_to_lm_training_input(
            ds,
            vocab=vocab,
            max_len=max_len,
            window_size=3,
            max_padding_fraction=0.0,
        )
        example = next(iter(ds))
        for key in ["input_ids", "target_labels"]:
            # Shape is as expected.
            self.assertEqual((max_len,), example[key].shape)
        self.assertTrue("target_num_bytes" in example)
        input_ids, target_labels = example["input_ids"], example["target_labels"]
        self.assertTrue(vocab.eos_id in input_ids)  # EOS somewhere in the inputs.
        self.assertTrue(vocab.eos_id in target_labels)  # EOS somewhere in the targets.
        # The inputs should be one-off the labels.
        self.assertNestedAllClose(target_labels[:-1], input_ids[1:])

    @parameterized.parameters(
        dict(
            expected_batches=[
                {
                    "input_ids": [
                        [1, 21820, 296, 2, 29],
                        [3155, 1, 21820, 8114, 2],
                    ],
                    "target_labels": [
                        [21820, 296, 2, 29, 3155],
                        [1, 21820, 8114, 2, 29],
                    ],
                    "target_num_bytes": [18, 17],
                },
                {
                    "input_ids": [
                        [29, 3155, 1, 21820, 296],
                        [2, 29, 3155, 0, 0],
                    ],
                    "target_labels": [
                        [3155, 1, 21820, 296, 2],
                        [29, 3155, 1, 0, 0],
                    ],
                    "target_num_bytes": [19, 3],
                },
            ],
            max_padding_fraction=1.0,  # Always pad
        ),
        dict(
            expected_batches=[
                {
                    "input_ids": [
                        [1, 21820, 296, 2, 29],
                        [3155, 1, 21820, 8114, 2],
                    ],
                    "target_labels": [
                        [21820, 296, 2, 29, 3155],
                        [1, 21820, 8114, 2, 29],
                    ],
                    "target_num_bytes": [18, 17],
                },
                {
                    "input_ids": [
                        [29, 3155, 1, 21820, 296],
                        [2, 29, 3155, 0, 0],
                    ],
                    "target_labels": [
                        [3155, 1, 21820, 296, 2],
                        [29, 3155, 1, 0, 0],
                    ],
                    "target_num_bytes": [19, 3],
                },
            ],
            max_padding_fraction=1.0,  # Always pad
        ),
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_fake_text_lm_training_data(
        self,
        expected_batches: list[dict[str, Tensor]],
        max_padding_fraction: float,
    ):
        examples = [
            {"text": "hello world\n"},
            {"text": "hello moon\n"},
        ]

        # window_size > len(texts) to repeat the sentence. 18 tokens in total.
        # [    1,  21820,   296,     2,    29,  3155,     1, 21820,  8114,
        #      2,     29,  3155,     1, 21820,   296,     2,    29,  3155]
        window_size = 3

        # Pad the concatenated sequence to 20 tokens:
        # [    1,  21820,   296,     2,    29,  3155,     1, 21820,  8114,    2
        #     29,  3155,     1, 21820,   296,     2,    29,  3155,      0,    0]
        #
        # Or, trim the sequence to 15 tokens:
        # [    1,  21820,   296,     2,    29,  3155,     1, 21820,  8114,    2
        #     29,  3155,     1, 21820,   296]
        batch_size, max_len = 2, 5

        vocab_cls = with_regex_mapping(
            seqio.SentencePieceVocabulary,
            encode_mapping=[("\n", "<n>")],
            decode_mapping=[("<n>", "\n")],
        )
        vocab = vocab_cls(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )

        def source():
            ds = fake_grain_source(examples)
            ds = text_to_lm_training_input(
                ds,
                vocab=vocab,
                max_len=max_len,
                window_size=window_size,
                max_padding_fraction=max_padding_fraction,
            )
            ds = ds.batch(batch_size=batch_size)
            ds = maybe_to_iter_dataset(
                ds,
                read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=2),
            )
            ds = prefetch_dataset(
                ds,
                multiprocessing_options=grain.MultiprocessingOptions(
                    num_workers=1,  # Set explicitly to avoid using all cpus for testing.
                    enable_profiling=False,
                ),
            )
            return ds

        # Test text_to_lm_training_input.
        cfg: Input.Config = Input.default_config().set(name="test_input", source=source)
        ds = cfg.instantiate(parent=None)
        for ix, example in enumerate(ds):
            self.assertIsNotNone(example)
            if ix < len(expected_batches):
                # Check for equality for provided batches.
                example = {k: v.tolist() for k, v in example.items()}
                self.assertNestedAllClose(expected_batches[ix], example)
            if ix >= 10 * len(expected_batches):
                # Expect to be able to repeat forever.
                break
