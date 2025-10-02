# Copyright © 2024 Apple Inc.

"""Tests language modeling inputs."""

import functools
import os
from typing import Callable

import grain.python as grain
import numpy as np
import pytest
import seqio
from absl.testing import parameterized

from axlearn.common.input_fake import fake_grain_source
from axlearn.common.input_grain import Input, Tensor, maybe_to_iter_dataset, prefetch_dataset
from axlearn.common.input_grain_lm import (
    _drop_empty_targets,
    _make_autoregressive_inputs,
    _trim_or_pad_and_batch,
    streaming_packing,
    text_to_lm_eval_input,
    text_to_lm_training_input,
    windowed_packing,
)
from axlearn.common.input_grain_text import with_regex_mapping
from axlearn.common.input_test_utils import t5_sentence_piece_vocab_file
from axlearn.common.test_utils import TestCase


class MakeAutoregressveInputsTest(TestCase):
    """Tests `make_autoregressive_inputs`."""

    def _test_checkpointing(self, ds: grain.DatasetIterator):
        """Utility to test ds checkpointing."""

        max_steps = 10
        values_without_interruption: list[dict] = []
        checkpoints = []

        for _ in range(max_steps):
            checkpoints.append(ds.get_state())
            values_without_interruption.append(next(ds))

        def check(starting_step, ds):
            for i in range(starting_step, max_steps):
                actual = next(ds)
                expected = values_without_interruption[i]
                for k, v in expected.items():
                    self.assertNestedEqual(v, actual[k])

        # Try resuming from an existing iterator, to ensure that entire state is reset.
        for starting_step in range(max_steps):
            ds.set_state(checkpoints[starting_step])  # Restore using the same iterator as above.
            check(starting_step, ds)

        # Try resuming from a fresh iterator from any step and validate that outcome is the same.
        for starting_step in range(max_steps):
            ds = iter(ds)
            ds.set_state(checkpoints[starting_step])
            check(starting_step, ds)

    @parameterized.parameters(
        dict(
            target_labels=[
                np.array([33, 33]),
                np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]),
                np.array([7, 8, 9]),
                np.array([71, 25, 323]),
            ],
            max_len=2,
            window_size=3,
            packing_fn=streaming_packing,
            max_padding_fraction=0.5,
        ),
        dict(
            target_labels=[
                np.array([33, 33]),
                np.array([1, 3]),
                np.array([7, 8]),
                np.array([1, 2]),
                np.array([10, 10]),
                np.array([9, 2]),
                np.array([4, 4]),
                np.array([9, 9]),
                np.array([42, 41]),
                np.array([16, 16]),
            ],
            max_len=2,
            window_size=1,
            packing_fn=streaming_packing,
            max_padding_fraction=0.5,
        ),
    )
    def test_make_autoregressive_checkpointing(
        self,
        target_labels: list,
        max_len: int,
        window_size: int = 1,
        packing_fn: Callable = windowed_packing,
        max_padding_fraction: float = 1.0,
    ):
        """Note that there is no real "checkpoints"."""
        split_fn = functools.partial(
            _trim_or_pad_and_batch, max_padding_fraction=max_padding_fraction
        )
        ds = fake_grain_source([{"target_labels": x} for x in target_labels])
        ds = _make_autoregressive_inputs(
            ds, max_len=max_len, split_fn=split_fn, window_size=window_size, packing_fn=packing_fn
        )
        self._test_checkpointing(iter(ds))

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
            packing_fn=streaming_packing,
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
            packing_fn=streaming_packing,
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
            packing_fn=streaming_packing,
        ),
        # Windowing across ragged with padding.
        dict(
            target_labels=[
                np.array([10, 11, 12, 1, 21]),
                np.array([100, 1, 102, 103, 104, 105, 106]),
            ],
            expected=[
                dict(
                    target_labels=np.array([10, 11, 12, 1, 21, 100]),
                    input_ids=np.array([105, 10, 11, 12, 1, 21]),
                ),
                dict(
                    target_labels=np.array([1, 102, 103, 104, 105, 106]),
                    input_ids=np.array([100, 1, 102, 103, 104, 105]),
                ),
            ],
            max_len=6,
            window_size=3,  # Does not divide number of inputs evenly.
            packing_fn=streaming_packing,
        ),
        # One sequence that's significantly longer than max_len.
        dict(
            target_labels=[
                np.array([33, 33]),
                np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]),
            ],
            expected=[
                dict(
                    target_labels=np.array([33, 33, 1, 3, 5, 7]),
                    input_ids=np.array([7, 33, 33, 1, 3, 5]),
                ),
                dict(
                    target_labels=np.array([9, 11, 13, 15, 17, 19]),
                    input_ids=np.array([19, 9, 11, 13, 15, 17]),
                ),
                dict(
                    target_labels=np.array([21, 23, 25, -1, -1, -1]),
                    input_ids=np.array([-1, 21, 23, -1, -1, -1]),
                ),
            ],
            max_len=6,
            window_size=6,
            packing_fn=streaming_packing,
        ),
        # Keep this example
        dict(
            target_labels=[
                np.array([33, 33]),
                np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]),
                np.array([7, 8, 9]),
            ],
            expected=[
                dict(
                    target_labels=np.array([33, 33, 1, 3, 5, 7]),
                    input_ids=np.array([7, 33, 33, 1, 3, 5]),
                ),
                dict(
                    target_labels=np.array([9, 11, 13, 15, 17, 19]),
                    input_ids=np.array([19, 9, 11, 13, 15, 17]),
                ),
                dict(
                    target_labels=np.array([21, 23, 25, 7, 8, 9]),
                    input_ids=np.array([9, 21, 23, 25, 7, 8]),
                ),
            ],
            max_len=6,
            window_size=3,
            packing_fn=streaming_packing,
            max_padding_fraction=0.5,
        ),
        # Drop this example.
        dict(
            target_labels=[
                np.array([33, 33]),
                np.array([1, 3, 5, 7, 9, 11, 13, 15, 17]),
                np.array([7, 8, 9]),
                np.array([1, 2, 3, 4, 5, 6]),
            ],
            expected=[
                dict(
                    target_labels=np.array([33, 33, 1, 3, 5, 7]),
                    input_ids=np.array([7, 33, 33, 1, 3, 5]),
                ),
                dict(
                    target_labels=np.array([9, 11, 13, 15, 17, 7]),
                    input_ids=np.array([7, 9, 11, 13, 15, 17]),
                ),
                dict(
                    target_labels=np.array([1, 2, 3, 4, 5, 6]),
                    input_ids=np.array([6, 1, 2, 3, 4, 5]),
                ),
            ],
            max_len=6,
            window_size=3,
            packing_fn=streaming_packing,
            max_padding_fraction=0.5,
        ),
        dict(
            target_labels=[
                np.array([33, 33]),
                np.array([1, 3]),
                np.array([1, 2, 3, 4, 5, 6]),
            ],
            expected=[
                dict(
                    target_labels=np.array([33, 33]),
                    input_ids=np.array([33, 33]),
                ),
                dict(
                    target_labels=np.array([1, 3]),
                    input_ids=np.array([3, 1]),
                ),
                dict(
                    target_labels=np.array([1, 2]),
                    input_ids=np.array([2, 1]),
                ),
                dict(
                    target_labels=np.array([3, 4]),
                    input_ids=np.array([4, 3]),
                ),
                dict(
                    target_labels=np.array([5, 6]),
                    input_ids=np.array([6, 5]),
                ),
            ],
            max_len=2,
            window_size=3,
            packing_fn=streaming_packing,
            max_padding_fraction=0.5,
        ),
    )
    def test_make_autoregressive_inputs(
        self,
        target_labels: list,
        expected: list,
        max_len: int,
        window_size: int = 1,
        packing_fn: Callable = windowed_packing,
        max_padding_fraction: float = 1.0,
    ):
        split_fn = functools.partial(
            _trim_or_pad_and_batch, max_padding_fraction=max_padding_fraction
        )
        ds = fake_grain_source([{"target_labels": x} for x in target_labels])
        ds = _make_autoregressive_inputs(
            ds, max_len=max_len, split_fn=split_fn, window_size=window_size, packing_fn=packing_fn
        )
        actual = list(ds)
        self.assertEqual(len(expected), len(actual))
        for exp, act in zip(expected, actual):
            self.assertNestedEqual(exp["target_labels"], act["target_labels"])
            # The first element of input_id is rolled over from the last element of this sequence.
            # Skipping it.
            self.assertNestedEqual(exp["input_ids"][1:], act["input_ids"][1:])

    @parameterized.parameters(
        dict(
            target_labels=[
                np.array([33, 33]),
                np.array([1, 3, 5, 7, 9, 11, 13, 15, 17]),
                np.array([7, 8, 9]),
                np.array([1, 2, 3, 4, 5, 6]),
            ],
            max_len=6,
            window_size=3,
            max_padding_fraction=0.5,
        ),
        dict(
            target_labels=[
                np.array([10, 11, 12, 1, 21]),
                np.array([100, 1, 102, 103, 104, 105]),
            ],
            max_len=8,
            window_size=2,  # Divides number of inputs evenly.
        ),
    )
    def test_windowed_packing_streaming_packing_parity(
        self,
        target_labels: list,
        max_len: int,
        window_size: int = 1,
        max_padding_fraction: float = 1.0,
    ):
        split_fn = functools.partial(
            _trim_or_pad_and_batch, max_padding_fraction=max_padding_fraction
        )
        ds = fake_grain_source([{"target_labels": x} for x in target_labels])
        ds = _make_autoregressive_inputs(
            ds,
            max_len=max_len,
            split_fn=split_fn,
            window_size=window_size,
            packing_fn=windowed_packing,
        )
        windowed_packing_results = list(ds)
        ds = fake_grain_source([{"target_labels": x} for x in target_labels])
        ds = _make_autoregressive_inputs(
            ds,
            max_len=max_len,
            split_fn=split_fn,
            window_size=window_size,
            packing_fn=streaming_packing,
        )
        streaming_packing_results = list(ds)
        self.assertEqual(len(windowed_packing_results), len(streaming_packing_results))
        for windowed_result, streaming_result in zip(
            windowed_packing_results, streaming_packing_results
        ):
            self.assertNestedEqual(
                windowed_result["target_labels"], streaming_result["target_labels"]
            )
            # The first element of input_id is rolled over from the last element of this sequence.
            # Skipping it.
            self.assertNestedEqual(
                windowed_result["input_ids"][1:], streaming_result["input_ids"][1:]
            )


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

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_training_lm_processor_infinite_dataset(self):
        max_len = 32
        vocab = seqio.SentencePieceVocabulary(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )
        examples = [{"text": f"test_str_#{i}", "index": i} for i in range(10)]
        ds = fake_grain_source(examples)
        ds = text_to_lm_training_input(
            ds.repeat(),  # check if infinite dataset breaks the pipeline
            vocab=vocab,
            max_len=max_len,
            window_size=3,
            max_padding_fraction=0.0,
        )

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_training_lm_processor_dataset_no_len_func(self):
        max_len = 32
        vocab = seqio.SentencePieceVocabulary(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )
        examples = [{"text": f"test_str_#{i}", "index": i} for i in range(10)]
        ds = prefetch_dataset(
            maybe_to_iter_dataset(fake_grain_source(examples)),
            multiprocessing_options=grain.MultiprocessingOptions(
                num_workers=1,
                per_worker_buffer_size=1,
                enable_profiling=False,
            ),
        )
        ds = text_to_lm_training_input(
            ds,  # check if prefetch dataset breaks the pipeline
            vocab=vocab,
            max_len=max_len,
            window_size=3,
            max_padding_fraction=0.0,
        )

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

        def source(dispatch_config):
            del dispatch_config
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


class LmEvalInputTest(TestCase):
    """Tests `text_to_lm_eval_input`."""

    def _source(self, texts: list[str], max_len: int, batch_size: int = 1):
        vocab_cls = with_regex_mapping(
            seqio.SentencePieceVocabulary,
            encode_mapping=[("\n", "<n>")],
            decode_mapping=[("<n>", "\n")],
        )
        vocab = vocab_cls(
            sentencepiece_model_file=t5_sentence_piece_vocab_file,
        )
        ds = fake_grain_source([{"text": text} for text in texts])
        ds = text_to_lm_eval_input(ds, vocab=vocab, max_len=max_len, stride=2)
        if batch_size > 1:
            ds = ds.batch(batch_size=batch_size)
        ds = maybe_to_iter_dataset(
            ds,
            read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=2),
        )
        return ds

    @parameterized.parameters(
        "How long is a piece of string?",
        "On the 20th of June",
        "Here we stand united",
    )
    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_eval_lm_processor_single_example(self, text):
        max_len = 12
        example = next(iter(self._source(texts=[text], max_len=max_len)))
        for key in ["input_ids", "target_labels"]:
            # Shape is as expected.
            self.assertEqual((max_len,), example[key].shape)
        self.assertTrue("target_num_bytes" in example)

        input_ids, target_labels = example["input_ids"], example["target_labels"]
        self.assertEqual(1, input_ids[0])  # Start of example.
        non_padded_length = target_labels.argmin()
        self.assertNotEqual(1, target_labels[0])  # No EOS at start.
        self.assertEqual(1, target_labels[non_padded_length - 1])  # EOS.
        # The inputs should be one-off the labels.
        self.assertNestedAllClose(
            target_labels[: non_padded_length - 1], input_ids[1:non_padded_length]
        )

    @pytest.mark.skipif(
        not os.path.exists(t5_sentence_piece_vocab_file), reason="Missing testdata."
    )
    def test_fake_text_lm_eval_data(self):
        texts = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit\n",
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        ]
        expected_batches = [
            {
                "input_ids": [
                    [1, 8410, 15, 51, 3, 15432, 440, 103, 322, 2561, 3, 9],
                    [15, 51, 3, 15432, 440, 103, 322, 2561, 3, 9, 3493, 6],
                    [3, 15432, 440, 103, 322, 2561, 3, 9, 3493, 6, 975, 7549],
                ],
                "target_labels": [
                    [8410, 15, 51, 3, 15432, 440, 103, 322, 2561, 3, 9, 3493],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 975],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7549, 17],
                ],
                "target_num_bytes": [26, 5, 4],
            },
            {
                "input_ids": [
                    [440, 103, 322, 2561, 3, 9, 3493, 6, 975, 7549, 17, 15],
                    [322, 2561, 3, 9, 3493, 6, 975, 7549, 17, 15, 2905, 3],
                    [3, 9, 3493, 6, 975, 7549, 17, 15, 2905, 3, 9, 21981],
                ],
                "target_labels": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 2905],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21981, 159],
                ],
                "target_num_bytes": [4, 1, 5],
            },
        ]
        ds = self._source(texts=texts, max_len=12, batch_size=3)
        for ix, batch in enumerate(ds):
            batch = {k: v.tolist() for k, v in batch.items()}
            self.assertNestedAllClose(expected_batches[ix], batch)
            if ix > 0:
                # Check the first two batches.
                break

    def test_drop_empty_targets(self):
        ds = fake_grain_source(
            [
                {
                    "target_ids": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                    "target_num_bytes": np.array([0, 1, 0]),
                },
            ]
        )
        ds = ds.map(_drop_empty_targets)
        actual = list(ds)
        self.assertNestedEqual(
            [{"target_ids": np.array([[4, 5, 6]]), "target_num_bytes": np.array([1])}], actual
        )
