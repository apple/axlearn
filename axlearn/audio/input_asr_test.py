# Copyright © 2023 Apple Inc.

"""Tests speech and text processors."""
# pylint: disable=protected-access

import os
from typing import Any, Dict, Optional, Sequence

import seqio
import tensorflow as tf
from absl.testing import parameterized

from axlearn.audio import input_asr
from axlearn.common import input_fake, input_tf_data
from axlearn.common.config import config_for_class, config_for_function
from axlearn.common.test_utils import TestCase

tokenizers_dir = os.path.join(
    os.path.dirname(__file__), "../experiments/testdata/tokenizers/sentencepiece"
)


class SpeechInputTest(TestCase, tf.test.TestCase):
    """Tests speech input processing."""

    @parameterized.parameters(
        dict(
            # Test a basic case with padding.
            max_len=5,
            expected=[
                {
                    "inputs": tf.constant([-29515.0, 0, 0, 0, 0]),
                    "paddings": tf.constant([0, 1, 1, 1, 1]),
                },
                {
                    "inputs": tf.constant([14620.0, -21206.0, 0, 0, 0]),
                    "paddings": tf.constant([0, 0, 1, 1, 1]),
                },
                {
                    "inputs": tf.constant([-3954.0, -15555.0, 18074.0, 0, 0]),
                    "paddings": tf.constant([0, 0, 0, 1, 1]),
                },
            ],
        ),
        dict(
            # Test a basic case with filtering.
            max_len=2,
            expected=[
                {"inputs": tf.constant([-29515.0, 0]), "paddings": tf.constant([0, 1])},
                {"inputs": tf.constant([14620.0, -21206.0]), "paddings": tf.constant([0, 0])},
            ],
        ),
        dict(
            # Test a basic case with truncation.
            max_len=2,
            truncate=True,
            expected=[
                {
                    "inputs": tf.constant([-29515.0, 0]),
                    "paddings": tf.constant([0, 1]),
                },
                {
                    "inputs": tf.constant([14620.0, -21206.0]),
                    "paddings": tf.constant([0, 0]),
                },
                {
                    "inputs": tf.constant([-3954.0, -15555.0]),
                    "paddings": tf.constant([0, 0]),
                },
            ],
        ),
        dict(
            # Test a basic case with normalization.
            max_len=5,
            scale=2**15,
            expected=[
                {
                    "inputs": tf.constant([-0.9007263, 0.0, 0.0, 0.0, 0.0]),
                    "paddings": tf.constant([0, 1, 1, 1, 1]),
                },
                {
                    "inputs": tf.constant([0.446167, -0.64715576, 0.0, 0.0, 0.0]),
                    "paddings": tf.constant([0, 0, 1, 1, 1]),
                },
                {
                    "inputs": tf.constant([-0.1206665, -0.47470093, 0.5515747, 0.0, 0.0]),
                    "paddings": tf.constant([0, 0, 0, 1, 1]),
                },
            ],
        ),
        dict(
            # Test a basic case with input_key.
            max_len=2,
            scale=2**15,
            input_key="input_speech",
            expected=[
                {
                    "inputs": tf.constant([-0.9007263, 0.0]),
                    "paddings": tf.constant([0, 1]),
                },
                {
                    "inputs": tf.constant([0.446167, -0.64715576]),
                    "paddings": tf.constant([0, 0]),
                },
            ],
        ),
    )
    def test_speech_input(
        self,
        max_len: int,
        expected: Dict[str, Any],
        truncate: bool = False,
        input_key: str = "speech",
        scale: Optional[float] = None,
    ):
        processor = input_asr.speech_input(
            max_len=max_len,
            input_key=input_key,
            normalize_by_scale=scale,
            truncate=truncate,
        )
        # Use a fake speech source with only speech inputs.
        source = input_tf_data.with_processor(
            config_for_function(input_fake.fake_speech_source).set(
                speech_key=input_key, num_examples=10
            ),
            processor=config_for_function(input_tf_data.select_fields).set(fields=[input_key]),
            is_training=False,
        )
        actual = list(processor(source()).take(3))
        tf.nest.map_structure(self.assertAllClose, expected, actual)


class TextInputTest(TestCase, tf.test.TestCase):
    """Tests text input processing."""

    @parameterized.parameters(
        dict(
            # A basic case with padding (no truncation).
            text="NO TRUNCATION",
            truncate=False,
            max_len=8,
            eos_id=1024,
            expected=[
                {
                    "text": "NO TRUNCATION",
                    "target_labels": ["▁NO", "▁TRU", "N", "C", "ATION", 1024, -1, -1],
                    "input_ids": [1024, "▁NO", "▁TRU", "N", "C", "ATION", 1024, -1],
                }
            ],
        ),
        dict(
            # The example is filtered since we don't have space for EOS.
            text="NO TRUNCATION",
            truncate=False,
            max_len=5,
            eos_id=1024,
            expected=[],
        ),
        dict(
            # A basic case with truncation.
            text="WITH TRUNCATION",
            truncate=True,
            max_len=5,
            eos_id=1024,
            expected=[
                {
                    "text": "WITH TRUNCATION",
                    "target_labels": ["▁WITH", "▁TRU", "N", "C", "ATION"],
                    "input_ids": [1024, "▁WITH", "▁TRU", "N", "C"],
                }
            ],
        ),
        dict(
            # A basic case with input_key.
            text="WITH INPUT KEY",
            truncate=True,
            max_len=5,
            input_key="input_text",
            eos_id=1024,
            expected=[
                {
                    "input_text": "WITH INPUT KEY",
                    "target_labels": ["▁WITH", "▁IN", "P", "U", "T"],
                    "input_ids": [1024, "▁WITH", "▁IN", "P", "U"],
                }
            ],
        ),
        dict(
            # Test specifying eos_id without truncation.
            text="NO TRUNCATION",
            truncate=True,
            max_len=8,
            input_key="input_text",
            eos_id=1,
            expected=[
                {
                    "input_text": "NO TRUNCATION",
                    "target_labels": ["▁NO", "▁TRU", "N", "C", "ATION", 1, -1, -1],
                    "input_ids": [1, "▁NO", "▁TRU", "N", "C", "ATION", 1, -1],
                }
            ],
        ),
        dict(
            # Test specifying eos_id with truncation.
            text="WITH TRUNCATION",
            truncate=True,
            max_len=5,
            input_key="input_text",
            eos_id=1,
            expected=[
                {
                    "input_text": "WITH TRUNCATION",
                    "target_labels": ["▁WITH", "▁TRU", "N", "C", "ATION"],
                    "input_ids": [1, "▁WITH", "▁TRU", "N", "C"],
                }
            ],
        ),
    )
    def test_text_input(
        self,
        expected: Sequence[Dict[str, Any]],
        text: str,
        max_len: int,
        eos_id: int,
        input_key: str = "text",
        truncate: bool = False,
    ):
        spm_file = os.path.join(tokenizers_dir, "librispeech_unigram_1024.model")
        vocab_cfg = config_for_class(seqio.SentencePieceVocabulary).set(
            sentencepiece_model_file=spm_file
        )
        vocab = vocab_cfg.instantiate()
        processor = input_tf_data.chain(
            input_asr.text_input(
                max_len=max_len,
                vocab=vocab_cfg,
                input_key=input_key,
                truncate=truncate,
                eos_id=eos_id,
            ),
            input_asr.make_autoregressive_inputs(vocab=vocab_cfg, bos_id=eos_id),
        )
        source = input_fake.fake_source(examples=[{input_key: text}], is_training=False)
        actual = list(processor(source()))
        for ex in expected:
            for k, v in ex.items():
                if k in ["input_ids", "target_labels"]:
                    v = [vocab.tokenizer.PieceToId(x) if isinstance(x, str) else x for x in v]
                ex[k] = tf.constant(v)
        tf.nest.map_structure(self.assertAllEqual, expected, actual)


class SpeechTextInputTest(parameterized.TestCase, tf.test.TestCase):
    """Tests speech-text input processing."""

    @parameterized.parameters(
        dict(
            # Test when truncate=False.
            max_speech_len=5,
            max_text_len=6,
            truncate=False,
            eos_id=1024,
            inputs=[
                # Test a basic case.
                dict(text="THIS IS OK", speech=tf.ones([4], dtype=tf.int16)),
                # Empty text should be filtered.
                dict(text="", speech=tf.ones([4], dtype=tf.int16)),
                # Empty speech should be filtered.
                dict(text="THIS IS OK", speech=tf.ones([0], dtype=tf.int16)),
                # Long text should be filtered.
                dict(text="THIS TEXT IS WAY TOO LONG", speech=tf.ones([4], dtype=tf.int16)),
                # Long speech should be filtered.
                dict(text="THIS IS OK", speech=tf.ones([10], dtype=tf.int16)),
            ],
            expected=[
                {
                    "text": tf.constant("THIS IS OK", dtype=tf.string),
                    "target_labels": ["▁THIS", "▁IS", "▁", "OK", 1024, -1],
                    "input_ids": [1024, "▁THIS", "▁IS", "▁", "OK", 1024],
                    "inputs": tf.constant([0.5, 0.5, 0.5, 0.5, 0.0]),
                    "paddings": tf.constant([0, 0, 0, 0, 1]),
                },
            ],
        ),
        dict(
            # Test filtering based on EOS.
            max_speech_len=5,
            max_text_len=4,
            truncate=False,
            eos_id=1024,
            inputs=[
                # Text will be filtered if we can't fit EOS.
                dict(text="THIS IS OK", speech=tf.ones([4], dtype=tf.int16)),
            ],
            expected=[],
        ),
        dict(
            # Test when truncate=True.
            max_speech_len=5,
            max_text_len=6,
            truncate=True,
            eos_id=1024,
            inputs=[
                # Test a basic case.
                dict(text="THIS IS OK", speech=tf.ones([4], dtype=tf.int16)),
                # Empty text should be filtered.
                dict(text="", speech=tf.ones([4], dtype=tf.int16)),
                # Empty speech should be filtered.
                dict(text="THIS IS OK", speech=tf.ones([0], dtype=tf.int16)),
                # Long text should be truncated.
                dict(text="THIS TEXT IS WAY TOO LONG", speech=tf.ones([4], dtype=tf.int16)),
                # Long speech should be filtered.
                dict(text="THIS IS OK", speech=tf.ones([10], dtype=tf.int16)),
            ],
            expected=[
                {
                    "text": tf.constant("THIS IS OK", dtype=tf.string),
                    "target_labels": ["▁THIS", "▁IS", "▁", "OK", 1024, -1],
                    "input_ids": [1024, "▁THIS", "▁IS", "▁", "OK", 1024],
                    "inputs": tf.constant([0.5, 0.5, 0.5, 0.5, 0.0]),
                    "paddings": tf.constant([0, 0, 0, 0, 1]),
                },
                {
                    "text": tf.constant("THIS TEXT IS WAY TOO LONG", dtype=tf.string),
                    "target_labels": ["▁THIS", "▁", "TE", "X", "T", "▁IS"],
                    "input_ids": [1024, "▁THIS", "▁", "TE", "X", "T"],
                    "inputs": tf.constant([0.5, 0.5, 0.5, 0.5, 0.0]),
                    "paddings": tf.constant([0, 0, 0, 0, 1]),
                },
            ],
        ),
        dict(
            # Test truncate=False with eos_id.
            max_speech_len=5,
            max_text_len=6,
            truncate=False,
            eos_id=1,
            inputs=[
                dict(text="THIS IS OK", speech=tf.ones([4], dtype=tf.int16)),
            ],
            expected=[
                {
                    "text": tf.constant("THIS IS OK", dtype=tf.string),
                    "target_labels": ["▁THIS", "▁IS", "▁", "OK", 1, -1],
                    "input_ids": [1, "▁THIS", "▁IS", "▁", "OK", 1],
                    "inputs": tf.constant([0.5, 0.5, 0.5, 0.5, 0.0]),
                    "paddings": tf.constant([0, 0, 0, 0, 1]),
                },
            ],
        ),
        dict(
            # Test truncate=True with eos_id.
            max_speech_len=5,
            max_text_len=6,
            truncate=True,
            eos_id=1,
            inputs=[
                # Test a basic case.
                dict(text="THIS IS OK", speech=tf.ones([4], dtype=tf.int16)),
                # Empty text should be filtered.
                dict(text="", speech=tf.ones([4], dtype=tf.int16)),
                # Empty speech should be filtered.
                dict(text="THIS IS OK", speech=tf.ones([0], dtype=tf.int16)),
                # Long text should be truncated.
                dict(text="THIS TEXT IS WAY TOO LONG", speech=tf.ones([4], dtype=tf.int16)),
                # Long speech should be filtered.
                dict(text="THIS IS OK", speech=tf.ones([10], dtype=tf.int16)),
            ],
            expected=[
                {
                    "text": tf.constant("THIS IS OK", dtype=tf.string),
                    "target_labels": ["▁THIS", "▁IS", "▁", "OK", 1, -1],
                    "input_ids": [1, "▁THIS", "▁IS", "▁", "OK", 1],
                    "inputs": tf.constant([0.5, 0.5, 0.5, 0.5, 0.0]),
                    "paddings": tf.constant([0, 0, 0, 0, 1]),
                },
                {
                    "text": tf.constant("THIS TEXT IS WAY TOO LONG", dtype=tf.string),
                    "target_labels": ["▁THIS", "▁", "TE", "X", "T", "▁IS"],
                    "input_ids": [1, "▁THIS", "▁", "TE", "X", "T"],
                    "inputs": tf.constant([0.5, 0.5, 0.5, 0.5, 0.0]),
                    "paddings": tf.constant([0, 0, 0, 0, 1]),
                },
            ],
        ),
    )
    def test_asr_input(
        self,
        inputs: Sequence[Dict],
        expected: Sequence[Dict],
        max_speech_len: int,
        max_text_len: int,
        truncate: bool,
        eos_id: int,
    ):
        spm_file = os.path.join(tokenizers_dir, "librispeech_unigram_1024.model")
        vocab_cfg = config_for_class(seqio.SentencePieceVocabulary).set(
            sentencepiece_model_file=spm_file
        )
        vocab = vocab_cfg.instantiate()

        source = input_fake.fake_source(
            is_training=False,
            examples=inputs,
            spec=dict(
                speech=tf.TensorSpec(shape=(None,), dtype=tf.int16),
                text=tf.TensorSpec(shape=(), dtype=tf.string),
            ),
        )
        processor = input_tf_data.chain(
            input_asr.speech_input(max_len=max_speech_len, normalize_by_scale=2.0),
            input_asr.text_input(
                max_len=max_text_len, vocab=vocab_cfg, truncate=truncate, eos_id=eos_id
            ),
            input_asr.make_autoregressive_inputs(vocab=vocab_cfg, bos_id=eos_id),
        )
        actual = list(processor(source()))
        self.assertEqual(len(expected), len(actual))
        for i, expect in enumerate(expected):
            # Compare text fields separately from assertAllClose.
            self.assertEqual(actual[i].pop("text"), expect.pop("text"))
            for k, v in expect.items():
                if k in ["input_ids", "target_labels"]:
                    v = [vocab.tokenizer.PieceToId(x) if isinstance(x, str) else x for x in v]
                expect[k] = tf.constant(v)
        tf.nest.map_structure(self.assertAllClose, expected, actual)


class FilterTest(parameterized.TestCase, tf.test.TestCase):
    """Tests filtering."""

    @parameterized.parameters(
        dict(
            input_key="inputs",
            inputs=[{"inputs": []}, {"inputs": [1]}, {"inputs": [1, 2]}],
            expected=[{"inputs": []}, {"inputs": [1]}, {"inputs": [1, 2]}],
        ),
        dict(
            input_key="inputs",
            inputs=[{"inputs": []}, {"inputs": [1]}, {"inputs": [1, 2]}],
            expected=[{"inputs": [1]}, {"inputs": [1, 2]}],
            min_len=1,
        ),
        dict(
            input_key="inputs",
            inputs=[{"inputs": []}, {"inputs": [1]}, {"inputs": [1, 2]}],
            expected=[{"inputs": [1, 2]}],
            min_len=2,
        ),
        dict(
            input_key="inputs",
            inputs=[{"inputs": []}, {"inputs": [1]}, {"inputs": [1, 2]}],
            expected=[{"inputs": []}, {"inputs": [1]}],
            max_len=1,
        ),
        dict(
            input_key="inputs",
            inputs=[{"inputs": []}, {"inputs": [1]}, {"inputs": [1, 2]}],
            expected=[{"inputs": [1]}],
            min_len=1,
            max_len=1,
        ),
    )
    def test_filter_by_length(
        self,
        *,
        inputs: Sequence[Dict],
        expected: Sequence[Dict],
        **kwargs,
    ):
        source = input_fake.fake_source(
            is_training=False,
            examples=inputs,
            spec=dict(inputs=tf.TensorSpec(shape=[None], dtype=tf.int32)),
        )
        processor = input_asr._filter_by_length(**kwargs)
        actual = list(processor(source()))
        self.assertEqual(len(expected), len(actual))
        expected = [
            {k: tf.constant(v, dtype=tf.int32) for k, v in expect.items()} for expect in expected
        ]
        tf.nest.map_structure(self.assertAllEqual, expected, actual)
