# Copyright © 2024 Apple Inc.

"""Tests Conformer config builders."""

import os

import pytest
import seqio
import tensorflow as tf
from absl.testing import parameterized

from axlearn.audio.encoder_asr import SpeechFeatureLayer
from axlearn.common.config import config_for_class
from axlearn.common.input_fake import fake_speech_source, fake_text_source
from axlearn.common.layers import Dropout
from axlearn.common.test_utils import TestCase
from axlearn.experiments.audio.conformer.common import asr_input, encoder_config, stack_config

_tokenizers_dir = os.path.join(os.path.dirname(__file__), "../../../data/tokenizers")
# Easier to test with than librispeech 1k vocab, since it treats many test inputs as UNK.
_bpe_vocab_file = os.path.join(_tokenizers_dir, "sentencepiece", "bpe_32k_c4.model")


class ConfigTest(TestCase):
    """Tests configs."""

    def test_encoder_config(self):
        dim, num_layers, num_heads, dropout_rate = 8, 2, 4, 0.2
        cfg = encoder_config(
            dim=dim,
            stack_cfg=stack_config(num_heads=num_heads, num_layers=num_layers),
            feature_cfg=SpeechFeatureLayer.default_config(),
            dropout_rate=dropout_rate,
        )

        def visit_fn(_, value):
            if isinstance(value, Dropout.Config):
                self.assertEqual(value.rate, dropout_rate)

        # Check that dropout rate is applied to context.
        cfg.context.visit(visit_fn=visit_fn, enter_fn=None)

    @parameterized.parameters(
        # Not dropping any.
        dict(max_source_len=5, max_target_len=6, expect_count=5),
        # Dropping all text.
        dict(max_source_len=5, max_target_len=5, expect_count=0),
        # Dropping some speech and pad text.
        dict(max_source_len=4, max_target_len=8, expect_count=4),
    )
    @pytest.mark.skipif(not os.path.exists(_bpe_vocab_file), reason="Missing testdata.")
    def test_asr_input(self, max_source_len: int, max_target_len: int, expect_count: int):
        vocab_cfg = config_for_class(seqio.SentencePieceVocabulary).set(  # noqa: F821
            sentencepiece_model_file=_bpe_vocab_file
        )
        vocab = vocab_cfg.instantiate()
        processor = asr_input(
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            vocab_cfg=vocab_cfg,
            bos_id=1,  # The default BPE vocab has no BOS.
            eos_id=vocab.eos_id,
        )

        def source():
            speech_ds = fake_speech_source(is_training=False, num_examples=5)()
            text_ds = fake_text_source(is_training=False, batch_size=5)()
            return tf.data.Dataset.zip((speech_ds, text_ds)).map(lambda s, t: {**s, **t})

        actual_count = 0
        for ex in processor(source()):
            actual_count += 1
            self.assertEqual(max_source_len, ex["source"]["inputs"].shape[0])
            # Paddings should be 0's and 1's.
            self.assertTrue(
                tf.reduce_all(
                    tf.logical_or(
                        tf.math.logical_not(ex["source"]["paddings"]), ex["source"]["paddings"]
                    )
                )
            )
            # Paddings for source should correspond to pad_id.
            self.assertTrue(
                tf.reduce_all(
                    tf.where(
                        tf.cast(ex["source"]["paddings"], tf.bool),
                        ex["source"]["inputs"],
                        vocab.pad_id,
                    )
                    == vocab.pad_id
                )
            )
            self.assertEqual(max_target_len, ex["target"]["input_ids"].shape[0])
            # At least one non-padding in targets.
            self.assertTrue(tf.reduce_any(ex["target"]["input_ids"] != -1))
            # target_labels and target_ids should match.
            self.assertTrue(
                tf.reduce_all(ex["target"]["input_ids"][1:] == ex["target_labels"][:-1])
            )
        self.assertEqual(expect_count, actual_count)
