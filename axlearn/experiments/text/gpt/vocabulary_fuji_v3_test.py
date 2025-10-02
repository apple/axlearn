# Copyright © 2024 Apple Inc.

"""Tests fuji v3 vocabulary."""

import numpy as np
import pytest
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from axlearn.common import input_text, input_tf_data
from axlearn.common.config import config_for_class, config_for_function
from axlearn.common.input_lm import (
    PackingMethodType,
    lm_text_preprocessor,
    text_to_lm_eval_input,
    text_to_lm_training_input,
)
from axlearn.common.input_test_utils import make_ds_fn
from axlearn.common.test_utils import TestCase
from axlearn.experiments.text.gpt.vocabulary_fuji_v3 import FujiV3Vocabulary


@pytest.mark.skip(reason="no tokenizer file.")
class FujiV3VocabularyTest(TestCase):
    """Tests FujiV3VocabularyTest."""

    @property
    def vocab_cfg(self):
        return config_for_class(FujiV3Vocabulary).set(filename="Llama-3-tokenizer.json")

    def test_encode_tf_and_decode_tf(self):
        vocab = self.vocab_cfg.instantiate()
        text = tf.constant(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit\n", dtype=tf.string
        )
        ids = vocab.encode_tf(text)
        recovered = vocab._decode_tf(ids)  # pylint: disable=W0212

        self.assertEqual(text.numpy().decode("utf-8"), recovered.numpy().decode("utf-8"))

    def test_tokenize_example(self):
        vocab = self.vocab_cfg.instantiate()
        newlines_replaced_with = "<n>"
        newlines_replaced_with_id = vocab.encode(newlines_replaced_with)

        # Test tokenize_example replaces newlines.
        tokens = input_text.tokenize_example(
            "Hello\n", sp_vocab=vocab, replace_newlines_with=newlines_replaced_with
        ).numpy()
        self.assertNestedAllClose(
            np.array(vocab.encode("Hello") + newlines_replaced_with_id), tokens
        )

    def test_num_bytes(self):
        vocab = self.vocab_cfg.instantiate()
        newlines_replaced_with = "\n"
        pad_id = vocab.pad_id
        newline_id = vocab.encode("\n").pop()
        newlines_replaced_with_id = vocab.encode(newlines_replaced_with).pop()

        # Test num_bytes computes expected value.
        ids = tf.constant(
            [vocab.eos_id, newlines_replaced_with_id, newline_id, pad_id, pad_id, pad_id],
            dtype=tf.int32,
        )
        self.assertEqual(
            3,
            input_text.num_bytes(
                ids, sp_vocab=vocab, newlines_replaced_with=newlines_replaced_with
            ),
        )

    @parameterized.parameters(
        dict(
            packing_method=PackingMethodType.EOS_DELIM_MASK,
            max_padding_fraction=1.0,  # Always pad
        ),
        dict(
            packing_method=PackingMethodType.EOS_DELIM_NO_MASK,
            max_padding_fraction=1.0,  # Always pad
        ),
        dict(
            packing_method=PackingMethodType.EOS_DELIM_MASK,
            max_padding_fraction=0.0,  # Do not pad
        ),
    )
    def test_fake_text_lm_training_data(
        self, packing_method: PackingMethodType, max_padding_fraction: float
    ):
        texts = [
            "hello world\n",
            "hello moon\n",
        ]

        # window_size > len(texts) to repeat the sentence. 18 tokens in total.
        window_size = 3

        # Pad the concatenated sequence to 20 tokens:
        # Or, trim the sequence to 15 tokens:
        batch_size, max_len = 2, 5

        # Disable shuffling to make results interpretable.
        shuffle_buffer_size = 0

        # Test text_to_lm_training_input.
        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=True,
            source=config_for_function(make_ds_fn).set(texts=texts),
            processor=config_for_function(text_to_lm_training_input).set(
                vocab_cfg=self.vocab_cfg,
                max_len=max_len,
                replace_newlines_with="\n",
                window_size=window_size,
                max_padding_fraction=max_padding_fraction,
                shuffle_buffer_size=shuffle_buffer_size,
                packing_method=packing_method,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        # Set TensorFlow seed.
        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for ix, batch in enumerate(dataset):
            self.assertIsNotNone(batch)
            batch = {k: v.tolist() for k, v in batch.items()}
            if ix >= 10:
                # Expect to be able to repeat forever.
                break

        # Test lm_text_preprocessor. Expect same results.
        cfg = input_tf_data.Input.default_config().set(
            name="test_input",
            is_training=True,
            source=config_for_function(make_ds_fn).set(
                texts=texts,
            ),
            processor=config_for_function(lm_text_preprocessor).set(
                vocab_cfg=self.vocab_cfg,
                max_sequence_length=max_len,
                replace_newlines_with="<n>",
                window_size=window_size,
                max_padding_fraction=max_padding_fraction,
                shuffle_buffer_size=shuffle_buffer_size,
                packing_method=packing_method,
            ),
            batcher=config_for_function(input_tf_data.batch).set(
                global_batch_size=batch_size,
                prefetch_buffer_size=2,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )

        # Reset TensorFlow seed.
        tf.random.set_seed(123)
        dataset = cfg.instantiate(parent=None)
        for ix, batch in enumerate(dataset):
            if ix >= 3:
                break
            batch = {k: v.tolist() for k, v in batch.items()}

    @parameterized.parameters(
        ("How long is a piece of string?", "index"),
        ("On the 20th of June", "not_index"),
        ("Here we stand united", None),
    )
    def test_eval_lm_processor_single_example(self, text, index_key):
        max_len = 12
        processor = text_to_lm_eval_input(
            vocab_cfg=self.vocab_cfg,
            max_len=max_len,
            replace_newlines_with="\n",
            stride=None,
            index_key="index",
        )
        ds_fn = (
            config_for_function(make_ds_fn)
            .set(is_training=False, texts=[text], repeat=1)
            .instantiate()
        )
        example = next(iter(processor(ds_fn())))
        for key in ["input_ids", "target_labels"]:
            # Shape is as expected.
            self.assertEqual((max_len,), example[key].numpy().shape)
        self.assertTrue("target_num_bytes" in example)
        # Index should have been passed through only for set value of `index_key`.
        self.assertEqual(index_key == "index", index_key in example)

        input_ids, target_labels = example["input_ids"].numpy(), example["target_labels"].numpy()
        self.assertEqual(128001, input_ids[0])  # EOS
        non_padded_length = (target_labels == 128004).argmax()
        self.assertEqual(128001, target_labels[non_padded_length - 1])  # EOS.
        # The inputs should be one-off the labels.
        self.assertNestedAllClose(
            target_labels[: non_padded_length - 1], input_ids[1:non_padded_length]
        )
