# Copyright © 2024 Apple Inc.

"""Tests Gala sigmoid methods."""
from absl.testing import absltest

from axlearn.common import input_tf_data, utils
from axlearn.common.config import config_for_function
from axlearn.common.test_utils import TestCase
from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments.text.gpt.common import mixture_train_input_source
from axlearn.experiments.text.gpt.gala_sigmoid import _set_seq_len_recursively


class SetConfigTest(TestCase):
    def test_set_seq_len_recursively(self):
        train_input_source = config_for_function(mixture_train_input_source).set(
            max_sequence_length=200
        )
        cfg = SpmdTrainer.default_config().set(
            input=input_tf_data.Input.default_config().set(source=train_input_source)
        )

        self.assertEqual(cfg.input.source.max_sequence_length, 200)
        _set_seq_len_recursively(cfg, max_sequence_length=100)
        self.assertEqual(cfg.input.source.max_sequence_length, 100)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
