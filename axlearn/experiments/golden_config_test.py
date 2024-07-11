# Copyright © 2023 Apple Inc.

"""This test compares against golden files to detect inadvertent changes."""

import pytest
from absl.testing import parameterized

from axlearn.experiments import test_utils
from axlearn.experiments.text import gpt
from axlearn.experiments.vision import resnet

_CONFIGS = [
    *test_utils.named_parameters(resnet.imagenet_trainer),
    *test_utils.named_parameters(gpt.c4_trainer),
    *test_utils.named_parameters(gpt.pajama_trainer),
    *test_utils.named_parameters(gpt.pajama_sigmoid_trainer),
]

_INITS = [
    *test_utils.named_parameters(resnet.imagenet_trainer),
    *test_utils.named_parameters(gpt.c4_trainer),
    *test_utils.named_parameters(gpt.pajama_trainer),
    *test_utils.named_parameters(gpt.pajama_sigmoid_trainer),
]

_REGULARIZERS = [
    *test_utils.named_parameters(resnet.imagenet_trainer),
    *test_utils.named_parameters(gpt.c4_trainer),
    *test_utils.named_parameters(gpt.pajama_trainer),
    *test_utils.named_parameters(gpt.pajama_sigmoid_trainer),
]

_RUNS = [
    *[
        tup
        for tup in test_utils.named_parameters(gpt.c4_trainer, data_dir="FAKE")
        if "golden-run-test" in tup[0]
    ],
]


class GoldenConfigTest(test_utils.BaseGoldenConfigTest):
    @parameterized.named_parameters(*_CONFIGS)
    @pytest.mark.golden_config
    def test_config(self, *args):
        self._test(*args, test_type=test_utils.GoldenTestType.CONFIG)

    @parameterized.named_parameters(*_INITS)
    @pytest.mark.golden_init
    def test_init(self, *args):
        self._test(*args, test_type=test_utils.GoldenTestType.INIT)

    @parameterized.named_parameters(*_REGULARIZERS)
    @pytest.mark.golden_regularizer
    def test_regularizer(self, *args):
        self._test(*args, test_type=test_utils.GoldenTestType.REGULARIZER)

    @parameterized.named_parameters(*_RUNS)
    @pytest.mark.golden_run
    @pytest.mark.skip("Intended to be run manually.")
    def test_run(self, *args):
        self._test(*args, test_type=test_utils.GoldenTestType.RUN)
