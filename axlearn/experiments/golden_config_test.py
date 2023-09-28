# Copyright © 2023 Apple Inc.

"""This test compares against golden files to detect inadvertent changes."""

import pytest
from absl.testing import parameterized

from axlearn.experiments import test_utils
from axlearn.experiments.vision import resnet

_CONFIGS = [
    *test_utils.named_parameters(resnet.imagenet_trainer),
]

_INITS = [
    *test_utils.named_parameters(resnet.imagenet_trainer),
]

_REGULARIZERS = [
    *test_utils.named_parameters(resnet.imagenet_trainer),
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
