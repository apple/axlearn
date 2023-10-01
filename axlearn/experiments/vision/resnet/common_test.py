# Copyright © 2023 Apple Inc.

"""Tests ResNet config builders."""

from absl.testing import parameterized

from axlearn.common import schedule
from axlearn.common.config import config_for_function
from axlearn.common.test_utils import TestCase
from axlearn.experiments.vision.resnet.common import learner_config, model_config
from axlearn.vision.resnet import ResNet


class ConfigTest(TestCase):
    """Tests configs."""

    @parameterized.product(
        learning_rate=[
            1.0,
            schedule.polynomial(end_value=10),
            config_for_function(schedule.polynomial).set(end_value=10),
        ],
        ema_decay=[None, 0.9],
    )
    def test_learner_config(self, **kwargs):
        cfg = learner_config(**kwargs)
        self.assertEqual(cfg.optimizer.learning_rate, kwargs["learning_rate"])
        # Make sure that we can instantiate.
        learner = cfg.set(name="test").instantiate(parent=None)
        if kwargs["ema_decay"] is not None:
            self.assertIsNotNone(learner.ema)

    def test_model_config(self):
        cfg = model_config()
        # We should be able to cfg.set(backbone=..., num_classes=...).
        cfg.set(backbone=ResNet.resnet18_config(), num_classes=100)
        # Make sure we can instantiate.
        cfg.set(name="test").instantiate(parent=None)
