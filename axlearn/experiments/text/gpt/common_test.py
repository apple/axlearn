# Copyright © 2024 Apple Inc.

"""Tests common GPT trainer utils."""

from axlearn.common.config import config_for_function
from axlearn.common.input_fake import fake_text_source
from axlearn.common.learner import Learner
from axlearn.common.test_utils import DummyForwardModel, TestCase
from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments.text.gpt.common import MESH_AXIS_NAMES, get_trainer_config_fn


class TrainerConfigTest(TestCase):
    """Tests trainer config utils."""

    def test_mesh_axes(self):
        config_fn = get_trainer_config_fn(
            model_cfg=DummyForwardModel.default_config(),
            learner_cfg=Learner.default_config(),
            max_step=1,
            train_batch_size=1,
            train_input_source=config_for_function(fake_text_source),
            evalers={},
            mesh_shape=(1,) * len(MESH_AXIS_NAMES),
        )
        cfg: SpmdTrainer.Config = config_fn()
        self.assertEqual(cfg.mesh_axis_names, MESH_AXIS_NAMES)
        self.assertNotIn("pipeline", cfg.batch_axis_names)
        self.assertNotIn("model", cfg.batch_axis_names)
