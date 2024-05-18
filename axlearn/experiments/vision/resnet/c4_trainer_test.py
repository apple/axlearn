# Copyright © 2023 Apple Inc.

"""c4 trainer config tests."""

from axlearn.common import test_utils
from axlearn.experiments.text.gpt import c4_trainer


class C4TrainerTest(test_utils.TrainerConfigTestCase):
    """Tests C4 trainer."""

    def test_trainer(self):
        self._test_with_trainer_config(
            c4_trainer.named_trainer_configs()["fuji-test-v1"](),
        )
