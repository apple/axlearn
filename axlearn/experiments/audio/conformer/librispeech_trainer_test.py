# Copyright © 2024 Apple Inc.

"""Tests Conformer LibriSpeech configs."""

from axlearn.common import test_utils
from axlearn.experiments.audio.conformer import librispeech_trainer


class LibriSpeechTrainerTest(test_utils.TrainerConfigTestCase):
    """Tests LibriSpeech trainer."""

    def test_trainer(self):
        self._test_with_trainer_config(
            librispeech_trainer.named_trainer_configs()["conformer-test-ctc"](),
        )
