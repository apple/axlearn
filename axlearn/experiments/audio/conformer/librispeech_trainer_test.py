# Copyright © 2024 Apple Inc.

"""Tests Conformer LibriSpeech configs."""

from absl.testing import absltest

from axlearn.experiments import test_utils
from axlearn.experiments.audio.conformer import librispeech_trainer


class LibriSpeechTrainerTest(test_utils.TrainerConfigTestCase):
    """Tests LibriSpeech trainer."""

    def test_trainer(self):
        self._test_with_trainer_config(
            librispeech_trainer.named_trainer_configs()["conformer-test-ctc"](),
        )


if __name__ == "__main__":
    absltest.main()
