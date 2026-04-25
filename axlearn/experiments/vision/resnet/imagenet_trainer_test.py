# Copyright © 2023 Apple Inc.

"""ResNet on ImageNet trainer config tests."""

from absl.testing import absltest

from axlearn.experiments import test_utils
from axlearn.experiments.vision.resnet import imagenet_trainer


class ImageNetTrainerTest(test_utils.TrainerConfigTestCase):
    """Tests ImageNet trainer."""

    def test_trainer(self):
        self._test_with_trainer_config(
            imagenet_trainer.named_trainer_configs()["ResNet-Test"](),
        )


if __name__ == "__main__":
    absltest.main()
