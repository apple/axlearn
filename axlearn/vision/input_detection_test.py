# Copyright © 2023 Apple Inc.

"""Tests object detection inputs."""
# pylint: disable=no-member,no-self-use
import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.config import config_for_function
from axlearn.vision import utils_detection
from axlearn.vision.input_detection import (
    DetectionInput,
    fake_detection_dataset,
    resize_and_crop_image,
)


class DetectionInputTest(parameterized.TestCase, tf.test.TestCase):
    """Tests DetectionInput."""

    def _input_config(
        self,
        is_training: bool,
        batch_size: int,
        image_size: int,
        output_stride: int,
        max_num_instances: int = 100,
    ):
        cfg = DetectionInput.default_config().set(name="test", is_training=is_training)
        cfg.source.set(
            dataset_name="coco/2017",
            split="validation",
            train_shuffle_buffer_size=100,
        )
        cfg.processor.set(
            image_size=(image_size, image_size),
            output_stride=output_stride,
            max_num_instances=max_num_instances,
        )
        cfg.batcher.set(global_batch_size=batch_size)
        return cfg

    def _expected_data_shape(
        self,
        is_training: bool,
        batch_size: int,
        image_size: int,
        max_num_detection: int = 100,
    ):
        expected_shape = {
            "image_data": {
                "image": (batch_size, image_size, image_size, 3),
                "image_info": (batch_size, 4, 2),
            },
        }
        if is_training:
            expected_shape.update(
                {
                    "labels": {
                        "groundtruth_classes": (batch_size, max_num_detection),
                        "groundtruth_boxes": (batch_size, max_num_detection, 4),
                    }
                }
            )
        else:
            expected_shape.update(
                {
                    "labels": {
                        "groundtruths": {
                            "areas": (batch_size, max_num_detection),
                            "boxes": (batch_size, max_num_detection, 4),
                            "classes": (batch_size, max_num_detection),
                            "height": (batch_size,),
                            "image_info": (batch_size, 4, 2),
                            "is_crowds": (batch_size, max_num_detection),
                            "num_detections": (batch_size, 1),
                            "source_id": (batch_size,),
                            "width": (batch_size,),
                        }
                    }
                }
            )
        return expected_shape

    @parameterized.product(is_training=(False, True))
    def test_fake_input(self, is_training):
        batch_size = 8
        image_size = 640
        cfg = self._input_config(
            is_training,
            batch_size=batch_size,
            image_size=image_size,
            output_stride=2**7,
        )
        cfg.source = config_for_function(fake_detection_dataset)
        if not is_training:
            cfg.source.set(total_num_examples=batch_size)
        dataset = cfg.instantiate(parent=None)
        expected_shape = self._expected_data_shape(
            is_training=is_training,
            batch_size=batch_size,
            image_size=image_size,
        )
        for batch in dataset:
            self.assertEqual(expected_shape, utils.shapes(batch))
            break

    @parameterized.parameters(
        (100, 200, 220, 220, 1.1, 1.1, 224, 224),
        (512, 512, 1024, 1024, 2.0, 2.0, 1024, 1024),
    )
    def test_resize_and_crop_image(
        self,
        input_height,
        input_width,
        desired_height,
        desired_width,
        scale_y,
        scale_x,
        output_height,
        output_width,
    ):
        image = tf.convert_to_tensor(np.random.rand(input_height, input_width, 3))

        desired_size = (desired_height, desired_width)
        resized_image, image_info = resize_and_crop_image(
            image,
            desired_size=desired_size,
            padded_size=utils_detection.compute_padded_size(desired_size, 32),
        )
        resized_image_shape = tf.shape(resized_image)

        self.assertAllEqual([output_height, output_width, 3], resized_image_shape.numpy())
        self.assertNDArrayNear(
            [
                [input_height, input_width],
                [desired_height, desired_width],
                [scale_y, scale_x],
                [0.0, 0.0],
            ],
            image_info.numpy(),
            1e-5,
        )


if __name__ == "__main__":
    absltest.main()
