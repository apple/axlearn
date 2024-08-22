# Copyright © 2023 Apple Inc.
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Object detection input modules.

Reference:
https://github.com/tensorflow/models/blob/5a0305c41304e8136e2056c589ab490a807dffa0/official/vision/dataloaders/retinanet_input.py
"""
from typing import Any, Literal, Optional

import numpy as np
import tensorflow as tf

from axlearn.common import input_tf_data
from axlearn.common.config import config_class, config_for_function
from axlearn.common.utils import NestedTensor, Tensor
from axlearn.vision import utils_detection
from axlearn.vision.input_image import whiten
from axlearn.vision.mask_generator import MaskingGenerator


def random_horizontal_flip(
    image: tf.Tensor, boxes: tf.Tensor, seed: Optional[int] = None
) -> tuple[tf.Tensor, tf.Tensor]:
    """Randomly flips the image and boxes horizontally with a probability of 50%.

    Args:
        image: rank 3 float32 tensor with shape [height, width, channels].
        boxes: rank 2 float32 tensor with shape [N, 4] containing the bounding boxes.
            Boxes are in normalized form meaning their coordinates
            vary between [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].
        seed: random seed.

    Returns:
        image: image which is the same shape as input image.
        boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
             Boxes are in normalized form meaning their coordinates vary
             between [0, 1].
    """

    def _flip_image(image):
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    def _flip_boxes_left_right(boxes):
        ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        # The x coordinates are in [0, 1].
        flipped_xmin = tf.subtract(1.0, xmax)
        flipped_xmax = tf.subtract(1.0, xmin)
        flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
        return flipped_boxes

    do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    # flip image
    image = tf.cond(
        pred=do_a_flip_random, true_fn=lambda: _flip_image(image), false_fn=lambda: image
    )
    # flip boxes
    boxes = tf.cond(
        pred=do_a_flip_random,
        true_fn=lambda: _flip_boxes_left_right(boxes),
        false_fn=lambda: boxes,
    )
    return image, boxes


def resize_and_crop_image(
    image: tf.Tensor,
    *,
    desired_size: tuple[int, int],
    padded_size: list[int],
    aug_scale_min: float = 1.0,
    aug_scale_max: float = 1.0,
    seed: int = 1,
    method: Literal = tf.image.ResizeMethod.BILINEAR,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Resizes the input image to output size (RetinaNet style).

    Resize and pad images given the desired output size of the image and
    stride size.
    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
        the largest rectangle to be bounded by the rectangle specified by the
        `desired_size`.
    2. Pad the rescaled image to the padded_size.

    Args:
        image: a `Tensor` of shape [height, width, 3] representing an image.
        desired_size: a tuple of two elements representing [height, width] of the desired actual
            output image size.
        padded_size: a tuple of two elements representing [height, width] of the padded output
            image size. Padding will be applied after scaling the image to the desired_size.
        aug_scale_min: a `float` with range between [0, 1.0] representing minimum
            random scale applied to desired_size for training scale jittering.
        aug_scale_max: a `float` with range between [1.0, inf] representing maximum
            random scale applied to desired_size for training scale jittering.
        seed: seed for random scale jittering.
        method: function to resize input image to scaled image. For more supported method,
            please refer to https://www.tensorflow.org/api_docs/python/tf/image/resize.

    Returns:
        output_image: `Tensor` of shape [height, width, 3] where [height, width]
            equals to `output_size`.
        image_info: a 2D `Tensor` that encodes the information of the image and the
            applied preprocessing. It is in the format of
            [[original_height, original_width], [desired_height, desired_width],
            [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
            desired_width] is the actual scaled image size, and [y_scale, x_scale] is
            the scaling factor, which is the ratio of
            scaled dimension / original dimension.
    """
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0

    if random_jittering:
        random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max, seed=seed)
        scaled_size = tf.round(random_scale * desired_size)
    else:
        scaled_size = desired_size

    scale = tf.minimum(scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
        max_offset = scaled_size - desired_size
        max_offset = tf.where(tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset)
        offset = max_offset * tf.random.uniform(
            shape=[
                2,
            ],
            minval=0,
            maxval=1,
            seed=seed,
        )
        offset = tf.cast(offset, tf.int32)
    else:
        offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
        scaled_image = scaled_image[
            offset[0] : offset[0] + desired_size[0], offset[1] : offset[1] + desired_size[1], :
        ]

    output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, padded_size[0], padded_size[1])

    image_info = tf.stack(
        [
            image_size,
            tf.constant(desired_size, dtype=tf.float32),
            image_scale,
            tf.cast(offset, tf.float32),
        ]
    )
    return output_image, image_info


# pylint: disable=unused-argument, unused-variable, too-many-function-args
def _parse_train_data(
    data: dict[str, Any],
    *,
    output_size: tuple[int, int],
    output_stride: int,
    aug_scale_min: float,
    aug_scale_max: float,
    aug_rand_hflip: bool = True,
    skip_crowd_during_training: bool = True,
    max_num_instances: int = 100,
    **kwargs,
) -> tf.Tensor:
    """Parses single data for training."""
    del kwargs  # Delete eval-specific kwargs, e.g. kwargs for padding.

    classes = data["groundtruth_classes"]
    boxes = data["groundtruth_boxes"]
    is_crowds = data["groundtruth_is_crowd"]

    # Skips annotations with `is_crowd` = True.
    if skip_crowd_during_training:
        num_groundtrtuhs = tf.shape(input=classes)[0]
        indices = tf.cond(
            pred=tf.greater(tf.size(input=is_crowds), 0),
            true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            false_fn=lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64),
        )
        classes = tf.gather(classes, indices)
        boxes = tf.gather(boxes, indices)

    # Gets original image.
    image = data["image"]
    image_shape = tf.shape(input=image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = whiten(image)

    # Flips image randomly during training.
    if aug_rand_hflip:
        image, boxes = random_horizontal_flip(image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = utils_detection.denormalize_boxes_tf(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = resize_and_crop_image(
        image,
        desired_size=output_size,
        padded_size=utils_detection.compute_padded_size(output_size, output_stride),
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
    )
    image_data = {"image": image, "image_info": image_info}
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = utils_detection.resize_and_crop_boxes(boxes, image_scale, image_info[1, :], offset)
    # Filters out ground truth boxes that are all zeros.
    indices = utils_detection.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    labels = {
        "groundtruth_boxes": utils_detection.clip_or_pad_to_fixed_size(
            boxes, max_num_instances, -1
        ),
        "groundtruth_classes": utils_detection.clip_or_pad_to_fixed_size(
            classes, max_num_instances, -1
        ),
    }
    return image_data, labels


def _parse_eval_data(
    data: dict[str, Any],
    *,
    output_size: tuple[int, int],
    output_stride: int,
    max_num_instances: int = 100,
    **kwargs,
):
    """Parses single data for evaluation."""
    del kwargs  # Delete train-specific kwargs, e.g. kwargs for data augmentation.

    classes = data["groundtruth_classes"]
    boxes = data["groundtruth_boxes"]

    # Gets original image and its size.
    image = data["image"]
    image_shape = tf.shape(input=image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = whiten(image)

    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = utils_detection.denormalize_boxes_tf(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = resize_and_crop_image(
        image,
        desired_size=output_size,
        padded_size=utils_detection.compute_padded_size(output_size, output_stride),
        aug_scale_min=1.0,  # No scale jitter for evaluation.
        aug_scale_max=1.0,  # No scale jitter for evaluation.
    )
    image_data = {"image": image, "image_info": image_info}
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = utils_detection.resize_and_crop_boxes(boxes, image_scale, image_info[1, :], offset)
    # Filters out ground truth boxes that are all zeros.
    indices = utils_detection.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    # Sets up groundtruth data for evaluation.
    groundtruths = {
        "source_id": data["source_id"],
        "height": data["height"],
        "width": data["width"],
        "num_detections": tf.shape(data["groundtruth_classes"]),
        "image_info": image_info,
        "boxes": utils_detection.denormalize_boxes_tf(data["groundtruth_boxes"], image_shape),
        "classes": data["groundtruth_classes"],
        "areas": data["groundtruth_area"],
        "is_crowds": tf.cast(data["groundtruth_is_crowd"], tf.int32),
    }
    groundtruths["source_id"] = utils_detection.process_source_id(groundtruths["source_id"])
    labels = {
        "groundtruths": utils_detection.pad_groundtruths_to_fixed_size(
            groundtruths, max_num_instances
        )
    }
    return image_data, labels


def _parser(
    example: dict[str, Any],
    *,
    is_training: bool,
    **kwargs,
) -> tf.Tensor:
    if is_training:
        image_data, labels = _parse_train_data(data=example, **kwargs)
    else:
        image_data, labels = _parse_eval_data(data=example, **kwargs)
    return image_data, labels


def _process_example(
    is_training: bool,
    image_size: tuple[int, int],
    output_stride: int,
    max_num_instances: int,
    aug_scale_min: float,
    aug_scale_max: float,
    num_parallel_calls: Optional[int] = None,
    label_offset: int = 0,
    mask_window_size: Optional[int] = None,
    num_masking_patches: Optional[int] = None,
):
    """Decode and parse examples in the dataset.

    Input Example structure must be as below:
    {
        "image": An uint8 [height, width, 3] tensor with input image.
        "image/filename": A bytes tensor with image filename.
        "image/id": An int identifier for the image.
        "objects": {
            "area": A float [num_boxes] tensor with box areas.
            "bbox": A float [num_boxes, 4] tensor with normalized box coordinates in the form
                [ymin, xmin, ymax, xmax].
            "id": A int [num_boxes] tensor with box ids.
            "is_crowd": A bool [num_boxes] tensor indicating if boxes contain crowd.
            "label": A int32 [num_boxes] tensor with classes.
        }
    }

    Args:
        is_training: A boolean indicating whether it is in the training mode.
        image_size: a tuple of [height, width] for output image.
        output_stride: Output feature stride. This is used to compute the amount of padding applied
            to resized images to make their heights and widths a multiple of `output_stride`.
        max_num_instances: Maximum size to pad or clip groundtruth boxes and classes.
        aug_scale_min: the minimum scale applied to image for data augmentation during training.
        aug_scale_max: the maximum scale applied to image for data augmentation during training.
        num_parallel_calls: the number of batches to compute asynchronously in parallel.
            If not specified, batches will be computed sequentially.
        label_offset: Detection models expect labels ids to be 1-indexed as `0` is implicitly
            used to represent background/negatives. If source dataset does not contain 1-indexed
            labels set `label_offset` to make it 1-indexed.
        mask_window_size: A number to specify the generated mask height and width.
        num_masking_patches: A number to specify the total number of masked patches.

    Returns:
        A DatasetToDatasetFn that yields the following nested dictionary of tensors:
            image_data: A dictionary of input tensors:
                image: A float [batch, height, width, 3]  image tensor.
                image_info: An integer [batch, 4, 2] tensor containing [[original_height,
                    original_width], [desired_height, desired_width], [scale_y, scale_x], [offset_y,
                    offset_x]].

        If `training` is True:
            labels: A dictionary of labels with the following fields:
                groundtruth_boxes: A float [batch, max_num_instances, 4] tensor containing padded
                    groundtruth boxes in image coordinates and in the form [ymin, xmin,
                    ymax, xmax]. Values of -1s indicate padding.
                groundtruth_classes: An integer [batch, max_num_instances] tensor containing padded
                    groundtruth classes. Values of -1s indicate padding.

        If `training` is False:
            labels: A dictionary of labels with the following fields:
                areas: A float [batch, max_num_instances] tensor with box areas.
                boxes: A float [batch, max_num_instances, 4] tensor with box coordinates.
                classes: A float [batch, max_num_instances] tensor with classes.
                height: A float [batch] tensor with image heights.
                width: A float [batch] tensor with image heights.
                image_info: A float [batch, 4, 2] tensor with image scaling info. A copy of
                    `image_data["image_info"]`.
                is_crowds: A bool [batch, max_num_instances] tensor indicating if the boxes contain
                    a crowd.
                num_detections: An integer [batch, 1] tensor indicating number of valid groundtruth
                    excluding the padding.
                source_id: A str [batch] tensor indicating the source id of the images.
    """

    def example_fn(example: dict[str, Tensor]) -> NestedTensor:
        decoded_example = {
            "image": example["image"],
            "source_id": utils_detection.process_source_id(
                tf.strings.as_string(example["image/id"])
            ),
            "height": tf.cast(tf.shape(example["image"])[0], tf.int64),
            "width": tf.cast(tf.shape(example["image"])[1], tf.int64),
            "groundtruth_classes": example["objects"]["label"] + label_offset,
            "groundtruth_is_crowd": example["objects"]["is_crowd"],
            "groundtruth_area": tf.cast(example["objects"]["area"], tf.float32),
            "groundtruth_boxes": example["objects"]["bbox"],
        }
        kwargs = dict(
            output_size=image_size,
            output_stride=output_stride,
            max_num_instances=max_num_instances,
            aug_scale_min=aug_scale_min,
            aug_scale_max=aug_scale_max,
        )
        image_data, labels = _parser(
            decoded_example,
            is_training=is_training,
            **kwargs,
        )
        data = {"image_data": image_data, "labels": labels}
        if mask_window_size is not None:
            if num_masking_patches is None:
                raise ValueError("num_masking_patches needs to be specified for mask generation.")
            data["is_masked"] = tf.py_function(
                func=MaskingGenerator(
                    input_size=(mask_window_size, mask_window_size),
                    num_masking_patches=num_masking_patches,
                ),
                inp=(),
                Tout=tf.bool,
            )
        return data

    def dataset_fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(example_fn, num_parallel_calls=num_parallel_calls)

    return dataset_fn


class DetectionInput(input_tf_data.Input):
    """Detection input module."""

    @config_class
    class Config(input_tf_data.Input.Config):
        """Configures DetectionInput."""

        image_size: tuple[int, int] = (640, 640)  # The image size.

    @classmethod
    def default_config(cls):
        cfg = super().default_config()  # type: DetectionInput.Config
        cfg.source = config_for_function(input_tf_data.tfds_dataset).set(
            dataset_name="coco/2017",
            train_shuffle_buffer_size=1024,  # to be tuned.
        )
        # Default processor settings for COCO detection.
        cfg.processor = config_for_function(_process_example).set(
            image_size=(640, 640),
            output_stride=2**7,
            max_num_instances=100,
            aug_scale_min=0.1,
            aug_scale_max=2.0,
        )
        # TODO(xianzhi): add pad_example_fn for eval.
        cfg.batcher.pad_example_fn = input_tf_data.default_pad_example_fn
        return cfg


def fake_detection_dataset(
    is_training: bool, total_num_examples: Optional[int] = None
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def example_fn(_) -> NestedTensor:
        image = np.random.randint(
            size=[640, 640, 3],
            low=0,
            high=256,
            dtype=np.int32,
        )
        # A random COCO sample.
        fake_example = {
            "image": tf.convert_to_tensor(image),
            "image/filename": b"000000460139.jpg",
            "image/id": 460139,
            "objects": {
                "area": tf.constant([17821, 16942, 4344]),
                "bbox": tf.constant(
                    [
                        [0.54380953, 0.13464062, 0.98651516, 0.33742186],
                        [0.50707793, 0.517875, 0.8044805, 0.891125],
                        [0.3264935, 0.36971876, 0.65203464, 0.4431875],
                    ],
                    dtype=tf.float32,
                ),
                "id": tf.constant([17821, 16942, 4344]),
                "is_crowd": tf.constant([False, False, False]),
                "label": tf.constant([11, 56, 3]),
            },
        }
        return fake_example

    def fn() -> tf.data.Dataset:
        counter_ds = tf.data.experimental.Counter()
        if total_num_examples is not None:
            counter_ds = counter_ds.take(total_num_examples)
        return counter_ds.map(example_fn)

    return fn
