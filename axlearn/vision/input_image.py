# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# pytorch/vision:
# Copyright (c) Soumith Chintala 2016, All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# google/flax:
# Copyright 2023 The Flax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google-research/vision_transformer:
# Copyright 2023 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Image input modules.

References:
https://github.com/pytorch/vision/blob/29418e34a94e2c43f861a321265f7f21035e7b19/torchvision/models/resnet.py
https://github.com/pytorch/vision/blob/29418e34a94e2c43f861a321265f7f21035e7b19/references/classification/presets.py
https://github.com/google/flax/blob/ce98f350c22599b31cce1b787f5ed2d5510f0706/examples/imagenet/input_pipeline.py
https://github.com/google-research/vision_transformer/blob/ac6e056f9da686895f9f0f6ac026d3b5a464e59e/vit_jax/input_pipeline.py#L195-L241
https://github.com/tensorflow/models/blob/5a0305c41304e8136e2056c589ab490a807dffa0/official/legacy/image_classification/augment.py
"""

from typing import Any, Optional

import numpy as np
import tensorflow as tf
from absl import logging

from axlearn.common import input_tf_data
from axlearn.common.config import config_for_function
from axlearn.common.utils import Tensor
from axlearn.vision import augment
from axlearn.vision.mask_generator import MaskingGenerator

# Mean and stddev RGB values from ImageNet
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

MEAN_RGB_NORMED = [x / 255.0 for x in MEAN_RGB]
STDDEV_RGB_NORMED = [x / 255.0 for x in STDDEV_RGB]


def filter_invalid_images(ds: tf.data.Dataset) -> tf.data.Dataset:
    # GIF image might have rank = 4.
    # Image with rank >=2 and rank <=4 are valid.
    return ds.filter(
        lambda x: tf.logical_and(
            tf.less_equal(tf.rank(x["image"]), 4), tf.greater_equal(tf.rank(x["image"]), 2)
        )
    )


def whiten(
    image: np.ndarray,
    *,
    mean_rgb: Optional[list[float]] = None,
    stddev_rgb: Optional[list[float]] = None,
) -> tf.Tensor:
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    image -= tf.constant(mean_rgb or MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(stddev_rgb or STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def de_whiten(
    image: np.ndarray,
    *,
    mean_rgb: Optional[list[float]] = None,
    stddev_rgb: Optional[list[float]] = None,
) -> tf.Tensor:
    """De-whitens given whitened image.

    Args:
        image: A float numpy array of shape [image_height, image_width, 3]
        mean_rgb: A float list of length 3 containing the mean RGB values to de-whiten the image.
        stddev_rgb: A float list of length 3 containing the standard deviation RGB values to
            de-whiten the image.

    Returns:
        A float tensor of shape [image_height, image_width, 3] containing the de-whitened image.
    """
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    image *= tf.constant(stddev_rgb or STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image += tf.constant(mean_rgb or MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def random_crop(
    image: tf.Tensor,
    aspect_ratio_range: tuple[float, float] = (0.75, 1.33),
    area_range: tuple[float, float] = (0.08, 1.0),
    max_attempts: int = 100,
):
    """Generates a randomly cropped image.

    Args:
        image: `Tensor` of shape [H, W, C].
        aspect_ratio_range: An optional list of `float`s. The cropped area of the image must have an
            aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image must contain a
            fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped region of the
            image of the specified constraints. After `max_attempts` failures, return the entire
            image.

    Returns:
        Cropped image `Tensor`, [H', W', C].
    """
    # A bounding box covering the entire image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # See `tf.image.sample_distorted_bounding_box` for more documentation.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.0,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    # Crop the image to the specified bounding box.
    return tf.slice(image, bbox_begin, bbox_size)


def central_crop(image, image_size):
    image_height, image_width, image_channels = image.shape
    offset_height = ((image_height - image_size[0]) + 1) // 2
    offset_width = ((image_width - image_size[1]) + 1) // 2
    return tf.slice(
        image, [offset_height, offset_width, 0], [image_size[0], image_size[1], image_channels]
    )


def central_crop_v2(image: tf.Tensor, center_crop_fraction: float = 0.875):
    """Center crop a square shape slice from the input image.

    It crops a square shape slice from the image. The side of the actual crop is
    224 / 256 = 0.875 of the short side of the original image.

    References:
    [1] Very Deep Convolutional Networks for Large-Scale Image Recognition
        https://arxiv.org/abs/1409.1556
    [2] Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385

    Args:
        image: a Tensor of shape [H, W, C] representing the input image.
        center_crop_fraction: the ratio between the crop and the short side of the original image.

    Returns:
        A Tensor representing the center cropped image.
    """
    image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    crop_size = center_crop_fraction * tf.math.minimum(image_size[0], image_size[1])
    crop_offset = tf.cast((image_size - crop_size) / 2.0, dtype=tf.int32)
    crop_size = tf.cast(crop_size, dtype=tf.int32)
    return image[
        crop_offset[0] : crop_offset[0] + crop_size, crop_offset[1] : crop_offset[1] + crop_size, :
    ]


def randaugment(
    image: Tensor,
    num_layers: int = 2,
    magnitude: float = 10.0,
    cutout_const: float = 40.0,
    translate_const: float = 100.0,
    exclude_ops: Optional[list[str]] = None,
):
    """Applies RandAugment from https://arxiv.org/abs/1909.13719.

    Reference:
    https://github.com/tensorflow/models/blob/5a0305c41304e8136e2056c589ab490a807dffa0/official/legacy/image_classification/augment.py
    """
    logging.info("Applying randaugment with %s layers and magnitude %s.", num_layers, magnitude)
    available_ops = [
        "AutoContrast",
        "Equalize",
        "Invert",
        "Rotate",
        "Posterize",
        "Solarize",
        "Color",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        "Cutout",
        "SolarizeAdd",
    ]
    if exclude_ops:
        logging.info("Excluding augmentation ops %s from RangAugment.", exclude_ops)
        available_ops = [op for op in available_ops if op not in exclude_ops]
    input_image_type = image.dtype
    if input_image_type != tf.uint8:
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, dtype=tf.uint8)

    replace_value = [128] * 3
    min_prob, max_prob = 0.2, 0.8
    for _ in range(num_layers):
        op_to_select = tf.random.uniform([], maxval=len(available_ops) + 1, dtype=tf.int32)

        branch_fns = []
        for i, op_name in enumerate(available_ops):
            prob = tf.random.uniform([], minval=min_prob, maxval=max_prob, dtype=tf.float32)
            func, _, args = augment.parse_policy_info(
                op_name, prob, magnitude, replace_value, cutout_const, translate_const
            )
            branch_fns.append(
                (
                    i,
                    lambda selected_func=func, selected_args=args: selected_func(
                        image, *selected_args
                    ),
                )
            )
        image = tf.switch_case(
            branch_index=op_to_select, branch_fns=branch_fns, default=lambda: tf.identity(image)
        )
    return tf.cast(image, dtype=input_image_type)


def random_erasing(
    image: Tensor,
    erasing_probability: float = 0.25,
):
    """Applies RandomErasing to a single image. Reference: https://arxiv.org/abs/1708.04896.

    Args:
        image: the input image of shape [H, W, C].
        erasing_probability: the probability of applying random erasing.

    Returns:
        The augmented image.
    """
    uniform_random = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
    mirror_cond = tf.less(uniform_random, erasing_probability)
    image = tf.cond(mirror_cond, lambda: augment.erase(image), lambda: image)
    return image


def crop_augment_whiten(
    image: Tensor,
    *,
    is_training: bool,
    image_size: tuple[int, int],
    eval_resize: Optional[tuple[int, int]] = None,
    augment_name: Optional[str] = None,
    randaug_num_layers: int = 2,
    randaug_magnitude: int = 10,
    randaug_exclude_ops: Optional[list[str]] = None,
    erasing_probability: Optional[float] = None,
    use_whitening: bool = True,
):
    if is_training:
        cropped_image = random_crop(image)
        # If random cropping failed, do center cropping instead.
        image = tf.cond(
            tf.reduce_all(tf.equal(tf.shape(cropped_image), tf.shape(image))),
            lambda: central_crop_v2(image),
            lambda: cropped_image,
        )
        image = tf.image.random_flip_left_right(image)
        # The resize changes aspect ratio of the crop during training.
        image = tf.image.resize([image], image_size, method=tf.image.ResizeMethod.BILINEAR)[0]
        # Apply RangAugment if set.
        if augment_name:
            if augment_name == "randaugment":
                image = randaugment(
                    image,
                    num_layers=randaug_num_layers,
                    magnitude=randaug_magnitude,
                    exclude_ops=randaug_exclude_ops,
                )
            else:
                raise ValueError(f"Augmentation type {augment_name} not supported.")
    else:
        # TODO(xianzhi): remove the second option if the performance of `_central_crop_v2` has
        # been verified for image captioning.
        if eval_resize is None:
            image = central_crop_v2(image)
            image = tf.image.resize([image], image_size, method=tf.image.ResizeMethod.BILINEAR)[0]
        else:
            image = tf.image.resize([image], eval_resize, method=tf.image.ResizeMethod.BILINEAR)[0]
            image = central_crop(image, image_size)

    # Whiten should be applied after augmentation, as augmentation methods assume image pixel
    # values to be within [0, 255].
    if use_whitening:
        image = whiten(image)
    else:
        image = tf.cast(image, tf.float32)
    # Apply random erasing after image normalization.
    if is_training and erasing_probability:
        image = random_erasing(image, erasing_probability=erasing_probability)

    return image


def _process_example(
    is_training: bool,
    image_size: tuple[int, int],
    eval_resize: Optional[tuple[int, int]] = None,
    num_parallel_calls: Optional[int] = None,
    augment_name: Optional[str] = None,
    randaug_num_layers: int = 2,
    randaug_magnitude: int = 10,
    randaug_exclude_ops: Optional[list[str]] = None,
    erasing_probability: Optional[float] = None,
    use_whitening: bool = True,
    mask_window_size: Optional[int] = None,
    num_masking_patches: Optional[int] = None,
    input_key: str = "image",
):
    def example_fn(example: dict[str, Tensor]) -> dict[str, Tensor]:
        image = example[input_key]
        image = crop_augment_whiten(
            image,
            is_training=is_training,
            image_size=image_size,
            eval_resize=eval_resize,
            augment_name=augment_name,
            randaug_num_layers=randaug_num_layers,
            randaug_magnitude=randaug_magnitude,
            randaug_exclude_ops=randaug_exclude_ops,
            erasing_probability=erasing_probability,
            use_whitening=use_whitening,
        )
        data = {"image": image, "label": example["label"]}

        # TODO(xianzhi): move mask generation to model implementation once verified performance.
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
        return ds.map(example_fn, num_parallel_calls=num_parallel_calls or tf.data.AUTOTUNE)

    return dataset_fn


def pad_with_negative_labels(element_spec: Any) -> Any:
    example = input_tf_data.default_pad_example_fn(element_spec)
    # For multilabel classification usecases, we support the plural version of label.
    for key in ("label", "labels", "text"):
        if key in example:
            example[key] = tf.negative(tf.ones_like(example[key]))
    return example


# TODO(markblee): Deprecate Input subclasses in favor of config builder pattern.
class ImagenetInput(input_tf_data.Input):
    """ImageNet input module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()  # type: ImagenetInput.Config
        cfg.source = config_for_function(input_tf_data.tfds_dataset).set(
            dataset_name="imagenet2012",
            train_shuffle_buffer_size=1024,  # to be tuned.
        )
        cfg.processor = config_for_function(_process_example).set(
            image_size=(224, 224),
            eval_resize=None,
            augment_name=None,
        )
        cfg.batcher.set(pad_example_fn=pad_with_negative_labels)  # pylint: disable=no-member
        return cfg


class Imagenetv2Input(ImagenetInput):
    """ImageNetV2 is a new testing data for the ImageNet benchmark.

    Reference: https://github.com/modestyachts/ImageNetV2
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.source = cfg.source.set(
            dataset_name="imagenet_v2/matched-frequency",
        )
        return cfg


def fake_image_dataset(
    is_training: bool,
    total_num_examples: Optional[int] = None,
    input_key: str = "image",
) -> input_tf_data.BuildDatasetFn:
    del is_training

    def example_fn(_) -> dict[str, Tensor]:
        rng = np.random.RandomState(0)
        image = rng.randint(
            low=0,
            high=256,
            size=[224, 224, 3],
            dtype=np.int32,
        )
        label = rng.randint(
            low=0,
            high=1000,
            dtype=np.int32,
        )
        return {input_key: image, "label": label}

    def fn() -> tf.data.Dataset:
        counter_ds = tf.data.experimental.Counter()
        if total_num_examples is not None:
            counter_ds = counter_ds.take(total_num_examples)
        return counter_ds.map(example_fn)

    return fn
