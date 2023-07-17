# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for randaugment.

Reference:
https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/augment_test.py
"""
import tensorflow as tf
from absl.testing import parameterized

from axlearn.vision import augment


def get_dtype_test_cases():
    return [
        ("uint8", tf.uint8),
        ("int32", tf.int32),
        ("float16", tf.float16),
        ("float32", tf.float32),
    ]


@parameterized.named_parameters(get_dtype_test_cases())
class TransformsTest(parameterized.TestCase, tf.test.TestCase):
    """Basic tests for fundamental transformations."""

    def test_to_from_4d(self, dtype):
        for shape in [(10, 10), (10, 10, 10), (10, 10, 10, 10)]:
            original_ndims = len(shape)
            image = tf.zeros(shape, dtype=dtype)
            image_4d = augment.to_4d(image)
            self.assertEqual(4, tf.rank(image_4d))
            self.assertAllEqual(image, augment.from_4d(image_4d, original_ndims))

    def test_transform(self, dtype):
        image = tf.constant([[1, 2], [3, 4]], dtype=dtype)
        self.assertAllEqual(augment.transform(image, transforms=[1] * 8), [[4, 4], [4, 4]])

    def test_translate(self, dtype):
        image = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtype)
        translations = [-1, -1]
        translated = augment.translate(image=image, translations=translations)
        expected = [[1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 1]]
        self.assertAllEqual(translated, expected)

    def test_translate_shapes(self, dtype):
        translation = [0, 0]
        for shape in [(3, 3), (5, 5), (224, 224, 3)]:
            image = tf.zeros(shape, dtype=dtype)
            self.assertAllEqual(image, augment.translate(image, translation))

    def test_translate_invalid_translation(self, dtype):
        image = tf.zeros((1, 1), dtype=dtype)
        invalid_translation = [[[1, 1]]]
        with self.assertRaisesRegex(TypeError, "rank 1 or 2"):
            _ = augment.translate(image, invalid_translation)

    def test_rotate(self, dtype):
        image = tf.reshape(tf.cast(tf.range(9), dtype), (3, 3))
        rotation = 90.0
        transformed = augment.rotate(image=image, degrees=rotation)
        expected = [[2, 5, 8], [1, 4, 7], [0, 3, 6]]
        self.assertAllEqual(transformed, expected)

    def test_rotate_shapes(self, dtype):
        degrees = 0.0
        for shape in [(3, 3), (5, 5), (224, 224, 3)]:
            image = tf.zeros(shape, dtype=dtype)
            self.assertAllEqual(image, augment.rotate(image, degrees))


class RandAugmentTest(tf.test.TestCase):
    def test_all_policy_ops(self):
        """Smoke test to be sure all augmentation functions can execute."""

        prob = 1
        magnitude = 10
        replace_value = [128] * 3
        cutout_const = 100
        translate_const = 250

        image = tf.ones((224, 224, 3), dtype=tf.uint8)

        for op_name in augment.NAME_TO_FUNC:
            func, _, args = augment.parse_policy_info(
                op_name, prob, magnitude, replace_value, cutout_const, translate_const
            )
            image = func(image, *args)

        self.assertEqual((224, 224, 3), image.shape)


class MixupAndCutmixTest(tf.test.TestCase, parameterized.TestCase):
    def test_mixup_and_cutmix_smoothes_labels(self):
        batch_size = 12
        num_classes = 1000
        label_smoothing = 0.1

        images = tf.random.normal((batch_size, 224, 224, 3), dtype=tf.float32)
        labels = tf.range(batch_size)
        augmenter = augment.MixupAndCutmix(num_classes=num_classes, label_smoothing=label_smoothing)

        aug_images, aug_labels = augmenter.distort(images, labels)

        self.assertEqual(images.shape, aug_images.shape)
        self.assertEqual(images.dtype, aug_images.dtype)
        self.assertEqual([batch_size, num_classes], aug_labels.shape)
        self.assertAllLessEqual(
            aug_labels, 1.0 - label_smoothing + 2.0 / num_classes
        )  # With tolerance
        self.assertAllGreaterEqual(
            aug_labels, label_smoothing / num_classes - 1e4
        )  # With tolerance

    def test_mixup_changes_image(self):
        batch_size = 12
        num_classes = 1000
        label_smoothing = 0.1

        images = tf.random.normal((batch_size, 224, 224, 3), dtype=tf.float32)
        labels = tf.range(batch_size)
        augmenter = augment.MixupAndCutmix(
            mixup_alpha=1.0, cutmix_alpha=0.0, num_classes=num_classes
        )

        aug_images, aug_labels = augmenter.distort(images, labels)

        self.assertEqual(images.shape, aug_images.shape)
        self.assertEqual(images.dtype, aug_images.dtype)
        self.assertEqual([batch_size, num_classes], aug_labels.shape)
        self.assertAllLessEqual(
            aug_labels, 1.0 - label_smoothing + 2.0 / num_classes
        )  # With tolerance
        self.assertAllGreaterEqual(
            aug_labels, label_smoothing / num_classes - 1e4
        )  # With tolerance
        self.assertFalse(tf.math.reduce_all(images == aug_images))

    def test_cutmix_changes_image(self):
        batch_size = 12
        num_classes = 1000
        label_smoothing = 0.1

        images = tf.random.normal((batch_size, 224, 224, 3), dtype=tf.float32)
        labels = tf.range(batch_size)
        augmenter = augment.MixupAndCutmix(
            mixup_alpha=0.0, cutmix_alpha=1.0, num_classes=num_classes
        )

        aug_images, aug_labels = augmenter.distort(images, labels)

        self.assertEqual(images.shape, aug_images.shape)
        self.assertEqual(images.dtype, aug_images.dtype)
        self.assertEqual([batch_size, num_classes], aug_labels.shape)
        self.assertAllLessEqual(
            aug_labels, 1.0 - label_smoothing + 2.0 / num_classes
        )  # With tolerance
        self.assertAllGreaterEqual(
            aug_labels, label_smoothing / num_classes - 1e4
        )  # With tolerance
        self.assertFalse(tf.math.reduce_all(images == aug_images))

    def test_mixup_and_cutmix_smoothes_labels_with_videos(self):
        batch_size = 12
        num_classes = 1000
        label_smoothing = 0.1

        images = tf.random.normal((batch_size, 8, 224, 224, 3), dtype=tf.float32)
        labels = tf.range(batch_size)
        augmenter = augment.MixupAndCutmix(num_classes=num_classes, label_smoothing=label_smoothing)

        aug_images, aug_labels = augmenter.distort(images, labels)

        self.assertEqual(images.shape, aug_images.shape)
        self.assertEqual(images.dtype, aug_images.dtype)
        self.assertEqual([batch_size, num_classes], aug_labels.shape)
        self.assertAllLessEqual(
            aug_labels, 1.0 - label_smoothing + 2.0 / num_classes
        )  # With tolerance
        self.assertAllGreaterEqual(
            aug_labels, label_smoothing / num_classes - 1e4
        )  # With tolerance

    def test_mixup_changes_video(self):
        batch_size = 12
        num_classes = 1000
        label_smoothing = 0.1

        images = tf.random.normal((batch_size, 8, 224, 224, 3), dtype=tf.float32)
        labels = tf.range(batch_size)
        augmenter = augment.MixupAndCutmix(
            mixup_alpha=1.0, cutmix_alpha=0.0, num_classes=num_classes
        )

        aug_images, aug_labels = augmenter.distort(images, labels)

        self.assertEqual(images.shape, aug_images.shape)
        self.assertEqual(images.dtype, aug_images.dtype)
        self.assertEqual([batch_size, num_classes], aug_labels.shape)
        self.assertAllLessEqual(
            aug_labels, 1.0 - label_smoothing + 2.0 / num_classes
        )  # With tolerance
        self.assertAllGreaterEqual(
            aug_labels, label_smoothing / num_classes - 1e4
        )  # With tolerance
        self.assertFalse(tf.math.reduce_all(images == aug_images))

    def test_cutmix_changes_video(self):
        batch_size = 12
        num_classes = 1000
        label_smoothing = 0.1

        images = tf.random.normal((batch_size, 8, 224, 224, 3), dtype=tf.float32)
        labels = tf.range(batch_size)
        augmenter = augment.MixupAndCutmix(
            mixup_alpha=0.0, cutmix_alpha=1.0, num_classes=num_classes
        )

        aug_images, aug_labels = augmenter.distort(images, labels)

        self.assertEqual(images.shape, aug_images.shape)
        self.assertEqual(images.dtype, aug_images.dtype)
        self.assertEqual([batch_size, num_classes], aug_labels.shape)
        self.assertAllLessEqual(
            aug_labels, 1.0 - label_smoothing + 2.0 / num_classes
        )  # With tolerance
        self.assertAllGreaterEqual(
            aug_labels, label_smoothing / num_classes - 1e4
        )  # With tolerance
        self.assertFalse(tf.math.reduce_all(images == aug_images))


class FillRectangleTest(parameterized.TestCase):
    @parameterized.parameters(
        (100, 100, 10, 10, True),
        (100, 100, 200, 200, False),
        (-1, -1, 10, 10, False),
    )
    def test_fill_rectangle(
        self, center_width, center_height, half_width, half_height, self_replace
    ):
        image = tf.random.normal((224, 224, 3), dtype=tf.float32)
        # pylint: disable-next=protected-access
        aug_image = augment._fill_rectangle(
            image,
            center_width=center_width,
            center_height=center_height,
            half_width=half_width,
            half_height=half_height,
            replace=image if self_replace else None,
        )
        if self_replace:
            self.assertTrue(tf.math.reduce_all(image == aug_image))
        else:
            self.assertFalse(tf.math.reduce_all(image == aug_image))

    @parameterized.parameters(
        (100, 100, 10, 10, True),
        (100, 100, 200, 200, False),
        (-1, -1, 10, 10, False),
    )
    def test_fill_rectangle_4d(
        self, center_width, center_height, half_width, half_height, self_replace
    ):
        image = tf.random.normal((8, 224, 224, 3), dtype=tf.float32)
        # pylint: disable-next=protected-access
        aug_image = augment._fill_rectangle_4d(
            image,
            center_width=center_width,
            center_height=center_height,
            half_width=half_width,
            half_height=half_height,
            replace=image if self_replace else None,
        )
        if self_replace:
            self.assertTrue(tf.math.reduce_all(image == aug_image))
        else:
            self.assertFalse(tf.math.reduce_all(image == aug_image))


class RandomErasingTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters((0, 0), (0, 1))
    def test_random_erase_replaces_some_pixels(self, min_area, max_area):
        image = tf.zeros((224, 224, 3), dtype=tf.float32)
        aug_image = augment.erase(image, min_area=min_area, max_area=max_area, max_count=100)
        self.assertEqual((224, 224, 3), aug_image.shape)

        if min_area == max_area == 0:
            self.assertTrue(tf.math.reduce_all(image == aug_image))
        else:
            self.assertFalse(tf.math.reduce_all(image == aug_image))


if __name__ == "__main__":
    tf.test.main()
