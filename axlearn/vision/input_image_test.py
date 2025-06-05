# Copyright © 2023 Apple Inc.

"""Tests image input modules."""
# pylint: disable=no-member,no-self-use
import jax.numpy as jnp
import tensorflow as tf
from absl import logging
from absl.testing import absltest, parameterized
from tensorflow_datasets.testing.mocking import mock_data

from axlearn.common import utils
from axlearn.common.config import config_for_function
from axlearn.vision.input_image import (
    ImagenetInput,
    Imagenetv2Input,
    crop_augment_whiten,
    fake_image_dataset,
    randaugment,
    random_erasing,
)


def _count_batches(dataset, max_batches=100):
    n = 0
    for _ in dataset:
        n += 1
        if n >= max_batches:
            return -1
    return n


class BaseImagenetInputTest(parameterized.TestCase):
    """Base class for ImagenetInputTest. Do not add tests here."""

    def _input_config(self, is_training: bool):
        cfg = ImagenetInput.default_config().set(name="test", is_training=is_training)
        cfg.source.set(
            dataset_name="imagenet2012",
            split="train",
            train_shuffle_buffer_size=100,
        )
        cfg.batcher.set(global_batch_size=8)
        return cfg

    def _build_input_config(self, dataset: str, *, is_training: bool):
        if dataset == "imagenet":
            cfg = self._input_config(is_training)
            cfg.source = config_for_function(fake_image_dataset)
        elif dataset == "imagenetv2":
            assert not is_training  # imagenetv2 is only for testing
            cfg = Imagenetv2Input.default_config().set(name="test", is_training=False)
            cfg.source = config_for_function(fake_image_dataset)
        else:
            raise NotImplementedError(dataset)
        return cfg

    def _test_fake_input(
        self, dataset, is_training, eval_resize, augment_name, erasing_probability, use_whitening
    ):
        cfg = self._build_input_config(dataset, is_training=is_training)
        cfg.processor.eval_resize = eval_resize
        cfg.processor.augment_name = augment_name
        cfg.processor.erasing_probability = erasing_probability
        cfg.processor.use_whitening = use_whitening
        if not is_training:
            cfg.source.set(total_num_examples=40)
        cfg.batcher.set(global_batch_size=8)
        dataset = cfg.instantiate(parent=None)
        if is_training:
            # For training, we loop over the dataset forever.
            self.assertEqual(-1, _count_batches(dataset, max_batches=24))
        else:
            # For evaluation, we loop over the dataset only once.
            self.assertEqual(40 // 8, _count_batches(dataset, max_batches=100))
        for batch in dataset:
            self.assertEqual({"image": (8, 224, 224, 3), "label": (8,)}, utils.shapes(batch))
            break

    def _test_fake_input_with_mask(self, dataset, is_training):
        batch_size = 2
        cfg = self._build_input_config(dataset, is_training=is_training)
        cfg.processor.mask_window_size = 14
        cfg.processor.num_masking_patches = 75
        if not is_training:
            cfg.source.set(total_num_examples=40)
        cfg.batcher.set(global_batch_size=batch_size)
        dataset = cfg.instantiate(parent=None)
        if is_training:
            # For training, we loop over the dataset forever.
            self.assertEqual(-1, _count_batches(dataset, max_batches=24))
        else:
            # For evaluation, we loop over the dataset only once.
            self.assertEqual(40 // batch_size, _count_batches(dataset, max_batches=100))
        cnt = 0
        prev_is_masked = None
        for batch in dataset:
            self.assertEqual(
                {
                    "image": (batch_size, 224, 224, 3),
                    "label": (batch_size,),
                    "is_masked": (batch_size, 14, 14),
                },
                utils.shapes(batch),
            )
            if cnt >= 1:
                assert not jnp.allclose(prev_is_masked, batch["is_masked"])
            prev_is_masked = batch["is_masked"]
            cnt += 1
            if cnt == 3:
                break


class ImagenetInputTest(BaseImagenetInputTest):
    @parameterized.parameters(False, True)
    def test_iteration(self, is_training):
        with mock_data(num_examples=40):
            cfg = self._input_config(is_training)
            dataset = cfg.instantiate(parent=None)
            if is_training:
                # For training, we loop over the dataset forever.
                self.assertEqual(-1, _count_batches(dataset, max_batches=24))
            else:
                # For evaluation, we loop over the dataset only once.
                self.assertEqual(40 // 8, _count_batches(dataset, max_batches=100))
            for batch in dataset:
                self.assertEqual({"image": (8, 224, 224, 3), "label": (8,)}, utils.shapes(batch))
                break

    @parameterized.parameters(25, 31)
    def test_indivisible(self, num_examples):
        batch_size = 8
        with mock_data(num_examples=num_examples):
            cfg = self._input_config(is_training=False)
            dataset = cfg.instantiate(parent=None)
            num_batches = 0
            last_batch = None
            for batch in dataset:
                logging.info(batch["label"])
                self.assertEqual(
                    {"image": (batch_size, 224, 224, 3), "label": (batch_size,)},
                    utils.shapes(batch),
                )
                last_batch = batch
                num_batches += 1
            self.assertEqual(num_batches, (num_examples + batch_size - 1) // batch_size)
            num_valid_examples_in_last_batch = num_examples % batch_size
            self.assertEqual(
                [-1] * (batch_size - num_valid_examples_in_last_batch),
                last_batch["label"][num_valid_examples_in_last_batch:].tolist(),
            )

    @parameterized.product(
        dataset=("imagenet", "imagenetv2"),
        is_training=(False, True),
        eval_resize=(None, (256, 256)),
        augment_name=(None, "randaugment"),
        erasing_probability=(None, 0.25),
        use_whitening=(False, True),
    )
    def test_fake_input(
        self, dataset, is_training, eval_resize, augment_name, erasing_probability, use_whitening
    ):
        if dataset == "imagenetv2" and is_training:
            return  # imagenetv2 is only used for testing.
        self._test_fake_input(
            dataset, is_training, eval_resize, augment_name, erasing_probability, use_whitening
        )

    @parameterized.product(
        dataset=("imagenet",),
        is_training=(False, True),
    )
    def test_fake_input_with_mask(self, dataset, is_training):
        self._test_fake_input_with_mask(dataset, is_training)


class AugmentationTest(parameterized.TestCase):
    """Tests augmentations."""

    @parameterized.product(
        magnitude=(10, 15),
        exclude_ops=(None, ["Cutout"]),
    )
    def test_randaugment(self, magnitude, exclude_ops):
        image = tf.random.normal((224, 224, 3), dtype=tf.float32)
        aug_image = randaugment(
            image,
            magnitude=magnitude,
            exclude_ops=exclude_ops,
        )
        self.assertFalse(tf.math.reduce_all(image == aug_image))

    @parameterized.parameters(0, 1, 0.25)
    def test_rand_erasing(self, erasing_probability):
        image = tf.random.normal((224, 224, 3), dtype=tf.float32)
        aug_image = random_erasing(
            image,
            erasing_probability=erasing_probability,
        )
        if erasing_probability == 0:
            self.assertTrue(tf.math.reduce_all(image == aug_image))
        elif erasing_probability == 1:
            self.assertFalse(tf.math.reduce_all(image == aug_image))


class CropAugmentWhitenTest(parameterized.TestCase):
    """Tests crop_augment_whiten."""

    @parameterized.parameters(True, False)
    def test_optional_whiten(self, use_whitening):
        image = tf.random.uniform(
            shape=[32, 32, 3],
            minval=0,
            maxval=256,
            dtype=tf.dtypes.int32,
            seed=0,
        )
        out_image = crop_augment_whiten(
            image, is_training=True, image_size=(32, 32), use_whitening=use_whitening
        )
        assert out_image.dtype == tf.float32
        min_value = tf.math.reduce_min(out_image)

        if use_whitening:
            self.assertTrue(min_value < 0.0)
        else:
            self.assertTrue(min_value >= 0.0)


if __name__ == "__main__":
    absltest.main()
