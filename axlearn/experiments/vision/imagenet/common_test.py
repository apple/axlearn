# Copyright © 2023 Apple Inc.

"""Tests ImageNet configs."""

from absl.testing import parameterized

from axlearn.common.test_utils import TestWithTemporaryCWD
from axlearn.common.utils import set_data_dir
from axlearn.experiments.vision import imagenet


class InputConfigTest(TestWithTemporaryCWD):
    """Tests input configs."""

    @parameterized.parameters(
        dict(split="train", global_batch_size=4, prefetch_buffer_size=2),
        dict(split="train[:50]", global_batch_size=10, prefetch_buffer_size=None),
    )
    def test_input_config(self, **kwargs):
        with set_data_dir(self._temp_root.name):
            cfg = imagenet.input_config(**kwargs)

            self.assertEqual(cfg.source.split, kwargs["split"])
            self.assertEqual(cfg.batcher.global_batch_size, kwargs["global_batch_size"])
            self.assertEqual(cfg.batcher.prefetch_buffer_size, kwargs["prefetch_buffer_size"])

    @parameterized.parameters([True, False])
    def test_fake_input_config(self, is_training):
        with set_data_dir("FAKE"):
            global_batch_size = 2
            cfg = imagenet.input_config(split="test", global_batch_size=global_batch_size)
            ds = cfg.set(name="test", is_training=is_training).instantiate(parent=None)

            num_examples = 0
            for example in ds.dataset().take(4):
                self.assertEqual(example["image"].shape, [global_batch_size, 224, 224, 3])
                num_examples += 1

            if is_training:
                # In the training case, expect to see > 2 batches.
                self.assertGreater(num_examples, 2)
            else:
                # In the eval case, expect to see only 2 batches.
                self.assertEqual(num_examples, 2)
