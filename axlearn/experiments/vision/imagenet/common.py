# Copyright © 2023 Apple Inc.

"""ImageNet training configs.

Reference:
https://arxiv.org/abs/1409.0575
"""

import logging
from typing import Optional

from axlearn.common.config import config_for_function
from axlearn.common.input_tf_data import tfds_read_config
from axlearn.common.utils import get_data_dir
from axlearn.vision.input_image import ImagenetInput, fake_image_dataset

NUM_TRAIN_EXAMPLES = 1_281_167
NUM_CLASSES = 1_000


def input_config(
    *, split: str, global_batch_size: int, prefetch_buffer_size: Optional[int] = None
) -> ImagenetInput.Config:
    """Constructs an input config for ImageNet.

    Args:
        split: The dataset split, e.g. "train", "validation", or "test".
        global_batch_size: The global batch size.
        prefetch_buffer_size: Size of prefetch buffer for training. This allows later elements to be
            prepared while the current element is being processed. If set to None,
            `tf.data.experimental.AUTOTUNE` is used.

    Returns:
        An input config.
    """

    cfg = ImagenetInput.default_config()
    if get_data_dir() == "FAKE":
        logging.warning("Using FAKE inputs!")
        cfg.source = config_for_function(fake_image_dataset).set(
            total_num_examples=global_batch_size * 2,
        )
    else:
        cfg.source.set(
            split=split,
            read_config=config_for_function(tfds_read_config).set(decode_parallelism=128),
        )
    cfg.processor.set(num_parallel_calls=1024)
    cfg.batcher.set(global_batch_size=global_batch_size, prefetch_buffer_size=prefetch_buffer_size)
    return cfg
