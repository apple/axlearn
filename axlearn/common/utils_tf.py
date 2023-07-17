# Copyright © 2023 Apple Inc.

"""Tensorflow utils."""
from typing import Union

import tensorflow as tf


def masked_fill(orig: tf.Tensor, mask: tf.Tensor, fill_value: Union[int, float, str]) -> tf.Tensor:
    """Replaces values in `orig` with `fill_value` where `mask` is true.

    Args:
        orig: A Tensor representing the original values.
        mask: A boolean Tensor of the same shape as `orig`,
            representing where the values should be replaced.
        fill_value: The value to fill where mask is True.

    Returns:
        A Tensor of the same size and dtype as `orig`, but with some values replaced with
        `fill_value` where `mask` is true.
    """
    fill = tf.cast(tf.fill(tf.shape(orig), fill_value), orig.dtype)
    return tf.where(mask, fill, orig)
