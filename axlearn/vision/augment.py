# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/models:
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# keras-team/keras:
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# facebookresearch/mvit:
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""RandAugment policies for enhanced image preprocessing.

Reference:
https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/augment.py
"""

# pylint: disable=too-many-lines
import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import tensorflow as tf

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0


def to_4d(image: tf.Tensor) -> tf.Tensor:
    """Converts an input Tensor to 4 dimensions.

    4D image => [N, H, W, C] or [N, C, H, W]
    3D image => [1, H, W, C] or [1, C, H, W]
    2D image => [1, H, W, 1]

    Args:
        image: The 2/3/4D input tensor.

    Returns:
        A 4D image tensor.

    Raises:
        `TypeError` if `image` is not a 2/3/4D tensor.
    """
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        0,
    )
    return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
    """Converts a 4D image back to `ndims` rank."""
    shape = tf.shape(image)
    begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
    """Converts translations to a projective transform.

    The translation matrix looks like this:
        [[1 0 -dx]
        [0 1 -dy]
        [0 0 1]]

    Args:
        translations: The 2-element list representing [dx, dy], or a list of
            2-element lists representing [dx dy] to translate for each image.
            The shape must be static.

    Returns:
        A transformation matrix of shape (num_images, 8) to be used by
        https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/layers/preprocessing/image_preprocessing.py#L898-L985

    Raises:
        TypeError: If
            - the shape of `translations` is not known or
            - the shape of `translations` is not rank 1 or 2.
    """
    translations = tf.convert_to_tensor(translations, dtype=tf.float32)
    if translations.get_shape().ndims is None:
        raise TypeError("translations rank must be statically known")
    if len(translations.get_shape()) == 1:
        translations = translations[None]
    elif len(translations.get_shape()) != 2:
        raise TypeError("translations should have rank 1 or 2.")
    num_translations = tf.shape(translations)[0]

    return tf.concat(
        [
            tf.ones((num_translations, 1), tf.dtypes.float32),
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            -translations[:, 0, None],
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            tf.ones((num_translations, 1), tf.dtypes.float32),
            -translations[:, 1, None],
            tf.zeros((num_translations, 2), tf.dtypes.float32),
        ],
        1,
    )


def _convert_angles_to_transform(
    angles: tf.Tensor, image_width: tf.Tensor, image_height: tf.Tensor
) -> tf.Tensor:
    """Converts an angle or angles to a projective transform.

    Args:
        angles: A scalar to rotate all images, or a vector to rotate a batch of
            images.
        image_width: The width of the image(s) to be transformed.
        image_height: The height of the image(s) to be transformed.

    Returns:
        A transformation matrix of shape (num_images, 8) to be used by
        https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/layers/preprocessing/image_preprocessing.py#L898-L985

    Raises:
        TypeError: If `angles` is not rank 0 or 1.
    """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if len(angles.get_shape()) == 0:
        angles = angles[None]
    elif len(angles.get_shape()) != 1:
        raise TypeError("Angles should have a rank 0 or 1.")
    x_offset = (
        (image_width - 1)
        - (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) * (image_height - 1))
    ) / 2.0
    y_offset = (
        (image_height - 1)
        - (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) * (image_height - 1))
    ) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        [
            tf.math.cos(angles)[:, None],
            -tf.math.sin(angles)[:, None],
            x_offset[:, None],
            tf.math.sin(angles)[:, None],
            tf.math.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.dtypes.float32),
        ],
        1,
    )


# TODO(xianzhi,markblee): Avoid copying this private function from Keras.
def _keras_image_processing_transform(
    images: tf.Tensor,
    transforms: Union[Sequence[float], tf.Tensor],
    fill_mode: str = "reflect",
    fill_value: float = 0.0,
    interpolation: str = "bilinear",
    output_shape: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Applies the given transform(s) to the image(s).

    Copied from
    https://github.com/keras-team/keras/blob/v2.14.0/keras/layers/preprocessing/image_preprocessing.py#L720-L815
    since it was removed from the public API, with minor adaptation.
    """
    with tf.name_scope(name or "transform"):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(output_shape, tf.int32, name="output_shape")

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                f"output_shape must be a 1-D Tensor of 2 elements: new_height, new_width, "
                f"instead got {output_shape} "
            )

        fill_value = tf.convert_to_tensor(fill_value, tf.float32, name="fill_value")

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )


def transform(image: tf.Tensor, transforms) -> tf.Tensor:
    """Prepares input data for `_keras_image_processing_transform`."""
    original_ndims = tf.rank(image)
    transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if transforms.shape.rank == 1:
        transforms = transforms[None]
    image = to_4d(image)
    image = _keras_image_processing_transform(
        images=image, transforms=transforms, interpolation="nearest"
    )
    return from_4d(image, original_ndims)


def translate(image: tf.Tensor, translations: tf.Tensor) -> tf.Tensor:
    """Translates image(s) by provided vectors.

    Args:
        image: An image Tensor of type uint8.
        translations: A vector or matrix representing [dx dy].

    Returns:
        The translated version of the image.
    """
    transforms = _convert_translation_to_transform(translations)
    return transform(image, transforms=transforms)


def rotate(image: tf.Tensor, degrees: float) -> tf.Tensor:
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
        image: An image Tensor of type uint8.
        degrees: Float, a scalar angle in degrees to rotate all images by. If
            degrees is positive the image will be rotated clockwise otherwise
            it will be rotated counterclockwise.

    Returns:
        The rotated version of image.
    """
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = tf.cast(degrees * degrees_to_radians, tf.float32)

    original_ndims = tf.rank(image)
    image = to_4d(image)

    image_height = tf.cast(tf.shape(image)[1], tf.float32)
    image_width = tf.cast(tf.shape(image)[2], tf.float32)
    transforms = _convert_angles_to_transform(
        angles=radians, image_width=image_width, image_height=image_height
    )
    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    image = transform(image, transforms=transforms)
    return from_4d(image, original_ndims)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
        image1: An image Tensor of type uint8.
        image2: An image Tensor of type uint8.
        factor: A floating point value above 0.0.

    Returns:
        A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if 0.0 < factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
        image: An image Tensor of type uint8.
        pad_size: Specifies how big the zero mask that will be generated is that is
            applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
        replace: What pixel value to fill in the image in the area that has the
            cutout mask applied to it.

    Returns:
        An image Tensor that is of type uint8.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height, dtype=tf.int32
    )

    cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image)
    return image


def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(image: tf.Tensor, addition: int = 0, threshold: int = 128) -> tf.Tensor:
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)


def color(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return blend(degenerate, image, factor)


def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
    """Equivalent of PIL Posterize."""
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def wrapped_rotate(image: tf.Tensor, degrees: float, replace: int) -> tf.Tensor:
    """Applies rotation with wrap/unwrap."""
    image = rotate(wrap(image), degrees=degrees)
    return unwrap(image, replace)


def translate_x(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
    """Equivalent of PIL Translate in X dimension."""
    image = translate(wrap(image), [-pixels, 0])
    return unwrap(image, replace)


def translate_y(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
    """Equivalent of PIL Translate in Y dimension."""
    image = translate(wrap(image), [0, -pixels])
    return unwrap(image, replace)


def shear_x(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    image = transform(image=wrap(image), transforms=[1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return unwrap(image, replace)


def shear_y(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    image = transform(image=wrap(image), transforms=[1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0])
    return unwrap(image, replace)


def autocontrast(image: tf.Tensor) -> tf.Tensor:
    """Implements Autocontrast function from PIL using TF ops.

    Args:
        image: A 3D uint8 tensor.

    Returns:
        The image after it has had autocontrast applied to it and will be of
        type uint8.
    """

    def scale_channel(image: tf.Tensor) -> tf.Tensor:
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = (
        tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]) / 13.0
    )
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding="VALID", dilations=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


def equalize(image: tf.Tensor) -> tf.Tensor:
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0), lambda: im, lambda: tf.gather(build_lut(histo, step), im)
        )

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


def invert(image: tf.Tensor) -> tf.Tensor:
    """Inverts the image pixels."""
    image = tf.convert_to_tensor(image)
    return 255 - image


def wrap(image: tf.Tensor) -> tf.Tensor:
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf.concat([image, extended_channel], 2)
    return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
    """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.


    Args:
        image: A 3D Image Tensor with 4 channels.
        replace: A one or three value 1D tensor to fill empty pixels.

    Returns:
        image: A 3D image Tensor with 3 channels.
    """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[2]])

    # Find all pixels where the last channel is zero.
    alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)

    replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(
        tf.equal(alpha_channel, 0),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace,
        flattened_image,
    )

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
    return image


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level: float):
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate_tensor(level)
    return (level,)


def _shrink_level_to_arg(level: float):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return (1.0,)  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2.0 / (_MAX_LEVEL / level) + 0.9
    return (level,)


def _enhance_level_to_arg(level: float):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level: float):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _translate_level_to_arg(level: float, translate_const: float):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _mult_to_arg(level: float, multiplier: float = 1.0):
    return (int((level / _MAX_LEVEL) * multiplier),)


def _apply_func_with_prob(func: Any, image: tf.Tensor, args: Any, prob: float):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    assert isinstance(args, tuple)

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
    augmented_image = tf.cond(should_apply_op, lambda: func(image, *args), lambda: image)
    return augmented_image


def select_and_apply_random_policy(policies: Any, image: tf.Tensor):
    """Select a random policy from `policies` and apply it to `image`."""
    policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for i, policy in enumerate(policies):
        image = tf.cond(
            tf.equal(i, policy_to_select),
            lambda selected_policy=policy: selected_policy(image),
            lambda: image,
        )
    return image


NAME_TO_FUNC = {
    "AutoContrast": autocontrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": wrapped_rotate,
    "Posterize": posterize,
    "Solarize": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "Contrast": contrast,
    "Brightness": brightness,
    "Sharpness": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x,
    "TranslateY": translate_y,
    "Cutout": cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset(
    {
        "Rotate",
        "TranslateX",
        "ShearX",
        "ShearY",
        "TranslateY",
        "Cutout",
    }
)


def level_to_arg(cutout_const: float, translate_const: float):
    """Creates a dict mapping image operation names to their arguments."""

    no_arg = lambda level: ()
    posterize_arg = lambda level: _mult_to_arg(level, 4)
    solarize_arg = lambda level: _mult_to_arg(level, 256)
    solarize_add_arg = lambda level: _mult_to_arg(level, 110)
    cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
    translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

    args = {
        "AutoContrast": no_arg,
        "Equalize": no_arg,
        "Invert": no_arg,
        "Rotate": _rotate_level_to_arg,
        "Posterize": posterize_arg,
        "Solarize": solarize_arg,
        "SolarizeAdd": solarize_add_arg,
        "Color": _enhance_level_to_arg,
        "Contrast": _enhance_level_to_arg,
        "Brightness": _enhance_level_to_arg,
        "Sharpness": _enhance_level_to_arg,
        "ShearX": _shear_level_to_arg,
        "ShearY": _shear_level_to_arg,
        "Cutout": cutout_arg,
        "TranslateX": translate_arg,
        "TranslateY": translate_arg,
    }
    return args


def parse_policy_info(
    name: str,
    prob: float,
    level: float,
    replace_value: list[int],
    cutout_const: float,
    translate_const: float,
) -> tuple[Any, float, Any]:
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(cutout_const, translate_const)[name](level)

    if name in REPLACE_FUNCS:
        # Add in replace arg if it is required for the function that is called.
        args = tuple(list(args) + [replace_value])

    return func, prob, args


def _fill_rectangle(image, center_width, center_height, half_width, half_height, replace=None):
    """Fills blank area for the input image.

    Args:
        image: the input image in [h, w, c].
        center_width: width coordinate of the center pixel of the filling area.
        center_height: height coordinate of the center pixel of the filling area.
        half_width: half width of the filling area.
        half_height: half height of the filling area.
        replace: a tensor of the same shape as the input image to be used to fill the blank area.
            If None, random noise is used.

    Returns:
        image: an output image in [h, w, c] with the filled area.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # The blank area will always be within the image.
    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])

    if replace is None:
        fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)

    return image


def _fill_rectangle_4d(image, center_width, center_height, half_width, half_height, replace=None):
    """Fills blank area for 4D images along the spatial dimensions.

    Args:
        image: the 4D input image in [t, h, w, c].
        center_width: width coordinate of the center pixel of the filling area.
        center_height: height coordinate of the center pixel of the filling area.
        half_width: half width of the filling area.
        half_height: half height of the filling area.
        replace: a tensor of the same shape as the input image to be used to fill the blank area.
            If None, random noise is used.

    Returns:
        image: an output 4D image in [t, h, w, c] with the filled area.
    """
    image_time = tf.shape(image)[0]
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]

    # The blank area will always be within the image.
    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [
        image_time,
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[0, 0], [lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 1, 3])

    if replace is None:
        fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    image = tf.where(tf.equal(mask, 0), fill, image)

    return image


# pylint: disable-next=too-many-instance-attributes
class MixupAndCutmix:
    """Applies Mixup and/or Cutmix to a batch of images.
    - Mixup: https://arxiv.org/abs/1710.09412
    - Cutmix: https://arxiv.org/abs/1905.04899

    Code reference:
    https://github.com/tensorflow/models/blob/983109490fcdd06bcea7d4c60f6a8a4b943aceda/official/vision/ops/augment.py#L2240-L2401
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
        num_classes: int = 1000,
    ):
        """Applies Mixup and/or Cutmix to a batch of images.

        Args:
            mixup_alpha (float, optional): For drawing a random lambda (`lam`) from a
                beta distribution (for each image). If zero Mixup is deactivated.
                Defaults to .8.
            cutmix_alpha (float, optional): For drawing a random lambda (`lam`) from a
                beta distribution (for each image). If zero Cutmix is deactivated.
                Defaults to 1..
            prob (float, optional): Of augmenting the batch. Defaults to 1.0.
            switch_prob (float, optional): Probability of applying Cutmix for the
                batch. Defaults to 0.5.
            label_smoothing (float, optional): Constant for label smoothing. Defaults
                to 0.1.
            num_classes (int, optional): Number of classes. Defaults to 1000.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = "batch"
        self.mixup_enabled = True

        if self.mixup_alpha and not self.cutmix_alpha:
            self.switch_prob = -1
        elif not self.mixup_alpha and self.cutmix_alpha:
            self.switch_prob = 1

    def __call__(self, images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self.distort(images, labels)

    def distort(self, images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Applies Mixup and/or Cutmix to batch of images and transforms labels.

        Args:
            images (tf.Tensor): Of shape [batch_size, height, width, 3] representing a
                batch of image, or [batch_size, time, height, width, 3] representing a
                batch of video.
            labels (tf.Tensor): Of shape [batch_size, ] representing the class id for
                each image of the batch.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The augmented version of `image` and `labels`.
        """
        labels = tf.reshape(labels, [-1])
        augment_cond = tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.mix_prob)
        augment_a = lambda: self._update_labels(
            *tf.cond(
                tf.less(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.switch_prob),
                lambda: self._cutmix(images, labels),
                lambda: self._mixup(images, labels),
            )
        )
        augment_b = lambda: (images, self._smooth_labels(labels))
        return tf.cond(augment_cond, augment_a, augment_b)

    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def _cutmix(
        self, images: tf.Tensor, labels: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Applies cutmix."""
        lam = MixupAndCutmix._sample_from_beta(
            self.cutmix_alpha, self.cutmix_alpha, tf.shape(labels)
        )

        ratio = tf.math.sqrt(1 - lam)

        batch_size = tf.shape(images)[0]

        if images.shape.rank == 4:
            image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
            fill_fn = _fill_rectangle
        elif images.shape.rank == 5:
            image_height, image_width = tf.shape(images)[2], tf.shape(images)[3]
            fill_fn = _fill_rectangle_4d
        else:
            raise ValueError(f"Bad image rank: {images.shape.rank}.")

        cut_height = tf.cast(ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)
        cut_width = tf.cast(ratio * tf.cast(image_height, dtype=tf.float32), dtype=tf.int32)

        random_center_height = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_height, dtype=tf.int32
        )
        random_center_width = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=image_width, dtype=tf.int32
        )

        bbox_area = cut_height * cut_width
        lam = 1.0 - bbox_area / (image_height * image_width)
        lam = tf.cast(lam, dtype=tf.float32)

        images = tf.map_fn(
            lambda x: fill_fn(*x),
            (
                images,
                random_center_width,
                random_center_height,
                cut_width // 2,
                cut_height // 2,
                tf.reverse(images, [0]),
            ),
            dtype=(images.dtype, tf.int32, tf.int32, tf.int32, tf.int32, images.dtype),
            fn_output_signature=tf.TensorSpec(images.shape[1:], dtype=images.dtype),
        )

        return images, labels, lam

    def _mixup(
        self, images: tf.Tensor, labels: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Applies mixup."""
        lam = MixupAndCutmix._sample_from_beta(self.mixup_alpha, self.mixup_alpha, tf.shape(labels))
        if images.shape.rank == 4:
            lam = tf.reshape(lam, [-1, 1, 1, 1])
        elif images.shape.rank == 5:
            lam = tf.reshape(lam, [-1, 1, 1, 1, 1])
        else:
            raise ValueError(f"Bad image rank: {images.shape.rank}.")

        lam_cast = tf.cast(lam, dtype=images.dtype)
        images = lam_cast * images + (1.0 - lam_cast) * tf.reverse(images, [0])

        return images, labels, tf.squeeze(lam)

    def _smooth_labels(self, labels: tf.Tensor) -> tf.Tensor:
        off_value = self.label_smoothing / self.num_classes
        on_value = 1.0 - self.label_smoothing + off_value

        smooth_labels = tf.one_hot(labels, self.num_classes, on_value=on_value, off_value=off_value)
        return smooth_labels

    def _update_labels(
        self, images: tf.Tensor, labels: tf.Tensor, lam: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        labels_1 = self._smooth_labels(labels)
        labels_2 = tf.reverse(labels_1, [0])

        lam = tf.reshape(lam, [-1, 1])
        labels = lam * labels_1 + (1.0 - lam) * labels_2

        return images, labels


def erase(
    image: tf.Tensor,
    min_area: float = 0.02,
    max_area: float = 1 / 3,
    min_aspect: float = 0.3,
    min_count: int = 1,
    max_count: int = 1,
    trials: int = 10,
    max_aspect: Optional[float] = None,
) -> tf.Tensor:
    """Erase an area.

    Args:
        image: The input image in [H, W, C].
        min_area: Minimum area of the random erasing rectangle.
        max_area: Maximum area of the random erasing rectangle.
        min_aspect: Minimum aspect rate of the random erasing rectangle.
        min_count: Minimum number of erased rectangles.
        max_count: Maximum number of erased rectangles.
        trials: Maximum number of trials to randomly sample a rectangle that
            fulfills constraint.
        max_aspect: Maximum aspect rate of the random erasing rectangle. If not set,
            max_aspect = 1 / min_aspect.

    Returns:
        The augmented image.

    Raises:
        ValueError: If min_count >= max_count.
    """
    if min_count == max_count:
        count = min_count
    elif min_count < max_count:
        count = tf.random.uniform(
            shape=[],
            minval=int(min_count),
            maxval=int(max_count + 1),
            dtype=tf.int32,
        )
    else:
        raise ValueError(
            f"min_count ({min_count}) should be no greater than max_count ({max_count})."
        )

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    area = tf.cast(image_width * image_height, tf.float32)

    for _ in range(count):
        # Work around since break is not supported in tf.function
        is_trial_successful = False
        for _ in range(trials):
            if not is_trial_successful:
                erase_area = tf.random.uniform(
                    shape=[], minval=area * min_area, maxval=area * max_area
                )
                aspect_ratio = tf.math.exp(
                    tf.random.uniform(
                        shape=[],
                        minval=math.log(min_aspect),
                        maxval=math.log(max_aspect or 1 / min_aspect),
                    )
                )

                half_height = tf.cast(
                    tf.math.round(tf.math.sqrt(erase_area * aspect_ratio) / 2), dtype=tf.int32
                )
                half_width = tf.cast(
                    tf.math.round(tf.math.sqrt(erase_area / aspect_ratio) / 2), dtype=tf.int32
                )

                if 2 * half_height < image_height and 2 * half_width < image_width:
                    center_height = tf.random.uniform(
                        shape=[],
                        minval=0,
                        maxval=int(image_height - 2 * half_height),
                        dtype=tf.int32,
                    )
                    center_width = tf.random.uniform(
                        shape=[],
                        minval=0,
                        maxval=int(image_width - 2 * half_width),
                        dtype=tf.int32,
                    )

                    image = _fill_rectangle(
                        image,
                        center_width,
                        center_height,
                        half_width,
                        half_height,
                        replace=None,
                    )

                    is_trial_successful = True
    return image
