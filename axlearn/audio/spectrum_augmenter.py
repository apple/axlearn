# Copyright © 2023 Apple Inc.
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Spectrum augmentation utilities."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.module import Module
from axlearn.common.utils import Tensor, safe_not


class MaskSampler(BaseLayer):
    """A layer to generate masks given input lengths.

    Reference: https://arxiv.org/abs/1912.05533 Section 2.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures MaskSampler."""

        # At least one of `max_num_masks` and `max_num_masks_ratio` should be set.
        # None means "no limit". `max_num_masks` will be set to the max sequence length if None.
        # num_masks = min(max_num_masks, floor(input_length * max_num_masks_ratio)).
        max_num_masks: Optional[int] = None
        max_num_masks_ratio: Optional[float] = None

        # At least one of `max_mask_length` and `max_mask_length_ratio` should be set.
        # None means "no limit". `max_mask_length` will be set to the max sequence length if None.
        # `max_mask_length_ratio` will be set to 1 if None.
        # mask_length = uniform(
        #     0, min(max_mask_length, floor(input_length * max_mask_length_ratio))
        # ).
        max_mask_length: Optional[int] = None
        max_mask_length_ratio: Optional[float] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        def check_one_of(a: str, b: str):
            if getattr(cfg, a) is None and getattr(cfg, b) is None:
                raise ValueError(f"Expected at least one of {a} or {b} to be set.")

        def check_positive(name: str):
            value = getattr(cfg, name)
            if not (value is None or value > 0):
                raise ValueError(f"Expected {name} to be greater than 0, got {value}.")

        def check_range(name: str):
            value = getattr(cfg, name)
            if not (value is None or 0 < value <= 1.0):
                raise ValueError(f"Expected {name} to be in (0, 1], got {value}.")

        check_one_of("max_num_masks", "max_num_masks_ratio")
        check_one_of("max_mask_length", "max_mask_length_ratio")
        check_positive("max_num_masks")
        check_positive("max_mask_length")
        check_range("max_num_masks_ratio")
        check_range("max_mask_length_ratio")

    def forward(self, input_lengths: Tensor, *, max_length: int) -> Tensor:
        """Generates masks based on the provided input lengths.

        Args:
            input_lengths: An int Tensor of shape [batch_size].
            max_length: Maximum length of inputs.

        Returns:
            A 0/1 Tensor of shape [batch_size, max_length]. 1 means masked position.
        """
        cfg: MaskSampler.Config = self.config
        batch_size = input_lengths.shape[0]
        length_key, start_key = jax.random.split(self.prng_key, num=2)

        max_mask_length_ratio = cfg.max_mask_length_ratio or 1.0
        max_mask_length = min(cfg.max_mask_length or max_length, max_length)
        max_num_masks = min(cfg.max_num_masks or max_length, max_length)

        # [batch_size, num_masks=1].
        input_lengths = input_lengths[:, None]
        # [batch_size, num_masks=1] with values in [1, max_length].
        max_mask_length = jnp.maximum(
            jnp.minimum(max_mask_length, max_mask_length_ratio * input_lengths), 1
        ).astype(input_lengths.dtype)

        # Sample lengths of the masks.
        # [batch_size, max_num_masks] with values in [0, max_mask_length].
        lengths = jax.random.randint(
            length_key, shape=[batch_size, max_num_masks], minval=0, maxval=max_mask_length + 1
        )
        # Sample start indices of the masks.
        # [batch_size, max_num_masks] with values in [0, input_length - lengths].
        starts = jax.random.randint(
            start_key,
            shape=[batch_size, max_num_masks],
            minval=0,
            maxval=input_lengths - lengths + 1,
        )
        # Construct masks of shape [batch_size, max_num_masks, max_length].
        mask_index = jnp.arange(max_length)
        masks = (starts[..., None] <= mask_index) & (mask_index < (starts + lengths)[..., None])

        # Sample adaptive number of masks.
        if cfg.max_num_masks_ratio:
            num_masks = (cfg.max_num_masks_ratio * input_lengths).astype(input_lengths.dtype)
            masks = masks * (jnp.arange(max_num_masks) < num_masks)[..., None]

        # Reduce over max_num_masks axis.
        masks = jnp.max(masks, axis=1)
        chex.assert_type(masks, jnp.bool)
        return masks


class SpectrumAugmenter(BaseLayer):
    """SpecAugment layer.

    References:
    https://arxiv.org/abs/1904.08779
    https://arxiv.org/abs/1912.05533
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures SpectrumAugmenter."""

        # TODO(markblee): Add time warping.
        freq_mask_sampler: MaskSampler.Config = MaskSampler.default_config()
        time_mask_sampler: MaskSampler.Config = MaskSampler.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("freq_mask_sampler", cfg.freq_mask_sampler)
        self._add_child("time_mask_sampler", cfg.time_mask_sampler)

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> dict[str, Tensor]:
        """Applies SpecAugment to inputs.

        Args:
            inputs: A Tensor of shape [batch_size, num_frames, num_freq, num_channels].
            paddings: A 0/1 Tensor of shape [batch_size, num_frames].

        Returns:
            A Tensor of shape [batch_size, num_frames, num_freq, num_channels].
        """
        if not self.is_training:
            return inputs

        batch_size, num_frames, num_freq = inputs.shape[:3]

        # [batch_size, num_freq].
        freq_masks = self.freq_mask_sampler(
            input_lengths=jnp.repeat(num_freq, batch_size),
            max_length=num_freq,
        )
        # [batch_size, num_frames].
        time_masks = self.time_mask_sampler(
            input_lengths=jnp.sum(safe_not(paddings), axis=1),
            max_length=num_frames,
        )

        # [batch_size, 1, num_freq, 1].
        freq_keep = safe_not(freq_masks)[:, None, :, None]
        # [batch_size, num_frames, 1, 1].
        time_keep = safe_not(time_masks)[:, :, None, None]

        return inputs * freq_keep * time_keep
