# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# microsoft/unilm:
# Copyright (c) 2021 Microsoft.
# Licensed under The MIT License.

"""A mask generator implementation.

Code reference: https://github.com/microsoft/unilm/blob/master/beit2/masking_generator.py
"""
import math
import random
from typing import Optional

import numpy as np


# pylint: disable-next=too-many-instance-attributes
class MaskingGenerator:
    """Generates a random mask that will be used to mask a portion of the patchified input image.

    This method selects a random area at a random aspect ratio to mask at each time. It stops when
    the total number of selected mask patches reach the specified number.

    Reference: https://arxiv.org/pdf/2106.08254.pdf
    """

    def __init__(
        self,
        *,
        input_size: tuple[int, int],
        num_masking_patches: int,
        num_attempts: int = 10,
        min_mask_patches: int = 16,
        min_aspect: float = 0.3,
        max_mask_patches: Optional[int] = None,
        max_aspect: Optional[float] = None,
    ):
        """Initializes MaskingGenerator.

        Args:
            input_size: an int tuple that represents (height, width) of the patchified target.
            num_masking_patches: the number of patches to be masked.
            num_attempts: the max number of attempts for one mask generation trial.
            min_mask_patches: the min number of patches for one masking area.
            max_mask_patches: the max number of patches for one masking area. If None, sets to
                num_masking_patches.
            min_aspect: the min aspect ratio (height/width) for one masking area.
            max_aspect: the max aspect ratio for one masking area. If None, sets to 1 / min_aspect.

        Raises:
            ValueError: if min_aspect or max_aspect are below 0 or max_aspect is smaller than
                min_aspect.
        """
        self.height, self.width = input_size
        # Total number of patches in the pachified input.
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.num_attempts = num_attempts
        self.min_mask_patches = min_mask_patches
        self.max_mask_patches = (
            num_masking_patches if max_mask_patches is None else max_mask_patches
        )
        max_aspect = max_aspect or 1 / min_aspect
        if min_aspect <= 0 or max_aspect <= 0:
            raise ValueError("Both min and max aspect ratios need to be positive.")
        if min_aspect > max_aspect:
            raise ValueError("min_aspect needs to be no greater than max_aspect.")
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self) -> str:
        # pylint: disable-next=consider-using-f-string
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_mask_patches,
            self.max_mask_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self) -> tuple[int, ...]:
        return self.height, self.width

    def _mask(self, mask, max_mask_patches) -> int:
        """Initializes MaskingGenerator.

        Args:
            mask: a boolean 2D mask that represents the masked positions.
            max_mask_patches: the max number of masked patches for the current trial.

        Returns:
            The number of masked positions in this trial.
        """
        delta = 0
        # pylint: disable-next=too-many-nested-blocks,unused-variable
        for attempt in range(self.num_attempts):
            target_area = random.uniform(self.min_mask_patches, max_mask_patches)
            # TODO(xianzhi): consider generating random aspect_ratio first and then bound the
            # target_area sampling so it doesn't get rejected.
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            # Note that h*w can potentially be out of [min_mask_patches, max_mask_patches].
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Assign 1 to selected patches if no overlap with existing masked patches.
                if 0 < h * w - num_masked <= max_mask_patches:
                    # TODO(xianzhi): consider vectorizing these loops.
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                # Stop at this attempt if a mask has been successfully generated.
                if delta > 0:
                    break
        return delta

    def __call__(self) -> np.ndarray:
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        # pylint: disable=no-else-break,unbalanced-tuple-unpacking
        # Keeps selecting one new random area if mask_count does not reach num_masking_patches.
        # TODO(xianzhi): consider consolidating the while and the if/else logics.
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_mask_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        # TODO(haotian): EVA does not have the following restrictions.
        # maintain a fix number {self.num_masking_patches}
        if mask_count > self.num_masking_patches:
            delta = mask_count - self.num_masking_patches
            mask_x, mask_y = mask.nonzero()
            to_vis = np.random.choice(mask_x.shape[0], delta, replace=False)
            # pylint: disable-next=unsubscriptable-object
            mask[mask_x[to_vis], mask_y[to_vis]] = 0

        elif mask_count < self.num_masking_patches:
            delta = self.num_masking_patches - mask_count
            mask_x, mask_y = (mask == 0).nonzero()
            to_mask = np.random.choice(mask_x.shape[0], delta, replace=False)
            # pylint: disable-next=unsubscriptable-object
            mask[mask_x[to_mask], mask_y[to_mask]] = 1

        assert mask.sum() == self.num_masking_patches, f"mask: {mask}, mask count {mask.sum()}"

        return mask
