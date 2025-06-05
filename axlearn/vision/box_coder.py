# Copyright © 2023 Apple Inc.

"""Box coder, for converting between image and anchor coordinates."""
import enum

import jax
import jax.numpy as jnp
import numpy as np

from axlearn.common.config import Configurable, config_class
from axlearn.common.utils import Tensor

# Max ratio between box height/width and anchor height/width in log scale.
BBOX_XFORM_CLIP_EXP = 1000.0 / 16.0
BBOX_XFORM_CLIP = np.log(BBOX_XFORM_CLIP_EXP)


class BoxClipMethod(str, enum.Enum):
    """Methods for clipping decoded boxes.

    NONE: Do not clip Y, X, height and width offsets.
    MaxHW: Clip height and width offset to maximum value of BBOX_XFORM_CLIP.
    MinMaxYXHW: Clip Y and X offset to to be within [-BBOX_XFORM_CLIP_EXP, BBOX_XFORM_CLIP_EXP],
        clip height and width offset to be within [-BBOX_XFORM_CLIP, BBOX_XFORM_CLIP]
    """

    # pylint: disable=invalid-name
    NONE = "None"
    MaxHW = "MaxHW"
    MinMaxYXHW = "MinMaxYXHW"
    # pylint: enable=invalid-name


class BoxCoder(Configurable):
    """Class to encode/decode boxes with respect to anchors.

    Reference: https://arxiv.org/pdf/1506.01497.pdf.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures BoxCoder."""

        # Epsilon value to add to box and anchor sizes before taking a log transform of their
        # ratios.
        eps: float = 1e-8
        # Minimum value for the box width and height before encoding.
        box_wh_min_value: float = 0.0
        # Scaling factor for encoded box coordinates of the form
        # (y_center, x_center, height, width).
        weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
        # Box clipping method.
        clip_boxes: BoxClipMethod = BoxClipMethod.MaxHW

    def encode(self, *, boxes: Tensor, anchors: Tensor) -> Tensor:
        """Encodes boxes with respect to anchors.

        Args:
            boxes: A [..., 4] float tensor with boxes to encode. Boxes must be of the form
                [ymin, xmin, ymax, xmax].
            anchors: A [..., 4] float tensor containing anchors. Anchors must be broadcastable to
                `boxes` and of the form [ymin, xmin, ymax, xmax].

        Returns:
            A [..., 4] float tensor containing encoded boxes.
        """
        cfg = self.config
        boxes = boxes.astype(anchors.dtype)
        ymin = boxes[..., 0:1]
        xmin = boxes[..., 1:2]
        ymax = boxes[..., 2:3]
        xmax = boxes[..., 3:4]
        box_h = ymax - ymin + cfg.eps
        box_w = xmax - xmin + cfg.eps

        box_h = jnp.maximum(box_h, cfg.box_wh_min_value)
        box_w = jnp.maximum(box_w, cfg.box_wh_min_value)

        box_yc = ymin + 0.5 * box_h
        box_xc = xmin + 0.5 * box_w

        anchor_ymin = anchors[..., 0:1]
        anchor_xmin = anchors[..., 1:2]
        anchor_ymax = anchors[..., 2:3]
        anchor_xmax = anchors[..., 3:4]
        anchor_h = anchor_ymax - anchor_ymin + cfg.eps
        anchor_w = anchor_xmax - anchor_xmin + cfg.eps
        anchor_yc = anchor_ymin + 0.5 * anchor_h
        anchor_xc = anchor_xmin + 0.5 * anchor_w

        encoded_dy = (box_yc - anchor_yc) / anchor_h
        encoded_dx = (box_xc - anchor_xc) / anchor_w
        encoded_dh = jnp.log(box_h / anchor_h)
        encoded_dw = jnp.log(box_w / anchor_w)

        encoded_dy *= cfg.weights[0]
        encoded_dx *= cfg.weights[1]
        encoded_dh *= cfg.weights[2]
        encoded_dw *= cfg.weights[3]

        encoded_boxes = jnp.concatenate([encoded_dy, encoded_dx, encoded_dh, encoded_dw], axis=-1)
        return encoded_boxes

    def decode(self, *, encoded_boxes: Tensor, anchors: Tensor) -> Tensor:
        """Decodes encoded boxes with respect to anchors.

        Args:
            encoded_boxes: A [..., 4] float tensor with encoded boxes.
            anchors: A [..., 4] float tensor containing anchors. Anchors must be broadcastable to
                `boxes` and of the form [ymin, xmin, ymax, xmax].

        Returns:
            A [..., 4] float tensor containing decoded boxes of the form [ymin, xmin, ymax, xmax].
        """
        # pylint: disable=invalid-name
        cfg = self.config
        encoded_boxes = encoded_boxes.astype(anchors.dtype)
        dy = encoded_boxes[..., 0:1]
        dx = encoded_boxes[..., 1:2]
        dh = encoded_boxes[..., 2:3]
        dw = encoded_boxes[..., 3:4]

        dy /= cfg.weights[0]
        dx /= cfg.weights[1]
        dh /= cfg.weights[2]
        dw /= cfg.weights[3]

        if cfg.clip_boxes == BoxClipMethod.MaxHW:
            dh = jnp.minimum(dh, BBOX_XFORM_CLIP)
            dw = jnp.minimum(dw, BBOX_XFORM_CLIP)
        elif cfg.clip_boxes == BoxClipMethod.MinMaxYXHW:
            dy = jax.lax.clamp(-BBOX_XFORM_CLIP_EXP, dy, BBOX_XFORM_CLIP_EXP)
            dx = jax.lax.clamp(-BBOX_XFORM_CLIP_EXP, dx, BBOX_XFORM_CLIP_EXP)
            dh = jax.lax.clamp(-BBOX_XFORM_CLIP, dh, BBOX_XFORM_CLIP)
            dw = jax.lax.clamp(-BBOX_XFORM_CLIP, dw, BBOX_XFORM_CLIP)

        anchor_ymin = anchors[..., 0:1]
        anchor_xmin = anchors[..., 1:2]
        anchor_ymax = anchors[..., 2:3]
        anchor_xmax = anchors[..., 3:4]
        anchor_h = anchor_ymax - anchor_ymin
        anchor_w = anchor_xmax - anchor_xmin
        anchor_yc = anchor_ymin + 0.5 * anchor_h
        anchor_xc = anchor_xmin + 0.5 * anchor_w

        decoded_boxes_yc = dy * anchor_h + anchor_yc
        decoded_boxes_xc = dx * anchor_w + anchor_xc
        decoded_boxes_h = jnp.exp(dh) * anchor_h
        decoded_boxes_w = jnp.exp(dw) * anchor_w

        decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
        decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
        decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h
        decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w

        decoded_boxes = jnp.concatenate(
            [decoded_boxes_ymin, decoded_boxes_xmin, decoded_boxes_ymax, decoded_boxes_xmax],
            axis=-1,
        )
        # pylint: enable=invalid-name
        return decoded_boxes
