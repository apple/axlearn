"""Image processing utils."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_text_to_image(img: Image.Image, text: str):
    """Add text to an image. We call such change as OCR-attack.

    Args:
        img: PIL image
        text: string to be written to image

    Returns:
        a new PIL image

    Note:
        img cannot be a tf tensor, otherwise will generate error
        "Cannot convert a symbolic Tensor  to a numpy array"
    """

    # We use the "Arial Unicode" font which is available on Mac.
    # You may change it to other font.
    font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 50)
    im2 = Image.fromarray(np.uint8(img))

    i1 = ImageDraw.Draw(im2)
    i1.text((50, 50), text, font=font, fill=(255, 0, 0))
    return im2


def to_rgb(img: np.ndarray):
    """
    Args:
        img: 3D numpy array, but shape can be (, , 4) or (, , 1)

    Returns
        3D shape array with shape (, , 3)
    """

    if img.ndim == 4:
        # This might be a gif image with leading rank indicating the number of frames.
        # Pick the first image in the gif.
        img = img[0]
    if img.shape[-1] == 1:
        # this is a gray image
        img = img.squeeze(axis=-1)
        img = np.stack([img] * 3, axis=-1)
    assert img.ndim == 3 and img.shape[-1] == 3

    return img
