# Copyright © 2023 Apple Inc.

"""Utils for image and annotation visualizations."""
import numpy as np

from axlearn.common.utils import NestedTensor, Tensor
from axlearn.vision.input_image import de_whiten
from axlearn.vision.utils_detection import clip_boxes


def draw_box_on_image(
    *,
    bbox: Tensor,
    image: Tensor,
    color: np.ndarray = np.array([255, 255, 255]),
    edge_thickness_fraction: float = 0.006,
) -> Tensor:
    """Draws box on image.

    Draws the given bbox in [y_min, x_min, y_max, x_max] format on the given image, with specified
    color and thickness.

    Args:
        bbox: A float tensor of shape 4 containing bounding box co-ordinates in the format
            [y_min, x_min, y_max, x_max].
        image: A float tensor of shape (image_height, image_width, 3)
        color: A float tensor of shape 3 containing RBG values to draw the box.
        edge_thickness_fraction: A float value representing the thickness of the box to draw as a
            fraction of image dimensions.

    Returns:
        A float tensor of shape (image_height, image_width, 3) with the bbox drawn.
    """

    image_height, image_width, _ = image.shape
    edge_thickness_pixels = min(image_height, image_width) * edge_thickness_fraction

    def _thick_left_edge(coordinate, bound):
        return max(int(coordinate - edge_thickness_pixels / 2), bound)

    def _thick_right_edge(coordinate, bound):
        return min(int(coordinate + edge_thickness_pixels / 2), bound - 1)

    y_min, x_min, y_max, x_max = bbox
    image[
        _thick_left_edge(y_min, 0) : _thick_right_edge(y_min, image_height) + 1, x_min : x_max + 1
    ] = color
    image[
        _thick_left_edge(y_max, 0) : _thick_right_edge(y_max, image_height) + 1, x_min : x_max + 1
    ] = color
    image[
        y_min : y_max + 1, _thick_left_edge(x_min, 0) : _thick_right_edge(x_min, image_width) + 1
    ] = color
    image[
        y_min : y_max + 1, _thick_left_edge(x_max, 0) : _thick_right_edge(x_max, image_width) + 1
    ] = color
    return image


def visualize_detections(
    *,
    images: Tensor,
    predictions: NestedTensor,
    groundtruths: NestedTensor,
    cutoff_score: float = 0.3,
) -> Tensor:
    """Draws predicted and groundtruth boxes on images.

    Draws prediction outputs of detector models along with groundtruth annotated boxes on
    one image per input batch.

    Args:
        images: A float tensor of shape [batch_size, image_height, image_width, 3] containing
            whitened images.
        predictions: A dictionary containing the output of the detector model containing
            predicted boxes, scores and classes.
        groundtruths: A dictionary containing the groundtruth annotations corresponding to the
            batch of images.
        cutoff_score: A float value indicating the cutoff for predicted score for visualizing the
            predicted boxes.

    Returns:
        A float tensor of shape [image_height, image_width, 3] with predicted and groundtruth boxes
            drawn on the first image from the input batch.
    """
    example_id = 0

    image = de_whiten(images[example_id]).numpy() / 255
    image_height, image_width, _ = image.shape

    groundtruth_boxes = groundtruths["boxes"][example_id]
    for i in range(groundtruths["num_detections"][example_id].item()):
        clipped_box = (
            clip_boxes(groundtruth_boxes[i], [image_height, image_width]).numpy().astype(np.int32)
        )
        # Draw groundtruth boxes in blue color
        image = draw_box_on_image(bbox=clipped_box, image=image, color=np.array([0, 0, 1]))

    prediction_boxes = predictions["detection_boxes"][example_id]
    prediction_scores = predictions["detection_scores"][example_id]
    for i in range(predictions["num_detections"][example_id].item()):
        # If the score of the predicted box is less than cutoff then do not visualize.
        if prediction_scores[i] < cutoff_score:
            continue

        clipped_box = (
            clip_boxes(prediction_boxes[i], [image_height, image_width]).numpy().astype(np.int32)
        )
        # Draw prediction boxes in red color
        image = draw_box_on_image(bbox=clipped_box, image=image, color=np.array([1, 0, 0]))

    return image
