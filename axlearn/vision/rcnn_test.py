# Copyright © 2023 Apple Inc.

"""Tests RCNN layers."""
import jax.random
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.config import config_for_function
from axlearn.common.module import functional as F
from axlearn.common.vision_transformer import build_vit_model_config
from axlearn.vision.input_detection import DetectionInput, fake_detection_dataset
from axlearn.vision.rcnn import FasterRCNN

REF_BACKBONE_PARAMS = 23561152
REF_FPN_PARAMS = 3873792
REF_RPN_HEAD_PARAMS = 606765
REF_RCNN_HEAD_PARAMS = 15680967


def _input_config(is_training: bool, batch_size: int, image_size: int, output_stride: int):
    cfg = DetectionInput.default_config().set(name="test", is_training=is_training)
    cfg.source.set(
        dataset_name="coco/2017",
        split="validation",
        train_shuffle_buffer_size=100,
    )
    cfg.processor.set(
        image_size=(image_size, image_size), max_num_instances=10, output_stride=output_stride
    )
    cfg.batcher.set(global_batch_size=batch_size)
    return cfg


# TODO: Add a test to compare forward pass against reference implementation.
# pylint: disable=no-self-use
class FasterRCNNTest(parameterized.TestCase):
    """Tests FasterRCNN."""

    def test_model_params(self):
        image_data = {
            "image": np.random.uniform(-1, 1, [2, 128, 128, 3]).astype(np.float32),
            "image_info": np.array(
                [
                    [[64, 64], [128, 128], [2, 2], [0, 0]],
                    [[64, 64], [128, 128], [2, 2], [0, 0]],
                ]
            ),
        }

        cfg = FasterRCNN.default_config().set(name="test", num_classes=91)
        cfg.fpn.set(hidden_dim=256)

        model: FasterRCNN = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=image_data),
            method="predict",
        )

        # Test backbone number of parameters against the TF implementation.
        self.assertEqual(
            REF_BACKBONE_PARAMS,
            utils.count_model_params(state["backbone"]),
        )
        # Test fpn number of parameters against the TF implementation.
        self.assertEqual(REF_FPN_PARAMS, utils.count_model_params(state["fpn"]))
        # Test head number of parameters against the TF implementation.
        self.assertEqual(REF_RPN_HEAD_PARAMS, utils.count_model_params(state["rpn_head"]))
        # Test head number of parameters against the TF implementation.
        self.assertEqual(
            REF_RCNN_HEAD_PARAMS, utils.count_model_params(state["rcnn_detection_head"])
        )

    @parameterized.product(
        is_training=(True, False),
        backbone_type=("resnet", "vit"),
    )
    def test_model_forward(self, is_training, backbone_type):
        image_size = 128
        batch_size = 2
        min_level = 3
        max_level = 4
        output_stride = 2**max_level

        input_cfg = _input_config(
            is_training=is_training,
            batch_size=batch_size,
            image_size=image_size,
            output_stride=output_stride,
        )
        input_cfg.source = config_for_function(fake_detection_dataset)
        if not is_training:
            input_cfg.source.set(total_num_examples=batch_size)
        dataset = input_cfg.instantiate(parent=None)
        inputs = next(iter(dataset))
        cfg = FasterRCNN.default_config().set(
            name="test_rcnn",
            num_classes=91,
        )
        if backbone_type == "vit":
            cfg.backbone = build_vit_model_config(
                num_layers=1,
                model_dim=8,
                num_heads=4,
                image_size=(image_size, image_size),
                patch_size=(8, 8),
            )
        cfg.fpn.set(hidden_dim=256, min_level=min_level, max_level=max_level)
        model: FasterRCNN = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=inputs if is_training else inputs["image_data"]),
            method="forward" if is_training else "predict",
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
