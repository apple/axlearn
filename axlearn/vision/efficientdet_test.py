# Copyright © 2023 Apple Inc.

"""Tests EfficientDet layers."""
import jax.random
import numpy as np
from absl.testing import parameterized

from axlearn.common.module import functional as F
from axlearn.common.test_utils import flatten_items
from axlearn.vision.efficientdet import (
    EFFICIENTDETVARIANTS,
    BoxClassHead,
    PredictionHead,
    efficientdet_boxclasshead_config,
    set_efficientdet_config,
)
from axlearn.vision.retinanet import RetinaNetModel


class PredictionHeadTest(parameterized.TestCase):
    """Tests PredictionHead."""

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        batch_size = 2
        hidden_dim = 8
        min_level = 3
        max_level = 5
        image_size = 32
        num_anchors = 13
        num_layers = 2
        targets_per_anchor = 4

        inputs = {}
        for level in range(min_level, max_level + 1):
            inputs[level] = np.random.uniform(
                -1,
                1,
                [batch_size, image_size // 2**level, image_size // 2**level, hidden_dim],
            ).astype(np.float32)

        cfg = PredictionHead.default_config().set(
            name="test",
            input_dim=hidden_dim,
            head_conv_output_dim=targets_per_anchor * num_anchors,
            min_level=min_level,
            max_level=max_level,
            num_layers=num_layers,
        )
        model: PredictionHead = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )

        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs)
            expected_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                num_anchors * targets_per_anchor,
            )
            self.assertEqual(expected_shape, outputs[level].shape)


class BoxClassHeadTest(parameterized.TestCase):
    """Tests BoxClassHead."""

    @staticmethod
    def _compute_num_params(
        input_dim: int,
        num_layers: int,
        num_levels: int,
        num_anchors: int,
        num_classes: int,
        add_redundant_bias: bool,
    ) -> int:
        expected_num_params = 0
        # norm layers: scale, bias, moving_mean, moving_variances
        expected_num_params += num_layers * num_levels * input_dim * 4
        # ds_conv layers with 3x3 conv
        expected_num_params += (3 * 3 * input_dim + input_dim * input_dim) * num_layers
        if add_redundant_bias:
            expected_num_params += input_dim * num_layers
        # same architecture for box and class head
        expected_num_params *= 2
        # box head convolution weight
        expected_num_params += 3 * 3 * input_dim * 4 * num_anchors
        # box head convolution bias
        expected_num_params += 4 * num_anchors
        # class head convolution weight
        expected_num_params += 3 * 3 * input_dim * num_anchors * num_classes
        # class head convolution bias
        expected_num_params += num_anchors * num_classes
        return expected_num_params

    @parameterized.product(
        add_redundant_bias=(False, True),
    )
    def test_num_parameters(self, add_redundant_bias: bool):
        input_dim = 8
        num_classes = 11
        num_anchors = 9
        num_layers = 3
        min_level = 3
        max_level = 4
        num_levels = max_level - min_level + 1

        cfg = efficientdet_boxclasshead_config(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_layers=num_layers,
            min_level=min_level,
            max_level=max_level,
            add_redundant_bias=add_redundant_bias,
            use_ds_conv_in_head=False,
        )
        cfg.set(name="test")
        model: BoxClassHead = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        num_params = 0
        for _, param in flatten_items(state):
            num_params += param.size

        expected_num_params = self._compute_num_params(
            input_dim=input_dim,
            num_layers=num_layers,
            num_levels=num_levels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            add_redundant_bias=add_redundant_bias,
        )
        self.assertEqual(expected_num_params, num_params)

    @parameterized.product(
        is_training=(False, True),
    )
    def test_model_forward(self, is_training):
        batch_size = 2
        input_dim = 8
        num_classes = 11
        num_anchors = 9
        num_layers = 2
        min_level = 3
        max_level = 5
        image_size = 32

        inputs = {}
        for level in range(min_level, max_level + 1):
            inputs[level] = np.random.uniform(
                -1,
                1,
                [batch_size, image_size // 2**level, image_size // 2**level, input_dim],
            ).astype(np.float32)

        cfg = efficientdet_boxclasshead_config(
            input_dim=input_dim,
            num_classes=num_classes,
            num_anchors=num_anchors,
            num_layers=num_layers,
            min_level=min_level,
            max_level=max_level,
        )
        cfg.set(name="test")
        model: BoxClassHead = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )

        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs["class_outputs"])
            self.assertIn(level, outputs["box_outputs"])
            expected_class_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                num_classes * num_anchors,
            )
            expected_box_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                4 * num_anchors,
            )
            self.assertEqual(expected_class_shape, outputs["class_outputs"][level].shape)
            self.assertEqual(expected_box_shape, outputs["box_outputs"][level].shape)


class EfficientDetTest(parameterized.TestCase):
    """Tests EfficientDet."""

    def test_model_forward(self, is_training=False):
        batch_size = 2
        image_size = 512
        min_level = 3
        max_level = 7
        num_anchors = 9
        num_classes = 11

        inputs = np.random.uniform(
            -1,
            1,
            [batch_size, image_size, image_size, 3],
        ).astype(np.float32)

        (
            _,
            backbone_variant,
            hidden_dim,
            num_bifpn_layers,
            num_head_layers,
        ) = EFFICIENTDETVARIANTS["test"]
        cfg = set_efficientdet_config(
            backbone_variant=backbone_variant,
            num_head_layers=num_head_layers,
            min_level=min_level,
            max_level=max_level,
            add_redundant_bias=False,
        )
        cfg.name = "test"
        cfg.num_classes = num_classes
        cfg.fpn_hidden_dim = hidden_dim
        cfg.head_hidden_dim = hidden_dim
        cfg.head.num_anchors = num_anchors
        cfg.fpn.num_bifpn_layers = num_bifpn_layers

        model: RetinaNetModel = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=dict(image=inputs)),
            method="_predict_raw",
        )

        for level in range(min_level, max_level + 1):
            self.assertIn(level, outputs["class_outputs"])
            self.assertIn(level, outputs["box_outputs"])
            expected_class_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                num_classes * num_anchors,
            )
            expected_box_shape = (
                batch_size,
                image_size // 2**level,
                image_size // 2**level,
                4 * num_anchors,
            )
            self.assertEqual(expected_class_shape, outputs["class_outputs"][level].shape)
            self.assertEqual(expected_box_shape, outputs["box_outputs"][level].shape)

    @parameterized.parameters(("d0",))
    def test_number_of_parameters(self, efficientdet_variant):
        # Only testing "d0" variant in order to speed up the test.
        min_level = 3
        max_level = 7
        num_anchors = 9
        num_classes = 80

        expected_num_params = {
            # Reference: EfficientDetBackbone with default args from
            # https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/tree/15403b5371a64defb2a7c74e162c6e880a7f462c
            "test": 31670,  # custom variant, this is not supported in the reference implementation.
            "d0": 3874217,
            "d1": 6617888,
            "d2": 8086869,
            "d3": 12017806,
            "d4": 20703425,
            "d5": 33627305,
            "d6": 51837284,
            "d7": 51837284,
        }

        efficientdet_variant_values = EFFICIENTDETVARIANTS[efficientdet_variant]
        (
            _,
            backbone_variant,
            hidden_dim,
            num_bifpn_layers,
            num_head_layers,
        ) = efficientdet_variant_values
        cfg = set_efficientdet_config(
            backbone_variant=backbone_variant,
            num_head_layers=num_head_layers,
            min_level=min_level,
            max_level=max_level,
            add_redundant_bias=True,
        )
        cfg.name = "test"
        cfg.num_classes = num_classes
        cfg.fpn_hidden_dim = hidden_dim
        cfg.head_hidden_dim = hidden_dim
        cfg.head.num_anchors = num_anchors
        cfg.fpn.num_bifpn_layers = num_bifpn_layers

        cfg = cfg.set(name="test")

        model: RetinaNetModel = cfg.instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        total_num_params = 0
        for name, param in flatten_items(state):
            if name.split("/")[-1] in {"moving_mean", "moving_variance"}:
                # Batch norm stats are not considered parameters in the torch reference.
                continue
            total_num_params += param.size

        self.assertEqual(expected_num_params[efficientdet_variant], total_num_params)
