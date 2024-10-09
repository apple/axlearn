# Copyright © 2023 Apple Inc.

"""Tests image classification models."""
import difflib
from functools import partial
from typing import Callable

import jax.random
import numpy as np
import pytest
import timm
import torch
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from axlearn.common import utils
from axlearn.common.config import InstantiableConfig
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor
from axlearn.common.vision_transformer import build_vit_model_config
from axlearn.vision.image_classification import ImageClassificationModel
from axlearn.vision.mobilenets import ModelNames, named_model_configs
from axlearn.vision.param_converter import MobileNetsParamConverter, ResNetParamConverter
from axlearn.vision.resnet import ResNet


# pylint: disable=no-self-use
class ClassificationModelTest(parameterized.TestCase):
    def _base_test(
        self,
        *,
        cfg: InstantiableConfig,
        ref: torch.nn.Module,
        params_from_model: Callable,
        is_training: bool = False,
    ):
        model = cfg.set(name="test", num_classes=1000).instantiate(parent=None)
        init_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))

        num_params = utils.count_model_params(init_params)
        logging.info("Model contains %s Million parameters.", num_params / 10.0**6)

        param_spec_strs = [
            f"{name}={list(param.shape)}" for name, param in utils.flatten_items(init_params)
        ]
        for name, param in ref.named_parameters():
            logging.info("ref param: %s=%s", name, list(param.shape))

        model_params = jax.tree.map(utils.as_tensor, params_from_model(ref=ref))
        model_param_strs = [
            f"{name}={list(param.shape)}" for name, param in utils.flatten_items(model_params)
        ]
        self.assertEqual(
            model_param_strs,
            param_spec_strs,
            "\n".join(difflib.ndiff(model_param_strs, param_spec_strs)),
        )
        batch_size = 2
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            "label": np.random.randint(0, 999, [batch_size]).astype(np.int32),
        }
        (_, aux), _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=model_params,
            inputs=dict(input_batch=jax.tree.map(jnp.asarray, inputs)),
        )
        if not is_training:
            ref.eval()
            ref_logits = ref(torch.as_tensor(inputs["image"]).permute(0, 3, 1, 2)).detach().numpy()
            np.testing.assert_allclose(aux["logits"], ref_logits, atol=1e-4, rtol=1e-3)

    def _get_resnet_cfg_model(self, model_depth):
        model = ImageClassificationModel.default_config()
        if model_depth == 18:
            cfg, ref = model.set(backbone=ResNet.resnet18_config()), resnet18
        elif model_depth == 34:
            cfg, ref = model.set(backbone=ResNet.resnet34_config()), resnet34
        elif model_depth == 50:
            cfg, ref = model.set(backbone=ResNet.resnet50_config()), resnet50
        elif model_depth == 101:
            cfg, ref = model.set(backbone=ResNet.resnet101_config()), resnet101
        elif model_depth == 152:
            cfg, ref = model.set(backbone=ResNet.resnet152_config()), resnet152
        else:
            raise ValueError(f"ResNet model depth {model_depth} not implemented.")
        return cfg, ref

    @parameterized.product(
        model_depth=(18, 34, 50, 101, 152),
        is_training=(False, True),
    )
    def test_resnets(self, model_depth, is_training):
        cfg, torch_model = self._get_resnet_cfg_model(model_depth)
        params_from_model = ResNetParamConverter.params_from_model
        self._base_test(
            cfg=cfg,
            ref=torch_model(pretrained=True),
            params_from_model=params_from_model,
            is_training=is_training,
        )

    def _get_mobilenet_cfg_model(self, model_name: ModelNames, variant: str, version: str):
        backbone_cfg = named_model_configs(model_name, variant, efficientnet_version=version)
        # Use TF default settings for BN and padding as we compare against TF reference checkpoint.
        # https://github.com/huggingface/pytorch-image-models/blob/394e8145551191ae60f672556936314a20232a35/timm/models/efficientnet.py#L1926-L2223
        cfg = ImageClassificationModel.default_config().set(backbone=backbone_cfg)

        timm_model_name = {
            (ModelNames.EFFICIENTNET, "V1"): "efficientnet",
            (ModelNames.EFFICIENTNET, "V2"): "efficientnetv2",
            (ModelNames.MOBILENETV3, None): "mobilenetv3",
        }[(model_name, version)]
        timm_variant = variant.replace("-", "_").lower()
        ref = timm.create_model(f"tf_{timm_model_name}_{timm_variant}", pretrained=True)

        return cfg, ref

    # TODO(edaxberger): Re-enable test after fixing failing tests due to limited CI resources.
    @parameterized.product(
        model_arch=(
            (ModelNames.MOBILENETV3, "small-minimal-100", None),
            (ModelNames.MOBILENETV3, "small-075", None),
            (ModelNames.MOBILENETV3, "small-100", None),
            (ModelNames.MOBILENETV3, "large-minimal-100", None),
            (ModelNames.MOBILENETV3, "large-075", None),
            (ModelNames.MOBILENETV3, "large-100", None),
            (ModelNames.EFFICIENTNET, "B0", "V1"),
            (ModelNames.EFFICIENTNET, "B1", "V1"),
            (ModelNames.EFFICIENTNET, "B2", "V1"),
            (ModelNames.EFFICIENTNET, "B3", "V1"),
            (ModelNames.EFFICIENTNET, "B4", "V1"),
            (ModelNames.EFFICIENTNET, "B5", "V1"),
            (ModelNames.EFFICIENTNET, "B6", "V1"),
            (ModelNames.EFFICIENTNET, "B7", "V1"),
            (ModelNames.EFFICIENTNET, "B8", "V1"),
            (ModelNames.EFFICIENTNET, "lite0", "V1"),
            (ModelNames.EFFICIENTNET, "lite1", "V1"),
            (ModelNames.EFFICIENTNET, "lite2", "V1"),
            (ModelNames.EFFICIENTNET, "lite3", "V1"),
            (ModelNames.EFFICIENTNET, "lite4", "V1"),
            (ModelNames.EFFICIENTNET, "B0", "V2"),
            (ModelNames.EFFICIENTNET, "B1", "V2"),
            (ModelNames.EFFICIENTNET, "B2", "V2"),
            (ModelNames.EFFICIENTNET, "B3", "V2"),
            (ModelNames.EFFICIENTNET, "S", "V2"),
            (ModelNames.EFFICIENTNET, "M", "V2"),
            (ModelNames.EFFICIENTNET, "L", "V2"),
        ),
        is_training=(False, True),
    )
    @pytest.mark.high_cpu
    def test_mobilenets(self, model_arch, is_training):
        cfg, torch_model = self._get_mobilenet_cfg_model(*model_arch)
        params_from_model = partial(MobileNetsParamConverter.params_from_model, cfg=cfg)
        self._base_test(
            cfg=cfg,
            ref=torch_model,
            params_from_model=params_from_model,
            is_training=is_training,
        )


class ViTClassificationModelTest(TestCase):
    @parameterized.parameters(0, 1, 2)
    def test_padded_examples(self, num_padded_examples):
        vit_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
            patch_size=(16, 16),
        )
        cfg = ImageClassificationModel.default_config().set(
            name="test", backbone=vit_cfg, num_classes=1000
        )
        model = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size = 2
        num_valid_examples = batch_size - num_padded_examples
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            "label": np.asarray(
                [99] * num_valid_examples + [-1] * num_padded_examples, dtype=np.int32
            ),
        }

        (loss, aux), output_collection = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=as_tensor(inputs)),
        )
        if num_valid_examples:
            self.assertGreater(loss, 0)
        else:
            self.assertEqual(loss, 0)
        self.assertIn("logits", aux)
        self.assertAlmostEqual(
            output_collection.summaries["metric"]["loss"], WeightedScalar(loss, num_valid_examples)
        )
        self.assertAlmostEqual(
            output_collection.summaries["metric"]["accuracy"],
            WeightedScalar(0.0, num_valid_examples),
        )


if __name__ == "__main__":
    absltest.main()
