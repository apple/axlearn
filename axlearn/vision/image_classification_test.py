"""Tests image classification models."""
import difflib

import jax.random
import numpy as np
import torch
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from axlearn.common import utils
from axlearn.common.config import InstantiableConfig
from axlearn.common.module import NestedTensor
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor
from axlearn.common.vision_transformer import build_vit_model_config
from axlearn.vision.image_classification import ImageClassificationModel
from axlearn.vision.resnet import ResNet


def params_from_conv(ref: torch.nn.Module) -> NestedTensor:
    return {
        "weight": ref.weight.permute(2, 3, 1, 0),
    }


def params_from_linear(ref: torch.nn.Module) -> NestedTensor:
    return {
        "weight": ref.weight.transpose(1, 0),
        "bias": ref.bias,
    }


def params_from_bn(ref: torch.nn.Module) -> NestedTensor:
    return {
        "scale": ref.weight,
        "bias": ref.bias,
        "moving_mean": ref.running_mean,
        "moving_variance": ref.running_var,
    }


def params_from_downsample(ref: torch.nn.ModuleList) -> NestedTensor:
    return {
        "conv": params_from_conv(ref[0]),
        "norm": params_from_bn(ref[1]),
    }


def params_from_block(ref: torch.nn.Module) -> NestedTensor:
    params = {
        "conv1": params_from_conv(ref.conv1),
        "norm1": params_from_bn(ref.bn1),
        "conv2": params_from_conv(ref.conv2),
        "norm2": params_from_bn(ref.bn2),
    }
    if hasattr(ref, "conv3"):
        params["conv3"] = params_from_conv(ref.conv3)
    if hasattr(ref, "bn3"):
        params["norm3"] = params_from_bn(ref.bn3)
    if getattr(ref, "downsample"):
        params["downsample"] = params_from_downsample(ref.downsample)
    return params


def params_from_stage(ref: torch.nn.ModuleList) -> NestedTensor:
    return {f"block{i}": params_from_block(block) for i, block in enumerate(ref)}


def params_from_stem(ref: torch.nn.ModuleList) -> NestedTensor:
    return {
        "conv1": params_from_conv(ref.conv1),
        "norm1": params_from_bn(ref.bn1),
    }


def params_from_backbone(ref: torch.nn.ModuleList) -> NestedTensor:
    return {
        "stem": params_from_stem(ref),
        **{f"stage{i}": params_from_stage(getattr(ref, f"layer{i + 1}")) for i in range(4)},
    }


def params_from_resnet(ref: torch.nn.Module) -> NestedTensor:
    return {
        "backbone": params_from_backbone(ref),
        "classifier": params_from_linear(ref.fc),
    }


# pylint: disable=no-self-use
class ResNetClassificationModelTest(parameterized.TestCase):
    def _base_test(self, cfg: InstantiableConfig, ref: torch.nn.Module, is_training=False):
        model = cfg.set(name="test", num_classes=1000).instantiate(parent=None)
        init_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))

        num_params = utils.count_model_params(init_params)
        logging.info("Model contains %s Million parameters.", num_params / 10.0**6)

        param_spec_strs = [
            f"{name}={list(param.shape)}" for name, param in utils.flatten_items(init_params)
        ]
        for name, param in ref.named_parameters():
            logging.info("ref param: %s=%s", name, list(param.shape))

        model_params = jax.tree_util.tree_map(utils.as_tensor, params_from_resnet(ref))
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
            inputs=dict(input_batch=jax.tree_util.tree_map(jnp.asarray, inputs)),
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
        self._base_test(cfg, torch_model(pretrained=True), is_training=is_training)


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
            output_collection.summaries["metric"]["loss"], (loss, num_valid_examples)
        )
        self.assertAlmostEqual(
            output_collection.summaries["metric"]["accuracy"],
            (0.0, num_valid_examples),
        )


if __name__ == "__main__":
    absltest.main()
