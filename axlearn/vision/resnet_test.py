# Copyright © 2023 Apple Inc.

"""Tests ResNet."""
import jax.random
import numpy as np
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import utils
from axlearn.common.config import InstantiableConfig
from axlearn.common.module import functional as F
from axlearn.vision import resnet
from axlearn.vision.resnet import ResNet


def get_resnet_cfg_model(model_depth):
    if model_depth == 18:
        cfg = ResNet.resnet18_config()
    elif model_depth == 34:
        cfg = ResNet.resnet34_config()
    elif model_depth == 50:
        cfg = ResNet.resnet50_config()
    elif model_depth == 101:
        cfg = ResNet.resnet101_config()
    elif model_depth == 152:
        cfg = ResNet.resnet152_config()
    elif model_depth == 200:
        cfg = ResNet.resnet200_config()
    elif model_depth == 270:
        cfg = ResNet.resnet270_config()
    elif model_depth == 350:
        cfg = ResNet.resnet350_config()
    elif model_depth == 420:
        cfg = ResNet.resnet420_config()
    else:
        raise ValueError(f"ResNet model depth {model_depth} not implemented.")
    return cfg


# pylint: disable=no-self-use
class ResNetTest(parameterized.TestCase):
    """Tests ResNet."""

    def _base_test(self, cfg: InstantiableConfig, is_training=False):
        model: resnet.ResNet = cfg.set(name="test").instantiate(parent=None)
        init_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)
        outputs, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=init_params,
            inputs=dict(image=jax.tree.map(jnp.asarray, inputs)),
        )
        for level in range(2, 6):
            self.assertEqual(outputs[str(level)].shape[-1], model.endpoints_dims[str(level)])
        self.assertEqual(outputs["embedding"].shape[-1], model.endpoints_dims["embedding"])

    @parameterized.product(
        model_depth=(18, 34, 50, 101, 152),
        is_training=(False, True),
    )
    def test_resnets(self, model_depth, is_training):
        cfg = get_resnet_cfg_model(model_depth)
        self._base_test(cfg, is_training=is_training)

    def _test_forward_pass(self, cfg: InstantiableConfig, is_training=False):
        model: resnet.ResNet = cfg.set(name="test").instantiate(parent=None)
        init_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)

        F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=init_params,
            inputs=dict(image=jax.tree.map(jnp.asarray, inputs)),
        )
        num_params = utils.count_model_params(init_params)
        logging.info("Model contains %s Million parameters.", num_params / 10.0**6)

    @parameterized.product(
        model_depth=(18, 50),
        is_training=(False, True),
        peak_rate=(None, 0.2),
    )
    def test_resnet_features(self, model_depth, is_training, peak_rate):
        cfg = get_resnet_cfg_model(model_depth)
        cfg.peak_stochastic_depth_rate = peak_rate
        self._test_forward_pass(cfg, is_training=is_training)

    @parameterized.product(
        is_training=(False, True), model_depth=(50, 101, 152, 200, 270, 350, 420)
    )
    def test_resnet_rs_architectures(self, is_training, model_depth):
        cfg = get_resnet_cfg_model(model_depth=model_depth)
        cfg.stem = resnet.StemV1.default_config()
        cfg.stage.block.downsample.downsample_op = "maxpool"
        cfg.stage.block.squeeze_excitation.se_ratio = 0.25
        cfg.peak_stochastic_depth_rate = 0.2
        cfg.stage.block.activation = "nn.swish"
        self._test_forward_pass(cfg, is_training=is_training)


if __name__ == "__main__":
    absltest.main()
