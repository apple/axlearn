# Copyright © 2023 Apple Inc.

"""Tests blocks for mobile networks."""
# pylint: disable=no-self-use,too-many-lines,too-many-public-methods
import jax.random
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common import utils
from axlearn.common.layers import SqueezeExcitation, StochasticDepth
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.vision.mobilenets_blocks import ConvBnAct, MobileBlock, MobileBlockType


# pylint: disable=too-many-public-methods
class MobileNetsBlocksTest(TestCase, tf.test.TestCase):
    """Tests blocks for mobile networks."""

    @parameterized.named_parameters(
        {
            "testcase_name": "CONV_BN_ACT",
            "block_class": ConvBnAct,
            "block_type": None,
            "se_ratio": 0.0,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.0,
            "num_params": 4736,
        },
        {
            "testcase_name": "DEPTHWISE_SEPARABLE",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.DEPTHWISE_SEPARABLE,
            "se_ratio": 0.0,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.0,
            "num_params": 848,
        },
        {
            "testcase_name": "DEPTHWISE_SEPARABLE_DROP",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.DEPTHWISE_SEPARABLE,
            "se_ratio": 0.5,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.5,
            "num_params": 1128,
        },
        {
            "testcase_name": "DEPTHWISE_SEPARABLE_SE",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.DEPTHWISE_SEPARABLE,
            "se_ratio": 0.5,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.0,
            "num_params": 1128,
        },
        {
            "testcase_name": "INVERTED_BOTTLENECK",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.INVERTED_BOTTLENECK,
            "se_ratio": 0.0,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.0,
            "num_params": 848,
        },
        {
            "testcase_name": "INVERTED_BOTTLENECK_DROP",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.INVERTED_BOTTLENECK,
            "se_ratio": 0.0,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.5,
            "num_params": 848,
        },
        {
            "testcase_name": "INVERTED_BOTTLENECK_SE",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.INVERTED_BOTTLENECK,
            "se_ratio": 0.5,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.0,
            "num_params": 1128,
        },
        {
            "testcase_name": "INVERTED_BOTTLENECK_EXP",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.INVERTED_BOTTLENECK,
            "se_ratio": 0.0,
            "exp_ratio": 2.0,
            "drop_path_rate": 0.0,
            "num_params": 2208,
        },
        {
            "testcase_name": "INVERTED_BOTTLENECK_EXP_SE",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.INVERTED_BOTTLENECK,
            "se_ratio": 0.5,
            "exp_ratio": 2.0,
            "drop_path_rate": 0.0,
            "num_params": 3280,
        },
        {
            "testcase_name": "FUSED_INVERTED_BOTTLENECK",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.FUSED_INVERTED_BOTTLENECK,
            "se_ratio": 0.0,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.0,
            "num_params": 640,
        },
        {
            "testcase_name": "FUSED_INVERTED_BOTTLENECK_DROP",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.FUSED_INVERTED_BOTTLENECK,
            "se_ratio": 0.0,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.5,
            "num_params": 640,
        },
        {
            "testcase_name": "FUSED_INVERTED_BOTTLENECK_SE",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.FUSED_INVERTED_BOTTLENECK,
            "se_ratio": 0.5,
            "exp_ratio": 1.0,
            "drop_path_rate": 0.0,
            "num_params": 920,
        },
        {
            "testcase_name": "FUSED_INVERTED_BOTTLENECK_EXP",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.FUSED_INVERTED_BOTTLENECK,
            "se_ratio": 0.0,
            "exp_ratio": 2.0,
            "drop_path_rate": 0.0,
            "num_params": 5888,
        },
        {
            "testcase_name": "FUSED_INVERTED_BOTTLENECK_EXP_SE",
            "block_class": MobileBlock,
            "block_type": MobileBlockType.FUSED_INVERTED_BOTTLENECK,
            "se_ratio": 0.5,
            "exp_ratio": 2.0,
            "drop_path_rate": 0.0,
            "num_params": 6960,
        },
    )
    def test_mobilenets_blocks(
        self,
        block_class,
        block_type: MobileBlockType,
        se_ratio: float,
        exp_ratio: float,
        drop_path_rate: float,
        num_params: int,
    ):
        # Initialize layer.
        input_dim = 16
        output_dim = 32
        cfg = block_class.default_config().set(
            name="test", input_dim=input_dim, output_dim=output_dim
        )
        if block_type is not None:
            cfg.mobile_block_type = block_type
        if se_ratio > 0.0:
            cfg.se_layer = SqueezeExcitation.default_config().set(se_ratio=se_ratio)
        if drop_path_rate > 0.0:
            cfg.drop_path = StochasticDepth.default_config().set(rate=drop_path_rate)
        if exp_ratio > 1.0:
            cfg.exp_ratio = exp_ratio
        layer: block_class = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        prng_key, input_key = jax.random.split(prng_key)
        # Random inputs.
        batch_size, input_size = 2, 32
        inputs = jax.random.normal(input_key, [batch_size, input_size, input_size, input_dim])
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        self.assertEqual(outputs.shape, (batch_size, input_size, input_size, output_dim))
        self.assertEqual(utils.count_model_params(layer_params), num_params)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
