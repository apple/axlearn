# Copyright © 2023 Apple Inc.

"""Tests multistream model."""
# pylint: disable=no-self-use
import copy

import jax
import numpy as np
from absl.testing import absltest
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.module import functional as F
from axlearn.common.multi_stream_model import FusionNetwork, MultiStreamModel, StreamEncoder
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import NestedTensor, Tensor


class MockImageEncoder(StreamEncoder):
    @config_class
    class Config(StreamEncoder.Config):
        count_encoder: int = 0

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        image = input_batch["image"]
        image_emb = jnp.mean(image, axis=1)
        return {"image_emb": image_emb, "count_encoder": self.config.count_encoder}


class MockFusionNetwork(FusionNetwork):
    @config_class
    class Config(FusionNetwork.Config):
        count_fusion: int = 0

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        image_emb = input_batch["encoder"]["image_emb"]
        loss = jnp.mean(image_emb, axis=1)
        return loss, {"image_emb": image_emb, "count_fusion": self.config.count_fusion}


class MockMultiStreamModel(MultiStreamModel):
    def _overwrite_config(
        self, child_name: str, child_config: BaseLayer.Config
    ) -> BaseLayer.Config:
        if child_name == "encoder":
            child_config.count_encoder = 3
        return child_config


class MultiStreamPipelineTest(TestCase):
    def test_stream_encoder(self):
        batch_size = 2
        num_len = 10
        dim = 12
        encoder_name = "encoder"
        mock_encoder_layer = (
            MockImageEncoder.default_config().set(name=encoder_name).instantiate(parent=None)
        )

        input_batch = {"image": jnp.asarray(np.random.random((batch_size, num_len, dim)))}
        input_batch[encoder_name] = mock_encoder_layer.forward(input_batch)
        assert encoder_name in input_batch
        assert_allclose(
            input_batch[encoder_name]["image_emb"], jnp.mean(input_batch["image"], axis=1)
        )

    def test_fusion_network(self):
        batch_size = 2
        num_len = 10
        dim = 12
        encoder_name = "encoder"
        fusion_name = "fusion"
        mock_encoder_layer = (
            MockImageEncoder.default_config().set(name=encoder_name).instantiate(parent=None)
        )
        mock_fusion_network = (
            MockFusionNetwork.default_config().set(name=fusion_name).instantiate(parent=None)
        )

        input_batch = {
            "input": {"image": jnp.asarray(np.random.random((batch_size, num_len, dim)))}
        }
        losses = {}
        input_batch[encoder_name] = mock_encoder_layer.forward(input_batch["input"])
        losses[fusion_name], input_batch[fusion_name] = mock_fusion_network.forward(input_batch)
        assert fusion_name in input_batch
        assert fusion_name in losses
        assert_allclose(
            losses[fusion_name], jnp.mean(jnp.mean(input_batch["input"]["image"], axis=1), axis=1)
        )

    def test_multi_stream_pipeline(self):
        batch_size = 2
        num_len = 10
        dim = 12
        encoder_name = "encoder"
        fusion_name = "fusion"
        mock_encoder_layer_cfg = MockImageEncoder.default_config()
        mock_fusion_network_cfg = MockFusionNetwork.default_config()
        mock_multi_stream_cfg = MultiStreamModel.default_config().set(
            name="mock_multi_stream_model"
        )
        mock_multi_stream_cfg.stream_encoder = {encoder_name: mock_encoder_layer_cfg}
        mock_multi_stream_cfg.fusion_network = {fusion_name: mock_fusion_network_cfg}
        mock_multi_stream_model = mock_multi_stream_cfg.instantiate(parent=None)

        input_batch = {
            "input": {"image": jnp.asarray(np.random.random((batch_size, num_len, dim)))}
        }
        outputs, _ = F(
            mock_multi_stream_model,
            state={},
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(input_batch=input_batch),
        )
        losses = outputs[0]
        assert_allclose(losses, jnp.mean(jnp.mean(input_batch["input"]["image"], axis=1), axis=1))

    def test_single_stream_forward(self):
        batch_size = 2
        num_len = 10
        dim = 12
        encoder_name = "encoder"
        fusion_name = "fusion"
        mock_encoder_layer_cfg = MockImageEncoder.default_config()
        mock_fusion_network_cfg = MockFusionNetwork.default_config()
        mock_multi_stream_cfg = MultiStreamModel.default_config().set(
            name="mock_multi_stream_model"
        )
        mock_multi_stream_cfg.stream_encoder = {encoder_name: mock_encoder_layer_cfg}
        mock_multi_stream_cfg.fusion_network = {fusion_name: mock_fusion_network_cfg}
        mock_multi_stream_model = mock_multi_stream_cfg.instantiate(parent=None)

        image = jnp.asarray(np.random.random((batch_size, num_len, dim)))

        input_batch = {"input": {"image": image}}
        outputs, _ = F(
            mock_multi_stream_model,
            state={},
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(input_batch=input_batch, encoder_name=encoder_name),
            method="forward_single_stream_encoder",
        )
        assert_allclose(outputs["image_emb"], jnp.mean(image, axis=1))

        with self.assertRaises(ValueError):
            F(
                mock_multi_stream_model,
                state={},
                is_training=True,
                prng_key=jax.random.PRNGKey(456),
                inputs=dict(input_batch=input_batch, encoder_name="abc"),
                method="forward_single_stream_encoder",
            )

    def test_overwrite(self):
        batch_size = 2
        num_len = 10
        dim = 12
        encoder_name = "encoder"
        fusion_name = "fusion"
        mock_encoder_layer_cfg = MockImageEncoder.default_config()
        mock_fusion_network_cfg = MockFusionNetwork.default_config()
        mock_multi_stream_cfg = MockMultiStreamModel.default_config().set(
            name="mock_multi_stream_model"
        )
        mock_multi_stream_cfg.stream_encoder = {encoder_name: mock_encoder_layer_cfg}
        mock_multi_stream_cfg.fusion_network = {fusion_name: mock_fusion_network_cfg}
        mock_multi_stream_model = mock_multi_stream_cfg.instantiate(parent=None)

        input_batch = {
            "input": {"image": jnp.asarray(np.random.random((batch_size, num_len, dim)))}
        }
        outputs, _ = F(
            mock_multi_stream_model,
            state={},
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(input_batch=input_batch),
            method="predict",
        )
        assert_allclose(outputs["encoder"]["count_encoder"], 3)

    def calculate_loss(self, input_batch, loss_weights):
        encoder_name = "encoder"
        fusion_1_name = "fusion_1"
        fusion_2_name = "fusion_2"
        mock_encoder_layer_cfg = MockImageEncoder.default_config()
        mock_fusion_network_cfg = MockFusionNetwork.default_config()
        mock_multi_stream_cfg = MultiStreamModel.default_config().set(
            name="mock_multi_stream_model", loss_weights=loss_weights
        )
        mock_multi_stream_cfg.stream_encoder = {encoder_name: mock_encoder_layer_cfg}
        mock_multi_stream_cfg.fusion_network = {
            fusion_1_name: mock_fusion_network_cfg,
            fusion_2_name: mock_fusion_network_cfg,
        }
        mock_multi_stream_model = mock_multi_stream_cfg.instantiate(parent=None)
        outputs, _ = F(
            mock_multi_stream_model,
            state={},
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(input_batch=input_batch),
        )
        losses = outputs[0]
        return losses

    def test_loss_weight(self):
        batch_size = 2
        num_len = 10
        dim = 12
        input_batch = {
            "input": {"image": jnp.asarray(np.random.random((batch_size, num_len, dim)))}
        }

        # The input_batch is updated during feedforward. Therefore, copy.deepcopy is required.
        loss_0 = self.calculate_loss(copy.deepcopy(input_batch), {"fusion_1": 0, "fusion_2": 1})
        loss_1 = self.calculate_loss(copy.deepcopy(input_batch), {"fusion_1": 1, "fusion_2": 0})
        loss_2 = self.calculate_loss(copy.deepcopy(input_batch), {"fusion_1": 1, "fusion_2": 1})
        assert_allclose(loss_0, loss_1)
        assert_allclose(loss_0 * 2, loss_2)


if __name__ == "__main__":
    absltest.main()
