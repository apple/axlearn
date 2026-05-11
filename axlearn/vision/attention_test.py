# Copyright © 2023 Apple Inc.

"""Tests for vision attention layers."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from axlearn.common.attention import BaseScaleQK
from axlearn.common.golden import load_golden
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import Tensor
from axlearn.vision.attention import WindowedAttention, WindowedSelfAttentionLayer, get_rel_pos_emb


class ScaleNoOp(BaseScaleQK):
    """Dummy Scale{Query|Key} layer that does not scale the projection."""

    def forward(self, proj: Tensor, *, positions: Optional[Tensor]) -> Tensor:
        """Scales the projected queries."""
        return proj


class RelativePositionTest(TestCase, tf.test.TestCase):
    @parameterized.parameters(
        [
            dict(q_size=14, k_size=14, length=27),
            dict(q_size=64, k_size=64, length=127),
            dict(q_size=7, k_size=7, length=12),
            dict(q_size=10, k_size=6, length=16),
            dict(q_size=20, k_size=20, length=20),
        ]
    )
    def test_get_rel_pos_emb(self, q_size, k_size, length):
        golden = load_golden(
            "axlearn.vision.attention_test",
            f"test_get_rel_pos_emb_q{q_size}_k{k_size}_l{length}",
        )
        inputs = jnp.array(golden["inputs"]["rel_pos"])

        outputs = get_rel_pos_emb(q_size, k_size, inputs)

        ref_outputs = golden["outputs"]["ref"]

        # Tests outputs Tensor shape and value
        self.assertAllEqual(outputs.shape, ref_outputs.shape)
        assert_allclose(outputs, ref_outputs)


class WindowedAttentionTest(TestCase, tf.test.TestCase):
    @parameterized.parameters(
        [
            dict(image_size=224, patch_size=16, use_rel_pos_emb=True),
            dict(image_size=224, patch_size=16, use_rel_pos_emb=False),
            dict(image_size=640, patch_size=16, use_rel_pos_emb=True),
            dict(image_size=640, patch_size=16, use_rel_pos_emb=False),
            dict(image_size=224, patch_size=32, use_rel_pos_emb=True),
            dict(image_size=224, patch_size=32, use_rel_pos_emb=False),
        ]
    )
    def test_windowed_attention_forward(self, image_size, patch_size, use_rel_pos_emb):
        num_heads, model_dim = 4, 8
        batch_size, height, width = 2, image_size // patch_size, image_size // patch_size
        rng = np.random.default_rng(seed=123)
        query = jnp.asarray(rng.random([batch_size, height, width, model_dim]))

        attention_cfg = WindowedAttention.default_config().set(
            name="test",
            input_size=(image_size // patch_size, image_size, patch_size),
            use_rel_pos_emb=use_rel_pos_emb,
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            output_dim=model_dim,
        )
        attention = attention_cfg.instantiate(parent=None)
        attention_state = attention.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )

        outputs, _ = F(
            attention,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=attention_state,
            inputs=dict(query=query, return_aux={"probs"}),
        )
        ref_data_shape = (batch_size, height * width, model_dim)
        ref_probs_shape = (batch_size, num_heads, height * width, height * width)
        self.assertEqual(outputs.data.shape, ref_data_shape)
        self.assertEqual(outputs.probs.shape, ref_probs_shape)


class WindowedAttentionLayerTest(TestCase, tf.test.TestCase):
    @parameterized.parameters(
        [
            dict(target_len=196, window_size=14),
            dict(target_len=1600, window_size=14),
            dict(target_len=196, window_size=0),
            dict(target_len=1600, window_size=0),
        ]
    )
    def test_windowed_attention_layer_forward(self, target_len, window_size):
        batch_size, num_heads, model_dim = 2, 4, 8
        height = width = int(np.sqrt(target_len))
        rng = np.random.default_rng(seed=123)
        target = jnp.asarray(rng.random([batch_size, target_len, model_dim]))
        attention_cfg = WindowedAttention.default_config().set(
            input_size=(height, width),
            use_rel_pos_emb=True,
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            output_dim=model_dim,
        )
        attention_layer_cfg = WindowedSelfAttentionLayer.default_config().set(
            name="test",
            window_size=window_size,
            attention=attention_cfg,
            target_dim=model_dim,
            source_dim=model_dim,
            structure="prenorm",
        )
        attention_layer = attention_layer_cfg.instantiate(parent=None)
        attention_state = attention_layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )

        outputs, _ = F(
            attention_layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=attention_state,
            inputs=dict(target=target),
        )

        ref_data_shape = (batch_size, target_len, model_dim)
        self.assertEqual(outputs.data.shape, ref_data_shape)

        noscale_attention_cfg = attention_cfg.clone(
            query_scale=ScaleNoOp.default_config(), key_scale=ScaleNoOp.default_config()
        )
        noscale_attention_layer_cfg = attention_layer_cfg.clone(attention=noscale_attention_cfg)
        noscale_attention_layer = noscale_attention_layer_cfg.instantiate(parent=None)
        noscale_outputs, _ = F(
            noscale_attention_layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=attention_state,
            inputs=dict(target=target),
        )
        self.assertNotAllClose(outputs.data, noscale_outputs.data)


if __name__ == "__main__":
    absltest.main()
