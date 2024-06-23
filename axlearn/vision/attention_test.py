# Copyright © 2023 Apple Inc.

"""Tests for vision attention layers."""
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from absl.testing import absltest, parameterized

from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.vision.attention import WindowedAttention, WindowedSelfAttentionLayer, get_rel_pos_emb


def get_rel_pos_torch(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = torch.nn.functional.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


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
        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        dim = 64
        inputs = jax.random.normal(input_key, [length, dim])

        outputs = get_rel_pos_emb(q_size, k_size, inputs)

        # Compute torch ref outputs
        torch_inputs = as_torch_tensor(inputs)
        ref_outputs = get_rel_pos_torch(q_size, k_size, torch_inputs)
        ref_outputs = ref_outputs.detach().numpy()

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


if __name__ == "__main__":
    absltest.main()
