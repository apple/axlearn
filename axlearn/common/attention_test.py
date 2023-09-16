# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# ofirpress/attention_with_linear_biases:
# Copyright (c) Facebook, Inc. and its affiliates.

"""Tests attention layers."""
# pylint: disable=too-many-lines,duplicate-code,no-self-use
import math
from itertools import combinations
from typing import List, Optional

import jax
import numpy as np
import optax
import pytest
import torch
from absl import logging
from absl.testing import absltest, parameterized
from jax import nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_policies as jax_remat_policies
from transformers.models.roberta import modeling_roberta as hf_roberta
from transformers.models.roformer import modeling_roformer as hf_roformer
from transformers.models.xlnet import modeling_xlnet as hf_xlnet

from axlearn.common import attention, utils
from axlearn.common.attention import (
    NEG_INF,
    BaseStackedTransformerLayer,
    BottleNeckAdapterTransformerLayer,
    FusedQKVLinear,
    LearnedPositionalEmbedding,
    MultiheadAttentionXL,
    MultiheadInputLinear,
    MultiheadOutputLinear,
    MultiheadRelativePositionLinear,
    PerDimScale,
    PipelinedTransformerLayer,
    QKVLinear,
    RepeatedTransformerLayer,
    RoFormerQKVLinear,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    TransformerLayer,
    _next_power_of_two,
    apply_attention_logit_biases,
    apply_rotary_position_embeddings,
    build_remat_spec,
    compute_padding_biases,
    make_causal_mask,
    rel_pos_to_abs_pos,
    scaled_hidden_dim,
    set_double_shard_weights_config,
    sinusoidal_positional_embeddings,
    xl_attention_logits,
)
from axlearn.common.base_layer import BaseLayer, FactorizationSpec, ParameterSpec, RematSpec
from axlearn.common.config import config_class
from axlearn.common.module import InvocationContext, Module
from axlearn.common.module import functional as F
from axlearn.common.module import new_output_collection, set_current_context
from axlearn.common.optimizer_base import OptParam
from axlearn.common.optimizers import adafactor_optimizer
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    DefaultInitializer,
    FanAxes,
    WeightInitializer,
)
from axlearn.common.test_utils import TestCase, assert_allclose, dummy_segments_positions
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import PartitionSpec, Tensor, as_tensor, flatten_items, shapes


def _random_mask(prng_key, tgt_len, src_len):
    """Returns a float mask of shape [tgt_len, src_len]."""
    key1, key2 = jax.random.split(prng_key)
    mask = jnp.logical_not(
        jax.random.randint(key1, minval=0, maxval=2, shape=[tgt_len, src_len])
        +
        # Ensure that every tgt position attends to at least one src position, otherwise
        # torch_modules.MultiheadAttention will generate NaN.
        nn.one_hot(jax.random.randint(key2, minval=0, maxval=src_len, shape=[tgt_len]), src_len)
    )
    return mask.astype(jnp.float32) * NEG_INF


class MaskTest(absltest.TestCase):
    """Tests mask implementations."""

    def test_causal_mask(self):
        expected = jnp.array([[0.0, NEG_INF, NEG_INF], [0.0, 0.0, NEG_INF], [0.0, 0.0, 0.0]])
        actual = attention.make_causal_mask(3)
        self.assertTrue(jnp.all(actual <= expected))

    def test_segment_mask(self):
        expected = jnp.array(
            [  # batch
                [  # num_heads
                    [
                        [NEG_INF, NEG_INF, NEG_INF, 0.0],
                        [NEG_INF, NEG_INF, NEG_INF, 0.0],
                        [0.0, 0.0, NEG_INF, NEG_INF],
                        [NEG_INF, NEG_INF, 0.0, NEG_INF],
                    ]
                ]
            ]
        )
        actual = attention.make_segment_mask(
            target_segments=jnp.asarray([[1, 1, 2, 0]]),
            source_segments=jnp.asarray([[2, 2, 0, 1]]),
        )
        self.assertTrue(jnp.all(actual <= expected))

    def test_apply_attention_logit_biases(self):
        batch_size = 10
        num_heads = 12
        dim = 32
        logits = jnp.asarray(np.random.random(size=[batch_size, num_heads, dim]))

        # Tesing for biases = None
        masked_logit = apply_attention_logit_biases(logits, attention_logit_biases=None)
        self.assertEqual(masked_logit.dtype, logits.dtype)
        np.testing.assert_array_equal(logits, masked_logit)

        # Testing for biases = random_float_biases
        for logit_float_type in [jnp.bfloat16, jnp.float32, jnp.float16]:
            for mask_float_type in [jnp.bfloat16, jnp.float32, jnp.float16]:
                logits = jnp.asarray(np.random.random(size=[batch_size, num_heads, dim])).astype(
                    logit_float_type
                )
                random_float_biases = jnp.asarray(
                    np.random.random(size=[batch_size, num_heads, dim])
                ).astype(mask_float_type)
                masked_logit = apply_attention_logit_biases(
                    logits, attention_logit_biases=random_float_biases
                )
                self.assertEqual(masked_logit.dtype, logits.dtype)
                np.testing.assert_array_equal(
                    masked_logit, logits + random_float_biases.astype(logits.dtype)
                )


class CausalAttentionLogitBiasLayerTest(TestCase):
    """Tests CausalAttentionLogitBiasLayer."""

    @parameterized.parameters(
        # Test the mask with all padding tokens.
        dict(
            token_ids=[[0, 0, 0], [0, 0, 0]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [0, 0, 0],
                ],
            ]
            * 2,
        ),
        # Test the mask with all valid tokens.
        dict(
            token_ids=[[1, 2, 3], [4, 5, 6]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [0, 0, 0],
                ],
            ]
            * 2,
        ),
        # Test the mask with some valid tokens and some padding tokens.
        dict(
            token_ids=[[10, 0, 0], [12, 33, 0]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [0, 0, 0],
                ],
            ]
            * 2,
        ),
        # Test the mask with additional padding biases.
        dict(
            token_ids=[[10, 0, 0], [12, 33, 0]],
            apply_padding_mask=True,
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
            ],
        ),
        # Test the mask with valid tokens AND paddings in between.
        dict(
            token_ids=[[10, 0, 11], [12, 33, 0]],
            apply_padding_mask=True,
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                    [0, NEG_INF, 0],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
            ],
        ),
        # Test a basic case with positions.
        dict(
            token_ids=[[10, 11, 12], [13, 14, 15]],
            segment_ids=[[1, 1, 2], [1, 2, 2]],
            positions=[[0, 1, 0], [0, 0, 1]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, 0],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, 0, NEG_INF],
                    [NEG_INF, 0, 0],
                ],
            ],
        ),
        # Test a case where some segments are empty.
        dict(
            token_ids=[[10, 11, 12], [13, 14, 15]],
            segment_ids=[[1, 2, 2], [2, 2, 2]],
            positions=[[0, 0, 1], [0, 1, 2]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, 0, NEG_INF],
                    [NEG_INF, 0, 0],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [0, 0, 0],
                ],
            ],
        ),
        # Test with positions and padding.
        # Note: we deliberately allow the last token to be 0, to test that without
        # apply_padding_mask, a 0-token is not necessarily padding if its segment_id != 0.
        dict(
            token_ids=[[10, 11, 0], [13, 14, 0]],
            segment_ids=[[1, 1, 0], [1, 2, 2]],
            positions=[[0, 1, 0], [0, 0, 1]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, 0],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, 0, NEG_INF],
                    [NEG_INF, 0, 0],
                ],
            ],
        ),
        # Test with segment IDs but not positions.
        # This should have the same result as the previous test.
        dict(
            token_ids=[[10, 11, 0], [13, 14, 0]],
            segment_ids=[[1, 1, 0], [1, 2, 2]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, 0],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, 0, NEG_INF],
                    [NEG_INF, 0, 0],
                ],
            ],
        ),
        # Test with positions and padding, and apply the padding mask.
        dict(
            token_ids=[[10, 11, 0], [13, 14, 0]],
            segment_ids=[[1, 1, 0], [1, 2, 0]],
            positions=[[0, 1, 0], [0, 0, 1]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, 0, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
            ],
            apply_padding_mask=True,
        ),
    )
    def test_causal_attention_mask_layer(
        self,
        token_ids: List,
        expected: List,
        segment_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        apply_padding_mask: Optional[bool] = False,
    ):
        causal_attention_mask_layer = (
            attention.CausalAttentionLogitBiasLayer.default_config()
            .set(name="test_causal_attention_mask")
            .instantiate(parent=None)
        )
        if token_ids is not None:
            token_ids = np.asarray(token_ids)
        if positions is None:
            positions = np.arange(token_ids.shape[-1])[None, :]
        else:
            positions = np.asarray(positions)
        if segment_ids is None:
            segment_ids = np.ones_like(token_ids)
        else:
            segment_ids = np.asarray(segment_ids)
        actual = causal_attention_mask_layer.forward(segment_ids=segment_ids, positions=positions)
        if apply_padding_mask:
            actual += compute_padding_biases(token_ids, pad_token_id=0)
        assert_allclose(jnp.exp(actual.squeeze(1)), jnp.exp(np.asarray(expected)))


class FullAttentionLogitBiasLayerTest(TestCase):
    """Tests FullAttentionLogitBiasLayer."""

    @parameterized.parameters(
        # Test the mask with all padding tokens.
        dict(
            token_ids=[[0, 0, 0], [0, 0, 0]],
            expected=[
                [
                    [NEG_INF, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
                [
                    [NEG_INF, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
            ],
        ),
        # Test the mask with all valid tokens.
        dict(
            token_ids=[[1, 2, 3], [4, 5, 6]],
            expected=[
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        ),
        # Test the mask with some valid tokens and some padding tokens.
        dict(
            token_ids=[[10, 0, 0], [12, 33, 0]],
            expected=[
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
                [
                    [0, 0, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
            ],
        ),
        # Test a basic case with segment IDs.
        dict(
            token_ids=[[10, 11, 12], [13, 14, 15]],
            segment_ids=[[1, 1, 2], [1, 2, 2]],
            positions=[[0, 1, 0], [0, 0, 1]],
            expected=[
                [
                    [0, 0, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, 0],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, 0, 0],
                    [NEG_INF, 0, 0],
                ],
            ],
        ),
        # Test a case where some segments are empty.
        dict(
            token_ids=[[10, 11, 12], [13, 14, 15]],
            segment_ids=[[1, 1, 2], [2, 2, 2]],
            positions=[[0, 1, 0], [0, 1, 2]],
            expected=[
                [
                    [0, 0, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ],
        ),
        # Test with segment IDs and padding.
        dict(
            token_ids=[[10, 11, 0], [13, 14, 0]],
            segment_ids=[[1, 1, 0], [1, 2, 0]],
            positions=[[0, 1, 0], [0, 0, 1]],
            expected=[
                [
                    [0, 0, NEG_INF],
                    [0, 0, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
                [
                    [0, NEG_INF, NEG_INF],
                    [NEG_INF, 0, NEG_INF],
                    [NEG_INF, NEG_INF, NEG_INF],
                ],
            ],
        ),
    )
    def test_full_attention_mask_layer(
        self,
        token_ids: List,
        expected: List,
        segment_ids: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ):
        full_attention_mask_layer = (
            attention.FullAttentionLogitBiasLayer.default_config()
            .set(name="test_full_attention_mask")
            .instantiate(parent=None)
        )
        if token_ids is not None:
            token_ids = np.asarray(token_ids)
        if positions is None:
            positions = np.arange(token_ids.shape[-1])[None, :]
        else:
            positions = np.asarray(positions)
        if segment_ids is None:
            segment_ids = token_ids != 0
        else:
            segment_ids = np.asarray(segment_ids)
        actual = full_attention_mask_layer.forward(segment_ids=segment_ids, positions=positions)
        actual += compute_padding_biases(token_ids, pad_token_id=0)
        assert_allclose(jnp.exp(np.asarray(expected)), jnp.exp(actual.squeeze(1)))


class ALiBiAttentionLogitBiasLayerTest(TestCase):
    """Tests ALiBiAttentionLogitBiasLayer."""

    def ref_alibi_implementation(self, batch_size, num_heads, max_len):
        # Slopes is in jax DeviceArray. Switch it to torch tensor as the ref code.
        slopes = torch.Tensor(attention.alibi_get_slopes(num_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_len).unsqueeze(0).unsqueeze(
            0
        ).expand(num_heads, -1, -1)
        alibi = alibi.view(num_heads, 1, max_len)

        # Post processing to translate alibi matrix into the jax format.
        # Alibi matrix shape [batch_size, num_heads, max_len, max_len].
        alibi = alibi.unsqueeze(0).expand(batch_size, -1, max_len, -1)
        # Translate from pytorch to jax.
        alibi = as_tensor(alibi)
        return alibi

    def test_alibi_attention_mask(self):
        num_heads = 12
        batch_size = 2
        max_len = 3

        # Test alibi implementation.
        alibi_attention_mask_layer = (
            attention.ALiBiAttentionLogitBiasLayer.default_config()
            .set(name="test_alibi_attention_mask", num_heads=num_heads)
            .instantiate(parent=None)
        )

        # Casual attention mask which will be applied to ref alibi mask.
        ref_causal_attention_mask_layer = (
            attention.CausalAttentionLogitBiasLayer.default_config()
            .set(name="ref_causal_attention_mask")
            .instantiate(parent=None)
        )

        token_ids = as_tensor(np.random.randint(low=1, high=20, size=[batch_size, max_len]))
        segment_ids = jnp.ones_like(token_ids)
        positions = jnp.arange(max_len)[None, :]

        ref_alibi_mask = self.ref_alibi_implementation(batch_size, num_heads, max_len)
        # Reshape causal_mask to [batch_size, num_heads, max_len, max_len].
        ref_causal_mask = ref_causal_attention_mask_layer.forward(
            segment_ids=segment_ids, positions=positions
        )
        ref_causal_mask = jnp.repeat(ref_causal_mask, num_heads, axis=1)

        # Prepare the ref and the test alibi mask.
        ref_alibi_mask = attention.apply_attention_logit_biases(ref_alibi_mask, ref_causal_mask)
        test_alibi_mask = alibi_attention_mask_layer.forward(
            segment_ids=segment_ids, positions=positions
        )

        # Ref and test alibi mask should be the same after applying it into a QK attention matrix.
        # e.g. softmax(QK + ref_alibi_mask) == softmax(QK + test_alibi_mask).
        random_qk_matrix = jnp.asarray(
            np.random.random(size=[batch_size, num_heads, max_len, max_len])
        )

        ref_alibi_softmax = jax.nn.softmax(random_qk_matrix + ref_alibi_mask, axis=-1)
        test_alibi_softmax = jax.nn.softmax(random_qk_matrix + test_alibi_mask, axis=-1)

        # The ref alibi implementation relies on the softmax property of invariance to translation.
        # e.g. ref_alibi = [[0, -inf, -inf], [0, 1, -inf], [0, 1, 2]]
        # test_alibi = [[0, -inf, -inf], [-1, 0, -inf], [-2, -1, 0]]
        # softmax(qk + test_alibi) = softmax (qk + [[0, -inf, -inf], [-1, 0, -inf], [-2, -1, 0]])
        #                          = softmax (qk + [[0, -inf, -inf], [0, 1, -inf+1], [0, 1, 2]])
        # As the numerical -inf is not perfect -inf defined in math.
        # Therefore, a very limit difference between those two after softmax, due to (-inf + x).
        # The rtol is set to 5e-7 to tolerate this difference.
        np.testing.assert_allclose(ref_alibi_softmax, test_alibi_softmax, rtol=5e-07)

    @parameterized.product(
        [
            dict(num_segments=1, max_len=3),
            dict(num_segments=3, max_len=3),
            dict(num_segments=3, max_len=8),
        ],
    )
    def test_packing(self, max_len: int, num_segments: int):
        # With packed inputs of shape [batch, seq_len], we form a block-diagonal matrix of shape
        # [batch, num_heads, seq_len, seq_len], where each (unpacked) input has blocks of shape
        # [batch, num_heads, segment_len, segment_len] (segment_len <= seq_len).
        # We test this by comparing each block against a freshly computed alibi mask of the same
        # shape, ensuring that packing is equivalent to treating each unpacked input separately.
        num_heads = 12
        batch_size = 2

        # Test alibi implementation.
        alibi_attention_mask_layer = (
            attention.ALiBiAttentionLogitBiasLayer.default_config()
            .set(name="test_alibi_attention_mask", num_heads=num_heads)
            .instantiate(parent=None)
        )

        # Construct inputs of shape [batch_size, max_len].
        input_segment_ids, positions = dummy_segments_positions(
            batch_size, max_len, num_segments=num_segments
        )

        # Compute the test alibi mask of shape [batch, num_heads, seq_len, seq_len].
        test_alibi_batch = alibi_attention_mask_layer.forward(
            segment_ids=input_segment_ids, positions=positions
        )
        # Apply segment mask and softmax (see notes above).
        test_alibi_batch = jax.nn.softmax(test_alibi_batch, axis=-1)

        for batch in range(batch_size):
            test_alibi = test_alibi_batch[batch]
            input_segments = input_segment_ids[batch]

            # Compute the reference alibi mask(s) for each segment separately.
            for segment in range(num_segments):
                # [seq_len].
                segment_mask = input_segments == segment
                segment_len = int(jnp.sum(segment_mask, dtype=jnp.int32))

                # Skip the segment if empty.
                if segment_len == 0:
                    continue

                # Select the submatrix in test_alibi corresponding to the current segment.
                # [seq_len, seq_len].
                segment_mask = jnp.logical_and(segment_mask[:, None], segment_mask[None, :])
                # [num_heads, seq_len, seq_len].
                segment_mask = jnp.repeat(segment_mask[None, ...], num_heads, 0)
                # [num_heads, segment_len, segment_len].
                test_alibi_segment = test_alibi[segment_mask.astype(jnp.bool_)].reshape(
                    (num_heads, segment_len, segment_len)
                )

                # Construct the ref_alibi for the current segment.
                # [num_heads, segment_len].
                ref_alibi = self.ref_alibi_implementation(1, num_heads, segment_len).squeeze(0)
                ref_causal_mask = jnp.repeat(make_causal_mask(segment_len)[None, ...], num_heads, 0)
                ref_alibi = attention.apply_attention_logit_biases(ref_alibi, ref_causal_mask)
                ref_alibi = jax.nn.softmax(ref_alibi, axis=-1)

                np.testing.assert_allclose(ref_alibi, test_alibi_segment, rtol=5e-07)


class SymmetricALiBiAttentionLogitBiasLayerTest(TestCase):
    """Tests SymmetricALiBiAttentionLogitBiasLayer."""

    def test_alibi_attention_mask(self):
        num_heads = 8
        batch_size = 2
        max_len = 3

        # Test alibi implementation.
        alibi_attention_mask_layer = (
            attention.SymmetricALiBiAttentionLogitBiasLayer.default_config()
            .set(name="test_symmetric_alibi_attention_mask", num_heads=num_heads)
            .instantiate(parent=None)
        )

        # [num_heads]
        slopes = jnp.array(attention.alibi_get_slopes(num_heads))

        # [max_len, max_len]
        base_alibi_mask = jnp.array(
            [
                [0, -1, -2],
                [-1, 0, -1],
                [-2, -1, 0],
            ],
            dtype=jnp.float32,
        )

        # [heads, max_len, max_len]
        expected_logits_bias = slopes[:, jnp.newaxis, jnp.newaxis] * base_alibi_mask
        # [batch, heads, max_len, max_len]
        expected_logits_bias = expected_logits_bias[jnp.newaxis, ...].repeat(batch_size, axis=0)

        segment_ids = jnp.ones((batch_size, max_len))
        positions = jnp.arange(max_len)[None, :]
        actual_logits_bias = alibi_attention_mask_layer(
            segment_ids=segment_ids, positions=positions
        )

        assert_allclose(actual_logits_bias, expected_logits_bias)


class RoFormerSinusoidalPositionalEmbeddingTest(TestCase):
    """Tests RoFormerSinusoidalPositionalEmbedding."""

    @parameterized.parameters(
        (2, 3, 10, 32),
        (2, 3, 8, 32),
        (2, 4, 6, 32),
        (2, 4, 8, 16),
        (2, 5, 8, 48),
        (2, 5, 8, 64),
    )
    def test_apply_rotary_position_embeddings(self, batch_size, num_heads, max_len, dim):
        # Unittest against the apply_rotary_position_embeddings in HF.
        token_ids = np.random.randint(low=1, high=20, size=[batch_size, max_len])
        sinusoidal_pos_layer = hf_roformer.RoFormerSinusoidalPositionalEmbedding(max_len, dim)
        sinusoidal_pos = sinusoidal_pos_layer(as_torch_tensor(token_ids).shape)[None, None, :, :]
        query = np.random.random([batch_size, num_heads, max_len, dim])
        key = np.random.random([batch_size, num_heads, max_len, dim])
        value = np.random.random([batch_size, num_heads, max_len, dim])
        ref_layer = hf_roformer.RoFormerSelfAttention.apply_rotary_position_embeddings
        test_layer = apply_rotary_position_embeddings
        ref_q_proj, ref_k_proj, ref_v_proj = ref_layer(
            sinusoidal_pos, as_torch_tensor(query), as_torch_tensor(key), as_torch_tensor(value)
        )
        kwargs = {
            "sinusoidal_pos": as_tensor(sinusoidal_pos),
            "query": query,
            "key": key,
            "value": value,
        }
        test_q_proj, test_k_proj, test_v_proj = test_layer(**kwargs)
        np.testing.assert_allclose(test_q_proj, ref_q_proj, atol=5e-7)
        np.testing.assert_allclose(test_k_proj, ref_k_proj, atol=5e-7)
        np.testing.assert_allclose(test_v_proj, ref_v_proj, atol=5e-7)

    @parameterized.parameters(
        (2, 10, 32),
        (2, 8, 32),
        (2, 6, 32),
        (2, 8, 16),
        (2, 8, 48),
        (2, 8, 64),
    )
    def test_rope_emb(self, batch_size, max_len, dim):
        # Token id is in the np format for easier transition.
        token_ids = np.random.randint(low=1, high=20, size=[batch_size, max_len])
        positions = jnp.arange(token_ids.shape[-1], dtype=jnp.int32)
        ref_layer = hf_roformer.RoFormerSinusoidalPositionalEmbedding(max_len, dim)
        ref_output = ref_layer(as_torch_tensor(token_ids).shape)
        # Set up the RoPE AXLearn configs.
        test_layer = (
            attention.RoFormerSinusoidalPositionalEmbedding.default_config()
            .set(name="test_rope_emb", max_len=max_len, dim=dim)
            .instantiate(parent=None)
        )
        test_output = test_layer.forward(positions)
        np.testing.assert_allclose(ref_output, test_output, atol=5e-7)

    def test_rope_emb_out_of_seq(self):
        # Unittest for sequence length > max_len.
        max_len = 8
        seq_len = max_len * 2
        dim = 32
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        test_layer = (
            attention.RoFormerSinusoidalPositionalEmbedding.default_config()
            .set(name="test_rope_emb", max_len=max_len, dim=dim)
            .instantiate(parent=None)
        )
        with self.assertRaises(Exception) as context:
            test_layer.forward(positions)
        self.assertTrue(
            ValueError(f"Seq. length ({seq_len}) should be less than max length ({max_len})"),
            context.exception,
        )

    def _compare_against_roformer_attention(
        self,
        ref,
        layer,
        tgt_len,
        batch_size,
        ref_rope_emb,
    ):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_param_shapes = jax.tree_util.tree_map(lambda x: x.shape, layer_params)
        print(f"layer state={layer_param_shapes}")
        layer_params = parameters_from_torch_layer(ref)
        model_dim, num_heads = layer.config.target_dim, layer.config.attention.num_heads
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        null_mask = jnp.zeros([tgt_len, tgt_len])
        rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)

        for mask in (None, null_mask, rand_mask):
            if mask is not None:
                mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
            layer_outputs, _ = F(
                layer,
                inputs=dict(target=jnp.asarray(target), attention_logit_biases=mask),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            attn_mask = None if mask is None else as_torch_tensor(mask)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                sinusoidal_pos=ref_rope_emb,
                output_attentions=False,
            )
            assert_allclose(layer_outputs.data, as_tensor(ref_outputs))

    def test_rope_self_attention(self):
        model_dim = 32
        num_heads = 4
        max_sequence_length = 12
        batch_size = 2
        rope_mha_cfg = attention.MultiheadAttention.default_config().set(
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            input_linear=RoFormerQKVLinear.default_config().set(max_seq_length=max_sequence_length),
        )
        rope_emb_layer = (
            attention.RoFormerSinusoidalPositionalEmbedding.default_config()
            .set(name="test_rope_emb", max_len=max_sequence_length, dim=model_dim // num_heads)
            .instantiate(parent=None)
        )
        ref_rope_emb = as_torch_tensor(rope_emb_layer.forward(jnp.arange(max_sequence_length)))
        layer = attention.TransformerAttentionLayer.default_config().set(
            source_dim=model_dim,
            target_dim=model_dim,
            name="rope_trans_attn",
            attention=rope_mha_cfg,
            structure="postnorm",
        )
        layer = layer.instantiate(parent=None)
        roformer_config = hf_roformer.RoFormerConfig(
            hidden_size=model_dim,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0,
            rotary_value=True,
        )
        print(f"roformer_config={roformer_config}")
        ref = hf_roformer.RoFormerAttention(roformer_config)
        self._compare_against_roformer_attention(
            ref, layer, max_sequence_length, batch_size, ref_rope_emb
        )


class MultiheadLinearInitTest(TestCase):
    """Tests MultiheadLinear initialization."""

    @parameterized.parameters(
        (
            MultiheadInputLinear,
            FanAxes(in_axis=0, out_axis=(1, 2)),
            {
                "fan_in": 4,
                "fan_out": 8 * 6,
                "fan_avg": (4 + 8 * 6) / 2,
            },
        ),
        (
            MultiheadOutputLinear,
            FanAxes(in_axis=(1, 2), out_axis=0),
            {
                "fan_in": 8 * 6,
                "fan_out": 4,
                "fan_avg": (8 * 6 + 4) / 2,
            },
        ),
        (
            MultiheadRelativePositionLinear,
            FanAxes(in_axis=0, out_axis=(1, 2)),
            {
                "fan_in": 4,
                "fan_out": 8 * 6,
                "fan_avg": (4 + 8 * 6) / 2,
            },
        ),
    )
    def test_compute_fan_axes(self, cls, fan_axes, fans):
        for dist in ("uniform", "normal", "truncated_normal"):
            for scale in (1.0, 2.0):
                for fan_type in ("fan_in", "fan_out", "fan_avg"):
                    cfg = cls.default_config().set(
                        name="test", model_dim=4, num_heads=8, per_head_dim=6
                    )
                    cfg.param_init = DefaultInitializer.default_config().set(
                        init_by_param_name={
                            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                                fan=fan_type, scale=scale, distribution=dist
                            )
                        }
                    )
                    layer: BaseLayer = cfg.instantiate(parent=None)
                    # pylint: disable-next=protected-access
                    param_spec_map = layer._create_layer_parameter_specs()
                    self.assertEqual(
                        # pylint: disable-next=protected-access
                        layer._compute_fan_axes("weight", param_spec_map["weight"]),
                        fan_axes,
                    )
                    layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(1))
                    weight = layer_params["weight"]
                    self.assertEqual(weight.dtype, jnp.float32)
                    fan = fans[fan_type]
                    expected_std = scale / math.sqrt(fan)
                    actual_std = np.std(weight)
                    self.assertBetween(actual_std, expected_std / 1.5, expected_std * 1.5)


class QKVLinearTest(TestCase):
    """Tests QKVLinear."""

    def test_qkv_fused_equality(self):
        with utils.numeric_checks(True):
            model_dim = 12
            num_heads = 4
            per_head_dim = model_dim // num_heads
            layer_kwargs = dict(
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                num_heads=num_heads,
                per_head_dim=per_head_dim,
            )
            qkv_linear = (
                attention.QKVLinear.default_config()
                .set(
                    name="qkv_test",
                    **layer_kwargs,
                )
                .instantiate(parent=None)
            )
            qkv_linear_state = qkv_linear.initialize_parameters_recursively(jax.random.PRNGKey(0))
            fused_qkv_linear = (
                attention.FusedQKVLinear.default_config()
                .set(
                    name="fused_qkv_test",
                    **layer_kwargs,
                )
                .instantiate(parent=None)
            )

            def fused_state_from(state):
                weight = jnp.array([state[el]["weight"] for el in ("q_proj", "k_proj", "v_proj")])
                bias = jnp.array([state[el]["bias"] for el in ("q_proj", "k_proj", "v_proj")])
                return {"qkv_proj": dict(weight=weight, bias=bias)}

            # Map state to fused version.
            fused_qkv_linear_state = fused_state_from(qkv_linear_state)

            batch_size, src_len, tgt_len = 2, 6, 6
            rng = np.random.default_rng(seed=123)
            query = jnp.asarray(rng.random([batch_size, tgt_len, model_dim]))
            key = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
            value = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
            inputs = dict(query=query, key=key, value=value)

            outputs = {}
            layer_names = ("qkv_linear", "fused_qkv_linear")
            for name, layer, state in zip(
                layer_names,
                (qkv_linear, fused_qkv_linear),
                (qkv_linear_state, fused_qkv_linear_state),
            ):
                outputs[name], _ = F(
                    layer,
                    state=state,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(456),
                    inputs=inputs,
                )
            for layer_a, layer_b in combinations(layer_names, 2):
                # Check that the outputs are close for all pairs.
                self.assertNestedAllClose(outputs[layer_a], outputs[layer_b])


class PerDimScaleTest(TestCase):
    """Tests PerDimScale."""

    @parameterized.parameters(jnp.float32, jnp.float16, jnp.bfloat16)
    def test_per_dim_scale(self, dtype: jnp.dtype):
        batch_size, tgt_len, num_head, model_dim = 3, 5, 2, 8
        per_head_dim = model_dim // num_head
        layer: PerDimScale = (
            PerDimScale.default_config()
            .set(
                name="test",
                dim=per_head_dim,
            )  # We do not set layer dtype.
            .instantiate(parent=None)
        )
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        query = jax.random.normal(
            jax.random.PRNGKey(456), [batch_size, tgt_len, num_head, per_head_dim], dtype=dtype
        )
        self.assertEqual(dict(param=(per_head_dim,)), shapes(state))
        outputs, _ = F(
            layer,
            state=state,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=(query,),
        )
        expected_outputs = query
        assert_allclose(outputs, expected_outputs)
        self.assertEqual(outputs.dtype, query.dtype)


class MultiheadAttentionTest(TestCase):
    """Tests MultiheadAttention."""

    def test_invalid_key_value_combinations_raise(self):
        model_dim = 12
        num_heads = 4
        layer_kwargs = dict(
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )
        multihead_attention = (
            attention.MultiheadAttention.default_config()
            .set(name="test_multihead_attention", **layer_kwargs)
            .instantiate(parent=None)
        )
        fused_multihead_attention = (
            attention.MultiheadAttention.default_config()
            .set(
                name="test_fused_multihead_attention",
                input_linear=attention.FusedQKVLinear.default_config(),
                **layer_kwargs,
            )
            .instantiate(parent=None)
        )
        rng = np.random.default_rng(seed=123)
        inputs = jnp.asarray(rng.random([2, 6, model_dim]))
        for layer in (multihead_attention, fused_multihead_attention):
            for query, key, value in [(inputs, None, inputs), (inputs, inputs, None)]:
                with self.assertRaisesRegex(
                    ValueError, "key and value must be both None or both set"
                ):
                    layer.forward(query, key=key, value=value)

    @parameterized.parameters(None, PerDimScale.default_config())
    def test_input_linear_variants(self, per_dim_scale):
        with utils.numeric_checks(True):
            model_dim = 12
            num_heads = 4
            layer_kwargs = dict(
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                num_heads=num_heads,
                per_dim_scale=per_dim_scale,
            )
            multihead_attention = (
                attention.MultiheadAttention.default_config()
                .set(name="test_multihead_attention", **layer_kwargs)
                .instantiate(parent=None)
            )
            multihead_attention_state = multihead_attention.initialize_parameters_recursively(
                jax.random.PRNGKey(0)
            )
            fused_multihead_attention = (
                attention.MultiheadAttention.default_config()
                .set(
                    name="test_fused_multihead_attention",
                    input_linear=attention.FusedQKVLinear.default_config(),
                    **layer_kwargs,
                )
                .instantiate(parent=None)
            )

            def fused_state_from(state):
                output_state = {}
                for k, v in state.items():
                    if k == "i_proj":
                        weight = jnp.array(
                            [v[el]["weight"] for el in ("q_proj", "k_proj", "v_proj")]
                        )
                        bias = jnp.array([v[el]["bias"] for el in ("q_proj", "k_proj", "v_proj")])
                        output_state[k] = {"qkv_proj": dict(weight=weight, bias=bias)}
                    else:
                        output_state[k] = v
                return output_state

            # Map state to fused version.
            fused_multihead_attention_state = fused_state_from(multihead_attention_state)

            batch_size, src_len, tgt_len = 2, 6, 6
            rng = np.random.default_rng(seed=123)
            query = jnp.asarray(rng.random([batch_size, tgt_len, model_dim]))
            key = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
            value = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
            attention_logit_biases = jnp.ones([batch_size, tgt_len, src_len]) * NEG_INF
            inputs = dict(
                query=query, key=key, value=value, attention_logit_biases=attention_logit_biases
            )

            outputs = {}
            layer_names = ("multihead_attention", "fused_multihead_attention")
            for name, layer, state in zip(
                layer_names,
                (multihead_attention, fused_multihead_attention),
                (multihead_attention_state, fused_multihead_attention_state),
            ):
                outputs[name], _ = F(
                    layer,
                    state=state,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(456),
                    inputs=inputs,
                )
                layer_output_data = outputs[name].data
                # No NaN.
                self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)
            for layer_a, layer_b in combinations(layer_names, 2):
                # Check that the outputs are close for all pairs.
                self.assertNestedAllClose(outputs[layer_a], outputs[layer_b])

    @parameterized.parameters(None, PerDimScale.default_config())
    def test_all_mask(self, per_dim_scale):
        with utils.numeric_checks(True):
            model_dim = 12
            num_heads = 4
            per_head_dim = model_dim // num_heads
            cfg = attention.MultiheadAttention.default_config().set(
                name="test",
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                num_heads=num_heads,
                per_dim_scale=per_dim_scale,
            )
            layer: attention.MultiheadAttention = cfg.instantiate(parent=None)
            self.assertContainsSubset(
                dict(
                    dropout={},
                    i_proj={
                        **{
                            proj: {
                                "weight": ParameterSpec(
                                    dtype=layer.dtype(),
                                    shape=(model_dim, num_heads, per_head_dim),
                                    mesh_axes=PartitionSpec(None, "model", None),
                                    factorization=FactorizationSpec(axes=("row", None, "col")),
                                ),
                                "bias": ParameterSpec(
                                    dtype=layer.dtype(),
                                    shape=(num_heads, per_head_dim),
                                    mesh_axes=PartitionSpec("model", None),
                                    factorization=None,
                                ),
                            }
                            for proj in ("q_proj", "k_proj", "v_proj")
                        },
                    },
                    o_proj={
                        "bias": ParameterSpec(
                            dtype=layer.dtype(),
                            shape=(model_dim,),
                            mesh_axes=PartitionSpec(
                                None,
                            ),
                            factorization=None,
                        ),
                        "weight": ParameterSpec(
                            dtype=layer.dtype(),
                            shape=(model_dim, num_heads, per_head_dim),
                            mesh_axes=PartitionSpec(None, "model", None),
                            factorization=FactorizationSpec(axes=("row", None, "col")),
                        ),
                    },
                ),
                layer.create_parameter_specs_recursively(),
            )

            layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            qkv_shapes = dict(
                weight=(model_dim, num_heads, per_head_dim), bias=(num_heads, per_head_dim)
            )
            expected_params = {
                "i_proj": {f"{x}_proj": qkv_shapes for x in ("q", "k", "v")},
                "o_proj": dict(weight=(model_dim, num_heads, per_head_dim), bias=(model_dim,)),
                "dropout": {},
            }
            if per_dim_scale:
                expected_params["per_dim_scale"] = dict(param=(per_head_dim,))
            self.assertEqual(
                expected_params,
                shapes(layer_params),
            )

            batch_size, src_len, tgt_len = 2, 4, 6
            rng = np.random.default_rng(seed=123)
            query = jnp.asarray(rng.random([batch_size, tgt_len, model_dim]))
            key = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
            value = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
            attention_logit_biases = jnp.ones([batch_size, tgt_len, src_len]) * NEG_INF
            inputs = dict(
                query=query, key=key, value=value, attention_logit_biases=attention_logit_biases
            )
            layer_outputs, _ = F(
                layer,
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
            )
            layer_output_data = layer_outputs.data
            # No NaN.
            self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
    )
    def test_data_types(self, dtype: jnp.dtype, per_dim_scale: Optional[PerDimScale.Config]):
        model_dim = 16
        num_heads = 4
        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            dtype=dtype,
            per_dim_scale=per_dim_scale,
        )
        layer = cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size, src_len, tgt_len = 2, 4, 6
        query = jnp.zeros([batch_size, tgt_len, model_dim], dtype=dtype)
        key = jnp.zeros([batch_size, src_len, model_dim], dtype=dtype)
        value = jnp.zeros([batch_size, src_len, model_dim], dtype=dtype)
        attention_logit_biases = jnp.ones([batch_size, tgt_len, src_len]) * NEG_INF
        inputs = dict(
            query=query, key=key, value=value, attention_logit_biases=attention_logit_biases
        )
        layer_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=inputs,
        )
        self.assertEqual(layer_outputs.data.dtype, dtype)

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
        atten_logit_cap=(0.0, 20.0),
    )
    def test_extend_step(
        self, dtype: jnp.dtype, per_dim_scale: Optional[PerDimScale.Config], atten_logit_cap: float
    ):
        model_dim = 16
        num_heads = 4
        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            per_dim_scale=per_dim_scale,
            atten_logit_cap=atten_logit_cap,
        )
        cfg.input_linear.set(dtype=dtype, cache_dtype=None)
        layer = cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size, tgt_len = 2, 6
        query = jax.random.normal(
            jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
        )
        key = query
        value = query
        attention_logit_biases = attention.make_causal_mask(tgt_len)
        inputs = dict(
            query=query, key=key, value=value, attention_logit_biases=attention_logit_biases
        )
        forward_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=inputs,
        )

        initial_state = layer.init_states(target_batch_size=batch_size, target_max_len=tgt_len)
        for k in ["key", "value"]:
            # Check that the cache dtype is inferred as the layer dtype.
            self.assertEqual(initial_state["i_proj"][k].dtype, dtype)
        inputs = dict(cached_states=initial_state)
        decoder_output = jnp.zeros(shape=[tgt_len, batch_size, model_dim])
        decoder_probs = jnp.zeros(shape=[tgt_len, batch_size, num_heads, tgt_len])
        for t in range(tgt_len):
            inputs["query"] = jnp.expand_dims(query[:, t, :], axis=1)
            inputs["attention_logit_biases"] = attention_logit_biases[
                jnp.newaxis, jnp.newaxis, t, :
            ]
            extend_step_outputs, _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            inputs["cached_states"] = extend_step_outputs[0]
            decoder_output = decoder_output.at[t].set(
                jnp.squeeze(extend_step_outputs[1].data, axis=1)
            )
            decoder_probs = decoder_probs.at[t].set(
                jnp.squeeze(extend_step_outputs[1].probs, axis=2)
            )
        decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
        decoder_probs_transposed = jnp.transpose(decoder_probs, [1, 2, 0, 3])
        assert_allclose(
            decoder_out_transposed,
            forward_outputs.data,
            atol=1e-6,
        )
        assert_allclose(
            decoder_probs_transposed,
            forward_outputs.probs,
            atol=1e-6,
        )

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
        atten_logit_cap=(0.0, 20.0),
    )
    def test_prefill_states(
        self, dtype: jnp.dtype, per_dim_scale: Optional[PerDimScale.Config], atten_logit_cap: float
    ):
        model_dim = 16
        num_heads = 4
        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            per_dim_scale=per_dim_scale,
            atten_logit_cap=atten_logit_cap,
        )
        cfg.input_linear.set(dtype=dtype, cache_dtype=None)
        layer = cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size, tgt_len = 3, 6
        query = jax.random.normal(
            jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
        )
        attention_logit_biases = attention.make_causal_mask(tgt_len)

        forward_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(query=query, attention_logit_biases=attention_logit_biases),
        )

        time_step = jnp.arange(batch_size)
        (initial_states, initial_output), _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(
                time_step=time_step, query=query, attention_logit_biases=attention_logit_biases
            ),
            method="prefill_states",
        )

        # Check time_step and shapes of state.
        self.assertEqual(["i_proj"], list(initial_states.keys()))
        self.assertTrue(jnp.all(time_step == initial_states["i_proj"]["time_step"]))
        for proj in ["key", "value"]:
            self.assertEqual(
                (batch_size, num_heads, model_dim // num_heads, tgt_len),
                initial_states["i_proj"][proj].shape,
            )
            self.assertEqual(
                dtype,
                initial_states["i_proj"][proj].dtype,
            )

        # Zero-out outputs starting from initial time_step, and test that we can recover the full
        # outputs by calling extend_step starting from time_step.
        # [batch, tgt_len].
        time_step_mask = jnp.arange(tgt_len) < time_step[:, None]
        # [batch, tgt_len, model_dim].
        decoder_output = initial_output.data * time_step_mask[..., None]
        # [batch, num_heads, tgt_len, src_len].
        decoder_probs = initial_output.probs * time_step_mask[:, None, :, None]

        # [batch, tgt_len, model_dim] --> [batch, model_dim, tgt_len].
        decoder_output = jnp.moveaxis(decoder_output, -2, -1)
        # [batch, num_heads, tgt_len, src_len] --> [batch, num_heads, src_len, tgt_len].
        decoder_probs = jnp.moveaxis(decoder_probs, -2, -1)

        # Call extend_step from time_step, ensuring that outputs match.
        inputs = dict(cached_states=initial_states)
        while jnp.any(time_step < tgt_len):
            # [batch, tgt_len=1, model_dim].
            inputs["query"] = jnp.take_along_axis(
                query, time_step[:, None, None], axis=1, mode="clip"
            )
            # [batch=1, tgt_len=1, tgt_len].
            inputs["attention_logit_biases"] = jnp.take_along_axis(
                attention_logit_biases[None, :, :], time_step[:, None, None], axis=1, mode="clip"
            )
            (updated_state, outputs), _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            inputs["cached_states"] = updated_state

            # [batch, model_dim, tgt_len=1]
            curr_outputs = jnp.moveaxis(outputs.data, -2, -1)
            # [batch, num_heads, src_len, tgt_len=1]
            curr_probs = jnp.moveaxis(outputs.probs, -2, -1)

            # [batch, 1, tgt_len].
            oh_indices = jax.nn.one_hot(time_step, tgt_len)[:, None, :]
            decoder_output = decoder_output + curr_outputs * oh_indices
            # [batch, 1, 1, tgt_len].
            oh_indices = oh_indices[..., None, :]
            decoder_probs = decoder_probs + curr_probs * oh_indices
            time_step = time_step + 1

        # [batch, model_dim, tgt_len] --> [batch, tgt_len, model_dim].
        decoder_output = jnp.moveaxis(decoder_output, -1, -2)
        # [batch, num_heads, src_len, tgt_len] --> [batch, num_heads, tgt_len, src_len].
        decoder_probs = jnp.moveaxis(decoder_probs, -1, -2)

        assert_allclose(decoder_output, forward_outputs.data)
        assert_allclose(decoder_probs, forward_outputs.probs)


def oracle_xl_attention_logits(
    query: np.ndarray,
    key: np.ndarray,
    relative_pos_emb: np.ndarray,
    content_bias: np.ndarray,
    positional_bias: np.ndarray,
) -> np.ndarray:
    """Computes expected attention logits using non-vectorized approach.

    Reference:
    https://github.com/tensorflow/lingvo/blob/41212226eac7a26491790c2bd476b78493f93ff6/lingvo/core/attention_util_test.py#L48-L73.

    Note that this implementation follows XLNet implementation and is different from the lingvo
    implementation in that here the relative_pos_emb index is computed from key_i - query_i,
    while lingvo computes from query_i - key_i.

    See comments on xl_attention_logits().
    """
    batch, seqlen, num_heads, _ = query.shape
    tgtlen, srclen = seqlen, seqlen

    logits = np.zeros((batch, num_heads, tgtlen, srclen))

    for b in range(batch):
        for n in range(num_heads):
            for i in range(tgtlen):
                for j in range(srclen):
                    offset = seqlen - 1
                    pos_emb = relative_pos_emb[j - i + offset]
                    logits[b][n][i][j] = np.dot(query[b][i][n], key[b][j][n])
                    logits[b][n][i][j] += np.dot(query[b][i][n], pos_emb[n])
                    logits[b][n][i][j] += np.dot(content_bias[n], key[b][j][n])
                    logits[b][n][i][j] += np.dot(positional_bias[n], pos_emb[n])
    return logits


class TransformerXLTest(TestCase):
    """Tests TransformerXL."""

    def test_rel_pos_to_abs_pos(self):
        seq_len = 5
        # rel_offset[:, i] = i - (seq_len - 1), i.e., in range [-seq_len + 1, seq_len - 1].
        rel_offset = jnp.tile(jnp.arange(-seq_len + 1, seq_len)[None, :], [seq_len, 1])
        # abs_pos[i, j] = j - i.
        abs_pos = rel_pos_to_abs_pos(rel_offset)
        expected = jnp.arange(seq_len)[None, :] - jnp.arange(seq_len)[:, None]
        assert_allclose(abs_pos, expected)

    def test_xl_attention_logits(self):
        num_heads, per_head_dim = 4, 3
        batch_size, tgt_len = 2, 5
        q = jax.random.normal(
            jax.random.PRNGKey(100),
            [batch_size, tgt_len, num_heads, per_head_dim],
            dtype=jnp.float32,
        )
        k = jax.random.normal(
            jax.random.PRNGKey(101),
            [batch_size, tgt_len, num_heads, per_head_dim],
            dtype=jnp.float32,
        )
        relative_pos_emb = jax.random.normal(
            jax.random.PRNGKey(102), [2 * tgt_len - 1, num_heads, per_head_dim], dtype=jnp.float32
        )
        u = jax.random.normal(jax.random.PRNGKey(103), [num_heads, per_head_dim], dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(104), [num_heads, per_head_dim], dtype=jnp.float32)
        expected = oracle_xl_attention_logits(
            query=q, key=k, relative_pos_emb=relative_pos_emb, content_bias=u, positional_bias=v
        )
        actual = xl_attention_logits(
            q_proj=q, k_proj=k, relative_pos_emb=relative_pos_emb, u=u, v=v
        )
        assert_allclose(actual, expected)

    @parameterized.product(
        per_dim_scale=(None, PerDimScale.default_config()),
        scale_position=(
            MultiheadAttentionXL.ScalePosition.LOGIT,
            MultiheadAttentionXL.ScalePosition.QUERY,
        ),
    )
    def test_per_dim_scale(self, per_dim_scale, scale_position):
        model_dim = 6
        num_heads = 2
        cfg = attention.TransformerAttentionLayer.default_config().set(
            name="test",
            target_dim=model_dim,
            source_dim=model_dim,
            structure="postnorm",
            attention=MultiheadAttentionXL.default_config().set(
                num_heads=num_heads, per_dim_scale=per_dim_scale, scale_position=scale_position
            ),
        )
        cfg.attention.output_linear.bias = False
        cfg.attention.vlog = 5

        layer: attention.TransformerAttentionLayer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(123))
        batch_size, tgt_len = 2, 5
        target = jax.random.normal(
            jax.random.PRNGKey(100), [batch_size, tgt_len, model_dim], dtype=jnp.float32
        )

        layer_params["attention"]["u_bias"] = jax.random.normal(
            jax.random.PRNGKey(0), [num_heads, model_dim // num_heads]
        )
        layer_params["attention"]["v_bias"] = jax.random.normal(
            jax.random.PRNGKey(1), [num_heads, model_dim // num_heads]
        )
        if per_dim_scale:
            layer_params["attention"]["per_dim_scale"]["param"] = jax.random.normal(
                jax.random.PRNGKey(2), [model_dim // num_heads]
            )
        layer_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(target=target),
        )
        expected_vals = {
            str(None): {
                MultiheadAttentionXL.ScalePosition.LOGIT.value: 48.55191,
                MultiheadAttentionXL.ScalePosition.QUERY.value: 48.91095,
            },
            str(PerDimScale.default_config()): {
                MultiheadAttentionXL.ScalePosition.LOGIT.value: 46.327015,
                MultiheadAttentionXL.ScalePosition.QUERY.value: 46.608315,
            },
        }
        assert_allclose(
            jnp.abs(layer_outputs.data).sum(),
            expected_vals[str(per_dim_scale)][scale_position.value],
        )

    def test_multihead_attention_xl(self):
        model_dim = 6
        num_heads = 2
        per_head_dim = model_dim // num_heads
        cfg = attention.TransformerAttentionLayer.default_config().set(
            name="test",
            target_dim=model_dim,
            source_dim=model_dim,
            structure="postnorm",
            attention=MultiheadAttentionXL.default_config().set(num_heads=num_heads),
        )
        cfg.attention.output_linear.bias = False
        cfg.attention.vlog = 5
        layer: attention.TransformerAttentionLayer = cfg.instantiate(parent=None)
        layer.initialize_parameters_recursively(jax.random.PRNGKey(123))
        ref_cfg = hf_xlnet.XLNetConfig(
            n_head=num_heads,
            d_model=model_dim,
            d_head=model_dim // num_heads,
            dropout=0,
            layer_norm_eps=cfg.norm.eps,
        )
        ref = hf_xlnet.XLNetRelativeAttention(ref_cfg)
        # XLNetRelativeAttention is not properly initialized.
        with torch.no_grad():
            for var in ("q", "k", "v", "o", "r"):
                getattr(ref, var).copy_(
                    torch.normal(0, np.sqrt(model_dim), [model_dim, num_heads, per_head_dim])
                )
            for var in ("r_w_bias", "r_r_bias"):
                getattr(ref, var).copy_(
                    torch.normal(0, np.sqrt(model_dim), [num_heads, model_dim // num_heads])
                )
        batch_size, tgt_len = 2, 5
        target = jax.random.normal(
            jax.random.PRNGKey(100), [batch_size, tgt_len, model_dim], dtype=jnp.float32
        )
        num_tokens = jax.random.randint(
            jax.random.PRNGKey(101),
            minval=2,
            maxval=tgt_len + 1,
            shape=[batch_size],
        )
        # [batch_size, tgt_len].
        is_valid_token = jnp.arange(tgt_len)[None, :] < num_tokens[:, None]
        # [batch_size, 1, tgt_len, tgt_len].
        attention_logit_biases = jnp.expand_dims(
            NEG_INF * (1 - jnp.einsum("bt,bs->bts", is_valid_token, is_valid_token)), 1
        )
        # [2 * tgt_len, model_dim].
        rel_pos_emb = sinusoidal_positional_embeddings(
            jnp.arange(tgt_len, -tgt_len, -1), dim=model_dim
        )
        ref_inputs = dict(
            g=None,
            h=target.transpose([1, 0, 2]),  # [qlen, bsz, d_model].
            r=rel_pos_emb[:, None, :],  # [rlen, 1, d_model].
            attn_mask_g=None,
            # [qlen, klen, bsz, n_head].
            attn_mask_h=attention_logit_biases.transpose([2, 3, 0, 1]) < 0,
            seg_mat=None,
        )
        logging.info("ref_inputs=%s", ref_inputs)

        test_outputs, ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref,
            test_inputs=dict(target=target, attention_logit_biases=attention_logit_biases),
            ref_inputs=as_torch_tensor(ref_inputs),
            parameters_from_ref_layer=parameters_from_torch_layer,
            require_same_num_params=False,
        )
        logging.info("test_outputs=%s", test_outputs)
        logging.info("ref_outputs=%s", ref_outputs)
        self.assertNestedAllClose(
            test_outputs.data, as_tensor(ref_outputs[0]).transpose([1, 0, 2]), atol=6e-6
        )


class TransformerTest(absltest.TestCase):
    """Tests TransformerLayer."""

    def _compare_against_roberta_attention(
        self, ref: hf_roberta.RobertaAttention, layer: TransformerAttentionLayer
    ):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_param_shapes = jax.tree_util.tree_map(lambda x: x.shape, layer_params)
        print(f"layer state={layer_param_shapes}")
        layer_params = parameters_from_torch_layer(ref)
        batch_size, tgt_len = 2, 6
        model_dim, num_heads = layer.config.target_dim, layer.config.attention.num_heads
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        null_mask = jnp.zeros([tgt_len, tgt_len])
        rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
        for mask in (None, null_mask, rand_mask):
            if mask is not None:
                mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
            layer_outputs, _ = F(
                layer,
                inputs=dict(target=jnp.asarray(target), attention_logit_biases=mask),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            attn_mask = None if mask is None else as_torch_tensor(mask)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                output_attentions=False,
            )
            assert_allclose(layer_outputs.data, as_tensor(ref_outputs))

    def test_against_roberta_attention(self):
        model_dim = 16
        num_heads = 4
        cfg = attention.TransformerAttentionLayer.default_config().set(
            name="test",
            target_dim=model_dim,
            source_dim=model_dim,
            structure="postnorm",
        )
        cfg.attention.set(num_heads=num_heads)
        layer = cfg.instantiate(parent=None)
        roberta_config = hf_roberta.RobertaConfig(
            hidden_size=model_dim,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0,
            classifier_dropout=0,
        )
        print(f"roberta_config={roberta_config}")
        ref = hf_roberta.RobertaAttention(roberta_config)
        self._compare_against_roberta_attention(ref, layer)

    def _compare_against_roberta_layer(self, ref: hf_roberta.RobertaLayer, layer: TransformerLayer):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_params = parameters_from_torch_layer(ref)
        batch_size, tgt_len = 2, 6
        model_dim, num_heads = (
            layer.config.input_dim,
            layer.config.self_attention.attention.num_heads,
        )
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        null_mask = jnp.zeros([tgt_len, tgt_len])
        rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
        for mask in (None, null_mask, rand_mask):
            if mask is not None:
                mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
            layer_outputs, _ = F(
                layer,
                inputs=dict(data=jnp.asarray(target), self_attention_logit_biases=mask),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            if layer_outputs.self_attention_probs is not None:
                self.assertEqual(
                    (batch_size, num_heads, tgt_len, tgt_len),
                    layer_outputs.self_attention_probs.shape,
                )
            attn_mask = None if mask is None else as_torch_tensor(mask)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                output_attentions=False,
            )
            assert_allclose(layer_outputs.data, as_tensor(ref_outputs))

    def test_against_roberta_layer(self):
        model_dim = 16
        num_heads = 4
        cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
        cfg.self_attention.set(structure="postnorm")
        cfg.feed_forward.set(
            structure="postnorm", activation="nn.silu", hidden_dim=scaled_hidden_dim(4)
        )
        cfg.feed_forward.linear1.set(bias=True)
        cfg.feed_forward.linear2.set(bias=True)
        cfg.self_attention.attention.set(num_heads=num_heads)
        cfg.self_attention.attention.input_linear.layer.set(bias=True)
        cfg.self_attention.attention.output_linear.set(bias=True)
        layer: TransformerLayer = cfg.instantiate(parent=None)
        roberta_config = hf_roberta.RobertaConfig(
            hidden_size=model_dim,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0,
            classifier_dropout=0,
            # Jax's gelu uses an approximation by default and is slightly different from
            # torch.nn.gelu.
            hidden_act="silu",
        )
        ref = hf_roberta.RobertaLayer(roberta_config)
        self._compare_against_roberta_layer(ref, layer)


class TestStackModel(BaseLayer):
    """A dummy transformer stack."""

    @config_class
    class Config(BaseLayer.Config):
        stack: Optional[BaseStackedTransformerLayer.Config] = None  # The transformer stack.

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("stack", cfg.stack)

    def forward(self, data, self_attention_logit_biases):
        # [batch, length, dim].
        x = self.stack(data, self_attention_logit_biases=self_attention_logit_biases).data
        x_mean = jnp.mean(x, axis=1, keepdims=True)
        # [batch, length].
        x_var = jnp.sum((x - x_mean) ** 2, axis=-1)
        loss = jnp.mean(x_var)
        return loss, {"mean": x_mean}


class StackedTransformerTest(TestCase):
    """Tests StackedTransformerLayer."""

    def _stack_config(
        self, stack_cfg, *, num_layers, model_dim, num_heads, dtype, remat_spec
    ) -> TestStackModel.Config:
        if isinstance(stack_cfg, type):
            stack_cfg = stack_cfg.default_config()
        if callable(remat_spec):
            remat_spec = remat_spec(stack_cfg)
        cfg = TestStackModel.default_config().set(
            name="test",
            stack=stack_cfg.set(
                input_dim=model_dim,
                num_layers=num_layers,
                vlog=5,
                dtype=dtype,
                layer=TransformerLayer.default_config().set(remat_spec=remat_spec),
            ),
        )
        layer_cfg = cfg.stack.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.feed_forward.hidden_dim = model_dim * 4
        layer_cfg.vlog = 5
        return cfg

    @parameterized.product(
        transformer_type=[StackedTransformerLayer, RepeatedTransformerLayer],
        # Also tests stack-of-stacks and repeat-of-stacks.
        layer_type=[TransformerLayer, StackedTransformerLayer],
    )
    def test_transformer_extend_step(self, transformer_type, layer_type):
        batch_size, src_len, tgt_len = 10, 4, 6
        num_dec_layers, model_dim, num_heads = 3, 16, 4

        cfg = transformer_type.default_config().set(
            name="test",
            input_dim=model_dim,
            num_layers=num_dec_layers,
        )
        cross_atten_cfg = TransformerAttentionLayer.default_config().set(
            source_dim=model_dim * 2,
            structure="postnorm",
        )
        cross_atten_cfg.attention.set(num_heads=num_heads)

        # Prepare layer config.
        if layer_type == StackedTransformerLayer:
            cfg.layer = layer_type.default_config().set(num_layers=2)
            layer_cfg = cfg.layer.layer
        else:
            layer_cfg = cfg.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.cross_attention = cross_atten_cfg
        layer_cfg.feed_forward.hidden_dim = model_dim * 4

        # Instantiate transformer stack.
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        target = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])
        source = jax.random.normal(jax.random.PRNGKey(456), [batch_size, src_len, model_dim * 2])

        self_attention_logit_biases = attention.make_causal_mask(tgt_len)
        cross_attention_logit_biases = (
            jnp.array(np.random.randint(0, 2, [tgt_len, src_len])) * NEG_INF
        )

        forward_outputs, _ = F(
            layer,
            inputs=dict(
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        initial_state = layer.init_states(target_batch_size=batch_size, target_max_len=tgt_len)
        inputs = dict(cached_states=initial_state, cross_attention_data=source)
        decoder_output = jnp.zeros(shape=[tgt_len, batch_size, model_dim])

        # [num_dec_layers, [num_stacked_layers,] batch_size, num_heads, tgt_len, tgt_len] -->
        # [tgt_len, num_dec_layers, [num_stacked_layers,] batch_size, num_heads, tgt_len].
        # The layer being stacked can itself be a stack, in which case we have an extra dim.
        decoder_self_attention_probs = jnp.moveaxis(
            jnp.zeros_like(forward_outputs.self_attention_probs),
            -2,
            0,
        )
        # [tgt_len, num_dec_layers, [num_stacked_layers,] batch_size, num_heads, src_len].
        decoder_cross_attention_probs = jnp.moveaxis(
            jnp.zeros_like(forward_outputs.cross_attention_probs),
            -2,
            0,
        )
        for t in range(tgt_len):
            inputs["data"] = jnp.expand_dims(target[:, t, :], axis=1)
            inputs["self_attention_logit_biases"] = self_attention_logit_biases[
                jnp.newaxis, jnp.newaxis, t, :
            ]
            inputs["cross_attention_logit_biases"] = cross_attention_logit_biases[
                jnp.newaxis, jnp.newaxis, t, :
            ]
            (updated_states, layer_outputs), _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            # Check that updated_states are VDicts for the Repeated layer.
            if transformer_type is RepeatedTransformerLayer:
                jax.tree_map(
                    lambda v: self.assertIsInstance(v, utils.VDict),
                    updated_states,
                    is_leaf=lambda v: isinstance(v, dict),
                )
            inputs["cached_states"] = updated_states
            decoder_output = decoder_output.at[t].set(jnp.squeeze(layer_outputs.data, axis=1))
            decoder_self_attention_probs = decoder_self_attention_probs.at[t].set(
                jnp.squeeze(layer_outputs.self_attention_probs, axis=-2)
            )
            decoder_cross_attention_probs = decoder_cross_attention_probs.at[t].set(
                jnp.squeeze(layer_outputs.cross_attention_probs, axis=-2)
            )
        decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
        decoder_self_attention_probs_transposed = jnp.moveaxis(decoder_self_attention_probs, 0, -2)
        decoder_cross_attention_probs_transposed = jnp.moveaxis(
            decoder_cross_attention_probs, 0, -2
        )

        assert_allclose(decoder_out_transposed, forward_outputs.data, atol=1e-6)
        assert_allclose(
            decoder_self_attention_probs_transposed, forward_outputs.self_attention_probs, atol=1e-6
        )
        assert_allclose(
            decoder_cross_attention_probs_transposed,
            forward_outputs.cross_attention_probs,
            atol=1e-6,
        )

    @parameterized.product(
        transformer_type=[StackedTransformerLayer, RepeatedTransformerLayer],
        # Also tests stack-of-stacks and repeat-of-stacks.
        layer_type=[TransformerLayer, StackedTransformerLayer],
    )
    # pylint: disable-next=too-many-statements
    def test_transformer_prefill_states(self, transformer_type, layer_type):
        batch_size, src_len, tgt_len = 10, 4, 6
        num_dec_layers, model_dim, num_heads = 3, 16, 4

        cfg = transformer_type.default_config().set(
            name="test",
            input_dim=model_dim,
            num_layers=num_dec_layers,
        )
        cross_atten_cfg = TransformerAttentionLayer.default_config().set(
            source_dim=model_dim * 2,
            structure="postnorm",
        )
        cross_atten_cfg.attention.set(num_heads=num_heads)

        # Prepare layer config.
        if layer_type == StackedTransformerLayer:
            cfg.layer = layer_type.default_config().set(num_layers=2)
            layer_cfg = cfg.layer.layer
        else:
            layer_cfg = cfg.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.cross_attention = cross_atten_cfg
        layer_cfg.feed_forward.hidden_dim = model_dim * 4

        # Instantiate transformer stack.
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        target = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])
        source = jax.random.normal(jax.random.PRNGKey(456), [batch_size, src_len, model_dim * 2])

        self_attention_logit_biases = attention.make_causal_mask(tgt_len)
        cross_attention_logit_biases = (
            jnp.array(np.random.randint(0, 2, [tgt_len, src_len])) * NEG_INF
        )

        forward_outputs, _ = F(
            layer,
            inputs=dict(
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        # Initialize state.
        time_step = jnp.arange(batch_size)
        (initial_states, initial_output), _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(
                time_step=time_step,
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
            ),
            method="prefill_states",
        )

        # Zero-out outputs starting from initial time_step, and test that we can recover the full
        # outputs by calling extend_step starting from time_step.
        time_step_mask = jnp.arange(tgt_len) < time_step[:, None]
        # [batch, tgt_len, model_dim].
        decoder_output = initial_output.data * time_step_mask[..., None]
        # [num_layers, batch, num_heads, tgt_len, tgt_len].
        decoder_self_attention_probs = (
            initial_output.self_attention_probs * time_step_mask[None, :, None, :, None]
        )
        # [num_layers, batch, num_heads, tgt_len, src_len].
        decoder_cross_attention_probs = (
            initial_output.cross_attention_probs * time_step_mask[None, :, None, :, None]
        )

        # Transpose for simpler updates during extend_step.
        # [batch, tgt_len, model_dim] --> [batch, model_dim, tgt_len].
        decoder_output = jnp.moveaxis(decoder_output, -2, -1)
        # [..., tgt_len, src_len] --> [..., src_len, tgt_len].
        decoder_self_attention_probs = jnp.moveaxis(decoder_self_attention_probs, -2, -1)
        decoder_cross_attention_probs = jnp.moveaxis(decoder_cross_attention_probs, -2, -1)

        # Call extend_step from time_step, ensuring that outputs match.
        inputs = dict(cached_states=initial_states, cross_attention_data=source)
        while jnp.any(time_step < tgt_len):
            # [batch, tgt_len=1, model_dim].
            inputs["data"] = jnp.take_along_axis(
                target, time_step[:, None, None], axis=1, mode="clip"
            )
            # [batch=1, tgt_len=1, tgt_len].
            inputs["self_attention_logit_biases"] = jnp.take_along_axis(
                self_attention_logit_biases[None, :, :],
                time_step[:, None, None],
                axis=1,
                mode="clip",
            )
            # [batch=1, tgt_len=1, src_len].
            inputs["cross_attention_logit_biases"] = jnp.take_along_axis(
                cross_attention_logit_biases[None, :, :],
                time_step[:, None, None],
                axis=1,
                mode="clip",
            )
            (updated_states, layer_outputs), _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            # Check that updated_states are VDicts for the Repeated layer.
            if transformer_type is RepeatedTransformerLayer:
                jax.tree_map(
                    lambda v: self.assertIsInstance(v, utils.VDict),
                    updated_states,
                    is_leaf=lambda v: isinstance(v, dict),
                )
            inputs["cached_states"] = updated_states

            # [batch, model_dim, tgt_len=1]
            curr_outputs = jnp.moveaxis(layer_outputs.data, -2, -1)
            # [..., tgt_len, tgt_len=1]
            curr_self_attention_probs = jnp.moveaxis(layer_outputs.self_attention_probs, -2, -1)
            # [..., src_len, tgt_len=1]
            curr_cross_attention_probs = jnp.moveaxis(layer_outputs.cross_attention_probs, -2, -1)

            # [batch, 1, tgt_len].
            oh_indices = jax.nn.one_hot(time_step, tgt_len)[:, None, :]
            decoder_output = decoder_output + curr_outputs * oh_indices
            # [num_layers=1, batch, num_heads=1, tgt_len=1, tgt_len].
            oh_indices = oh_indices[None, :, None, :, :]
            decoder_self_attention_probs = (
                decoder_self_attention_probs + curr_self_attention_probs * oh_indices
            )
            decoder_cross_attention_probs = (
                decoder_cross_attention_probs + curr_cross_attention_probs * oh_indices
            )
            time_step = time_step + 1

        # [batch, model_dim, tgt_len] --> [batch, tgt_len, model_dim].
        decoder_output = jnp.moveaxis(decoder_output, -1, -2)
        # [..., src_len, tgt_len] --> [..., tgt_len, src_len].
        decoder_self_attention_probs = jnp.moveaxis(decoder_self_attention_probs, -1, -2)
        decoder_cross_attention_probs = jnp.moveaxis(decoder_cross_attention_probs, -1, -2)

        assert_allclose(decoder_output, forward_outputs.data)
        assert_allclose(decoder_self_attention_probs, forward_outputs.self_attention_probs)
        assert_allclose(decoder_cross_attention_probs, forward_outputs.cross_attention_probs)

    def test_stack_vs_repeat(self):
        self._compare_layers(StackedTransformerLayer, RepeatedTransformerLayer)

    def test_stack_vs_repeat_bfloat16(self):
        # FIXME(rpang): fix the following test, which is caused by different behaviors of bfloat16
        # to float32 casting.
        # self._compare_layers(StackedTransformerLayer, RepeatedTransformerLayer,
        # dtype=jnp.bfloat16)
        pass

    def test_stack_vs_repeat_remat_everything_saveable(self):
        self._compare_layers(
            StackedTransformerLayer,
            RepeatedTransformerLayer,
            remat_spec=RematSpec(policy=jax_remat_policies.everything_saveable),
        )

    def test_stack_vs_repeat_with_build_remat_spec(self):
        self._compare_layers(
            StackedTransformerLayer,
            RepeatedTransformerLayer,
            remat_spec=build_remat_spec,
        )

    def test_stack_vs_pipeline_of_stacks(self):
        self._compare_layers(
            StackedTransformerLayer,
            PipelinedTransformerLayer.default_config().set(
                stage=StackedTransformerLayer.default_config().set(layer=None)
            ),
        )

    def test_stack_vs_pipeline_of_repeats(self):
        self._compare_layers(
            StackedTransformerLayer,
            PipelinedTransformerLayer.default_config().set(
                stage=RepeatedTransformerLayer.default_config().set(layer=None)
            ),
        )

    def test_stack_vs_pipeline_remat_everything_saveable(self):
        self._compare_layers(
            StackedTransformerLayer,
            PipelinedTransformerLayer,
            remat_spec=RematSpec(policy=jax_remat_policies.everything_saveable),
        )

    # pylint: disable-next=too-many-statements,too-many-branches
    def _compare_layers(self, *stack_configs, dtype=jnp.float32, remat_spec=None):
        with utils.numeric_checks(False):
            batch_size, tgt_len = 10, 5
            num_layers, model_dim, num_heads = 6, 8, 4

            target = jax.random.normal(
                jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
            )
            rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
            rand_mask = jnp.tile(rand_mask[None, None, :, :], (batch_size, num_heads, 1, 1))

            all_params = []
            all_outputs = []
            all_summaries = []
            all_gradients = []
            all_updates = []
            for stack_cfg in stack_configs:
                cfg = self._stack_config(
                    stack_cfg,
                    num_layers=num_layers,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    dtype=dtype,
                    remat_spec=remat_spec,
                )
                cls = cfg.stack.klass
                if cls == PipelinedTransformerLayer:
                    cfg.stack.num_microbatches = batch_size // 2
                    cfg.stack.num_stages = num_layers // 2
                layer: TestStackModel = cfg.instantiate(parent=None)

                param_specs = layer.create_parameter_specs_recursively()
                logging.info(
                    "%s.factorization_specs=%s",
                    cls,
                    jax.tree_util.tree_map(lambda x: x.factorization, param_specs),
                )
                layer_params = layer.initialize_parameters_recursively(
                    prng_key=jax.random.PRNGKey(123)
                )
                logging.info(
                    "%s.params=%s",
                    cls,
                    jax.tree_util.tree_map(lambda x: f"{x.dtype}({x.shape})", layer_params),
                )

                def _loss(layer_params, data, mask, layer=layer):
                    layer_outputs, layer_output_collection = F(
                        layer,
                        inputs=dict(data=data, self_attention_logit_biases=mask),
                        state=layer_params,
                        is_training=True,
                        prng_key=jax.random.PRNGKey(0),
                    )
                    loss, aux = layer_outputs
                    return loss, (aux, layer_output_collection)

                value, grads = jax.value_and_grad(_loss, has_aux=True)(
                    layer_params, jnp.asarray(target), rand_mask
                )
                loss, (aux, layer_output_collection) = value
                layer_outputs = (loss, aux)

                summaries = layer_output_collection.summaries
                logging.info(
                    "layer_outputs=%s summaries=%s",
                    shapes(flatten_items(layer_outputs)),
                    shapes(flatten_items(summaries)),
                )
                logging.info(
                    "global_grad_norm=%s, grads=%s",
                    optax.global_norm(grads),
                    shapes(flatten_items(grads)),
                )

                optimizer = adafactor_optimizer(
                    learning_rate=0.1,
                    b1=0.9,
                    b2=0.98,
                    multiply_by_parameter_scale=False,
                    clipping_threshold=1.0,
                    eps=1e-2,
                )
                opt_params = jax.tree_util.tree_map(
                    lambda spec, p: OptParam(
                        value=p,
                        factorization_spec=spec.factorization,
                        weight_decay_scale=spec.weight_decay_scale,
                    ),
                    param_specs,
                    layer_params,
                )
                opt_state = optimizer.init(opt_params)
                logging.info("opt_state=%s", shapes(opt_state))
                updates, opt_state = optimizer.update(grads, opt_state, opt_params)

                def rms_norm(x):
                    return jnp.sqrt(jnp.mean(x**2))

                if cls == StackedTransformerLayer:
                    update_norms = jax.tree_util.tree_map(rms_norm, updates)
                else:
                    update_norms = jax.vmap(
                        lambda x, norm=rms_norm: jax.tree_util.tree_map(norm, x)
                    )(updates)
                logging.info(
                    "global_update_norm=%s update_norms=%s",
                    optax.global_norm(updates),
                    dict(utils.flatten_items(update_norms)),
                )

                def recursive_stack(stacked, axis=0):
                    return {
                        "layer": utils.vectorized_tree_map(
                            lambda *xs: jnp.stack(xs, axis=axis),
                            *stacked.values(),
                        )
                    }

                if cls == StackedTransformerLayer:
                    for x in (layer_params, grads, summaries, updates):
                        x["stack"] = recursive_stack(x["stack"])

                if cls == RepeatedTransformerLayer:
                    for x in (layer_params, grads, summaries, updates):
                        x["stack"] = x["stack"]["repeat"]

                if cls == PipelinedTransformerLayer:
                    for x in (layer_params, grads, summaries, updates):
                        if cfg.stack.stage.klass == StackedTransformerLayer:
                            # First stack within each stage.
                            x["stack"]["pipeline"]["layer"] = recursive_stack(
                                x["stack"]["pipeline"]["layer"], axis=1
                            )
                            logging.info("x=%s", shapes(x))
                        elif cfg.stack.stage.klass == RepeatedTransformerLayer:
                            x["stack"]["pipeline"]["layer"] = x["stack"]["pipeline"]["layer"][
                                "repeat"
                            ]
                        else:
                            raise NotImplementedError(cfg.stack.stage.klass)

                        # Then reshape across stages.
                        x["stack"] = jax.tree_util.tree_map(
                            lambda x: x.reshape([num_layers] + list(x.shape[2:])),
                            x["stack"]["pipeline"]["layer"],
                        )

                all_params.append(layer_params)
                all_outputs.append(layer_outputs)
                all_summaries.append(summaries)
                all_gradients.append(grads)
                all_updates.append(updates)

                if cls == StackedTransformerLayer:
                    one_layer = layer.stack.layer0
                elif cls == RepeatedTransformerLayer:
                    one_layer = layer.stack.repeat.layer
                else:
                    one_layer = None

                # pylint: disable=protected-access
                if one_layer is not None:
                    logging.info(
                        "%s._remat_methods = %s", one_layer.path(), one_layer._remat_methods
                    )
                    if remat_spec is not None:
                        self.assertSequenceEqual(
                            one_layer._remat_methods, ["forward"], msg=one_layer.path()
                        )
                    else:
                        self.assertEmpty(one_layer._remat_methods, msg=one_layer.path())
                # pylint: enable=protected-access

            self.assertNestedAllClose(all_params[0], all_params[1])
            self.assertNestedAllClose(all_summaries[0], all_summaries[1])
            self.assertNestedAllClose(all_outputs[0], all_outputs[1])
            self.assertNestedAllClose(all_gradients[0], all_gradients[1])
            self.assertNestedAllClose(all_updates[0], all_updates[1])

    @parameterized.parameters(None, 0.0, 0.2, 1.0)
    def test_stochastic_depth(self, rate):
        batch_size, tgt_len = 10, 6
        num_dec_layers, model_dim, num_heads = 3, 16, 4
        model_dim = 16
        num_heads = 4
        cfg = StackedTransformerLayer.default_config().set(
            name="test",
            input_dim=model_dim,
            num_layers=num_dec_layers,
            peak_stochastic_depth_rate=rate,
        )
        layer_cfg = cfg.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.feed_forward.hidden_dim = model_dim * 4

        if rate is None or 0 <= rate < 1:
            layer = cfg.instantiate(parent=None)
            layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            target = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])
            F(
                layer,
                inputs=dict(data=target),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
        else:
            with self.assertRaises(ValueError):
                cfg.instantiate(parent=None)

    @parameterized.product(is_training=(True, False))
    def test_stacked_transformer_with_seq_layer_cfgs(self, is_training):
        batch_size = 2
        seq_len = 16
        input_dim = 4
        hidden_dim = 16
        num_layers = 4
        num_heads = 4

        # Create a StackedTransformerLayer by specifying a sequence of layer configs.
        cfg = StackedTransformerLayer.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.num_layers = num_layers
        transformer_cfg = TransformerLayer.default_config()
        transformer_cfg.self_attention.attention.num_heads = num_heads
        transformer_cfg.feed_forward.hidden_dim = hidden_dim
        cfg.layer = (transformer_cfg,) * num_layers
        layer: StackedTransformerLayer = cfg.instantiate(parent=None)
        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(data=inputs),
        )
        # Create a ref StackedTransformerLayer with repeating the default layer cfg.
        ref_cfg = StackedTransformerLayer.default_config().set(name="test")
        ref_cfg.input_dim = input_dim
        ref_cfg.num_layers = num_layers
        ref_cfg.layer.self_attention.attention.num_heads = num_heads
        ref_cfg.layer.feed_forward.hidden_dim = hidden_dim
        ref_layer: StackedTransformerLayer = ref_cfg.instantiate(parent=None)
        ref_outputs, _ = F(
            ref_layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(data=inputs),
        )
        assert_allclose(outputs.data, ref_outputs.data)
        assert_allclose(outputs.self_attention_probs, ref_outputs.self_attention_probs)


class ConfigHelperTest(TestCase):
    """Tests config utils."""

    @parameterized.product(
        self_attention_input_linear_cfg=(
            QKVLinear.default_config(),
            FusedQKVLinear.default_config(),
        ),
        cross_attention_cfg=(None, TransformerAttentionLayer.default_config()),
        batch_axis_names=("data", ("replica", "data", "fsdp")),
        fsdp_axis_names=("fsdp",),
        tp_axis_names=("model",),
    )
    def test_set_double_shard_weights_config(
        self,
        self_attention_input_linear_cfg,
        cross_attention_cfg,
        batch_axis_names,
        fsdp_axis_names,
        tp_axis_names,
    ):
        cfg: TransformerLayer.Config = TransformerLayer.default_config().set(
            cross_attention=cross_attention_cfg
        )
        cfg.self_attention.attention.input_linear = self_attention_input_linear_cfg
        set_double_shard_weights_config(
            cfg,
            batch_axis_names=batch_axis_names,
            fsdp_axis_names=fsdp_axis_names,
            tp_axis_names=tp_axis_names,
        )

        ff_layer = cfg.feed_forward
        self.assertSequenceEqual(
            ff_layer.linear1.param_partition_spec, (fsdp_axis_names, tp_axis_names)
        )
        self.assertSequenceEqual(
            ff_layer.linear2.param_partition_spec, (tp_axis_names, fsdp_axis_names)
        )
        self.assertSequenceEqual(
            ff_layer.linear1.output_partition_spec, (batch_axis_names, None, tp_axis_names)
        )
        self.assertSequenceEqual(
            ff_layer.linear2.output_partition_spec, (batch_axis_names, None, tp_axis_names)
        )

        self_atten = cfg.self_attention.attention
        # Shard weights.
        self.assertSequenceEqual(
            self_atten.input_linear.layer.param_partition_spec,
            (fsdp_axis_names, tp_axis_names, None),
        )
        self.assertSequenceEqual(
            self_atten.output_linear.param_partition_spec, (fsdp_axis_names, tp_axis_names, None)
        )

        if cross_attention_cfg is None:
            self.assertIsNone(cfg.cross_attention)
        else:
            cross_atten = cfg.self_attention.attention
            # Shard weights.
            self.assertSequenceEqual(
                cross_atten.input_linear.layer.param_partition_spec,
                (fsdp_axis_names, tp_axis_names, None),
            )
            self.assertSequenceEqual(
                cross_atten.output_linear.param_partition_spec,
                (fsdp_axis_names, tp_axis_names, None),
            )


class PositionalEmbeddingTest(TestCase):
    """Tests PositionalEmbedding."""

    def test_learned_positional_embedding_1d(self):
        """
        Simple test that LearnedPositionalEmbedding returns expected outputs for a 1d sequence.
        """
        positions = np.arange(10)
        dim = 8
        pos_emb_cfg = LearnedPositionalEmbedding.default_config().set(
            name="test",
            dim=dim,
            shape=(len(positions),),
        )
        pos_emb = pos_emb_cfg.instantiate(parent=None)

        state = pos_emb.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        outputs, _ = F(
            pos_emb,
            is_training=True,
            prng_key=jax.random.PRNGKey(1),
            state=state,
            inputs={"positions": positions},
        )

        context = InvocationContext(
            name="root",
            parent=None,
            module=pos_emb,
            state=state,
            output_collection=new_output_collection(),
            is_training=True,
            prng_key=jax.random.PRNGKey(2),
        )
        with set_current_context(context):
            embeddings_tensor = pos_emb.embeddings()
            assert embeddings_tensor.shape == (len(positions), dim)

        for position in positions:
            assert_allclose(outputs[position], embeddings_tensor[position])


@pytest.mark.parametrize("x, output", [(300, 512), (127.1, 128), (128, 128), (0.1, 2)])
def test_next_power_of_two(x, output):
    assert _next_power_of_two(x) == output


class BottleNeckAdapterTransformerLayerTest(TestCase):
    """Tests BottleNeckAdapterTransformerLayer."""

    @parameterized.parameters(
        {"bottleneck_ratio": 0.1},
        {"bottleneck_ratio": 0.5},
        {"bottleneck_ratio": 1.0},
    )
    def test_forward(self, bottleneck_ratio):
        batch_size, tgt_len, model_dim, num_heads = 2, 3, 32, 1

        layer_cfg = TransformerLayer.default_config().set(name="layer", input_dim=model_dim)
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.feed_forward.hidden_dim = model_dim

        adapter_cfg = BottleNeckAdapterTransformerLayer.default_config().set(
            input_dim=model_dim, name="adapter", bottleneck_ratio=bottleneck_ratio
        )
        adapter_cfg.layer = layer_cfg

        adapter = adapter_cfg.instantiate(parent=None)

        state = adapter.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        data = jax.random.normal(jax.random.PRNGKey(1), [batch_size, tgt_len, model_dim])
        self_attention_logit_biases = attention.make_causal_mask(tgt_len)

        outputs, _ = F(
            adapter,
            is_training=True,
            prng_key=jax.random.PRNGKey(2),
            state=state,
            inputs=dict(
                data=data,
                self_attention_logit_biases=self_attention_logit_biases,
            ),
        )

        # Output shape is left unchanged.
        assert outputs.data.shape == (2, 3, 32)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
