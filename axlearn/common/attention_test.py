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
#
# facebookresearch/llama:
# Copyright (c) Facebook, Inc. and its affiliates.

"""Tests attention layers."""

import contextlib
import copy
import itertools

# pylint: disable=too-many-lines,duplicate-code,no-self-use
import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Callable, Optional, Union
from unittest import mock

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
from transformers.configuration_utils import PretrainedConfig as HFPretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as HF_ROPE_INIT_FUNCTIONS
from transformers.models.roberta import modeling_roberta as hf_roberta
from transformers.models.roformer import modeling_roformer as hf_roformer
from transformers.models.xlnet import modeling_xlnet as hf_xlnet

from axlearn.common import attention, attention_bias, test_utils, utils
from axlearn.common.attention import (
    BaseQKVLinear,
    BaseStackedTransformerLayer,
    BaseTransformerLayer,
    BottleNeckAdapterTransformerLayer,
    ForwardMode,
    FusedGroupedQKVLinear,
    FusedQKVLinear,
    KVCache,
    KVState,
    LearnedPositionalEmbedding,
    MultiheadAttentionXL,
    MultiheadInputLinear,
    MultiheadOutputLinear,
    MultiheadRelativePositionLinear,
    NormPosition,
    ParallelTransformerLayer,
    PerDimScale,
    PipelinedTransformerLayer,
    QKVLinear,
    QLinear,
    RematRegexSavePatterns,
    RepeatedTransformerLayer,
    RoFormerQKVLinear,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
    TransformerLayer,
    _next_power_of_two,
    apply_attention_logit_biases,
    apply_rotary_position_embeddings,
    build_remat_spec,
    compute_padding_biases,
    rel_pos_to_abs_pos,
    scaled_hidden_dim,
    set_double_shard_weights_config,
    sinusoidal_positional_embeddings,
    update_data_with_skip_connection,
    xl_attention_logits,
)
from axlearn.common.attention_bias import (
    NEG_INF,
    CausalAttentionBias,
    SlidingWindowAttentionBias,
    bool_to_bias,
    causal_mask,
    make_causal_biases,
    make_sliding_window_causal_biases,
)
from axlearn.common.base_layer import (
    BaseLayer,
    DefaultTensorStats,
    FactorizationSpec,
    ParameterSpec,
    RematSpec,
)
from axlearn.common.config import (
    InstantiableConfig,
    UnknownFieldError,
    config_class,
    config_for_function,
    maybe_set_config,
)
from axlearn.common.decoder import Decoder, TransformerTextEmbeddings
from axlearn.common.kv_cache.paged_kv_cache import PagedKVCache
from axlearn.common.kv_cache.sliding_window_kv_cache import enable_sliding_window_attention
from axlearn.common.layers import RMSNorm, set_bias_recursively
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
from axlearn.common.pipeline import BaseSchedule, GPipeSchedule, StreamSchedule
from axlearn.common.quantized_dot_general.layers import (
    DotGeneralQuantizationType,
    QuantizedDotGeneral,
    set_quantized_dot_general_recursively,
)
from axlearn.common.test_utils import (
    TestCase,
    assert_allclose,
    dummy_segments_positions,
    is_supported_mesh_shape,
    set_threefry_partitionable,
)
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import (
    Nested,
    PartitionSpec,
    Tensor,
    TensorSpec,
    VDict,
    as_tensor,
    flatten_items,
    save_and_offload_only_these_names_regex,
    shapes,
)


def all_subsets(given_set):
    "Generate all subsets of a list `given_set`."
    s = list(given_set)
    return list(
        itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
    )


def make_index_position_biases(query_len: int, kv_len: int) -> Tensor:
    """Generates attention logit biases where query indices cannot attend to larger key indices.

    Args:
        query_len: The sequence length.
        kv_len: The key's length.

    Returns:
        A float tensor of shape [query_len, kv_len] where the value at
        [i, j] = -inf if i < j, 0 otherwise.
    """

    return bool_to_bias(
        causal_mask(
            jnp.arange(query_len)[:, None],
            jnp.arange(kv_len)[None, :],
        )
    )


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
        actual = attention_bias.make_causal_biases(3)
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
        actual = attention_bias.make_segment_mask(
            target_segments=jnp.asarray([[1, 1, 2, 0]]),
            source_segments=jnp.asarray([[2, 2, 0, 1]]),
        )
        self.assertTrue(jnp.all(actual <= expected))

    def test_apply_attention_logit_biases(self):
        batch_size = 10
        num_heads = 12
        dim = 32
        logits = jnp.asarray(np.random.random(size=[batch_size, num_heads, dim]))

        # Testing for biases = None
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
        token_ids: list,
        expected: list,
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
        token_ids: list,
        expected: list,
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
                ref_causal_mask = jnp.repeat(
                    make_causal_biases(segment_len)[None, ...], num_heads, 0
                )
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

    @parameterized.product(
        tensor_dimensions=(
            (2, 3, 10, 32),
            (2, 3, 8, 32),
            (2, 4, 6, 32),
            (2, 4, 8, 16),
            (2, 5, 8, 48),
            (2, 5, 8, 64),
        ),
        rotary_key=(True, False),
        rotary_value=(True, False),
    )
    def test_apply_rotary_position_embeddings(
        self, tensor_dimensions: tuple[int, int, int, int], rotary_key: bool, rotary_value: bool
    ):
        # Unittest against the apply_rotary_position_embeddings in HF.
        batch_size, num_heads, max_len, dim = tensor_dimensions

        token_ids = np.random.randint(low=1, high=20, size=[batch_size, max_len])
        sinusoidal_pos_layer = hf_roformer.RoFormerSinusoidalPositionalEmbedding(max_len, dim)
        sinusoidal_pos = sinusoidal_pos_layer(as_torch_tensor(token_ids).shape)[None, :, None, :]
        query = np.random.random([batch_size, max_len, num_heads, dim])
        key = np.random.random([batch_size, max_len, num_heads, dim])
        value = np.random.random([batch_size, max_len, num_heads, dim])
        ref_layer = hf_roformer.RoFormerSelfAttention.apply_rotary_position_embeddings
        test_layer = apply_rotary_position_embeddings
        if rotary_value:
            ref_q_proj, ref_k_proj, ref_v_proj = ref_layer(
                sinusoidal_pos,
                as_torch_tensor(query),
                as_torch_tensor(key),
                as_torch_tensor(value),
            )
        else:
            # If rotary_value is set to False, value keeps unchanged.
            # pylint: disable-next=unbalanced-tuple-unpacking
            ref_q_proj, ref_k_proj = ref_layer(
                sinusoidal_pos, as_torch_tensor(query), as_torch_tensor(key)
            )
            ref_v_proj = as_torch_tensor(value)
        if not rotary_key:
            ref_k_proj = as_torch_tensor(key)

        test_q_proj, test_k_proj, test_v_proj = test_layer(
            sinusoidal_pos=as_tensor(sinusoidal_pos),
            query=query,
            key=key,
            value=value,
            rotary_key=rotary_key,
            rotary_value=rotary_value,
        )
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
        positions = jnp.expand_dims(jnp.arange(token_ids.shape[-1], dtype=jnp.int32), 0)
        ref_layer = hf_roformer.RoFormerSinusoidalPositionalEmbedding(max_len, dim)
        # In recent transformers API, PE's `_init_weight` is called recursively by parent module.
        # Since we only initialize the PE layer here, we need to manually call it.
        ref_layer._init_weight()  # pylint: disable=protected-access, no-value-for-parameter
        ref_output = ref_layer(as_torch_tensor(token_ids).shape)
        # Set up the RoPE AXLearn configs.
        test_layer = (
            attention.RoFormerSinusoidalPositionalEmbedding.default_config()
            .set(name="test_rope_emb", dim=dim)
            .instantiate(parent=None)
        )
        test_output = test_layer.forward(positions=positions)
        np.testing.assert_allclose(np.expand_dims(ref_output, 0), test_output, atol=5e-7)

    @parameterized.parameters(
        (None, True),
        (10, False),
    )
    def test_rope_emb_no_pos(self, max_len, should_raise):
        test_layer = (
            attention.RoFormerSinusoidalPositionalEmbedding.default_config()
            .set(name="test_rope_emb", dim=10)
            .instantiate(parent=None)
        )
        if should_raise:
            with self.assertRaises(ValueError):
                test_layer.forward(max_seq_len=max_len)
        else:
            test_layer.forward(max_seq_len=max_len)

    @parameterized.parameters(
        (2, 10, 32, 4),
    )
    def test_default_rope_emb(self, batch_size, max_len, dim, num_heads):
        rng = np.random.default_rng(seed=123)
        query = jnp.asarray(rng.random([batch_size, max_len, dim]))
        key = jnp.asarray(rng.random([batch_size, max_len, dim]))
        value = jnp.asarray(rng.random([batch_size, max_len, dim]))
        per_head_dim = dim // num_heads

        emb_layer_cfg = attention.RoFormerSinusoidalPositionalEmbedding.default_config().set(
            dim=per_head_dim,
        )
        linear_layer_cfg = attention.RoFormerQKVLinear.default_config().set(
            query_dim=dim,
            key_dim=dim,
            value_dim=dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
            rope_pos_emb_layer=emb_layer_cfg,
            rotary_value=False,
            name="test_rope_linear",
        )
        rope_linear_layer = linear_layer_cfg.instantiate(parent=None)
        state = rope_linear_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        rope_emb_layer = emb_layer_cfg.set(name="test_rope_emb").instantiate(parent=None)
        default_positions = rope_emb_layer.default_query_positions(max_len)

        input_dict = dict(query=query, key=key, value=value)

        layer_outputs_no_position, _ = F(
            rope_linear_layer,
            inputs=input_dict,
            state=state,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        layer_outputs, _ = F(
            rope_linear_layer,
            inputs=dict(**input_dict, query_positions=default_positions),
            state=state,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        # test RoFormerQKVLinear uses default positions in RoFormerSinusoidalPositionalEmbedding
        np.testing.assert_allclose(layer_outputs_no_position, layer_outputs, atol=1e-5)

    def _compare_against_roformer_attention(
        self,
        ref,
        layer,
        tgt_len,
        batch_size,
        ref_rope_emb,
        positions,
    ):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_param_shapes = jax.tree.map(lambda x: x.shape, layer_params)
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
                inputs=dict(
                    target=jnp.asarray(target),
                    attention_logit_biases=mask,
                    target_positions=positions,
                ),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            attn_mask = None if mask is None else as_torch_tensor(mask)
            print("ref_rope_emb", ref_rope_emb.shape)
            print("target", target.shape)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                sinusoidal_pos=ref_rope_emb,
                output_attentions=False,
            )
            assert_allclose(layer_outputs.data, as_tensor(ref_outputs))

    @parameterized.product(rotary_value=[True, False], override_positions=[True, False])
    def test_rope_self_attention(self, rotary_value: bool, override_positions: bool):
        model_dim = 32
        num_heads = 4
        max_sequence_length = 12
        batch_size = 2
        rope_mha_cfg = attention.MultiheadAttention.default_config().set(
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            input_linear=RoFormerQKVLinear.default_config().set(rotary_value=rotary_value),
        )
        rope_emb_layer = (
            attention.RoFormerSinusoidalPositionalEmbedding.default_config()
            .set(name="test_rope_emb", dim=model_dim // num_heads)
            .instantiate(parent=None)
        )
        positions = (
            jax.random.randint(
                jax.random.PRNGKey(0),
                shape=(batch_size, max_sequence_length),
                minval=0,
                maxval=max_sequence_length,
            )
            if override_positions
            else jnp.expand_dims(jnp.arange(max_sequence_length), 0)
        )
        ref_rope_emb = as_torch_tensor(rope_emb_layer.forward(positions=positions)).unsqueeze(1)
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
            rotary_value=rotary_value,
        )
        print(f"roformer_config={roformer_config}")
        ref = hf_roformer.RoFormerAttention(roformer_config)
        self._compare_against_roformer_attention(
            ref,
            layer,
            max_sequence_length,
            batch_size,
            ref_rope_emb,
            positions if override_positions else None,
        )


class YarnScaleRopeParametersTest(TestCase):
    """Tests YarnScaledRoFormerSinusoidalPositionalEmbedding."""

    @parameterized.parameters(
        (2, 10, 32),
        (2, 8, 32),
        (2, 6, 32),
        (2, 8, 16),
        (2, 8, 48),
        (2, 8, 64),
    )
    def test_yarn_emb_basic(self, batch_size, max_len, dim):
        # Here we test yarn embedding within the original max length, which translates to
        # the original RoFormer embedding.
        # Token id is in the np format for easier transition.
        token_ids = np.random.randint(low=1, high=20, size=[batch_size, max_len])
        positions = jnp.expand_dims(jnp.arange(token_ids.shape[-1], dtype=jnp.int32), 0)
        ref_layer = hf_roformer.RoFormerSinusoidalPositionalEmbedding(max_len, dim)
        # In recent transformers API, PE's `_init_weight` is called recursively by parent module.
        # Since we only initialize the PE layer here, we need to manually call it.
        ref_layer._init_weight()  # pylint: disable=protected-access, no-value-for-parameter
        ref_output = ref_layer(as_torch_tensor(token_ids).shape)
        # Set up the RoPE AXLearn configs.
        test_layer = (
            attention.YaRNSinusoidalPositionalEmbedding.default_config()
            .set(name="test_rope_emb", dim=dim, original_max_seq_length=max_len)
            .instantiate(parent=None)
        )
        test_output = test_layer.forward(positions=positions)
        np.testing.assert_allclose(np.expand_dims(ref_output, 0), test_output, atol=5e-7)

    @parameterized.parameters(
        (2, 8, 32, 32, 1.0, 32.0, 10000.0),
        (2, 16, 32, 32, 1.0, 32.0, 10000.0),
        (2, 8, 48, 32, 1.0, 32.0, 10000.0),
        (2, 8, 64, 32, 2.0, 32.0, 10000.0),
        (2, 8, 64, 32, 2.0, 16.0, 10000.0),
        (2, 8, 48, 32, 1.0, 16.0, 10000.0),
    )
    def test_yarn_emb_extend(
        self, batch_size, original_max_len, dim, new_max_len, beta_slow, beta_fast, theta
    ):
        # Here we test yarn embedding within the original max length, which translates to
        # the original RoFormer embedding.
        # Token id is in the np format for easier transition.
        token_ids = np.random.randint(low=1, high=20, size=[batch_size, new_max_len])
        scaling_factor = new_max_len / original_max_len
        ref_config = HFPretrainedConfig(
            rope_theta=theta,
            head_dim=dim,
            hidden_size=dim,
            num_attention_heads=1,
            max_position_embeddings=new_max_len,
            rope_scaling={
                "rope_type": "yarn",
                "factor": scaling_factor,
                "original_max_position_embeddings": original_max_len,
                "beta_slow": beta_slow,
                "beta_fast": beta_fast,
            },
        )
        ref_rope_fn = HF_ROPE_INIT_FUNCTIONS[ref_config.rope_scaling["rope_type"]]
        ref_tokens = as_torch_tensor(token_ids)
        inv_freq, attention_factor = ref_rope_fn(ref_config, device=ref_tokens.device)
        # In recent transformers API, PE's `_init_weight` is called recursively by parent module.
        # Since we only initialize the PE layer here, we need to manually call it.
        # Set up the RoPE AXLearn configs.
        test_layer = (
            attention.YaRNSinusoidalPositionalEmbedding.default_config()
            .set(
                name="test_rope_emb",
                scaling_factor=scaling_factor,
                dim=dim,
                theta=theta,
                original_max_seq_length=original_max_len,
                beta_slow=beta_slow,
                beta_fast=beta_fast,
            )
            .instantiate(parent=None)
        )
        test_output, test_output2 = test_layer.compute_rope_params()
        np.testing.assert_allclose(as_tensor(inv_freq), test_output, atol=5e-7)
        np.testing.assert_allclose(attention_factor, test_output2, atol=5e-7)


def llama_reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """LLaMA reshape for broadcast function.

    Ref:
    https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L55-L60
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1  # pylint: disable=consider-using-in
        for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def llama_apply_rotary_emb(
    *,
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LLaMA apply rotary embeddings to input tensors using the given frequency tensor.

    Ref:
    https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L63-L73
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = llama_reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RefLLaMAAttention(torch.nn.Module):
    """Reference Implementation of LLaMA-1.

    Ref:
    https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L76

    The modifications are removing the dependency of ColumnParallelLinear and RowParallelLinear.
    """

    def __init__(self, n_heads: int, dim: int, max_batch_size: int, max_seq_len: int):
        super().__init__()

        self.n_local_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = torch.nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wk = torch.nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wv = torch.nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )
        self.wo = torch.nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=False,
        )

        self.cache_k = torch.zeros((max_batch_size, max_seq_len, self.n_local_heads, self.head_dim))
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, self.n_local_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = llama_apply_rotary_emb(xq=xq, xk=xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class RoFormerSinusoidalPositionalEmbeddingAgainstLLaMATest(TestCase):
    def llama_ref_precompute_freqs_cis(
        self, *, dim: int, end: int, theta: float = 10000.0
    ) -> torch.Tensor:
        """Reference LLaMA-1 implementation.

        Ref:
        https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47-L52
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    @parameterized.parameters([10000.0, 1000000.0])
    def test_against_llama_for_precompute_freqs_cis(self, theta: float):
        max_len = 100
        dim = 32
        positions = jnp.expand_dims(jnp.arange(max_len), 0)
        axlearn_rope_cfg = attention.RoFormerSinusoidalPositionalEmbedding.default_config().set(
            dim=dim,
            theta=theta,
        )
        axlearn_rope_layer = axlearn_rope_cfg.set(name="rope").instantiate(parent=None)
        axlearn_rope, _ = F(
            axlearn_rope_layer,
            inputs=dict(positions=positions),
            state=axlearn_rope_layer.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(0)
            ),
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        llama_rope = self.llama_ref_precompute_freqs_cis(dim=dim, end=max_len, theta=theta)
        axlearn_imag, axlearn_real = jnp.split(axlearn_rope, 2, axis=-1)
        llama_real, llama_imag = llama_rope.real, llama_rope.imag
        # [0] is added, as axlearn_real and axlearn_imag has a batch_size=1 dimension.
        assert_allclose(llama_real, as_tensor(axlearn_real)[0])
        assert_allclose(llama_imag, as_tensor(axlearn_imag)[0])

    @parameterized.product(
        dtype=(jnp.float32, jnp.bfloat16),
        input_linear=(
            None,
            attention.QKVLinear.default_config(),
            attention.GroupedQKVLinear.default_config(),
        ),
        has_query_positions=(True, False),
        use_query_scale=(True, False),
        use_key_scale=(True, False),
        partial_rope_factor=(1.0, 0.5, None),
    )
    def test_roformer_qkv_linear(
        self,
        dtype: jnp.dtype,
        input_linear: attention.BaseQKVLinear.Config,
        has_query_positions: bool,
        use_query_scale: bool,
        use_key_scale: bool,
        partial_rope_factor: Optional[float],
    ):
        seq_len = 6
        batch_size = 2
        model_dim = 16
        num_heads = 4
        per_head_dim = model_dim // num_heads
        roformer_qkv_linear_kwargs = {
            "name": "roformer_qkv_linear",
            "query_dim": model_dim,
            "key_dim": model_dim,
            "value_dim": model_dim,
            "num_heads": num_heads,
            "per_head_dim": per_head_dim,
            "rotary_value": False,
        }
        num_kv_heads = num_heads
        if input_linear is not None:
            if isinstance(input_linear, attention.GroupedQKVLinear.Config):
                num_kv_heads = num_heads // 2
                input_linear = input_linear.set(num_kv_heads=num_kv_heads)
            roformer_qkv_linear_kwargs["input_linear"] = input_linear

        if use_key_scale:
            roformer_qkv_linear_kwargs["key_scale"] = attention.ScaleKey.default_config()
        if use_query_scale:
            roformer_qkv_linear_kwargs["query_scale"] = attention.ScaleQuery.default_config()
        roformer_qkv_linear_kwargs["partial_rope_factor"] = partial_rope_factor

        roformer_qkv_linear = (
            RoFormerQKVLinear.default_config()
            .set(**roformer_qkv_linear_kwargs)
            .instantiate(parent=None)
        )

        # Check that we see the num kv heads is propagated from child input linear.
        self.assertEqual(roformer_qkv_linear.num_kv_heads, num_kv_heads)

        query = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, model_dim))
        key = jax.random.uniform(jax.random.PRNGKey(2), shape=(batch_size, seq_len, model_dim))
        value = jax.random.uniform(jax.random.PRNGKey(3), shape=(batch_size, seq_len, model_dim))
        roformer_qkv_linear_state = roformer_qkv_linear.initialize_parameters_recursively(
            jax.random.PRNGKey(0)
        )
        input_batch = dict(query=query, key=key, value=value)
        if has_query_positions:
            input_batch["query_positions"] = jax.random.permutation(
                jax.random.PRNGKey(1),
                jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0),
                axis=1,
                independent=True,
            )

        layer_outputs, _ = F(
            roformer_qkv_linear,
            inputs=utils.cast_floats(input_batch, to_dtype=dtype),
            state=utils.cast_floats(roformer_qkv_linear_state, to_dtype=dtype),
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertEqual(layer_outputs.query.dtype, dtype)
        self.assertEqual(layer_outputs.key.dtype, dtype)
        self.assertEqual(layer_outputs.value.dtype, dtype)

    def test_against_llama_for_apply_rotary_emb(self):
        max_len = 100
        dim = 32
        batch_size = 4
        positions = jnp.expand_dims(jnp.arange(max_len), 0)
        axlearn_rope_cfg = attention.RoFormerSinusoidalPositionalEmbedding.default_config().set(
            dim=dim
        )
        axlearn_rope_layer = axlearn_rope_cfg.set(name="rope").instantiate(parent=None)
        axlearn_rope, _ = F(
            axlearn_rope_layer,
            inputs=dict(positions=positions),
            state=axlearn_rope_layer.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(0)
            ),
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        llama_rope = self.llama_ref_precompute_freqs_cis(dim=dim, end=max_len)
        rng = np.random.default_rng(seed=123)
        query = rng.random([batch_size, max_len, dim])
        key = rng.random([batch_size, max_len, dim])
        value = rng.random([batch_size, max_len, dim])
        llama_q, llama_k = llama_apply_rotary_emb(
            xq=torch.Tensor(query), xk=torch.Tensor(key), freqs_cis=llama_rope
        )
        axlearn_q, axlearn_k, _ = attention.apply_rotary_position_embeddings(
            query=jnp.asarray(query)[:, :, None],  # [B, T, D] -> [B, T, 1, D]
            key=jnp.asarray(key)[:, :, None],
            value=jnp.asarray(value)[:, :, None],
            sinusoidal_pos=axlearn_rope[:, :, None],
            rotary_key=True,
            rotary_value=False,
        )

        assert_allclose(as_tensor(llama_q.reshape(*axlearn_q.shape)), axlearn_q, atol=5e-6)
        assert_allclose(as_tensor(llama_k.reshape(*axlearn_k.shape)), axlearn_k, atol=5e-6)

    def test_against_llama_for_attention(self):
        max_len = 100
        dim = 32
        batch_size = 4
        n_heads = 4
        rng = np.random.default_rng(seed=123)
        x = rng.random([batch_size, max_len, dim])
        ref_llama = RefLLaMAAttention(
            n_heads=n_heads, dim=dim, max_batch_size=batch_size, max_seq_len=max_len
        )
        llama_rope = self.llama_ref_precompute_freqs_cis(dim=dim // n_heads, end=max_len)
        llama_output = ref_llama.forward(torch.Tensor(x), 0, llama_rope, mask=None)

        rope_mha_cfg = attention.MultiheadAttention.default_config().set(
            query_dim=dim,
            key_dim=dim,
            value_dim=dim,
            num_heads=n_heads,
            input_linear=RoFormerQKVLinear.default_config().set(
                rotary_value=False,
            ),
        )

        rope_mha = rope_mha_cfg.set(name="rope").instantiate(parent=None)

        state = rope_mha.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        state["i_proj"]["i_proj"]["q_proj"]["weight"] = jnp.asarray(
            ref_llama.wq.weight.transpose(0, 1)
            .reshape(dim, n_heads, dim // n_heads)
            .detach()
            .numpy()
        )
        state["i_proj"]["i_proj"]["k_proj"]["weight"] = jnp.asarray(
            ref_llama.wk.weight.transpose(0, 1)
            .reshape(dim, n_heads, dim // n_heads)
            .detach()
            .numpy()
        )
        state["i_proj"]["i_proj"]["v_proj"]["weight"] = jnp.asarray(
            ref_llama.wv.weight.transpose(0, 1)
            .reshape(dim, n_heads, dim // n_heads)
            .detach()
            .numpy()
        )
        state["o_proj"]["weight"] = jnp.asarray(
            ref_llama.wo.weight.reshape(dim, n_heads, dim // n_heads).detach().numpy()
        )

        axlearn_output, _ = F(
            rope_mha,
            inputs=dict(query=jnp.asarray(x)),
            state=state,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(
            as_tensor(llama_output.reshape(batch_size, max_len, -1)), axlearn_output.data
        )


class MultiheadLinearInitTest(TestCase):
    """Tests MultiheadLinear initialization."""

    def test_unique_config_classes(self):
        self.assertFalse(
            isinstance(MultiheadInputLinear.default_config(), MultiheadOutputLinear.Config)
        )
        self.assertFalse(
            isinstance(MultiheadOutputLinear.default_config(), MultiheadInputLinear.Config)
        )

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
    """Tests QKVLinear, FusedQKVLinear, and associated layers."""

    @parameterized.product(
        test_cls=[
            attention.FusedQKVLinear,
            attention.GroupedQKVLinear,
            attention.FusedGroupedQKVLinear,
        ],
        with_positions=[True, False],
        value_dim_ratio=[1, 2],
        fp8_amax_history_length=[None, 0, 1],
        cross_attention=[True, False],
    )
    def test_qkv_equality(
        self,
        test_cls: type[attention.BaseQKVLinear],
        with_positions: bool,
        value_dim_ratio: int,
        fp8_amax_history_length: int,
        cross_attention: bool,
    ):
        """Tests that the QKVLinear variants are equivalent when num_kv_heads=num_heads."""
        if value_dim_ratio != 1 and test_cls in (
            attention.FusedQKVLinear,
            attention.FusedGroupedQKVLinear,
        ):
            self.skipTest("Fused QKV doesn't support different value dim.")
        if fp8_amax_history_length is not None and jax.default_backend() != "gpu":
            self.skipTest("FP8 is only supported on H100!")
        if cross_attention and test_cls is attention.FusedGroupedQKVLinear:
            self.skipTest("FusedGroupedQKVLinear doesn't support cross attention.")
        if value_dim_ratio != 1 and not cross_attention:
            self.skipTest("Value dim ratio only makes sense for cross attention.")
        with utils.numeric_checks(True):
            model_dim = 12
            num_heads = 4
            per_head_dim = model_dim // num_heads
            layer_kwargs = dict(
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim * value_dim_ratio,
                num_heads=num_heads,
                per_head_dim=per_head_dim,
            )
            base_cfg = QKVLinear.default_config().set(**layer_kwargs)
            test_cfg = test_cls.default_config().set(**layer_kwargs)

            if fp8_amax_history_length is not None:
                fp8_config = QuantizedDotGeneral.default_config().set(
                    quantization_type=DotGeneralQuantizationType.FP_8,
                    fp8_amax_history_length=fp8_amax_history_length,
                )
                set_quantized_dot_general_recursively(
                    base_cfg,
                    quantized_dot_general=fp8_config,
                )
                set_quantized_dot_general_recursively(
                    test_cfg,
                    quantized_dot_general=fp8_config,
                )
            maybe_set_config(test_cfg, num_kv_heads=num_heads)
            base_layer = base_cfg.set(name="base").instantiate(parent=None)
            test_layer = test_cfg.set(name="test").instantiate(parent=None)

            # Construct base layer state.
            base_state = base_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
            test_state = test_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))

            # Map state to fused version.
            if test_cls == attention.FusedQKVLinear:
                weight = jnp.array(
                    [base_state[el]["weight"] for el in ("q_proj", "k_proj", "v_proj")]
                )
                bias = jnp.array([base_state[el]["bias"] for el in ("q_proj", "k_proj", "v_proj")])
                test_state["qkv_proj"].update(weight=weight, bias=bias)
            elif test_cls == attention.FusedGroupedQKVLinear:
                # Concatenate along the num_heads dim.
                weight = jnp.concatenate(
                    [base_state[el]["weight"] for el in ("q_proj", "k_proj", "v_proj")], axis=1
                )
                bias = jnp.concatenate(
                    [base_state[el]["bias"] for el in ("q_proj", "k_proj", "v_proj")], axis=0
                )
                test_state["qkv_proj"].update(weight=weight, bias=bias)
            else:
                for el in ("q_proj", "k_proj", "v_proj"):
                    test_state[el].update(base_state[el])

            # Construct test inputs.
            batch_size, src_len, tgt_len = 2, 6, 6
            query = jax.random.uniform(jax.random.PRNGKey(0), [batch_size, tgt_len, model_dim])
            if cross_attention:
                key = jax.random.uniform(jax.random.PRNGKey(1), [batch_size, src_len, model_dim])
                value = jax.random.uniform(
                    jax.random.PRNGKey(2), [batch_size, src_len, model_dim * value_dim_ratio]
                )
            else:
                key = value = None

            positions = jnp.arange(tgt_len)[None] if with_positions else None
            inputs = dict(query=query, key=key, value=value, query_positions=positions)
            outputs = {}
            layer_names = ("base", "test")
            for name, layer, state in zip(
                layer_names, (base_layer, test_layer), (base_state, test_state)
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
                atol = 0.000001
                if fp8_amax_history_length is not None and (
                    test_cls is FusedGroupedQKVLinear
                    or (test_cls is FusedQKVLinear and not cross_attention)
                ):
                    # When using FP8, the QKV input and weight in QKVLinear has their own
                    # scaling factor. FusedGroupedQKVLinear only has one scaling factor per weight.
                    # FusedQKVLinear has one scaling factor when using self attention, and one per
                    # QKV when using cross attention.
                    # When the number of scaling factor of the test cls differ from the baseline,
                    # we expect higher tolerance.
                    atol = 0.075
                self.assertNestedAllClose(outputs[layer_a], outputs[layer_b], atol=atol)

    @parameterized.parameters(
        dict(layer_cls=attention.QKVLinear, expected=4),
        dict(layer_cls=attention.FusedQKVLinear, expected=4),
        dict(
            layer_cls=attention.QKVLinear,
            num_kv_heads=2,
            expected=UnknownFieldError("num_kv_heads"),
        ),
        dict(
            layer_cls=attention.FusedQKVLinear,
            num_kv_heads=2,
            expected=UnknownFieldError("num_kv_heads"),
        ),
        dict(
            layer_cls=attention.GroupedQKVLinear,
            num_kv_heads=3,
            expected=ValueError("should divide"),
        ),
        dict(
            layer_cls=attention.FusedGroupedQKVLinear,
            num_kv_heads=3,
            expected=ValueError("should divide"),
        ),
        dict(layer_cls=attention.GroupedQKVLinear, num_kv_heads=2, expected=2),
        dict(layer_cls=attention.FusedGroupedQKVLinear, num_kv_heads=2, expected=2),
    )
    def test_num_kv_heads(
        self,
        layer_cls: type[attention.BaseQKVLinear],
        expected: Union[int, Exception],
        num_kv_heads: Optional[int] = None,
    ):
        model_dim = 12
        num_heads = 4
        per_head_dim = model_dim // num_heads
        common_kwargs = dict(
            query_dim=model_dim, key_dim=model_dim, value_dim=model_dim, per_head_dim=per_head_dim
        )
        cfg = layer_cls.default_config().set(name="test", num_heads=num_heads, **common_kwargs)

        if isinstance(expected, Exception):
            ctx = self.assertRaisesRegex(type(expected), str(expected))
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            if num_kv_heads is not None:
                cfg.set(num_kv_heads=num_kv_heads)
            layer = cfg.instantiate(parent=None)
            self.assertEqual(expected, layer.num_kv_heads)

    @parameterized.parameters(
        (QKVLinear.default_config(), QLinear.default_config()),
        (
            RoFormerQKVLinear.default_config().set(
                input_linear=QKVLinear.default_config(), rotary_value=False
            ),
            RoFormerQKVLinear.default_config().set(
                input_linear=QLinear.default_config(), rotary_value=False
            ),
        ),
    )
    def test_qlinear(self, base_cfg, test_cfg):
        """Tests that QLinear is equivalent to QKVLinear with the same kv_state."""
        with utils.numeric_checks(True):
            model_dim = 12
            num_heads = 3
            per_head_dim = model_dim // num_heads
            layer_kwargs = dict(
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                num_heads=num_heads,
                per_head_dim=per_head_dim,
            )
            base_cfg = base_cfg.set(**layer_kwargs)
            test_cfg = test_cfg.set(**layer_kwargs)
            maybe_set_config(test_cfg, num_kv_heads=num_heads)
            base_layer = base_cfg.set(name="base").instantiate(parent=None)
            test_layer = test_cfg.set(name="test").instantiate(parent=None)

            # Construct base layer state.
            base_state = base_layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
            # Map state to QLinear.
            if "q_proj" in base_state:
                test_state = {"q_proj": base_state["q_proj"]}
            elif "i_proj" in base_state:
                test_state = {"i_proj": {"q_proj": base_state["i_proj"]["q_proj"]}}
            else:
                raise ValueError("Cannot find expected q_proj state.")

            # Construct test inputs.
            batch_size, src_len, tgt_len = 2, 6, 6
            query = jax.random.uniform(jax.random.PRNGKey(0), [batch_size, tgt_len, model_dim])
            key = jax.random.uniform(jax.random.PRNGKey(1), [batch_size, src_len, model_dim])
            value = jax.random.uniform(jax.random.PRNGKey(2), [batch_size, src_len, model_dim])

            outputs = {}
            layer_names = ("base", "test")
            kv_kwargs = {"key": key, "value": value}
            for name, layer, state in zip(
                layer_names, (base_layer, test_layer), (base_state, test_state)
            ):
                outputs[name], _ = F(
                    layer,
                    state=state,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(456),
                    inputs=dict(query=query, **kv_kwargs),
                )
                if name == "base":
                    kv_kwargs = {
                        "kv_state": KVState(
                            k_proj=outputs[name].key,
                            v_proj=outputs[name].value,
                            key_positions=jnp.arange(src_len)[None],
                        )
                    }
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


class ScaleQueryTest(TestCase):
    """Tests ScaleQuery."""

    @parameterized.product(
        scale_factor=[None, 7],
        norm=[None, RMSNorm.default_config()],
        per_dim_scale=[
            None,
            PerDimScale.default_config(),
        ],
    )
    def test_scale_query(
        self,
        *,
        scale_factor: Optional[float],
        norm: Optional[RMSNorm.Config],
        per_dim_scale: Optional[PerDimScale.Config],
    ):
        kwargs = self._scale_kwargs(
            scale_factor=scale_factor, norm=norm, per_dim_scale=per_dim_scale
        )
        forward_outputs, _ = F(**kwargs)

        self.assertEqual(forward_outputs.shape, kwargs["inputs"]["proj"].shape)
        q_proj_scaled = kwargs["inputs"]["proj"]
        if norm is not None:
            assert isinstance(norm, RMSNorm.Config)
            moment2 = (q_proj_scaled * q_proj_scaled).mean(axis=-1, keepdims=True)
            q_proj_scaled = q_proj_scaled * jax.lax.rsqrt(moment2 + norm.eps)
        if per_dim_scale is not None:
            assert isinstance(per_dim_scale, PerDimScale.Config)
            # We overrode the initializer for PerDimScale so we can measure the effect.
            q_proj_scaled = q_proj_scaled * jax.nn.softplus(1.0) * 1.442695041

        if scale_factor is None:
            scale_factor = kwargs["module"].config.per_head_dim ** -0.5
        scale_factor = float(scale_factor)
        q_proj_scaled = q_proj_scaled * scale_factor

        self.assertNestedAllClose(forward_outputs, q_proj_scaled)

    def _scale_kwargs(
        self,
        *,
        scale_factor: Union[None, int, float, InstantiableConfig[attention.ScaleFn]],
        norm: Optional[InstantiableConfig],
        per_dim_scale: Optional[PerDimScale.Config],
    ):
        model_dim = 16
        if isinstance(scale_factor, (int, float)):
            scale_factor = config_for_function(attention.constant_scale_fn).set(value=scale_factor)

        num_heads = 2
        per_head_dim = model_dim // num_heads
        if per_dim_scale is not None:
            per_dim_scale = per_dim_scale.set(dim=per_head_dim)

        cfg = attention.ScaleQuery.default_config().set(
            name="test",
            per_head_dim=per_head_dim,
            norm=norm,
            scale_factor=scale_factor,
            per_dim_scale=per_dim_scale,
        )
        layer = cfg.instantiate(parent=None)

        param_specs = layer.create_parameter_specs_recursively()
        layer_params = jax.tree.map(
            lambda spec: jnp.ones(spec.shape, dtype=spec.dtype), param_specs
        )

        batch_size = 3
        tgt_len = 10
        q_proj = jnp.concatenate(
            (
                jnp.ones([batch_size, tgt_len // 2, num_heads, per_head_dim]),
                jnp.zeros([batch_size, tgt_len // 2, num_heads, per_head_dim]),
            ),
            axis=1,
        )
        kwargs = dict(
            module=layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(proj=q_proj, positions=jnp.arange(tgt_len)[None, :]),
        )
        return kwargs


class ScaleKeyTest(TestCase):
    """Tests ScaleKey."""

    @parameterized.product(
        scale_factor=[None, 7],
        norm=[None, RMSNorm.default_config()],
    )
    def test_scale_key(
        self,
        *,
        scale_factor: Optional[float],
        norm: Optional[RMSNorm.Config],
    ):
        kwargs = self._scale_kwargs(scale_factor=scale_factor, norm=norm)
        forward_outputs, _ = F(**kwargs)

        self.assertEqual(forward_outputs.shape, kwargs["inputs"]["proj"].shape)
        q_proj_scaled = kwargs["inputs"]["proj"]
        if norm is not None:
            assert isinstance(norm, RMSNorm.Config)
            moment2 = (q_proj_scaled * q_proj_scaled).mean(axis=-1, keepdims=True)
            q_proj_scaled = q_proj_scaled * jax.lax.rsqrt(moment2 + norm.eps)

        if scale_factor is None:
            scale_factor = 1.0
        scale_factor = float(scale_factor)
        q_proj_scaled = q_proj_scaled * scale_factor
        self.assertNestedAllClose(forward_outputs, q_proj_scaled)

    def _scale_kwargs(
        self,
        *,
        scale_factor: Union[None, int, float, InstantiableConfig[attention.ScaleFn]],
        norm: Optional[InstantiableConfig],
    ):
        model_dim = 16
        if isinstance(scale_factor, (int, float)):
            scale_factor = config_for_function(attention.constant_scale_fn).set(value=scale_factor)

        num_heads = 2
        per_head_dim = model_dim // num_heads

        cfg = attention.ScaleKey.default_config().set(
            name="test",
            per_head_dim=per_head_dim,
            norm=norm,
            scale_factor=scale_factor,
        )
        layer = cfg.instantiate(parent=None)

        param_specs = layer.create_parameter_specs_recursively()
        layer_params = jax.tree.map(
            lambda spec: jnp.ones(spec.shape, dtype=spec.dtype), param_specs
        )

        batch_size = 4
        tgt_len = 12
        k_proj = jnp.concatenate(
            (
                jnp.ones([batch_size, tgt_len // 2, num_heads, per_head_dim]),
                jnp.zeros([batch_size, tgt_len // 2, num_heads, per_head_dim]),
            ),
            axis=1,
        )
        kwargs = dict(
            module=layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            inputs=dict(proj=k_proj, positions=jnp.arange(tgt_len)[None, :]),
        )
        return kwargs


def _convert_to_qkv_linear(
    base_state: Nested[Tensor], *, input_linear_layer_class: type
) -> Nested[Tensor]:
    """Converts the params of a MultiheadAttention layer

    ... to params of a MultiheadAttention layer with input_linear of the given type."""
    test_state = copy.deepcopy(base_state)

    if issubclass(
        input_linear_layer_class, (attention.FusedQKVLinear, attention.FusedGroupedQKVLinear)
    ):

        def combine_qkv(param_name: str) -> Tensor:
            qkv_params = [
                utils.get_recursively(base_state, f"i_proj/{proj}/{param_name}")
                for proj in ("q_proj", "k_proj", "v_proj")
            ]
            if issubclass(input_linear_layer_class, attention.FusedQKVLinear):
                return jnp.stack(qkv_params)
            else:
                return jnp.concatenate(qkv_params, axis=-2)

        qkv_proj = {"weight": combine_qkv("weight")}
        if "bias" in base_state["i_proj"]["q_proj"]:
            qkv_proj["bias"] = combine_qkv("bias")
        test_state["i_proj"] = VDict({"qkv_proj": qkv_proj})

    return test_state


class _QLinearWithKvUpdate(QLinear):
    """QLinear that adjusts the external KV for testing purposes."""

    def forward(
        self,
        query: Tensor,
        *,
        kv_state: KVState,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_positions: Optional[Tensor] = None,
    ) -> BaseQKVLinear.Output:
        query, key, value = super().forward(
            query, kv_state=kv_state, key=key, value=value, query_positions=query_positions
        )
        key = key + 1.0
        value = value - 1.0
        return BaseQKVLinear.Output(query, key, value)


class MultiheadAttentionTest(TestCase):
    """Tests MultiheadAttention, GroupedQueryAttention, and associated layers."""

    def test_add_tensor_stats(self):
        model_dim = 12
        num_heads = 4
        cfg = attention.MultiheadAttention.default_config().set(
            name="attn",
            query_dim=12,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            tensor_stats=DefaultTensorStats.default_config(),
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        batch_size, src_len, tgt_len = 2, 6, 6
        rng = np.random.default_rng(seed=123)
        query = jnp.asarray(rng.random([batch_size, tgt_len, model_dim]))
        key = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        value = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        attention_logit_biases = jnp.ones([batch_size, tgt_len, src_len]) * NEG_INF
        x = dict(query=query, key=key, value=value, attention_logit_biases=attention_logit_biases)
        _, output_collection = F(
            layer,
            inputs=x,
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        if "tensor_stats" in output_collection.summaries:
            output_stats = output_collection.summaries["tensor_stats"]
        else:
            output_stats = {}
        expected_stats = ["o_proj_outputs"]
        for k in expected_stats:
            assert k in output_stats

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
                query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
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
                query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
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
            expected_scale_query_params = {}
            if per_dim_scale:
                expected_scale_query_params["per_dim_scale"] = dict(param=(per_head_dim,))
            expected_params = {
                "i_proj": {f"{x}_proj": qkv_shapes for x in ("q", "k", "v")},
                "o_proj": dict(weight=(model_dim, num_heads, per_head_dim), bias=(model_dim,)),
                "kv_cache": {},
                "dropout": {},
                "scale_key": {},
                "scale_query": expected_scale_query_params,
            }
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
            query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
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
        base_cfg=(
            attention.MultiheadAttention.default_config(),
            attention.GroupedQueryAttention.default_config().set(
                input_linear=attention.GroupedQKVLinear.default_config().set(num_kv_heads=2)
            ),
            attention.GroupedQueryAttention.default_config().set(
                input_linear=attention.FusedGroupedQKVLinear.default_config().set(num_kv_heads=2)
            ),
            attention.GroupedQueryAttention.default_config().set(
                input_linear=attention.RoFormerQKVLinear.default_config().set(rotary_value=False)
            ),
            attention.SigmoidAttention.default_config().set(
                input_linear=attention.RoFormerQKVLinear.default_config().set(rotary_value=False),
                seq_len=4,
            ),
            attention.SigmoidAttention.default_config().set(
                # Used in ALiBi position encoding.
                input_linear=FusedQKVLinear.default_config(),
                seq_len=4,
            ),
        ),
        attention_logit_biases_fn=(
            lambda query_len, kv_len: None,
            lambda query_len, kv_len: _random_mask(jax.random.PRNGKey(1), query_len, kv_len),
        ),
        kv_length_multiplier=(0.5, 1, 2),
        has_query_positions=(False, True),
    )
    def test_causal(
        self,
        base_cfg: attention.MultiheadAttention.Config,
        attention_logit_biases_fn: Callable[[int, int], Tensor],
        kv_length_multiplier: float,
        has_query_positions: bool,
    ):
        """Tests that base_cfg(causal=True) is equivalent to applying a causal mask."""
        if (
            has_query_positions
            and not isinstance(base_cfg.input_linear, RoFormerQKVLinear.Config)
            or kv_length_multiplier != 1
            and isinstance(
                base_cfg.input_linear,
                (FusedGroupedQKVLinear.Config, RoFormerQKVLinear.Config, FusedQKVLinear.Config),
            )
        ):
            self.skipTest("Incompatible test setting that does not need testing.")

        model_dim = 16
        num_heads = 4
        ref_cfg = base_cfg.clone(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )
        self.assertFalse(ref_cfg.causal)
        ref_layer = ref_cfg.instantiate(parent=None)
        layer_params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        test_cfg = ref_cfg.clone(causal=True)
        test_layer = test_cfg.instantiate(parent=None)

        batch_size, query_len = 2, 4
        query = jnp.zeros([batch_size, query_len, model_dim], dtype=jnp.float32)
        outputs = []

        if has_query_positions:
            query_positions = jax.random.permutation(
                jax.random.PRNGKey(1),
                jnp.arange(query_len)[None, :].repeat(batch_size, axis=0),
                axis=1,
                independent=True,
            )

        for layer in (ref_layer, test_layer):
            inputs = dict(query=query)
            kv_len = int(kv_length_multiplier * query_len)
            if kv_length_multiplier < 1:
                inputs["key"] = query[:, :kv_len]
                inputs["value"] = query[:, :kv_len]
            elif kv_length_multiplier > 1:
                inputs["key"] = jnp.tile(query, [1, int(kv_length_multiplier), 1])
                inputs["value"] = jnp.tile(query, [1, int(kv_length_multiplier), 1])

            attention_logit_biases = attention_logit_biases_fn(inputs["query"].shape[1], kv_len)
            if layer is ref_layer:
                # Apply causal mask on top of the logit biases for `ref_layer`.
                causal_biases = make_index_position_biases(inputs["query"].shape[1], kv_len=kv_len)
                if attention_logit_biases is None:
                    attention_logit_biases = causal_biases
                else:
                    attention_logit_biases = apply_attention_logit_biases(
                        attention_logit_biases, causal_biases
                    )
            inputs["attention_logit_biases"] = attention_logit_biases
            if has_query_positions:
                inputs["query_positions"] = query_positions

            layer_outputs, _ = F(
                layer,
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
            )
            outputs.append(layer_outputs)
        # The outputs are equivalent.
        self.assertNestedAllClose(outputs[0], outputs[1])

    @parameterized.product(
        base_cfg=(
            attention.MultiheadAttention.default_config(),
            attention.GroupedQueryAttention.default_config().set(
                input_linear=attention.GroupedQKVLinear.default_config().set(num_kv_heads=2)
            ),
            attention.GroupedQueryAttention.default_config().set(
                input_linear=attention.FusedGroupedQKVLinear.default_config().set(num_kv_heads=2)
            ),
            attention.GroupedQueryAttention.default_config().set(
                input_linear=attention.RoFormerQKVLinear.default_config().set(rotary_value=False)
            ),
            attention.SigmoidAttention.default_config().set(
                input_linear=attention.RoFormerQKVLinear.default_config().set(rotary_value=False),
                seq_len=4,
            ),
            attention.SigmoidAttention.default_config().set(
                # Used in ALiBi position encoding.
                input_linear=FusedQKVLinear.default_config(),
                seq_len=4,
            ),
        ),
        attention_logit_biases_fn=(
            lambda seq_len: None,
            lambda seq_len: _random_mask(jax.random.PRNGKey(1), seq_len, seq_len),
        ),
        has_query_positions=(False, True),
    )
    def test_sliding_window(
        self,
        base_cfg: attention.MultiheadAttention.Config,
        attention_logit_biases_fn: Callable[[int], Tensor],
        has_query_positions: bool,
    ):
        """
        Tests that base_cfg with sliding window causal mask fns is equivalent to applying a
        causal sliding window mask.
        """
        if has_query_positions and not isinstance(base_cfg.input_linear, RoFormerQKVLinear.Config):
            return

        model_dim = 16
        num_heads = 4
        ref_cfg = base_cfg.clone(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )
        self.assertFalse(ref_cfg.causal)
        ref_layer = ref_cfg.instantiate(parent=None)
        layer_params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        sliding_window_size = 2
        test_cfg = enable_sliding_window_attention(ref_cfg, sliding_window_size)
        test_layer = test_cfg.instantiate(parent=None)

        batch_size, seq_len = 2, 4
        query = jnp.zeros([batch_size, seq_len, model_dim], dtype=jnp.float32)
        outputs = []

        if has_query_positions:
            query_positions = jax.random.permutation(
                jax.random.PRNGKey(1),
                jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0),
                axis=1,
                independent=True,
            )

        for layer in (ref_layer, test_layer):
            attention_logit_biases = attention_logit_biases_fn(seq_len)
            if layer is ref_layer:
                # Apply causal and sliding window mask on top of the logit biases for `ref_layer`.
                attention_logit_biases = apply_attention_logit_biases(
                    make_sliding_window_causal_biases(seq_len, sliding_window_size),
                    attention_logit_biases,
                )
            inputs = dict(query=query, attention_logit_biases=attention_logit_biases)
            if has_query_positions:
                inputs["query_positions"] = query_positions
            layer_outputs, _ = F(
                layer,
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
            )
            outputs.append(layer_outputs)
        # The outputs are equivalent.
        self.assertNestedAllClose(outputs[0], outputs[1])

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
        atten_logit_cap=(0.0, 20.0),
        input_linear=(
            None,  # Use the default linear.
            attention.QKVLinear.default_config(),
            attention.FusedQKVLinear.default_config(),
            attention.GroupedQKVLinear.default_config().set(num_kv_heads=4),
            attention.FusedGroupedQKVLinear.default_config().set(num_kv_heads=4),
        ),
        bias=(True, False),
        use_legacy_attention_logit_biases=(True, False),
    )
    def test_gqa_forward(
        self,
        dtype: jnp.dtype,
        per_dim_scale: Optional[PerDimScale.Config],
        atten_logit_cap: float,
        input_linear: attention.BaseQKVLinear.Config,
        bias: bool,
        use_legacy_attention_logit_biases: bool,
    ):
        """When num_kv_heads=num_heads, GQA should be equivalent to MHA."""
        model_dim = 16
        num_heads = 4
        layer_kwargs = dict(
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
            atten_logit_cap=atten_logit_cap,
            dtype=dtype,
        )
        init_key = jax.random.PRNGKey(123)
        # Initialize MultiheadAttention.
        base_cfg = attention.MultiheadAttention.default_config().set(**layer_kwargs)
        set_bias_recursively(base_cfg, bias=bias)
        base_layer = base_cfg.set(name="base").instantiate(parent=None)
        base_state = base_layer.initialize_parameters_recursively(prng_key=init_key)
        # Initialize GroupedQueryAttenion.
        mask = None if use_legacy_attention_logit_biases else CausalAttentionBias.default_config()
        cfg = attention.GroupedQueryAttention.default_config().set(**layer_kwargs, mask=mask)
        if input_linear is not None:
            cfg.set(input_linear=input_linear)
        set_bias_recursively(cfg, bias=bias)
        test_layer = cfg.set(name="test").instantiate(parent=None)
        logging.info("base_state=%s", shapes(base_state))
        # We convert 'base_state' to 'test_state' because JAX does not ensure that RNG behavior
        # remains the same with vs. without vmap. So test_layer initialization may behave
        # differently even with the same seed.
        test_state = _convert_to_qkv_linear(
            base_state, input_linear_layer_class=cfg.input_linear.klass
        )
        logging.info("transformed_test_state=%s", shapes(test_state))

        # Dummy inputs.
        batch_size, tgt_len = 2, 6
        base_inputs = dict(
            query=jax.random.normal(
                jax.random.PRNGKey(124),
                [batch_size, tgt_len, model_dim],
                dtype=dtype,
            ),
            key=None,
            value=None,
            attention_logit_biases=attention_bias.make_causal_biases(tgt_len),
        )
        test_inputs = base_inputs.copy()
        if not use_legacy_attention_logit_biases:
            test_inputs["attention_logit_biases"] = None
        # Get outputs.
        forward_key = jax.random.PRNGKey(456)
        base_outputs, _ = F(
            base_layer,
            state=base_state,
            is_training=False,
            prng_key=forward_key,
            inputs=base_inputs,
        )
        test_outputs, _ = F(
            test_layer,
            state=test_state,
            is_training=False,
            prng_key=forward_key,
            inputs=test_inputs,
        )
        self.assertNestedAllClose(base_outputs, test_outputs)

    @parameterized.product(kv_part=[None, ("fsdp", None, "model", None)])
    @pytest.mark.for_8_devices
    def test_qkvo_partition_spec(self, kv_part):
        """Tests that QKVO partition spec are applied correctly when specified."""
        mesh_shape = (2, 2, 2)
        if not is_supported_mesh_shape(mesh_shape):
            self.skipTest(f"Unsupported mesh shape {mesh_shape}")
        model_dim = 16
        num_heads = 4
        mesh = jax.make_mesh(mesh_shape, axis_names=("fsdp", "seq", "model"))
        q_part = ("fsdp", "seq", "model", None)
        o_part = ("fsdp", "seq", None)

        layer_kwargs = dict(
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            dtype=jnp.float32,
            q_partition_spec=q_part,
            o_partition_spec=o_part,
            k_partition_spec=kv_part,
            v_partition_spec=kv_part,
        )
        init_key = jax.random.PRNGKey(123)
        base_cfg = attention.MultiheadAttention.default_config().set(**layer_kwargs)
        base_layer = base_cfg.set(name="base").instantiate(parent=None)
        base_state = base_layer.initialize_parameters_recursively(prng_key=init_key)

        # Dummy inputs.
        batch_size, tgt_len = 2, 6
        base_inputs = dict(
            query=jax.random.normal(
                jax.random.PRNGKey(124),
                [batch_size, tgt_len, model_dim],
                dtype=jnp.float32,
            ),
            key=None,
            value=None,
        )
        forward_key = jax.random.PRNGKey(456)

        def patched_remat_name(_, tensor, name):
            def callback(sharding):
                # pylint: disable-next=protected-access
                normalize_spec = sharding.spec._normalized_spec_for_aval(len(tensor.shape))
                if name == "q_proj":
                    self.assertEqual(normalize_spec, PartitionSpec(*q_part))
                elif name == "o_proj":
                    self.assertEqual(normalize_spec, PartitionSpec(*o_part))
                elif name in ["k_proj", "v_proj"]:
                    if kv_part is None:
                        self.assertEqual(normalize_spec, PartitionSpec(*q_part))
                    else:
                        self.assertEqual(normalize_spec, PartitionSpec(*kv_part))

            jax.debug.inspect_array_sharding(tensor, callback=callback)
            return tensor

        with (
            mesh,
            mock.patch.object(attention.MultiheadAttention, "_remat_name", patched_remat_name),
        ):

            @jax.jit
            def jit_fn():
                base_outputs, _ = F(
                    base_layer,
                    state=base_state,
                    is_training=False,
                    prng_key=forward_key,
                    inputs=base_inputs,
                )
                return base_outputs

            jit_fn()

    def _test_extend_step(
        self,
        attention_cfg: attention.MultiheadAttention.Config,
        *,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dtype: jnp.dtype,
        bias: bool,
        extend_step_len: int,
        page_size: Optional[int],  # None means not using PagedKVCache
    ):
        cfg = attention_cfg.set(
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            dtype=dtype,
        )
        set_bias_recursively(cfg, bias=bias)
        layer: attention.MultiheadAttention = cfg.set(name="test").instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size = 2
        tgt_len = 6 if page_size is None else max(6, page_size * 3)
        head_dim = model_dim // num_heads
        query = jax.random.normal(
            jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
        )
        key = value = kv_state = None
        if attention_cfg.klass == attention.GroupedQueryAttention:
            pass
        elif attention_cfg.input_linear.klass in (QLinear, _QLinearWithKvUpdate):
            kv_state = KVState(
                k_proj=jax.random.normal(
                    jax.random.PRNGKey(124), [batch_size, tgt_len, num_heads, head_dim], dtype=dtype
                ),
                v_proj=jax.random.normal(
                    jax.random.PRNGKey(125), [batch_size, tgt_len, num_heads, head_dim], dtype=dtype
                ),
                key_positions=jnp.arange(tgt_len)[None],
            )
        else:
            # Make key and value distinct from query. Otherwise, it is equivalent
            # to the query only case.
            key = value = query + 0.1
        return_aux = {"probs", "kv_state"}
        inputs = dict(
            query=query,
            key=key,
            value=value,
            kv_state=kv_state,
            return_aux=return_aux,
        )
        forward_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=inputs,
        )

        initial_state, initial_output = layer.init_states(
            time_step=None,
            query=TensorSpec([batch_size, tgt_len], dtype=dtype),
            kv_state=kv_state,
        )
        if page_size is not None:
            # populate the kv_pages and page_indices
            max_pages_each_request = (tgt_len + page_size - 1) // page_size
            num_global_pages = batch_size * max_pages_each_request
            for k in ["key", "value"]:
                initial_state["kv_cache"][k] = jnp.zeros(
                    shape=[num_kv_heads, num_global_pages, page_size, head_dim]
                ).astype(dtype)
            initial_state["kv_cache"]["page_indices"] = jnp.arange(num_global_pages).reshape(
                (batch_size, max_pages_each_request)
            )

        self.assertIsNone(initial_output)
        if kv_state is None:
            for k in ["key", "value"]:
                # Check that the cache dtype is inferred as the layer dtype.
                self.assertEqual(initial_state["kv_cache"][k].dtype, dtype)
        else:
            self.assertNotIn("kv_cache", initial_state)
        inputs = dict(cached_states=initial_state, kv_state=kv_state, return_aux=return_aux)
        decoder_output = []
        decoder_probs = []
        for t in range(0, tgt_len, extend_step_len):
            inputs["query"] = query[:, t : t + extend_step_len, :]
            if key is not None:
                inputs["key"] = key[:, t : t + extend_step_len, :]
            if value is not None:
                inputs["value"] = value[:, t : t + extend_step_len, :]
            (cached_states, extend_step_outputs), _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
                method="extend_step",
            )
            inputs["cached_states"] = cached_states
            decoder_output.append(extend_step_outputs.data)
            decoder_probs.append(extend_step_outputs.probs)
        decoder_output = jnp.concatenate(decoder_output, axis=1)
        assert_allclose(
            decoder_output, forward_outputs.data, atol=1e-6 if dtype == jnp.float32 else 1e-3
        )
        # When not using KVCache, the source positions are not always full positions.
        # e.g., SlidingWindowKVCache returns probs for only sliding window size.
        if cfg.kv_cache is None or cfg.kv_cache.klass in [KVCache, PagedKVCache]:
            decoder_probs = jnp.concatenate(decoder_probs, axis=2)
            assert decoder_output.shape == forward_outputs.data.shape
            assert decoder_probs.shape == forward_outputs.probs.shape
            assert_allclose(decoder_probs, forward_outputs.probs, atol=1e-6)
            test_k_proj, test_v_proj = layer.kv_cache.maybe_normalize_kv(
                extend_step_outputs.kv_state
            )
            self.assertNestedAllClose(test_k_proj, forward_outputs.kv_state.k_proj, atol=1e-6)
            self.assertNestedAllClose(test_v_proj, forward_outputs.kv_state.v_proj, atol=1e-6)

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
        atten_logit_cap=(0.0, 20.0),
        input_linear=(QKVLinear, RoFormerQKVLinear, QLinear, _QLinearWithKvUpdate),
        bias=(True, False),
        causal_type=("causal", "sliding_window"),
        extend_step_len=(1, 4),
        scale_kv_before_cache_update=(False, True),
        page_size=(None, 16),  # None means not using PagedKVCache
    )
    def test_extend_step(
        self,
        dtype: jnp.dtype,
        per_dim_scale: Optional[PerDimScale.Config],
        atten_logit_cap: float,
        input_linear: attention.BaseQKVLinear,
        bias: bool,
        causal_type: str,
        extend_step_len: int,
        scale_kv_before_cache_update: bool,
        page_size: Optional[int],
    ):
        if input_linear in (QLinear, _QLinearWithKvUpdate):
            if causal_type == "sliding_window":
                self.skipTest("QLinear variants don't support sliding window mask.")
            if scale_kv_before_cache_update:
                self.skipTest("QLinear variants don't support scale_kv_before_cache_update=True")
        if page_size is not None:
            if extend_step_len > 1:
                self.skipTest("PagedKVCache doesn't support extending multiple steps yet.")
            if input_linear in (QLinear, _QLinearWithKvUpdate):
                self.skipTest("PagedKVCache doesn't support QLinear yet.")
        model_dim = 16
        num_heads = 4
        if input_linear == attention.RoFormerQKVLinear:
            input_linear = input_linear.default_config().set(rotary_value=False)
        else:
            input_linear = input_linear.default_config()

        kv_cache_class = PagedKVCache if page_size is not None else KVCache
        cfg = attention.MultiheadAttention.default_config().set(
            query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
            atten_logit_cap=atten_logit_cap,
            input_linear=input_linear,
            kv_cache=kv_cache_class.default_config().set(cache_dtype=dtype),
            scale_kv_before_cache_update=scale_kv_before_cache_update,
        )
        if causal_type == "causal":
            cfg.mask = CausalAttentionBias.default_config()
        elif causal_type == "sliding_window":
            if page_size is not None:
                cfg.mask = SlidingWindowAttentionBias.default_config(sliding_window_size=4)
            else:
                cfg = enable_sliding_window_attention(cfg, sliding_window_size=4)
        else:
            raise ValueError(f"{causal_type} is not supportd.")
        self._test_extend_step(
            cfg,
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            dtype=dtype,
            bias=bias,
            extend_step_len=extend_step_len,
            page_size=page_size,
        )

    @parameterized.parameters(False, True)
    def test_prescaled_kv_share(self, scale_kv_before_cache_update: bool):
        """Tests that we always output the pre-scaled kv_state when input kv_state is set.

        That is, the output kv_state should be equal to the input kv_state if it's set.
        """
        model_dim = 16
        num_heads = 4
        dtype = jnp.float32
        batch_size = 2
        tgt_len = 6
        head_dim = model_dim // num_heads

        cfg = attention.MultiheadAttention.default_config().set(
            input_linear=QLinear.default_config(),
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            dtype=dtype,
            scale_kv_before_cache_update=scale_kv_before_cache_update,
        )
        cfg.key_scale.norm = RMSNorm.default_config().set(input_dim=head_dim)

        layer = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        query = jax.random.normal(
            jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
        )
        kv_state = KVState(
            k_proj=jax.random.normal(
                jax.random.PRNGKey(124), [batch_size, tgt_len, num_heads, head_dim], dtype=dtype
            ),
            v_proj=jax.random.normal(
                jax.random.PRNGKey(125), [batch_size, tgt_len, num_heads, head_dim], dtype=dtype
            ),
            key_positions=jnp.arange(tgt_len)[None],
        )
        return_aux = {"probs", "kv_state"}
        inputs = dict(
            query=query,
            key=None,
            value=None,
            kv_state=kv_state,
            return_aux=return_aux,
        )
        with (
            self.assertRaises(ValueError)
            if scale_kv_before_cache_update
            else contextlib.nullcontext()
        ):
            forward_outputs, _ = F(
                layer,
                state=layer_params,
                is_training=False,
                prng_key=jax.random.PRNGKey(456),
                inputs=inputs,
            )
        if scale_kv_before_cache_update:
            return
        self.assertNestedEqual(forward_outputs.kv_state.k_proj, kv_state.k_proj)
        self.assertNestedEqual(forward_outputs.kv_state.v_proj, kv_state.v_proj)

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
        atten_logit_cap=(0.0, 20.0),
        num_kv_heads=(1, 2, 4),
        input_linear=(attention.GroupedQKVLinear, attention.FusedGroupedQKVLinear),
        bias=(True, False),
        causal_type=("causal", "sliding_window"),
        extend_step_len=(1, 4),
        page_size=(None, 1, 16),  # None means not using PagedKVCache.
    )
    def test_gqa_extend_step(
        self,
        dtype: jnp.dtype,
        per_dim_scale: Optional[PerDimScale.Config],
        atten_logit_cap: float,
        num_kv_heads: int,
        input_linear: type[attention.BaseQKVLinear],
        bias: bool,
        causal_type: str,
        extend_step_len: int,
        page_size: Optional[int],
    ):
        if page_size is not None:
            if extend_step_len > 1:
                self.skipTest("PagedKVCache doesn't support extending multiple steps.")
        model_dim = 16
        num_heads = 4
        kv_cache_class = PagedKVCache if page_size is not None else KVCache
        cfg = attention.GroupedQueryAttention.default_config().set(
            query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
            atten_logit_cap=atten_logit_cap,
            input_linear=input_linear.default_config().set(num_kv_heads=num_kv_heads),
            kv_cache=kv_cache_class.default_config().set(cache_dtype=dtype),
        )
        if causal_type == "causal":
            cfg.mask = CausalAttentionBias.default_config()
        elif causal_type == "sliding_window":
            if page_size is not None:
                cfg.mask = SlidingWindowAttentionBias.default_config(sliding_window_size=4)
            else:
                cfg = enable_sliding_window_attention(cfg, sliding_window_size=4)
        else:
            raise ValueError(f"{causal_type} is not supportd.")
        self._test_extend_step(
            cfg,
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            bias=bias,
            extend_step_len=extend_step_len,
            page_size=page_size,
        )

    def _test_prefill_states(
        self,
        attention_cfg: attention.MultiheadAttention.Config,
        *,
        model_dim: int,
        num_heads: int,
        dtype: jnp.dtype,
        bias: bool,
        num_kv_heads: Optional[int] = None,
    ):
        cfg = attention_cfg.set(
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            dtype=dtype,
        )
        set_bias_recursively(cfg, bias=bias)
        layer: attention.MultiheadAttention = cfg.set(name="test").instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size, tgt_len = 3, 6
        query = jax.random.normal(
            jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
        )
        if attention_cfg.klass == attention.GroupedQueryAttention:
            key = value = None
        else:
            # Make key and value distinct from query. Otherwise, it is equivalent
            # to the query only case.
            key = value = query + 0.1
        return_aux = {"probs"}

        forward_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(
                query=query,
                key=key,
                value=value,
                return_aux=return_aux,
            ),
        )

        time_step = jnp.arange(batch_size)
        (initial_states, initial_output), _ = F(
            layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(
                time_step=time_step,
                query=query,
                key=key,
                value=value,
                return_aux=return_aux,
            ),
            method="init_states",
        )

        # Check time_step and shapes of state.
        self.assertEqual(["time_step", "kv_cache"], list(initial_states.keys()))
        self.assertTrue(jnp.all(time_step == initial_states["time_step"]))

        # Zero-out outputs starting from initial time_step, and test that we can recover the full
        # outputs by calling extend_step starting from time_step.
        # [batch, tgt_len].
        time_step_mask = jnp.arange(tgt_len) < time_step[:, None]
        # [batch, tgt_len, model_dim].
        decoder_output = initial_output.data * time_step_mask[..., None]
        # [batch, tgt_len, model_dim] --> [batch, model_dim, tgt_len].
        decoder_output = jnp.moveaxis(decoder_output, -2, -1)

        # When not using KVCache, the source positions are not always full positions.
        # e.g., SlidingWindowKVCache returns probs for only sliding window size.
        vanilla_kvcache = cfg.kv_cache is None or cfg.kv_cache.klass in [KVCache, PagedKVCache]
        if vanilla_kvcache:
            for proj in ["key", "value"]:
                self.assertEqual(
                    (batch_size, num_kv_heads or num_heads, model_dim // num_heads, tgt_len),
                    initial_states["kv_cache"][proj].shape,
                )
                self.assertEqual(
                    dtype,
                    initial_states["kv_cache"][proj].dtype,
                )

        # [batch, num_heads, tgt_len, src_len].
        if initial_output.probs is not None and vanilla_kvcache:
            decoder_probs = initial_output.probs * time_step_mask[:, None, :, None]
            # [batch, num_heads, tgt_len, src_len] --> [batch, num_heads, src_len, tgt_len].
            decoder_probs = jnp.moveaxis(decoder_probs, -2, -1)
        else:
            decoder_probs = None

        # Call extend_step from time_step, ensuring that outputs match.
        inputs = dict(cached_states=initial_states, return_aux=return_aux)
        while jnp.any(time_step < tgt_len):
            # [batch, tgt_len=1, model_dim].
            inputs["query"] = jnp.take_along_axis(
                query, time_step[:, None, None], axis=1, mode="clip"
            )
            if key is not None:
                inputs["key"] = jnp.take_along_axis(
                    key, time_step[:, None, None], axis=1, mode="clip"
                )
                inputs["value"] = jnp.take_along_axis(
                    value, time_step[:, None, None], axis=1, mode="clip"
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
            if decoder_probs is not None:
                # [batch, 1, 1, tgt_len].
                oh_indices = oh_indices[..., None, :]
                decoder_probs = decoder_probs + curr_probs * oh_indices
            time_step = time_step + 1

        # [batch, model_dim, tgt_len] --> [batch, tgt_len, model_dim].
        decoder_output = jnp.moveaxis(decoder_output, -1, -2)
        assert_allclose(decoder_output, forward_outputs.data)
        if decoder_probs is not None:
            # [batch, num_heads, src_len, tgt_len] --> [batch, num_heads, tgt_len, src_len].
            decoder_probs = jnp.moveaxis(decoder_probs, -1, -2)
            assert_allclose(decoder_probs, forward_outputs.probs)

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
        atten_logit_cap=(0.0, 20.0),
        bias=(True, False),
        causal_type=("causal", "sliding_window"),
        input_linear=(attention.QKVLinear, attention.RoFormerQKVLinear),
    )
    def test_prefill_states(
        self,
        dtype: jnp.dtype,
        per_dim_scale: Optional[PerDimScale.Config],
        atten_logit_cap: float,
        bias: bool,
        causal_type: str,
        input_linear: attention.BaseQKVLinear,
    ):
        model_dim = 16
        num_heads = 4
        if input_linear == attention.RoFormerQKVLinear:
            input_linear = input_linear.default_config().set(rotary_value=False)
        else:
            input_linear = input_linear.default_config()
        cfg = attention.MultiheadAttention.default_config().set(
            query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
            atten_logit_cap=atten_logit_cap,
            input_linear=input_linear,
        )
        if causal_type == "causal":
            cfg.mask = CausalAttentionBias.default_config()
        elif causal_type == "sliding_window":
            cfg = enable_sliding_window_attention(cfg, sliding_window_size=4)
        else:
            raise ValueError(f"{causal_type} is not supportd.")
        self._test_prefill_states(
            cfg, model_dim=model_dim, num_heads=num_heads, dtype=dtype, bias=bias
        )

    @parameterized.product(
        dtype=(jnp.float32, jnp.float16, jnp.bfloat16),
        per_dim_scale=(None, PerDimScale.default_config()),
        atten_logit_cap=(0.0, 20.0),
        num_kv_heads=(1, 2, 4),
        input_linear=(attention.GroupedQKVLinear, attention.FusedGroupedQKVLinear),
        bias=(True, False),
        causal_type=("causal", "sliding_window"),
    )
    def test_gqa_prefill_states(
        self,
        dtype: jnp.dtype,
        per_dim_scale: Optional[PerDimScale.Config],
        atten_logit_cap: float,
        num_kv_heads: int,
        input_linear: type[attention.BaseQKVLinear],
        bias: bool,
        causal_type: str,
    ):
        model_dim = 16
        num_heads = 4
        cfg = attention.GroupedQueryAttention.default_config().set(
            query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
            atten_logit_cap=atten_logit_cap,
            input_linear=input_linear.default_config().set(num_kv_heads=num_kv_heads),
        )
        if causal_type == "causal":
            cfg.mask = CausalAttentionBias.default_config()
        elif causal_type == "sliding_window":
            cfg = enable_sliding_window_attention(cfg, sliding_window_size=4)
        else:
            raise ValueError(f"{causal_type} is not supportd.")
        self._test_prefill_states(
            cfg,
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
            bias=bias,
        )

    @parameterized.product(
        mode=[ForwardMode.FORWARD, ForwardMode.EXTEND_STEP],
        kv_dtype=[jnp.float64, jnp.float32, jnp.bfloat16],
    )
    def test_gqa_against_mha(self, mode, kv_dtype):
        model_dim = 16
        num_heads = 4
        num_kv_heads = 2
        ref_cfg = attention.MultiheadAttention.default_config().set(
            name="mha",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )
        ref_layer = ref_cfg.instantiate(parent=None)

        test_cfg = attention.GroupedQueryAttention.default_config().set(
            name="gqa",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            input_linear=attention.GroupedQKVLinear.default_config().set(num_kv_heads=num_kv_heads),
        )
        test_layer = test_cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key, data_key = jax.random.split(prng_key, num=3)
        state = ref_layer.initialize_parameters_recursively(init_key)

        batch, seq_len = 2, 10
        per_head_dim = ref_layer.per_head_dim()
        q = jax.random.uniform(data_key, (batch, seq_len, num_heads, per_head_dim))
        k = jax.random.uniform(data_key, (batch, seq_len, num_kv_heads, per_head_dim))
        v = jax.random.uniform(data_key, (batch, seq_len, num_kv_heads, per_head_dim))
        q, k, v = q.astype(jnp.float32), k.astype(kv_dtype), v.astype(kv_dtype)
        attention_logit_biases = attention_logit_biases = attention_bias.ZeroAttentionBias()
        pos = jnp.arange(seq_len)[None, :]
        kv_state = KVState(
            k_proj=k,
            v_proj=v,
            key_positions=pos,
        )

        (test_context, ref_probs), _ = F(
            test_layer,
            method="_compute_attention",
            state=state,
            is_training=False,
            prng_key=prng_key,
            inputs=dict(
                mode=mode,
                q_proj=q,
                attention_logit_biases=attention_logit_biases,
                kv_state=kv_state,
            ),
        )

        k = jnp.repeat(k, num_heads // num_kv_heads, axis=2)
        v = jnp.repeat(v, num_heads // num_kv_heads, axis=2)

        kv_state = KVState(
            k_proj=k,
            v_proj=v,
            key_positions=pos,
        )

        (ref_context, ref_probs), _ = F(
            ref_layer,
            method="_compute_attention",
            state=state,
            is_training=False,
            prng_key=prng_key,
            inputs=dict(
                mode=mode,
                q_proj=q,
                attention_logit_biases=attention_logit_biases,
                kv_state=kv_state,
            ),
        )

        assert_allclose(ref_context, test_context)
        self.assertEqual(ref_context.dtype, q.dtype)
        self.assertEqual(test_context.dtype, q.dtype)
        assert_allclose(ref_probs, ref_probs)


class ScaleFunctionsTest(TestCase):
    """Tests Scale Functions."""

    def _scale_query_kwargs(
        self,
        *,
        query_scale_factor: Union[None, int, float, InstantiableConfig[attention.ScaleFn]],
        key_scale_factor: Union[None, int, float, InstantiableConfig[attention.ScaleFn]],
    ):
        model_dim = 16
        if isinstance(query_scale_factor, (int, float)):
            query_scale_factor = config_for_function(attention.constant_scale_fn).set(
                value=query_scale_factor
            )
        if isinstance(key_scale_factor, (int, float)):
            key_scale_factor = config_for_function(attention.constant_scale_fn).set(
                value=key_scale_factor
            )

        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=2,
            query_scale=attention.ScaleQuery.default_config().set(scale_factor=query_scale_factor),
            key_scale=attention.ScaleKey.default_config().set(scale_factor=key_scale_factor),
        )
        cfg.input_linear.layer.bias = False
        cfg.output_linear.bias = False
        layer = cfg.instantiate(parent=None)

        param_specs = layer.create_parameter_specs_recursively()
        layer_params = jax.tree.map(
            lambda spec: jnp.ones(spec.shape, dtype=spec.dtype), param_specs
        )

        batch_size = 3
        tgt_len = 10  # Must be even.
        query = jnp.concatenate(
            (
                jnp.ones([batch_size, tgt_len // 2, model_dim]),
                jnp.zeros([batch_size, tgt_len // 2, model_dim]),
            ),
            axis=1,
        )
        kwargs = dict(
            module=layer,
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
            inputs=dict(query=query),
        )
        return kwargs

    @parameterized.product(query_scale_factor=[None, 7], key_scale_factor=[None, 11])
    def test_scale_query_key(
        self, *, query_scale_factor: Optional[float], key_scale_factor: Optional[float]
    ):
        kwargs = self._scale_query_kwargs(
            query_scale_factor=query_scale_factor, key_scale_factor=key_scale_factor
        )
        kwargs["inputs"]["return_aux"] = {"probs"}
        forward_outputs, _ = F(**kwargs)
        if query_scale_factor is None:
            query_scale_factor = kwargs["module"].per_head_dim() ** -0.5
        if key_scale_factor is None:
            key_scale_factor = 1
        query_scale_factor = float(query_scale_factor)
        key_scale_factor = float(key_scale_factor)
        self.assertNestedAllClose(
            forward_outputs.probs[0, 0, 0, 0],
            # All ones matrix times all ones vector has l2 norm dim ** 1.5.
            # Half of input tokens are all ones, half are all zeros.
            jax.nn.sigmoid(
                kwargs["inputs"]["query"].shape[-1] ** 3 * query_scale_factor * key_scale_factor,
            )
            / (kwargs["inputs"]["query"].shape[1] // 2),
        )

    def test_scale_query_key_dim_dependence(self):
        query_scale_factor = config_for_function(attention.pow_scale_fn).set(exp=1)
        key_scale_factor = config_for_function(attention.pow_scale_fn).set(exp=-1)
        kwargs = self._scale_query_kwargs(
            query_scale_factor=query_scale_factor, key_scale_factor=key_scale_factor
        )
        kwargs["inputs"]["return_aux"] = {"probs"}
        forward_outputs, _ = F(**kwargs)
        self.assertNestedAllClose(
            forward_outputs.probs[0, 0, 0, 0],
            # All ones matrix times all ones vector has l2 norm dim ** 1.5.
            # Half of input tokens are all ones, half are all zeros.
            jax.nn.sigmoid(float(kwargs["inputs"]["query"].shape[-1] ** 3))
            / (kwargs["inputs"]["query"].shape[1] // 2),
        )

    def test_scale_query_key_barrier(self):
        """Tests that the scale factors are not combined.

        Note that even without the barrier, it's not clear that they would be combined.
        (They aren't on CPU even without the barrier.)
        """
        query_scale_factor = 7
        key_scale_factor = 11
        kwargs = self._scale_query_kwargs(
            query_scale_factor=query_scale_factor, key_scale_factor=key_scale_factor
        )

        # Check optimized HLO scales by query_scale_factor and key_scale_factor as separate
        # multiplications. This only checks the default backend, so it doesn't check
        # what happens on gpu/tpu unless jax is configured to use them.
        f = jax.jit(F, static_argnames=("module", "is_training"))
        compile_options = dict(
            xla_cpu_enable_fast_math=True,
            xla_cpu_fast_math_honor_nans=False,
            xla_cpu_fast_math_honor_infs=False,
            xla_cpu_fast_math_honor_functions=False,
            xla_cpu_fast_math_honor_division=False,
        )
        hlo = f.lower(**kwargs).compile(compile_options).as_text()
        hlo = test_utils.clean_hlo(hlo)
        self.assertIn(str(query_scale_factor), hlo)
        self.assertIn(str(key_scale_factor), hlo)
        self.assertNotIn(str(query_scale_factor * key_scale_factor), hlo)

    @parameterized.product(
        [
            dict(
                qkv_value=1.0,
                expected_value=jax.nn.sigmoid((1.0 * 1.0) * 2 - jnp.log(6)),
                seq_len=6,
            ),
            dict(
                qkv_value=1.0,
                expected_value=jax.nn.sigmoid((1.0 * 1.0) * 2 - jnp.log(4)),
                seq_len=4,
            ),
            dict(
                qkv_value=2.0,
                expected_value=jax.nn.sigmoid((2.0 * 2.0) * 2 - jnp.log(6)),
                seq_len=6,
            ),
        ],
        mode=[ForwardMode.FORWARD, ForwardMode.EXTEND_STEP],
        kv_dtype=[jnp.float64, jnp.float32, jnp.bfloat16],
    )
    def test_sigmoid_compute_attention(
        self,
        qkv_value: float,
        expected_value: float,
        seq_len: int,
        mode: ForwardMode,
        kv_dtype: jnp.dtype,
    ):
        model_dim = 16
        num_heads = 4
        batch_size = 2
        init_key = jax.random.PRNGKey(123)

        cfg = attention.SigmoidAttention.default_config().set(
            seq_len=seq_len,
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            query_scale=attention.ScaleQuery.default_config(),
            atten_logit_cap=0.0,
            dtype=jnp.float32,
        )
        sigmoid_attention = cfg.set(name="sigmoid_attention").instantiate(parent=None)
        state = sigmoid_attention.initialize_parameters_recursively(prng_key=init_key)

        qkv_shape = [batch_size, seq_len, num_heads, num_heads]

        # Get outputs.
        forward_key = jax.random.PRNGKey(456)

        q_proj = jnp.full(qkv_shape, fill_value=qkv_value, dtype=jnp.float32)
        k_proj = jnp.full(qkv_shape, fill_value=qkv_value, dtype=kv_dtype)
        v_proj = jnp.full(qkv_shape, fill_value=qkv_value, dtype=kv_dtype)
        pos = jnp.arange(seq_len)[None, :]

        (q_proj, k_proj), _ = F(
            sigmoid_attention,
            method="_scale_qk",
            state=state,
            is_training=False,
            prng_key=forward_key,
            inputs=dict(q_proj=q_proj, k_proj=k_proj, query_positions=pos, key_positions=pos),
        )

        kv_state = KVState(
            k_proj=k_proj,
            v_proj=v_proj,
            key_positions=pos,
        )

        inputs = dict(
            mode=mode,
            q_proj=q_proj,
            attention_logit_biases=attention_bias.CausalAttentionBias(
                target_positions=jnp.arange(seq_len)[None],
                source_positions=jnp.arange(seq_len)[None],
            ),
            kv_state=kv_state,
        )

        (context, probs), _ = F(
            sigmoid_attention,
            method="_compute_attention",
            state=state,
            is_training=False,
            prng_key=forward_key,
            inputs=inputs,
        )

        self.assertEqual(context.dtype, inputs["q_proj"].dtype)
        output_shape = [batch_size, num_heads, seq_len, seq_len]
        indexes = jnp.arange(seq_len)
        # Zeros outside of the causal triangle.
        causal_biases = jax.lax.ge(indexes[:, None], indexes[None, :])
        expected_output = jnp.full(output_shape, fill_value=expected_value) * causal_biases

        self.assertNestedAllClose(probs, expected_output)


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

    @parameterized.parameters(5, 2, 1)
    def test_rel_pos_to_abs_pos(self, seq_len):
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
    @set_threefry_partitionable(True)  # TODO(mhopkins): remove decorator after jax 0.5.0
    def test_per_dim_scale(self, per_dim_scale, scale_position):
        model_dim = 6
        num_heads = 2
        cfg = attention.TransformerAttentionLayer.default_config().set(
            name="test",
            target_dim=model_dim,
            source_dim=model_dim,
            structure="postnorm",
            attention=MultiheadAttentionXL.default_config().set(
                num_heads=num_heads,
                query_scale=attention.ScaleQuery.default_config().set(per_dim_scale=per_dim_scale),
                scale_position=scale_position,
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
            layer_params["attention"]["scale_query"]["per_dim_scale"]["param"] = jax.random.normal(
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
                MultiheadAttentionXL.ScalePosition.LOGIT.value: 51.232643,
                MultiheadAttentionXL.ScalePosition.QUERY.value: 51.397125,
            },
            str(PerDimScale.default_config()): {
                MultiheadAttentionXL.ScalePosition.LOGIT.value: 50.681373,
                MultiheadAttentionXL.ScalePosition.QUERY.value: 50.898140,
            },
        }
        assert_allclose(
            expected_vals[str(per_dim_scale)][scale_position.value],
            jnp.abs(layer_outputs.data).sum(),
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


class TransformerAttentionLayerTest(TestCase):
    @parameterized.parameters([False, True])
    def test_forward_vs_extend_step(self, with_source: bool):
        init_prng, target_prng, source_prng = jax.random.split(jax.random.PRNGKey(0), 3)

        model_dim = 8
        layer_kwargs = dict(target_dim=model_dim, source_dim=model_dim)
        cfg: TransformerAttentionLayer.Config = TransformerAttentionLayer.default_config().set(
            **layer_kwargs
        )
        cfg.attention.set(num_heads=2, mask=CausalAttentionBias.default_config())
        layer: TransformerAttentionLayer = cfg.set(name="test").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=init_prng)

        batch, decode_len = 2, 6
        target = jax.random.uniform(target_prng, shape=[batch, decode_len, model_dim])
        input_kwargs = {}

        if with_source:
            input_kwargs.update(
                source=jax.random.uniform(source_prng, shape=[batch, decode_len, model_dim])
            )

        forward_outputs, _ = F(
            layer,
            inputs=dict(target=jnp.asarray(target), **input_kwargs),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        for start_time_step in (-1, 0, 2, decode_len):
            if start_time_step < 0:
                (cached_states, init_outputs), _ = F(
                    layer,
                    inputs=dict(
                        time_step=None,
                        target=TensorSpec(target.shape, target.dtype),
                        **input_kwargs,
                    ),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                    method="init_states",
                )
                self.assertIsNone(init_outputs)
                data = jnp.zeros([batch, decode_len, model_dim])
                start_time_step = 0
            else:
                (cached_states, prefill_outputs), _ = F(
                    layer,
                    inputs=dict(
                        time_step=jnp.array([start_time_step] * batch, dtype=jnp.int32),
                        target=target,
                        **input_kwargs,
                    ),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                    method="init_states",
                )
                data = prefill_outputs.data

            data = jnp.einsum("btd->tbd", data)

            for time_step in range(start_time_step, decode_len):
                extend_kwargs = {}
                for k, v in input_kwargs.items():
                    extend_kwargs[k] = jnp.asarray(v[:, time_step : time_step + 1, :])

                (cached_states, extend_outputs), _ = F(
                    layer,
                    inputs=dict(
                        target=jnp.asarray(target[:, time_step : time_step + 1, :]),
                        cached_states=cached_states,
                        **extend_kwargs,
                    ),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                    method="extend_step",
                )
                data = data.at[time_step].set(jnp.squeeze(extend_outputs.data, axis=1))

            data = jnp.einsum("tbd->btd", data)

            # Prefill + extend_step == forward.
            assert_allclose(forward_outputs.data, data)

    @parameterized.product(
        structure=("prenorm", "postnorm", "hybridnorm"), with_source=(False, True)
    )
    def test_v2_structure(self, structure, with_source: bool):
        # Test equivalence bewteen (prenorm, postnorm, hybridnorm) and v2 structure.
        # prenorm == in_norm (v2)
        # postnorm == out_norm (v2)
        # hybridnorm == (in_norm, res_norm) (v2)
        init_prng, target_prng, source_prng = jax.random.split(jax.random.PRNGKey(0), 3)
        batch, decode_len, model_dim = 2, 6, 8
        target = jax.random.uniform(target_prng, shape=[batch, decode_len, model_dim])

        input_kwargs = {}
        if with_source:
            input_kwargs.update(
                source=jax.random.uniform(source_prng, shape=[batch, decode_len, model_dim])
            )

        ref_layer_kwargs = dict(
            target_dim=model_dim,
            source_dim=model_dim,
            structure=structure,
            norm=RMSNorm.default_config(),
        )

        ref_cfg: TransformerAttentionLayer.Config = TransformerAttentionLayer.default_config().set(
            **ref_layer_kwargs
        )
        ref_cfg.attention.set(num_heads=2, mask=CausalAttentionBias.default_config())
        ref_layer: TransformerAttentionLayer = ref_cfg.set(name="ref").instantiate(parent=None)
        ref_layer_params = ref_layer.initialize_parameters_recursively(prng_key=init_prng)

        ref_outputs, _ = F(
            ref_layer,
            inputs=dict(target=jnp.asarray(target), **input_kwargs),
            state=ref_layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        norm = {}
        if structure == "prenorm":
            norm = {NormPosition.IN_NORM: RMSNorm.default_config()}
        elif structure == "postnorm":
            norm = {NormPosition.OUT_NORM: RMSNorm.default_config()}
        elif structure == "hybridnorm":
            norm = {
                NormPosition.IN_NORM: RMSNorm.default_config(),
                NormPosition.RES_NORM: RMSNorm.default_config(),
            }
        else:
            raise ValueError(f"No equivalent structure is available in v2 to {structure}.")

        layer_kwargs = dict(target_dim=model_dim, source_dim=model_dim, structure="v2", norm=norm)

        cfg: TransformerAttentionLayer.Config = TransformerAttentionLayer.default_config().set(
            **layer_kwargs
        )
        cfg.attention.set(num_heads=2, mask=CausalAttentionBias.default_config())
        layer: TransformerAttentionLayer = cfg.set(name="test").instantiate(parent=None)
        layer_params = ref_layer_params
        if structure == "prenorm":
            layer_params["in_norm"] = layer_params["norm"]
            del layer_params["norm"]
        elif structure == "postnorm":
            layer_params["out_norm"] = layer_params["norm"]
            del layer_params["norm"]
        elif structure == "hybridnorm":
            layer_params["in_norm"] = layer_params["prenorm"]
            layer_params["res_norm"] = layer_params["postnorm"]
            del layer_params["prenorm"]
            del layer_params["postnorm"]
        else:
            raise ValueError(f"No equivalent structure is available in v2 to {structure}.")

        outputs, _ = F(
            layer,
            inputs=dict(target=jnp.asarray(target), **input_kwargs),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertNestedAllClose(outputs, ref_outputs)


class TransformerFeedForwardLayerTest(TestCase):
    @parameterized.parameters(
        dict(rms_norm_summary=[]),
        dict(rms_norm_summary=["linear2_outputs"]),
        dict(rms_norm_summary=["final_outputs"], expected_raise_regex="add_value_rms_norm_summary"),
    )
    @set_threefry_partitionable(True)  # TODO(mhopkins): remove after jax 0.5.0
    def test_add_value_rms_norm_summary(
        self, rms_norm_summary: list[str], *, expected_raise_regex=None
    ):
        batch, seq_len, dim = 2, 3, 4
        cfg = TransformerFeedForwardLayer.default_config().set(
            name="ffn",
            input_dim=dim,
            hidden_dim=dim * 4,
            add_value_rms_norm_summary=rms_norm_summary,
            tensor_stats=DefaultTensorStats.default_config(),
        )
        if expected_raise_regex is not None:
            with self.assertRaisesRegex(NotImplementedError, expected_raise_regex):
                layer = cfg.instantiate(parent=None)
            return
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), shape=[batch, seq_len, dim])
        y, output_collection = F(
            layer,
            inputs=dict(inputs=x),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertSequenceEqual(x.shape, y.shape)
        self.assertNestedAllClose(2.990841, jnp.sum(y))
        if "tensor_stats" in output_collection.summaries:
            output_stats = output_collection.summaries["tensor_stats"]
        else:
            output_stats = {}
        for k in rms_norm_summary:
            assert k in output_stats

    @parameterized.parameters(
        dict(activation_fn="nn.relu"),
        dict(activation_fn=("nn.relu", "linear")),
        dict(activation_fn=("linear", "quick_gelu")),
        dict(activation_fn=("linear", "exact_gelu")),
        dict(activation_fn=("linear", "nn.silu")),
    )
    def test_add_dead_neuron_summary(self, activation_fn: Union[str, list[str]]):
        batch, seq_len, dim = 2, 3, 4
        cfg = TransformerFeedForwardLayer.default_config().set(
            name="ffn",
            input_dim=dim,
            hidden_dim=dim * 4,
            activation=activation_fn,
            add_dead_neuron_summary=True,
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), shape=[batch, seq_len, dim])
        y, output_collection = F(
            layer,
            inputs=dict(inputs=x),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertSequenceEqual(x.shape, y.shape)
        if isinstance(activation_fn, str):
            activation_fn = [activation_fn]
        self.assertSetEqual(
            {k for k in output_collection.summaries.keys() if k.startswith("dead_neurons/")},
            {
                f"dead_neurons/{k}"
                for k in activation_fn
                if k in ("nn.relu", "quick_gelu", "exact_gelu", "nn.silu")
            },
        )

    def test_linear_remat(self):
        batch, seq_len, dim = 2, 3, 4
        cfg = TransformerFeedForwardLayer.default_config().set(
            name="ffn",
            input_dim=dim,
            hidden_dim=dim * 4,
            add_value_rms_norm_summary=[],
            tensor_stats=DefaultTensorStats.default_config(),
            activation=("nn.relu", "nn.relu"),
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), shape=[batch, seq_len, dim])

        def f(x, layer_params):
            y, _ = F(
                layer,
                inputs=dict(inputs=x),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            return y

        _, save_name_backward = jax.linearize(
            jax.remat(
                f,
                policy=save_and_offload_only_these_names_regex(
                    names_which_can_be_saved=RematRegexSavePatterns.FEED_FORWARD.value,
                    names_which_can_be_offloaded=None,
                    offload_src="device",
                    offload_dst="pinned_host",
                ),
            ),
            x,
            layer_params,
        )
        _, save_dots_backward = jax.linearize(
            jax.remat(f, policy=jax_remat_policies.dots_saveable), x, layer_params
        )

        self.assertEqual(str(save_name_backward).count(" dot_general"), 6)
        self.assertEqual(
            str(save_name_backward).count(" dot_general"),
            str(save_dots_backward).count(" dot_general"),
        )

    @parameterized.parameters(
        "prenorm",
        "postnorm",
        "hybridnorm",
    )
    def test_v2_structure(self, structure):
        # Test equivalence bewteen (prenorm, postnorm, hybridnorm) and v2 structure.
        # prenorm == in_norm (v2)
        # postnorm == out_norm (v2)
        # hybridnorm == (in_norm, res_norm) (v2)
        batch, seq_len, dim = 2, 3, 4
        ref_cfg = TransformerFeedForwardLayer.default_config().set(
            name="ffn",
            input_dim=dim,
            hidden_dim=dim * 4,
            structure=structure,
            norm=RMSNorm.default_config(),
        )
        ref_layer = ref_cfg.instantiate(parent=None)
        ref_layer_params = ref_layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(0)
        )
        x = jax.random.normal(jax.random.PRNGKey(1), shape=[batch, seq_len, dim])
        ref_y, _ = F(
            ref_layer,
            inputs=dict(inputs=x),
            state=ref_layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        norm = {}
        if structure == "prenorm":
            norm = {NormPosition.IN_NORM: RMSNorm.default_config()}
        elif structure == "postnorm":
            norm = {NormPosition.OUT_NORM: RMSNorm.default_config()}
        elif structure == "hybridnorm":
            norm = {
                NormPosition.IN_NORM: RMSNorm.default_config(),
                NormPosition.RES_NORM: RMSNorm.default_config(),
            }
        else:
            raise ValueError(f"No equivalent structure is available in v2 to {structure}.")

        cfg = TransformerFeedForwardLayer.default_config().set(
            name="ffn",
            input_dim=dim,
            hidden_dim=dim * 4,
            structure="v2",
            norm=norm,
        )
        layer = cfg.instantiate(parent=None)
        layer_params = ref_layer_params
        if structure == "prenorm":
            layer_params["in_norm"] = layer_params["norm"]
            del layer_params["norm"]
        elif structure == "postnorm":
            layer_params["out_norm"] = layer_params["norm"]
            del layer_params["norm"]
        elif structure == "hybridnorm":
            layer_params["in_norm"] = layer_params["prenorm"]
            layer_params["res_norm"] = layer_params["postnorm"]
            del layer_params["prenorm"]
            del layer_params["postnorm"]
        else:
            raise ValueError(f"No equivalent structure is available in v2 to {structure}.")

        y, _ = F(
            layer,
            inputs=dict(inputs=x),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertNestedAllClose(y, ref_y)


class BaseTransformerTest(TestCase):
    def _test_decoder_with_transformer(self, transformer_cfg: BaseTransformerLayer.Config):
        prefix_length = jnp.asarray([0, 2])
        batch_size, num_decodes, seq_len, vocab_size = prefix_length.shape[0], 3, 7, 6
        bos_id = eos_id = 1
        pad_token_id = 0

        cfg = Decoder.default_config().set(
            transformer=transformer_cfg.clone(name="transformer"),
            dim=transformer_cfg.input_dim,
            vocab_size=vocab_size,
            emb=TransformerTextEmbeddings.default_config().set(
                pos_emb=LearnedPositionalEmbedding.default_config().set(shape=(seq_len,))
            ),
            # output_norm=LayerNorm.default_config().set(eps=layer_norm_epsilon),
            # dropout_rate=dropout_rate,
            pad_token_id=pad_token_id,
            eos_token_id=eos_id,
        )

        decoder: Decoder = cfg.set(name="decoder").instantiate(parent=None)
        decoder_state = decoder.initialize_parameters_recursively(jax.random.PRNGKey(0))

        prefix = jax.random.randint(
            jax.random.PRNGKey(124),
            shape=[batch_size, seq_len],
            # Prefix can consist of any tokens, including pad and eos.
            minval=0,
            maxval=vocab_size,
        )
        # Explicitly fill positions >= prefix_length with pad_token_id.
        # Note that each batch example may have a different prefix length.
        # [batch_size, seq_len].
        prefix_mask = jnp.arange(seq_len) < prefix_length[:, None]
        prefix = prefix * prefix_mask + pad_token_id * (1 - prefix_mask)
        # Set last token to a non-pad token, to fix the prefix length.
        oh_indices = jax.nn.one_hot(prefix_length - 1, seq_len, dtype=prefix.dtype)
        prefix = prefix * (1 - oh_indices) + bos_id * oh_indices
        inputs = dict(
            input_batch=dict(prefix=prefix),
            max_sequence_length=seq_len,
            # cross_attention_data=None,
            # cross_attention_logit_biases=None,
            num_decodes=num_decodes,
        )
        outputs, _ = F(
            decoder,
            inputs=inputs,
            state=decoder_state,
            is_training=False,
            prng_key=jax.random.PRNGKey(2),
            method="sample_decode",
        )
        sequences = outputs.sequences
        self.assertEqual(sequences.shape, (batch_size, num_decodes, seq_len))

    def _test_forward_vs_extend_step(
        self,
        cfg: BaseTransformerLayer.Config,
        *,
        input_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Tests that {init,prefill}_states + extend_step is equivalent to forward for `cfg`."""
        if input_kwargs is None:
            input_kwargs = {}
        layer: BaseTransformerLayer = cfg.clone(name="layer").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        batch_size, tgt_len = 2, 5
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, cfg.input_dim], dtype=np.float32)

        forward_outputs, _ = F(
            layer,
            inputs=dict(
                data=jnp.asarray(target),
                **input_kwargs,
            ),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        for start_time_step in (-1, 0, 2, tgt_len):
            if start_time_step > tgt_len:
                continue
            print(f"start_time_step={start_time_step} layer={type(layer)}")
            if start_time_step < 0:
                (cached_states, init_outputs), _ = F(
                    layer,
                    inputs=dict(
                        time_step=None,
                        data=TensorSpec([batch_size, tgt_len], dtype=target.dtype),
                        **input_kwargs,
                    ),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                    method="init_states",
                )
                self.assertIsNone(init_outputs)
                decoder_output = jnp.zeros_like(target)
                start_time_step = 0
            else:
                (cached_states, prefill_outputs), _ = F(
                    layer,
                    inputs=dict(
                        time_step=jnp.array([start_time_step] * batch_size, dtype=jnp.int32),
                        data=jnp.asarray(target),
                        **input_kwargs,
                    ),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                    method="init_states",
                )
                decoder_output = prefill_outputs.data
            # Transpose to [tgt_len, batch_size, model_dim].
            decoder_output = jnp.einsum("bsd->sbd", decoder_output)
            for time_step in range(start_time_step, tgt_len):
                (cached_states, extend_step_outputs), _ = F(
                    layer,
                    inputs=dict(
                        data=jnp.asarray(target[:, time_step : time_step + 1, :]),
                        cached_states=cached_states,
                        **input_kwargs,
                    ),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                    method="extend_step",
                )
                decoder_output = decoder_output.at[time_step].set(
                    jnp.squeeze(extend_step_outputs.data, axis=1)
                )
            # Transpose to [batch_size, tgt_len, model_dim].
            decoder_output = jnp.einsum("sbd->bsd", decoder_output)
            # Prefill + extend_step == forward.
            assert_allclose(forward_outputs.data, decoder_output)


class TransformerTest(BaseTransformerTest):
    """Tests TransformerLayer."""

    def _compare_against_roberta_attention(
        self, ref: hf_roberta.RobertaAttention, layer: TransformerAttentionLayer
    ):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_param_shapes = jax.tree.map(lambda x: x.shape, layer_params)
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
            layer_outputs, output_collection = F(
                layer,
                inputs=dict(data=jnp.asarray(target), self_attention_logit_biases=mask),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
                drop_output_collections=(),
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
            self.assertNestedEqual(layer_outputs.data, output_collection.module_outputs["output"])

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

    def test_decoding(self):
        model_dim, num_heads = 6, 2
        cfg = TransformerLayer.default_config().set(input_dim=model_dim)
        cfg.self_attention.attention.set(num_heads=num_heads, causal=True)
        cfg.feed_forward.hidden_dim = model_dim * 4
        cfg.vlog = 5
        self._test_forward_vs_extend_step(cfg)

    def test_self_attention_kv_state(self):
        """Tests TransformerLayer with explicit self_attention_kv_state.

        Creates a base TransformerLayer and a test TransformerLayer with QLinear. Uses the kv_state
        of the base layer as the explicit kv_state for the test layer. Checks that the outputs are
        identical.
        """
        model_dim = 16
        num_heads = 4
        base_cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
        base_cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
        base_cfg.self_attention.attention.set(num_heads=num_heads, causal=True)
        base_layer: TransformerLayer = base_cfg.instantiate(parent=None)
        base_layer_params = base_layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(0)
        )

        test_cfg = base_cfg.clone()
        test_cfg.self_attention.attention.input_linear = QLinear.default_config()
        test_layer: TransformerLayer = test_cfg.instantiate(parent=None)
        # Let test_layer_params to be identical to base_layer_params except removing {k,v}_proj.
        test_layer_params = copy.deepcopy(base_layer_params)
        for k in ("k_proj", "v_proj"):
            test_layer_params["self_attention"]["attention"]["i_proj"].pop(k)
        self.assertEqual(
            shapes(test_layer_params),
            shapes(test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))),
        )

        batch_size, tgt_len = 2, 5
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        base_layer_outputs, _ = F(
            base_layer,
            inputs=dict(data=jnp.asarray(target), return_aux={"self_attention_kv_state"}),
            state=base_layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        test_layer_outputs, _ = F(
            test_layer,
            # Explicitly pass `self_attention_kv_state` from `base_layer_outputs` as inputs to
            # test_layer.
            inputs=dict(
                data=jnp.asarray(target),
                self_attention_kv_state=base_layer_outputs.self_attention_kv_state,
                return_aux={"self_attention_kv_state"},
            ),
            state=test_layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(base_layer_outputs.data, test_layer_outputs.data)

        # Tests prefill_state and extend_step.
        self._test_forward_vs_extend_step(
            test_cfg,
            input_kwargs=dict(
                # Explicitly pass `self_attention_kv_state`.
                self_attention_kv_state=base_layer_outputs.self_attention_kv_state,
            ),
        )


class ParallelTransformerTest(TestCase):
    """Tests ParallelTransformerLayer."""

    @set_threefry_partitionable(True)  # TODO(mhopkins): remove after jax 0.5.0
    def test_with_golden_value(self):
        """A test of ParallelTransformerLayer by comparing results to a golden value."""
        model_dim = 16
        num_heads = 4
        cfg = ParallelTransformerLayer.default_config().set(name="test", input_dim=model_dim)
        cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
        cfg.self_attention.set(num_heads=num_heads)
        cfg.norm = RMSNorm.default_config()
        set_bias_recursively(cfg, bias=False)
        layer: TransformerLayer = cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        self.assertEqual(
            {
                "feed_forward": {
                    "dropout1": {},
                    "dropout2": {},
                    "linear1": {"weight": (16, 64)},
                    "linear2": {"weight": (64, 16)},
                    "stochastic_depth": {},
                },
                "norm": {"scale": (16,)},
                "self_attention": {
                    "dropout": {},
                    "i_proj": {
                        "k_proj": {"weight": (16, 4, 4)},
                        "q_proj": {"weight": (16, 4, 4)},
                        "v_proj": {"weight": (16, 4, 4)},
                    },
                    "kv_cache": {},
                    "o_proj": {"weight": (16, 4, 4)},
                    "scale_key": {},
                    "scale_query": {},
                },
            },
            utils.shapes(layer_params),
        )

        batch_size, tgt_len = 2, 6
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        mask = attention_bias.make_causal_biases(tgt_len)
        mask = jnp.tile(mask[None, None, :, :], (batch_size, num_heads, 1, 1))
        layer_outputs, _ = F(
            layer,
            inputs=dict(data=jnp.asarray(target), self_attention_logit_biases=mask),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertEqual(target.shape, layer_outputs.data.shape)
        self.assertNestedAllClose(0.165421, np.mean(layer_outputs.data))

    def test_build_remat_spec(self):
        model_dim, num_heads = 6, 2
        cfg: TransformerLayer.Config = TransformerLayer.default_config().set(input_dim=model_dim)
        cfg.self_attention.attention.set(num_heads=num_heads, causal=True)
        cfg.feed_forward.hidden_dim = model_dim * 4
        cfg.vlog = 5

        layer: BaseTransformerLayer = cfg.clone(name="layer").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        batch_size, tgt_len = 2, 5
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, cfg.input_dim], dtype=np.float32)

        def f(x, layer_params):
            forward_outputs, _ = F(
                layer,
                inputs=dict(
                    data=x,
                ),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            return forward_outputs

        # Ignore type errors.
        spec: Any = build_remat_spec(mock.MagicMock())

        _, default_policy_backward = jax.linearize(
            jax.remat(f, policy=spec.policy.instantiate(), prevent_cse=spec.prevent_cse),
            jnp.asarray(target),
            layer_params,
        )
        _, full_remat_backward = jax.linearize(
            jax.remat(f),
            jnp.asarray(target),
            layer_params,
        )
        # Eliminated the remat of qkv_proj, context and o_proj = 5 dots. This assumes
        # FlashAttention is not enabled.
        self.assertEqual(
            str(full_remat_backward).count(" dot_general")
            - str(default_policy_backward).count(" dot_general"),
            5,
        )

    def test_build_remat_spec_neuron(self):
        model_dim, num_heads = 6, 2
        cfg: TransformerLayer.Config = TransformerLayer.default_config().set(input_dim=model_dim)
        cfg.self_attention.attention.set(num_heads=num_heads, causal=True)
        cfg.feed_forward.hidden_dim = model_dim * 4
        cfg.vlog = 5

        layer: BaseTransformerLayer = cfg.clone(name="layer").instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))

        batch_size, tgt_len = 2, 5
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, cfg.input_dim], dtype=np.float32)

        def f(x, layer_params):
            forward_outputs, _ = F(
                layer,
                inputs=dict(
                    data=x,
                ),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            return forward_outputs

        # Ignore type errors.
        spec: Any = build_remat_spec(mock.MagicMock())

        policy = (
            config_for_function(save_and_offload_only_these_names_regex)
            .set(
                names_which_can_be_saved="|".join(
                    [
                        RematRegexSavePatterns.QKV_PROJ.value,
                        RematRegexSavePatterns.LINEAR1_X.value,
                    ]
                ),
                names_which_can_be_offloaded=None,
                offload_src=None,
                offload_dst=None,
            )
            .instantiate()
        )

        _, default_policy_backward = jax.linearize(
            jax.remat(f, policy=policy, prevent_cse=spec.prevent_cse),
            jnp.asarray(target),
            layer_params,
        )
        _, full_remat_backward = jax.linearize(
            jax.remat(f),
            jnp.asarray(target),
            layer_params,
        )

        # Eliminated the remat of qkv_proj and linear1_0 = 4 dots.
        self.assertEqual(
            str(full_remat_backward).count(" dot_general")
            - str(default_policy_backward).count(" dot_general"),
            4,
        )


class _StackModel(BaseLayer):
    """A dummy transformer stack."""

    @config_class
    class Config(BaseLayer.Config):
        stack: Optional[BaseStackedTransformerLayer.Config] = None  # The transformer stack.
        output_self_attention_kv_state: bool = False

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("stack", cfg.stack)

    def forward(self, data, **layer_kwargs):
        cfg = self.config

        # [batch, length, dim].
        output = self.stack(data, **layer_kwargs)
        x = output.data
        x_mean = jnp.mean(x, axis=1, keepdims=True)
        # [batch, length].
        x_var = jnp.sum((x - x_mean) ** 2, axis=-1)
        loss = jnp.mean(x_var)
        if cfg.output_self_attention_kv_state:
            return loss, {"mean": x_mean, "self_attention_kv_state": output.self_attention_kv_state}
        return loss, {"mean": x_mean}


def _recursive_stack(inputs: Nested[Tensor], axis=0):
    def stack(*xs):
        return jnp.stack(xs, axis=axis)

    return {"layer": utils.vectorized_tree_map(stack, *inputs.values())}


def _convert_from_stacked_params(
    layer_params: Nested[Tensor], *, target_stack_cfg: BaseStackedTransformerLayer.Config
) -> Nested[Tensor]:
    """Converts params of a StackedTransformerLayer to params for `target_stack_cfg`."""
    # First stack to params of a RepeatedTransformerLayer.
    layer_params = {"stack": {"repeat": VDict(_recursive_stack(layer_params["stack"]))}}
    if target_stack_cfg.klass == RepeatedTransformerLayer:
        return layer_params
    elif target_stack_cfg.klass == PipelinedTransformerLayer:
        pipeline_stage_cfg = target_stack_cfg.stage
        num_layers_per_stage = target_stack_cfg.num_layers // target_stack_cfg.num_stages

        def reshape(x):
            """Reshapes x from [num_layers, ...] to [num_stages, num_layers_per_stage, ...]."""
            x_shape = list(x.shape)
            return jnp.reshape(x, [target_stack_cfg.num_stages, num_layers_per_stage] + x_shape[1:])

        pipeline_params = jax.tree.map(reshape, layer_params["stack"].pop("repeat"))

        if pipeline_stage_cfg.klass == RepeatedTransformerLayer:
            layer_params["stack"]["pipeline"] = VDict({"layer": {"repeat": pipeline_params}})
        elif pipeline_stage_cfg.klass == StackedTransformerLayer:
            layer_params["stack"]["pipeline"] = VDict(
                {
                    "layer": {
                        f"layer{i}": jax.tree.map(lambda x, i=i: x[:, i], pipeline_params["layer"])
                        for i in range(num_layers_per_stage)
                    }
                }
            )
        else:
            raise NotImplementedError(target_stack_cfg)
        return layer_params
    else:
        raise NotImplementedError(target_stack_cfg)


class NonUniformStack(StackedTransformerLayer):
    def _aggregate_layer_outputs(
        self, layer_outputs: Sequence[BaseTransformerLayer.Output]
    ) -> BaseTransformerLayer.Output:
        return BaseTransformerLayer.Output(
            # Use data and self_attention_kv_state from the final layer outputs.
            data=layer_outputs[-1].data,
            self_attention_kv_state=layer_outputs[-1].self_attention_kv_state,
            # Do not aggregate *_attention_probs.
            self_attention_probs=None,
            cross_attention_probs=None,
        )


class _StackedTransformerLayerWithKVState(NonUniformStack):
    """A class with a simple override of _update_layer_kwargs for unit testing."""

    def _update_layer_kwargs(
        self,
        layer_kwargs: dict[str, Any],
        *,
        all_layer_outputs: list[BaseTransformerLayer.Output],
        external_self_attention_kv_state: Optional[KVState] = None,
    ):
        del external_self_attention_kv_state

        layer_index = len(all_layer_outputs)
        if layer_index == 1:
            layer_kwargs["self_attention_kv_state"] = all_layer_outputs[-1].self_attention_kv_state
        elif layer_index == 2:
            layer_kwargs["self_attention_kv_state"] = None


class _StackedTransformerLayerWithSkipConnection(StackedTransformerLayer):
    """A class that outputs all layers' output for unit testing."""

    def _aggregate_layer_outputs(
        self,
        layer_outputs: Sequence[BaseTransformerLayer.Output],
    ) -> Sequence[BaseTransformerLayer.Output]:
        return layer_outputs


class StackedTransformerTest(BaseTransformerTest):
    """Tests StackedTransformerLayer."""

    def _stack_config(
        self,
        stack_cfg,
        *,
        num_layers,
        model_dim,
        num_heads,
        dtype,
        remat_spec,
        output_self_attention_kv_state=False,
    ) -> _StackModel.Config:
        if isinstance(stack_cfg, type):
            stack_cfg = stack_cfg.default_config()
        if callable(remat_spec):
            remat_spec = remat_spec(stack_cfg)
        cfg = _StackModel.default_config().set(
            name="test",
            stack=stack_cfg.set(
                input_dim=model_dim,
                num_layers=num_layers,
                vlog=5,
                dtype=dtype,
                layer=TransformerLayer.default_config().set(remat_spec=remat_spec),
            ),
            output_self_attention_kv_state=output_self_attention_kv_state,
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

        cfg: BaseStackedTransformerLayer.Config = transformer_type.default_config().set(
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
        layer: BaseStackedTransformerLayer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        target = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])
        source = jax.random.normal(jax.random.PRNGKey(456), [batch_size, src_len, model_dim * 2])

        self_attention_logit_biases = attention_bias.make_causal_biases(tgt_len)
        cross_attention_logit_biases = (
            jnp.array(np.random.randint(0, 2, [tgt_len, src_len])) * NEG_INF
        )
        return_aux = {"self_attention_probs", "cross_attention_probs"}

        forward_outputs, _ = F(
            layer,
            inputs=dict(
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
                return_aux=return_aux,
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(0),
        )
        initial_state, initial_output = layer.init_states(
            time_step=None,
            data=TensorSpec([batch_size, tgt_len], dtype=target.dtype),
        )
        self.assertIsNone(initial_output)
        inputs = dict(
            cached_states=initial_state, cross_attention_data=source, return_aux=return_aux
        )
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
                jax.tree.map(
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

        self_attention_logit_biases = attention_bias.make_causal_biases(tgt_len)
        cross_attention_logit_biases = (
            jnp.array(np.random.randint(0, 2, [tgt_len, src_len])) * NEG_INF
        )
        return_aux = {"self_attention_probs", "cross_attention_probs"}

        forward_outputs, _ = F(
            layer,
            inputs=dict(
                data=target,
                self_attention_logit_biases=self_attention_logit_biases,
                cross_attention_data=source,
                cross_attention_logit_biases=cross_attention_logit_biases,
                return_aux=return_aux,
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
                return_aux=return_aux,
            ),
            method="init_states",
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
        inputs = dict(
            cached_states=initial_states, cross_attention_data=source, return_aux=return_aux
        )
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
                jax.tree.map(
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

    def test_skip_connection(self):
        batch_size = 2
        seq_len = 6
        num_heads = 2
        input_dim = 4
        hidden_dim = 8
        num_layers = 5
        layer_with_skip_input = 3

        cfg = _StackedTransformerLayerWithSkipConnection.default_config().set(
            name="test", input_dim=input_dim, num_layers=num_layers
        )

        transformer_cfg = TransformerLayer.default_config()
        transformer_cfg.self_attention.attention.num_heads = num_heads
        transformer_cfg.feed_forward.hidden_dim = hidden_dim
        cfg.layer = transformer_cfg

        test_cfg = cfg.clone().set(
            data_merger=config_for_function(update_data_with_skip_connection).set(
                skip_connections={layer_with_skip_input: 1}
            )
        )

        base_layer = cfg.instantiate(parent=None)
        test_layer = test_cfg.instantiate(parent=None)

        random_inputs = jax.random.uniform(
            jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim)
        )
        state = base_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        base_output, _ = F(
            base_layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(data=random_inputs),
        )
        test_output, _ = F(
            test_layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(data=random_inputs),
        )

        for i in range(layer_with_skip_input):
            self.assertNestedAllClose(
                base_output[i].data,
                test_output[i].data,
            )
        for i in range(layer_with_skip_input, num_layers):
            self.assertNotAlmostEqual(
                jnp.min(jnp.abs(base_output[i].data - test_output[i].data)),
                0.0,
            )

    def test_passthrough_update_layer_kwargs(self):
        num_heads = 2
        input_dim = 4
        hidden_dim = 8
        num_layers = 3

        cfg = StackedTransformerLayer.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.num_layers = num_layers

        transformer_cfg = TransformerLayer.default_config()
        transformer_cfg.self_attention.attention.num_heads = num_heads
        transformer_cfg.feed_forward.hidden_dim = hidden_dim
        cfg.layer = transformer_cfg

        layer: StackedTransformerLayer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        input_all_layer_outputs = [BaseTransformerLayer.Output(data=jnp.ones([2, 3]))]
        expected_all_layer_outputs = [BaseTransformerLayer.Output(data=jnp.ones([2, 3]))]
        k_proj = jnp.zeros([3, 3])
        v_proj = jnp.ones([3, 3])
        input_self_attention_kv_state = KVState(
            k_proj=k_proj, v_proj=v_proj, key_positions=jnp.arange(3)[None]
        )
        expected_self_attention_kv_state = KVState(
            k_proj=k_proj, v_proj=v_proj, key_positions=jnp.arange(3)[None]
        )
        F(
            layer,
            prng_key=jax.random.PRNGKey(0),
            state=state,
            inputs=dict(
                layer_kwargs={},
                all_layer_outputs=[],
                external_self_attention_kv_state=input_self_attention_kv_state,
            ),
            method="_update_layer_kwargs",
            is_training=True,
        )
        self.assertNestedAllClose(
            input_all_layer_outputs,
            expected_all_layer_outputs,
        )
        self.assertNestedAllClose(
            input_self_attention_kv_state,
            expected_self_attention_kv_state,
        )

    def test_update_layer_kwargs(self):
        batch_size = 2
        seq_len = 6
        num_heads = 2
        input_dim = 4
        per_head_dim = input_dim // num_heads
        hidden_dim = 8
        num_layers = 3

        # Create a StackedTransformerLayer by specifying a sequence of non-uniform layer configs.
        cfg = _StackedTransformerLayerWithKVState.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.num_layers = num_layers
        cfg.layer = []
        for i in range(num_layers):
            transformer_cfg = TransformerLayer.default_config()
            transformer_cfg.self_attention.attention.num_heads = num_heads
            transformer_cfg.feed_forward.hidden_dim = hidden_dim

            if i == 1:
                transformer_cfg.self_attention.attention.input_linear = QLinear.default_config()

            cfg.layer.append(transformer_cfg)

        layer: StackedTransformerLayer = cfg.instantiate(parent=None)
        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(data=inputs, return_aux={"self_attention_kv_state"}),
        )
        self.assertEqual(
            BaseTransformerLayer.Output(
                data=(batch_size, seq_len, input_dim),
                self_attention_probs=None,
                self_attention_kv_state=KVState(
                    k_proj=(batch_size, seq_len, num_heads, per_head_dim),
                    v_proj=(batch_size, seq_len, num_heads, per_head_dim),
                    key_positions=(1, seq_len),
                ),
                cross_attention_probs=None,
            ),
            shapes(outputs),
        )

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

    @parameterized.product(
        stage_cls=[StackedTransformerLayer, RepeatedTransformerLayer],
        schedule_cls=[GPipeSchedule, StreamSchedule],
        remat_spec=[None, RematSpec(policy=jax_remat_policies.everything_saveable)],
    )
    def test_stack_vs_pipeline(
        self,
        stage_cls: type[BaseTransformerLayer],
        schedule_cls: type[BaseSchedule],
        remat_spec: Optional[RematSpec],
    ):
        pipelined_cfg: PipelinedTransformerLayer.Config = PipelinedTransformerLayer.default_config()
        pipelined_cfg.stage = stage_cls.default_config().set(layer=None)
        pipelined_cfg.pipeline.schedule = schedule_cls.default_config()

        # If using StreamSchedule, we expect `num_microbatches` to be divisible by `num_stages`.
        if schedule_cls is StreamSchedule:
            # num_microbatches = 6, num_stages = 3, microbatch_size = 2
            batch_size, num_layers = 12, 6
        else:
            # num_microbatches = 5, num_stages = 3, microbatch_size = 2
            batch_size, num_layers = 10, 6

        pipelined_cfg.num_microbatches = batch_size // 2
        pipelined_cfg.num_stages = num_layers // 2
        self._compare_layers(
            StackedTransformerLayer,
            pipelined_cfg,
            remat_spec=remat_spec,
            batch_size=batch_size,
            num_layers=num_layers,
        )

    # pylint: disable-next=too-many-statements,too-many-branches
    def _compare_layers(
        self,
        *stack_configs,
        dtype=jnp.float32,
        remat_spec=None,
        batch_size: int = 10,
        num_layers: int = 6,
    ):
        assert stack_configs[0] == StackedTransformerLayer, stack_configs[0]
        with utils.numeric_checks(False):
            tgt_len, model_dim, num_heads = 5, 8, 4

            target = jax.random.normal(
                jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
            )
            rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
            rand_mask = jnp.tile(rand_mask[None, None, :, :], (batch_size, num_heads, 1, 1))

            all_params = []
            all_outputs = []
            all_gradients = []
            all_updates = []
            stacked_layer_params = None
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
                layer: _StackModel = cfg.instantiate(parent=None)

                param_specs = layer.create_parameter_specs_recursively()
                logging.info(
                    "%s.factorization_specs=%s",
                    cls,
                    jax.tree.map(lambda x: x.factorization, param_specs),
                )
                layer_params = layer.initialize_parameters_recursively(
                    prng_key=jax.random.PRNGKey(123)
                )
                logging.info(
                    "%s.params=%s",
                    cls,
                    [
                        f"{path}={value.dtype}({value.shape})"
                        for path, value in flatten_items(layer_params)
                    ],
                )
                if cls == StackedTransformerLayer:
                    stacked_layer_params = copy.deepcopy(layer_params)
                else:
                    layer_params = _convert_from_stacked_params(
                        stacked_layer_params, target_stack_cfg=cfg.stack
                    )
                    logging.info(
                        "Converted: %s.params=%s",
                        cls,
                        [
                            f"{path}={value.dtype}({value.shape})"
                            for path, value in flatten_items(layer_params)
                        ],
                    )

                def _loss(layer_params, data, mask, layer=layer):
                    layer_outputs, layer_output_collection = F(
                        layer,
                        inputs=dict(
                            data=data, self_attention_logit_biases=mask, target_segment_ids=None
                        ),
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

                # Note that we do not compare summaries across stack layer types because:
                # (1) attention layers do not emit summaries yet;
                # (2) pipelines emit per-microbatch summaries which have a different structure
                #     than summaries from other stack layers.
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
                opt_params = jax.tree.map(
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
                    update_norms = jax.tree.map(rms_norm, updates)
                else:
                    update_norms = jax.vmap(lambda x, norm=rms_norm: jax.tree.map(norm, x))(updates)
                logging.info(
                    "global_update_norm=%s update_norms=%s",
                    optax.global_norm(updates),
                    dict(utils.flatten_items(update_norms)),
                )

                if cls == StackedTransformerLayer:
                    for x in (layer_params, grads, updates):
                        x["stack"] = _recursive_stack(x["stack"])

                if cls == RepeatedTransformerLayer:
                    for x in (layer_params, grads, updates):
                        x["stack"] = x["stack"]["repeat"]

                if cls == PipelinedTransformerLayer:
                    for x in (layer_params, grads, updates):
                        logging.info("x=%s", shapes(x))
                        if cfg.stack.stage.klass == StackedTransformerLayer:
                            # First stack within each stage.
                            x["stack"]["pipeline"]["layer"] = _recursive_stack(
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
                        x["stack"] = jax.tree.map(
                            lambda x: x.reshape([num_layers] + list(x.shape[2:])),
                            x["stack"]["pipeline"]["layer"],
                        )

                all_params.append(layer_params)
                all_outputs.append(layer_outputs)
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
            self.assertNestedAllClose(all_outputs[0], all_outputs[1])
            self.assertNestedAllClose(all_gradients[0], all_gradients[1])
            self.assertNestedAllClose(all_updates[0], all_updates[1])

    @parameterized.parameters(StackedTransformerLayer, RepeatedTransformerLayer)
    def test_stacked_decoding(self, stack_cls):
        model_dim, num_heads = 6, 2
        cfg = stack_cls.default_config().set(num_layers=5, input_dim=model_dim)
        layer_cfg = cfg.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads, causal=True)
        layer_cfg.feed_forward.hidden_dim = model_dim * 4
        self._test_forward_vs_extend_step(cfg)
        self._test_decoder_with_transformer(cfg)

    @parameterized.product(
        outer_stack_cls=(StackedTransformerLayer, RepeatedTransformerLayer),
        inner_stack_cls=(StackedTransformerLayer, RepeatedTransformerLayer),
    )
    def test_nested_stacked_decoding(self, outer_stack_cls, inner_stack_cls):
        model_dim, num_heads = 6, 2
        cfg = outer_stack_cls.default_config().set(num_layers=2, input_dim=model_dim)
        cfg.layer = inner_stack_cls.default_config().set(num_layers=3)
        layer_cfg = cfg.layer.layer
        layer_cfg.self_attention.attention.set(num_heads=num_heads, causal=True)
        layer_cfg.feed_forward.hidden_dim = model_dim * 4
        self._test_forward_vs_extend_step(cfg)
        self._test_decoder_with_transformer(cfg)

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

    @parameterized.product(is_training=(True, False))
    def test_stacked_transformer_with_non_uniform_layers(self, is_training):
        """Tests that a custom StackedTransformerLayer can support non-uniform layers."""
        batch_size = 2
        seq_len = 16
        input_dim = 4
        hidden_dim = 16
        num_layers = 2

        # Create a StackedTransformerLayer by specifying a sequence of non-uniform layer configs.
        cfg = NonUniformStack.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.num_layers = num_layers
        cfg.layer = []
        for i in range(num_layers):
            transformer_cfg = TransformerLayer.default_config()
            # Different numbers of heads between the layers.
            transformer_cfg.self_attention.attention.num_heads = 2 if i == 0 else 1
            transformer_cfg.feed_forward.hidden_dim = hidden_dim
            cfg.layer.append(transformer_cfg)
        layer: StackedTransformerLayer = cfg.instantiate(parent=None)
        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        outputs, _ = F(
            layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(data=inputs, return_aux={"self_attention_kv_state"}),
        )
        self.assertEqual(
            BaseTransformerLayer.Output(
                data=(batch_size, seq_len, input_dim),
                self_attention_probs=None,
                self_attention_kv_state=KVState(
                    k_proj=(batch_size, seq_len, 1, 4),
                    v_proj=(batch_size, seq_len, 1, 4),
                    key_positions=(1, seq_len),
                ),
                cross_attention_probs=None,
            ),
            shapes(outputs),
        )

    @parameterized.parameters(
        [None, False],
        [("data",), False],
        [("data",), True],
        [("data", "self_attention_kv_state"), True],
    )
    @set_threefry_partitionable(True)  # TODO(mhopkins): remove after jax 0.5.0
    def test_repeated_layer_with_custom_carry(self, repeat_carry, precomputed_kv_state):
        """Tests RepeatedTransformerLayer with customized `carry`."""
        batch_size = 1
        seq_len = 16
        input_dim = 4
        num_heads = 2
        head_dim = input_dim // num_heads
        num_layers = 3

        cfg = self._stack_config(
            RepeatedTransformerLayer,
            num_layers=num_layers,
            model_dim=input_dim,
            num_heads=num_heads,
            dtype=jnp.float32,
            remat_spec=None,
            output_self_attention_kv_state=True,
        )
        cfg.stack.repeat.carry = repeat_carry
        cfg.stack.layer.remat_spec = build_remat_spec(cfg.stack)
        if precomputed_kv_state:
            kv_shape = (batch_size, seq_len, num_heads, head_dim)
            kv_state = KVState(
                k_proj=jax.random.normal(key=jax.random.PRNGKey(1), shape=kv_shape),
                v_proj=jax.random.normal(key=jax.random.PRNGKey(2), shape=kv_shape),
                key_positions=jnp.arange(seq_len)[None],
            )
            cfg.stack.layer.self_attention.attention.input_linear = QLinear.default_config()
            expected_output = 0.7333336
        else:
            kv_state = None
            # carry=None and carry=("data",) are equivalent.
            expected_output = 0.9357959

        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        outputs, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(
                data=inputs,
                self_attention_kv_state=kv_state,
                return_aux={"self_attention_kv_state"},
            ),
        )
        self.assertNestedAllClose(expected_output, outputs[0])
        if precomputed_kv_state:
            self.assertNestedAllClose(kv_state, outputs[1]["self_attention_kv_state"])
        else:
            self.assertIsInstance(outputs[1]["self_attention_kv_state"], KVState)

    def test_pipeline_return_aux(self):
        batch_size, num_heads, seq_len, dim = 2, 3, 4, 6

        class DummyTransformerLayer(TransformerLayer):
            def forward(self, data, **kwargs):
                return TransformerLayer.Output(
                    data=data,
                    self_attention_probs=jnp.empty([batch_size, num_heads, seq_len, seq_len]),
                    self_attention_kv_state=KVState(
                        k_proj=jnp.empty([batch_size, seq_len, num_heads, dim]),
                        v_proj=jnp.empty([batch_size, seq_len, num_heads, dim]),
                        key_positions=jnp.arange(seq_len)[None],
                    ),
                )

        cfg: PipelinedTransformerLayer.Config = PipelinedTransformerLayer.default_config().set(
            num_stages=2,
            num_microbatches=2,
            num_layers=2,
            input_dim=dim,
            layer=DummyTransformerLayer.default_config(),
        )
        cfg.layer.self_attention.attention.set(num_heads=num_heads)
        cfg.layer.feed_forward.hidden_dim = scaled_hidden_dim(4)

        with test_utils.bind_layer(cfg) as layer:
            data = jax.random.uniform(layer.prng_key, shape=[2, 3, 4])
            out = layer(data, return_aux={"self_attention_kv_state"})
            self.assertNestedAllClose(data, out.data)
            self.assertIsNone(out.self_attention_probs)
            self.assertIsNotNone(out.self_attention_kv_state)

    @parameterized.parameters(
        ([],),
        (["self_attention"],),
        (["feed_forward"],),
        (["self_attention", "feed_forward"],),
    )
    def test_initialize_parameters_recursively(self, prebuilt_layers: list[str]):
        """Tests initialize_parameters_recursively with various prebuilt layers."""
        input_dim = 4
        num_heads = 2
        num_layers = 3

        cfg = self._stack_config(
            RepeatedTransformerLayer,
            num_layers=num_layers,
            model_dim=input_dim,
            num_heads=num_heads,
            dtype=jnp.float32,
            remat_spec=None,
            output_self_attention_kv_state=True,
        )
        cfg.stack.layer.remat_spec = build_remat_spec(cfg.stack)
        layer = cfg.instantiate(parent=None)
        param_specs = layer.create_parameter_specs_recursively()
        initialized_from_scratch = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )
        jax.tree_util.tree_map_with_path(
            lambda path, spec, param: self.assertEqual(param.shape, spec.shape, path),
            param_specs,
            initialized_from_scratch,
        )

        def has_prebuilt_layers(path):
            for prebuilt_layer in prebuilt_layers:
                for part in path:
                    if prebuilt_layer == part.key:
                        return True
            return False

        # ParameterSpec for a prebuilt param, None otherwise.
        prebuilt_specs = jax.tree_util.tree_map_with_path(
            lambda path, spec: spec if has_prebuilt_layers(path) else None, param_specs
        )
        if prebuilt_layers:
            self.assertNotEmpty(jax.tree_util.tree_leaves(prebuilt_specs))
        initialized_state = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123), prebuilt=prebuilt_specs
        )

        def validate_initialized(path, spec, initialized, prebuilt):
            if prebuilt is None:
                self.assertEqual(spec.shape, initialized.shape, path)
            else:
                self.assertIsNone(initialized)

        jax.tree_util.tree_map_with_path(
            validate_initialized, param_specs, initialized_state, prebuilt_specs
        )


class ConfigHelperTest(TestCase):
    """Tests config utils."""

    @parameterized.product(
        input_linear_cfg=(
            QKVLinear.default_config(),
            FusedQKVLinear.default_config(),
            RoFormerQKVLinear.default_config().set(input_linear=FusedQKVLinear.default_config()),
        ),
        fsdp_axis_names=("fsdp",),
        tp_axis_names=("model",),
    )
    def test_set_attn_partition_specs(
        self,
        input_linear_cfg,
        fsdp_axis_names,
        tp_axis_names,
    ):
        cfg = attention.MultiheadAttention.default_config()
        cfg.input_linear = input_linear_cfg
        attention.set_attention_partition_specs(
            cfg,
            fsdp_axis_names=fsdp_axis_names,
            tp_axis_names=tp_axis_names,
        )

        input_linear = cfg.input_linear
        if isinstance(input_linear_cfg, RoFormerQKVLinear.Config):
            input_linear = input_linear.input_linear
        # Shard weights.
        self.assertSequenceEqual(
            input_linear.layer.param_partition_spec, (fsdp_axis_names, tp_axis_names, None)
        )
        self.assertSequenceEqual(
            cfg.output_linear.param_partition_spec, (fsdp_axis_names, tp_axis_names, None)
        )

    @parameterized.product(
        batch_axis_names=("data", ("replica", "data", "fsdp")),
        fsdp_axis_names=("fsdp",),
        tp_axis_names=("model",),
        seq_axis_names=("seq",),
    )
    def test_set_ffn_partition_specs(
        self,
        batch_axis_names,
        fsdp_axis_names,
        tp_axis_names,
        seq_axis_names,
    ):
        cfg = TransformerFeedForwardLayer.default_config()
        attention.set_feed_forward_partition_specs(
            cfg,
            batch_axis_names=batch_axis_names,
            fsdp_axis_names=fsdp_axis_names,
            tp_axis_names=tp_axis_names,
            seq_axis_names=seq_axis_names,
        )

        self.assertSequenceEqual(cfg.linear1.param_partition_spec, (fsdp_axis_names, tp_axis_names))
        self.assertSequenceEqual(cfg.linear2.param_partition_spec, (tp_axis_names, fsdp_axis_names))
        self.assertSequenceEqual(
            cfg.linear1.output_partition_spec, (batch_axis_names, seq_axis_names, tp_axis_names)
        )
        self.assertSequenceEqual(
            cfg.linear2.output_partition_spec, (batch_axis_names, seq_axis_names, tp_axis_names)
        )

    @parameterized.product(
        self_attention_input_linear_cfg=(
            QKVLinear.default_config(),
            FusedQKVLinear.default_config(),
            RoFormerQKVLinear.default_config().set(input_linear=FusedQKVLinear.default_config()),
        ),
        cross_attention_cfg=(None, TransformerAttentionLayer.default_config()),
        batch_axis_names=("data", ("replica", "data", "fsdp")),
        fsdp_axis_names=("fsdp",),
        tp_axis_names=("model",),
        seq_axis_names=("seq",),
    )
    def test_set_double_shard_weights_config(
        self,
        self_attention_input_linear_cfg,
        cross_attention_cfg,
        batch_axis_names,
        fsdp_axis_names,
        tp_axis_names,
        seq_axis_names,
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
            seq_axis_names=seq_axis_names,
        )

        ff_layer = cfg.feed_forward
        self.assertSequenceEqual(
            ff_layer.linear1.param_partition_spec, (fsdp_axis_names, tp_axis_names)
        )
        self.assertSequenceEqual(
            ff_layer.linear2.param_partition_spec, (tp_axis_names, fsdp_axis_names)
        )
        self.assertSequenceEqual(
            ff_layer.linear1.output_partition_spec,
            (batch_axis_names, seq_axis_names, tp_axis_names),
        )
        self.assertSequenceEqual(
            ff_layer.linear2.output_partition_spec,
            (batch_axis_names, seq_axis_names, tp_axis_names),
        )

        self_atten = cfg.self_attention.attention
        input_linear = self_atten.input_linear
        if isinstance(self_attention_input_linear_cfg, RoFormerQKVLinear.Config):
            input_linear = input_linear.input_linear
        # Shard weights.
        self.assertSequenceEqual(
            input_linear.layer.param_partition_spec,
            (fsdp_axis_names, tp_axis_names, None),
        )
        self.assertSequenceEqual(
            self_atten.output_linear.param_partition_spec, (fsdp_axis_names, tp_axis_names, None)
        )

        if cross_attention_cfg is None:
            self.assertIsNone(cfg.cross_attention)
        else:
            cross_atten = cfg.cross_attention.attention
            # Shard weights.
            self.assertSequenceEqual(
                cross_atten.input_linear.layer.param_partition_spec,
                (fsdp_axis_names, tp_axis_names, None),
            )
            self.assertSequenceEqual(
                cross_atten.output_linear.param_partition_spec,
                (fsdp_axis_names, tp_axis_names, None),
            )

    @parameterized.product(
        self_attention_input_linear_cfg=(
            QKVLinear.default_config(),
            FusedQKVLinear.default_config(),
        ),
        cross_attention_cfg=(None, TransformerAttentionLayer.default_config()),
        batch_axis_names=("data", ("replica", "data", "fsdp")),
        fsdp_axis_names=("fsdp",),
        tp_axis_names=("model",),
        seq_axis_names=("seq",),
    )
    def test_set_double_shard_weights_config_for_list_of_configs(
        self,
        self_attention_input_linear_cfg,
        cross_attention_cfg,
        batch_axis_names,
        fsdp_axis_names,
        tp_axis_names,
        seq_axis_names,
    ):
        cfg_layer: TransformerLayer.Config = TransformerLayer.default_config().set(
            cross_attention=cross_attention_cfg
        )
        cfg_layer.self_attention.attention.input_linear = self_attention_input_linear_cfg
        cfg_layers = [cfg_layer, cfg_layer]
        set_double_shard_weights_config(
            cfg_layers,
            batch_axis_names=batch_axis_names,
            fsdp_axis_names=fsdp_axis_names,
            tp_axis_names=tp_axis_names,
            seq_axis_names=seq_axis_names,
        )

        for cfg in cfg_layers:
            ff_layer = cfg.feed_forward
            self.assertSequenceEqual(
                ff_layer.linear1.param_partition_spec, (fsdp_axis_names, tp_axis_names)
            )
            self.assertSequenceEqual(
                ff_layer.linear2.param_partition_spec, (tp_axis_names, fsdp_axis_names)
            )
            self.assertSequenceEqual(
                ff_layer.linear1.output_partition_spec,
                (batch_axis_names, seq_axis_names, tp_axis_names),
            )
            self.assertSequenceEqual(
                ff_layer.linear2.output_partition_spec,
                (batch_axis_names, seq_axis_names, tp_axis_names),
            )

            self_atten = cfg.self_attention.attention
            # Shard weights.
            self.assertSequenceEqual(
                self_atten.input_linear.layer.param_partition_spec,
                (fsdp_axis_names, tp_axis_names, None),
            )
            self.assertSequenceEqual(
                self_atten.output_linear.param_partition_spec,
                (fsdp_axis_names, tp_axis_names, None),
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
        self_attention_logit_biases = attention_bias.make_causal_biases(tgt_len)

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


class LogitSinkTest(TestCase):
    """Tests logit_sink functionality in MultiheadAttention."""

    def test_logit_sink_basic_functionality(self):
        """Test that logit_sink is properly applied in softmax computation."""
        model_dim = 16
        num_heads = 4
        batch_size = 2
        seq_len = 6

        # Create attention layer with logit_sink enabled
        cfg = attention.MultiheadAttention.default_config().set(
            name="test_attention",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            logit_sink=True,
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Check that sink parameter exists and has correct shape
        self.assertIn("sink", layer_params)
        self.assertEqual(layer_params["sink"].shape, (num_heads,))

        # Test forward pass
        query = jax.random.normal(
            jax.random.PRNGKey(0), [batch_size, seq_len, model_dim], dtype=jnp.float32
        )

        outputs, _ = F(
            layer,
            inputs=dict(query=query, return_aux={"probs"}),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
        )

        # Check output shapes
        self.assertEqual(outputs.data.shape, (batch_size, seq_len, model_dim))
        self.assertEqual(outputs.probs.shape, (batch_size, num_heads, seq_len, seq_len))

        # Check that probabilities sum to less than 1.0.
        prob_sums = jnp.sum(outputs.probs, axis=-1)
        np.testing.assert_array_less(prob_sums, 1.0)

    def test_logit_sink_vs_no_logit_sink(self):
        """Test that logit_sink affects attention probabilities."""
        model_dim = 8
        num_heads = 2
        batch_size = 1
        seq_len = 4

        # Create two identical layers, one with logit_sink, one without
        base_cfg = attention.MultiheadAttention.default_config().set(
            name="test_attention",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )

        layer_no_sink = base_cfg.set(logit_sink=False).instantiate(parent=None)
        layer_with_sink = base_cfg.set(logit_sink=True).instantiate(parent=None)

        # Initialize with same parameters (except sink)
        init_key = jax.random.PRNGKey(123)
        params_no_sink = layer_no_sink.initialize_parameters_recursively(prng_key=init_key)
        params_with_sink = layer_with_sink.initialize_parameters_recursively(prng_key=init_key)

        # Copy non-sink parameters to ensure they're identical
        for key in params_no_sink:
            if key in params_with_sink:
                params_with_sink[key] = params_no_sink[key]

        # Set sink to non-zero values to see effect
        params_with_sink["sink"] = jnp.array([1.0, -0.5], dtype=jnp.float32)

        query = jax.random.normal(
            jax.random.PRNGKey(0), [batch_size, seq_len, model_dim], dtype=jnp.float32
        )

        # Get outputs from both layers
        outputs_no_sink, _ = F(
            layer_no_sink,
            inputs=dict(query=query, return_aux={"probs"}),
            state=params_no_sink,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
        )

        outputs_with_sink, _ = F(
            layer_with_sink,
            inputs=dict(query=query, return_aux={"probs"}),
            state=params_with_sink,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
        )

        # Probabilities should be different due to logit sink
        self.assertFalse(jnp.allclose(outputs_no_sink.probs, outputs_with_sink.probs, atol=1e-6))

        prob_sums = jnp.sum(outputs_with_sink.probs, axis=-1)
        np.testing.assert_array_less(prob_sums, 1.0)

        prob_sums = jnp.sum(outputs_no_sink.probs, axis=-1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

    @parameterized.parameters(
        dict(
            model_dim=8,
            num_heads=2,
            batch_size=1,
            seq_len=4,
            sink_values=[0.5, -1.0],
        ),
        dict(
            model_dim=16,
            num_heads=4,
            batch_size=2,
            seq_len=6,
            sink_values=[1.0, -0.5, 0.0, 0.2],
        ),
    )
    def test_logit_sink_with_attention_biases(
        self, model_dim, num_heads, batch_size, seq_len, sink_values
    ):
        """Test logit_sink works correctly with attention biases."""
        cfg = attention.MultiheadAttention.default_config().set(
            name="test_attention",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            logit_sink=True,
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Set sink values
        layer_params["sink"] = jnp.array(sink_values, dtype=jnp.float32)

        query = jax.random.normal(
            jax.random.PRNGKey(0), [batch_size, seq_len, model_dim], dtype=jnp.float32
        )

        # Create causal attention biases
        attention_logit_biases = attention_bias.make_causal_biases(seq_len)

        outputs, _ = F(
            layer,
            inputs=dict(
                query=query, attention_logit_biases=attention_logit_biases, return_aux={"probs"}
            ),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
        )

        # Check that causal masking is still applied (upper triangle should be ~0)
        probs = outputs.probs[0, 0]  # First batch, first head
        upper_triangle = jnp.triu(probs, k=1)
        self.assertTrue(jnp.all(upper_triangle < 1e-6))

        # Check that probabilities still sum to less than 1
        prob_sums = jnp.sum(outputs.probs, axis=-1)
        np.testing.assert_array_less(prob_sums, 1.0)

    def test_logit_sink_parameter_initialization(self):
        """Test that logit_sink parameters are initialized correctly."""
        model_dim = 12
        num_heads = 3

        cfg = attention.MultiheadAttention.default_config().set(
            name="test_attention",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            logit_sink=True,
        )
        layer = cfg.instantiate(parent=None)

        # Check parameter specs
        param_specs = layer.create_parameter_specs_recursively()
        self.assertIn("sink", param_specs)
        sink_spec = param_specs["sink"]
        self.assertEqual(sink_spec.shape, (num_heads,))
        self.assertEqual(sink_spec.mesh_axes, ("model",))
        self.assertEqual(sink_spec.weight_decay_scale, 0.0)

    def test_logit_sink_disabled_by_default(self):
        """Test that logit_sink is disabled by default."""
        cfg = attention.MultiheadAttention.default_config().set(
            name="test_attention",
            query_dim=8,
            key_dim=8,
            value_dim=8,
            num_heads=2,
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Should not have sink parameter when logit_sink is None/False
        self.assertNotIn("sink", layer_params)

    @parameterized.parameters(
        dict(dtype=jnp.float32),
        dict(dtype=jnp.bfloat16),
        dict(dtype=jnp.float16),
    )
    def test_logit_sink_with_different_dtypes(self, dtype):
        """Test logit_sink works with different data types."""
        model_dim = 8
        num_heads = 2
        batch_size = 1
        seq_len = 3

        cfg = attention.MultiheadAttention.default_config().set(
            name="test_attention",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            logit_sink=True,
            dtype=dtype,
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        query = jax.random.normal(
            jax.random.PRNGKey(0), [batch_size, seq_len, model_dim], dtype=dtype
        )

        outputs, _ = F(
            layer,
            inputs=dict(query=query, return_aux={"probs"}),
            state=layer_params,
            is_training=False,
            prng_key=jax.random.PRNGKey(456),
        )

        # Check output dtype matches input
        self.assertEqual(outputs.data.dtype, dtype)
        prob_sums = jnp.sum(outputs.probs, axis=-1)
        np.testing.assert_array_less(prob_sums, 1.0)

    @parameterized.parameters(
        dict(use_attention_biases=False),
        dict(use_attention_biases=True),
    )
    def test_softmax_with_biases_logit_sink(self, use_attention_biases):
        """Test the softmax_with_biases function with logit_sink in various scenarios."""
        batch_size = 2
        num_heads = 3
        seq_len = 4
        logit_sink_values = [1.0, -0.5, 0.0]

        # Create test logits
        logits = jax.random.normal(
            jax.random.PRNGKey(0), [batch_size, num_heads, seq_len, seq_len], dtype=jnp.float32
        )

        # Create logit sink
        logit_sink = jnp.array(logit_sink_values, dtype=jnp.float32)

        # Create attention biases if needed
        attention_logit_biases = None
        if use_attention_biases:
            attention_logit_biases = attention_bias.make_causal_biases(seq_len)
        else:
            attention_logit_biases = None

        # Test without logit sink
        probs_no_sink = attention.softmax_with_biases(
            logits, attention_logit_biases=attention_logit_biases
        )

        # Test with logit sink
        probs_with_sink = attention.softmax_with_biases(
            logits, attention_logit_biases=attention_logit_biases, logit_sink=logit_sink
        )

        # Both should have same shape
        self.assertEqual(probs_no_sink.shape, probs_with_sink.shape)

        # Check causal masking is applied (upper triangle should be ~0)
        if use_attention_biases:
            upper_triangle = jnp.triu(probs_with_sink[0, 0], k=1)
            self.assertTrue(jnp.all(upper_triangle < 1e-6))

        prob_sums = jnp.sum(probs_with_sink, axis=-1)
        np.testing.assert_array_less(prob_sums, 1.0)

        prob_sums = jnp.sum(probs_no_sink, axis=-1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

        # Results should be different
        self.assertFalse(jnp.allclose(probs_no_sink, probs_with_sink, atol=1e-6))


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
