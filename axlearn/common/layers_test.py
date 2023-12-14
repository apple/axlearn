# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests basic layers."""
# pylint: disable=no-self-use,too-many-lines,too-many-public-methods
import copy
import itertools
import math
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax.random
import numpy as np
import tensorflow as tf
import torch
from absl.testing import absltest, parameterized
from jax import nn
from jax import numpy as jnp
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score

from axlearn.common import module, utils
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import config_class
from axlearn.common.decoder import Decoder
from axlearn.common.layers import (
    BatchNorm,
    BinaryClassificationMetric,
    CategoricalHingeLossMetric,
    ClassificationMetric,
    Conv1D,
    Conv2D,
    Conv2DTranspose,
    Conv2DWith1DPadding,
    Conv3D,
    DepthwiseConv1D,
    DropToken,
    Embedding,
    GroupNorm,
    L2Norm,
    LayerNorm,
    LayerNormStateless,
    Linear,
    MaxPool2D,
    MultiLinear,
    RedirectToSharedModule,
    RMSNorm,
    SeparableSpaceTimePositionalEmbedding,
    SqueezeExcitation,
    StackOverTime,
    StochasticDepth,
    UnitNormLinear,
    VariationalNoise,
    _compute_conv_output_1d_padding,
    _compute_moments_with_paddings,
    get_activation_fn,
    get_stochastic_depth_linear_rate,
    set_bias_recursively,
    set_dropout_rate_recursively,
    set_layer_norm_eps_recursively,
    set_norm_recursively,
)
from axlearn.common.module import Module, Tensor, child_context
from axlearn.common.module import functional as F
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.param_init import ConstantInitializer, FanAxes
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import as_tensor, flatten_items, shapes


def _copy(src: jnp.ndarray, dst: torch.nn.Parameter):
    with torch.no_grad():
        src = np.asarray(src).copy()
        src = torch.as_tensor(src)
        dst.copy_(src)


# pylint: disable=too-many-public-methods
class LayerTest(TestCase, tf.test.TestCase):
    @parameterized.parameters(
        "linear",
        "nn.relu",
        "nn.sigmoid",
        "nn.silu",
        "nn.swish",
        "nn.gelu",
        "nn.tanh",
        "jnp.tanh",
    )
    def test_get_activation_fn(self, name):
        fn = get_activation_fn(name)
        # Random inputs.
        inputs = jax.random.normal(jax.random.PRNGKey(123), [10])
        outputs = fn(inputs)
        if name == "linear":
            self.assertNestedAllClose(outputs, inputs)
        elif name.startswith("nn."):
            self.assertNestedAllClose(outputs, getattr(jax.nn, name[3:])(inputs))
        else:
            self.assertNestedAllClose(outputs, getattr(jax.numpy, name[4:])(inputs))

    def test_layer_norm(self):
        dim = 6
        cfg = LayerNorm.default_config().set(name="norm", input_dim=dim)
        layer = cfg.instantiate(parent=None)  # type: LayerNorm

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(dict(scale=(dim,), bias=(dim,)), shapes(layer_params))

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        # forward() should not mutate 'inputs' in-place.
        assert_allclose(inputs, orig_inputs)
        # The output mean should be close to 0.
        output_mean = outputs.mean(axis=-1, keepdims=True)
        assert_allclose(output_mean, np.zeros_like(output_mean))
        # The output variance should be close to 1.
        output_var = ((outputs - output_mean) ** 2).mean(axis=-1)
        assert_allclose(output_var, np.ones_like(output_var))

        # Set scales to 2.
        layer_params2 = copy.deepcopy(layer_params)
        layer_params2["scale"] *= 2
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params2,
            prng_key=prng_key,
        )
        # The output mean should be close to 0.
        output_mean = outputs.mean(axis=-1, keepdims=True)
        assert_allclose(output_mean, np.zeros_like(output_mean))
        # The output variance should be close to 4.
        output_var = ((outputs - output_mean) ** 2).mean(axis=-1)
        assert_allclose(output_var, np.ones_like(output_var) * 4)

    def test_layer_norm_statefree(self):
        dim = 6
        cfg = LayerNormStateless.default_config().set(name="norm", input_dim=dim)
        layer = cfg.instantiate(parent=None)  # type: LayerNormStateless

        # Random inputs.
        prng_key = jax.random.PRNGKey(123)
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state={},
            prng_key=prng_key,
        )

        ref_ln = torch.nn.LayerNorm(dim, eps=1e-8, elementwise_affine=False)
        # forward() should not mutate 'inputs' in-place.
        assert_allclose(inputs, orig_inputs)
        assert_allclose(outputs, as_tensor(ref_ln(as_torch_tensor(orig_inputs))))

    @parameterized.parameters(
        [
            dict(inputs_shape=[2, 3, 6]),
            dict(inputs_shape=[2, 3, 9], paddings=jnp.array([[0, 1, 1], [0, 0, 0]])),
            dict(inputs_shape=[3, 3, 4, 12], paddings=jnp.array([[1, 1, 1], [0, 0, 1], [0, 1, 1]])),
            dict(inputs_shape=[2, 3, 6], scale_params=jnp.array([0, 0, 1, 1, 2, 2])),
            dict(
                inputs_shape=[2, 3, 4],
                paddings=jnp.array([[0, 1, 1], [0, 0, 0]]),
                num_groups=2,
                scale_params=jnp.array([1, 1, 5, 5]),
            ),
            dict(
                inputs_shape=[3, 3, 7, 4],
                paddings=jnp.array([[1, 1, 1], [0, 0, 1], [0, 1, 1]]),
                num_groups=2,
                scale_params=jnp.array([2, 2, 3, 3]),
            ),
        ]
    )
    # pylint: disable-next=too-many-statements,too-many-branches
    def test_group_norm(self, inputs_shape, *, paddings=None, num_groups=3, scale_params=None):
        batch_size = inputs_shape[0]
        dim = inputs_shape[-1]
        cfg = GroupNorm.default_config().set(name="norm", input_dim=dim, num_groups=num_groups)
        layer = cfg.instantiate(parent=None)  # type: GroupNorm

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        if scale_params is not None:
            # Set scales.
            layer_params["scale"] = scale_params

        self.assertEqual(dict(scale=(dim,), bias=(dim,)), shapes(layer_params))

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, inputs_shape)
        inputs = orig_inputs.copy()

        outputs, _ = F(
            layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        # forward() should not mutate "inputs" in-place.
        assert_allclose(inputs, orig_inputs)
        reduction_axis = (1, -1) if len(inputs_shape) == 3 else (1, 2, -1)
        outputs_by_group = jnp.reshape(outputs, inputs_shape[:-1] + [num_groups, dim // num_groups])

        if paddings is None:
            output_mean = outputs_by_group.mean(axis=reduction_axis, keepdims=True)
            output_var = ((outputs_by_group - output_mean) ** 2).mean(axis=(1, -1), keepdims=True)
        else:
            expanded_paddings = (
                paddings[:, :, None, None]
                if len(outputs_by_group.shape) == 4
                else paddings[:, :, None, None, None]
            )
            output_sum = jnp.sum(
                outputs_by_group * (1 - expanded_paddings), axis=reduction_axis, keepdims=True
            )
            output_count = jnp.sum(
                jnp.ones_like(outputs_by_group) * (1 - expanded_paddings),
                axis=reduction_axis,
                keepdims=True,
            )
            output_mean = output_sum / jnp.maximum(output_count, 1.0)
            output_var = jnp.sum(
                (outputs_by_group * (1 - expanded_paddings) - output_mean) ** 2,
                axis=reduction_axis,
                keepdims=True,
            ) / jnp.maximum(output_count, 1.0)

        if len(inputs_shape) == 3:
            self.assertEqual(output_mean.shape, (batch_size, 1, num_groups, 1))
            self.assertEqual(output_var.shape, (batch_size, 1, num_groups, 1))
        else:
            self.assertEqual(output_mean.shape, (batch_size, 1, 1, num_groups, 1))
            self.assertEqual(output_var.shape, (batch_size, 1, 1, num_groups, 1))
        # The output group mean should be close to 0.
        assert_allclose(output_mean, np.zeros_like(output_mean), rtol=1e-6, atol=1e-6)
        if scale_params is None:
            # The output variance should be close to 1.
            expected_var = np.ones_like(output_var)
        else:
            # [num_groups].
            expected_var = jnp.reshape(scale_params, [num_groups, dim // num_groups])[:, 0] ** 2
            expected_var = jnp.tile(expected_var, [batch_size, 1])
            if len(inputs_shape) == 3:
                expected_var = jnp.expand_dims(expected_var, axis=(1, 3))
            else:
                expected_var = jnp.expand_dims(expected_var, axis=(1, 2, 4))

        if paddings is not None:
            expected_var = expected_var * (
                jnp.sum(1 - expanded_paddings, axis=1, keepdims=True) > 0
            )
        assert_allclose(output_var, expected_var, atol=1e-6, rtol=1e-6)

    @parameterized.parameters(
        ((2, 10, 4, 3, 2), [1, 2, -1]),
        ((4, 20, 2, 3), [1, -1]),
        ((4, 15, 3, 2), [0, 1, 2]),
        ((2, 10, 2, 4, 3), [0, 1, 2, 3]),
    )
    def test_compute_moments(self, input_shape, reduction_axis):
        batch_size, seq_len = input_shape[:2]
        k1, k2 = jax.random.split(jax.random.PRNGKey(721))
        x = jax.random.normal(key=k1, shape=input_shape) * 10 + 2
        lengths = jax.random.randint(key=k2, shape=(batch_size - 1,), minval=0, maxval=seq_len)
        lengths = jnp.append(lengths, 0)
        paddings = jnp.arange(seq_len)[None, :] >= lengths[:, None]
        mean, variance = _compute_moments_with_paddings(
            x=x,
            paddings=paddings,
            reduction_axis=reduction_axis,
            keepdims=True,
        )
        if -1 in reduction_axis:
            group_size = input_shape[-2]
            expected_mean = np.zeros([batch_size, group_size])
            expected_var = np.zeros([batch_size, group_size])
            for i in range(batch_size):
                if lengths[i] > 0:
                    expected_mean[i] = jnp.mean(
                        x[i, : lengths[i]], axis=list(range(0, len(input_shape) - 3)) + [-1]
                    )
                    expected_var[i] = jnp.var(
                        x[i, : lengths[i]], axis=list(range(0, len(input_shape) - 3)) + [-1]
                    )
                else:
                    expected_mean[i] = np.zeros(group_size)
                    expected_var[i] = np.zeros(group_size)
        else:
            sum_x = np.zeros(input_shape[-1])
            sum_x2 = np.zeros(input_shape[-1])
            count = 0
            for i in range(batch_size):
                if lengths[i] > 0:
                    sum_x += jnp.sum(x[i, : lengths[i]], axis=list(range(0, len(input_shape) - 2)))
                    sum_x2 += jnp.sum(
                        x[i, : lengths[i]] ** 2, axis=list(range(0, len(input_shape) - 2))
                    )
                    count += lengths[i] * math.prod(input_shape[2:-1])
            expected_mean = sum_x / np.maximum(count, 1)
            expected_var = sum_x2 / np.maximum(count, 1) - expected_mean**2

        self.assertAllClose(jnp.squeeze(mean, axis=reduction_axis), expected_mean)
        self.assertAllClose(jnp.squeeze(variance, axis=reduction_axis), expected_var)

    def test_layer_norm_against_torch(self):
        dim = 6
        cfg = LayerNorm.default_config().set(name="norm", input_dim=dim)
        layer = cfg.instantiate(parent=None)  # type: LayerNorm
        ref_layer = torch.nn.LayerNorm(dim, eps=cfg.eps)
        inputs = jax.random.normal(jax.random.PRNGKey(123), [2, 5, dim])
        test_outputs, ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(x=as_tensor(inputs)),
            ref_inputs=as_torch_tensor(inputs),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )
        assert_allclose(test_outputs, as_tensor(ref_outputs))

    def test_rms_norm(self):
        dim = 6
        cfg = RMSNorm.default_config().set(name="norm", input_dim=dim)
        layer: RMSNorm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(dict(scale=(dim,)), shapes(layer_params))

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        # forward() should not mutate 'inputs' in-place.
        assert_allclose(inputs, orig_inputs)
        # The output_norm should be close to sqrt(dim).
        output_norm = jnp.sqrt((outputs**2).sum(axis=-1))
        assert_allclose(output_norm, np.ones_like(output_norm) * math.sqrt(dim))

        # Set scales to 2.
        layer_params2 = copy.deepcopy(layer_params)
        layer_params2["scale"] *= 2
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params2,
            prng_key=prng_key,
        )
        output_norm = jnp.sqrt((outputs**2).sum(axis=-1))
        # The output_norm should be close to 2 * sqrt(dim).
        assert_allclose(output_norm, np.ones_like(output_norm) * 2.0 * math.sqrt(dim))

    def test_l2_norm(self):
        cfg = L2Norm.default_config().set(name="norm")
        layer: L2Norm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, 10])
        inputs = orig_inputs.copy()

        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )
        # forward() should not mutate 'inputs' in-place.
        assert_allclose(inputs, orig_inputs)
        # The output_norm should be close to sqrt(dim).
        output_norm = jnp.sqrt((outputs**2).sum(axis=-1))
        assert_allclose(output_norm, np.ones_like(output_norm))

    @parameterized.parameters(
        [
            dict(inputs_shape=[2, 3, 6], paddings=None),
            dict(inputs_shape=[2, 5, 6], paddings=jnp.array([[0, 0, 0, 0, 1], [0, 0, 1, 1, 1]])),
            dict(inputs_shape=[2, 3, 6], paddings=jnp.array([[1, 1, 1], [1, 1, 1]])),
        ]
    )
    def test_batch_norm(self, inputs_shape, paddings):
        dim = inputs_shape[-1]
        cfg = BatchNorm.default_config().set(name="norm", input_dim=dim)
        layer: BatchNorm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(scale=(dim,), bias=(dim,), moving_mean=(dim,), moving_variance=(dim,)),
            shapes(layer_params),
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, inputs_shape)
        inputs = orig_inputs.copy()

        for is_training in (True, False):
            outputs, output_collection = F(
                layer,
                inputs=dict(x=inputs, paddings=paddings),
                is_training=is_training,
                state=layer_params,
                prng_key=prng_key,
            )
            param_updates = output_collection.state_updates
            if is_training:
                if paddings is None:
                    # The output mean should be close to 0.
                    output_mean = jnp.mean(outputs, axis=(0, 1), keepdims=True)
                    # The output variance should be close to 1.
                    output_var = jnp.mean((outputs - output_mean) ** 2, axis=(0, 1))
                else:
                    output_mean, output_var = _compute_moments_with_paddings(
                        x=outputs, paddings=paddings, reduction_axis=[0, 1]
                    )

                assert_allclose(output_mean, np.zeros_like(output_mean))
                expected_var = np.ones_like(output_var)
                if paddings is not None:
                    # var is 0 if there is no valid frame in the batch.
                    expected_var *= jnp.sum(1 - paddings) > 0
                assert_allclose(output_var, expected_var)
                # Check parameter updates.
                self.assertCountEqual(["moving_mean", "moving_variance"], param_updates.keys())
                self.assertEqual((dim,), param_updates["moving_mean"].shape)
                self.assertEqual((dim,), param_updates["moving_variance"].shape)
                if paddings is None or jnp.sum(1 - paddings) > 0:
                    self.assertNotAlmostEqual(
                        jnp.abs(param_updates["moving_mean"] - layer_params["moving_mean"]).max(),
                        0,
                    )
                    self.assertNotAlmostEqual(
                        jnp.abs(
                            param_updates["moving_variance"] - layer_params["moving_variance"]
                        ).max(),
                        0,
                    )
                else:
                    # No valid frame in the batch.
                    assert_allclose(
                        jnp.abs(
                            param_updates["moving_mean"] - layer_params["moving_mean"] * cfg.decay
                        ).max(),
                        0,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                    assert_allclose(
                        jnp.abs(
                            param_updates["moving_variance"]
                            - layer_params["moving_variance"] * cfg.decay
                        ).max(),
                        0,
                        atol=1e-6,
                        rtol=1e-6,
                    )
            else:
                assert_allclose(outputs, inputs)
                self.assertCountEqual([], param_updates.keys())

    def test_linear(self):
        input_dim, output_dim = 4, 6
        cfg = Linear.default_config().set(name="test", input_dim=input_dim, output_dim=output_dim)
        layer: Linear = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(input_dim, output_dim), bias=(output_dim,)),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, input_dim])
        inputs = orig_inputs.copy()

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        ref = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
        # torch.nn.Linear.weight is of shape (output_dim, input_dim).
        _copy(layer_params["weight"].transpose(), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(torch.as_tensor(np.asarray(inputs)))
        assert_allclose(outputs, ref_outputs.detach().numpy())

    def test_unit_norm_linear(self):
        layer = (
            UnitNormLinear.default_config()
            .set(name="test", input_dim=3, output_dim=2)
            .instantiate(parent=None)
        )
        prng_key = jax.random.PRNGKey(123)
        layer_params = layer.initialize_parameters_recursively(prng_key)
        # Input identity matrix to get weights values.
        outputs, _ = F(
            layer,
            inputs=(jnp.eye(3),),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        self.assertAllClose(jnp.linalg.norm(outputs, axis=0), jnp.array([1.0, 1.0]))

    @parameterized.named_parameters(
        {
            "testcase_name": "2x2",
            "window": (2, 2),
            "strides": (1, 1),
        },
        {"testcase_name": "2x2_S2", "window": (2, 2), "strides": (2, 2)},
        {
            "testcase_name": "3x3",
            "window": (3, 3),
            "strides": (1, 1),
        },
        {
            "testcase_name": "3x3_S2",
            "window": (3, 3),
            "strides": (2, 2),
        },
    )
    def test_maxpool2d(
        self,
        window: Tuple[int, int],
        strides: Tuple[int, int],
    ):
        input_dim = 4
        cfg = MaxPool2D.default_config().set(
            name="test",
            window=window,
            strides=strides,
        )
        layer: MaxPool2D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 10, 7, input_dim])

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        ref = torch.nn.MaxPool2d(
            kernel_size=window,
            stride=strides,
            padding=0,
        )
        ref_outputs = ref(as_torch_tensor(inputs.transpose(0, 3, 1, 2)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 1))
        # Tests output_shape.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertAllEqual(outputs.shape, output_shape)

    # Fails if tolerance is made smaller.
    @parameterized.named_parameters(
        {
            "testcase_name": "1x1",
            "window": (1, 1),
            "strides": (1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2_VALID",
            "window": (2, 2),
            "strides": (1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2_SAME",
            "window": (2, 2),
            "strides": (1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2_S2_VALID",
            "window": (2, 2),
            "strides": (2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_VALID",
            "window": (3, 3),
            "strides": (1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_SAME",
            "window": (3, 3),
            "strides": (1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_S2_VALID",
            "window": (3, 3),
            "strides": (2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_S2_PADDING1",
            "window": (3, 3),
            "strides": (2, 2),
            "padding": (1, 1),
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3_GROUPS4",
            "window": (3, 3),
            "strides": (1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 4,
        },
    )
    def test_conv2d(
        self,
        window: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Union[str, Tuple[int, int]],
        num_input_dim_groups: int,
    ):
        input_dim, output_dim = 256, 128
        if isinstance(padding, tuple):
            conv_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            conv_padding = padding
        cfg = Conv2D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=conv_padding,
            num_input_dim_groups=num_input_dim_groups,
        )
        layer: Conv2D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(
                weight=(window[0], window[1], input_dim // num_input_dim_groups, output_dim),
                bias=(output_dim,),
            ),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 10, 7, input_dim])

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        ref_padding = padding.lower() if isinstance(padding, str) else padding
        ref = torch.nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
            groups=num_input_dim_groups,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(3, 2, 0, 1), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(inputs.transpose(0, 3, 1, 2)))
        # We currently don't match PyTorch as closely as we would like.
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 1), atol=4e-6)
        # Tests output_shape.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertAllEqual(outputs.shape, output_shape)

    @parameterized.parameters(
        ([0, 0, 0, 1], [0, 0, 0, 1], 1, "SAME"),
        ([0, 0, 0, 0], [0, 0], 1, "VALID"),
        ([0, 0, 0, 1], [0, 1], 1, "VALID"),
        ([0, 0, 0, 0], [0], 2, "VALID"),
        ([0, 0, 0, 1], [0], 2, "VALID"),
        ([0, 0, 1, 1], [1], 2, "VALID"),
        ([0, 0, 0, 0, 0], [0, 0], 2, "VALID"),
        ([0, 0, 0, 0, 1], [0, 1], 2, "VALID"),
        ([0, 0, 0, 1, 1], [0, 1], 2, "VALID"),
        ([0, 0, 1, 1, 1], [1, 1], 2, "VALID"),
        ([0, 0, 0, 0, 0, 0], [0, 0], 2, "VALID"),
        ([0, 0, 0, 0, 0, 1], [0, 0], 2, "VALID"),
        ([0, 0, 0, 0, 1, 1], [0, 1], 2, "VALID"),
        ([0, 0, 0, 1, 1, 1], [0, 1], 2, "VALID"),
        ([0, 0, 1, 1, 1, 1], [1, 1], 2, "VALID"),
    )
    def test_conv_padding_lingvo(
        self, input_paddings, expected_paddings, stride: int, padding_cfg: str
    ):
        """Tests _compute_conv_output_1d_padding() with SAME and VALID padding cfg."""
        # This test is from lingvo
        # https://github.com/tensorflow/lingvo/blob/master/lingvo/core/conv_layers_with_time_padding_test.py#L157.
        window = 3
        out_paddings = _compute_conv_output_1d_padding(
            jnp.array([input_paddings]), window=window, stride=stride, conv_padding_cfg=padding_cfg
        )
        assert_allclose(out_paddings[0], expected_paddings)

    @parameterized.parameters(
        (3, 2, (1, 1)),
        (5, 2, (1, 1)),
        (5, 2, (2, 2)),
    )
    def test_conv_output_1d_padding(self, window: int, stride: int, padding_cfg: Tuple[int, int]):
        """Tests _compute_conv_output_1d_padding() with explicit padding cfg."""
        batch_size = 5
        seq_len = 5
        paddings = jnp.triu(jnp.ones((batch_size, seq_len)), k=1)
        out_paddings = _compute_conv_output_1d_padding(
            paddings, window=window, stride=stride, conv_padding_cfg=padding_cfg
        )

        # The reference is computed using 1d convolution.
        # We first pad the padding sequence with 0 on the left, and 1 on the right.
        paddings = jnp.pad(paddings, ((0, 0), (padding_cfg[0], 0)), constant_values=0)
        paddings = jnp.pad(paddings, ((0, 0), (0, padding_cfg[1])), constant_values=1)
        conv_paddings = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(paddings, -1),  # [batch, seq, 1]
            rhs=jnp.ones([window, 1, 1], paddings.dtype),
            window_strides=(stride,),
            dimension_numbers=("NWC", "WIO", "NWC"),
            padding="VALID",
        )
        ref_paddings = (jnp.squeeze(conv_paddings, axis=-1) > 0).astype(paddings.dtype)
        self.assertAllEqual(out_paddings, ref_paddings)

    @parameterized.named_parameters(
        ("1x1", (1, 1), (1, 1), "VALID"),
        ("2x2_VALID", (2, 2), (1, 1), "VALID"),
        ("2x2_SAME", (2, 2), (1, 1), "SAME"),
        ("2x2_S2_VALID", (2, 2), (2, 2), "VALID"),
        ("3x3_VALID", (3, 3), (1, 1), "VALID"),
        ("3x3_SAME", (3, 3), (1, 1), "SAME"),
        ("3x3_S2_VALID", (3, 3), (2, 2), "VALID"),
        ("3x3_S2_PADDING1", (3, 3), (2, 2), (1, 1)),
    )
    def test_conv2d_with_1d_padding(
        self,
        window: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Union[str, Tuple[int, int]],
    ):
        """Tests that Conv2DWith1DPadding has consistent outputs under different padding lengths.

        Generates a batch of input sequences. Pads the sequences under different lengths.
        Checks that the outputs are the same.
        """
        input_dim, input_channel, output_dim = 4, 7, 6
        if isinstance(padding, tuple):
            conv_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            conv_padding = padding
        cfg = Conv2DWith1DPadding.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=conv_padding,
        )
        layer: Conv2DWith1DPadding = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window[0], window[1], input_dim, output_dim), bias=(output_dim,)),
            shapes(layer_params),
        )
        # Generate a batch of 10 input sequences.
        batch_size, max_seq_len = 10, 10

        prng_key, input_key = jax.random.split(prng_key)
        inputs = (
            jax.random.normal(input_key, [batch_size, max_seq_len, input_channel, input_dim]) * 100
        )

        # The 10 sequences have length 1 to 10.
        paddings = jnp.triu(jnp.ones((batch_size, max_seq_len)), k=1)

        # Compute layer outputs.
        (ref_outputs, ref_paddings), _ = F(
            layer,
            inputs=dict(x=inputs, paddings=paddings),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        random_keys = jax.random.split(input_key, num=2 * max_seq_len)
        for seq_len in range(1, max_seq_len):
            # We create a new batch. The time axis of the new batch is of length seq_len.
            permute_idx = jax.random.permutation(random_keys[2 * (seq_len - 1)], seq_len)
            inputs_batch = jnp.take_along_axis(inputs, permute_idx[:, None, None, None], axis=0)[
                :, :seq_len
            ]
            paddings_batch = jnp.take_along_axis(paddings, permute_idx[:, None], axis=0)[
                :, :seq_len
            ]

            # Generate random data at padding positions.
            random_data = (
                jax.random.normal(
                    random_keys[2 * seq_len - 1],
                    [len(permute_idx), seq_len, input_channel, input_dim],
                )
                * 1000
            )
            inputs_new_batch = jnp.where(
                paddings_batch[:, :, None, None], random_data, inputs_batch
            )

            (outputs_batch, output_paddings_batch), _ = F(
                layer,
                inputs=dict(x=inputs_new_batch, paddings=paddings_batch),
                is_training=True,
                state=layer_params,
                prng_key=prng_key,
            )
            output_len = output_paddings_batch.shape[1]
            if output_len > 0:
                assert_allclose(
                    outputs_batch,
                    jnp.take_along_axis(ref_outputs, permute_idx[:, None, None, None], axis=0)[
                        :, :output_len
                    ],
                )
                self.assertAllEqual(
                    output_paddings_batch,
                    jnp.take_along_axis(ref_paddings, permute_idx[:, None], axis=0)[:, :output_len],
                )

    @parameterized.named_parameters(
        {
            "testcase_name": "2x2",
            "window": (2, 2),
            "strides": (1, 1),
            "padding": "VALID",
        },
        {
            "testcase_name": "2x2_S2",
            "window": (2, 2),
            "strides": (2, 2),
            "padding": "VALID",
        },
        {
            "testcase_name": "3x3_S2",
            "window": (3, 3),
            "strides": (2, 2),
            "padding": "VALID",
        },
    )
    def test_deconv2d(
        self,
        window: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Union[str, Tuple[int, int]],
    ):
        input_dim, output_dim = 4, 8
        if isinstance(padding, tuple):
            deconv_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            deconv_padding = padding
        cfg = Conv2DTranspose.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=deconv_padding,
        )
        layer: Conv2DTranspose = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(
                weight=(window[0], window[1], output_dim, input_dim),
                bias=(output_dim,),
            ),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 10, 7, input_dim])
        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        if isinstance(padding, Tuple):
            ref_padding = padding[0]
        elif isinstance(padding, str):
            ref_padding = padding.lower()
            if ref_padding == "valid":
                ref_padding = 0
        else:
            ref_padding = 0

        ref = torch.nn.ConvTranspose2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(3, 2, 0, 1), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(inputs.transpose(0, 3, 1, 2)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 1))
        # Tests output_shape.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertAllEqual(outputs.shape, output_shape)

    @parameterized.named_parameters(
        {
            "testcase_name": "1x1x1",
            "window": (1, 1, 1),
            "strides": (1, 1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2x2_VALID",
            "window": (2, 2, 2),
            "strides": (1, 1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2x2_SAME",
            "window": (2, 2, 2),
            "strides": (1, 1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "2x2x2_S2_VALID",
            "window": (2, 2, 2),
            "strides": (2, 2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_VALID",
            "window": (3, 3, 3),
            "strides": (1, 1, 1),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_SAME",
            "window": (3, 3, 3),
            "strides": (1, 1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_S2_VALID",
            "window": (3, 3, 3),
            "strides": (2, 2, 2),
            "padding": "VALID",
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_S2_PADDING1",
            "window": (3, 3, 3),
            "strides": (2, 2, 2),
            "padding": (1, 1, 1),
            "num_input_dim_groups": 1,
        },
        {
            "testcase_name": "3x3x3_GROUPS4",
            "window": (3, 3, 3),
            "strides": (1, 1, 1),
            "padding": "SAME",
            "num_input_dim_groups": 4,
        },
    )
    def test_conv3d(
        self,
        window: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Union[str, Tuple[int, int]],
        num_input_dim_groups: int,
    ):
        input_dim, output_dim = 4, 8
        if isinstance(padding, tuple):
            conv_padding = (
                (padding[0], padding[0]),
                (padding[1], padding[1]),
                (padding[2], padding[2]),
            )
        else:
            conv_padding = padding
        cfg = Conv3D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=conv_padding,
            num_input_dim_groups=num_input_dim_groups,
        )
        layer: Conv3D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        expected = dict(
            weight=(window[0], window[1], window[2], input_dim // num_input_dim_groups, output_dim),
            bias=(output_dim,),
        )
        self.assertEqual(
            expected,
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)

        batch_size = 2
        inputs = jax.random.normal(input_key, [batch_size, 10, 7, 4, input_dim])

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        ref_padding = padding.lower() if isinstance(padding, str) else padding
        ref = torch.nn.Conv3d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
            groups=num_input_dim_groups,
        )

        # weight.shape: (H, W, D, I, O)
        # ref.weight.shape: (O, I, H, W, D)
        _copy(layer_params["weight"].transpose(4, 3, 0, 1, 2), ref.weight)
        _copy(layer_params["bias"], ref.bias)

        ref_outputs = ref(as_torch_tensor(inputs.transpose(0, 4, 1, 2, 3)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 4, 1))

        # Tests output_shape.
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertAllEqual(outputs.shape, output_shape)

    @parameterized.named_parameters(
        ("w3s1_VALID", 3, 1, "VALID"),
        ("w3s1_SAME", 3, 1, "SAME"),
        ("w4s1_SAME", 4, 1, "SAME"),
        ("w4s1_CAUSAL", 4, 1, (3, 0)),
    )
    def test_conv1d(
        self,
        window: int,
        strides: int,
        padding: Union[str, Tuple[int, int]],
    ):
        input_dim, output_dim = 4, 6
        cfg = Conv1D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=padding,
        )
        layer: Conv1D = cfg.instantiate(parent=None)
        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, input_dim, output_dim), bias=(output_dim,)),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 7, input_dim])
        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        if isinstance(padding, str):
            ref_padding = padding.lower()
            ref_inputs = inputs
        else:
            # torch.nn.Conv1d does not support asymmetric padding, so pad manually and use "valid".
            ref_padding = "valid"
            ref_inputs = jnp.pad(inputs, ((0, 0), padding, (0, 0)))
        ref = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            groups=1,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(2, 1, 0), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(ref_inputs.transpose(0, 2, 1)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 1))

    @parameterized.named_parameters(
        ("w3s1_VALID", 3, 1, "VALID"),
        ("w3s1_SAME", 3, 1, "SAME"),
        ("w4s1_SAME", 4, 1, "SAME"),
        ("w4s1_CAUSAL", 4, 1, (3, 0)),
    )
    def test_depthwise_conv1d(
        self,
        window: int,
        strides: int,
        padding: Union[str, Tuple[int, int]],
    ):
        input_dim = 4
        cfg = DepthwiseConv1D.default_config().set(
            name="test",
            input_dim=input_dim,
            window=window,
            strides=strides,
            padding=padding,
        )
        layer: DepthwiseConv1D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(window, 1, input_dim), bias=(input_dim,)),
            shapes(layer_params),
        )
        bias = layer_params["bias"]
        assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 7, input_dim])

        # Compute layer outputs.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        # Compute ref outputs.
        if isinstance(padding, str):
            ref_padding = padding.lower()
            ref_inputs = inputs
        else:
            # torch.nn.Conv1d does not support asymmetric padding, so pad manually and use "valid".
            ref_padding = "valid"
            ref_inputs = jnp.pad(inputs, ((0, 0), padding, (0, 0)))
        ref = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            groups=input_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(2, 1, 0), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(as_torch_tensor(ref_inputs.transpose(0, 2, 1)))
        assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 1))

    @parameterized.parameters(
        itertools.product(
            (None, 0.0, 0.2, 1.0, -0.1),
            ("row", "batch", "test"),
            (True, False),
        )
    )
    def test_stochastic_depth(self, rate, mode, is_training):
        # Initialize layer.
        cfg = StochasticDepth.default_config().set(name="test", rate=rate, mode=mode)
        layer: StochasticDepth = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        prng_key, input_key = jax.random.split(prng_key)
        # Random inputs.
        inputs = jax.random.normal(input_key, [2, 32, 32, 16])

        test_fn = partial(
            F,
            layer,
            inputs=(inputs,),
            is_training=is_training,
            state=layer_params,
            prng_key=prng_key,
        )

        if not is_training or rate is None or rate == 0:
            outputs, _ = test_fn()
            assert_allclose(inputs, outputs)
        elif rate < 0.0 or rate >= 1.0 or mode not in ["row", "batch"]:
            with self.assertRaises(ValueError):
                test_fn()
        else:
            test_fn()

    @parameterized.parameters(itertools.product((1.0, -0.1, 0.2), (-1, 6, 2)))
    def test_get_stochastic_depth_linear_rate(self, peak_rate, stage_order):
        num_stages = 5
        if peak_rate < 0 or peak_rate >= 1 or stage_order < 0 or stage_order > num_stages:
            with self.assertRaises(ValueError):
                get_stochastic_depth_linear_rate(peak_rate, stage_order, num_stages)
        else:
            rate = get_stochastic_depth_linear_rate(peak_rate, stage_order, num_stages)
            self.assertEqual(rate, peak_rate * stage_order / num_stages)

    @parameterized.product(
        se_ratio=(0, 0.25),
        activation=("nn.relu", "nn.sigmoid"),
        gating=("nn.sigmoid", "nn.hard_sigmoid"),
        num_reduced_filters=(None, 32),
    )
    def test_squeeze_excitation(self, se_ratio, activation, gating, num_reduced_filters):
        # Initialize layer.
        input_dim = 64
        cfg = SqueezeExcitation.default_config().set(
            name="test",
            input_dim=input_dim,
            se_ratio=se_ratio,
            activation=activation,
            gating=gating,
            num_reduced_filters=num_reduced_filters,
        )
        layer: SqueezeExcitation = cfg.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        prng_key, input_key = jax.random.split(prng_key)
        # Random inputs.
        inputs = jax.random.normal(input_key, [2, 32, 32, input_dim])
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        if se_ratio == 0:
            assert (inputs == outputs).all()
        else:
            self.assertEqual(inputs.shape, outputs.shape)
            # Check number of parameters.
            if num_reduced_filters is None:
                num_reduced_filters = max(1, int(input_dim * se_ratio))
            num_params_se = input_dim * (num_reduced_filters + 1) + num_reduced_filters * (
                input_dim + 1
            )
            init_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(1))
            self.assertEqual(num_params_se, utils.count_model_params(init_params))

    @parameterized.product(
        label_smoothing=(0.0, 0.1),
        use_soft_labels=(True, False),
    )
    def test_classification_metric(self, label_smoothing, use_soft_labels):
        batch_size = 2
        num_classes = 1000
        logits = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        labels = np.random.randint(0, num_classes - 1, [batch_size]).astype(np.int32)

        cfg = ClassificationMetric.default_config().set(
            name="test", num_classes=1000, label_smoothing=label_smoothing
        )
        layer: ClassificationMetric = cfg.instantiate(parent=None)
        if use_soft_labels:
            soft_labels = jax.nn.one_hot(labels, num_classes, dtype=logits.dtype)
        else:
            soft_labels = None

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        if use_soft_labels and label_smoothing:
            with self.assertRaises(ValueError):
                F(
                    layer,
                    inputs=dict(logits=logits, labels=labels, soft_labels=soft_labels),
                    is_training=True,
                    state=layer_params,
                    prng_key=prng_key,
                )
        else:
            loss, _ = F(
                layer,
                inputs=dict(logits=logits, labels=labels, soft_labels=soft_labels),
                is_training=True,
                state=layer_params,
                prng_key=prng_key,
            )
            self.assertTrue(loss is not None)

    @parameterized.product(use_soft_labels=(False, False, False, True), num_labels=(50, 25, 1, 1))
    def test_binary_classification_metric(self, use_soft_labels: bool, num_labels: int):
        prediction_threshold = 0.5
        batch_size = 200
        average = "micro"
        if num_labels == 1:
            average = "binary"
        logits = np.random.uniform(-1, 1, [batch_size, num_labels]).astype(np.float32)
        labels = np.random.randint(0, 2, [batch_size, num_labels]).astype(np.int32)

        # Enforce last label to be masked.
        test_labels = np.copy(labels)
        test_labels[-1, :] = -1

        cfg = BinaryClassificationMetric.default_config().set(
            name="test",
            prediction_threshold=prediction_threshold,
        )
        layer: BinaryClassificationMetric = cfg.instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        if use_soft_labels:
            with self.assertRaisesRegex(
                ValueError, "soft_labels for binary cross entropy must be None"
            ):
                F(
                    layer,
                    inputs=dict(logits=logits, labels=test_labels, soft_labels=test_labels),
                    is_training=True,
                    state=layer_params,
                    prng_key=prng_key,
                )
        else:
            loss, output_collection = F(
                layer,
                inputs=dict(logits=logits, labels=test_labels),
                is_training=True,
                state=layer_params,
                prng_key=prng_key,
            )
            self.assertIsNotNone(loss)
            summaries = output_collection.summaries
            preds = jnp.where(jnp.greater(nn.sigmoid(logits), prediction_threshold), 1, 0)
            self.assertEqual(
                sklearn_recall_score(labels[:-1], preds[:-1], average=average),
                summaries["recall"].mean,
            )
            self.assertEqual(
                sklearn_precision_score(labels[:-1], preds[:-1], average=average),
                summaries["precision"].mean,
            )

    @parameterized.product(use_soft_labels=(False, True, True, True), num_classes=(50, 2, 1000, 5))
    def test_hingeloss_metric(self, use_soft_labels: bool, num_classes: int):
        batch_size = 2
        logits = np.random.uniform(0, 1, [batch_size, num_classes]).astype(np.float32)
        labels = np.random.randint(0, num_classes, [batch_size]).astype(np.int32)
        soft_labels = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)

        cfg = CategoricalHingeLossMetric.default_config().set(name="test", num_classes=num_classes)
        layer: CategoricalHingeLossMetric = cfg.instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        if use_soft_labels:
            with self.assertRaisesRegex(
                ValueError, "soft_labels for category hinge loss must be None"
            ):
                F(
                    layer,
                    inputs=dict(logits=logits, labels=labels, soft_labels=soft_labels),
                    is_training=True,
                    state=layer_params,
                    prng_key=prng_key,
                )
        else:
            loss, _ = F(
                layer,
                inputs=dict(logits=logits, labels=labels, soft_labels=None),
                is_training=True,
                state=layer_params,
                prng_key=prng_key,
            )
            self.assertTrue(loss is not None)

    @parameterized.parameters(1.0, -1.0)
    def test_variational_noise(self, vn_std):
        param_noise_cfg = VariationalNoise.default_config().set(vn_std=vn_std)
        test_layer: SqueezeExcitation = (
            SqueezeExcitation.default_config()
            .set(name="test", param_noise=param_noise_cfg, input_dim=8, se_ratio=0.1)
            .instantiate(parent=None)
        )
        params = test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        noisy_params = test_layer.apply_parameter_noise_recursively(
            prng_key=jax.random.PRNGKey(456), params=params
        )
        if param_noise_cfg.vn_std <= 0.0:
            self.assertNestedAllClose(params, noisy_params)
        else:
            for (orig_path, orig_value), (noisy_path, noisy_value) in zip(
                flatten_items(params), flatten_items(noisy_params)
            ):
                self.assertEqual(orig_path, noisy_path)
                self.assertNotAllClose(orig_value, noisy_value)

    @parameterized.product(drop_rate=(0, 0.5), num_cls_tokens=(0, 6))
    def test_drop_tokens(self, drop_rate, num_cls_tokens):
        batch_size, len_tokens, dim = 32, 50, 32
        cfg = DropToken.default_config().set(
            name="test", rate=drop_rate, num_cls_tokens=num_cls_tokens
        )
        layer: DropToken = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        inputs = jax.random.normal(init_key, [batch_size, len_tokens, dim])

        # Compute layer outputs during training.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )

        assert_allclose(
            outputs.shape,
            [
                batch_size,
                num_cls_tokens + int((len_tokens - num_cls_tokens) * (1 - drop_rate)),
                dim,
            ],
        )

        # Compute layer outputs during evaluation.
        outputs, _ = F(
            layer,
            inputs=(inputs,),
            is_training=False,
            state=layer_params,
            prng_key=prng_key,
        )

        assert_allclose(outputs.shape, [batch_size, len_tokens, dim])

    @parameterized.parameters(
        (
            2,
            (0, 0),
            [[[1, 1, 2, 2], [3, 3, 4, 4]], [[7, 7, 8, 8], [0, 0, 0, 0]]],
            [[0, 0], [0, 1]],
        ),
        (
            3,
            (0, 0),
            [[[1, 1, 2, 2, 3, 3]], [[0, 0, 0, 0, 0, 0]]],
            [[0], [1]],
        ),
        (
            3,
            (2, 0),
            [[[0, 0, 0, 0, 1, 1], [2, 2, 3, 3, 4, 4]], [[0, 0, 0, 0, 7, 7], [0, 0, 0, 0, 0, 0]]],
            [[0, 0], [0, 1]],
        ),
    )
    def test_stack_over_time(self, stride, pad, expected_outputs, expected_output_paddings):
        # Input shape [2, 5, 2].
        inputs = jnp.array(
            [[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], [[7, 7], [8, 8], [0, 0], [0, 0], [0, 0]]],
            dtype=jnp.float32,
        )
        paddings = jnp.array([[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]])
        layer: StackOverTime = (
            StackOverTime.default_config()
            .set(
                name="test",
                stride=stride,
                padding=pad,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        (outputs, output_paddings), _ = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=False,
            state=layer_params,
            prng_key=jax.random.PRNGKey(5),
        )
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertAllEqual(outputs.shape, output_shape)
        self.assertAllClose(jnp.array(expected_outputs, dtype=jnp.float32), outputs)
        self.assertAllClose(jnp.array(expected_output_paddings, dtype=jnp.int32), output_paddings)

    def test_stack_over_time_data_change(self):
        """Tests that the stacked outputs is masked with the output paddings."""
        np.random.seed(500)
        inputs = np.random.normal(size=[2, 21, 16])
        paddings = np.ones([2, 21], dtype=np.float32)
        paddings[0, :9] = 0
        paddings[1, :14] = 0
        inputs = inputs * (1 - paddings)[:, :, None]

        layer: StackOverTime = (
            StackOverTime.default_config()
            .set(
                name="test",
                stride=2,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        (outputs, output_paddings), _ = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=False,
            state=layer_params,
            prng_key=jax.random.PRNGKey(5),
        )
        output_shape = layer.output_shape(input_shape=inputs.shape)
        self.assertAllEqual(outputs.shape, output_shape)
        self.assertAllEqual(np.array([4, 7], dtype=np.float32), np.sum(1 - output_paddings, axis=1))
        self.assertAllClose(
            np.sum(inputs**2, (1, 2)),
            np.sum(outputs**2, (1, 2)) + np.array([np.sum(inputs[0][8] ** 2), 0.0]),
        )

    @parameterized.product(stride=(2, 3, 4), pad=((0, 0), (1, 1), (2, 0)))
    def test_stack_consistent_outputs(self, stride, pad):
        """Tests that StackOverTime has consistent outputs under different padding lengths."""
        batch_size, input_dim = 2, 1
        input_length = 7
        expected_output_length = (input_length + pad[0]) // stride
        layer: StackOverTime = (
            StackOverTime.default_config()
            .set(
                name="test",
                stride=stride,
                padding=pad,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        for ll in range(4, 11):
            # Batch with another example of length ll.
            length = max(input_length, ll)
            inputs = jnp.ones([batch_size, length, input_dim])
            paddings = jnp.arange(length)[None, :] >= jnp.array([7, ll])[:, None]
            (outputs, output_paddings), _ = F(
                layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                is_training=False,
                state=layer_params,
                prng_key=jax.random.PRNGKey(5),
            )
            output_shape = layer.output_shape(input_shape=inputs.shape)
            self.assertAllEqual(outputs.shape, output_shape)
            self.assertEqual(expected_output_length, np.sum(1 - output_paddings, axis=1)[0])

    @parameterized.parameters(((0, 1), (0, 0)), ((1, 1), (3, 0)), ((1, 1), (0, 3)))
    def test_stack_vs_conv2d_output_len_match(self, conv_padding, stack_padding):
        # Note that to get the same output length, we need to pad the sequence differently
        # for convolution and stacking layer.
        for audio_seq_len in [16000, 16160, 16320, 16480, 16640, 16800, 16960, 17120]:
            sampling_rate, window_size_ms, window_step_ms = 16000, 25, 10
            window_size = window_size_ms * sampling_rate // 1000
            window_step = window_step_ms * sampling_rate // 1000
            seq_len = max(audio_seq_len - window_size, 0) // window_step + 1
            conv_layer: Conv2DWith1DPadding = (
                Conv2DWith1DPadding.default_config()
                .set(
                    name="test_conv",
                    input_dim=3,
                    output_dim=3,
                    window=(3, 3),
                    strides=(2, 2),
                    padding=(conv_padding, (0, 1)),
                )
                .instantiate(parent=None)
            )
            stack_layer: StackOverTime = (
                StackOverTime.default_config()
                .set(name="test_stack", stride=4, padding=stack_padding)
                .instantiate(parent=None)
            )
            # Computes downsampler output shape.
            down_sample_shape1 = conv_layer.output_shape(input_shape=[None, seq_len, 80, 3])
            down_sample_shape2 = conv_layer.output_shape(input_shape=down_sample_shape1)

            # Computes stack output shape.
            stack_shape = stack_layer.output_shape(input_shape=[None, seq_len, 80])
            # Tests that the sequence length dimension matches.
            self.assertEqual(down_sample_shape2[1], stack_shape[1])

    def test_multilinear_fan_axes(self):
        input_dim, num_outputs, output_dim = 3, 4, 6
        layer: MultiLinear = (
            MultiLinear.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                num_outputs=4,
                output_dim=output_dim,
            )
            .instantiate(parent=None)
        )
        # pylint: disable-next=protected-access
        param_spec_map = layer._create_layer_parameter_specs()
        self.assertSequenceEqual(
            [input_dim, num_outputs, output_dim], param_spec_map["weight"].shape
        )
        # pylint: disable-next=protected-access
        fan_axes = layer._compute_fan_axes("weight", param_spec_map["weight"])
        self.assertEqual(FanAxes(in_axis=0, out_axis=(1, 2)), fan_axes)

    @parameterized.named_parameters(
        {
            "testcase_name": "simple",
            "positions": None,
        },
        {
            "testcase_name": "first_patch_positions",
            "positions": [0, 1],
        },
        {
            "testcase_name": "batched_positions",
            "positions": [[0, 1], [5, 7]],
        },
    )
    def test_separable_space_time_positional_embedding(
        self, positions: Optional[Union[List[int], List[List[int]]]]
    ):
        dim = 4
        video_size = (64, 64, 4)
        patch_size = (16, 16, 2)
        num_patches = tuple(v // p for v, p in zip(video_size, patch_size))

        height, width, time = num_patches
        if positions is None:
            positions = np.arange(height * width * time)
        else:
            positions = np.array(positions)

        pos_emb_cfg = SeparableSpaceTimePositionalEmbedding.default_config().set(
            name="test",
            dim=dim,
            spatial_embeddings=Embedding.default_config().set(num_embeddings=height * width),
            temporal_embeddings=Embedding.default_config().set(num_embeddings=time),
        )

        layer: SeparableSpaceTimePositionalEmbedding = pos_emb_cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(42))
        outputs, _ = F(
            layer,
            inputs=(positions,),
            is_training=True,
            state=layer_params,
            prng_key=jax.random.PRNGKey(123),
        )
        self.assertAllEqual([*positions.shape, dim], outputs.shape)

        context = module.InvocationContext(
            name="root",
            parent=None,
            module=layer,
            state=layer_params,
            output_collection=module.new_output_collection(),
            is_training=True,
            prng_key=jax.random.PRNGKey(124),
        )

        with module.set_current_context(context):
            with module.child_context("spatial", module=layer.spatial):
                spatial_embeddings = layer.spatial.embeddings()
            assert list(spatial_embeddings.shape) == [height * width, dim]

            with module.child_context("temporal", module=layer.temporal):
                temporal_embeddings = layer.temporal.embeddings()
            assert list(temporal_embeddings.shape) == [time, dim]

            with module.child_context("combined", module=layer):
                combined_embeddings = layer.embeddings()
            assert list(combined_embeddings.shape) == [height * width * time, dim]

        positions = positions.flatten()
        outputs = outputs.reshape(positions.size, dim)
        for pos, output in zip(positions, outputs):
            # Reminder: input has HWT layout.
            spatial_idx = math.floor(pos / temporal_embeddings.shape[0])
            temporal_idx = pos % temporal_embeddings.shape[0]
            expected_embed = spatial_embeddings[spatial_idx] + temporal_embeddings[temporal_idx]
            assert_allclose(output, expected_embed)


class EmbedTest(parameterized.TestCase):
    @staticmethod
    def build_embedder(dim, num_embeddings, rng):
        cfg = Embedding.default_config()
        cfg.dim = dim
        cfg.num_embeddings = num_embeddings
        cfg.name = "embed"
        emb = cfg.instantiate(parent=None)
        state = emb.initialize_parameters_recursively(rng)
        return (emb, state)

    @parameterized.parameters(itertools.product((5, 7), (2, 16), (10, 100), (True, False)))
    def test_embed_lookup(self, seq_len, dim, num_embeddings, is_training):
        rng = jax.random.PRNGKey(1)
        embedder, state = EmbedTest.build_embedder(dim, num_embeddings, rng)
        ixs = jax.random.randint(rng, minval=0, maxval=num_embeddings, shape=(3, seq_len))
        actual_embeds, _ = module.functional(
            embedder, rng, state=state, inputs=[ixs], is_training=is_training
        )
        np.testing.assert_array_equal(state["weight"][ixs], actual_embeds)

    @parameterized.parameters(itertools.product((5, 7), (2, 16), (10, 100), (True, False)))
    def test_embed_attend(self, seq_len, dim, num_embeddings, is_training):
        rng = jax.random.PRNGKey(1)
        embedder, state = EmbedTest.build_embedder(dim, num_embeddings, rng)
        x = jax.random.normal(rng, shape=(3, seq_len, dim))
        actual_attends = module.functional(
            embedder, rng, state=state, inputs=[x], is_training=is_training, method="attend"
        )[0]
        assert_allclose(jnp.dot(x, state["weight"].T), actual_attends)


class BiasLayer(BaseLayer):
    """A test layer with bias."""

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        return dict(bias=ParameterSpec(shape=[], mesh_axes=None))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.state["bias"]

    def forward_twice(self, x: Tensor) -> Tensor:
        with child_context("forward_1", module=self):
            x = self.forward(x)
        with child_context("forward_2", module=self):
            x = self.forward(x)
        return x


class ParentLayer(BaseLayer):
    """A test parent layer."""

    @config_class
    class Config(BaseLayer.Config):
        children: Dict[str, BaseLayer.Config] = {}
        shared_modules: Sequence[str] = []

    def __init__(self, cfg: BaseLayer.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        for name, child_cfg in cfg.children.items():
            child = self._add_child(name, child_cfg)
            if name in cfg.shared_modules:
                self._share_with_descendants(child, shared_module_name=name)

    def forward(self, path: Sequence[str], child_method="forward", **kwargs) -> Tensor:
        child_name = path[0]
        child = self.children[child_name]
        if len(path) == 1:
            return getattr(child, child_method)(**kwargs)
        else:
            return child(**kwargs, path=path[1:], child_method=child_method)


class RedirectToSharedModuleTest(TestCase):
    def test_deeply_nested(self):
        with jax.checking_leaks():
            cfg = ParentLayer.default_config().set(
                shared_modules=["shared_bias"],
                children=dict(
                    shared_bias=BiasLayer.default_config().set(
                        param_init=ConstantInitializer.default_config().set(value=1)
                    ),
                    parent_a=ParentLayer.default_config().set(
                        children=dict(
                            redirect_1=RedirectToSharedModule.default_config().set(
                                shared_module="shared_bias",
                            ),
                            redirect_2=RedirectToSharedModule.default_config().set(
                                shared_module="shared_bias",
                                method_map=dict(forward="forward_twice"),
                            ),
                        ),
                    ),
                ),
            )
            layer: ParentLayer = cfg.set(name="test").instantiate(parent=None)
            inputs = jnp.ones([]) * 100
            state = layer.initialize_parameters_recursively(jax.random.PRNGKey(123))
            self.assertNestedAllClose(
                state,
                dict(shared_bias=dict(bias=jnp.ones([]))),
            )

            @partial(jax.jit, static_argnames=("path", "child_method"))
            def jit_forward(inputs: Tensor, *, path: Tuple[str], child_method: str = "forward"):
                outputs, _ = F(
                    layer,
                    state=state,
                    prng_key=jax.random.PRNGKey(0),
                    is_training=True,
                    inputs=dict(x=inputs, path=path, child_method=child_method),
                )
                return outputs

            # Call bias_1.forward.
            outputs = jit_forward(inputs, path=("shared_bias",))
            self.assertNestedAllClose(outputs, jnp.ones([]) * 101)
            # Call parent_a.redirect_1.forward.
            outputs = jit_forward(inputs, path=("parent_a", "redirect_1"))
            self.assertNestedAllClose(outputs, jnp.ones([]) * 101)
            # Call shared_bias.forward_twice.
            outputs = jit_forward(inputs, path=("shared_bias",), child_method="forward_twice")
            self.assertNestedAllClose(outputs, jnp.ones([]) * 102)
            # Call parent_a.redirect_2.forward -> shared_bias.forward_twice
            outputs = jit_forward(inputs, path=("parent_a", "redirect_2"), child_method="forward")
            self.assertNestedAllClose(outputs, jnp.ones([]) * 102)


class SetConfigTest(TestCase):
    def test_set_bias(self):
        cfg = Decoder.default_config()
        self.assertTrue(cfg.transformer.layer.self_attention.attention.input_linear.layer.bias)
        self.assertTrue(cfg.transformer.layer.self_attention.attention.output_linear.bias)
        self.assertTrue(cfg.transformer.layer.feed_forward.linear1.bias)
        self.assertTrue(cfg.transformer.layer.feed_forward.linear2.bias)
        set_bias_recursively(cfg, bias=False)
        self.assertFalse(cfg.transformer.layer.self_attention.attention.input_linear.layer.bias)
        self.assertFalse(cfg.transformer.layer.self_attention.attention.output_linear.bias)
        self.assertFalse(cfg.transformer.layer.feed_forward.linear1.bias)
        self.assertFalse(cfg.transformer.layer.feed_forward.linear2.bias)

    def test_set_norm(self):
        cfg = Decoder.default_config()
        self.assertIsInstance(cfg.transformer.layer.self_attention.norm, LayerNorm.Config)
        self.assertIsInstance(cfg.transformer.layer.feed_forward.norm, LayerNorm.Config)
        self.assertIsInstance(cfg.output_norm, LayerNorm.Config)
        set_norm_recursively(cfg, norm_cfg=RMSNorm.default_config())
        self.assertIsInstance(cfg.transformer.layer.self_attention.norm, RMSNorm.Config)
        self.assertIsInstance(cfg.transformer.layer.feed_forward.norm, RMSNorm.Config)
        self.assertIsInstance(cfg.output_norm, RMSNorm.Config)

    def test_set_dropout_rate_recursively(self):
        cfg = Decoder.default_config()
        self.assertIsNone(cfg.transformer.layer.self_attention.attention.dropout.rate)
        self.assertIsNone(cfg.transformer.layer.self_attention.dropout.rate)
        self.assertIsNone(cfg.transformer.layer.feed_forward.dropout.rate)
        self.assertIsNone(cfg.output_dropout.rate)
        set_dropout_rate_recursively(cfg, 0.1)
        self.assertEqual(0.1, cfg.transformer.layer.self_attention.attention.dropout.rate)
        self.assertEqual(0.1, cfg.transformer.layer.self_attention.dropout.rate)
        self.assertEqual(0.1, cfg.transformer.layer.feed_forward.dropout.rate)
        self.assertEqual(0.1, cfg.output_dropout.rate)

    def test_set_layer_norm_eps_recursively(self):
        cfg = Decoder.default_config()
        self.assertEqual(1e-8, cfg.transformer.layer.self_attention.norm.eps)
        self.assertEqual(1e-8, cfg.transformer.layer.feed_forward.norm.eps)
        set_layer_norm_eps_recursively(cfg, 1e-5)
        self.assertEqual(1e-5, cfg.transformer.layer.self_attention.norm.eps)
        self.assertEqual(1e-5, cfg.transformer.layer.feed_forward.norm.eps)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
