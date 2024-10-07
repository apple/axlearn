# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/praxis:
# Copyright 2022 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests quantization layers and metrics."""

# pylint: disable=no-self-use,wrong-import-position,missing-module-docstring
import os

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

# pylint: disable-next=protected-access
from jax._src import prng as prng_interal

from axlearn.common import schedule
from axlearn.common.module import functional as F
from axlearn.common.normalize import l2_normalize
from axlearn.common.param_init import ConstantInitializer, DefaultInitializer
from axlearn.common.quantizer import (
    BaseQuantizer,
    GumbelSoftmaxVectorQuantizer,
    KmeansVectorQuantizer,
    RandomVectorQuantizer,
    SimilarityMetric,
    compute_code_coverage,
    compute_code_pplx,
    quantize_by_nearest_neighbor,
)
from axlearn.common.test_utils import TestCase, assert_allclose, prng_impl
from axlearn.common.utils import Tensor, shapes

testdata_dir = os.path.join(os.path.dirname(__file__), "../experiments/testdata")

_CODE_BOOK = jnp.array(
    [
        [0.116230249, 0.0104732513, -0.409445882, -0.153374314],
        [-0.0672334433, -0.430877686, -0.280010223, 0.394074917],
        [-0.360892653, -0.153173685, -0.45321393, -0.176380157],
        [0.406187773, 0.304340839, 0.439772606, 0.368542314],
    ]
)


def _create_prngkeyarray(key_data: list[int]) -> Tensor:
    return jax.random.wrap_key_data(
        jnp.array(key_data, dtype=jnp.uint32),
        impl=prng_interal.threefry_prng_impl,
    )


class HelpersTest(TestCase):
    @parameterized.parameters(
        (1, 0.0, SimilarityMetric.DOT_PRODUCT),
        (1, -0.5, SimilarityMetric.L2_DISTANCE),
        (2, 0.0, SimilarityMetric.L2_DISTANCE),
        (2, -0.5, SimilarityMetric.DOT_PRODUCT),
        (2, 0.1, -3),
    )
    def test_quantize(self, num_groups, input_mean, metric):
        vocab_size, dim_from_all_codebooks = 4, 4
        codebook_dim = dim_from_all_codebooks // num_groups

        # [vocab_size, num_codebooks, codebook_dim].
        codebook = jnp.reshape(_CODE_BOOK, [vocab_size, num_groups, codebook_dim])
        codebook = l2_normalize(codebook, eps=1e-12, axis=-1)
        batch_size, seq_len = 2, 4
        np.random.seed(2021)
        # When codebook is normalized, normalizing inputs or not yield the same results.
        inputs = (
            np.random.rand(batch_size, seq_len, num_groups, codebook_dim).astype(np.float32)
            + input_mean
        )
        paddings = jnp.zeros([batch_size, seq_len])
        if metric not in (SimilarityMetric.L2_DISTANCE, SimilarityMetric.DOT_PRODUCT):
            with self.assertRaisesRegex(ValueError, "Expect DOT_PRODUCT metric"):
                quantize_by_nearest_neighbor(inputs=inputs, codebook=codebook, metric=metric)
        else:
            q_outputs = quantize_by_nearest_neighbor(
                inputs=inputs, codebook=codebook, metric=metric
            )
            # Compute codebook metrics.
            coverage = compute_code_coverage(onehots=q_outputs.onehots, paddings=paddings)
            pplx, entropy = compute_code_pplx(onehots=q_outputs.onehots, paddings=paddings)

            # Check shapes.
            self.assertEqual(q_outputs.ids.shape, (batch_size, seq_len, num_groups))
            self.assertEqual(q_outputs.onehots.shape, (batch_size, seq_len, num_groups, vocab_size))
            self.assertEqual(
                q_outputs.quantized_vectors.shape, (batch_size, seq_len, num_groups, codebook_dim)
            )
            # Expected outputs is generated from praxis.
            # https://github.com/google/praxis/blob/179774fb688aa8fe048307d2184c9f2b338e935f/praxis/layers/quantizer_test.py#L50-L99
            expected_outputs = {
                1: {
                    0.0: [15.861525, 24, 0.25, 1.0, 0.0],
                    -0.5: [-1.936341, 10, 1.0, 3.746748, 1.3208883],
                },
                2: {
                    0.0: [21.840881, 42, 0.375, 1.3773826, 0.32018507],
                    -0.5: [0.58745056, 24, 0.875, 3.1573687, 1.149739],
                },
            }

            assert_allclose(
                expected_outputs[num_groups][input_mean][0],
                np.sum(q_outputs.quantized_vectors),
                rtol=1e-06,
                atol=1e-06,
            )
            assert_allclose(
                expected_outputs[num_groups][input_mean][1],
                np.sum(q_outputs.ids * (1 - paddings[:, :, None])),
                rtol=1e-06,
                atol=1e-06,
            )
            assert_allclose(
                expected_outputs[num_groups][input_mean][2], coverage, rtol=1e-06, atol=1e-06
            )
            assert_allclose(
                expected_outputs[num_groups][input_mean][3], pplx, rtol=1e-06, atol=1e-06
            )
            assert_allclose(
                expected_outputs[num_groups][input_mean][4], entropy, rtol=1e-06, atol=1e-06
            )

    def test_compute_code_pplx(self):  # pylint: disable=no-self-use
        vocab_size = 11

        codes = jnp.array(
            [
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [0, 0]],
                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [0, 0]],
            ]
        )
        paddings = jnp.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])

        pplx, entropy = compute_code_pplx(
            onehots=jax.nn.one_hot(codes, num_classes=vocab_size, axis=-1), paddings=paddings
        )

        assert_allclose(pplx, 5.000, rtol=1e-06, atol=1e-06)
        assert_allclose(entropy, 1.609438, rtol=1e-06, atol=1e-06)

    def test_codebook_initializer(self):
        vocab_size, dim_from_all_codebooks, num_groups = 10, 100, 5
        codebook_dim = dim_from_all_codebooks // num_groups
        layer: BaseQuantizer = (
            BaseQuantizer.default_config()
            .set(
                name="test",
                codebook_dim=codebook_dim,
                codebook_size=vocab_size,
                num_codebooks=num_groups,
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        ".*codebook$": ConstantInitializer.default_config().set(value=0.0),
                    }
                ),
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        self.assertNestedAllClose(
            layer_params, dict(codebook=jnp.zeros((vocab_size, num_groups, codebook_dim)))
        )


class RandomVectorQuantizerTest(TestCase):
    @parameterized.parameters(
        (2, 4, 20, 16, 4, 1, True, False),
        (2, 4, 20, 16, 4, 1, False, False),
        (3, 4, 20, 32, 4, 2, True, True),
        (3, 4, 20, 32, 4, 2, False, True),
        (4, 7, 16, 256, 20, 8, True, False),
        (4, 7, 16, 256, 20, 8, False, False),
    )
    def test_forward(
        self,
        batch_size,
        seq_len,
        input_dim,
        dim_from_all_codebooks,
        vocab_size,
        num_groups,
        normalize_codebook,
        normalize_inputs,
    ):
        # The output is obtained from:
        # https://github.com/google/praxis/blob/179774fb688aa8fe048307d2184c9f2b338e935f/praxis/layers/quantizer_test.py#L109
        cfg = RandomVectorQuantizer.default_config().set(
            name="test",
            input_dim=input_dim,
            normalize_codebook=normalize_codebook,
            normalize_inputs=normalize_inputs,
            codebook_dim=dim_from_all_codebooks // num_groups,
            codebook_size=vocab_size,
            num_codebooks=num_groups,
        )
        layer: RandomVectorQuantizer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
        self.assertEqual(
            dict(
                rand_proj=dict(weight=(input_dim, dim_from_all_codebooks)),
                codebook=(vocab_size, num_groups, dim_from_all_codebooks // num_groups),
            ),
            shapes(layer_params),
        )
        with prng_impl("threefry2x32"):
            proj_key = _create_prngkeyarray([3077990774, 2166202870])
            codebook_key = _create_prngkeyarray([791337683, 1373966058])

        layer_params["rand_proj"]["weight"] = jax.random.uniform(
            key=proj_key, shape=(input_dim, dim_from_all_codebooks), minval=-1.0, maxval=1.0
        ) * np.sqrt(6 / (input_dim + dim_from_all_codebooks))
        if normalize_codebook:
            assert_allclose(
                jnp.sum(layer_params["codebook"] ** 2, axis=-1),
                jnp.ones((vocab_size, num_groups)),
                atol=1e-6,
                rtol=1e-6,
            )
        layer_params["codebook"] = jax.random.normal(
            key=codebook_key,
            shape=(vocab_size, num_groups, dim_from_all_codebooks // num_groups),
        )
        if normalize_codebook:
            layer_params["codebook"] = l2_normalize(
                layer_params["codebook"],
                axis=-1,
                eps=1e-12,
            )
        expected_values = {
            2: {
                True: dict(
                    weight=17.409603,
                    codebook=num_groups * vocab_size,
                    q_vecs=0.11563152,
                    ids=3,
                    onehots=batch_size * seq_len * num_groups,
                    coverage=0.5,
                    pplx=1.4575692,
                    entropy=0.37677014,
                ),
                False: dict(
                    weight=17.409603,
                    codebook=67.37621,
                    q_vecs=1.3286973,
                    ids=14,
                    onehots=batch_size * seq_len * num_groups,
                    coverage=0.75,
                    pplx=2.6493511,
                    entropy=0.97431475,
                ),
            },
            3: {
                True: dict(
                    weight=25.727945,
                    codebook=num_groups * vocab_size,
                    q_vecs=-20.109955,
                    ids=26,
                    onehots=batch_size * seq_len * num_groups,
                    coverage=0.625,
                    pplx=2.1354918,
                    entropy=0.758697,
                ),
                False: dict(
                    weight=25.727945,
                    codebook=121.483154,
                    q_vecs=-54.850475,
                    ids=50,
                    onehots=batch_size * seq_len * num_groups,
                    coverage=0.375,
                    pplx=1.2845963,
                    entropy=0.25044453,
                ),
            },
            4: {
                True: dict(
                    weight=29.666399,
                    codebook=num_groups * vocab_size,
                    q_vecs=-27.134514,
                    ids=2008,
                    onehots=batch_size * seq_len * num_groups,
                    coverage=0.2,
                    pplx=2.9827762,
                    entropy=1.0928545,
                ),
                False: dict(
                    weight=29.666399,
                    codebook=5148.8306,
                    q_vecs=-345.86197,
                    ids=1793,
                    onehots=batch_size * seq_len * num_groups,
                    coverage=0.1,
                    pplx=1.5364327,
                    entropy=0.42946333,
                ),
            },
        }
        assert_allclose(
            np.sum(layer_params["rand_proj"]["weight"] ** 2),
            expected_values[batch_size][normalize_codebook]["weight"],
            atol=1e-6,
            rtol=1e-5,
        )
        assert_allclose(
            np.sum(layer_params["codebook"] ** 2),
            expected_values[batch_size][normalize_codebook]["codebook"],
            atol=1e-6,
            rtol=1e-6,
        )

        np.random.seed(2022)
        inputs = np.random.rand(batch_size, seq_len, input_dim).astype(np.float32)
        paddings = np.zeros((batch_size, seq_len)).astype(np.float32)
        q_outputs, output_collections = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=True,
            prng_key=jax.random.PRNGKey(10),
            state=layer_params,
        )
        self.assertEqual(
            (batch_size, seq_len, num_groups, dim_from_all_codebooks // num_groups),
            q_outputs.quantized_vectors.shape,
        )
        self.assertEqual((batch_size, seq_len, num_groups), q_outputs.ids.shape)
        self.assertEqual((batch_size, seq_len, num_groups, vocab_size), q_outputs.onehots.shape)
        assert_allclose(
            np.sum(
                jnp.reshape(
                    q_outputs.quantized_vectors, [batch_size, seq_len, dim_from_all_codebooks]
                )
            ),
            expected_values[batch_size][normalize_codebook]["q_vecs"],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            np.sum(q_outputs.ids * (1 - paddings[:, :, None])),
            expected_values[batch_size][normalize_codebook]["ids"],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            np.sum(q_outputs.onehots),
            expected_values[batch_size][normalize_codebook]["onehots"],
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(
            output_collections.summaries["codebook/num_frames"].mean,
            jnp.sum(1 - paddings) / batch_size,
        )
        assert_allclose(
            output_collections.summaries["codebook/coverage"].mean,
            expected_values[batch_size][normalize_codebook]["coverage"],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            output_collections.summaries["codebook/pplx"].mean,
            expected_values[batch_size][normalize_codebook]["pplx"],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            output_collections.summaries["codebook/entropy"].mean,
            expected_values[batch_size][normalize_codebook]["entropy"],
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(q_outputs.loss, None)

    @parameterized.product(normalize_inputs=(True, False), normalize_codebook=(True, False))
    def test_backward(self, normalize_inputs, normalize_codebook):
        batch_size, seq_len = 3, 4
        input_dim, dim_from_all_codebooks, vocab_size, num_groups = 10, 8, 4, 2
        cfg = RandomVectorQuantizer.default_config().set(
            name="test",
            input_dim=input_dim,
            normalize_inputs=normalize_inputs,
            normalize_codebook=normalize_codebook,
            codebook_dim=dim_from_all_codebooks // num_groups,
            codebook_size=vocab_size,
            num_codebooks=num_groups,
        )
        layer: RandomVectorQuantizer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        def _loss(params, inputs, paddings, layer=layer):
            q_outputs, o_col = F(
                layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                state=params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            return (
                jnp.sum(q_outputs.quantized_vectors**2)
                + jnp.sum(q_outputs.ids)
                + o_col.summaries["codebook/entropy"].mean
            )

        np.random.seed(2000)
        inputs = np.random.rand(batch_size, seq_len, input_dim).astype(np.float32)
        paddings = np.zeros((batch_size, seq_len)).astype(np.float32)

        _, (grad_params, grad_inputs) = jax.value_and_grad(_loss, argnums=(0, 1), has_aux=False)(
            layer_params, jnp.asarray(inputs), jnp.asarray(paddings)
        )
        self.assertNestedAllClose(grad_params, jax.tree.map(jnp.zeros_like, layer_params))
        assert_allclose(grad_inputs, jnp.zeros_like(inputs), atol=1e-6, rtol=1e-6)


class KmeansVectorQuantizerTest(TestCase):
    @parameterized.product(num_groups=(1, 2), input_mean=(0.0, -0.5))
    def test_forward(self, num_groups, input_mean):
        # The output is obtained from:
        # https://github.com/google/praxis/blob/179774fb688aa8fe048307d2184c9f2b338e935f/praxis/layers/quantizer_test.py#L32
        vocab_size, dim_from_all_codebooks = 4, 4
        codebook_dim = dim_from_all_codebooks // num_groups
        layer: KmeansVectorQuantizer = (
            KmeansVectorQuantizer.default_config()
            .set(
                name="test",
                codebook_dim=codebook_dim,
                codebook_size=vocab_size,
                num_codebooks=num_groups,
                beta=0.1,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        # [vocab_size, num_codebooks, codebook_dim].
        layer_params["codebook"] = jnp.reshape(_CODE_BOOK, [vocab_size, num_groups, codebook_dim])
        batch_size, seq_len = 2, 4
        np.random.seed(2021)
        inputs = (
            np.random.rand(batch_size, seq_len, dim_from_all_codebooks).astype(np.float32)
            + input_mean
        )
        paddings = jnp.arange(seq_len)[None, :] >= jnp.array([2, 3])[:, None]
        inputs = inputs * (1 - paddings)[:, :, None]
        outputs, output_collections = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=True,
            prng_key=jax.random.PRNGKey(1),
            state=layer_params,
            drop_output_collections=[],
        )
        expected_outputs = {
            1: {
                0.0: [7.5942173, 15, 0.25, 1.0, 0.0],
                -0.5: [-2.076443, 2, 0.5, 1.9601316, 0.67301166],
            },
            2: {
                0.0: [7.5942173, 30, 0.25, 1.0, 0.0],
                -0.5: [-0.8678032, 6, 0.625, 2.2732706, 0.82121956],
            },
        }
        self.assertEqual(
            (batch_size, seq_len, num_groups, dim_from_all_codebooks // num_groups),
            outputs.quantized_vectors.shape,
        )
        self.assertEqual((batch_size, seq_len, num_groups), outputs.ids.shape)
        self.assertEqual((batch_size, seq_len, num_groups, vocab_size), outputs.onehots.shape)

        assert_allclose(
            expected_outputs[num_groups][input_mean][0],
            np.sum(outputs.quantized_vectors),
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            expected_outputs[num_groups][input_mean][1],
            np.sum(outputs.ids * (1 - paddings[:, :, None])),
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            expected_outputs[num_groups][input_mean][2],
            output_collections.summaries["codebook/coverage"].mean,
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            expected_outputs[num_groups][input_mean][3],
            output_collections.summaries["codebook/pplx"].mean,
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            expected_outputs[num_groups][input_mean][4],
            output_collections.summaries["codebook/entropy"].mean,
            atol=1e-6,
            rtol=1e-6,
        )

        expected_losses = {
            1: {
                0.0: [0.08721543, 0.09593698],
                -0.5: [0.07469131, 0.08216044],
            },
            2: {
                0.0: [0.08721543, 0.09593698],
                -0.5: [0.053814936, 0.05919643],
            },
        }
        assert_allclose(
            output_collections.module_outputs["kmeans_loss"],
            expected_losses[num_groups][input_mean][0],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            output_collections.module_outputs["commitment_loss"],
            expected_losses[num_groups][input_mean][0],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            outputs.loss,
            expected_losses[num_groups][input_mean][1],
            atol=1e-6,
            rtol=1e-6,
        )

    @parameterized.product(norm_inputs=(False, True), norm_codebook=(False, True))
    def test_forward_with_normalization(self, norm_inputs, norm_codebook):
        num_groups = 2
        vocab_size, dim_from_all_codebooks = 4, 4
        codebook_dim = dim_from_all_codebooks // num_groups
        layer: KmeansVectorQuantizer = (
            KmeansVectorQuantizer.default_config()
            .set(
                name="test",
                codebook_dim=codebook_dim,
                codebook_size=vocab_size,
                num_codebooks=num_groups,
                beta=0.1,
                normalize_inputs=norm_inputs,
                normalize_codebook=norm_codebook,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        # [vocab_size, num_codebooks, codebook_dim].
        layer_params["codebook"] = jnp.reshape(_CODE_BOOK, [vocab_size, num_groups, codebook_dim])
        batch_size, seq_len = 2, 4
        inputs = jax.random.uniform(
            jax.random.PRNGKey(1), shape=(batch_size, seq_len, dim_from_all_codebooks)
        )
        paddings = jnp.arange(seq_len)[None, :] >= jnp.array([2, 3])[:, None]
        inputs = inputs * (1 - paddings)[:, :, None]
        F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=True,
            prng_key=jax.random.PRNGKey(1),
            state=layer_params,
            drop_output_collections=[],
        )
        # TODO(xianzhi): add tests against reference code and tests for backward pass.

    def test_backward(self):
        batch_size, seq_len = 4, 6
        dim_from_all_codebooks, vocab_size, num_groups = 15, 4, 3
        codebook_dim = dim_from_all_codebooks // num_groups
        beta = 0.5
        layer: KmeansVectorQuantizer = (
            KmeansVectorQuantizer.default_config()
            .set(
                name="test",
                codebook_dim=codebook_dim,
                codebook_size=vocab_size,
                num_codebooks=num_groups,
                beta=beta,
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        alpha = 3

        def _loss(params, inputs, paddings, layer=layer):
            outputs, _ = F(
                layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                state=params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            return (
                # Loss = kmeans_loss + commitment_loss + decoder_loss.
                outputs.loss + alpha * jnp.sum(jnp.exp(outputs.quantized_vectors)),
                outputs,
            )

        np.random.seed(2000)
        inputs = np.random.rand(batch_size, seq_len, dim_from_all_codebooks).astype(np.float32)
        paddings = np.arange(seq_len)[None, :] >= np.array([5, 6, 3, 4])[:, None]
        inputs = inputs * (1 - paddings)[:, :, None]

        (_, outputs), (grad_params, grad_inputs) = jax.value_and_grad(
            _loss, argnums=(0, 1), has_aux=True
        )(layer_params, jnp.asarray(inputs), jnp.asarray(paddings))
        # Check that padding is applied on the outputs.
        assert_allclose(
            outputs.ids * paddings[:, :, None],
            jnp.full_like(outputs.ids, fill_value=-1) * paddings[:, :, None],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            outputs.onehots * paddings[:, :, None, None],
            jnp.zeros_like(outputs.onehots),
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            outputs.quantized_vectors * paddings[:, :, None, None],
            jnp.zeros_like(outputs.quantized_vectors),
            atol=1e-6,
            rtol=1e-6,
        )

        num_frames = jnp.sum(1 - paddings)
        mask = (1 - paddings)[:, :, None]
        reshape_quantized_vectors = jnp.reshape(
            outputs.quantized_vectors, [batch_size, seq_len, -1]
        )
        grad_l2_loss = (
            2 * (inputs * mask - reshape_quantized_vectors) / dim_from_all_codebooks / num_frames
        )
        # Gradient of the dummy decoder loss on quantized_vectors.
        grad_q_vecs = jnp.exp(reshape_quantized_vectors)
        # Gradient w.r.t inputs come from two sources: commitment_loss and decoder loss.
        expected_grad_inputs = beta * grad_l2_loss + alpha * grad_q_vecs * mask
        assert_allclose(grad_inputs, expected_grad_inputs, atol=1e-6, rtol=1e-6)
        # Tests that no gradient on pad positions.
        assert_allclose(
            grad_inputs * paddings[:, :, None], jnp.zeros_like(grad_inputs), atol=1e-6, rtol=1e-6
        )

        # [batch_size, seq_len, num_groups, dim].
        # Gradient w.r.t codebook comes from kmeans_loss.
        grad_kmeans = -jnp.reshape(grad_l2_loss, [batch_size, seq_len, num_groups, codebook_dim])
        expected_grad_codebook = jnp.einsum("btgh,btgv->vgh", grad_kmeans, outputs.onehots)
        self.assertNestedAllClose(grad_params, dict(codebook=expected_grad_codebook))


class GumbelSoftmaxVectorQuantizerTest(TestCase):
    @parameterized.parameters(True, False)
    def test_forward(self, is_training):
        dim_from_all_codebooks, vocab_size, num_groups = 15, 5, 3
        input_dim = 10
        step = 5
        begin_step, begin_value, end_step, end_value = 0, 21, 10, 1
        codebook_dim = dim_from_all_codebooks // num_groups
        layer: GumbelSoftmaxVectorQuantizer = (
            GumbelSoftmaxVectorQuantizer.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                codebook_dim=codebook_dim,
                codebook_size=vocab_size,
                num_codebooks=num_groups,
                temperature_schedule=schedule.polynomial(
                    begin_step=begin_step,
                    begin_value=begin_value,
                    end_step=end_step,
                    end_value=end_value,
                ),
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        layer_params["step"] = step
        batch_size, seq_len = 2, 4
        np.random.seed(2021)
        inputs = np.random.rand(batch_size, seq_len, input_dim).astype(np.float32)
        paddings = np.array(
            np.arange(seq_len)[None, :] >= np.array([2, 3])[:, None], dtype=np.float32
        )
        inputs = inputs * (1 - paddings)[:, :, None]

        outputs, output_collections = F(
            layer,
            inputs=dict(inputs=inputs, paddings=paddings),
            is_training=is_training,
            prng_key=jax.random.PRNGKey(1),
            state=layer_params,
        )
        # Tests that the outputs are padded.
        assert_allclose(
            outputs.ids * paddings[:, :, None],
            jnp.full_like(outputs.ids, fill_value=-1) * paddings[:, :, None],
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            outputs.onehots * paddings[:, :, None, None],
            jnp.zeros_like(outputs.onehots),
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            outputs.quantized_vectors * paddings[:, :, None, None],
            jnp.zeros_like(outputs.quantized_vectors),
            atol=1e-6,
            rtol=1e-6,
        )
        if is_training:
            # Tests the temperature schedule.
            assert_allclose(
                output_collections.summaries["codebook/temperature"],
                begin_value
                + ((step - begin_step) / (end_step - begin_step)) * (end_value - begin_value),
                atol=1e-6,
                rtol=1e-6,
            )
            # Tests that the step is updated.
            self.assertEqual(output_collections.state_updates["step"], step + 1)

    def test_forward_against_fairseq(self):
        dim_from_all_codebooks, vocab_size, num_groups = 15, 4, 3
        input_dim = 8
        codebook_dim = dim_from_all_codebooks // num_groups
        constant_temperature = 1.0
        constant_schedule = schedule.as_schedule_fn(constant_temperature)
        layer: GumbelSoftmaxVectorQuantizer = (
            GumbelSoftmaxVectorQuantizer.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                codebook_dim=codebook_dim,
                codebook_size=vocab_size,
                num_codebooks=num_groups,
                temperature_schedule=constant_schedule,
            )
            .instantiate(parent=None)
        )
        batch_size, seq_len = 2, 4
        testcase = jnp.load(
            os.path.join(testdata_dir, __name__, "test_forward_against_fairseq.npy"),
            allow_pickle=True,
        ).item()
        ref_outputs = testcase["outputs"]
        paddings = testcase["paddings"]
        outputs, output_collections = F(
            layer,
            inputs=dict(inputs=testcase["inputs"], paddings=paddings),
            is_training=False,
            prng_key=jax.random.PRNGKey(1),
            state=testcase["params"],
        )
        assert_allclose(
            ref_outputs["code_perplexity"],
            output_collections.summaries["codebook/pplx"].mean * num_groups,
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            ref_outputs["x"].detach().numpy(),
            jnp.reshape(
                outputs.quantized_vectors, [batch_size, seq_len, num_groups * codebook_dim]
            ),
            atol=1e-6,
            rtol=1e-6,
        )
        assert_allclose(
            ref_outputs["targets"].detach().numpy(),
            outputs.ids * (1 - paddings[:, :, None]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_backward(self):
        batch_size, seq_len = 4, 6
        input_dim = 2
        dim_from_all_codebooks, vocab_size, num_groups = 6, 2, 3
        codebook_dim = dim_from_all_codebooks // num_groups
        layer: GumbelSoftmaxVectorQuantizer = (
            GumbelSoftmaxVectorQuantizer.default_config()
            .set(
                name="test",
                input_dim=input_dim,
                codebook_dim=codebook_dim,
                codebook_size=vocab_size,
                num_codebooks=num_groups,
                temperature_schedule=schedule.exponential(
                    begin_step=0, begin_value=2, end_step=100, end_value=0.01
                ),
            )
            .instantiate(parent=None)
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        def _loss(params, inputs, paddings, layer=layer):
            outputs, o_col = F(
                layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                state=params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
                drop_output_collections=[],
            )
            return jnp.sum(jnp.exp(outputs.quantized_vectors)), (outputs, o_col)

        np.random.seed(2000)
        inputs = np.random.rand(batch_size, seq_len, input_dim).astype(np.float32)
        paddings = np.arange(seq_len)[None, :] >= np.array([5, 6, 3, 4])[:, None]
        inputs = inputs * (1 - paddings)[:, :, None]

        (_, (outputs, side_outputs)), (grad_params, grad_inputs) = jax.value_and_grad(
            _loss, argnums=(0, 1), has_aux=True
        )(layer_params, jnp.asarray(inputs), jnp.asarray(paddings))

        # [batch_size, seq_len, num_groups, codebook_dim].
        # Gradient w.r.t. quantized_vectors.
        grad_q_vecs = jnp.exp(outputs.quantized_vectors)
        # Computes gradient w.r.t inputs using Gumbel softmax trick and chain rule.
        grad_onehots = jnp.einsum(
            "btgh,vgh->btgv",
            grad_q_vecs * (1 - paddings)[:, :, None, None],
            layer_params["codebook"],
        )
        # [batch_size, seq_len, num_groups, vocab_size, vocab_size].
        grad_softmax = side_outputs.module_outputs["probs"][:, :, :, :, None] * (
            jnp.identity(vocab_size)[None, None, None, :, :]
            - side_outputs.module_outputs["probs"][:, :, :, None, :]
        )
        grad_proj_out = (
            jnp.einsum("btgu,btguv->btgv", grad_onehots, grad_softmax)
            / side_outputs.summaries["codebook/temperature"]
        )
        expected_grad_inputs = jnp.einsum(
            "btgu,igu->bti", grad_proj_out, layer_params["input_proj"]["weight"]
        )
        assert_allclose(grad_inputs, expected_grad_inputs, atol=1e-6, rtol=1e-6)

        # [batch_size, seq_len, num_groups, dim].
        # Gradient w.r.t codebook.
        expected_grad_codebook = jnp.einsum("btgh,btgv->vgh", grad_q_vecs, outputs.onehots)
        assert_allclose(grad_params["codebook"], expected_grad_codebook, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    absltest.main()
