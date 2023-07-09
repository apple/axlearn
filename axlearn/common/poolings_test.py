"""Tests pooling layers."""
# pylint: disable=no-self-use
import itertools

import jax
import jax.random
import numpy as np
import torch
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import utils
from axlearn.common.layers import Linear
from axlearn.common.module import functional as F
from axlearn.common.poolings import (
    AttentionPooling,
    AveragePooling,
    FirstNTokenPooling,
    LastNTokenPooling,
    MaxPooling,
    PoolingWithProjection,
    SpladePooling,
)
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import shapes


class PoolingTest(TestCase):
    # TODO(bwzhang@) add a stronger unittest for attention pooling layers.
    @parameterized.parameters(itertools.product([1, 3], [6, 10], [10], [jnp.bfloat16, jnp.float32]))
    def test_attention_pooling(self, num_outputs, input_dim, output_dim, dtype):
        batch_size = 2
        seq_len = 10

        cfg = AttentionPooling.default_config().set(
            name="attention_pooling",
            num_outputs=num_outputs,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        pooler = cfg.instantiate(parent=None)  # type: AttentionPooling

        # Initialize pooler parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        pooler_params = pooler.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [batch_size, seq_len, input_dim])
        inputs = orig_inputs.copy()

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs,), dtype),
            is_training=True,
            state=utils.cast_floats(pooler_params, dtype),
            prng_key=prng_key,
        )

        # forward() should not mutate 'inputs' in-place.
        assert_allclose(inputs, orig_inputs)

        # Check the output shape.
        self.assertEqual(shapes(outputs), (batch_size, num_outputs, output_dim))
        self.assertEqual(outputs.dtype, dtype)

    @parameterized.parameters((jnp.float32), (jnp.bfloat16))
    def test_average_pooling(self, dtype):
        atol = 5e-3 if dtype == jnp.bfloat16 else 1e-6
        batch_size = 2
        seq_len = 10
        input_dim, output_dim = 10, 10

        cfg = AveragePooling.default_config().set(
            name="average_pooling", input_dim=input_dim, output_dim=output_dim
        )

        pooler = cfg.instantiate(parent=None)  # type: AveragePooling

        # Random inputs.
        prng_key = jax.random.PRNGKey(123)
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [batch_size, seq_len, input_dim])
        inputs = orig_inputs.copy()

        # test w/o mask
        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs,), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        assert_allclose(inputs, orig_inputs)  # forward() should not mutate 'inputs' in-place.

        outputs_expected = jnp.mean(orig_inputs, axis=1, keepdims=True)
        assert_allclose(outputs, outputs_expected, atol=atol)
        self.assertEqual(outputs.dtype, dtype)

        # test w/ mask
        paddings = jnp.hstack(
            [jnp.zeros((inputs.shape[0], inputs.shape[1] - 1)), jnp.ones((inputs.shape[0], 1))]
        )

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        outputs_expected = jnp.mean(inputs[:, :-1, :], axis=1, keepdims=True)
        assert_allclose(outputs, outputs_expected, atol=atol)

    @parameterized.parameters((3, 10, 10), (1, 6, 10))
    def test_average_pooling_exception(self, num_outputs, input_dim, output_dim):
        batch_size = 2
        seq_len = 10

        cfg = AveragePooling.default_config().set(
            name="average_pooling",
            num_outputs=num_outputs,
            input_dim=input_dim,
            output_dim=output_dim,
        )

        pooler = cfg.instantiate(parent=None)  # type: AveragePooling

        # Random inputs.
        prng_key = jax.random.PRNGKey(123)
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [batch_size, seq_len, input_dim])

        with self.assertRaises(ValueError):
            F(
                pooler,
                inputs=(inputs,),
                is_training=True,
                state=None,
                prng_key=prng_key,
            )

    @parameterized.parameters((jnp.float32), (jnp.bfloat16))
    def test_max_pooling(self, dtype):
        atol = 5e-3 if dtype == jnp.bfloat16 else 1e-6
        batch_size = 2
        seq_len = 10
        input_dim, output_dim = 10, 10

        cfg = MaxPooling.default_config().set(
            name="max_pooling", input_dim=input_dim, output_dim=output_dim
        )

        pooler = cfg.instantiate(parent=None)  # type: MaxPooling

        # Random inputs.
        prng_key = jax.random.PRNGKey(123)
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [batch_size, seq_len, input_dim])
        inputs = orig_inputs.copy()

        # test w/o mask
        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs,), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        assert_allclose(inputs, orig_inputs)  # forward() should not mutate 'inputs' in-place.

        outputs_expected = jnp.max(orig_inputs, axis=1, keepdims=True)
        assert_allclose(outputs, outputs_expected, atol=atol)
        self.assertEqual(outputs.dtype, dtype)

        # test w/ mask
        paddings = jnp.hstack(
            [jnp.zeros((inputs.shape[0], inputs.shape[1] - 1)), jnp.ones((inputs.shape[0], 1))]
        )

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        outputs_expected = jnp.max(inputs[:, :-1, :], axis=1, keepdims=True)
        assert_allclose(outputs, outputs_expected, atol=atol)

    @parameterized.parameters(itertools.product([1, 2, 3, 4, 5], [jnp.float32, jnp.bfloat16]))
    def test_first_n_pooling(self, n, dtype):
        atol = 5e-3 if dtype == jnp.bfloat16 else 1e-6
        batch_size = 2
        seq_len = 10
        input_dim, output_dim = 10, 10

        cfg = FirstNTokenPooling.default_config().set(
            name="first_n_pooling", input_dim=input_dim, output_dim=output_dim, num_outputs=n
        )

        pooler = cfg.instantiate(parent=None)  # type: FirstNTokenPooling

        # Random inputs.
        prng_key = jax.random.PRNGKey(123)
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [batch_size, seq_len, input_dim])
        inputs = orig_inputs.copy()

        # Test w/o mask.
        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs,), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        assert_allclose(inputs, orig_inputs)  # forward() should not mutate 'inputs' in-place.
        outputs_expected = inputs[:, :n]
        assert_allclose(outputs, outputs_expected, atol=atol)
        self.assertEqual(outputs.dtype, dtype)

        # Test w/ mask.
        paddings = jnp.hstack(
            [jnp.zeros((inputs.shape[0], inputs.shape[1] - 1)), jnp.ones((inputs.shape[0], 1))]
        )

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        outputs_expected = inputs[:, :n]
        assert_allclose(outputs, outputs_expected, atol=atol)
        self.assertEqual(outputs.dtype, dtype)

        # Test w/ jit.
        jit_f = jax.jit(pooler.forward)
        outputs = jit_f(inputs, paddings)
        assert_allclose(outputs, outputs_expected, atol=atol)

        # Test w/ all zero masks.
        paddings = jnp.ones((inputs.shape[0], inputs.shape[1]))

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        assert_allclose(outputs, jnp.zeros((inputs.shape[0], n, inputs.shape[2])), atol=atol)
        self.assertEqual(outputs.dtype, dtype)

    @parameterized.parameters((jnp.float32), (jnp.bfloat16))
    def test_last_pooling(self, dtype):
        atol = 5e-3 if dtype == jnp.bfloat16 else 1e-6
        batch_size = 2
        seq_len = 10
        input_dim, output_dim = 10, 10

        cfg = LastNTokenPooling.default_config().set(
            name="last_n_pooling",
            input_dim=input_dim,
            output_dim=output_dim,
            num_outputs=2,
        )

        pooler = cfg.instantiate(parent=None)  # type: LastNTokenPooling

        # Random inputs.
        prng_key = jax.random.PRNGKey(123)
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [batch_size, seq_len, input_dim])

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs,), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        # Test w/o mask.
        outputs_expected = inputs[:, -2:][:, ::-1]
        assert_allclose(outputs, outputs_expected, atol=atol)

        # Test w/ mask.
        paddings = jnp.hstack(
            [jnp.zeros((inputs.shape[0], inputs.shape[1] - 1)), jnp.ones((inputs.shape[0], 1))]
        )

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        outputs_expected = inputs[:, -3:-1][:, ::-1]
        assert_allclose(outputs, outputs_expected, atol=atol)

        # Test w/ specific mask.
        inputs = jax.random.normal(input_key, [3, 3, input_dim])
        paddings = jnp.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=jnp.int32)
        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )
        outputs_expected = jnp.stack((inputs[0, :2], inputs[1, :2], inputs[2, 1:]), axis=0)[:, ::-1]
        assert_allclose(outputs, outputs_expected, atol=atol)

        # Test w/ invalid masks.
        inputs = jnp.asarray(
            [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
        )
        paddings = jnp.asarray([[0, 1], [0, 0]], dtype=jnp.int32)

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=None,
            prng_key=prng_key,
        )

        outputs_expected = jnp.asarray(
            [[[0, 0, 0], [0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
        )[:, ::-1]
        assert_allclose(outputs, outputs_expected, atol=atol)

        # Test w/ jit.
        jit_f = jax.jit(pooler.forward)
        outputs = jit_f(inputs, paddings)
        assert_allclose(outputs, outputs_expected)

    @parameterized.parameters(
        itertools.product([1, 2, 3, 4, 5], [jnp.float32, jnp.bfloat16], [8, 10, 12])
    )
    def test_pooling_with_projection(self, n, dtype, output_dim):
        atol = 0.02 if dtype == jnp.bfloat16 else 1e-6
        batch_size = 2
        seq_len = 10
        input_dim = 10

        cfg = PoolingWithProjection.default_config().set(
            name="pooling_with_projection",
            input_dim=input_dim,
            output_dim=output_dim,
            pooler=FirstNTokenPooling.default_config().set(
                num_outputs=n,
            ),
            proj=Linear.default_config(),
        )
        pooler = cfg.instantiate(parent=None)  # type: PoolingWithProjection

        # Initialize pooler parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        pooler_params = pooler.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [batch_size, seq_len, input_dim])
        inputs = orig_inputs.copy()

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs,), dtype),
            is_training=True,
            state=utils.cast_floats(pooler_params, dtype),
            prng_key=prng_key,
        )

        # forward() should not mutate 'inputs' in-place.
        assert_allclose(inputs, orig_inputs)

        # [input_dim, output_dim]
        proj_weight = pooler_params["proj"]["weight"]
        # [output_dim]
        proj_bias = pooler_params["proj"]["bias"]

        # Test w/o mask.
        outputs_expected = inputs[:, :n] @ proj_weight + proj_bias

        assert_allclose(outputs, outputs_expected, atol=atol)
        self.assertEqual(outputs.dtype, dtype)

        # Test w/ mask.
        paddings = jnp.hstack(
            [jnp.zeros((inputs.shape[0], inputs.shape[1] - 1)), jnp.ones((inputs.shape[0], 1))]
        )

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=utils.cast_floats(pooler_params, dtype),
            prng_key=prng_key,
        )

        outputs_expected = inputs[:, :n] @ proj_weight + proj_bias
        assert_allclose(outputs, outputs_expected, atol=atol)
        self.assertEqual(outputs.dtype, dtype)

        # Test w/ all zero masks.
        paddings = jnp.ones((inputs.shape[0], inputs.shape[1]))

        outputs, _ = F(
            pooler,
            inputs=utils.cast_floats((inputs, paddings), dtype),
            is_training=True,
            state=utils.cast_floats(pooler_params, dtype),
            prng_key=prng_key,
        )

        assert_allclose(outputs, jnp.zeros((inputs.shape[0], n, output_dim)), atol=atol)
        self.assertEqual(outputs.dtype, dtype)


class SpladePoolingTest(TestCase):
    def ref_splade_implementation(
        self, inputs, attention_mask, agg, model_args
    ):  # pylint: disable=no-self-use
        """Reference splade implementation.

        Ref: https://github.com/naver/splade/blob/main/splade/models/transformer_rep.py#L188-L193
        """
        mapping_layer = torch.nn.Sequential(
            torch.nn.Linear(model_args["input_dim"], model_args["intermediate_dim"]),
            torch.nn.GELU("tanh"),
            torch.nn.LayerNorm([model_args["intermediate_dim"]], eps=1e-8),
            torch.nn.Linear(model_args["intermediate_dim"], model_args["output_dim"]),
        )
        out = mapping_layer(inputs)
        if agg == "sum":
            values = torch.sum(
                torch.log(1 + torch.relu(out)) * attention_mask.unsqueeze(-1), dim=1, keepdims=True
            )
        else:
            values, _ = torch.max(
                torch.log(1 + torch.relu(out)) * attention_mask.unsqueeze(-1), dim=1, keepdims=True
            )
        return (
            values,
            {
                "input_mapping": {
                    "linear": mapping_layer[0],
                    "norm": mapping_layer[2],
                },
                "vocab_mapping": mapping_layer[3],
            },
        )

    def verify_splade_against_ref(self, inputs, splade_layer, paddings, agg, model_args):
        # Process the paddings.
        if paddings is None:
            torch_paddings = torch.ones((inputs.shape[:-1]))
            axlearn_paddings = None
        else:
            torch_paddings = torch.from_numpy(paddings.astype(np.float))
            # axlearn_paddings = True means padded tokens.
            axlearn_paddings = jnp.asarray((1 - paddings).astype(np.bool))

        # Reference output.
        ref_output, ref_model_params = self.ref_splade_implementation(
            torch.from_numpy(inputs), torch_paddings, agg, model_args
        )

        # Set Splade layer parameters.
        layer_params = {
            "vocab_mapping": {
                "input_mapping": parameters_from_torch_layer(
                    ref_model_params["input_mapping"]["linear"]
                ),
                "input_norm": parameters_from_torch_layer(
                    ref_model_params["input_mapping"]["norm"]
                ),
                "vocab_mapping": parameters_from_torch_layer(ref_model_params["vocab_mapping"]),
            }
        }

        layer_output, _ = F(
            splade_layer,
            inputs=dict(tokens=jnp.asarray(inputs), paddings=axlearn_paddings),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_output, ref_output.detach().numpy())

    def test_splade_lm_head(self, mode="max"):
        # set up the env.
        batch_size = 2
        length = 12
        dim = 32
        intermediate_dim = 48
        vocab_size = 64

        # test w/o paddings
        splade_layer_cfg = SpladePooling.default_config().set(
            name="splade_head",
            input_dim=dim,
            output_dim=vocab_size,
            splade_mode=mode,
        )
        splade_layer_cfg.vocab_mapping.intermediate_dim = intermediate_dim
        splade_layer = splade_layer_cfg.instantiate(parent=None)
        model_args = {
            "input_dim": dim,
            "intermediate_dim": intermediate_dim,
            "output_dim": vocab_size,
        }
        inputs = np.random.random((batch_size, length, dim)).astype(np.float32)
        self.verify_splade_against_ref(
            inputs, splade_layer, paddings=None, agg=mode, model_args=model_args
        )

        # test w/ paddings
        inputs = np.random.random((batch_size, length, dim)).astype(np.float32)
        paddings = np.random.randint(0, 2, (batch_size, length))
        self.verify_splade_against_ref(
            inputs, splade_layer, paddings=paddings, agg=mode, model_args=model_args
        )

    def test_all_splade_mode(self):
        self.test_splade_lm_head(mode="max")
        self.test_splade_lm_head(mode="sum")


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
