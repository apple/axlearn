# Copyright © 2023 Apple Inc.

"""Tests BEiT image tokenizer."""
import jax
from absl.testing import absltest, parameterized

from axlearn.common.module import functional as F
from axlearn.common.param_init import ConstantInitializer
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.vision.beit_image_tokenizer import (
    BEiTStochasticDepth,
    set_beit_image_tokenizer_encoder_config,
)


# TODO (bwzhang@) add reference implementation comparison.
class BEITImageQuantizerTest(TestCase):
    """Tests BEIT utils."""

    def test_forward(self):  # pylint: disable=no-self-use
        batch_size = 2
        codebook_dim = 8
        codebook_size = 4
        tokenizer_cfg = set_beit_image_tokenizer_encoder_config(
            num_layers=2,
            model_dim=16,
            num_heads=4,
            image_size=(24, 24),
            patch_size=(6, 6),
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )
        tokenizer_cfg.set(name="test")
        tokenizer = tokenizer_cfg.instantiate(parent=None)

        inputs = jax.random.uniform(key=jax.random.PRNGKey(123), shape=(batch_size, 24, 24, 3))
        layer_params = tokenizer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        (output, aux), _ = F(
            tokenizer,
            inputs=dict(inputs=inputs),
            is_training=True,
            prng_key=jax.random.PRNGKey(10),
            state=layer_params,
        )
        assert_allclose(aux["quantized_vectors"].shape, (batch_size, 16, codebook_dim))
        assert_allclose(aux["quantized_codebook_onehots"].shape, (batch_size, 16, codebook_size))
        assert_allclose(output.shape, (batch_size, 16))

    @parameterized.parameters(None, 0.0, 0.2, 1.0)
    def test_beit_stochastic_depth(self, rate):
        batch_size, tgt_len = 10, 6
        model_dim = 16
        cfg = BEiTStochasticDepth.default_config().set(
            name="test",
            input_dim=model_dim,
            rate=rate,
            param_init=ConstantInitializer.default_config().set(value=2),
        )
        layer = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        target = jax.random.normal(jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim])

        if rate is None or 0 <= rate < 1:
            output, _ = F(
                layer,
                inputs=dict(x=target),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            if rate is None or rate == 0:
                assert_allclose(output, target * 2)
        else:
            with self.assertRaises(ValueError):
                F(
                    layer,
                    inputs=dict(x=target),
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )


if __name__ == "__main__":
    absltest.main()
