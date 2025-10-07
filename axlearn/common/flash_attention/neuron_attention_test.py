# Copyright © 2024 Amazon Inc.
"""Tests for Flash attention on Neuron. Tested on trn1 & trn2."""

from typing import Literal

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common.attention_bias import causal_mask
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.test_utils import generate_attention_data


class NeuronAttentionTest(parameterized.TestCase):
    """Tests Neuron FlashAttention kernels."""

    def setUp(self):
        super().setUp()
        if jax.default_backend() != "neuron":
            self.skipTest("Incompatible hardware, AWS Neuron only test.")

    @parameterized.product(
        [
            dict(batch_size=1, seq_len=2048, num_heads=1, per_head_dim=64),
            dict(batch_size=2, seq_len=2048, num_heads=2, per_head_dim=64),
            dict(batch_size=1, seq_len=2048, num_heads=1, per_head_dim=128),
            dict(batch_size=2, seq_len=2048, num_heads=2, per_head_dim=128),
            dict(batch_size=1, seq_len=2048, num_heads=8, per_head_dim=128),
            dict(batch_size=2, seq_len=2048, num_heads=8, per_head_dim=128),
        ],
        causal=[True, False],
        attention_bias_type=[None, "2d"],
        input_dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
    )
    def test_fwd_against_ref(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        per_head_dim: int,
        causal: bool,
        input_dtype: jnp.dtype,
        attention_bias_type: Literal[None, "2d"],
    ):
        # On demand import only if test is needed.
        # pylint: disable=import-outside-toplevel
        from axlearn.common.flash_attention.neuron_attention import NeuronFlashAttention

        q, k, v, bias = generate_attention_data(
            batch_size,
            seq_len,
            seq_len,
            num_heads,
            per_head_dim,
            mask_fn=causal_mask if causal else None,
            attention_bias_type=attention_bias_type,
            dtype=input_dtype,
        )

        cfg = dict(
            softmax_scale=q.shape[-1] ** -0.5,
        )
        # Compare outputs.
        test_fn = NeuronFlashAttention.default_config().set(**cfg).instantiate()
        ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()
        input_batch = dict(query=q, key=k, value=v, bias=bias, logit_sink=None)
        o = test_fn(input_batch)
        o_ref = ref_fn(input_batch)
        if input_dtype == jnp.float16:
            chex.assert_trees_all_close(o, o_ref, atol=0.07)
        elif input_dtype == jnp.float32:
            chex.assert_trees_all_close(o, o_ref, atol=0.03)

    @parameterized.product(
        [
            dict(batch_size=1, num_heads=1, seq_len=2048, per_head_dim=64),
            dict(batch_size=2, num_heads=2, seq_len=2048, per_head_dim=64),
            dict(batch_size=1, num_heads=1, seq_len=2048, per_head_dim=128),
            dict(batch_size=2, num_heads=2, seq_len=2048, per_head_dim=128),
            dict(batch_size=1, num_heads=8, seq_len=2048, per_head_dim=128),
            dict(batch_size=2, num_heads=8, seq_len=2048, per_head_dim=128),
        ],
        causal=[True, False],
        input_dtype=[jnp.bfloat16, jnp.float16, jnp.float32],
        attention_bias_type=[None, "2d"],
    )
    def test_bwd_against_ref(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        per_head_dim: int,
        causal: bool,
        input_dtype: jnp.dtype,
        attention_bias_type: Literal[None, "2d"],
    ):
        # On demand import only if test is needed.
        # pylint: disable=import-outside-toplevel
        from axlearn.common.flash_attention.neuron_attention import NeuronFlashAttention

        q, k, v, bias = generate_attention_data(
            batch_size,
            seq_len,
            seq_len,
            num_heads,
            per_head_dim,
            mask_fn=causal_mask if causal else None,
            attention_bias_type=attention_bias_type,
            dtype=input_dtype,
        )

        cfg = dict(
            softmax_scale=q.shape[-1] ** -0.5,
        )
        # Compare outputs.
        test_fn = NeuronFlashAttention.default_config().set(**cfg).instantiate()
        ref_fn = ReferenceMHA.default_config().set(**cfg).instantiate()

        jax_ref_grads = jax.grad(
            lambda q, k, v, b: ref_fn(dict(query=q, key=k, value=v, bias=b)).mean(),
            argnums=(0, 1, 2),
        )(q, k, v, bias)
        jax_grads = jax.grad(
            lambda q, k, v, b: test_fn(dict(query=q, key=k, value=v, bias=b)).mean(),
            argnums=(0, 1, 2),
        )(q, k, v, bias)
        chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.07)


if __name__ == "__main__":
    absltest.main()
