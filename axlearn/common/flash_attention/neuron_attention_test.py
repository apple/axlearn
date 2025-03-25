# Copyright © 2024 Amazon Inc.
"""Tests for Flash attention on Neuron. Tested on trn1 & trn2."""

from typing import Literal

import chex
import jax
import jax.numpy as jnp
import pytest

from axlearn.common.attention_bias import causal_mask
from axlearn.common.flash_attention.common import ReferenceMHA
from axlearn.common.flash_attention.test_utils import generate_attention_data

if jax.default_backend() != "neuron":
    pytestmark = pytest.skip(
        reason="Incompatible hardware, AWS Neuron only test.", allow_module_level=True
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        (1, 2048, 1, 64),
        (2, 2048, 2, 64),
        (1, 2048, 1, 128),
        (2, 2048, 2, 128),
        (1, 2048, 8, 128),
        (2, 2048, 8, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("attention_bias_type", [None, "2d"])
@pytest.mark.parametrize("input_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_fwd_against_ref(
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

    o = test_fn(q, k, v, bias)
    o_ref = ref_fn(q, k, v, bias)
    if input_dtype == jnp.float16:
        chex.assert_trees_all_close(o, o_ref, atol=0.07)
    elif input_dtype == jnp.float32:
        chex.assert_trees_all_close(o, o_ref, atol=0.03)


@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 1, 2048, 64),
        (2, 2, 2048, 64),
        (1, 1, 2048, 128),
        (2, 2, 2048, 128),
        (1, 8, 2048, 128),
        (2, 8, 2048, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("input_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
@pytest.mark.parametrize("attention_bias_type", [None, "2d"])
def test_bwd_against_ref(
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

    jax_grads = jax.grad(lambda *args: test_fn(*args).mean(), argnums=(0, 1, 2))(q, k, v, bias)
    jax_ref_grads = jax.grad(lambda *args: ref_fn(*args).mean(), argnums=(0, 1, 2))(q, k, v, bias)
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.07)
