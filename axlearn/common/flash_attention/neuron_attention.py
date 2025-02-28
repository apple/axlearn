# Copyright © 2024 Amazon Inc.
"""Flash attention Kernels using NKI on Neuron. Tested on trn1 & trn2."""
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

# TODO(apoorvtintin): remove pytype disable when dependencies are public.
# pytype: disable=import-error
# Import needed to enable JAX cache on Neuron.
import jax_neuronx  # pylint: disable=unused-import
import neuronxcc.nki.language as nl
from jax import custom_vjp
from neuronxcc.nki.kernels.attention import flash_attn_bwd, flash_fwd

# pytype: enable=import-error

Tensor = jax.Array
lnc = 2 if jax.devices()[0].device_kind == "NC_v3d" else 1


# TODO(apoorvtintin): Add segment IDs as an argument when the kernel supports it.
@partial(custom_vjp, nondiff_argnums=(5, 6, 7))
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor] = None,
    prng_key: Optional[Tensor] = None,
    causal: bool = False,
    softmax_scale: float = 1.0,
    dropout_rate: float = 0.0,
):
    """Wraps _mha_forward for custom vjp.

    Args:
        query: Query of shape [batch_size, target_length, num_heads, per_head_dim].
        key: Key of shape [batch_size, source_length, num_heads, per_head_dim].
        value: Value of shape [batch_size, source_length, num_heads, per_head_dim].
        bias: Optional logit biases of shape [1, 1, target_length, source_length].
        prng_key: PRNG key used for dropout. Must be specified when dropout_rate > 0.0.
        causal: Whether to apply causal mask.
        softmax_scale: Optional scale to apply to softmax. Defaults to 1.
        dropout_rate: Dropout rate. Default to 0.0 (no dropout).

    Returns:
        The attention outputs of shape [batch_size, target_length, num_heads, per_head_dim].
    """
    out, _ = _mha_forward(query, key, value, bias, prng_key, causal, softmax_scale, dropout_rate)
    return out


def _mha_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Tensor,
    prng_key: Tensor,
    causal: bool,
    softmax_scale: float,
    dropout_rate: float,
):
    """Computes attention outputs following FlashAttention.

    See also `_mha_backward` for the backward pass.

    Args:
        query: Input query.
        key: Input key.
        value: Input value.
        bias: Input bias.
        prng_key: PRNG key used for dropout. Must be specified when dropout_rate > 0.0.
        causal: Whether to apply causal mask.
        softmax_scale: Softmax scale to use in the kernel.
        dropout_rate: Dropout rate to use in the kernel.
    """
    # Get the batch size, sequence lengths, number of heads, and hidden dimension.
    batch_size, _, num_heads, _ = query.shape

    # Transpose the query, key, and value tensors.
    q = query.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, q_seq_len].
    k = key.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, kv_seq_len].
    v = value.transpose(0, 2, 1, 3)  # [batch_size, num_heads, kv_seq_len, d_model].

    # TODO(apoorvtintin): Remove when kernel supports dropout.
    if prng_key is not None:
        raise NotImplementedError("PRNG key specified cannot be used since dropout is unsupported.")

    if dropout_rate > 0:
        if dropout_rate < 1:
            raise ValueError(f"dropout rate must be between 0 and 1, but got value {dropout_rate}.")
        if prng_key is None:
            raise ValueError("prng_key must be specified if dropout is enabled, got None.")
    else:
        # Dummy unused key.
        prng_key = jax.random.key(0)

    # TODO(apoorvtintin): Pass rbg key to kernel directly when kernel is ready to accept it.
    # Currenlty NKI kernel supports a single 32 bit key, temporarily override this till support
    # for 128 bit keys is added. Till then dropout is not supported.
    prng_key = jnp.array([1])

    # Call the NKI kernel, duplicate the kernel if we cannot shard on num_heads.
    if num_heads > 0 and num_heads % lnc == 0:
        grid = batch_size, nl.nc(lnc) * (num_heads // lnc)
    else:
        grid = batch_size, num_heads

    if bias is not None:
        if bias.ndim != 4:
            raise ValueError(
                f"Neuron flash_attention is only expecting bias.ndim = 4 but got {bias.ndim}"
            )
        if bias.shape[0] != 1 and bias.shape[1] != 1:
            raise ValueError(
                f"Bias is only supported when batch and num_heads are both 1, "
                f"batch is {bias.shape[0]} and num_heads is {bias.shape[1]}"
            )
        attn_output, lse = flash_fwd[grid](
            q,
            k,
            v,
            prng_key,
            bias,
            use_causal_mask=causal,
            softmax_scale=softmax_scale,
            mixed_precision=True,
            dropout_p=dropout_rate,
        )
    else:
        attn_output, lse = flash_fwd[grid](
            q,
            k,
            v,
            prng_key,
            use_causal_mask=causal,
            softmax_scale=softmax_scale,
            mixed_precision=True,
            dropout_p=dropout_rate,
        )
    # Transpose the output back to the original shape.
    attn_output = attn_output.transpose(0, 2, 1, 3)  # [batch_size, q_seq_len, num_heads, d_model].

    return attn_output, (lse, attn_output, q, k, v, bias, prng_key)


def _mha_backward(
    causal: bool,
    softmax_scale: float,
    dropout_rate: float,
    res,
    d_attn_output: Tensor,
):
    lse, o, q, k, v, bias, prng_key = res
    batch_size, num_heads, _, _ = q.shape

    # Transpose the input tensors.
    o = o.transpose(0, 2, 3, 1)
    dy = d_attn_output.transpose(0, 2, 3, 1)

    # Transpose v tensor.
    v = jnp.transpose(v, axes=(0, 1, 3, 2))

    # Call the NKI kernel, duplicate the kernel if we cannot shard on num_heads.
    if num_heads > 0 and num_heads % lnc == 0:
        grid = batch_size, nl.nc(lnc) * (num_heads // lnc)
    else:
        grid = batch_size, num_heads

    if bias is not None:
        assert (
            bias.ndim == 4
        ), f"Neuron flash_attention is only expecting bias.ndim = 4 but got {bias.ndim}"
        assert bias.shape[0] == 1 and bias.shape[1] == 1, (
            f"Bias is only supported when batch and num_heads are both 1, "
            f"batch is {bias.shape[0]} and num_heads is {bias.shape[1]}"
        )
        d_query, d_key, d_value = flash_attn_bwd[grid](
            q,
            k,
            v,
            o,
            dy,
            lse,
            prng_key,
            bias,
            use_causal_mask=causal,
            mixed_precision=True,
            dropout_p=dropout_rate,
            softmax_scale=softmax_scale,
        )
    else:
        d_query, d_key, d_value = flash_attn_bwd[grid](
            q,
            k,
            v,
            o,
            dy,
            lse,
            prng_key,
            use_causal_mask=causal,
            mixed_precision=True,
            dropout_p=dropout_rate,
            softmax_scale=softmax_scale,
        )

    # Transpose the gradients back to the original shape.
    d_query = d_query.transpose(0, 3, 1, 2)  # [batch_size, q_seq_len, num_heads, d_model]
    d_key = d_key.transpose(0, 3, 1, 2)  # [batch_size, kv_seq_len, num_heads, d_model]
    d_value = d_value.transpose(0, 3, 1, 2)  # [batch_size, kv_seq_len, num_heads, d_model]

    return d_query, d_key, d_value, None, None


flash_attention.defvjp(_mha_forward, _mha_backward)
