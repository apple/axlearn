from absl import logging
from functools import partial
import jax
import jax.numpy as jnp
import jax.numpy as jnp
from jax import custom_vjp
import os

lnc = 2 if jax.devices()[0].device_kind == "NC_v3d" else 1

@partial(custom_vjp, nondiff_argnums=(4, 5))
def flash_attention(query, key, value, bias, causal, softmax_scale):
    # NOTE : Merge with upstream. Old code supports both 2d and 4d bias but upstream code only supports 4d.
    #       We no longer need 2d logit_bias but should sync how we merge this check with upstream.
    #   assert bias.ndim == 4, f"Neuron flash_attention is only expecting bias.ndim = 4 but got {bias.ndim}"
    out, _ = _mha_forward(query, key, value, bias, causal, softmax_scale)
    return out


def _mha_forward(query, key, value, bias, causal, softmax_scale):
    # Get the batch size, sequence lengths, number of heads, and hidden dimension
    batch_size, q_seq_len, num_heads, d_model = query.shape
    _, kv_seq_len, _, _ = key.shape

    # Transpose the query, key, and value tensors
    q = query.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, q_seq_len]
    k = key.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, kv_seq_len]
    v = value.transpose(0, 2, 1, 3)  # [batch_size, num_heads, kv_seq_len, d_model]

    import neuronxcc.nki.language as nl
    from neuronxcc.nki.kernels.attention import flash_fwd
    seed = jnp.array([1])

    # Call the NKI kernel
    assert (num_heads % 2) == 0 and (num_heads // 2 > 0), f"unexpected num_heads: {num_heads}"
    
    if bias != None:
        attn_output, lse = flash_fwd[batch_size, nl.nc(lnc) * (num_heads // lnc)](
            q,
            k,
            v,
            seed,
            bias,
            use_causal_mask=causal,
            softmax_scale=softmax_scale,
            mixed_precision=True,
            dropout_p=0.0,
        )
    else:
        attn_output, lse = flash_fwd[batch_size, nl.nc(lnc) * (num_heads // lnc)](
            q,
            k,
            v,
            seed,
            use_causal_mask=causal,
            softmax_scale=softmax_scale,
            mixed_precision=True,
            dropout_p=0.0,
        )
    # Transpose the output back to the original shape
    attn_output = attn_output.transpose(0, 2, 1, 3)  # [batch_size, q_seq_len, num_heads, d_model]

    return attn_output, (lse, attn_output, q, k, v, bias)


def _mha_backward(causal, softmax_scale, res, d_attn_output):
    lse, o, q, k, v, bias = res
    batch_size, num_heads, d_model, seq_len = q.shape

    # Transpose the input tensors
    o = o.transpose(0, 2, 3, 1)
    dy = d_attn_output.transpose(0, 2, 3, 1)

    # Transpose v tensor
    v = jnp.transpose(v, axes=(0, 1, 3, 2))
    seed = jnp.array([1])

    from neuronxcc.nki.kernels.attention import flash_attn_bwd
    import neuronxcc.nki.language as nl

    # Call the NKI kernel
    assert (num_heads % 2) == 0 and (num_heads // 2 > 0), f"unexpected num_heads: {num_heads}"
    if bias != None:
        d_query, d_key, d_value = flash_attn_bwd[batch_size, nl.nc(lnc) * (num_heads // lnc)](
            q,
            k,
            v,
            o,
            dy,
            lse,
            seed,
            bias,
            use_causal_mask=causal,
            mixed_precision=True,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
        )
    else:
        d_query, d_key, d_value = flash_attn_bwd[batch_size, nl.nc(lnc) * (num_heads // lnc)](
            q,
            k,
            v,
            o,
            dy,
            lse,
            seed,
            use_causal_mask=causal,
            mixed_precision=True,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
        )

    # Batch seq_len heads, head_dim
    # Transpose the gradients back to the original shape
    d_query = d_query.transpose(0, 3, 1, 2)
    d_key = d_key.transpose(0, 3, 1, 2)
    d_value = d_value.transpose(0, 3, 1, 2)

    return d_query, d_key, d_value, None


flash_attention.defvjp(_mha_forward, _mha_backward)
