# Copyright © 2023 Apple Inc.

"""Implements FlashAttention kernel dispatch."""
from typing import Literal, Optional

from absl import logging

from axlearn.common.attention_bias import BaseAttentionBias
from axlearn.common.flash_attention.common import BaseFlashAttention, ReferenceMHA
from axlearn.common.flash_attention.gpu_attention import (
    CuDNNGPUFlashAttention,
    CuDNNGPUFlashAttentionWithExplicitBias,
    PallasGPUFlashAttention,
)
from axlearn.common.flash_attention.gpu_decoding import GPUDecoding
from axlearn.common.flash_attention.tpu_attention import LegacyTPUFlashAttention, TPUSplashAttention
from axlearn.common.flash_attention.tpu_decoding import TPUDecoding
from axlearn.common.utils import Tensor

BACKENDS = dict(
    # Always try decoding kernel first, then regular attention kernels.
    # For TPU, prefer SplashAttention whenever possible, as it's faster than legacy.
    tpu=[TPUDecoding, TPUSplashAttention, LegacyTPUFlashAttention],
    gpu=[
        GPUDecoding,
        # For GPU, prefer cuDNN (without bias) whenever possible, as it's the fastest.
        CuDNNGPUFlashAttention,
        # Fallbacks to Pallas if cuDNN cannot be used without instantiating bias tensors.
        PallasGPUFlashAttention,
        # If Pallas is not supported, fallback to cuDNN with bias as the last resort before we
        # fallback to plain XLA.
        CuDNNGPUFlashAttentionWithExplicitBias,
    ],
    cpu=[ReferenceMHA],
    xla=[ReferenceMHA],
)


def flash_attention_implementation(
    backend: Literal["cpu", "tpu", "gpu", "xla", "neuron"],
    *,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: BaseAttentionBias,
    softmax_scale: float = 1.0,
    is_decoding: bool = False,
    tpu_block_size: int = 512,
    gpu_block_size: int = 128,
    dropout_rate: Optional[float] = 0.0,
) -> Optional[BaseFlashAttention]:
    """Returns a jitted "flash" multihead-attention implementation for the given backend.

    The first matching kernel will be picked for each backend.

    Args:
        backend: A valid XLA backend name. 'cpu' intended for testing only.
        query: Query of shape [batch_size, target_length, num_heads, per_head_dim].
        key: Key of shape [batch_size, source_length, num_kv_heads, per_head_dim].
        value: Value of shape [batch_size, source_length, num_kv_heads, per_head_dim].
        bias: Attention bias to apply.
        softmax_scale: A scalar value applied to the logits before softmax.
        is_decoding: Whether it is in decoding.
        tpu_block_size: The size of the computation-block unit for 'tpu' backend.
            A multiple of 128, and should be less than the target sequence length.
            Smaller values are more memory efficient but less compute efficient.
        gpu_block_size: Block size for GPU Pallas kernels. The default value of 128 should be the
            best value for almost all cases.
        dropout_rate: The optional dropout rate.

    Returns:
        A jitted function implementing multi-head attention for the given backend.
        This jitted function may raise ValueError: If the given configuration doesn't logically
        make sense, e.g. if the shapes of q/k/v do not satisfy the requirement of a standard
        attention.
    """
    if dropout_rate is None:
        dropout_rate = 0.0

    if backend == "neuron":
        # Register neuron kernel at runtime due to extra dependencies.
        # pylint: disable-next=import-outside-toplevel
        from axlearn.common.flash_attention.neuron_attention import NeuronFlashAttention

        BACKENDS["neuron"] = [NeuronFlashAttention]
    attn_configs = BACKENDS.get(backend, [])
    common_cfg = dict(
        is_decoding=is_decoding,
        dropout_rate=dropout_rate,
        interpret=_interpret(backend),
        softmax_scale=softmax_scale,
        tpu_block_size=tpu_block_size,
        gpu_block_size=gpu_block_size,
    )
    for cfg in attn_configs:
        attn_fn = cfg.default_config().set(**common_cfg).instantiate()
        if attn_fn.is_supported(query=query, key=key, value=value, bias=bias):
            return attn_fn
    # Fall back to standard attention if no backend kernels are supported for the given
    # configuration.
    logging.warning("Using standard attention as flash attention fallback.")
    return None


def _interpret(backend: str):
    return backend == "cpu"
