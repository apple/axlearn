# Copyright © 2023 Apple Inc.

"""Speech testing utils."""

import jax
import jax.numpy as jnp

from axlearn.common.utils import Tensor


def fake_audio(
    *,
    batch_size: int,
    seq_len: int,
    prng_key: Tensor,
    scale: float = 32768.0,
    dtype: jnp.dtype = jnp.float32,
):
    """Generates fake audio data with a fixed seed."""
    input_key, length_key = jax.random.split(prng_key)
    inputs = jax.random.uniform(
        input_key,
        shape=[batch_size, seq_len],
        minval=-scale,
        maxval=scale,
        dtype=jnp.float32,
    ).astype(dtype)
    lengths = jax.random.randint(length_key, shape=[batch_size, 1], minval=0, maxval=seq_len)
    paddings = (jnp.arange(seq_len)[None, :] >= lengths).astype(jnp.int32)
    return inputs, paddings
