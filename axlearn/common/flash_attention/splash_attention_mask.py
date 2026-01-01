"""Splash attention mask.

This code is adapted from the jax_ml/jax library, specifically from the
https://github.com/jax-ml/jax/blob/9bcfac6542a330b77f29d5cc5dcf4a57f55b2947/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask.py
TODO(dhwang2): Delete this file once JAX is upgraded and LocalMask becomes a computable mask.
"""

from typing import Callable

import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import _ComputableMask

from axlearn.common.utils import Tensor


class ComputableMask(_ComputableMask):
    """Computable mask for splash attention that supports custom mask functions.

    This mask accepts any Jax/Numpy exchangeable mask function following the MaskFn protocol
    from attention_bias.py, such as causal_mask or sliding_window_causal_mask.

    Attributes:
      mask_fn: A callable mask function that takes query_position and key_position
        tensors and returns a boolean mask tensor.
    """

    mask_fn: Callable[[Tensor, Tensor], Tensor]

    def __init__(
        self,
        shape: tuple[int, int],
        mask_fn: Callable[[Tensor, Tensor], Tensor],
        shard_count: int = 1,
    ):
        """Initialize ComputableMask.

        Args:
            shape: The shape of the attention mask (q_len, kv_len).
            mask_fn: A callable that implements the MaskFn protocol from attention_bias.py.
                Takes (query_position, key_position) and returns a boolean mask.
            shard_count: Number of shards.
        """
        self.mask_fn = mask_fn

        def mask_function(q_ids, kv_ids):
            """Computes the attention mask using the provided mask_fn."""
            assert q_ids.ndim == 2
            assert kv_ids.ndim == 2
            return self.mask_fn(q_ids, kv_ids)

        super().__init__(
            shape=shape,
            mask_function=mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return False

        return (
            self.shape == other.shape
            and self.mask_fn == other.mask_fn
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                id(self.mask_fn),
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )
