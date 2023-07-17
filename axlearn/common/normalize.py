# Copyright © 2023 Apple Inc.

"""Normalization utils."""
import jax

from axlearn.common.utils import Tensor


def l2_normalize(x: Tensor, eps: float = 1e-8, axis: int = -1) -> Tensor:
    """l2_normalize Normalizes along the dimension `axis` using an L2 norm.

    Args:
        x: Input tensor.
        axis: Dimension along which to normalize.
        eps: A lower bound value for the norm. Defaults to 1e-8.

    Returns:
        A Tensor with the same shape as x.
    """
    sum2 = (x * x).sum(axis=axis, keepdims=True)
    return x * jax.lax.rsqrt(sum2 + eps)
