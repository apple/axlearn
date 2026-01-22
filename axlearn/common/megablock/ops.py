# Copyright © 2025 Apple Inc.
"""Grouped matrix multiplication operations with custom VJPs."""

from typing import Optional

import jax
import jax.numpy as jnp

import axlearn.common.megablock.gmm_gpu as backend
from axlearn.common.utils import Tensor

_gmm_impl_cache = None
GmmResidual = tuple[Tensor, Tensor, Tensor, Tensor | None, int]

gmm_gpu = jax.custom_vjp(
    backend.gmm,
    nondiff_argnums=(3, 4, 6, 7),
)

# _gmm_fwd: lhs x rhs = out
# _gmm_bwd: lhs_grad = out_grad x rhs.T, rhs_grad = lhs.T x out_grad

CAST_DTYPE = jnp.float32  # pylint: disable=invalid-name


def _gmm_fwd(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] = (128, 128, 128),
    group_offset: Optional[Tensor] = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> tuple[
    Tensor,
    GmmResidual,
]:
    """Forward function for GMM VJP."""
    # Overflow may happen on GPU with bf16, using fp32 for matmul calculation as a temporary
    # solution.
    # TODO(yiping): Investigate a better way to avoid overflow on GPU.

    out = backend.gmm(
        lhs.astype(CAST_DTYPE),  # cast to fp32
        rhs.astype(CAST_DTYPE),  # cast to fp32
        group_sizes,
        CAST_DTYPE,  # set preferred_element_type to fp32
        tiling,
        group_offset,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )
    out = out.astype(preferred_element_type)
    return out, (lhs, rhs, group_sizes, group_offset, rhs.shape[0])


def _gmm_bwd(
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    residual: GmmResidual,
    grad: Tensor,
) -> tuple[Tensor, Tensor, None, None]:
    """Backward function for throughput GMM VJP."""
    del preferred_element_type
    lhs, rhs, group_sizes, group_offset, num_actual_groups = residual
    # Overflow may happen on GPU with bf16, using fp32 for matmul calculation as a temporary
    # solution.
    # TODO(yiping): Investigate a better way to avoid overflow on GPU.
    grad_lhs = backend.gmm(
        grad.astype(CAST_DTYPE),
        rhs.astype(CAST_DTYPE),
        group_sizes,
        CAST_DTYPE,  # lhs[0].dtype
        tiling,
        group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
    )
    grad_rhs = backend.tgmm(
        lhs.swapaxes(0, 1).astype(CAST_DTYPE),
        grad.astype(CAST_DTYPE),
        group_sizes,
        CAST_DTYPE,  # rhs.dtype
        tiling,
        group_offset,
        num_actual_groups,
        interpret=interpret,
    )

    # NOTE: If the rhs transposition is fused into the forward pass we need to
    # return the transpose of the rhs gradient that we calculated above.
    grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
    # casting grad back to input original dtype
    grad_lhs = grad_lhs.astype(lhs[0].dtype)
    grad_rhs = grad_rhs.astype(rhs.dtype)
    return grad_lhs, grad_rhs, None, None


gmm_gpu.defvjp(_gmm_fwd, _gmm_bwd)


def _select_and_cache_gmm_backend():
    """Internal function to detect backend, select implementation, and cache it."""
    # pylint: disable-next=global-statement
    global _gmm_impl_cache

    backend_name = jax.default_backend()
    print(f"[gmm interface] Selecting GMM implementation for backend: {backend_name}")

    if backend_name == "gpu":
        _gmm_impl_cache = gmm_gpu
    else:
        # pylint: disable-next=ungrouped-imports,import-outside-toplevel
        import jax.experimental.pallas.ops.tpu.megablox as mblx

        _gmm_impl_cache = mblx.gmm

    return _gmm_impl_cache


def gmm(*args, **kwargs):
    """
    Universal GMM interface. Selects and caches the backend-specific
    implementation on the first call.

    Assumes jax.distributed.initialize() has completed before its first execution.
    """
    selected_impl = _gmm_impl_cache or _select_and_cache_gmm_backend()
    return selected_impl(*args, **kwargs)
