# Copyright © 2025 Apple Inc.
"""Remat policy for FlashAttention kernels."""

from jax._src.cudnn.fused_attention_stablehlo import _dot_product_attention_fwd_p_wrapper
from jax.custom_derivatives import custom_vjp_call_jaxpr_p
from jax.experimental.pallas import pallas_call_p

from axlearn.common.utils import Recompute, RematPolicy, RematType, Saveable


def save_or_offload_flash_attention_policy(remat_type: RematType = Saveable) -> RematPolicy:
    """Returns a remat policy for FlashAttention output.

    This remat policy allows saving attention output, which is the tensor before out projection
    commonly named "context". More precisely, it saves the attention output of GPU Pallas kernel,
    TPU Legacy Pallas kernel, TPU SplashAttention kernel, and cuDNN FlashAttention kernel.

    Because cuDNN FlashAttention and TPU SplashAttention invocations are in Jax source code, it's
    not feasible to save the output using `checkpoint_name`. Therefore, we match the Jax primitives
    to implement this save policy.

    Note for users: for context length >= 4096, FlashAttention kernel takes noticeably longer on
    both TPU and GPU to execute than o_proj. Therefore, saving the output of FlashAttention is
    more advantages than saving o_proj since they have roughly the same memory footprint if the HBM
    capacity doesn't allow saving both.

    Args:
        remat_type: Remat type. Defaults to Saveable (save to HBM) and only supports Saveable.

    Returns:
        A RematPolicy. Users can combine this remat policy with any existing policy with
            `axlearn.common.utils.combine_remat_policies`.
    """
    # Jax bug: https://github.com/jax-ml/jax/issues/25841.
    # TODO(hanzhi-zhou): add support for Offloadable when jax supports it.
    if remat_type is not Saveable:
        raise NotImplementedError(f"{remat_type=} is not implemented.")

    def policy(prim, *_, **params):
        src_info = ""
        # Primitives could be copies if modules are reinitialized, so `is` check is unreliable.
        # Use string equality instead.
        prim_s = str(prim)
        if prim_s == str(pallas_call_p):
            src_info = str(params.get("name_and_src_info", ""))
        if prim_s == str(custom_vjp_call_jaxpr_p):
            src_info = str(params.get("fun_jaxpr", ""))
        # GPU Pallas kernel.
        if "_mha_forward_kernel" in src_info:
            return remat_type
        # TPU new and legacy Pallas kernel.
        if "flash_attention_kernel" in src_info:
            return remat_type
        # cuDNN kernel.
        if prim_s == str(_dot_product_attention_fwd_p_wrapper):
            return remat_type
        return Recompute

    return policy
