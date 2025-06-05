# Copyright © 2025 Apple Inc.
"""Remat policy for FlashAttention kernels."""

from jax._src.ad_checkpoint import name_p
from jax._src.cudnn.fused_attention_stablehlo import _dot_product_attention_fwd_p_wrapper

from axlearn.common.utils import Offloadable, Recompute, RematPolicy, RematType, Saveable

FLASH_ATTN_RESIDUAL_NAME = "flash_residuals"


# TODO(hanzhi-zhou): simplify after cudnn attention supports passing checkpoint names.
def save_or_offload_flash_attention_policy(remat_type: RematType = Saveable) -> RematPolicy:
    """Returns a remat policy for FlashAttention output.

    This remat policy allows saving attention output, which is the tensor before out projection
    commonly named "context". More precisely, it saves the attention output of GPU Pallas kernel,
    TPU Legacy Pallas kernel, TPU SplashAttention kernel, and cuDNN FlashAttention kernel.

    Because cuDNN FlashAttention invocation is in Jax source code, it's not feasible to save the
    output using `checkpoint_name`. Therefore, we match the Jax primitives to implement this save
    policy.

    Note for users: for context length >= 4096, FlashAttention kernel takes noticeably longer on
    both TPU and GPU to execute than o_proj. Therefore, saving the output of FlashAttention is
    more advantages than saving o_proj since they have roughly the same memory footprint if the HBM
    capacity doesn't allow saving both.

    Args:
        remat_type: Remat type. Defaults to Saveable (save to HBM). Note that Offloadable is not
            supported when using cuDNN flash attention.

    Returns:
        A RematPolicy. Users can combine this remat policy with any existing policy with
            `axlearn.common.utils.combine_remat_policies`.
    """

    def policy(prim, *_, **params):
        # Primitives could be copies if modules are reinitialized, so `is` check is unreliable.
        # Use string equality instead.
        prim_s = str(prim)
        if prim_s == str(name_p):
            if FLASH_ATTN_RESIDUAL_NAME in params["name"]:
                return remat_type
        # cuDNN kernel.
        if prim_s == str(_dot_product_attention_fwd_p_wrapper):
            if isinstance(remat_type, Offloadable):
                # Raise a nice error message rather than a Jax internal error.
                # See https://github.com/jax-ml/jax/issues/25841.
                # TODO(hanzhi-zhou): the bug is fixed in nightly. Remove this restriction
                # once we upgrade to jax 0.4.39 or newer.
                raise NotImplementedError(
                    "Offloading cuDNN attention is not supported due to a Jax bug."
                )
            return remat_type
        return Recompute

    return policy
