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
from neuronxcc.nki._private_kernels.blockwise_mm import (
        blockwise_mm_baseline as blockwise_mm_nki,
        check_blockwise_mm_kernel_compatibility,
    )
from neuronxcc.nki._private_kernels.blockwise_mm_bwd import (
    blockwise_mm_bwd as blockwise_mm_bwd_nki,
    check_blockwise_mm_bwd_kernel_compatibility,
)
from neuronxcc.nki._private_kernels.blockwise_mm import ExpertAffinityScaleMode
Tensor = jax.Array
lnc = 2 if jax.devices()[0].device_kind == "NC_v3d" else 1

def can_use_blockwise_matmul_nki(
    hidden_size,
    intermediate_size_tp,
    block_size,
    glu_mlp,
):
    if not glu_mlp:
        print("Blockwise NKI kernel incompatible with glu_mlp=False")
        return False

    if blockwise_mm_nki is None:
        print("Failed to load Blockwise NKI kernel.")
        return False

    try:
        check_blockwise_mm_kernel_compatibility(
            hidden_size=hidden_size,
            block_size=block_size,
            intermediate_size_tp=intermediate_size_tp,
        )
    except AssertionError as e:
        print(f"Blockwise kernel not compatible with model config. Reason: {str(e)}")
        return False

    return True

@partial(custom_vjp, nondiff_argnums=(4,12,13,14))
def blockwise_mm(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    block_size: int, 
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    gate_up_proj_scale: Tensor=None,
    down_proj_scale: Tensor=None,
    gate_up_activations_T: Tensor=None,
    down_activations_T: Tensor=None,
    skip_dma: bool=False,
    is_tensor_update_accumulating: bool=True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode=ExpertAffinityScaleMode.POST_SCALE,
):
    # TODO handle O>1, G>1
    # Remove O, G dimensions
    hidden_states = jnp.squeeze(hidden_states, axis=(0,1,))
    expert_affinities_masked = jnp.squeeze(expert_affinities_masked, axis=(0,1,))
    token_position_to_id = jnp.squeeze(token_position_to_id, axis=(0,1,))
    block_to_expert = jnp.squeeze(block_to_expert, axis=(0,1,))
    
    # (N, 1)
    block_to_expert = jnp.expand_dims(block_to_expert, axis=1)

    # add +1 for padding
    padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
    # (S+1, H)
    hidden_states = jnp.concat(hidden_states, padding_h, dim=0)

    # expert_affinities_masked =

    gate_up_weight = jnp.stack([gate_weight, up_proj_weight], 
        axis=2
    )
    output = jnp.empty(hidden_states.shape, dtype=hidden_states.dtype)

    # out: (S+1, H)
    out, gate_up_activations_T, down_activations = blockwise_mm_nki(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        output=output,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        gate_up_activations_T=gate_up_activations_T,
        down_activations_T=down_activations_T,
        skip_dma=skip_dma,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        lnc=lnc,
    )
    # remove the additional row for padding
    # out: (O, G, S, H)
    out = jnp.expand_dims(out, axis=(0, 1))[:, :, :-1, :]

    # gate_up_activations = jnp.empty(
    #     dp,
    #     num_block,
    #     2,
    #     intermediate_size,
    #     block_size,
    #     dtype=hidden_states.dtype,
    #     device=gate_up_proj_weight.device,
    # )
    # # (DP, N, B, H)
    # down_activations = jnp.empty(
    #     dp, num_block, block_size, hidden_size, dtype=hidden_states.dtype, device=down_proj_weight.device
    # )

    return out


blockwise_mm.defvjp(blockwise_mm_nki, blockwise_mm_bwd_nki)