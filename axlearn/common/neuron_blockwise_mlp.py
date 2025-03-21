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
        blockwise_mm as blockwise_mm_nki,
        # blockwise_mm_baseline_shard_hidden as blockwise_mm_nki,
        check_blockwise_mm_kernel_compatibility,
    )
from neuronxcc.nki._private_kernels.blockwise_mm_bwd import (
    blockwise_mm_bwd as blockwise_mm_bwd_nki,
    check_blockwise_mm_bwd_kernel_compatibility,
)
from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC
import neuronxcc.nki as nki

_blockwise_mm_nki_call = nki.jit(show_compiler_tb=True)(blockwise_mm_nki)

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

@partial(custom_vjp, nondiff_argnums=(5,10))
def blockwise_mm(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    block_size: int, 
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    gate_up_activations_T: Tensor=None,
    down_activations: Tensor=None,
    skip_dma: bool=True,
):
    
    # TODO handle O>1, G>1
    # Remove O, G dimensions
    hidden_states = jnp.squeeze(hidden_states, axis=(0,1,))
    expert_affinities_masked = jnp.squeeze(expert_affinities_masked, axis=(0,1,))
    token_position_to_id = jnp.squeeze(token_position_to_id, axis=(0,1,))
    block_to_expert = jnp.squeeze(block_to_expert, axis=(0,1,))
    
    # (N, 1)
    block_to_expert = jnp.expand_dims(block_to_expert, axis=1)

    # # add +1 for padding
    # padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
    # # (S+1, H)
    # hidden_states = jnp.concat(hidden_states, padding_h, dim=0)
    output = jnp.empty(hidden_states.shape, dtype=hidden_states.dtype)
    expert_affinities_masked = jnp.reshape(expert_affinities_masked, (-1, 1))
    gate_up_weight = jnp.stack([gate_weight, up_proj_weight], 
        axis=2
    )

    # out: (S+1, H)
    # out = blockwise_mm_nki[VNC(2)](
    out = _blockwise_mm_nki_call[VNC(2)](
        # Inputs
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        # MLP Weights
        gate_up_proj_weight=gate_up_weight,
        down_proj_weight=down_proj_weight,
        # Block Related
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        # Output
        output=output,
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        # LNC
        skip_dma=skip_dma,
        lnc=lnc,
    )
    return out

def _blockwise_mm_bwd(
    hidden_states,
    expert_affinities_masked,
    token_position_to_id,
    block_to_expert,
    gate_up_proj_weight,
    down_proj_weight,
    gate_up_activations,
    down_activations,
    grad_output,
    total_tokens,
    intermediate_size,
):
    # Get shapes
    _, E = expert_affinities_masked.shape
    T = total_tokens
    H = hidden_states.shape[-1]

    # Initialize gradients with zeros
    hidden_states_grad = jnp.zeros((T, H), dtype=hidden_states.dtype)
    affinities_grad = jnp.zeros((T, E), dtype=expert_affinities_masked.dtype)
    gate_up_proj_weight_grad = jnp.zeros_like(gate_up_proj_weight)
    down_weight_grad = jnp.zeros_like(down_proj_weight)

    # Compute gradients
    hidden_states_grad, affinities_grad, gate_up_proj_weight_grad, down_weight_grad = blockwise_mm_bwd_nki(
        hidden_states,
        hidden_states_grad,
        expert_affinities_masked.reshape(-1, 1),
        affinities_grad.reshape(-1, 1),
        gate_up_proj_weight,
        gate_up_proj_weight_grad,
        gate_up_activations,
        down_proj_weight,
        down_weight_grad,
        down_activations,
        token_position_to_id.astype(jnp.int32),
        block_to_expert.astype(jnp.int32),
        grad_output,
    )

    # Take only the relevant portion of hidden_states_grad
    hidden_states_grad = hidden_states_grad[:T]

    return (
        hidden_states_grad[:T],
        affinities_grad[:T],
        gate_up_proj_weight_grad.reshape(E, H, 2 * intermediate_size),
        down_weight_grad,
    )

blockwise_mm.defvjp(blockwise_mm, _blockwise_mm_bwd)