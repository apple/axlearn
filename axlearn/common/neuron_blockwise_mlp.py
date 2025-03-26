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
        # blockwise_mm as blockwise_mm_nki,
        blockwise_mm_selective_cp as blockwise_mm_nki,
        # blockwise_mm_baseline_shard_hidden as blockwise_mm_nki,
        check_blockwise_mm_kernel_compatibility,
    )
from neuronxcc.nki._private_kernels.blockwise_mm_bwd import (
    # blockwise_mm_bwd as blockwise_mm_bwd_nki,
    blockwise_mm_bwd_selective_cp as blockwise_mm_bwd_nki,
    check_blockwise_mm_bwd_kernel_compatibility,
)
from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC
import neuronxcc.nki as nki

_blockwise_mm_nki_call = nki.jit(show_compiler_tb=True)(blockwise_mm_nki)
_blockwise_mm_bwd_nki_call = nki.jit(show_compiler_tb=True)(blockwise_mm_bwd_nki)


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

@partial(custom_vjp, nondiff_argnums=(5,))
def blockwise_mm(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    block_size: int, 
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
):    
    out, _ = _blockwise_mm_fwd(hidden_states, expert_affinities_masked, gate_weight, up_proj_weight,
                            down_proj_weight, block_size, token_position_to_id, block_to_expert)
    return out

def _blockwise_mm_fwd(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    block_size: int, 
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
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
    padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
    padding_e = jnp.zeros((1,expert_affinities_masked.shape[1]), dtype=expert_affinities_masked.dtype)
    # # (S+1, H)
    hidden_states = jnp.concat([hidden_states, padding_h], axis=0)
    expert_affinities_masked = jnp.concat([expert_affinities_masked, padding_e], axis=0)
    expert_affinities_masked = jnp.reshape(expert_affinities_masked, (-1, 1))

    gate_up_weight = jnp.stack([gate_weight, up_proj_weight], 
        axis=2
    )

    print("gate:::", gate_up_weight)
    print("hidden_states:::", hidden_states)
    print("down:::", down_proj_weight)

    # hidden_states = jnp.zeros_like(hidden_states)
    # expert_affinities_masked = jnp.zeros_like(expert_affinities_masked)
    # gate_up_weight = jnp.zeros_like(gate_up_weight)
    # down_proj_weight = jnp.zeros_like(down_proj_weight)
    # block_size = block_size  # If it's an integer, just use the same value
    # token_position_to_id = jnp.zeros_like(token_position_to_id)
    # block_to_expert = jnp.zeros_like(block_to_expert)

    # out: (S+1, H)
    # out = blockwise_mm_nki[VNC(2)](
    out, gate_up_activations_T, down_activations = _blockwise_mm_nki_call[VNC(2)](
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
    )

    print("out:::", out)
    print("gate_up_activations_T:::", gate_up_activations_T)
    print("down_activations:::", down_activations)

    return out[:-1, :], (hidden_states, expert_affinities_masked, gate_up_weight, 
                down_proj_weight, down_activations, gate_up_activations_T, 
                token_position_to_id, block_to_expert)

def _blockwise_mm_bwd(
    block_size,
    res,
    grad_output
):
    (hidden_states, expert_affinities_masked, gate_up_proj_weight, 
     down_proj_weight, down_activations, gate_up_activations_T, 
     token_position_to_id, block_to_expert) = res
    
    T,H = hidden_states.shape
    E, _, _, _ = gate_up_proj_weight.shape

    print("gate grads::::", grad_output)

    print("hidden states::, ", hidden_states.shape)

    print("block sizeeee:::" , block_size)
    # Compute gradients
    hidden_states_grad, affinities_grad, gate_up_proj_weight_grad, down_weight_grad = _blockwise_mm_bwd_nki_call[VNC(2)](
        hidden_states,
        expert_affinities_masked,
        gate_up_proj_weight,
        gate_up_activations_T,
        down_proj_weight,
        down_activations,
        token_position_to_id.astype(jnp.int32),
        block_to_expert.astype(jnp.int32),
        grad_output,
        block_size=block_size,
    )

    print("gate grads::::", gate_up_proj_weight_grad)
    print("down_weight_grad grads::::", down_weight_grad)
    
    print("gate hidden_states_grad::::", hidden_states_grad)
    print("affinities_grad grads::::", affinities_grad)

    sliced_tensor = hidden_states_grad[:-1,:]
    hidden_states_grad = sliced_tensor.reshape(1, 1, -1, H)
    
    # sliced_tensor = affinities_grad.reshape(T,E)
    # affinities_grad = sliced_tensor[:T-1, :]

    affinities_grad = jnp.reshape(affinities_grad, (-1, E))
    affinities_grad = affinities_grad[:-1, :].reshape(1, 1, -1, E)

    print("gate hidden_states_grad::::", hidden_states_grad)
    print("affinities_grad grads::::", affinities_grad)

    gate_proj_weight_grad = gate_up_proj_weight_grad[:, :, 0, :]  # Shape: (E, H, I)
    up_proj_weight_grad = gate_up_proj_weight_grad[:, :, 1, :]

    # exit()


    return (
        hidden_states_grad,
        affinities_grad,
        gate_proj_weight_grad,
        up_proj_weight_grad,
        down_weight_grad,
        token_position_to_id,
        block_to_expert
    )

blockwise_mm.defvjp(_blockwise_mm_fwd, _blockwise_mm_bwd)