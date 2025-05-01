from functools import partial

import jax
import jax.numpy as jnp
# TODO(apoorvtintin): remove pytype disable when dependencies are public.
# pytype: disable=import-error
# Import needed to enable JAX cache on Neuron.
import jax_neuronx  # pylint: disable=unused-import
import neuronxcc.nki.language as nl
from jax import custom_vjp

from neuronxcc.nki._private_kernels.blockwise_mm import (
# from axlearn.common.blockwise_mm import (
        # blockwise_mm as blockwise_mm_nki,
        blockwise_mm_selective_cp as blockwise_mm_nki,
        # blockwise_mm_baseline_shard_hidden as blockwise_mm_nki,
        #check_blockwise_mm_kernel_compatibility,
    )
from neuronxcc.nki._private_kernels.blockwise_mm_bwd import (
#from axlearn.common.blockwise_mm_bwd import (
    # blockwise_mm_bwd as blockwise_mm_bwd_nki,
    blockwise_mm_bwd_selective_cp as blockwise_mm_bwd_nki,
    #check_blockwise_mm_bwd_kernel_compatibility,
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
    '''
    try:
        check_blockwise_mm_kernel_compatibility(
            hidden_size=hidden_size,
            block_size=block_size,
            intermediate_size_tp=intermediate_size_tp,
        )
    except AssertionError as e:
        print(f"Blockwise kernel not compatible with model config. Reason: {str(e)}")
        return False
    '''
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
    with jax.named_scope("take_out_OG"):
        hidden_states = jnp.squeeze(hidden_states, axis=(0,1,))
        expert_affinities_masked = jnp.squeeze(expert_affinities_masked, axis=(0,1,))
        token_position_to_id = jnp.squeeze(token_position_to_id, axis=(0,1,))
        block_to_expert = jnp.squeeze(block_to_expert, axis=(0,1,))

    # # add +1 for padding
    with jax.named_scope("add padding"):
        padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
        padding_e = jnp.zeros((1,expert_affinities_masked.shape[1]), dtype=expert_affinities_masked.dtype)
        # # (S+1, H)
        hidden_states = jnp.concat([hidden_states, padding_h], axis=0)
        expert_affinities_masked = jnp.concat([expert_affinities_masked, padding_e], axis=0)
        expert_affinities_masked = jnp.reshape(expert_affinities_masked, (-1, 1))

    with jax.named_scope("setupweight"):
        gate_up_weight = jnp.stack([gate_weight, up_proj_weight], 
            axis=2
        )

    print(
        hidden_states,
        expert_affinities_masked,
        gate_up_weight,
        down_proj_weight,
        token_position_to_id,
        block_to_expert,
        block_size
    )
    with jax.named_scope("make NKI call"):
        out, gate_up_activations_T, down_activations = _blockwise_mm_nki_call[VNC(2)](
            hidden_states,
            expert_affinities_masked,
            gate_up_weight,
            down_proj_weight,
            token_position_to_id,
            block_to_expert,
            block_size=block_size,
        )

    return out[None, None, None, :-1, :], (hidden_states, expert_affinities_masked, gate_up_weight, 
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

    grad_output =  jnp.squeeze(grad_output, axis=(0,1,2))
    padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
    grad_output = jnp.concat([grad_output, padding_h], axis=0)
    
    # jax.debug.print("grad_output: {x}", x=grad_output)
    # jax.debug.print("hidden_states: {x}", x=hidden_states)
    # breakpoint()
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

    # jax.debug.print("hidden_states_grad: {x}", x=hidden_states_grad)
    # jax.debug.print("affinities_grad: {x}", x=affinities_grad)
    # jax.debug.print("gate_up_proj_weight_grad: {x}", x=gate_up_proj_weight_grad)
    # jax.debug.print("down_weight_grad: {x}", x=down_weight_grad)
    
    sliced_tensor = hidden_states_grad[:-1,:]
    hidden_states_grad = sliced_tensor.reshape(1, 1, -1, H)
    
    # sliced_tensor = affinities_grad.reshape(T,E)
    # affinities_grad = sliced_tensor[:T-1, :]

    affinities_grad = jnp.reshape(affinities_grad, (-1, E))
    affinities_grad = affinities_grad[:-1, :].reshape(1, 1, -1, E)

    gate_proj_weight_grad = gate_up_proj_weight_grad[:, :, 0, :]  # Shape: (E, H, I)
    up_proj_weight_grad = gate_up_proj_weight_grad[:, :, 1, :]

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
