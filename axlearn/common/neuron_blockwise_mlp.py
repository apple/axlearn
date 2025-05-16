from functools import partial

import jax
import jax.numpy as jnp
# TODO(apoorvtintin): remove pytype disable when dependencies are public.
# pytype: disable=import-error
# Import needed to enable JAX cache on Neuron.
import jax_neuronx  # pylint: disable=unused-import
import neuronxcc.nki.language as nl
from jax import custom_vjp
from jax._src.mesh import thread_resources
from neuronxcc.nki._private_kernels.blockwise_mm import SkipMode
from neuronxcc.nki._private_kernels.blockwise_mm import (
        blockwise_mm_selective_cp as blockwise_mm_nki,
        check_blockwise_mm_kernel_compatibility,
    )
from neuronxcc.nki._private_kernels.blockwise_mm_bwd import (
    blockwise_mm_bwd_selective_cp as blockwise_mm_bwd_nki,
    # check_blockwise_mm_bwd_kernel_compatibility,
)
from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC
import neuronxcc.nki as nki

_blockwise_mm_nki_call = nki.jit(show_compiler_tb=True)(blockwise_mm_nki)
_blockwise_mm_bwd_nki_call = nki.jit(show_compiler_tb=True)(blockwise_mm_bwd_nki)


Tensor = jax.Array
lnc = 2 if jax.devices()[0].device_kind == "NC_v3d" else 1

def _backend():
    # For compatibility with AOT compilation, we obtain the backend type from physical_mesh.
    global_mesh = thread_resources.env.physical_mesh
    if len(global_mesh.devices):
        backend = global_mesh.devices.flat[0].platform
    else:
        # Fall back to jax.default_backend() if no device is found in physical_mesh.
        backend = jax.default_backend()
    return backend

def can_use_blockwise_matmul_nki(
    hidden_size,
    intermediate_size_tp,
    block_size,
    glu_mlp,
):
    if _backend() != "neuron":
        return False

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


@partial(custom_vjp, nondiff_argnums=(7,))
def blockwise_mm(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    block_size: int,
):
    out, _ = _blockwise_mm_fwd(hidden_states, expert_affinities_masked, gate_weight, up_proj_weight,
                            down_proj_weight, token_position_to_id, block_to_expert, block_size)
    return out

def _blockwise_mm_fwd(
    hidden_states: Tensor,
    expert_affinities_masked: Tensor,
    gate_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    block_size: int, 
):
    # Remove O, G dimensions
    with jax.named_scope("take_out_OG"):
        hidden_states = jnp.squeeze(hidden_states, axis=(0,1,))
        expert_affinities_masked = jnp.squeeze(expert_affinities_masked, axis=(0,1,))
        token_position_to_id = jnp.squeeze(token_position_to_id, axis=(0,1,))
        block_to_expert = jnp.squeeze(block_to_expert, axis=(0,1,))

    # # add +1 for padding
    # with jax.named_scope("add padding"):
    #     padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
    #     padding_e = jnp.zeros((1,expert_affinities_masked.shape[1]), dtype=expert_affinities_masked.dtype)
    #     # (S+1, H)
    #     hidden_states = jnp.concat([hidden_states, padding_h], axis=0)
    #     expert_affinities_masked = jnp.concat([expert_affinities_masked, padding_e], axis=0)
    expert_affinities_masked = jnp.reshape(expert_affinities_masked, (-1, 1))

    with jax.named_scope("setupweight"):
        gate_up_weight = jnp.stack([gate_weight, up_proj_weight], 
            axis=2
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
            skip_dma=SkipMode(True, False)
        )

    return out[None, None, None, :, :], (hidden_states, expert_affinities_masked, gate_weight, up_proj_weight, 
                down_proj_weight, down_activations, gate_up_activations_T, 
                token_position_to_id, block_to_expert)

def _blockwise_mm_bwd(
    block_size,
    res,
    grad_output
):
    (hidden_states, expert_affinities_masked, gate_weight, up_proj_weight,
     down_proj_weight, down_activations, gate_up_activations_T, 
     token_position_to_id, block_to_expert) = res

    gate_up_proj_weight = jnp.stack([gate_weight, up_proj_weight], 
        axis=2
    )
    
    T,H = hidden_states.shape
    E, _, _, _ = gate_up_proj_weight.shape

    grad_output =  jnp.squeeze(grad_output, axis=(0,1,2))
    padding_h = jnp.zeros((1, hidden_states.shape[1]), dtype=hidden_states.dtype)
    grad_output = jnp.concat([grad_output, padding_h], axis=0)
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
        skip_dma=SkipMode(True, False),
    )
    
    sliced_tensor = hidden_states_grad[:,:]
    hidden_states_grad = sliced_tensor.reshape(1, 1, -1, H)
    
    affinities_grad = jnp.reshape(affinities_grad, (-1, E))
    affinities_grad = affinities_grad[:, :].reshape(1, 1, -1, E)

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
