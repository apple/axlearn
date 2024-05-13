# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/flax:
# Copyright 2024 The Flax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/maxtext:
# Copyright 2024 The MaxText Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/aqt:
# Copyright 2024 The AQT Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""QuantizedDotGeneral Utilities. Hosts default quantization configuration.

"""
import functools

import jax
from aqt.jax.v2.config import DotGeneral, config_v3, set_stochastic_rounding
from jax import numpy as jnp


def lhs_activation_aqt_config() -> DotGeneral:
    """Default AQT config for when lhs is activation, rhs is weight.
        1. Sets int8 for fwd and dlhs. None for drhs.
        2. Sets int32 accumulation for fwd and dlhs. None for drhs.
        3. Sets stochastic rounding for lhs.

    See Also: https://github.com/google/maxtext/blob/
                718d9e796733faa9b49b018ddfad5061de46fcb2/MaxText/layers/quantizations.py#L95

    Returns:
        Default AQT config for when lhs is activation, rhs is weight.
    """
    return config_v3(
        fwd_bits=8,
        dlhs_bits=8,
        drhs_bits=None,
        rng_type="jax.uniform",
        dlhs_local_aqt=None,
        drhs_local_aqt=None,
        fwd_accumulator_dtype=jnp.int32,
        dlhs_accumulator_dtype=jnp.int32,
        drhs_accumulator_dtype=None,
    )


def rhs_activation_aqt_config() -> DotGeneral:
    """Default AQT config for when lhs is weight, rhs is activation.
        1. Sets int8 for fwd and drhs. None for dlhs.
        2. Sets int32 accumulation for fwd and drhs. None for dlhs.
        3. Sets stochastic rounding for rhs.

    This is necessary for when einsum swapped the original operands
    before calling dot_general.

    See Also: https://github.com/google/aqt/blob/
                4fb6e09f847ae4f3ea91351c54d63b8bdc69feda/aqt/jax/v2/flax/aqt_flax.py#L531

    Returns:
        Default AQT config for when lhs is weight, rhs is activation.
    """
    rng_type = "jax.uniform"
    config = config_v3(
        fwd_bits=8,
        dlhs_bits=None,
        drhs_bits=8,
        rng_type=rng_type,
        dlhs_local_aqt=None,
        drhs_local_aqt=None,
        fwd_accumulator_dtype=jnp.int32,
        dlhs_accumulator_dtype=None,
        drhs_accumulator_dtype=jnp.int32,
    )
    set_stochastic_rounding(
        config,
        vjp_lhs_stochastic_rounding=False,
        vjp_rhs_stochastic_rounding=True,
        implementation=rng_type,
    )
    return config


def is_einsum_swapped_operands(subscripts, /, *operands) -> bool:
    """Utility function checking if einsum swapped input operands
    before calling dot_general

    See Also: https://github.com/google/aqt/blob/
                4fb6e09f847ae4f3ea91351c54d63b8bdc69feda/aqt/jax/v2/flax/aqt_flax.py#L494

    Args:
        subscripts:     Specifies the subscripts for summation as comma separated
                        list of subscript labels.
                        An implicit (classical Einstein summation) calculation is
                        performed unless the explicit indicator ‘->’ is included
                        as well as subscript labels of the precise output form.
        *operands:      These are the arrays for the operation.

    Returns:
        Boolean, if True, einsum swaps operands before calling einsum
    """
    einsum = functools.partial(jnp.einsum, subscripts)
    if len(operands) != 2:
        raise ValueError(
            f"is_einsum_swapped_operands can only work on 2 operands, found {len(operands)}"
        )
    einsum_jaxpr = jax.make_jaxpr(einsum)(*operands)
    # Check if the input order to the first equation (dot_general)
    # matches the input order of the whole jaxpr.
    # If not, the operands were swapped
    not_swapped: bool = einsum_jaxpr.eqns[0].invars == einsum_jaxpr.jaxpr.invars
    swapped: bool = einsum_jaxpr.eqns[0].invars[::-1] == einsum_jaxpr.jaxpr.invars
    assert swapped ^ not_swapped
    return swapped
