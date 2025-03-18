# Copyright © 2024 Apple Inc.
"""This module provides functions to decorate a ForwardFn to allow for a minibatched
version that enables gradient accumulation.
"""
import functools
from typing import Any, Callable, Optional

import jax
import numpy as np
from jax import numpy as jnp

from axlearn.common import utils
from axlearn.common.config import ConfigOr, maybe_instantiate
from axlearn.common.metrics import MetricAccumulator
from axlearn.common.update_transformation import ForwardFn, ForwardOutputs
from axlearn.common.utils import Nested, Tensor


def _compute_minibatch_size(input_batch: Nested[Tensor], *, steps: int) -> int:
    """Utility function to compute minibatch size from input batch.

    Args:
        input_batch: A ForwardFn input_batch.
        steps: Integer steps to divide inputs with.

    Returns:
        An integer representing minibatch size.

    Raises:
        ValueError if the input batch is not divisible by steps, input batch
        is empty or otherwise contain ambiguous (heterogeneus) sizes.
    """
    if steps <= 0:
        raise ValueError("Accumulation steps need to be a positive integer.")

    input_batch_sizes = jax.tree_leaves(jax.tree.map(lambda x: x.shape[0], input_batch))

    if len(input_batch_sizes) == 0:
        raise ValueError("Input batch is empty.")

    if not (np.array(input_batch_sizes) == input_batch_sizes[0]).all():
        raise ValueError(f"Ambiguous input batch sizes: {input_batch_sizes}")

    input_batch_size = input_batch_sizes[0]
    if input_batch_size % steps != 0:
        raise ValueError(
            f"Input batch size {input_batch_size} "
            f"must be divisible by number of steps: {steps}."
        )
    return input_batch_size // steps


def _make_scan_minibatch_inputs(
    inputs: Nested[Tensor],
    *,
    forward_key: Tensor,
    param_noise_key: Tensor,
    minibatch_index: int,
) -> tuple[Nested[Tensor], Tensor, Tensor]:
    """Creates minibatch inputs from inputs.

    This is a utility function that is only meant to be called from
    within a scan function body and is meant to return sliced minibatches
    to run the ForwardFn on.

    Args:
        inputs: Same pytree as ForwardFn inputs.
        forward_key: The `forward_key` from the ForwardFn inputs
        param_noise_key: The `param_noise_key` from the ForwardFn inputs
        minibatch_index: Current scan minibatch index.

    Returns:
        A tuple of minibatch inputs which of the same structure as `inputs`
        and new (carry) forward_key and param_noise_key.
    """
    minibatch_input = jax.tree.map(
        lambda x: x[minibatch_index],
        inputs["input_batch"],
    )

    next_forward_key, forward_key = jax.random.split(forward_key)
    next_param_noise_key, param_noise_key = jax.random.split(param_noise_key)

    minibatch_inputs = dict(
        input_batch=minibatch_input,
        forward_key=forward_key,
        param_noise_key=param_noise_key,
    )
    return minibatch_inputs, next_forward_key, next_param_noise_key


def with_minibatch_steps(
    *,
    steps: int,
    metric_accumulator: ConfigOr[MetricAccumulator],
    grad_dtype: Optional[jnp.dtype] = None,
) -> Callable[[ForwardFn], ForwardFn]:
    """Decorate a ForwardFn to accumulate gradients over minibatch steps.

    At the core lies a jax.lax.scan implementation where we slice the input batch B into b
    sized chunks where b = B // steps and scan over the ForwardFn for number of `steps` times
    and compute the minibatch gradients. These partial gradients are then accumulated and
    averaged to get the final gradient of the batch B.

    The decorated function has the same semantics as the original function but internally
    computes its forward and backward pass using a sequential gradient accumulation
    described in more detail below. We assume that the original function's loss is a mean of
    the loss over each batch element. If this assumption is not true (e.g. maskes loss with
    unequal loss weights) then the decorated function will not be equivalent to the original(*).

    TODO(cemkoc): Extend the gradient accumulation function from average to more.

    To make it memory efficient we implement a jax custom vjp rule for the forward (primal)
    function where we compute the minibatch gradients as well as primal output in the forward
    pass using a jax.lax.scan and pass this gradient to backward pass to be used as is. Since
    we compute the gradients using minibatches in the forward pass and pass that into the backward
    pass we no longer need to compute it again which is what enables memory efficiency.

    (*): Empirically we see that loss curves can still be slightly mismatched even if the above
    assumption about the loss function holds true and the input dataset is a deterministically
    batched dataset. We are not sure what causes this slight mismatch and will investigate it more.

    TODO(cemkoc): Investigate the slight difference in loss curves when decorated.

    Outputs of the decorated ForwardFn are accumulated based on the provided metric_accumulator.
    The accumulated outputs of the decorated ForwardFn are the same as ForwardFn only if the
    specific output does not rely on batch size of the input. For example if a summary output
    of a ForwardFn is of the shape [batch_size, ] with value [1,2,3,4] where batch_size is 4,
    after decoration with minibatch_size of 2 the metric output will be of shape
    [minibatch_size, ] with value [1+3, 2+4] instead.

    Args:
        steps: Number of gradient accumulation steps.
        metric_accumulator: A `MetricAccumulator` to accumulate minibatch summaries from the
            forward output.
        grad_dtype: Optional dtype to cast the grads back to after accumulating in fp32.

    Returns:
        Decorated ForwardFn.
    """

    def decorator(fn: ForwardFn) -> ForwardFn:
        # We define a positional arg only version of the original function
        # that is passed because jax.value_and_grad does not accept
        # keyword-only arguments. This is used when computing the gradient in
        # the f_fwd function below.
        def original_func_positional_args(
            model_params: Nested[Tensor], inputs: Any
        ) -> tuple[Tensor, ForwardOutputs]:
            output = fn(model_params=model_params, inputs=inputs)
            return output.loss, output

        def fwd_helper(
            model_params: Nested[Tensor], inputs: Any, compute_grad: bool
        ) -> tuple[ForwardOutputs, Optional[Nested[Tensor]]]:
            """Helper function that scans a ForwardFn over minibatches.

            Args:
                model_params: Model parameters for the ForwardFn
                inputs: Inputs for the ForwardFn.
                compute_grad: Whether to compute and return grads as
                    part of the scanned forward pass.

            Returns:
                A tuple where first element is the accumulated `ForwardOutputs`
                and second is the accumulated grads (if `compute_grad` is True)
                otherwise None.
            """
            minibatch_size = _compute_minibatch_size(inputs["input_batch"], steps=steps)

            def reshape_for_scan(x: Tensor):
                """Helper function that adds a minibatch dimension while evenly dividing
                batches across gradient accumulation iterations.

                Input dimension is [Global logical Batch Size, Sequence], this first reshaped to
                [Minibatch Size, Steps, Sequence],
                then transposed to [steps, Minibatch Size, Sequence] this ensures that
                batches picked up from the global batch in a staggered pattern.

                The main benefit is that this avoids extra communication incurred in reshard
                for every minibatch.

                Args:
                    x: Tensor to be reshaped.

                Returns:
                    The reshaped tensor.
                """
                if x.shape[0] % minibatch_size != 0:
                    raise ValueError(
                        f"minibatch_size {minibatch_size} does not evenly divide "
                        f"global batch size of {x.shape[0]}"
                    )

                x = x.reshape(minibatch_size, -1, *x.shape[1:])
                return jnp.swapaxes(x, 0, 1)

            inputs["input_batch"] = jax.tree_map(reshape_for_scan, inputs["input_batch"])

            # Create a sample minibatch for the carry buffer creation below
            (
                sample_minibatch_inputs,
                _,
                _,
            ) = _make_scan_minibatch_inputs(
                inputs,
                forward_key=inputs["forward_key"],
                param_noise_key=inputs["param_noise_key"],
                minibatch_index=0,
            )

            # Carry initialization for the lax.scan procedure. Since we are passing a
            # `MetricAccumulator` into carry and carry input/output shapes must match
            # we need initialize the `MetricAccumulator` summary with the right PyTree
            # structure.
            _, primal_output_shape = jax.eval_shape(
                original_func_positional_args, model_params, sample_minibatch_inputs
            )
            init_primal_out = jax.tree.map(jnp.zeros_like, primal_output_shape)
            init_accumulator = maybe_instantiate(metric_accumulator)
            init_accumulator.update(init_primal_out.output_collection.summaries)
            # Init carry here with: primal_output, grads (optional), prngkeys, metric_accumulator.
            if compute_grad:
                # Accumulate gradients with fp32.
                init_grads = jax.tree.map(lambda x: jnp.zeros(x.shape, jnp.float32), model_params)
            else:
                init_grads = None

            carry = (
                init_primal_out,
                init_grads,
                inputs["forward_key"],
                inputs["param_noise_key"],
                init_accumulator,
            )

            def scan_body(
                carry: tuple[Nested[Tensor], Nested[Tensor], Tensor, Tensor, MetricAccumulator],
                minibatch_index: int,
            ):
                """Computes minibatch forward outputs and, optionally, gradients."""
                primal_out, grads, forward_key, param_noise_key, accumulator = carry
                (
                    minibatch_inputs,
                    next_forward_key,
                    next_param_noise_key,
                ) = _make_scan_minibatch_inputs(
                    inputs,
                    forward_key=forward_key,
                    param_noise_key=param_noise_key,
                    minibatch_index=minibatch_index,
                )
                minibatch_args = (model_params, minibatch_inputs)

                if compute_grad:
                    # Compute the minibatch primal output and gradients
                    (_, primal_out_minibatch), grads_minibatch = jax.value_and_grad(
                        original_func_positional_args, has_aux=True
                    )(
                        *minibatch_args,
                    )
                    grads = jax.tree.map(jnp.add, grads, grads_minibatch)
                else:
                    _, primal_out_minibatch = original_func_positional_args(
                        *minibatch_args,
                    )

                # Update the metric accumulator with minibatch output summaries
                accumulator.update(primal_out_minibatch.output_collection.summaries)
                # Accumulate the primal output and grads and pass them into carry.
                primal_out = jax.tree.map(jnp.add, primal_out, primal_out_minibatch)
                return (
                    primal_out,
                    grads,
                    next_forward_key,
                    next_param_noise_key,
                    accumulator,
                ), None

            (primal_out, grads, _, _, accumulator), _ = jax.lax.scan(
                scan_body,
                init=carry,
                xs=jnp.arange(steps),
            )
            # Since we summed during accumulation we take the average here to rescale.
            grads = jax.tree.map(lambda x: x / steps, grads)
            primal_out = jax.tree.map(lambda x: x / steps, primal_out)
            primal_out.output_collection.summaries.update(accumulator.summaries())
            return primal_out, grads

        def sequential_vmap(func: ForwardFn) -> ForwardFn:
            """Decorates a ForwardFn to process the input in minibatches."""

            @functools.wraps(func)
            def wrapper(model_params: Nested[Tensor], inputs: Any) -> ForwardOutputs:
                forward_outputs, _ = fwd_helper(
                    model_params=model_params, inputs=inputs, compute_grad=False
                )
                return forward_outputs

            return wrapper

        @jax.custom_vjp
        @sequential_vmap
        # We define a positional arg only version to decorate with custom vjp
        # because custom vjp also does not support keyword-only arguments.
        def func(model_params: Nested[Tensor], inputs: Any) -> ForwardOutputs:
            """Wrap original function to pass in key-word args."""
            return fn(model_params=model_params, inputs=inputs)

        def func_fwd(model_params: Nested[Tensor], inputs: Any) -> tuple[ForwardOutputs, tuple]:
            """Defines forward pass for the custom vjp based gradient computation."""
            args_ = (model_params, inputs)
            forward_outputs, grads = fwd_helper(
                model_params=model_params, inputs=inputs, compute_grad=True
            )
            # Cast grads to grad_dtype (if specified).
            grads = utils.cast_floats(grads, to_dtype=grad_dtype)
            saved_fwd_state = grads, len(args_)
            return forward_outputs, saved_fwd_state

        def func_bwd(saved_fwd_state, grad_from_later_in_network) -> tuple[Nested[Tensor], None]:
            """Defines backward pass for the custom vjp based gradient computation."""
            grad_from_earlier, num_args = saved_fwd_state
            # Compute the backward pass gradient value.
            grad = jax.tree_map(lambda x: x * grad_from_later_in_network.loss, grad_from_earlier)
            # Return gradient along with None so the output length equals to that of primal input.
            return (grad,) + (None,) * (num_args - 1)

        func.defvjp(func_fwd, func_bwd)
        return func

    return decorator
