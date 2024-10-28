# Copyright © 2023 Apple Inc.

"""A custom op that provides an optimization barrier."""

from typing import Any

import jax
from jax._src import ad_checkpoint  # pylint: disable=protected-access


@jax.custom_jvp
@jax.custom_batching.custom_vmap  # Must be wrapped in this before custom_jvp.
# PyTrees are defined by whether they are registered, not based on their type.
def forward_optimization_barrier(pytree: Any) -> Any:
    """Returns `pytree` after transparently wrapping the computation in an XLA optimization barrier.

    If the output of this function is transformed by autodifferentiation,
    only the forward pass will be wrapped in a barrier. This is because putting a barrier
    on the derivative computation would force explicit Jacobian computation, which is not
    computationally tractable.

    Example:
        ```
        # Suppose we are writing a function where we multiply
        # a very small input `x` by a very large constant `a`
        import jax
        import jax.numpy as jnp
        from axlearn.common import ops
        from axlearn.common.utils import Tensor

        a = jnp.array(2**8, dtype=jnp.float16)
        x = jnp.array(2**-8, dtype=jnp.float16)


        @jax.jit
        @jax.value_and_grad
        def f_without_barrier(x):
            return (a * x) ** 2


        @jax.jit
        @jax.value_and_grad
        def f_with_barrier(x):
            return ops.forward_optimization_barrier(a * x) ** 2


        @jax.jit
        @jax.value_and_grad
        def g_without_barrier(x):
            return a * (a * x**2)


        @jax.jit
        @jax.value_and_grad
        def g_with_barrier(x):
            return a * ops.forward_optimization_barrier(a * x**2)


        def print_result(msg: str, scalars: Tensor):
            scalars = jax.tree.map(lambda x: x.item(), scalars)
            print(msg, scalars)


        print("f:")
        print("(value, grad):")
        print_result("  without barrier:", f_without_barrier(x))
        print_result("  with barrier:", f_with_barrier(x))
        print_result("  if there were a barrier on grad:", ((a * x) ** 2, a * 2 * (a * x)))
        print()

        print("g:")
        print("(value, grad):")
        print_result("  without barrier:", g_without_barrier(x))
        print_result("  with barrier:", g_with_barrier(x))
        print_result("  if there were a barrier on grad:", (a * (a * x**2), a * 2 * (a * x)))

        # Output:
        # f:
        # (value, grad):
        #   without barrier: (inf, inf)
        #   with barrier: (1.0, 512.0)
        #   if there were a barrier on grad: (1.0, 512.0)
        #
        # g:
        # (value, grad):
        #   without barrier: (inf, inf)
        #   with barrier: (1.0, inf)
        #   if there were a barrier on grad: (1.0, 512.0)
        ```

    Args:
        pytree: The pytree to wrap.

    Returns:
        `pytree` transparently wrapped in an XLA optimization barrier.
    """
    return ad_checkpoint._optimization_barrier(pytree)  # pylint: disable=protected-access


@forward_optimization_barrier.defjvp
def forward_optimization_barrier_jvp(primals: tuple, tangents: tuple) -> tuple[Any, Any]:
    """The JVP for `optimization_barrier`.

    Args:
        primals: The JVP primals. A tuple with the same structure as the input arguments of the
                 original function. Contains the values of each input the original function was
                 called with in the forward pass.
        tangents: The JVP tangents. A tuple with the same structure as the input arguments of the
                  original function. Contains a direction to compute the directional derivative
                  with respect to.

    Returns:
        `(primal_out, tangent_out)` which are respectively the output of the original function and
        its derivative in the direction given by `tangents`.
    """
    (primal_out,) = primals
    primal_out = forward_optimization_barrier(primal_out)
    (tangent_out,) = tangents
    return primal_out, tangent_out


@forward_optimization_barrier.def_vmap
def forward_optimization_barrier_vmap(
    # pylint: disable-next=unused-argument
    batch_axis_size: int,
    in_batched: tuple,
    pytree: Any,
) -> tuple[Any, Any]:
    """VMAP rule for`optimization_barrier`.

    Args:
        pytree: The arguments to evaluate the vmapped function with. These have batch axes already.
        batch_axis_size: The size of the batch axis being vmapped over.
        in_batched: For each input, whether that input has a batch dimension at index 0.
                    Has the same pytree structure as a tuple of the arguments of the original
                    function.

    Returns:
        `(out, out_batched)` which are respectively the result of mapping the function over
        the batch axes and which axes are batched. `out_batched` has the same pytree structure
        as `out`.
    """
    (out_batched,) = in_batched
    return forward_optimization_barrier(pytree), out_batched
