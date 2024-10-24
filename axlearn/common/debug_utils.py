"""Utilities for debugging training."""

from typing import Any, Callable, Optional, Protocol, Union

import jax
from jax._src.pjit import pjit
from jax.experimental import checkify


class _CheckifyCompiledFnWrapper:
    """Performs error handling on a "checkified" compiled function during the call."""

    def __init__(self, compiled: jax.stages.Compiled):
        self._compiled = compiled

    def __call__(self, *args, **kwargs) -> Any:
        """Calls the compiled function and raises on detected checkify error."""
        err, result = self._compiled(*args, **kwargs)
        checkify.check_error(err)
        return result


class _CheckifyLoweredFnWrapper:
    """Wraps a lowered checkified function."""

    def __init__(
        self,
        lowered: jax.stages.Lowered,
    ):
        self._lowered = lowered

    def compile(
        self, compiler_options: Optional[dict[str, Union[str, bool]]] = None
    ) -> _CheckifyCompiledFnWrapper:
        """Compile the function with provided options."""
        compiled = self._lowered.compile(compiler_options=compiler_options)
        return _CheckifyCompiledFnWrapper(compiled)


class CheckifyJitFnWrapper:
    """A checkify wrapper for use in place of JAX's implementation of `pjit` in some cases."""

    def __init__(
        self,
        checkified_jit_handle: jax.stages.Wrapped,
    ):
        self._checkified_jit_handle = checkified_jit_handle

    def __call__(
        self, *args, compiler_options: Optional[dict[str, Union[str, bool]]] = None, **kwargs
    ) -> Any:
        """Lowers, compiles, and runs the function with provided arguments and keyword arguments."""
        lowered = self.lower(*args, **kwargs)
        compiled = lowered.compile(compiler_options)
        return compiled(*args, **kwargs)

    def lower(self, *args, **kwargs) -> _CheckifyLoweredFnWrapper:
        """Traces and lowers the function using the provided arguments."""
        lowered = self._checkified_jit_handle.lower(*args, **kwargs)
        return _CheckifyLoweredFnWrapper(lowered)


class CheckifyJitFn(Protocol):
    """Mirrors the call signature of JAX's `pjit` definition."""

    def __call__(self, fun: Callable, *args, **kwargs) -> CheckifyJitFnWrapper:
        """Return a jit-fn wrapped version of `fun`.

        The arguments accepted are the same as JAX's `pjit`.
        """


# pylint: disable=redefined-outer-name,
# pylint: disable-next=protected-access
JaxException = jax._src.checkify.JaxException  # type: ignore[module-attr]


def checkify_pjit(errors: frozenset[JaxException]) -> CheckifyJitFn:
    """Produce a checkified version of pjit.

    Example:
    ```py
    pjit_with_nan_check = checkify_pjit(errors=checkify.nan_checks)

    @pjit_with_nan_check
    def fn(x):
        return jnp.log(x)

    assert fn(jnp.exp(1)) == 1.0

    # Raises a JaxRuntimeError with the source of the undefined logarithm call.
    fn(-1)
    ```

    Args:
        errors: The checkify checks to run.

    Returns:
        A checkified version of pjit.
    """

    def pjit_checkify_decorator(
        fun, *args, out_shardings: Any = None, **kwargs
    ) -> CheckifyJitFnWrapper:
        # We need to update out_shardings to handle the checkify result.
        out_shardings = (None, out_shardings)
        checkified_fun = CheckifyJitFnWrapper(
            pjit(
                checkify.checkify(fun, errors=errors), *args, out_shardings=out_shardings, **kwargs
            ),
        )
        return checkified_fun

    return pjit_checkify_decorator
