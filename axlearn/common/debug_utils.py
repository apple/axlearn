"""Utilities for debugging training."""
import functools
import gc
from typing import Any, Callable, FrozenSet, Protocol

import chex
import jax
from absl import logging
from jax._src.pjit import pjit
from jax.experimental import checkify


class JitFn(Protocol):
    """A "pjit-like" function suitable for using in place of JAX's implementation of `pjit`."""

    def __call__(self, fun: Callable, *args, **kwargs) -> Callable:
        """Return a wrapped version of fun using pjit or similar.

        The arguments accepted are the same as JAX's `pjit`.
        """


# pylint: disable=redefined-outer-name,
# pylint: disable-next=protected-access
JaxException = jax._src.checkify.JaxException  # type: ignore[module-attr]


def checkify_pjit(errors: FrozenSet[JaxException], pjit: JitFn = pjit) -> JitFn:
    """Produce a checkified version of pjit.

    See the docstring of `checkify_and_rerun_on_nonfinite` for a usage example.

    Args:
        errors: The checkify checks to run.
        pjit: The pjit function to wrap.

    Returns:
        A checkified version of pjit.
    """

    def pjit_checkify_decorator(fun, *args, **kwargs):
        checkified_fun = checkify.checkify(pjit(fun, *args, **kwargs), errors=errors)

        def pjit_checkify_wrapper(*args, **kwargs):
            err, result = checkified_fun(*args, **kwargs)
            checkify.check_error(err)
            return result

        return pjit_checkify_wrapper

    return pjit_checkify_decorator


def checkify_and_rerun_for_float_errors(pjit: JitFn = pjit) -> JitFn:
    """Produce a pjit-like transformation that runs jax checkify float checks.

    Args:
        pjit: The pjit function to wrap.

    Returns:
        A checkified version of pjit.
    """
    return checkify_and_rerun_on_nonfinite(checkify.float_checks, pjit=pjit)


def checkify_and_rerun_on_nonfinite(
    errors: FrozenSet[JaxException], *, pjit: JitFn = pjit
) -> JitFn:
    """Produce a pjit-like transformation that detects if the output contains nonfinite values
    and if found, rerurns with additional error instrumentation.

    This is similar to `jax_debug_nans` but it works properly with jit and pjit.
    Despite claims on the jax documentation,there are cases where `jax_debug_nans` failed to locate
    the nans when using jit.

    Note: Unlike ordinary pjit, this prevents donating the arguments.

    Example:
        ```
        pjit_with_rerun = checkify_and_rerun_on_nonfinite(errors=checkify.float_checks)

        @pjit_with_rerun
        def fn(x,y):
            return x / y

        assert fn(8,2) == 4

        # Raises a JaxRuntimeError with the source of the division by 0.
        fn(3,0)
        ```

    Args:
        errors: The checkify error checks to enable when rerunning.

    Returns:
        A function suitable for `SpmdTrainer.Config.dynamic_rerun`.

    Raises:
        JaxRuntimeException: If a nonfinite value is found in the original run and the checkify
                             checks fail in the rerun.
    """

    def pjit_and_rerun_decorator(fun, *args, donate_argnums: Any = None, **kwargs):
        # donate_argnums cannot be used because we need to be able to rerun on the original inputs.
        if donate_argnums:
            logging.warning(
                "Ignoring donate_argnums=%s because it is incompatible with rerunning.",
                donate_argnums,
            )
        jit_fun = pjit(fun, *args, **kwargs)

        @functools.wraps(fun)
        def pjit_and_rerun_wrapper(*args, **kwargs):
            # Ensure leftover arrays from previous invocations are collected.
            gc.collect()
            # Run function first time.
            result = jit_fun(*args, **kwargs)
            try:
                chex.assert_tree_all_finite(result)
            except AssertionError as e:
                logging.error("Got nonfinite results from pjit function %s", e)
                checkified_fun = checkify.checkify(jit_fun, errors=errors)
                # Run function second time.
                err, result = checkified_fun(*args, **kwargs)
                checkify.check_error(err)
                # If no error raised from above line:
                logging.warning("Bad call failed to reproduce the issue when rerun. Continuing...")
            return result

        return pjit_and_rerun_wrapper

    return pjit_and_rerun_decorator


def noop_pjit() -> JitFn:
    """Produces a noop function that does not jit."""

    def no_pjit_decorator(fun, *args, **kwargs):
        del args, kwargs
        return fun

    return no_pjit_decorator


def checking_leaks_pjit(*, pjit: JitFn = pjit) -> JitFn:
    """Prdouces a pjit-like transformation with jax tracer leak detection."""

    def checking_leaks_pjit_decorator(fun, *args, **kwargs):
        jit_fun = pjit(fun, *args, **kwargs)

        @functools.wraps(jit_fun)
        def checking_leaks_pjit_wrapper(*args, **kwargs):
            with jax.checking_leaks():
                return jit_fun(*args, **kwargs)

        return checking_leaks_pjit_wrapper

    return checking_leaks_pjit_decorator
