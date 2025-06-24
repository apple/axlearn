# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""A generic repeat layer.

Adapted from:
https://github.com/tensorflow/lingvo/blob/2d46faf8/lingvo/core/repeat_layer.py
https://github.com/tensorflow/lingvo/blob/2d46faf8/lingvo/jax/layers/repeats.py

A repeat layer consists a stack of N identical sub layers, where
  * The variables are stacked across layers. Each stacked variable has shape [N, ...].
  * The computation is performed with a recurrent loop across layers.

Compared with a layer stack, a repeat layer's XLA code size does not grow
proportional to the number of layers. It also reduces HBM usage but incurs
additional computation through rematerialization.

Repeat._run() allows its subclasses to describe arbitrary
computation across sub layers.

Inputs to repeat layer computation fall into two categories:

  * carry: iterative input to the first sub layer, e.g., hidden vectors.
  * xs: separate inputs for each sub layer, specified by tensors of shape [N, ...],
      where T[i, ...] is the input for sub layer i, e.g., states for auto-regressive inference.

The output of a sub layer can include:

  * carry: iterative input for the next sub layer.
  * ys: layer-wise outputs, to be stacked for the final output, e.g.,
      updated_states from auto-regressive inference.

The final output of a repeat layer will include:

  * carry: the iterative output of the final sub layer.
  * ys: stacked tensors of layer-wise outputs, of shape [N, ...],
      where T[i, ...] is a layer-wise output of sub layer i.

In pseudo code:

  def _run(theta, fn, carry, xs):
      for i in range(p.num_layers):
          carry, ys[i, ...] = fn(carry, xs[i, ...])
      return carry, ys
"""
import dataclasses
from typing import Callable, NamedTuple, Optional, Sequence, Union

import jax

from axlearn.common import param_init
from axlearn.common.base_layer import (
    BaseLayer,
    FactorizationSpec,
    NestedParameterSpec,
    ParameterSpec,
    PartitionSpec,
)
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)
from axlearn.common.module import Module, child_context, new_output_collection, scan_in_context
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    Tensor,
    VDict,
    get_or_none,
    match_regex_rules,
    split_prng_key,
)


def _drop_by_regex(rules: Sequence[str]) -> Callable[[str], bool]:
    """Returns a drop that regex-matches inputs against `rules`."""
    return lambda x: match_regex_rules(x, rules=[(rule, True) for rule in rules])


class Repeat(BaseLayer):
    """A layer which repeats a sub layer sequentially using a jax.lax.scan loop."""

    @config_class
    class Config(BaseLayer.Config):
        """Config class for the Repeat layer."""

        # The config for the sub layer.
        layer: Required[InstantiableConfig] = REQUIRED
        # Repeat layers specified in `layer` this many times.
        num_layers: Required[int] = REQUIRED
        # A callable that drops outputs from the layer's output_collection based on path. By
        # default, we drop all module outputs. See `scan_in_context` for details.
        # TODO(markblee): Converge on dropping no outputs by default.
        drop_output: InstantiableConfig[Callable[[str], bool]] = config_for_function(
            _drop_by_regex
        ).set(rules=["module_outputs.*"])
        # An optional positive integer or boolean argument for `jax.lax.scan`.
        # If a positive integer is provided, it determines how many unrolled loop iterations to run
        # within a single rolled iteration of the loop. If a boolean is provided, it will determine
        # if the loop is competely unrolled or left completely rolled.
        # If None, defaults to 1 (same as jax.lax.scan's default value).
        unroll: Optional[Union[bool, int]] = None
        # remat: Whether to apply `jax.checkpoint` to the scanned function for memory-efficient
        # training. If True, wraps the scan body function with `jax.checkpoint` to trade
        # increased compute for reduced memory usage by recomputing intermediate activations
        # during backward pass.
        # The default is None for backward compatibility, but it’s recommended to set it to True.
        remat_in_scan: Optional[bool] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("layer", self._layer_config())

        self._drop_output = None
        if cfg.drop_output is not None:
            self._drop_output = cfg.drop_output.instantiate()

    def _layer_config(self):
        return self.config.layer

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        cfg = self.config
        specs = VDict(**super().create_parameter_specs_recursively())

        def transform_factorization_spec(
            spec: Optional[FactorizationSpec],
        ) -> Optional[FactorizationSpec]:
            if spec is None:
                return None
            return FactorizationSpec(axes=[None] + list(spec.axes))

        return jax.tree.map(
            lambda spec: dataclasses.replace(
                spec,
                shape=(cfg.num_layers, *spec.shape),
                mesh_axes=PartitionSpec(None, *spec.mesh_axes),
                factorization=transform_factorization_spec(spec.factorization),
                fan_axes=param_init.maybe_prepend_axis(
                    spec.fan_axes, axis_type=param_init.FanAxes.AxisType.BATCH_AXIS
                ),
            ),
            specs,
        )

    def initialize_parameters_recursively(
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
    ) -> NestedTensor:
        def init(prng_key_i):
            return VDict(
                layer=self.layer.initialize_parameters_recursively(
                    # `prebuilt` must be consistent across all layers.
                    prng_key_i,
                    prebuilt=get_or_none(prebuilt, "layer"),
                )
            )

        cfg = self.config
        return jax.vmap(init)(split_prng_key(prng_key, cfg.num_layers).keys)

    class Output(NamedTuple):
        carry: NestedTensor
        ys: NestedTensor

    def _run(self, fn, carry=None, *, xs=None):
        """Invokes 'fn' for each sub-layer.

        Note, the number of sub-layers used for the computation might be smaller than
        `cfg.num_layers` depending on the invocation context.

        Args:
            fn: A function with args (carry, x) returning a dict(carry=..., y=...).
                `fn` will be run in the context of `self.layer`.
            carry: a nested tensor for the iterative input of the 0'th sub-layer.
            xs: a nested tensor with separate inputs for each sub-layer,
                where each leaf value T is a tensor of shape [num_layers, ...]
                and T[i, ...] represents layer-wise inputs to the i'th sub-layer.

        Returns:
            A dict with the following keys:
            - carry: a nested tensor with the same structure as iterative_input_0
                representing the iterative output of the last sub-layer.
            - ys: a nested tensor where each leaf value T is a tensor of shape [num_layers, ...]
                and T[i, ...] represents layer-wise output from the i'th sub-layer.
        """
        cfg = self.config
        prng_key = self.prng_key

        if carry is None:
            carry = {}
        if xs is None:
            xs = {}

        layer_output_collection = new_output_collection()
        with child_context("layer", output_collection=layer_output_collection) as layer_context:
            # Note, actual `num_layers` might be smaller than `cfg.num_layers` depending on
            # the invocation context.
            num_layers = jax.tree_util.tree_reduce(
                lambda num, x: min(num, x.shape[0]),
                tree=(layer_context.state, xs),
                initializer=cfg.num_layers,
            )

            if cfg.remat_in_scan:
                remat_kwargs = dict(
                    prevent_cse=False,
                    policy=maybe_instantiate(cfg.remat_spec.policy)
                    if cfg.remat_spec is not None
                    else None,
                )
            else:
                remat_kwargs = None
            carry, ys = scan_in_context(
                fn,
                carry=carry,
                xs=dict(
                    xs=xs,
                    prng_key=split_prng_key(prng_key, num_layers).keys,
                    state=layer_context.state,
                ),
                drop_output=self._drop_output,
                child_name_prefix="layer",
                unroll=cfg.unroll if cfg.unroll is not None else 1,
                remat_kwargs=remat_kwargs,
            )

        self.get_invocation_context().output_collection.update(layer_output_collection)
        return self.Output(carry=carry, ys=ys)
