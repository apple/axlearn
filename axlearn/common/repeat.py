"""A generic repeat layer.

Adapted from:
https://github.com/tensorflow/lingvo/blob/master/lingvo/core/repeat_layer.py
https://github.com/tensorflow/lingvo/blob/master/lingvo/jax/layers/repeats.py

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
from typing import NamedTuple, Optional

import jax

from axlearn.common.base_layer import (
    BaseLayer,
    FactorizationSpec,
    NestedParameterSpec,
    PartitionSpec,
)
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.module import (
    Module,
    NestedTensor,
    child_context,
    current_context,
    new_output_collection,
)
from axlearn.common.utils import VDict, get_or_none, split_prng_key


class Repeat(BaseLayer):
    """A layer which repeats a sub layer sequentially using a jax.lax.scan loop."""

    @config_class
    class Config(BaseLayer.Config):
        layer: Required[InstantiableConfig] = REQUIRED  # The config for the sub layer.
        num_layers: Required[int] = REQUIRED  # Repeat layers specified in `layer` this many times.

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._add_child("layer", self._layer_config())

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

        return jax.tree_util.tree_map(
            lambda spec: dataclasses.replace(
                spec,
                shape=(cfg.num_layers, *spec.shape),
                mesh_axes=PartitionSpec(None, *spec.mesh_axes),
                factorization=transform_factorization_spec(spec.factorization),
            ),
            specs,
        )

    def initialize_parameters_recursively(
        self,
        prng_key: jax.random.KeyArray,
        *,
        prebuilt: Optional[NestedTensor] = None,
    ) -> NestedTensor:
        def init(prng_key_i, prebuilt_i):
            return VDict(
                layer=self.layer.initialize_parameters_recursively(
                    prng_key_i, prebuilt=get_or_none(prebuilt_i, "layer")
                )
            )

        cfg = self.config
        return jax.vmap(init)(split_prng_key(prng_key, cfg.num_layers).keys, prebuilt)

    class Output(NamedTuple):
        carry: NestedTensor
        ys: NestedTensor

    def _run(self, fn, carry=None, *, xs=None):
        """Invokes 'fn' for each sub-layer.

        Args:
            fn: A function with args (carry, x) returning a dict(carry=..., y=...).
            carry: a nested tensor for the iterative input of the 0'th sub-layer.
            xs: a nested tensor with separate inputs for each sub-layer,
                where each leaf value T is a tensor of shape [cfg.num_layers, ...]
                and T[i, ...] represents layer-wise inputs to the i'th sub-layer.

        Returns:
            A dict with the following keys:
            - carry: a nested tensor with the same structure as iterative_input_0
                representing the iterative output of the last sub-layer.
            - ys: a nested tensor where each leaf value T is a tensor of shape [cfg.num_layers, ...]
                and T[i, ...] represents layer-wise output from the i'th sub-layer.
        """
        cfg = self.config

        if xs is None:
            xs = {}
        if carry is None:
            carry = {}

        context = current_context()
        assert context is not None
        prng_key = context.prng_key
        with child_context("layer") as layer_context:

            def scan_fn(carry_i, scan_i):
                prng_key_i, layer_state_i, x_i = scan_i
                output_collection_i = new_output_collection()
                with child_context(
                    "iter",
                    module=layer_context.module,
                    state=layer_state_i,
                    prng_key=prng_key_i,
                    output_collection=output_collection_i,
                ):
                    carry_i, y_i = fn(carry_i, x_i)
                # TODO(adesai22): Find way to avoid clearing intermediate outputs.
                output_collection_i.module_outputs.clear()
                return carry_i, dict(y_i=y_i, output_collection=output_collection_i)

            carry, scan_ys = jax.lax.scan(
                scan_fn,
                init=carry,
                xs=(split_prng_key(prng_key, cfg.num_layers).keys, layer_context.state, xs),
            )

            output_collection = layer_context.output_collection
            output_collection.update(scan_ys["output_collection"])

        return self.Output(carry=carry, ys=scan_ys["y_i"])
