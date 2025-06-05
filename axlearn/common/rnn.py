# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Layers for recurrent neural networks.

References:
https://github.com/tensorflow/lingvo/blob/7dcd8e0b5704b19b3197674c856ac7a0ae3f965f/lingvo/core/rnn_cell.py
"""
from collections.abc import Sequence
from typing import Optional

import jax
from absl import logging
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.layers import Linear, MultiLinear
from axlearn.common.module import Module, child_context, new_output_collection
from axlearn.common.repeat import Repeat
from axlearn.common.utils import Nested, Tensor, VDict, get_or_none, split_prng_key


class BaseRNNCell(BaseLayer):
    """An abstract class to define the common interface of all RNN cell layers, including:

    * All subclasses must have `input_dim` and `output_dim` in its Config;
    * The common method signature for `init_states()` and `extend_step()`.
    * A common implementation for `forward()`.
    """

    @config_class
    class Config(BaseLayer.Config):
        input_dim: Required[int] = REQUIRED  # Input feature dim.
        output_dim: Optional[int] = None  # The output dim. If None, use input_dim.

    @property
    def output_dim(self):
        cfg = self.config
        return cfg.output_dim if cfg.output_dim is not None else cfg.input_dim

    def init_states(self, *, batch_size: int) -> Nested[Tensor]:
        """Returns the initial states, to be used by `extend_step`."""
        raise NotImplementedError(type(self))

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        data: Tensor,
    ) -> tuple[Nested[Tensor], Tensor]:
        """Computes the outputs and state updates for one step.

        Args:
            cached_states: A NestedTensor returned by `init_states()` or `extend_step()`.
            data: A Tensor of shape [batch_size, input_dim], the inputs for the current step.

        Returns:
            (updated_cached_states, outputs), where:
            `updated_cached_states` represents the new cached states incorporating `data`;
            `outputs` is a Tensor of shape [batch_size, output_dim].
        """
        raise NotImplementedError(type(self))

    def forward(self, time_major_inputs: Tensor) -> Tensor:
        """Computes RNN outputs given full-sequence inputs.

        For incremental computation, use init_states() and extend_step().

        Args:
            time_major_inputs: The sequence inputs, often a Tensor of shape
                [seq_len, batch_size, input_dim].

        Returns:
            A Tensor of shape [seq_len, batch_size, output_dim].
        """
        batch_size = self._batch_size(time_major_inputs)
        seq_len = self._seq_len(time_major_inputs)
        with child_context("init_states", module=self):
            initial_states = self.init_states(batch_size=batch_size)

        context = self.get_invocation_context()

        def scan_fn(carry_i, scan_i):
            prng_key_i, x_i = scan_i
            output_collection_i = new_output_collection()
            with child_context(
                "iter",
                module=self,
                state=context.state,
                prng_key=prng_key_i,
                output_collection=output_collection_i,
            ):
                next_states, y_i = self.extend_step(cached_states=carry_i, data=x_i)
            return next_states, dict(y_i=y_i, output_collection=output_collection_i)

        final_states, scan_ys = jax.lax.scan(
            scan_fn,
            init=initial_states,
            xs=(split_prng_key(context.prng_key, seq_len).keys, time_major_inputs),
        )

        output_collection = context.output_collection
        output_collection.update(scan_ys["output_collection"])
        self.add_module_output("final_states", final_states)
        return scan_ys["y_i"]

    # pylint: disable-next=no-self-use
    def _batch_size(self, inputs: Tensor) -> int:
        """Infers batch size from `inputs`."""
        raise NotImplementedError(type(self))

    # pylint: disable-next=no-self-use
    def _seq_len(self, inputs: Tensor) -> int:
        """Infers sequence length from `inputs`."""
        raise NotImplementedError(type(self))


class LSTMCell(BaseRNNCell):
    """Implements a variant of LSTM that supports normalization and output projection.

    Sepp Hochreiter; Jürgen Schmidhuber (1997). "Long short-term memory".

    Compared with the "vanilla" LSTM
    (as described in https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html),
    our implementation has the following differences:

    - We don't use biases (for simplicity)
    - We apply optional normalization on the gates i/f/g/o (for training stability)
    - We apply an additional projection on h', which allows hidden_dim >> output_dim
      (for parameter efficiency)

    All above changes are also present in the Lingvo code.
    https://github.com/tensorflow/lingvo/blob/06553f297bbc38eb369503a421d07515d114cbb4/lingvo/core/rnn_cell.py#L594-L638
    """

    @config_class
    class Config(BaseRNNCell.Config):
        # If hidden_dim is None, use output_dim and no output_proj is applied.
        hidden_dim: Optional[int] = None
        input_proj: MultiLinear.Config = MultiLinear.default_config().set(bias=False)
        output_proj: Linear.Config = Linear.default_config().set(bias=False)
        norm: Optional[BaseLayer.Config] = None
        max_cell_value: Optional[float] = None

    @property
    def hidden_dim(self):
        cfg = self.config
        return cfg.hidden_dim if cfg.hidden_dim is not None else self.output_dim

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "input_proj",
            cfg.input_proj.set(
                input_dim=cfg.input_dim + self.output_dim,
                num_outputs=4,
                output_dim=self.hidden_dim,
            ),
        )
        if cfg.hidden_dim:
            if cfg.output_proj is None:
                raise ValueError(
                    "cfg.output_proj should not be None if cfg.hidden_dim is not None."
                )
            self._add_child(
                "output_proj",
                cfg.output_proj.set(input_dim=self.hidden_dim, output_dim=self.output_dim),
            )
        if cfg.norm is not None:
            self._add_child("norm", cfg.norm.set(input_dim=self.hidden_dim))

    def init_states(self, *, batch_size: int) -> Nested[Tensor]:
        cfg = self.config
        # Using the naming convention of:
        # https://github.com/tensorflow/lingvo/blob/06553f297bbc38eb369503a421d07515d114cbb4/lingvo/core/rnn_cell.py#L223.
        # 'm' is aka 'h' in https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html.
        return dict(
            m=jnp.zeros([batch_size, self.output_dim], dtype=cfg.dtype),
            c=jnp.zeros([batch_size, self.hidden_dim], dtype=cfg.dtype),
        )

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        data: Tensor,
    ) -> tuple[Nested[Tensor], Tensor]:
        cfg = self.config
        # [batch_size, input_dim + output_dim].
        inputs_and_m = jnp.concatenate([data, cached_states["m"]], axis=-1)
        # [batch_size, 4, hidden_dim].
        input_proj = self.input_proj(inputs_and_m)
        if "norm" in self.children:
            input_proj = self.norm(input_proj)
        assert input_proj.shape == (data.shape[0], 4, self.hidden_dim)
        # The "input, forget, gate, output" gates, each of shape [batch, hidden_dim].
        proj_i, proj_f, proj_g, proj_o = (
            gate.squeeze(axis=-2) for gate in jnp.split(input_proj, indices_or_sections=4, axis=-2)
        )
        # [batch, hidden_dim].
        old_c = cached_states["c"]
        new_c = jax.nn.sigmoid(proj_f) * old_c + jax.nn.sigmoid(proj_g) * jax.nn.tanh(proj_i)
        if cfg.max_cell_value is not None:
            new_c = jnp.clip(new_c, a_min=-cfg.max_cell_value, a_max=cfg.max_cell_value)
        # [batch, output_dim].
        # Output_nonlinearity is default to True in Lingvo.
        # https://github.com/tensorflow/lingvo/blob/06553f297bbc38eb369503a421d07515d114cbb4/lingvo/core/rnn_cell.py#L247.
        new_m = jax.nn.sigmoid(proj_o) * jax.nn.tanh(new_c)
        if cfg.hidden_dim:
            new_m = self.output_proj(new_m)

        return dict(m=new_m, c=new_c), new_m

    # pylint: disable-next=no-self-use
    def _batch_size(self, inputs: Tensor) -> int:
        assert isinstance(inputs, Tensor)
        assert inputs.ndim == 3, inputs.shape
        return inputs.shape[1]

    # pylint: disable-next=no-self-use
    def _seq_len(self, inputs: Tensor) -> int:
        assert isinstance(inputs, Tensor)
        assert inputs.ndim == 3, inputs.shape
        return inputs.shape[0]


class StackedRNNLayer(BaseRNNCell):
    """Stacked RNN layer."""

    @config_class
    class Config(BaseRNNCell.Config):
        """Configures StackedRNNLayer."""

        # Sequence of rnn cell configs in the stack.
        layers: Required[Sequence[BaseRNNCell.Config]] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: StackedRNNLayer.Config = self.config
        if len(cfg.layers) <= 0:
            raise ValueError(
                f"Number of layers in the stack must be greater than 0, get {len(cfg.layers)}."
            )
        self._layers = []
        # First layer input dim.
        input_dim = cfg.input_dim
        for i, layer_cfg in enumerate(cfg.layers):
            layer = self._add_child(f"layer{i}", layer_cfg.set(input_dim=input_dim))
            self._layers.append(layer)
            logging.info(
                "Add rnn layer%s: input_dim = %s, hidden_dim = %s, output_dim=%s",
                i,
                input_dim,
                layer_cfg.hidden_dim,
                layer.output_dim,
            )
            # Next layer input dim.
            input_dim = layer.output_dim
        if cfg.output_dim is not None and cfg.output_dim != input_dim:
            raise ValueError(
                f"StackedRNNLayer: self.config.output_dim = {cfg.output_dim} is set, but "
                f"does not match last rnn's output_dim = {input_dim}."
            )

    def initialize_parameters_recursively(
        self,
        prng_key: Tensor,
        *,
        prebuilt: Optional[Nested[Tensor]] = None,
    ) -> Nested[Tensor]:
        cfg = self.config  # type: StackedRNNLayer.Config
        prng_key = split_prng_key(prng_key, len(cfg.layers))
        state = {}
        for i, layer in enumerate(self._layers):
            key = jax.tree.map(lambda x, index=i: x[index], prng_key.keys)
            state[layer.name] = layer.initialize_parameters_recursively(
                key, prebuilt=get_or_none(prebuilt, layer.name)
            )
        return state

    def init_states(self, *, batch_size: int) -> list[Nested[Tensor]]:
        """Returns a list of initial step states from all layers."""
        states_list = [layer.init_states(batch_size=batch_size) for layer in self._layers]
        return states_list

    def extend_step(
        self,
        *,
        cached_states: list[Nested[Tensor]],
        data: Tensor,
    ) -> tuple[list[Nested[Tensor]], Tensor]:
        """Computes the outputs and all layers state updates for one step.

        Args:
            cached_states: A list of cached states from all layers returned by `init_states`
                or `extend_step`.
            data: A Tensor of shape [batch_size, input_dim], the inputs for the current step.

        Returns:
            (updated_cached_states, outputs), where:
            `updated_cached_states` is a list of states from all layers;
             `outputs` is a Tensor of shape [batch_size, output_dim].
        """
        outputs = data
        updated_states_list = []
        for i, layer in enumerate(self._layers):
            states, outputs = layer.extend_step(cached_states=cached_states[i], data=outputs)
            updated_states_list.append(states)
        return updated_states_list, outputs

    def _batch_size(self, inputs: Tensor) -> int:
        # pylint: disable-next=protected-access
        return self._layers[0]._batch_size(inputs=inputs)

    def _seq_len(self, inputs: Tensor) -> int:
        # pylint: disable-next=protected-access
        return self._layers[0]._seq_len(inputs=inputs)

    @property
    def output_dim(self):
        last_cfg = self.config.layers[-1]
        last_rnn_output_dim = (
            last_cfg.output_dim if last_cfg.output_dim is not None else last_cfg.input_dim
        )
        logging.info(
            "StackedRNNLayer: output_dim = %s, from the output_dim of last rnn in the stack.",
            last_rnn_output_dim,
        )
        return last_rnn_output_dim


class _RNNRepeat(Repeat):
    """A Repeat layer with layer = children class of BaseRNNCell."""

    def init_states(self, *, batch_size: int) -> Nested[Tensor]:
        """Returns the initial states of all layers."""

        def layer_fn(_):
            return VDict(self.layer.init_states(batch_size=batch_size))

        cfg = self.config
        return jax.vmap(layer_fn)(jnp.empty(cfg.num_layers))

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        data: Tensor,
    ) -> tuple[Nested[Tensor], Tensor]:
        """Computes the outputs and state updates for one step for all layers.

        Args:
            cached_states: A NestedTensor returned by `init_states()` or `extend_step()`.
            data: A Tensor of shape [batch_size, input_dim], the inputs for the current step.

        Returns:
            (updated_cached_states, outputs), where `outputs` are usually a Tensor of shape
            [batch_size, output_dim].
        """

        def layer_fn(carry, x_i):
            updated_layer_states, layer_outputs = self.layer.extend_step(
                cached_states=x_i, data=carry
            )
            return layer_outputs, VDict(updated_layer_states)

        repeat_outputs: Repeat.Output = self._run(layer_fn, carry=data, xs=cached_states)
        return repeat_outputs.ys, repeat_outputs.carry


class RepeatedRNNLayer(BaseRNNCell):
    """Repeated RNN layer."""

    @config_class
    class Config(BaseRNNCell.Config):
        """Configures RepeatedRNNLayer."""

        # The number of layers in the rnn stack.
        num_layers: Required[int] = REQUIRED
        # Rnn cell of the layer in the stack.
        layer: BaseRNNCell.Config = LSTMCell.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config  # type: RepeatedRNNLayer.Config
        if cfg.num_layers <= 0:
            raise ValueError(
                f"num_layers must be greater than 0, get cfg.num_layers = {cfg.num_layers}."
            )
        repeat_cfg = _RNNRepeat.default_config().set(
            layer=cfg.layer.set(input_dim=cfg.input_dim),
            num_layers=cfg.num_layers,
        )
        self._add_child("repeat", repeat_cfg)

    def _batch_size(self, inputs: Tensor) -> int:
        # pylint: disable-next=protected-access
        return self.repeat.layer._batch_size(inputs)

    def _seq_len(self, inputs: Tensor) -> int:
        # pylint: disable-next=protected-access
        return self.repeat.layer._seq_len(inputs)

    def initialize_parameters_recursively(
        self,
        prng_key: Tensor,
        *,
        prebuilt: Optional[Nested[Tensor]] = None,
    ) -> Nested[Tensor]:
        # We need to call self.repeat.initialize_parameters_recursively() with the same prng_key
        # to ensure initialization parity with StackedRNNLayer.
        return dict(
            repeat=self.repeat.initialize_parameters_recursively(
                prng_key, prebuilt=get_or_none(prebuilt, "repeat")
            )
        )

    def init_states(self, *, batch_size: int) -> Nested[Tensor]:
        return self.repeat.init_states(batch_size=batch_size)

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        data: Tensor,
    ) -> tuple[Nested[Tensor], Tensor]:
        return self.repeat.extend_step(cached_states=cached_states, data=data)


class IdentityCell(BaseRNNCell):
    """Identity RNN cell."""

    def __init__(self, cfg: BaseRNNCell.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.output_dim and cfg.output_dim != cfg.input_dim:
            raise ValueError(
                "IdentityCell requires input_dim = output_dim, but got "
                f"input_dim = {cfg.input_dim}, output_dim = {cfg.output_dim}."
            )

    def init_states(self, *, batch_size: int) -> Nested[Tensor]:
        """Returns the initial states, to be used by `extend_step`."""
        return {}

    def extend_step(
        self,
        *,
        cached_states: Nested[Tensor],
        data: Tensor,
    ) -> tuple[Nested[Tensor], Tensor]:
        new_states = {}
        outputs = data
        return new_states, outputs

    def _batch_size(self, inputs: Tensor) -> int:
        assert isinstance(inputs, Tensor)
        assert inputs.ndim == 3, inputs.shape
        return inputs.shape[1]

    def _seq_len(self, inputs: Tensor) -> int:
        assert isinstance(inputs, Tensor)
        assert inputs.ndim == 3, inputs.shape
        return inputs.shape[0]
