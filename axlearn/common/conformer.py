# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

"""An implementation of Conformer layers.

References:
https://arxiv.org/abs/2005.08100
https://github.com/tensorflow/lingvo/blob/d2f1e1b3cccdac8f73ae20f86afb03560b1c176d/lingvo/core/conformer_layer.py
"""

from typing import Literal, Optional, Union

from jax import numpy as jnp

from axlearn.common.attention import (
    MultiheadAttention,
    MultiheadAttentionXL,
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import (
    BatchNorm,
    DepthwiseConv1D,
    Dropout,
    GroupNorm,
    LayerNorm,
    Linear,
    get_activation_fn,
)
from axlearn.common.module import Module
from axlearn.common.repeat import Repeat
from axlearn.common.utils import Tensor


class LConvLayer(BaseLayer):
    r"""Lightweight conv layer.

    architecture::
      input
      /   \
      |   linear1_norm(.)         # input_dim
      |   linear1(.)              # 2 * input_dim
      |   linear1_activation(.)   # input_dim
      |   conv(.)
      |   conv_norm(.)
      |   conv_activation(.)
      |   linear2(.)
      |   dropout(.)
      \   /
        +
        |
      output
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures LConvLayer."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        linear1_norm: LayerNorm.Config = LayerNorm.default_config()
        linear1_activation: tuple[str, str] = ("linear", "nn.sigmoid")
        linear1: Linear.Config = Linear.default_config().set(bias=True)
        conv: DepthwiseConv1D.Config = DepthwiseConv1D.default_config().set(
            # See Table 2 and 7.
            window=32,
            bias=False,
            padding="SAME",
        )
        conv_norm: Union[BatchNorm.Config, GroupNorm.Config] = BatchNorm.default_config()
        conv_activation: str = "nn.silu"  # aka. Swish
        linear2: Linear.Config = Linear.default_config().set(bias=True)
        dropout: Dropout.Config = Dropout.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: LConvLayer.Config = self.config
        self._add_child("linear1_norm", cfg.linear1_norm.set(input_dim=cfg.input_dim))

        assert len(cfg.linear1_activation) == 2, cfg.linear1_activation
        # Create a linear1 projection for each activation.
        for i in range(len(cfg.linear1_activation)):
            self._add_child(
                f"linear1_{i}",
                cfg.linear1.set(input_dim=cfg.input_dim, output_dim=cfg.input_dim),
            )

        self._add_child("conv", cfg.conv.set(input_dim=cfg.input_dim))
        self._add_child("conv_norm", cfg.conv_norm.set(input_dim=cfg.input_dim))
        self._add_child(
            "linear2",
            cfg.linear2.set(input_dim=cfg.input_dim, output_dim=cfg.input_dim),
        )
        self._add_child("dropout", cfg.dropout)

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Tensor:
        """Computes LConvLayer outputs.

        Args:
            inputs: of shape [batch, seq_len, input_dim].
            paddings: boolean tensor of shape [batch, seq_len]. True iff it's a padding position.

        Returns:
            The output feature of shape [batch, seq_len, input_dim].
        """
        cfg = self.config
        x = self.linear1_norm(inputs)
        activations = [
            get_activation_fn(activation)(self.children[f"linear1_{i}"](x))
            for i, activation in enumerate(cfg.linear1_activation)
        ]
        assert len(activations) == 2, cfg.linear1_activation
        x = activations[0] * activations[1]
        # We need to clear padded positions in 'x' before feeding into `conv` to ensure padding
        # doesn't affect results.
        x = self.conv(x * jnp.expand_dims(1 - paddings, axis=-1))
        x = self.conv_norm(x, paddings=paddings)
        x = get_activation_fn(cfg.conv_activation)(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + inputs


def compute_attention_logit_biases(
    paddings: Tensor,
    *,
    left_context: Optional[int] = None,
    right_context: Optional[int] = None,
    neg_inf: float = -1.0e9,
) -> Tensor:
    """Computes attention logit biases.

    Adapted from
    https://github.com/google/praxis/blob/097b862d883e15cf3eb9df83bf5194c9052c6576/praxis/layers/attentions.py#L52-L82.
    Empirically we find that neg_inf of the logit biases effects self-supervised pre-training
    optimization behavior.

    Args:
        paddings: 0/1 tensor of shape [batch_size, seq_len].
        left_context: integer of history steps. If None, use all history steps.
        right_context: integer of future steps. If None, use all future steps.
        neg_inf: -inf for masked logits value.

    Returns:
        Tensor of shape [batch_size, 1, seq_len, seq_len]. Output[b,i,j] is -inf
            iff attention is disabled with query=input[b, i] and key=input[b, j].

    Raises:
        ValueError: if left_context < 0 or right_context < 0.
    """
    if left_context and left_context < 0:
        raise ValueError(f"left_context must be greater or equal to 0, get {left_context}.")
    if right_context and right_context < 0:
        raise ValueError(f"right_context must be greater or equal to 0, get {right_context}.")
    seq_len = paddings.shape[1]
    # [batch_size, 1, seq_len, seq_len]. True if attention is disabled between positions.
    atten_mask = jnp.logical_or(paddings[:, None, :, None], paddings[:, None, None, :])
    idx = jnp.arange(seq_len)
    if left_context is not None:
        left_mask = idx[:, None] > (idx + left_context)[None, :]
        atten_mask = jnp.logical_or(atten_mask, left_mask[None, None, :, :])

    if right_context is not None:
        right_mask = idx[:, None] < (idx - right_context)[None, :]
        atten_mask = jnp.logical_or(atten_mask, right_mask[None, None, :, :])

    attention_logit_biases = atten_mask * neg_inf
    return attention_logit_biases


class ConformerLayer(BaseLayer):
    """The Conformer layer.

    Canonical version (with default params.)
      x = ff_start(x)
      x = attention(x)
      x = lconv(x)
      x = ff_end(x)
      y = norm(x)
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures ConformerLayer."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        ff_start: TransformerFeedForwardLayer.Config = TransformerFeedForwardLayer.default_config()
        ff_end: TransformerFeedForwardLayer.Config = TransformerFeedForwardLayer.default_config()
        self_attention: TransformerAttentionLayer.Config = (
            TransformerAttentionLayer.default_config()
        )
        lconv: LConvLayer.Config = LConvLayer.default_config()
        norm: LayerNorm.Config = LayerNorm.default_config()
        # Layer order. If None, default to "mhsa_before_conv", i.e., conformer layer order as
        # secified in https://arxiv.org/abs/2005.08100.
        # If not None, specify the layer order regarding conv and multihead self attention (mhsa).
        # e.g., lconv_before_mhsa can be found in Figure 1 https://arxiv.org/pdf/2011.10798.
        layer_order: Optional[
            Literal["lconv_before_ff", "lconv_before_mhsa", "mhsa_before_lconv"]
        ] = None

        # Config for computing relative position embeddings for range [-seq_len + 1, seq_len - 1].
        # It should only be used when attention is of class MultiheadAttention.
        rel_pos_emb: Optional[InstantiableConfig] = None
        left_context: Optional[int] = None
        right_context: Optional[int] = None
        neg_inf: float = -1.0e15

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        for ff_cfg in (cfg.ff_start, cfg.ff_end):
            ff_cfg.hidden_dim = scaled_hidden_dim(scale=4)
            ff_cfg.residual_weight = 0.5
            ff_cfg.activation = "nn.silu"
        # Conformer uses attention with XLNet relative positional embedding by default.
        # See ablation study in Table 3.
        cfg.self_attention.attention = MultiheadAttentionXL.default_config()
        cfg.self_attention.attention.input_linear.layer.bias = True
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: ConformerLayer.Config = self.config
        self._add_child("ff_start", cfg.ff_start.set(input_dim=cfg.input_dim))
        self._add_child("ff_end", cfg.ff_start.set(input_dim=cfg.input_dim))
        self._add_child(
            "self_attention",
            cfg.self_attention.set(target_dim=cfg.input_dim, source_dim=cfg.input_dim),
        )
        self._add_child("lconv", cfg.lconv.set(input_dim=cfg.input_dim))
        self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))

        if cfg.rel_pos_emb:
            if not cfg.self_attention.attention.klass == MultiheadAttention:
                raise ValueError(
                    "rel_pos_emb should only be set in MultiheadAttention, "
                    f"but got {cfg.self_attention.attention.klass}."
                )
            pos_emb_dim = cfg.self_attention.attention.num_heads
            self._add_child("rel_pos_emb", cfg.rel_pos_emb.set(dim=pos_emb_dim))

        if cfg.left_context and cfg.left_context < 0:
            raise ValueError(
                f"cfg.left_context must be greater or equal to 0, get {cfg.left_context}."
            )
        if cfg.right_context and cfg.right_context < 0:
            raise ValueError(
                f"cfg.right_context must be greater or equal to 0, get {cfg.right_context}."
            )

        if cfg.layer_order is not None:
            supported_layer_order = ["lconv_before_ff", "lconv_before_mhsa", "mhsa_before_lconv"]
            if cfg.layer_order not in supported_layer_order:
                raise ValueError(f"Only {supported_layer_order} is allowed, got {cfg.layer_order}")

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Tensor:
        """Computes ConformerLayer outputs.

        Args:
            inputs: of shape [batch, seq_len, input_dim].
            paddings: boolean tensor of shape [batch, seq_len]. True iff it's a padding position.

        Returns:
            The output feature of shape [batch, seq_len, input_dim].
        """
        cfg = self.config
        x = inputs

        layer_order = cfg.layer_order
        if layer_order is None:
            layer_order = "mhsa_before_lconv"

        if layer_order == "lconv_before_ff":
            x = self.lconv(x, paddings=paddings)
        x = self.ff_start(x)
        attention_logit_biases = compute_attention_logit_biases(
            paddings=paddings,
            left_context=cfg.left_context,
            right_context=cfg.right_context,
            neg_inf=cfg.neg_inf,
        )
        # ToDo(zhiyunlu): test limited context mask with rel_pos_emb.
        if self.config.rel_pos_emb:
            attention_logit_biases = self.rel_pos_emb(attention_logit_biases)
        if layer_order == "lconv_before_mhsa":
            x = self.lconv(x, paddings=paddings)
        x = self.self_attention(target=x, attention_logit_biases=attention_logit_biases).data
        if layer_order == "mhsa_before_lconv":
            x = self.lconv(x, paddings=paddings)
        x = self.ff_end(x)
        x = self.norm(x)
        return x


class _ConformerRepeat(Repeat):
    """A Repeat layer with layer=ConformerLayer.

    See axlearn/common/repeat.py for more details.
    """

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Tensor:
        # Note the padding does not change.
        def layer_fn(carry, _):
            # layer-wise side input is {}.
            # carry_i, y_i = fn(carry_i, x_i)
            layer_outputs: Tensor = self.layer(inputs=carry, paddings=paddings)
            return layer_outputs, _

        repeat_outputs: Repeat.Output = self._run(fn=layer_fn, carry=inputs)
        # repeat_outputs.ys is {}.
        return repeat_outputs.carry


class RepeatedConformerLayer(BaseLayer):
    """A stack of ConformerLayer."""

    @config_class
    class Config(BaseLayer.Config):
        input_dim: Required[int] = REQUIRED  # Input feature dim.
        # The number of layers in the stack.
        num_layers: Required[int] = REQUIRED
        # Config for each layer in the stack.
        layer: ConformerLayer.Config = ConformerLayer.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config  # type: RepeatedConformerLayer.Config
        if cfg.num_layers <= 0:
            raise ValueError(
                f"num_layers must be greater than 0, get cfg.num_layers = {cfg.num_layers}."
            )
        repeat_cfg = _ConformerRepeat.default_config().set(
            layer=cfg.layer.set(input_dim=cfg.input_dim),
            num_layers=cfg.num_layers,
        )
        self._add_child("repeat", repeat_cfg)

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Tensor:
        return self.repeat(inputs=inputs, paddings=paddings)
