# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# facebookresearch/DiT:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under CC-BY-NC.

"""Scalable Diffusion Models with Transformers (DiT).

Ref: https://github.com/facebookresearch/DiT
"""

from typing import Optional, Union

import chex
import einops
import jax
import jax.numpy as jnp

from axlearn.common.attention import MultiheadAttention, scaled_hidden_dim
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import (
    REQUIRED,
    FunctionConfigBase,
    InstantiableConfig,
    Required,
    config_class,
)
from axlearn.common.layers import Dropout, Embedding, LayerNormStateless, Linear, get_activation_fn
from axlearn.common.module import Module, Tensor
from axlearn.common.utils import NestedTensor, TensorSpec


def modulate(*, x, shift, scale):
    """Modulates the input x tensor.

    Note: shift and scale must have the same shape.

    Args:
        x: input tensor with shape [batch_size, num_length, input_dim].
        shift: shifting the norm tensor with shape [batch_size, 1|num_length, input_dim].
        scale: scaling the norm tensor with shape [batch_size, 1|num_length, input_dim].

    Returns:
        A tensor with shape [batch_size, num_length, input_dim].
    """
    chex.assert_equal_shape((shift, scale))
    chex.assert_equal_rank((x, shift, scale))
    return x * (1 + scale) + shift


class TimeStepEmbedding(BaseLayer):
    """Timestep Embedding layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures TimeStepEmbedding."""

        output_dim: Required[int] = REQUIRED
        # Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L31
        pos_embed_dim: int = 256
        embed_proj: Linear.Config = Linear.default_config()
        output_proj: Linear.Config = Linear.default_config()
        activation: str = "nn.silu"
        max_timescale: float = 10000
        output_norm: Optional[InstantiableConfig] = None

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "embed_proj", cfg.embed_proj.set(input_dim=cfg.pos_embed_dim, output_dim=cfg.output_dim)
        )
        self._add_child(
            "output_proj", cfg.output_proj.set(input_dim=cfg.output_dim, output_dim=cfg.output_dim)
        )
        if cfg.output_norm is not None:
            self._add_child("output_norm", cfg.output_norm.set(input_dim=cfg.output_dim))

    def dit_sinusoidal_positional_embeddings(self, positions: Tensor):
        """DiT Specific Sinusoidal Positional Embeddings.

        The DiT specific sinusoidal positional embedding is different to the standard.
        1. The log_timescale_increment is calculated as:
            jnp.log(max_timescale) / jnp.maximum(1, pos_emb_dim // 2)
        2. The output is the concatenation of [sin, cos] emb, which is not the standard [cos, sin].

        Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L27-L64

        Args:
            positions: an int Tensor of shape [batch_size, 1].

        Returns:
            Sinusoidal positional embeddings of shape [batch_size, pos_emb_dim]
        """
        cfg = self.config
        num_timescales = cfg.pos_embed_dim // 2
        log_timescale_increment = jnp.log(cfg.max_timescale) / jnp.maximum(1, num_timescales)

        # [num_timescales].
        inv_timescales = jnp.exp(jnp.arange(num_timescales) * -log_timescale_increment)

        # [..., num_timescales].
        scaled_time = jnp.expand_dims(positions, -1) * inv_timescales

        # [..., pos_embed_dim].
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=-1)
        if cfg.pos_embed_dim % 2:
            signal = jnp.concatenate([signal, jnp.zeros_like(signal[:, :1])], axis=-1)
        return signal

    def forward(
        self,
        positions: Tensor,
    ) -> Tensor:
        """Computes time step positional embeddings.

        Args:
            positions: an int Tensor of shape [batch_size, 1].

        Returns:
            A Tensor of shape [batch_size, output_dim].
        """
        cfg = self.config
        # pos_emb shape [batch_size, pos_embed_dim]
        pos_emb = self.dit_sinusoidal_positional_embeddings(positions)
        # x shape [batch_size, output_dim]
        x = self.embed_proj(pos_emb)
        x = get_activation_fn(self.config.activation)(x)
        # output shape [batch_size, output_dim]
        output = self.output_proj(x)
        if cfg.output_norm is not None:
            output = self.output_norm(output)
        return output


class LabelEmbedding(BaseLayer):
    """Label Embedding layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures LabelEmbedding."""

        num_classes: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        dropout_rate: Optional[float] = None
        emb: Embedding.Config = Embedding.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # Ref: https://github.com/facebookresearch/DiT/blob/main/models.py#L74
        # The num_embeddings = cfg.num_classes + (dropout_rate > 0)
        # Additional token is added to label dropout for classifier-free guidance.
        use_drop_rate = int(cfg.dropout_rate > 0)
        self._add_child(
            "emb", cfg.emb.set(dim=cfg.output_dim, num_embeddings=cfg.num_classes + use_drop_rate)
        )

    def _mask_label(self, label: Tensor) -> Tensor:
        """This function is activated only when dropout_rate > 0 and is_training=False.

        Args:
            label: an int Tensor of shape [batch_size].

        Returns:
            Masked label with shape [batch_size].
        """
        cfg = self.config
        if not self.is_training or cfg.dropout_rate is None or cfg.dropout_rate == 0:
            return label
        samples = jax.random.uniform(
            self.prng_key, shape=label.shape, dtype=jnp.float32, minval=0.0, maxval=1.0
        )
        # The official implementation uses sample-wise dropout.
        drop = (samples < cfg.dropout_rate).astype(jnp.int32)
        # The last_id = cfg.num_classes is used for classifier-free condition.
        masked_label = label * (1 - drop) + cfg.num_classes * drop
        return masked_label

    def forward(
        self,
        labels: Tensor,
    ) -> Tensor:
        """Computes label embeddings.

        Args:
            labels: an int Tensor of shape [batch_size].

        Returns:
            A Tensor of shape [batch_size, output_dim].
        """
        # pos_emb shape [batch_size, pos_embed_dim]
        masked_label = self._mask_label(labels)
        # output shape [batch_size, output_dim]
        output = self.emb(masked_label)
        return output


class AdaptiveLayerNormModulation(BaseLayer):
    """Adaptive Layer Norm Modulation Layer

    This layer generates the means and scales that modulate
        the follow-up layers.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures AdaptiveLayerNormModulation."""

        dim: Required[int] = REQUIRED
        # The number of parameters to generate.
        num_outputs: Required[int] = REQUIRED
        linear: Linear.Config = Linear.default_config()
        activation: str = "nn.silu"

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "linear", cfg.linear.set(input_dim=cfg.dim, output_dim=cfg.num_outputs * cfg.dim)
        )

    # pylint: disable-next=redefined-builtin
    def forward(self, input: Tensor) -> Tensor:
        """Generate the parameters for modulation.

        Args:
            input: A tensor with shape [batch_size, dim] or [batch_size, num_length, dim].

        Returns:
            A list of tensors with length num_outputs.
                Each tensor has shape [batch_size, 1|num_length, dim].
        """
        cfg = self.config
        if input.ndim not in (2, 3):
            raise ValueError(f"The input must be rank 2 or 3, but got the {input.shape} tensor.")
        x = get_activation_fn(cfg.activation)(input)
        output = self.linear(x)
        if output.ndim == 2:
            output = einops.rearrange(output, "b d -> b 1 d")
        output = jnp.split(output, cfg.num_outputs, axis=-1)
        return output


class DiTFeedForwardLayer(BaseLayer):
    """The DiT feed forward layer.

    prenorm: output = input + gate * mlp(norm(input) * (1 + scale) + shift)
    postnorm: output = input + gate * norm(mlp(input * (1 + scale) + shift))
    hybridnorm: output = input + gate * postnorm(mlp(prenorm(input) * (1 + scale) + shift))
        where mlp = linear2(act(linear1(x)).
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures DiTFeedForwardLayer."""

        input_dim: Required[int] = REQUIRED  # Input feature dim.
        # The hidden dim.
        # It should be given either as an integer or a function config that instantiates
        # a dim-to-dim function, e.g., scaled_hidden_dim(4).
        hidden_dim: Union[int, FunctionConfigBase] = scaled_hidden_dim(scale=4)
        # Config for the first linear layer.
        linear1: InstantiableConfig = Linear.default_config().set(
            param_partition_spec=[None, "model"]
        )
        # Config for the second linear layer.
        linear2: InstantiableConfig = Linear.default_config().set(
            param_partition_spec=["model", None]
        )
        # Noted this is a STATELESS layer norm.
        norm: InstantiableConfig = (
            LayerNormStateless.default_config()
        )  # The normalization layer config.
        activation: Union[str, tuple[str, str]] = "nn.gelu"
        dropout1: InstantiableConfig = Dropout.default_config()
        dropout2: InstantiableConfig = Dropout.default_config()
        structure: str = "prenorm"

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if isinstance(cfg.hidden_dim, int):
            hidden_dim = cfg.hidden_dim
        else:
            hidden_dim = cfg.hidden_dim.set(input_dim=cfg.input_dim).instantiate()

        if cfg.structure in ["prenorm", "postnorm"]:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))
        elif cfg.structure == "hybridnorm":
            self._add_child("prenorm", cfg.norm.clone(input_dim=cfg.input_dim))
            self._add_child("postnorm", cfg.norm.clone(input_dim=cfg.input_dim))
        else:
            raise NotImplementedError(cfg.structure)

        self._add_child(
            "linear1",
            cfg.linear1.set(input_dim=cfg.input_dim, output_dim=hidden_dim),
        )
        self._add_child(
            "linear2",
            cfg.linear2.set(input_dim=hidden_dim, output_dim=cfg.input_dim),
        )
        self._add_child("dropout1", cfg.dropout1)
        self._add_child("dropout2", cfg.dropout2)

    # pylint: disable-next=redefined-builtin
    def forward(self, *, input: Tensor, shift: Tensor, scale: Tensor, gate: Tensor) -> Tensor:
        """The forward function of DiTFeedForwardLayer.

        Args:
            input: input tensor with shape [batch_size, num_length, input_dim].
            shift: shifting the norm tensor with shape [batch_size, 1|num_length, input_dim].
            scale: scaling the norm tensor with shape [batch_size, 1|num_length, input_dim].
            gate: applying before the residual addition with shape
                [batch_size, 1|num_length, input_dim].

        Returns:
            A tensor with shape [batch_size, num_length, input_dim].
        """
        chex.assert_equal_shape((shift, scale, gate))
        chex.assert_equal_rank((input, shift))
        cfg = self.config
        remat_pt1 = "linear1_0"
        remat_pt2 = "linear2"

        if cfg.structure == "prenorm":
            x = self.norm(input)
        elif cfg.structure == "hybridnorm":
            x = self.prenorm(input)
        elif cfg.structure == "postnorm":
            x = input

        x = modulate(x=x, shift=shift, scale=scale)
        x = self.linear1(x)
        x = self._remat_name(x, remat_pt1)
        x = get_activation_fn(cfg.activation)(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self._remat_name(x, remat_pt2)

        if cfg.structure == "postnorm":
            x = self.norm(x)
        elif cfg.structure == "hybridnorm":
            x = self.postnorm(x)

        x = self.dropout2(x)
        x = x * gate
        x += input
        return x


class DiTAttentionLayer(BaseLayer):
    """The DiT attention layer.

    prenorm: output = input + gate * multihead_atten(norm(input) * (1 + scale) + shift)
    postnorm: output = input + gate * norm(multihead_atten(input * (1 + scale) + shift))
    hybridnorm: output = input + gate * postnorm(multihead_atten(
        prenorm(input) * (1 + scale) + shift))
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures DiTAttentionLayer."""

        target_dim: Required[int] = REQUIRED  # Input target feature dim.
        source_dim: Required[int] = REQUIRED  # Input source feature dim.
        norm: LayerNormStateless.Config = LayerNormStateless.default_config()
        attention: InstantiableConfig = MultiheadAttention.default_config()
        structure: str = "prenorm"

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.target_dim != cfg.source_dim:
            raise ValueError(
                f"Target dim ({cfg.target_dim}) should match source dim ({cfg.source_dim}."
            )

        if cfg.structure in ["prenorm", "postnorm"]:
            self._add_child("norm", cfg.norm.set(input_dim=cfg.target_dim))
        elif cfg.structure == "hybridnorm":
            self._add_child("prenorm", cfg.norm.clone(input_dim=cfg.target_dim))
            self._add_child("postnorm", cfg.norm.clone(input_dim=cfg.target_dim))
        else:
            raise NotImplementedError(cfg.structure)

        self._add_child(
            "attention",
            cfg.attention.set(
                query_dim=cfg.target_dim,
                key_dim=cfg.source_dim,
                value_dim=cfg.source_dim,
                output_dim=cfg.target_dim,
            ),
        )

    def forward(
        self,
        *,
        # pylint: disable-next=redefined-builtin
        input: Tensor,
        shift: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        gate: Optional[Tensor] = None,
        attention_logit_biases: Optional[Tensor] = None,
    ) -> Tensor:
        """The forward function of DiTAttentionLayer.

        Args:
            input: input tensor with shape [batch_size, num_length, target_dim].
            shift: If provided, shifting the norm tensor with shape [batch_size, 1|num_length,
                target_dim] and scale should be provided.
            scale: If provided, scaling the norm tensor with shape [batch_size, 1|num_length,
                target_dim] and shift should be provided.
            gate: If provided, applying before the residual addition with shape
                [batch_size, 1|num_length, target_dim].
            attention_logit_biases: Optional Tensor representing the self attention biases.

        Returns:
            A tensor with shape [batch_size, num_length, target_dim].

        Raises:
            ValueError: shift and scale are not both provided or both None.
        """
        if (shift is None) != (scale is None):
            raise ValueError("shift and scale must be both provided or both None.")

        cfg = self.config

        if cfg.structure == "prenorm":
            x = self.norm(input)
        elif cfg.structure == "hybridnorm":
            x = self.prenorm(input)
        elif cfg.structure == "postnorm":
            x = input

        if shift is not None and scale is not None:
            x = modulate(x=x, shift=shift, scale=scale)

        x = self.attention(query=x, attention_logit_biases=attention_logit_biases).data

        if cfg.structure == "postnorm":
            x = self.norm(x)
        elif cfg.structure == "hybridnorm":
            x = self.postnorm(x)

        if gate is not None:
            x = x * gate

        output = input + x
        return output

    def init_states(self, input_spec: TensorSpec) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            input_spec: TensorSpec [batch, num_length, target_dim] corresponding to query vector.

        Returns:
            init_states: A Nested Tensor state depending on the `attention` layer implementation.
        """
        states = dict()
        states["attention"], _ = self.attention.init_states(
            time_step=None, query=input_spec, attention_logit_biases=None
        )
        return states

    def extend_step(
        self,
        cached_states: NestedTensor,
        target: Tensor,
        *,
        shift: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        gate: Optional[Tensor] = None,
    ) -> tuple[NestedTensor, Tensor]:
        """Computes the value vector given the query of the current step.
        This function is used by autoregressive decoding.

        Args:
            cached_states: A `NestedTensor` object containing tensors which are the
                results of previous attentions, and index used for fast decoding. Contains
                "attention" cached states.
            target: target tensor with shape [batch_size, step_length, target_dim].
            shift: If provided, shifting the norm tensor with shape [batch_size, 1|num_length,
                target_dim] and scale should be provided.
            scale: If provided, scaling the norm tensor with shape [batch_size, 1|num_length,
                target_dim] and shift should be provided.
            gate: If provided, applying before the residual addition with shape
                [batch_size, 1|num_length, target_dim].

        Returns:
            A tuple (cached_states, output):
            * cached_states: A NestedTensor of cache states.
            * output: A output tensor of shape [batch, step_length, target_dim]
        """
        if (shift is None) != (scale is None):
            raise ValueError("shift and scale must be both provided or both None.")

        cfg = self.config
        if cfg.structure == "prenorm":
            x = self.norm(target)
        elif cfg.structure == "hybridnorm":
            x = self.prenorm(target)
        elif cfg.structure == "postnorm":
            x = target

        if shift is not None and scale is not None:
            x = modulate(x=x, shift=shift, scale=scale)

        # It supports only the (sliding window) causal case, which is handled by attention itself.
        attention_logit_biases = None
        attn_states, attn_output = self.attention.extend_step(
            cached_states=cached_states["attention"],
            query=x,
            attention_logit_biases=attention_logit_biases,
        )
        x = attn_output.data

        if cfg.structure == "postnorm":
            x = self.norm(x)
        elif cfg.structure == "hybridnorm":
            x = self.postnorm(x)

        if gate is not None:
            x = x * gate

        output = target + x
        return dict(attention=attn_states), output


class DiTBlock(BaseLayer):
    """The DiT block layer.

    This is the concatenation of DiTAttentionLayer and DiTFeedForwardLayer.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures DiTBlock."""

        input_dim: Required[int] = REQUIRED
        attention: DiTAttentionLayer.Config = DiTAttentionLayer.default_config()
        feed_forward: DiTFeedForwardLayer.Config = DiTFeedForwardLayer.default_config()
        adaln: AdaptiveLayerNormModulation.Config = (
            AdaptiveLayerNormModulation.default_config().set(num_outputs=6)
        )

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "attention", cfg.attention.set(target_dim=cfg.input_dim, source_dim=cfg.input_dim)
        )
        self._add_child("feed_forward", cfg.feed_forward.set(input_dim=cfg.input_dim))
        self._add_child("adaln", cfg.adaln.set(dim=cfg.input_dim))

    # pylint: disable-next=redefined-builtin
    def forward(self, *, input: Tensor, condition: Tensor) -> Tensor:
        """The forward function of DiTBlock.

        Args:
            input: input tensor with shape [batch_size, num_length, input_dim].
            condition: tensor with shape [batch_size, input_dim] or [batch_size, num_length,
                input_dim] for generating layer norm shift, scale, and gate.

        Returns:
            A tensor with shape [batch_size, num_length, input_dim].
        """
        layer_norm_params = self.adaln(condition)
        shift_attn, scale_attn, gate_attn = layer_norm_params[0:3]
        shift_ffn, scale_ffn, gate_ffn = layer_norm_params[3:6]
        x = self.attention(input=input, shift=shift_attn, scale=scale_attn, gate=gate_attn)
        x = self.feed_forward(input=x, shift=shift_ffn, scale=scale_ffn, gate=gate_ffn)

        return x

    def init_states(self, input_spec: TensorSpec) -> NestedTensor:
        """Initializes cache for autoregressive cached decoding.

        Args:
            input_spec: TensorSpec [batch, target_length, target_dim] corresponding to query vector.

        Returns:
            init_states: A Nested Tensor state depending on the `attention` layer implementation.
        """
        states = dict()
        states["attention"] = self.attention.init_states(input_spec=input_spec)
        return states

    def extend_step(
        self,
        cached_states: NestedTensor,
        target: Tensor,
        *,
        condition: Tensor,
    ) -> tuple[NestedTensor, Tensor]:
        """Computes the value vector given the query of the current step.
        This function is used by autoregressive decoding.

        Args:
            cached_states: A `NestedTensor` object containing tensors which are the
                results of previous attentions, and index used for fast decoding. Contains
                "attention" cached states.
            target: target tensor with shape [batch_size, step_length, input_dim].
            condition: tensor with shape [batch_size, input_dim] or [batch_size, step_length,
                input_dim] for generating layer norm shift, scale, and gate.

        Returns:
            A tuple (cached_states, output):
            * cached_states: A NestedTensor of cache states.
            * output: A output tensor of shape [batch, step_length, target_dim]
        """
        layer_norm_params = self.adaln(condition)
        shift_attn, scale_attn, gate_attn = layer_norm_params[0:3]
        shift_ffn, scale_ffn, gate_ffn = layer_norm_params[3:6]
        attn_states, x = self.attention.extend_step(
            cached_states=cached_states["attention"],
            target=target,
            shift=shift_attn,
            scale=scale_attn,
            gate=gate_attn,
        )
        x = self.feed_forward(input=x, shift=shift_ffn, scale=scale_ffn, gate=gate_ffn)
        return dict(attention=attn_states), x


class DiTFinalLayer(BaseLayer):
    """The DiT final layer.

    output = linear(norm(input) * (1 + scale) + shift)
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures DiTFinalLayer."""

        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        norm: LayerNormStateless.Config = LayerNormStateless.default_config()
        linear: Linear.Config = Linear.default_config()
        adaln: AdaptiveLayerNormModulation.Config = (
            AdaptiveLayerNormModulation.default_config().set(num_outputs=2)
        )

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("norm", cfg.norm.set(input_dim=cfg.input_dim))
        self._add_child("adaln", cfg.adaln.set(dim=cfg.input_dim))
        self._add_child(
            "linear", cfg.linear.set(input_dim=cfg.input_dim, output_dim=cfg.output_dim)
        )

    # pylint: disable-next=redefined-builtin
    def forward(self, *, input: Tensor, condition: Tensor) -> Tensor:
        """The forward function of DiTFinalLayer.

        Args:
            input: input tensor with shape [batch_size, num_length, input_dim].
            condition: tensor with shape [batch_size, input_dim] or [batch_size, num_length,
                input_dim] for generating layer norm shift and scale.

        Returns:
            A tensor with shape [batch_size, num_length, output_dim].
        """
        layer_norm_params = self.adaln(input=condition)
        x = self.norm(input)
        x = modulate(x=x, shift=layer_norm_params[0], scale=layer_norm_params[1])
        output = self.linear(x)
        return output
