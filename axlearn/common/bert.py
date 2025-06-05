# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/bert:
# Copyright 2018 The Google AI Language Team Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# facebookresearch/fairseq:
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.
#
# huggingface/transformers:
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# tensorflow/tensorflow:
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""BERT layer implementations.

https://arxiv.org/abs/1810.04805
"""
from typing import Optional

import jax.numpy as jnp

from axlearn.common.attention import (
    BaseStackedTransformerLayer,
    LearnedPositionalEmbedding,
    StackedTransformerLayer,
    TransformerLayer,
    scaled_hidden_dim,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.encoder import Encoder, EncoderModel
from axlearn.common.layers import (
    BaseClassificationHead,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    RedirectToSharedModule,
    get_activation_fn,
    set_layer_norm_eps_recursively,
)
from axlearn.common.loss import mean_squared_error
from axlearn.common.module import Module, child_context
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    DefaultInitializer,
    WeightInitializer,
    constant_initializer,
)
from axlearn.common.utils import NestedTensor, Tensor


class BertPooler(BaseLayer):
    """BERT Pooling Layer. This is used for example for the NSP task."""

    @config_class
    class Config(BaseLayer.Config):
        input_dim: Required[int] = REQUIRED  # Input dimension.
        linear: InstantiableConfig = Linear.default_config()
        dropout: InstantiableConfig = Dropout.default_config()
        activation: str = "jnp.tanh"

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("linear", cfg.linear.set(input_dim=cfg.input_dim, output_dim=cfg.input_dim))
        self._add_child("dropout", cfg.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        """Runs a forward pass of the pooler.

        Reference:
        https://github.com/facebookresearch/fairseq/blob/956fcf495b2d5d696ba114520363f82148a8a649/fairseq/models/roberta/model.py#L526-L533

        Args:
            inputs: a float Tensor of shape [..., seq_len, hidden_dim].

        Returns:
            A float Tensor of shape [..., hidden_dim].
        """
        cfg = self.config
        # "Pool" outputs by simply taking the first token embeddings.
        pooled_output = inputs[..., 0, :]
        with child_context("dropout1", module=self.dropout):
            pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)
        pooled_output = get_activation_fn(cfg.activation)(pooled_output)
        with child_context("dropout2", module=self.dropout):
            pooled_output = self.dropout(pooled_output)
        return pooled_output


class NonLinear(BaseLayer):
    """Non-linear transformation."""

    @config_class
    class Config(BaseLayer.Config):
        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        linear: InstantiableConfig = Linear.default_config()
        activation: str = "nn.gelu"
        norm: InstantiableConfig = LayerNorm.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "linear", cfg.linear.set(input_dim=cfg.input_dim, output_dim=cfg.output_dim)
        )
        self._add_child("norm", cfg.norm.set(input_dim=cfg.output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies a non-linear transform to the inputs.

        Args:
            inputs: a float Tensor of shape [..., input_dim].

        Returns:
            A float Tensor of shape [..., output_dim].
        """
        cfg = self.config
        x = self.linear(inputs)
        x = get_activation_fn(cfg.activation)(x)
        x = self.norm(x)
        return x


class BertLMHead(BaseClassificationHead):
    """BERT LM Head."""

    @config_class
    class Config(BaseClassificationHead.Config):
        inner_head: Required[InstantiableConfig] = REQUIRED  # Output layer.
        transform: InstantiableConfig = NonLinear.default_config()  # Output transform.
        ignored_target_id: int = -100  # Value of target that should be ignored.

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if hasattr(cfg.inner_head, "input_dim"):
            self._add_child("inner_head", cfg.inner_head.set(input_dim=cfg.input_dim))
        else:
            self._add_child("inner_head", cfg.inner_head.set(dim=cfg.input_dim))
        self._add_child(
            "transform", cfg.transform.set(input_dim=cfg.input_dim, output_dim=cfg.input_dim)
        )

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            # Output layer weights are shared with encoder, but bias is a separate parameter.
            output_bias=ParameterSpec(
                shape=[cfg.num_classes],
                mesh_axes=(None,),
                initializer=constant_initializer(0.0),
            ),
        )

    def forward(self, input_batch: NestedTensor) -> Tensor:
        """Produces prediction scores from BERT output features.

        Args:
            input_batch: A dict with the following entries:
                hidden_states: A Tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            A Tensor of shape [batch_size, seq_len, vocab_size] representing logits.
        """
        x = self.transform(input_batch["hidden_states"])
        x = self.inner_head(x)
        x = x + self.parameters["output_bias"]
        return x

    def loss(
        self, *, logits: Tensor, target_labels: Tensor, soft_labels: Optional[Tensor] = None
    ) -> Tensor:
        """Computes cross-entropy loss.

        Args:
            logits: a float Tensor of shape [..., num_classes].
            target_labels: an int Tensor of shape [...].
                Targets should contain the ground truth token ids in the range [0, num_classes).
                Targets with value equal to `config.ignored_target_id` are mapped outside of this
                range and effectively ignored in the loss calculation.
            soft_labels: Optional labels that are already smoothed/in one-hot form. If provided,
                target_labels will only be used for inferring the mask during loss calculation.

        Returns:
            A scalar loss value.
        """
        # Map ignored targets out of the valid range [0, num_classes).
        target_labels = jnp.where(target_labels == self.config.ignored_target_id, -1, target_labels)
        return super().loss(logits=logits, target_labels=target_labels, soft_labels=soft_labels)


class BertSequenceClassificationHead(BaseClassificationHead):
    """BERT Sequence Classification Head, for tasks like GLUE.

    Note: This does not strictly handle classification tasks; for example, the STSB GLUE task is a
    regression problem. We still refer to this as a "classification head" keeping in line with
    fairseq and Hugging Face. Regression tasks will have `num_classes = 1`.
    """

    @config_class
    class Config(BaseClassificationHead.Config):
        pooler: BertPooler.Config = BertPooler.default_config()  # Pooler layer.
        output: Linear.Config = Linear.default_config()  # Output projection layer.

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("pooler", cfg.pooler.set(input_dim=cfg.input_dim))
        self._add_child(
            "output",
            cfg.output.set(input_dim=cfg.input_dim, output_dim=cfg.num_classes),
        )

    def forward(self, input_batch: NestedTensor) -> Tensor:
        """Produces prediction scores from BERT output features.

        Args:
            input_batch: A dict with the following entries:
                hidden_states: A Tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            A Tensor of shape [batch_size, num_classes] representing logits.
        """
        x = input_batch["hidden_states"]
        x = self.pooler(x)
        x = self.output(x)
        return x

    def loss(
        self, *, logits: Tensor, target_labels: Tensor, soft_labels: Optional[Tensor] = None
    ) -> Tensor:
        """Computes the loss based on the number of classes.

        For regression tasks (num_classes = 1), computes MSE loss.
        Otherwise, computes cross-entropy loss.

        Args:
            logits: A float Tensor of shape [..., num_classes].
            target_labels: An int Tensor of shape [...] with values in the range [0, num_classes)
                for classification; or a float Tensor of shape [...] for regression.
            soft_labels: Optional labels that are already smoothed/in one-hot form. If provided,
                target_labels will only be used for inferring the mask during loss calculation.

        Returns:
            A scalar loss value.
        """
        cfg = self.config
        if cfg.num_classes == 1:  # Regression.
            weighted_loss = mean_squared_error(logits.squeeze(axis=-1), target_labels)
            self.add_summary("loss", weighted_loss)
            return weighted_loss.mean
        return super().loss(logits=logits, target_labels=target_labels, soft_labels=soft_labels)


class BertMultipleChoiceHead(BaseClassificationHead):
    """BERT Multiple Choice Head, for tasks like COPA or SWAG.

    Each choice typically corresponds to a separate embedding, resulting in an input shape
    [batch_size, num_classes, seq_len, hidden_dim], where each class is a multiple choice option.
    The loss is identical to `BertSequenceClassificationHead`.

    Reference:
    https://github.com/huggingface/transformers/blob/35a7052b61579cfe8df1a059d4cd3359310ec2d1/src/transformers/models/roberta/modeling_roberta.py#L1271
    """

    @config_class
    class Config(BaseClassificationHead.Config):
        pooler: BertPooler.Config = BertPooler.default_config()  # Pooler layer.
        output: Linear.Config = Linear.default_config()  # Output projection layer.

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # Ensures that we don't treat this as a regression problem.
        if cfg.num_classes < 2:
            raise ValueError(f"cfg.num_classes should be at least 2, got {cfg.num_classes}.")
        self._add_child("pooler", cfg.pooler.set(input_dim=cfg.input_dim))
        self._add_child("output", cfg.output.set(input_dim=cfg.input_dim, output_dim=1))

    def forward(self, input_batch: NestedTensor) -> Tensor:
        """Produces prediction scores from BERT output features.

        Args:
            input_batch: A dict with the following entries:
                hidden_states: A Tensor of shape [batch_size, num_classes, seq_len, hidden_dim].

        Returns:
            A Tensor of shape [batch_size, num_classes] representing logits.
        """
        x = input_batch["hidden_states"]
        # [batch, num_classes, hidden_dim].
        x = self.pooler(x)
        # [batch, num_classes, 1].
        x = self.output(x)
        # [batch, num_classes].
        return x.squeeze(axis=-1)


class BertModel(EncoderModel):
    """BERT model."""

    Config = EncoderModel.Config

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._share_with_descendants(
            self.encoder.emb.token_emb, shared_module_name="shared_token_emb"
        )

    @classmethod
    def default_config(cls):
        cfg: BertModel.Config = super().default_config()
        # param_init matches original tf implementation. The Hugging Face implementation uses plain
        # GaussianInitializer which empirically does not work as well (gradients quickly vanish).
        # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L377
        # https://github.com/huggingface/transformers/blob/aa6cfe9c4b073b2c058a78fc2d26fe3fbe0ad70b/src/transformers/models/bert/modeling_bert.py#L726
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None,
                    distribution="truncated_normal",
                    scale=0.02,
                )
            }
        )
        # By default, assume `head` employs tied weights.
        cfg.head = BertLMHead.default_config().set(
            inner_head=RedirectToSharedModule.default_config().set(
                shared_module="shared_token_emb",
                # Map the method name here so the child can just call forward directly.
                method_map=dict(forward="attend"),
            )
        )
        return cfg


def bert_embedding_config(
    *,
    max_position_embeddings: Optional[int] = 512,
    layer_norm_epsilon: Optional[float] = None,
    type_vocab_size: Optional[int] = None,
) -> TransformerTextEmbeddings.Config:
    """Builds configs for BERT Embedding layer.

    Defaults are from the BERT-BASE model.

    Args:
        max_position_embeddings: Number of positional embeddings.
            If None, position embedding is not used.
        layer_norm_epsilon: Optional epsilon for layer norm.
        type_vocab_size: Optional number of token type embeddings.

    Returns:
        The embedding configs.
    """
    cfg = TransformerTextEmbeddings.default_config().set(
        norm=LayerNorm.default_config().set(eps=layer_norm_epsilon),
    )
    if max_position_embeddings is not None:
        cfg.set(
            pos_emb=LearnedPositionalEmbedding.default_config().set(
                shape=(max_position_embeddings,)
            )
        )
    if type_vocab_size is not None:
        cfg.set(
            type_emb=Embedding.default_config().set(num_embeddings=type_vocab_size),
        )
    return cfg


# Linter thinks `ignored_target_id` is an unused param and complains that we document it, so we
# ignore the warning.
# pylint: disable-next=useless-param-doc
def bert_lm_head_config(
    *,
    vocab_size: int,
    activation: str = "nn.gelu",
    ignored_target_id: int = 0,
    layer_norm_epsilon: Optional[float] = None,
    base_cfg: Optional[BertLMHead.Config] = None,  # Keep this at the end.
) -> BertLMHead.Config:
    """Builds configs for BertLMHead.

    Defaults are from the BERT-BASE model.

    Args:
        vocab_size: Vocab size.
        activation: Type of output activation.
        ignored_target_id: Value of target that should be ignored.
        layer_norm_epsilon: Optional epsilon for layer normalization.
        base_cfg: Optional base config. Will be cloned.

    Returns:
        The output head configs.
    """
    base_cfg = base_cfg.clone() if base_cfg else BertLMHead.default_config()
    return base_cfg.set(
        transform=NonLinear.default_config().set(
            activation=activation,
            norm=LayerNorm.default_config().set(eps=layer_norm_epsilon),
        ),
        ignored_target_id=ignored_target_id,
        num_classes=vocab_size,
    )


def bert_transformer_config(
    *,
    num_layers: int = 12,
    num_heads: int = 12,
    layer_norm_epsilon: Optional[float] = None,
    base_cfg: Optional[BaseStackedTransformerLayer.Config] = None,
) -> BaseStackedTransformerLayer.Config:
    """Builds configs for BERT transformer stack.

    Defaults are from the BERT-BASE model.

    Args:
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads per transformer layer.
        layer_norm_epsilon: Epsilon for layer normalization.
        base_cfg: Optional base config. Will be cloned.

    Returns:
        The stack config.
    """
    base_cfg = base_cfg.clone() if base_cfg else StackedTransformerLayer.default_config()
    layer_norm_epsilon = (
        layer_norm_epsilon
        if layer_norm_epsilon is not None
        else bert_layer_norm_epsilon(dtype=base_cfg.dtype)
    )

    layer_cfg = TransformerLayer.default_config()
    # Feed-forward transformer layer config.
    layer_cfg.feed_forward.activation = "nn.gelu"
    layer_cfg.feed_forward.norm.eps = layer_norm_epsilon
    layer_cfg.feed_forward.hidden_dim = scaled_hidden_dim(4)
    layer_cfg.feed_forward.structure = "postnorm"
    # Self attention transformer layer config.
    layer_cfg.self_attention.norm.eps = layer_norm_epsilon
    layer_cfg.self_attention.attention.num_heads = num_heads
    layer_cfg.self_attention.structure = "postnorm"

    return base_cfg.set(num_layers=num_layers, layer=layer_cfg)


def bert_model_config(
    *,
    vocab_size: int,
    hidden_dim: int = 768,
    dropout_rate: float = 0.0,
    dtype: jnp.dtype = jnp.float32,
    embedding_cfg: Optional[Embedding.Config] = None,
    stack_cfg: Optional[BaseStackedTransformerLayer.Config] = None,
    head_cfg: Optional[BaseClassificationHead.Config] = None,
    encoder_cfg: Optional[Encoder.Config] = None,
    base_cfg: Optional[BertModel.Config] = None,
) -> BertModel.Config:
    """Builds configs for BERT model.

    Defaults are from the BERT-BASE model.

    Args:
        vocab_size: Vocab size.
        hidden_dim: Hidden dim.
        dropout_rate: Dropout rate.
        dtype: Model dtype.
        embedding_cfg: Optional embedding config. Defaults to a BERT-base Embedding.
        stack_cfg: Optional transformer stack config. Defaults to a StackedTransformerLayer.
        head_cfg: Optional head config. Defaults to a BertLMHead.Config.
        encoder_cfg: Optional encoder config. Defaults to a Encoder.Config.
        base_cfg: Optional base config. Will be cloned.

    Returns:
        The stack config.
    """
    base_cfg = base_cfg.clone() if base_cfg else BertModel.default_config()

    encoder_cfg = encoder_cfg or Encoder.default_config().set(
        dim=hidden_dim,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        emb=embedding_cfg or bert_embedding_config(),
        transformer=stack_cfg or bert_transformer_config(),
        pad_token_id=0,
    )
    head_cfg = head_cfg or bert_lm_head_config(
        base_cfg=base_cfg.head,  # pylint: disable=no-member
        vocab_size=vocab_size,
    )
    base_cfg.set(
        dtype=dtype, vocab_size=vocab_size, dim=hidden_dim, encoder=encoder_cfg, head=head_cfg
    )
    set_layer_norm_eps_recursively(base_cfg, bert_layer_norm_epsilon(dtype=dtype))
    return base_cfg


def bert_layer_norm_epsilon(dtype=None):
    # https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/contrib/layers/python/layers/layers.py#L2322
    return 1e-12 if dtype != jnp.float16 else 1e-3
