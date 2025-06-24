# Copyright © 2023 Apple Inc.

"""Layers for Splade.

https://arxiv.org/pdf/2109.10086.pdf
"""
from typing import Optional

import jax.numpy as jnp

from axlearn.common.attention_bias import NEG_INF
from axlearn.common.bert import BertLMHead
from axlearn.common.config import config_class
from axlearn.common.layers import BaseClassificationHead, RedirectToSharedModule, get_activation_fn
from axlearn.common.module import Module
from axlearn.common.poolings import BasePoolingLayer
from axlearn.common.utils import Tensor, safe_not


class SpladePooling(BasePoolingLayer):
    """Splade pooling layer."""

    SHARED_EMB_NAME = "shared_token_emb"

    @config_class
    class Config(BasePoolingLayer.Config):
        # Splade activation function. ReLU by default.
        splade_activation_fn: str = "nn.relu"
        splade_mode: str = "max"
        vocab_mapping: BaseClassificationHead.Config = BertLMHead.default_config()

    @classmethod
    def default_config(cls):
        cfg: SpladePooling.Config = super().default_config()
        # By default, assume `inner_head` employs tied token embedding weights.
        cfg.vocab_mapping = BertLMHead.default_config().set(
            inner_head=RedirectToSharedModule.default_config().set(
                shared_module=cls.SHARED_EMB_NAME,
                method_map=dict(forward="attend"),
            ),
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "vocab_mapping",
            cfg.vocab_mapping.set(
                input_dim=cfg.input_dim,
                num_classes=cfg.output_dim,
            ),
        )

    def forward(  # pylint:disable=arguments-renamed
        self, tokens: Tensor, paddings: Tensor = None
    ) -> Tensor:
        """Calculate the Splade Pooler.

        Args:
            tokens: A Tensor of shape [batch_size, seq_len, hidden_dim].
            paddings: A Tensor of shape [batch_size, seq_len].

        Returns:
            A Tensor of shape [batch_size, num_outputs, vocab_size] representing Splade features,
            where num_outputs is determined by the pooling mode. Currently cfg.splade_mode supports
            max and sum. For both, num_outputs = 1.

        Raises:
            ValueError: If cfg.splade_mode is not supported.
            NotImplementedError: If cfg.num_outputs > 1.
        """
        cfg = self.config
        if paddings is None:
            paddings = jnp.zeros((tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype)
        # paddings shape is expanded to [batch_size, seq_len, 1].
        paddings = jnp.expand_dims(paddings, -1)
        if cfg.splade_mode not in ["max", "sum"]:
            raise ValueError(f"({cfg.splade_mode}) is not supported in Splade pooling.")
        if cfg.num_outputs != 1:
            raise NotImplementedError(
                f"SPLADE pooling currently doesn't support num_outputs = ({cfg.num_outputs})."
            )
        # Output shape: [batch_size, seq_len, vocab_size].
        x = self.vocab_mapping({"hidden_states": tokens})
        # Splade = max(log(1 + relu(x)), dim=1)
        if cfg.splade_mode == "max":
            # Doing max pooling first to help reduce memory consumption.
            x += paddings * NEG_INF  # Set padded values to -inf.
            x = jnp.max(x, axis=1, keepdims=True)
            splade_output = jnp.log1p(get_activation_fn(cfg.splade_activation_fn)(x))
        elif cfg.splade_mode == "sum":
            splade_output = jnp.log1p(get_activation_fn(cfg.splade_activation_fn)(x))
            splade_output *= safe_not(paddings)  # Set padded values to 0.
            splade_output = jnp.sum(splade_output, axis=1, keepdims=True)
        return splade_output
