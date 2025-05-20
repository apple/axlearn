# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# naver/splade:
# Copyright (c) 2021-present NAVER Corp.
# Licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

"""Tests Splade modules."""
import itertools
from typing import Optional

import jax
import jax.random
import numpy as np
import torch
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.layers import Embedding, Linear
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.poolings import BasePoolingLayer
from axlearn.common.splade import SpladePooling
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import Tensor, safe_not


class SpladePoolingParentLayer(BaseLayer):
    """A test parent layer to test SpladePooling when token_emb is shared with it."""

    @config_class
    class Config(BaseLayer.Config):
        token_emb: Embedding.Config = Embedding.default_config()
        sparse_pooler: BasePoolingLayer.Config = SpladePooling.default_config()

    def __init__(self, cfg: BaseLayer.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("token_emb", cfg.token_emb)
        self._share_with_descendants(
            self.token_emb,
            shared_module_name=SpladePooling.SHARED_EMB_NAME,
        )
        self._add_child("sparse_pooler", cfg.sparse_pooler)

    def forward(self, tokens: Tensor, paddings: Tensor = None) -> Tensor:
        return self.sparse_pooler(tokens, paddings)


class SpladePoolingTest(TestCase):
    def ref_splade_implementation(
        self, inputs, attention_mask, splade_mode, model_args
    ):  # pylint: disable=no-self-use
        """Reference splade implementation.

        Ref: https://github.com/naver/splade/blob/main/splade/models/transformer_rep.py#L188-L193
        """
        mapping_layer = torch.nn.Sequential(
            torch.nn.Linear(model_args["input_dim"], model_args["input_dim"]),
            torch.nn.GELU("tanh"),
            torch.nn.LayerNorm([model_args["input_dim"]], eps=1e-8),
            torch.nn.Linear(model_args["input_dim"], model_args["output_dim"]),
        )
        out = mapping_layer(inputs)
        if splade_mode == "sum":
            values = torch.sum(
                torch.log(1 + torch.relu(out)) * attention_mask.unsqueeze(-1), dim=1, keepdims=True
            )
        else:
            values, _ = torch.max(
                torch.log(1 + torch.relu(out)) * attention_mask.unsqueeze(-1), dim=1, keepdims=True
            )
        return (
            values,
            {
                "token_emb": mapping_layer[3],
                "sparse_pooler": {
                    "vocab_mapping": {
                        "transform": {
                            "linear": mapping_layer[0],
                            "norm": mapping_layer[2],
                        },
                    },
                    "inner_head": mapping_layer[3],
                },
            },
        )

    def verify_splade_against_ref(self, inputs, splade_layer, paddings, splade_mode, model_args):
        # Process the paddings.
        if paddings is None:
            torch_paddings = torch.ones(inputs.shape[:-1])
            axlearn_paddings = None
        else:
            torch_paddings = torch.from_numpy(paddings.astype(float))
            # axlearn_paddings = True means padded tokens.
            axlearn_paddings = safe_not(paddings)

        # Reference output.
        ref_output, ref_model_params = self.ref_splade_implementation(
            torch.from_numpy(inputs), torch_paddings, splade_mode, model_args
        )
        layer_params = {
            "vocab_mapping": {
                "transform": {
                    "linear": parameters_from_torch_layer(
                        ref_model_params["sparse_pooler"]["vocab_mapping"]["transform"]["linear"]
                    ),
                    "norm": parameters_from_torch_layer(
                        ref_model_params["sparse_pooler"]["vocab_mapping"]["transform"]["norm"]
                    ),
                },
            }
        }
        if isinstance(splade_layer, SpladePooling):
            layer_params["vocab_mapping"]["output_bias"] = parameters_from_torch_layer(
                ref_model_params["sparse_pooler"]["inner_head"]
            )["bias"]
            layer_params["vocab_mapping"]["inner_head"] = {}
            layer_params["vocab_mapping"]["inner_head"]["weight"] = parameters_from_torch_layer(
                ref_model_params["sparse_pooler"]["inner_head"]
            )["weight"]
        elif isinstance(splade_layer, SpladePoolingParentLayer):
            layer_params["vocab_mapping"]["output_bias"] = parameters_from_torch_layer(
                ref_model_params["token_emb"]
            )["bias"]
            layer_params = {
                "token_emb": {
                    "weight": parameters_from_torch_layer(ref_model_params["token_emb"])[
                        "weight"
                    ].transpose()
                },
                "sparse_pooler": layer_params,
            }
        else:
            raise ValueError("Unsupported splade_layer class!")

        layer_output, _ = F(
            splade_layer,
            inputs=dict(tokens=jnp.asarray(inputs), paddings=axlearn_paddings),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(layer_output, ref_output.detach().numpy())

    @parameterized.parameters(
        itertools.product(
            ["max", "sum"],
            [True, False],
        )
    )
    def test_splade_pooling(self, mode, share_token_emb):
        batch_size = 2
        length = 12
        dim = 32
        vocab_size = 64

        if share_token_emb:
            splade_pooling_parent_layer_cfg = SpladePoolingParentLayer.default_config().set(
                name="splade_pooling_parent_layer"
            )
            splade_pooling_parent_layer_cfg.token_emb.set(num_embeddings=vocab_size, dim=dim)
            splade_pooling_parent_layer_cfg.sparse_pooler.set(
                input_dim=dim,
                output_dim=vocab_size,
                splade_mode=mode,
            )
            splade_layer = splade_pooling_parent_layer_cfg.instantiate(parent=None)
        else:
            splade_pooling_layer_cfg = SpladePooling.default_config().set(
                name="splade_pooling_layer",
                input_dim=dim,
                output_dim=vocab_size,
                splade_mode=mode,
            )
            splade_pooling_layer_cfg.vocab_mapping.inner_head = Linear.default_config().set(
                bias=False, input_dim=dim, output_dim=5
            )
            splade_layer = splade_pooling_layer_cfg.instantiate(parent=None)

        model_args = {
            "input_dim": dim,
            "output_dim": vocab_size,
        }
        # test w/o paddings
        inputs = np.random.random((batch_size, length, dim)).astype(np.float32)
        self.verify_splade_against_ref(
            inputs, splade_layer, paddings=None, splade_mode=mode, model_args=model_args
        )

        # test w/ paddings
        inputs = np.random.random((batch_size, length, dim)).astype(np.float32)
        paddings = np.random.randint(0, 2, (batch_size, length))
        self.verify_splade_against_ref(
            inputs, splade_layer, paddings=paddings, splade_mode=mode, model_args=model_args
        )
