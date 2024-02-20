# Copyright © 2024 Apple Inc.

"""Tests utilities for loading pre-trained Hugging Face models."""
from typing import Any, Optional

import jax
import numpy as np
import pytest
from absl.testing import parameterized
from jax.sharding import Mesh

from axlearn.common.base_layer import BaseLayer
from axlearn.common.deberta_test import build_cfg as build_deberta_cfg
from axlearn.common.deberta_test import build_model_config as build_deberta_model_config
from axlearn.common.layers import set_layer_norm_eps_recursively
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.state_builder import Builder
from axlearn.common.test_utils import TestCase
from axlearn.common.trainer import TrainerState
from axlearn.huggingface.hf_pretrained_loaders import (
    auto_model_from_pretrained,
    hf_pretrained_builder_config,
)


class TestDeBERTaBuilder(TestCase):
    """Tests for loading DeBERTa from HF checkpoints."""

    @parameterized.parameters(
        {
            "hidden_dim": 384,
            "num_heads": 6,
            "num_layers": 12,
            "max_distance": 512,
            "vocab_size": 128100,
            "num_directional_buckets": 256,
            "model_path": "microsoft/deberta-v3-xsmall",
        },
    )
    @pytest.mark.gs_login
    def test_deberta_hf_pretrained_builder(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_distance: int,
        vocab_size: int,
        num_directional_buckets: Optional[int],
        model_path: str,
    ):
        with Mesh(np.array(jax.devices()).reshape(-1, 1), ("data", "model")):
            _, test_cfg = build_deberta_cfg(
                share_projections=True,
                # This is equivalent to max_relative_positions in HF.
                # https://github.com/huggingface/transformers/blob/c51dc4f92755c67a83f3fc8a0bd6b3e64df199e4/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L426-L428
                max_distance=max_distance,
                query_len=max_distance,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                # This is equivalent to position_buckets in HF.
                # https://huggingface.co/microsoft/deberta-v3-small/commit/52978a199180611f3eabdcf5bec2e5912050bfb1#d2h-025836
                num_directional_buckets=num_directional_buckets,
                # HF pre-trained models have this as false:
                # https://huggingface.co/microsoft/deberta-v3-small/blob/main/config.json.
                position_biased_input=False,
                num_classes=1,
            )
            model_cfg = build_deberta_model_config(test_cfg=test_cfg)

            # Two config changes are important for matching output with HF:
            # 1) Most important: axlearn default DeBERTa uses approximated gelu activation
            # while HF uses exact gelu activation.
            # 2) axlearn default LayerNorm epsilon is different from HF's 1e-7.
            # Reference:
            # https://huggingface.co/microsoft/deberta-v3-small/commit/52978a199180611f3eabdcf5bec2e5912050bfb1#d2h-025836.
            set_layer_norm_eps_recursively(model_cfg, 1e-7)
            model_cfg.encoder.transformer.layer.feed_forward.activation = "exact_gelu"

            model = model_cfg.set(name="model").instantiate(parent=None)

            model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(0))
            trainer_state = TrainerState(
                prng_key=jax.random.PRNGKey(1),
                model=model_params,
                learner=None,
            )

            # Only restore encoder parameters, ignore head (absent from pre-trained).
            builder_cfg = hf_pretrained_builder_config(
                model_name_or_path=model_path,
                target_scope=["encoder"],
                source_scope=["encoder"],
            ).set(name="builder")
            builder = builder_cfg.instantiate(parent=None)
            restored_state = builder(
                Builder.State(step=0, trainer_state=trainer_state, built_keys=set()),
            )

            # Load the matching Hugging Face pretrained checkpoint.
            ref_model = auto_model_from_pretrained(model_path)

            # Compare encoder outputs.
            input_ids = jax.random.randint(
                jax.random.PRNGKey(123),
                shape=[8, 64],
                minval=0,
                maxval=test_cfg.vocab_size,
            )

            def parameters_from_ref_layer(src: Any, dst_layer: BaseLayer):
                del src, dst_layer
                return restored_state.trainer_state.model["encoder"]

            test_outputs, ref_outputs = self._compute_layer_outputs(
                test_layer=model.encoder,
                ref_layer=ref_model,
                test_inputs=dict(input_ids=input_ids),
                ref_inputs=as_torch_tensor(input_ids),
                parameters_from_ref_layer=parameters_from_ref_layer,
            )

            self.assertNestedAllClose(test_outputs, ref_outputs.last_hidden_state, atol=1e-4)
