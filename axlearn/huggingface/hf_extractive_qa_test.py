# Copyright © 2023 Apple Inc.

"""Tests HF Extractive QA wrappers."""
import os

import jax.random
import numpy as np
import torch
from absl.testing import parameterized
from jax import numpy as jnp
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig

from axlearn.common.config import RequiredFieldMissingError
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.huggingface.hf_extractive_qa import (
    HfBertForExtractiveQuestionAnsweringWrapper,
    HfRobertaForExtractiveQuestionAnsweringWrapper,
    HfXLMRobertaForExtractiveQuestionAnsweringWrapper,
    _HfExtractiveQuestionAnsweringWrapper,
)

testdata_dir = os.path.join(os.path.dirname(__file__), "../experiments/testdata/huggingface")


# pylint: disable=no-self-use
class HfExtractiveQuestionAnsweringWrapperTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            "extractive_qa_wrapper": HfBertForExtractiveQuestionAnsweringWrapper,
            "hf_config_cls": BertConfig,
            "hidden_dim": 24,
        },
        {
            "extractive_qa_wrapper": HfXLMRobertaForExtractiveQuestionAnsweringWrapper,
            "hf_config_cls": XLMRobertaConfig,
            "hidden_dim": 24,
        },
        {
            "extractive_qa_wrapper": HfRobertaForExtractiveQuestionAnsweringWrapper,
            "hf_config_cls": RobertaConfig,
            "hidden_dim": 24,
        },
    )
    def test_feed_forward(  # pylint: disable=too-many-statements
        self,
        extractive_qa_wrapper: type[_HfExtractiveQuestionAnsweringWrapper],
        hf_config_cls: type[PretrainedConfig],
        hidden_dim: int,
    ):
        batch_size, seq_len = 4, 8
        dropout_rate = 0.1
        cfg = extractive_qa_wrapper.default_config().set(
            name="test",
            dtype=jnp.float32,
            hf_config=hf_config_cls(hidden_size=hidden_dim, hidden_dropout_prob=dropout_rate),
        )
        model = cfg.instantiate(parent=None)
        model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))

        inputs = jnp.ones([batch_size, seq_len], dtype=jnp.float32)

        # Test predict().
        outputs, _ = F(
            model,
            inputs=dict(input_batch=dict(input_ids=inputs, token_type_ids=inputs)),
            state=model_params,
            is_training=False,
            method="predict",
            prng_key=jax.random.PRNGKey(0),
        )
        start_logits = outputs["start_logits"]
        self.assertEqual((batch_size, seq_len), start_logits.shape)
        self.assertFalse(jnp.isnan(start_logits).any().item())
        end_logits = outputs["end_logits"]
        self.assertEqual((batch_size, seq_len), end_logits.shape)
        self.assertFalse(jnp.isnan(end_logits).any().item())

        # Test forward() with full batch.
        outputs, _ = F(
            model,
            inputs=dict(
                input_batch=dict(
                    input_ids=inputs,
                    token_type_ids=jnp.zeros_like(inputs),
                    start_positions=jnp.array([0] * batch_size),
                    end_positions=jnp.array([1] * batch_size),
                )
            ),
            state=model_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertLen(outputs, 2)
        start_logits = outputs[1]["start_logits"]
        self.assertEqual((batch_size, seq_len), start_logits.shape)
        self.assertFalse(jnp.isnan(start_logits).any().item())

        end_logits = outputs[1]["end_logits"]
        self.assertEqual((batch_size, seq_len), end_logits.shape)
        self.assertFalse(jnp.isnan(end_logits).any().item())

        torch_start_logits = torch.tensor(np.array(start_logits))
        torch_end_logits = torch.tensor(np.array(end_logits))
        num_examples = 4
        torch_ce_loss = (
            torch.nn.functional.cross_entropy(
                torch_start_logits,
                torch.tensor([0] * batch_size),
                reduction="none",
                ignore_index=-1,
            ).sum()
            / num_examples
            + torch.nn.functional.cross_entropy(
                torch_end_logits, torch.tensor([1] * batch_size), reduction="none", ignore_index=-1
            ).sum()
            / num_examples
        ) / 2
        self.assertAlmostEqual(torch_ce_loss.item(), outputs[0].item(), delta=1e-6)

        # Test forward with end_position clipping.
        outputs, _ = F(
            model,
            inputs=dict(
                input_batch=dict(
                    input_ids=inputs,
                    token_type_ids=jnp.zeros_like(inputs),
                    start_positions=jnp.array([0] * batch_size),
                    # Some randomly large number will be clipped to seq_len-1.
                    end_positions=jnp.array([1024] * batch_size),
                )
            ),
            state=model_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertLen(outputs, 2)
        start_logits = outputs[1]["start_logits"]
        self.assertEqual((batch_size, seq_len), start_logits.shape)
        self.assertFalse(jnp.isnan(start_logits).any().item())

        end_logits = outputs[1]["end_logits"]
        self.assertEqual((batch_size, seq_len), end_logits.shape)
        self.assertFalse(jnp.isnan(end_logits).any().item())

        torch_start_logits = torch.tensor(np.array(start_logits))
        torch_end_logits = torch.tensor(np.array(end_logits))
        num_examples = 4
        torch_ce_loss = (
            torch.nn.functional.cross_entropy(
                torch_start_logits,
                torch.tensor([0] * batch_size),
                reduction="none",
                ignore_index=-1,
            ).sum()
            / num_examples
            + torch.nn.functional.cross_entropy(
                torch_end_logits,
                torch.tensor([seq_len - 1] * batch_size),
                reduction="none",
                ignore_index=-1,
            ).sum()
            / num_examples
        ) / 2
        self.assertAlmostEqual(torch_ce_loss.item(), outputs[0].item(), delta=1e-6)

        # Test forward with half batch.
        outputs, _ = F(
            model,
            inputs=dict(
                input_batch=dict(
                    input_ids=inputs,
                    token_type_ids=jnp.zeros_like(inputs),
                    # Last two are paddings, we use start_position = end_position = -1.
                    start_positions=jnp.array([0, 0, -1, -1]),
                    end_positions=jnp.array([1, 1, -1, -1]),
                )
            ),
            state=model_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertLen(outputs, 2)
        start_logits = outputs[1]["start_logits"]
        self.assertEqual((batch_size, seq_len), start_logits.shape)
        self.assertFalse(jnp.isnan(start_logits).any().item())

        end_logits = outputs[1]["end_logits"]
        self.assertEqual((batch_size, seq_len), end_logits.shape)
        self.assertFalse(jnp.isnan(end_logits).any().item())

        torch_start_logits = torch.tensor(np.array(start_logits))
        torch_end_logits = torch.tensor(np.array(end_logits))
        num_examples = 2
        torch_ce_loss = (
            torch.nn.functional.cross_entropy(
                torch_start_logits, torch.tensor([0, 0, -1, -1]), reduction="none", ignore_index=-1
            ).sum()
            / num_examples
            + torch.nn.functional.cross_entropy(
                torch_end_logits, torch.tensor([1, 1, -1, -1]), reduction="none", ignore_index=-1
            ).sum()
            / num_examples
        ) / 2
        self.assertAlmostEqual(torch_ce_loss.item(), outputs[0].item(), delta=1e-6)

    @parameterized.parameters(
        {
            "extractive_qa_wrapper": HfBertForExtractiveQuestionAnsweringWrapper,
            "hf_config_cls": BertConfig,
            "hidden_dim": 24,
            "pad_token_id": 0,
        },
        {
            "extractive_qa_wrapper": HfXLMRobertaForExtractiveQuestionAnsweringWrapper,
            "hf_config_cls": XLMRobertaConfig,
            "hidden_dim": 24,
            "pad_token_id": 0,
        },
        {
            "extractive_qa_wrapper": HfRobertaForExtractiveQuestionAnsweringWrapper,
            "hf_config_cls": RobertaConfig,
            "hidden_dim": 24,
            "pad_token_id": 0,
        },
    )
    def test_attention_mask(
        self,
        extractive_qa_wrapper: type[_HfExtractiveQuestionAnsweringWrapper],
        hf_config_cls: type[PretrainedConfig],
        hidden_dim: int,
        pad_token_id: int,
    ):  # pylint: disable=duplicate-code
        cfg = extractive_qa_wrapper.default_config().set(
            name="test",
            dtype=jnp.float32,
            hf_config=hf_config_cls(hidden_size=hidden_dim, pad_token_id=pad_token_id),
        )
        model = cfg.instantiate(parent=None)
        model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        forward_kwargs, _ = F(
            model,
            inputs=dict(
                input_batch=dict(
                    input_ids=jnp.array(
                        [[101, 2023, 2003, 6207, 102], [101, 2228, 2367, 102, 0]], dtype=jnp.int32
                    )
                )
            ),
            state=model_params,
            is_training=False,
            method="_forward_kwargs",
            prng_key=jax.random.PRNGKey(0),
        )
        assert_allclose(
            forward_kwargs["attention_mask"],
            jnp.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=jnp.int32),
        )

    @parameterized.parameters(
        {
            "extractive_qa_wrapper": HfBertForExtractiveQuestionAnsweringWrapper,
        },
        {
            "extractive_qa_wrapper": HfXLMRobertaForExtractiveQuestionAnsweringWrapper,
        },
        {
            "extractive_qa_wrapper": HfRobertaForExtractiveQuestionAnsweringWrapper,
        },
    )
    def test_exception_when_missing_hf_config(
        self,
        extractive_qa_wrapper: type[_HfExtractiveQuestionAnsweringWrapper],
    ):
        with self.assertRaises(RequiredFieldMissingError) as e:
            cfg = extractive_qa_wrapper.default_config().set(
                name="test",
                dtype=jnp.float32,
            )
            cfg.instantiate(parent=None)
        self.assertEqual(
            "pretrained_model_path is required when hf_config is not specified.", str(e.exception)
        )
