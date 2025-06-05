# Copyright © 2023 Apple Inc.

"""HuggingFace sequence classification tests."""
from typing import Any, Callable

import jax.random
import numpy as np
import torch
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from transformers.configuration_utils import PretrainedConfig
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig

from axlearn.common import utils
from axlearn.common.config import RequiredFieldMissingError
from axlearn.common.module import functional as F
from axlearn.common.test_utils import assert_allclose
from axlearn.huggingface.hf_module import HF_MODULE_KEY
from axlearn.huggingface.hf_sequence_classification import (
    HfAlbertForSequenceClassificationWrapper,
    HfBertForSequenceClassificationWrapper,
    HfRobertaForSequenceClassificationWrapper,
    HfSequenceClassificationWrapper,
)


# pylint: disable=no-self-use
class HfSequenceClassificationWrapperTest(parameterized.TestCase):
    """Tests HfSequenceClassificationWrappers."""

    @parameterized.parameters(
        {
            "seq_cls_wrapper": HfBertForSequenceClassificationWrapper,
            "hf_config_cls": BertConfig,
            "hidden_dim": 24,
            "classifier_shape": lambda model_shapes: model_shapes["classifier"],
        },
        {
            "seq_cls_wrapper": HfAlbertForSequenceClassificationWrapper,
            "hf_config_cls": AlbertConfig,
            "hidden_dim": 64,
            "classifier_shape": lambda model_shapes: model_shapes["classifier"],
        },
        {
            "seq_cls_wrapper": HfRobertaForSequenceClassificationWrapper,
            "hf_config_cls": RobertaConfig,
            "hidden_dim": 24,
            "classifier_shape": lambda model_shapes: model_shapes["classifier"]["out_proj"],
        },
    )
    def test_feed_forward(
        self,
        seq_cls_wrapper: type[HfSequenceClassificationWrapper],
        hf_config_cls: type[PretrainedConfig],
        hidden_dim: int,
        classifier_shape: Callable[[dict[str, Any]], dict[str, tuple]],
    ):
        batch_size, seq_len = 4, 8
        num_labels = 2
        cfg = seq_cls_wrapper.default_config().set(
            name="test",
            dtype=jnp.float32,
            hf_config=hf_config_cls(num_labels=num_labels, hidden_size=hidden_dim),
        )
        model = cfg.instantiate(parent=None)
        model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        model_shapes = utils.shapes(model_params)[HF_MODULE_KEY]["params"]

        self.assertEqual(
            {
                "bias": (num_labels,),
                "kernel": (hidden_dim, num_labels),
            },
            classifier_shape(model_shapes),
        )

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

        self.assertEqual(type(outputs), dict)
        logits = outputs["logits"]
        self.assertEqual((batch_size, num_labels), logits.shape)
        self.assertFalse(jnp.isnan(logits).any().item())

        # Test forward().
        outputs, _ = F(
            model,
            inputs=dict(
                input_batch=dict(
                    input_ids=inputs,
                    token_type_ids=jnp.zeros_like(inputs),
                    target_labels=jnp.array([0] * batch_size),
                )
            ),
            state=model_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )
        self.assertLen(outputs, 2)
        logits = outputs[1]["logits"]
        self.assertEqual((batch_size, num_labels), logits.shape)
        self.assertFalse(jnp.isnan(logits).any().item())

        torch_logits = torch.tensor(np.array(logits))  # pylint: disable=no-member
        torch_ce_loss = torch.nn.functional.cross_entropy(
            torch_logits, torch.tensor([0] * batch_size)  # pylint: disable=no-member
        )
        self.assertAlmostEqual(torch_ce_loss.item(), outputs[0].item(), delta=1e-6)

    def test_attention_mask(self):  # pylint: disable=duplicate-code
        cfg = HfBertForSequenceClassificationWrapper.default_config().set(
            name="test",
            dtype=jnp.float32,
            hf_config=BertConfig(hidden_size=24, pad_token_id=0),
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

        # Check that additional padding at the end doesn't affect model outputs.
        outputs = [
            F(
                model,
                inputs=dict(input_batch=dict(input_ids=input_ids)),
                state=model_params,
                is_training=False,
                method="predict",
                prng_key=jax.random.PRNGKey(0),
            )
            for input_ids in [
                jnp.array([[101, 2228, 2367, 102]], dtype=jnp.int32),
                jnp.array([[101, 2228, 2367, 102, 0]], dtype=jnp.int32),
            ]
        ]
        assert_allclose(outputs[0][0]["logits"], outputs[1][0]["logits"])

    def test_padding_examples_ignored(self):
        cfg = HfBertForSequenceClassificationWrapper.default_config().set(
            name="test",
            dtype=jnp.float32,
            hf_config=BertConfig(hidden_size=12, pad_token_id=0),
        )
        model = cfg.instantiate(parent=None)
        model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))

        # Check that padding example at the end doesn't affect loss.
        outputs = [
            F(
                model,
                inputs=dict(input_batch=dict(input_ids=input_ids, target_labels=target_labels)),
                state=model_params,
                is_training=False,
                method="forward",
                prng_key=jax.random.PRNGKey(0),
            )
            for input_ids, target_labels in zip(
                [
                    jnp.array([[101, 2228, 2367, 102], [101, 1234, 456, 789]], dtype=jnp.int32),
                    jnp.array(
                        [[101, 2228, 2367, 102], [101, 1234, 456, 789], [101, 4, 5, 6]],
                        dtype=jnp.int32,
                    ),
                ],
                [jnp.array([0, 1], dtype=jnp.int32), jnp.array([0, 1, -1], dtype=jnp.int32)],
            )
        ]
        assert_allclose(outputs[0][0][0], outputs[1][0][0])

    def test_exception_when_missing_hf_config(self):
        with self.assertRaises(RequiredFieldMissingError) as e:
            cfg = HfBertForSequenceClassificationWrapper.default_config().set(
                name="test",
                dtype=jnp.float32,
            )
            cfg.instantiate(parent=None)
        self.assertEqual(
            "pretrained_model_path is required when hf_config is not specified.", str(e.exception)
        )


if __name__ == "__main__":
    absltest.main()
