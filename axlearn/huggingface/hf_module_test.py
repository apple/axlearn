# Copyright © 2023 Apple Inc.

"""General tests for HuggingFace module wrapper."""
import jax.numpy as jnp
from absl.testing import parameterized
from transformers.models.bert.configuration_bert import BertConfig

# pylint: disable=no-self-use
from axlearn.huggingface.hf_sequence_classification import HfBertForSequenceClassificationWrapper


class HfWrapperTest(parameterized.TestCase):
    def test_exception_for_unspecified_uri_scheme(self):
        with self.assertRaisesRegex(
            ValueError,
            r"pretrained_model_path .* has URI scheme s3 which is not supported in "
            r"uri_scheme_handlers\. Supported ones are: [a-z,]*\.",
        ):
            cfg = HfBertForSequenceClassificationWrapper.default_config().set(
                name="test",
                dtype=jnp.float32,
                hf_config=BertConfig(hidden_size=24, pad_token_id=0),
                pretrained_model_path="s3://non-existent-bucket/bert-base-uncased",
            )
            cfg.instantiate(parent=None)
