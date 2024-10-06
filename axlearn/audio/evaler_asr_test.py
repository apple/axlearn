# Copyright © 2024 Apple Inc.

"""Tests ASR evaluation utilities."""

import os
from typing import Optional

import jax
import jax.numpy as jnp
import seqio
import tensorflow as tf
from absl.testing import parameterized
from jax.experimental import mesh_utils

from axlearn.audio.asr_decoder import DecodeOutputs
from axlearn.audio.evaler_asr import (
    WordErrorRateMetricCalculator,
    WordErrors,
    compute_word_errors,
    normalize_text,
)
from axlearn.audio.input_asr import text_input
from axlearn.common.base_model import BaseModel
from axlearn.common.config import InstantiableConfig, config_for_class, config_for_function
from axlearn.common.decoding import BrevityPenaltyFn, brevity_penalty_fn
from axlearn.common.input_fake import fake_source
from axlearn.common.input_tf_data import with_processor
from axlearn.common.metrics import WeightedScalar
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor

_1K_VOCAB_FILE = os.path.join(
    os.path.dirname(__file__),
    "../data/tokenizers/sentencepiece/librispeech_unigram_1024.model",
)
# A text spm with larger vocab to exercise text normalization.
_32K_VOCAB_FILE = os.path.join(
    os.path.dirname(__file__),
    "../data/tokenizers/sentencepiece/bpe_32k_c4.model",
)


class _DummyAsrModel(BaseModel):
    """A dummy model which implements beam_search_decode method."""

    # pylint: disable-next=no-self-use
    def beam_search_decode(
        self,
        input_batch,
        *,
        num_decodes: int,
        max_decode_len: int,
        brevity_penalty: Optional[BrevityPenaltyFn] = None,
    ):
        token_ids = jnp.reshape(input_batch["mock_decodes"], [-1, num_decodes, max_decode_len])
        if brevity_penalty is not None:
            # Fake brevity penalty to test it is set and applied.
            token_ids = jnp.round(
                brevity_penalty(length=jnp.array(max_decode_len), raw_scores=token_ids)
            ).astype(jnp.int32)

        return DecodeOutputs(
            sequences=token_ids,
            raw_sequences=None,  # Not used.
            paddings=None,  # Not used.
            scores=None,  # Not used.
        )


def _text_dataset(
    texts: list[str],
    *,
    input_key: str,
    vocab: InstantiableConfig[seqio.SentencePieceVocabulary],
    max_len: int,
) -> tf.data.Dataset:
    source = config_for_function(fake_source).set(
        examples=[{input_key: text} for text in texts], shuffle_buffer_size=0
    )
    processor = config_for_function(text_input).set(
        max_len=max_len,
        vocab=vocab,
        input_key=input_key,
        truncate=True,
        # Explicitly retain empty target_labels to test dropping at the evaler.
        min_len=0,
    )
    return with_processor(source, processor=processor, is_training=False)()


def _asr_text_dataset(
    hypotheses: list[str],
    references: list[str],
    *,
    vocab: InstantiableConfig[seqio.SentencePieceVocabulary],
    num_decodes: int,
    max_decode_len: int,
) -> tf.data.Dataset:
    """Generates dummy input batch and decode outputs for word error rate eval."""
    batch_size, max_target_len = 3, 16
    assert len(hypotheses) / len(references) == num_decodes

    hyp_ds = _text_dataset(hypotheses, input_key="hypothesis", vocab=vocab, max_len=max_decode_len)
    ref_ds = _text_dataset(references, input_key="reference", vocab=vocab, max_len=max_target_len)

    # Batch for beam width. [num_decodes, max_decode_len].
    hyp_ds = hyp_ds.batch(num_decodes)

    # Merge keys from datasets.
    ds = tf.data.Dataset.zip(hyp_ds, ref_ds)
    # Construct examples.
    ds = ds.map(
        lambda hyp, ref: dict(mock_decodes=hyp["target_labels"], target_labels=ref["target_labels"])
    )
    # Batch to produce:
    # mock_decodes: [batch_size, num_decodes, max_decode_len]
    # target_labels: [batch_size, max_target_len]
    return ds.batch(batch_size)


def _compute_metrics(
    hypotheses: list[str],
    references: list[str],
    *,
    vocab_file: str,
    num_decodes: int = 1,
    brevity_penalty: Optional[BrevityPenaltyFn] = None,
) -> dict[str, WeightedScalar]:
    max_decode_len = 32
    with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
        model = _DummyAsrModel.default_config().set(name="test-model").instantiate(parent=None)
        decode_kwargs = dict(num_decodes=num_decodes, max_decode_len=max_decode_len)
        if brevity_penalty:
            decode_kwargs["brevity_penalty"] = brevity_penalty

        cfg: WordErrorRateMetricCalculator.Config = (
            WordErrorRateMetricCalculator.default_config().set(
                vocab=config_for_class(seqio.SentencePieceVocabulary).set(
                    sentencepiece_model_file=vocab_file,
                ),
                model_method_kwargs=decode_kwargs,
            )
        )
        calculator: WordErrorRateMetricCalculator = cfg.set(name="test-metric").instantiate(
            parent=None, model=model, model_param_partition_specs={}
        )
        state = calculator.init_state(prng_key=jax.random.PRNGKey(0), model_params={})
        ds = _asr_text_dataset(
            hypotheses=hypotheses,
            references=references,
            vocab=cfg.vocab,
            num_decodes=num_decodes,
            max_decode_len=max_decode_len,
        )
        for input_batch in ds:
            input_batch = as_tensor(input_batch)
            forward_outputs = calculator.forward(input_batch, model_params={}, state=state)
            state = forward_outputs["state"]

        return calculator.get_summaries(model_params={}, state=state, all_forward_outputs=[])


class WordErrorRateMetricCalculatorTest(TestCase):
    @parameterized.parameters(
        dict(
            reference="",
            hypothesis="",
            expected=WordErrors(num_deletions=0, num_insertions=0, num_substitutions=0),
        ),
        dict(
            reference="",
            hypothesis="test",
            expected=WordErrors(num_deletions=0, num_insertions=1, num_substitutions=0),
        ),
        dict(
            reference="test",
            hypothesis="",
            expected=WordErrors(num_deletions=1, num_insertions=0, num_substitutions=0),
        ),
        dict(
            reference="hello world",
            hypothesis="hello tiger",
            expected=WordErrors(num_deletions=0, num_insertions=0, num_substitutions=1),
        ),
        dict(
            reference="hello this is a more complicated test example",
            hypothesis="there are more complicated test examples",
            expected=WordErrors(num_deletions=2, num_insertions=0, num_substitutions=3),
        ),
    )
    def test_compute_word_errors(self, reference: str, hypothesis: str, expected: WordErrors):
        actual = compute_word_errors(hypothesis.split(), reference.split())
        self.assertEqual(expected.num_total, actual.num_total)
        self.assertEqual(expected.num_deletions, actual.num_deletions)
        self.assertEqual(expected.num_insertions, actual.num_insertions)
        self.assertEqual(expected.num_substitutions, actual.num_substitutions)

    @parameterized.parameters(
        dict(
            text='<s>This is a test.\t  (This is a "<unk>" token!)',
            expected="this is a test this is a token!",
        ),
        dict(
            text="<TEST__//>! !  ..(\n\t\n],.?",
            expected="",
        ),
    )
    def test_normalize_text(self, text: str, expected: list[str]):
        self.assertEqual(expected, normalize_text()(text))

    @parameterized.parameters(
        [
            # Basic cases.
            dict(
                references=[
                    "HELLO WORLD",
                    "THIS IS CORRECT",
                    "CARPE DIEM",
                    "WHAT TIME IS IT",
                    "VIVA LA VIDA",
                ],
                # There are 2 decodes for each reference.
                # We should only consider the top decode of each.
                hypotheses=[
                    "HALO HELLO WORLD",  # 1 insertion.
                    "HELLO HELLO",
                    "THIS IS CORRECT",  # 0 error.
                    "IS CORRECT",
                    "CAPE DIEM",  # 1 substitution.
                    "CARPE DIEM",
                    "WHAT TIME IS",  # 1 deletion.
                    "WHAT TIME",
                    "VIVA VIDA",  # 1 deletion.
                    "VIVA LA VIDA",
                ],
                num_decodes=2,
                expected={
                    "word_errors/wer": WeightedScalar(4.0 / 14, 14),
                    "word_errors/deletions": WeightedScalar(2.0 / 14, 14),
                    "word_errors/insertions": WeightedScalar(1.0 / 14, 14),
                    "word_errors/substitutions": WeightedScalar(1.0 / 14, 14),
                    "word_errors/sentence_accuracy": WeightedScalar(1.0 / 5, 5),
                },
                vocab_file=_1K_VOCAB_FILE,
            ),
            # Test with normalization.
            dict(
                references=[
                    "Hello, world!! Good-bye",
                    "<s>This is:\nCORRECT</s>",
                    " <body>\t\t?reference\n?  (</body>)",
                ],
                # There are 2 decodes for each reference.
                # We should only consider the top decode of each.
                hypotheses=[
                    "HELLO world! Goodbye",  # Good-bye -> Goodbye.
                    '<s>this is "correct" correct ...',  # is -> is:, delete correct.
                    "reference",  # OK.
                ],
                num_decodes=1,
                expected={
                    "word_errors/wer": WeightedScalar(3.0 / 7, 7),
                    "word_errors/deletions": WeightedScalar(0.0 / 7, 7),
                    "word_errors/insertions": WeightedScalar(1.0 / 7, 7),
                    "word_errors/substitutions": WeightedScalar(2.0 / 7, 7),
                    "word_errors/sentence_accuracy": WeightedScalar(1.0 / 3, 3),
                },
                vocab_file=_32K_VOCAB_FILE,
            ),
            # Test empty references/hypotheses.
            dict(
                references=["", "HELLO"],  # First reference is dropped. Second is incorrect.
                hypotheses=["INVALID", "", "", "HELLO"],
                num_decodes=2,
                expected={
                    "word_errors/wer": WeightedScalar(1.0 / 1, 1),
                    "word_errors/deletions": WeightedScalar(1.0 / 1, 1),
                    "word_errors/insertions": WeightedScalar(0.0 / 1, 1),
                    "word_errors/substitutions": WeightedScalar(0.0 / 1, 1),
                    "word_errors/sentence_accuracy": WeightedScalar(0.0 / 1, 1),
                },
                vocab_file=_1K_VOCAB_FILE,
            ),
        ]
    )
    def test_word_error_rate_metrics(
        self,
        hypotheses: list[str],
        references: list[str],
        num_decodes: int,
        expected: dict[str, WeightedScalar],
        vocab_file: str,
    ):
        actual = _compute_metrics(
            hypotheses, references, num_decodes=num_decodes, vocab_file=vocab_file
        )
        self.assertNestedAllClose(expected, actual)

    def test_brevity_penalty(self):
        references = ["HELLO", "HELLO WORLD"]
        hypotheses = ["HELLO", "HELLO WORLD"]

        # Without brevity penalty, outputs match exactly.
        outputs = _compute_metrics(hypotheses, references, vocab_file=_1K_VOCAB_FILE)
        self.assertNestedAllClose(outputs["word_errors/wer"].mean, 0.0)
        self.assertNestedAllClose(outputs["word_errors/sentence_accuracy"].mean, 1.0)

        # With brevity penalty, DummyModel should emit fewer tokens.
        brevity_penalty = brevity_penalty_fn(alpha=0.9, bp_type="t5")
        outputs = _compute_metrics(
            hypotheses, references, brevity_penalty=brevity_penalty, vocab_file=_1K_VOCAB_FILE
        )
        self.assertGreater(outputs["word_errors/wer"].mean, 0.0)
        self.assertNestedAllClose(outputs["word_errors/sentence_accuracy"].mean, 0.0)
