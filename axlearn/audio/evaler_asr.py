# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""ASR evaluation utilities."""

import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import jax.numpy as jnp
import seqio
from jax.sharding import PartitionSpec
from Levenshtein import opcodes as levenshtein

from axlearn.audio.asr_decoder import DecodeOutputs
from axlearn.common.base_model import BaseModel
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
)
from axlearn.common.evaler import ModelSummaryAccumulator
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module
from axlearn.common.utils import Nested, Tensor, replicate_to_local_data


@dataclass
class WordErrors:
    """Word error metrics.

    Each metric describes the number of operations needed to transform a hypothesis into a reference
    sequence; e.g., `num_deletions` indicates the minimum number of elements of the hypothesis
    sequence that must be deleted to produce the reference sequence.
    """

    num_deletions: int
    num_insertions: int
    num_substitutions: int

    @property
    def num_total(self):
        """The sum of `num_deletions`, `num_insertions`, and `num_substitutions`."""
        return self.num_deletions + self.num_insertions + self.num_substitutions


def compute_word_errors(hypothesis: Sequence[str], reference: Sequence[str]) -> WordErrors:
    """Computes word errors from hypotheses and references using `levenshtein`.

    See also `WordErrors` for more details.
    """
    opcodes = levenshtein(hypothesis, reference)
    num_sub, num_ins, num_del = 0, 0, 0

    # Each opcode is a 5-tuple indicating how to transform hyp to ref:
    # https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.get_opcodes
    for opcode, hyp_i, hyp_j, ref_i, ref_j in opcodes:
        hyp_len = hyp_j - hyp_i
        ref_len = ref_j - ref_i

        if opcode == "equal":
            continue
        elif opcode == "delete":
            num_ins += hyp_len  # If we have to delete from hyp to ref, we have insertion errors.
        elif opcode == "insert":
            num_del += ref_len
        elif opcode == "replace":
            num_sub += min(hyp_len, ref_len)
            num_ins += max(0, hyp_len - ref_len)
            num_del += max(0, ref_len - hyp_len)
        else:
            raise ValueError(f"Unexpected {opcode=}.")

    return WordErrors(num_deletions=num_del, num_insertions=num_ins, num_substitutions=num_sub)


def normalize_text() -> Callable[[str], str]:
    """Normalizes text for WER computation.

    Specifically:
    1. Lowercase.
    2. Replace \t and \n with a space.
    3. Remove special tags, like <s> </s> and <unk>.
    4. Replace punctuations before space, before the end, or after space with a space.
    5. Remove double quotes, [, ], ( and ).
    6. Replace consecutive spaces with a space.

    Reference:
    https://github.com/tensorflow/lingvo/blob/f910d4fcab1fa57386aa40645b8814833377b531/lingvo/tasks/asr/tools/simple_wer_v2.py#L49

    Returns:
        A function that takes an input string and returns a normalized string.
    """

    def fn(text: str) -> str:
        # Replace \t and \n with a space.
        text = re.sub(r"[\t\n]", " ", text)
        # Remove special tags, like <s> </s> and <unk>.
        text = re.sub(r"\<[\w_\/\\.]+\>", "", text)
        # Remove punctuation before space.
        text = re.sub(r"[,.\?!]+ ", " ", text)
        # Remove punctuation after space.
        text = re.sub(r" [,.\?!]+", " ", text)
        # Remove punctuation before end.
        text = re.sub(r"[,.\?!]+$", " ", text)
        # Remove double quotes, [, ], ( and ).
        text = re.sub(r'["\(\)\[\]]', "", text)
        # Replace consecutive spaces with a space.
        text = re.sub(" +", " ", text.strip())
        # Lowercase.
        text = text.lower()
        return text

    return fn


class WordErrorRateMetricCalculator(ModelSummaryAccumulator):
    """Computes word error rate (WER) for ASR models."""

    @config_class
    class Config(ModelSummaryAccumulator.Config):
        """Configures WordErrorRateMetricCalculator.

        Attributes:
            vocab: A config instantiating to a text vocab.
            text_normalizer: A config instantiating to a text normalization function.
            scorer: A config instantiating to a scoring function.
        """

        vocab: Required[InstantiableConfig[seqio.Vocabulary]] = REQUIRED
        text_normalizer: InstantiableConfig[Callable[[str], str]] = config_for_function(
            normalize_text
        )
        scorer: InstantiableConfig["WordErrorRateMetricCalculator.Scorer"] = config_for_function(
            lambda: compute_word_errors
        )

    @classmethod
    def default_config(cls):
        """Returns a calculator that invokes `model.beam_search_decode` to generate hypotheses."""
        return super().default_config().set(model_method="beam_search_decode")

    class Scorer(Protocol):
        """A function that computes word errors."""

        def __call__(self, hypotheses: list[str], reference: list[str]) -> WordErrors:
            ...

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: Nested[PartitionSpec],
    ):
        super().__init__(
            cfg, parent=parent, model=model, model_param_partition_specs=model_param_partition_specs
        )
        cfg = self.config
        self._vocab = cfg.vocab.instantiate()
        self._scorer = cfg.scorer.instantiate()
        self._text_normalizer = cfg.text_normalizer.instantiate()

    def forward(
        self,
        input_batch: Nested[Tensor],
        *,
        model_params: Nested[Tensor],
        state: Nested[Tensor],
    ) -> dict[str, Nested[Tensor]]:
        outputs = self._jit_forward(model_params, state["prng_key"], input_batch)
        self._compute_word_error_rate(input_batch, outputs["per_example"])
        return dict(state=dict(prng_key=outputs["replicated"]["prng_key"]), output={})

    def _forward_in_pjit(self, *args, **kwargs) -> dict[str, Nested[Tensor]]:
        out = super()._forward_in_pjit(*args, **kwargs)
        return dict(
            replicated=dict(prng_key=out["replicated"]["prng_key"]), per_example=out["per_example"]
        )

    def _per_example_outputs(self, model_outputs: Nested[Tensor]) -> Nested[Tensor]:
        return model_outputs

    def _compute_word_error_rate(
        self, input_batch: Nested[Tensor], per_example_outputs: DecodeOutputs
    ) -> Nested[Tensor]:
        """Computes WER metrics.

        Args:
            input_batch: A nested Tensor containing "target_labels", a Tensor of shape
                [batch_size, max_sequence_length] representing the reference text tokens.
                Out of vocab labels will be mapped to `vocab.pad_id`.
            per_example_outputs: Per-example model decodings (as produced by invoking
                `model_method`). Outputs will be replicated for metric calculation.

        Returns:
            An empty dict. Metrics will be directly accumulated via `metric_accumulator`.
        """
        references = replicate_to_local_data(input_batch["target_labels"])
        hypotheses = replicate_to_local_data(per_example_outputs.sequences[:, 0])

        # Map out-of-vocab labels to pad_id.
        references: Tensor = jnp.where(references < 0, self._vocab.pad_id, references)
        assert references.shape[0] == hypotheses.shape[0]

        # Batch decode the hypotheses and references.
        hypotheses = self._vocab.decode_tf(hypotheses).numpy()
        references = self._vocab.decode_tf(references).numpy()

        for hyp, ref in zip(hypotheses, references):
            hyp_words = self._text_normalizer(hyp.decode("utf-8")).split()
            ref_words = self._text_normalizer(ref.decode("utf-8")).split()
            num_words = len(ref_words)
            denom = max(1.0, num_words)
            metrics = self._scorer(hyp_words, ref_words)
            # NOTE: we drop utterance of empty references in WER metrics.
            # TODO(zhiyunlu): investigate whether we should keep empty references.
            self._metric_accumulator.update(
                {
                    "word_errors/wer": WeightedScalar(metrics.num_total / denom, num_words),
                    "word_errors/deletions": WeightedScalar(
                        metrics.num_deletions / denom, num_words
                    ),
                    "word_errors/insertions": WeightedScalar(
                        metrics.num_insertions / denom, num_words
                    ),
                    "word_errors/substitutions": WeightedScalar(
                        metrics.num_substitutions / denom, num_words
                    ),
                    "word_errors/sentence_accuracy": WeightedScalar(
                        int(metrics.num_total == 0), int(num_words > 0)
                    ),
                }
            )

        return {}
