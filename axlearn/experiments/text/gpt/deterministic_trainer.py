# Copyright © 2024 Apple Inc.

"""GPT models on Deterministic datasets."""

from typing import Callable, Dict

from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.input_tf_data import tfds_dataset
from axlearn.experiments.text.common import vocab
from axlearn.experiments.text.gpt import honeycrisp
from axlearn.experiments.text.gpt.common import REPLACE_NEWLINES_WITH, tfds_input
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

_SENTENCEPIECE_MODEL_NAME = {
    48 * 1024: "bpe_48k_honeycrisp.model",
}

_TRAINING_DATASETS = {
    "pajama-15t-49k": dict(
        dataset_name="pajama_honeycrisp_15t_202408072230",
        vocab_size=48 * 1024,
        max_sequence_length=4096,
    ),
}


def _eval_input_sources(
    *, vocab_size: int, max_sequence_length: int
) -> Dict[str, InstantiableConfig]:
    return {
        name: config_for_function(tfds_input).set(
            dataset_name=dataset_name,
            split=split,
            is_training=False,
            vocab_cfg=config_for_function(vocab).set(
                sentencepiece_model_name=_SENTENCEPIECE_MODEL_NAME[vocab_size],
            ),
            max_sequence_length=max_sequence_length,
            replace_newlines_with=REPLACE_NEWLINES_WITH,
        )
        for dataset_name, name, split in (
            ("pile/openwebtext_2:1.0.0", "openwebtext_2", "test[:5000]"),
        )
    }


def _train_input_source_fn(
    *, dataset_name: str, **dataset_info
) -> Callable[[int, int], InstantiableConfig]:
    def fn(vocab_size: int, max_sequence_length: int) -> InstantiableConfig:
        if vocab_size != dataset_info["vocab_size"]:
            raise ValueError(
                f"vocab_size mismatch for {dataset_name}: {vocab_size} vs. {dataset_info}"
            )
        if max_sequence_length > dataset_info["max_sequence_length"]:
            raise ValueError(
                f"max_sequence_length mismatch for {dataset_name}: "
                f"{max_sequence_length} vs. {dataset_info}"
            )
        return config_for_function(tfds_dataset).set(
            dataset_name=dataset_name,
            split="train",
            is_training=True,
            train_shuffle_buffer_size=0,
            train_shuffle_files=False,
        )

    return fn


def named_trainer_configs() -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    config_map = {}
    for name, dataset_info in _TRAINING_DATASETS.items():
        for k, fn in honeycrisp.trainer_configs(
            _train_input_source_fn(**dataset_info), _eval_input_sources
        ).items():
            cfg = fn()

            def trainer_config_fn(cfg=cfg) -> InstantiableConfig:
                return cfg.clone(save_input_iterator=True)

            config_map[f"deterministic-{name}-{k}"] = trainer_config_fn
    return config_map
