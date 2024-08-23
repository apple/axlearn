# Copyright © 2024 Apple Inc.

"""Utilities for loading pre-trained Hugging Face models."""
from collections.abc import Sequence
from typing import Any, Callable

from axlearn.common.config import config_for_function
from axlearn.common.state_builder import HuggingFacePreTrainedBuilder
from axlearn.huggingface.hf_module import download_hf_models_from_remote


def auto_model_from_pretrained(  # pytype: disable=name-error
    model_name_or_path: str,
) -> "AutoModel":
    """Downloads and loads pre-trained HF model.

    Args:
        model_name_or_path: Model name or local or GCS path to the pre-trained model.

    Returns:
        A Hugging Face transformers PreTrainedModel.
    """
    # Lazily import to avoid introducing a dependency otherwise.
    # pylint: disable-next=import-outside-toplevel
    from transformers import AutoModel

    if model_name_or_path.startswith("gs://"):
        model_name_or_path = download_hf_models_from_remote(model_name_or_path)

    return AutoModel.from_pretrained(model_name_or_path)


def hf_pretrained_builder_config(
    *,
    model_name_or_path: str,
    target_scope: Sequence[str] = tuple(),
    source_scope: Sequence[str] = ("encoder",),
    from_pretrained_fn: Callable[[str], Any] = auto_model_from_pretrained,
) -> HuggingFacePreTrainedBuilder.Config:
    """Constructs a HuggingFacePreTrainedBuilder config to initialize from HF models.

    The builder will replace the target model's parameters under
    target_scope1->target_scope2->... to the HF model's parameters under
    source_scope1->source_scope2->...

    Args:
        model_name_or_path: Model name or location of the model's artifacts folder.
        target_scope: A list of strings with multiple scope names.
            If empty, it means the whole model state parameters will be replaced.
        source_scope: A list of strings with multiple scope names.
            If empty, it means the whole HF model parameters will be used for replacement.
        from_pretrained_fn: A function that takes a model identifier and returns an HF model.

    Returns:
        HuggingFacePreTrainedBuilder config to initialize from HF models.
    """
    builder = HuggingFacePreTrainedBuilder.default_config().set(
        hf_layer_config=config_for_function(from_pretrained_fn).set(
            model_name_or_path=model_name_or_path
        ),
        target_scope=target_scope,
        source_scope=source_scope,
    )
    return builder
