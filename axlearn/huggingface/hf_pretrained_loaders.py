# Copyright © 2023 Apple Inc.

"""HuggingFace pretraiend loaders."""
import torch

from axlearn.huggingface.hf_module import download_hf_models_from_gs


def load_pytorch_deberta_v2_from_pretrained(model_path: str):
    """Downloads model_path and loads pre-trained DeBERTa v2 model from path.

    Args:
        model_path: Path to the pre-trained model.

    Returns:
        A Hugging Face DebertaV2Model.

    Raises:
        NotImplementedError: The model path scheme is unsupported.
    """
    # Lazily import to avoid introducing a dependency otherwise.
    # pylint: disable-next=import-outside-toplevel
    from transformers import DebertaV2Model

    if model_path.startswith("gs://"):
        model_path = download_hf_models_from_gs(model_path)
    else:
        raise NotImplementedError(f"Unsupported scheme for model path: {model_path}.")

    return DebertaV2Model.from_pretrained(model_path)


def load_pytorch_mt0_from_pretrained(model_path: str):
    """Downloads model_path and loads pre-trained mT0/mT5 model from path.

    Args:
        model_path: Path to the pre-trained model.

    Returns:
        A Hugging Face t0 model.

    Raises:
        NotImplementedError: The model path scheme is unsupported.
    """
    # Lazily import to avoid introducing a dependency otherwise.
    # pylint: disable-next=import-outside-toplevel
    from transformers.models.mt5 import modeling_mt5

    if model_path.startswith("gs://"):
        model_path_local = download_hf_models_from_gs(model_path)
        return modeling_mt5.MT5ForConditionalGeneration.from_pretrained(
            model_path_local, torch_dtype=torch.bfloat16
        )

    return modeling_mt5.MT5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
