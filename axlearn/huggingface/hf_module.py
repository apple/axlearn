"""HuggingFace module wrappers."""
import json
import os
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import jax.numpy as jnp
import jax.random
import tensorflow as tf
from absl import logging
from flax.core.frozen_dict import freeze, unfreeze
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from axlearn.common.adapter_flax import config_for_flax_module
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, Required, RequiredFieldMissingError, config_class
from axlearn.common.module import Module, NestedTensor  # pylint: disable=unused-import
from axlearn.common.utils import Tensor

HF_MODULE_KEY = "hf_module"


def get_hf_models_cache_dir() -> Path:
    """Returns the directory where HF models are cached.

    The directory is not guaranteed to exist.
    """
    hf_models_cache_dir = (
        Path(os.environ["HOME"]) / ".cache" / "axlearn" / "huggingface" / "models"
        if "HF_MODELS_CACHE_DIR" not in os.environ
        else Path(os.environ["HF_MODELS_CACHE_DIR"])
    )
    return hf_models_cache_dir


def download_hf_models_from_gs(gs_path: str) -> str:
    """Downloads HuggingFace model artifacts from gs buckets.

    Args:
        gs_path: Model artifacts location.

    Returns:
        Local path of downloaded models.
    """
    model_name = gs_path.rstrip("/").split("/")[-1]
    cache_dir = get_hf_models_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_pretrained_model_path = cache_dir / model_name
    if not local_pretrained_model_path.exists():
        local_pretrained_model_path.mkdir(parents=True)
        for filename in tf.io.gfile.listdir(gs_path):
            logging.info(
                "Downloading %s to %s",
                os.path.join(gs_path, filename),
                local_pretrained_model_path / filename,
            )
            tf.io.gfile.copy(
                os.path.join(gs_path, filename),
                str(local_pretrained_model_path / filename),
            )
    return str(local_pretrained_model_path)


class HfModuleWrapper(BaseModel, ABC):
    """A wrapper for Hugging Face Flax modules so that they can be used within AXLearn.

    This is the super class for all Hugging Face Flax module wrappers, such as
    `HfSequenceClassificationWrapper`.

    Note that it wraps a Hugging Face Flax linen `nn.Module`, such as `FlaxBertModule`, and not a
    Hugging Face model, such as `FlaxBertPreTrainedModel`.

    The Hugging Face Flax module is added as a child under HF_MODULE_KEY.

    Initializing parameters of this module will create a local cache directory
    ~/.cache/axlearn/huggingface/models and download Hugging Face pretrained models to the local
    cache directory if they do not exist.

    The location of the local cache directory can be overridden by the env var HF_MODELS_CACHE_DIR.
    """

    @config_class
    class Config(BaseModel.Config):
        """Configures HfModuleWrapper."""

        dtype: Required[jnp.dtype] = REQUIRED
        # Type of HF pretrained model.
        hf_model_type: Required[Type[FlaxPreTrainedModel]] = REQUIRED
        # A local or GCS path to the Hugging Face model directory.
        pretrained_model_path: Optional[str] = None
        # Initialize model parameters from PyTorch checkpoint rather than Flax checkpoint.
        from_pt: Optional[bool] = False
        # hf_config is typically constructed from config.json inside the pretrained_model_path.
        # When pretrained_model_path is specified, hf_config is optional; it's required otherwise.
        # Both pretrained_model_path and hf_config can also be specified, in which case
        # hf_config will override fields in pretrained_model_path's config.json.
        # This HF config does not adhere to the composability
        # principle, but we use this as-is to reduce the amount of code to use the HF module.
        hf_config: Optional[PretrainedConfig] = None
        # Keys to skip when copying weights from pre-trained models.
        # Key names refer to dictionary key names in the model's config.json, and has no dots.
        pretrained_keys_to_skip: Optional[Sequence[str]] = []

    def __init__(self, cfg: Config, *, parent: Optional["Module"]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        if cfg.pretrained_model_path is not None:
            hf_config_cls = cfg.hf_model_type.config_class
            with tf.io.gfile.GFile(os.path.join(cfg.pretrained_model_path, "config.json")) as f:
                final_hf_config = hf_config_cls(**json.load(f))

            if cfg.hf_config is not None:
                for k, v in cfg.hf_config.to_dict().items():
                    if v != getattr(final_hf_config, k):
                        self.vlog(
                            1,
                            "Overriding hf_config %s=%s, previously it was %s.",
                            k,
                            v,
                            getattr(final_hf_config, k),
                        )
                        setattr(final_hf_config, k, v)
        elif cfg.hf_config is not None:
            final_hf_config = cfg.hf_config
        else:
            raise RequiredFieldMissingError(
                "pretrained_model_path is required when hf_config is not specified."
            )

        self._hf_config = final_hf_config
        self._add_child(
            HF_MODULE_KEY,
            config_for_flax_module(
                cfg.hf_model_type.module_class, self.create_dummy_input_fn()
            ).set(
                create_module_kwargs=dict(config=self._hf_config, dtype=cfg.dtype),
            ),
        )

    def initialize_parameters_recursively(
        self, prng_key: jax.random.KeyArray, *, prebuilt: Optional[NestedTensor] = None
    ) -> NestedTensor:
        if self._use_prebuilt_params(prebuilt):
            return prebuilt
        params = super().initialize_parameters_recursively(prng_key, prebuilt=prebuilt)
        cfg = self.config
        if cfg.pretrained_model_path is not None:
            if cfg.pretrained_model_path.startswith("gs://"):
                # Cache the model locally, if it is not already cached.
                model_name = cfg.pretrained_model_path.rstrip("/").split("/")[-1]
                cache_dir = get_hf_models_cache_dir()
                cache_dir.mkdir(parents=True, exist_ok=True)
                local_pretrained_model_path = cache_dir / model_name
                if not local_pretrained_model_path.exists():
                    local_pretrained_model_path.mkdir(parents=True)
                    for filename in tf.io.gfile.listdir(cfg.pretrained_model_path):
                        logging.info(
                            "Downloading %s to %s",
                            os.path.join(cfg.pretrained_model_path, filename),
                            local_pretrained_model_path / filename,
                        )
                        tf.io.gfile.copy(
                            os.path.join(cfg.pretrained_model_path, filename),
                            str(local_pretrained_model_path / filename),
                        )
                else:
                    logging.info(
                        "Found cache %s, skipping downloading model from %s",
                        local_pretrained_model_path,
                        cfg.pretrained_model_path,
                    )
            else:
                local_pretrained_model_path = cfg.pretrained_model_path
            hf_model = cfg.hf_model_type.from_pretrained(
                local_pretrained_model_path, from_pt=cfg.from_pt
            )
            hf_module_params = unfreeze(params[HF_MODULE_KEY])
            hf_module_params["params"] = {
                key: (
                    value
                    if key not in cfg.pretrained_keys_to_skip
                    else hf_module_params["params"][key]
                )
                for key, value in hf_model.params.items()
            }
            params[HF_MODULE_KEY] = freeze(hf_module_params)
        return params

    def _dummy_input_kwargs(self) -> Dict[str, Optional[Tensor]]:  # pylint: disable=no-self-use
        """Returns a dictionary of kwargs to pass to linen.Module.init."""
        return dict(
            # The first dim is batch size.
            input_ids=jnp.zeros([1, 8], dtype=jnp.int32),
            attention_mask=jnp.zeros([1, 8], dtype=jnp.int32),
            token_type_ids=jnp.zeros([1, 8], dtype=jnp.int32),
            position_ids=jnp.zeros([1, 8], dtype=jnp.int32),
        )

    def create_dummy_input_fn(self) -> Callable[[], Tuple[Sequence, Dict]]:
        """Returns a function that returns (args, kwargs) for linen.Module.init."""

        def create_dummy_input():
            return tuple(), self._dummy_input_kwargs()

        return create_dummy_input

    def _forward_kwargs(self, input_batch: Dict[str, Tensor]) -> Dict[str, Any]:
        """Returns a dictionary of kwargs for HF module's forward __call__.

        Args:
            input_batch: a dict with the following entries:
                input_ids: an int Tensor of shape [batch_size, seq_len]
                    representing indices of input sequence tokens in the vocabulary.
                    Values should be in the range [0, vocab_size].
                token_type_ids: an optional int Tensor of shape [batch_size, seq_len]
                    indicating the token type.

        Returns:
            A dictionary of kwargs for HF module's forward __call__.
        """
        input_ids: Tensor = input_batch["input_ids"]
        # In HF attention, mask 1 represents a valid position and 0 represents a padding position.
        # See https://huggingface.co/docs/transformers/glossary#attention-mask.
        attention_mask = jnp.not_equal(input_ids, self._hf_config.pad_token_id).astype(
            input_ids.dtype
        )
        _, dropout_subkey = jax.random.split(self.prng_key)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=input_batch.get("token_type_ids"),
            position_ids=None,
            # Whether we should apply dropout mask (deterministic=False) or not (True).
            deterministic=not self.is_training,
            rngs={"params": self.prng_key, "dropout": dropout_subkey},
        )

    def predict(self, input_batch: Dict[str, Tensor]) -> NestedTensor:
        """Runs model prediction with the given inputs.

        Args:
            input_batch: a dict with at minimum the following entries:
                input_ids: an int Tensor of shape [batch_size, seq_len]
                    representing indices of input sequence tokens in the vocabulary.
                    Values should be in the range [0, vocab_size].
                token_type_ids: an optional int Tensor of shape [batch_size, seq_len]
                    indicating the token type.

        Returns:
            A NestedTensor containing model-specific auxiliary outputs.
        """
        hf_module = self.children[HF_MODULE_KEY]
        hf_output = hf_module(**self._forward_kwargs(input_batch))
        return hf_output
