# Copyright © 2025 Apple Inc.

"""Builds trainer states from Hugging Face models."""

import functools
from importlib import import_module
from typing import Optional, Sequence, Union

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)
from axlearn.common.module import Module
from axlearn.common.state_builder import Builder, traverse_and_set_target_state_parameters
from axlearn.common.utils import check_param_shape_alignment, flatten_items


def torch_to_axlearn_converter(
    module: str = "axlearn.common.param_converter",
    dst_layer: Optional[ConfigOr[Union[BaseLayer, type]]] = None,
):
    """See HuggingFacePreTrainedBuilder.converter."""
    # Lazily import to avoid introducing a dependency otherwise.
    # TODO(bwzhang@): The fairseq layer is not supported in TPU.
    # pylint: disable-next=import-outside-toplevel
    torch_to_axlearn = import_module(module).torch_to_axlearn
    return functools.partial(torch_to_axlearn, dst_layer=maybe_instantiate(dst_layer))


class HuggingFacePreTrainedBuilder(Builder):
    """Replaces model state with params from a Hugging Face layer.

    This builder supports replacing parameters for parts of the module.
    To use this builder, the user can call spec_to_config function with spec defined as
    "hf_pretrained:model_name:target_scope1/target_scope2:source_scope1/source_scope2"

    The user can also configure and initialize the builder by setting the hf_layer_config,
    target_scope, and source_scope.
    The hf_layer_config is required. The target and source scopes are optional.

    In the following context, the target refers to the model state.
    The source refers to the HF model.

    The builder will replace the target model's parameters under
    target_scope1->target_scope2->... to the HF model's parameters under
    source_scope1->source_scope2->...
    target[target_scope1][target_scope2] = hf_model[source_scope1][source_scope2]
    """

    SPEC_PREFIX = "hf_pretrained:"

    @config_class
    class Config(Builder.Config):
        """Configures HuggingFacePreTrainedBuilder."""

        # A config that instantiates to a Hugging Face layer.
        hf_layer_config: Required[InstantiableConfig] = REQUIRED
        # The target_scope is defined as a list of strings with multiple scope names.
        # If target_scope == [], it means the whole model state parameters will be replaced.
        target_scope: Sequence[str] = []
        # The source_scope is defined as a list of strings with multiple scope names.
        # If source_scope == [], it means the whole HF model parameters will be
        # used for replacement.
        source_scope: Sequence[str] = []
        # A config that instantiates to a param converter, which takes a torch module and emits
        # axlearn model params.
        converter: InstantiableConfig = config_for_function(torch_to_axlearn_converter)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._converter = cfg.converter.instantiate()

    @classmethod
    def spec_to_config(cls, spec: str) -> Config:
        # Lazily import to avoid introducing a dependency otherwise.
        # pylint: disable-next=import-outside-toplevel
        from transformers import AutoModel

        spec_split = spec.split(":")
        if len(spec_split) != 3:
            raise ValueError(
                "The spec should contains three components, model_name,"
                "source_scope, and target_scope. The source_scope and"
                "target scope can be empty. For example, 'hf_pretrained:Model::'"
                "This means the model_name is Model. The source_scope and"
                "target_scope are not set."
            )

        spec = spec_split[0]
        target_scope = spec_split[1]
        if target_scope == "":
            target_scope = []  # directly replacing empty string with [].
        else:
            target_scope = target_scope.split("/")
        source_scope = spec_split[2]
        if source_scope == "":
            source_scope = []
        else:
            source_scope = source_scope.split("/")

        def auto_from_pretrained(model_name: str):
            return AutoModel.from_pretrained(model_name)

        return cls.default_config().set(
            hf_layer_config=config_for_function(auto_from_pretrained).set(model_name=spec),
            target_scope=target_scope,
            source_scope=source_scope,
        )

    def input_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSORS

    def __call__(self, state: Builder.State) -> Builder.State:
        """Copies model params from Hugging Face layer."""
        cfg = self.config

        hf_layer = cfg.hf_layer_config.instantiate()
        model_params = self._converter(hf_layer)

        restored_model_state = traverse_and_set_target_state_parameters(
            target_state=state.trainer_state.model,
            target_scope=cfg.target_scope,
            source_params=model_params,
            source_scope=cfg.source_scope,
        )
        # Check the shape between the original state and the new state.
        shape_check_msg = check_param_shape_alignment(
            state.trainer_state.model, restored_model_state
        )
        if shape_check_msg:
            raise ValueError(shape_check_msg)

        # The restored_model_state should return a full model state. Part of the model parameters
        # might be set to HF model's parameters.
        restored_state = state.trainer_state._replace(model=restored_model_state)

        built_keys = state.built_keys.union({key for key, _ in flatten_items(restored_state)})
        return Builder.State(step=0, trainer_state=restored_state, built_keys=built_keys)
