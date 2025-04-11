# Copyright © 2023 Apple Inc.

# pylint: disable=too-many-lines
"""A library to build trainer states, e.g., from checkpoints of other models."""

import copy
import enum
import functools
import re
from collections.abc import Mapping, Sequence
from importlib import import_module
from tempfile import mkdtemp
from typing import Any, Optional, Union

import jax
import optax
from absl import logging
from chex import dataclass  # tree_map-friendly dataclass
from jax import numpy as jnp
from jax._src.tree_util import KeyEntry
from jax.experimental.pjit import pjit

from axlearn.common import utils
from axlearn.common.base_layer import BaseLayer
from axlearn.common.checkpointer import (
    CheckpointValidationType,
    StateStorage,
    TensorStoreStateStorage,
    build_step_dir,
    check_state_structure,
    parse_step_from_dir,
)
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
    maybe_instantiate,
)
from axlearn.common.input_fake import EmptyInput
from axlearn.common.module import Module
from axlearn.common.optimizer_base import OptStateSpec
from axlearn.common.optimizers import ParamEmaState
from axlearn.common.utils import (
    NestedTensor,
    NestedTensorSpec,
    PartitionSpec,
    Tensor,
    _key_entry_to_str,
    check_param_shape_alignment,
    flatten_items,
    get_data_dir,
    infer_mesh_shape,
    set_data_dir,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn


class Builder(Module):
    """An abstract class for building trainer states."""

    class StateType(enum.Enum):
        TENSORS = "tensors"
        TENSOR_SPECS = "tensor_specs"

    @config_class
    class Config(Module.Config):
        pass

    @dataclass
    class State:
        # The current step.
        step: int
        # The trainer state. A nested structure with Tensors or TensorSpec as leaf nodes.
        trainer_state: Union[NestedTensor, NestedTensorSpec]
        # A set of paths in `trainer_state` that have been built by this builder.
        built_keys: set[str]

    def input_state_type(self) -> StateType:
        raise NotImplementedError(type(self))

    # Use a different __call__() API from Module.__call__.
    # pylint: disable-next=arguments-differ
    def __call__(self, state: State) -> State:
        """Builds trainer state.

        Args:
            state: The input state.
                If `self.input_state_type()` returns TENSORS,
                `state.trainer_state` is expected to have Tensors as leaf nodes.
                If `self.input_state_type()` returns TENSOR_SPECS,
                `state.trainer_state` is expected to have TensorSpecs as leaf nodes.

        Returns:
            Built state.
        """
        raise NotImplementedError(type(self))


class ChainBuilder(Builder):
    """A Builder that chains a sequence of Builders.

    The chained builder's output state.trainer_state tree structure is validated against
    the desired validation type if set.
    """

    @config_class
    class Config(Builder.Config):
        builders: Required[Sequence[Builder.Config]] = REQUIRED
        # If validation is None, no structure check will be enforced.
        validation: Optional[CheckpointValidationType] = CheckpointValidationType.EXACT

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._builders = []

        for i, builder_cfg in enumerate(cfg.builders):
            self._builders.append(self._add_child(f"builder{i:03d}", builder_cfg))

    def input_state_type(self) -> Builder.StateType:
        return self._builders[0].input_state_type()

    def __call__(self, state: Builder.State) -> Builder.State:
        cfg = self.config
        initial_trainer_state_list = list(
            (key, None) for key, _ in flatten_items(state.trainer_state)
        )

        for builder in self._builders:
            logging.info("Applying builder: %s", builder)
            state = builder(state)
        # Check the model tree structure if set.
        if cfg.validation is not None:
            check_state_structure(
                initial_trainer_state_list,
                list((key, None) for key, _ in flatten_items(state.trainer_state)),
                validation=cfg.validation,
            )
        return state


class Converter(Module):
    """Converts builder state from one version to another.

    This can be used to migrate legacy trainer state structures to updated structures in the case
    of backwards-incompatible changes. Often used in `RestoreAndConvertBuilder`.
    """

    @config_class
    class Config(Module.Config):
        pass

    def target_state_type(self) -> Builder.StateType:
        """Whether `target` given to `target_to_source` will contain Tensors or TensorSpecs

        ...as leaf nodes.
        """
        raise NotImplementedError(type(self))

    def target_to_source(self, target: Builder.State) -> tuple[Builder.State, Any]:
        """Converts a target (e.g. new) state to a source (e.g. legacy) state.
        This should be called before `source_to_target`.

        Args:
            target: State to be converted. Should contain Tensors or TensorSpecs as leaf nodes if
                `target_state_type()` returns TENSORS or TENSORSPECS, respectively.

        Returns:
            A tuple of (converted state, auxiliary outputs).
        """
        raise NotImplementedError(type(self))

    def source_to_target(self, source: Builder.State, aux: Any) -> Builder.State:
        """Converts a source (e.g. legacy) state to a target (e.g. new) state.
        This should be called after `target_to_source`.

        Args:
            source: State to be converted.
            aux: Auxiliary outputs from corresponding `target_to_source` call.

        Returns:
            The converted state.
        """
        raise NotImplementedError(type(self))


class ChainConverter(Converter):
    """A Converter that chains a sequence of Converters."""

    @config_class
    class Config(Converter.Config):
        converters: Required[Sequence[Converter.Config]] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._converters = []
        for i, converter_cfg in enumerate(cfg.converters):
            self._converters.append(self._add_child(f"converter{i:03d}", converter_cfg))

    def target_state_type(self) -> Builder.StateType:
        for converter in self._converters:
            if converter.target_state_type() == Builder.StateType.TENSORS:
                return Builder.StateType.TENSORS
            else:
                assert (
                    converter.target_state_type() == Builder.StateType.TENSOR_SPECS
                ), converter.target_state_type()
        return Builder.StateType.TENSOR_SPECS

    def target_to_source(self, target: Builder.State) -> tuple[Builder.State, Any]:
        """Converts a target state by applying a sequence of converters.

        Args:
            target: State to be converted.

        Returns:
            A tuple of (converted state, stack of auxiliary outputs).
        """
        aux_stack = []
        # Apply converters in reverse, maintaining a stack of aux outputs.
        for converter in self._converters[::-1]:
            logging.info("Converting target to source: %s", type(converter))
            target, aux = converter.target_to_source(target)
            aux_stack.append(aux)
        return target, aux_stack

    def source_to_target(self, source: Builder.State, aux: Any) -> Builder.State:
        """Converts a source state by applying a sequence of converters.

        Args:
            source: State to be converted.
            aux: Stack of auxiliary outputs from `target_to_source`.

        Returns:
            The converted state.
        """
        for converter in self._converters:
            logging.info("Converting source to target: %s", type(converter))
            source = converter.source_to_target(source, aux.pop())
        return source


class MergeStateSelection(str, enum.Enum):
    TARGET = "TARGET"  # Keep the target state unchanged.
    SOURCE = "SOURCE"  # Load from the source state.


class MergeStateConverter(Converter):
    """A converter allowing merging of states. The state is cloned in target_to_source and returned
    in the aux variable, and passed back in source_to_target.

    selection_regexes is a list of (regex, MergeStateSelection.TARGET | MergeStateSelection.SOURCE).
    If the regex matches the path to a leaf, the TARGET or SOURCE state is used for that leaf in the
    resulting tree. If no regexes match a path, the SOURCE state is chosen.

    Arguments:
        selection_regexes: a list of (regex, MergeStateSelection.TARGET |
            MergeStateSelection.SOURCE) used to choose if the target or source state is used in
            the resulting tree.
    """

    @config_class
    class Config(Converter.Config):
        selection_regexes: list[tuple[str, Union[str, MergeStateSelection]]] = []

    def __init__(self, cfg: "MergeStateConverter.Config", *, parent: Optional[Module] = None):
        super().__init__(cfg, parent=parent)

        self.selection_regexes = [
            (re.compile(r), MergeStateSelection(s)) for r, s in cfg.selection_regexes
        ]

    def _selector(self, path, target, source):
        for regex, state in self.selection_regexes:
            if regex.match(path) is not None:
                if state == MergeStateSelection.TARGET:
                    return target
                else:
                    return source
        return source

    def target_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSORS

    def target_to_source(self, target: Builder.State) -> tuple[Builder.State, Builder.State]:
        return target, clone_tree(target)

    def source_to_target(self, source: Builder.State, aux: Builder.State) -> Builder.State:
        """Source is newly loaded state, aux is original state."""
        new_trainer_state = jax.tree.map(
            self._selector,
            utils.tree_paths(aux.trainer_state),
            aux.trainer_state,
            source.trainer_state,
        )
        return Builder.State(
            step=source.step, trainer_state=new_trainer_state, built_keys=source.built_keys
        )


class RestoreAndConvertBuilder(Builder):
    """Builds state by restoring a checkpoint and applying a Converter to the state.

    This can be useful for restoring legacy checkpoints.

    Conversion always happens via the following strategy:
    1. Roll-back to checkpoint-compatible state via `target_to_source`.
    2. Restore (build) checkpoint.
    3. Fast-forward to current state via `source_to_target`.
    """

    @config_class
    class Config(Builder.Config):
        builder: Required[Builder.Config] = REQUIRED
        converter: Required[Converter.Config] = REQUIRED

    @classmethod
    def spec_to_config(cls, spec: str) -> Config:
        cfg = cls.default_config()
        cfg.builder = cfg.builder.klass.spec_to_config(spec)
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("builder", cfg.builder)
        self._add_child("converter", cfg.converter)

    def input_state_type(self) -> Builder.StateType:
        return self.converter.target_state_type()

    def __call__(self, state: Builder.State) -> Builder.State:
        input_state = state.trainer_state
        input_keys = {key for key, _ in flatten_items(input_state)}
        source_state, aux = self.converter.target_to_source(state)
        restored_state = self.builder(source_state)
        converted_state = self.converter.source_to_target(restored_state, aux)
        output_state = converted_state.trainer_state
        output_keys = {key for key, _ in flatten_items(output_state)}
        assert input_keys == output_keys, (
            f"input-only keys: {input_keys - output_keys} "
            f"output-only keys: {output_keys - input_keys}"
        )
        for (input_key, input_value), (output_key, output_value) in zip(
            flatten_items(input_state), flatten_items(output_state)
        ):
            assert input_key == output_key, f"Key mismatch for {input_key} vs. {output_key}"
            assert tuple(input_value.shape) == tuple(
                output_value.shape
            ), f"Shape mismatch for {input_key}: {input_value.shape} vs. {output_value.shape}"

        return converted_state


def is_dict(x: Any, has_key: Union[str, Sequence[str]]) -> bool:
    """Return whether x is a dict with specified key(s)."""
    if isinstance(has_key, str):
        has_key = [has_key]
    # pylint: disable=use-a-generator
    return isinstance(x, dict) and all([key in x for key in has_key])


def clone_tree(in_tree: NestedTensor) -> NestedTensor:
    # Tensors are not pickleable, so deepcopy isn't possible.
    # They should be immutable and exist beyond the lifetime of this fn, however,
    # so we copy by reference.
    return jax.tree.map(lambda x: x if isinstance(x, Tensor) else copy.deepcopy(x), in_tree)


# pylint: disable-next=abstract-method
class BaseStateStorageBuilder(Builder):
    """Builds state by restoring from a checkpoint."""

    @config_class
    class Config(Builder.Config):
        """Configures BaseStateStorageBuilder.

        Attributes:
            base_dir: Base directory that contains checkpoints of a trainer, usually containing
                subdirs, one for each checkpointed step.
            step: Step number to load. Required if `base_dir` is specified.
            validation: Checkpoint validation type.
            concurrent_gb: Memory limit of the in-flight reads.
        """

        base_dir: Optional[str] = None
        step: Optional[int] = None
        validation: CheckpointValidationType = CheckpointValidationType.EXACT
        concurrent_gb: int = 32

    def input_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSOR_SPECS


class TensorStoreStateStorageBuilder(BaseStateStorageBuilder):
    """Builds state by restoring from TensorStoreStateStorage."""

    SPEC_PREFIX = "tensor_store_state_storage:"

    @config_class
    class Config(BaseStateStorageBuilder.Config):
        """Configures TensorStoreStateStorageBuilder.

        Attributes:
            dir: Full checkpoint path that contains a checkpoint of a single step.
                This is supported for backward compatibility purposes.
                It's recommended to use `base_dir` and `step` if possible.
            storage: Config for the underlying storage used during checkpoint loading. Defaults
                to `TensorStoreStateStorage`.
        """

        dir: Optional[str] = None
        storage: StateStorage.Config = TensorStoreStateStorage.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        if not (cfg.dir is None) ^ (cfg.base_dir is None and cfg.step is None):
            raise ValueError("Error in config. You must either specify dir or base_dir and step.")

    @classmethod
    def spec_to_config(cls, spec: str) -> Config:
        return cls.default_config().set(dir=spec)

    def __call__(self, state: Builder.State) -> Builder.State:
        """Restores state from TensorStoreStateStorage.

        Args:
            state: Uninitialized state.

        Returns:
            The restored state.
        """
        cfg: TensorStoreStateStorageBuilder.Config = self.config
        if cfg.base_dir:
            ckpt_dir = build_step_dir(cfg.base_dir, step=cfg.step)
            step = cfg.step
        else:
            ckpt_dir = cfg.dir
            step = parse_step_from_dir(cfg.dir)
        cfg.storage.max_concurrent_restore_gb = cfg.concurrent_gb
        storage = cfg.storage.instantiate()
        restored_state = storage.restore_from_dir(
            step=step,
            state=state.trainer_state,
            ckpt_dir=ckpt_dir,
            validation=cfg.validation,
        )
        built_keys = state.built_keys.union({key for key, _ in flatten_items(restored_state)})
        return Builder.State(step=step, trainer_state=restored_state, built_keys=built_keys)


# pylint: disable=abstract-method
class BaseConverterFromPretrainedModel(Converter):
    """A base Converter class for converting from a pretrained model.

    Note that `target_to_source` does not produce NestedTensors but NestedTensorSpec, which
    are sufficient for loading checkpoints.
    """

    @config_class
    class Config(Converter.Config):
        # A config that instantiates to a TrainerConfigFn that defines the source pretrained model.
        source_trainer_config: Required[InstantiableConfig[TrainerConfigFn]] = REQUIRED
        source_data_dir: Optional[str] = None
        mesh_axis_names: Optional[Sequence[str]] = None
        mesh_shape: Optional[Sequence[int]] = None

    def target_to_source(self, target: Builder.State) -> tuple[Builder.State, Any]:
        """Produces state specs compatible with the pretrained checkpoint to be restored."""
        cfg = self.config
        # Initialize the model and learner states for the pretrained model.
        # pytype: disable=attribute-error
        cfg_fn: TrainerConfigFn = cfg.source_trainer_config.instantiate()

        with set_data_dir(cfg.source_data_dir or get_data_dir()):
            trainer_cfg = cfg_fn()
            logging.info(
                "Initialize model and learner states for the pretrained model: %s", trainer_cfg.name
            )
            trainer_cfg.dir = mkdtemp()
            trainer_cfg.mesh_axis_names = (
                cfg.mesh_axis_names or trainer_cfg.mesh_axis_names or ("data", "model")
            )
            trainer_cfg.mesh_shape = infer_mesh_shape(
                cfg.mesh_shape or trainer_cfg.mesh_shape or (len(jax.devices()), 1)
            )
            # Reset datasets and evalers for the pretrained model config.
            # This input is not used. Set global_batch_size to 0 by default.
            trainer_cfg.input = EmptyInput.default_config().set(global_batch_size=0)
            trainer_cfg.evalers = {}

            trainer = trainer_cfg.instantiate(parent=None)
            # pytype: enable=attribute-error
            source = Builder.State(
                step=trainer.step,
                trainer_state=trainer.trainer_state_specs,
                built_keys=set(),
            )
            return source, target


class BertSequenceClassificationHeadConverter(BaseConverterFromPretrainedModel):
    """Replaces a BertLMHead with a BertSequenceClassificationHead.

    Main use-case is to convert a pretrained BertModel checkpoint by dropping the MLM head and
    loading a task-specific finetuning head in its place. Note that we also reset optimizer state
    (e.g. accumulators) corresponding to the base pretrained model.
    """

    def target_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSORS

    def source_to_target(self, source: Builder.State, aux: Any) -> Builder.State:
        """Produces state compatible with the new classification head."""
        restored_model = jax.tree.map(
            lambda s, t: self._swap_heads(s, t) if self._is_bert_lm_head(s) else s,
            source.trainer_state.model,
            aux.trainer_state.model,
        )
        restored_state = source.trainer_state._replace(model=restored_model)
        # Reset old optimizer state following fairseq, e.g:
        # https://github.com/facebookresearch/fairseq/blob/acd9a53607d1e5c64604e88fc9601d0ee56fd6f1/examples/roberta/config/finetuning/cola.yaml#L21
        # https://github.com/facebookresearch/fairseq/blob/10b797a44f1d724465cd66ce1bb92d6b8fa052eb/fairseq/trainer.py#L587
        restored_state = restored_state._replace(learner=aux.trainer_state.learner)
        built_keys = source.built_keys.union({key for key, _ in flatten_items(restored_state)})
        return Builder.State(step=aux.step, trainer_state=restored_state, built_keys=built_keys)

    # pylint: disable-next=no-self-use
    def _is_bert_lm_head(self, state: dict[str, Any]) -> bool:
        return is_dict(state, "head") and is_dict(state["head"], ["transform", "output_bias"])

    # pylint: disable-next=no-self-use
    def _swap_heads(self, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        out = clone_tree(source)
        out["head"] = target["head"]
        return out


def traverse_and_set_target_state_parameters(
    *,
    target_state: dict[str, Any],
    target_scope: list[str],
    source_params: dict[str, Any],
    source_scope: list[str],
) -> dict[str, Any]:
    """Traverse the target model state and source params and set the new parameters.

    The function traverses the target state based on ``target_scope``.
    The target state parameters are set to ``source_params`` under ``source_scope``.

    Args:
        target_state: A nested dictionary containing
            *parameter_name: A Tensor containing the parameter.
        target_scope: A list of string defining the traverse path for the target state.
            For example, ["linear", "weight"] means the traverse path is linear->weight.
            With this target scope, the function will replace the parameters of
            target_state["linear"]["weight"] to the source_params.
        source_params: A nested dictionary containing
            *parameter_name: A DeviceArray containing the parameter.
        source_scope: A list of string defining the traverse path for the source params.
            With this scope, the function will use source_params under source scope to
            replace the target_state parameters.

    Returns:
        A nested dictionary containing
            *parameter_name: A Tensor containing the new parameter.
    """
    if len(target_scope) == 0:
        # Traverse the source_params using source_scope.
        while len(source_scope):
            source_params = source_params[source_scope.pop(0)]

        # Set the target_state with the source_params.
        def _to_target_sharding(src: Tensor, tgt: Tensor) -> Tensor:
            assert tuple(src.shape) == tuple(tgt.shape), f"{src.shape} vs. {tgt.shape}"
            return jax.make_array_from_callback(
                shape=tuple(tgt.shape),
                sharding=tgt.sharding,
                data_callback=lambda index: src[index],
            )

        target_state = jax.tree.map(_to_target_sharding, source_params, target_state)
    else:
        scope = target_scope.pop(0)  # The item should be popped from the bottom of the list.
        target_state[scope] = traverse_and_set_target_state_parameters(
            target_state=target_state[scope],
            target_scope=target_scope,
            source_params=source_params,
            source_scope=source_scope,
        )
    return target_state


class FlaxPretrainedBuilder(Builder):
    """Builds (partial) model state from supplied flax states.

    The function reads model parameters from configured state supplier, and
    sets the corresponding model state to these parameters.

    In the following context, the target refers to the model state.
    The source refers to the pretrained model.

    The builder will replace the target model's parameters under
    target_scope1->target_scope2->... with source parameters under
    source_scope1->source_scope2->...
    target[target_scope1][target_scope2] = source[source_scope1][source_scope2]

    Note that target[target_scope1][target_scope2] should address to a Flax module, not
    regular axlearn module.
    """

    @config_class
    class Config(Builder.Config):
        flax_state_supplier_config: Required[InstantiableConfig] = REQUIRED
        # The target_scope is defined as a list of strings with multiple scope names.
        # If target_scope == [], it means the whole model state parameters will be replaced.
        target_scope: Sequence[str] = []
        # The source_scope is defined as a list of strings with multiple scope names.
        # If source_scope == [], it means the whole diffusers model parameters will be
        # used for replacement.
        source_scope: Sequence[str] = []

    def input_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSORS

    def __call__(self, state: Builder.State) -> Builder.State:
        cfg = self.config
        source_params = cfg.flax_state_supplier_config.instantiate()

        state.step = 0

        parent_state = state.trainer_state.model
        if len(cfg.target_scope) == 0:
            target_flax_state = parent_state["params"]
        else:
            for scope in cfg.target_scope[:-1]:
                if scope not in parent_state:
                    raise ValueError()
                parent_state = parent_state[scope]
            target_flax_state = parent_state[cfg.target_scope[-1]]["params"]

        restored_target_state = traverse_and_set_target_state_parameters(
            target_state=target_flax_state,
            target_scope=[],
            source_params=source_params,
            source_scope=[],
        )

        # Check the shape between the original state and the new state.
        shape_check_msg = check_param_shape_alignment(target_flax_state, restored_target_state)
        if shape_check_msg:
            raise ValueError(shape_check_msg)

        if len(cfg.target_scope) == 0:
            new_trainer_state = state.trainer_state._replace(
                model={"params": restored_target_state}
            )
        else:
            parent_state[cfg.target_scope[-1]] = {"params": restored_target_state}
            new_trainer_state = state.trainer_state

        built_keys = state.built_keys.union({key for key, _ in flatten_items(new_trainer_state)})
        return Builder.State(step=0, trainer_state=new_trainer_state, built_keys=built_keys)


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


class ModelStateScopeConverter(BaseConverterFromPretrainedModel):
    """Restore model state from the source under the same scope or a new scope.

    This is commonly used when loading a pre-trained model as part of a new model, e.g. load
    a pre-trained backbone to a downstream model or load a pre-trained teacher to a
    distillation model.
    """

    @config_class
    class Config(BaseConverterFromPretrainedModel.Config):
        # Loads source model to the given `scope` in the target model.
        # Alternatively a mapping from target scopes to corresponding source scopes
        # can be provided. E.g. {"a/b": "c/d", "e": "f"}' loads 'a/b' from 'c/d' and 'e' from 'f'.
        # If None, restores to `target_trainer.model.`
        scope: Union[str, Mapping[str, str], None] = None

    @property
    def scopes(self) -> Mapping[str, str]:
        """Returns a mapping from target scopes to source scopes."""
        cfg = self.config
        return cfg.scope if isinstance(cfg.scope, Mapping) else {(cfg.scope or ""): ""}

    def target_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSOR_SPECS

    def target_to_source(self, target: Builder.State) -> tuple[Builder.State, Any]:
        """Produces state compatible with the pretrained checkpoint to be restored."""
        source, aux = super().target_to_source(target)
        source_model = source.trainer_state.model
        pruned_model = type(source_model)()
        for source_scope in self.scopes.values():
            pruned_model = utils.copy_recursively(
                source=source_model,
                target=pruned_model,
                path=source_scope,
            )
        # Prune model and reset learner and prng_key.
        pruned_trainer_state = source.trainer_state._replace(
            model=pruned_model,
            learner={},
            prng_key={},
        )
        logging.info(
            "pruned_state=%s", sorted(key for key, _ in flatten_items(pruned_trainer_state))
        )
        source.trainer_state = pruned_trainer_state
        return source, aux

    def source_to_target(self, source: Builder.State, aux: Any) -> Builder.State:
        """Load model state from source to target.

        Note: If a Tensor from `source` is included by multiple source scopes, makes deep copy of
              the Tensor so that the target Tensors do not refer to the same underlying Tensor.
        """
        restored_state = clone_tree(aux.trainer_state)

        copied_leaf_paths = set()

        def _copy_leaf(
            path: tuple[KeyEntry], leaf: Tensor, source_scope: str, separator: str = "/"
        ) -> Tensor:
            path = separator.join(_key_entry_to_str(k) for k in path)
            if path:
                # If path is not empty, concatenate with source_scope.
                path = f"{source_scope}{separator}{path}"
            else:
                # If path is empty, it means source_scope is leaf.
                path = source_scope
            if path in copied_leaf_paths:
                return leaf.copy()
            copied_leaf_paths.add(path)
            return leaf

        for target_scope, source_scope in self.scopes.items():
            orig_source_model = utils.get_recursively(source.trainer_state.model, source_scope)
            source_model = jax.tree.map_with_path(
                lambda path, leaf, source_scope=source_scope: _copy_leaf(
                    path, leaf, source_scope=source_scope
                ),
                orig_source_model,
            )

            if target_scope:
                if "/" in target_scope:
                    target_path, last_scope = target_scope.rsplit("/", 1)
                else:
                    target_path, last_scope = "", target_scope
                utils.get_recursively(restored_state.model, target_path)[last_scope] = source_model
            else:
                restored_state = restored_state._replace(model=source_model)

        logging.info("restored_state=%s", sorted(key for key, _ in flatten_items(restored_state)))
        built_keys = source.built_keys.union(
            {key for key, value in flatten_items(restored_state) if isinstance(value, Tensor)}
        )
        return Builder.State(step=aux.step, trainer_state=restored_state, built_keys=built_keys)


class PosEmbeddingConverter(BaseConverterFromPretrainedModel):
    """Interpolates positional embedding of the source model to the target's shape.

    This is commonly used when loading a checkpoint trained at a different input resolution or
    sequence length.
    """

    @config_class
    class Config(BaseConverterFromPretrainedModel.Config):
        # A strategy to convert source positional embedding to target positional embedding.
        # If "keep_target", will keep target positional embeddings unchanged.
        # If "replace_target_prefix_with_source", will replace the first source_len target
        # positional embeddings with the whole source positional embeddings where
        # source_len == source.shape[1]. The target sequence length must be >= source sequence
        # length.
        # If "truncate", will truncate the source sequence length to the target sequence length.
        #     Note that the source sequence length must be >= the target sequence length.
        strategy: Required[str] = REQUIRED

    def target_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSORS

    def source_to_target(self, source: Builder.State, aux: Any) -> Builder.State:
        """Produces state compatible with the new positional embedding."""

        def restored_state(src: NestedTensor, tgt: NestedTensor) -> NestedTensor:
            if self._is_pos_emb(src):
                return self._derive_compat_source_pos_emb(src, tgt)
            return src if isinstance(src, Tensor) else tgt

        restored_state = jax.tree.map(
            restored_state,
            source.trainer_state,
            aux.trainer_state,
            is_leaf=self._is_pos_emb,
        )
        restored_state = restored_state._replace(learner=aux.trainer_state.learner)
        built_keys = source.built_keys.union({key for key, _ in flatten_items(restored_state)})
        return Builder.State(step=aux.step, trainer_state=restored_state, built_keys=built_keys)

    # pylint: disable-next=no-self-use
    def _is_pos_emb(self, state: dict[str, Any]) -> bool:
        return is_dict(state, "pos_emb") and is_dict(state["pos_emb"], ["weight"])

    # pylint: disable-next=no-self-use
    def _derive_compat_source_pos_emb(
        self, source: NestedTensor, target: NestedTensor
    ) -> dict[str, Any]:
        cfg = self.config
        source_weight = source["pos_emb"]["weight"]
        target_weight = target["pos_emb"]["weight"]

        if not isinstance(source_weight, Tensor):
            # Sometimes we only build a subset of the source model's weights. Use target_weight
            # in this case.
            return clone_tree(target)

        source_shape = source_weight.shape
        target_shape = target_weight.shape

        out = clone_tree(source)
        if not (
            len(source_shape) == 3
            and len(target_shape) == 3
            and source_shape[0] == target_shape[0]
            and source_shape[2] == target_shape[2]
        ):
            raise ValueError(
                f"Incompatible shapes: source {source_shape} vs. target {target_shape}."
            )

        # The length dimension of embedding is always on axis index 1.
        source_len = source_shape[1]
        target_len = target_shape[1]

        if cfg.strategy.startswith("truncate"):
            if target_len > source_len:
                raise ValueError(f"Target length {target_len} must be <= source len {source_len}.")

            def truncate(x):
                if cfg.strategy == "truncate_left":
                    return x[:, source_len - target_len :, :]
                else:
                    return x[:, :target_len, :]

            out["pos_emb"]["weight"] = pjit(
                truncate,
                in_shardings=source_weight.sharding,
                out_shardings=target_weight.sharding,
            )(source_weight)
        elif cfg.strategy == "replace_target_prefix_with_source":
            if target_len < source_len:
                raise ValueError(f"Target length {target_len} must be >= source len {source_len}.")

            def replace_prefix(source_weight, target_weight):
                return jnp.concatenate([source_weight, target_weight[:, source_len:, :]], axis=1)

            out["pos_emb"]["weight"] = pjit(
                replace_prefix,
                in_shardings=(
                    source_weight.sharding,
                    target_weight.sharding,
                ),
                out_shardings=target_weight.sharding,
            )(source_weight, target_weight)
        elif cfg.strategy == "keep_target":
            out["pos_emb"] = target["pos_emb"]
        # TODO(xianzhi): Implement interpolation for shape matching.
        else:
            raise NotImplementedError(
                f"Strategy {cfg.strategy} is not implemented for PosEmbeddingConverter."
            )
        return out


class EmaParamsConverter(Converter):
    """Copies source model's learner ema weight to target's model weight.

    This can be used together in RestoreAndConvertBuilder to load ema weight into model.
    Note when using EmaParamsConverter, source learner optimizer state is not copied.
    """

    def target_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSOR_SPECS

    def target_to_source(self, target: Builder.State) -> tuple[Builder.State, Any]:
        """Builds a source that contains ema state the same as target.model.

        The source model and optimizer state are empty trees.
        """
        ema_state = ParamEmaState(
            count=OptStateSpec(dtype=jnp.int32, shape=[], mesh_axes=PartitionSpec()),
            ema=clone_tree(target.trainer_state.model),
        )
        source_trainer_state = target.trainer_state._replace(
            # Only keep ema state.
            learner=dict(ema=ema_state),
            # Empty model.
            model={},
        )
        built_keys = {key for key, _ in flatten_items(source_trainer_state)}
        source = Builder.State(
            step=target.step,
            trainer_state=source_trainer_state,
            built_keys=built_keys,
        )
        return source, target

    def source_to_target(self, source: Builder.State, aux: Any) -> Builder.State:
        """Copies the source ema weight into target model and ema if target has a non-empty ema."""
        # Load model weight from learner ema.
        restored_state = aux.trainer_state._replace(
            model=source.trainer_state.learner["ema"].ema,
            prng_key=source.trainer_state.prng_key,
        )
        if (
            aux.trainer_state.learner is not None
            and "ema" in aux.trainer_state.learner
            and aux.trainer_state.learner["ema"].ema is not optax.EmptyState()
        ):
            # Load ema weight to target ema.
            restored_state.learner["ema"] = restored_state.learner["ema"]._replace(
                ema=source.trainer_state.learner["ema"].ema,
            )

        built_keys = aux.built_keys.union({key for key, _ in flatten_items(restored_state)})
        return Builder.State(step=aux.step, trainer_state=restored_state, built_keys=built_keys)


class PruneEmaStateBuilder(Builder):
    """Prune learner ema state from trainer_state.

    We can use PruneEmaStateBuilder when we load from a checkpoint with ema weight
    to a target trainer which does not use ema.
    """

    def input_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSORS

    def __call__(self, state: Builder.State) -> Builder.State:
        # Remove learner ema state.
        del state.trainer_state.learner["ema"]
        prune_state = state.trainer_state._replace(
            learner=state.trainer_state.learner,
        )
        built_keys = state.built_keys.union({key for key, _ in flatten_items(prune_state)})
        return Builder.State(
            step=state.step,
            trainer_state=prune_state,
            built_keys=built_keys,
        )


class SimpleWeightModifierBuilder(Builder):
    """
    A builder for cases where you just want to modify the `weight` in some specific layer,
    such as reshaping/tiling/etc.
    """

    @config_class
    class Config(Builder.Config):
        layer_name: Required[str] = REQUIRED

    def __call__(self, state: Builder.State) -> Builder.State:
        """
        Args:
            state: The current builder state.

        Returns:
            The builder state with a trainer_state containing potentially modified
            layers (as determined by subclass).
        """
        cfg = self.config

        def maybe_modify(x: NestedTensor) -> NestedTensor:
            if self._should_modify(x):
                weight = x[cfg.layer_name]["weight"]
                out = clone_tree(x)
                out[cfg.layer_name]["weight"] = self._modify_weight(weight)
                return out
            else:
                return x

        modified_trainer_state = jax.tree.map(
            maybe_modify,
            state.trainer_state,
            is_leaf=self._should_modify,
        )
        modified_state = Builder.State(
            step=state.step,
            trainer_state=modified_trainer_state,
            built_keys=state.built_keys,
        )
        return modified_state

    def _modify_weight(self, weight: Tensor) -> Tensor:
        """
        Args:
            weight: The weight tensor associated with layers that trigger `self._should_modify`
                being True.

        Returns:
            A potentially modified version of `weight`, as determined by subclass implementation.
        """
        raise NotImplementedError(type(self))

    def _should_modify(self, state: NestedTensor) -> bool:
        """This is called by `tree_map` in __call__ (above) as it traverses trainer_state.

        Args:
            state: An arbitrary sub-tree of the trainer_state.

        Returns:
            True if this state represents the desired layer in the tree that
            contains a weight.
        """
        cfg = self.config
        return is_dict(state, cfg.layer_name) and is_dict(state[cfg.layer_name], ["weight"])


class BaseConv2DStateBuilder(SimpleWeightModifierBuilder):
    """Base class for modifying Conv2D states."""

    @config_class
    class Config(SimpleWeightModifierBuilder.Config):
        layer_name: str = "conv"

    def _should_modify(self, state: NestedTensor) -> bool:
        """
        Args:
            state: A sub-tree of original trainer state.

        Returns:
            True if this state represents a Conv2D layer, else False.
        """
        cfg = self.config
        return super()._should_modify(state) and state[cfg.layer_name]["weight"].ndim == 4


class Reduction(enum.Enum):
    SUM = 0
    MEAN = 1


class Conv2DKernelReducerBuilder(BaseConv2DStateBuilder):
    """Reduces the kernel of Conv2D along the input channel dimension.

    This enables loading pretrained models applied on inputs with three channels (e.g. RGB)
    to a target model for single-channel input (e.g. audio).
    """

    @config_class
    class Config(BaseConv2DStateBuilder.Config):
        reduction: Reduction = Reduction.SUM

    def _modify_weight(self, weight: Tensor) -> Tensor:
        """Reduce (by summation) the input dimension of the source kernel.
        This gives us the converted kernel with desired input dimension of 1.
        """
        cfg = self.config
        if cfg.reduction == Reduction.SUM:
            return weight.sum(axis=2, keepdims=True)
        elif cfg.reduction == Reduction.MEAN:
            return weight.mean(axis=2, keepdims=True)
        else:
            raise ValueError(f"Unrecognized reduction={cfg.reduction}")


class Conv2DToConv3DBuilder(BaseConv2DStateBuilder):
    """
    Tiles Conv2D weights to Conv3D by repeating along the depth dimension,
    along with dividing by the depth dimension in order to keep the outputs on the same
    scale as the original model.
    """

    @config_class
    class Config(BaseConv2DStateBuilder.Config):
        depth: Required[int] = REQUIRED

    def _modify_weight(self, weight: Tensor) -> Tensor:
        cfg = self.config
        # Divide by `depth` to ensure output values remain on the same scale.
        return weight[:, :, jnp.newaxis, :, :].repeat(cfg.depth, axis=2) / cfg.depth


_BUILDERS = [
    TensorStoreStateStorageBuilder,
    HuggingFacePreTrainedBuilder,
]


class UnknownBuilderError(NotImplementedError):
    pass


def get_builder_config(spec: Union[str, Builder], builders: list[type[Builder]]) -> Builder.Config:
    """Match the spec against a list of Builders.

    Args:
        spec: Spec to match.
        builders: List of candidate Builders.

    Returns:
        The Builder with the given spec.

    Raises:
        UnknownBuilderError: If a builder could not be inferred from spec.
    """
    # Get builder by spec from registered builders.
    for builder_class in builders:
        if spec.startswith(builder_class.SPEC_PREFIX):
            # spec is expected to look like <spec_prefix><builder_dir>.
            return builder_class.spec_to_config(spec[len(builder_class.SPEC_PREFIX) :])
    raise UnknownBuilderError(f"Unknown builder: {spec}")


def get_builder(spec: Union[str, Builder]) -> Builder:
    if isinstance(spec, Builder):
        return spec
    builder_cfg = get_builder_config(spec, builders=_BUILDERS)
    builder = builder_cfg.set(name="builder").instantiate(parent=None)
    return builder


class OrbaxStateBuilder(BaseStateStorageBuilder):
    """Build trainer state from Orbax checkpoints."""

    SPEC_PREFIX = "orbax_state_builder:"

    def __init__(self, cfg: BaseStateStorageBuilder.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        if cfg.base_dir is None or cfg.step is None:
            raise ValueError("OrbaxStateBuilder requires base_dir and step to be specified.")

    def __call__(self, state: BaseStateStorageBuilder.State) -> BaseStateStorageBuilder.State:
        cfg: OrbaxStateBuilder.Config = self.config
        # Use lazy-import to avoid global dependency on Orbax.
        # pylint: disable-next=import-outside-toplevel
        from axlearn.common.checkpointer_orbax import OrbaxCheckpointer

        reader_cfg: OrbaxCheckpointer.Config = OrbaxCheckpointer.default_config()
        reader_cfg.name = cfg.name + "-reader"
        reader_cfg.validation_type = cfg.validation
        reader_cfg.dir = cfg.base_dir
        reader_cfg.max_concurrent_restore_gb = cfg.concurrent_gb
        ckpt_reader: OrbaxCheckpointer = reader_cfg.instantiate(parent=self)
        step, restored_state = ckpt_reader.restore(step=cfg.step, state=state.trainer_state)
        assert step == cfg.step
        built_keys = state.built_keys.union({key for key, _ in flatten_items(restored_state)})
        return BaseStateStorageBuilder.State(
            step=step, trainer_state=restored_state, built_keys=built_keys
        )
