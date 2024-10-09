# Copyright © 2023 Apple Inc.

"""Tests state builders."""

# pylint: disable=no-self-use,too-many-lines
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import optax
import pytest
import torch
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

from axlearn.common import bert
from axlearn.common.adapter_flax import config_for_flax_module
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
)
from axlearn.common.input_fake import FakeLmInput
from axlearn.common.layers import Conv2D, Linear
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.param_converter import torch_to_axlearn
from axlearn.common.repeat import Repeat
from axlearn.common.state_builder import (
    _BUILDERS,
    BaseConv2DStateBuilder,
    BaseConverterFromPretrainedModel,
    Builder,
    ChainBuilder,
    ChainConverter,
    Conv2DKernelReducerBuilder,
    Conv2DToConv3DBuilder,
    Converter,
    EmaParamsConverter,
    FlaxPretrainedBuilder,
    HuggingFacePreTrainedBuilder,
    MergeStateConverter,
    MergeStateSelection,
    ModelStateScopeConverter,
    PosEmbeddingConverter,
    RestoreAndConvertBuilder,
    clone_tree,
    get_builder,
    traverse_and_set_target_state_parameters,
)
from axlearn.common.test_utils import TestCase, mock_trainer_config
from axlearn.common.trainer import SpmdTrainer, TrainerState
from axlearn.common.utils import (
    NestedTensor,
    PartitionSpec,
    Tensor,
    VDict,
    as_tensor,
    flatten_items,
    get_recursively,
    replicate_to_local_data,
)


class MergeStateConverterTest(TestCase):
    """Tests MergeStateConverter."""

    def test_converter_string_and_enum_args(self):
        class DummyBuilder(Builder):
            """A dummy state builder."""

            def input_state_type(self) -> Builder.StateType:
                return Builder.StateType.TENSORS

            def __call__(self, state: Builder.State) -> Builder.State:
                return Builder.State(
                    step=0,
                    trainer_state={"a": jnp.array(1), "b": {"a": jnp.array(1000)}},
                    built_keys=set(),
                )

        cfg = RestoreAndConvertBuilder.default_config().set(
            name="test",
            builder=DummyBuilder.default_config().set(),
            # Tests both string and enum arguments
            converter=MergeStateConverter.default_config().set(
                selection_regexes=[(r"b/a", MergeStateSelection.SOURCE), (r".*", "TARGET")]
            ),
        )
        builder = cfg.instantiate(parent=None)

        state = Builder.State(
            step=0, trainer_state={"a": jnp.array(0), "b": {"a": jnp.array(0)}}, built_keys=set()
        )
        state = builder(state)
        self.assertEqual(state.trainer_state, {"a": jnp.array(0), "b": {"a": jnp.array(1000)}})

    def test_converter_default(self):
        class DummyBuilder(Builder):
            """A dummy state builder."""

            def input_state_type(self) -> Builder.StateType:
                return Builder.StateType.TENSORS

            def __call__(self, state: Builder.State) -> Builder.State:
                return Builder.State(
                    step=0,
                    trainer_state={"a": jnp.array(1), "b": {"a": jnp.array(1000)}},
                    built_keys=set(),
                )

        cfg = RestoreAndConvertBuilder.default_config().set(
            name="test",
            builder=DummyBuilder.default_config().set(),
            converter=MergeStateConverter.default_config(),
        )
        builder = cfg.instantiate(parent=None)

        state = Builder.State(
            step=0, trainer_state={"a": jnp.array(0), "b": {"a": jnp.array(0)}}, built_keys=set()
        )
        state = builder(state)
        self.assertEqual(state.trainer_state, {"a": jnp.array(1), "b": {"a": jnp.array(1000)}})


class ConverterTest(TestCase):
    """Tests converters."""

    def test_chain(self):
        # pylint: disable=missing-class-docstring
        class DummyConverter(Converter):
            @config_class
            class Config(Builder.Config):
                id: Required[int] = REQUIRED

            def target_state_type(self) -> Builder.StateType:
                return Builder.StateType.TENSORS

            def target_to_source(self, target: Builder.State) -> tuple[Builder.State, Any]:
                target.step += 1
                return target, self.config.id

            def source_to_target(self, source: Builder.State, aux: Any) -> Builder.State:
                assert aux == self.config.id
                source.step -= 1
                return source

        class DummyBuilder(Builder):
            @config_class
            class Config(Builder.Config):
                expected_step: Required[int] = REQUIRED

            def input_state_type(self) -> Builder.StateType:
                return Builder.StateType.TENSORS

            def __call__(self, state: Builder.State) -> Builder.State:
                assert state.step == self.config.expected_step
                return state

        num_converters = 5
        cfg = RestoreAndConvertBuilder.default_config().set(
            name="test",
            converter=ChainConverter.default_config().set(
                converters=[
                    DummyConverter.default_config().set(id=i) for i in range(num_converters)
                ]
            ),
            builder=DummyBuilder.default_config().set(expected_step=num_converters),
        )
        builder = cfg.instantiate(parent=None)

        state = Builder.State(step=0, trainer_state={}, built_keys=set())
        state = builder(state)
        self.assertEqual(state.step, 0)
        # pylint: enable=missing-class-docstring


class GetBuilderTest(TestCase):
    """Tests get_builder."""

    # Parameterize the tests so we can run in parallel.
    @parameterized.parameters(_BUILDERS)
    def test_get_builder(self, builder_cls: type[Builder]):
        self._test_get_builder(builder_cls)

    def _get_builder(self, builder_cls: type[Builder], spec: str):
        if builder_cls.SPEC_PREFIX == "hf_pretrained:":
            # HF builder requires source and target scopes.
            spec = spec + "::"
        return get_builder(spec=f"{builder_cls.SPEC_PREFIX}{spec}")

    def _test_get_builder(self, builder_cls: type[Builder]):
        spec = "test_spec"
        default_cfg = builder_cls.default_config()
        builder = self._get_builder(builder_cls, spec)
        self.assertIsInstance(builder, builder_cls)
        self.assertEqual(type(builder), builder_cls)
        if isinstance(builder, RestoreAndConvertBuilder):
            self.assertEqual(type(builder.builder), default_cfg.builder.cls)
            self.assertEqual(type(builder.converter), default_cfg.converter.cls)
            self.assertEqual(builder.builder.config.dir, spec)
        elif isinstance(builder, HuggingFacePreTrainedBuilder):
            self.assertEqual(builder.config.hf_layer_config.model_name, spec)
        else:
            self.assertEqual(builder.config.dir, spec)


class ChainBuilderTest(TestCase):
    """Tests ChainBuilder."""

    def test_simple_chain_builder(self):
        class BiasPlusOneBuilder(Builder):
            """Simple builder that will increase bias by one."""

            def input_state_type(self) -> Builder.StateType:
                return Builder.StateType.TENSORS

            def __call__(self, state: Builder.State) -> Builder.State:
                cloned_state = clone_tree(state.trainer_state)
                cloned_state.model["bias"] += 1.0
                return Builder.State(step=0, trainer_state=cloned_state, built_keys=set())

        model = (
            Linear.default_config()
            .set(name="model", input_dim=1, output_dim=1)
            .instantiate(parent=None)
        )
        prng_key = jax.random.PRNGKey(0)
        init_trainer_state = TrainerState(
            prng_key=prng_key,
            model=model.initialize_parameters_recursively(prng_key=prng_key),
            learner=None,
        )
        builder = (
            ChainBuilder.default_config()
            .set(
                name="builder",
                builders=[BiasPlusOneBuilder.default_config(), BiasPlusOneBuilder.default_config()],
            )
            .instantiate(parent=None)
        )
        new_trainer_state = builder(
            Builder.State(step=0, trainer_state=init_trainer_state, built_keys=set())
        )
        self.assertEqual(new_trainer_state.trainer_state.model["bias"][0], 2.0)

    def test_chain_builder_error(self):
        class RemoveBiasBuilder(Builder):
            """Simple builder that removes bias and hence fails the state structure validation."""

            def input_state_type(self) -> Builder.StateType:
                return Builder.StateType.TENSORS

            def __call__(self, state: Builder.State) -> Builder.State:
                cloned_state = clone_tree(state.trainer_state)
                del cloned_state.model["bias"]
                return Builder.State(step=0, trainer_state=cloned_state, built_keys=set())

        model = (
            Linear.default_config()
            .set(name="model", input_dim=1, output_dim=1)
            .instantiate(parent=None)
        )
        prng_key = jax.random.PRNGKey(0)
        init_trainer_state = TrainerState(
            prng_key=prng_key,
            model=model.initialize_parameters_recursively(prng_key=prng_key),
            learner=None,
        )
        builder = (
            ChainBuilder.default_config()
            .set(name="builder", builders=[RemoveBiasBuilder.default_config()])
            .instantiate(parent=None)
        )
        with self.assertRaises(ValueError):
            builder(Builder.State(step=0, trainer_state=init_trainer_state, built_keys=set()))


class PosEmbeddingConverterTest(TestCase):
    """Tests PosEmbeddingConverter."""

    def _mock_bert_trainer_config(
        self, max_len: int = 512, hidden_dim: int = 32
    ) -> SpmdTrainer.Config:
        vocab_size = 3000
        return mock_trainer_config(
            input_config=FakeLmInput.default_config().set(
                global_batch_size=128,
                source_length=max_len,
            ),
            model_config=bert.bert_model_config(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                embedding_cfg=bert.bert_embedding_config(max_position_embeddings=max_len),
                stack_cfg=bert.bert_transformer_config(num_layers=3, num_heads=2),
            ),
        )

    def _mock_bert_trainer_config_and_state(
        self, max_len: int = 512, hidden_dim: int = 32
    ) -> tuple[SpmdTrainer.Config, Builder.State]:
        trainer_config = self._mock_bert_trainer_config(max_len=max_len, hidden_dim=hidden_dim)
        trainer = trainer_config.instantiate(parent=None)
        trainer.init(prng_key=jax.random.PRNGKey(0))
        source_state = Builder.State(step=0, trainer_state=trainer.trainer_state, built_keys=set())
        return trainer_config, source_state

    @parameterized.parameters(["truncate", "truncate_left"])
    def test_truncation(self, strategy):
        source_trainer_config, source_state = self._mock_bert_trainer_config_and_state(max_len=512)
        _, target_state = self._mock_bert_trainer_config_and_state(max_len=64)

        cfg = PosEmbeddingConverter.default_config().set(
            name="pos_emb_tester",
            source_trainer_config=source_trainer_config,
            strategy=strategy,
        )
        converter: PosEmbeddingConverter = cfg.instantiate(parent=None)

        converted_state = converter.source_to_target(source_state, target_state)
        self.assertEqual(
            (1, 512, 32),
            source_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"].shape,
        )
        self.assertEqual(
            (1, 64, 32),
            converted_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"].shape,
        )

        source_weight = replicate_to_local_data(
            source_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"],
        )
        converted_weight = replicate_to_local_data(
            converted_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"],
        )
        # Check weights after truncation are from source.
        if strategy == "truncate":
            self.assertNestedAllClose(source_weight[:, :64, :], converted_weight)
        else:
            self.assertNestedAllClose(source_weight[:, 448:, :], converted_weight)

    @parameterized.parameters(["truncate", "truncate_left"])
    def test_truncation_target_longer_than_source(self, strategy):
        source_trainer_config, source_state = self._mock_bert_trainer_config_and_state(max_len=32)
        _, target_state = self._mock_bert_trainer_config_and_state(max_len=64)
        cfg = PosEmbeddingConverter.default_config().set(
            name="pos_emb_tester",
            source_trainer_config=source_trainer_config,
            strategy=strategy,
        )

        with self.assertRaisesWithLiteralMatch(
            ValueError, "Target length 64 must be <= source len 32."
        ):
            converter: PosEmbeddingConverter = cfg.instantiate(parent=None)
            converter.source_to_target(source_state, target_state)

    @parameterized.parameters(
        ["truncate", "truncate_left", "keep_target", "replace_target_prefix_with_source"]
    )
    def test_truncation_incompatible_shape(self, strategy: str):
        source_trainer_config, source_state = self._mock_bert_trainer_config_and_state(
            max_len=512, hidden_dim=128
        )
        _, target_state = self._mock_bert_trainer_config_and_state(max_len=64, hidden_dim=256)
        cfg = PosEmbeddingConverter.default_config().set(
            name="pos_emb_tester",
            source_trainer_config=source_trainer_config,
            strategy=strategy,
        )

        with self.assertRaisesWithLiteralMatch(
            ValueError, "Incompatible shapes: source (1, 512, 128) vs. target (1, 64, 256)."
        ):
            converter: PosEmbeddingConverter = cfg.instantiate(parent=None)
            converter.source_to_target(source_state, target_state)

    def test_replace_target_prefix_with_source(self):
        source_trainer_config, source_state = self._mock_bert_trainer_config_and_state(max_len=32)
        _, target_state = self._mock_bert_trainer_config_and_state(max_len=64)

        cfg = PosEmbeddingConverter.default_config().set(
            name="pos_emb_tester",
            source_trainer_config=source_trainer_config,
            strategy="replace_target_prefix_with_source",
        )
        converter: PosEmbeddingConverter = cfg.instantiate(parent=None)

        converted_state = converter.source_to_target(source_state, target_state)
        self.assertEqual(
            (1, 32, 32),
            source_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"].shape,
        )
        self.assertEqual(
            (1, 64, 32),
            converted_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"].shape,
        )

        source_weight = replicate_to_local_data(
            source_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"],
        )
        target_weight = replicate_to_local_data(
            target_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"],
        )
        converted_weight = replicate_to_local_data(
            converted_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"],
        )
        # Check the prefix are from source.
        self.assertNestedAllClose(source_weight, converted_weight[:, :32, :])
        # Check the rest are from target.
        self.assertNestedAllClose(target_weight[:, 32:, :], converted_weight[:, 32:, :])

    def test_keep_target(self):
        source_trainer_config, source_state = self._mock_bert_trainer_config_and_state(max_len=512)
        _, target_state = self._mock_bert_trainer_config_and_state(max_len=64)

        cfg = PosEmbeddingConverter.default_config().set(
            name="pos_emb_tester",
            source_trainer_config=source_trainer_config,
            strategy="keep_target",
        )
        converter: PosEmbeddingConverter = cfg.instantiate(parent=None)

        converted_state = converter.source_to_target(source_state, target_state)
        target_weight = replicate_to_local_data(
            target_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"]
        )
        converted_weight = replicate_to_local_data(
            converted_state.trainer_state.model["encoder"]["emb"]["pos_emb"]["weight"]
        )
        # Check weights after truncation are from target.
        self.assertNestedAllClose(target_weight, converted_weight)

    def test_non_existent_strategy(self):
        source_trainer_config, source_state = self._mock_bert_trainer_config_and_state(max_len=512)
        _, target_state = self._mock_bert_trainer_config_and_state(max_len=64)
        cfg = PosEmbeddingConverter.default_config().set(
            name="pos_emb_tester",
            source_trainer_config=source_trainer_config,
            strategy="nonexistent",
        )

        with self.assertRaisesWithLiteralMatch(
            NotImplementedError,
            "Strategy nonexistent is not implemented for PosEmbeddingConverter.",
        ):
            converter: PosEmbeddingConverter = cfg.instantiate(parent=None)
            converter.source_to_target(source_state, target_state)


class DummyRepeatLayer(Repeat):
    """A dummy repeat layer"""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.layer = Linear.default_config().set(input_dim=2, output_dim=3)
        cfg.num_layers = 2
        return cfg


class DummyModel(BaseModel):  # pylint: disable=abstract-method
    """A dummy model."""

    @config_class
    class Config(BaseModel.Config):
        """Configures DummyModel."""

        layer: InstantiableConfig = Linear.default_config().set(input_dim=5, output_dim=2)
        child: Optional[InstantiableConfig] = None
        layer_name: str = "linear"

    def __init__(self, cfg: BaseModel.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(cfg.layer_name, cfg.layer)
        if cfg.child:
            self._add_child("child", cfg.child)


class DummyNestedLayer(BaseLayer):
    """A dummy nested layer."""

    @config_class
    class Config(BaseModel.Config):
        """Configures DummyNestedLayer."""

        layer: InstantiableConfig = Linear.default_config().set(input_dim=5, output_dim=2)
        path: Required[str] = REQUIRED
        path2: Optional[str] = None

    def __init__(self, cfg: BaseModel.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        def add_nested_child(path):
            if "/" not in path:
                self._add_child(path, cfg.layer)
            else:
                name, sub_path = path.split("/", maxsplit=1)
                self._add_child(
                    name,
                    DummyNestedLayer.default_config().set(
                        layer=cfg.layer,
                        path=sub_path,
                    ),
                )

        add_nested_child(cfg.path)
        if cfg.path2 is not None:
            add_nested_child(cfg.path2)


class DummyNestedModel(BaseModel):
    """A dummy nested model."""

    @config_class
    class Config(BaseModel.Config):
        """Configures DummyNestedModel."""

        layer: InstantiableConfig = Linear.default_config().set(input_dim=5, output_dim=2)
        path: Required[str] = REQUIRED
        path2: Optional[str] = None

    def __init__(self, cfg: BaseModel.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "model",
            DummyNestedLayer.default_config().set(
                path=cfg.path,
                path2=cfg.path2,
                layer=cfg.layer,
            ),
        )

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        raise NotImplementedError(type(self))


def _mock_state(trainer_cfg, seed: int = 0) -> Builder.State:
    trainer = trainer_cfg.instantiate(parent=None)
    trainer.init(prng_key=jax.random.PRNGKey(seed))
    state = Builder.State(
        step=0,
        trainer_state=trainer.trainer_state,
        built_keys=set(),
    )
    return state


class _FakeMultimodalImageInput(Module):
    """A fake multimodal image input."""

    @config_class
    class Config(Module.Config):
        """Configures _FakeMultimodalImageInput."""

        is_training: Required[bool] = REQUIRED
        input_size: Required[list[int]] = REQUIRED
        batch_size: Required[int] = REQUIRED

    def __iter__(self):
        cfg = self.config
        data = dict(
            inputs=jax.random.normal(
                self.prng_key(),
                shape=[cfg.batch_size, 1, *cfg.input_size],
            )
        )
        yield data

    def dataset(self):
        return self.__iter__()  # pylint: disable=unnecessary-dunder-call


class TestConv2DStateBuilders(TestCase):
    """Tests Conv2DStateBuilders."""

    def test_conv2d_to_conv3d(self):
        """Basic sanity check that we can convert Conv2D to Conv3D dummy models."""
        height, width, input_dim, output_dim = 64, 64, 3, 32
        patch_size = (16, 16, 2)

        conv_path = "source/path/to/conv"

        # Create Conv2D model.
        source_cfg = self._mock_image_config(
            patch_size[:-1],
            (height, width, input_dim),
            output_dim,
            conv_path=conv_path,
        )

        # Convert Conv2D kernel to Conv3D kernel.
        source_weight, converted_weight = self._run_builder(
            Conv2DToConv3DBuilder,
            source_cfg=source_cfg,
            conv_path=conv_path,
            depth=patch_size[-1],
        )
        self.assertEqual(
            (*patch_size[:-1], input_dim, output_dim),
            source_weight.shape,
        )
        self.assertEqual(
            (*patch_size, input_dim, output_dim),
            converted_weight.shape,
        )

        source_weight = replicate_to_local_data(source_weight)
        converted_weight = replicate_to_local_data(converted_weight)
        for depth_idx in range(patch_size[-1]):
            self.assertNestedAllClose(
                source_weight,
                patch_size[-1] * converted_weight[:, :, depth_idx, :, :],
            )

    def test_conv2d_kernel_reducer(self):
        height, width, output_dim = 128, 64, 32
        patch_size = (16, 16)

        conv_path = "source/path/to/conv"

        # Create Conv2D model with input_dim=3.
        source_cfg = self._mock_image_config(
            patch_size,
            (height, width, 3),
            output_dim,
            conv_path=conv_path,
        )

        # Convert RGB kernel to single-channel kernel.
        source_weight, converted_weight = self._run_builder(
            Conv2DKernelReducerBuilder,
            source_cfg=source_cfg,
            conv_path=conv_path,
        )

        self.assertEqual(
            (*patch_size, 3, output_dim),
            source_weight.shape,
        )
        self.assertEqual(
            (*patch_size, 1, output_dim),
            converted_weight.shape,
        )
        self.assertNestedAllClose(
            source_weight.sum(axis=2, keepdims=True),
            converted_weight,
        )

    def _run_builder(
        self,
        builder_cls: type[BaseConv2DStateBuilder],
        *,
        source_cfg,
        conv_path: str,
        **extra_converter_config_kwargs,
    ):
        source_state = _mock_state(source_cfg, seed=0)
        initial_trainer_state_tree_structure = jax.tree_util.tree_structure(
            source_state.trainer_state
        )

        builder = (
            builder_cls.default_config()
            .set(
                name="builder",
                **extra_converter_config_kwargs,
            )
            .instantiate(parent=None)
        )

        source_model = source_state.trainer_state.model

        converted_state = builder(deepcopy(source_state))
        assert initial_trainer_state_tree_structure == jax.tree_util.tree_structure(
            converted_state.trainer_state
        )
        converted_model = converted_state.trainer_state.model

        source_weight = get_recursively(source_model, f"model/{conv_path}")["weight"]
        converted_weight = get_recursively(converted_model, f"model/{conv_path}")["weight"]

        source_weight = replicate_to_local_data(source_weight)
        converted_weight = replicate_to_local_data(converted_weight)
        return source_weight, converted_weight

    def _dummy_model_config(
        self,
        patch_size: tuple[int, ...],
        input_size: tuple[int, ...],
        output_dim: int,
        conv_path: str,
        conv_cls: type[BaseLayer],
    ):
        return DummyNestedModel.default_config().set(
            layer=conv_cls.default_config().set(
                window=patch_size,
                strides=patch_size,
                input_dim=input_size[-1],
                output_dim=output_dim,
                bias=False,
            ),
            path=conv_path,
            dtype=jnp.float32,
        )

    def _mock_image_config(
        self,
        patch_size: tuple[int, ...],
        input_size: tuple[int, ...],
        output_dim: int,
        conv_path: str,
        batch_size: int = 8,
    ):
        model_cfg = self._dummy_model_config(
            patch_size=patch_size,
            input_size=input_size,
            output_dim=output_dim,
            conv_path=conv_path,
            conv_cls=Conv2D,
        )
        input_cfg = _FakeMultimodalImageInput.default_config().set(
            is_training=True,
            batch_size=batch_size,
            input_size=input_size,
        )
        return mock_trainer_config(input_config=input_cfg, model_config=model_cfg)


class BaseConverterFromPretrainedModelTest(TestCase):
    """Sanity checks for BaseConverterFromPretrainedModel."""

    def test_mesh_shape(self):
        mock_trainer_cfg, mock_state = _create_dummy_state(jax.random.PRNGKey(1))
        cfg = BaseConverterFromPretrainedModel.default_config().set(
            source_trainer_config=mock_trainer_cfg,
            mesh_shape=(-1, 1),
        )
        # Ensure that we're able to instantiate mock_trainer_cfg with -1 in the mesh.
        converter = cfg.set(name="test_converter").instantiate(parent=None)
        converter.target_to_source(mock_state)


class DiffusersPretrainedBuilderTest(TestCase):
    """Tests FlaxPretrainedBuilder for a diffusers model."""

    def _init_state(self, model, prng_key: Tensor):
        prng_key, init_key = jax.random.split(prng_key)
        model_params = model.initialize_parameters_recursively(init_key)
        return TrainerState(
            prng_key=prng_key,
            model=model_params,
            learner=None,
        )

    def test_load_local_ckpts(self):
        ckpt_folder = os.path.join(
            os.path.dirname(__file__), "testdata/axlearn.common.state_builder_test"
        )
        if not os.path.exists(ckpt_folder):
            pytest.skip(reason="Missing testdata.")

        # pylint: disable-next=import-outside-toplevel,import-error
        from diffusers.models.vae_flax import FlaxAutoencoderKL  # pytype: disable=import-error

        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            # Set up a minimal sized diffusers model.
            auto_encoder_config_dict = dict(
                sample_size=16,
                in_channels=1,
                out_channels=1,
                latent_channels=1,
                block_out_channels=(2,),
                layers_per_block=1,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                norm_num_groups=1,
            )

            def dummy_inputs_bchw():
                return (jnp.zeros([1, 1, 16, 16], dtype=jnp.float32),), {}

            autoencoder_cfg = config_for_flax_module(
                FlaxAutoencoderKL,
                dummy_inputs_bchw,
                create_module_kwargs=auto_encoder_config_dict,
            ).set(name="autoencoder")

            model = autoencoder_cfg.instantiate(parent=None)
            prng_key = jax.random.PRNGKey(0)
            trainer_state_specs = TrainerState(
                prng_key=ParameterSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
                model=model.create_parameter_specs_recursively(),
                learner=None,
            )
            trainer_state_partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, trainer_state_specs
            )

            def _init_state(*args):
                return self._init_state(model, *args)

            init_computation = pjit(
                _init_state,
                in_shardings=(None,),
                out_shardings=trainer_state_partition_specs,
            )

            trainer_state = init_computation(prng_key)
            builder_state = Builder.State(step=0, trainer_state=trainer_state, built_keys=set())

            def flax_state_supplier():
                _, source_params = FlaxAutoencoderKL.from_pretrained(
                    ckpt_folder, subfolder="vae", from_pt=True
                )
                return source_params

            builder_config = FlaxPretrainedBuilder.default_config().set(
                name="builder",
                flax_state_supplier_config=config_for_function(flax_state_supplier),
                target_scope=[],
            )
            builder = builder_config.instantiate(parent=None)

            restored_state = builder(builder_state)

            # Check the value of a specific kernel.
            restored_kernel = restored_state.trainer_state.model["params"]["encoder"]["conv_in"][
                "kernel"
            ]
            restored_kernel = jnp.reshape(jnp.transpose(restored_kernel, (3, 2, 0, 1)), [18])

            self.assertNestedAllClose(jnp.array(range(18), dtype=jnp.float32), restored_kernel)

            # Check the parity of model output.
            flax_model, flax_params = FlaxAutoencoderKL.from_pretrained(
                ckpt_folder, subfolder="vae", from_pt=True
            )
            random_image = jax.random.uniform(
                jax.random.PRNGKey(0), [1, 1, 16, 16], dtype=jnp.float32
            )
            flax_output = flax_model.apply({"params": flax_params}, random_image)

            axlearn_output, _ = F(
                model,
                inputs=(random_image,),
                state=restored_state.trainer_state.model,
                is_training=False,
                prng_key=jax.random.PRNGKey(0),
            )
            self.assertNestedAllClose(flax_output.sample, axlearn_output.sample)


class HuggingFacePreTrainedBuilderTest(TestCase):
    """Tests HuggingFacePreTrainedBuilder."""

    # TODO(bwzhang@) add a HF builder test by setting a temporary path for downloading the models.
    def _init_state(self, model, prng_key: Tensor):
        prng_key, init_key = jax.random.split(prng_key)
        model_params = model.initialize_parameters_recursively(init_key)
        return TrainerState(
            prng_key=prng_key,
            model=model_params,
            learner=None,
        )

    def test_traverse_and_set_state(self):
        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            model = DummyModel.default_config().set(name="model").instantiate(parent=None)
            prng_key = jax.random.PRNGKey(0)
            trainer_state_specs = TrainerState(
                prng_key=ParameterSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
                model=model.create_parameter_specs_recursively(),
                learner=None,
            )
            trainer_state_partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, trainer_state_specs
            )

            def _init_state(*args):
                return self._init_state(model, *args)

            init_computation = pjit(
                _init_state,
                in_shardings=(None,),
                out_shardings=trainer_state_partition_specs,
            )

            trainer_state = init_computation(prng_key)
            linear_model = torch.nn.Linear(in_features=5, out_features=2, bias=True)
            linear_model_jax_parameter = torch_to_axlearn(linear_model)

            # Replace all model states
            new_trainer_state = traverse_and_set_target_state_parameters(
                target_state=trainer_state.model,
                target_scope=["linear"],
                source_params=linear_model_jax_parameter,
                source_scope=[],
            )
            self.assertNestedAllClose(new_trainer_state["linear"], linear_model_jax_parameter)

            # Replace bias only and not traverse the source_params
            new_trainer_state = traverse_and_set_target_state_parameters(
                target_state=trainer_state.model,
                target_scope=["linear", "bias"],
                source_params=linear_model_jax_parameter["bias"],
                source_scope=[],
            )
            self.assertNestedAllClose(
                new_trainer_state["linear"]["bias"], linear_model_jax_parameter["bias"]
            )

            # Replace weights only and traverse the source_params
            new_trainer_state = traverse_and_set_target_state_parameters(
                target_state=trainer_state.model,
                target_scope=["linear", "weight"],
                source_params=linear_model_jax_parameter,
                source_scope=["weight"],
            )
            self.assertNestedAllClose(
                new_trainer_state["linear"]["weight"], linear_model_jax_parameter["weight"]
            )

    def test_repeat(self):
        """Test with repeat layer, which uses VDict."""

        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            repeat_cfg = DummyRepeatLayer.default_config()
            model_cfg = DummyModel.default_config().set(child=repeat_cfg)
            model = model_cfg.set(name="model").instantiate(parent=None)
            prng_key = jax.random.PRNGKey(0)
            trainer_state_specs = TrainerState(
                prng_key=ParameterSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
                model=model.create_parameter_specs_recursively(),
                learner=None,
            )
            trainer_state_partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, trainer_state_specs
            )

            def _init_state(*args):
                return self._init_state(model, *args)

            init_computation = pjit(
                _init_state,
                in_shardings=(None,),
                out_shardings=trainer_state_partition_specs,
            )

            trainer_state = init_computation(prng_key)

            # Construct params for linear layer.
            ref_linear = torch.nn.Linear(in_features=5, out_features=2, bias=True)
            ref_linear_params = torch_to_axlearn(ref_linear)

            # Construct params for child repeat layer.
            ref_repeat = torch.nn.Linear(in_features=2, out_features=3, bias=True)
            ref_repeat_params = torch_to_axlearn(ref_repeat)
            # Tile the params across repeat dim.
            ref_repeat_params = jax.tree_map(
                lambda x: jnp.tile(x, [repeat_cfg.num_layers] + [1] * x.ndim), ref_repeat_params
            )

            # Construct final params.
            ref_params = dict(linear=ref_linear_params, child=VDict(layer=ref_repeat_params))

            # Replace all model states
            new_trainer_state = traverse_and_set_target_state_parameters(
                target_state=trainer_state.model,
                target_scope=[],
                source_params=ref_params,
                source_scope=[],
            )
            self.assertNestedAllClose(new_trainer_state, ref_params)

    @parameterized.parameters(
        [
            None,
            torch.nn.Linear,
            config_for_function(
                lambda: Linear.default_config()
                .set(input_dim=5, output_dim=2, name="test")
                .instantiate(parent=None)
            ),
        ]
    )
    def test_converter(self, dst_layer):
        x = torch.nn.Linear(in_features=5, out_features=2)

        def dummy_layer():
            return x

        cfg = HuggingFacePreTrainedBuilder.default_config().set(
            name="test",
            hf_layer_config=config_for_function(dummy_layer),
        )
        if dst_layer is not None:
            cfg.converter.dst_layer = dst_layer

        builder = cfg.instantiate(parent=None)
        init_state = TrainerState(
            model=dict(weight=jnp.zeros([5, 2]), bias=jnp.zeros([2])), prng_key=None, learner=None
        )
        output = builder(Builder.State(step=0, trainer_state=init_state, built_keys=set()))
        self.assertNestedAllClose(
            dict(weight=as_tensor(x.weight.transpose(0, 1)), bias=x.bias),
            output.trainer_state.model,
        )


def _create_dummy_state(prng_key, model_config=DummyModel.default_config(), use_ema=False):
    trainer_config = mock_trainer_config(
        input_config=FakeLmInput.default_config().set(
            global_batch_size=128,
            source_length=64,
        ),
        model_config=model_config.set(name="model", dtype=jnp.float32),
    )
    if use_ema:
        trainer_config.learner.ema.decay = 0.99
    trainer = trainer_config.instantiate(parent=None)
    trainer.init(prng_key=prng_key)

    def trainer_cfg_fn():
        def fn() -> InstantiableConfig:
            return trainer_config

        return fn

    trainer_state = trainer.trainer_state
    if use_ema:
        trainer_state.learner["ema"] = trainer_state.learner["ema"]._replace(
            ema=jax.tree.map(lambda p: -jnp.ones_like(p), trainer.trainer_state.learner["ema"].ema)
        )
    return config_for_function(trainer_cfg_fn), Builder.State(
        step=0, trainer_state=trainer_state, built_keys=set()
    )


class ModelStateScopeConverterTest(TestCase):
    """Tests ModelStateScopeConverter."""

    def test_scope_none(self):
        source_cfg, source_state = _create_dummy_state(jax.random.PRNGKey(0))
        _, target_state = _create_dummy_state(jax.random.PRNGKey(1))
        converter: ModelStateScopeConverter = (
            ModelStateScopeConverter.default_config()
            .set(name="test", source_trainer_config=source_cfg)
            .instantiate(parent=None)
        )
        # The pruned state has no learner entries.
        pruned_source_state, _ = converter.target_to_source(target_state)
        self.assertCountEqual([], pruned_source_state.built_keys)
        self.assertCountEqual(
            ["model/linear/bias", "model/linear/weight"],
            [path for path, _ in flatten_items(pruned_source_state.trainer_state)],
        )
        converted_state = converter.source_to_target(source_state, target_state)
        self.assertCountEqual(
            {
                "prng_key",
                "model/linear/weight",
                "model/linear/bias",
                "learner/optimizer/0/nu/linear/weight",
                "learner/optimizer/0/nu/linear/bias",
                "learner/optimizer/2/count",
                "learner/optimizer/0/mu/linear/weight",
                "learner/optimizer/0/mu/linear/bias",
                "learner/optimizer/0/count",
            },
            converted_state.built_keys,
        )
        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear"],
            source_state.trainer_state.model["linear"],
        )

    def test_target_scope_only(self):
        source_cfg, source_state = _create_dummy_state(jax.random.PRNGKey(0))
        target_model_cfg = DummyModel.default_config().set(
            layer=DummyModel.default_config(), layer_name="nested"
        )
        _, target_state = _create_dummy_state(jax.random.PRNGKey(1), target_model_cfg)
        converter = (
            ModelStateScopeConverter.default_config()
            .set(name="test", source_trainer_config=source_cfg, scope="nested")
            .instantiate(parent=None)
        )
        converted_state = converter.source_to_target(source_state, target_state)
        self.assertNestedAllClose(
            converted_state.trainer_state.model["nested"]["linear"],
            source_state.trainer_state.model["linear"],
        )

    def test_cross_scopes(self):
        source_cfg, source_state = _create_dummy_state(jax.random.PRNGKey(0))
        _, target_state = _create_dummy_state(
            jax.random.PRNGKey(1), DummyModel.default_config().set(layer_name="linear2")
        )
        converter = (
            ModelStateScopeConverter.default_config()
            .set(name="test", source_trainer_config=source_cfg, scope={"linear2": "linear"})
            .instantiate(parent=None)
        )
        converted_state = converter.source_to_target(source_state, target_state)
        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear2"],
            source_state.trainer_state.model["linear"],
        )

    def test_cross_scopes_many(self):
        source_cfg, source_state = _create_dummy_state(jax.random.PRNGKey(0))
        _, target_state = _create_dummy_state(
            jax.random.PRNGKey(1), DummyModel.default_config().set(layer_name="linear2")
        )
        converter = (
            ModelStateScopeConverter.default_config()
            .set(
                name="test",
                source_trainer_config=source_cfg,
                scope={"linear2/bias": "linear/bias", "linear2/weight": "linear/weight"},
            )
            .instantiate(parent=None)
        )
        converted_state = converter.source_to_target(source_state, target_state)
        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear2"],
            source_state.trainer_state.model["linear"],
        )

    def test_only_linear_weight(self):
        source_cfg, source_state = _create_dummy_state(jax.random.PRNGKey(0))
        _, target_state = _create_dummy_state(
            jax.random.PRNGKey(1), DummyModel.default_config().set(layer_name="linear2")
        )
        converter = (
            ModelStateScopeConverter.default_config()
            .set(
                name="test",
                source_trainer_config=source_cfg,
                scope={"linear2/weight": "linear/weight"},
            )
            .instantiate(parent=None)
        )
        # The pruned state has only 'linear/weight' under 'model'.
        pruned_source_state, _ = converter.target_to_source(target_state)
        self.assertCountEqual(
            ["model/linear/weight"],
            [path for path, _ in flatten_items(pruned_source_state.trainer_state)],
        )
        converted_state = converter.source_to_target(source_state, target_state)
        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear2"]["weight"],
            source_state.trainer_state.model["linear"]["weight"],
        )
        # linear/bias remains unchanged from target_state.
        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear2"]["bias"],
            target_state.trainer_state.model["linear2"]["bias"],
        )

    def _create_fake_state_and_convert(self, scope_mapping: Dict[str, str]):
        # Create fake source_state and target_state with nested layers.
        source_cfg, source_state = _create_dummy_state(
            jax.random.PRNGKey(0),
            DummyNestedModel.default_config().set(path="linear", path2="linear2"),
        )
        _, target_state = _create_dummy_state(
            jax.random.PRNGKey(1),
            DummyModel.default_config().set(
                child=DummyNestedLayer.default_config().set(path="linear")
            ),
        )

        converter = (
            ModelStateScopeConverter.default_config()
            .set(
                name="test",
                source_trainer_config=source_cfg,
                scope=scope_mapping,
            )
            .instantiate(parent=None)
        )
        converted_state = converter.source_to_target(source_state, target_state)
        return source_state, converted_state

    @parameterized.parameters(
        {"scope_mapping": {"linear": "model/linear", "child": "model"}},
        {"scope_mapping": {"linear/bias": "model/linear/bias", "child": "model"}},
        {
            "scope_mapping": {
                "linear/bias": "model/linear/bias",
                "child/linear": "model/linear",
                "child": "model",
            }
        },
        {
            "scope_mapping": {
                "linear/bias": "model/linear/bias",
                "child": "model",
                "child/linear": "model/linear",
            }
        },
    )
    def test_duplicate_source_scopes_leaf_first(self, scope_mapping):
        # Map leaf first.
        # Create fake source_state and target_state with nested layers and perform conversion.
        source_state, converted_state = self._create_fake_state_and_convert(scope_mapping)

        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )
        self.assertNestedAllClose(
            converted_state.trainer_state.model["child"]["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )
        # converted_state's "linear/bias" is donated.
        self.assertIs(
            converted_state.trainer_state.model["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )
        # converted_state's "child" is a copy and has different memory.
        self.assertIsNot(
            converted_state.trainer_state.model["child"]["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )

    @parameterized.parameters(
        {"scope_mapping": {"child/linear": "model/linear", "linear": "model/linear"}},
        {"scope_mapping": {"child/linear": "model/linear", "linear/bias": "model/linear/bias"}},
        {"scope_mapping": {"child": "model", "linear/bias": "model/linear/bias"}},
    )
    def test_duplicate_source_scopes_leaf_last(self, scope_mapping):
        # Map leaf at last.

        # Create fake source_state and target_state with nested layers and perform conversion.
        source_state, converted_state = self._create_fake_state_and_convert(scope_mapping)

        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )
        self.assertNestedAllClose(
            converted_state.trainer_state.model["child"]["linear"],
            source_state.trainer_state.model["model"]["linear"],
        )
        # source_state's "model" is donated to coverted_state's "child".
        self.assertIs(
            converted_state.trainer_state.model["child"]["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )
        self.assertIs(
            converted_state.trainer_state.model["child"]["linear"]["weight"],
            source_state.trainer_state.model["model"]["linear"]["weight"],
        )
        # converted_state's "linear/bias" is a copy and has different memory.
        self.assertIsNot(
            converted_state.trainer_state.model["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )

    def test_duplicate_source_scopes_edge_case(self):
        # Create fake source_state and target_state with nested layers and perform conversion.
        scope_mapping = {"linear": "model/linear", "child/linear": "model/linear2"}
        source_state, converted_state = self._create_fake_state_and_convert(scope_mapping)

        self.assertNestedAllClose(
            converted_state.trainer_state.model["linear"],
            source_state.trainer_state.model["model"]["linear"],
        )
        self.assertNestedAllClose(
            converted_state.trainer_state.model["child"]["linear"],
            source_state.trainer_state.model["model"]["linear2"],
        )
        # converted_state's "linear" is donated.
        self.assertIs(
            converted_state.trainer_state.model["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear"]["bias"],
        )
        self.assertIs(
            converted_state.trainer_state.model["linear"]["weight"],
            source_state.trainer_state.model["model"]["linear"]["weight"],
        )
        # converted_state's "child/linear" is also donated.
        self.assertIs(
            converted_state.trainer_state.model["child"]["linear"]["bias"],
            source_state.trainer_state.model["model"]["linear2"]["bias"],
        )
        self.assertIs(
            converted_state.trainer_state.model["child"]["linear"]["weight"],
            source_state.trainer_state.model["model"]["linear2"]["weight"],
        )

    @parameterized.parameters(None, "FAKE")
    def test_source_data_dir(self, source_data_dir):
        source_cfg, source_state = _create_dummy_state(jax.random.PRNGKey(0))
        _, target_state = _create_dummy_state(jax.random.PRNGKey(1))
        converter: ModelStateScopeConverter = (
            ModelStateScopeConverter.default_config()
            .set(name="test", source_trainer_config=source_cfg, source_data_dir=source_data_dir)
            .instantiate(parent=None)
        )
        converted_state = converter.source_to_target(source_state, target_state)
        self.assertNestedAllClose(
            converted_state.trainer_state.model,
            source_state.trainer_state.model,
        )


class EmaParamsConverterTest(TestCase):
    """Tests EmaParamsConverter."""

    @parameterized.parameters(["with_target_ema", "with_learner_no_ema", "with_no_learner"])
    def test_ema_params_converter(self, target_ema):
        _, source_state = _create_dummy_state(jax.random.PRNGKey(0), use_ema=True)
        target_state = clone_tree(source_state)
        if target_ema == "with_no_learner":
            target_state.trainer_state = target_state.trainer_state._replace(learner=None)
        elif target_ema == "with_learner_no_ema":
            del target_state.trainer_state.learner["ema"]

        converter = (
            EmaParamsConverter.default_config()
            .set(
                name="test",
            )
            .instantiate(parent=None)
        )
        convert_state, _ = converter.target_to_source(target_state)
        # Test that model is empty.
        self.assertNestedAllClose(
            convert_state.trainer_state.model,
            optax.EmptyState(),
        )
        self.assertEqual(["ema"], list(convert_state.trainer_state.learner.keys()))
        # Target model is copied to convert state ema.
        self.assertNestedAllClose(
            convert_state.trainer_state.learner["ema"].ema,
            target_state.trainer_state.model,
        )

        output_state = converter.source_to_target(source_state, target_state)

        self.assertNestedAllClose(
            output_state.trainer_state.model,
            source_state.trainer_state.learner["ema"].ema,
        )

        if target_ema == "with_target_ema":
            self.assertNestedAllClose(
                output_state.trainer_state.learner["ema"],
                source_state.trainer_state.learner["ema"],
            )


if __name__ == "__main__":
    absltest.main()
