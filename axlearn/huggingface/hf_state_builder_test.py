# Copyright © 2025 Apple Inc.

"""Tests Hugging Face state builder."""

from typing import Optional

import jax
import jax.numpy as jnp
import torch
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

from axlearn.common.base_layer import ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.config import InstantiableConfig, config_class, config_for_function
from axlearn.common.layers import Linear
from axlearn.common.module import Module
from axlearn.common.param_converter import torch_to_axlearn
from axlearn.common.repeat import Repeat
from axlearn.common.state_builder import Builder, traverse_and_set_target_state_parameters
from axlearn.common.test_utils import TestCase
from axlearn.common.trainer import TrainerState
from axlearn.common.utils import PartitionSpec, Tensor, VDict, as_tensor
from axlearn.huggingface.hf_state_builder import HuggingFacePreTrainedBuilder


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


class DummyRepeatLayer(Repeat):
    """A dummy repeat layer"""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.layer = Linear.default_config().set(input_dim=2, output_dim=3)
        cfg.num_layers = 2
        return cfg


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
            ref_repeat_params = jax.tree.map(
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
