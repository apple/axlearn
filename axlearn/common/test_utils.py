# Copyright © 2023 Apple Inc.

"""Utilites used for testing."""
import contextlib
import copy
import dataclasses
import os
import tempfile
from collections import OrderedDict
from functools import partial
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union
from unittest.mock import patch

import jax
import jax.random
import numpy as np
from absl import logging
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.common import decoding, optimizers, schedule, utils_spmd
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import every_n_steps_policy
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
)
from axlearn.common.evaler import every_n_steps_policy as eval_every_n_steps_policy
from axlearn.common.learner import Learner
from axlearn.common.logit_modifiers import LogitsToLogitsFn
from axlearn.common.module import functional as F
from axlearn.common.optimizer_base import OptParam
from axlearn.common.optimizers import opt_param_values
from axlearn.common.param_init import FanAxes, Initializer, Shape
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import (
    NestedTensor,
    NestedTree,
    Tensor,
    as_tensor,
    complete_partition_spec_tree,
    flatten_items,
    pop_data_dir,
    push_data_dir,
    set_data_dir,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

# Instead of running setup() in TestCase.setUp(), we need to run it here, because setUp() runs
# after parameterized.parameters(), so if we have
#     @parameterized.parameters(
#         {
#             "data": jnp.array([[1.0, 0.8, 0, 0]], dtype=jnp.float32),
#         }
#     )
# `data` will be of type jax.DeviceArray() instead of jax.Array() because we haven't called
# jax.config.update("jax_array", True) yet.
utils_spmd.setup()

# See utils_spmd.py for where we set "jax_default_prng_impl".
_default_prng_impl = "rbg"
_PYTEST_OPT_REGISTERED = {}


def assert_allclose(actual, desired, atol=1e-6, rtol=1e-3, err_msg=""):
    actual = jnp.asarray(actual).astype(np.float32)
    desired = jnp.asarray(desired).astype(np.float32)
    # Checks if 'actual' and 'desired' are within (atol + rtol * abs(desired)).
    diff: np.ndarray = np.abs(actual - desired)
    if diff.size > 0:
        diff = diff.max()
    np.testing.assert_allclose(
        actual,
        desired,
        atol=atol,
        rtol=rtol,
        err_msg=f"{err_msg}: {diff}.\nactual={actual}\ndesired={desired}",
    )


def is_supported_platform(target_platform: str) -> bool:
    """Checks if a function intended for a specific platform can be executed on the current one."""
    devices = jax.devices()
    supported = all(device.platform == target_platform for device in devices)
    if not supported:
        logging.info(
            "Skipping test for %s on %s",
            target_platform,
            [device.platform for device in devices],
        )
    return supported


def is_supported_mesh_shape(mesh_shape: Tuple[int, int]) -> bool:
    """Checks if a function intended for a mesh shape is compatible with the current device(s)."""
    device_count = jax.device_count()
    supported = device_count == np.prod(mesh_shape)
    if not supported:
        logging.info("Skipping mesh_shape=%s with device_count=%s", mesh_shape, device_count)
    return supported


def as_local_tensor(x: Tensor) -> NestedTensor:
    if isinstance(x, Tensor):
        return x
    raise NotImplementedError(f"{type(x)}: {x}")


class ParameterConversionFn(Protocol):
    def __call__(self, src: Any, *, dst_layer: BaseLayer) -> NestedTensor:
        """Converts parameters from `src` to parameters for `dst_layer`."""


class TestCase(parameterized.TestCase):
    """Base test class."""

    @property
    def data_dir(self):
        return "FAKE"

    def setUp(self):
        push_data_dir(self.data_dir)

    def tearDown(self) -> None:
        self.assertEqual(pop_data_dir(), self.data_dir)

    def _compute_layer_outputs(
        self,
        *,
        test_layer: BaseLayer,
        ref_layer: Any,
        test_inputs: Any,
        ref_inputs: Any,
        parameters_from_ref_layer: ParameterConversionFn,
        require_same_num_params: bool = True,
        require_same_tree_structure: bool = True,
    ) -> Tuple[Any, Any]:
        layer_params = test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        total_num_layer_params = 0
        for name, param in flatten_items(layer_params):
            logging.info("Test: %s=%s", name, param.shape)
            total_num_layer_params += param.size
        logging.info("Total test layer params: %s", total_num_layer_params)

        total_num_ref_params = 0
        for name, param in ref_layer.named_parameters(recurse=True):
            logging.info("Ref: %s=%s", name, param.shape)
            total_num_ref_params += param.numel()
        for name, param in ref_layer.named_buffers(recurse=True):
            logging.info("Ref buffer: %s=%s", name, param.shape)
        logging.info("Total ref layer params: %s", total_num_ref_params)

        if require_same_num_params:
            self.assertEqual(total_num_layer_params, total_num_ref_params)

        params_from_ref = parameters_from_ref_layer(ref_layer, dst_layer=test_layer)

        # Test that leaves have same paths and shapes. Note that trees with different structures can
        # still have the same leaves: {"a": 1, "b": {}} will match {"a": 1}, because there are no
        # leaves under subtree "b".
        self.assertSequenceEqual(
            [f"{name}={value.shape}" for name, value in flatten_items(params_from_ref)],
            [f"{name}={value.shape}" for name, value in flatten_items(layer_params)],
        )
        # Optionally, test that trees also have the same structure.
        if require_same_tree_structure:
            ref_structure = jax.tree_util.tree_structure(params_from_ref)
            test_structure = jax.tree_util.tree_structure(layer_params)
            self.assertEqual(
                ref_structure, test_structure, msg=f"\nRef: {ref_structure}\nTest: {test_structure}"
            )
        del layer_params

        test_outputs, _ = F(
            test_layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=params_from_ref,
            inputs=test_inputs,
        )
        ref_layer.eval()
        ref_outputs = (
            ref_layer(**ref_inputs) if isinstance(ref_inputs, dict) else ref_layer(ref_inputs)
        )
        return test_outputs, ref_outputs

    def assertNestedAllClose(self, a, b, atol=1e-6, rtol=1e-3):
        a_items = sorted(flatten_items(a), key=lambda x: x[0])
        b_items = sorted(flatten_items(b), key=lambda x: x[0])
        self.assertEqual([name for name, _ in a_items], [name for name, _ in b_items])
        for (a_name, a_value), (b_name, b_value) in zip(a_items, b_items):
            self.assertEqual(a_name, b_name)
            if isinstance(a_value, Tensor):
                a_value = as_local_tensor(a_value)
            if isinstance(b_value, Tensor):
                b_value = as_local_tensor(b_value)
            if isinstance(a_value, (np.ndarray, jnp.ndarray)) or isinstance(
                b_value, (np.ndarray, jnp.ndarray)
            ):
                a_value, b_value = as_tensor(a_value), as_tensor(b_value)
                self.assertEqual(a_value.dtype, b_value.dtype, msg=f"{a_name}")
                self.assertEqual(a_value.shape, b_value.shape, msg=f"{a_name}")
                assert_allclose(a_value, b_value, atol=atol, rtol=rtol, err_msg=f"{a_name}")
            else:
                self.assertAlmostEqual(a_value, b_value)

    def assertNestedEqual(self, a, b):
        a_kv = flatten_items(a)
        b_kv = flatten_items(b)
        self.assertCountEqual([k for k, _ in a_kv], [k for k, _ in b_kv])
        a_dict = dict(a_kv)
        b_dict = dict(b_kv)
        for k in a_dict:
            a_value = a_dict[k]
            b_value = b_dict[k]
            np.testing.assert_array_equal(a_value, b_value, err_msg=k)
            if hasattr(a_value, "dtype"):
                self.assertEqual(a_value.dtype, b_value.dtype)


class TrainerConfigTestCase(TestCase):
    """Base class for testing trainer configs."""

    def _test_with_trainer_config(self, trainer_config, mesh_size: Optional[Dict[str, int]] = None):
        with jax.checking_leaks(), set_data_dir("FAKE"):
            if mesh_size is None:
                mesh_size = {}
            cfg = copy.deepcopy(trainer_config)
            cfg.dir = cfg.dir or tempfile.mkdtemp()
            cfg.mesh_axis_names = cfg.mesh_axis_names or ("data", "model")
            cfg.mesh_shape = cfg.mesh_shape or (len(jax.devices()), 1)
            cfg.max_step = 3
            for evaler_cfg in cfg.evalers.values():
                if getattr(evaler_cfg.eval_policy, "fn", None) is eval_every_n_steps_policy:
                    evaler_cfg.eval_policy.n = 2
                evaler_cfg.vlog = max(evaler_cfg.vlog, 3)
            if getattr(cfg.checkpointer.save_policy, "fn", None) is every_n_steps_policy:
                cfg.checkpointer.save_policy.n = 2
            logging.info("_test_with_trainer_config: %s", trainer_config)
            trainer: SpmdTrainer = cfg.instantiate(parent=None)
            trainer.run(jax.random.PRNGKey(123))

            state_spec_map = dict(flatten_items(trainer.trainer_state_specs))
            for path, value in flatten_items(trainer.trainer_state):
                state_spec = state_spec_map.get(path)
                logging.info(
                    "State: %s=%s(%s) state_spec=%s", path, value.dtype, value.shape, state_spec
                )
                if state_spec is None:
                    continue
                self.assertSequenceEqual(value.shape, state_spec.shape)
                self.assertLen(
                    state_spec.mesh_axes,
                    len(value.shape),
                    msg=f"{path}: {state_spec} vs {value.shape}",
                )
                for dim_size, dim_name in zip(value.shape, state_spec.mesh_axes):
                    if dim_name is None:
                        continue
                    mesh_dim_size = mesh_size.get(dim_name, 1)
                    self.assertEqual(
                        dim_size % mesh_dim_size,
                        0,
                        msg=(
                            f"{path}: {dim_size} % {mesh_dim_size} != 0 "
                            f"for {dim_name} in {value.shape} vs. {state_spec}"
                        ),
                    )


class DummyForwardModel(BaseModel):
    """A dummy model whose ``forward`` returns (0, input_batch["aux"]).

    ``predict`` returns input_batch["aux"].
    """

    def forward(self, input_batch: NestedTensor, **kwargs) -> Tuple[Tensor, NestedTensor]:
        del kwargs
        return jnp.zeros([], dtype=jnp.float32), input_batch.get("aux", {})

    def predict(self, input_batch: NestedTensor) -> NestedTensor:
        return self.forward(input_batch)[1]


# forward() is not implemented.
# pylint: disable-next=abstract-method
class DummyDecodingModel(BaseModel):
    """A dummy model whose `beam_search_decode` and `sample_decode` returns ids from
    input_batch["predicted"]."""

    @config_class
    class Config(BaseModel.Config):
        vocab_size: Required[int] = REQUIRED
        # bos_id: Required[int] = REQUIRED
        eos_id: Required[int] = REQUIRED

    def _token_to_scores(
        self,
        predicted_sequences: Tensor,
        num_decodes: int,
        logits_modifier: Optional[LogitsToLogitsFn] = None,
    ):
        cfg = self.config

        print(f"predicted_sequences={predicted_sequences}")
        batch_size = predicted_sequences.shape[0]
        vocab_size = cfg.vocab_size

        # This fn aims to duplicate the input sequences.
        def tokens_to_scores(
            token_indices: Tensor, state_cache: NestedTensor  # pylint: disable=unused-argument
        ) -> Tuple[Tensor, NestedTensor]:
            cur_step = state_cache["cur_step"]
            # Gets the golden token for the next time step.
            # [batch_size, num_decodes]
            # Adds EOS after predicted_sequences.shape[1] (all tokens would be equally likely).
            cur_golden_token = jnp.take_along_axis(
                predicted_sequences,
                jnp.reshape(cur_step[:, 0] + 1, (batch_size, num_decodes)),
                axis=1,
            )
            print(f"cur_golden_token={cur_golden_token}, vocab_size={vocab_size}")
            # Convert tokens to the logits by the one hot operation.
            # The golden token's logit will be 1000 and others' are zeros.
            # [batch_size * num_decodes, vocab_size]
            logits_for_tokens = jnp.reshape(
                jax.nn.one_hot(cur_golden_token, vocab_size, axis=-1) * 1000,
                (batch_size * num_decodes, vocab_size),
            )
            log_probs_for_tokens = jax.nn.log_softmax(logits_for_tokens)
            # Update state in the cache.
            new_cache = state_cache.copy()
            new_cache["cur_step"] = cur_step + 1
            # Apply logits_modifier.
            if logits_modifier is not None:
                log_probs_for_tokens = logits_modifier(log_probs_for_tokens)
            return log_probs_for_tokens, new_cache

        return tokens_to_scores

    def beam_search_decode(
        self,
        input_batch: NestedTensor,
        *,
        num_decodes: int = 1,
        max_len: int = 114,
    ) -> decoding.BeamSearchOutputs:
        cfg = self.config

        if "beam_search_outputs" in input_batch:
            return input_batch["beam_search_outputs"]

        predicted_sequences = input_batch["predicted"]
        batch_size = predicted_sequences.shape[0]

        # Initializes the cache for the beam search.
        init_cache = {}
        init_cache["cur_step"] = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        inputs = jnp.zeros_like(predicted_sequences)
        if "prefix" in input_batch:
            inputs = inputs.at[:, : input_batch["prefix"].shape[1]].set(input_batch["prefix"])
        return decoding.beam_search_decode(
            inputs=inputs,
            cache=init_cache,
            tokens_to_scores=self._token_to_scores(predicted_sequences, num_decodes),
            eos_id=cfg.eos_id,
            num_decodes=num_decodes,
            max_decode_len=max(max_len, predicted_sequences.shape[-1]),
        )

    def sample_decode(
        self,
        input_batch: NestedTensor,
        *,
        num_decodes: int,
        max_len: int,
        logits_modifier: LogitsToLogitsFn,
    ) -> decoding.SampleOutputs:
        cfg = self.config

        if "sample_outputs" in input_batch:
            return input_batch["sample_outputs"]

        predicted_sequences = input_batch["predicted"]
        batch_size = predicted_sequences.shape[0]

        # Initializes the cache for the beam search.
        init_cache = {}
        init_cache["cur_step"] = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        inputs = jnp.zeros_like(predicted_sequences)
        inputs = inputs.at[:, 0].set(cfg.eos_id)
        return decoding.sample_decode(
            inputs=inputs,
            cache=init_cache,
            tokens_to_scores=self._token_to_scores(
                predicted_sequences,
                num_decodes,
                logits_modifier=logits_modifier,
            ),
            stop_decoding_condition=decoding.StopOnSubsequence([[cfg.eos_id]]),
            num_decodes=num_decodes,
            max_decode_len=max(max_len, predicted_sequences.shape[-1]),
            prng_key=jax.random.PRNGKey(123),
        )


class TestWithTemporaryCWD(TestCase):
    """Run all tests in a temp directory to isolate from local env."""

    def run(self, result=None):
        temp_root = tempfile.TemporaryDirectory()
        # Note that using "as" will only return the dir name.
        with temp_root:
            # pylint: disable-next=attribute-defined-outside-init
            self._temp_root = temp_root
            temp_root = os.path.realpath(self._temp_root.name)
            os.chdir(temp_root)
            super().run(result)


@contextlib.contextmanager
def prng_impl(new_prng_impl: str):
    old_prng_impl = _default_prng_impl

    def switch(value):
        global _default_prng_impl  # pylint: disable=global-statement
        _default_prng_impl = value
        jax.config.update("jax_default_prng_impl", value)

    switch(new_prng_impl)
    yield
    switch(old_prng_impl)


# Use dataclass so that jax.tree_util.tree_map does not expand it.
@dataclasses.dataclass
class ParamInitSpec:
    shape: Optional[Sequence[int]]
    initializer: Initializer
    fan_axes: Optional[FanAxes]


# initialize is intentionally left unimplemented.
# pylint: disable=abstract-method
class ThirdPartyInitializer(Initializer):
    """An stand-in initializer that indicates that initialization is delegated to a third party
    library, like HuggingFace."""

    def __init__(self, library: str):
        self._library = library

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        return f"delegated({self._library})"


# When pytype supports recursive types, switch to:
# Optional[Union[ParamInitSpec, Dict[str, "NestedParamInitSpec"]]]
NestedParamInitSpec = Optional[Union[ParamInitSpec, Dict[str, Any]]]


def _cast_ordered_dict(params: NestedTensor):
    if isinstance(params, dict):
        params = OrderedDict({k: _cast_ordered_dict(v) for k, v in params.items()})
    return params


def _complete_param_init_spec_tree(
    params: NestedTensor, param_init_specs: List[ParamInitSpec], delegates: Dict[str, ParamInitSpec]
):
    """Completes the param_init_specs by replacing certain param paths with proxy Initializers.

    For example, HF modules can have params keyed under HF_MODULE_KEY which have no corresponding
    param_init_spec. In these cases, to ensure that we can construct a tree of `param_init_specs` of
    the same treedef, we must either prune params or complete the `param_init_specs`. This function
    does the latter. To achieve the former, see `utils.prune_tree`.

    Args:
        params: The tree of params, as obtained by `initialize_parameters_recursively`.
        param_init_specs: A list of `param_init_specs`, corresponding to leaves of a tree prefix of
            `params`.
        delegates: A mapping from param name to a `param_init_spec`. Params with the given name are
            assumed to have no corresponding `param_init_spec` in `param_init_specs`.

    Returns:
        A tree with the same treedef as `params`, with leaves from `param_init_specs`. Keys that are
        present in `delegates` will have the corresponding `param_init_spec` injected.
    """

    def is_leaf(v):
        return not isinstance(v, dict) or any(k in v for k in delegates)

    def replace_keys(v, mapping):
        if isinstance(v, dict):
            for k, new_v in mapping.items():
                if k in v:
                    v[k] = new_v
        return v

    # Convert to ordered dict, to work around https://github.com/google/jax/issues/4085.
    params = _cast_ordered_dict(params)

    # Complete the param_init_specs to match the params treedef.
    # Replace with Nones so that jax doesn't treat them as leaves.
    params_with_nones = jax.tree_map(
        partial(replace_keys, mapping={k: None for k in delegates}), params, is_leaf=is_leaf
    )
    _, treedef = jax.tree_util.tree_flatten(params_with_nones)
    inits_with_nones = jax.tree_util.tree_unflatten(treedef, param_init_specs)

    # Replace the Nones with a delegate.
    return jax.tree_map(partial(replace_keys, mapping=delegates), inits_with_nones, is_leaf=is_leaf)


def read_param_init_specs_recursively(
    layer: BaseLayer, *, delegates: Optional[Dict[str, ParamInitSpec]] = None
) -> NestedParamInitSpec:
    """Given a layer, returns all nested parameter initialization specs.

    Args:
        layer: The layer to read from recursively.
        delegates: An optional mapping from param name to a `param_init_spec`.
            See `_complete_param_init_spec_tree` for details.

    Returns:
        A tree of ParamInitSpec.
    """
    # A flat list of ParamInitSpec in init (traversal) order.
    all_param_init_specs = []

    # pylint: disable-next=unused-argument
    def patched_init(self, name, *, prng_key, parameter_spec):
        # pylint: disable-next=protected-access
        fan_axes = self._compute_fan_axes(name, parameter_spec)
        all_param_init_specs.append(
            ParamInitSpec(
                shape=parameter_spec.shape,
                initializer=parameter_spec.initializer,
                fan_axes=fan_axes,
            )
        )
        return np.empty(0)  # Return a dummy init value.

    patch_init = patch(
        "axlearn.common.base_layer.BaseLayer._initialize_parameter",
        side_effect=patched_init,
        autospec=True,
    )

    orig_vmap = jax.vmap

    def patched_vmap(fn):
        def wrapped_fn(*args, **kwargs):
            return _cast_ordered_dict(fn(*args, **kwargs))

        return orig_vmap(wrapped_fn)

    patch_vmap = patch("jax.vmap", side_effect=patched_vmap, autospec=True)

    with patch_init, patch_vmap:
        params = layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
    return _complete_param_init_spec_tree(params, all_param_init_specs, delegates or {})


def read_per_param_settings(
    module: Any, config_name: str, trainer_config: Optional[TrainerConfigFn] = None
) -> Dict[str, NestedTensor]:
    """Extracts per-param settings for the given trainer config.

    Given a trainer config specified by `module` and `config_name`, initializes the trainer
    and intercepts calls to `register_per_param_settings` to extract `description` and `settings`
    for each call.

    Args:
        module: training config module.
        config_name: training config name.
        trainer_config: Optional, the pre-cached trainer config used in golden config test.

    Returns:
        A dictionary of trees, where the key is the description of the `register_per_param_settings`
        call, and the value is a tree of the same structure as the model parameter, and has
        per-parameter settings, e.g., a float number representing weight decay scale of the
        parameter.
        It can be empty if the trainer config does not call `register_per_param_settings`.
    """
    # Define patchers.

    # pylint: disable-next=unused-argument
    def patched_init(self, name, *, prng_key, parameter_spec):
        return np.zeros(1)  # Return a dummy init value.

    # Use dummy values to speed up the initialization.
    patch_init = patch(
        "axlearn.common.base_layer.BaseLayer._initialize_parameter",
        side_effect=patched_init,
        autospec=True,
    )

    all_param_settings = {}

    def patched_register_per_param_settings(
        settings: NestedTree, *, description: str
    ) -> NestedTree:
        if description in all_param_settings:
            raise ValueError(
                f"{description} already populated:\n"
                f"Existing: {all_param_settings[description]}\n"
                f"New: {settings}\n"
            )
        all_param_settings[description] = settings
        return settings

    with patch(
        "axlearn.common.utils._register_per_param_settings",
        side_effect=patched_register_per_param_settings,
        autospec=True,
    ), patch_init:
        if trainer_config is not None:
            config_fn = trainer_config
        else:
            config_fn = getattr(module, "named_trainer_configs")()[config_name]
        trainer_cfg: SpmdTrainer.Config = config_fn()
        model: BaseModel = trainer_cfg.model.set(name="model").instantiate(parent=None)

        model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(0))

        model_specs = model.create_parameter_specs_recursively()
        model_specs = complete_partition_spec_tree(
            jax.tree_util.tree_structure(model_params), model_specs
        )
        opt_params = jax.tree_util.tree_map(
            lambda param, spec: OptParam(
                value=param,
                # Disable factored second moment since we use a dummy weight value.
                factorization_spec=None,
                weight_decay_scale=spec.weight_decay_scale if spec is not None else 1.0,
            ),
            model_params,
            model_specs,
        )
        # Sets gradients to dummy values.
        zero_grads = jax.tree_util.tree_map(lambda p: jnp.zeros(1), opt_param_values(opt_params))
        learner = trainer_cfg.learner.set(name="learner").instantiate(parent=None)
        learner_state = learner.init(opt_params)
        zero_grads = jax.tree_util.tree_map(
            lambda use_opt, g: g if use_opt else None,
            learner.should_update_with_optimizers(opt_params),
            zero_grads,
        )
        F(
            module=learner,
            state=learner_state,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
            method="update",
            inputs=dict(model_params=opt_params, gradients=zero_grads, state_updates={}),
        )

    return all_param_settings


def dummy_padding_mask(*, batch_size: int, max_seq_len: int) -> Tensor:
    """Builds a dummy attention mask where non-padding tokens are followed by padding tokens.

    Example:
        batch_size: 3
        max_seq_len: 3
        output: [[1, 1, 0], [1, 0, 0], [1, 1, 1]]

    Args:
        batch_size: Batch size.
        max_seq_len: Sequence length.

    Returns:
        A bool attention mask of shape [batch, max_seq_len]. A value of 0 indicates a padding
        token, whereas 1 indicates non-padding. Each example has at least one non-padding,
        followed by padding up to seq_len.
    """
    lower_diag = jnp.arange(max_seq_len)
    lower_diag = lower_diag[None, :] <= lower_diag[:, None]
    input_len = jax.random.randint(
        jax.random.PRNGKey(123), shape=(batch_size,), minval=0, maxval=max_seq_len
    )
    return lower_diag[input_len]


def dummy_segments_positions(
    batch: int, seq_len: int, *, num_segments: int
) -> Tuple[Tensor, Tensor]:
    """Builds dummy segment IDs and corresponding positions.

    Example:
        batch: 2
        seq_len: 4
        num_segments: 3
        output: (
            [[0, 1, 1, 1], [1, 1, 2, 2]],  # segment_ids
            [[0, 0, 1, 2], [0, 1, 0, 1]],  # positions
        )

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        num_segments: Number of segments.

    Returns:
        A tuple of:
        - A Tensor of shape [batch, seq_len] with values in [0, num_segments). Note that not all
            segment IDs may be present, i.e., some segments may be empty.
        - A Tensor of shape [batch, seq_len] with values in [0, seq_len) representing positions
            within each segment.

    Raises:
        ValueError: If num_segments is not strictly positive.
    """
    if num_segments < 1:
        raise ValueError("num_segments must be strictly positive.")
    # Generate dummy segment IDs of shape [batch, seq_len].
    segment_starts = jnp.pad(jnp.arange(seq_len - 1) < num_segments - 1, [[1, 0]])
    segment_starts = jnp.tile(segment_starts, [batch, 1])
    segment_starts = jax.random.shuffle(
        jax.random.PRNGKey(401),
        segment_starts,
        axis=-1,
    )
    segment_ids = 1 + jnp.cumsum(segment_starts, axis=-1)

    # Generate dummy positions corresponding to the segments.
    segment_idx = jnp.where(segment_starts)[1].reshape((batch, num_segments - 1))
    start_idx = jnp.column_stack([jnp.zeros(batch, dtype=jnp.int32), segment_idx])
    end_idx = jnp.column_stack([segment_idx, jnp.ones(batch, dtype=jnp.int32) * seq_len]).astype(
        jnp.int32
    )
    segment_lens = end_idx - start_idx
    segment_lens = jnp.repeat(start_idx, segment_lens).reshape((batch, seq_len))
    positions = jnp.tile(jnp.arange(seq_len), (batch, 1)) - segment_lens

    return segment_ids, positions


def take_segment(inputs: Tensor, mask: Tensor, *, pad_value: int = 0) -> Tensor:
    """Selects elements from the input corresponding to the given segment.

    Example:
        inputs: [[1, 2, 3], [4, 5, 6]]
        mask: [[0, 0, 1], [0, 1, 1]]
        pad_value: -100
        output: [[3, -100, -100], [5, 6, -100]]

    Args:
        inputs: A Tensor of any shape.
        mask: A bool Tensor of same shape as `input_batch`. True values are kept, False
            values are replaced with padding.
        pad_value: Value to pad with.

    Returns:
        A Tensor of same shape as `inputs`, with chosen values left-aligned.
    """
    lens = mask.sum(axis=-1, keepdims=True)
    shifted = jnp.tile(jnp.arange(mask.shape[-1]), [mask.shape[-2], 1]) < lens
    new_inputs = jnp.ones_like(inputs) * pad_value
    return new_inputs.at[shifted.astype(jnp.bool_)].set(inputs[mask.astype(jnp.bool_)])


def pytest_addoption_atomic(parser, option, **kwargs):
    """This function allows pytest_addoption to be invoked multiple times atomically.

    Reference:
    https://github.com/huggingface/transformers/blob/92ce53aab859012f7714dae6d6fce7a7d701e75f/src/transformers/testing_utils.py#L1331
    """
    if option not in _PYTEST_OPT_REGISTERED:
        _PYTEST_OPT_REGISTERED[option] = True
        parser.addoption(option, **kwargs)


def mock_trainer_config(
    input_config: InstantiableConfig,
    model_config: BaseModel.Config,
    mesh_axis_names: Sequence[str] = ("data", "model"),
) -> SpmdTrainer.Config:
    cfg = SpmdTrainer.default_config()
    cfg.name = "mock_trainer_config"
    cfg.input = input_config
    cfg.dir = mkdtemp()
    cfg.mesh_axis_names = mesh_axis_names
    cfg.mesh_shape = (len(jax.devices()), 1)
    cfg.model = model_config
    cfg.learner = Learner.default_config().set(
        optimizer=config_for_function(optimizers.adamw_optimizer).set(
            learning_rate=config_for_function(schedule.polynomial).set(
                begin_step=0,
                begin_value=0.00001,
                end_step=200,
                end_value=1e-7,
            ),
            b1=0.9,
            b2=0.999,
            eps=1e-4,
        )
    )
    return cfg
