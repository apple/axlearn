# Copyright © 2023 Apple Inc.

"""Utilities used for testing."""

import contextlib
import copy
import dataclasses
import os
import pathlib
import re
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Sequence
from functools import partial
from tempfile import mkdtemp
from typing import Any, NamedTuple, Optional, Protocol, TypeVar, Union
from unittest.mock import patch

import jax
import jax.random
import numpy as np
import optax
from absl import logging
from absl.testing import parameterized
from jax import numpy as jnp

from axlearn.common import optimizers, schedule, utils_spmd
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import every_n_steps_policy
from axlearn.common.config import (
    REQUIRED,
    ConfigOr,
    InstantiableConfig,
    Required,
    RequiredFieldValue,
    config_class,
    config_for_function,
)
from axlearn.common.evaler import every_n_steps_policy as eval_every_n_steps_policy
from axlearn.common.learner import Learner
from axlearn.common.module import InvocationContext, Module, current_context
from axlearn.common.module import functional as F
from axlearn.common.module import new_output_collection, set_current_context
from axlearn.common.optimizer_base import OptParam
from axlearn.common.optimizers import opt_param_values
from axlearn.common.param_init import FanAxes, Initializer, Shape
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.update_transformation import Updates
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    NestedTree,
    Tensor,
    as_tensor,
    complete_partition_spec_tree,
    flatten_items,
    pop_data_dir,
    prune_empty,
    prune_tree,
    push_data_dir,
    set_data_dir,
    shapes,
)
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

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


def is_supported_mesh_shape(
    mesh_shape: Sequence[int], devices: Optional[list[jax.Device]] = None
) -> bool:
    """Checks if a function intended for a mesh shape is compatible with the current device(s)."""
    device_count = jax.device_count() if devices is None else len(devices)
    supported = device_count == np.prod(mesh_shape)
    if not supported:
        logging.info("Skipping mesh_shape=%s with device_count=%s", mesh_shape, device_count)
    return supported


def as_local_tensor(x: Tensor) -> NestedTensor:
    if isinstance(x, Tensor):
        return x
    raise NotImplementedError(f"{type(x)}: {x}")


def clean_hlo(hlo: str) -> str:
    """Returns a cleaned version of `hlo` with non-functional parts that may impact test reliability
    removed.

    Args:
        hlo: The hlo to clean.

    Returns:
        A cleaned version of `hlo`.
    """
    # Matches an escaped string literal. E.g., "hello, world\"\\"
    escaped_str = '"' + r"""([^"\\]|\\\\|\\")*""" + '"'
    metadata_value = "(" + escaped_str + "|" + r"\d+" + ")"
    pattern = r"metadata=\{(\w+=" + metadata_value + r"\s*)*\}"
    return re.sub(pattern=pattern, repl="", string=hlo)


class ParameterConversionFn(Protocol):
    def __call__(self, src: Any, *, dst_layer: BaseLayer) -> NestedTensor:
        """Converts parameters from `src` to parameters for `dst_layer`."""


class Tolerance(NamedTuple):
    rtol: float = 0.001
    atol: float = 0.001


class TestCase(parameterized.TestCase):
    """Base test class."""

    @property
    def data_dir(self):
        return "FAKE"

    def setUp(self):
        super().setUp()
        push_data_dir(self.data_dir)
        utils_spmd.setup(jax_backend=self._jax_backend())

    def _jax_backend(self) -> str:
        # Setup without distributed initialization by default.
        return "cpu"

    def tearDown(self) -> None:
        super().tearDown()
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
    ) -> tuple[Any, Any]:
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
            # Prune empty subtrees so we don't require empty dicts for layers with no params.
            ref_structure = jax.tree_util.tree_structure(prune_empty(params_from_ref))
            test_structure = jax.tree_util.tree_structure(prune_empty(layer_params))
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
                self.assertAlmostEqual(a_value, b_value, msg=f"{a_name}")

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

    def assertAllCloseWithOutliers(self, actual, desired, *, tolerance_map: dict[float, Tolerance]):
        """Like np.testing.assert_allclose, but allows outlier percentiles to be specified.

        `tolerance_map` is mapping of percentile values (between 0 and 1) to `Tolerance` objects.
        Each entry defines the acceptable tolerance for a certain percentile of elements in the
        difference `abs(actual - desired)`. The specified tolerance should be met within the given
        percentile of total elements in `actual` or `desired`.

        Example:
        ```python
        self.assertAllCloseWithOutliers(x, y, tolerance_map={
            1.0: Tolerance(atol=0.2),
            0.95: Tolerance(atol=0.05),
        })
        ```
        This example asserts 100% elements of `abs(x - y)` should be within atol=0.2, and 95%
        elements of `abs(x - y)` should be within atol=0.05.
        """
        assert len(tolerance_map) > 0
        self.assertEqual(actual.shape, desired.shape)
        self.assertEqual(actual.dtype, desired.dtype)
        actual = actual.astype(np.float32)
        desired = desired.astype(np.float32)
        diff = np.abs(actual - desired)
        for percentile, tol in tolerance_map.items():
            percentile = 1 - percentile
            tolerance = tol.atol + tol.rtol * np.abs(desired)
            expected_num_ele = round(diff.size * percentile)
            actual_num_ele = np.count_nonzero(diff > tolerance)
            actual_percent = actual_num_ele / diff.size
            self.assertLessEqual(
                actual_num_ele,
                expected_num_ele,
                msg=f"Expected the number of elements over {tol} to be less than {percentile:.3%}"
                f" of total elements (or {expected_num_ele}), but got {actual_percent:.3%} "
                f"(or {actual_num_ele}). These differences are {diff[diff > tolerance]}. "
                f"Max difference = {diff.max()}",
            )


# TODO(markblee): Move this to axlearn/experiments/test_utils.py, where it's used.
class TrainerConfigTestCase(TestCase):
    """Base class for testing trainer configs."""

    def _test_with_trainer_config(self, trainer_config, mesh_size: Optional[dict[str, int]] = None):
        with jax.checking_leaks(), set_data_dir("FAKE"):
            if mesh_size is None:
                mesh_size = {}
            cfg = copy.deepcopy(trainer_config)
            cfg.dir = cfg.dir or tempfile.mkdtemp()
            cfg.mesh_axis_names = cfg.mesh_axis_names or ("data", "model")
            cfg.mesh_shape = cfg.mesh_shape or (len(jax.devices()), 1)
            cfg.max_step = 3

            # TODO(kelvin-zou): Remove this once bfloat16 bug on CPU is fixed.
            if jax.devices()[0].platform == "cpu":
                if cfg.train_dtype == jnp.bfloat16:
                    cfg.train_dtype = jnp.float32
                for evaler_cfg in cfg.evalers.values():
                    if evaler_cfg.eval_dtype == jnp.bfloat16:
                        evaler_cfg.eval_dtype = jnp.float32

            for evaler_cfg in cfg.evalers.values():
                if getattr(evaler_cfg.eval_policy, "fn", None) is eval_every_n_steps_policy:
                    evaler_cfg.eval_policy.n = 2
                evaler_cfg.vlog = max(evaler_cfg.vlog or 0, 3)

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
                if state_spec.mesh_axes is None:
                    state_spec.mesh_axes = [None] * len(value.shape)
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

    def forward(self, input_batch: NestedTensor, **kwargs) -> tuple[Tensor, NestedTensor]:
        del kwargs
        return jnp.zeros([], dtype=jnp.float32), input_batch.get("aux", {})

    def predict(self, input_batch: NestedTensor) -> NestedTensor:
        return self.forward(input_batch)[1]


class TestWithTemporaryCWD(TestCase):
    """Run all tests in a temp directory to isolate from local env."""

    def run(self, result=None):
        temp_root = tempfile.TemporaryDirectory()
        # Note that using "with temp_root as ..." will only return the dir name.
        # pylint: disable-next=attribute-defined-outside-init
        self._temp_root = temp_root
        with temp_root, temp_chdir(os.path.realpath(self._temp_root.name)):
            super().run(result)


@contextlib.contextmanager
def prng_impl(new_prng_impl: str):
    old_prng_impl = jax.config.jax_default_prng_impl

    def switch(value):
        jax.config.update("jax_default_prng_impl", value)

    switch(new_prng_impl)
    yield
    switch(old_prng_impl)


# Use dataclass so that jax.tree.map does not expand it.
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

    @config_class
    class Config(Initializer.Config):
        library: Required[str] = REQUIRED

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        return f"delegated({self.config.library})"


# For new code, use Nested[ParamInitSpec].
NestedParamInitSpec = Optional[Union[ParamInitSpec, dict[str, Any]]]


def _cast_ordered_dict(params: NestedTensor):
    if isinstance(params, dict):
        params = OrderedDict({k: _cast_ordered_dict(v) for k, v in params.items()})
    return params


def _complete_param_init_spec_tree(
    params: NestedTensor, param_init_specs: list[ParamInitSpec], delegates: dict[str, ParamInitSpec]
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
    layer: BaseLayer, *, delegates: Optional[dict[str, ParamInitSpec]] = None
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
        fan_axes = (
            parameter_spec.fan_axes
            if parameter_spec.fan_axes is not None
            # pylint: disable-next=protected-access
            else self._compute_fan_axes(name, parameter_spec)
        )
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

    def patched_vmap(fn, **vmap_kwargs):
        def wrapped_fn(*args, **kwargs):
            return _cast_ordered_dict(fn(*args, **kwargs))

        return orig_vmap(wrapped_fn, **vmap_kwargs)

    patch_vmap = patch("jax.vmap", side_effect=patched_vmap, autospec=True)

    with patch_init, patch_vmap:
        params = layer.initialize_parameters_recursively(jax.random.PRNGKey(0))
    return _complete_param_init_spec_tree(params, all_param_init_specs, delegates or {})


def read_per_param_settings(
    module: Any, config_name: str, trainer_config: Optional[TrainerConfigFn] = None
) -> dict[str, dict[str, NestedTensor]]:
    """Extracts per-param settings for the given trainer config.

    Given a trainer config specified by `module` and `config_name`, initializes the trainer
    and intercepts calls to `register_per_param_settings` to extract `description` and `settings`
    for each call.

    Learners that require realistic input to their `update()` function may not work with this
    function. E.g., expects specific state updates, gradient shapes, a named forward pass.

    Args:
        module: training config module.
        config_name: training config name.
        trainer_config: Optional, the pre-cached trainer config used in golden config test.

    Returns:
        A nested dict, where the key is the description of the `register_per_param_settings` call,
        and the value is a dictionary. The value maps the learner path to a tree of
        the same structure as the corresponding model parameters, which records the
        per-parameter settings (such as the weight decay scales of each parameter).
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

    all_param_settings = defaultdict(dict)

    def patched_register_per_param_settings(
        settings: NestedTree, *, description: str, path: Optional[str] = None
    ) -> NestedTree:
        # Prune MaskedNode subtrees. If a tree would be made empty by removal of its subtrees,
        # it will also be pruned.
        pruned_settings = prune_tree(
            settings,
            lambda _, v: isinstance(v, optax.MaskedNode) or (isinstance(v, dict) and not v),
        )
        if all_param_settings[description]:
            logging.info("There are multiple per_param_settings of %s.", description)
        # Use the path if provided, else a counter.
        key = path or str(len(all_param_settings[description]))

        all_param_settings[description][key] = pruned_settings
        return settings

    with (
        patch(
            "axlearn.common.utils._register_per_param_settings",
            side_effect=patched_register_per_param_settings,
            autospec=True,
        ),
        patch_init,
    ):
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
        opt_params = jax.tree.map(
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
        zero_grads = jax.tree.map(lambda p: jnp.zeros(1), opt_param_values(opt_params))
        learner = trainer_cfg.learner.set(name="learner").instantiate(parent=None)
        learner_state = learner.init(opt_params)
        zero_grads = jax.tree.map(
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
            inputs=[Updates(opt_params=opt_params, delta_updates=zero_grads, inplace_updates={})],
        )

    return all_param_settings


# TODO(markblee): Update to take prng_key explicitly.
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
    return lower_diag[input_len].astype(jnp.int32)


# TODO(markblee): Update to take prng_key explicitly.
def dummy_segments_positions(
    batch: int, seq_len: int, *, num_segments: int
) -> tuple[Tensor, Tensor]:
    """Builds dummy segment IDs and corresponding positions.

    Example:
        batch: 2
        seq_len: 4
        num_segments: 3
        output: (
            [[1, 2, 2, 2], [1, 1, 2, 2]],  # segment_ids
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
    segment_starts = jax.random.permutation(
        jax.random.PRNGKey(401),
        segment_starts,
        axis=-1,
        independent=True,
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


@contextlib.contextmanager
def temp_chdir(new_cwd: Union[pathlib.Path, str]):
    """Changes into a temp CWD only within the context."""
    old_cwd = os.getcwd()
    os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


L = TypeVar("L", bound=BaseLayer)
M = TypeVar("M", bound=Module)


@contextlib.contextmanager
def bind_module(
    module: ConfigOr[M],
    *,
    is_training: bool = True,
    prng_key: Optional[jax.random.PRNGKey] = None,
    state: Nested[Tensor],
) -> Iterator[M]:
    """Creates a context in which `module` has `state`.`

    This lets you write tests that make calls to a module without needing to call `functional()`
    yourself.

    It is similar in spirit to FLAX's `module.bind()` although that works differently due to the
    fact that FLAX state is only associated with an instance of a module, whereas AXLearn state is
    global.

    Example:
        ```
        cfg = MyModule.default_config()
        with test_utils.bind_layer(cfg) as module:
            result = module.do_something(some_args)
        ```

    Args:
        module: The module to create a context for.
        is_training: Tell the module it is in training or not.
        prng_key: The PRNG key to use. If None, `jax.random.PRNGKey(0)`.
        state: The state to use.

    Returns:
        The initialized module.
    """
    if prng_key is None:
        prng_key = jax.random.PRNGKey(0)

    if isinstance(module, InstantiableConfig):
        if isinstance(module, Module.Config) and isinstance(
            getattr(module, "name", None), RequiredFieldValue
        ):
            setattr(module, "name", "tmp")
        module = module.instantiate(parent=None)
    ctx = InvocationContext(
        name="root",
        parent=None,
        module=module,
        is_training=is_training,
        prng_key=prng_key,
        state=state,
        output_collection=new_output_collection(),
    )
    with set_current_context(ctx):
        yield module


@contextlib.contextmanager
def bind_layer(
    layer: ConfigOr[L],
    *,
    is_training: bool = True,
    prng_key: Optional[jax.random.PRNGKey] = None,
    state: Optional[Nested[Tensor]] = None,
) -> Iterator[L]:
    """Creates a context in which `layer` has state initialized using
    `initialize_parameters_recursively`.

    The only difference between this and `bind_module()` is this calls
    `initialize_parameters_recursively`.

    Example:
        ```
        cfg = Linear.default_config().set(input_dim=5, output_dim=7)
        with test_utils.bind_layer(cfg) as layer:
            result = layer(jnp.ones(5))
        assert result.shape == (7,)
        ```

    Args:
        layer: The layer to initialize.
        is_training: Tell the layer it is in training or not.
        prng_key: The PRNG key to use. If None, `jax.random.PRNGKey(0)`.
        state: The state to use. If None, call `initialize_parameters_recursively()` to initialize
               the state.

    Returns:
        The Initialized module.
    """
    if prng_key is None:
        prng_key = jax.random.PRNGKey(0)

    init_key, ctx_key = jax.random.split(prng_key)

    with bind_module(layer, is_training=is_training, prng_key=ctx_key, state={}) as instance:
        if state is None:
            state = instance.initialize_parameters_recursively(prng_key=init_key)
        current_context().state = state
        yield instance


def initialize_parameters_with_prebuilt(
    layer: BaseLayer, *, prng_key: Tensor, prebuilt: Nested[Union[Tensor, ParameterSpec]]
) -> Nested[Tensor]:
    """Initializes parameters with given prebuilt parameters.

    This is different from `BaseLayer.initialize_parameters_recursively` in two ways:
    - `prebuilt` contains Tensors for prebuilt parameters, ParameterSpec otherwise;
    - This function merges `prebuilt` into the returned tree.

    Args:
        layer: The `layer` for which to initialize parameters.
        prng_key: The random key.
        prebuilt: A Nested tree whose leaf nodes are Tensors if the parameters are prebuilt,
            ParameterSpecs if the parameters should be initialized.

    Returns:
        A Nested Tree with Tensors as leaf nodes. The Tensors will come from `prebuilt` if
        provided, otherwise initialized via `initialize_parameters_recursively`.
    """
    if prebuilt is None:
        return layer.initialize_parameters_recursively(prng_key)
    # A tree where a leaf is a ParameterSpec for a prebuilt param, None otherwise.
    # This is used for `initialize_parameters_recursively`.
    prebuilt_param_specs = jax.tree.map(
        lambda value: (
            ParameterSpec(
                shape=value.shape,
                dtype=value.dtype,
                mesh_axes=value.sharding,
            )
            if isinstance(value, Tensor)
            else None
        ),
        prebuilt,
    )
    logging.debug("prebuilt_param_specs: %s", shapes(prebuilt_param_specs))
    initialized = layer.initialize_parameters_recursively(prng_key, prebuilt=prebuilt_param_specs)
    # Merge `prebuilt` and `initialized`.
    logging.info("prebuilt params: %s", shapes(prebuilt))
    logging.info("initialized params: %s", shapes(initialized))
    return jax.tree.map(
        lambda prebuilt, initialized: (prebuilt if isinstance(prebuilt, Tensor) else initialized),
        prebuilt,
        initialized,
    )
