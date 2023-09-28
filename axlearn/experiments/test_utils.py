# Copyright © 2023 Apple Inc.

"""Utilities for testing experiments."""
# pylint: disable=no-self-use
import enum
import os.path
import re
import sys
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from axlearn.common.test_utils import (
    ParamInitSpec,
    TestCase,
    ThirdPartyInitializer,
    read_param_init_specs_recursively,
    read_per_param_settings,
)
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import flatten_items, set_data_dir
from axlearn.experiments.trainer_config_utils import TrainerConfigFn
from axlearn.huggingface.hf_module import HF_MODULE_KEY


def named_parameters(
    module: ModuleType, *, match_by_name: Optional[str] = None
) -> List[Tuple[str, ModuleType, str, TrainerConfigFn]]:
    """Obtains the named parameters for golden config test.

    Args:
        module: A module.
        match_by_name: A string which will be used for selecting config_names for golden config.

    Returns:
        A list of
        - a string with {module}.{name}
        - a ModuleType module
        - a config name
        - the trainer config for the module.config_name.
    """
    with set_data_dir("$DATA_DIR"):
        final_named_trainer_configs = {}
        named_trainer_configs = getattr(module, "named_trainer_configs")()
        if match_by_name:
            # If match_by_name is not None, filter config_names based on match_by_name.
            for name in named_trainer_configs.keys():
                if re.fullmatch(match_by_name, name):
                    final_named_trainer_configs[name] = named_trainer_configs[name]
        else:
            final_named_trainer_configs = named_trainer_configs
        return [
            (f"{module.__name__}.{name}", module, name, final_named_trainer_configs[name])
            for name in final_named_trainer_configs.keys()
        ]


def param_init_debug_string(
    cfg: SpmdTrainer.Config, *, kv_separator=": ", field_separator="\n"
) -> str:
    layer = cfg.model.set(name="init_debug_string").instantiate(parent=None)
    param_init_specs = read_param_init_specs_recursively(
        layer,
        delegates={
            HF_MODULE_KEY: ParamInitSpec(
                shape=None,
                initializer=ThirdPartyInitializer.default_config().set(library="hf"),
                fan_axes=None,
            ),
        },
    )
    lines = []
    for name, init_specs in flatten_items(param_init_specs):
        init_str = init_specs.initializer.debug_string(
            name=name, shape=init_specs.shape, axes=init_specs.fan_axes
        )
        lines.append(f"{name}{kv_separator}{init_str}")
    return field_separator.join(lines)


def per_param_setting_debug_string(
    module: str,
    config_name: str,
    trainer_config: TrainerConfigFn,
    *,
    kv_separator=": ",
    field_separator="\n",
) -> Dict[str, str]:
    """Given module and config, returns a dictionary of read_per_param_settings debug strings.

    If no register_per_param_settings is called, an assertion error is raised.
    """
    all_param_settings = read_per_param_settings(
        module=module, config_name=config_name, trainer_config=trainer_config
    )
    if not all_param_settings:
        raise AssertionError(
            f"{module}.{config_name} does not use per_param_setting_by_path."
            "It is safe to remove this test."
        )

    def _settings_to_string(settings):
        lines = []
        for param_name, param_setting in flatten_items(settings):
            lines.append(f"{param_name}{kv_separator}{param_setting}")
        return field_separator.join(lines)

    return {key: _settings_to_string(settings) for key, settings in all_param_settings.items()}


@enum.unique
class GoldenTestType(enum.Enum):
    CONFIG = "config"
    INIT = "init"
    # We cover both coupled (l2_regularizer) and decoupled weight decay in this test.
    REGULARIZER = "regularizer"
    PARAM_UPDATE = "param_update"


class BaseGoldenConfigTest(TestCase):
    """Tests against golden configs."""

    @property
    def data_dir(self):
        return "$DATA_DIR"

    # This method is automatically invoked by pytest. We assume the existence of a conftest.py in
    # the same directory, which defines these options.
    @pytest.fixture(autouse=True)
    def _flags(self, request):
        # pylint: disable-next=attribute-defined-outside-init
        self._update = request.config.getoption("--update")

    def _test(
        self,
        module: str,
        config_name: str,
        trainer_config: TrainerConfigFn,
        *,
        test_type: GoldenTestType,
    ):
        if self._update:
            self._update_golden_file(
                module, config_name, trainer_config=trainer_config, test_type=test_type
            )
        else:
            self._check_against_golden_file(
                module, config_name, trainer_config=trainer_config, test_type=test_type
            )

    @property
    def _filepath(self):
        # We read __file__ from the module in which the class (or subclass) is defined. This
        # allows subclasses in other directories to define their own golden tests.
        return sys.modules[self.__module__].__file__

    @property
    def _testdata(self):
        return os.path.join(os.path.dirname(self._filepath), "testdata")

    def _golden_file_path(self, module: str, config_name: str, test_type: GoldenTestType) -> str:
        if test_type == GoldenTestType.CONFIG:
            suffix = ""
        elif test_type in (
            GoldenTestType.INIT,
            GoldenTestType.REGULARIZER,
            GoldenTestType.PARAM_UPDATE,
        ):
            suffix = "_" + test_type.value
        else:
            raise ValueError(f"{test_type} is not supported.")
        return os.path.join(self._testdata, module.__name__, f"{config_name}{suffix}.txt")

    # pylint: disable-next=too-many-branches
    def _to_string(
        self,
        *,
        module: str,
        config_name: str,
        trainer_config: TrainerConfigFn,
        test_type: GoldenTestType,
    ) -> str:
        if test_type == GoldenTestType.CONFIG:
            cfg = trainer_config()
            debug_str = cfg.debug_string()
        elif test_type == GoldenTestType.INIT:
            cfg = trainer_config()
            debug_str = param_init_debug_string(cfg)
        elif test_type == GoldenTestType.REGULARIZER:
            debug_str = self._per_param_settings(
                module=module,
                config_name=config_name,
                trainer_config=trainer_config,
                setting_types=("weight_decay_scale", "l2_regularizer_scale"),
            )
        elif test_type == GoldenTestType.PARAM_UPDATE:
            debug_str = self._per_param_settings(
                module=module,
                config_name=config_name,
                trainer_config=trainer_config,
                setting_types=("update_scale", "gradient_scale", "learner_update_type"),
            )
        else:
            raise ValueError(f"{test_type} is not supported.")
        return debug_str

    def _per_param_settings(
        self,
        *,
        module: str,
        config_name: str,
        trainer_config: TrainerConfigFn,
        setting_types: Sequence[str],
    ) -> str:
        valid_per_param_setting_keys = {
            "weight_decay_scale",
            "l2_regularizer_scale",
            "update_scale",
            "gradient_scale",
            "learner_update_type",
        }
        settings_dict = per_param_setting_debug_string(
            module=module, config_name=config_name, trainer_config=trainer_config
        )
        invalid_keys = set(settings_dict.keys()).difference(valid_per_param_setting_keys)
        if invalid_keys:
            raise ValueError(
                f"Got invalid param_update keys: {invalid_keys}. "
                f"Valid options are: {valid_per_param_setting_keys}."
            )

        def sep_line(name: str):
            return "=" * 20 + name + "=" * 20 + "\n"

        debug_str = ""
        for setting_type in setting_types:
            assert setting_type in valid_per_param_setting_keys, setting_type
            if setting_type in settings_dict:
                debug_str += sep_line(name=setting_type)
                debug_str += settings_dict[setting_type]
                debug_str += "\n"
        if not debug_str:
            raise ValueError(f"No per param settings for {setting_types} is found.")
        return debug_str

    def _check_against_golden_file(
        self,
        module: str,
        config_name: str,
        *,
        trainer_config: TrainerConfigFn,
        test_type: GoldenTestType,
    ):
        with open(self._golden_file_path(module, config_name, test_type), encoding="utf-8") as f:
            golden_str = f.read()
        try:
            actual_str = self._to_string(
                module=module,
                config_name=config_name,
                trainer_config=trainer_config,
                test_type=test_type,
            )
            self.assertListEqual(golden_str.split("\n"), actual_str.split("\n"))
        except AssertionError as e:
            raise AssertionError(
                f"Golden {test_type.value} files have changed. If this is expected, run "
                f"`pytest -n auto {self._filepath} --update` to update golden files."
            ) from e

    def _update_golden_file(
        self,
        module: str,
        config_name: str,
        *,
        trainer_config: TrainerConfigFn,
        test_type: GoldenTestType,
    ):
        config_path = self._golden_file_path(module, config_name, test_type)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                self._to_string(
                    module=module,
                    config_name=config_name,
                    trainer_config=trainer_config,
                    test_type=test_type,
                )
            )
