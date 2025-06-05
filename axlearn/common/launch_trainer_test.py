# Copyright © 2024 Apple Inc.

"""Tests trainer launch utilities."""
import copy
from collections.abc import Sequence
from typing import Optional
from unittest import mock

from absl import flags
from absl.testing import parameterized

from axlearn.common import launch_trainer
from axlearn.common.test_utils import TestCase


def _mock_get_named_trainer_config(config_name: str, *, config_module: str) -> mock.MagicMock:
    valid_module_paths = {
        "local_module": {"config": mock.MagicMock()},
        "root_module.local_module": {"config": mock.MagicMock()},
        "axlearn.experiments.experiment_module": {"experiment_config": mock.MagicMock()},
    }
    if config_module not in valid_module_paths:
        raise ImportError(config_module)

    return valid_module_paths[config_module][config_name]


def _flag_values_from_dict(flag_values: dict) -> flags.FlagValues:
    # Avoid mutating global FLAGS.
    fv = copy.deepcopy(flags.FLAGS)
    for k, v in flag_values.items():
        fv.set_default(k, v)
    fv.mark_as_parsed()
    return fv


class GetTrainerConfigTest(TestCase):
    """Tests get_trainer_config."""

    @parameterized.parameters(
        # Attempt to import from local module (which exists).
        dict(
            flag_values=dict(config="config", config_module="local_module"),
            expect_args=("config",),
            expect_kwargs=dict(config_module="local_module"),
        ),
        # Import from axlearn.experiments, since local module is not found.
        dict(
            flag_values=dict(config="experiment_config", config_module="experiment_module"),
            expect_args=("experiment_config",),
            expect_kwargs=dict(config_module="axlearn.experiments.experiment_module"),
        ),
        # If specified, attempt to import from root_module directly.
        dict(
            flag_values=dict(config="config", config_module="root_module.local_module"),
            expect_args=("config",),
            expect_kwargs=dict(config_module="root_module.local_module"),
        ),
        # If specified, use trainer_config_fn directly.
        dict(
            flag_values=dict(config="config", config_module="root_module.local_module"),
            trainer_config=mock.MagicMock(),
            expect_args=(),
            expect_kwargs={},
        ),
    )
    def test_get_trainer_config(
        self,
        flag_values: dict,
        expect_args: Sequence[str],
        expect_kwargs: dict,
        trainer_config: Optional[mock.Mock] = None,
    ):
        fv = _flag_values_from_dict(flag_values)

        # Mock the actual call to avoid depending on experiments.
        with mock.patch(
            f"{launch_trainer.__name__}.get_named_trainer_config",
            side_effect=_mock_get_named_trainer_config,
        ) as mock_fn:
            trainer_config_fn = None
            if trainer_config is not None:
                trainer_config_fn = lambda: trainer_config

            cfg = launch_trainer.get_trainer_config(
                flag_values=fv, trainer_config_fn=trainer_config_fn
            )
            if trainer_config is not None:
                self.assertEqual(cfg, trainer_config)
                mock_fn.assert_not_called()
            else:
                mock_fn.assert_called_with(*expect_args, **expect_kwargs)

    @parameterized.parameters(
        # If config cannot be found in local module, we fallback to axlearn.experiments.
        dict(
            flag_values=dict(config="unknown_config", config_module="local_module"),
            expect_raises=ImportError("axlearn.experiments.local_module"),
        ),
    )
    def test_get_trainer_config_failure(self, flag_values: dict, expect_raises: Exception):
        fv = _flag_values_from_dict(flag_values)

        # Mock the actual call to avoid depending on experiments.
        patch_fn = mock.patch(
            f"{launch_trainer.__name__}.get_named_trainer_config",
            side_effect=_mock_get_named_trainer_config,
        )
        with patch_fn, self.assertRaisesRegex(type(expect_raises), str(expect_raises)):
            launch_trainer.get_trainer_config(flag_values=fv)
