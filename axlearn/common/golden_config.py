# Copyright © 2026 Apple Inc.

"""Utilities for per-module golden config tests.

Example test file::

    import os
    from axlearn.common import golden_config
    from axlearn.experiments.text.gpt import c4_trainer

    if __name__ == "__main__":
        golden_config.test_main(c4_trainer, os.path.join(os.path.dirname(__file__), "testdata"))

With filtering::

    golden_config.test_main(c4_trainer, testdata_dir, match=r"fuji-.*")

To update golden files::

    bazel test //axlearn/experiments/... \\
      --test_env=UPDATE_GOLDEN=1 \\
      --test_env=WORKSPACE_ROOT=$(pwd) \\
      --cache_test_results=no \\
      --strategy=TestRunner=local
"""

from __future__ import annotations

import os
import re
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Optional

from axlearn.common.module import HF_MODULE_KEY
from axlearn.common.test_utils import (
    ParamInitSpec,
    ThirdPartyInitializer,
    read_param_init_specs_recursively,
    read_per_param_settings,
)
from axlearn.common.utils import flatten_items, set_data_dir

if TYPE_CHECKING:
    from axlearn.common.config import TrainerConfigFn


def _source_path(golden_file: str) -> str:
    """Map a Bazel runfiles path back to the source tree.

    When WORKSPACE_ROOT is set, extracts the workspace-relative portion from
    the runfiles path and joins it with WORKSPACE_ROOT.
    """
    workspace_root = os.environ.get("WORKSPACE_ROOT")
    if not workspace_root:
        return golden_file
    # Runfiles path: .../foo.runfiles/{workspace_name}/{rel_path}
    marker = ".runfiles" + os.sep
    idx = golden_file.find(marker)
    if idx < 0:
        return golden_file
    after_marker = golden_file[idx + len(marker) :]
    sep_idx = after_marker.find(os.sep)
    if sep_idx < 0:
        return golden_file
    return os.path.join(workspace_root, after_marker[sep_idx + 1 :])


def _check_or_update(golden_file: str, actual: str):
    if actual and not actual.endswith("\n"):
        actual += "\n"

    if os.environ.get("UPDATE_GOLDEN"):
        dest = _source_path(golden_file)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(actual)
        return

    try:
        with open(golden_file, "r", encoding="utf-8") as f:
            expected = f.read()
    except FileNotFoundError:
        raise AssertionError(
            f"Golden file not found: {golden_file}\nRun with UPDATE_GOLDEN=1 to generate."
        ) from None

    if actual != expected:
        raise AssertionError(
            f"Golden file mismatch: {golden_file}\nRun with UPDATE_GOLDEN=1 to update."
        )


def check(config_fn: TrainerConfigFn, golden_file: str):
    """Assert that the trainer config debug string matches the golden file."""
    with set_data_dir("$DATA_DIR"):
        cfg = config_fn()
    _check_or_update(golden_file, cfg.debug_string())


def check_init(config_fn: TrainerConfigFn, golden_file: str):
    """Assert that parameter init specs match the golden file."""
    with set_data_dir("$DATA_DIR"):
        cfg = config_fn()
    layer = cfg.model.set(name="init_debug_string").instantiate(parent=None)
    specs = read_param_init_specs_recursively(
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
    for name, s in flatten_items(specs):
        lines.append(
            f"{name}: {s.initializer.debug_string(name=name, shape=s.shape, axes=s.fan_axes)}"
        )
    _check_or_update(golden_file, "\n".join(lines))


def _per_param_text(config_fn: TrainerConfigFn, setting_keys: tuple[str, ...]) -> str:
    """Generate per-param settings text filtered by setting_keys."""
    with set_data_dir("$DATA_DIR"):
        settings = read_per_param_settings(module=None, config_name="", trainer_config=config_fn)
    lines = []
    for description, by_learner in settings.items():
        if description not in setting_keys:
            continue
        for learner_path, param_tree in by_learner.items():
            lines.append("=" * 20 + f"{description} {learner_path}" + "=" * 20)
            for param_name, value in flatten_items(param_tree):
                lines.append(f"{param_name}: {value}")
    return "\n".join(lines)


def check_regularizer(config_fn: TrainerConfigFn, golden_file: str):
    """Assert that weight decay / L2 regularizer scales match the golden file."""
    text = _per_param_text(config_fn, ("weight_decay_scale", "l2_regularizer_scale"))
    _check_or_update(golden_file, text)


def check_param_update(config_fn: TrainerConfigFn, golden_file: str):
    """Assert that parameter update specs match the golden file."""
    text = _per_param_text(config_fn, ("update_scale", "gradient_scale", "learner_update_type"))
    _check_or_update(golden_file, text)


def check_learner_rule(config_fn: TrainerConfigFn, golden_file: str):
    """Assert that learner rule configs match the golden file."""
    text = _per_param_text(config_fn, ("learner_rule",))
    _check_or_update(golden_file, text)


_SUFFIXES = {
    check: ".txt",
    check_init: "_init.txt",
    check_regularizer: "_regularizer.txt",
    check_param_update: "_param_update.txt",
    check_learner_rule: "_learner_rule.txt",
}


def test_main(
    module: ModuleType,
    testdata_dir: str,
    *,
    match: Optional[str] = None,
    checks: tuple = (check, check_init),
):
    """Run golden config tests for all configs in a module.

    Args:
        module: Module with a named_trainer_configs() function.
        testdata_dir: Path to the testdata directory.
        match: Optional regex to select config names.
        checks: Tuple of check functions to run. Each takes (config_fn, golden_file).
            Defaults to (check, check_init, check_regularizer).
    """
    with set_data_dir("$DATA_DIR"):
        configs = module.named_trainer_configs()

    if match:
        configs = {n: fn for n, fn in configs.items() if re.fullmatch(match, n)}

    failures = []
    for name, config_fn in configs.items():
        for check_fn in checks:
            golden_file = os.path.join(testdata_dir, f"{name}{_SUFFIXES[check_fn]}")
            try:
                check_fn(config_fn, golden_file)
            except (AssertionError, KeyError, TypeError, ValueError) as e:
                failures.append(f"{name} ({check_fn.__name__}): {e}")

    if failures:
        print(f"FAILED: {len(failures)} failures out of {len(configs)} configs:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"PASSED: {len(configs)} configs, {len(checks)} checks each")
