# Copyright © 2023 Apple Inc.

"""General-purpose utilities."""

import collections
import copy
import dataclasses
import functools
import logging as pylogging
import os
import shlex
import signal
import subprocess
import uuid
from collections.abc import Sequence
from typing import Any, Callable, Optional, TypeVar, Union

import pkg_resources
import psutil
from absl import app, flags, logging

from axlearn.cloud import ROOT_MODULE_NAME
from axlearn.cloud.common.types import ResourceMap
from axlearn.cloud.gcp.tpu import infer_tpu_resources
from axlearn.common.config import REQUIRED, ConfigBase, Configurable, Required, config_class


class FilterDiscoveryLogging(pylogging.Filter):
    """Filters out noisy 'discovery' logging."""

    def filter(self, record):
        if record.levelno == 20:
            return record.module != "discovery"
        return True


def configure_logging(level: int):
    """Configures the logging level and adds FilterDiscoveryLogging.

    Args:
        level: Logging verbosity.
    """
    # Utility to configure logging.
    logging.set_verbosity(level)
    logging.get_absl_handler().addFilter(FilterDiscoveryLogging())


def handle_popen(proc: subprocess.Popen):
    """Waits for subprocess to exit, if return != 0 raises with stdout/stderr.

    Args:
        proc: Subprocess to handle.

    Raises:
        RuntimeError: If subprocess returncode != 0.
    """
    proc.wait()
    if proc.returncode != 0:
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Popen command {proc.args} returned non-zero exit code {proc.returncode}:\n"
            f"stdout={stdout}\n"
            f"stderr={stderr}"
        )


def generate_job_name() -> str:
    """Generate a unique job name."""
    prefix = os.environ.get("USER", "job").replace("_", "")
    return f"{prefix}-{uuid.uuid4().hex.lower()[:6]}"


def generate_job_id() -> str:
    """Generate a unique job uuid."""
    return str(uuid.uuid4())


# TODO(markblee): Consider using git python.
def get_git_revision(revision: str) -> str:
    """Gets the commit hash for the revision."""
    return subprocess.check_output(["git", "rev-parse", revision]).decode("ascii").strip()


def get_package_root(root_module_name: str = ROOT_MODULE_NAME) -> str:
    """Returns the absolute path of the package root, as defined by the directory with name
    `root_module_name`.

    Note that the installed package may not include pyproject.toml or a git directory (as it is
    rooted at `root_module_name`, not the root of the project).

    Args:
        root_module_name: Name of the root module.

    Returns:
        The absolute path of the project root.

    Raises:
        ValueError: If run from outside the package.
    """
    init = curr = os.path.dirname(os.path.realpath(__file__))
    while curr and curr != "/":
        if os.path.basename(curr) == root_module_name:
            return curr
        curr = os.path.dirname(curr)
    raise ValueError(f"Not running within {root_module_name} (searching up from '{init}').")


def get_pyproject_version() -> str:
    """Returns the project version, e.g. X.Y.Z."""
    # TODO(markblee): Fix for nightly
    return pkg_resources.get_distribution(ROOT_MODULE_NAME).version


def parse_kv_flags(kv_flags: Sequence[str], *, delimiter: str = ":") -> dict[str, str]:
    """Parses sequence of k:v into a dict.

    Args:
        kv_flags: A sequence of strings in the format "k:v". If a key appears twice, the last
            occurrence "wins".
        delimiter: The separator between the key and value.

    Returns:
        A dict where keys and values are parsed from "k:v".

    Raises:
        ValueError: If a member of `kv_flags` isn't in the format "k:v".
    """
    metadata = {}
    for kv in kv_flags:
        parts = kv.split(delimiter, maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Expected key{delimiter}value, got {kv}")
        metadata[parts[0]] = parts[1]
    return metadata


def format_table(*, headings: list[str], rows: list[list[str]]) -> str:
    """Formats headings and rows as a table.

    Args:
        headings: Sequence of headings, one for each column.
        rows: Sequence of rows. Each row is itself a sequence of strings, containing values for each
            column. It is assumed that each row has the same number of columns as `headings`.

    Returns:
        A string formatted as a table, consisting of the provided headings and rows.
    """
    rows = [[h.upper() for h in headings]] + rows
    max_lens = [max(len(str(row[i])) for row in rows) for i in range(len(headings))]
    fmt = "".join([f"{{:<{max_len + 6}}}" for max_len in max_lens])
    return "\n" + "\n".join([fmt.format(*[str(v) for v in row]) for row in rows]) + "\n"


def infer_cli_name() -> str:
    """Attempts to infer the CLI name."""
    return os.path.basename(os.environ.get("AXLEARN_CLI_NAME", ROOT_MODULE_NAME))


def subprocess_run(argv: Union[str, Sequence[str]], *args, **kwargs) -> subprocess.CompletedProcess:
    """Runs a command via subprocess.run.

    Main differences are:
    - Automatically splits or joins argv depending on whether shell=True.
    - Logs errors if capture_output=True and subprocess raises.

    Args:
        argv: The command. Can be a string or sequence of strings.
        *args: Forwarded to `subprocess.run`.
        **overrides: Forwarded to `subprocess.run`.

    Returns:
        A completed process.

    Raises:
        CalledProcessError: If the command fails.
    """
    try:
        if not kwargs.get("shell") and isinstance(argv, str):
            argv = shlex.split(argv)
        elif kwargs.get("shell") and isinstance(argv, list):
            argv = shlex.join(argv)
        # pylint: disable-next=subprocess-run-check
        return subprocess.run(argv, *args, **kwargs)
    except subprocess.CalledProcessError as e:
        # Emit the captured stdout/stderr.
        error_msg = (
            f"Command {e.cmd} failed: code={e.returncode}, stdout={e.stdout}, stderr={e.stderr}"
        )
        if kwargs.get("capture_output"):
            logging.error(error_msg)
        raise ValueError(error_msg) from e  # Re-raise.


def canonicalize_to_list(v: Union[str, Sequence[str]], *, delimiter: str = ",") -> list[str]:
    """Converts delimited strings to lists."""
    if not v:
        return []  # Note: "".split(",") returns [""].
    if isinstance(v, str):
        v = [elem.strip() for elem in v.split(delimiter)]
    return list(v)


def canonicalize_to_string(v: Union[str, Sequence[str]], *, delimiter: str = ",") -> str:
    """Converts lists to delimited strings."""
    if not v:
        return ""
    if not isinstance(v, str) and isinstance(v, Sequence):
        v = delimiter.join([elem.strip() for elem in v])
    return str(v)


def parse_action(
    argv: Sequence[str], *, options: Sequence[str], default: Optional[str] = None
) -> str:
    """Parses action from argv, or exits with usage info.

    The action is inferred from the first positional arg in argv[1:] (where argv[0] is interpreted
    as the CLI name).

    Args:
        argv: CLI arguments, possibly including --flags.
        options: Possible actions.
        default: Optional default action if unable to infer action from argv.

    Returns:
        The chosen action.

    Raises:
        absl.app.UsageError: if an invalid action (or no action and default is None) is provided.
    """
    assert default is None or default in options
    action = None
    for arg in argv[1:]:
        arg = arg.strip()
        if not arg.startswith("--"):
            action = arg
            break
    if action not in options:
        action = default
    if action is None or action not in options:  # No action nor default provided.
        raise app.UsageError(f"Invalid action: {action}. Expected one of [{','.join(options)}].")
    return action


def send_signal(popen: subprocess.Popen, sig: int = signal.SIGKILL):
    """Sends a signal (default SIGKILL) to the process (and child processes)."""
    # Note: kill() might leave orphan processes if proc spawned child processes.
    # We use psutil to recursively kill() all children.
    # If changing this fn, please run the `test_send_signal` test manually.
    try:
        parent = psutil.Process(popen.pid)
    except psutil.NoSuchProcess:
        return  # Nothing to do.
    for child in parent.children(recursive=True):
        try:
            child.send_signal(sig)
        except psutil.NoSuchProcess:
            pass  # Ignore NoSuchProcess exception and continue with the next child.
    popen.send_signal(sig)


def copy_blobs(from_prefix: str, *, to_prefix: str):
    """Replicates blobs with the from_prefix to the to_prefix."""

    # tf.io, which `fs` uses for some APIs, increases import time significantly, which hurts CLI
    # experience.
    # pylint: disable-next=import-outside-toplevel
    from axlearn.common import file_system as fs

    # As file_system.copy requires a path to a file when reading from cloud storage,
    # we traverse the `from_prefix` to find and copy all suffixes.
    if not fs.isdir(from_prefix):
        # Copy the file.
        logging.debug("Copying file %s", from_prefix)
        fs.copy(from_prefix, to_prefix, overwrite=True)
        return
    for blob in fs.glob(os.path.join(from_prefix, "*")):
        if fs.isdir(blob):
            sub_directory = os.path.basename(blob)
            logging.info("Copying sub-directory %s", sub_directory)
            to_prefix = os.path.join(to_prefix, sub_directory)
            fs.makedirs(to_prefix)
        copy_blobs(blob, to_prefix=to_prefix)


def merge(base: dict, overrides: dict):
    """Recursively merge overrides into base."""
    if not isinstance(base, dict):
        return overrides
    for k, v in overrides.items():
        base[k] = merge(base.get(k), v)
    return base


_Row = list[Any]


@dataclasses.dataclass(repr=False)
class Table:
    """A table which can be pretty-printed."""

    headings: _Row
    rows: list[_Row]

    def __post_init__(self):
        if not isinstance(self.headings, Sequence):
            raise ValueError(f"Expected headings to be a sequence: {self.headings}")
        if not isinstance(self.rows, Sequence):
            raise ValueError(f"Expected rows to be a sequence: {self.rows}")
        for row in self.rows:
            self._check_row(row)

    def _check_row(self, row: _Row):
        if not isinstance(row, Sequence):
            raise ValueError(f"Expected row to be a sequence: {row}")
        if len(self.headings) != len(row):
            raise ValueError(f"Expected row to have {len(self.headings)} columns.")

    def add_row(self, row: _Row):
        """Adds a row to the table."""
        self._check_row(row)
        self.rows.append(row)

    def add_col(self, key: str, col: list[Any]):
        """Adds a named column to the table. The name will be added as a heading."""
        col = list(col)
        if not self.rows:
            self.headings.append(key)
            self.rows = col
        elif len(self.rows) != len(col):
            raise ValueError(f"Expected column to have {len(self.rows)} rows.")
        else:
            self.headings.append(key)
            for i, row in enumerate(self.rows):
                row.append(col[i])

    def get_col(self, *keys: str) -> list[_Row]:
        """Gets one or more named columns from the table."""
        idx = [self.headings.index(k) for k in keys]
        return [[row[i] for i in idx] for row in self.rows]

    def sort(self, key: Callable[[_Row], Any], reverse: bool = False):
        """Sorts the table. Heading remains unchanged."""
        self.rows.sort(key=key, reverse=reverse)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Table) and other.headings == self.headings and other.rows == self.rows
        )

    def __repr__(self) -> str:
        """Formats the table for printing."""
        return format_table(headings=self.headings, rows=self.rows)


@config_class
class AcceleratorConfig(ConfigBase):
    """Configures job resources, e.g. TPU or GPU.

    Attributes:
        instance_type: Instance type, e.g. tpu-v4-8. The format of instance type is
            `<accelerator_type>-<user_facing_name>`. As an example, a list of accelerator types and
            user facing names for GCP can be found in `axlearn.cloud.gcp.system_characteristics`.
        num_replicas: Number of replicas, e.g. TPU slices.
    """

    instance_type: Required[str] = REQUIRED
    num_replicas: int = 1


def accelerator_flags(flag_values: flags.FlagValues, **kwargs):
    """Defines resource flags, e.g. --instance_type and --num_replicas."""
    flags.DEFINE_string("instance_type", None, "Instance type.", flag_values=flag_values, **kwargs)
    flags.DEFINE_integer(
        "num_replicas", 1, "Number of replicas.", flag_values=flag_values, **kwargs
    )


def infer_resources(cfg: ConfigBase) -> ResourceMap[int]:
    """Traverses a job config to identify resources based on `AcceleratorConfig`.

    Args:
        cfg: An arbitrary config. Resources should be configured via `AcceleratorConfig`.

    Returns:
        A mapping from resource type to usage.

    Raises:
        NotImplementedError: If unable to infer resources for an `instance_type`.
    """

    total_resources = collections.defaultdict(int)

    def visit_fn(_, value):
        if isinstance(value, AcceleratorConfig):
            if value.instance_type.startswith("tpu-"):
                resources = infer_tpu_resources(value.instance_type, value.num_replicas)
                for resource, usage in resources.items():
                    total_resources[resource] += usage
            else:
                raise NotImplementedError(value.instance_type)

    def enter_fn(_, value, default_kv):
        return None if isinstance(value, AcceleratorConfig) else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)
    return dict(total_resources)


_FLAG_NAMESPACE_ATTRIBUTE = "__axlearn_flag_namespace_mapping"


def define_flags(cfg: ConfigBase, fv: flags.FlagValues):
    """Define flags on `fv` by recursively invoking `cfg.klass.define_flags`.

    Flags are defined in topological order, i.e., parent flags will be defined prior to defining
    child flags.

    Args:
        cfg: A config instance. It does not need to be a `FlagConfigurable` config.
        fv: Parsed flag values instance. The same `fv` is used for the entire config hierarchy.

    Raises:
        ValueError: If `cfg` is not a config.
    """
    # pylint: disable=protected-access
    if not isinstance(cfg, ConfigBase):
        raise ValueError(f"Expected {ConfigBase}, got: {type(cfg)}")

    def visit_fn(_, value):
        if not isinstance(value, FlagConfigurable.Config):
            return
        for namespace, child in _get_namespaced_config(value).items():
            sub_fv = flags.FlagValues()
            define_flags(child, sub_fv)
            # Flatten the child flags into `fv`. See `FlagValues.append_flag_values` for ref.
            # The main difference is that we namespace the flags by child name.
            for flag_name, flag in sub_fv._flags().items():
                # absl flattens short names into `fv` during __setattr__.
                # Keep things simple for now by limiting to verbose names.
                if flag.short_name:
                    raise NotImplementedError(
                        f"Short names are currently not supported: {flag.short_name}"
                    )
                fv[f"{namespace}.{flag_name}"] = flag

    def enter_fn(_, value, default_kv):
        if not isinstance(value, Configurable.Config) or not hasattr(value.klass, "define_flags"):
            return default_kv
        klass: FlagConfigurable = value.klass
        klass.define_flags(fv)
        if hasattr(klass, _FLAG_NAMESPACE_ATTRIBUTE):
            return None  # Enter visit_fn.
        return default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)


_C = TypeVar("_C", bound=ConfigBase)


def from_flags(cfg: _C, fv: flags.FlagValues, **kwargs) -> _C:
    """Read values from `fv` by recursively invoking `cfg.klass.from_flags`.

    The config precedence is `kwargs` followed by `fv` followed by `cfg`. In other words, `kwargs`
    override any values specified as flags, which override any existing values on `cfg`.

    Args:
        cfg: A config instance. It does not need to be a `FlagConfigurable` config.
        fv: Parsed flag values instance. The same `fv` is used for the entire config hierarchy.
        **kwargs: Forwarded to `cfg.klass.from_flags(...)`.

    Returns:
        The modified config instance. The modifications happen in-place and is also returned for
        convenience (consistent with `cfg.set()`).

    Raises:
        ValueError: If `cfg` is not a config.
    """
    # pylint: disable=protected-access
    if not isinstance(cfg, ConfigBase):
        raise ValueError(f"Expected {ConfigBase}, got: {type(cfg)}")

    def visit_fn(_, value, method):
        if not isinstance(value, FlagConfigurable.Config):
            return
        for namespace, child in _get_namespaced_config(value).items():
            sub_fv = copy.deepcopy(fv)
            fv_flags_dict = fv._flags()

            # Propagate the following flags from fv to sub_fv:
            # 1. flags starting with `namespace` will have the namespace prefix stripped;
            # 2. if the resulting flag shares a name with an ancestor, inherit the default.
            for flag_name, flag in fv_flags_dict.items():
                parts = flag_name.split(".", maxsplit=1)
                if len(parts) < 2 or parts[0] != namespace:
                    continue
                sub_fv.remove_flag_values([flag_name])
                if parts[1] in fv_flags_dict:
                    flag._set_default(fv_flags_dict[parts[1]].value)
                sub_fv[parts[1]] = flag

            method(child, sub_fv)

    def enter_set_defaults(_, value, default_kv):
        if not isinstance(value, Configurable.Config) or not hasattr(value.klass, "set_defaults"):
            return default_kv
        klass: FlagConfigurable = value.klass
        klass.set_defaults(fv)
        if hasattr(klass, _FLAG_NAMESPACE_ATTRIBUTE):
            return None  # Enter visit_fn.
        return default_kv

    def enter_from_flags(_, value, default_kv):
        if not isinstance(value, Configurable.Config) or not hasattr(value.klass, "from_flags"):
            return default_kv
        klass: FlagConfigurable = value.klass
        klass.from_flags(fv, prebuilt_cfg=value, **kwargs)
        if hasattr(klass, _FLAG_NAMESPACE_ATTRIBUTE):
            return None  # Enter visit_fn.
        return default_kv

    visit_set_defaults = functools.partial(
        visit_fn, method=lambda child, fv: child.klass.set_defaults(fv)
    )
    visit_from_flags = functools.partial(
        visit_fn, method=lambda child, fv, kwargs=kwargs: from_flags(child, fv, **kwargs)
    )

    # Set all defaults across the hierarchy first, so that default override can happen.
    # This ensures that fv defaults are consistent.
    cfg.visit(visit_fn=visit_set_defaults, enter_fn=enter_set_defaults)
    # Read configs from flags.
    cfg.visit(visit_fn=visit_from_flags, enter_fn=enter_from_flags)
    return cfg


class FlagConfigurable(Configurable):
    """A Configurable object that also supports flag-based configuration."""

    Config = Configurable.Config

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        """Subclasses can override this method to define absl flags to be read by `from_flags()`.

        This method should only define flags that are used by this class, and not any child classes,
        which allows each class to be encapsulated.

        To define flags recursively, use `utils.define_flags`.
        """
        del fv

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> Config:
        """Populate config partially using parsed absl flags.

        This method should only set configs that are used by this class, and not any child classes,
        which allows each class to be encapsulated.

        To read flags recursively, use `utils.from_flags`.
        """
        cfg: FlagConfigurable.Config = kwargs.pop("prebuilt_cfg", cls.default_config())
        flag_values = {**fv.flag_values_dict(), **kwargs}
        return cfg.set(
            **{field: flag_values[field] for field in cfg.keys() if field in flag_values}
        )

    @classmethod
    def set_defaults(cls, fv: flags.FlagValues):
        """Sets default values for `fv`.

        Instead of setting defaults in `define_flags` or `from_flags`, this method applies defaults
        after flag parsing (allowing access to values in `fv`) while ensuring that a parent's
        `set_defaults` is invoked before the child's.

        This allows the child to have more flexibility in choosing whether a default value should be
        overridden, inherited, or inferred from another flag. For example, to inherit a default, one
        can do:
        ```
        # Calling the super method defines parent flags first.
        super().set_defaults(fv)

        # Use the existing default (if any), else use our own default.
        fv.set_default("my_flag", fv.my_flag or "backup-default")
        ```
        On the other hand, to override a default, one can do:
        ```
        # Register parent defaults first.
        super().set_defaults(fv)

        # Override the default for "my_flag".
        fv.set_default("my_flag", "override-default")
        ```
        Or, to infer a flag from another:
        ```
        # Register parent defaults first.
        super().set_defaults(fv)

        # Override the default for "my_flag" using the value of another flag.
        fv.set_default("my_flag", f"{fv.my_other_flag}-with-suffix")
        ```
        """
        del fv


_F = TypeVar("_F", bound=FlagConfigurable)


# This is provided as a decorator so that users do not need to modify inheritance chains.
def namespaced(mapping: str):
    """A class decorator that wraps `FlagConfigurable`s with flag namespaces.

    The config field `mapping` should define a mapping from namespace to child `FlagConfigurable`.
    When using `define_flags` and `from_flags`, each child will automatically have its flags
    namespaced, s.t. the same configs can be set to different values across children.

    For example, one can define a composite job that uses a config field "inner" to define the
    namespace to child mapping:
    ```
    @namespaced('inner')
    class CompositeFlagConfigurable(FlagConfigurable):

        @config_class
        class Config(FlagConfigurable.Config):
            inner: Required[dict[str, FlagConfigurable]] = REQUIRED

        def __call__(self):
            outputs = []
            for name, child in self._inner.items():
                outputs.extend(child(...))
            return outputs

    JobAB = CompositeFlagConfigurable.default_config().set(inner={"a": JobA, "b": JobB})
    ```

    Supposing that `JobA` and `JobB` both define the flags `--name` and `--command`, the
    corresponding composite flags will be defined:
    ```
    --a.name --a.command
    --b.name --b.command
    ```
    This allows for providing flags to specific nested jobs even if they follow similar interfaces.
    """

    def wrapper(cls: type[_F]) -> type[_F]:
        existing = getattr(cls, _FLAG_NAMESPACE_ATTRIBUTE, None)
        if existing is not None and existing != mapping:
            raise ValueError(f"{cls} already defines a namespace: {existing}")
        setattr(cls, _FLAG_NAMESPACE_ATTRIBUTE, mapping)
        return cls

    return wrapper


def _get_namespaced_config(cfg: ConfigBase) -> dict[str, ConfigBase]:
    """Obtains the value of cfg that defines the {namespace: child} mapping.

    If the cfg doesn't define a namespace attribute, an empty dict will be returned.
    """
    config_key = getattr(cfg.klass, _FLAG_NAMESPACE_ATTRIBUTE, None)
    if config_key is None:
        return {}
    mapping = getattr(cfg, config_key, None)
    if not isinstance(mapping, dict):
        raise ValueError(f"{type(cfg)} does not define a mapping at {config_key}.")
    return mapping
