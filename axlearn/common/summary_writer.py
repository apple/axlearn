# Copyright © 2023 Apple Inc.

"""Utilities for writing summaries."""

import contextlib
import enum
import numbers
import os
from functools import wraps
from types import FunctionType
from typing import Any, Callable, Literal, Optional, Sequence, Union

import jax
import numpy as np
from absl import logging
from tensorflow import summary as tf_summary

from axlearn.common import file_system as fs
from axlearn.common.config import REQUIRED, ConfigBase, Required, RequiredFieldValue, config_class
from axlearn.common.module import Module
from axlearn.common.summary import AudioSummary, ImageSummary, Summary
from axlearn.common.utils import Nested, NestedTensor, Tensor, tree_paths

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class CheckpointerAction(str, enum.Enum):
    """Represents the checkpointer action corresponding to a checkpoint summary.

    Attributes:
        RESTORE: The model was restored from the checkpoint.
        SAVE: The model was saved to the checkpoint.
    """

    RESTORE = "RESTORE"
    SAVE = "SAVE"


def processor_zero_only(fn: Callable) -> Callable:
    """Decorator to use for operations that should only happen on the main process.

    If the function is called from a worker process, this decorator will return None.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
        if jax.process_index() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper


class BaseWriter(Module):
    """Base summary writer."""

    @config_class
    class Config(Module.Config):
        """Configures BaseWriter."""

        dir: Required[str] = REQUIRED  # The output directory.

    def log_config(self, config: ConfigBase, step: int = 0):
        """Log the config.

        Args:
            config: The config to log.
            step: The step to log the config.
        """
        raise NotImplementedError

    def log_checkpoint(
        self,
        ckpt_dir: str,
        *,
        state: NestedTensor,
        action: CheckpointerAction,
        step: int = 0,
    ):
        """Log a checkpoint. The default implementation is no-op.

        Args:
            ckpt_dir: The location of the checkpoint to log.
            state: The state to store.
            action: Represents a type of checkpoint action.
            step: Training step.
        """

    # We adapt the args and kwargs from base Module to arguments specific to summary writer,
    # and drop the method argument since the caller does not decide which method to call.
    # pylint: disable=arguments-differ
    def __call__(self, step: int, values: dict[str, Any]):
        """Log data to disk.

        Args:
            step: The step.
            values: The values to log.
        """
        raise NotImplementedError


class CompositeWriter(BaseWriter):
    """A collection of named writers.

    This class automatically sets the `dir` field of any child writers.
    """

    @config_class
    class Config(BaseWriter.Config):
        writers: Required[dict[str, BaseWriter.Config]] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional["Module"]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._writers: list[BaseWriter] = []
        for writer_name, writer_cfg in cfg.writers.items():
            self._writers.append(
                self._add_child(writer_name, writer_cfg.set(dir=os.path.join(cfg.dir, writer_name)))
            )

    @property
    def writers(self) -> list[BaseWriter]:
        """A list of writers."""
        return self._writers

    def log_config(self, config: ConfigBase, step: int = 0):
        writer: BaseWriter
        for writer in self._writers:
            writer.log_config(config, step=step)

    def __call__(self, step: int, values: dict[str, Any]):
        writer: BaseWriter
        for writer in self._writers:
            writer(step, values)

    def log_checkpoint(
        self,
        ckpt_dir: str,
        *,
        state: NestedTensor,
        action: CheckpointerAction,
        step: int = 0,
    ):
        for writer in self._writers:
            writer.log_checkpoint(ckpt_dir=ckpt_dir, state=state, action=action, step=step)


class NoOpWriter(BaseWriter):
    """A writer that does nothing. Used by testing."""

    def log_config(self, config: ConfigBase, step: int = 0):
        pass

    def __call__(self, step: int, values: dict[str, Any]):
        pass


# Enumeration of summary types for configurable write intervals.
# Used to specify the summary category when configuring type-specific `write_every_n_steps_map`
# in `SummaryWriter.Config`.
# We use `Literal` instead of `Enum` because W&B config logging only supports Python primitives.
# Values:
#     Scalar: Scalar values, e.g., loss, learning rate.
#     Tensor: Tensor values whose ndim is neither 0 (SCALAR) nor 4 (IMAGE).
#     Text: Text summaries, such as configs or notes.
#     Audio: Audio samples, such as waveform outputs.
#     Image: Image summaries, including input/output visualizations.
SummaryKind = Literal["Scalar", "Tensor", "Text", "Audio", "Image"]


def _match_summary_type(
    kind: SummaryKind,
    *,
    value: Union[Summary, Tensor],
    raw_value: Union[np.ndarray, numbers.Number, str],
) -> bool:
    """Checks whether a given value is appropriate for the specified summary kind.

    This is used to determine whether a particular summary value (or wrapper) matches
    the expected structure for a given summary type. This ensures that logging backends
    (like TensorBoard or WandB) can handle the value correctly.

    Args:
        kind: The target summary kind, such as "Scalar", "Tensor", "Text", "Image", or "Audio".
        value: The wrapped or annotated summary object (e.g., ImageSummary, AudioSummary),
            or the raw array itself.
        raw_value: The underlying unwrapped value, typically a NumPy array or scalar.

    Returns:
        True if the value matches the expected shape/type for the summary kind.

    Raises:
        ValueError: If the provided `kind` is unrecognized.
    """
    if kind == "Text":
        return isinstance(raw_value, str)
    elif kind == "Scalar":
        return isinstance(raw_value, numbers.Number) or (
            isinstance(raw_value, np.ndarray) and raw_value.ndim == 0
        )
    elif kind == "Tensor":
        return isinstance(raw_value, np.ndarray) and raw_value.ndim != 4
    elif kind == "Image":
        return isinstance(value, ImageSummary) or (
            isinstance(raw_value, np.ndarray) and raw_value.ndim == 4
        )
    elif kind == "Audio":
        return isinstance(value, AudioSummary)
    else:
        raise ValueError(f"Invalid summary kind: {kind}")


class SummaryWriter(BaseWriter):
    """Tensorflow summary writer."""

    @config_class
    class Config(BaseWriter.Config):
        """Configures SummaryWriter.

        See also: https://www.tensorflow.org/api_docs/python/tf/summary/create_file_writer

        Attributes:
            write_every_n_steps: Writes summary every N steps.
            write_every_n_steps_map: Optional per-summary-type interval override. Keys are
                members of `SummaryType`, and values are integers indicating how frequently
                that type of summary should be logged (e.g., log images every 1000 steps).
                If a type is not listed, `write_every_n_steps` is used as fallback. Each value must
                be a positive integer multiple of `write_every_n_steps`.
            max_queue: Configures maximum number of summaries before flush.
                If None, uses the `tf_summary` default (10).
            flush_ms: Largest interval between flushes in milliseconds.
                If None, uses the `tf_summary` default (120,000, i.e. 2 minutes).
        """

        write_every_n_steps: int = 1
        write_every_n_steps_map: Optional[dict[SummaryKind, int]] = None
        max_queue: Optional[int] = None
        flush_ms: Optional[float] = None

    def __init__(self, cfg: BaseWriter.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg: SummaryWriter.Config = self.config
        self.summary_writer: tf_summary.SummaryWriter = (
            tf_summary.create_file_writer(
                cfg.dir, max_queue=cfg.max_queue, flush_millis=cfg.flush_ms
            )
            if jax.process_index() == 0
            else tf_summary.create_noop_writer()
        )

    @contextlib.contextmanager
    def as_default(self):
        with self.summary_writer.as_default() as writer:
            yield writer

    def log_config(self, config: ConfigBase, step: int = 0):
        with self.as_default():
            config_lines = config.debug_string().split("\n")
            tf_summary.text("trainer_config", config_lines, step=step)
            for line in config_lines:
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    tf_summary.text(f"trainer_config/{parts[0]}", parts[1], step=step)

    def _time_to_write(self, step: int, kind: SummaryKind) -> bool:
        cfg = self.config
        if cfg.write_every_n_steps_map is None:
            return True
        else:
            n_steps = cfg.write_every_n_steps_map.get(kind, cfg.write_every_n_steps)
            return step % n_steps == 0

    def __call__(self, step: int, values: dict[str, Any]):
        cfg = self.config
        if step % cfg.write_every_n_steps != 0:
            return

        with self.summary_writer.as_default(step=step):

            def write(path: str, value: jax.Array):
                if isinstance(value, Summary):
                    raw_value = value.value()
                else:
                    raw_value = value

                self.vlog(3, "SummaryWriter %s: %s=%s", self.path(), path, raw_value)

                if isinstance(raw_value, Tensor) and not raw_value.is_fully_replicated:
                    logging.warning(
                        "SummaryWriter: %s: %s is not fully replicated", path, raw_value
                    )
                    return

                if isinstance(raw_value, jax.Array):
                    raw_value = np.asarray(raw_value)

                if _match_summary_type("Image", value=value, raw_value=raw_value):
                    if self._time_to_write(step, "Image"):
                        tf_summary.image(path, raw_value, step=step, max_outputs=32)
                    return

                if _match_summary_type("Audio", value=value, raw_value=raw_value):
                    if self._time_to_write(step, "Audio"):
                        tf_summary.audio(
                            path,
                            raw_value[None],
                            value.sample_rate,
                            step=step,
                            max_outputs=1,
                            encoding="wav",
                        )
                    return

                if _match_summary_type("Text", value=value, raw_value=raw_value):
                    if self._time_to_write(step, "Text"):
                        tf_summary.text(path, raw_value, step=step)
                    return

                if _match_summary_type("Scalar", value=value, raw_value=raw_value):
                    if self._time_to_write(step, "Scalar"):
                        tf_summary.scalar(path, raw_value, step=step)
                    return

                # Note: The tensor check must come after the audio check, since audio is a tensor.
                if _match_summary_type("Tensor", value=value, raw_value=raw_value):
                    if self._time_to_write(step, "Tensor"):
                        tf_summary.histogram(path, raw_value, step=step)
                    return

                logging.warning(
                    "SummaryWriter: Does not know how to " 'log "%s" (%s).',
                    path,
                    raw_value.__class__,
                )

            def is_leaf(x):
                return isinstance(x, Summary)

            paths = tree_paths(values, separator="/", is_leaf=is_leaf)
            jax.tree.map(write, paths, values, is_leaf=is_leaf)
            self.summary_writer.flush()


class WandBWriter(BaseWriter):
    """Utility for logging with Weights and Biases.

    Note:
        This utility does not support restarts gracefully.
        If the job is preempted, the logger will create a new run.

    TODO(adesai22): Add support for restarts.
    """

    _FLAT_CONFIG_KEY = "flat_config"

    @config_class
    class Config(BaseWriter.Config):
        """Configures WandBWriter."""

        write_every_n_steps: int = 1  # Writes summary every N steps.
        # Optional per-summary-type interval override. Keys are members of `SummaryType`,
        # and values are integers indicating how frequently that type of summary should be logged
        # (e.g., log images every 1000 steps).
        # If a type is not listed, `write_every_n_steps` is used as fallback. Each value must be
        # a positive integer multiple of `write_every_n_steps`.
        write_every_n_steps_map: Optional[dict[SummaryKind, int]] = None
        prefix: Optional[str] = None  # A prefix to prepend to all metric keys.

        # Weights and Biases init arguments.
        # These only need to be specified once per experiment.
        # Given the current structure of summary writers being different for
        # trainers and evaluators, we keep these as optional arguments that will
        # be set the first time the WandBLogger is instantiated.
        # (Recommended) Set these fields via environment variables.

        # The wandb experiment name. Defaults to random name generated by wandb.
        exp_name: Optional[str] = None
        # The wandb project. Defaults to wandb "Uncategorized" project.
        project: Optional[str] = None
        # The wandb entity. Defaults to personal wandb account.
        entity: Optional[str] = None
        # The group to put the experiment in.
        group: Optional[str] = None
        # The tags to categorize the experiment.
        tags: Optional[Sequence[str]] = None
        # A brief description of the experiment.
        notes: Optional[str] = None
        # One of 'online', 'offline', 'disabled'.
        # 'online': The metrics will be uploaded to W&B. Requires internet.
        # 'offline': The metrics will be written locally only. Does not require internet.
        # 'disabled': Disable wandb logging. Useful for debugging purposes.
        mode: str = "online"
        # Resume determines whether W&B will try automatically resuming the same run if
        # init is called on the same machine as a previous run.
        # Options: "allow", "must", "never", "auto" or None.
        resume: str = "auto"

        # If True, convert any 2D Tensors to wandb.Image before passing to wandb.log.
        convert_2d_to_image: bool = False

    @classmethod
    def default_config(cls: Config) -> Config:
        cfg = super().default_config()
        cfg.exp_name = os.environ.get("WANDB_NAME")
        cfg.project = os.environ.get("WANDB_PROJECT")
        cfg.entity = os.environ.get("WANDB_ENTITY")
        cfg.group = os.environ.get("WANDB_GROUP")
        cfg.notes = os.environ.get("WANDB_NOTES", "")
        tags = os.environ.get("WANDB_TAGS")
        cfg.tags = tags.split(",") if tags else None
        cfg.dir = os.environ.get("WANDB_DIR")
        return cfg

    def __init__(self, cfg: SummaryWriter.Config, *, parent: Optional[Module]):
        if wandb is None:
            raise ModuleNotFoundError(
                "To use the Weights & Biases logger, please install wandb "
                "with `pip install wandb`."
            )
        super().__init__(cfg, parent=parent)

        if wandb.run is None:
            self._initialize_run()

    @processor_zero_only
    def _initialize_run(self):
        cfg = self.config

        wandb_file = os.path.join(cfg.dir, "wandb_id")
        if cfg.resume == "never":
            exp_id = None
        elif fs.exists(wandb_file):  # pytype: disable=module-attr
            with fs.open(wandb_file, "r") as f:  # pytype: disable=module-attr
                exp_id = f.read().strip()
        elif os.getenv("WANDB_RUN_ID", None):
            exp_id = os.environ["WANDB_RUN_ID"]
        else:
            exp_id = wandb.util.generate_id()
            fs.makedirs(cfg.dir)  # pytype: disable=module-attr
            with fs.open(wandb_file, "w") as f:  # pytype: disable=module-attr
                f.write(exp_id)

        wandb.init(
            id=exp_id,
            name=cfg.exp_name,
            tags=cfg.tags if cfg.tags else None,
            project=cfg.project,
            entity=cfg.entity,
            notes=cfg.notes,
            mode=cfg.mode,
            sync_tensorboard=False,  # do not synchronize with tensorboard.
            resume=cfg.resume,
            dir=cfg.dir,
            group=cfg.group,
        )

    @staticmethod
    def format_config(val) -> Union[Nested[str], list[Nested[str]]]:
        """Helper function to format config for wandb logging."""
        if isinstance(val, RequiredFieldValue):
            return "REQUIRED"
        elif isinstance(val, enum.Enum):
            return str(val)
        elif isinstance(val, dict):
            return type(val)({str(k): WandBWriter.format_config(v) for k, v in val.items()})
        elif isinstance(val, (tuple, list)):
            # wandb config stores tuple as list so no type(val)(...)
            return [WandBWriter.format_config(v) for v in val]
        elif isinstance(val, (type, FunctionType)):
            # wandb config stores type as fully qualified str (same as Configurable.debug_string())
            return f"{val.__module__}.{val.__name__}"
        else:
            return val

    @processor_zero_only
    def log_config(self, config: ConfigBase, step: int = 0):
        assert wandb.run is not None, "A wandb run must be initialized."
        wandb.config.update(WandBWriter.format_config(config.to_dict()), allow_val_change=True)
        wandb.config.update(
            # .to_flat_dict(omit_default_values={},...) matches behavior of .to_dict()
            {
                self._FLAT_CONFIG_KEY: WandBWriter.format_config(
                    config.to_flat_dict(omit_default_values={})
                )
            },
            allow_val_change=True,
        )

    def _time_to_write(self, step: int, kind: SummaryKind) -> bool:
        cfg = self.config
        if cfg.write_every_n_steps_map is None:
            return True
        else:
            n_steps = cfg.write_every_n_steps_map.get(kind, cfg.write_every_n_steps)
            return step % n_steps == 0

    @processor_zero_only
    def __call__(self, step: int, values: dict[str, Any]) -> None:
        """Convert nested summary values to wandb acceptable format and upload run data."""
        cfg = self.config
        if step % cfg.write_every_n_steps != 0:
            return

        def convert(path: str, value: Any):
            if isinstance(value, Summary):
                raw_value = value.value()
            else:
                raw_value = value

            self.vlog(3, "WandbWriter %s: %s=%s", self.path(), path, raw_value)

            # Ensure all arrays are cast to numpy.
            # Wandb will crash if jax.Array is present.
            if isinstance(raw_value, jax.Array):
                raw_value = np.asarray(raw_value)

            if _match_summary_type("Image", value=value, raw_value=raw_value):
                if self._time_to_write(step, "Image"):
                    return [wandb.Image(el) for el in raw_value]
                return

            if _match_summary_type("Audio", value=value, raw_value=raw_value):
                if self._time_to_write(step, "Audio"):
                    # W&B calls soundfile.write and saves a wav file with int16 dtype.
                    sample_rate = value.sample_rate
                    assert raw_value.ndim == 2, raw_value.shape
                    assert np.issubdtype(raw_value.dtype, np.floating), raw_value.dtype
                    raw_value = (raw_value * 32768).clip(-32768, 32767).astype(np.int16)
                    return wandb.Audio(raw_value, sample_rate=sample_rate)
                return

            if _match_summary_type("Text", value=value, raw_value=raw_value):
                if self._time_to_write(step, "Text"):
                    return raw_value
                return

            if _match_summary_type("Scalar", value=value, raw_value=raw_value):
                if self._time_to_write(step, "Scalar"):
                    return raw_value
                return

            # Note: The tensor check must come after the audio check, since audio is a tensor.
            if _match_summary_type("Tensor", value=value, raw_value=raw_value):
                if self._time_to_write(step, "Tensor"):
                    return raw_value
                return

            logging.warning(
                "WandBWriter: Does not know how to " 'log "%s" (%s).', path, raw_value.__class__
            )

        def is_leaf(x):
            return isinstance(x, Summary)

        paths = tree_paths(values, separator="/", is_leaf=is_leaf)
        values = jax.tree.map(convert, paths, values, is_leaf=is_leaf)

        # Flatten nested dicts and join the keys with "/"
        flat_paths_and_values, _ = jax.tree_util.tree_flatten_with_path(values)
        values = {
            jax.tree_util.keystr(key_path, separator="/", simple=True): value
            for key_path, value in flat_paths_and_values
        }

        if cfg.prefix:
            values = {f"{cfg.prefix}/{k}": v for k, v in values.items()}

        # Wandb doesn't recognize dot-delimited structures, but does recognize `/`
        # and will create the proper nesting if we replace `.` with `/`.
        values = {k.replace(".", "/"): v for k, v in values.items() if v is not None}

        wandb.log(values, step=step)
