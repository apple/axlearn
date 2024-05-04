# Copyright © 2023 Apple Inc.

"""Utilities for writing summaries."""
import contextlib
import numbers
import os
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import numpy as np
import tensorflow as tf
from absl import logging
from jax import numpy as jnp
from tensorflow import summary as tf_summary

from axlearn.common.config import REQUIRED, ConfigBase, Required, RequiredFieldValue, config_class
from axlearn.common.module import Module
from axlearn.common.summary import ImageSummary, Summary
from axlearn.common.utils import Tensor, tree_paths

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

Tensor = jnp.ndarray


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

    def writing_at_step(self, step: int) -> bool:
        """Returns whether to write summaries at `step`."""
        raise NotImplementedError(type(self))

    # We adapt the args and kwargs from base Module to arguments specific to summary writer,
    # and drop the method argument since the caller does not decide which method to call.
    # pylint: disable=arguments-differ
    def __call__(self, step: int, values: Dict[str, Any]):
        """Log data to disk.

        Args:
            step: The step.
            values: The values to log.
        """
        raise NotImplementedError(type(self))


class CompositeWriter(BaseWriter):
    """A collection of named writers.

    This class automatically sets the `dir` field of any child writers.
    """

    @config_class
    class Config(BaseWriter.Config):
        writers: Required[Dict[str, BaseWriter.Config]] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional["Module"]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._writers = []
        for writer_name, writer_cfg in cfg.writers.items():
            self._writers.append(
                self._add_child(writer_name, writer_cfg.set(dir=os.path.join(cfg.dir, writer_name)))
            )

    @property
    def writers(self) -> List[BaseWriter]:
        """A list of writers."""
        return self._writers

    def log_config(self, config: ConfigBase, step: int = 0):
        writer: BaseWriter
        for writer in self._writers:
            writer.log_config(config, step=step)

    def writing_at_step(self, step: int) -> bool:
        """Returns whether to write summaries at `step`."""
        writing = False
        for writer in self._writers:
            writing = writing or writer.writing_at_step(step)
        return writing

    def __call__(self, step: int, values: Dict[str, Any]):
        writer: BaseWriter
        for writer in self._writers:
            writer(step, values)


class NoOpWriter(BaseWriter):
    """A writer that does nothing. Used by testing."""

    def log_config(self, config: ConfigBase, step: int = 0):
        pass

    def writing_at_step(self, step: int) -> bool:
        """Returns whether to write summaries at `step`."""
        return False

    def __call__(self, step: int, values: Dict[str, Any]):
        pass


class SummaryWriter(BaseWriter):
    """Tensorflow summary writer."""

    @config_class
    class Config(BaseWriter.Config):
        """Configures SummaryWriter."""

        write_every_n_steps: int = 1  # Writes summary every N steps.

    def __init__(self, cfg: BaseWriter.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self.summary_writer: tf_summary.SummaryWriter = (
            tf_summary.create_file_writer(cfg.dir)
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

    def writing_at_step(self, step: int) -> bool:
        return step % self.config.write_every_n_steps == 0

    def __call__(self, step: int, values: Dict[str, Any]):
        if not self.writing_at_step(step):
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
                elif isinstance(value, ImageSummary):
                    tf_summary.image(path, raw_value, step=step)
                elif isinstance(raw_value, str):
                    tf_summary.text(path, raw_value, step=step)
                elif isinstance(raw_value, numbers.Number) or raw_value.ndim == 0:
                    tf_summary.scalar(path, raw_value, step=step)
                elif isinstance(raw_value, np.ndarray) and raw_value.ndim == 4:
                    tf_summary.image(path, raw_value, step=step, max_outputs=25)
                elif isinstance(raw_value, jax.Array):
                    tf_summary.histogram(path, raw_value, step=step)
                else:
                    logging.warning(
                        "SummaryWriter: Does not know how to " 'log "%s" (%s).',
                        path,
                        raw_value.__class__,
                    )

            def is_leaf(x):
                return isinstance(x, Summary)

            paths = tree_paths(values, separator="/", is_leaf=is_leaf)
            jax.tree_util.tree_map(write, paths, values, is_leaf=is_leaf)
            self.summary_writer.flush()


class WandBWriter(BaseWriter):
    """Utility for logging with Weights and Biases.

    Note:
        This utility does not support restarts gracefully.
        If the job is pre-empted, the logger will create a new run.

    TODO(adesai22): Add support for restarts.
    """

    @config_class
    class Config(BaseWriter.Config):
        """Configures WandBWriter."""

        write_every_n_steps: int = 1  # Writes summary every N steps.
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
        elif tf.io.gfile.exists(wandb_file):  # pytype: disable=module-attr
            with tf.io.gfile.GFile(wandb_file, "r") as f:  # pytype: disable=module-attr
                exp_id = f.read().strip()
        else:
            exp_id = wandb.util.generate_id()
            tf.io.gfile.makedirs(cfg.dir)  # pytype: disable=module-attr
            with tf.io.gfile.GFile(wandb_file, "w") as f:  # pytype: disable=module-attr
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

    @processor_zero_only
    def log_config(self, config: ConfigBase, step: int = 0):
        def fmt(val):
            if isinstance(val, RequiredFieldValue):
                return "REQUIRED"
            elif isinstance(val, dict):
                return type(val)({k: fmt(v) for k, v in val.items()})
            elif isinstance(val, (tuple, list)):
                return type(val)([fmt(v) for v in val])
            else:
                return val

        assert wandb.run is not None, "A wandb run must be initialized."
        wandb.config.update(fmt(config.to_dict()), allow_val_change=True)

    def writing_at_step(self, step: int) -> bool:
        return step % self.config.write_every_n_steps == 0

    @processor_zero_only
    def __call__(self, step: int, values: Dict[str, Any]):
        cfg = self.config
        if not self.writing_at_step(step):
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

            if isinstance(value, ImageSummary):
                return [wandb.Image(el) for el in raw_value]
            return raw_value

        def is_leaf(x):
            return isinstance(x, Summary)

        paths = tree_paths(values, separator="/", is_leaf=is_leaf)
        values = jax.tree_util.tree_map(convert, paths, values, is_leaf=is_leaf)

        if cfg.prefix:
            values = {f"{cfg.prefix}/{k}": v for k, v in values.items()}

        # Wandb doesn't recognize dot-delimited structures, but does recognize `/`
        # and will create the proper nesting if we replace `.` with `/`.
        values = {k.replace(".", "/"): v for k, v in values.items()}

        wandb.log(values, step=step)
