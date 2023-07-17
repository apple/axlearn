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
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module
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

    # We adapt the args and kwargs from base Module to arguments specific to summary writer,
    # and drop the method argument since the caller does not decide which method to call.
    # pylint: disable=arguments-differ
    def __call__(self, step: int, values: Dict[str, Any]):
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

    def __call__(self, step: int, values: Dict[str, Any]):
        writer: BaseWriter
        for writer in self._writers:
            writer(step, values)


class NoOpWriter(BaseWriter):
    """A writer that does nothing. Used by testing."""

    def log_config(self, config: ConfigBase, step: int = 0):
        pass

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
            tf_summary.text("trainer_config", config.debug_string().split("\n"), step=step)

    def __call__(self, step: int, values: Dict[str, Any]):
        cfg = self.config
        if step % cfg.write_every_n_steps != 0:
            return
        with self.summary_writer.as_default(step=step):
            values = jax.tree_util.tree_map(
                lambda v: v.mean if isinstance(v, WeightedScalar) else v,
                values,
                is_leaf=lambda x: isinstance(x, WeightedScalar),
            )

            def write(path: str, value: jnp.ndarray):
                self.vlog(3, "SummaryWriter %s: %s=%s", self.path(), path, value)
                if isinstance(value, Tensor) and not value.is_fully_replicated:
                    logging.warning("SummaryWriter: %s: %s is not fully replicated", path, value)
                elif isinstance(value, str):
                    tf_summary.text(path, value, step=step)
                elif isinstance(value, numbers.Number) or value.ndim == 0:
                    tf_summary.scalar(path, value, step=step)
                elif isinstance(value, np.ndarray) and value.ndim == 4:
                    tf_summary.image(path, value, step=step, max_outputs=25)
                else:
                    tf_summary.histogram(path, value, step=step)

            jax.tree_util.tree_map(write, tree_paths(values, separator="/"), values)
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

    @processor_zero_only
    def __call__(self, step: int, values: Dict[str, Any]):
        cfg = self.config
        if step % cfg.write_every_n_steps != 0:
            return

        values = jax.tree_util.tree_map(
            lambda v: v.mean if isinstance(v, WeightedScalar) else v,
            values,
            is_leaf=lambda x: isinstance(x, WeightedScalar),
        )
        # Ensure all arrays are cast to numpy.
        # Wandb will crash if jax.Array is present.
        values = jax.tree_util.tree_map(
            lambda v: np.asarray(v) if isinstance(v, jax.Array) else v,
            values,
        )
        if cfg.prefix:
            values = {f"{cfg.prefix}/{k}": v for k, v in values.items()}

        # wandb doesn't recognize dot-delimited structures, but does recognize `/`
        # and will create the proper nesting if we replace `.` with `/`.
        values = {k.replace(".", "/"): v for k, v in values.items()}
        if cfg.convert_2d_to_image:
            # TODO(bmckinzie): support kwargs in add_summary (e.g. `cast_to=wandb.Image`) as a
            # more general solution for things like this.
            values = jax.tree_util.tree_map(
                lambda v: wandb.Image(v) if (isinstance(v, np.ndarray) and v.ndim == 2) else v,
                values,
            )
        wandb.log(values, step=step)
