# Copyright © 2023 Apple Inc.

"""Summary objects that can be logged."""

from typing import Any, Callable, Optional, Union

import jax
import numpy as np

from axlearn.common import flax_struct
from axlearn.common.utils import NestedTensor, Tensor


class Summary(flax_struct.PyTreeNode):
    """Base class for a summary value.

    Subclasses should implement value() and, optionally, validate().
    """

    def value(self) -> Optional[Union[NestedTensor, int, float]]:
        """Returns a value for logging."""
        raise NotImplementedError(type(self))

    def validate(self):
        """Validates that the summary was constructed with valid data.

        Raises:
            Exception: If the summary is invalid.
        """

    def accumulate(self, other: "Summary") -> "Summary":
        """The default way this summary should be accumulated.

        Args:
            other: The summary from later in training/eval to accumulate into this summary.

        Returns:
            A single accumulated summary.
        """
        raise NotImplementedError(type(self))


class ImageSummary(Summary):
    """A summary that should be logged as a batch of images.

    The shape should either be (batch, height, width, channels)  or (batch, height, width).

    Supported channels values are 1 for grayscale, 3 for RGB, and 4 for RGBA.

    The image returned by `value()` will be padded to `ndim==4` if this was instantiated with a
    tensor with `ndim==3`.
    """

    _value: Tensor

    def validate(self):
        val = self._value
        if val.ndim not in (3, 4):
            raise ValueError(
                f"ImageSummary value has invalid shape:\n"
                f"expected val.ndim in (3, 4), got {val.ndim}"
            )
        if val.ndim == 4 and val.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"ImageSummary value has invalid shape:\n"
                f"expected channels (val.shape[-1]) in (1, 3, 4), got {val.shape[-1]}"
            )

    def value(self) -> Tensor:
        # Add dimension representing 1 grayscale channel if the image is grayscale.
        val = self._value
        if val.ndim == 3:
            val = val[..., None]
        return val

    def accumulate(self, other: Summary) -> Summary:
        return self


class AudioSummary(Summary):
    """Audio summary.

    Attributes:
        _value: A Tensor representing audio data with shape [t,] or [t,c], t is the number
            of frames, and c is the number of channels. Elements should be floating-point
            values in [-1.0,1.0].
        sample_rate: An int or rank-0 int32 Tensor that represents the sample rate, in Hz.
            Must be positive. When not passed, default value 16kHz will be used.

    Example:
        ```
        # A 10 second 16kHz audio
        audio_16k = jax.numpy.ones((16000*10,))
        add_summary("audio_16k", AudioSummary(audio_16k))
        # A 10 second 24kHz 2 channels audio
        audio_24k = jax.numpy.ones((24000*10,2))
        add_summary("audio_24k", AudioSummary(audio_24k, sample_rate=24000))
        ```
    """

    _value: Tensor
    sample_rate: int = 16000

    def validate(self):
        val = self._value
        if val.ndim not in (1, 2):
            raise ValueError(
                f"Audio value has invalid shape: expected val.ndim in (1, 2), got {val.ndim}"
            )

    def value(self) -> Tensor:
        """Returns the audio tensor in shape [t, c], with floating-point values in [-1.0, 1.0]"""
        val = self._value
        # tf.summary.audio takes a tensor representing audio data with shape [k, t, c],
        # where k is the number of audio clips, t is the number of frames, and c is
        # the number of channels. Multiple audio clips are not supported, we set
        # max_outputs=1 in tf_summary.audio
        if val.ndim == 1:
            # Add the audio clips and channels dimension.
            return val[:, None]
        return val

    def accumulate(self, other: Summary) -> Summary:
        return self


class CallbackSummary(Summary):
    # pylint: disable=not-callable,super-init-not-called
    """A summary defined using a callback that is only called outside of JIT. The arguments
    to the callback are treated as pytrees whose leaves are converted to numpy arrays before
    calling the callback.

    Example:
        ```
        # This logs a 7 row table with two columns where each cell contains a 16 x 16 color image
        # Shape: num examples x table columns x image height x image width x channels
        images = jax.numpy.ones((7, 2, 16, 16, 3))

        def create_table(images: np.ndarray):
            return wandb.Table(
                ["output", "target"], [[wandb.Image(img) for img in row] for row in images]
            )

        add_summary("my_summary", CallbackSummary(create_table, images))
        ```
    """

    fn: Callable = flax_struct.field(pytree_node=False)
    args: tuple
    kwargs: dict[str, Any]

    def __init__(self, fn: Callable, *args, **kwargs):
        """Initializes the class.

        This sets self.fn=fn, self.args=args, self.kwargs=kwargs.
        If kwargs with the names 'args' or 'kwargs' are present, the values of self.args or
        self.kwargs will be set using those values instead.

        Args:
            fn: The function to call with the given arguments that should return an object that
                is compatible with WandB's logger.
            *args: The positional arguments to pass to fn. JAX arrays will be converted to Numpy.
            **kwargs: The keyword arguments to pass to fn. JAX arrays will be converted to Numpy.
        """
        # The tree flattening and unflattening methods generated by the PyTreeNode superclass
        # expect that the class can be constructed using its members as keyword arguments.
        # Therefore, we need to support constructing this class using "args" and "kwargs"
        # as keyword arguments.
        # __setattr__ is the only way to initialize a frozen dataclass's fields.
        super().__setattr__("fn", fn)
        if "args" in kwargs:
            super().__setattr__("args", kwargs["args"])
        else:
            super().__setattr__("args", args)
        if "kwargs" in kwargs:
            super().__setattr__("kwargs", kwargs["kwargs"])
        else:
            super().__setattr__("kwargs", kwargs)

    def value(self) -> Tensor:
        args = tuple(np.asarray(x) for x in self.args)
        kwargs = jax.tree.map(lambda x: np.asarray(x) if isinstance(x, Tensor) else x, self.kwargs)
        return self.fn(*args, **kwargs)

    def accumulate(self, other: Summary) -> Summary:
        return self
