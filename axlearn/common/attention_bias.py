# Copyright © 2024 Apple Inc.

"""Data structures for working with different kinds of attention logit biases.

Some downstream optimizations, e.g., flash attention, rely on specific subclasses of
BaseAttentionBias being used. E.g., CausalAttentionBias should be used to specify the causal mask.
The optimizations will not happen if a BoolTensorAttentionBias is used to specify the causal mask
instead.
These "special" bias classes also include SegmentIdAttentionBias and MaskFnAttentionBias.

Note that the various `BaseAttentionBias` classes are not intended to be instantiated at
configuration time. Instead, they are geared towards people developing layers who can then return
an instance of `BaseAttentionBias` where they would have before returned an explicit attention
bias Tensor. Currently, we don't have support for AttentionLogitBiasLayer returning these objects
instead of explicit bias Tensors, but such support can be added in the future if needed in a
fully backwards-compatible manner.
"""

import dataclasses
import functools
import typing
from typing import (
    Callable,
    Generic,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    final,
)

import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common import flax_struct
from axlearn.common.config import ClassConfigBase, ConfigOr, config_for_class, maybe_instantiate
from axlearn.common.utils import Tensor, safe_not

NEG_INF = -1e15

# We use OpT when we have a function like fn(x: OpT) -> OpT where we want to annotate
# that the functions return value is None/not None according to whether the input is None/not None.
# This makes e.g. IDE inspections able to understand that fn(jnp.ones(5)) is in fact a Tensor and
# not None.
OpT = typing.TypeVar("OpT", type(None), Tensor)
B = TypeVar("B", bound="BaseAttentionBias")


@functools.partial(flax_struct.dataclass, eq=False)
class BaseAttentionBias:
    """Base class representing attention logit biases."""

    # The dtype of the biases to return in `value()`.
    # If None, do not cast the dtype.
    dtype: Optional[jnp.dtype] = flax_struct.field(kw_only=True, default=None, pytree_node=False)

    @final
    def eval_shape(self) -> tuple[int, int, int, int]:
        """Return the shape of the bias tensor.

        Note: this doesn't materialize the value. jax.eval_shape calls value(), but it only does so
        using tracers.

        Returns
            shape: [batch or 1, num_heads or 1, target_len, source_len].

        Raises:
            ValueError: If the bias has no value.
        """
        if not self.has_value():
            raise ValueError("AttentionBias has no value.")
        return jax.eval_shape(type(self).value, self).shape

    @final
    def has_value(self) -> bool:
        """Return whether to the bias has a value."""
        return jax.eval_shape(type(self).value, self) is not None

    @final
    def value(self) -> Optional[Tensor]:
        """Return a tensor with the biases or None if there are no biases.

        Shape: [batch or 1, num_heads or 1, target_len, source_len].

        The dtype will be cast to `self.dtype` if it is not None.
        """
        value = self._value()
        if self.dtype is not None and value is not None:
            value = value.astype(self.dtype)
        return self._broadcast_value(value)

    def _value(self) -> Optional[Tensor]:
        """Internal version of `value()` without the casting and broadcasting done in the public
        method.

        Subclasses must implement this.

        Shape: Any of:
        * [target_len, source_len]
        * [batch or 1, target_len, source_len]
        * [batch or 1, num_heads or 1, target_len, source_len].
        """
        raise NotImplementedError

    def __add__(self, other: "BaseAttentionBias") -> "CompositeAttentionBias":
        """Returns a bias tensor representing the sum of `self` and `other`.

        The implementation lazily adds them by creating a CompositeAttentionBias
        from the biases being added.
        """
        return CompositeAttentionBias([self, other])

    def astype(self, dtype: jnp.dtype) -> "BaseAttentionBias":
        """Return a new bias whose dtype is `dtype`."""
        result = dataclasses.replace(self, dtype=dtype)
        result = cast(BaseAttentionBias, result)
        return result

    @classmethod
    def _broadcast_value(cls, value: OpT) -> OpT:
        """Broadcasts `value` to a canonical 4 dimensional attention bias shape.

        Raises:
            ValueError: If the shape of `value` is not 2, 3, or 4 dimensional.
        """
        if value is None or value.ndim == 4:
            return value
        if value.ndim == 2:
            # Shape: [1, 1, target_length, source_length].
            return value[None, None, :, :]
        elif value.ndim == 3:
            # Shape: [batch, 1, target_length, source_length].
            return value[:, None, :, :]
        raise ValueError(f"Invalid attention_logit_biases shape: {value.shape}.")

    def partition_spec(
        self, mha_dim_to_partition_spec: dict[str, PartitionSpec]
    ) -> Union["BaseAttentionBias", PartitionSpec]:
        """Compute a partition spec for this bias."""
        raise NotImplementedError

    def bias_and_residual(self, cls: Type[B]) -> "BiasAndResidual[B]":
        """Split this bias into a bias of type `cls` and a residual.

        If the two returned biases are added together, the result is equivalent to
        the value of this bias.

        The default implementation returns `self` as either the bias or residual.
        Which field is set to `self` is based on whether this is an instance of `cls`.
        If it is an instance, `self` is returned in the bias field and the `residual` field will be
        a `BaseAttentionBias` with value() None.
        If not, it is returned in the `residual` field, and the `bias` field is set to `None`.
        """
        if isinstance(self, cls):
            return BiasAndResidual(bias=self, residual=CompositeAttentionBias([]))
        return BiasAndResidual(bias=None, residual=self)

    @classmethod
    def from_sequence(cls, biases: Sequence["BaseAttentionBias"]) -> Optional["BaseAttentionBias"]:
        """Constructs a single combined attention bias of the same type as this class
        from a sequence of such biases.

        If the sequence is empty, returns None.

        The default implementation returns the bias if the sequence has length one and
        raises for length > 1.

        Raises:
            NotImplementedError: If the sequence has length > 1.
            TypeError: If `seq` contains a type of bias that is not an instance of this class.
            ValueError: If `eval_shape()` is not the same for every bias.
        """

        if not biases:
            return None

        shape = biases[0].eval_shape()
        for bias in biases:
            if not isinstance(bias, cls):
                raise TypeError(f"Got bias type {type(bias)}, not instance of {cls}.")
            if bias.eval_shape() != shape:
                raise ValueError(f"Got shape mismatch {bias.eval_shape()} != {shape}.")

        if len(biases) == 1:
            return biases[0]
        raise NotImplementedError


@flax_struct.dataclass
class BiasAndResidual(BaseAttentionBias, Generic[B]):
    """A bias and residual where the bias has type `B` (or is None) and the residual
    has any type.

    Used to represent an original bias that has been split into the sum of two biases.

    See `BaseAttentionBias.bias_and_residual()`.
    """

    bias: Optional[B]
    residual: BaseAttentionBias

    def _value(self) -> Optional[Tensor]:
        biases = [self.bias] if self.bias is not None else []
        biases.append(self.residual)
        return CompositeAttentionBias(biases).value()

    def __iter__(self):
        return iter((self.bias, self.residual))


@flax_struct.dataclass
class CompositeAttentionBias(BaseAttentionBias):
    """A lazily evaluated list of biases that are added together to get the final bias.

    The implementation automatically flattens nested instances of `CompositeAttentionBias`.

    Biases that have a `value()` of None or are equal to None are automatically omitted
    when iterating over this instance. However, they remain in the `biases` list
    and are therefore still part of the pytree structure.
    """

    # The biases to add to obtain the final bias.
    biases: Sequence[BaseAttentionBias]

    def _value(self) -> Optional[Tensor]:
        """Returns the sum of the biases.

        If all biases have value None, this is guaranteed to also return None.

        Shape: [batch or 1, num_heads or 1, target_len, source_len].

        Raises:
            ValueError: If one of the biases in the sum has the wrong shape.
        """
        biases = self._nonzero()
        if not biases:
            return None

        result = 0.0
        for bias in biases:
            result += bias.value()
        return result

    def __add__(self, other: BaseAttentionBias) -> "CompositeAttentionBias":
        return self.__class__([self, other])

    def _nonzero(self) -> Sequence[BaseAttentionBias]:
        """Returns an sequence of biases in this collection except those detected as zero.

        Returned biases are not guaranteed to be nonzero, but are guaranteed to not return None.
        """
        filt = lambda b: b.has_value()
        return list(filter(filt, self.biases))

    def bias_and_residual(self, cls: Type[B]) -> "BiasAndResidual[B]":
        """Split this bias into a bias of type `cls` and a residual.

        Compared to the default implementation, this determines which instance of `cls` to return
        by calling `bias_and_residual()` on each member of this collection. It also recursively
        calls `bias_and_residual` on any residuals obtained from such BiasAndResidual objects.

        All non-None biases returned from doing this are then merged using `cls.from_sequence()`
        before returning.
        """
        bias_and_residual = super().bias_and_residual(cls)
        if bias_and_residual.bias is not None:
            return bias_and_residual
        remaining_biases = list(self._nonzero())
        cls_biases = []
        residuals = []
        while remaining_biases:
            bias = remaining_biases.pop()
            bias_and_residual = bias.bias_and_residual(cls)
            if bias_and_residual.bias is not None:
                cls_biases.append(bias_and_residual.bias)
                send_residual_to = remaining_biases
            else:
                send_residual_to = residuals
            if bias_and_residual.residual.has_value():
                send_residual_to.append(bias_and_residual.residual)
        return BiasAndResidual(
            bias=cls.from_sequence(cls_biases), residual=CompositeAttentionBias(residuals)
        )

    def partition_spec(
        self, mha_dim_to_partition_spec: dict[str, PartitionSpec]
    ) -> Union[BaseAttentionBias, PartitionSpec]:
        return CompositeAttentionBias(
            [
                b.partition_spec(mha_dim_to_partition_spec) if b is not None else PartitionSpec()
                for b in self.biases
            ],
            dtype=self.dtype,
        )

    def _flatten(self) -> "CompositeAttentionBias":
        """Returns a flattened version of this instance.

        Used only for testing/debugging
        """
        remaining = [self]
        biases = []
        while remaining:
            bias = remaining.pop()
            if isinstance(bias, CompositeAttentionBias):
                remaining.extend(bias._nonzero())  # pylint: disable=protected-access
            else:
                biases.append(bias)
        return CompositeAttentionBias(biases)


def split(bias: BaseAttentionBias, *cls: Type[BaseAttentionBias]) -> Iterable[BaseAttentionBias]:
    """Split `bias` into an iterable of biases of `len(cls) + 1` instances, where the ith instances
    has type cls[i] or ZeroAttentionBias.

    Each bias will only be present once in the output, with ties broken based on the first
    matching type in `cls`.

    The correctness of this function requires that `bias_and_residual(cls)`
    has the property that future calls to `bias_and_residual()` on any residuals
    obtained directly or indirectly from the original residual will never return an instance
    of `cls`.

    Raises:
        NotImplementedError: If any residual is encountered with a type in `cls`.
    """
    for c in cls:
        result, bias = bias.bias_and_residual(c)
        if isinstance(bias, cls):
            raise NotImplementedError("Got a residual of type in `cls`.")
        if result is None:
            yield ZeroAttentionBias()
        else:
            yield result
    yield bias


@flax_struct.dataclass
class TensorAttentionBias(BaseAttentionBias):
    """An attention bias represented as an explicit Tensor."""

    # The explicit value of the bias.
    # Shape: [batch, num_heads, target_len, source_len].
    _internal_value: Tensor

    def __post_init__(self):
        # Because TensorAttentionBias is a flax_struct.dataclass and the automatically generated
        # pytree flattening methods for all flax_struct.dataclasses always flatten to a list of the
        # dataclass fields. (I.e., not the result of calling value().)  Therefore, we enforce a
        # consistent shape so that the partition spec correctly lines up wit the dimensions of the
        # stored Tensor.
        if getattr(self._internal_value, "ndim", 4) != 4:
            raise ValueError(f"Invalid shape {self._internal_value.shape}.")

    def _value(self) -> Tensor:
        return self._internal_value

    def partition_spec(
        self, mha_dim_to_partition_spec: dict[str, PartitionSpec]
    ) -> Union[BaseAttentionBias, PartitionSpec]:
        shape = self.eval_shape()
        spec = mha_dim_to_partition_spec["bnts"]
        return _spec_for_explicit_bias(spec=spec, shape=shape)

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "TensorAttentionBias":
        """Constructs an instance of this class, automatically canonicalizing the shape of
        `tensor` to the required form.

        Unlike a CompositeAttentionBias, this can be used as a mask in SplashAttention.
        """
        return cls(cls._broadcast_value(tensor))


def _spec_for_explicit_bias(
    spec: PartitionSpec, shape: tuple[int, ...]
) -> Union[BaseAttentionBias, PartitionSpec]:
    """Return a PartionSpec for an explicit bias tensor of the given shape baed on `spec`."""
    # Explicit attention bias: [batch_size, num_heads, target_len, source_len].
    if spec != PartitionSpec(None):
        if shape[0] == 1:
            spec = PartitionSpec(None, *spec[1:])
        if shape[1] == 1:
            spec = PartitionSpec(spec[0], None, *spec[2:])
    return spec


@flax_struct.dataclass
class BoolAttentionBias(BaseAttentionBias):
    """An attention bias represented as a boolean mask."""

    @final
    def _value(self) -> Optional[Tensor]:
        bool_value = self.bool_value()
        if bool_value is None:
            return None
        return bool_to_bias(bool_value)

    @final
    def bool_value(self) -> Optional[Tensor]:
        """Return a tensor with the boolean values from `self.mask` before they have been converted
        to biases.

        Shape: Same as `self.value()`.
        """
        bool_value = self._bool_value()
        if bool_value is not None:
            bool_value = bool_value.astype(bool)
        return self._broadcast_value(bool_value)

    def _bool_value(self) -> Optional[Tensor]:
        """Internal version of `bool_value()` without the casting and broadcasting done in the
        public method.

        Subclasses must implement this.

        Shape: Same as `self._value()`.
        """
        raise NotImplementedError


@flax_struct.dataclass
class SegmentIdAttentionBias(BoolAttentionBias):
    """An attention bias defined by segment ids."""

    # See ``on segment ids'' in the module docstring.
    segment_ids: Tensor

    def _bool_value(self) -> Optional[Tensor]:
        return _make_bool_segment_mask(
            target_segments=self.segment_ids, source_segments=self.segment_ids
        )

    def partition_spec(
        self, mha_dim_to_partition_spec: dict[str, PartitionSpec]
    ) -> Union[BaseAttentionBias, PartitionSpec]:
        # Segment IDs: [batch_size, seq_len].
        q_spec = mha_dim_to_partition_spec["btnh"]
        if q_spec == PartitionSpec(None):
            return PartitionSpec(None)
        return PartitionSpec(q_spec[0], q_spec[1])


class MaskFn(Protocol):
    """A broadcastable function for computing a boolean logit mask."""

    def __call__(self, query_position: Tensor, key_position: Tensor) -> Tensor:
        """Returns a bool Tensor of whether the query token at `query_position` should attend
        to the key token at `key_position`.

        Implementations have the following contract:
        * Must support scalar arguments.
        * If given non-scalar arguments of the same shape, the result must be the same as
          applying the function elementwise over these arugments. I.e.,
          ```
          x = f(jnp.asarray([1,2]), jnp.asarray([3,4]))
          assert x[0] == f(jnp.asarray(1), jnp.asarray(3))[None]
          ```
        * Both tensors have the same rank (either 2 or 3), as batch dim is optional.
        * If given non-scalar arguments of different shapes, the result must be the same if we
          first broadcast the arguments against each other to make them have the same shape.
        * Beyond requiring broadcastability, must not impose any constraints on the shapes of its
          arguments.
        * Must return a non-tracer value when run in `jax.ensure_compile_time_eval()`. This will
          typically be the case as long as you don't access tensors stored in global variables.

        Args:
            query_position: The index in the sequence of query vectors.
            key_position: The index in the sequence of key vectors.

        Returns:
            Whether the query and key vectors with the given index should attend to one another.
            True means they should attend. False means they should not.
            The shape is the same as the shape obtained after broadcasting the inputs against each
            other according to numpy broadcasting semantics.
            For example, a common usage pattern is that `query_position` has shape
            `[batch, tgt_seq, 1]` and `key_position` has shape `[batch, 1, src_seq]` and the mask
            will have shape  `[batch, tgt_seq, src_seq]`.
            Reference for bradcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
        """


@flax_struct.dataclass
class MaskFnAttentionBias(BoolAttentionBias):
    """An attention bias represented as an implicit boolean mask."""

    # The function defining the contents of the mask.
    mask: MaskFn = flax_struct.field(pytree_node=False)

    # The positions in the query sequence that the mask should be computed for.
    # I.e., `self.value()[batch, num_heads, i]` is the mask specifying what the query token at
    # `target_positions[batch, i]`  may attend to.
    # This is typically used during decoding to specify the locations in the sequence being
    # being decoded.
    # E.g., if we are decoding position 5 and 7 of the first and second batch entry respectively,
    # we would set `target_positions = jnp.arange(steps)[None] + jnp.asarray([5, 7])`.
    # The motivation for supporting such shapes is for use cases where time_step in transformers
    # is not necessarily contiguous. E.g., speculative decoding, non-contiguous prompts,
    # various papers that need it.
    # The index in the sequence of query vectors, [1|batch, target_len].
    target_positions: Tensor = flax_struct.field(kw_only=True)
    # The index in the sequence of key vectors, [1|batch, source_len].
    source_positions: Tensor = flax_struct.field(kw_only=True)

    @classmethod
    def default_config(cls, mask: MaskFn) -> ClassConfigBase["MaskFnAttentionBias"]:
        return config_for_class(MaskFnAttentionBias).set(mask=mask)

    def _bool_value(self) -> Optional[Tensor]:
        """Return a tensor with the boolean values from `self.mask` before they have been converted
        to biases.

        Shape: [batch, target_len, source_len].

        Raises:
            ValueError. If `(target|source)_positions.ndim not == 2`.
        """
        target_positions, source_positions = self.target_positions, self.source_positions
        if target_positions.ndim != source_positions.ndim != 2:
            raise ValueError(
                f"{target_positions.shape=} or {source_positions.shape=} is not rank 2."
            )

        target_positions = jnp.expand_dims(target_positions, axis=2)  # [batch, target_length, 1]
        source_positions = jnp.expand_dims(source_positions, axis=1)  # [batch, 1, source_length]
        return self.mask(target_positions, source_positions)  # pylint: disable=not-callable

    @classmethod
    def from_sequence(
        cls, biases: Sequence["MaskFnAttentionBias"]
    ) -> Optional["MaskFnAttentionBias"]:
        """Constructs a single combined `MaskFnAttentionBias` from a Sequence of them.

        All biases use the same query and key, so target_positions and source_positions are same
        per bias.

        The sequence is first filtered to remove biases that are detected as all zero.

        If the sequence only has one element after doing so, that one element is returned without
        modification.

        If the sequence is empty, returns None.

        Raises:
            ValueError: If `target_positions` is set for any bias.
        """
        try:
            return super().from_sequence(biases)
        except NotImplementedError:
            pass

        # Combine masks.
        mask = lambda query_position, key_position: jnp.all(
            jnp.stack([b.mask(query_position, key_position) for b in biases]), axis=0
        )
        return MaskFnAttentionBias(
            mask=mask,
            target_positions=biases[0].target_positions,
            source_positions=biases[0].source_positions,
        )

    def partition_spec(
        self, mha_dim_to_partition_spec: dict[str, PartitionSpec]
    ) -> Union[BaseAttentionBias, PartitionSpec]:
        if mha_dim_to_partition_spec["bnts"] == PartitionSpec(None):
            batch = target = source = None
        else:
            batch, _, target, source = mha_dim_to_partition_spec["bnts"]
        return dataclasses.replace(
            self,
            target_positions=PartitionSpec(
                None if self.target_positions.shape[0] == 1 else batch, target
            ),
            source_positions=PartitionSpec(
                None if self.source_positions.shape[0] == 1 else batch, source
            ),
        )


@flax_struct.dataclass
class BoolTensorAttentionBias(BoolAttentionBias):
    """An attention bias represented as an explicit boolean mask."""

    # The explicit bool value of the bias.
    _internal_bool_value: Tensor

    def __post_init__(self):
        if getattr(self._internal_bool_value, "ndim", 4) != 4:
            raise ValueError(f"Invalid shape {self._internal_bool_value.shape}.")
        if getattr(self._internal_bool_value, "dtype", bool) != bool:
            raise ValueError(f"Invalid dtype {self._internal_bool_value.dtype}, expected bool.")

    def _bool_value(self) -> Tensor:
        """Return a tensor with the boolean values from `self.mask` before they have been converted
        to biases.
        """
        return self._internal_bool_value

    def partition_spec(
        self, mha_dim_to_partition_spec: dict[str, PartitionSpec]
    ) -> Union[BaseAttentionBias, PartitionSpec]:
        shape = self.eval_shape()
        spec = mha_dim_to_partition_spec["bnts"]
        return _spec_for_explicit_bias(spec=spec, shape=shape)

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "BoolTensorAttentionBias":
        """Constructs an instance of this class, automatically canonicalizing the shape of
        `tensor` to the required form.
        """
        return cls(cls._broadcast_value(tensor))


def as_attention_bias(bias: Union[None, Tensor, B]) -> B:
    """Converts `bias` to an instance of `BaseAttentionBias`.

    Raises:
        ValueError: If `bias` is a Tensor but does not have a floating point dtype.
        NotImplementedError: If `bias` is an unknown type.
    """
    if bias is None:
        return ZeroAttentionBias()
    if isinstance(bias, Tensor):
        if not jnp.issubdtype(bias.dtype, jnp.floating):
            raise ValueError(f"bias must have a floating dtype, got {bias.dtype}.")
        return TensorAttentionBias.from_tensor(bias)
    if isinstance(bias, BaseAttentionBias):
        return bias
    raise NotImplementedError(type(bias))


def causal_mask(query_position: Tensor, key_position: Tensor) -> Tensor:
    """Returns the given entry of a causal attention mask.

    Implements the `MaskFn` protocol.
    See that and `MultiheadAttention.Config.mask`.
    """
    return query_position >= key_position


@flax_struct.dataclass
@final
class CausalAttentionBias(MaskFnAttentionBias):  # pylint: disable=final-error
    """A causal attention mask."""

    mask: Optional[MaskFn] = flax_struct.field(pytree_node=False, default=causal_mask)

    @classmethod
    def default_config(cls) -> ClassConfigBase[MaskFnAttentionBias]:
        return config_for_class(CausalAttentionBias)

    @classmethod
    def from_sequence(
        cls, biases: Sequence["CausalAttentionBias"]
    ) -> Optional["CausalAttentionBias"]:
        try:
            return super().from_sequence(biases)
        except NotImplementedError:
            pass
        return biases[0]


@flax_struct.dataclass
@final
class SlidingWindowAttentionBias(MaskFnAttentionBias):  # pylint: disable=final-error
    """A sliding window attention mask."""

    # A left context size for sliding window attention. Total window size = sliding_window_size + 1
    sliding_window_size: int = flax_struct.field(kw_only=True, pytree_node=False)

    @classmethod
    # pylint: disable-next=arguments-renamed
    def default_config(
        cls, sliding_window_size: int
    ) -> ClassConfigBase["SlidingWindowAttentionBias"]:
        return config_for_class(SlidingWindowAttentionBias).set(
            mask=sliding_window_causal_mask(sliding_window_size=sliding_window_size),
            sliding_window_size=sliding_window_size,
        )


@flax_struct.dataclass
@final
class ZeroAttentionBias(BoolAttentionBias):
    """ "Attention bias that adds zero.

    It is better to check whether a bias has `value()` None rather than using
    an isinstacne check on this class, since the former is more general.
    """

    def _bool_value(self) -> None:
        return None

    def partition_spec(
        self, mha_dim_to_partition_spec: dict[str, PartitionSpec]
    ) -> Union[BaseAttentionBias, PartitionSpec]:
        # Nothing to shard.
        return PartitionSpec()

    def __eq__(self, other):
        return type(other) is type(self)


def _composite_masks(op: Callable[[Tensor, Tensor], Tensor], *mask_fns: ConfigOr[MaskFn]):
    if len(mask_fns) == 0:
        raise RuntimeError(f"Input must not be empty: {mask_fns}")

    def mask(query_position: Tensor, key_position: Tensor):
        fns = [maybe_instantiate(arg) for arg in mask_fns]
        result = fns[0](query_position, key_position)
        for mask in fns[1:]:
            result = op(result, mask(query_position, key_position))
        return result

    return mask


def or_masks(*mask_fns: ConfigOr[MaskFn]) -> MaskFn:
    """Returns a MaskFn that's the union of provided MaskFn's."""
    return _composite_masks(jnp.logical_or, *mask_fns)


def and_masks(*mask_fns: ConfigOr[MaskFn]) -> MaskFn:
    """Returns a MaskFn that's the intersection of provided MaskFn's."""
    return _composite_masks(jnp.logical_and, *mask_fns)


def sliding_window_causal_mask(sliding_window_size: int) -> MaskFn:
    """Returns a causal MaskFn for sliding window attentions of a given window size.

    Implements the `MaskFn` protocol.

    Note: Setting sliding_window_size = 8 results in attending to 9 tokens - it attends to itself
    and sliding_window_size tokens to the left.

    Args:
        sliding_window_size: Left context of sliding window mask.
    """

    def mask(query_position: Tensor, key_position: Tensor):
        pos_mask = query_position - key_position <= sliding_window_size
        return pos_mask

    fun = and_masks(causal_mask, mask)
    return fun


def make_causal_biases(seq_len: int) -> Tensor:
    """Generates attention logit biases for causal masking.

    Args:
        seq_len: Sequence length.

    Returns:
        A float tensor of shape [seq_len, seq_len] where the value at [i, j] = -inf if i < j,
        0 otherwise.
    """
    # TODO(sneha): Support batching.
    return bool_to_bias(causal_mask(jnp.arange(seq_len)[:, None], jnp.arange(seq_len)[None, :]))


def make_sliding_window_causal_biases(seq_len: int, sliding_window_size: int) -> Tensor:
    """Generates attention logit biases for sliding window attention.

    Note: Setting sliding_window_size = 8 results in attending to 9 tokens - it attends to itself
    and sliding_window_size tokens to the left.

    Args:
        seq_len: Sequence length.
        sliding_window_size: Left context of sliding window mask.

    Returns:
        A float tensor of shape [seq_len, seq_len] where the value at [i, j] = -inf
        if i - j > sliding_window_size or i < j, 0 otherwise.
    """
    mask_fn = sliding_window_causal_mask(sliding_window_size)
    return bool_to_bias(mask_fn(jnp.arange(seq_len)[:, None], jnp.arange(seq_len)[None, :]))


def bool_to_bias(mask: OpT) -> OpT:
    """Converts a bool mask tensor to a bias mask tensor.

    Maps:
    0 -> -NEG_INF
    1 -> 0.
    """
    if mask is None:
        return None
    if mask.dtype != jnp.bool:
        raise ValueError("mask must be a Boolean tensor.")
    return safe_not(mask) * NEG_INF


def _make_bool_segment_mask(*, source_segments: Tensor, target_segments: Tensor) -> Tensor:
    """The same as `make_segment_mask()` but returns a bool mask tensor instead of a flaot
    bias tensor, where True corresponds to a bias of 0 and False corresponds to a bias of NEG_INF>
    """
    target_segments = jnp.expand_dims(target_segments, -1)
    source_segments = jnp.expand_dims(source_segments, -2)
    return jax.lax.eq(source_segments, target_segments)[:, None, ...]


def make_segment_mask(*, source_segments: Tensor, target_segments: Tensor) -> Tensor:
    """Generates attention logit biases given the segment ids.

    ... such that positions belonging to different segments cannot attend to each other.

    Args:
        source_segments: An integer tensor of shape [batch, ..., source_length].
        target_segments: An integer tensor of shape [batch, ..., target_length].

    Returns:
        A float Tensor of shape [batch, 1, ..., target_length, source_length] where the
        value at [..., i, j] = 0 if target_segments[..., i] == source_segments[..., j], or -inf
        otherwise.
    """
    return NEG_INF * ~_make_bool_segment_mask(
        source_segments=source_segments, target_segments=target_segments
    )
