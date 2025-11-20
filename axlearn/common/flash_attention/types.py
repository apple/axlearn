# Copyright © 2025 Apple Inc.

"""FlashAttention types for all backends."""

from dataclasses import dataclass, field
from typing import Callable

from jax.sharding import PartitionSpec

from axlearn.common.utils import Tensor


@dataclass
class FlashAttentionWithShardMapSpecs:
    """Wraps a function and its additional arguments for use in shard_map.

    Attributes:
        fn: A callable flash attention function for use in shard_map.
        additional_in_specs: Dict mapping argument names to PartitionSpecs for sharding.
        additional_kwargs: Dict mapping argument names to tensor values (e.g., mask infos).

    Example usage:
        ```python
        input_batch_specs = {
            # Standard specs for query, key, value, bias, prng_key, ...
            ...,
            **additional_in_specs,  # Add additional specs
        }
        partitioned_fn = shard_map(fn, in_specs=(input_batch_specs,), ...)

        input_batch = {
            # Standard inputs: query, key, value, bias, prng_key, ...
            ...,
            **additional_kwargs,  # Add additional kwargs
        }
        outputs = partitioned_fn(input_batch)
        ```
    """

    fn: Callable
    additional_in_specs: dict[str, PartitionSpec] = field(default_factory=dict)
    additional_kwargs: dict[str, Tensor] = field(default_factory=dict)
