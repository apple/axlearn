# Copyright © 2025 Apple Inc.

"""FlashAttention types for all backends."""

from dataclasses import dataclass
from typing import Optional, Protocol

from jax.sharding import PartitionSpec

from axlearn.common.attention_bias import BaseAttentionBias
from axlearn.common.utils import Tensor


class FlashAttentionShardMapFn(Protocol):
    """Protocol for flash attention that can be called in shard_map."""

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias: BaseAttentionBias,
        prng_key: Optional[Tensor],
        *args: Tensor,
    ):
        ...


@dataclass
class FlashAttentionShardMapSpecs:
    """Wraps a function and its additional arguments for use in shard_map.

    Specifically, fn can be called in shard_map like this:
    ```python
    shard_map(
        fn,
        in_specs=(
            ..., # partition specs for query, key, value, bias, prng_key
            *additional_in_specs
        )
    )(query, key, value, bias, prng_key, *additional_args)

    By default, additional_in_specs and additional_args are empty tuples.
    ```
    """

    fn: FlashAttentionShardMapFn
    additional_in_specs: tuple[PartitionSpec, ...] = ()
    additional_args: tuple[Tensor, ...] = ()
