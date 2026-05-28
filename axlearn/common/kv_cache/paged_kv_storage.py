# Copyright © 2026 Apple Inc.

"""Paged KV cache storage protocols and layout primitives.

Defines a structural `PagedKVStorage` protocol for paged KV cache
variants, plus `Bf16PagedStorage`, the concrete NamedTuple emitted
by `PagedKVCache`. Each variant exposes three methods:

- `kernel_inputs`: paged-specific entries for the attention kernel's
  input dict (merged with query/bias by the caller).
- `as_dense`: materialised `(k, v)` for fallback / debug paths.
- `kernel_for`: classmethod returning the backend kernel registered
  for this variant.

Adding a new variant (quantised, compressed, mixed-precision) means one
new NamedTuple plus its kernel registration — no edits to
`KVState`, `FlashAttention`, or the flash-attention
dispatch.

Also hosts the dense-reconstruction and scatter-update primitives
(`reconstruct_kv`, `scatter_update_pages`,
`scatter_update_pages_kernel`) that operate on a paged layout.
They live here rather than in `paged_kv_cache` so that storage
methods (e.g. `Bf16PagedStorage.as_dense`) can call them without
forming an import cycle against the cache Layer that consumes them.
"""

from typing import Any, Callable, Mapping, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common.utils import Tensor, get_current_abstract_or_physical_mesh


def reconstruct_kv(page_tables: Tensor, pages: Tensor) -> Tensor:
    """Retrieve key/value from page tables given pages.

    Args:
        page_tables: [batch_size, pages_per_sequence], specifying page indices.
        pages: [num_kv_heads, total_num_pages, page_size, head_dim], k/v pages.

    Returns:
        Retrieved actual key / value of shape [batch_size, kv_seq_len, n_kv_heads, head_dim],
            where kv_seq_len = pages_per_sequence * page_size.
    """
    temp = jnp.einsum("nbpsh->bpsnh", pages.at[:, page_tables].get(mode="fill", fill_value=0))
    b, _, _, n, h = temp.shape
    return temp.reshape(b, -1, n, h)


def scatter_update_pages(
    kv_pages: Tensor, kv_proj: Tensor, page_indices: Tensor, key_positions: Tensor
) -> Tensor:
    """Scatter kv_proj into kv_pages according to key_positions.

    Args:
        kv_pages: A tensor of shape [num_heads, num_pages, page_size, head_dim].
        kv_proj: A tensor of shape [num_heads, batch_size, 1, head_dim].
        page_indices: A tensor of shape [batch_size, pages_per_batch].
        key_positions: A tensor of shape [batch_size, 1].

    Returns:
        A tensor with the same shape as `kv_pages`.
    """
    page_size = kv_pages.shape[-2]
    offset_in_page = key_positions % page_size  # (batch, step)
    page_idx = key_positions // page_size  # (batch, step)
    page_idx = jnp.take_along_axis(page_indices, page_idx, axis=1)
    # unique_indices=True provides 10x faster perf for the scatter kernel on GPU.
    kv_pages = kv_pages.at[:, page_idx, offset_in_page].set(
        kv_proj, unique_indices=True, mode="drop"
    )
    return kv_pages


def scatter_update_pages_kernel(
    *,
    kv_pages: Tensor,
    kv_proj: Tensor,
    page_indices: Tensor,
    key_positions: Tensor,
    shmap_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    """Equivalent to `scatter_update_pages` but much faster on TPU and GPU.

    Args:
        kv_pages: A tensor of shape [num_heads, num_pages, page_size, head_dim].
        kv_proj: A tensor of shape [num_heads, batch_size, 1, head_dim].
        page_indices: A tensor of shape [batch_size, pages_per_batch].
        key_positions: A tensor of shape [batch_size, 1].
        shmap_fn: A callable that takes in the tensors above and returns an updated `kv_pages` in
            a shard_map context.

    Returns:
        A tensor with the same shape as `kv_pages`.
    """
    key_positions = key_positions.squeeze(1)
    mesh = get_current_abstract_or_physical_mesh()
    num_kv_heads = kv_pages.shape[0]
    model_axis = (
        "model" if "model" in mesh.axis_names and num_kv_heads % mesh.shape["model"] == 0 else None
    )
    # pylint: disable-next=too-many-function-args
    return jax.shard_map(
        shmap_fn,
        mesh=mesh,
        in_specs=(
            PartitionSpec(model_axis, None, None, None),
            PartitionSpec(model_axis, None, None, None),
            PartitionSpec(None, None),
            PartitionSpec(None),
        ),
        out_specs=PartitionSpec(model_axis, None, None, None),
        check_vma=False,
    )(kv_pages, kv_proj, page_indices, key_positions)


@runtime_checkable
class PagedKVStorage(Protocol):
    """Structural protocol for a paged KV storage variant.

    Concrete implementations are NamedTuples whose fields are exactly the
    tensors the matching attention kernel consumes. Every paged variant
    exposes `page_indices` (the per-request page table) and
    `key_positions`; other fields (pages, scales, zero-points, ...)
    differ per variant.
    """

    page_indices: Tensor
    key_positions: Tensor

    def kernel_inputs(
        self,
        mha_dim_to_partition_spec: Mapping[str, PartitionSpec],
    ) -> tuple[dict[str, Any], dict[str, PartitionSpec]]:
        """Produce the paged-specific entries of the kernel's input dict.

        Args:
            mha_dim_to_partition_spec: Mapping from axlearn's dim-pattern
                names (e.g. `"nbph"`, `"bs"`) to their
                `PartitionSpec`, as configured on
                `FlashAttention`.

        Returns:
            A pair `(input_batch, input_batch_specs)`: dicts of tensors
            and their partition specs, keyed by the name the attention
            kernel expects. The caller merges them with its own query /
            bias / prng-key entries and any backend extras.
        """

    def as_dense(self) -> tuple[Tensor, Tensor]:
        """Materialise dense `(k, v)` for fallback / debug paths.

        Returns:
            `(k, v)` of shape `[batch, kv_seq_len, num_kv_heads, head_dim]`.
        """

    @classmethod
    def kernel_for(cls, backend: str) -> Callable:
        """Return the attention kernel registered for this storage + backend."""


_BF16_KERNEL_REGISTRY: dict[str, Callable] = {}


def register_bf16_kernel(backend: str) -> Callable[[Callable], Callable]:
    """Register `fn` as the bf16 paged attention kernel for `backend`.

    Decorator form::

        @register_bf16_kernel("tpu")
        def my_tpu_paged_attention(...): ...

    Invoked as an import-time side effect from each kernel module; this
    keeps dispatch wiring local to the kernel file and avoids circular
    imports with `paged_kv_cache`.
    """

    def decorator(fn: Callable) -> Callable:
        _BF16_KERNEL_REGISTRY[backend] = fn
        return fn

    return decorator


class Bf16PagedStorage(NamedTuple):
    """Paged KV storage emitted by `PagedKVCache` (bf16 pages).

    The fields `k_proj` / `v_proj` mirror `KVState`'s naming so
    call sites that accessed `kv_state.k_proj` in the pre-refactor world
    continue to work — but for paged storage these tensors are the **page
    pool**, shape `[num_kv_heads, total_num_pages, page_size, head_dim]`,
    not dense projections. Use `as_dense` to materialise a dense
    `(k, v)` view.

    TODO: rename `k_proj` / `v_proj` back to `k_pages` / `v_pages`
    once the `scale_kv_before_cache_update=False` path in
    `axlearn.common.attention` no longer reads `kv_state.k_proj`
    and calls `kv_state._replace(k_proj=...)` directly — i.e., once
    that code routes through `BaseKVCache.as_dense_kv`. The
    current naming is a compat lie for that single call site.

    Fields:
        k_proj: Key page pool, `[num_kv_heads, total_num_pages, page_size, head_dim]`.
        v_proj: Value page pool, `[num_kv_heads, total_num_pages, page_size, head_dim]`.
        page_indices: `[batch, max_pages_per_request]` per-request page tables.
        key_positions: `[batch, source_length]`.
    """

    k_proj: Tensor
    v_proj: Tensor
    page_indices: Tensor
    key_positions: Tensor

    def kernel_inputs(
        self,
        mha_dim_to_partition_spec: Mapping[str, PartitionSpec],
    ) -> tuple[dict[str, Any], dict[str, PartitionSpec]]:
        input_batch = {
            "key": self.k_proj,
            "value": self.v_proj,
            "page_tables": self.page_indices,
        }
        kv_spec = mha_dim_to_partition_spec["nbph"]
        page_tables_spec = mha_dim_to_partition_spec.get("bs", PartitionSpec(None))
        input_batch_specs = {
            "key": kv_spec,
            "value": kv_spec,
            "page_tables": page_tables_spec,
        }
        return input_batch, input_batch_specs

    def as_dense(self) -> tuple[Tensor, Tensor]:
        return (
            reconstruct_kv(self.page_indices, self.k_proj),
            reconstruct_kv(self.page_indices, self.v_proj),
        )

    @classmethod
    def kernel_for(cls, backend: str) -> Callable:
        try:
            return _BF16_KERNEL_REGISTRY[backend]
        except KeyError as e:
            raise KeyError(
                f"No bf16 paged attention kernel registered for backend {backend!r}. "
                f"Registered backends: {sorted(_BF16_KERNEL_REGISTRY)}"
            ) from e
