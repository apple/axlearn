# Copyright © 2026 Apple Inc.

"""Tests for `paged_kv_storage`."""

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax.sharding import PartitionSpec

from axlearn.common.kv_cache.base_kv_cache import BaseKVCache, KVState
from axlearn.common.kv_cache.paged_kv_storage import (
    _BF16_KERNEL_REGISTRY,
    Bf16PagedStorage,
    PagedKVStorage,
    reconstruct_kv,
    register_bf16_kernel,
)
from axlearn.common.test_utils import TestCase


def _make_bf16_storage(
    *,
    batch: int = 2,
    num_heads: int = 3,
    num_pages: int = 8,
    page_size: int = 4,
    head_dim: int = 16,
    pages_per_request: int = 3,
    seed: int = 0,
) -> Bf16PagedStorage:
    k_key, v_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    k_pages = jax.random.normal(
        k_key, (num_heads, num_pages, page_size, head_dim), dtype=jnp.bfloat16
    )
    v_pages = jax.random.normal(
        v_key, (num_heads, num_pages, page_size, head_dim), dtype=jnp.bfloat16
    )
    # Distinct per-row assignments, skipping page 0 (reserved as padding).
    page_indices = jnp.arange(1, batch * pages_per_request + 1, dtype=jnp.int32).reshape(
        batch, pages_per_request
    )
    key_positions = jnp.broadcast_to(
        jnp.arange(pages_per_request * page_size, dtype=jnp.int32),
        (batch, pages_per_request * page_size),
    )
    return Bf16PagedStorage(
        k_proj=k_pages,
        v_proj=v_pages,
        page_indices=page_indices,
        key_positions=key_positions,
    )


class PagedKVStorageProtocolTest(TestCase):
    """Structural-typing tests for the protocol."""

    def test_bf16_storage_is_paged_storage(self):
        storage = _make_bf16_storage()
        self.assertIsInstance(storage, PagedKVStorage)

    def test_bf16_storage_field_order(self):
        # PyTree flattening is positional; downstream code (kernel dispatch,
        # `KVState(*cache_output)` migrations) relies on this order.
        self.assertEqual(
            Bf16PagedStorage._fields,
            ("k_proj", "v_proj", "page_indices", "key_positions"),
        )


class Bf16PagedStorageMethodsTest(TestCase):

    def test_kernel_inputs_keys_and_tensors(self):
        storage = _make_bf16_storage()
        mha = {
            "nbph": PartitionSpec("data", None, None, None),
            "bs": PartitionSpec("data", None),
        }
        input_batch, input_specs = storage.kernel_inputs(mha)

        # Keys must match today's FlashAttention.layer paged branch so the
        # caller can hand them straight through to the attention kernel.
        self.assertEqual(set(input_batch), {"key", "value", "page_tables"})
        self.assertEqual(set(input_specs), {"key", "value", "page_tables"})

        self.assertIs(input_batch["key"], storage.k_proj)
        self.assertIs(input_batch["value"], storage.v_proj)
        self.assertIs(input_batch["page_tables"], storage.page_indices)

        self.assertEqual(input_specs["key"], mha["nbph"])
        self.assertEqual(input_specs["value"], mha["nbph"])
        self.assertEqual(input_specs["page_tables"], mha["bs"])

    def test_kernel_inputs_page_tables_spec_defaults_to_replicated(self):
        storage = _make_bf16_storage()
        mha = {"nbph": PartitionSpec()}  # no "bs" key
        _, input_specs = storage.kernel_inputs(mha)
        self.assertEqual(input_specs["page_tables"], PartitionSpec(None))

    def test_as_dense_matches_reconstruct_kv(self):
        storage = _make_bf16_storage()
        dense_k, dense_v = storage.as_dense()

        expected_k = reconstruct_kv(storage.page_indices, storage.k_proj)
        expected_v = reconstruct_kv(storage.page_indices, storage.v_proj)

        self.assertNestedAllClose(dense_k, expected_k)
        self.assertNestedAllClose(dense_v, expected_v)

    def test_kernel_for_missing_backend_raises_with_hint(self):
        backend = "backend_that_is_definitely_not_registered"
        self.assertNotIn(backend, _BF16_KERNEL_REGISTRY)
        with self.assertRaises(KeyError) as cm:
            Bf16PagedStorage.kernel_for(backend)
        self.assertIn(backend, str(cm.exception))
        self.assertIn("Registered backends", str(cm.exception))


class RegisterBf16KernelTest(TestCase):

    def setUp(self):
        super().setUp()
        self._snapshot = dict(_BF16_KERNEL_REGISTRY)

    def tearDown(self):
        _BF16_KERNEL_REGISTRY.clear()
        _BF16_KERNEL_REGISTRY.update(self._snapshot)
        super().tearDown()

    def test_register_and_lookup(self):
        @register_bf16_kernel("test_backend_xyz")
        def kernel(**kwargs):
            return kwargs

        self.assertIs(Bf16PagedStorage.kernel_for("test_backend_xyz"), kernel)


class AsDenseKvDispatchTest(TestCase):
    """Freeze `BaseKVCache.as_dense_kv` dispatch.

    Downstream callers (`rattention.py` after the paged-SWA fix,
    `attention.py`'s base `_compute_attention`) rely on this method
    to hand back dense `(k, v)` regardless of whether the cache's
    `extend_step` emitted a `KVState` or a paged variant. This test
    ensures the dispatch never quietly regresses — if someone adds a
    third `AttentionInputs` arm and forgets to teach `as_dense_kv`
    about it, this test catches it.
    """

    def test_dense_kv_state_returns_inputs_unchanged(self):
        k = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 2, 8), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 2, 8), dtype=jnp.bfloat16)
        kp = jnp.arange(4, dtype=jnp.int32)[None]
        state = KVState(k_proj=k, v_proj=v, key_positions=kp)

        k_out, v_out = BaseKVCache.as_dense_kv(state)

        # Dense path: identity — no copy, no reconstruction.
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_paged_storage_returns_dense_reconstruction(self):
        storage = _make_bf16_storage()  # Bf16PagedStorage fixture

        k_out, v_out = BaseKVCache.as_dense_kv(storage)

        # Paged path: must materialise dense tensors from the page pool —
        # NOT hand back the pool (`[num_kv_heads, total_num_pages,
        # page_size, head_dim]`) as-is. That was the silent-corruption
        # failure mode flagged in review.
        batch, pages_per_request = storage.page_indices.shape
        num_kv_heads, _, page_size, head_dim = storage.k_proj.shape
        self.assertEqual(
            k_out.shape, (batch, pages_per_request * page_size, num_kv_heads, head_dim)
        )
        self.assertEqual(
            v_out.shape, (batch, pages_per_request * page_size, num_kv_heads, head_dim)
        )
        self.assertTrue(
            jnp.array_equal(k_out, reconstruct_kv(storage.page_indices, storage.k_proj))
        )
        self.assertTrue(
            jnp.array_equal(v_out, reconstruct_kv(storage.page_indices, storage.v_proj))
        )


if __name__ == "__main__":
    absltest.main()
