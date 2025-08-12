# Copyright © 2023 Apple Inc.

"""Segment-tree library."""

from typing import Callable, Generic, Optional, TypeVar

import numpy as np

T = TypeVar("T")


# TODO(tlu7): Add negative number support.
class SegmentTree(Generic[T]):
    """Implements a segment tree data structure for efficient range queries (sum, min, max)
    and find the first index that satisfies the request criteria."""

    def __init__(
        self,
        max_index: int,
        operation: Callable[[T, T], T],
        default_value: T,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        """
        Initialize segment tree.

        Args:
            max_index: Maximum index (inclusive) that the tree will handle.
            operation: Binary operation to combine values (e.g., max, min, sum).
            default_value: Default value for empty/uninitialized nodes.
            dtype: NumPy data type for internal storage, auto-inferred if not provided.
        """
        self.max_index = max_index
        # Number of elements (0 to max_index inclusive)
        self.size = max_index + 1
        self.operation = operation
        self.default_value = default_value

        # Tree size needs to be power of 2
        self.tree_size = 1
        while self.tree_size < self.size:
            self.tree_size *= 2

        if dtype is None:
            dtype = self._infer_dtype(default_value)

        self.dtype = dtype
        self.tree = np.full(2 * self.tree_size, default_value, dtype=dtype)

    def _infer_dtype(self, default_value: T) -> np.dtype:
        """Infer appropriate NumPy dtype from default_value type."""
        if isinstance(default_value, (int, np.integer)):
            return np.int32
        elif isinstance(default_value, (float, np.floating)):
            return np.float64
        elif isinstance(default_value, bool):
            return np.bool_
        else:
            return object

    def update(self, index: int, value: T) -> None:
        """Update value at given index and propagate changes upward."""
        if not 0 <= index < self.size:
            raise IndexError(f"Index {index} out of bounds for size {self.size}")

        is_max_operation = self.operation is max
        is_non_negative_default = (
            isinstance(self.default_value, (int, float)) and self.default_value >= 0
        )
        is_negative_value = isinstance(value, (int, float)) and value < 0
        if is_max_operation and is_non_negative_default and is_negative_value:
            raise ValueError(
                f"Negative value {value} not supported for max operations with "
                f"non-negative default_value={self.default_value}. This causes incorrect "
                f"query results. Use default_value=float('-inf') for negative numbers."
            )

        leaf_idx = self.tree_size + index
        self.tree[leaf_idx] = value

        parent = leaf_idx // 2
        while parent > 0:
            left_child = 2 * parent
            right_child = 2 * parent + 1
            self.tree[parent] = self.operation(self.tree[left_child], self.tree[right_child])
            parent //= 2

    def query_range(self, left: int, right: int) -> T:
        """Query operation result over range [left, right] inclusive."""
        if left > right or left < 0 or right >= self.size:
            return self.default_value
        return self._query_range(1, 0, self.tree_size - 1, left, right)

    def _query_range(self, node: int, start: int, end: int, left: int, right: int) -> T:
        """Recursive implementation of range query."""
        if right < start or left > end:
            return self.default_value

        if left <= start and end <= right:
            return self.tree[node]

        mid = (start + end) // 2
        left_result = self._query_range(2 * node, start, mid, left, right)
        right_result = self._query_range(2 * node + 1, mid + 1, end, left, right)
        return self.operation(left_result, right_result)

    def find_first(self, predicate: Callable[[T], bool], start_index: int = 0) -> int:
        """Find first index >= start_index where predicate is True."""
        if start_index >= self.size:
            return -1
        return self._find_first(1, 0, self.tree_size - 1, predicate, start_index)

    def _find_first(
        self, node: int, start: int, end: int, predicate: Callable[[T], bool], search_start: int
    ) -> int:
        """Recursive implementation of find_first."""
        if end < search_start:
            return -1

        # Leaf node
        if start == end:
            if start >= search_start and predicate(self.tree[node]):
                return start
            return -1

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        # Search left subtree first (for smallest index)
        if mid >= search_start and predicate(self.tree[left_child]):
            result = self._find_first(left_child, start, mid, predicate, search_start)
            if result != -1:
                return result

        # Search right subtree
        if predicate(self.tree[right_child]):
            return self._find_first(
                right_child, mid + 1, end, predicate, max(search_start, mid + 1)
            )

        return -1

    def get_value(self, index: int) -> T:
        """Get value at a specific index."""
        if not 0 <= index < self.size:
            return self.default_value
        return self.tree[self.tree_size + index]


class BestFitSegmentTree:
    """A segment tree adapter for best-fit decreasing bin packing."""

    def __init__(self, max_capacity: int):
        """
        Initialize best-fit segment tree.

        Args:
            max_capacity: Maximum capacity that any bin can have (must be positive).
        """
        if max_capacity <= 0:
            raise ValueError("max_capacity must be positive")

        self.max_capacity = max_capacity

        self.tree: SegmentTree[int] = SegmentTree(
            max_index=max_capacity, operation=max, default_value=0
        )

        self.bins_by_capacity = np.zeros(max_capacity + 1, dtype=np.int32)

        self.bins_by_capacity[max_capacity] = 1
        self.tree.update(max_capacity, max_capacity)

    def find_best_fit(self, item_weight: int) -> int:
        """Find the smallest available capacity >= item_weight."""
        if item_weight <= 0 or item_weight > self.max_capacity:
            return -1

        return self.tree.find_first(
            predicate=lambda capacity: capacity >= item_weight, start_index=item_weight
        )

    def update_capacity(self, old_capacity: int, new_capacity: int) -> None:
        """Update when a bin's capacity changes."""
        if old_capacity < 0 or new_capacity < 0:
            return

        # Remove old capacity
        if old_capacity <= self.max_capacity:
            self.bins_by_capacity[old_capacity] = max(0, self.bins_by_capacity[old_capacity] - 1)

            if self.bins_by_capacity[old_capacity] == 0:
                self.tree.update(old_capacity, 0)
            else:
                self.tree.update(old_capacity, old_capacity)

        # Add new capacity
        if new_capacity <= self.max_capacity:
            self.bins_by_capacity[new_capacity] += 1
            self.tree.update(new_capacity, new_capacity)

    def add_new_bin(self, capacity: int) -> None:
        """Add a new bin with the given capacity."""
        if 0 <= capacity <= self.max_capacity:
            self.bins_by_capacity[capacity] += 1
            self.tree.update(capacity, capacity)

    def get_max_available_capacity(self) -> int:
        """Get the maximum available capacity across all bins."""
        return self.tree.query_range(0, self.max_capacity)

    def get_bin_by_capacity(self) -> np.ndarray:
        """Get copy of bins_by_capacity."""
        return self.bins_by_capacity.copy()

    def get_bin_count_at_capacity(self, capacity: int) -> int:
        """Get number of bins with exact capacity."""
        if 0 <= capacity <= self.max_capacity:
            return self.bins_by_capacity[capacity]
        return 0

    def get_tree_value_at_capacity(self, capacity: int) -> int:
        """Get tree value for specific capacity."""
        return self.tree.get_value(capacity)
