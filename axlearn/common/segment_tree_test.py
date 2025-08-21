# Copyright © 2023 Apple Inc.

"""Tests segment-tree library."""

# pylint: disable=no-self-use
from absl.testing import absltest

from axlearn.common.segment_tree import BestFitSegmentTree, SegmentTree
from axlearn.common.test_utils import TestCase


class SegmentTreeTest(TestCase):
    """Test cases for the generic SegmentTree data structure."""

    def test_initialization(self):
        """Test that segment tree initializes correctly."""
        tree = SegmentTree[int](max_index=5, operation=max, default_value=0)
        # All values should be default initially
        for i in range(6):
            self.assertEqual(tree.get_value(i), 0)
        self.assertEqual(tree.query_range(0, 5), 0)

    def test_single_update_and_query(self):
        """Test single updates and point queries."""
        tree = SegmentTree[int](max_index=5, operation=max, default_value=0)
        tree.update(2, 10)

        self.assertEqual(tree.get_value(2), 10)
        self.assertEqual(tree.get_value(1), 0)  # Other values unchanged
        self.assertEqual(tree.query_range(2, 2), 10)  # Single element range
        self.assertEqual(tree.query_range(0, 1), 0)  # Range without updated element

    def test_range_maximum_queries(self):
        """Test range maximum queries."""
        tree = SegmentTree[int](max_index=5, operation=max, default_value=0)
        values = [1, 3, 5, 7, 9]

        for i, val in enumerate(values):
            tree.update(i, val)

        self.assertEqual(tree.query_range(0, 4), 9)  # Entire range
        self.assertEqual(tree.query_range(1, 3), 7)  # Middle range
        self.assertEqual(tree.query_range(0, 0), 1)  # Single element
        self.assertEqual(tree.query_range(2, 4), 9)  # End range
        self.assertEqual(tree.query_range(0, 2), 5)  # Start range

    def test_range_sum_queries(self):
        """Test range sum queries."""
        tree = SegmentTree[int](max_index=5, operation=lambda a, b: a + b, default_value=0)
        values = [1, 3, 5, 7, 9]

        for i, val in enumerate(values):
            tree.update(i, val)

        self.assertEqual(tree.query_range(0, 4), 25)  # Sum of all: 1+3+5+7+9
        self.assertEqual(tree.query_range(1, 3), 15)  # Sum of 3+5+7
        self.assertEqual(tree.query_range(0, 0), 1)  # Single element
        self.assertEqual(tree.query_range(2, 4), 21)  # Sum of 5+7+9

    def test_range_minimum_queries(self):
        """Test range minimum queries."""
        tree = SegmentTree[float](max_index=5, operation=min, default_value=float("inf"))
        values = [9, 3, 7, 1, 5]

        for i, val in enumerate(values):
            tree.update(i, val)

        self.assertEqual(tree.query_range(0, 4), 1)  # Min of all
        self.assertEqual(tree.query_range(0, 2), 3)  # Min of 9,3,7
        self.assertEqual(tree.query_range(2, 4), 1)  # Min of 7,1,5
        self.assertEqual(tree.query_range(1, 1), 3)  # Single element

    def test_find_first_functionality(self):
        """Test find_first method."""
        tree = SegmentTree[int](max_index=5, operation=max, default_value=0)
        values = [1, 3, 5, 7, 9]

        for i, val in enumerate(values):
            tree.update(i, val)

        self.assertEqual(tree.find_first(lambda x: x >= 6), 3)  # Index of 7
        self.assertEqual(tree.find_first(lambda x: x >= 5), 2)  # Index of 5
        self.assertEqual(tree.find_first(lambda x: x >= 1), 0)  # Index of 1
        self.assertEqual(tree.find_first(lambda x: x >= 10), -1)  # No such value

        self.assertEqual(tree.find_first(lambda x: x >= 5, start_index=3), 3)  # Index of 7
        self.assertEqual(tree.find_first(lambda x: x >= 1, start_index=2), 2)  # Index of 5
        self.assertEqual(tree.find_first(lambda x: x >= 1, start_index=5), -1)  # Past end

    def test_find_first_edge_cases(self):
        """Test edge cases for find_first."""
        tree = SegmentTree[int](max_index=3, operation=max, default_value=0)
        tree.update(0, 2)
        tree.update(1, 0)  # Explicitly set to 0
        tree.update(2, 4)

        # Should skip index 1 (value 0) when looking for > 0
        self.assertEqual(tree.find_first(lambda x: x > 0), 0)  # Index of 2
        self.assertEqual(tree.find_first(lambda x: x > 2), 2)  # Index of 4
        self.assertEqual(tree.find_first(lambda x: x >= 0), 0)  # Index of 2 (first >= 0)

    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        tree = SegmentTree[int](max_index=0, operation=max, default_value=0)

        # Operations on empty tree
        self.assertEqual(tree.query_range(0, 0), 0)
        self.assertEqual(tree.find_first(lambda x: x > 0), -1)

        # Single element tree
        single_tree = SegmentTree[int](max_index=0, operation=max, default_value=5)
        single_tree.update(0, 10)
        self.assertEqual(single_tree.get_value(0), 10)
        self.assertEqual(single_tree.query_range(0, 0), 10)

    def test_invalid_indices(self):
        """Test behavior with invalid indices."""
        tree = SegmentTree[int](max_index=5, operation=max, default_value=0)

        with self.assertRaises(IndexError):
            tree.update(-1, 100)
        with self.assertRaises(IndexError):
            tree.update(6, 100)

        # Invalid range queries should return default
        self.assertEqual(tree.query_range(-1, 0), 0)
        self.assertEqual(tree.query_range(5, 6), 0)
        self.assertEqual(tree.query_range(3, 2), 0)  # left > right

        # Invalid get_value should return default
        self.assertEqual(tree.get_value(-1), 0)
        self.assertEqual(tree.get_value(6), 0)

    def test_negative_value_validation(self):
        """Test that negative values are properly validated for max operations."""
        # Test the specific problematic case: max operation with non-negative default
        tree = SegmentTree[int](max_index=5, operation=max, default_value=0)

        with self.assertRaises(ValueError) as context:
            tree.update(0, -5)

        error_msg = str(context.exception)
        self.assertIn("Negative value -5 not supported", error_msg)
        self.assertIn("max operations", error_msg)
        self.assertIn("default_value=0", error_msg)
        self.assertIn("float('-inf')", error_msg)

        # Test edge case: zero should be allowed
        tree.update(0, 0)
        self.assertEqual(tree.get_value(0), 0)

        # Test positive values work fine
        tree.update(1, 5)
        self.assertEqual(tree.get_value(1), 5)

        # Test that other operations with negative values work fine
        sum_tree = SegmentTree[int](max_index=5, operation=lambda x, y: x + y, default_value=0)
        sum_tree.update(0, -5)  # Should not raise for sum operation
        self.assertEqual(sum_tree.get_value(0), -5)

        # Test max with proper negative default works fine
        max_tree_neg = SegmentTree[float](max_index=5, operation=max, default_value=float("-inf"))
        max_tree_neg.update(0, -5)  # Should not raise with proper default
        self.assertEqual(max_tree_neg.get_value(0), -5)

    def test_updates_propagate_correctly(self):
        """Test that updates propagate correctly through the tree."""
        tree = SegmentTree[int](max_index=3, operation=max, default_value=0)

        # Update values and check propagation
        tree.update(0, 5)
        self.assertEqual(tree.query_range(0, 2), 5)  # Max should be 5

        tree.update(1, 10)
        self.assertEqual(tree.query_range(0, 2), 10)  # Max should now be 10

        tree.update(2, 3)
        self.assertEqual(tree.query_range(0, 2), 10)  # Max should still be 10

        # Update the max value to something smaller
        tree.update(1, 2)
        self.assertEqual(tree.query_range(0, 2), 5)  # Max should now be 5


class BestFitSegmentTreeTest(TestCase):
    """Test cases for the BestFitSegmentTree data structure."""

    def test_initialization(self):
        """Test that segment tree initializes correctly."""
        max_capacity = 8
        tree = BestFitSegmentTree(max_capacity)

        # Should be able to find max capacity initially
        self.assertEqual(tree.find_best_fit(8), 8)
        self.assertEqual(tree.find_best_fit(4), 8)
        self.assertEqual(tree.find_best_fit(1), 8)

    def test_find_best_fit_basic(self):
        """Test basic best-fit functionality."""
        tree = BestFitSegmentTree(8)

        # Should find exact capacity or larger
        self.assertEqual(tree.find_best_fit(8), 8)
        self.assertEqual(tree.find_best_fit(7), 8)
        self.assertEqual(tree.find_best_fit(1), 8)

        # Should return -1 for items too large
        self.assertEqual(tree.find_best_fit(9), -1)
        self.assertEqual(tree.find_best_fit(10), -1)

    def test_update_capacity(self):
        """Test capacity updates work correctly."""
        tree = BestFitSegmentTree(8)

        # Initially can find capacity 8 (we have 1 bin with capacity 8)
        self.assertEqual(tree.find_best_fit(8), 8)

        # After using capacity 8 -> 4, we should have 1 bin with capacity 4
        tree.update_capacity(8, 4)
        self.assertEqual(tree.find_best_fit(4), 4)
        self.assertEqual(tree.find_best_fit(3), 4)
        self.assertEqual(tree.find_best_fit(5), -1)

        # After using capacity 4 -> 2, should find 2 for items <= 2
        tree.update_capacity(4, 2)
        self.assertEqual(tree.find_best_fit(2), 2)
        self.assertEqual(tree.find_best_fit(3), -1)

        # Add a new bin with capacity 8
        tree.add_new_bin(8)
        self.assertEqual(tree.find_best_fit(8), 8)
        self.assertEqual(tree.find_best_fit(5), 8)

    def test_multiple_bins_same_capacity(self):
        """Test handling multiple bins with same remaining capacity."""
        tree = BestFitSegmentTree(8)

        tree.add_new_bin(8)

        # Simulate two bins both going from 8 -> 4
        tree.update_capacity(8, 4)  # First bin: 8 -> 4
        # Second bin: 8 -> 4 (creates another 8 capacity)
        tree.update_capacity(8, 4)

        # Should still be able to find capacity 4
        self.assertEqual(tree.find_best_fit(4), 4)
        self.assertEqual(tree.find_best_fit(3), 4)

    def test_internal_state_consistency(self):
        """Test get state of tree."""
        tree = BestFitSegmentTree(10)
        tree.add_new_bin(5)
        tree.update_capacity(10, 7)

        # Test capacity counts
        counts = tree.get_bin_by_capacity()
        self.assertEqual(counts[10], 0)  # No bins with capacity 10
        self.assertEqual(counts[7], 1)  # One bin with capacity 7
        self.assertEqual(counts[5], 1)  # One bin with capacity 5

        # Test tree values match counts
        self.assertEqual(tree.get_tree_value_at_capacity(10), 0)  # No bins = 0
        self.assertEqual(tree.get_tree_value_at_capacity(7), 7)  # Has bins = capacity
        self.assertEqual(tree.get_tree_value_at_capacity(5), 5)  # Has bins = capacity

    def test_best_fit_selects_smallest_capacity(self):
        """Test that best-fit decreasing selects the smallest available capacity."""
        tree = BestFitSegmentTree(10)

        # Add bins with various capacities: 10, 6, 3
        # Now we have bins with capacities: [6, 3, 8]
        tree.update_capacity(10, 6)  # Original bin: 10 -> 6
        tree.add_new_bin(3)  # Add bin with capacity 3
        tree.add_new_bin(8)  # Add bin with capacity 8

        # For item weight 2: should choose capacity 3 (smallest that fits)
        self.assertEqual(tree.find_best_fit(2), 3)

        # For item weight 4: should choose capacity 6 (smallest that fits)
        self.assertEqual(tree.find_best_fit(4), 6)

        # For item weight 7: should choose capacity 8 (only one that fits)
        self.assertEqual(tree.find_best_fit(7), 8)

        # For item weight 9: should return -1 (no capacity large enough)
        self.assertEqual(tree.find_best_fit(9), -1)

    def test_best_fit_with_multiple_same_capacities(self):
        """Test best-fit when multiple bins have the same capacity."""
        tree = BestFitSegmentTree(10)

        # Create multiple bins with capacity 5
        tree.update_capacity(10, 5)  # 10 -> 5
        tree.add_new_bin(5)  # Add another capacity 5
        tree.add_new_bin(7)  # Add capacity 7

        # For item weight 3: should choose 5 (smallest fit), not 7
        self.assertEqual(tree.find_best_fit(3), 5)

        # For item weight 6: should choose 7 (only one that fits)
        self.assertEqual(tree.find_best_fit(6), 7)

    def test_best_fit_ordering_property(self):
        """Test that best-fit consistently returns smallest suitable capacity."""
        tree = BestFitSegmentTree(20)

        # Create bins with capacities in non-sorted order: [15, 5, 10, 8, 12]
        tree.update_capacity(20, 15)  # 20 -> 15
        tree.add_new_bin(5)
        tree.add_new_bin(10)
        tree.add_new_bin(8)
        tree.add_new_bin(12)

        # Test that it always picks the smallest suitable capacity
        self.assertEqual(tree.find_best_fit(1), 5)  # Smallest: 5
        self.assertEqual(tree.find_best_fit(5), 5)  # Exact fit: 5
        self.assertEqual(tree.find_best_fit(6), 8)  # Smallest >= 6: 8
        self.assertEqual(tree.find_best_fit(9), 10)  # Smallest >= 9: 10
        self.assertEqual(tree.find_best_fit(11), 12)  # Smallest >= 11: 12
        self.assertEqual(tree.find_best_fit(13), 15)  # Smallest >= 13: 15


if __name__ == "__main__":
    absltest.main()
