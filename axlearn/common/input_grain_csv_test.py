# Copyright © 2024 Apple Inc.

"""Tests for CSV/TSV input processing in input_grain.py."""

import csv
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

from axlearn.common.input_grain import csv_dataset, tsv_dataset


class CsvDatasetTest(unittest.TestCase):
    """Tests for CSV/TSV dataset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.exit_stack = ExitStack()
        self.temp_dir = self.exit_stack.enter_context(tempfile.TemporaryDirectory())
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        self.exit_stack.close()

    def _create_csv_file(self, filename: str, data: list[list[str]]):
        """Helper to create a CSV file for testing."""
        file_path = self.temp_path / filename
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
        return str(file_path)

    def _create_tsv_file(self, filename: str, data: list[list[str]]):
        """Helper to create a TSV file for testing."""
        file_path = self.temp_path / filename
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in data:
                writer.writerow(row)
        return str(file_path)

    def test_csv_dataset_with_header(self):
        """Test CSV dataset with header row."""
        data = [
            ["name", "age", "city"],
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
            ["Charlie", "35", "Chicago"],
        ]
        csv_path = self._create_csv_file("test.csv", data)

        ds = csv_dataset(csv_path)
        examples = list(ds)

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0], {"name": "Alice", "age": "25", "city": "New York"})
        self.assertEqual(examples[1], {"name": "Bob", "age": "30", "city": "San Francisco"})
        self.assertEqual(examples[2], {"name": "Charlie", "age": "35", "city": "Chicago"})

    def test_csv_dataset_without_header(self):
        """Test CSV dataset without header row."""
        data = [
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
            ["Charlie", "35", "Chicago"],
        ]
        csv_path = self._create_csv_file("test.csv", data)

        ds = csv_dataset(csv_path, has_header=False, column_names=["name", "age", "city"])
        examples = list(ds)

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0], {"name": "Alice", "age": "25", "city": "New York"})
        self.assertEqual(examples[1], {"name": "Bob", "age": "30", "city": "San Francisco"})
        self.assertEqual(examples[2], {"name": "Charlie", "age": "35", "city": "Chicago"})

    def test_csv_dataset_custom_column_names(self):
        """Test CSV dataset with custom column names overriding header."""
        data = [
            ["old_name", "old_age", "old_city"],
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
        ]
        csv_path = self._create_csv_file("test.csv", data)

        ds = csv_dataset(csv_path, has_header=True, column_names=["name", "age", "city"])
        examples = list(ds)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0], {"name": "Alice", "age": "25", "city": "New York"})
        self.assertEqual(examples[1], {"name": "Bob", "age": "30", "city": "San Francisco"})

    def test_csv_dataset_skip_rows(self):
        """Test CSV dataset with skip_rows parameter."""
        data = [
            ["name", "age", "city"],
            ["# This is a comment"],
            ["# Another comment"],
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
        ]
        csv_path = self._create_csv_file("test.csv", data)

        ds = csv_dataset(csv_path, skip_rows=2)
        examples = list(ds)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0], {"name": "Alice", "age": "25", "city": "New York"})
        self.assertEqual(examples[1], {"name": "Bob", "age": "30", "city": "San Francisco"})

    def test_csv_dataset_multiple_files(self):
        """Test CSV dataset with multiple files."""
        data1 = [
            ["name", "age"],
            ["Alice", "25"],
            ["Bob", "30"],
        ]
        data2 = [
            ["name", "age"],
            ["Charlie", "35"],
            ["David", "40"],
        ]

        csv_path1 = self._create_csv_file("test1.csv", data1)
        csv_path2 = self._create_csv_file("test2.csv", data2)

        ds = csv_dataset([csv_path1, csv_path2])
        examples = list(ds)

        self.assertEqual(len(examples), 4)
        self.assertEqual(examples[0], {"name": "Alice", "age": "25"})
        self.assertEqual(examples[1], {"name": "Bob", "age": "30"})
        self.assertEqual(examples[2], {"name": "Charlie", "age": "35"})
        self.assertEqual(examples[3], {"name": "David", "age": "40"})

    def test_csv_dataset_malformed_rows(self):
        """Test CSV dataset handling of malformed rows."""
        # Create a CSV with inconsistent column counts
        file_path = self.temp_path / "malformed.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            f.write("name,age,city\n")
            f.write("Alice,25,New York\n")
            f.write("Bob,30\n")  # Missing city
            f.write("Charlie,35,Chicago,Extra\n")  # Extra column

        ds = csv_dataset(str(file_path))
        examples = list(ds)

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0], {"name": "Alice", "age": "25", "city": "New York"})
        self.assertEqual(examples[1], {"name": "Bob", "age": "30", "city": ""})  # Padded
        self.assertEqual(
            examples[2], {"name": "Charlie", "age": "35", "city": "Chicago"}
        )  # Truncated

    def test_tsv_dataset(self):
        """Test TSV dataset functionality."""
        data = [
            ["name", "age", "city"],
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
        ]
        tsv_path = self._create_tsv_file("test.tsv", data)

        ds = tsv_dataset(tsv_path)
        examples = list(ds)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0], {"name": "Alice", "age": "25", "city": "New York"})
        self.assertEqual(examples[1], {"name": "Bob", "age": "30", "city": "San Francisco"})

    def test_csv_dataset_with_seed(self):
        """Test CSV dataset with seed for reproducible shuffling."""
        data = [
            ["name", "age"],
            ["Alice", "25"],
            ["Bob", "30"],
            ["Charlie", "35"],
            ["David", "40"],
        ]
        csv_path = self._create_csv_file("test.csv", data)

        # Create two datasets with the same seed
        ds1 = csv_dataset(csv_path, seed=42).shuffle(buffer_size=10)
        ds2 = csv_dataset(csv_path, seed=42).shuffle(buffer_size=10)

        examples1 = list(ds1)
        examples2 = list(ds2)

        # Should be the same order due to same seed
        self.assertEqual(examples1, examples2)
        self.assertEqual(len(examples1), 4)

    def test_csv_dataset_indexing(self):
        """Test CSV dataset supports indexing."""
        data = [
            ["name", "age"],
            ["Alice", "25"],
            ["Bob", "30"],
            ["Charlie", "35"],
        ]
        csv_path = self._create_csv_file("test.csv", data)

        ds = csv_dataset(csv_path)

        # Test length
        self.assertEqual(len(ds), 3)

        # Test indexing
        self.assertEqual(ds[0], {"name": "Alice", "age": "25"})
        self.assertEqual(ds[1], {"name": "Bob", "age": "30"})
        self.assertEqual(ds[2], {"name": "Charlie", "age": "35"})

    def test_csv_dataset_error_no_column_names(self):
        """Test CSV dataset raises error when no column names provided and no header."""
        data = [
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"],
        ]
        csv_path = self._create_csv_file("test.csv", data)

        with self.assertRaises(ValueError) as cm:
            csv_dataset(csv_path, has_header=False)

        self.assertIn("column_names must be provided", str(cm.exception))

    def test_csv_dataset_empty_file(self):
        """Test CSV dataset with empty file."""
        file_path = self.temp_path / "empty.csv"
        file_path.touch()

        with self.assertRaises(StopIteration):
            # Should raise StopIteration when trying to read header from empty file
            csv_dataset(str(file_path))

    def test_csv_dataset_encoding(self):
        """Test CSV dataset with different encoding."""
        # Create a file with UTF-8 content
        file_path = self.temp_path / "utf8.csv"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("name,description\n")
            f.write("Alice,Café owner\n")
            f.write("Bob,Naïve user\n")

        ds = csv_dataset(str(file_path), encoding="utf-8")
        examples = list(ds)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0], {"name": "Alice", "description": "Café owner"})
        self.assertEqual(examples[1], {"name": "Bob", "description": "Naïve user"})


if __name__ == "__main__":
    unittest.main()
