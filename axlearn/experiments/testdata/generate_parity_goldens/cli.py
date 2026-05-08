# Copyright © 2025 Apple Inc.

"""CLI entry point for generating parity test golden files.

Usage:
    bazel run //axlearn/experiments/testdata/generate_parity_goldens:generate
    bazel run //axlearn/experiments/testdata/generate_parity_goldens:generate -- --module distilbert
    bazel run //axlearn/experiments/testdata/generate_parity_goldens:generate -- --list
"""

import argparse
import importlib
import os
import sys

from axlearn.experiments.testdata.generate_parity_goldens._common import set_output_dir

MODULES = [
    "distilbert",
    "neural_retrieval",
]


def main():
    parser = argparse.ArgumentParser(description="Generate parity test golden files.")
    parser.add_argument(
        "--module",
        choices=MODULES,
        help="Generate golden files for a specific module only.",
    )
    parser.add_argument("--list", action="store_true", help="List available modules and exit.")
    parser.add_argument(
        "--output-dir",
        help="Output directory for golden files. Defaults to axlearn/experiments/testdata/ "
        "in the workspace root.",
    )
    args = parser.parse_args()

    if args.list:
        for name in MODULES:
            print(f"  {name}")
        return

    # Determine output directory.
    # BUILD_WORKSPACE_DIRECTORY is set by `bazel run` and points to the source workspace.
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
    if args.output_dir:
        output_dir = args.output_dir
    elif workspace:
        output_dir = os.path.join(workspace, "axlearn", "experiments", "testdata")
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "..")

    set_output_dir(output_dir)
    print(f"Output directory: {output_dir}")

    targets = [args.module] if args.module else MODULES

    for name in targets:
        print(f"Generating: {name}")
        mod = importlib.import_module(
            f"axlearn.experiments.testdata.generate_parity_goldens._{name}"
        )
        mod.generate()

    print(f"\nDone. Generated golden files for {len(targets)} module(s).")


if __name__ == "__main__":
    sys.exit(main() or 0)
