# Copyright © 2023 Apple Inc.

"""AXLearn cloud modules."""
import pathlib

# Root of the cloud module, e.g. /path/to/axlearn.
ROOT_MODULE = pathlib.Path(__file__).resolve().parent.parent
ROOT_MODULE_NAME = ROOT_MODULE.name

# Distribution name as published on PyPI (may differ from the Python module name).
DISTRIBUTION_NAME = "axlearn"
