# Copyright © 2023 Apple Inc.

"""AXLearn GCP module."""
import pathlib

# Root of the GCP module, i.e. /path/to/gcp.
GCP_ROOT = pathlib.Path(__file__).resolve().parent
ROOT_MODULE_NAME = GCP_ROOT.parent.name
