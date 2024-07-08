# Copyright © 2023 Apple Inc.

"""Tests TPU job cleaner."""
# pylint: disable=unused-argument

import contextlib
import datetime
from unittest import mock

from absl.testing import parameterized

from axlearn.cloud.common.bastion import new_jobspec
from axlearn.cloud.common.types import JobMetadata
from axlearn.cloud.gcp import tpu_cleaner
from axlearn.cloud.gcp.tpu import TpuInfo


class TestTPUCleaner(parameterized.TestCase):
    """Tests TPUCleaner."""

    def test_sweep(self):
        tpus = {
            "a": TpuInfo(name="a", accelerator_type="v4-8", state="", metadata={}),
            "c": TpuInfo(name="c", accelerator_type="v3-8", state="", metadata={}),
            "d": TpuInfo(name="d", accelerator_type="v4-64", state="", metadata={}),
        }
        jobs = {
            "a": dict(v4=8),  # Standard case.
            "b": dict(v4=32),  # Already cleaned up.
            "c": dict(v4=8),  # Inconsistent resource assignment.
            "d": dict(v4=64),  # Different TPU cores.
        }
        jobs = {
            k: new_jobspec(
                name="a",
                command="cmd",
                metadata=JobMetadata(
                    user_id="user",
                    project_id="jetpack",
                    creation_time=datetime.datetime(1900, 1, 1, 0, 0, 0),
                    resources=v,
                ),
            )
            for k, v in jobs.items()
        }

        def mock_get_credentials(*args, **kwargs):
            return None

        def mock_list_tpu_info(*args, **kwargs):
            return []

        def mock_list_queued_resource_info(*args, **kwargs):
            return list(tpus.values())

        def mock_delete_batch(tpu_names, **kwargs):
            for tpu_name in list(tpus.keys()):
                if tpu_name in tpu_names:
                    tpus.pop(tpu_name)

        cfg = tpu_cleaner.TPUCleaner.default_config()
        cleaner = cfg.instantiate()

        module_name = tpu_cleaner.__name__
        mocks = [
            mock.patch(f"{module_name}.qrm_resource", return_value=mock.MagicMock()),
            mock.patch(f"{module_name}.tpu_resource", return_value=mock.MagicMock()),
            mock.patch(f"{module_name}.get_credentials", side_effect=mock_get_credentials),
            mock.patch(f"{module_name}.list_tpu_info", side_effect=mock_list_tpu_info),
            mock.patch(
                f"{module_name}.list_queued_resource_info",
                side_effect=mock_list_queued_resource_info,
            ),
            mock.patch.object(cleaner, "_delete_batch", side_effect=mock_delete_batch),
        ]
        # Boilerplate to register multiple mocks at once.
        with contextlib.ExitStack() as stack:
            for m in mocks:
                stack.enter_context(m)

            # In the first sweep, only "b" will be returned (already cleaned up).
            # "a", "c", and "d" will be terminating.
            terminated = cleaner.sweep(jobs)
            self.assertCountEqual(terminated, ["b"])  # Checks for list equality, without order.

            # In the second sweep, all should be terminated.
            terminated = cleaner.sweep(jobs)
            self.assertCountEqual(terminated, ["a", "b", "c", "d"])
