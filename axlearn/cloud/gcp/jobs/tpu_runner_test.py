# Copyright © 2023 Apple Inc.

"""Tests TPU runner job."""
import contextlib
import tempfile
from typing import Dict
from unittest import mock

from absl.testing import parameterized

from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp import job as gcp_job
from axlearn.cloud.gcp.job_test import mock_job
from axlearn.cloud.gcp.jobs import tpu_runner
from axlearn.cloud.gcp.test_utils import mock_gcp_settings


@contextlib.contextmanager
def mock_tpu(module_name: str):
    """Mocks out TPU get, create, and delete."""
    running_tpus = {}

    def mock_create_tpu(name: str, *args, **kwargs):
        del args, kwargs
        if name not in running_tpus:
            tpu = mock.MagicMock()
            tpu.status = "CREATED"
            running_tpus[name] = tpu

    def mock_get_tpu_node(name: str, *args, **kwargs):
        del args, kwargs
        return running_tpus.get(name, None)

    def mock_delete_tpu(name: str, *args, **kwargs):
        del args, kwargs
        running_tpus.pop(name, None)

    def mock_running_from_vm():
        return True

    def mock_tpu_resource(*args, **kwargs):
        del args, kwargs

    mocks = [
        mock.patch(f"{module_name}.create_tpu", side_effect=mock_create_tpu),
        mock.patch(f"{module_name}.get_tpu_node", side_effect=mock_get_tpu_node),
        mock.patch(f"{module_name}.get_queued_tpu_node", side_effect=mock_get_tpu_node),
        mock.patch(f"{module_name}._tpu_resource", side_effect=mock_tpu_resource),
        mock.patch(f"{module_name}.delete_tpu", side_effect=mock_delete_tpu),
        mock.patch(f"{module_name}.running_from_vm", side_effect=mock_running_from_vm),
    ]

    with contextlib.ExitStack() as stack:
        # Boilerplate to register multiple mocks at once.
        for m in mocks:
            stack.enter_context(m)
        yield


@contextlib.contextmanager
def mock_tpu_statuses(
    job: gcp_job.GCPJob, *, statuses: Dict[str, str], returncodes: Dict[str, int]
):
    assert statuses.keys() == returncodes.keys()

    def mock_execute_remote_cmd(*args, **kwargs):
        del args, kwargs
        procs = []
        for worker_id, status in statuses.items():
            proc = mock.MagicMock()
            proc.stdout = f"STATUS_{worker_id}_{status}"
            proc.returncode = returncodes[worker_id]
            procs.append(proc)
        return procs

    with mock.patch.object(job, "_execute_remote_cmd", side_effect=mock_execute_remote_cmd):
        yield


class TPURunnerJobTest(parameterized.TestCase):
    """Tests TPURunnerJob."""

    def test_get_status(self):
        mocks = [
            mock_tpu(tpu_runner.__name__),
            mock_gcp_settings(bundler.__name__, settings={"ttl_bucket": "ttl_bucket"}),
            mock_job(gcp_job.__name__),
        ]

        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory() as temp_dir:
            # Boilerplate to register multiple mocks at once.
            for m in mocks:
                stack.enter_context(m)

            cfg = tpu_runner.TPURunnerJob.default_config().set(
                name="test",
                tpu_type="v4-16",  # 2 workers.
                command="",
                max_tries=1,
                project="test_project",
                zone="test_zone",
                retry_interval=1,
                output_dir=temp_dir,
            )
            job = cfg.instantiate()

            # pylint: disable=protected-access
            with mock_tpu_statuses(job, statuses={"0": "", "1": ""}, returncodes={"0": 0, "1": 0}):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.NOT_STARTED, job._get_status())

            job._start()

            # Invalid status value.
            with mock_tpu_statuses(job, statuses={"0": "", "1": ""}, returncodes={"0": 0, "1": 0}):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.UNKNOWN, job._get_status())

            # Non-zero exit code.
            with mock_tpu_statuses(
                job, statuses={"0": "RUNNING", "1": "RUNNING"}, returncodes={"0": 0, "1": 1}
            ):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.UNKNOWN, job._get_status())

            # All statuses agree.
            with mock_tpu_statuses(
                job, statuses={"0": "RUNNING", "1": "RUNNING"}, returncodes={"0": 0, "1": 0}
            ):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.RUNNING, job._get_status())

            # Not all statuses agree.
            with mock_tpu_statuses(
                job, statuses={"0": "RUNNING", "1": "NOT_RUNNING"}, returncodes={"0": 0, "1": 0}
            ):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.UNKNOWN, job._get_status())
