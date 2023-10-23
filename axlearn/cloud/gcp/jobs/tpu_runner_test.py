# Copyright © 2023 Apple Inc.

"""Tests TPU runner job."""
# pylint: disable=protected-access
import contextlib
import os
import tempfile
from typing import List
from unittest import mock

import pytest
from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp import job as gcp_job
from axlearn.cloud.gcp.jobs import tpu_runner
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.cloud.gcp.tpu import TpuInfo
from axlearn.common.config import config_for_function
from axlearn.common.test_utils import TestWithTemporaryCWD


@contextlib.contextmanager
def mock_tpu(module_name: str, running_from_vm: bool = True):
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
        return running_from_vm

    def mock_tpu_resource(*args, **kwargs):
        del args, kwargs

    def mock_list_tpu_info(creds):
        del creds
        return [
            TpuInfo(name=tpu.name, accelerator_type="", state="", metadata={})
            for tpu in running_tpus.values()
        ]

    mocks = {
        "create_tpu": mock_create_tpu,
        "get_tpu_node": mock_get_tpu_node,
        "get_queued_tpu_node": mock_get_tpu_node,
        "_tpu_resource": mock_tpu_resource,
        "delete_tpu": mock_delete_tpu,
        "running_from_vm": mock_running_from_vm,
        "list_tpu_info": mock_list_tpu_info,
    }
    with contextlib.ExitStack() as stack:
        # Boilerplate to register multiple mocks at once, and return the mocks.
        mocks = {
            name: stack.enter_context(mock.patch(f"{module_name}.{name}", side_effect=method))
            for name, method in mocks.items()
        }
        yield mocks


@contextlib.contextmanager
def mock_tpu_statuses(
    job: tpu_runner.TPURunnerJob,
    *,
    num_booted: int,
    statuses: List[str],
    returncodes: List[int],
):
    num_vms = job._num_workers()
    # num_booted should contain number of VMs booted across all workers.
    # statuses[i] should contain status for worker i;
    # returncodes[i] should contain execute_remote_cmd returncode for worker i;
    assert 0 <= num_booted <= num_vms
    assert len(statuses) == len(returncodes) == num_vms

    def mock_get_tpu_node_status(name, *, node, **kwargs):
        del name, kwargs
        assert node is not None
        return dict(num_booted=num_booted)

    def mock_execute_remote_cmd(*args, **kwargs):
        del args, kwargs
        procs = []
        for worker_id, status in enumerate(statuses):
            proc = mock.MagicMock()
            proc.stdout = f"STATUS_{worker_id}_{status}"
            proc.returncode = returncodes[worker_id]
            procs.append(proc)
        return procs

    mock_tpu_status = mock.patch.multiple(
        tpu_runner.__name__,
        get_tpu_node_status=mock.Mock(side_effect=mock_get_tpu_node_status),
        get_queued_tpu_node_status=mock.Mock(side_effect=mock_get_tpu_node_status),
    )
    mock_ssh_status = mock.patch.object(
        job,
        "_execute_remote_cmd",
        side_effect=mock_execute_remote_cmd,
    )
    with mock_tpu_status, mock_ssh_status:
        yield


def _mock_config():
    mock_bundler = mock.MagicMock()
    mock_bundler.install_command.return_value = "test_install"
    return tpu_runner.TPURunnerJob.default_config().set(
        name="test-name",
        output_dir="test_output",
        tpu_type="v4-8",
        num_slices=2,
        project="test_project",
        zone="test_zone",
        max_tries=1,
        retry_interval=60,
        bundler=config_for_function(lambda: mock_bundler),
    )


@contextlib.contextmanager
def _mock_credentials():
    mocks = [
        mock.patch(f"{gcp_job.__name__}.get_credentials"),
        mock.patch(f"{tpu_runner.__name__}.get_credentials"),
    ]
    with contextlib.ExitStack() as stack:
        for m in mocks:
            stack.enter_context(m)
        yield


class TPURunnerJobTest(TestWithTemporaryCWD):
    """Tests TPURunnerJob."""

    @parameterized.parameters(
        dict(num_workers=1, tpu_type="v4-8", num_slices=1),
        dict(num_workers=2, tpu_type="v4-8", num_slices=2),
        dict(num_workers=2, tpu_type="v4-16", num_slices=1),
        dict(num_workers=4, tpu_type="v4-16", num_slices=2),
    )
    def test_num_workers(self, num_workers, tpu_type, num_slices):
        cfg = _mock_config()
        job = cfg.set(command="", tpu_type=tpu_type, num_slices=num_slices).instantiate()
        self.assertEqual(num_workers, job._num_workers())

    @parameterized.parameters(True, False)
    def test_start(self, running_from_vm):
        cfg = _mock_config()
        job = cfg.set(command="").instantiate()

        mock_execute = mock.patch.object(job, "_execute_remote_cmd")
        mock_credentials = mock.patch.object(job, "_get_job_credentials")

        with mock_execute, mock_credentials, mock_tpu(
            tpu_runner.__name__, running_from_vm
        ) as mocks:
            # Create a dummy TPU.
            mocks["create_tpu"](cfg.name)
            # Issue start command.
            job._start()
            mocks["delete_tpu"].assert_called()
            mocks["create_tpu"].assert_called()
            # Bundling should happen if not on VM.
            self.assertEqual(not running_from_vm, job._bundler.bundle.called)

    def test_delete(self):
        cfg = _mock_config()
        job = cfg.set(command="").instantiate()

        mock_execute = mock.patch.object(job, "_execute_remote_cmd")
        mock_credentials = mock.patch.object(job, "_get_job_credentials")

        with mock_credentials, mock_tpu(tpu_runner.__name__) as mocks, mock_execute as mock_exec:
            # Create a dummy TPU.
            mocks["create_tpu"](cfg.name)
            job._delete()
            # Outputs should be copied. call_args get the args of the last call.
            self.assertIn("gsutil cp", mock_exec.call_args.args[0])
            self.assertIn(cfg.name, mocks["delete_tpu"].call_args.args)

    def test_get_status(self):
        mocks = [
            mock_tpu(tpu_runner.__name__),
            mock_gcp_settings(bundler.__name__, settings={"ttl_bucket": "ttl_bucket"}),
            _mock_credentials(),
        ]

        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory() as temp_dir:
            # Boilerplate to register multiple mocks at once.
            for m in mocks:
                stack.enter_context(m)

            cfg = _mock_config().set(output_dir=temp_dir, command="")
            job = cfg.instantiate()

            # TPUs haven't started yet (_start() not called).
            with mock_tpu_statuses(job, num_booted=0, statuses=["", ""], returncodes=[0, 0]):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.NOT_STARTED, job._get_status())

            # Start the TPU.
            job._start()

            # TPUs haven't booted yet (_start() called, but num_booted < num_vms).
            num_vms = job._num_workers()
            with mock_tpu_statuses(
                job, num_booted=num_vms - 1, statuses=["", ""], returncodes=[0, 0]
            ):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.NOT_STARTED, job._get_status())

            # TPUs have booted, but invalid status value.
            with mock_tpu_statuses(job, num_booted=num_vms, statuses=["", ""], returncodes=[0, 0]):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.UNKNOWN, job._get_status())

            # TPUs have booted, statuses are valid, but got non-zero exit code.
            with mock_tpu_statuses(
                job,
                num_booted=num_vms,
                statuses=["RUNNING", "RUNNING"],
                returncodes=[0, 1],
            ):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.UNKNOWN, job._get_status())

            # TPUs have booted, statuses are valid, and all statuses agree.
            with mock_tpu_statuses(
                job,
                num_booted=num_vms,
                statuses=["RUNNING", "RUNNING"],
                returncodes=[0, 0],
            ):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.RUNNING, job._get_status())

            # TPUs have booted, statuses are valid, but not all statuses agree.
            with mock_tpu_statuses(
                job,
                num_booted=num_vms,
                statuses=["RUNNING", "NOT_RUNNING"],
                returncodes=[0, 0],
            ):
                self.assertEqual(tpu_runner.TPURunnerJob.Status.UNKNOWN, job._get_status())

    def test_execute(self):
        cfg = _mock_config()
        job = cfg.set(command="").instantiate()

        @contextlib.contextmanager
        def mock_status(status):
            mocks = {
                "_get_status": mock.patch.object(
                    job, "_get_status", side_effect=[status, StopIteration()]
                ),
                "_delete": mock.patch.object(job, "_delete", return_value=None),
                "_start": mock.patch.object(job, "_start", return_value=None),
                "_run_command": mock.patch.object(job, "_run_command", return_value=None),
            }
            with contextlib.ExitStack() as stack:
                yield {name: stack.enter_context(patch) for name, patch in mocks.items()}

        try:
            with mock_status(tpu_runner.TPURunnerJob.Status.SUCCESS) as mocks:
                job._execute()
                mocks["_delete"].assert_called()

            with self.assertRaisesRegex(ValueError, "failed"), mock_status(
                tpu_runner.TPURunnerJob.Status.FAILED
            ) as mocks:
                job._execute()
                mocks["_delete"].assert_called()

            with mock_status(tpu_runner.TPURunnerJob.Status.NOT_STARTED) as mocks:
                job._execute()
                mocks["_start"].assert_called()

            with mock_status(tpu_runner.TPURunnerJob.Status.NOT_RUNNING) as mocks:
                job._execute()
                mocks["_run_command"].assert_called()
        except StopIteration:
            pass  # Expected.

    @parameterized.product(
        name=[None, "test-name"],
        output_dir=[None, "test-output"],
        bundler_spec=[None, "find_links=/custom/python/archives"],
    )
    def test_from_flags(self, name, output_dir, bundler_spec):
        if name is None and os.getenv("USER") is None:
            pytest.skip(reason="No USER in env.")

        # Construct flags.
        fv = flags.FlagValues()
        tpu_runner.launch_flags(flag_values=fv)
        argv = ["cli"]
        if name is not None:
            argv.append(f"--name={name}")
        if output_dir is not None:
            argv.append(f"--output_dir={output_dir}")
        if bundler_spec is not None:
            argv.append(f"--bundler_spec={bundler_spec}")

        # Parse argv.
        fv(argv)
        assert fv.name == name

        # Construct config.
        mock_settings = {"ttl_bucket": "ttl_bucket"}
        with mock_gcp_settings(tpu_runner.__name__, settings=mock_settings), mock_gcp_settings(
            bundler.__name__, settings=mock_settings
        ):
            cfg = tpu_runner.TPURunnerJob.from_flags(fv)

        # If name is not provided, there should be a default.
        if name is None:
            self.assertIsNotNone(cfg.name)
        else:
            self.assertEqual(name, cfg.name)

        # If output_dir is not provided, it should use the right name.
        if output_dir is None:
            self.assertEqual(f"gs://ttl_bucket/axlearn/jobs/{cfg.name}", cfg.output_dir)
        else:
            self.assertEqual(output_dir, cfg.output_dir)

        # If find_links is not provided, it should be a default.
        if bundler_spec is None:
            self.assertEqual(
                ["https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
                cfg.bundler.find_links,
            )
        else:
            self.assertEqual(
                [
                    "/custom/python/archives",
                    "https://storage.googleapis.com/jax-releases/libtpu_releases.html",
                ],
                cfg.bundler.find_links,
            )

        # It should be instantiable.
        cfg.set(command="").instantiate()


@contextlib.contextmanager
def _mock_job(running_from_vm: bool):
    mock_job = mock.MagicMock()
    mock_cfg = mock.MagicMock(**{"instantiate.return_value": mock_job})
    patch = mock.patch.object(tpu_runner.TPURunnerJob, "from_flags", return_value=mock_cfg)

    with mock_tpu(tpu_runner.__name__, running_from_vm=running_from_vm), patch:
        yield mock_job


class TPURunnerMainTest(TestWithTemporaryCWD):
    """Tests CLI entrypoint."""

    def test_launch_flags(self):
        fv = flags.FlagValues()
        tpu_runner.launch_flags(flag_values=fv)
        # Basic sanity check.
        self.assertEqual(fv["num_slices"].default, 1)

    @parameterized.parameters(True, False)
    def test_list(self, running_from_vm):
        # Test that list can be invoked without additional flags.
        with _mock_job(running_from_vm), _mock_credentials():
            tpu_runner.main(["cli", "list"])

    @parameterized.parameters(True, False)
    def test_stop(self, running_from_vm):
        # Test that stop can be invoked with just --name.
        with _mock_job(running_from_vm):
            tpu_runner.main(["cli", "stop", "--name=test"])

    @parameterized.parameters(True, False)
    def test_start(self, running_from_vm):
        fv = flags.FlagValues()
        tpu_runner.launch_flags(flag_values=fv)
        fv.mark_as_parsed()
        self.assertEqual(fv.bundler_type, bundler.GCSTarBundler.TYPE)

        with _mock_job(running_from_vm):
            with self.assertRaisesRegex(app.UsageError, "Invalid action"):
                tpu_runner.main(["cli"], flag_values=fv)

            with self.assertRaisesRegex(app.UsageError, "tpu_type is required"):
                tpu_runner.main(["cli", "start"], flag_values=fv)

            with self.assertRaisesRegex(app.UsageError, "Command is required"):
                fv.set_default("tpu_type", "v4-8")
                tpu_runner.main(["cli", "start"], flag_values=fv)

            fv.set_default("tpu_type", "v4-8")
            tpu_runner.main(["cli", "start", "--", "test_command"], flag_values=fv)
