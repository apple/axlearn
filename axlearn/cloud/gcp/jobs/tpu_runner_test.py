# Copyright © 2023 Apple Inc.

"""Tests TPU runner job."""

# pylint: disable=protected-access
import contextlib
import tempfile
from typing import Optional
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp import job as gcp_job
from axlearn.cloud.gcp.jobs import tpu_runner
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.cloud.gcp.tpu import TpuInfo
from axlearn.common.config import config_for_function, maybe_set_config
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
        "create_queued_tpu": mock_create_tpu,
        "get_queued_tpu_node": mock_get_tpu_node,
        "tpu_resource": mock_tpu_resource,
        "qrm_resource": mock_tpu_resource,
        "delete_queued_tpu": mock_delete_tpu,
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
    statuses: list[str],
    returncodes: list[int],
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
        get_queued_tpu_node_status=mock.Mock(side_effect=mock_get_tpu_node_status),
    )
    mock_ssh_status = mock.patch.object(
        job,
        "_execute_remote_cmd",
        side_effect=mock_execute_remote_cmd,
    )
    with mock_tpu_status, mock_ssh_status:
        yield


def _mock_config() -> tpu_runner.TPURunnerJob.Config:
    mock_bundler = mock.MagicMock()
    mock_bundler.install_command.return_value = "test_install"
    return tpu_runner.TPURunnerJob.default_config().set(
        name="test-name",
        output_dir="test_output",
        accelerator=gcp_job.AcceleratorConfig(
            instance_type="tpu-v4-8",
            num_replicas=2,
        ),
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
        dict(num_workers=1, tpu_type="v4-8", num_replicas=1),
        dict(num_workers=2, tpu_type="v4-8", num_replicas=2),
        dict(num_workers=2, tpu_type="v4-16", num_replicas=1),
        dict(num_workers=4, tpu_type="v4-16", num_replicas=2),
    )
    def test_num_workers(self, num_workers, tpu_type, num_replicas):
        cfg = _mock_config()
        cfg.accelerator.set(instance_type=f"tpu-{tpu_type}", num_replicas=num_replicas)
        job = cfg.instantiate()
        self.assertEqual(num_workers, job._num_workers())

    @parameterized.parameters(
        bundler.DockerBundler.default_config(),
        bundler.ArtifactRegistryBundler.default_config(),
        bundler.CloudBuildBundler.default_config(),
    )
    def test_wrap(self, bundler_cfg: bundler.BaseDockerBundler.Config):
        # Test compat with different docker bundler types.
        cfg: tpu_runner.TPURunnerJob.Config = _mock_config().set(
            bundler=bundler_cfg.set(
                image="test-image",
                repo="test-repo",
                dockerfile="test-dockerfile",
            ),
        )
        maybe_set_config(cfg.bundler, project="test-project")
        job: tpu_runner.TPURunnerJob = cfg.set(command="test-command").instantiate()
        cmd = job._wrap(cfg.command, env={"TEST_ENV": "123"})
        self.assertStartsWith(cmd, "TEST_ENV=123 docker run")
        self.assertIn("-e TEST_ENV", cmd)
        self.assertIn(cfg.command, cmd)

    @parameterized.parameters(
        [
            dict(
                running_from_vm=True,
                env={"BASTION_TIER": "0"},
                reserved_tpu_gcp_setting=None,
                expect_reserved=True,
                expected_label="reserved",
            ),
            dict(
                running_from_vm=True,
                env={"BASTION_TIER": "0"},
                reserved_tpu_gcp_setting=True,
                expect_reserved=True,
                expected_label="reserved",
            ),
            dict(
                running_from_vm=True,
                env={"BASTION_TIER": "0"},
                reserved_tpu_gcp_setting=False,
                expect_reserved=True,
                expected_label="reserved",
            ),
            dict(
                running_from_vm=True,
                env={"BASTION_TIER": "1"},
                reserved_tpu_gcp_setting=None,
                expect_reserved=False,
                expected_label="spot",
            ),
            dict(
                running_from_vm=True,
                env={"BASTION_TIER": "1"},
                reserved_tpu_gcp_setting=True,
                expect_reserved=False,
                expected_label="spot",
            ),
            dict(
                running_from_vm=True,
                env={"BASTION_TIER": "1"},
                reserved_tpu_gcp_setting=False,
                expect_reserved=False,
                expected_label="spot",
            ),
            dict(
                running_from_vm=False,
                reserved_tpu_gcp_setting=None,
                expect_reserved=None,
                expected_label="spot",
            ),
            dict(
                running_from_vm=False,
                reserved_tpu_gcp_setting=True,
                expect_reserved=True,
                expected_label="reserved",
            ),
            dict(
                running_from_vm=False,
                reserved_tpu_gcp_setting=False,
                expect_reserved=False,
                expected_label="spot",
            ),
        ]
    )
    def test_start(
        self,
        running_from_vm: bool,
        expect_reserved: bool,
        expected_label: str,
        env: Optional[dict] = None,
        reserved_tpu_gcp_setting: Optional[bool] = None,
    ):
        cfg = _mock_config()
        job = cfg.set(command="").instantiate()

        mock_execute = mock.patch.object(job, "_execute_remote_cmd")
        mock_credentials = mock.patch.object(job, "_get_job_credentials")
        mock_env = mock.patch.dict("os.environ", env or {})

        with (
            mock_env,
            mock_execute,
            mock_credentials,
            mock_gcp_settings(
                tpu_runner.__name__, settings={"reserved_tpu": reserved_tpu_gcp_setting}
            ),
            mock_tpu(tpu_runner.__name__, running_from_vm) as mocks,
        ):
            # Create a dummy TPU.
            mocks["create_queued_tpu"](cfg.name)
            # Issue start command.
            job._start()
            mocks["delete_queued_tpu"].assert_called()
            mocks["create_queued_tpu"].assert_called()
            # TPU should be created with the right reservation.
            self.assertEqual(expect_reserved, mocks["create_queued_tpu"].call_args[1]["reserved"])
            self.assertEqual(
                expected_label, mocks["create_queued_tpu"].call_args[1]["labels"]["bastion_tier"]
            )
            # Bundling should happen if not on VM.
            self.assertEqual(not running_from_vm, job._bundler.bundle.called)

    def test_delete(self):
        cfg = _mock_config()
        job = cfg.set(command="").instantiate()

        mock_execute = mock.patch.object(job, "_execute_remote_cmd")
        mock_credentials = mock.patch.object(job, "_get_job_credentials")

        with mock_credentials, mock_tpu(tpu_runner.__name__) as mocks, mock_execute as mock_exec:
            # Create a dummy TPU.
            mocks["create_queued_tpu"](cfg.name)
            job._delete()
            # Outputs should be copied. call_args get the args of the last call.
            self.assertIn("gsutil cp", mock_exec.call_args.args[0])
            self.assertIn(cfg.name, mocks["delete_queued_tpu"].call_args.args)

    def test_get_status(self):
        mocks = [
            mock_tpu(tpu_runner.__name__),
            mock_gcp_settings(bundler.__name__, settings={"ttl_bucket": "ttl_bucket"}),
            mock_gcp_settings(tpu_runner.__name__, settings={"reserved_tpu": True}),
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

            with (
                self.assertRaisesRegex(ValueError, "failed"),
                mock_status(tpu_runner.TPURunnerJob.Status.FAILED) as mocks,
            ):
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
        # Construct flags.
        fv = flags.FlagValues()
        tpu_runner.TPURunnerJob.define_flags(fv)
        argv = ["cli"]
        if name is not None:
            argv.append(f"--name={name}")
        if output_dir is not None:
            argv.append(f"--output_dir={output_dir}")
        if bundler_spec is not None:
            argv.append(f"--bundler_spec={bundler_spec}")

        # Parsing without instance_type should be OK, e.g. for help/list/stop.
        fv(argv)
        argv.append("--instance_type=tpu-v4-8")

        # Parse argv.
        fv(argv)
        self.assertEqual(fv.name, name)

        # Construct config.
        mock_settings = {"ttl_bucket": "ttl_bucket"}
        mock_generate_job_name = mock.patch(
            f"{tpu_runner.__name__}.generate_job_name", return_value="test-name"
        )
        with (
            mock_generate_job_name,
            mock_gcp_settings(tpu_runner.__name__, settings=mock_settings),
            mock_gcp_settings(bundler.__name__, settings=mock_settings),
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

    def test_define_flags(self):
        fv = flags.FlagValues()
        tpu_runner.TPURunnerJob.define_flags(fv)
        # Basic sanity check.
        self.assertEqual(fv["num_replicas"].default, 1)

    @parameterized.parameters(True, False)
    def test_list(self, running_from_vm):
        fv = flags.FlagValues()
        tpu_runner.TPURunnerJob.define_flags(fv)
        fv.mark_as_parsed()

        # Test that list can be invoked without additional flags.
        with _mock_job(running_from_vm), _mock_credentials():
            tpu_runner.main(["cli", "list"], flag_values=fv)

    @parameterized.parameters(True, False)
    def test_stop(self, running_from_vm):
        fv = flags.FlagValues()
        tpu_runner.TPURunnerJob.define_flags(fv)
        fv.set_default("name", "test")
        fv.mark_as_parsed()

        # Test that stop can be invoked with just --name.
        with _mock_job(running_from_vm):
            tpu_runner.main(["cli", "stop"], flag_values=fv)

    @parameterized.parameters(True, False)
    def test_start(self, running_from_vm):
        fv = flags.FlagValues()
        tpu_runner.TPURunnerJob.define_flags(fv)
        fv.mark_as_parsed()

        with _mock_job(running_from_vm):
            with self.assertRaisesRegex(app.UsageError, "Invalid action"):
                tpu_runner.main(["cli"], flag_values=fv)

            with self.assertRaisesRegex(app.UsageError, "instance_type is required"):
                tpu_runner.main(["cli", "start"], flag_values=fv)

            with self.assertRaisesRegex(app.UsageError, "Command is required"):
                fv.set_default("instance_type", "tpu-v4-8")
                tpu_runner.main(["cli", "start"], flag_values=fv)

            fv.set_default("instance_type", "tpu-v4-8")
            tpu_runner.main(["cli", "start", "--", "test_command"], flag_values=fv)
