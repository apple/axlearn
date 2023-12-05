# Copyright © 2023 Apple Inc.

"""Tests dataflow launch."""
import contextlib
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import DockerBundler
from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler
from axlearn.cloud.gcp.jobs import cpu_runner, dataflow
from axlearn.cloud.gcp.jobs.cpu_runner_test import mock_vm
from axlearn.cloud.gcp.jobs.dataflow import (
    DataflowJob,
    _docker_bundler_to_flags,
    launch_flags,
    main,
)
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.test_utils import TestWithTemporaryCWD


@contextlib.contextmanager
def _mock_gcp_settings():
    mock_settings = {
        "project": "test_project",
        "zone": "test_zone",
        "docker_repo": "test_repo",
        "default_dockerfile": "test_dockerfile",
        "ttl_bucket": "test_ttl_bucket",
        "service_account_email": "test_service_account",
        "subnetwork": "projects/test_project/regions/test_region/subnetworks/test_subnetwork",
    }
    mocks = [
        mock_gcp_settings(module.__name__, settings=mock_settings)
        for module in [dataflow, bundler, cpu_runner]
    ]
    with contextlib.ExitStack() as stack:
        for m in mocks:
            stack.enter_context(m)
        yield mock_settings


def _mock_flags():
    fv = flags.FlagValues()
    launch_flags(flag_values=fv)
    fv.mark_as_parsed()
    fv.set_default("bundler_spec", ["image=test_image"])
    fv.set_default("name", "test_name")
    return fv


def _mock_dataflow(module_name: str):
    return mock.patch.multiple(
        module_name,
        get_credentials=mock.MagicMock(),
        _dataflow_resource=mock.MagicMock(),
        _get_dataflow_jobs=mock.MagicMock(),
    )


class DataflowJobTest(TestWithTemporaryCWD):
    def test_from_flags_bundler(self):
        with _mock_gcp_settings():
            fv = _mock_flags()
            cfg = DataflowJob.from_flags(fv)

            # Ensure worker bundler is constructed properly.
            self.assertEqual(cfg.bundler.klass.TYPE, ArtifactRegistryBundler.TYPE)
            self.assertEqual("test_image", cfg.bundler.image)
            self.assertEqual("test_dockerfile", cfg.bundler.dockerfile)
            self.assertEqual("test_repo", cfg.bundler.repo)

    def test_dataflow_spec_from_flags(self):
        with _mock_gcp_settings() as settings:
            fv = _mock_flags()
            cfg = DataflowJob.from_flags(fv)
            # pylint: disable-next=protected-access
            dataflow_spec, _ = DataflowJob._dataflow_spec_from_flags(cfg, fv)

            # Test defaults.
            self.assertEqual("DataflowRunner", dataflow_spec["runner"])

            # Test specs read from settings.
            self.assertEqual(settings["project"], dataflow_spec["project"])
            self.assertEqual(settings["zone"], dataflow_spec["region"])
            self.assertEqual(
                settings["service_account_email"], dataflow_spec["service_account_email"]
            )
            self.assertEqual(
                f"{settings['docker_repo']}/test_image:test_name",
                dataflow_spec["sdk_container_image"],
            )
            self.assertEqual(
                f"gs://{settings['ttl_bucket']}/tmp/test_name/", dataflow_spec["temp_location"]
            )
            self.assertEqual(
                f"https://www.googleapis.com/compute/v1/{settings['subnetwork']}",
                dataflow_spec["subnetwork"],
            )

            # Test specs read from flags.
            self.assertEqual("test_name", dataflow_spec["job_name"])
            self.assertEqual(fv.vm_type, dataflow_spec["worker_machine_type"])

            # Test overridding specs (including a multi-flag)
            fv.set_default(
                "dataflow_spec",
                ["project=other_project", "temp_location=other_location", "experiments=exp1,exp2"],
            )
            cfg = DataflowJob.from_flags(fv)
            # pylint: disable-next=protected-access
            dataflow_spec, _ = DataflowJob._dataflow_spec_from_flags(cfg, fv)

            self.assertEqual("other_project", dataflow_spec["project"])
            self.assertEqual("other_location", dataflow_spec["temp_location"])
            self.assertEqual(["exp1", "exp2"], dataflow_spec["experiments"])

            # Check the final command.
            self.assertIn("--experiments=exp1 --experiments=exp2", cfg.command)


class UtilsTest(TestWithTemporaryCWD):
    """Tests util functions."""

    def test_docker_bundler_to_flags(self):
        cfg = DockerBundler.default_config().set(
            dockerfile="test_dockerfile",
            image="test_image",
            repo="test_repo",
            build_args={"a": "test", "b": 123},
        )
        self.assertEqual(
            [
                "--bundler_spec=a=test",
                "--bundler_spec=b=123",
                "--bundler_spec=dockerfile=test_dockerfile",
                "--bundler_spec=image=test_image",
                "--bundler_spec=platform=linux/amd64",
                "--bundler_spec=repo=test_repo",
                "--bundler_type=docker",
            ],
            sorted(_docker_bundler_to_flags(cfg)),
        )


@contextlib.contextmanager
def _mock_job(running_from_vm: bool, **kwargs):
    mock_job = mock.MagicMock()
    mock_cfg = mock.MagicMock(**{"instantiate.return_value": mock_job, **kwargs})
    patch = mock.patch.object(DataflowJob, "from_flags", return_value=mock_cfg)

    with mock_vm(cpu_runner.__name__, running_from_vm=running_from_vm), patch:
        yield mock_job


class DataflowMainTest(TestWithTemporaryCWD):
    """Tests CLI entrypoint."""

    def test_launch_flags(self):
        fv = flags.FlagValues()
        launch_flags(flag_values=fv)
        # Basic sanity check.
        self.assertEqual(fv["bundler_type"].default, ArtifactRegistryBundler.TYPE)
        self.assertEqual(fv["vm_type"].default, "n2-standard-2")

    @parameterized.parameters(True, False)
    def test_start(self, running_from_vm):
        # Test that command is required for 'start'.
        with _mock_job(running_from_vm):
            with self.assertRaisesRegex(app.UsageError, "Invalid action"):
                main(["cli"])

            with self.assertRaisesRegex(app.UsageError, "Command is required"):
                main(["cli", "start"])

        # Test that repo and image are required for 'start'.
        mock_cfg = {"bundler.repo": None, "bundler.image": None}
        with _mock_job(running_from_vm, **mock_cfg):
            with self.assertRaisesRegex(app.UsageError, "repo and image are required"):
                main(["cli", "start", "--", "cmd"])

        # Test success case.
        mock_cfg = {"bundler.repo": "a", "bundler.image": "b"}
        with _mock_job(running_from_vm, **mock_cfg):
            main(["cli", "start", "--", "cmd"])
