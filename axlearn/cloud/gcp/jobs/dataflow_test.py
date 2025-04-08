# Copyright © 2023 Apple Inc.

"""Tests dataflow launch."""

import contextlib
from typing import Optional, Type, cast
from unittest import mock

from absl import app, flags
from absl.testing import parameterized

from axlearn.cloud.common.bundler import BUNDLE_EXCLUDE, BaseDockerBundler, _bundlers
from axlearn.cloud.common.utils import canonicalize_to_string
from axlearn.cloud.gcp import bundler, job
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler
from axlearn.cloud.gcp.jobs import cpu_runner, dataflow
from axlearn.cloud.gcp.jobs.cpu_runner_test import mock_vm
from axlearn.cloud.gcp.jobs.dataflow import DataflowJob, _docker_bundler_to_flags, main
from axlearn.cloud.gcp.test_utils import default_mock_settings, mock_gcp_settings
from axlearn.common.config import maybe_set_config
from axlearn.common.test_utils import TestWithTemporaryCWD


@contextlib.contextmanager
def _mock_gcp_settings():
    mock_settings = default_mock_settings()
    mocks = [
        mock_gcp_settings(module.__name__, settings=mock_settings)
        for module in [dataflow, bundler, cpu_runner]
    ]
    mocks += [
        mock.patch(f"{job.__name__}.default_project", return_value=mock_settings["project"]),
        mock.patch(f"{job.__name__}.default_zone", return_value=mock_settings["zone"]),
    ]
    with contextlib.ExitStack() as stack:
        for m in mocks:
            stack.enter_context(m)
        yield mock_settings


def _mock_flags():
    fv = flags.FlagValues()
    DataflowJob.define_flags(fv)
    fv.mark_as_parsed()
    fv.bundler_spec = ["image=test_image"]
    fv.name = "test_name"
    return fv


class DataflowJobTest(TestWithTemporaryCWD):
    def test_from_flags_bundler(self):
        with _mock_gcp_settings():
            fv = _mock_flags()
            cfg = DataflowJob.from_flags(fv)

            # Ensure worker bundler is constructed properly.
            self.assertEqual(cfg.bundler.klass.TYPE, ArtifactRegistryBundler.TYPE)
            self.assertEqual("test_image", cfg.bundler.image)
            self.assertEqual("settings-dockerfile", cfg.bundler.dockerfile)
            self.assertEqual("settings-repo", cfg.bundler.repo)

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
            self.assertTrue(settings["zone"].startswith(dataflow_spec["region"]))
            self.assertEqual(
                settings["service_account_email"],
                dataflow_spec["service_account_email"],
            )
            self.assertEqual(
                f"{settings['docker_repo']}/test_image:test_name",
                dataflow_spec["sdk_container_image"],
            )
            self.assertEqual(
                f"gs://{settings['ttl_bucket']}/tmp/test_name/",
                dataflow_spec["temp_location"],
            )
            self.assertEqual(
                f"https://www.googleapis.com/compute/v1/{settings['subnetwork']}",
                dataflow_spec["subnetwork"],
            )

            # Test specs read from flags.
            self.assertEqual("test_name", dataflow_spec["job_name"])
            self.assertEqual(fv.vm_type, dataflow_spec["worker_machine_type"])

            # Test overriding specs (including a multi-flag)
            fv.dataflow_spec = [
                "project=other_project",
                "temp_location=other_location",
                "experiments=exp1,exp2",
            ]
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

    @parameterized.product(
        bundler_klass=[
            bundler_klass
            for bundler_klass in _bundlers.values()
            if issubclass(bundler_klass, BaseDockerBundler)
        ],
        private_worker_pool=[None, "test-pool"],
    )
    def test_docker_bundler_to_flags(
        self,
        bundler_klass: Type[BaseDockerBundler],
        private_worker_pool: Optional[str] = None,
    ):
        if bundler_klass is not CloudBuildBundler and private_worker_pool:
            return
        cfg = bundler_klass.default_config().set(
            image="test_image",
            repo="test_repo",
            dockerfile="test_dockerfile",
            build_args={"a": "test", "b": "123"},
            allow_dirty=True,
            cache_from=("cache1", "cache2"),
        )
        maybe_set_config(cfg, private_worker_pool=private_worker_pool)
        fv = flags.FlagValues()
        fv.mark_as_parsed()
        spec_flags = [
            "--bundler_spec=a=test",
            "--bundler_spec=allow_dirty=True",
            "--bundler_spec=b=123",
            "--bundler_spec=dockerfile=test_dockerfile",
            f"--bundler_spec=exclude={canonicalize_to_string(BUNDLE_EXCLUDE)}",
            "--bundler_spec=image=test_image",
            "--bundler_spec=platform=linux/amd64",
            "--bundler_spec=repo=test_repo",
            "--bundler_spec=cache_from=cache1,cache2",
        ]
        if bundler_klass is CloudBuildBundler:
            cfg = cast(CloudBuildBundler.Config, cfg)
            if cfg.private_worker_pool:
                spec_flags.append(f"--bundler_spec=private_worker_pool={cfg.private_worker_pool}")
            spec_flags.append(f"--bundler_spec=is_async={cfg.is_async}")

        all_flags = [f"--bundler_type={bundler_klass.TYPE}"] + spec_flags
        actual = _docker_bundler_to_flags(cfg, fv=fv)
        self.assertSameElements(all_flags, actual)

        re_cfg = bundler_klass.from_spec(
            [x.replace("--bundler_spec=", "") for x in spec_flags], fv=fv
        )
        for name, value in re_cfg.items():
            self.assertEqual(
                canonicalize_to_string(getattr(cfg, name, None)),
                canonicalize_to_string(value),
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

    def test_define_flags(self):
        fv = flags.FlagValues()
        DataflowJob.define_flags(fv)
        # Basic sanity check.
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
