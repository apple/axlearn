# Copyright © 2023 Apple Inc.

"""Tests bundling utilities."""

import contextlib
from unittest import mock

from absl.testing import parameterized

from axlearn.cloud.common.bundler import BaseDockerBundler, _bundlers, get_bundler_config
from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler, GCSTarBundler
from axlearn.cloud.gcp.cloud_build import CloudBuildStatus
from axlearn.cloud.gcp.test_utils import mock_gcp_settings
from axlearn.common.test_utils import TestCase


class RegistryTest(TestCase):
    """Tests retrieving bundlers from registry."""

    def test_get_gcs_bundler(self):
        # Test overriding settings.
        with mock_gcp_settings(bundler.__name__, settings={"ttl_bucket": "default_bucket"}):
            cfg = get_bundler_config(
                bundler_type=GCSTarBundler.TYPE,
                spec=[
                    "remote_dir=test_bucket",
                    # Make sure parent configs can be set from spec.
                    "external=test_external",
                ],
                fv=None,
            )
            self.assertEqual(cfg.remote_dir, "test_bucket")
            self.assertEqual(cfg.external, "test_external")

        # Test using defaults.
        with mock_gcp_settings(bundler.__name__, settings={"ttl_bucket": "default_bucket"}):
            cfg = get_bundler_config(
                bundler_type=GCSTarBundler.TYPE,
                spec=["external=test_external"],
                fv=None,
            )
            self.assertEqual(cfg.remote_dir, "gs://default_bucket/axlearn/jobs")
            self.assertEqual(cfg.external, "test_external")

    @parameterized.parameters(ArtifactRegistryBundler, CloudBuildBundler)
    def test_get_docker_bundler(self, bundler_cls):
        # Test without settings.
        with mock_gcp_settings(bundler.__name__, settings={}):
            cfg: BaseDockerBundler.Config = get_bundler_config(
                bundler_type=bundler_cls.TYPE,
                spec=[
                    "image=test_image",
                    "repo=test_repo",
                    "dockerfile=test_dockerfile",
                    "build_arg1=test_build_arg",
                    # Make sure parent configs can be set from spec.
                    "external=test_external",
                    "target=test_target",
                    "cache_from=test_cache1,test_cache2",
                ],
                fv=None,
            )
            self.assertEqual(cfg.image, "test_image")
            self.assertEqual(cfg.repo, "test_repo")
            self.assertEqual(cfg.dockerfile, "test_dockerfile")
            self.assertEqual(cfg.build_args, {"build_arg1": "test_build_arg"})
            self.assertEqual(cfg.external, "test_external")
            self.assertEqual(cfg.target, "test_target")
            self.assertEqual(cfg.cache_from, ["test_cache1", "test_cache2"])
            self.assertEqual(cfg.instantiate().TYPE, bundler_cls.TYPE)

        # Test with default settings.
        with mock_gcp_settings(
            bundler.__name__,
            settings={
                "project": "default_project",
                "docker_repo": "default_repo",
                "default_dockerfile": "default_dockerfile",
            },
        ):
            cfg = get_bundler_config(
                bundler_type=bundler_cls.TYPE,
                spec=[
                    "image=test_image",
                    "build_arg1=test_build_arg",
                    "external=test_external",
                    "target=test_target",
                ],
                fv=None,
            )
            self.assertEqual(cfg.image, "test_image")
            self.assertEqual(cfg.repo, "default_repo")
            self.assertEqual(cfg.dockerfile, "default_dockerfile")
            self.assertEqual(cfg.build_args, {"build_arg1": "test_build_arg"})
            self.assertEqual(cfg.external, "test_external")
            self.assertEqual(cfg.target, "test_target")
            if hasattr(cfg, "project"):
                self.assertEqual(cfg.project, "default_project")

    @parameterized.parameters([bundler_klass.TYPE for bundler_klass in _bundlers.values()])
    def test_with_tpu_extras(self, bundler_type):
        # Test configuring bundle for TPU.
        with mock_gcp_settings(bundler.__name__, settings={"ttl_bucket": "default_bucket"}):
            cfg = get_bundler_config(
                bundler_type=bundler_type, spec=["find_links=test", "extras=test"]
            )
            cfg = bundler.with_tpu_extras(cfg)
            if hasattr(cfg, "find_links"):
                self.assertSameElements(
                    [
                        "test",
                        "https://storage.googleapis.com/jax-releases/libtpu_releases.html",
                    ],
                    cfg.find_links,
                )
            self.assertSameElements(["tpu", "test"], cfg.extras)


class CloudBuildBundlerTest(TestCase):
    """Tests CloudBuildBundler."""

    @contextlib.contextmanager
    def _mock_status(self, *side_effect: tuple[CloudBuildStatus]):
        with (
            mock.patch("time.sleep"),
            mock.patch(
                f"{bundler.__name__}.get_cloud_build_status", side_effect=side_effect
            ) as mock_status,
        ):
            yield mock_status

    def _get_test_cloud_build_bundler(self) -> CloudBuildBundler.Config:
        return CloudBuildBundler.default_config().set(
            image="test-image",
            repo="test-repo",
            dockerfile="test-dockerfile",
            project="test-project",
        )

    def test_wait_until_finished_is_no_op_if_async_false(self):
        # Should be a no-op if is_async=False.
        cfg = self._get_test_cloud_build_bundler()

        with self._mock_status(None) as mock_status:
            b = cfg.set(is_async=False).instantiate()
            b.wait_until_finished("test-name")
            self.assertFalse(mock_status.called)

    def test_wait_until_finished_is_called_with_happy_status_path(self):
        # Tests happy path: transitions from no status -> pending -> success.
        cfg = self._get_test_cloud_build_bundler()

        with self._mock_status(
            None, CloudBuildStatus.PENDING, CloudBuildStatus.SUCCESS
        ) as mock_status:
            b = cfg.set(is_async=True).instantiate()
            b.wait_until_finished("test-name")
            self.assertEqual(3, mock_status.call_count)

    def test_wait_until_finished_raises_runtime_error_with_cloud_build_status_failure(self):
        # Tests that we raise a runtime error if CloudBuildStatus.FAILURE status is returned.
        cfg = self._get_test_cloud_build_bundler()

        with self._mock_status(
            None, CloudBuildStatus.PENDING, CloudBuildStatus.FAILURE
        ) as mock_status:
            b = cfg.set(is_async=True).instantiate()
            with self.assertRaisesRegex(
                RuntimeError, "CloudBuild for test-name failed: CloudBuildStatus.FAILURE"
            ):
                b.wait_until_finished("test-name")
                self.assertEqual(3, mock_status.call_count)

    def test_wait_until_finished_retries_with_runtime_error(self):
        # Tests that the query is retried if retrieving status fails with a RuntimeError.
        cfg = self._get_test_cloud_build_bundler()

        with self._mock_status(RuntimeError("fake error"), CloudBuildStatus.SUCCESS) as mock_status:
            b = cfg.set(is_async=True).instantiate()
            b.wait_until_finished("test-name")
            self.assertEqual(2, mock_status.call_count)
