# Copyright © 2023 Apple Inc.

"""Tests bundling utilities."""

from absl.testing import parameterized

from axlearn.cloud.common.bundler import get_bundler_config
from axlearn.cloud.gcp import bundler
from axlearn.cloud.gcp.bundler import ArtifactRegistryBundler, CloudBuildBundler, GCSTarBundler
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
            )
            self.assertEqual(cfg.remote_dir, "test_bucket")
            self.assertEqual(cfg.external, "test_external")

        # Test using defaults.
        with mock_gcp_settings(bundler.__name__, settings={"ttl_bucket": "default_bucket"}):
            cfg = get_bundler_config(
                bundler_type=GCSTarBundler.TYPE,
                spec=["external=test_external"],
            )
            self.assertEqual(cfg.remote_dir, "gs://default_bucket/axlearn/jobs")
            self.assertEqual(cfg.external, "test_external")

    @parameterized.parameters(ArtifactRegistryBundler, CloudBuildBundler)
    def test_get_docker_bundler(self, bundler_cls):
        # Test without settings.
        with mock_gcp_settings(bundler.__name__, settings={}):
            cfg = get_bundler_config(
                bundler_type=bundler_cls.TYPE,
                spec=[
                    "image=test_image",
                    "repo=test_repo",
                    "dockerfile=test_dockerfile",
                    "build_arg1=test_build_arg",
                    # Make sure parent configs can be set from spec.
                    "external=test_external",
                    "target=test_target",
                ],
            )
            self.assertEqual(cfg.image, "test_image")
            self.assertEqual(cfg.repo, "test_repo")
            self.assertEqual(cfg.dockerfile, "test_dockerfile")
            self.assertEqual(cfg.build_args, {"build_arg1": "test_build_arg"})
            self.assertEqual(cfg.external, "test_external")
            self.assertEqual(cfg.target, "test_target")

            self.assertEqual(cfg.instantiate().TYPE, bundler_cls.TYPE)

        # Test with default settings.
        with mock_gcp_settings(
            bundler.__name__,
            settings={"docker_repo": "default_repo", "default_dockerfile": "default_dockerfile"},
        ):
            cfg = get_bundler_config(
                bundler_type=bundler_cls.TYPE,
                spec=[
                    "image=test_image",
                    "build_arg1=test_build_arg",
                    "external=test_external",
                    "target=test_target",
                ],
            )
            self.assertEqual(cfg.image, "test_image")
            self.assertEqual(cfg.repo, "default_repo")
            self.assertEqual(cfg.dockerfile, "default_dockerfile")
            self.assertEqual(cfg.build_args, {"build_arg1": "test_build_arg"})
            self.assertEqual(cfg.external, "test_external")
            self.assertEqual(cfg.target, "test_target")
