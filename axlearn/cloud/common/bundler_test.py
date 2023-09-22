# Copyright © 2023 Apple Inc.

"""Tests bundling utilities."""

import pathlib
import tarfile
import tempfile
from contextlib import contextmanager
from unittest import mock

import toml

from axlearn.cloud.common import bundler
from axlearn.cloud.common.bundler import BaseTarBundler, Bundler, DockerBundler, get_bundler_config
from axlearn.cloud.common.config import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG_FILE
from axlearn.cloud.common.config_test import create_default_config
from axlearn.common.test_utils import TestCase, TestWithTemporaryCWD


@contextmanager
def _fake_dockerfile():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dockerfile = pathlib.Path(temp_dir) / "FAKE_DOCKERFILE"
        temp_dockerfile.touch()
        yield temp_dockerfile


class BundlerTest(TestWithTemporaryCWD):
    """Tests Bundler."""

    def test_local_dir_context_config(self):
        b = Bundler.default_config().instantiate()

        # Fail if no config file exists.
        with self.assertRaises(SystemExit):
            # pylint: disable-next=protected-access
            with b._local_dir_context():
                pass

        # Create a dummy config.
        config_file = pathlib.Path(self._temp_root.name) / CONFIG_DIR / CONFIG_FILE
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.touch()

        # Ensure the config file is copied to temp bundle.
        # pylint: disable-next=protected-access
        with b._local_dir_context() as temp_bundle:
            self.assertTrue(
                (pathlib.Path(temp_bundle) / "axlearn" / CONFIG_DIR / CONFIG_FILE).exists()
            )


class RegistryTest(TestCase):
    """Tests retrieving bundlers from registry."""

    def test_get_docker_bundler(self):
        cfg = get_bundler_config(
            bundler_type=DockerBundler.TYPE,
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


class BaseTarBundlerTest(TestWithTemporaryCWD):
    """Tests BaseTarBundler."""

    def test_bundle(self):
        with tempfile.TemporaryDirectory() as remote_dir:
            # Create dummy file and requirements in CWD.
            for f in ["test1.txt", "test2.txt"]:
                dummy_file = pathlib.Path(self._temp_root.name) / f
                dummy_file.write_text("hello world")

            # Create a config file.
            configs = {"test_namespace": {"hello": 123}}
            create_default_config(self._temp_root.name, contents=configs)

            # Create the bundle.
            cfg = BaseTarBundler.default_config().set(
                remote_dir=remote_dir,
                exclude=["test1.txt"],
                extras=["test.whl", "dev"],
                find_links=["link1", "link2"],
            )
            b = cfg.instantiate()
            bundle_name = "test_bundle"
            bundle_id = b.bundle(bundle_name)

            # Check that bundle exists.
            self.assertTrue(pathlib.Path(bundle_id).exists())
            # Check that bundle includes the right files.
            with tarfile.open(bundle_id, "r") as tar:
                contents = tar.getnames()
                # test1 is excluded.
                self.assertNotIn("test1.txt", contents)
                self.assertIn("test2.txt", contents)
                self.assertIn(f"{CONFIG_DIR}/requirements.txt", contents)
                self.assertIn(f"{CONFIG_DIR}/{DEFAULT_CONFIG_FILE}", contents)

                # Make sure requirements has the right contents.
                f = tar.extractfile(f"{CONFIG_DIR}/requirements.txt")
                assert f is not None  # Explicit assert so pytype understands.
                with f:
                    # fmt: off
                    expected = (
                        "--find-links link1\n"
                        "--find-links link2\n"
                        "test.whl\n"
                        ".[dev]"
                    )
                    # fmt: on
                    self.assertEqual(expected, f.read().decode("utf-8"))

                # Make sure config has the right contents.
                f = tar.extractfile(f"{CONFIG_DIR}/{DEFAULT_CONFIG_FILE}")
                assert f is not None  # Explicit assert so pytype understands.
                with f:
                    self.assertEqual(toml.loads(f.read().decode("utf-8")), configs)


class DockerBundlerTest(TestWithTemporaryCWD):
    """Tests DockerBundler."""

    def test_required(self):
        cfg = DockerBundler.default_config().set(image="", repo="", dockerfile="")

        with self.assertRaisesRegex(ValueError, "image cannot be empty"):
            cfg.set(repo="test", image="", dockerfile="test")
            cfg.instantiate()

        with self.assertRaisesRegex(ValueError, "repo cannot be empty"):
            cfg.set(repo="", image="test", dockerfile="test")
            cfg.instantiate()

        with self.assertRaisesRegex(ValueError, "dockerfile cannot be empty"):
            cfg.set(dockerfile="", repo="test", image="test")
            cfg.instantiate()

        cfg.set(image="test", repo="test", dockerfile="test")
        cfg.instantiate()

    @mock.patch(f"{bundler.__name__}.running_from_source", return_value=True)
    @mock.patch(f"{bundler.__name__}.get_git_revision", return_value="FAKE_REVISION")
    @mock.patch(f"{bundler.__name__}.get_git_status", return_value=["FAKE_FILE"])
    def test_call_unclean(self, get_git_status, get_git_revision, running_from_source):
        # Create a dummy config.
        config_file = pathlib.Path(self._temp_root.name) / CONFIG_DIR / CONFIG_FILE
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.touch()

        with _fake_dockerfile() as dockerfile:
            b = (
                DockerBundler.default_config()
                .set(image="test", repo="FAKE_REPO", dockerfile=str(dockerfile))
                .instantiate()
            )
            with self.assertRaisesRegex(RuntimeError, "commit your changes"):
                b.bundle("FAKE_TAG")

        self.assertGreater(running_from_source.call_count, 0)
        self.assertGreater(get_git_status.call_count, 0)
        self.assertEqual(get_git_revision.call_count, 0)
