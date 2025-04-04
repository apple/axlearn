# Copyright © 2023 Apple Inc.

"""Tests bundling utilities."""

import contextlib
import os
import pathlib
import tarfile
import tempfile
from contextlib import contextmanager
from unittest import mock

import toml
from absl import flags
from absl.testing import parameterized

from axlearn.cloud.common import bundler, config, git_summary
from axlearn.cloud.common.bundler import BaseTarBundler, Bundler, DockerBundler, get_bundler_config
from axlearn.cloud.common.config import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG_FILE
from axlearn.cloud.common.config_test import create_default_config
from axlearn.common.test_utils import TestCase, TestWithTemporaryCWD, temp_chdir

git_summary_path = f"{git_summary.__name__}.{git_summary.GitSummary.__name__}"


@contextmanager
def _fake_dockerfile():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dockerfile = pathlib.Path(temp_dir) / "FAKE_DOCKERFILE"
        temp_dockerfile.touch()
        yield temp_dockerfile


def _create_dummy_config(temp_dir: str):
    # Create a dummy config.
    config_file = pathlib.Path(temp_dir) / CONFIG_DIR / CONFIG_FILE
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.touch()
    return config_file


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
        dummy_config_file = _create_dummy_config(self._temp_root.name)

        # Ensure the config file is copied to temp bundle.
        # pylint: disable-next=protected-access
        with b._local_dir_context() as temp_bundle:
            self.assertTrue(
                (pathlib.Path(temp_bundle) / "axlearn" / CONFIG_DIR / CONFIG_FILE).exists()
            )

        # Test copying config file from a path which is not relative to the package dir.
        b2 = Bundler.default_config().instantiate()
        with tempfile.TemporaryDirectory() as temp_new_cwd:
            mock_search = mock.patch(
                f"{config.__name__}._config_search_paths",
                return_value=[str(dummy_config_file)],
            )
            # pylint: disable-next=protected-access
            with mock_search, temp_chdir(temp_new_cwd), b2._local_dir_context() as temp_bundle:
                # Ensure the config file is copied to temp bundle.
                self.assertTrue(
                    (pathlib.Path(temp_bundle) / "axlearn" / CONFIG_DIR / CONFIG_FILE).exists()
                )

    def test_local_dir_context_external(self):
        # Create a dummy config.
        _create_dummy_config(self._temp_root.name)

        with tempfile.TemporaryDirectory() as temp_dir:
            (pathlib.Path(temp_dir) / "a" / "b").mkdir(parents=True)
            for f in ["test1.txt", "a/b/d.txt"]:
                (pathlib.Path(temp_dir) / f).write_text("hello world")

            # Using --bundler_spec=external=/path/to/dir should copy the dir.
            b = Bundler.default_config().set(external=[os.path.join(temp_dir, "a")]).instantiate()
            # pylint: disable-next=protected-access
            with b._local_dir_context() as temp_bundle:
                temp_bundle = pathlib.Path(temp_bundle) / "axlearn"
                self.assertTrue((temp_bundle / "a").is_dir())
                self.assertTrue((temp_bundle / "a" / "b").is_dir())
                self.assertEqual("hello world", (temp_bundle / "a" / "b" / "d.txt").read_text())
                self.assertFalse((temp_bundle / "test1.txt").exists())

            # Using --bundler_spec=external=/path/to/dir/ should copy the dir contents.
            b = Bundler.default_config().set(external=[os.path.join(temp_dir, "a/")]).instantiate()
            # pylint: disable-next=protected-access
            with b._local_dir_context() as temp_bundle:
                temp_bundle = pathlib.Path(temp_bundle) / "axlearn"
                self.assertFalse((temp_bundle / "a").exists())
                self.assertTrue((temp_bundle / "b").is_dir())
                self.assertEqual("hello world", (temp_bundle / "b" / "d.txt").read_text())

    def test_package_dir(self):
        # Create a dummy config.
        _create_dummy_config(self._temp_root.name)

        temp_file = pathlib.Path(self._temp_root.name) / "test.txt"
        temp_file.write_text("hello world")

        b = Bundler.default_config().instantiate()
        # pylint: disable-next=protected-access
        with b._local_dir_context() as temp_bundle:
            temp_bundle = pathlib.Path(temp_bundle) / "axlearn"
            self.assertEqual("hello world", (temp_bundle / "test.txt").read_text())
            self.assertTrue((temp_bundle / CONFIG_DIR / CONFIG_FILE).exists())

        # Try excluding the temp dir itself.
        b2 = Bundler.default_config().set(exclude=["."]).instantiate()
        # pylint: disable-next=protected-access
        with b2._local_dir_context() as temp_bundle:
            temp_bundle = pathlib.Path(temp_bundle) / "axlearn"
            self.assertFalse((temp_bundle / "test.txt").exists())
            self.assertTrue((temp_bundle / CONFIG_DIR / CONFIG_FILE).exists())

        # Try excluding the temp dir itself, but include the external file.
        b2 = (
            Bundler.default_config()
            .set(
                exclude=["."],
                external=[str(temp_file)],
            )
            .instantiate()
        )
        # pylint: disable-next=protected-access
        with b2._local_dir_context() as temp_bundle:
            temp_bundle = pathlib.Path(temp_bundle) / "axlearn"
            self.assertEqual("hello world", (temp_bundle / "test.txt").read_text())
            self.assertTrue((temp_bundle / CONFIG_DIR / CONFIG_FILE).exists())


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
            fv=flags.FlagValues(),
        )
        self.assertEqual(cfg.image, "test_image")
        self.assertEqual(cfg.repo, "test_repo")
        self.assertEqual(cfg.dockerfile, "test_dockerfile")
        self.assertEqual(cfg.build_args, {"build_arg1": "test_build_arg"})
        self.assertEqual(cfg.external, "test_external")
        self.assertEqual(cfg.target, "test_target")


class BaseTarBundlerTest(TestWithTemporaryCWD):
    """Tests BaseTarBundler."""

    @parameterized.product(
        index_url=[None, "url1"],
    )
    def test_bundle(self, index_url):
        with tempfile.TemporaryDirectory() as remote_dir:
            # Create dummy file and requirements in CWD.
            (pathlib.Path(self._temp_root.name) / "a" / "b").mkdir(parents=True)
            (pathlib.Path(self._temp_root.name) / "a" / "f").mkdir(parents=True)
            for f in [
                "test1.txt",
                "test2.txt",
                "a/c.txt",
                "a/b/d.txt",
                "a/e.txt",
                "a/f/g.txt",
            ]:
                dummy_file = pathlib.Path(self._temp_root.name) / f
                dummy_file.write_text("hello world")

            # Create a config file.
            configs = {"test_namespace": {"hello": 123}}
            create_default_config(self._temp_root.name, contents=configs)

            # Create the bundle.
            cfg = BaseTarBundler.default_config().set(
                remote_dir=remote_dir,
                exclude=["test1.txt", "./a/b", "a/c.txt", "f"],
                extras=["test.whl", "dev"],
                find_links=["link1", "link2"],
                index_url=index_url,
            )
            b = cfg.instantiate()
            bundle_name = "test_bundle"

            (pathlib.Path(self._temp_root.name) / "pyproject.toml").touch()
            bundle_id = b.bundle(bundle_name)

            # Check that bundle exists.
            self.assertTrue(pathlib.Path(bundle_id).exists())
            # Check that bundle includes the right files.
            with tarfile.open(bundle_id, "r") as tar:
                contents = tar.getnames()
                # test1 is excluded.
                self.assertNotIn("test1.txt", contents)
                self.assertNotIn("a/b", contents)
                self.assertNotIn("a/b/d.txt", contents)
                self.assertNotIn("a/c.txt", contents)
                self.assertNotIn("a/f/g.txt", contents)
                self.assertIn("test2.txt", contents)
                self.assertIn("a/e.txt", contents)
                self.assertIn("a", contents)
                self.assertIn(f"{CONFIG_DIR}/requirements.txt", contents)
                self.assertIn(f"{CONFIG_DIR}/{DEFAULT_CONFIG_FILE}", contents)

                # Make sure requirements has the right contents.
                f = tar.extractfile(f"{CONFIG_DIR}/requirements.txt")
                assert f is not None  # Explicit assert so pytype understands.
                expected_index_url = f"--index-url {index_url}\n" if index_url else ""
                with f:
                    # fmt: off
                    expected = (
                        "--find-links link1\n"
                        "--find-links link2\n"
                        f"{expected_index_url}"
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

    @parameterized.product(
        platform=[None, "test-platform"],
        target=[None, "test-target"],
    )
    def test_build_and_push(self, platform, target):
        def mock_build_fn(**kwargs):
            # All build args should be strings.
            self.assertTrue(all(isinstance(x, str) for x in kwargs["args"].values()))
            self.assertEqual(kwargs["target"], target)
            self.assertEqual(kwargs["platform"], platform)

        mock_build = mock.Mock(side_effect=mock_build_fn)
        with mock.patch.multiple(
            bundler.__name__,
            docker_push=mock.Mock(return_value=123),
            docker_build=mock_build,
        ), mock.patch.multiple(
            f"{git_summary.__name__}.{git_summary.GitSummary.__name__}",
            is_valid=mock.Mock(return_value=False),
        ):
            _create_dummy_config(self._temp_root.name)

            # Ensure that docker bundler works whether build args are specified as strings or lists.
            build_args = dict(a="a,b", b=("a", "b"), c=["a", "b"])

            with _fake_dockerfile() as dockerfile:
                cfg = DockerBundler.default_config().set(
                    image="test",
                    repo="FAKE_REPO",
                    dockerfile=str(dockerfile),
                    build_args=build_args,
                    platform=platform,
                    target=target,
                )
                b = cfg.instantiate()
                self.assertEqual(b.bundle("FAKE_TAG"), 123)
                self.assertEqual(mock_build.call_count, 1)

                # Test skip bundle.
                b = cfg.set(skip_bundle=True).instantiate()
                self.assertEqual(b.bundle("FAKE_TAG"), "FAKE_REPO/test:FAKE_TAG")
                # Build should not be called.
                self.assertEqual(mock_build.call_count, 1)

    @parameterized.product(running_from_source=[True, False], allow_dirty=[False, True])
    @mock.patch(
        f"{git_summary_path}.to_labels",
        return_value={"fake-kabel": "fake-label-value"},
    )
    @mock.patch(f"{git_summary_path}.to_disk", return_value=[])
    def test_call_unclean(
        self,
        to_disk,
        to_labels,
        running_from_source,
        allow_dirty,
    ):
        _create_dummy_config(self._temp_root.name)

        mock_running_from_source = mock.patch(
            f"{git_summary_path}.is_valid",
            return_value=running_from_source,
        )
        mock_is_dirty = mock.patch(f"{git_summary_path}.is_dirty", return_value=True)
        with (
            mock_running_from_source as mock_source,
            mock_is_dirty as mock_dirty,
            _fake_dockerfile() as dockerfile,
        ):
            b = (
                DockerBundler.default_config()
                .set(
                    image="test",
                    repo="FAKE_REPO",
                    dockerfile=str(dockerfile),
                    allow_dirty=allow_dirty,
                )
                .instantiate()
            )
            should_raise = running_from_source and not allow_dirty

            if should_raise:
                ctx = self.assertRaisesRegex(RuntimeError, "commit your changes")
            else:
                ctx = contextlib.nullcontext()

            mock_build_and_push = mock.patch.object(b, "_build_and_push", return_value=None)
            with ctx, mock_build_and_push as mock_push:
                b.bundle("FAKE_TAG")
                self.assertEqual(not should_raise, mock_push.called)

            self.assertGreater(mock_source.call_count, 0)
            self.assertEqual(running_from_source, mock_dirty.called)
            self.assertEqual(running_from_source and allow_dirty, to_disk.called)
            self.assertEqual(running_from_source and allow_dirty, to_labels.called)
