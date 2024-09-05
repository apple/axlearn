# Copyright © 2023 Apple Inc.

"""Code bundling utilities.

The type of bundler to use is determined by `--bundler_type`. Bundlers can also be configured via
`--bundler_spec` flags; see the corresponding bundler class' `from_spec` method for details.

Example (docker):

    # Docker build and push to repo.
    # Note: assumes that you have already auth'ed to registry.
    axlearn cloud bundle --bundler_type=docker \
        --name=my-tag \
        --bundler_spec=image=my-image \
        --bundler_spec=repo=my-repo \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=build_arg1=my-build-arg

    # Bundlers support packaging external directories (besides just CWD) via the `external` spec.
    # For example, you can copy a specific file into the bundle root:
    axlearn cloud bundle ... --bundler_spec=external=/path/to/file.txt

    # When pointing `external` to a directory, bundling behaves similarly to the linux `cp` command:
    # specifying a trailing slash copies directory contents, and omitting a trailing slash copies
    # the directory itself.
    #
    # As an example, suppose we have a directory like:
    # external_dir/
    # -- my_file.txt

    # The following command produces a structure like:
    # bundle_root/
    # -- external_dir/
    axlearn cloud bundle ... --bundler_spec=external=/path/to/external_dir

    # The following command produces a structure like:
    # bundle_root/
    # -- my_file.txt
    axlearn cloud bundle ... --bundler_spec=external=/path/to/external_dir/

"""

import os
import pathlib
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from typing import Optional, TypeVar, Union
from urllib.parse import urlparse

import prefixed
from absl import app, flags, logging
from tensorflow import io as tf_io

from axlearn.cloud.common import config
from axlearn.cloud.common.docker import build as docker_build
from axlearn.cloud.common.docker import push as docker_push
from axlearn.cloud.common.utils import (
    canonicalize_to_list,
    canonicalize_to_string,
    copy_blobs,
    get_git_branch,
    get_git_revision,
    get_git_status,
    get_pyproject_version,
    parse_kv_flags,
    running_from_source,
)
from axlearn.common.config import REQUIRED, Configurable, Required, config_class

BUNDLE_EXCLUDE = [
    # Each entry below specifies a subdir/file name or a relative path from the src dir whose
    # contents should be excluded.
    "venv",
    ".git",
    ".idea",
    ".cache",
    ".pytest_cache",
    ".pytype",
    "__pycache__",
    ".ruff_cache",
    ".DS_Store",
]
FLAGS = flags.FLAGS
_DEFAULT_DOCKER_PLATFORM = "linux/amd64"


class Bundler(Configurable):
    """The base class of a bundler."""

    # To appease pytype.
    TYPE = None

    @config_class
    class Config(Configurable.Config):
        """Configures Bundler."""

        # Extra dependencies to include in the bundle, either specified as a comma separated string
        # or a sequence of strings.
        # Each string can be a pyproject section or a path to a wheel (relative to bundle root).
        extras: Optional[Union[str, Sequence[str]]] = None
        # File/directory names to exclude from the bundle.
        # These are matched against all levels of the directory tree, i.e., a subdirectory whose
        # name appears in the exclude list will also be excluded, even if the full path from root is
        # not in the list.
        exclude: Union[str, Sequence[str]] = BUNDLE_EXCLUDE
        # File/directory paths outside of CWD to include in the bundle, either specified as a comma
        # separated string or a sequence of strings.
        # For example, one can include custom pip wheels as part of the tarball.
        # The `exclude` rules also apply to these directories.
        external: Optional[Union[str, Sequence[str]]] = None

    def _local_dir_context(self) -> tempfile.TemporaryDirectory:
        """Copies contents of local directory to `target_dir`, excluding `exclude` paths,
        and returns the directory.

        Caller is expected to use as a context manager to ensure proper cleanup.

        Returns:
            The temporary directory.
        """
        cfg: Bundler.Config = self.config
        config_file, configs = config.load_configs(required=True)
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        exclude_paths = set(canonicalize_to_list(cfg.exclude))

        def copytree(src: pathlib.Path, dst: pathlib.Path, exclude: Iterable[str], root=None):
            """Recursively copies `src` to `dst`.

            Args:
                src: A path to a file or directory.
                dst: A path to a file or directory.
                exclude: Patterns to exclude.
                root: The `src` of the original  call to `copytree` before recursing.
                      I.e., the root of the tree being copied.
            """
            if root is None:
                root = src
            for s in src.glob("*"):
                d = dst / s.name
                relative_s = s.relative_to(root)
                # 1. Check if the final component of the path `s` is in exclude (which since we
                # are walking the tree recursively, ultimately checks whether any component
                # of the path is in exclude)
                # 2. Check if the exclusion pattern specifies a parent path of `relative_s`,
                # where both paths are relative to `root`. (`relative_s` is considered
                # a parent of itself.)
                if s.name in exclude or any(relative_s.is_relative_to(e) for e in exclude):
                    continue
                if s.is_dir():
                    d.mkdir()
                    copytree(s, d, exclude, root)
                else:
                    shutil.copy2(s, d, follow_symlinks=True)

        # Copy local dir except exclude list to temporary directory.
        package_dir = pathlib.Path.cwd()
        temp_root = pathlib.Path(temp_dir.name) / "axlearn"
        temp_root.mkdir()

        logging.info("Packaging %s.", package_dir)
        copytree(package_dir, temp_root, exclude_paths)

        # Copy any external files/dirs.
        for dep in canonicalize_to_list(cfg.external):
            dep = dep.strip()
            logging.info("Packaging external path %s.", dep)
            if urlparse(dep).scheme:
                # Has scheme so we try to use copy_blobs to handle.
                copy_blobs(dep, to_prefix=temp_root.as_posix())
                continue
            # We rely on shutil for local-to-local copy as it follows symlinks.
            dep_src = pathlib.Path(dep.strip())
            if dep_src.is_file():
                shutil.copy2(dep_src, temp_root / dep_src.name, follow_symlinks=True)
            else:
                # If the external dir ends with /, copy only the contents of the directory.
                # Otherwise, copy the directory itself. This is similar to linux `cp`.
                dep_dst = temp_root
                if not dep.endswith("/"):
                    dep_dst = dep_dst / dep_src.name
                    dep_dst.mkdir()
                copytree(dep_src, dep_dst, exclude_paths)

        # Copy the configs to the bundle directory, since the config file(s) may not be in cwd.
        # Note that configs may comprise of multiple config files, so we serialize the full configs
        # to a new file in the search paths, instead of copying config_file.
        # We also always copy to a standard path `CONFIG_DIR / CONFIG_FILE` within the bundle.
        # Note that this may override the existing config if it originated from `cwd / CONFIG_DIR /
        # CONFIG_FILE` in the first place (but this new config will have merged the original).
        rel_config_file = temp_root / config.CONFIG_DIR / config.CONFIG_FILE
        rel_config_file.parent.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Copying the config file from %s to the package under %s.",
            config_file,
            rel_config_file,
        )
        config.write_configs_with_header(str(rel_config_file), configs)

        dir_size = sum(f.stat().st_size for f in temp_root.glob("**/*"))
        dir_size = f"{prefixed.Float(dir_size):!.2k}B"
        logging.info("Uncompressed size: %s", dir_size)

        return temp_dir

    @classmethod
    def from_spec(cls, spec: list[str], *, fv: Optional[flags.FlagValues]) -> Config:
        """Converts a spec to a bundler."""
        raise NotImplementedError(cls)

    def id(self, name: str) -> str:
        """Returns a unique identifier for the bundle."""
        raise NotImplementedError(type(self))

    # pylint: disable-next=arguments-differ
    def bundle(self, name: str) -> str:
        """Produces a code bundle for the local directory.

        Args:
            name: Bundle name.

        Returns:
            The bundle identifier.
        """
        raise NotImplementedError(type(self))

    def install_command(self, bundle_id: str) -> str:
        """Emits a command to install the bundle.

        Can be executed by user via shell, passed to subprocess.run, or used in remote job
        execution.

        Args:
            bundle_id: ID as produced by `id`. Can be a gs:// path, docker image, etc. depending on
                bundler implementation.

        Returns:
            The command to install the bundle.
        """
        raise NotImplementedError(type(self))

    def wait_until_finished(self, name: str):
        """Blocks until bundling has completed.

        This can be used in cases where `bundle()` is async.

        Args:
            name: Bundle name.
        """
        pass


_bundlers: dict[str, type[Bundler]] = {}
T = TypeVar("T")


def register_bundler(cls: T) -> T:
    """Registers a bundler class for `get_bundler_config`."""
    _bundlers[cls.TYPE] = cls
    return cls


class BaseDockerBundler(Bundler):
    """Bundles local directory into a Docker container.

    Subclasses should implement `_build_and_push` -- see `CloudBuildBundler` as an example.
    """

    @config_class
    class Config(Bundler.Config):
        """Configures BaseDockerBundler."""

        # Image name.
        image: Required[str] = REQUIRED
        # Docker repo to push to. Must be writable by the user.
        repo: Required[str] = REQUIRED
        # Path to Dockerfile; either an absolute path, or a path relative to CWD.
        dockerfile: Required[str] = REQUIRED
        # Docker build args.
        build_args: dict[str, str] = {}
        # Build target.
        target: Optional[str] = None
        # Build target platform.
        # Usually the image is to be run on the cloud on x86 machines, so "linux/amd64" is the
        # default even on arm64 machines like Apple Silicon. If None, let docker pick the platform.
        platform: Optional[str] = _DEFAULT_DOCKER_PLATFORM
        # Allow git status to be dirty if bundling from source. This is sometimes necessary, e.g.
        # during bundling-time modifications of files.
        allow_dirty: bool = False
        # Additional image(s) to cache from.
        cache_from: Optional[Sequence[str]] = None
        # Skip the build + push step (e.g., using a pre-built image).
        skip_bundle: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config

        if not cfg.image:
            raise ValueError(
                "image cannot be empty. Please provide one via --bundler_spec=image=my-image."
            )
        if not cfg.repo:
            raise ValueError(
                "repo cannot be empty. Either configure docker_repo in settings, "
                "or provide one explicitly via --bundler_spec=repo=my-repo."
            )
        if not cfg.dockerfile:
            raise ValueError(
                "dockerfile cannot be empty. Either configure default_dockerfile in settings, "
                "or provide one explicitly via --bundler_spec=dockerfile=/path/to/Dockerfile."
            )

    @classmethod
    def from_spec(cls, spec: list[str], *, fv: Optional[flags.FlagValues]) -> Config:
        """Converts a spec to a bundler.

        Possible options:
        - image: The image name.
        - repo: The docker repo.
        - dockerfile: The Dockerfile path relative to project root.
        - target: The build target.
        - platform: The image target platform.
        - allow_dirty: Whether to ignore dirty git status.
        - cache_from: A comma-separated list of cache sources.
        - skip_bundle: Whether to skip the build + push. This option is intended to be used when an
            image has already been pre-built offline, in which case we may still want to leverage
            the install commands implemented by the bundler.

        All other specs are treated as build args.
        """
        del fv  # Not used.
        cfg: BaseDockerBundler.Config = cls.default_config()
        kwargs = parse_kv_flags(spec, delimiter="=")
        cache_from = canonicalize_to_list(kwargs.pop("cache_from", None))
        # Non-config specs are treated as build args.
        build_args = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in cfg}
        return cfg.set(build_args=build_args, cache_from=cache_from, **kwargs)

    # pylint: disable-next=arguments-renamed
    def id(self, tag: str) -> str:
        """Returns the full image identifier from the tag."""
        cfg: BaseDockerBundler.Config = self.config
        return f"{cfg.repo}/{cfg.image}:{tag}"

    # pylint: disable-next=arguments-renamed
    def bundle(self, tag: str) -> str:
        """Docker builds and pushes the local directory.

        Args:
            tag: The docker image tag.

        Returns:
            The full identifier of the remote image, in the format {repo}/{image}:{tag}.

        Raises:
            ValueError: If image, repo, or dockerfile are invalid.
            RuntimeError: If attempting to bundle with dirty git status.
        """
        cfg: BaseDockerBundler.Config = self.config
        if cfg.skip_bundle:
            bundle_id = self.id(tag)
            logging.info("Skipping build + push and using: %s.", bundle_id)
            return bundle_id

        # Fail early if git status is dirty.
        if running_from_source() and (status := get_git_status()):
            if cfg.allow_dirty:
                logging.warning("Bundling with local changes:\n%s", status)
            else:
                raise RuntimeError("Please commit your changes or gitignore them.")

        # If path is relative, assume it is relative to CWD.
        dockerfile = pathlib.Path(cfg.dockerfile).expanduser()
        if not dockerfile.is_absolute():
            dockerfile = (os.getcwd() / dockerfile).resolve()
        assert dockerfile.is_absolute(), dockerfile

        with self._local_dir_context() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            temp_root = temp_dir / "axlearn"

            # If the builder is invoked outside of the project, the Dockerfile may not exist in CWD
            # (hence temp_root). We explicitly copy the Dockerfile into the build context. Note that
            # dockerfile is an absolute path, and we are copying it into temp_root. It should
            # not really matter where we place it, as long as docker_build points to the right path.
            temp_dockerfile_path = temp_root / dockerfile.name
            shutil.copy2(dockerfile, temp_dockerfile_path)

            # Label with package version.
            labels = dict(version=get_pyproject_version())

            # If running from source, also label with git metadata.
            if running_from_source():
                labels.update(
                    {
                        "git-branch": get_git_branch(),
                        "git-commit-head": get_git_revision("HEAD"),
                    }
                )

            build_args = {**cfg.build_args}
            if cfg.extras:
                build_args["extras"] = cfg.extras
            # Ensure that build args are specified strings.
            build_args = {k: canonicalize_to_string(v) for k, v in build_args.items()}
            bundle_path = self._build_and_push(
                dockerfile=str(temp_dockerfile_path),
                image=self.id(tag),
                args=build_args,
                context=str(temp_root),
                labels=labels,
            )
        return bundle_path

    def install_command(self, bundle_id: str) -> str:
        """Emits a command to install the bundle.

        Args:
            bundle_id: Full docker image identifier, including the remote repo.

        Returns:
            The command to install the bundle.
        """
        return f"docker pull {bundle_id}"

    # pylint: disable-next=no-self-use
    def _build_and_push(
        self,
        *,
        dockerfile: str,
        image: str,
        args: dict[str, str],
        context: str,
        labels: dict[str, str],
    ) -> str:
        """Builds and pushes the docker image.

        Args:
            dockerfile: The full path to the Dockerfile in the temporary build context.
            image: The full image tag.
            args: Docker build args, e.g. as supplied via `--bundler_spec`.
            context: The full path to the temporary build context.
            labels: Docker labels.

        Returns:
            The full image tag of the built image. Will be returned from `bundle` as the bundle ID.
        """
        raise NotImplementedError(type(self))


@register_bundler
class DockerBundler(BaseDockerBundler):
    """A bundler that uses docker."""

    TYPE = "docker"

    # pylint: disable-next=no-self-use
    def _build_and_push(
        self,
        *,
        dockerfile: str,
        image: str,
        args: dict[str, str],
        context: str,
        labels: dict[str, str],
    ) -> str:
        cfg: DockerBundler.Config = self.config
        return docker_push(
            docker_build(
                dockerfile=dockerfile,
                image=image,
                args=args,
                context=context,
                target=cfg.target,
                labels=labels,
                platform=cfg.platform,
                cache_from=cfg.cache_from,
            )
        )


class BaseTarBundler(Bundler):
    """Bundles local directory into a tarball and ships to remote storage.

    Subclasses should implement `_copy_to_local_command` -- see `GCSTarBundler` as an example.
    """

    @config_class
    class Config(Bundler.Config):
        """Configures TarBundler."""

        # Remote directory to upload bundle to.
        remote_dir: Required[str] = REQUIRED
        # Optional list of --find-links to use in pip install.
        find_links: Optional[Union[str, Sequence[str]]] = None
        # Optional --index-url to use in pip install.
        index_url: Optional[str] = None
        # Whether to install in editable mode.
        editable: bool = False

    @classmethod
    def from_spec(cls, spec: list[str], *, fv: Optional[flags.FlagValues]) -> Config:
        """Converts a spec to a bundler.

        Possible options:
        - remote_dir: The remote directory to copy the bundle to. Must be compatible with tf_io.
        """
        del fv  # Not used.
        return cls.default_config().set(**parse_kv_flags(spec, delimiter="="))

    def id(self, name: str) -> str:
        """Returns the full image identifier from the tag."""
        # TODO(markblee): Consider directly uploading as `{name}.tar.gz`.
        return f"{self.config.remote_dir}/{name}/axlearn.tar.gz"

    def bundle(self, name: str) -> str:
        """Bundles and copies the bundler to `cfg.remote_dir`.

        Args:
            name: The directory name to copy the tar bundle to. This will be namespaced under
                `{cfg.remote_dir}/{name}/axlearn.tar.gz`.

        Returns:
            The remote path.
        """
        cfg: BaseTarBundler.Config = self.config

        with self._local_dir_context() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            temp_root = temp_dir / "axlearn"

            # Tar bundling installs via `pip install`, which requires a pyproject or setup.py.
            if not ((temp_root / "pyproject.toml").exists() or (temp_root / "setup.py").exists()):
                logging.warning(
                    "No pyproject.toml or setup.py found in the bundle root -- "
                    "This means that bundle installation will likely fail!"
                )

            # Add a requirements file indicating which deps/extras to install. This allows
            # install_command() to know how to install the bundle given just the bundle_id, without
            # having to know which extras the bundler was configured with when bundle() was called.
            # This is similar to docker bundling the deps at build time rather than install time.
            requirements = temp_root / config.CONFIG_DIR / "requirements.txt"
            requirements.parent.mkdir(parents=True, exist_ok=True)
            with requirements.open("w", encoding="utf-8") as f:
                for find_links in canonicalize_to_list(cfg.find_links):
                    f.write(f"--find-links {find_links}\n")
                if cfg.index_url:
                    f.write(f"--index-url {cfg.index_url}\n")
                pyproject_extras = []
                for extra in canonicalize_to_list(cfg.extras):
                    # NOTE: .whl can also end with a pyproject section, e.g. axlearn.whl[dev].
                    if ".whl" in extra:
                        f.write(f"{extra}\n")
                    else:
                        pyproject_extras.append(extra)
                deps = f".[{','.join(pyproject_extras)}]" if pyproject_extras else "."
                f.write(f"-e {deps}" if cfg.editable else deps)

            # Tar it up.
            tar_path = temp_dir / "axlearn.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                for f in temp_root.glob("*"):
                    tar.add(f, arcname=f.name)
            tar_size = f"{prefixed.Float(tar_path.stat().st_size):!.2k}B"
            logging.info("Compressed size: %s", tar_size)

            # Upload to remote.
            remote_path = self.id(name)
            logging.info("Submitting to %s.", remote_path)
            self._copy_to_remote(local_path=str(tar_path), remote_path=remote_path)

        return remote_path

    def _copy_to_remote(self, *, local_path: str, remote_path: str):
        if tf_io.gfile.exists(remote_path):
            logging.info("Overwriting existing bundle at %s", remote_path)
        else:
            logging.info("Uploading bundle to: %s", remote_path)
        tf_io.gfile.makedirs(os.path.dirname(remote_path))
        tf_io.gfile.copy(local_path, remote_path, overwrite=True)

    def _copy_to_local_command(self, *, remote_bundle_id: str, local_bundle_id: str) -> str:
        """Emits a command to copy a bundle from remote to local.

        By default, we make no assumptions about what's already installed, as the command can be
        executed on arbitrary compute environments.

        Args:
            remote_bundle_id: The remote bundle path to copy from.
            local_bundle_id: The local bundle path to copy to.

        Returns:
            The command to perform the copy.
        """
        raise NotImplementedError(type(self))

    def install_command(self, bundle_id: str) -> str:
        """Emits a command to install the bundle.

        Args:
            bundle_id: Path to tarball.

        Returns:
            The command to install the bundle.
        """
        copy_cmd = self._copy_to_local_command(
            remote_bundle_id=bundle_id, local_bundle_id="axlearn.tar.gz"
        )
        pip_install_cmd = (
            f"if [[ -f {config.CONFIG_DIR}/requirements.txt ]]; then "
            f"python3 -m pip install -r {config.CONFIG_DIR}/requirements.txt; "
            "else python3 -m pip install .; fi"
        )
        return (
            f"{copy_cmd} && tar -xzf axlearn.tar.gz && "
            f"python3 -m pip install --upgrade pip && {pip_install_cmd}"
        )


def get_bundler_config(
    *,
    bundler_type: str,
    spec: list[str],
    fv: Optional[flags.FlagValues] = None,
) -> Bundler.Config:
    """Constructs a bundler config from the given spec.

    Bundlers must be registered via `register_bundler`.

    Args:
        bundler_type: Type of bundler class.
        spec: Bundler specs. See the corresponding `from_spec` method of the bundler class.
        fv: The flag values.

    Returns:
        The bundler config.
    """
    if bundler_class := _bundlers.get(bundler_type, None):
        return bundler_class.from_spec(spec, fv=fv)
    raise NotImplementedError(
        f"Unknown bundler type: {bundler_type}. "
        f"Supported types are {sorted(list(_bundlers.keys()))}"
    )


def bundler_flags(required: bool = True, **kwargs):
    """Common bundler flags. Keyword args will be forwarded to flag definitions."""

    flags.DEFINE_string("bundler_type", None, "Bundler type.", required=required, **kwargs)
    flags.DEFINE_multi_string(
        "bundler_spec",
        [],
        "Bundler spec provided as key=value. "
        "Refer to each bundler's `from_spec` method docstring for details.",
        **kwargs,
    )
    flags.DEFINE_multi_string(
        "bundler_exclude",
        BUNDLE_EXCLUDE,
        "Files/folders in root to exclude from code bundle.",
        **kwargs,
    )


def main_flags():
    """Bundler flags required for `main`."""
    bundler_flags()
    # This flag is only used if invoking the bundler CLI, so we leave it out of `bundler_flags`.
    flags.DEFINE_string(
        "name",
        None,
        "Bundle name. If None, a unique name is generated.",
        required=True,
    )


def main(_):
    cfg = get_bundler_config(
        bundler_type=FLAGS.bundler_type, spec=FLAGS.bundler_spec, fv=FLAGS
    ).set(exclude=FLAGS.bundler_exclude)
    bundler = cfg.instantiate()
    bundler.bundle(FLAGS.name)


if __name__ == "__main__":
    main_flags()
    app.run(main)
