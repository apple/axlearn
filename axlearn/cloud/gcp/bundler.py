# Copyright © 2023 Apple Inc.

"""Code bundling utilities.

The type of bundler to use is determined by `--bundler_type`. Bundlers can also be configured via
`--bundler_spec` flags; see the corresponding bundler class' `from_spec` method for details.

Examples (gcs):

    # Tar and upload to GCS with default bucket.
    axlearn gcp bundle --bundler_type=gcs --name=$USER-test

    # Tar and upload to GCS with custom bucket.
    axlearn gcp bundle --bundler_type=gcs --name=$USER-test \
        --bundler_spec=gs_bucket=my-custom-bucket

    # Tar and upload to GCS with a locally built wheel.
    axlearn gcp bundle --bundler_type=gcs --name=$USER-test \
        --bundler_spec=external=/path/to/dist/ \
        --bundler_spec=extras=dev,axlearn.whl

    # Tar and upload to GCS with --find-links.
    axlearn gcp bundle --bundler_type=gcs --name=$USER-test \
        --bundler_spec=extras=tpu \
        --bundler_spec=find_links=https://storage.googleapis.com/jax-releases/libtpu_releases.html

Examples (artifactregistry):

    # Docker build and push to repo.
    axlearn gcp bundle --bundler_type=artifactregistry \
        --name=my-tag \
        --bundler_spec=image=my-image \
        --bundler_spec=repo=my-repo \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=build_arg1=my-build-arg

Examples (cloudbuild):

    # Docker build and push to repo.
    axlearn gcp bundle --bundler_type=cloudbuild \
        --name=my-tag \
        --bundler_spec=image=my-image \
        --bundler_spec=repo=my-repo \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=build_arg1=my-build-arg

"""

import os
import subprocess
import time
from typing import Optional

from absl import app, flags, logging

from axlearn.cloud.common.bundler import BaseDockerBundler, BaseTarBundler, Bundler, DockerBundler
from axlearn.cloud.common.bundler import main as bundler_main
from axlearn.cloud.common.bundler import main_flags as bundler_main_flags
from axlearn.cloud.common.bundler import register_bundler
from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.utils import canonicalize_to_list
from axlearn.cloud.gcp.cloud_build import get_cloud_build_status
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.utils import common_flags
from axlearn.common.config import REQUIRED, Required, config_class, maybe_set_config

FLAGS = flags.FLAGS


@register_bundler
class GCSTarBundler(BaseTarBundler):
    """A TarBundler that reads configs from gcp_settings, and uploads to GCS."""

    TYPE = "gcs"

    @classmethod
    def from_spec(cls, spec: list[str], *, fv: Optional[flags.FlagValues]) -> BaseTarBundler.Config:
        """Converts a spec to a bundler.

        Possible options:
        - remote_dir: The remote directory to copy the bundle to. Must be compatible with tf_io.
        """
        cfg = super().from_spec(spec, fv=fv)
        # Read from settings by default, if available.
        if not cfg.remote_dir:
            ttl_bucket = gcp_settings("ttl_bucket", required=False, fv=fv)
            if ttl_bucket:
                cfg.remote_dir = f"gs://{ttl_bucket}/axlearn/jobs"
        return cfg

    def _copy_to_local_command(self, *, remote_bundle_id: str, local_bundle_id: str) -> str:
        """Assumes that bundling happens in an environment where `gsutil` is available."""
        return f"gsutil -q cp {remote_bundle_id} {local_bundle_id}"


@register_bundler
class ArtifactRegistryBundler(DockerBundler):
    """A DockerBundler that reads configs from gcp_settings, and auths to Artifact Registry."""

    TYPE = "artifactregistry"

    @classmethod
    def from_spec(cls, spec: list[str], *, fv: Optional[flags.FlagValues]) -> DockerBundler.Config:
        cfg = super().from_spec(spec, fv=fv)
        cfg.repo = cfg.repo or gcp_settings("docker_repo", required=False, fv=fv)
        cfg.dockerfile = cfg.dockerfile or gcp_settings("default_dockerfile", required=False, fv=fv)
        return cfg

    def _build_and_push(self, *args, **kwargs):
        cfg = self.config
        subprocess.run(
            ["gcloud", "auth", "configure-docker", registry_from_repo(cfg.repo)],
            check=True,
        )
        return super()._build_and_push(*args, **kwargs)


@register_bundler
class CloudBuildBundler(BaseDockerBundler):
    """A bundler that uses CloudBuild."""

    TYPE = "cloudbuild"

    @config_class
    class Config(BaseDockerBundler.Config):
        """Configures CloudBuildBundler.

        Attributes:
            project: GCP project. If using `from_spec`, this is inferred from `gcp_settings` or
                from flags.
            is_async: Whether to build asynchronously. If True, callers should invoke
                `wait_until_finished()` to wait for bundling to complete.
        """

        # GCP project.
        project: Required[str] = REQUIRED
        # Build image asynchronously.
        is_async: bool = True
        # If provided, should be the identifier of a private worker pool.
        # See: https://cloud.google.com/build/docs/private-pools/private-pools-overview
        private_worker_pool: Optional[str] = None

    @classmethod
    def from_spec(
        cls, spec: list[str], *, fv: Optional[flags.FlagValues]
    ) -> BaseDockerBundler.Config:
        cfg: CloudBuildBundler.Config = super().from_spec(spec, fv=fv)
        cfg.project = cfg.project or gcp_settings("project", required=False, fv=fv)
        cfg.repo = cfg.repo or gcp_settings("docker_repo", required=False, fv=fv)
        cfg.dockerfile = cfg.dockerfile or gcp_settings("default_dockerfile", required=False, fv=fv)
        # The value from from_spec is a str and will result in wrong condition.
        if isinstance(cfg.is_async, str):
            cfg.is_async = cfg.is_async.lower() != "false"
        return cfg

    # pylint: disable-next=no-self-use,unused-argument
    def _build_and_push(
        self,
        *,
        dockerfile: str,
        image: str,
        args: dict[str, str],
        context: str,
        labels: dict[str, str],
    ):
        cfg: CloudBuildBundler.Config = self.config
        logging.info("CloudBuild build args: %s", args)
        build_args = "\n".join(
            [f'"--build-arg", "{k.strip().upper()}={v.strip()}",' for k, v in args.items()]
        )
        labels = "\n".join([f'"--label", "{k.strip()}={v.strip()}",' for k, v in labels.items()])
        build_target = f'"--target", "{cfg.target}",' if cfg.target else ""
        build_platform = f'"--platform", "{cfg.platform}",' if cfg.platform else ""
        cache_from = (
            "\n".join([f'"--cache-from", "{cache_source}",' for cache_source in cfg.cache_from])
            if cfg.cache_from
            else ""
        )
        image_path, image_tag = image.rsplit(":", maxsplit=1)
        latest_tag = f"{image_path}:latest"
        cloudbuild_yaml = f"""
steps:
- name: "gcr.io/cloud-builders/docker"
  args: [
    "build",
    "-f", "{os.path.relpath(dockerfile, context)}",
    "-t", "{image}",
    "-t", "{latest_tag}",
    "--cache-from", "{image}",
    "--cache-from", "{latest_tag}",
    {cache_from}
    {build_target}
    {build_platform}
    {build_args}
    {labels}
    "."
  ]
  env:
  - "DOCKER_BUILDKIT=1"
timeout: 3600s
images:
- "{image}"
- "{latest_tag}"
tags: [{image_tag}]
options:
  logging: CLOUD_LOGGING_ONLY
        """
        cloudbuild_yaml_file = os.path.join(context, "cloudbuild.yaml")
        logging.info("CloudBuild YAML:\n%s", cloudbuild_yaml)
        with open(cloudbuild_yaml_file, "w", encoding="utf-8") as f:
            f.write(cloudbuild_yaml)
        cmd = [
            "gcloud",
            "builds",
            "submit",
            "--project",
            cfg.project,
            "--config",
            cloudbuild_yaml_file,
            context,
        ]
        if cfg.private_worker_pool:
            cmd.extend(["--worker-pool", cfg.private_worker_pool])
        if cfg.is_async:
            cmd.append("--async")
        logging.info("Running %s", cmd)
        print(subprocess.run(cmd, check=True))
        return image

    def wait_until_finished(self, name: str):
        """Waits for async CloudBuild to finish by polling for status.

        Is a no-op if `cfg.is_async` is False.

        Args:
            name: Bundle name.

        Raises:
            ValueError: If async build failed.
        """
        cfg: CloudBuildBundler.Config = self.config
        while cfg.is_async:
            try:
                build_status = get_cloud_build_status(
                    project_id=cfg.project, image_name=self.id(name), tags=[name]
                )
            except Exception as e:  # pylint: disable=broad-except
                # TODO(liang-he,markblee): Distinguish transient from non-transient errors.
                logging.warning("Failed to get the CloudBuild status, will retry: %s", e)
            else:
                if not build_status:
                    logging.warning("CloudBuild for %s does not exist yet.", name)
                elif build_status.is_pending():
                    logging.info("CloudBuild for %s is pending: %s.", name, build_status)
                elif build_status.is_success():
                    logging.info("CloudBuild for %s is successful: %s.", name, build_status)
                    return
                else:
                    # Unknown status is also considered a failure.
                    raise RuntimeError(f"CloudBuild for {name} failed: {build_status}.")
            time.sleep(30)


def with_tpu_extras(bundler: Bundler.Config) -> Bundler.Config:
    """Configures bundler to install 'tpu' extras."""
    # Note: find_links is only applicable for tar bundlers.
    # For docker bundlers, point to the TPU build target.
    find_links = canonicalize_to_list(getattr(bundler, "find_links", []))
    find_links.append("https://storage.googleapis.com/jax-releases/libtpu_releases.html")
    maybe_set_config(bundler, find_links=find_links)
    extras = canonicalize_to_list(bundler.extras)
    extras.append("tpu")
    bundler.set(extras=extras)
    return bundler


if __name__ == "__main__":
    common_flags()
    bundler_main_flags()
    app.run(bundler_main)
