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
from typing import Dict, List

from absl import app, flags, logging

from axlearn.cloud.common.bundler import BaseDockerBundler, BaseTarBundler, Bundler, DockerBundler
from axlearn.cloud.common.bundler import main as bundler_main
from axlearn.cloud.common.bundler import main_flags as bundler_main_flags
from axlearn.cloud.common.bundler import register_bundler
from axlearn.cloud.common.docker import registry_from_repo
from axlearn.cloud.common.utils import canonicalize_to_list
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.utils import common_flags
from axlearn.common.config import maybe_set_config

FLAGS = flags.FLAGS


@register_bundler
class GCSTarBundler(BaseTarBundler):
    """A TarBundler that reads configs from gcp_settings, and uploads to GCS."""

    TYPE = "gcs"

    @classmethod
    def from_spec(cls, spec: List[str], *, fv: flags.FlagValues) -> BaseTarBundler.Config:
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
    def from_spec(cls, spec: List[str], *, fv: flags.FlagValues) -> DockerBundler.Config:
        cfg = super().from_spec(spec, fv=fv)
        if not cfg.repo:
            cfg.repo = gcp_settings("docker_repo", required=False, fv=fv)
        if not cfg.dockerfile:
            cfg.dockerfile = gcp_settings("default_dockerfile", required=False, fv=fv)
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

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.repo = gcp_settings("docker_repo", required=False)
        cfg.dockerfile = gcp_settings("default_dockerfile", required=False)
        return cfg

    # pylint: disable-next=no-self-use,unused-argument
    def _build_and_push(
        self,
        *,
        dockerfile: str,
        image: str,
        args: Dict[str, str],
        context: str,
        labels: Dict[str, str],
    ):
        cfg = self.config
        build_args = "\n".join(
            [f'    "--build-arg", "{key}={value}",' for key, value in args.items()]
        )
        labels = "\n".join([f'    "--label", "{key}={value}",' for key, value in labels.items()])
        build_target = f'    "--target", "{cfg.target}",' if cfg.target else ""
        build_platform = f'    "--platform", "{cfg.platform}",' if cfg.platform else ""
        cloudbuild_yaml = f"""
steps:
- name: "gcr.io/cloud-builders/docker"
  args: [
    "build",
    "-f", "{os.path.relpath(dockerfile, context)}",
    "-t", "{image}",
    "--cache-from", "{image}",
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
            gcp_settings("project"),
            "--config",
            cloudbuild_yaml_file,
            context,
        ]
        logging.info("Running %s", cmd)
        print(subprocess.run(cmd, check=False))
        return image


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
