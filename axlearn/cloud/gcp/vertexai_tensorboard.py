# Copyright © 2023 Apple Inc.

"""Tools to upload model summaries to VertexAI Tensorboard."""

import multiprocessing
import re

from absl import logging
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.tensorboard import uploader, uploader_utils
from google.cloud.aiplatform.utils import TensorboardClientWithOverride

from axlearn.cloud.gcp import config as gcp_config
from axlearn.common.config import REQUIRED, Configurable, Required, config_class


def _vertexai_experiment_name_from_output_dir(output_dir: str) -> str:
    """Creates Vertex AI experiment name from output_dir."""
    pattern = r"gs://[^/]+/([a-zA-Z0-9][a-zA-Z0-9-_/\.]+)"
    match = re.fullmatch(pattern=pattern, string=output_dir)
    if not match:
        raise ValueError(rf"{output_dir} does not match '{pattern}'.")
    # Vertex AI Tensorboard requires experiment_name to match "[a-z0-9][a-z0-9-]+".
    experiment_name = match.group(1).lower().replace("/", "-").replace("_", "-").replace(".", "-")
    return experiment_name


def is_vertexai_tensorboard_configured() -> bool:
    """Checks the config to see whether VertexAI Tensorboard should be enabled."""
    return bool(
        gcp_config.gcp_settings("vertexai_tensorboard")
        and gcp_config.gcp_settings("vertexai_region")
        and gcp_config.gcp_settings("project")
    )


class VertexAITensorboardUploader(Configurable):
    """Vertex AI Tensorboard Uploader.

    Spins up VertexAI tensorboard uploader in a separate process and ensures that the it is alive
    each time `upload` is called. If the process is not alive, kills it and restarts the process to
    avoid any gaps in uploading summaries.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures VertexAITensorboardUploader."""

        # Vertex AI Tensorboard Instance Id.
        instance_id: Required[str] = REQUIRED
        # Vertex AI region.
        region: Required[str] = REQUIRED
        # GCP project id.
        project_id: Required[str] = REQUIRED
        # Directory containing summaries.
        summary_dir: Required[str] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._resource_name = (
            f"projects/{cfg.project_id}/locations/{cfg.region}/tensorboards/{cfg.instance_id}"
        )
        self._uploader_proc = None

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        instance_id = gcp_config.gcp_settings(key="vertexai_tensorboard")
        region = gcp_config.gcp_settings(key="vertexai_region")
        project_id = gcp_config.gcp_settings(key="project")
        return cfg.set(instance_id=instance_id, region=region, project_id=project_id)

    def upload(self):
        """Uploads summaries in output_dir to Vertex AI. Runs/Keeps uploader process alive."""
        cfg = self.config

        def fn():
            api_client = initializer.global_config.create_client(
                client_class=TensorboardClientWithOverride,
                location_override=cfg.region,
            )

            (
                blob_storage_bucket,
                blob_storage_folder,
            ) = uploader_utils.get_blob_storage_bucket_and_folder(
                api_client, self._resource_name, cfg.project_id
            )
            tb_uploader = uploader.TensorBoardUploader(
                experiment_name=_vertexai_experiment_name_from_output_dir(cfg.summary_dir),
                experiment_display_name=_vertexai_experiment_name_from_output_dir(cfg.summary_dir),
                description=None,
                tensorboard_resource_name=self._resource_name,
                blob_storage_bucket=blob_storage_bucket,
                blob_storage_folder=blob_storage_folder,
                allowed_plugins=["scalars", "histograms", "distributions", "hparams", "text"],
                writer_client=api_client,
                logdir=cfg.summary_dir,
                one_shot=False,
                event_file_inactive_secs=None,
                verbosity=0,
                run_name_prefix=None,
            )
            tb_uploader.create_experiment()
            logging.info(
                "View your Tensorboard at: "
                "https://%s.tensorboard.googleusercontent.com/experiment/%s",
                cfg.region,
                tb_uploader.get_experiment_resource_name().replace("/", "+"),
            )
            tb_uploader.start_uploading()

        # TODO(vivekrathod): Refactor this pattern into a reusable class so we can apply it to all
        # the helper processes associated with a Jobs such as syncing logs, uploading summaries
        # among others.
        if self._uploader_proc is not None:
            self._uploader_proc.join(timeout=0)
            if not self._uploader_proc.is_alive():
                logging.info("VertexAI Tensorboard uploader process died, removing...")
                self._uploader_proc.kill()
                self._uploader_proc.join()
                self._uploader_proc = None
                logging.info("VertexAI Tensorboard uploader removed. Will restart...")

        if self._uploader_proc is None:
            logging.info("Starting VertexAI Tensorboard uploader.")
            self._uploader_proc = multiprocessing.Process(target=fn, daemon=True)
            self._uploader_proc.start()
            logging.info("VertexAI Tensorboard uploader started.")
        else:
            logging.info("VertexAI Tensorboard uploader is still running.")
