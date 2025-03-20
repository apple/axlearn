# Copyright © 2023 Apple Inc.

"""Tools to upload model summaries to VertexAI Tensorboard."""

import multiprocessing
import re

from absl import flags, logging

try:
    from google.cloud.aiplatform import initializer
    from google.cloud.aiplatform.tensorboard import uploader, uploader_utils
    from google.cloud.aiplatform.utils import TensorboardClientWithOverride

    _VERTEXAI_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    _VERTEXAI_INSTALLED = False


from axlearn.cloud.gcp import config as gcp_config
from axlearn.common.config import REQUIRED, Configurable, Required, config_class

_VERTEXAI_EXP_NAME_MAX_LEN = 128


def _vertexai_experiment_name_from_output_dir(output_dir: str) -> str:
    """Creates Vertex AI experiment name from output_dir."""
    pattern = r"gs://[^/]+/([a-zA-Z0-9][a-zA-Z0-9-_/\.]+)"
    match = re.fullmatch(pattern=pattern, string=output_dir)
    if not match:
        raise ValueError(rf"{output_dir} does not match '{pattern}'.")
    # Vertex AI Tensorboard requires experiment_name to match "[a-z0-9][a-z0-9-]+".
    experiment_name = match.group(1).lower().replace("/", "-").replace("_", "-").replace(".", "-")
    if len(experiment_name) >= _VERTEXAI_EXP_NAME_MAX_LEN:  # Vertex AI length limit.
        raise ValueError(
            rf"Experiment name must be less than {_VERTEXAI_EXP_NAME_MAX_LEN} chars long."
            rf"{experiment_name} is {len(experiment_name)} chars."
        )
    return experiment_name


def is_vertexai_tensorboard_configured(flag_values: flags.FlagValues) -> bool:
    """Checks the config to see whether VertexAI Tensorboard should be enabled."""
    return _VERTEXAI_INSTALLED and bool(
        gcp_config.gcp_settings("vertexai_tensorboard", required=False, fv=flag_values)
        and gcp_config.gcp_settings("vertexai_region", required=False, fv=flag_values)
        and gcp_config.gcp_settings("project", required=False, fv=flag_values)
    )


# Keep as a top-level function so that it is pickleable.
def _start_vertexai_tensorboard(*, project_id: str, region: str, resource_name: str, logdir: str):
    api_client = initializer.global_config.create_client(
        client_class=TensorboardClientWithOverride,
        location_override=region,
    )

    (
        blob_storage_bucket,
        blob_storage_folder,
    ) = uploader_utils.get_blob_storage_bucket_and_folder(api_client, resource_name, project_id)
    tb_uploader = uploader.TensorBoardUploader(
        experiment_name=_vertexai_experiment_name_from_output_dir(logdir),
        experiment_display_name=_vertexai_experiment_name_from_output_dir(logdir),
        description=None,
        tensorboard_resource_name=resource_name,
        blob_storage_bucket=blob_storage_bucket,
        blob_storage_folder=blob_storage_folder,
        allowed_plugins=["scalars", "histograms", "distributions", "hparams", "text"],
        writer_client=api_client,
        logdir=logdir,
        one_shot=False,
        event_file_inactive_secs=None,
        verbosity=0,
        run_name_prefix=None,
        logdir_poll_rate_limiter=uploader_utils.RateLimiter(interval_secs=30),
    )
    tb_uploader.create_experiment()
    tb_uploader.start_uploading()


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
    def from_flags(cls, fv: flags.FlagValues):
        cfg = super().default_config()
        instance_id = gcp_config.gcp_settings(key="vertexai_tensorboard", fv=fv)
        region = gcp_config.gcp_settings(key="vertexai_region", fv=fv)
        project_id = gcp_config.gcp_settings(key="project", fv=fv)
        return cfg.set(instance_id=instance_id, region=region, project_id=project_id)

    def upload(self):
        """Uploads summaries in output_dir to Vertex AI. Runs/Keeps uploader process alive."""
        cfg = self.config
        # TODO(vivekrathod,markblee): Use Uploader.
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
            kwargs = dict(
                project_id=cfg.project_id,
                region=cfg.region,
                resource_name=self._resource_name,
                logdir=cfg.summary_dir,
            )
            self._uploader_proc = multiprocessing.Process(
                target=_start_vertexai_tensorboard, kwargs=kwargs, daemon=True
            )
            self._uploader_proc.start()
            logging.info("VertexAI Tensorboard uploader started.")

        # For Tensorboard experiment format, see:
        # https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/projects.locations.tensorboards.experiments
        experiment = (
            f"projects/{cfg.project_id}/locations/{cfg.region}/tensorboards/{cfg.instance_id}/"
            f"experiments/{_vertexai_experiment_name_from_output_dir(cfg.summary_dir)}"
        )
        logging.info(
            "VertexAI Tensorboard is running. View your Tensorboard at: "
            "https://%s.tensorboard.googleusercontent.com/experiment/%s",
            cfg.region,
            experiment.replace("/", "+"),
        )
