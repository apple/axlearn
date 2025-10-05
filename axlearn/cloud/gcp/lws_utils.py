# Copyright © 2025 Apple Inc.

"""Utilities for building LeaderWorkerSet specs"""

from typing import Any, Optional, Sequence

from absl import flags

from axlearn.cloud.common.bundler import Bundler
from axlearn.cloud.common.utils import AcceleratorConfig, FlagConfigurable, accelerator_flags
from axlearn.cloud.gcp.config import gcp_settings
from axlearn.cloud.gcp.jobset_utils import TPUJobBuilder
from axlearn.cloud.gcp.system_characteristics import USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.utils import Nested


class BaseLeaderWorkerTemplate(FlagConfigurable):
    """
    Common base class for LeaderWorker Templates
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """
        Configures BaseLeaderWorker.
        Attributes:
            name: Name of the LeaderWorkerSet
            command: Command to be executed.
            accelerator: Accelerator configuration.
            env_vars: Optional env vars to set.
            service_account: Optional service account to execute the job as.
            output_dir: An optional GCS path to upload LWS outputs to.
        """

        name: Required[str] = REQUIRED
        # TODO: Change this to be a list of str[], to support different commands
        # between leader and workers
        command: Required[str] = REQUIRED
        accelerator: AcceleratorConfig = AcceleratorConfig()
        env_vars: dict[str, str] = {}
        service_account: Optional[str] = None
        output_dir: Optional[str] = None
        image_id: Optional[str] = None

    @classmethod
    def define_flags(cls, fv):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        accelerator_flags(**common_kwargs)
        # NOTE: the parent typically sets these flags, so we leave them as None.
        flags.DEFINE_string("name", None, "Name of the LWS.", **common_kwargs)
        flags.DEFINE_string("command", None, "Command to execute.", **common_kwargs)
        flags.DEFINE_multi_string("env", [], "Env var in the format key:value.", **common_kwargs)
        flags.DEFINE_string(
            "service_account",
            None,
            "If specified, will run job as the service account.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "output_dir",
            None,
            "If specified, the directory to store outputs (such as logs).",
            **common_kwargs,
        )
        flags.DEFINE_boolean(
            "enable_pre_provisioner", None, "Whether to enable pre-provisioner.", **common_kwargs
        )
        flags.DEFINE_string(
            "image_id", None, "Image used for starting the container.", **common_kwargs
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: BaseLeaderWorkerTemplate.Config = super().from_flags(fv, **kwargs)
        cfg.service_account = cfg.service_account or gcp_settings(
            "k8s_service_account", default="default", fv=fv
        )
        cfg.accelerator.set(instance_type=fv.instance_type, num_replicas=fv.num_replicas)
        return cfg

    def __init__(self, cfg: Config, *, bundler: Bundler):
        super().__init__(cfg)
        self._bundler = bundler

    def __call__(self) -> Sequence[Nested[Any]]:
        """Builds LeaderWorkerTemplate for the LWS API.

        Returns:
        A nested dict corresponding to a LeaderWorkerTemplate config.
        """
        raise NotImplementedError(type(self))


class TPULeaderWorkerTemplate(TPUJobBuilder):
    """Builds a LeaderWorkerTemplate spec for a generic TPU workload"""

    Config = TPUJobBuilder.Config

    def __call__(self) -> Sequence[Nested[Any]]:
        system = USER_FACING_NAME_TO_SYSTEM_CHARACTERISTICS[self._tpu_type]
        return dict(
            size=system.vms_per_slice,
            workerTemplate=self._build_pod(),
        )
