# Copyright © 2025 Apple Inc.

"""Kubernetes readiness probe configuration for GKE job containers."""

from typing import Optional

from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable
from axlearn.common.config import config_class


class ReadinessProbe(FlagConfigurable):
    """Configures a Kubernetes readiness probe (gRPC or HTTP) for a container.

    Activate gRPC probing by setting grpc_port; activate HTTP probing by setting http_port.
    The two probe types are mutually exclusive.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures ReadinessProbe.

        Attributes:
            grpc_port: gRPC port. Setting this activates the gRPC probe type.
            grpc_service: Optional gRPC service name for the probe.
            http_port: HTTP port. Setting this activates the HTTP probe type.
            http_path: Optional HTTP path (e.g. "/healthz").
            http_scheme: Optional HTTP scheme ("HTTP" or "HTTPS").
            initial_delay_seconds: Seconds after container start before probes are initiated.
            period_seconds: How often (in seconds) to perform the probe.
            timeout_seconds: Seconds after which the probe times out.
            success_threshold: Minimum consecutive successes for probe to be considered successful.
            failure_threshold: Consecutive failures before the pod is marked Unready.
        """

        # gRPC-specific fields. Non-None grpc_port activates the gRPC probe.
        grpc_port: Optional[int] = None
        grpc_service: Optional[str] = None

        # HTTP-specific fields. Non-None http_port activates the HTTP probe.
        http_port: Optional[int] = None
        http_path: Optional[str] = None
        http_scheme: Optional[str] = None

        # Common timing fields.
        initial_delay_seconds: Optional[int] = None
        period_seconds: Optional[int] = None
        timeout_seconds: Optional[int] = None
        success_threshold: Optional[int] = None
        failure_threshold: Optional[int] = None

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_integer(
            "readiness_probe_grpc_port",
            None,
            "gRPC port for the readiness probe. Activates gRPC probe type.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "readiness_probe_grpc_service",
            None,
            "gRPC service name for the readiness probe.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "readiness_probe_http_port",
            None,
            "HTTP port for the readiness probe. Activates HTTP probe type.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "readiness_probe_http_path",
            None,
            "HTTP path for the readiness probe (e.g. '/healthz').",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "readiness_probe_http_scheme",
            None,
            "HTTP scheme for the readiness probe ('HTTP' or 'HTTPS').",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "readiness_probe_initial_delay_seconds",
            None,
            "Seconds after container start before probes are initiated.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "readiness_probe_period_seconds",
            None,
            "How often (in seconds) to perform the probe.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "readiness_probe_timeout_seconds",
            None,
            "Seconds after which the probe times out.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "readiness_probe_success_threshold",
            None,
            "Minimum consecutive successes for probe to be considered successful.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "readiness_probe_failure_threshold",
            None,
            "Consecutive failures before the pod is marked Unready.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: ReadinessProbe.Config = super().from_flags(fv, **kwargs)
        cfg.grpc_port = fv.readiness_probe_grpc_port
        cfg.grpc_service = fv.readiness_probe_grpc_service
        cfg.http_port = fv.readiness_probe_http_port
        cfg.http_path = fv.readiness_probe_http_path
        cfg.http_scheme = fv.readiness_probe_http_scheme
        cfg.initial_delay_seconds = fv.readiness_probe_initial_delay_seconds
        cfg.period_seconds = fv.readiness_probe_period_seconds
        cfg.timeout_seconds = fv.readiness_probe_timeout_seconds
        cfg.success_threshold = fv.readiness_probe_success_threshold
        cfg.failure_threshold = fv.readiness_probe_failure_threshold
        return cfg

    def is_configured(self) -> bool:
        """Returns True if a probe type has been configured (grpc_port or http_port is set)."""
        cfg: ReadinessProbe.Config = self.config
        return cfg.grpc_port is not None or cfg.http_port is not None

    def build_readiness_probe(self) -> dict:
        """Builds a k8s readiness probe dict, omitting any None fields.

        Returns:
            A dict corresponding to a k8s readiness probe configuration.

        Raises:
            ValueError: If both grpc_port and http_port are set, or if neither is set.
        """
        cfg: ReadinessProbe.Config = self.config

        if cfg.grpc_port is not None and cfg.http_port is not None:
            raise ValueError("grpc_port and http_port are mutually exclusive; both are set.")
        if cfg.grpc_port is None and cfg.http_port is None:
            raise ValueError(
                "Either grpc_port or http_port must be set to build a readiness probe."
            )

        probe: dict = {}

        if cfg.grpc_port is not None:
            grpc_spec: dict = {"port": cfg.grpc_port}
            if cfg.grpc_service is not None:
                grpc_spec["service"] = cfg.grpc_service
            probe["grpc"] = grpc_spec
        else:
            http_spec: dict = {"port": cfg.http_port}
            if cfg.http_path is not None:
                http_spec["path"] = cfg.http_path
            if cfg.http_scheme is not None:
                http_spec["scheme"] = cfg.http_scheme
            probe["httpGet"] = http_spec

        if cfg.initial_delay_seconds is not None:
            probe["initialDelaySeconds"] = cfg.initial_delay_seconds
        if cfg.period_seconds is not None:
            probe["periodSeconds"] = cfg.period_seconds
        if cfg.timeout_seconds is not None:
            probe["timeoutSeconds"] = cfg.timeout_seconds
        if cfg.success_threshold is not None:
            probe["successThreshold"] = cfg.success_threshold
        if cfg.failure_threshold is not None:
            probe["failureThreshold"] = cfg.failure_threshold

        return probe
