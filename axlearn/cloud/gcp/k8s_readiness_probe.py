# Copyright © 2025 Apple Inc.

"""Kubernetes readiness and liveness probe configuration for GKE job containers."""

from typing import Optional

from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable
from axlearn.common.config import config_class


class _ContainerProbe(FlagConfigurable):
    """Base class for Kubernetes container probes (readiness, liveness).

    Subclasses specialise by providing a ``_flag_prefix`` class attribute and
    implementing a public ``build_*_probe()`` method that delegates to
    ``_build_probe()``.
    """

    # Override in subclasses (e.g. "readiness_probe" or "liveness_probe").
    _flag_prefix: str = ""

    @config_class
    class Config(FlagConfigurable.Config):
        """Shared probe configuration fields.

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
            failure_threshold: Consecutive failures before the pod is marked Unready/Unhealthy.
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
        if not cls._flag_prefix:
            raise ValueError(f"{cls.__name__} must set a non-empty _flag_prefix.")
        p = cls._flag_prefix
        common_kwargs = dict(flag_values=fv, allow_override=True)
        flags.DEFINE_integer(
            f"{p}_grpc_port",
            None,
            f"gRPC port for the {p}. Activates gRPC probe type.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            f"{p}_grpc_service",
            None,
            f"gRPC service name for the {p}.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            f"{p}_http_port",
            None,
            f"HTTP port for the {p}. Activates HTTP probe type.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            f"{p}_http_path",
            None,
            f"HTTP path for the {p} (e.g. '/healthz').",
            **common_kwargs,
        )
        flags.DEFINE_string(
            f"{p}_http_scheme",
            None,
            f"HTTP scheme for the {p} ('HTTP' or 'HTTPS').",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            f"{p}_initial_delay_seconds",
            None,
            "Seconds after container start before probes are initiated.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            f"{p}_period_seconds",
            None,
            "How often (in seconds) to perform the probe.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            f"{p}_timeout_seconds",
            None,
            "Seconds after which the probe times out.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            f"{p}_success_threshold",
            None,
            "Minimum consecutive successes for probe to be considered successful.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            f"{p}_failure_threshold",
            None,
            "Consecutive failures before the pod is marked Unready/Unhealthy.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs):
        cfg: _ContainerProbe.Config = super().from_flags(fv, **kwargs)
        p = cls._flag_prefix
        cfg.grpc_port = getattr(fv, f"{p}_grpc_port")
        cfg.grpc_service = getattr(fv, f"{p}_grpc_service")
        cfg.http_port = getattr(fv, f"{p}_http_port")
        cfg.http_path = getattr(fv, f"{p}_http_path")
        cfg.http_scheme = getattr(fv, f"{p}_http_scheme")
        cfg.initial_delay_seconds = getattr(fv, f"{p}_initial_delay_seconds")
        cfg.period_seconds = getattr(fv, f"{p}_period_seconds")
        cfg.timeout_seconds = getattr(fv, f"{p}_timeout_seconds")
        cfg.success_threshold = getattr(fv, f"{p}_success_threshold")
        cfg.failure_threshold = getattr(fv, f"{p}_failure_threshold")
        return cfg

    def is_configured(self) -> bool:
        """Returns True if a probe type has been configured (grpc_port or http_port is set)."""
        cfg: _ContainerProbe.Config = self.config
        return cfg.grpc_port is not None or cfg.http_port is not None

    def _build_probe(self) -> dict:
        """Builds the shared k8s probe dict, omitting any None fields.

        Returns:
            A dict corresponding to a k8s probe configuration.

        Raises:
            ValueError: If both grpc_port and http_port are set, or if neither is set.
        """
        cfg: _ContainerProbe.Config = self.config

        if cfg.grpc_port is not None and cfg.http_port is not None:
            raise ValueError("grpc_port and http_port are mutually exclusive; both are set.")
        if cfg.grpc_port is None and cfg.http_port is None:
            raise ValueError("Either grpc_port or http_port must be set to build a probe.")

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


class ReadinessProbe(_ContainerProbe):
    """Configures a Kubernetes readiness probe (gRPC or HTTP) for a container.

    Activate gRPC probing by setting grpc_port; activate HTTP probing by setting http_port.
    The two probe types are mutually exclusive.

    A failing readiness probe marks the pod as not-ready (e.g. removes it from Service
    endpoints), but does NOT restart the container. Use LivenessProbe if you want
    container restarts on failure.
    """

    _flag_prefix = "readiness_probe"

    Config = _ContainerProbe.Config

    def build_readiness_probe(self) -> dict:
        """Builds a k8s readiness probe dict, omitting any None fields.

        Returns:
            A dict corresponding to a k8s readiness probe configuration.

        Raises:
            ValueError: If both grpc_port and http_port are set, or if neither is set.
        """
        return self._build_probe()


class LivenessProbe(_ContainerProbe):
    """Configures a Kubernetes liveness probe (gRPC or HTTP) for a container.

    Activate gRPC probing by setting grpc_port; activate HTTP probing by setting http_port.
    The two probe types are mutually exclusive.

    A failing liveness probe causes the kubelet to kill and restart the container.
    Use ReadinessProbe if you only want to gate traffic without restarting.

    Flags are registered with the ``liveness_probe_`` prefix (e.g.
    ``liveness_probe_grpc_port``, ``liveness_probe_http_port``).
    """

    _flag_prefix = "liveness_probe"

    Config = _ContainerProbe.Config

    def build_liveness_probe(self) -> dict:
        """Builds a k8s liveness probe dict, omitting any None fields.

        Returns:
            A dict corresponding to a k8s liveness probe configuration.

        Raises:
            ValueError: If both grpc_port and http_port are set, or if neither is set.
        """
        return self._build_probe()


class StartupProbe(_ContainerProbe):
    """Configures a Kubernetes startup probe (gRPC or HTTP) for a container.

    Activate gRPC probing by setting grpc_port; activate HTTP probing by setting http_port.
    The two probe types are mutually exclusive.

    A startup probe delays the start of liveness and readiness probes until the container
    has finished initialising. Once the startup probe succeeds, liveness/readiness probes
    take over. If the startup probe never succeeds within failureThreshold * periodSeconds,
    the kubelet kills and restarts the container.

    Note: Kubernetes requires successThreshold=1 for startup probes; omitting
    success_threshold (the default) satisfies this constraint.

    Flags are registered with the ``startup_probe_`` prefix (e.g.
    ``startup_probe_grpc_port``, ``startup_probe_http_port``).
    """

    _flag_prefix = "startup_probe"

    Config = _ContainerProbe.Config

    def build_startup_probe(self) -> dict:
        """Builds a k8s startup probe dict, omitting any None fields.

        Returns:
            A dict corresponding to a k8s startup probe configuration.

        Raises:
            ValueError: If both grpc_port and http_port are set, or if neither is set.
        """
        return self._build_probe()
