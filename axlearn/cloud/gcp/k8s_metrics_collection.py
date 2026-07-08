# Copyright © 2026 Apple Inc.

"""Metrics collection configurables for HPA-driven autoscaling.

`MetricsCollection` is an abstract Configurable for K8s resources that tell
a metrics backend (e.g., Google Managed Prometheus) which pods to scrape.
The concrete `PodMonitoring` implementation creates a GMP `PodMonitoring`
custom resource. Alternative implementations (Prometheus Operator
`ServiceMonitor`, OpenTelemetry, etc.) can plug into the same runner field.

The runner instantiates and calls `execute()` after the workload is
created, passing the workload's owner reference so the scraping resource
is garbage-collected when the workload is deleted.
"""

import logging
from typing import Optional, Union

import kubernetes as k8s
from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable
from axlearn.common.config import REQUIRED, Required, config_class

_GMP_GROUP = "monitoring.googleapis.com"
_GMP_VERSION = "v1"
_GMP_PODMONITORING_PLURAL = "podmonitorings"


def _make_owner_reference(
    *,
    api_version: str,
    kind: str,
    name: str,
    uid: str,
) -> dict:
    """Build a k8s ownerReference dict for cascading deletion."""
    return {
        "apiVersion": api_version,
        "kind": kind,
        "name": name,
        "uid": uid,
        "controller": False,
        "blockOwnerDeletion": False,
    }


class MetricsCollection(FlagConfigurable):
    """Abstract base for resources that configure metric scraping.

    Concrete subclasses implement `execute()` to create the backing K8s
    resource (e.g., GMP `PodMonitoring`, Prometheus `ServiceMonitor`).
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Common metrics-collection fields.

        Attributes:
            name: Name for the K8s resource.
            namespace: Kubernetes namespace.
        """

        name: Required[str] = REQUIRED
        namespace: str = "default"

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # MetricsCollection is auto-enabled when `--scaling_metrics_port`
        # is set. The runner is responsible for picking a concrete impl
        # (e.g. `PodMonitoring` for GMP) and populating name / selector.
        flags.DEFINE_string(
            "scaling_metrics_port",
            None,
            "Pod port (name or number) to scrape metrics from.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "scaling_metrics_path",
            None,
            "HTTP path to scrape. Default /metrics.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "scaling_metrics_interval",
            None,
            "Scrape interval. Default 30s.",
            **common_kwargs,
        )

    def compose_metric_name(self, name: str) -> str:
        """Transforms a bare metric name to the form the metrics adapter exposes.

        Default is identity — subclasses tied to a specific metrics
        adapter override this to apply the adapter's name conventions
        (e.g. Stackdriver's `<prefix>|<name>|<kind>` path).
        """
        return name

    def execute(self, *, owner: Optional[dict] = None) -> None:
        """Create the metrics-collection resource. Must be idempotent."""
        raise NotImplementedError(type(self))


class PodMonitoring(MetricsCollection):
    """Google Managed Prometheus PodMonitoring CR.

    Creates a `monitoring.googleapis.com/v1/PodMonitoring` CR that tells
    GMP to scrape the matching pods on the configured port and path.
    """

    @config_class
    class Config(MetricsCollection.Config):
        """Configures a GMP PodMonitoring.

        Attributes:
            selector_labels: Pod label selector. The CR scrapes pods whose
                labels match all entries in this map.
            port: Port name or number to scrape (matches the pod's
                container port name or numeric value).
            path: HTTP path for the scrape endpoint.
            interval: Scrape interval (e.g. "30s", "1m").
            metric_relabeling: Optional list of `endpoint.metricRelabeling`
                rules applied to scraped metrics. Each entry is a dict
                with the GMP relabeling fields (action, sourceLabels,
                targetLabel, replacement, regex, separator). Useful to
                inject a unique metric label so HPA selectors can isolate
                this PodMonitoring's scrape from any other scrape config
                that picks up the same pods.
            metric_name_prefix: Prefix that the Stackdriver custom-metrics
                adapter prepends to every Prometheus metric scraped by
                GMP. The adapter exposes metrics under
                `<prefix>|<name>|<kind>` (the Stackdriver type path with
                `/` replaced by `|`, since `/` isn't allowed in URL path
                segments). Use an empty string to opt out of prefixing.
            metric_name_kind: Suffix appended to the composed metric name
                (`gauge` for instantaneous values, `counter` for
                monotonic). Empty string opts out of the suffix.
        """

        selector_labels: dict = {}
        port: Required[Union[int, str]] = REQUIRED
        path: str = "/metrics"
        interval: str = "30s"
        metric_relabeling: list = []
        metric_name_prefix: str = "prometheus.googleapis.com"
        metric_name_kind: str = "gauge"

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # Metric-name composition is specific to the Stackdriver
        # custom-metrics adapter that fronts GMP, so the flags live on
        # this Configurable rather than on HPA.
        flags.DEFINE_string(
            "scaling_metric_prefix",
            None,
            "Prefix that GMP prepends to scraped Prometheus metrics. "
            "Empty string disables the prefix. Default prometheus.googleapis.com.",
            **common_kwargs,
        )
        flags.DEFINE_string(
            "scaling_metric_kind",
            None,
            "Suffix appended to the composed metric name. Use `counter` "
            "for monotonic metrics. Empty string disables the suffix. Default gauge.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> "PodMonitoring.Config":
        """Populate a `PodMonitoring.Config` from `--scaling_metrics_*` flags.

        Returns the prebuilt cfg unchanged when no `--scaling_metrics_port`
        is set — the caller decides whether to create the resource without
        a configured port.
        """
        cfg: PodMonitoring.Config = super().from_flags(fv, **kwargs)
        port_value = fv.scaling_metrics_port
        if not port_value:
            return cfg
        # Try numeric port first, fall back to named port string.
        try:
            cfg.port = int(port_value)
        except ValueError:
            cfg.port = port_value
        # Override only when the flag was explicitly provided (flags default to
        # None). The Config field holds the default. `is not None` — not
        # falsiness — so an explicit empty string (opt-out for prefix/kind)
        # still overrides.
        if fv.scaling_metrics_path is not None:
            cfg.path = fv.scaling_metrics_path
        if fv.scaling_metrics_interval is not None:
            cfg.interval = fv.scaling_metrics_interval
        if fv.scaling_metric_prefix is not None:
            cfg.metric_name_prefix = fv.scaling_metric_prefix
        if fv.scaling_metric_kind is not None:
            cfg.metric_name_kind = fv.scaling_metric_kind
        return cfg

    def compose_metric_name(self, name: str) -> str:
        """Compose `<prefix>|<name>|<kind>` for the Stackdriver adapter.

        If `name` already contains a `|`, it is assumed to be
        pre-composed and returned unchanged. Empty prefix or kind are
        omitted.
        """
        if "|" in name:
            return name
        cfg: PodMonitoring.Config = self.config
        parts = [p for p in (cfg.metric_name_prefix, name, cfg.metric_name_kind) if p]
        return "|".join(parts)

    def execute(self, *, owner: Optional[dict] = None) -> None:
        cfg: PodMonitoring.Config = self.config
        endpoint: dict = {
            "port": cfg.port,
            "path": cfg.path,
            "interval": cfg.interval,
        }
        if cfg.metric_relabeling:
            endpoint["metricRelabeling"] = list(cfg.metric_relabeling)
        body = {
            "apiVersion": f"{_GMP_GROUP}/{_GMP_VERSION}",
            "kind": "PodMonitoring",
            "metadata": {"name": cfg.name, "namespace": cfg.namespace},
            "spec": {
                "selector": {"matchLabels": dict(cfg.selector_labels)},
                "endpoints": [endpoint],
            },
        }
        if owner is not None:
            body["metadata"]["ownerReferences"] = [owner]
        api = k8s.client.CustomObjectsApi()
        try:
            api.create_namespaced_custom_object(
                group=_GMP_GROUP,
                version=_GMP_VERSION,
                namespace=cfg.namespace,
                plural=_GMP_PODMONITORING_PLURAL,
                body=body,
            )
            logging.info("Created PodMonitoring %s/%s", cfg.namespace, cfg.name)
        except k8s.client.ApiException as e:
            if e.status != 409:
                raise
            logging.info("PodMonitoring %s/%s already exists; skipping", cfg.namespace, cfg.name)
