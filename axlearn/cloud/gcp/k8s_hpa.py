# Copyright © 2026 Apple Inc.

"""HorizontalPodAutoscaler Configurable for axlearn workloads.

Creates a `autoscaling/v2 HorizontalPodAutoscaler` targeting an arbitrary
scale subresource. The runner is responsible for filling in
`scale_target_*` fields with the right reference (typically
`apps/v1 Deployment`, `leaderworkerset.x-k8s.io/v1 LeaderWorkerSet`, or a
custom CRD that exposes a `scale` subresource).

`HPA` supports a list of metrics: any combination of Pods / Resource /
Object / External metric sources, mapped from `HPAMetric` configs.
"""

import logging
from typing import Optional

import kubernetes as k8s
from absl import flags

from axlearn.cloud.common.utils import FlagConfigurable
from axlearn.common.config import REQUIRED, Configurable, Required, config_class

_VALID_METRIC_TYPES = ("Pods", "Resource", "Object", "External")
_VALID_TARGET_TYPES = ("AverageValue", "Value", "Utilization")


class DescribedObject(Configurable):
    """The cluster object referenced by an HPA Object-type metric.

    Maps directly to the K8s HPA `object.describedObject` field. Used as
    a child config on an `HPAMetric` with `type="Object"`.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures a DescribedObject.

        Attributes:
            api_version: apiVersion of the referenced object
                (e.g. `networking.k8s.io/v1`).
            kind: Kind of the referenced object (e.g. `Ingress`).
            name: Name of the referenced object.
        """

        api_version: Required[str] = REQUIRED
        kind: Required[str] = REQUIRED
        name: Required[str] = REQUIRED


class HPAMetric(Configurable):
    """One metric in an HPA spec.

    The HPA controller scales the target so the average value of this
    metric across pods matches `target_value`. Supports the four standard
    HPA metric source types: `Pods`, `Resource`, `Object`, `External`.

    For `Pods` and `External` metrics, the `metric_name` is a custom
    metric name registered with the cluster's metrics API.
    """

    @config_class
    class Config(Configurable.Config):
        """Configures one HPA metric.

        Attributes:
            type: Source type — Pods | Resource | Object | External.
            metric_name: Metric name (for Pods / Object / External) or
                resource name like `cpu` / `memory` (for Resource).
            target_type: AverageValue | Value | Utilization. Most custom
                metrics use AverageValue; Resource uses Utilization.
            target_value: Target value as a string. For Utilization this
                is a percent ("80"); for AverageValue it's a quantity
                ("100" or "500m"); for Value it's an absolute quantity.
            selector: Optional label selector that filters which time
                series the metrics adapter returns for this metric.
                Useful when multiple scrape configs ingest the same
                metric — by selecting on a unique label injected by
                your own metrics-collection resource (e.g. `job`),
                HPA only sees your time series and other scrapers
                don't contribute duplicate values.
            described_object: Required for `type=Object` metrics. The
                cluster object whose metric drives scaling (e.g. an
                Ingress whose `requests-per-second` triggers scale).
                Maps to HPA `object.describedObject`.
        """

        type: str = "Pods"
        metric_name: Required[str] = REQUIRED
        target_type: str = "AverageValue"
        target_value: Required[str] = REQUIRED
        selector: Optional[dict] = None
        described_object: Optional[DescribedObject.Config] = None

    def build_identifier(self) -> dict:
        """Build the `metric` dict (name + optional selector) for HPA specs."""
        cfg: HPAMetric.Config = self.config
        identifier: dict = {"name": cfg.metric_name}
        if cfg.selector:
            identifier["selector"] = {"matchLabels": dict(cfg.selector)}
        return identifier

    def build_spec(self) -> dict:
        """Translate this metric to an HPA `metrics[]` entry."""
        cfg: HPAMetric.Config = self.config
        if cfg.type not in _VALID_METRIC_TYPES:
            raise ValueError(
                f"Unsupported HPA metric type: {cfg.type!r}. Expected one of {_VALID_METRIC_TYPES}."
            )
        if cfg.target_type not in _VALID_TARGET_TYPES:
            raise ValueError(
                f"Unsupported HPA target_type: {cfg.target_type!r}. "
                f"Expected one of {_VALID_TARGET_TYPES}."
            )
        target = {"type": cfg.target_type}
        if cfg.target_type == "Utilization":
            target["averageUtilization"] = int(cfg.target_value)
        elif cfg.target_type == "AverageValue":
            target["averageValue"] = cfg.target_value
        else:  # Value
            target["value"] = cfg.target_value

        identifier = self.build_identifier()

        if cfg.type == "Pods":
            return {"type": "Pods", "pods": {"metric": identifier, "target": target}}
        if cfg.type == "Resource":
            # Resource metrics don't use the selector field.
            return {
                "type": "Resource",
                "resource": {"name": cfg.metric_name, "target": target},
            }
        if cfg.type == "External":
            return {"type": "External", "external": {"metric": identifier, "target": target}}
        # cfg.type == "Object" — last remaining valid value.
        if cfg.described_object is None:
            raise ValueError(
                "HPA metric type=Object requires `described_object` "
                "(api_version, kind, name) so HPA can fetch the metric "
                "from the referenced resource."
            )
        return {
            "type": "Object",
            "object": {
                "metric": identifier,
                "target": target,
                "describedObject": {
                    "apiVersion": cfg.described_object.api_version,
                    "kind": cfg.described_object.kind,
                    "name": cfg.described_object.name,
                },
            },
        }


class HPA(FlagConfigurable):
    """HorizontalPodAutoscaler resource manager.

    Creates an `autoscaling/v2 HorizontalPodAutoscaler` targeting the
    referenced scale subresource. Multiple metrics are supported and
    combined per HPA semantics (HPA uses the recommendation that
    requires the largest replica count across all metrics).

    Lifecycle: caller invokes `execute()` after the target resource
    exists, passing the workload's owner reference so the HPA is
    garbage-collected on workload deletion.
    """

    @config_class
    class Config(FlagConfigurable.Config):
        """Configures an HPA.

        Attributes:
            name: Name for the HPA resource.
            namespace: Kubernetes namespace.
            scale_target_api_version: apiVersion of the scale-target
                resource (e.g. `apps/v1`, `leaderworkerset.x-k8s.io/v1`).
            scale_target_kind: Kind of the scale-target resource (e.g.
                `Deployment`, `LeaderWorkerSet`).
            scale_target_name: Name of the scale-target resource.
            min_replicas: Minimum replica count.
            max_replicas: Maximum replica count.
            metrics: Metric specs that drive scaling decisions.
            scale_up_stabilization_seconds: Time to wait before applying
                a new scale-up recommendation. Smooths spikes.
            scale_down_stabilization_seconds: Same for scale-down.
                Larger values prevent thrashing on noisy metrics.
            scale_up_percent: Per-period cap on scale-up rate, as a
                percentage of currentReplicas. Default 100 (fleet may
                double each `scale_up_period_seconds`).
            scale_up_period_seconds: Window over which `scale_up_percent`
                is applied.
            scale_down_percent: Per-period cap on scale-down rate, as a
                percentage of currentReplicas. Default 20.
            scale_down_period_seconds: Window over which
                `scale_down_percent` is applied. Default 300.
        """

        name: Required[str] = REQUIRED
        namespace: str = "default"

        scale_target_api_version: Required[str] = REQUIRED
        scale_target_kind: Required[str] = REQUIRED
        scale_target_name: Required[str] = REQUIRED

        min_replicas: int = 1
        max_replicas: int = 1

        metrics: list[HPAMetric.Config] = []

        scale_up_stabilization_seconds: Optional[int] = None
        scale_down_stabilization_seconds: Optional[int] = None

        scale_up_percent: int = 100
        scale_up_period_seconds: int = 60
        scale_down_percent: int = 20
        scale_down_period_seconds: int = 300

    @classmethod
    def define_flags(cls, fv: flags.FlagValues):
        super().define_flags(fv)
        common_kwargs = dict(flag_values=fv, allow_override=True)
        # HPA is auto-enabled when at least one `--scaling_metric_name` is
        # provided. `--min_replicas` / `--max_replicas` (defined by the
        # launcher) are reused as the HPA min/max bounds.
        flags.DEFINE_multi_string(
            "scaling_metric_name",
            None,
            "Metric name to drive autoscaling. Repeat for multiple metrics; "
            "each occurrence pairs with the i-th --scaling_metric_target_value "
            "(and optional --scaling_metric_type / --scaling_metric_target_type).",
            **common_kwargs,
        )
        flags.DEFINE_multi_string(
            "scaling_metric_target_value",
            None,
            "Target value for the corresponding --scaling_metric_name. "
            "Quantity string for AverageValue/Value (e.g. '100', '500m'); "
            "integer percent for Utilization (e.g. '80').",
            **common_kwargs,
        )
        flags.DEFINE_multi_string(
            "scaling_metric_type",
            None,
            "Metric source type: Pods | Resource | Object | External. "
            "Defaults to 'Pods' when omitted.",
            **common_kwargs,
        )
        flags.DEFINE_multi_string(
            "scaling_metric_target_type",
            None,
            "Metric target type: AverageValue | Value | Utilization. "
            "Defaults to 'AverageValue' when omitted.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "scaling_scale_up_stabilization_seconds",
            None,
            "Stabilization window for scale-up decisions, in seconds.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "scaling_scale_down_stabilization_seconds",
            None,
            "Stabilization window for scale-down decisions, in seconds.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "scaling_scale_up_percent",
            None,
            "Per-period cap on scale-up rate, as a percentage of "
            "currentReplicas. Default 100 (fleet may double each "
            "--scaling_scale_up_period_seconds).",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "scaling_scale_up_period_seconds",
            None,
            "Window over which --scaling_scale_up_percent is applied. Default 60.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "scaling_scale_down_percent",
            None,
            "Per-period cap on scale-down rate, as a percentage of currentReplicas. Default 20.",
            **common_kwargs,
        )
        flags.DEFINE_integer(
            "scaling_scale_down_period_seconds",
            None,
            "Window over which --scaling_scale_down_percent is applied. Default 300.",
            **common_kwargs,
        )

    @classmethod
    def from_flags(cls, fv: flags.FlagValues, **kwargs) -> "HPA.Config":
        """Populate an `HPA.Config` from `--scaling_*` flags.

        Reads metric names / targets positionally (the i-th occurrence of
        each flag describes the i-th metric). `metric_name` is taken as
        the bare name — the metrics-collection backend is responsible
        for transforming it (e.g. into a Stackdriver type path) before
        HPA queries the metrics adapter.

        Reuses `--min_replicas` / `--max_replicas` (defined by the launcher)
        for the HPA bounds when present.

        Returns the prebuilt cfg unchanged when no `--scaling_metric_name`
        is provided — the caller is responsible for deciding whether to
        actually instantiate / execute an HPA without metrics.
        """
        cfg: HPA.Config = super().from_flags(fv, **kwargs)

        metric_names = list(fv.scaling_metric_name or [])
        if not metric_names:
            return cfg

        target_values = list(fv.scaling_metric_target_value or [])
        if len(target_values) != len(metric_names):
            raise ValueError(
                f"--scaling_metric_target_value count ({len(target_values)}) must "
                f"match --scaling_metric_name count ({len(metric_names)})."
            )
        metric_types = list(fv.scaling_metric_type or [])
        target_types = list(fv.scaling_metric_target_type or [])
        if metric_types and len(metric_types) != len(metric_names):
            raise ValueError(
                f"--scaling_metric_type count ({len(metric_types)}) must match "
                f"--scaling_metric_name count ({len(metric_names)}) when set."
            )
        if target_types and len(target_types) != len(metric_names):
            raise ValueError(
                f"--scaling_metric_target_type count ({len(target_types)}) must "
                f"match --scaling_metric_name count ({len(metric_names)}) when set."
            )

        cfg.metrics = [
            HPAMetric.default_config().set(
                type=metric_types[i] if metric_types else "Pods",
                metric_name=metric_names[i],
                target_type=target_types[i] if target_types else "AverageValue",
                target_value=str(target_values[i]),
            )
            for i in range(len(metric_names))
        ]
        # --min_replicas / --max_replicas are defined by the launcher, not
        # HPA itself, so they may be absent (e.g. in tests or non-launcher
        # contexts).
        cfg.min_replicas = getattr(fv, "min_replicas", None) or 1
        cfg.max_replicas = getattr(fv, "max_replicas", None) or 1
        # Tuning flags default to None (see define_flags): an unset flag is
        # omitted from the forwarded bastion command, and the Config field
        # keeps its own default — the single source of truth. Only override
        # when the flag was explicitly provided.
        if fv.scaling_scale_up_stabilization_seconds is not None:
            cfg.scale_up_stabilization_seconds = fv.scaling_scale_up_stabilization_seconds
        if fv.scaling_scale_down_stabilization_seconds is not None:
            cfg.scale_down_stabilization_seconds = fv.scaling_scale_down_stabilization_seconds
        if fv.scaling_scale_up_percent is not None:
            cfg.scale_up_percent = fv.scaling_scale_up_percent
        if fv.scaling_scale_up_period_seconds is not None:
            cfg.scale_up_period_seconds = fv.scaling_scale_up_period_seconds
        if fv.scaling_scale_down_percent is not None:
            cfg.scale_down_percent = fv.scaling_scale_down_percent
        if fv.scaling_scale_down_period_seconds is not None:
            cfg.scale_down_period_seconds = fv.scaling_scale_down_period_seconds
        return cfg

    def execute(self, *, owner: Optional[dict] = None) -> None:
        cfg: HPA.Config = self.config
        if not cfg.metrics:
            raise ValueError(f"HPA {cfg.namespace}/{cfg.name} requires at least one metric.")

        spec: dict = {
            "scaleTargetRef": {
                "apiVersion": cfg.scale_target_api_version,
                "kind": cfg.scale_target_kind,
                "name": cfg.scale_target_name,
            },
            "minReplicas": cfg.min_replicas,
            "maxReplicas": cfg.max_replicas,
            "metrics": [m.instantiate().build_spec() for m in cfg.metrics],
        }
        behavior: dict = {
            # Cap scale-up velocity to scale_up_percent of currentReplicas
            # per scale_up_period_seconds. Defaults to 100% / 60s — the
            # fleet may double each minute. Percent applies to
            # currentReplicas, so this is exponential. The pre-provisioner
            # / GKE will throttle the actual provisioning rate.
            "scaleUp": {
                "policies": [
                    {
                        "type": "Percent",
                        "value": cfg.scale_up_percent,
                        "periodSeconds": cfg.scale_up_period_seconds,
                    },
                ],
            },
            # Cap scale-down velocity to scale_down_percent of currentReplicas
            # per scale_down_period_seconds. Defaults to 20% / 300s. Slower
            # than scale-up so a transient drop in metrics can't tear down
            # the workload all at once. The cluster autoscaler /
            # pre-provisioner reclaims node pools as replicas are removed.
            "scaleDown": {
                "policies": [
                    {
                        "type": "Percent",
                        "value": cfg.scale_down_percent,
                        "periodSeconds": cfg.scale_down_period_seconds,
                    },
                ],
            },
        }
        if cfg.scale_up_stabilization_seconds is not None:
            behavior["scaleUp"]["stabilizationWindowSeconds"] = cfg.scale_up_stabilization_seconds
        if cfg.scale_down_stabilization_seconds is not None:
            behavior["scaleDown"]["stabilizationWindowSeconds"] = (
                cfg.scale_down_stabilization_seconds
            )
        spec["behavior"] = behavior

        body = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": cfg.name, "namespace": cfg.namespace},
            "spec": spec,
        }
        if owner is not None:
            body["metadata"]["ownerReferences"] = [owner]

        api = k8s.client.AutoscalingV2Api()
        try:
            api.create_namespaced_horizontal_pod_autoscaler(namespace=cfg.namespace, body=body)
            logging.info(
                "Created HPA %s/%s targeting %s/%s (replicas %d..%d, %d metric(s))",
                cfg.namespace,
                cfg.name,
                cfg.scale_target_kind,
                cfg.scale_target_name,
                cfg.min_replicas,
                cfg.max_replicas,
                len(cfg.metrics),
            )
        except k8s.client.ApiException as e:
            if e.status != 409:
                raise
            logging.info("HPA %s/%s already exists; skipping", cfg.namespace, cfg.name)
