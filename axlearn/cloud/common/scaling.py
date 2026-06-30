# Copyright © 2026 Apple Inc.

"""Utilities related to scaling."""

from collections.abc import Sequence

from axlearn.cloud.common.job_types import ResourceType, ScalingSpec, Topology


def aggregate_min_resources(scaling_specs: Sequence[ScalingSpec]) -> dict[ResourceType, int]:
    """Aggregates `min_replicas × resources_per_replica` across scaling specs."""
    totals: dict[ResourceType, int] = {}
    for spec in scaling_specs:
        for resource, count in spec.resources_per_replica.items():
            totals[resource] = totals.get(resource, 0) + count * spec.min_replicas
    return totals


def aggregate_topology(scaling_specs: Sequence[ScalingSpec]) -> list[Topology]:
    """Builds one `Topology(topology_per_replica, min_replicas)` per spec.

    Skips specs without a `topology_per_replica`. Returns `[]` when no spec
    carries a topology.
    """
    return [
        Topology(topology=spec.topology_per_replica, replicas=spec.min_replicas)
        for spec in scaling_specs
        if spec.topology_per_replica
    ]
