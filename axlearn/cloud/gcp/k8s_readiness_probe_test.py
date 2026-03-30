# Copyright © 2025 Apple Inc.

"""Tests for the ReadinessProbe class."""

import pytest
from absl import flags

from axlearn.cloud.common.utils import define_flags, from_flags
from axlearn.cloud.gcp.k8s_readiness_probe import ReadinessProbe


def _make_probe(**kwargs) -> ReadinessProbe:
    return ReadinessProbe.default_config().set(**kwargs).instantiate()


# --- build_readiness_probe tests ---


def test_grpc_port_only():
    assert _make_probe(grpc_port=8080).build_readiness_probe() == {"grpc": {"port": 8080}}


def test_grpc_with_service():
    assert _make_probe(grpc_port=8080, grpc_service="svc").build_readiness_probe() == {
        "grpc": {"port": 8080, "service": "svc"}
    }


def test_grpc_all_fields():
    result = _make_probe(
        grpc_port=8080,
        grpc_service="my-svc",
        initial_delay_seconds=5,
        period_seconds=10,
        timeout_seconds=3,
        success_threshold=1,
        failure_threshold=3,
    ).build_readiness_probe()
    assert result == {
        "grpc": {"port": 8080, "service": "my-svc"},
        "initialDelaySeconds": 5,
        "periodSeconds": 10,
        "timeoutSeconds": 3,
        "successThreshold": 1,
        "failureThreshold": 3,
    }


def test_http_port_only():
    assert _make_probe(http_port=8080).build_readiness_probe() == {"httpGet": {"port": 8080}}


def test_http_all_fields():
    result = _make_probe(
        http_port=9090,
        http_path="/healthz",
        http_scheme="HTTPS",
        initial_delay_seconds=10,
        period_seconds=15,
        timeout_seconds=5,
        success_threshold=2,
        failure_threshold=4,
    ).build_readiness_probe()
    assert result == {
        "httpGet": {"port": 9090, "path": "/healthz", "scheme": "HTTPS"},
        "initialDelaySeconds": 10,
        "periodSeconds": 15,
        "timeoutSeconds": 5,
        "successThreshold": 2,
        "failureThreshold": 4,
    }


def test_none_fields_omitted():
    """Only grpc_port and period_seconds set; other timing fields must be absent."""
    result = _make_probe(grpc_port=8080, period_seconds=10).build_readiness_probe()
    assert "periodSeconds" in result
    assert "initialDelaySeconds" not in result
    assert "timeoutSeconds" not in result
    assert "successThreshold" not in result
    assert "failureThreshold" not in result


def test_both_ports_raises():
    with pytest.raises(ValueError):
        _make_probe(grpc_port=8080, http_port=8080).build_readiness_probe()


def test_neither_port_raises():
    with pytest.raises(ValueError):
        _make_probe().build_readiness_probe()


# --- is_configured tests ---


def test_is_configured_grpc():
    assert _make_probe(grpc_port=8080).is_configured()


def test_is_configured_http():
    assert _make_probe(http_port=9000).is_configured()


def test_is_configured_none():
    assert not _make_probe().is_configured()


# --- define_flags / from_flags tests ---


def _flags_config(**flag_overrides) -> ReadinessProbe.Config:
    fv = flags.FlagValues()
    cfg = ReadinessProbe.default_config()
    define_flags(cfg, fv)
    for key, value in flag_overrides.items():
        fv[key].value = value
    fv.mark_as_parsed()
    return from_flags(cfg, fv)


def test_define_flags_registers_without_error():
    fv = flags.FlagValues()
    cfg = ReadinessProbe.default_config()
    define_flags(cfg, fv)
    fv.mark_as_parsed()
    assert "readiness_probe_grpc_port" in fv
    assert "readiness_probe_http_port" in fv
    assert "readiness_probe_http_path" in fv


def test_from_flags_grpc_port():
    assert _flags_config(readiness_probe_grpc_port=8080).grpc_port == 8080


def test_from_flags_http_fields():
    cfg = _flags_config(readiness_probe_http_port=9000, readiness_probe_http_path="/health")
    assert cfg.http_port == 9000
    assert cfg.http_path == "/health"


def test_from_flags_all_none():
    cfg = _flags_config()
    assert cfg.grpc_port is None
    assert cfg.grpc_service is None
    assert cfg.http_port is None
    assert cfg.http_path is None
    assert cfg.http_scheme is None
    assert cfg.initial_delay_seconds is None
    assert cfg.period_seconds is None
    assert cfg.timeout_seconds is None
    assert cfg.success_threshold is None
    assert cfg.failure_threshold is None
