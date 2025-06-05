# Copyright © 2023 Apple Inc.

"""Tests job scheduler."""
# pylint: disable=unused-argument

import collections
from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence, Union
from unittest import mock

from absl.testing import absltest, parameterized

from axlearn.cloud.common.quota import QuotaInfo
from axlearn.cloud.common.scheduler import (
    BaseScheduler,
    JobMetadata,
    JobQueue,
    JobScheduler,
    ProjectJobSorter,
    ReporterFn,
    ReportingScheduler,
    TierScheduler,
    _compute_total_limits,
    _job_verdict,
    _normalize_quotas,
    _recursively_to_dict,
    composite_reporter,
)
from axlearn.cloud.common.types import ResourceMap
from axlearn.common.config import ConfigOr, InstantiableConfig, config_for_function
from axlearn.common.test_utils import TestCase


class ProjectJobSorterTest(absltest.TestCase):
    """Tests ProjectJobSorter."""

    def test_basic(self):
        sorter: ProjectJobSorter = ProjectJobSorter.default_config().instantiate()
        yesterday = datetime.now() - timedelta(days=1)
        jobs = {
            "a1": JobMetadata(
                user_id="a",
                project_id="p1",
                priority=1,
                creation_time=yesterday + timedelta(seconds=1),
                resources={"tpu": 10},
            ),
            "a2": JobMetadata(
                user_id="a",
                project_id="p1",
                priority=2,
                creation_time=yesterday + timedelta(seconds=2),
                resources={"tpu": 20},
            ),
            "b3": JobMetadata(
                user_id="b",
                project_id="p1",
                priority=2,
                creation_time=yesterday + timedelta(seconds=3),
                resources={"tpu": 5},
            ),
            "b4": JobMetadata(
                user_id="b",
                project_id="p1",
                priority=2,
                creation_time=yesterday + timedelta(seconds=4),
                resources={"tpu": 3},
            ),
            "b5": JobMetadata(
                user_id="b",
                project_id="p1",
                priority=1,
                creation_time=yesterday + timedelta(seconds=1),
                resources={"tpu": 3},
            ),
        }
        job_queue: JobQueue = sorter.sort(jobs)
        self.assertSequenceEqual(
            [
                # Among priority=1, "a1" goes before "b5" because it is created earlier.
                "a1",
                # "b5" goes before "b3" because it has priority=1.
                "b5",
                # Among priority=2, user "b" has less usage than "a" at this point.
                # "b3" goes next because it's earlier than "b4".
                "b3",
                # "b4" goes next because user "b" still has less usage than "a" at this point.
                "b4",
                # "a2" goes last.
                "a2",
            ],
            [job_id for job_id, _ in job_queue],
        )
        for job_id, job_metadata in job_queue:
            self.assertDictEqual(jobs[job_id].resources, job_metadata.resources, msg=job_id)


def _mock_job_metadata(resources, creation_time=None):
    m = mock.MagicMock()
    m.resources = resources
    if creation_time:
        m.creation_time = creation_time
    return m


class UtilsTest(TestCase):
    def test_recursively_to_dict(self):
        x = dict(a=dict(b=[1, 2, 3]))
        self.assertEqual(x, _recursively_to_dict(x))

        y = collections.defaultdict(lambda: collections.defaultdict(list))
        y["a"]["b"].extend([1, 2, 3])
        self.assertEqual(x, _recursively_to_dict(y))

        # Test "None" as input.
        self.assertIsNone(_recursively_to_dict(None))

    @parameterized.parameters(
        dict(
            quotas={"a": {"tpu": 0.1, "gpu": 3}, "b": {"tpu": 0.3, "gpu": 1}},
            resource_limits={"tpu": 4, "gpu": 4},
            expected={"a": {"tpu": 1.0, "gpu": 3.0}, "b": {"tpu": 3.0, "gpu": 1.0}},
        ),
        # Test when some limits are missing.
        dict(
            quotas={"a": {"tpu": 0.1, "gpu": 3}, "b": {"tpu": 0.3, "gpu": 1}},
            resource_limits={"tpu": 4},
            expected={"a": {"tpu": 1.0, "gpu": 0.0}, "b": {"tpu": 3.0, "gpu": 0.0}},
        ),
        # Test when some quotas are missing.
        dict(
            quotas={"a": {"tpu": 0.1}, "b": {"tpu": 0.3, "gpu": 1}},
            resource_limits={"tpu": 4, "gpu": 4},
            expected={"a": {"tpu": 1.0}, "b": {"tpu": 3.0, "gpu": 4.0}},
        ),
        # Test when total quotas are 0.
        dict(
            quotas={"a": {"tpu": 0.1, "gpu": 0.0}, "b": {"tpu": 0, "gpu": 0.0}},
            resource_limits={"tpu": 4, "gpu": 0},
            expected={"a": {"tpu": 4.0, "gpu": 0.0}, "b": {"tpu": 0.0, "gpu": 0.0}},
        ),
    )
    def test_normalize_quotas(self, quotas: dict, resource_limits: dict, expected: dict):
        actual = _normalize_quotas(quotas, resource_limits)
        self.assertNestedAllClose(expected, _recursively_to_dict(actual))
        # Make sure they sum to total limits.
        for resource_type, value in resource_limits.items():
            self.assertAlmostEqual(
                value, sum(quota.get(resource_type, 0) for quota in actual.values())
            )

    @parameterized.parameters(
        # Some basic cases.
        dict(
            resources={"tpu": 1},
            limits={"tpu": 2},
            over_limits=None,
        ),
        dict(
            resources={"tpu": 3},
            limits={"tpu": 2},
            over_limits={"tpu"},
        ),
        # Multiple resource cases.
        dict(
            resources={"tpu": 1, "gpu": 2},
            limits={"tpu": 2, "gpu": 2},
            over_limits=None,
        ),
        dict(
            resources={"tpu": 3, "gpu": 2},
            limits={"tpu": 2, "gpu": 2},
            over_limits={"tpu"},
        ),
        # Missing limits cases.
        dict(
            resources={"tpu": 1, "gpu": 2},
            limits={"tpu": 2},
            over_limits={"gpu"},
        ),
        dict(
            resources={"tpu": 3},
            limits={},
            over_limits={"tpu"},
        ),
        dict(
            resources={},
            limits={},
            over_limits=None,
        ),
    )
    def test_job_verdict(self, resources: dict, limits: dict, over_limits: set):
        self.assertEqual(over_limits, _job_verdict(resources, limits).over_limits)

    @parameterized.parameters(
        dict(resource_limits=[], expected={}),
        dict(
            resource_limits=[{"v4": 1}, {"v3": 1}, {"v4": 1, "v3": 1}], expected={"v4": 2, "v3": 2}
        ),
    )
    def test_compute_total_limits(self, resource_limits: list, expected: dict):
        self.assertEqual(expected, _compute_total_limits(resource_limits))


class TierSchedulerTest(parameterized.TestCase):
    def test_validate_tiers(self):
        sched: TierScheduler = TierScheduler.default_config().instantiate()
        common_kwargs = dict(project_quotas={}, project_jobs={})
        with self.assertRaisesRegex(ValueError, "at least one tier"):
            sched.schedule(resource_limits=[], **common_kwargs)

        with self.assertRaisesRegex(ValueError, "sequence"):
            sched.schedule(resource_limits={"v4": 10, "v3": 3}, **common_kwargs)

        sched.schedule(resource_limits=[{"v4": 10, "v3": 3}], **common_kwargs)

    @parameterized.named_parameters(
        # Test a case where all jobs fit into tier 0.
        dict(
            testcase_name="all_tier_0",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 5})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 2})),
                    ("b2", _mock_job_metadata({"v4": 3})),
                ),
            },
            expected_project_limits={"a": {"v4": 5}, "b": {"v4": 5}},
            expected_verdicts={"a1": True, "b1": True, "b2": True},
            # The order of entries in `expected_tiers` should reflect the job priorities.
            # Here "b1" is ahead of "a1" because it requests fewer resources.
            expected_tiers={"b1": 0, "a1": 0, "b2": 0},
        ),
        # Test tie-break. Although both are requesting the same amount, "b" is requesting more
        # relative to its quota, so "a" will be prioritized.
        dict(
            testcase_name="tiebreak_by_relative_demand",
            project_jobs={
                "b": (("b1", _mock_job_metadata({"v4": 7})),),
                "a": (("a1", _mock_job_metadata({"v4": 7})),),
            },
            project_quotas={"b": {"v4": 5}, "a": {"v4": 10}},
            expected_project_limits={"a": {"v4": 7}, "b": {"v4": 7}},
            expected_verdicts={"a1": True, "b1": True},
            expected_tiers={"a1": 0, "b1": 1},
        ),
        # Test tie-break. Since both have the same quotas, we will tie-break using creation time.
        # In this case, a1 is created first.
        dict(
            testcase_name="tiebreak_by_creation_time",
            project_jobs={
                "a": (
                    (
                        "a1",
                        _mock_job_metadata({"v4": 7}, creation_time=datetime(1900, 1, 1, 0, 0, 0)),
                    ),
                ),
                "b": (
                    (
                        "b1",
                        _mock_job_metadata({"v4": 7}, creation_time=datetime(1900, 1, 2, 0, 0, 0)),
                    ),
                ),
            },
            project_quotas={"b": {"v4": 10}, "a": {"v4": 10}},
            expected_project_limits={"a": {"v4": 7}, "b": {"v4": 7}},
            expected_verdicts={"a1": True, "b1": True},
            expected_tiers={"a1": 0, "b1": 1},
        ),
        # Test when a higher priority job does not fit into tier 0, thus allowing a lower priority
        # job to schedule onto tier 0.
        dict(
            testcase_name="high_priority_forced_into_tier_1",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 12})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 2})),
                    ("b2", _mock_job_metadata({"v4": 1})),
                ),
            },
            expected_project_limits={"a": {"v4": 12}, "b": {"v4": 3}},
            expected_verdicts={"a1": True, "b1": True, "b2": True},
            expected_tiers={"b1": 0, "b2": 0, "a1": 1},
        ),
        # In this case, "a" is requesting much more relative to its quota than "b", so "b" gets to
        # go first (so "a" doesn't fit).
        dict(
            testcase_name="lower_demand_goes_first",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 13})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 4})),
                    ("b2", _mock_job_metadata({"v4": 2})),
                ),
            },
            expected_project_limits={"a": {"v4": 0}, "b": {"v4": 6}},
            expected_verdicts={"a1": False, "b1": True, "b2": True},
            expected_tiers={"b1": 0, "b2": 0, "a1": None},
        ),
        # Test that leftover resources from reserved tier are schedulable by subsequent tiers.
        dict(
            testcase_name="leftover_from_reserved_tier",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 7})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 1})),
                    ("b2", _mock_job_metadata({"v4": 7})),
                ),
            },
            expected_project_limits={"a": {"v4": 7}, "b": {"v4": 8}},
            expected_verdicts={"a1": True, "b1": True, "b2": True},
            # "b1" is ahead of "a1" since it requests fewer resource.
            expected_tiers={"b1": 0, "a1": 0, "b2": 1},
        ),
        # Test load balance.
        dict(
            testcase_name="load_balance",
            project_jobs={
                "a": tuple((f"a{i}", _mock_job_metadata({"v4": 1})) for i in range(3)),
                "b": tuple((f"b{i}", _mock_job_metadata({"v4": 1})) for i in range(3)),
                "c": tuple((f"c{i}", _mock_job_metadata({"v4": 1})) for i in range(3)),
            },
            resource_limits=[{"v4": 3}, {"v4": 2}],
            project_quotas={"a": {"v4": 5}, "b": {"v4": 5}, "c": {"v4": 5}},
            expected_project_limits={
                "a": {"v4": 2},
                "b": {"v4": 2},
                "c": {"v4": 1},
            },
            expected_verdicts={
                f"{project}{i}": f"{project}{i}" in ["a0", "b0", "c0", "a1", "b1"]
                for project in ["a", "b", "c"]
                for i in range(3)
            },
            expected_tiers={
                "a0": 0,
                "b0": 0,
                "c0": 0,
                "a1": 1,
                "b1": 1,
                # While the rest of the jobs are not scheduled, the order still reflects
                # their priorities.
                # Since "c" gets the least resource, its jobs take priority over those from a/b.
                "c1": None,
                "c2": None,
                "a2": None,
                "b2": None,
            },
        ),
        # Test projects with no quotas.
        dict(
            testcase_name="projects_with_no_quota",
            project_jobs={
                "a": tuple((f"a{i}", _mock_job_metadata({"v4": 1})) for i in range(3)),
                "b": tuple((f"b{i}", _mock_job_metadata({"v4": 1})) for i in range(3)),
                "c": tuple((f"c{i}", _mock_job_metadata({"v4": 1})) for i in range(3)),
            },
            resource_limits=[{"v4": 2}, {"v4": 1}, {"v4": 2}],
            # Note a and c both have no quotas (c has quotas for v3 only).
            project_quotas={"b": {"v4": 3}, "c": {"v3": 4}},
            expected_project_limits={
                "b": {"v4": 3},
                "a": {"v4": 1},
                "c": {"v4": 1},
            },
            expected_verdicts={
                f"{project}{i}": f"{project}{i}" in ["b0", "b1", "b2", "a0", "c0"]
                for project in ["a", "b", "c"]
                for i in range(3)
            },
            expected_tiers={
                # "b" has quota for "v4", so its jobs get priorities.
                "b0": 0,
                "b1": 0,
                "b2": 1,
                # "a" and "c" jobs are interleaved in scheduling.
                "a0": 2,
                "c0": 2,
                "a1": None,
                "c1": None,
                "a2": None,
                "c2": None,
            },
        ),
        # Test quotas of different scales.
        dict(
            testcase_name="quotas_of_different_scales",
            project_jobs={
                "a": (
                    ("a1", _mock_job_metadata({"v3": 1})),
                    ("a2", _mock_job_metadata({"v4": 1})),
                    ("a3", _mock_job_metadata({"v3": 1, "v4": 1})),
                ),
                "b": (
                    ("b1", _mock_job_metadata({"v3": 1})),
                    ("b2", _mock_job_metadata({"v4": 1})),
                    ("b3", _mock_job_metadata({"v3": 1, "v4": 1})),
                ),
            },
            resource_limits=[{"v3": 2}, {"v4": 2}, {"v3": 1, "v4": 1}],
            # Note v3 and v4 have different quota scales.
            project_quotas={
                "a": {"v3": 0.1, "v4": 2},  # Effectively {"v3": 1, "v4": 2}
                "b": {"v3": 0.2, "v4": 1},  # Effectively {"v3": 2, "v4": 1}
            },
            expected_project_limits={"a": {"v3": 2, "v4": 2}, "b": {"v3": 1, "v4": 1}},
            expected_verdicts={
                "b1": True,
                "a1": True,
                "a2": True,
                "b2": True,
                "a3": True,
                "b3": False,
            },
            expected_tiers={"b1": 0, "a1": 0, "a2": 1, "b2": 1, "a3": 2, "b3": None},
        ),
        # Test that we cannot exceed total limit.
        # Note that missing resource types implicitly have limit 0.
        dict(
            testcase_name="cannot_exceed_total_limit",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 1, "v3": 2})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 2})),
                    ("b2", _mock_job_metadata({"v4": 7})),
                ),
            },
            resource_limits=[{"v4": 3}],
            expected_project_limits={"a": {"v4": 0, "v3": 0}, "b": {"v4": 2}},
            expected_verdicts={"a1": False, "b1": True, "b2": False},
            expected_tiers={"b1": 0, "b2": None, "a1": None},
        ),
        # Test that we can accumulate across tiers. Jobs should schedule onto the final tier.
        dict(
            testcase_name="accumulation_across_tiers",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 1, "v3": 2})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 3})),
                    ("b2", _mock_job_metadata({"v4": 7})),
                ),
            },
            resource_limits=[{"v4": 1}, {"v4": 1}, {"v4": 1}],
            expected_project_limits={"a": {"v4": 0, "v3": 0}, "b": {"v4": 3}},
            expected_verdicts={"a1": False, "b1": True, "b2": False},
            # While both "a1" and "b2" are not scheduled, "b2" is ranked ahead of "a1" because
            # its demand/limit ratio is lower (there's no v3 resource, so a1's ratio is infinite).
            expected_tiers={"b1": 2, "b2": None, "a1": None},
        ),
        # Test that we can accumulate across tiers across resource types.
        dict(
            testcase_name="accumulation_across_tiers_resource_types",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 1, "v3": 2})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 3})),
                    ("b2", _mock_job_metadata({"v4": 7})),
                ),
            },
            resource_limits=[{"v3": 1}, {"v4": 1}, {"v3": 1}, {"v4": 3}],
            project_quotas={"a": {}, "b": {}},
            expected_project_limits={"a": {"v4": 1, "v3": 2}, "b": {"v4": 3}},
            expected_verdicts={"a1": True, "b1": True, "b2": False},
            expected_tiers={"a1": 2, "b1": 3, "b2": None},
        ),
        # Test that we acquire resources in reverse-tier-order.
        dict(
            testcase_name="reverse_tier_order",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 1})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 3})),
                    ("b2", _mock_job_metadata({"v4": 1})),
                ),
            },
            # b1 initially spans tier 0 and 1, but because it fits entirely into 1, only occupies 1
            # and allows b2 to schedule onto tier 0.
            resource_limits=[{"v4": 2}, {"v4": 3}],
            project_quotas={"a": {}, "b": {}},
            expected_project_limits={"a": {"v4": 1}, "b": {"v4": 4}},
            expected_verdicts={"a1": True, "b1": True, "b2": True},
            expected_tiers={"a1": 0, "b1": 1, "b2": 0},
        ),
        # Test that we acquire resources in reverse-tier-order (multiple resources).
        dict(
            testcase_name="reverse_tier_order_multi_resource",
            project_jobs={
                "a": (("a1", _mock_job_metadata({"v4": 1, "v3": 2})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 3})),
                    ("b2", _mock_job_metadata({"v4": 1})),
                ),
            },
            # a1 spans tier 0-2, but only acquires resources from 2.
            # This lets b1 schedule onto 0, and b2 onto 1.
            resource_limits=[{"v4": 3}, {"v4": 2}, {"v3": 2}],
            project_quotas={"a": {}, "b": {}},
            expected_project_limits={"a": {"v4": 1, "v3": 2}, "b": {"v4": 4}},
            expected_verdicts={"a1": True, "b1": True, "b2": True},
            expected_tiers={"a1": 2, "b1": 0, "b2": 1},
        ),
        # Test scheduling jobs with no demands.
        dict(
            testcase_name="jobs_with_no_demands",
            project_jobs={
                "a": (("a1", _mock_job_metadata({})),),
                "b": (
                    ("b1", _mock_job_metadata({"v4": 2})),
                    ("b2", _mock_job_metadata({})),
                ),
            },
            expected_project_limits={"a": {}, "b": {"v4": 2}},
            expected_verdicts={"a1": True, "b1": True, "b2": True},
            expected_tiers={"a1": 0, "b1": 0, "b2": 0},
        ),
        # Test a case where some resource types are invalid.
        dict(
            testcase_name="invalid_resource_types",
            project_jobs={
                "a": (
                    ("a1", _mock_job_metadata({"v4": 5})),
                    ("a2", _mock_job_metadata({"v4": 10})),
                ),
                "b": (
                    # b1 is unscheduable due to its demand on "unknown", but it will not prevent
                    # b2 from being scheduled.
                    ("b1", _mock_job_metadata({"unknown": 1})),
                    ("b2", _mock_job_metadata({"v4": 1})),
                ),
            },
            expected_project_limits={"a": {"v4": 5}, "b": {"unknown": 0, "v4": 1}},
            expected_verdicts={"a1": True, "a2": False, "b1": False, "b2": True},
            expected_tiers={"b2": 0, "a1": 0, "a2": None, "b1": None},
        ),
    )
    def test_schedule(
        self,
        *,
        project_jobs: dict,
        expected_project_limits: dict,
        expected_verdicts: dict,
        expected_tiers: dict,
        resource_limits: Optional[dict] = None,
        project_quotas: Optional[dict] = None,
    ):
        sched: TierScheduler = TierScheduler.default_config().instantiate()
        resource_limits = resource_limits or [{"v4": 10}, {"v4": 5}]
        project_quotas = project_quotas or {"a": {"v4": 10}, "b": {"v4": 5}}

        now = datetime.now()
        for jobs in project_jobs.values():
            for i, (_, job_metadata) in enumerate(reversed(jobs)):
                # Jobs ordered in terms of oldest to newest.
                job_metadata.creation_time = now - timedelta(seconds=i)

        results = sched.schedule(
            resource_limits=resource_limits,
            project_quotas=project_quotas,
            project_jobs=project_jobs,
        )
        # project_limits should reflect limits across tiers.
        self.assertEqual(expected_project_limits, results.project_limits)
        job_verdicts = results.job_verdicts
        # Check that verdicts are expected.
        self.assertEqual(
            expected_verdicts,
            {job_name: job_verdict.should_run() for job_name, job_verdict in job_verdicts.items()},
        )
        # Check that the tiers are expected.
        self.assertEqual(
            expected_tiers,
            {
                job_name: job_verdict.metadata.get("tier", None)
                for job_name, job_verdict in job_verdicts.items()
            },
        )
        # Check that the order of jobs in `job_verdicts` matches that in `expected_tiers`.
        self.assertEqual(list(job_verdicts.keys()), list(expected_tiers.keys()))

    @parameterized.parameters(
        {"unused_limits": None},
        {"unused_limits": [{"v4": 4}, {"v4": 8}]},
    )
    def test_schedule_result(self, unused_limits: Optional[Sequence[ResourceMap[int]]]):
        schedule_result = BaseScheduler.ScheduleResults(
            project_limits={"a": {"v4": 5}, "b": {"unknown": 0, "v4": 1}},
            project_usages={"a": {"v4": 5}, "b": {"unknown": 0, "v4": 1}},
            job_verdicts={"a1": True, "a2": False, "b1": False, "b2": True},
            unused_limits=_recursively_to_dict(unused_limits),
        )
        self.assertEqual(schedule_result.unused_limits, unused_limits)


def _mock_get_resource_limits(*args):
    del args
    return QuotaInfo(
        total_resources=[{"v4": 15, "v3": 8, "v5": 5}],
        project_resources={
            "project1": {"v4": 10, "v5": 5},
            "project2": {"v4": 5, "v3": 5},
            "project3": {"v3": 3},
        },
        project_membership={"project1": ["user1"], "project2": [], "project3": []},
    )


def mock_quota_config():
    return _mock_get_resource_limits


def _dummy_reporter_cfg_impl(**kwargs):
    pass


def _dummy_reporter_as_cfg() -> InstantiableConfig[ReporterFn]:
    """Returns a reporter as an instantiable config of ReporterFn."""

    def reporter_factory(dummy_arg) -> ReporterFn:
        # Using wrapper to make lookup of mocked function dynamic.
        def wrapper(**kwargs):
            _dummy_reporter_cfg_impl(**kwargs)

        return wrapper

    return config_for_function(reporter_factory).set(dummy_arg="dummy_arg_val")


def _dummy_reporter_fn_impl(**kwargs):
    pass


def _dummy_reporter_as_fn() -> ReporterFn:
    """Returns a reporter as a ReporterFn."""

    # A wrapper is needed because in the tests below, we need to mock _impl(),
    # and maybe_instantiate(MagicMockObj) would result in infinite loop.
    def wrapper(**kwargs):
        _dummy_reporter_fn_impl(**kwargs)

    return wrapper


class TestJobScheduler(parameterized.TestCase):
    """Tests JobScheduler."""

    @parameterized.product(
        dry_run=[False, True],
        # Make sure calling various reporter without mock works fine.
        reporter=[None, _dummy_reporter_as_cfg(), _dummy_reporter_as_fn()],
    )
    def test_init(
        self,
        dry_run: bool,
        reporter: Optional[ConfigOr[ReporterFn]],
    ):
        # Initial scheduler set up.
        quota = config_for_function(mock_quota_config)
        if reporter:
            cfg = JobScheduler.default_config().set(
                quota=quota,
                scheduler=ReportingScheduler.default_config().set(reporter=reporter),
            )
        else:
            cfg = JobScheduler.default_config().set(quota=quota)

        # Set up candidate jobs.
        yesterday = datetime.now() - timedelta(days=1)
        jobs = {
            # Should be deprioritized in favor of b, since it's using part of p2's v4 quota.
            "a": JobMetadata(
                user_id="a",
                project_id="project1",
                creation_time=yesterday + timedelta(seconds=1),
                resources={"v4": 12},
            ),
            # Should run since there's v4 capacity in p2 after a is pre-empted.
            "b": JobMetadata(
                user_id="b",
                project_id="project2",
                creation_time=yesterday + timedelta(seconds=2),
                resources={"v4": 5},
            ),
            # Should run, due to available v3 quota in p2 and p3.
            "c": JobMetadata(
                user_id="c",
                project_id="project2",
                creation_time=yesterday + timedelta(seconds=3),
                resources={"v3": 6},
            ),
            # Should not run -- the excess v5 quota allocated is only 2.5.
            "d": JobMetadata(
                user_id="d",
                project_id="project2",
                creation_time=yesterday + timedelta(seconds=4),
                resources={"v5": 4},
            ),
            # Should run -- within the 2.5 excess v5 quota.
            "e": JobMetadata(
                user_id="e",
                project_id="project3",
                creation_time=yesterday + timedelta(seconds=5),
                resources={"v5": 2},
            ),
            # Should run. Even though it has no project, there is excess v3 quota.
            "f": JobMetadata(
                user_id="f",
                project_id="",
                creation_time=yesterday + timedelta(seconds=1),
                resources={"v3": 2},
            ),
        }

        # Test initialization.
        sched: JobScheduler = cfg.instantiate()
        # pylint: disable-next=protected-access
        self.assertEqual(sched._quota(), _mock_get_resource_limits())

        # Test scheduling. Get schedule results.
        results = sched.schedule(jobs, dry_run=dry_run, verbosity=1)
        # Get expected results.
        if dry_run:
            # All of the jobs should be scheduled, regardless.
            expected_verdicts = {"a": True, "b": True, "c": True, "d": True, "e": True, "f": True}
            expected_unused_limits = None
        else:
            expected_verdicts = {"a": False, "b": True, "c": True, "d": False, "e": True, "f": True}
            expected_unused_limits = [{"v4": 10, "v3": 0, "v5": 3}]

        self.assertEqual(
            expected_verdicts,
            {
                job_name: job_verdict.should_run()
                for job_name, job_verdict in results.job_verdicts.items()
            },
        )
        self.assertEqual(expected_unused_limits, results.unused_limits)

    def test_leftover(self):
        quota_info = QuotaInfo(
            total_resources=[{"gpu": 12}],
            project_resources={
                "project_a": {"gpu": 1},
                "project_b": {"gpu": 1},
                "project_c": {"gpu": 1},
            },
            project_membership={},
        )
        cfg = JobScheduler.default_config().set(
            quota=config_for_function(lambda: lambda *args: quota_info),
        )

        # Test initialization.
        sched: JobScheduler = cfg.instantiate()
        # pylint: disable-next=protected-access
        self.assertEqual(sched._quota(), quota_info)

        yesterday = datetime.now() - timedelta(days=1)

        # Test basic left-over scheduling.
        jobs = {
            proj: JobMetadata(
                user_id=f"user_{proj}",
                project_id=f"project_{proj}",
                creation_time=yesterday + timedelta(seconds=-index),
                resources={"gpu": 5},
            )
            for index, proj in enumerate(["a", "b", "c"])
        }
        results = sched.schedule(jobs)
        job_verdicts = results.job_verdicts
        # Two of the older jobs should run, even though every job's demand exceeds the project
        # limit.
        self.assertEqual(
            {"a": False, "b": True, "c": True},
            {job_name: verdict.should_run() for job_name, verdict in job_verdicts.items()},
        )

        # Test more complicated scheduling.
        jobs = {}
        creation_time = {
            "a1": 1,
            "a2": 13,
            "a3": 21,
            "b1": 2,
            "b2": 11,
            "b3": 22,
            "c1": 3,
            "c2": 12,
            "c3": 23,
        }
        for proj in ("a", "b", "c"):
            # Each project has three jobs, with demand of 1, 5, 4 gpus each.
            jobs.update(
                {
                    f"{proj}1": JobMetadata(
                        user_id=f"user_{proj}",
                        project_id=f"project_{proj}",
                        creation_time=yesterday + timedelta(seconds=creation_time[f"{proj}1"]),
                        resources={"gpu": 1},
                    ),
                    f"{proj}2": JobMetadata(
                        user_id=f"user_{proj}",
                        project_id=f"project_{proj}",
                        creation_time=yesterday + timedelta(seconds=creation_time[f"{proj}2"]),
                        resources={"gpu": 5},
                    ),
                    f"{proj}3": JobMetadata(
                        user_id=f"user_{proj}",
                        project_id=f"project_{proj}",
                        creation_time=yesterday + timedelta(seconds=creation_time[f"{proj}3"]),
                        resources={"gpu": 4},
                    ),
                }
            )
        results = sched.schedule(jobs)

        # Get verdicts by job name.
        job_verdicts = results.job_verdicts
        expected = {
            # The first job of each project will get scheduled.
            "a1": True,
            "b1": True,
            "c1": True,
            # Only "b2" will get scheduled since it's older than "a2" and "c2".
            # "c2" cannot be scheduled because we will exceed the total resources.
            "a2": False,
            "b2": True,
            "c2": False,
            # At this point, "a2", "c3" and "b3" are at the front of their queues.
            # "a2" is oldest but is deprioritized because it uses 5 gpus, compared to 4 gpus for
            # "b3" and "c3", which would result in a larger "a" usage ratio.
            # "b3" is next oldest but "b" is already utilizing 6 gpus, compared to just 1 for "c".
            # Thus, we schedule "c3".
            "a3": False,
            "b3": False,
            "c3": True,
        }
        self.assertEqual(
            expected,
            {job_name: job_verdict.should_run() for job_name, job_verdict in job_verdicts.items()},
        )
        self.assertEqual(
            # Actual usages can be higher than the limits due to left-over scheduling.
            {"project_a": {"gpu": 1}, "project_b": {"gpu": 6}, "project_c": {"gpu": 5}},
            results.project_usages,
        )

    @parameterized.product(
        [
            # Test no reporter.
            {"reporter": None, "expect_report_as_cfg": False, "expect_report_as_fn": False},
            # Test using reporter without composition.
            {
                "reporter": _dummy_reporter_as_cfg(),
                "expect_report_as_cfg": True,
                "expect_report_as_fn": False,
            },
            {
                "reporter": _dummy_reporter_as_fn(),
                "expect_report_as_cfg": False,
                "expect_report_as_fn": True,
            },
            # Test using reporter with composition.
            {
                "reporter": [_dummy_reporter_as_cfg()],
                "expect_report_as_cfg": True,
                "expect_report_as_fn": False,
            },
            {
                "reporter": [_dummy_reporter_as_fn()],
                "expect_report_as_cfg": False,
                "expect_report_as_fn": True,
            },
            {
                "reporter": [_dummy_reporter_as_cfg(), _dummy_reporter_as_fn()],
                "expect_report_as_cfg": True,
                "expect_report_as_fn": True,
            },
        ],
        lazy_instantiate=[True, False],
    )
    def test_scheduler_result_reporter(
        self,
        reporter: Optional[Union[ConfigOr[ReporterFn], Sequence[ConfigOr[ReporterFn]]]],
        expect_report_as_cfg: bool,
        expect_report_as_fn: bool,
        lazy_instantiate: bool,
    ):
        quota = config_for_function(mock_quota_config)
        if reporter:
            if isinstance(reporter, Iterable):
                if lazy_instantiate:
                    reporter = config_for_function(composite_reporter).set(reporters=reporter)
                else:
                    reporter = composite_reporter(reporters=reporter)
            scheduler_config = ReportingScheduler.default_config().set(reporter=reporter)
            cfg = JobScheduler.default_config().set(quota=quota, scheduler=scheduler_config)
        else:
            cfg = JobScheduler.default_config().set(quota=quota)
        # Job details don't matter here.
        jobs = {
            "dummy_job": JobMetadata(
                user_id="e",
                project_id="project3",
                creation_time=datetime.now(),
                resources={"v5": 2},
            ),
        }

        with (
            mock.patch(f"{__name__}.TierScheduler.schedule") as mock_tier_schedule,
            mock.patch(f"{__name__}._dummy_reporter_cfg_impl") as mock_reporter_as_cfg,
            mock.patch(f"{__name__}._dummy_reporter_fn_impl") as mock_reporter_as_fn,
        ):
            scheduler_instance: JobScheduler = cfg.instantiate()
            scheduler_instance.schedule(jobs, verbosity=1)

            # Check that TierScheduler is triggered as inner scheduler.
            mock_tier_schedule.assert_called_once()
            # Check that customized reporter is triggered on-demand.
            if expect_report_as_cfg:
                mock_reporter_as_cfg.assert_called_once()
            else:
                mock_reporter_as_cfg.assert_not_called()

            if expect_report_as_fn:
                mock_reporter_as_fn.assert_called_once()
            else:
                mock_reporter_as_fn.assert_not_called()
