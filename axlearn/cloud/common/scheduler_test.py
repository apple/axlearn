# Copyright © 2023 Apple Inc.

"""Tests job scheduler."""
# pylint: disable=unused-argument

from datetime import datetime, timedelta
from typing import Dict

from absl.testing import absltest, parameterized

from axlearn.cloud.common.quota import QuotaInfo
from axlearn.cloud.common.scheduler import (
    JobMetadata,
    JobQueue,
    JobScheduler,
    JobVerdict,
    ProjectJobSorter,
    ResourceLimitCalculator,
    Scheduler,
)
from axlearn.common.config import config_for_function


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
        for job_id, resources in job_queue:
            self.assertDictEqual(jobs[job_id].resources, resources, msg=job_id)


class ResourceLimitCalculatorTest(absltest.TestCase):
    """Tests ResourceLimitCalculator."""

    def test_proportional_quotas(self):
        calculator: ResourceLimitCalculator = ResourceLimitCalculator.default_config().instantiate()
        quotas = {"a": 0.4, "b": 0.2, "c": 0.1}
        self.assertDictEqual(
            # Quota will be allocated proportionally.
            {
                "a": 4,
                "b": 2,
                "c": 1,
            },
            # `quotas` do not have to add up to `limit`.
            calculator.calculate(limit=7, quotas=quotas, demands={"a": 100, "b": 100, "c": 100}),
        )
        self.assertDictEqual(
            {
                "a": 5,  # quota limit is rounded up in this case.
                "b": 2,
                "c": 1,
            },
            calculator.calculate(limit=8, quotas=quotas, demands={"a": 100, "b": 100, "c": 100}),
        )
        self.assertDictEqual(
            {
                "a": 5,  # quota limit is rounded up in this case.
                "b": 3,  # quota limit is rounded up in this case.
                "c": 1,
            },
            calculator.calculate(limit=9, quotas=quotas, demands={"a": 100, "b": 100, "c": 100}),
        )
        self.assertDictEqual(
            {
                "a": 6,  # quota limit is rounded up in this case.
                "b": 3,  # quota limit is rounded up in this case.
                "c": 1,
            },
            calculator.calculate(limit=10, quotas=quotas, demands={"a": 100, "b": 100, "c": 100}),
        )

    def test_unallocated_resources(self):
        calculator: ResourceLimitCalculator = ResourceLimitCalculator.default_config().instantiate()
        self.assertDictEqual(
            {
                "a": 1,
                "b": 1,
                "c": 2,  # An arbitrary project gets the remaining quota.
            },
            calculator.calculate(limit=4, quotas={}, demands={"a": 100, "b": 100, "c": 100}),
        )

    def test_extreme_cases(self):
        calculator: ResourceLimitCalculator = ResourceLimitCalculator.default_config().instantiate()
        # Empty demands.
        self.assertDictEqual(
            {}, calculator.calculate(limit=10, quotas={"a": 8, "b": 2}, demands={})
        )
        # Empty quota.
        self.assertDictEqual(
            {"a": 8, "b": 2}, calculator.calculate(limit=10, quotas={}, demands={"a": 8, "b": 2})
        )
        # Demand from one project only.
        self.assertDictEqual(
            # All resources will be allocated to the project.
            {"b": 10},
            calculator.calculate(limit=10, quotas={"a": 8, "b": 2}, demands={"b": 12}),
        )

    def test_spare_resources(self):
        calculator: ResourceLimitCalculator = ResourceLimitCalculator.default_config().instantiate()
        quotas = {"a": 8, "b": 4, "c": 2}
        self.assertDictEqual(
            {
                # When "a" doesn't use its full quota, its limit is equal to its demand.
                "a": 6,
                # The remaining quota are shared proportionally between "b" and "c".
                "b": 8,
                "c": 4,
            },
            calculator.calculate(limit=18, quotas=quotas, demands={"a": 6, "b": 100, "c": 100}),
        )
        quotas = {"a": 8, "b": 4, "c": 2, "d": 2}
        self.assertDictEqual(
            {
                # It may take multiple rounds to arrive at the final limits.
                # First, we divide the quota proportionally among b/c/d: {"b": 8, "c": 4, "d": 4}.
                # Next, the spare capacity of 2 from "b" is split among c and d.
                "b": 6,
                "c": 5,
                "d": 5,
            },
            calculator.calculate(limit=16, quotas=quotas, demands={"b": 6, "c": 10, "d": 10}),
        )

    def test_best_effort_demands(self):
        calculator: ResourceLimitCalculator = ResourceLimitCalculator.default_config().instantiate()
        quotas = {"a": 8, "b": 4, "c": 2}
        self.assertDictEqual(
            {
                "a": 4,
                # Projects "d" and "e" do not any quota, but since there are a spare capacity of
                # 14 left, it's proportionally divided.
                "d": 7,
                "e": 7,
            },
            calculator.calculate(limit=18, quotas=quotas, demands={"a": 4, "d": 100, "e": 100}),
        )


class SchedulerTest(absltest.TestCase):
    """Tests Scheduler."""

    def test_basics(self):
        sched: Scheduler = Scheduler.default_config().instantiate()
        resource_limits = {"tpu": 12}
        project_quotas = {"a": {"tpu": 6}, "b": {"tpu": 3}, "c": {"tpu": 1}}
        # With demands only from project "a".
        results = sched.schedule(
            resource_limits=resource_limits,
            project_quotas=project_quotas,
            project_jobs={"a": (("a1", {"tpu": 8}), ("a2", {"tpu": 2}), ("a3", {"tpu": 5}))},
        )
        self.assertDictEqual(
            {"a": {"tpu": 12}},
            results.project_limits,
        )
        self.assertDictEqual(
            {
                "a": {
                    "a1": JobVerdict(),
                    "a2": JobVerdict(),
                    "a3": JobVerdict(over_limits={"tpu"}),
                }
            },
            results.job_verdicts,
        )
        # With demands from both "a" and "b".
        results = sched.schedule(
            resource_limits=resource_limits,
            project_quotas=project_quotas,
            project_jobs={
                "a": (("a1", {"tpu": 8}), ("a2", {"tpu": 2}), ("a3", {"tpu": 5})),
                "b": (("b1", {"tpu": 6}), ("b2", {"tpu": 2})),
            },
        )
        self.assertDictEqual(
            {"a": {"tpu": 8}, "b": {"tpu": 4}},
            results.project_limits,
        )
        self.assertDictEqual(
            {
                "a": {
                    # "a1" uses 8 tpus and can fit into the limit.
                    "a1": JobVerdict(),
                    # "a2" and "a3" cannot fit into the limits.
                    "a2": JobVerdict(over_limits={"tpu"}),
                    "a3": JobVerdict(over_limits={"tpu"}),
                },
                "b": {
                    "b1": JobVerdict(over_limits={"tpu"}),
                    "b2": JobVerdict(),
                },
            },
            results.job_verdicts,
        )

    def test_multiple_resource_types(self):
        sched: Scheduler = Scheduler.default_config().instantiate()
        resource_limits = {"tpu": 12, "gpu": 9}
        project_quotas = {
            "a": {"tpu": 6, "gpu": 4},
            "b": {"tpu": 3, "gpu": 1},
            "c": {"tpu": 1, "gpu": 2},
        }
        # With demands from both "a" and "b".
        results = sched.schedule(
            resource_limits=resource_limits,
            project_quotas=project_quotas,
            project_jobs={
                "a": (("a1", {"tpu": 8, "gpu": 10}), ("a2", {"tpu": 2}), ("a3", {"gpu": 4})),
                "b": (("b1", {"tpu": 6}), ("b2", {"tpu": 2})),
            },
        )
        self.assertDictEqual(
            {"a": {"tpu": 8, "gpu": 9}, "b": {"tpu": 4, "gpu": 0}},
            results.project_limits,
        )
        self.assertDictEqual(
            {
                "a": {
                    # "a1" is over the GPU limit.
                    "a1": JobVerdict(over_limits={"gpu"}),
                    # "a2" and "a3" can fit into the limits.
                    "a2": JobVerdict(),
                    "a3": JobVerdict(),
                },
                "b": {
                    "b1": JobVerdict(over_limits={"tpu"}),
                    "b2": JobVerdict(),
                },
            },
            results.job_verdicts,
        )


def _mock_get_resource_limits(*args):
    del args
    return QuotaInfo(
        total_resources={"v4": 15, "v3": 8, "v5": 5},
        project_resources={
            "project1": {"v4": 10, "v5": 5},
            "project2": {"v4": 5, "v3": 5},
            "project3": {"v3": 3},
        },
    )


def mock_quota_config():
    return _mock_get_resource_limits


class TestJobScheduler(parameterized.TestCase):
    """Tests JobScheduler."""

    @parameterized.parameters([False, True])
    def test_init(self, dry_run: bool):
        cfg = JobScheduler.default_config().set(
            quota=config_for_function(mock_quota_config),
        )

        # Test initialization.
        sched: JobScheduler = cfg.instantiate()
        # pylint: disable-next=protected-access
        self.assertEqual(sched._quota(), _mock_get_resource_limits())

        # Test scheduling.
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
        results = sched.schedule(jobs, dry_run=dry_run)

        # Get verdicts by job name.
        job_verdicts: Dict[str, JobVerdict] = {
            job_name: verdict
            for project_verdicts in results.job_verdicts.values()
            for job_name, verdict in project_verdicts.items()
        }
        if dry_run:
            # All of the jobs should be scheduled, regardless.
            expected = {"a": True, "b": True, "c": True, "d": True, "e": True, "f": True}
        else:
            expected = {"a": False, "b": True, "c": True, "d": False, "e": True, "f": True}

        self.assertEqual(
            expected,
            {job_name: job_verdict.should_run() for job_name, job_verdict in job_verdicts.items()},
        )

    def test_leftover(self):
        quota_info = QuotaInfo(
            total_resources={"gpu": 12},
            project_resources={
                "project_a": {"gpu": 1},
                "project_b": {"gpu": 1},
                "project_c": {"gpu": 1},
            },
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
                creation_time=yesterday + timedelta(seconds=1),
                resources={"gpu": 5},
            )
            for proj in ("a", "b", "c")
        }
        results = sched.schedule(jobs)
        # Each project gets 4 GPUs as its resource limit.
        self.assertEqual(
            {f"project_{proj}": {"gpu": 4} for proj in ("a", "b", "c")},
            results.project_limits,
        )
        job_verdicts: Dict[str, JobVerdict] = {
            job_name: verdict
            for project_verdicts in results.job_verdicts.values()
            for job_name, verdict in project_verdicts.items()
        }
        # Two of the jobs should run, even though every job's demand exceeds the project limit.
        self.assertEqual(2, sum(verdict.should_run() for verdict in job_verdicts.values()))

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
        job_verdicts: Dict[str, JobVerdict] = {
            job_name: verdict
            for project_verdicts in results.job_verdicts.values()
            for job_name, verdict in project_verdicts.items()
        }
        expected = {
            # The first job of each project will get scheduled.
            "a1": True,
            "b1": True,
            "c1": True,
            # Only "b2" will get scheduled since it's older than "a2" and "c2".
            "a2": False,
            "b2": True,
            "c2": False,
            # Only "a3" will get scheduled since it can fit into the last gpu ("a2" cannot) and
            # it's older than "b3" and "c3".
            "a3": True,
            "b3": False,
            "c3": False,
        }
        self.assertEqual(
            expected,
            {job_name: job_verdict.should_run() for job_name, job_verdict in job_verdicts.items()},
        )
        self.assertEqual(
            # Each project gets 4 GPUs according to their quotas.
            {"project_a": {"gpu": 4}, "project_b": {"gpu": 4}, "project_c": {"gpu": 4}},
            results.project_limits,
        )
        self.assertEqual(
            # Actual usages can be higher than the limits due to left-over scheduling.
            {"project_a": {"gpu": 5}, "project_b": {"gpu": 6}, "project_c": {"gpu": 1}},
            results.project_usages,
        )
