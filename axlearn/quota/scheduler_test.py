"""Unittests for scheduler.py."""
from datetime import datetime, timedelta

from absl.testing import absltest

from axlearn.quota.scheduler import (
    JobMetadata,
    JobQueue,
    JobVerdict,
    ProjectJobSorter,
    ResourceLimitCalculator,
    Scheduler,
)


class ProjectJobSorterTest(absltest.TestCase):
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
    def test_extreme_cases(self):
        calculator: ResourceLimitCalculator = ResourceLimitCalculator.default_config().instantiate()
        # Empty demands.
        self.assertDictEqual(
            {}, calculator.calculate(limit=10, quotas={"a": 8, "b": 2}, demands={})
        )
        # Empty quota.
        self.assertDictEqual(
            {}, calculator.calculate(limit=10, quotas={}, demands={"a": 8, "b": 2})
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
    def test_basics(self):
        scheduler: Scheduler = Scheduler.default_config().instantiate()
        resource_limits = {"tpu": 12}
        project_quotas = {"a": {"tpu": 6}, "b": {"tpu": 3}, "c": {"tpu": 1}}
        # With demands only from project "a".
        results = scheduler.schedule(
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
        results = scheduler.schedule(
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
        scheduler: Scheduler = Scheduler.default_config().instantiate()
        resource_limits = {"tpu": 12, "gpu": 9}
        project_quotas = {
            "a": {"tpu": 6, "gpu": 4},
            "b": {"tpu": 3, "gpu": 1},
            "c": {"tpu": 1, "gpu": 2},
        }
        # With demands from both "a" and "b".
        results = scheduler.schedule(
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


if __name__ == "__main__":
    absltest.main()
