#!/usr/bin/env python3
"""Unit tests for the AnaTuple cost model, job packing and entry-range partitioning."""

import json
import os
import sys
import tempfile
import unittest

# Parent of the FLAF repo so `import FLAF.AnaProd...` resolves.
flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.AnaProd.CostModel import (
    CostModel,
    chunk_bounds,
    entry_key,
    merged_params,
    pack_units,
    scaled_bounds,
)


class TestChunkBounds(unittest.TestCase):
    """The chunk partition is what makes split files reproduce the whole-file result."""

    def test_partition_is_exact(self):
        for n_total in (0, 1, 2, 7, 1000, 526062, 999983):
            for n_chunks in (1, 2, 3, 5, 8, 17):
                bounds = [chunk_bounds(n_total, i, n_chunks) for i in range(n_chunks)]
                self.assertEqual(bounds[0][0], 0)
                self.assertEqual(bounds[-1][1], n_total)
                for (_, end), (begin, _) in zip(bounds, bounds[1:]):
                    self.assertEqual(
                        end, begin, "chunks must not overlap or leave gaps"
                    )
                self.assertEqual(
                    sum(end - begin for begin, end in bounds),
                    n_total,
                    "every entry is covered exactly once",
                )

    def test_single_chunk_is_the_whole_file(self):
        self.assertEqual(chunk_bounds(1234, 0, 1), (0, 1234))

    def test_out_of_range_index_is_clamped(self):
        self.assertEqual(chunk_bounds(100, 9, 4), chunk_bounds(100, 3, 4))
        self.assertEqual(chunk_bounds(100, -3, 4), chunk_bounds(100, 0, 4))

    def test_degenerate_chunk_count(self):
        self.assertEqual(chunk_bounds(100, 0, 0), (0, 100))


class TestScaledBounds(unittest.TestCase):
    """The not-selected tree must be sliced by the same fractions, or the per-chunk
    denominators do not add back up to the whole-file denominators."""

    def test_partition_is_exact(self):
        n_selected, n_not_selected = 526062, 106595
        for n_chunks in (1, 2, 3, 7, 13):
            mapped = [
                scaled_bounds(
                    *chunk_bounds(n_selected, i, n_chunks), n_selected, n_not_selected
                )
                for i in range(n_chunks)
            ]
            self.assertEqual(mapped[0][0], 0)
            self.assertEqual(mapped[-1][1], n_not_selected)
            for (_, end), (begin, _) in zip(mapped, mapped[1:]):
                self.assertEqual(end, begin)
            self.assertEqual(sum(end - begin for begin, end in mapped), n_not_selected)

    def test_empty_source(self):
        self.assertEqual(scaled_bounds(0, 0, 0, 100), (0, 0))


class TestPackUnits(unittest.TestCase):
    CAPACITY = 6 * 3600.0

    def _units(self, costs, tier="job"):
        return [(i, cost, tier) for i, cost in enumerate(costs)]

    def test_every_unit_is_placed_exactly_once(self):
        units = self._units([37.0 * (i % 23) + 11.0 for i in range(500)])
        groups = pack_units(units, self.CAPACITY, 50)
        placed = [key for group in groups for key in group]
        self.assertEqual(sorted(placed), [u[0] for u in units])
        self.assertEqual(len(placed), len(set(placed)), "no unit may be duplicated")

    def test_no_group_exceeds_capacity(self):
        units = self._units([1000.0 * (i % 30 + 1) for i in range(300)])
        costs = {u[0]: u[1] for u in units}
        groups = pack_units(units, self.CAPACITY, 50)
        for group in groups:
            if len(group) == 1:
                continue  # a unit larger than the capacity is submitted alone
            total = sum(costs[k] for k in group)
            self.assertLessEqual(total, self.CAPACITY)

    def test_oversized_unit_gets_its_own_group(self):
        units = [(0, 10 * self.CAPACITY, "job"), (1, 60.0, "job"), (2, 60.0, "job")]
        groups = pack_units(units, self.CAPACITY, 50)
        self.assertIn([0], groups)

    def test_max_units_per_job_is_respected(self):
        units = self._units([1.0] * 400)
        groups = pack_units(units, self.CAPACITY, 7)
        self.assertTrue(all(len(g) <= 7 for g in groups))
        self.assertEqual(sum(len(g) for g in groups), 400)

    def test_uncertain_estimates_are_packed_more_conservatively(self):
        """A wrong prior must not be able to rebuild an over-long job."""
        known = pack_units(self._units([600.0] * 60, tier="job"), self.CAPACITY, 100)
        guessed = pack_units(
            self._units([600.0] * 60, tier="default"), self.CAPACITY, 100
        )
        self.assertGreater(
            len(guessed), len(known), "guessed costs must yield smaller groups"
        )
        safety = merged_params()["tier_safety"]["default"]
        for group in guessed:
            if len(group) > 1:
                self.assertLessEqual(len(group) * 600.0 * safety, self.CAPACITY)

    def test_expensive_groups_are_submitted_first(self):
        units = [(0, 60.0, "job"), (1, 5000.0, "job"), (2, 120.0, "job")]
        groups = pack_units(units, 6000.0, 1)
        self.assertEqual(groups[0], [1])

    def test_empty_input(self):
        self.assertEqual(pack_units([], self.CAPACITY, 10), [])


class TestCostModel(unittest.TestCase):
    NANO = "HLepRare"

    def _probe(
        self, loop_seconds=335.0, n_scanned=5000, n_total=575756, size=1071260150
    ):
        return {
            "ok": True,
            "loop_seconds": loop_seconds,
            "setup_seconds": 180.0,
            "n_scanned_events": n_scanned,
            "n_original_events": n_total,
            "input_size": size,
            "n_trees": 21,
            "era": "Run3_2022EE",
        }

    def test_probe_sets_the_per_event_rate(self):
        model = CostModel()
        self.assertTrue(model.set_from_probe(self.NANO, "DY", self._probe()))
        rate, tier = model.sec_per_event(self.NANO, "DY")
        self.assertAlmostEqual(rate, 335.0 / 5000.0)
        self.assertEqual(tier, "probe")

    def test_failed_probe_is_ignored(self):
        model = CostModel()
        self.assertFalse(model.set_from_probe(self.NANO, "DY", {"ok": False}))
        self.assertFalse(
            model.set_from_probe(self.NANO, "DY", dict(self._probe(), loop_seconds=0))
        )
        self.assertEqual(model.sec_per_event(self.NANO, "DY")[1], "default")

    def test_estimate_uses_catalogued_event_counts(self):
        model = CostModel()
        model.set_from_probe(self.NANO, "DY", self._probe())
        seconds, tier = model.estimate(
            self.NANO, "DY", {"n_events": 500000, "size": 1000000000}
        )
        self.assertEqual(tier, "probe")
        self.assertAlmostEqual(seconds, model.overhead_sec + 0.067 * 500000)

    def test_estimate_falls_back_to_file_size(self):
        model = CostModel()
        model.set_from_probe(self.NANO, "DY", self._probe())
        seconds, tier = model.estimate(self.NANO, "DY", {"size": 1071260150})
        # events_per_byte came from the probe, so the inferred count is the probed file's
        self.assertEqual(tier, "catalogue")
        self.assertAlmostEqual(seconds, model.overhead_sec + 0.067 * 575756, places=3)

    def test_estimate_without_any_metadata(self):
        model = CostModel()
        model.set_from_probe(self.NANO, "DY", self._probe())
        seconds, tier = model.estimate(self.NANO, "DY", None)
        self.assertEqual(tier, "default")
        self.assertGreater(seconds, model.overhead_sec)

    def test_tier_ladder(self):
        model = CostModel()
        model.set_from_probe(self.NANO, "DY_1J", self._probe())
        info = {"n_events": 100000}
        # a sibling of the same process borrows its rate
        _, tier = model.estimate(self.NANO, "DY_2J", info, peers=["DY_1J", "DY_2J"])
        self.assertEqual(tier, "process")
        # otherwise the process group
        _, tier = model.estimate(self.NANO, "TT", info, group_peers=["DY_1J", "TT"])
        self.assertEqual(tier, "group")
        # and finally the configured prior
        _, tier = model.estimate(self.NANO, "TT", info)
        self.assertEqual(tier, "default")

    def test_a_different_nano_source_is_not_reused(self):
        """The same dataset costs a very different amount per event as a channel skim
        and as central unskimmed NanoAOD."""
        model = CostModel()
        model.set_from_probe(self.NANO, "DY", self._probe())
        self.assertEqual(model.sec_per_event("v15", "DY")[1], "default")

    def test_measurement_overrides_the_probe_and_averages(self):
        model = CostModel(params={"overhead_sec": 300.0, "measurement_weight": 0.5})
        # setup_seconds equal to the configured overhead keeps the overhead estimate put,
        # so this test is about the averaging of the per-event rate alone
        model.set_from_probe(self.NANO, "DY", dict(self._probe(), setup_seconds=300.0))
        self.assertTrue(model.add_measurement(self.NANO, "DY", 300.0 + 1000.0, 10000))
        rate, tier = model.sec_per_event(self.NANO, "DY")
        self.assertEqual(tier, "job")
        self.assertAlmostEqual(rate, 0.1)
        # a second measurement is averaged in, it does not replace
        model.add_measurement(self.NANO, "DY", 300.0 + 3000.0, 10000)
        self.assertAlmostEqual(model.sec_per_event(self.NANO, "DY")[0], 0.2)

    def test_short_jobs_carry_no_signal(self):
        model = CostModel(params={"overhead_sec": 300.0})
        self.assertFalse(model.add_measurement(self.NANO, "DY", 60.0, 10000))
        self.assertFalse(model.add_measurement(self.NANO, "DY", 1000.0, 0))

    def test_events_per_byte_from_catalogue(self):
        model = CostModel()
        info = {
            "a.root": {"n_events": 200, "size": 1000},
            "b.root": {"n_events": 300, "size": 1000},
            "c.root": {"size": 1000},  # no count, must be ignored
        }
        self.assertTrue(model.set_events_per_byte_from_catalogue(self.NANO, "DY", info))
        self.assertAlmostEqual(model.events_per_byte(self.NANO, "DY"), 0.25)

    def test_round_trip(self):
        model = CostModel()
        model.set_from_probe(self.NANO, "DY", self._probe())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "cost_model.json")
            model.save(path)
            self.assertFalse(model.dirty)
            with open(path) as f:
                self.assertIn(entry_key(self.NANO, "DY"), json.load(f)["entries"])
            reloaded = CostModel.load(path)
            self.assertEqual(
                reloaded.sec_per_event(self.NANO, "DY"),
                model.sec_per_event(self.NANO, "DY"),
            )

    def test_missing_and_corrupt_stores_are_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                CostModel.load(os.path.join(tmp, "absent.json")).entries, {}
            )
            broken = os.path.join(tmp, "broken.json")
            with open(broken, "w") as f:
                f.write("{not json")
            self.assertEqual(CostModel.load(broken).entries, {})

    def test_user_params_override_defaults(self):
        params = merged_params({"target_job_hours": 2.0, "tier_safety": {"job": 1.5}})
        self.assertEqual(params["target_job_hours"], 2.0)
        self.assertEqual(params["tier_safety"]["job"], 1.5)
        self.assertEqual(params["tier_safety"]["default"], 4.0)


try:
    from FLAF.run_tools.law_customizations import (
        LawProxyState,
        _BundleAwareHTCondorWorkflowProxy,
    )

    HAVE_LAW = True
except Exception:  # law/luigi not importable outside the analysis environment
    HAVE_LAW = False
    LawProxyState = object


class _StubJobData:
    def __init__(self, jobs=None, unsubmitted=None):
        self.jobs = dict(jobs or {})
        self.unsubmitted_jobs = dict(unsubmitted or {})
        self.attempts = {}


class _StubJobDataCls:
    @staticmethod
    def job_data(branches=None, **kwargs):
        return {"branches": list(branches or []), "status": None}


class _StubTask:
    def __init__(self, costs, capacity=6 * 3600.0):
        self.costs = costs
        self.capacity = capacity
        self.messages = []

    def branch_cost_map(self):
        return self.costs

    def cost_params(self):
        return merged_params()

    def cost_capacity_seconds(self):
        return self.capacity

    def publish_message(self, msg):
        self.messages.append(msg)


class _StubProxy(LawProxyState):
    """Carries just the state the packing and retry helpers touch, so their bookkeeping
    can be exercised without a batch system."""

    def __init__(self, task, jobs=None, unsubmitted=None):
        import collections

        self.task = task
        self.job_data = _StubJobData(jobs, unsubmitted)
        self.job_data_cls = _StubJobDataCls
        self._job_retries = collections.defaultdict(int)
        self._skip_jobs = {}
        self._cost_max_job_num = 0
        self._cost_own_jobs = set()
        self._cost_poll_started = False
        self._cost_repack_pending = True

    def _cost_scheduling_enabled(self):
        return True

    _next_job_num = (
        _BundleAwareHTCondorWorkflowProxy._next_job_num if HAVE_LAW else None
    )
    _cost_repack_unsubmitted = (
        _BundleAwareHTCondorWorkflowProxy._cost_repack_unsubmitted if HAVE_LAW else None
    )


@unittest.skipUnless(HAVE_LAW, "law is not importable in this environment")
class TestProxyBookkeeping(unittest.TestCase):
    """The job-number bookkeeping is the risky part: `_can_skip_job`, `job_retries` and
    `attempts` are all keyed by job number, so a recycled number applies stale state."""

    def _repack(self, proxy):
        _BundleAwareHTCondorWorkflowProxy._cost_repack_unsubmitted(proxy)

    def test_repack_preserves_every_unsubmitted_branch(self):
        costs = {b: (600.0, "job") for b in range(40)}
        proxy = _StubProxy(
            _StubTask(costs),
            jobs={1: {"branches": [100]}},
            unsubmitted={2: list(range(0, 20)), 3: list(range(20, 40))},
        )
        self._repack(proxy)
        packed = sorted(b for g in proxy.job_data.unsubmitted_jobs.values() for b in g)
        self.assertEqual(packed, list(range(40)))
        self.assertNotIn(100, packed, "submitted jobs must not be touched")

    def test_repack_never_recycles_a_job_number(self):
        costs = {b: (600.0, "job") for b in range(40)}
        proxy = _StubProxy(
            _StubTask(costs),
            jobs={1: {"branches": [100]}, 7: {"branches": [101]}},
            unsubmitted={2: list(range(0, 20)), 3: list(range(20, 40))},
        )
        self._repack(proxy)
        new_nums = set(proxy.job_data.unsubmitted_jobs)
        self.assertFalse(new_nums & {1, 7}, "must not collide with submitted jobs")
        self.assertTrue(min(new_nums) > 7)
        self.assertFalse(set(proxy._skip_jobs), "stale skip verdicts must be dropped")

    def test_repack_number_survives_a_job_pushed_back_to_unsubmitted(self):
        """law moves a job out of `jobs` when a retry cannot be submitted, so the live
        dicts alone would let the highest number be handed out a second time."""
        costs = {b: (600.0, "job") for b in range(4)}
        proxy = _StubProxy(_StubTask(costs), unsubmitted={1: [0, 1], 2: [2, 3]})
        self._repack(proxy)
        first = set(proxy.job_data.unsubmitted_jobs)
        # law submits the highest one, it fails, and it is pushed back to unsubmitted
        top = max(first)
        branches = proxy.job_data.unsubmitted_jobs[top]
        proxy._job_retries[top] = 1
        proxy.job_data.attempts[top] = 1
        proxy.job_data.unsubmitted_jobs = {top: branches}
        proxy._cost_repack_pending = True
        self._repack(proxy)
        second = set(proxy.job_data.unsubmitted_jobs)
        self.assertFalse(
            second & first, "a number carrying stale retry state must not be reused"
        )

    def test_durations_are_harvested_once_per_finished_job(self):
        """Only single-branch jobs this process submitted, seen running then finished."""
        import law

        task = _StubTask({})
        recorded = []
        task.record_job_durations = lambda samples: (recorded.extend(samples) or False)
        proxy = _StubProxy(task)
        proxy._job_started_at = {}
        proxy._job_harvested = set()
        proxy._cost_own_jobs = {1, 2, 3, 4}
        proxy.job_manager = law.job.base.BaseJobManager
        proxy.job_data.jobs = {
            1: {"branches": [0], "status": "running"},
            2: {"branches": [2], "status": "pending"},
            3: {"branches": [3], "status": "finished"},
            4: {"branches": [4, 5], "status": "running"},
        }
        harvest = _BundleAwareHTCondorWorkflowProxy.harvest_job_durations
        harvest(proxy)
        # job 3 was never seen running, so its duration is unknown and not invented
        self.assertEqual(recorded, [])
        proxy.job_data.jobs[1]["status"] = "finished"
        proxy.job_data.jobs[4]["status"] = "finished"
        harvest(proxy)
        self.assertEqual(len(recorded), 1, "the multi-branch job yields no sample")
        self.assertEqual(recorded[0][0], [0])
        self.assertGreaterEqual(recorded[0][1], 0.0)
        harvest(proxy)
        self.assertEqual(len(recorded), 1, "a job is harvested only once")

    def test_durations_of_jobs_from_an_earlier_run_are_ignored(self):
        """A job already running when the workflow restarted would be timed from the
        restart, not from its real start."""
        import law

        task = _StubTask({})
        recorded = []
        task.record_job_durations = lambda samples: (recorded.extend(samples) or False)
        proxy = _StubProxy(task)
        proxy._job_started_at = {}
        proxy._job_harvested = set()
        proxy._cost_own_jobs = set()
        proxy.job_manager = law.job.base.BaseJobManager
        proxy.job_data.jobs = {9: {"branches": [0], "status": "running"}}
        harvest = _BundleAwareHTCondorWorkflowProxy.harvest_job_durations
        harvest(proxy)
        proxy.job_data.jobs[9]["status"] = "finished"
        harvest(proxy)
        self.assertEqual(recorded, [])

    def test_resumed_run_repacks_before_polling(self):
        """law calls submit() only for a fresh submission, so a resumed workflow would
        never be regrouped if poll() did not do it."""
        costs = {b: (600.0, "job") for b in range(40)}
        proxy = _StubProxy(_StubTask(costs), unsubmitted={2: list(range(40))})
        _BundleAwareHTCondorWorkflowProxy._cost_repack_once(proxy)
        self.assertGreater(len(proxy.job_data.unsubmitted_jobs), 1)
        self.assertFalse(proxy._cost_repack_pending)
        packed = sorted(b for g in proxy.job_data.unsubmitted_jobs.values() for b in g)
        self.assertEqual(packed, list(range(40)))
        # and it is not repeated once polling has started
        proxy._cost_poll_started = True
        proxy._cost_repack_pending = True
        before = dict(proxy.job_data.unsubmitted_jobs)
        _BundleAwareHTCondorWorkflowProxy._cost_repack_once(proxy)
        self.assertEqual(proxy.job_data.unsubmitted_jobs, before)

    def test_no_repacking_once_polling_started(self):
        """law's poll() snapshots the job count, so the number of jobs must not change
        after it starts."""
        costs = {b: (600.0, "job") for b in range(40)}
        proxy = _StubProxy(_StubTask(costs), unsubmitted={2: list(range(40))})
        proxy._cost_poll_started = True
        before = dict(proxy.job_data.unsubmitted_jobs)
        submit = _BundleAwareHTCondorWorkflowProxy.submit
        try:
            submit(proxy)
        except Exception:
            pass  # the stub has no batch system; only the repack decision is under test
        self.assertEqual(proxy.job_data.unsubmitted_jobs, before)


if __name__ == "__main__":
    unittest.main()
