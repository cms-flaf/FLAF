#!/usr/bin/env python3
"""The CRAB wave gate: when retries are held back, where they wait, and what frees them.

Ported from DSProd (`test/test_retry_release_and_budget.py`, classes `TestWaveGate` and
`TestRetryParking`). The numbers come from the DSProd Run3_2023BPix production: 4800 branches
reached 99.4 % in 68.4 h, and retries held back by the gate waited 11.35 h at the median --
roughly one job length per retry generation, ~10.5 h of the total -- because the gate weighed a
handful of retries against a wave they could never fill.

The proxy is FLAF's own `_FLAFCrabWorkflowProxy`, built by its real constructor over a real
FLAF workflow task (the MRO of the analysis tasks), and every submission round runs law's own
`submit()` and, where a resumed run is exercised, law's own `poll()`. Only what a runner does
not have is replaced: the configuration loader (`Setup.getGlobal`), the CRAB call itself
(`_submit_group`) and, in the poll, the grid credentials (taken as set up) and the Kerberos
renewal; the stall watchdog, which lists remote storage, is switched off by its own setting.

FLAF deliberately keeps law's `retries` and `tolerance` defaults (DSProd raised them); that
is pinned at the end, in place of DSProd's failure-budget tests.
"""

import contextlib
import itertools
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from collections import OrderedDict
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

try:
    import ROOT  # noqa: F401
except ImportError:
    # FLAF.Common imports ROOT at module level; nothing exercised here uses it, and the
    # unit-test runner has none
    sys.modules["ROOT"] = mock.MagicMock()

import law  # noqa: E402
import law.workflow.base  # noqa: E402
import law.workflow.remote  # noqa: E402
from law.job.dashboard import NoJobDashboard  # noqa: E402

import FLAF.run_tools.law_customizations as lc  # noqa: E402
from FLAF.RunKit.law_gfal import GFALFileInterface  # noqa: E402

#: the production the numbers in the module docstring come from
N_BRANCHES = 4800
#: `_CRAB_DEFAULT_PARALLEL_JOBS`, and the default `refill_fraction` 0.2 of it: a wave
#: needs 1000 jobs
PARALLEL_JOBS = 5000
MIN_WAVE = 1000
#: `_CRAB_DEFAULT_RETRY_RELEASE_MINUTES`
RELEASE_MINUTES = 45


class _WaveGateTask(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    """A FLAF workflow with the bases of the analysis tasks (e.g. AnaTupleMergeTask)."""

    bundle_flavours = ["flaf"]

    def create_branch_map(self):
        return {0: 0}

    def output(self):
        return law.LocalFileTarget(
            os.path.join(self.ana_data_path(), "products", f"{self.branch}.txt")
        )

    def run(self):
        pass


class _WaveGateDownstreamTask(
    lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow
):
    """A second FLAF workflow, to see what `req()` hands from one task to another."""

    create_branch_map = _WaveGateTask.create_branch_map
    output = _WaveGateTask.output
    run = _WaveGateTask.run


#: a fresh `version` per task, so luigi's instance cache never hands one test another's task
_versions = itertools.count()


def setUpModule():
    # the round is skipped when a job source cannot be read (SubmissionGuards); every source
    # must be readable here, or the gate would never be reached and every test below would
    # fail for an unrelated reason
    with mock.patch.dict(os.environ, {"FLAF_PATH": flaf_repo}):
        missing = lc.missing_job_source(retries=0)
    if missing is not None:
        raise RuntimeError(f"job source {missing} is not readable in this checkout")


def jobs(*job_nums):
    return OrderedDict((job_num, [job_num]) for job_num in job_nums)


class _FlafCrabCase(unittest.TestCase):
    """A real FLAF CRAB proxy over a real FLAF task, in a data area of its own."""

    #: the `crab:` block the task is built with (the watchdog is built with the proxy)
    crab_settings = {}

    def setUp(self):
        self.data_dir = tempfile.mkdtemp(prefix="flaf_wave_gate_")
        self.addCleanup(shutil.rmtree, self.data_dir, True)
        env = mock.patch.dict(
            os.environ,
            {"ANALYSIS_DATA_PATH": self.data_dir, "FLAF_PATH": flaf_repo},
        )
        env.start()
        self.addCleanup(env.stop)
        #: the `crab:` block of the configuration, read through the real `_crab_cfg`
        self.crab_cfg = dict(self.crab_settings)
        self.setup = types.SimpleNamespace(global_params={"crab": self.crab_cfg})
        # the configuration loader needs a full analysis checkout; the tasks here read
        # nothing but `global_params` from it
        loader = mock.patch.object(lc.Setup, "getGlobal", return_value=self.setup)
        loader.start()
        self.addCleanup(loader.stop)
        self.task = self.make_task()
        self.proxy = self.make_proxy()

    def make_task(self, cls=_WaveGateTask, **kwargs):
        kwargs.setdefault("version", f"v{next(_versions)}")
        kwargs.setdefault("period", "Run3_2022")
        kwargs.setdefault("workflow", "crab")
        return cls(**kwargs)

    def make_proxy(self, task=None):
        """A fresh proxy, in the state law's `_run_impl` leaves it in before submitting."""
        p = lc._FLAFCrabWorkflowProxy(task=task or self.task)
        p._set_parallel_jobs(PARALLEL_JOBS)
        p.dashboard = NoJobDashboard()
        # nothing has been produced, so no job is skippable and no storage is consulted
        p._existing_branches = set()
        p.get_cached_output()["jobs"].parent.touch()
        return p

    def submission_file(self, p=None):
        return (p or self.proxy).get_cached_output()["jobs"]

    def resume(self, task=None):
        """A second driver, reading the submission file the first one wrote.

        Loaded with the same calls law's `_run_impl` makes; law resubmits nothing itself on
        a resumed run, so this is the whole of what the new driver knows about parked work.
        """
        p = self.make_proxy(task)
        output = self.submission_file(p)
        p._submitted = output.exists()
        self.assertTrue(p._submitted, "the first driver wrote no submission file")
        p.job_data.update(output.load(formatter="json"))
        for data in p.job_data.jobs.values():
            data["job_id"] = p.job_manager.cast_job_id(data["job_id"])
        return p

    @staticmethod
    def crab_job_id(job_num):
        """A job id of the shape law's CRAB job manager hands out."""
        return lc.FLAFCrabJobManager.JobId(
            job_num, "260101_000000:user_flaf_wave_gate", "/crab_projects/wave_gate"
        )

    @contextlib.contextmanager
    def law_submission(self, p=None):
        """Run law's own `submit()` underneath the proxy, with only the CRAB call replaced.

        What a release depends on is law's slot arithmetic -- it fills `unsubmitted_jobs` in
        dict order and stops at `n_parallel`, so a release can be truncated -- and that is
        exercised here rather than described by a double. Yields the job numbers submitted.
        """
        p = p or self.proxy
        submitted = []

        def submit_group(submit_jobs, **kwargs):
            submitted.extend(submit_jobs)
            return (
                [self.crab_job_id(job_num) for job_num in submit_jobs],
                OrderedDict(
                    (job_num, {"job": "job.py", "config": {}, "log": None})
                    for job_num in submit_jobs
                ),
            )

        with mock.patch.object(type(p), "_submit_group", side_effect=submit_group):
            yield submitted


class TestWaveGate(_FlafCrabCase):
    """The decision table of `_should_submit_crab_group`, at a 1000-job wave in 5000 slots."""

    def decide(self, n_backlog, n_retry, n_active, parked_min_ago=None):
        self.proxy.poll_data.n_active = n_active
        self.proxy._retry_parked_since = (
            None if parked_min_ago is None else time.monotonic() - parked_min_ago * 60
        )
        return self.proxy._should_submit_crab_group(n_backlog, n_retry)

    def test_the_constructor_sets_the_crab_wave(self):
        # the numbers every other test is written against: no parallel_jobs on the CLI or in
        # the configuration gives the CRAB default, and no clock is running yet
        p = lc._FLAFCrabWorkflowProxy(task=self.make_task())
        self.assertEqual(p.poll_data.n_parallel, PARALLEL_JOBS)
        self.assertEqual(p._crab_refill_fraction() * PARALLEL_JOBS, MIN_WAVE)
        self.assertEqual(p._crab_retry_release_minutes(), RELEASE_MINUTES)
        self.assertIsNone(p._retry_parked_since)

    def test_nothing_waiting_is_not_held_back(self):
        self.assertTrue(self.decide(n_backlog=0, n_retry=0, n_active=N_BRANCHES))

    def test_a_full_wave_of_backlog_with_room_goes_out(self):
        self.assertTrue(self.decide(n_backlog=N_BRANCHES, n_retry=0, n_active=0))
        self.assertTrue(self.decide(n_backlog=MIN_WAVE, n_retry=0, n_active=0))

    def test_a_full_wave_without_room_waits(self):
        # 500 free slots cannot take a 1000-job wave, and the running jobs will free more
        self.assertFalse(self.decide(n_backlog=MIN_WAVE, n_retry=0, n_active=4500))

    def test_a_handful_of_retries_is_parked(self):
        # the incident: gating on free slots alone opened the gate from the first poll, since
        # 3270 of 5000 slots taken leaves 1730 free -- one CRAB task per retry generation
        self.assertFalse(self.decide(n_backlog=0, n_retry=5, n_active=3270))

    def test_a_wave_that_can_no_longer_be_filled_goes_out_at_once(self):
        # the tail: 5 retries and 100 running jobs can never reach 1000, so waiting for a
        # wave would only delay them -- this is what keeps a large production's tail short
        self.assertTrue(self.decide(n_backlog=0, n_retry=5, n_active=100))
        self.assertTrue(self.decide(n_backlog=5, n_retry=0, n_active=994))
        self.assertFalse(self.decide(n_backlog=5, n_retry=0, n_active=995))

    def test_a_small_production_is_never_batched(self):
        # 12 jobs can never fill a wave, so a failure there is resubmitted on the next poll
        self.assertTrue(self.decide(n_backlog=12, n_retry=0, n_active=0))
        self.assertTrue(self.decide(n_backlog=0, n_retry=1, n_active=11))

    def test_unlimited_parallelism_keeps_laws_behaviour(self):
        self.proxy.poll_data.n_parallel = self.proxy.n_parallel_max
        self.assertTrue(self.decide(n_backlog=0, n_retry=1, n_active=N_BRANCHES - 1))

    def test_a_parked_retry_waits_out_its_window(self):
        self.assertFalse(
            self.decide(n_backlog=5, n_retry=0, n_active=4795, parked_min_ago=10)
        )
        self.assertFalse(
            self.decide(n_backlog=5, n_retry=0, n_active=4795, parked_min_ago=44)
        )

    def test_a_parked_retry_is_released_when_the_window_is_up(self):
        # what the 11.35 h median parking cost: without this the 5 retries wait for a wave of
        # 1000 that only the next era could ever bring
        self.assertTrue(
            self.decide(n_backlog=5, n_retry=0, n_active=4795, parked_min_ago=46)
        )

    def test_the_timer_only_runs_for_parked_work(self):
        # a retry offered this poll has waited for nothing yet, however long the driver ran
        self.assertFalse(self.decide(n_backlog=0, n_retry=5, n_active=4795))

    def test_the_release_window_is_configurable(self):
        self.crab_cfg["retry_release_minutes"] = 5
        self.assertTrue(
            self.decide(n_backlog=5, n_retry=0, n_active=4795, parked_min_ago=6)
        )
        self.assertFalse(
            self.decide(n_backlog=5, n_retry=0, n_active=4795, parked_min_ago=4)
        )

    def test_an_unreadable_release_window_falls_back_to_the_default(self):
        # a nan or an inf passes `float()` and would then disable the release for ever, since
        # `waited >= nan` is never true; an empty yaml value arrives as None
        for value in ("soon", float("nan"), float("inf"), None, [45]):
            with self.subTest(value=value):
                self.crab_cfg["retry_release_minutes"] = value
                self.assertEqual(
                    self.proxy._crab_retry_release_minutes(), RELEASE_MINUTES
                )
                self.assertTrue(
                    self.decide(
                        n_backlog=5,
                        n_retry=0,
                        n_active=4795,
                        parked_min_ago=RELEASE_MINUTES + 1,
                    )
                )

    def test_the_size_bar_ignores_the_retries_offered_this_poll(self):
        # a whole generation of retries is parked first and goes out on the next poll as
        # backlog, one poll interval later, rather than opening the gate before it waited
        self.assertFalse(self.decide(n_backlog=0, n_retry=MIN_WAVE, n_active=3800))
        self.assertTrue(self.decide(n_backlog=MIN_WAVE, n_retry=0, n_active=3800))


class _ParkingCase(_FlafCrabCase):
    """Retry generations offered to the proxy the way law's poll() offers them."""

    def setUp(self):
        super().setUp()
        self.dumped = self.record_dumps(self.proxy)

    def record_dumps(self, p):
        """Keep law's real dump to the submission file, and note what each one held."""
        dumped = []
        dump = p.dump_job_data

        def recorded():
            dump()
            dumped.append(list(p.job_data.unsubmitted_jobs))

        p.dump_job_data = recorded
        return dumped

    def offer(self, *job_nums, p=None):
        """A retry generation, in the state law's poll() leaves it in before submit()."""
        p = p or self.proxy
        for job_num in job_nums:
            p.job_data.jobs[job_num] = p.job_data_cls.job_data(
                job_id=self.crab_job_id(job_num),
                branches=[job_num],
                status=p.job_manager.RETRY,
            )
            # law counts the attempt before it offers the job to submit()
            # (`law/workflow/remote.py`, poll), which is what marks it a retry from here on
            p.job_data.attempts[job_num] = p.job_data.attempts.get(job_num, 0) + 1
        return jobs(*job_nums)

    def park(self, *job_nums, n_active=None, p=None):
        p = p or self.proxy
        retry_jobs = self.offer(*job_nums, p=p)
        p.poll_data.n_active = (
            N_BRANCHES - len(job_nums) if n_active is None else n_active
        )
        return p.submit(retry_jobs)

    def add_backlog(self, job_nums, p=None):
        """Never-submitted branches, as law's `_run_impl` lays them out: no attempts entry."""
        (p or self.proxy).job_data.unsubmitted_jobs.update({n: [n] for n in job_nums})

    def wind_back(self, minutes, p=None):
        p = p or self.proxy
        p._retry_parked_since -= minutes * 60


class TestRetryParking(_ParkingCase):
    """Where a parked retry lives, in what order, and what frees it again."""

    def test_parked_retries_stay_in_the_job_data(self):
        # a proxy-side dict would orphan them on a driver kill and, since law snapshots
        # `len(job_data)` as the poll loop's n_jobs, drop them out of the production's total
        with self.law_submission() as submitted:
            result = self.park(1, 2, 3)
        self.assertEqual(dict(result), {})
        self.assertEqual(submitted, [])
        self.assertEqual(list(self.proxy.job_data.unsubmitted_jobs), [1, 2, 3])
        self.assertEqual(dict(self.proxy.job_data.jobs), {})
        self.assertEqual(len(self.proxy.job_data), 3)
        self.assertEqual(
            self.dumped, [[1, 2, 3]], "a killed driver reads the dump, not memory"
        )
        on_disk = self.submission_file().load(formatter="json")
        self.assertEqual(list(on_disk["unsubmitted_jobs"]), ["1", "2", "3"])
        self.assertEqual(on_disk["attempts"], {"1": 1, "2": 1, "3": 1})

    def test_a_fresh_generation_of_a_full_wave_is_parked_once(self):
        # the size bar measures only the backlog: a generation as large as a wave has still
        # waited for nothing, so it is parked on the poll that offers it and leaves on the
        # next one as backlog -- one poll interval later, in one CRAB task
        with self.law_submission() as submitted:
            self.park(*range(1, MIN_WAVE + 1), n_active=3800)
        self.assertEqual(submitted, [])
        self.assertEqual(len(self.proxy.job_data.unsubmitted_jobs), MIN_WAVE)
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(submitted, list(range(1, MIN_WAVE + 1)))
        self.assertEqual(dict(self.proxy.job_data.unsubmitted_jobs), {})
        self.assertIsNone(self.proxy._retry_parked_since)

    def test_the_tail_goes_out_on_the_poll_that_offers_it(self):
        # 5 retries and 100 running jobs can never fill a wave: nothing is parked and no
        # clock is started
        with self.law_submission() as submitted:
            self.park(1, 2, 3, 4, 5, n_active=100)
        self.assertEqual(submitted, [1, 2, 3, 4, 5])
        self.assertEqual(dict(self.proxy.job_data.unsubmitted_jobs), {})
        self.assertIsNone(self.proxy._retry_parked_since)

    def test_a_parked_retry_goes_out_when_its_window_is_up(self):
        self.park(1, 2, 3, 4, 5)
        self.proxy.poll_data.n_active = N_BRANCHES - 5
        self.wind_back(RELEASE_MINUTES - 1)
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(submitted, [], "released before its window was up")
        self.wind_back(2)
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(submitted, [1, 2, 3, 4, 5])
        self.assertEqual(dict(self.proxy.job_data.unsubmitted_jobs), {})
        self.assertIsNone(self.proxy._retry_parked_since)

    def test_parked_retries_go_in_front_of_the_backlog(self):
        # law's submit() fills the next wave from `unsubmitted_jobs` in dict order, so a retry
        # appended behind a large backlog is not reached and the release timer cannot free it
        self.add_backlog(range(100, 140))
        self.park(1, 2, n_active=N_BRANCHES - 42)
        self.assertEqual(list(self.proxy.job_data.unsubmitted_jobs)[:2], [1, 2])
        self.assertEqual(len(self.proxy.job_data.unsubmitted_jobs), 42)
        # and that is what a release with two free slots takes
        self.wind_back(RELEASE_MINUTES + 1)
        self.proxy.poll_data.n_active = PARALLEL_JOBS - 2
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(submitted, [1, 2])
        self.assertEqual(
            list(self.proxy.job_data.unsubmitted_jobs), list(range(100, 140))
        )

    def test_the_clock_starts_with_the_oldest_parked_retry(self):
        self.park(1, 2)
        self.wind_back(30)
        first = self.proxy._retry_parked_since
        self.park(3)
        self.assertEqual(self.proxy._retry_parked_since, first)
        self.assertEqual(list(self.proxy.job_data.unsubmitted_jobs), [3, 1, 2])

    def test_a_release_takes_this_polls_retries_along(self):
        # the retries offered by this poll are handed to law with the release, rather than
        # parked behind it for a window of their own
        self.park(1, n_active=N_BRANCHES - 1)
        self.wind_back(RELEASE_MINUTES + 1)
        retry_jobs = self.offer(4)
        with self.law_submission() as submitted:
            self.proxy.submit(retry_jobs)
        self.assertEqual(sorted(submitted), [1, 4])
        self.assertEqual(dict(self.proxy.job_data.unsubmitted_jobs), {})
        self.assertIsNone(self.proxy._retry_parked_since)

    def test_a_wave_that_takes_every_parked_retry_stops_the_clock(self):
        # 5000 free slots take the 2 parked retries and the backlog behind them, so nothing
        # is parked afterwards. A clock left running with nothing on it would open the gate on
        # every poll for the rest of the run, i.e. one CRAB task per polling interval.
        self.park(1, 2)
        self.add_backlog(range(100, 1100))
        self.proxy.poll_data.n_active = 0
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(len(submitted), 1002)
        self.assertEqual(submitted[:2], [1, 2])
        self.assertEqual(dict(self.proxy.job_data.unsubmitted_jobs), {})
        self.assertIsNone(self.proxy._retry_parked_since)

    def test_a_release_with_fewer_slots_than_retries_keeps_the_rest_on_a_clock(self):
        # law fills only up to `n_parallel`, so one free slot releases one of 400 parked
        # retries. Clearing the clock here would leave the other 399 with no clock at all:
        # free slots opening up later could not release them, only the size bar or the tail.
        self.park(*range(1, 401), n_active=PARALLEL_JOBS - 1)
        self.assertEqual(len(self.proxy.job_data), 400)
        self.wind_back(RELEASE_MINUTES + 1)
        released_at = time.monotonic()
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(submitted, [1])
        self.assertEqual(len(self.proxy.job_data.unsubmitted_jobs), 399)
        self.assertEqual(len(self.proxy.job_data), 400)
        self.assertIsNotNone(self.proxy._retry_parked_since)
        # a fresh window, not the expired one, so the next poll does not open the gate again
        self.assertGreaterEqual(self.proxy._retry_parked_since, released_at)
        self.proxy.poll_data.n_active = 3000
        self.assertFalse(self.proxy._should_submit_crab_group(399, 0))
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(submitted, [], "a second CRAB task on the very next poll")
        self.wind_back(RELEASE_MINUTES + 1)
        with self.law_submission() as submitted:
            self.proxy.submit()
        self.assertEqual(submitted, list(range(2, 401)))
        self.assertIsNone(self.proxy._retry_parked_since)

    def test_a_restarted_driver_releases_the_retries_it_finds_parked(self):
        # the case that costs the most: a driver dies about daily, and a resumed leg parks
        # nothing itself -- law calls submit() with an empty retry generation on every poll
        # -- so a clock that only parking can start never runs, leaving the retries to the
        # size bar
        self.park(1, 2, 3, n_active=3270)
        resumed = self.resume()
        self.assertEqual(list(resumed.job_data.unsubmitted_jobs), [1, 2, 3])
        self.assertEqual(dict(resumed.job_data.attempts), {1: 1, 2: 1, 3: 1})
        self.assertIsNone(resumed._retry_parked_since)

        resumed.poll_data.n_active = 3270
        with self.law_submission(resumed) as submitted:
            self.assertEqual(dict(resumed.submit()), {})
        self.assertEqual(submitted, [])
        self.assertIsNotNone(resumed._retry_parked_since)

        self.wind_back(RELEASE_MINUTES + 1, p=resumed)
        with self.law_submission(resumed) as submitted:
            resumed.submit()
        self.assertEqual(submitted, [1, 2, 3])
        self.assertEqual(dict(resumed.job_data.unsubmitted_jobs), {})
        self.assertEqual(dict(resumed.job_data.attempts), {1: 1, 2: 1, 3: 1})
        self.assertIsNone(resumed._retry_parked_since)

    def test_the_backlog_alone_never_starts_a_clock(self):
        # a never-submitted branch has no `attempts` entry and must not get a clock: the timer
        # would then break a large production into a CRAB task per release window, which is
        # the tiny-task problem the wave size exists to solve
        self.add_backlog(range(1, 500))
        self.proxy.poll_data.n_active = 4700
        with self.law_submission() as submitted:
            self.assertEqual(dict(self.proxy.submit()), {})
        self.assertEqual(submitted, [])
        self.assertIsNone(self.proxy._retry_parked_since)
        self.assertEqual(len(self.proxy.job_data.unsubmitted_jobs), 499)
        # nor does a restarted driver that finds only backlog in the submission file
        self.proxy.dump_job_data()
        resumed = self.resume()
        resumed.poll_data.n_active = 4700
        with self.law_submission(resumed) as submitted:
            resumed.submit()
        self.assertEqual(submitted, [])
        self.assertIsNone(resumed._retry_parked_since)

    def test_no_poll_never_holds_anything_back(self):
        # a --no-poll invocation resubmits failures exactly once and then returns, so a parked
        # job would not be offered again until someone runs the task anew
        task = self.make_task(no_poll=True)
        p = self.make_proxy(task)
        with self.law_submission(p) as submitted:
            self.park(7, n_active=3270, p=p)
        self.assertEqual(submitted, [7])
        self.assertEqual(dict(p.job_data.unsubmitted_jobs), {})
        self.assertIsNone(p._retry_parked_since)
        # the same generation is parked without --no-poll
        self.park(7, n_active=3270)
        self.assertEqual(list(self.proxy.job_data.unsubmitted_jobs), [7])

    def test_no_poll_submits_what_an_earlier_driver_parked(self):
        self.park(1, 2, 3, n_active=3270)
        resumed = self.resume(self.make_task(no_poll=True, version=self.task.version))
        resumed.poll_data.n_active = 3270
        with self.law_submission(resumed) as submitted:
            resumed.submit()
        self.assertEqual(submitted, [1, 2, 3])
        self.assertEqual(dict(resumed.job_data.unsubmitted_jobs), {})


class _PollWentOn(AssertionError):
    """The poll loop went on past the iterations in which the run had to stop."""


class TestBrakeBeforeGate(_ParkingCase):
    """A resumed run that lost most of its outputs stops; nothing may park them first.

    Holding jobs back returns without delegating to law, so a brake consulted only on the
    way to law would let a mass retry be parked here and released later, unseen.
    """

    #: jobs a first driver recorded as finished, and how many of them lost their outputs
    N_RECORDED = 2000
    N_LOST = 1500

    # the stall watchdog lists remote storage from the poll callback, and there is none here
    crab_settings = {"watchdog": {"enabled": False}}

    def resume_after_mass_loss(self):
        """A second driver, whose first driver recorded every job as finished.

        The outputs of the first `N_LOST` jobs are gone; those of the rest are found.
        """
        for job_num in range(1, self.N_RECORDED + 1):
            self.proxy.job_data.jobs[job_num] = self.proxy.job_data_cls.job_data(
                job_id=self.proxy.job_data.dummy_job_id,
                branches=[job_num],
                status=self.proxy.job_manager.FINISHED,
                code=0,
            )
        self.proxy.dump_job_data()
        resumed = self.resume()
        resumed._existing_branches = set(range(self.N_LOST + 1, self.N_RECORDED + 1))
        return resumed

    def poll(self, resumed, iterations, before_iteration=None):
        """Run law's own poll() on `resumed`, which must stop the run within `iterations`.

        law takes the snapshot of the recorded-finished jobs, turns those without outputs
        into "unknown job id" retries and offers them to submit(). `before_iteration(i)` runs
        at the start of the poll callback of iteration `i`, before that iteration submits.
        A submission, or an iteration past `iterations`, fails the test where it happens.
        """
        # the credentials are set up once per run (FLAF's setup_job_manager probes the CMSSW
        # sandbox, the VOMS proxy and MyProxy): taken as done, as law caches it
        resumed._job_manager_setup_kwargs = {}
        # what a luigi worker attaches to a running task; there is no scheduler to talk to
        self.task.scheduler_messages = None
        # an iteration past the expected ones must fail the test at once, not sleep out the
        # CRAB poll interval first
        self.task.poll_interval = 0
        callback = self.task.crab_poll_callback
        self.poll_iterations = 0

        def guarded(poll_data):
            if self.poll_iterations >= iterations:
                raise _PollWentOn(
                    f"the poll loop reached iteration {self.poll_iterations} instead of "
                    "stopping the run"
                )
            if before_iteration is not None:
                before_iteration(self.poll_iterations)
            self.poll_iterations += 1
            return callback(poll_data)

        self.task.crab_poll_callback = guarded

        def refuse(submit_jobs, **kwargs):
            # raised where it happens: a job id made up here would only fail the next
            # iteration's status query, for a reason of its own
            raise _PollWentOn(
                f"{len(submit_jobs)} jobs were submitted in iteration "
                f"{self.poll_iterations - 1} instead of stopping the run"
            )

        # the Kerberos renewal shells out to `kinit -R`, and the completeness check moves a
        # process-wide epoch that is put back afterwards
        with mock.patch.object(
            type(resumed), "_submit_group", side_effect=refuse
        ), mock.patch.object(lc, "update_kinit"), mock.patch.object(
            GFALFileInterface,
            "negatives_valid_after",
            GFALFileInterface.negatives_valid_after,
        ):
            with self.assertRaisesRegex(
                RuntimeError, rf"\b{self.N_LOST} of {self.N_RECORDED}\b"
            ):
                resumed.poll()
        self.assertIsNone(resumed._retry_parked_since)
        self.assertEqual(len(resumed.job_data), self.N_RECORDED)

    def test_a_resumed_mass_loss_raises_instead_of_parking(self):
        # with nothing running, a 1500-job generation is not a tail and has not waited, so
        # the gate would park it -- and release it as backlog on the next poll
        resumed = self.resume_after_mass_loss()
        self.poll(resumed, iterations=1)
        self.assertEqual(dict(resumed.job_data.unsubmitted_jobs), {})
        on_disk = self.submission_file(resumed).load(formatter="json")
        self.assertEqual(on_disk["unsubmitted_jobs"], {})

    # A mass retry offered while the software tree cannot be read would be parked by the
    # skipped round and released later as backlog, where the brake no longer sees it; the
    # brake therefore runs before the round is skipped (it needs no job source).
    def test_a_mass_loss_parked_by_a_skipped_round_still_stops_the_run(self):
        # the first round finds the software tree unreadable (an AFS token lapsing, an EOS
        # mount blinking) and is skipped; the tree is back for the next one
        resumed = self.resume_after_mass_loss()
        tree = os.environ["FLAF_PATH"]
        os.environ["FLAF_PATH"] = os.path.join(self.data_dir, "unmounted", "FLAF")

        def tree_is_back(iteration):
            if iteration == 1:
                os.environ["FLAF_PATH"] = tree

        self.poll(resumed, iterations=2, before_iteration=tree_is_back)

    def test_a_resumed_handful_of_losses_is_still_parked(self):
        # the brake judges the first generation only and only above `max_lost_fraction`: a
        # few lost outputs are ordinary retries, held back by the gate like any other
        resumed = self.resume_after_mass_loss()
        resumed._snapshot_recorded_finished()
        with self.law_submission(resumed) as submitted:
            self.park(1, 2, 3, 4, 5, n_active=3270, p=resumed)
        self.assertEqual(submitted, [])
        self.assertEqual(list(resumed.job_data.unsubmitted_jobs), [1, 2, 3, 4, 5])
        self.assertIsNotNone(resumed._retry_parked_since)


class TestLawFailureBudgetKept(_FlafCrabCase):
    """FLAF keeps law's `retries` and `tolerance`, and hands them on through `req()`.

    DSProd raised both for its production task and stopped `req()` from copying them; FLAF
    deliberately does neither, so a change to either is a decision to be made here.
    """

    def test_the_defaults_are_laws(self):
        params = dict(_WaveGateTask.get_params())
        self.assertIs(params["retries"], law.workflow.remote.BaseRemoteWorkflow.retries)
        self.assertIs(params["tolerance"], law.workflow.base.BaseWorkflow.tolerance)
        self.assertIs(params["acceptance"], law.workflow.base.BaseWorkflow.acceptance)
        self.assertEqual(self.task.retries, 5)
        self.assertEqual(self.task.tolerance, 0.0)
        self.assertEqual(self.task.acceptance, 1.0)

    def test_req_hands_them_on(self):
        upstream = self.make_task(retries=7, tolerance=0.25)
        params = _WaveGateDownstreamTask.req_params(upstream)
        self.assertEqual(params["retries"], 7)
        self.assertEqual(params["tolerance"], 0.25)
        downstream = _WaveGateDownstreamTask.req(upstream)
        self.assertEqual(downstream.retries, 7)
        self.assertEqual(downstream.tolerance, 0.25)


if __name__ == "__main__":
    unittest.main()
