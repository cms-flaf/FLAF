#!/usr/bin/env python3
"""The CRAB backend must survive what a real production throws at it.

Each test here encodes a failure observed in the 115k-job DSProd CRAB production
(cms-flaf/DSProd #5, #7, #11, #16, #18, #19) or found while auditing FLAF against it:
an unreadable `crab status` response, retries escaping as tiny CRAB tasks, a blacklist
silently defeated by the whitelist, a worker deleting its own delegated proxy, resource
parameters leaking through req(), and a worker rebuilding a live bundle.

The later DSProd productions changed what several of them expect, and the tests follow:
a site is charged only for a failure carrying a job-level code (#23, #46), a parked retry
goes out once its release window is up and only the backlog is weighed against the wave
(#25), a submission round is skipped rather than lost when the software tree cannot be
read (#39), a lifted quarantine keeps the record that earned it (#40), the CRAB site list
is the CRIC Processing Site Name list and is refused when it is short (#42), and a
condition that must end the run is recorded for the poll callback instead of being raised
from a query, where law swallows it (#42; FLAF applies it to an unreadable status too).
The site list, the quarantine record and the other CRAB mechanics are covered in depth by
`test_crab_sites.py` and the other `test_crab_*.py` files.
"""

import contextlib
import importlib.util
import os
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

# law_customizations reaches ROOT only at import time (FLAF.Common.Setup -> Utilities) and
# nothing tested here calls into it; the unit-test runner has no ROOT, so an empty module
# stands in for it there.
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    sys.modules["ROOT"] = types.ModuleType("ROOT")

import law
import law.workflow.remote

from FLAF.run_tools import law_customizations as lc
from FLAF.run_tools.crab_sites import SiteStats, processing_sites, resolve_whitelist
from FLAF.RunKit import grid_helper_tasks

#: `_CRAB_DEFAULT_PARALLEL_JOBS`, and the default `refill_fraction` 0.2 of it: a wave
#: needs 1000 jobs
PARALLEL_JOBS = 5000
MIN_WAVE = 1000


def make_manager():
    return lc.FLAFCrabJobManager(
        sandbox_name="cmssw::CMSSW_14_0_0::arch=el9_amd64_gcc12"
    )


def make_crab_proxy(n_active=0, n_parallel=PARALLEL_JOBS, no_poll=False):
    """A FLAF CRAB proxy over law's real job data and FLAF's real job manager.

    Built with object.__new__: law's __init__ wants a real workflow task, and a FLAF one
    needs an analysis setup. The task stands in with what the proxy and law's submit()
    read from it, with no `crab:` settings, so the defaults apply. `dump_job_data` writes
    to the task's output, so it records what it would have written instead.
    """
    proxy = object.__new__(lc._FLAFCrabWorkflowProxy)
    proxy.task = types.SimpleNamespace(
        _crab_cfg=lambda: {},
        no_poll=no_poll,
        shuffle_jobs=False,
        append_retry_jobs=False,
        publish_message=lambda msg: None,
        forward_dashboard_event=lambda *args: None,
        crab_destination_info=lambda info: info,
    )
    proxy.poll_data = law.workflow.remote.PollData(
        n_parallel=n_parallel, n_finished_min=-1, n_failed_max=-1, n_active=n_active
    )
    proxy.job_data = law.workflow.remote.JobData()
    proxy.job_manager = make_manager()
    proxy.dashboard = None
    proxy._submitted = False
    proxy._skip_jobs = {}
    # nothing has been produced yet: no job is skippable and no storage is consulted
    proxy._existing_branches = set()
    # as __init__ leaves it: no retry is parked, so no release window is running
    proxy._retry_parked_since = None
    proxy.dumped = []
    proxy.dump_job_data = lambda: proxy.dumped.append(
        list(proxy.job_data.unsubmitted_jobs)
    )
    return proxy


def should_submit(n_backlog, n_retry, n_active, parked_min_ago=None):
    proxy = make_crab_proxy(n_active=n_active)
    if parked_min_ago is not None:
        proxy._retry_parked_since = time.monotonic() - parked_min_ago * 60
    return proxy._should_submit_crab_group(n_backlog, n_retry)


@contextlib.contextmanager
def law_submission(proxy):
    """Run law's own submit() underneath the proxy, with only the CRAB call replaced.

    Yields the job numbers law actually handed to CRAB.
    """
    submitted = []

    def submit_group(submit_jobs, **kwargs):
        submitted.extend(submit_jobs)
        return (
            [f"crab_{job_num}" for job_num in submit_jobs],
            OrderedDict(
                (job_num, {"job": "job.jdl", "config": {}, "log": None})
                for job_num in submit_jobs
            ),
        )

    with mock.patch.object(type(proxy), "_submit_group", side_effect=submit_group):
        yield submitted


class TestWaveGate(unittest.TestCase):
    """The gate must aggregate on jobs waiting, not on free slots.

    Waiting work comes in two parts (DSProd #25): the backlog in `unsubmitted_jobs`, which
    alone is weighed against the wave size, and the retries this poll offers, which have not
    waited for anything yet. A parked retry is not held for ever: it goes out once its
    release window (45 min by default) is up. Here a wave needs 1000 of 5000 slots.
    """

    def test_retry_trickle_is_held_in_part_filled_pool(self):
        # The DSProd incident: 3270 of 5000 slots taken, so 1730 slots free — the old
        # free-slot rule was permanently open and each poll's retry handful became its
        # own CRAB task. A handful of retries must be held, whether offered this poll or
        # parked by an earlier one whose release window still runs.
        self.assertFalse(should_submit(0, 5, n_active=3270))
        self.assertFalse(should_submit(5, 0, n_active=3270, parked_min_ago=10))

    def test_parked_trickle_goes_out_once_its_window_is_up(self):
        # held, but not until a wave it can never fill: that cost one job length per retry
        # generation, ~10.5 h of a 68.4 h DSProd production
        self.assertTrue(should_submit(5, 0, n_active=3270, parked_min_ago=46))

    def test_retries_offered_this_poll_do_not_count_towards_the_wave(self):
        # a whole generation of retries is parked once and goes out on the next poll as
        # backlog, rather than opening the gate before it has waited at all
        self.assertFalse(should_submit(0, MIN_WAVE, n_active=0))
        self.assertTrue(should_submit(MIN_WAVE, 0, n_active=0))

    def test_full_wave_with_room_submits(self):
        self.assertTrue(should_submit(3000, 0, n_active=0))

    def test_full_wave_without_room_is_held(self):
        # 5000 jobs waiting but only 500 slots free: no full wave can run yet.
        self.assertFalse(should_submit(5000, 0, n_active=4500))

    def test_tail_is_released(self):
        # Running + waiting can never fill a wave again — holding only delays the tail.
        self.assertTrue(should_submit(0, 5, n_active=300))

    def test_small_production_never_batches(self):
        self.assertTrue(should_submit(100, 0, n_active=0))

    def test_first_wave_of_large_production_submits(self):
        self.assertTrue(should_submit(20000, 0, n_active=0))

    def test_nothing_waiting_submits(self):
        self.assertTrue(should_submit(0, 0, n_active=3270))

    def test_unlimited_parallelism_keeps_law_behaviour(self):
        proxy = make_crab_proxy(n_active=0)
        proxy.poll_data.n_parallel = proxy.n_parallel_max
        self.assertTrue(proxy._should_submit_crab_group(0, 1))


class TestSubmitParking(unittest.TestCase):
    """Parked retries must move to unsubmitted without changing len(job_data).

    law's poll loop snapshots len(job_data) once; changing it mid-poll hangs the loop
    or ends it early. Every submission round first probes the files a job is built from
    (DSProd #39); FLAF_PATH points at this checkout, so that probe reads real files.
    """

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"FLAF_PATH": flaf_repo})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def offer_retry(proxy, job_num):
        """One job law hands back for retry: it counts the attempt before submit() sees it."""
        proxy.job_data.jobs[job_num] = law.workflow.remote.JobData.job_data(
            job_id="x", branches=[job_num], status="retry"
        )
        proxy.job_data.attempts[job_num] = 1
        return OrderedDict([(job_num, [job_num])])

    def test_parking_preserves_job_data_length(self):
        proxy = make_crab_proxy(n_active=3270)
        proxy.job_data.unsubmitted_jobs[9] = [9]
        retry_jobs = self.offer_retry(proxy, 7)

        n_before = len(proxy.job_data)
        with law_submission(proxy) as submitted:
            result = proxy.submit(retry_jobs=retry_jobs)

        self.assertEqual(dict(result), {})
        self.assertEqual(submitted, [])
        self.assertEqual(len(proxy.job_data), n_before)
        self.assertNotIn(7, proxy.job_data.jobs)
        # in front of the backlog: law fills a wave from it in dict order, and a retry
        # behind a large backlog would not be reached for hours
        self.assertEqual(list(proxy.job_data.unsubmitted_jobs), [7, 9])
        self.assertEqual(
            proxy.dumped[-1], [7, 9], "a killed driver reads the dump, not memory"
        )
        self.assertIsNotNone(
            proxy._retry_parked_since, "a parked retry must be on its release clock"
        )

    def test_no_poll_bypasses_the_gate(self):
        # a --no-poll invocation resubmits failures exactly once and then returns;
        # parking would silently skip that documented one-shot resubmission
        proxy = make_crab_proxy(n_active=3270, no_poll=True)
        retry_jobs = self.offer_retry(proxy, 7)
        with law_submission(proxy) as submitted:
            result = proxy.submit(retry_jobs=retry_jobs)
        self.assertEqual(submitted, [7])
        self.assertEqual(list(result), [7])
        self.assertEqual(proxy.job_data.jobs[7]["job_id"], "crab_7")
        self.assertEqual(
            dict(proxy.job_data.unsubmitted_jobs),
            {},
            "nothing may be parked under no_poll",
        )


class TestPollInterval(unittest.TestCase):
    """CRAB polls must default to 5 minutes even when HTCondor's param wins the MRO."""

    def apply(self, poll_interval, cfg=None, cli=False):
        proxy = object.__new__(lc._FLAFCrabWorkflowProxy)
        proxy.task = types.SimpleNamespace(
            poll_interval=poll_interval,
            _crab_cfg=lambda: cfg or {},
            get_task_family=lambda: "MyTask",
        )
        with mock.patch.object(lc, "_cli_has_param", return_value=cli):
            lc._FLAFCrabWorkflowProxy._apply_crab_poll_interval(proxy)
        return proxy.task.poll_interval

    def test_htcondor_default_is_replaced(self):
        htc_default = float(lc.HTCondorWorkflow.poll_interval._default)
        self.assertEqual(self.apply(htc_default), lc._CRAB_DEFAULT_POLL_INTERVAL)

    def test_explicit_value_is_kept(self):
        self.assertEqual(self.apply(7.0), 7.0)

    def test_yaml_wins_over_default(self):
        self.assertEqual(self.apply(2.0, cfg={"poll_interval": 3}), 3.0)

    def test_cli_wins_over_yaml(self):
        self.assertEqual(self.apply(2.0, cfg={"poll_interval": 3}, cli=True), 2.0)


class TestCliHasParam(unittest.TestCase):
    """An option addressed to one task must not disable the yaml value or the CRAB
    default for every other task in the graph."""

    def has(self, tokens, family="MyTask"):
        stub = types.SimpleNamespace(cmdline_args=tokens)
        with mock.patch.object(
            lc.luigi.cmdline_parser.CmdlineParser, "get_instance", return_value=stub
        ):
            return lc._cli_has_param("poll-interval", family)

    def test_bare_option_matches(self):
        self.assertTrue(self.has(["--poll-interval", "3"]))
        self.assertTrue(self.has(["--poll-interval=3"]))
        self.assertTrue(self.has(["--poll_interval", "3"]))

    def test_own_task_prefix_matches(self):
        self.assertTrue(self.has(["--MyTask-poll-interval", "3"]))
        self.assertTrue(self.has(["--MyTask-poll-interval=3"]))

    def test_other_task_prefix_does_not_match(self):
        self.assertFalse(self.has(["--OtherTask-poll-interval", "3"]))
        self.assertFalse(self.has(["--OtherTask-poll-interval=3"]))


class TestCostParallelJobs(unittest.TestCase):
    """The HTCondor cost scheduler must run through the shared CLI matcher — a
    dangling reference here crashed every HTCondor run at task init."""

    def apply(self, cost_enabled, cost_params=None, tokens=()):
        proxy = object.__new__(lc._BundleAwareHTCondorWorkflowProxy)
        proxy.task = types.SimpleNamespace(
            get_task_family=lambda: "MyTask",
            cost_params=lambda: cost_params or {},
        )
        proxy._cost_scheduling_enabled = lambda: cost_enabled
        proxy.poll_data = types.SimpleNamespace(n_parallel=1_000_000)
        proxy.n_parallel_max = 1_000_000
        applied = []
        proxy._set_parallel_jobs = lambda n: applied.append(n)
        stub = types.SimpleNamespace(cmdline_args=list(tokens))
        with mock.patch.object(
            lc.luigi.cmdline_parser.CmdlineParser, "get_instance", return_value=stub
        ):
            lc._BundleAwareHTCondorWorkflowProxy._apply_cost_parallel_jobs(proxy)
        return applied

    def test_disabled_cost_scheduling_returns_without_crashing(self):
        self.assertEqual(self.apply(False), [])

    def test_cost_parallel_jobs_applied(self):
        self.assertEqual(self.apply(True, {"parallel_jobs": 2000}), [2000])

    def test_cli_parallel_jobs_wins(self):
        self.assertEqual(
            self.apply(True, {"parallel_jobs": 2000}, ["--parallel-jobs", "5"]), []
        )

    def test_other_task_cli_flag_does_not_disable(self):
        self.assertEqual(
            self.apply(
                True, {"parallel_jobs": 2000}, ["--OtherTask-parallel-jobs", "5"]
            ),
            [2000],
        )


class TestResolveWhitelist(unittest.TestCase):
    """CRAB gives the whitelist precedence, so exclusions must be cut out of it.

    The glob-blacklist cases and the everything-excluded error are pinned, with the same
    inputs, in test_crab_sites.py.
    """

    SITES = ["T1_DE_KIT", "T2_CH_CERN", "T2_EE_Estonia", "T2_US_MIT", "T3_CH_PSI"]

    def test_no_blacklist_keeps_globs(self):
        self.assertEqual(
            resolve_whitelist(["T1_*", "T2_*"], [], self.SITES), ["T1_*", "T2_*"]
        )

    def test_blacklisted_site_is_cut_out_of_matching_glob_only(self):
        out = resolve_whitelist(["T1_*", "T2_*", "T3_*"], ["T2_EE_Estonia"], self.SITES)
        self.assertIn("T1_*", out)
        self.assertIn("T3_*", out)
        self.assertNotIn("T2_*", out)
        self.assertNotIn("T2_EE_Estonia", out)
        self.assertIn("T2_CH_CERN", out)
        self.assertIn("T2_US_MIT", out)

    def test_explicitly_whitelisted_and_blacklisted_site_disappears(self):
        out = resolve_whitelist(["T2_CH_CERN", "T2_US_MIT"], ["T2_US_MIT"], self.SITES)
        self.assertEqual(out, ["T2_CH_CERN"])

    def test_glob_blacklist_expands_partially_covered_glob(self):
        out = resolve_whitelist(["T2_*"], ["T2_US_*"], self.SITES)
        self.assertEqual(out, ["T2_CH_CERN", "T2_EE_Estonia"])


class TestProcessingSites(unittest.TestCase):
    """With CRIC unreachable and nothing cached, the site list must fail loudly.

    The cache paths (fresh, stale, corrupt, short) are pinned in test_crab_sites.py.
    """

    UNREACHABLE = "http://127.0.0.1:9/nope"

    def test_no_cache_and_cric_down_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "sites.json")
            with self.assertRaises(RuntimeError):
                processing_sites(cache, url=self.UNREACHABLE, timeout=1)


class TestSiteStats(unittest.TestCase):
    """One black-hole node must be quarantined before it eats the production —
    measured over jobs sent (ended + in flight), never over finished jobs alone."""

    def make(self, tmp, cfg=None):
        return SiteStats(os.path.join(tmp, "stats.json"), cfg)

    def feed_black_hole(self, stats, now):
        # a black hole fails 30 jobs in seconds while the healthy sites' jobs are
        # still running (in flight), with a few ordinary completions elsewhere
        for _ in range(30):
            stats.record("T2_EE_Estonia", False, now=now)
        for _ in range(10):
            stats.record("T2_CH_CERN", True, now=now)
        for _ in range(10):
            stats.record("T1_DE_KIT", True, now=now)
        stats.set_in_flight({"T2_CH_CERN": 50, "T1_DE_KIT": 50})

    def test_black_hole_is_quarantined(self):
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp)
            self.feed_black_hole(stats, now)
            self.assertEqual(stats.blacklist(now=now), ["T2_EE_Estonia"])

    def test_own_failures_need_a_baseline(self):
        # the first site to collect failures must not be judged against nothing
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp)
            for _ in range(30):
                stats.record("T2_EE_Estonia", False, now=now)
            self.assertEqual(stats.blacklist(now=now), [])

    def test_a_bug_of_our_own_blacklists_nothing(self):
        # every site failing at the same rate points at our code, not at a site
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp)
            for site in ("T2_EE_Estonia", "T2_CH_CERN", "T1_DE_KIT"):
                for _ in range(20):
                    stats.record(site, False, now=now)
            self.assertEqual(stats.blacklist(now=now), [])

    def test_quarantine_expires_without_forgetting_the_record(self):
        # The quarantine still expires, but no longer wipes the record (DSProd #40): a
        # wiped record let three broken sites cycle back into the whitelist every six
        # hours. The site returns, is not re-quarantined on the evidence its ban was
        # served for, and keeps both that evidence and the count its next ban doubles
        # on. The ban is shorter than the 24 h window, so the window cannot be what
        # removes the evidence.
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp, cfg={"quarantine_hours": 1.0})
            self.feed_black_hole(stats, now)
            self.assertEqual(stats.blacklist(now=now), ["T2_EE_Estonia"])
            later = stats.sites["T2_EE_Estonia"]["quarantined_until"] + 1
            self.assertEqual(stats.blacklist(now=later), [])
            record = stats.sites["T2_EE_Estonia"]
            self.assertEqual(
                len(record["events"]), 30, "the evidence that earned the ban was lost"
            )
            self.assertEqual(record["quarantines"], 1)

    def test_placeholder_site_names_are_ignored(self):
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp)
            for _ in range(30):
                stats.record("Unknown", False, now=now)
            stats.set_in_flight({"Unknown": 10, "T2_CH_CERN": 10})
            self.assertNotIn("Unknown", stats.sites)
            self.assertNotIn("Unknown", stats.in_flight)

    def test_disabled_returns_nothing(self):
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp, cfg={"enabled": False})
            self.feed_black_hole(stats, now)
            self.assertEqual(stats.blacklist(now=now), [])

    def test_load_skips_records_with_a_foreign_schema(self):
        # a stats file written by a different version must be dropped like corrupt
        # JSON, not kill the workflow before the first submission
        import json as _json

        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stats.json")
            with open(path, "w") as f:
                _json.dump(
                    {
                        "version": 99,
                        "sites": {
                            "T2_CH_CERN": {
                                "events": [{"t": now, "ok": 1}],
                                "quarantined_until": 0.0,
                            },
                            "T1_DE_KIT": {
                                "events": [[now, 1]],
                                "quarantined_until": "abc",
                            },
                            "T2_EE_Estonia": {
                                "events": [[now, 0]],
                                "quarantined_until": 0.0,
                            },
                        },
                    },
                    f,
                )
            stats = SiteStats(path)
            self.assertEqual(list(stats.sites), ["T2_EE_Estonia"])

    def test_persistence_roundtrip(self):
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp)
            self.feed_black_hole(stats, now)
            stats.blacklist(now=now)
            stats.save()
            reloaded = self.make(tmp)
            reloaded.set_in_flight({"T2_CH_CERN": 50, "T1_DE_KIT": 50})
            self.assertEqual(reloaded.blacklist(now=now), ["T2_EE_Estonia"])


class TestCrabJobManager(unittest.TestCase):
    """One unreadable `crab status` must not fail every job of the task, must report
    what crab returned, and must not kill the workflow before the tolerance is spent."""

    def test_parse_error_reports_what_crab_returned(self):
        out = (
            "Something went sideways\nCRAB is unhappy\n"
            + '{"json": "'
            + "x" * 4096
            + '"}\n'
        )
        with self.assertRaises(Exception) as ctx:
            lc.FLAFCrabJobManager.parse_query_output(out, "/tmp/proj", [])
        msg = str(ctx.exception)
        self.assertIn("first lines of what crab returned", msg)
        self.assertIn("Something went sideways", msg)
        self.assertNotIn('"json"', msg, "the multi-MB JSON must not be attached")

    def test_parse_accepts_fresh_task_without_per_job_info(self):
        manager_cls = lc.FLAFCrabJobManager
        out = "Status on the CRAB server:\tSUBMITTED\n"
        job_ids = [manager_cls.JobId(1, "task", "/tmp/proj")]
        result = manager_cls.parse_query_output(out, "/tmp/proj", job_ids)
        self.assertEqual(result[job_ids[0]]["status"], manager_cls.PENDING)

    def test_unreadable_status_degrades_to_pending_and_recovers(self):
        m = make_manager()
        jid = m.JobId(1, "task", "/tmp/proj")
        boom = Exception("no server status")
        with mock.patch.object(
            law.cms.CrabJobManager, "query", side_effect=boom
        ) as base_query, mock.patch("time.sleep") as sleep:
            result = m.query("/tmp/proj", job_ids=[jid])
        self.assertEqual(result[jid]["status"], m.PENDING)
        self.assertEqual(base_query.call_count, m.query_retries + 1)
        self.assertEqual(sleep.call_count, m.query_retries)
        self.assertEqual(m._unreadable["/tmp/proj"], 1)

        # a successful poll clears the strike counter
        with mock.patch.object(law.cms.CrabJobManager, "query", return_value={}):
            m.query("/tmp/proj", job_ids=[jid])
        self.assertNotIn("/tmp/proj", m._unreadable)

    def test_unreadable_status_stops_the_run_after_tolerance(self):
        # The stop is recorded in `stop_reason` and raised by the poll callback, not from
        # query(): law runs queries in a thread pool and turns an exception there into the
        # poll's result, so a raise from query() would count as one more failed poll while
        # every other CRAB task lost that poll's status.
        m = make_manager()
        jid = m.JobId(1, "task", "/tmp/proj")
        with mock.patch.object(
            law.cms.CrabJobManager, "query", side_effect=Exception("still broken")
        ), mock.patch("time.sleep"):
            # the last tolerated poll still only degrades
            m._unreadable["/tmp/proj"] = m.max_unreadable_polls - 1
            m.query("/tmp/proj", job_ids=[jid])
            self.assertIsNone(m.stop_reason, "the run was stopped before the tolerance")
            result = m.query("/tmp/proj", job_ids=[jid])
        self.assertEqual(result[jid]["status"], m.PENDING)
        self.assertIn("unreadable", m.stop_reason)
        self.assertIn("still broken", m.stop_reason)

        workflow = mock.Mock(spec=lc.CrabWorkflow)
        workflow._flaf_crab_job_manager = m
        with self.assertRaises(RuntimeError) as ctx:
            lc.CrabWorkflow.crab_poll_callback(workflow, mock.Mock())
        self.assertIn("unreadable", str(ctx.exception))

        # and a manager with nothing to report lets the poll loop go on
        workflow._flaf_crab_job_manager = make_manager()
        self.assertTrue(lc.CrabWorkflow.crab_poll_callback(workflow, mock.Mock()))

    def test_degrade_without_job_ids_reraises_without_crab_log(self):
        # a proj dir with no readable crab.log leaves nothing to degrade to
        m = make_manager()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            law.cms.CrabJobManager, "query", side_effect=Exception("boom")
        ), mock.patch("time.sleep"):
            with self.assertRaises(Exception) as ctx:
                m.query(tmp, job_ids=None)
        self.assertIn("boom", str(ctx.exception))


class TestSiteStatsHarvest(unittest.TestCase):
    """Site outcomes must be keyed by the per-attempt job id from the query result:
    law's poll attaches per-job `extra` to job_data positionally, so with several live
    CRAB projects the site info there can sit on the wrong job."""

    def make(self, tmp):
        m = make_manager()
        m.site_stats = SiteStats(os.path.join(tmp, "stats.json"))
        return m

    @staticmethod
    def job(m, num, proj, status, site, code=None):
        jid = m.JobId(num, "task", proj)
        return jid, {
            "status": status,
            "code": code,
            "extra": {"site_history": ["T0_X", site]},
        }

    def test_terminal_jobs_recorded_once_in_flight_refreshed(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self.make(tmp)
            result = dict(
                [
                    self.job(m, 1, "/p1", m.FINISHED, "T2_CH_CERN"),
                    # what CRAB reports for a payload that failed at the site
                    self.job(m, 2, "/p1", m.FAILED, "T2_EE_Estonia", code=5),
                    self.job(m, 3, "/p1", m.RUNNING, "T1_DE_KIT"),
                    self.job(m, 4, "/p1", m.PENDING, "T1_DE_KIT"),
                ]
            )
            m._harvest_site_stats("/p1", result)
            m._harvest_site_stats("/p1", result)  # the same poll result again

            stats = m.site_stats
            self.assertEqual(len(stats.sites["T2_CH_CERN"]["events"]), 1)
            self.assertEqual(len(stats.sites["T2_EE_Estonia"]["events"]), 1)
            self.assertEqual(stats.sites["T2_EE_Estonia"]["events"][0][1], 0)
            self.assertEqual(stats.in_flight, {"T1_DE_KIT": 2})
            self.assertTrue(os.path.exists(stats.path), "record must be persisted")

    def test_a_failure_without_a_code_is_not_the_sites_doing(self):
        # a killed task, a refused submission or law's own bookkeeping reports jobs failed
        # without a job-level code; counted, a mass kill drove every site's baseline to
        # ~100 % and the quarantine could no longer fire (DSProd #23, #46)
        with tempfile.TemporaryDirectory() as tmp:
            m = self.make(tmp)
            m._harvest_site_stats(
                "/p1", dict([self.job(m, 1, "/p1", m.FAILED, "T2_CH_CERN")])
            )
            self.assertNotIn("T2_CH_CERN", m.site_stats.sites)
            self.assertEqual(m.site_stats.in_flight, {}, "nor is it still in flight")

    def test_in_flight_is_combined_across_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self.make(tmp)
            m._harvest_site_stats(
                "/p1", dict([self.job(m, 1, "/p1", m.RUNNING, "T1_DE_KIT")])
            )
            m._harvest_site_stats(
                "/p2", dict([self.job(m, 1, "/p2", m.RUNNING, "T1_DE_KIT")])
            )
            self.assertEqual(m.site_stats.in_flight, {"T1_DE_KIT": 2})

    def test_jobs_without_site_history_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self.make(tmp)
            jid = m.JobId(1, "task", "/p1")
            m._harvest_site_stats("/p1", {jid: {"status": m.FINISHED, "extra": {}}})
            self.assertEqual(m.site_stats.sites, {})


class TestCrabHome(unittest.TestCase):
    """CRAB rewrites ~/.crab3 on every command: HOME must be off AFS, and crab.log
    must not land in the working area — while submit keeps its cwd."""

    def test_cmssw_env_moves_home_and_wraps_crab(self):
        base_env = {"PATH": "/cvmfs/x/bin:/usr/bin", "HOME": "/afs/cern.ch/user/x/xyz"}
        with tempfile.TemporaryDirectory() as tmp:
            old_tempdir = tempfile.tempdir
            tempfile.tempdir = tmp
            try:
                with mock.patch.object(
                    law.cms.CrabJobManager,
                    "cmssw_env",
                    property(lambda self: base_env),
                ):
                    m = make_manager()
                    env = m.cmssw_env
            finally:
                tempfile.tempdir = old_tempdir

            self.assertTrue(env["HOME"].startswith(tmp), env["HOME"])
            self.assertEqual(
                base_env["HOME"], "/afs/cern.ch/user/x/xyz", "base env must not mutate"
            )
            bin_dir = env["PATH"].split(":", 1)[0]
            self.assertTrue(env["PATH"].endswith(base_env["PATH"]))
            wrapper = os.path.join(bin_dir, "crab")
            self.assertTrue(os.access(wrapper, os.X_OK))
            content = open(wrapper).read()
            self.assertIn("submit) ;;", content, "submit must keep its cwd")
            self.assertIn('cd "$HOME"', content)
            self.assertIn("/cvmfs/cms.cern.ch/common/crab", content)


class TestResourceParamIsolation(unittest.TestCase):
    """A requiring task's max_runtime / n_cpus must not leak onto what it requires,
    while explicit pins and workflow<->branch conversion keep working."""

    class _WFA(lc.HTCondorWorkflow, law.LocalWorkflow):
        def create_branch_map(self):
            return {0: 0}

        def run(self):
            pass

    class _WFB(_WFA):
        max_runtime = lc.copy_param(lc.HTCondorWorkflow.max_runtime, 30.0)
        n_cpus = lc.copy_param(lc.HTCondorWorkflow.n_cpus, 4)

    def make_a(self):
        return self._WFA(max_runtime=2.0, n_cpus=1, workflow="local")

    def test_resources_do_not_leak_through_req(self):
        params = self._WFB.req_params(self.make_a())
        self.assertNotIn("max_runtime", params)
        self.assertNotIn("n_cpus", params)
        b = self._WFB.req(self.make_a())
        self.assertEqual(float(b.max_runtime), 30.0)
        self.assertEqual(int(b.n_cpus), 4)

    def test_explicit_pin_still_works(self):
        b = self._WFB.req(self.make_a(), max_runtime=9.0, n_cpus=2)
        self.assertEqual(float(b.max_runtime), 9.0)
        self.assertEqual(int(b.n_cpus), 2)

    def test_workflow_branch_conversion_keeps_resources(self):
        # law passes _skip_task_excludes for workflow<->branch conversion, so a
        # CLI-given per-task value still reaches that task's branches
        params = self._WFB.req_params(self.make_a(), _skip_task_excludes=True)
        self.assertEqual(float(params["max_runtime"]), 2.0)
        self.assertEqual(int(params["n_cpus"]), 1)


class TestWorkerGuards(unittest.TestCase):
    """A worker must never require (and possibly rebuild) a live bundle, and must
    never delete the proxy the batch system delegated."""

    def uses_bundles(self, env):
        stub = types.SimpleNamespace(
            bundle_flavours=["core"], effective_workflow="crab", bundle=True
        )
        with mock.patch.dict(os.environ, env, clear=False):
            if "LAW_JOB_HOME" not in env:
                os.environ.pop("LAW_JOB_HOME", None)
            return lc.HTCondorWorkflow._uses_bundles(stub)

    def test_uses_bundles_on_submit_node(self):
        self.assertTrue(self.uses_bundles({}))

    def test_never_uses_bundles_on_worker(self):
        self.assertFalse(self.uses_bundles({"LAW_JOB_HOME": "/srv/job"}))

    def test_delegated_proxy_survives_task_instantiation(self):
        # The DSProd incident: CRAB delegates a ~23:59 h proxy, below the interactive
        # 24 h renewal threshold; instantiating the task on a worker deleted it and
        # every remote-storage call in the job failed.
        with tempfile.TemporaryDirectory() as tmp:
            proxy_path = os.path.join(tmp, "x509up")
            with open(proxy_path, "w") as f:
                f.write("delegated proxy")
            env = {"X509_USER_PROXY": proxy_path, "LAW_JOB_HOME": "/srv/job"}
            with mock.patch.dict(os.environ, env), mock.patch.object(
                grid_helper_tasks,
                "get_voms_proxy_info",
                return_value={"timeleft": 5.0},
            ):
                task = grid_helper_tasks.CreateVomsProxy()
                self.assertTrue(
                    os.path.exists(proxy_path), "the delegated proxy was deleted"
                )
                self.assertTrue(task.complete())
                with self.assertRaises(RuntimeError):
                    task.run()
            self.assertTrue(os.path.exists(proxy_path))

    def test_interactive_short_proxy_is_incomplete_but_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            proxy_path = os.path.join(tmp, "x509up")
            with open(proxy_path, "w") as f:
                f.write("old proxy")
            env = {"X509_USER_PROXY": proxy_path}
            with mock.patch.dict(os.environ, env), mock.patch.object(
                grid_helper_tasks,
                "get_voms_proxy_info",
                return_value={"timeleft": 5.0},
            ):
                os.environ.pop("LAW_JOB_HOME", None)
                task = grid_helper_tasks.CreateVomsProxy()
                self.assertFalse(task.complete())
                self.assertTrue(
                    os.path.exists(proxy_path),
                    "complete() must judge, not delete",
                )


if __name__ == "__main__":
    unittest.main()
