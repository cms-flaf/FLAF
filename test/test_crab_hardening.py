#!/usr/bin/env python3
"""The CRAB backend must survive what a real production throws at it.

Each test here encodes a failure observed in the 115k-job DSProd CRAB production
(cms-flaf/DSProd #5, #7, #11, #16, #18, #19) or found while auditing FLAF against it:
an unreadable `crab status` response, retries escaping as tiny CRAB tasks, a blacklist
silently defeated by the whitelist, a worker deleting its own delegated proxy, resource
parameters leaking through req(), and a worker rebuilding a live bundle.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import law
import law.workflow.remote

from FLAF.run_tools import law_customizations as lc
from FLAF.run_tools.crab_sites import SiteStats, processing_sites, resolve_whitelist
from FLAF.RunKit import grid_helper_tasks


def make_proxy_stub(n_parallel, n_active, n_parallel_max=1_000_000, refill=0.2):
    stub = types.SimpleNamespace()
    stub.poll_data = types.SimpleNamespace(n_parallel=n_parallel, n_active=n_active)
    stub.n_parallel_max = n_parallel_max
    stub.task = types.SimpleNamespace(_crab_cfg=lambda: {"refill_fraction": refill})
    stub._crab_refill_fraction = (
        lambda: lc._FLAFCrabWorkflowProxy._crab_refill_fraction(stub)
    )
    return stub


def should_submit(n_waiting, **kwargs):
    stub = make_proxy_stub(**kwargs)
    return lc._FLAFCrabWorkflowProxy._should_submit_crab_group(stub, n_waiting)


class TestWaveGate(unittest.TestCase):
    """The gate must aggregate on jobs waiting, not on free slots."""

    def test_retry_trickle_is_held_in_part_filled_pool(self):
        # The DSProd incident: 3270 of 5000 slots taken, so 1730 slots free — the old
        # free-slot rule was permanently open and each poll's retry handful became its
        # own CRAB task. A handful of retries must be held.
        self.assertFalse(should_submit(5, n_parallel=5000, n_active=3270))

    def test_full_wave_with_room_submits(self):
        self.assertTrue(should_submit(3000, n_parallel=5000, n_active=0))

    def test_full_wave_without_room_is_held(self):
        # 5000 jobs waiting but only 500 slots free: no full wave can run yet.
        self.assertFalse(should_submit(5000, n_parallel=5000, n_active=4500))

    def test_tail_is_released(self):
        # Running + waiting can never fill a wave again — holding only delays the tail.
        self.assertTrue(should_submit(5, n_parallel=5000, n_active=300))

    def test_small_production_never_batches(self):
        self.assertTrue(should_submit(100, n_parallel=5000, n_active=0))

    def test_first_wave_of_large_production_submits(self):
        self.assertTrue(should_submit(20000, n_parallel=5000, n_active=0))

    def test_nothing_waiting_submits(self):
        self.assertTrue(should_submit(0, n_parallel=5000, n_active=3270))

    def test_unlimited_parallelism_keeps_law_behaviour(self):
        self.assertTrue(
            should_submit(1, n_parallel=1_000_000, n_active=0, n_parallel_max=1_000_000)
        )


class TestSubmitParking(unittest.TestCase):
    """Parked retries must move to unsubmitted without changing len(job_data).

    law's poll loop snapshots len(job_data) once; changing it mid-poll hangs the loop
    or ends it early.
    """

    def test_parking_preserves_job_data_length(self):
        proxy = object.__new__(lc._FLAFCrabWorkflowProxy)
        proxy.poll_data = types.SimpleNamespace(n_parallel=5000, n_active=3270)
        proxy.n_parallel_max = 1_000_000
        proxy.task = types.SimpleNamespace(_crab_cfg=lambda: {})
        proxy.job_data = law.workflow.remote.JobData()
        proxy.job_data.jobs[7] = {"job_id": "x", "branches": [7], "status": "retry"}
        proxy.job_data.unsubmitted_jobs[9] = [9]
        proxy._can_skip_job = lambda job_num, branches: False
        dumped = []
        proxy.dump_job_data = lambda: dumped.append(True)

        n_before = len(proxy.job_data)
        result = lc._FLAFCrabWorkflowProxy.submit(proxy, retry_jobs={7: [7]})

        self.assertEqual(result, {})
        self.assertEqual(len(proxy.job_data), n_before)
        self.assertNotIn(7, proxy.job_data.jobs)
        self.assertEqual(proxy.job_data.unsubmitted_jobs[7], [7])
        self.assertTrue(dumped)

    def test_no_poll_bypasses_the_gate(self):
        # a --no-poll invocation resubmits failures exactly once and then returns;
        # parking would silently skip that documented one-shot resubmission
        proxy = object.__new__(lc._FLAFCrabWorkflowProxy)
        proxy.poll_data = types.SimpleNamespace(n_parallel=5000, n_active=3270)
        proxy.n_parallel_max = 1_000_000
        proxy.task = types.SimpleNamespace(_crab_cfg=lambda: {}, no_poll=True)
        proxy.job_data = law.workflow.remote.JobData()
        proxy.job_data.jobs[7] = {"job_id": "x", "branches": [7], "status": "retry"}
        with mock.patch.object(
            lc._FLAFCrabWorkflowProxyBase, "submit", return_value={"submitted": True}
        ) as base_submit:
            result = lc._FLAFCrabWorkflowProxy.submit(proxy, retry_jobs={7: [7]})
        self.assertEqual(result, {"submitted": True})
        base_submit.assert_called_once()
        self.assertIn(7, proxy.job_data.jobs, "nothing may be parked under no_poll")


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
    """CRAB gives the whitelist precedence, so exclusions must be cut out of it."""

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

    def test_everything_excluded_raises(self):
        with self.assertRaises(RuntimeError):
            resolve_whitelist(["T2_US_MIT"], ["T2_US_MIT"], self.SITES)

    def test_glob_blacklist_is_not_inverted_into_a_whitelist(self):
        # a pattern in the blacklist must exclude what it matches — a literal
        # membership test would instead expand the tier into explicitly
        # whitelisted names, silently defeating the exclusion
        out = resolve_whitelist(["T1_*", "T2_*", "T3_*"], ["T3_*"], self.SITES)
        self.assertEqual(out, ["T1_*", "T2_*"])

    def test_glob_blacklist_excludes_a_concrete_whitelist_entry(self):
        out = resolve_whitelist(["T2_CH_CERN", "T2_US_MIT"], ["T2_US_*"], self.SITES)
        self.assertEqual(out, ["T2_CH_CERN"])

    def test_glob_blacklist_expands_partially_covered_glob(self):
        out = resolve_whitelist(["T2_*"], ["T2_US_*"], self.SITES)
        self.assertEqual(out, ["T2_CH_CERN", "T2_EE_Estonia"])


class TestProcessingSites(unittest.TestCase):
    """The site list must come from cache when CRIC is unreachable, and fail loudly
    only when there is nothing to fall back on."""

    UNREACHABLE = "http://127.0.0.1:9/nope"

    def test_fresh_cache_avoids_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "sites.json")
            with open(cache, "w") as f:
                f.write('["T1_DE_KIT", "T2_CH_CERN"]')
            sites = processing_sites(cache, url=self.UNREACHABLE, timeout=1)
            self.assertEqual(sites, ["T1_DE_KIT", "T2_CH_CERN"])

    def test_stale_cache_used_when_cric_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "sites.json")
            with open(cache, "w") as f:
                f.write('["T2_CH_CERN"]')
            os.utime(cache, (0, 0))  # far in the past
            sites = processing_sites(cache, url=self.UNREACHABLE, timeout=1)
            self.assertEqual(sites, ["T2_CH_CERN"])

    def test_corrupt_stale_cache_still_raises_the_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "sites.json")
            with open(cache, "w") as f:
                f.write("not json {")
            os.utime(cache, (0, 0))
            with self.assertRaises(RuntimeError):
                processing_sites(cache, url=self.UNREACHABLE, timeout=1)

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

    def test_quarantine_expires_and_record_restarts(self):
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = self.make(tmp)
            self.feed_black_hole(stats, now)
            self.assertEqual(stats.blacklist(now=now), ["T2_EE_Estonia"])
            later = now + stats.cfg["quarantine_hours"] * 3600.0 + 1
            self.assertEqual(stats.blacklist(now=later), [])
            self.assertEqual(
                stats.sites["T2_EE_Estonia"]["events"], [], "record must restart clean"
            )

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


def make_manager():
    return lc.FLAFCrabJobManager(
        sandbox_name="cmssw::CMSSW_14_0_0::arch=el9_amd64_gcc12"
    )


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

    def test_unreadable_status_raises_after_tolerance(self):
        m = make_manager()
        jid = m.JobId(1, "task", "/tmp/proj")
        m._unreadable["/tmp/proj"] = m.max_unreadable_polls
        with mock.patch.object(
            law.cms.CrabJobManager, "query", side_effect=Exception("still broken")
        ), mock.patch("time.sleep"):
            with self.assertRaises(Exception) as ctx:
                m.query("/tmp/proj", job_ids=[jid])
        self.assertIn("unreadable", str(ctx.exception))

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
    def job(m, num, proj, status, site):
        jid = m.JobId(num, "task", proj)
        return jid, {"status": status, "extra": {"site_history": ["T0_X", site]}}

    def test_terminal_jobs_recorded_once_in_flight_refreshed(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self.make(tmp)
            result = dict(
                [
                    self.job(m, 1, "/p1", m.FINISHED, "T2_CH_CERN"),
                    self.job(m, 2, "/p1", m.FAILED, "T2_EE_Estonia"),
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
