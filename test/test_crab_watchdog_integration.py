#!/usr/bin/env python3
"""How the stall watchdog is wired into FLAF's CRAB backend, and what reaches it.

`test_crab_watchdog.py` pins the watchdog itself. What is pinned here is the wiring a unit
test of that module cannot see, each piece of which turns a verdict into a lost branch or a
silent no-op when it is wrong:

- the job manager rewrites a condemned job as failed on the status law just fetched, so law's
  own retry path resubmits it -- and the site it hung at is charged once, by the watchdog,
  never a second time by the harvest that follows (DSProd, where a mass of code-less failures
  drove every site's baseline to ~100 % and the quarantine could no longer fire);
- the poll callback lists the flag directory once per interval and hands the watchdog law's
  job map, and the flag directory is resolved only when it is first listed, since building
  the remote file system shells out to the grid tools;
- the job side writes a flag only inside a real CRAB job, for the branch it was submitted to
  run, and luigi's START/SUCCESS/FAILURE events start and stop it once per task;
- a stale flag left by an earlier attempt of the same branch does not condemn the attempt now
  running before it has had a chance to beat (FLAF fix in `StallWatchdog._age`);
- `--<Task>-parallel-jobs` / `--<Task>-poll-interval` reach the task they name through
  `req()` instead of being overwritten by the requiring task's own value.

Ported in part from DSProd `test/test_site_stats_harvest.py` and `test/test_watchdog.py`.
"""

import datetime
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

# law_customizations imports FLAF.Common.Setup, which imports ROOT at module level; the CI
# unit-test environment has no ROOT, and nothing exercised here touches it.
try:
    import ROOT  # noqa: F401
except ImportError:
    sys.modules["ROOT"] = mock.MagicMock()

import law  # noqa: E402
import luigi  # noqa: E402
from luigi.cmdline_parser import CmdlineParser  # noqa: E402

from FLAF.run_tools import law_customizations as lc  # noqa: E402
from FLAF.run_tools.crab_sites import SiteStats  # noqa: E402
from FLAF.run_tools.crab_watchdog import (  # noqa: E402
    HEARTBEAT_DIR,
    Heartbeat,
    StallWatchdog,
    watchdog_config,
)

WD = "FLAF.run_tools.crab_watchdog"
TASK = "261008_120000:kandroso_crab_WatchdogUp_v1_P_0123abcd"
PROJ = "/crab/projects/crab_WatchdogUp_v1_P_0123abcd"
SITE = "T2_XX_Hung"
SANDBOX = "cmssw::CMSSW_14_0_0::arch=el9_amd64_gcc12"


class Flag:
    """One entry of `gfal_ls`: a name and a modification time."""

    def __init__(self, name, date, is_dir=False):
        self.name = str(name)
        self.date = date
        self.is_dir = is_dir


def minutes(n):
    return datetime.timedelta(minutes=n)


def utcnow():
    """The naive UTC time the watchdog itself compares listed modification times with."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def listed(w, *flags):
    """Let the watchdog list its directory once, seeing `flags`."""
    with mock.patch(f"{WD}.gfal_ls_safe", return_value=list(flags)):
        return w.refresh()


def job_map(*specs):
    """law's job_data.jobs for (job_num, branch) pairs, with json-style job ids."""
    return {
        num: {"job_id": [num, TASK, PROJ], "branches": [branch]}
        for num, branch in specs
    }


def manager():
    """A real FLAF CRAB job manager; the sandbox is built lazily, so nothing runs here."""
    return lc.FLAFCrabJobManager(sandbox_name=SANDBOX)


def running(m, *nums, site=SITE):
    """A parsed `crab status` response with every job of `nums` running at `site`."""
    out = {}
    for num in nums:
        jid = m.JobId(num, TASK, PROJ)
        out[jid] = m.job_status_dict(
            job_id=jid, status=m.RUNNING, extra={"site_history": ["T2_XX_Before", site]}
        )
    return out


def stalled_watchdog(m, specs, flag_age=99, cfg=None):
    """A real watchdog whose flags for `specs` are `flag_age` minutes old, and which has
    watched those jobs run for long enough that the grace for a first beat is over.

    The verdict is formed on the real clock, because the job manager asks for it without
    a time of its own.
    """
    now = utcnow()
    # `cfg=False` switches it off, which is not the same as giving no settings
    w = StallWatchdog(
        "root://x//flags", watchdog_config({"watchdog": {} if cfg is None else cfg})
    )
    w.messages = []
    w.publish = w.messages.append
    flags = [Flag(branch, now - minutes(flag_age)) for _, branch in specs]
    assert listed(w, *flags) == w.enabled
    w.set_jobs(job_map(*specs))
    # first seen running long ago: the grace clock starts on the first running poll
    nums = [num for num, _ in specs]
    assert w.verdicts(running(m, *nums), now=now - minutes(999)) == {}
    return w


def outcomes(stats, site):
    return [ok for _, ok in stats.sites.get(site, {}).get("events", [])]


# --------------------------------------------------------------------------------------
# FLAF-shaped task classes, as every FLAF task that can run on CRAB is built
# --------------------------------------------------------------------------------------


class StubSetup:
    """What `Task.__init__` needs from `Setup`: the global parameters and a file system.

    `get_fs` hands out a plain local path, which `remote_target` turns into a local target;
    the calls are counted, because building the real remote file system is what talks to
    the grid.
    """

    def __init__(self, fs_root, crab_cfg=None):
        self.global_params = {"crab": dict(crab_cfg or {})}
        self.get_fs = mock.Mock(return_value=fs_root)


class WatchdogUp(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    def create_branch_map(self):
        return {0: 0, 1: 1}

    def run(self):
        pass


class WatchdogRoot(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    def create_branch_map(self):
        return {0: 0}

    def workflow_requires(self):
        return {"up": WatchdogUp.req(self)}

    def run(self):
        pass


def reset_cli_state():
    """Forget every process-wide cache a command line leaves behind.

    law caches the parsed global options per process and FLAF caches the prefer-cli drop
    set per parser identity, which a later parser can reuse; luigi caches task instances by
    their parameters, workflow proxy included. A real run has one command line per process,
    a test module has many.
    """
    law.parser._reset()
    lc.Task._req_prefer_cli_drop_cache.clear()
    luigi.task_register.Register.clear_instance_cache()


class FlafTaskTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data_dir = tmp.name
        self.fs_root = os.path.join(self.data_dir, "fs")
        env = mock.patch.dict(
            os.environ,
            {"ANALYSIS_DATA_PATH": self.data_dir, "ANALYSIS_PATH": self.data_dir},
        )
        env.start()
        self.addCleanup(env.stop)
        for name in ("LAW_CRAB_JOB_NUMBER", "LAW_JOB_HOME", "X509_USER_PROXY"):
            if name in os.environ:
                os.environ.pop(name)
        self.setup = StubSetup(self.fs_root)
        patcher = mock.patch.object(
            lc.Setup, "getGlobal", side_effect=lambda *a, **k: self.setup
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        reset_cli_state()
        self.addCleanup(reset_cli_state)
        self.addCleanup(self._forget_site_records)

    def _forget_site_records(self):
        for key in list(SiteStats._shared):
            if key.startswith(self.data_dir):
                SiteStats._shared.pop(key, None)

    def crab_cfg(self, cfg):
        self.setup.global_params["crab"] = dict(cfg or {})

    def workflow(self, **kwargs):
        kwargs.setdefault("version", "v1")
        kwargs.setdefault("period", "P")
        kwargs.setdefault("workflow", "crab")
        return WatchdogUp(**kwargs)

    def branch(self, branch=0, **kwargs):
        return self.workflow(branch=branch, **kwargs)


# --------------------------------------------------------------------------------------
# the job manager
# --------------------------------------------------------------------------------------


class AVerdictReachesLawAsAFailure(unittest.TestCase):
    """`_apply_watchdog` rewrites the status law just fetched, so law's own retry path does
    the resubmission -- there is deliberately no second mechanism."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.m = manager()
        self.stats = SiteStats(os.path.join(tmp.name, "stats.json"))
        self.m.site_stats = self.stats

    def test_a_stalled_running_job_becomes_failed_with_the_reason_and_no_code(self):
        self.m.watchdog = stalled_watchdog(self.m, [(1, 7)])
        result = running(self.m, 1)
        self.m._apply_watchdog(result)
        data = result[self.m.JobId(1, TASK, PROJ)]
        self.assertEqual(data["status"], self.m.FAILED)
        self.assertIsNone(data["code"])
        self.assertIn("stalled", data["error"])
        self.assertIn("heartbeat 99 min old", data["error"])

    def test_the_site_it_hung_at_is_recorded_once_as_a_failure(self):
        self.m.watchdog = stalled_watchdog(self.m, [(1, 7)])
        self.m._apply_watchdog(running(self.m, 1))
        self.assertEqual(outcomes(self.stats, SITE), [0])
        self.assertEqual(outcomes(self.stats, "T2_XX_Before"), [])
        self.assertTrue(os.path.exists(self.stats.path), "the record must be saved")

    def test_the_same_job_id_condemned_again_is_not_charged_twice(self):
        # a verdict per branch is capped by the watchdog; with the cap raised the manager's
        # own once-per-job-id record is what keeps the site from being charged again
        w = stalled_watchdog(self.m, [(1, 7)], cfg={"max_per_branch": 5})
        self.m.watchdog = w
        self.m._apply_watchdog(running(self.m, 1))
        # the job id was forgotten; start its grace clock again, long ago
        again = running(self.m, 1)
        now = utcnow()
        self.assertEqual(w.verdicts(running(self.m, 1), now=now - minutes(999)), {})
        self.m._apply_watchdog(again)
        self.assertEqual(again[self.m.JobId(1, TASK, PROJ)]["status"], self.m.FAILED)
        self.assertEqual(outcomes(self.stats, SITE), [0])

    def test_each_condemned_job_is_charged_to_its_own_site(self):
        # two healthy jobs alongside, or two stale flags out of two running jobs would read
        # as a storage fault and condemn nothing
        w = stalled_watchdog(self.m, [(1, 7), (2, 8), (3, 9), (4, 10)])
        now = utcnow()
        listed(
            w,
            Flag(7, now - minutes(99)),
            Flag(8, now - minutes(99)),
            Flag(9, now - minutes(3)),
            Flag(10, now - minutes(3)),
        )
        self.m.watchdog = w
        result = running(self.m, 1, 3, 4)
        result.update(running(self.m, 2, site="T2_YY_AlsoHung"))
        self.m._apply_watchdog(result)
        failed = sorted(
            jid.crab_num for jid, d in result.items() if d["status"] == self.m.FAILED
        )
        self.assertEqual(failed, [1, 2])
        self.assertEqual(outcomes(self.stats, SITE), [0])
        self.assertEqual(outcomes(self.stats, "T2_YY_AlsoHung"), [0])

    def test_a_job_that_finished_meanwhile_is_left_alone(self):
        w = stalled_watchdog(self.m, [(1, 7)])
        self.m.watchdog = w
        original = w.verdicts

        def finish_then_report(res, now=None):
            # the job completes between the verdict being formed and being applied
            out = original(res, now=now)
            for data in res.values():
                data["status"] = self.m.FINISHED
            return out

        w.verdicts = finish_then_report
        result = running(self.m, 1)
        with mock.patch.object(w, "forget", wraps=w.forget) as forget:
            self.m._apply_watchdog(result)
        data = result[self.m.JobId(1, TASK, PROJ)]
        self.assertEqual(data["status"], self.m.FINISHED)
        self.assertNotIn("stalled", str(data["error"]))
        self.assertEqual(outcomes(self.stats, SITE), [])
        forget.assert_not_called()

    def test_the_condemned_job_id_is_forgotten(self):
        w = stalled_watchdog(self.m, [(1, 7)])
        self.m.watchdog = w
        with mock.patch.object(w, "forget", wraps=w.forget) as forget:
            self.m._apply_watchdog(running(self.m, 1))
        forget.assert_called_once_with(self.m.JobId(1, TASK, PROJ))
        # its grace clock starts again: the same job id seen running now is given time
        self.assertEqual(w.verdicts(running(self.m, 1)), {})

    def test_the_harvest_that_follows_does_not_charge_it_again(self):
        self.m.watchdog = stalled_watchdog(self.m, [(1, 7)])
        result = running(self.m, 1)
        self.m._apply_watchdog(result)
        self.m._harvest_site_stats(PROJ, result)
        self.m._harvest_site_stats(PROJ, result)
        self.assertEqual(outcomes(self.stats, SITE), [0])
        # and it is no longer in flight there either
        self.assertEqual(self.stats.in_flight, {})

    def test_a_job_without_a_verdict_is_untouched(self):
        now = utcnow()
        w = stalled_watchdog(self.m, [(1, 7)])
        listed(w, Flag(7, now - minutes(3)))
        self.m.watchdog = w
        result = running(self.m, 1)
        self.m._apply_watchdog(result)
        self.assertEqual(result[self.m.JobId(1, TASK, PROJ)]["status"], self.m.RUNNING)
        self.assertEqual(outcomes(self.stats, SITE), [])

    def test_nothing_happens_without_a_watchdog_or_with_it_off(self):
        for w in (None, stalled_watchdog(self.m, [(1, 7)], cfg=False)):
            self.m.watchdog = w
            result = running(self.m, 1)
            self.m._apply_watchdog(result)
            self.assertEqual(
                result[self.m.JobId(1, TASK, PROJ)]["status"], self.m.RUNNING
            )
        self.assertEqual(outcomes(self.stats, SITE), [])

    def test_without_a_site_record_the_job_is_still_failed(self):
        self.m.site_stats = None
        self.m.watchdog = stalled_watchdog(self.m, [(1, 7)])
        result = running(self.m, 1)
        self.m._apply_watchdog(result)
        self.assertEqual(result[self.m.JobId(1, TASK, PROJ)]["status"], self.m.FAILED)


class TheQueryAppliesTheWatchdogBeforeHarvesting(unittest.TestCase):
    """Harvested first, a condemned job would be counted in flight at the site on the very
    poll that charges the site for it."""

    def test_a_condemned_job_is_charged_once_and_not_counted_in_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = manager()
            m.site_stats = SiteStats(os.path.join(tmp, "stats.json"))
            m.watchdog = stalled_watchdog(m, [(1, 7)])
            jid = m.JobId(1, TASK, PROJ)
            with mock.patch.object(
                law.cms.CrabJobManager,
                "query",
                side_effect=lambda *a, **k: running(m, 1),
            ):
                result = m.query(PROJ, job_ids=[jid])
            self.assertEqual(result[jid]["status"], m.FAILED)
            self.assertEqual(outcomes(m.site_stats, SITE), [0])
            self.assertEqual(m.site_stats.in_flight, {})

    def test_a_healthy_job_is_still_counted_in_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = manager()
            m.site_stats = SiteStats(os.path.join(tmp, "stats.json"))
            w = stalled_watchdog(m, [(1, 7)])
            listed(w, Flag(7, utcnow()))
            m.watchdog = w
            jid = m.JobId(1, TASK, PROJ)
            with mock.patch.object(
                law.cms.CrabJobManager,
                "query",
                side_effect=lambda *a, **k: running(m, 1),
            ):
                result = m.query(PROJ, job_ids=[jid])
            self.assertEqual(result[jid]["status"], m.RUNNING)
            self.assertEqual(m.site_stats.in_flight, {SITE: 1})


# --------------------------------------------------------------------------------------
# ported from DSProd test/test_site_stats_harvest.py
# --------------------------------------------------------------------------------------


def harvest_manager(stats):
    m = manager()
    m.site_stats = stats
    return m


def job(m, num, site, status, code=None):
    """One entry of a parsed `crab status` response."""
    return m.JobId(num, "task", "/proj"), {
        "status": status,
        "code": code,
        "extra": {"site_history": ["T0_X", site]},
    }


class TestHarvestSiteStats(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stats = SiteStats(os.path.join(self.tmp.name, "stats.json"))
        self.m = harvest_manager(self.stats)

    def tearDown(self):
        self.tmp.cleanup()

    def outcomes(self, site):
        return outcomes(self.stats, site)

    def test_finished_and_failed_with_a_code_are_recorded(self):
        self.m._harvest_site_stats(
            "/proj",
            dict(
                [
                    job(self.m, 1, "T2_CH_CERN", self.m.FINISHED),
                    job(self.m, 2, "T2_EE_Estonia", self.m.FAILED, code=5),
                ]
            ),
        )
        self.assertEqual(self.outcomes("T2_CH_CERN"), [1])
        self.assertEqual(self.outcomes("T2_EE_Estonia"), [0])

    def test_a_failure_without_a_code_is_not_the_sites_doing(self):
        # a killed task reports its jobs as failed with no job-level error; counting those is
        # what let an operator's `crab kill` poison every site in the production
        self.m._harvest_site_stats(
            "/proj", dict([job(self.m, 1, "T2_CH_CERN", self.m.FAILED, code=None)])
        )
        self.assertEqual(self.outcomes("T2_CH_CERN"), [])
        self.assertEqual(self.stats.in_flight, {})

    def test_jobs_in_flight_are_the_denominator_not_an_outcome(self):
        self.m._harvest_site_stats(
            "/proj",
            dict(
                [
                    job(self.m, 1, "T2_CH_CERN", self.m.RUNNING),
                    job(self.m, 2, "T2_CH_CERN", self.m.PENDING),
                ]
            ),
        )
        self.assertEqual(self.outcomes("T2_CH_CERN"), [])
        self.assertEqual(self.stats.in_flight, {"T2_CH_CERN": 2})

    def test_the_same_response_twice_counts_once(self):
        result = dict([job(self.m, 1, "T2_EE_Estonia", self.m.FAILED, code=5)])
        self.m._harvest_site_stats("/proj", result)
        self.m._harvest_site_stats("/proj", result)
        self.assertEqual(self.outcomes("T2_EE_Estonia"), [0])

    def test_a_retry_that_then_succeeds_is_recorded_separately(self):
        self.m._harvest_site_stats(
            "/proj", dict([job(self.m, 1, "T2_EE_Estonia", self.m.FAILED, code=5)])
        )
        self.m._harvest_site_stats(
            "/proj", dict([job(self.m, 1, "T2_EE_Estonia", self.m.FINISHED)])
        )
        self.assertEqual(sorted(self.outcomes("T2_EE_Estonia")), [0, 1])

    def test_in_flight_is_combined_over_projects(self):
        self.m._harvest_site_stats(
            "/p1", dict([job(self.m, 1, "T1_DE_KIT", self.m.RUNNING)])
        )
        self.m._harvest_site_stats(
            "/p2", dict([job(self.m, 1, "T1_DE_KIT", self.m.RUNNING)])
        )
        self.assertEqual(self.stats.in_flight, {"T1_DE_KIT": 2})

    def test_jobs_without_a_site_history_are_skipped(self):
        jid = self.m.JobId(1, "task", "/proj")
        self.m._harvest_site_stats(
            "/proj", {jid: {"status": self.m.FINISHED, "code": 0, "extra": {}}}
        )
        self.assertEqual(self.stats.sites, {})

    def test_no_record_configured_is_a_no_op(self):
        m = harvest_manager(None)
        m._harvest_site_stats("/proj", dict([job(m, 1, "T2_CH_CERN", m.FINISHED)]))

    def test_an_unreadable_response_is_a_no_op(self):
        self.m._harvest_site_stats("/proj", None)
        self.assertEqual(self.stats.sites, {})


class TestQuarantineWithACleanBaseline(unittest.TestCase):
    """The end the harvest exists for: one bad node must stand out from healthy sites."""

    def test_a_black_hole_is_quarantined_once_the_baseline_is_real(self):
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = SiteStats(os.path.join(tmp, "stats.json"))
            # the shape actually observed: 65 % failures at one site, a healthy pool elsewhere
            for _ in range(311):
                stats.record("T2_EE_Estonia", False, now=now)
            for _ in range(169):
                stats.record("T2_EE_Estonia", True, now=now)
            for site in ("T2_UK_London_IC", "T2_CH_CSCS", "T2_IT_Legnaro"):
                for _ in range(100):
                    stats.record(site, True, now=now)
            self.assertEqual(stats.blacklist(now=now), ["T2_EE_Estonia"])

    def test_the_poisoned_baseline_is_what_disabled_it(self):
        # every site at ~100 % failure, as the old harvest recorded after a mass retry: the
        # failing site no longer stands out and nothing is quarantined
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stats = SiteStats(os.path.join(tmp, "stats.json"))
            for _ in range(311):
                stats.record("T2_EE_Estonia", False, now=now)
            for _ in range(169):
                stats.record("T2_EE_Estonia", True, now=now)
            for site in ("T2_UK_London_IC", "T2_CH_CSCS", "T2_IT_Legnaro"):
                for _ in range(100):
                    stats.record(site, False, now=now)
            self.assertEqual(stats.blacklist(now=now), [])


# --------------------------------------------------------------------------------------
# a stale flag left by an earlier attempt of the same branch (FLAF fix in StallWatchdog._age)
# --------------------------------------------------------------------------------------

T0 = datetime.datetime(2026, 10, 8, 12, 0, 0)
NEW_TASK = "261008_115500:kandroso_crab_WatchdogUp_v1_P_feedbeef"


def at(n):
    return T0 + minutes(n)


class AStaleFlagLeftByAnEarlierAttempt(unittest.TestCase):
    """Flags are named by branch, so a worker that died without removing its flag leaves one
    behind under the name the next attempt of the same branch will use. Read as evidence about
    the attempt running now, it condemns that attempt on the first poll that sees it running,
    before its own first beat -- and spends one of the branch's attempts for nothing.

    Defaults throughout: a beat every 30 min, stale after 2 missed (60 min), and a job that
    never beats is given 90 min.
    """

    def setUp(self):
        self.w = StallWatchdog("root://x//flags", watchdog_config({}))
        self.w.publish = lambda msg: None
        # the new attempt: job 1 of a fresh CRAB task, running branch 7
        self.w.set_jobs({1: {"job_id": [1, NEW_TASK, PROJ], "branches": [7]}})
        self.key = (1, NEW_TASK, PROJ)

    def poll(self, n):
        return self.w.verdicts({self.key: {"status": "running"}}, now=at(n))

    def test_it_does_not_condemn_the_new_attempt_on_its_first_running_poll(self):
        self.assertTrue(listed(self.w, Flag(7, at(-120))))
        self.assertEqual(self.poll(0), {})

    def test_nor_on_any_later_poll_inside_the_grace_for_a_first_beat(self):
        listed(self.w, Flag(7, at(-120)))
        for n in (0, 30, 60, 89):
            self.assertEqual(self.poll(n), {}, f"condemned {n} min after first seen")

    def test_a_new_attempt_that_never_beats_is_still_condemned_after_the_grace(self):
        listed(self.w, Flag(7, at(-120)))
        self.assertEqual(self.poll(0), {})
        verdict = self.poll(90)
        self.assertEqual(list(verdict), [self.key])
        self.assertIn("no heartbeat", verdict[self.key])

    def test_the_new_attempts_own_flag_going_stale_still_condemns_it(self):
        listed(self.w, Flag(7, at(-120)))
        self.assertEqual(self.poll(0), {})
        # its first beat overwrites the old flag in place
        listed(self.w, Flag(7, at(5)))
        self.assertEqual(self.poll(10), {})
        self.assertEqual(self.poll(64), {})
        verdict = self.poll(65)
        self.assertEqual(list(verdict), [self.key])
        self.assertIn("heartbeat 60 min old", verdict[self.key])

    def test_a_beat_listed_just_before_the_job_was_first_seen_running_counts(self):
        # listed modification times have minute resolution, so the first beat of a job that
        # started within the minute before the poll lists as up to a minute early
        listed(self.w, Flag(7, at(-1)))
        self.assertEqual(self.poll(0), {})
        verdict = self.poll(61)
        self.assertEqual(list(verdict), [self.key])
        self.assertIn("heartbeat 62 min old", verdict[self.key])

    def test_the_same_poll_condemns_it_when_attempts_are_not_told_apart(self):
        # the DSProd form, which judges every flag of the branch whoever wrote it: this is
        # the verdict the FLAF fix exists to withhold
        original = StallWatchdog._age

        def age_without_since(self, branches, now, since=None):
            return original(self, branches, now)

        listed(self.w, Flag(7, at(-120)))
        with mock.patch.object(StallWatchdog, "_age", age_without_since):
            verdict = self.poll(0)
        self.assertEqual(list(verdict), [self.key])
        self.assertIn("heartbeat 120 min old", verdict[self.key])


# --------------------------------------------------------------------------------------
# the workflow: flag directory, poll callback, job-side heartbeat, luigi events
# --------------------------------------------------------------------------------------


class WhereTheFlagsLive(FlafTaskTestCase):
    def test_one_flat_directory_per_task_and_era_under_the_version(self):
        task = self.workflow()
        parts = ["v1", HEARTBEAT_DIR, "WatchdogUp", "P"]
        self.assertEqual(HEARTBEAT_DIR, "heartbeat")
        self.assertEqual(task._heartbeat_dir_parts(), parts)
        self.assertEqual(
            task.heartbeat_dir_target().path, os.path.join(self.fs_root, *parts)
        )
        self.assertEqual(
            task.heartbeat_target(3).path, os.path.join(self.fs_root, *parts, "3")
        )

    def test_a_producer_gets_a_directory_of_its_own(self):
        for attr in ("producer_to_run", "producer_to_aggregate"):
            task = self.workflow()
            setattr(task, attr, "btagShape")
            self.assertEqual(
                task._heartbeat_dir_parts(),
                ["v1", HEARTBEAT_DIR, "WatchdogUp", "P", "btagShape"],
                attr,
            )
            delattr(task, attr)

    def test_the_flags_go_to_the_default_file_system(self):
        self.workflow().heartbeat_dir_target()
        self.setup.get_fs.assert_called_with("default")

    def test_building_the_watchdog_does_not_touch_the_storage(self):
        # resolving the uri builds the remote file system, which shells out to the grid
        # tools; the watchdog is built with the job manager, for --print-status too
        task = self.workflow()
        proxy = task.workflow_proxy
        w = task.job_watchdog()
        self.assertIs(proxy.job_manager.watchdog, w)
        self.setup.get_fs.assert_not_called()
        self.assertEqual(w.flag_dir, task.heartbeat_dir_target().uri())
        self.assertTrue(w.flag_dir.endswith("/v1/heartbeat/WatchdogUp/P"), w.flag_dir)

    def test_the_watchdog_is_built_once_per_workflow_from_the_crab_settings(self):
        self.crab_cfg({"watchdog": {"interval_minutes": 7, "missed_checks": 3}})
        task = self.workflow()
        w = task.job_watchdog()
        self.assertIs(task.job_watchdog(), w)
        self.assertEqual(w.interval_seconds, 7 * 60)
        self.assertEqual(w.stale_seconds, 21 * 60)


class TheJobManagerIsWiredToTheWorkflow(FlafTaskTestCase):
    def test_the_manager_shares_the_workflows_record_watchdog_and_site_list(self):
        task = self.workflow()
        m = task.workflow_proxy.job_manager
        self.assertIsInstance(m, lc.FLAFCrabJobManager)
        self.assertIs(task._flaf_crab_job_manager, m)
        self.assertIs(m.watchdog, task.job_watchdog())
        self.assertIs(m.site_stats, task.site_stats())
        self.assertEqual(
            m.site_stats.path, os.path.join(self.data_dir, "crab_site_stats.json")
        )
        self.assertEqual(
            m.site_cache_path, os.path.join(self.data_dir, "cms_psn_sites.json")
        )


class ThePollCallback(FlafTaskTestCase):
    def setUp(self):
        super().setUp()
        kinit = mock.patch.object(lc, "update_kinit")
        self.kinit = kinit.start()
        self.addCleanup(kinit.stop)

    def callback(self, task, n=1, flags=()):
        with mock.patch(f"{WD}.gfal_ls_safe", return_value=list(flags)) as ls:
            for _ in range(n):
                self.assertTrue(task.crab_poll_callback(task.workflow_proxy.poll_data))
        return ls

    def test_the_flag_directory_is_listed_once_per_interval(self):
        task = self.workflow()
        ls = self.callback(task, n=3)
        self.assertEqual(ls.call_count, 1)
        self.assertEqual(ls.call_args.args[0], task.heartbeat_dir_target().uri())

    def test_the_timed_refresh_is_built_once(self):
        task = self.workflow()
        self.callback(task)
        refresh = task._watchdog_refresh
        self.assertIsNotNone(refresh)
        ls = self.callback(task, n=2)
        self.assertIs(task._watchdog_refresh, refresh)
        ls.assert_not_called()

    def test_the_job_map_is_fed_from_law_job_data_on_every_poll(self):
        task = self.workflow()
        proxy = task.workflow_proxy
        jid = proxy.job_manager.JobId(1, TASK, PROJ)
        proxy.job_data.jobs[1] = proxy.job_data.job_data(
            job_id=jid, branches=[0], status=proxy.job_manager.RUNNING
        )
        w = task.job_watchdog()
        with mock.patch.object(w, "set_jobs", wraps=w.set_jobs) as set_jobs:
            self.callback(task, n=2)
        self.assertEqual(set_jobs.call_count, 2)
        for call in set_jobs.call_args_list:
            self.assertIs(call.args[0], proxy.job_data.jobs)

    def test_a_job_law_is_polling_is_condemned_on_the_next_query(self):
        """End to end: the callback lists a stale flag and publishes law's job map, and the
        manager the workflow built fails the job on the status it fetches next."""
        task = self.workflow()
        proxy = task.workflow_proxy
        m = proxy.job_manager
        jid = m.JobId(1, TASK, PROJ)
        proxy.job_data.jobs[1] = proxy.job_data.job_data(
            job_id=jid, branches=[0], status=m.RUNNING
        )
        now = utcnow()
        self.callback(task, flags=[Flag(0, now - minutes(99))])
        w = task.job_watchdog()
        # first seen running long ago, so the grace for a first beat is over
        self.assertEqual(w.verdicts(running(m, 1), now=now - minutes(999)), {})
        with mock.patch.object(
            law.cms.CrabJobManager, "query", side_effect=lambda *a, **k: running(m, 1)
        ):
            result = m.query(PROJ, job_ids=[jid])
        self.assertEqual(result[jid]["status"], m.FAILED)
        self.assertIn("stalled", result[jid]["error"])
        self.assertEqual(outcomes(m.site_stats, SITE), [0])

    def test_a_watchdog_switched_off_lists_nothing(self):
        self.crab_cfg({"watchdog": False})
        task = self.workflow()
        w = task.job_watchdog()
        with mock.patch.object(w, "set_jobs") as set_jobs:
            ls = self.callback(task, n=2)
        ls.assert_not_called()
        set_jobs.assert_not_called()
        self.assertIsNone(task._watchdog_refresh)


class TheJobSideHeartbeat(FlafTaskTestCase):
    """`crab_heartbeat()` is reached only inside a real CRAB job, so every test that does not
    set LAW_CRAB_JOB_NUMBER takes an early return -- which is how a NameError in the one
    branch that constructs the Heartbeat survived 187 passing DSProd tests and was found by
    the first real job instead. The constructing branch is executed here."""

    def heartbeat(self, task, env=True):
        values = {"LAW_CRAB_JOB_NUMBER": "1"} if env else {}
        with mock.patch.dict(os.environ, values):
            return task.crab_heartbeat()

    def test_a_crab_job_gets_a_real_heartbeat_on_its_branch_flag(self):
        task = self.branch(1)
        with mock.patch.dict(os.environ, {"X509_USER_PROXY": "/tmp/x509up_test"}):
            hb = self.heartbeat(task)
        self.assertIsInstance(hb, Heartbeat)
        self.assertEqual(hb.uri, task.heartbeat_target(1).uri())
        self.assertTrue(hb.uri.endswith("/v1/heartbeat/WatchdogUp/P/1"), hb.uri)
        self.assertEqual(hb.interval, 30 * 60)
        self.assertEqual(hb.voms_token, "/tmp/x509up_test")
        self.assertEqual(hb.label, {"task": "WatchdogUp", "branch": 1})

    def test_the_interval_comes_from_the_crab_settings(self):
        self.crab_cfg({"watchdog": {"interval_minutes": 5}})
        self.assertEqual(self.heartbeat(self.branch()).interval, 5 * 60)

    def test_a_job_of_the_submitted_family_beats(self):
        task = self.branch()
        with CmdlineParser.global_instance(
            ["WatchdogUp", "--version", "v1", "--period", "P", "--branch", "0"]
        ):
            self.assertIsInstance(self.heartbeat(task), Heartbeat)

    def test_nothing_outside_a_crab_job(self):
        self.assertIsNone(self.heartbeat(self.branch(), env=False))

    def test_nothing_for_the_workflow_itself(self):
        self.assertIsNone(self.heartbeat(self.workflow()))

    def test_nothing_for_a_task_the_job_was_not_submitted_to_run(self):
        # a requirement luigi decided to run inside the job: its flag would name a branch
        # of a workflow the driver is not watching
        task = self.branch()
        with CmdlineParser.global_instance(
            ["WatchdogRoot", "--version", "v1", "--period", "P", "--branch", "0"]
        ):
            self.assertIsNone(self.heartbeat(task))

    def test_nothing_with_the_watchdog_switched_off(self):
        for cfg in (False, {"enabled": False}):
            self.crab_cfg({"watchdog": cfg})
            self.assertIsNone(self.heartbeat(self.branch()), cfg)


class FakeHeartbeat:
    """Records what the event handlers do with it; `__enter__` returns itself, as the real
    one does."""

    def __init__(
        self, made, uri, interval_seconds, voms_token=None, label=None, log=None
    ):
        self.uri = uri
        self.interval = interval_seconds
        self.label = label
        self.entered = 0
        self.exited = 0
        made.append(self)

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False


class TheLuigiEventsStartAndStopIt(FlafTaskTestCase):
    """The heartbeat is started and stopped by luigi events, so it covers every FLAF task
    without touching its run(). luigi swallows an exception raised by an event handler into
    its log, so the effects are checked, and the log is checked to be clean."""

    def setUp(self):
        super().setUp()
        self.made = []
        patcher = mock.patch.object(
            lc,
            "Heartbeat",
            side_effect=lambda *a, **k: FakeHeartbeat(self.made, *a, **k),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"LAW_CRAB_JOB_NUMBER": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.tasks = []
        self.addCleanup(self._drop_heartbeats)

    def _drop_heartbeats(self):
        for task in self.tasks:
            lc._crab_heartbeats.pop(task.task_id, None)

    def task(self, branch=0):
        task = self.branch(branch)
        self.tasks.append(task)
        return task

    def fire(self, task, event, *args):
        with self.assertNoLogs("luigi-interface", level="ERROR"):
            task.trigger_event(event, task, *args)

    def test_start_begins_beating_on_the_branch_flag(self):
        task = self.task()
        self.fire(task, luigi.Event.START)
        self.assertEqual(len(self.made), 1)
        hb = self.made[0]
        self.assertEqual((hb.entered, hb.exited), (1, 0))
        self.assertEqual(hb.uri, task.heartbeat_target(0).uri())
        self.assertIs(lc._crab_heartbeats[task.task_id], hb)

    def test_a_second_start_of_the_same_task_does_not_start_another(self):
        # a run() that yields new requirements fires START again when it is resumed
        task = self.task()
        self.fire(task, luigi.Event.START)
        self.fire(task, luigi.Event.START)
        self.assertEqual(len(self.made), 1)
        self.assertEqual(self.made[0].entered, 1)

    def test_success_stops_it_and_removes_it(self):
        task = self.task()
        self.fire(task, luigi.Event.START)
        self.fire(task, luigi.Event.SUCCESS)
        self.assertEqual(self.made[0].exited, 1)
        self.assertNotIn(task.task_id, lc._crab_heartbeats)
        # nothing is left to stop twice
        self.fire(task, luigi.Event.SUCCESS)
        self.assertEqual(self.made[0].exited, 1)

    def test_failure_stops_it_too(self):
        task = self.task()
        self.fire(task, luigi.Event.START)
        self.fire(task, luigi.Event.FAILURE, RuntimeError("payload died"))
        self.assertEqual(self.made[0].exited, 1)
        self.assertNotIn(task.task_id, lc._crab_heartbeats)

    def test_each_task_has_its_own_and_stopping_one_leaves_the_other(self):
        first, second = self.task(0), self.task(1)
        self.fire(first, luigi.Event.START)
        self.fire(second, luigi.Event.START)
        self.assertEqual(len(self.made), 2)
        self.fire(first, luigi.Event.SUCCESS)
        self.assertEqual([hb.exited for hb in self.made], [1, 0])
        self.assertIn(second.task_id, lc._crab_heartbeats)

    def test_a_task_that_must_not_beat_starts_nothing(self):
        workflow = self.workflow()
        self.tasks.append(workflow)
        self.fire(workflow, luigi.Event.START)
        self.fire(workflow, luigi.Event.SUCCESS)
        self.assertEqual(self.made, [])
        self.assertNotIn(workflow.task_id, lc._crab_heartbeats)


# --------------------------------------------------------------------------------------
# --<Task>-parallel-jobs / --<Task>-poll-interval reach the task they name
# --------------------------------------------------------------------------------------


class TheCommandLineReachesTheRequiredTask(FlafTaskTestCase):
    """luigi resolves `--WatchdogUp-parallel-jobs` for WatchdogUp only when nothing passes
    the parameter explicitly, and `req()` does: it copies the requiring task's own value,
    unlimited (-1) by default. So without `prefer_params_cli` the value given for the task
    that runs on CRAB was replaced by the root's, and the CRAB proxy -- seeing the option on
    the command line -- then left it unlimited instead of applying its default."""

    def resolve(self, args, cfg=None):
        self.crab_cfg(cfg)
        reset_cli_state()
        argv = [
            "WatchdogRoot",
            "--version",
            "v1",
            "--period",
            "P",
            "--workflow",
            "crab",
        ]
        with CmdlineParser.global_instance(argv + list(args)) as cp:
            root = cp.get_task_obj()
            up = WatchdogUp.req(root)
            out = types.SimpleNamespace(
                up_parallel_jobs=up.parallel_jobs,
                up_poll_param=float(up.poll_interval),
            )
            # the proxies apply the CRAB defaults, and read the command line to do so
            out.up_n_parallel = up.workflow_proxy.poll_data.n_parallel
            out.up_poll = float(up.poll_interval)
            out.root_n_parallel = root.workflow_proxy.poll_data.n_parallel
            out.root_poll = float(root.poll_interval)
            out.unlimited = up.workflow_proxy.n_parallel_max
        return out

    def test_a_prefixed_value_reaches_the_task_it_names(self):
        r = self.resolve(
            ["--WatchdogUp-parallel-jobs", "7", "--WatchdogUp-poll-interval", "9"]
        )
        self.assertEqual(r.up_parallel_jobs, 7)
        self.assertEqual(r.up_poll_param, 9.0)
        self.assertEqual(r.up_n_parallel, 7)
        self.assertEqual(r.up_poll, 9.0)

    def test_and_not_the_task_that_requires_it(self):
        r = self.resolve(
            ["--WatchdogUp-parallel-jobs", "7", "--WatchdogUp-poll-interval", "9"]
        )
        self.assertEqual(r.root_n_parallel, lc._CRAB_DEFAULT_PARALLEL_JOBS)
        self.assertEqual(r.root_poll, float(lc._CRAB_DEFAULT_POLL_INTERVAL))

    def test_the_crab_proxy_keeps_the_command_line_value_over_the_yaml(self):
        r = self.resolve(
            ["--WatchdogUp-parallel-jobs", "7", "--WatchdogUp-poll-interval", "9"],
            cfg={"parallel_jobs": 3000, "poll_interval": 4},
        )
        self.assertEqual(r.up_n_parallel, 7)
        self.assertEqual(r.up_poll, 9.0)

    def test_a_bare_value_reaches_every_required_task(self):
        r = self.resolve(
            ["--parallel-jobs", "7", "--poll-interval", "9"],
            cfg={"parallel_jobs": 3000, "poll_interval": 4},
        )
        self.assertEqual(r.up_parallel_jobs, 7)
        self.assertEqual((r.up_n_parallel, r.up_poll), (7, 9.0))
        self.assertEqual((r.root_n_parallel, r.root_poll), (7, 9.0))

    def test_without_any_option_both_get_the_crab_defaults(self):
        r = self.resolve([])
        self.assertEqual(r.up_n_parallel, lc._CRAB_DEFAULT_PARALLEL_JOBS)
        self.assertEqual(r.up_poll, float(lc._CRAB_DEFAULT_POLL_INTERVAL))
        self.assertEqual(r.root_n_parallel, lc._CRAB_DEFAULT_PARALLEL_JOBS)

    def test_a_value_addressed_to_the_root_task_counts_like_a_bare_one(self):
        # for the root task luigi reads `--WatchdogRoot-parallel-jobs` exactly as the bare
        # option, and req() copies it on just the same, so the required task keeps it too
        # instead of falling back to the yaml with or without one
        for cfg in ({"parallel_jobs": 3000, "poll_interval": 4}, None):
            with self.subTest(cfg=cfg):
                r = self.resolve(
                    [
                        "--WatchdogRoot-parallel-jobs",
                        "7",
                        "--WatchdogRoot-poll-interval",
                        "9",
                    ],
                    cfg=cfg,
                )
                self.assertEqual((r.root_n_parallel, r.root_poll), (7, 9.0))
                self.assertEqual((r.up_n_parallel, r.up_poll), (7, 9.0))

    def test_without_the_prefer_cli_entries_the_value_is_lost(self):
        # what these tests would see without the fix: the root's unlimited value wins, and
        # the proxy, finding the option on the command line, leaves it unlimited
        names = [
            p
            for p in lc.Task.prefer_params_cli
            if p not in ("parallel_jobs", "poll_interval")
        ]
        with mock.patch.object(lc.Task, "prefer_params_cli", names):
            r = self.resolve(
                ["--WatchdogUp-parallel-jobs", "7", "--WatchdogUp-poll-interval", "9"]
            )
        self.assertEqual(r.up_parallel_jobs, -1)
        self.assertEqual(r.up_n_parallel, r.unlimited)


if __name__ == "__main__":
    unittest.main()
