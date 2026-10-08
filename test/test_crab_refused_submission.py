#!/usr/bin/env python3
"""What a FLAF production does when the CRAB server refuses a submission, or has not scheduled it yet.

Ported from DSProd (test/test_refused_submission.py), where on 2026-09-12 a 36000-branch
production died. The CRAB server had refused its task -- `Status on the CRAB server:
SUBMITREFUSED`, `Warning: A site name T3_CH_CERN_HelixNebula_REHA that user specified is not in
the list of known CMS Processing Site Names` -- because the `T3_*` whitelist glob had been
expanded against a site list built by the wrong rule. A refused task never produces per-job
information, law reports that as an unreadable status, and the task was retried as if it were
slow: 4 attempts per poll with 15 s between them, for 16 consecutive polls, until the driver died.

Two things are pinned here. A task the server has *refused* is terminal, so its jobs are failed
and law submits them again as a new task. A task the server has merely not *scheduled* yet is
normal, so it costs neither the retry delay nor a step towards `max_unreadable_polls` -- law
0.1.20 does not know the `WAITING` status that every task now starts in.

A third is FLAF's own: a task that stays unreadable past `max_unreadable_polls` stops the run
through `CrabWorkflow.crab_poll_callback`, never by raising from `query()`, where law's thread
pool would swallow the exception and skip the callback for that poll.

Wherever a status response is involved, law's own `CrabJobManager.query` runs: only the `crab`
subprocess it starts is replaced (by `CrabCli`), together with the CMSSW sandbox environment,
which needs cvmfs.
"""

import collections
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import law  # noqa: E402

law.contrib.load("cms")

import law.contrib.cms.job as law_crab_job  # noqa: E402
from law.job.base import get_async_result_silent  # noqa: E402
from law.job.dashboard import NoJobDashboard  # noqa: E402
from law.parameter import NO_FLOAT  # noqa: E402
from law.workflow.remote import JobData, PollData  # noqa: E402

# The CI unit-test environment has no ROOT, and law_customizations reaches
# FLAF.Common.Utilities, which imports it at module level without using it on import. A
# placeholder stands in for that import only and is removed again, so that every other test
# module still sees ROOT as absent.
_root_placeholder = (
    "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None
)
if _root_placeholder:
    sys.modules["ROOT"] = types.ModuleType("ROOT")
try:
    from FLAF.run_tools import law_customizations as lc  # noqa: E402
finally:
    if _root_placeholder:
        sys.modules.pop("ROOT", None)

SANDBOX = "cmssw::CMSSW_14_0_0::arch=el9_amd64_gcc12"
TASK_NAME = "260912_054021:kandroso_crab_AnaTupleFileTask_v1_Run3_2022EE_a741224d"

#: the response of the DSProd incident, trimmed to the lines that decide the outcome
REFUSED_OUTPUT = """Rucio client intialized for account kandroso
CRAB project directory:\t\t/eos/.../crab_AnaTupleFileTask_v1_Run3_2022EE_a741224d
Task name:\t\t\t260912_054021:kandroso_crab_AnaTupleFileTask_v1_Run3_2022EE_a741224d
Grid scheduler - Task Worker:\tN/A yet - crab-prod-tw01
Status on the CRAB server:\tSUBMITREFUSED
Warning:\t\tA site name T3_CH_CERN_HelixNebula_REHA that user specified is not in the list of known CMS Processing Site Names
Log file is /eos/.../crab.log
"""

WAITING_OUTPUT = REFUSED_OUTPUT.replace(
    "Status on the CRAB server:\tSUBMITREFUSED",
    "Status on the CRAB server:\tWAITING on command SUBMIT",
)

SUBMITTED_OUTPUT = REFUSED_OUTPUT.replace(
    "Status on the CRAB server:\tSUBMITREFUSED",
    "Status on the CRAB server:\tSUBMITTED",
)

#: a task on a scheduler, publishing its per-job states
SCHEDULED_OUTPUT = (
    SUBMITTED_OUTPUT.replace(
        "Grid scheduler - Task Worker:\tN/A yet - crab-prod-tw01",
        "Grid scheduler - Task Worker:\tcrab3@vocms0199.cern.ch - crab-prod-tw01",
    )
    + "Status on the scheduler:\tSUBMITTED\n"
    + json.dumps({str(i): {"State": "running"} for i in (1, 2, 3)})
    + "\n"
)

#: what law genuinely cannot read: no server status at all. A `SUBMITTED` response without job
#: data is NOT this case -- law accepts that status itself and reports the jobs pending.
UNREADABLE_OUTPUT = (
    "Rucio client intialized for account kandroso\nSome unexpected output\n"
)

#: what the workflow hands every status query (CrabWorkflow.crab_job_kwargs_query)
QUERY_KWARGS = dict(lc.CrabWorkflow.crab_job_kwargs_query)


def manager(**attrs):
    m = lc.FLAFCrabJobManager(sandbox_name=SANDBOX)
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


def job_ids(m, n=3, proj_dir="/proj"):
    return [m.JobId(i, TASK_NAME, proj_dir) for i in range(1, n + 1)]


def printed_text(printed):
    return [str(c.args[0]) for c in printed.call_args_list if c.args]


class CrabCli:
    """The `crab` command law runs, answering every call with one scripted response.

    Only the subprocess is replaced: law's `CrabJobManager.query` builds the command, raises on a
    non-zero exit with the output in its message, and hands a zero exit to FLAF's
    `parse_query_output` -- the path a production takes.
    """

    def __init__(self, out, code=0):
        self.out = out
        self.code = code
        self.commands = []

    def __call__(self, cmd, *args, **kwargs):
        self.commands.append(cmd)
        return self.code, self.out, None


def crab_answers(out, code=0):
    """Patches for a status query answered by `out`; returns (patchers, cli)."""
    cli = CrabCli(out, code)
    patchers = [
        mock.patch.object(law_crab_job, "interruptable_popen", side_effect=cli),
        # the sandbox env is built from cvmfs, which nothing here needs
        mock.patch.object(
            lc.FLAFCrabJobManager,
            "cmssw_env",
            new_callable=mock.PropertyMock,
            return_value={},
        ),
    ]
    return patchers, cli


def query(m, out, proj_dir="/proj", n=3, code=0, ids=True):
    """One poll of one CRAB task through law's real query; returns (result, ids, slept, cli)."""
    ids = job_ids(m, n, proj_dir) if ids else None
    patchers, cli = crab_answers(out, code)
    with patchers[0], patchers[1], mock.patch.object(lc.time, "sleep") as slept:
        result = m.query(proj_dir, job_ids=ids, **QUERY_KWARGS)
    return result, ids, slept, cli


class ReadingTheServerStatus(unittest.TestCase):
    """law's regex is anchored and applied per line; the obvious way to use it finds nothing."""

    def test_the_status_is_found_the_way_law_finds_it(self):
        self.assertEqual(
            lc.FLAFCrabJobManager.server_status(REFUSED_OUTPUT), "SUBMITREFUSED"
        )
        self.assertEqual(
            lc.FLAFCrabJobManager.server_state(REFUSED_OUTPUT), "SUBMITREFUSED"
        )

    def test_searching_the_whole_response_would_have_found_nothing(self):
        """The trap this pins: `query_server_status_cre` carries no `re.MULTILINE`."""
        self.assertIsNone(
            lc.FLAFCrabJobManager.query_server_status_cre.search(REFUSED_OUTPUT)
        )

    def test_the_command_half_is_dropped(self):
        self.assertEqual(lc.FLAFCrabJobManager.server_state(WAITING_OUTPUT), "WAITING")

    def test_the_reason_comes_from_the_warning_line(self):
        """A refused task sets no `Failure message from server`; the reason is a task warning."""
        warnings = lc.FLAFCrabJobManager.server_warnings(REFUSED_OUTPUT)
        self.assertEqual(len(warnings), 1)
        self.assertIn("T3_CH_CERN_HelixNebula_REHA", warnings[0])

    def test_an_empty_response_says_nothing_rather_than_raising(self):
        self.assertIsNone(lc.FLAFCrabJobManager.server_status(""))
        self.assertEqual(lc.FLAFCrabJobManager.server_state(None), "")


class TellingApartTheThreeUnreadableResponses(unittest.TestCase):
    """All three reach law's "no per-job information" error; only one is terminal."""

    def parse(self, out):
        return lc.FLAFCrabJobManager.parse_query_output(out, "/proj", [1, 2, 3])

    def test_a_refused_task_is_raised_as_refused(self):
        with self.assertRaises(lc.CrabTaskRefused) as caught:
            self.parse(REFUSED_OUTPUT)
        self.assertEqual(caught.exception.state, "SUBMITREFUSED")
        self.assertIn("T3_CH_CERN_HelixNebula_REHA", str(caught.exception))

    def test_a_waiting_task_is_raised_as_not_scheduled_yet(self):
        with self.assertRaises(lc.CrabTaskNotScheduledYet) as caught:
            self.parse(WAITING_OUTPUT)
        self.assertEqual(caught.exception.state, "WAITING")

    def test_anything_else_keeps_the_old_diagnostic(self):
        with self.assertRaises(Exception) as caught:
            self.parse(UNREADABLE_OUTPUT)
        for kind in (lc.CrabTaskRefused, lc.CrabTaskNotScheduledYet):
            self.assertNotIsInstance(caught.exception, kind)
        self.assertIn("first lines of what crab returned", str(caught.exception))
        self.assertIn("Some unexpected output", str(caught.exception))

    def test_a_task_that_still_reports_its_jobs_is_never_touched(self):
        """The expensive mistake: blanket-failing a task whose real per-job states are right there.

        The check runs only after law has refused the response, so a response law *can* parse --
        whatever the server status says -- is returned unchanged.
        """
        parsed = {"1": "real per-job states"}
        with mock.patch.object(
            law.cms.CrabJobManager, "parse_query_output", return_value=parsed
        ):
            self.assertEqual(self.parse(REFUSED_OUTPUT), parsed)


class WhenTheServerRefuses(unittest.TestCase):
    def query(self, m, out=REFUSED_OUTPUT, n=3, proj_dir="/proj", ours=True):
        if ours:
            m._submitted_projects.add(proj_dir)
        result, ids, slept, _ = query(m, out, proj_dir=proj_dir, n=n)
        return result, ids, slept

    def test_every_job_is_reported_failed_so_law_submits_them_again(self):
        m = manager()
        result, ids, _ = self.query(m)
        self.assertEqual(set(result), set(ids))
        for data in result.values():
            self.assertEqual(data["status"], m.FAILED)

    def test_no_site_is_blamed_for_a_task_that_never_ran(self):
        """`_harvest_site_stats` charges a site only for a failure with a job-level code."""
        m = manager()
        result, _, _ = self.query(m)
        self.assertTrue(all(data["code"] is None for data in result.values()))

    def test_the_error_names_the_server_state(self):
        m = manager()
        result, _, _ = self.query(m)
        for data in result.values():
            self.assertIn("SUBMITREFUSED", str(data["error"]))

    def test_nothing_waits_for_a_verdict_that_cannot_change(self):
        m = manager()
        result, _, slept, cli = query(m, REFUSED_OUTPUT)
        slept.assert_not_called()
        self.assertEqual(len(cli.commands), 1, "a refusal is not queried again")

    def test_it_does_not_count_towards_the_unreadable_ceiling(self):
        m = manager()
        m._unreadable["/proj"] = 7
        self.query(m)
        self.assertNotIn("/proj", m._unreadable)

    def test_the_reason_is_reported_once_per_task_not_once_per_poll(self):
        m = manager(max_refused_submissions=99)
        with mock.patch("builtins.print") as printed:
            self.query(m)
            self.query(m)
        said = "\n".join(printed_text(printed))
        self.assertEqual(said.count("T3_CH_CERN_HelixNebula_REHA"), 1)
        self.assertIn("preset=site-names", said)

    def test_the_cached_site_list_is_dropped_so_the_next_wave_re_reads_cric(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cms_psn_sites.json")
            with open(cache, "w") as f:
                json.dump(["T3_CH_CERN_HelixNebula_REHA"], f)
            m = manager(site_cache_path=cache)
            self.query(m)
            self.assertFalse(os.path.exists(cache))

    def test_one_refusal_alone_does_not_stop_the_run(self):
        """It can be a stale site list, which is dropped here, so the next wave differs."""
        m = manager()
        self.query(m)
        self.assertIsNone(m.stop_reason)

    def test_a_second_refused_submission_records_a_stop_reason_that_says_why(self):
        m = manager()
        self.query(m, proj_dir="/proj-1")
        self.query(m, proj_dir="/proj-2")
        self.assertIn("refused", m.stop_reason)
        self.assertIn("T3_CH_CERN_HelixNebula_REHA", m.stop_reason)

    def test_polling_one_refused_task_again_is_one_refusal_not_two(self):
        """The counter is per submission; law re-queries a project every poll."""
        m = manager()
        for _ in range(5):
            self.query(m, proj_dir="/proj-1")
        self.assertIsNone(m.stop_reason)

    def test_the_jobs_are_still_failed_on_the_poll_that_stops_the_run(self):
        """law must not be left with a task whose jobs were never marked, whichever way it ends."""
        m = manager()
        self.query(m, proj_dir="/proj-1")
        result, ids, _ = self.query(m, proj_dir="/proj-2")
        self.assertEqual(set(result), set(ids))
        self.assertTrue(all(d["status"] == m.FAILED for d in result.values()))
        self.assertIsNotNone(m.stop_reason)


class ARefusalBehindANonZeroExit(unittest.TestCase):
    """law raises before parsing when `crab status` exits non-zero, with the output inside its
    message: the refusal must be recognised there too, or the retry storm comes back whenever
    crab reports a failing exit."""

    def test_law_puts_the_response_into_its_error(self):
        """The premise, pinned against the installed law rather than assumed."""
        m = manager()
        patchers, _ = crab_answers(REFUSED_OUTPUT, code=1)
        with patchers[0], patchers[1], self.assertRaises(Exception) as caught:
            law.cms.CrabJobManager.query(m, "/proj", job_ids=job_ids(m), **QUERY_KWARGS)
        self.assertNotIsInstance(caught.exception, lc.CrabTaskRefused)
        self.assertEqual(
            lc.FLAFCrabJobManager.server_state(str(caught.exception)), "SUBMITREFUSED"
        )

    def test_the_jobs_are_failed_at_once_without_a_retry(self):
        m = manager()
        m._submitted_projects.add("/proj")
        result, ids, slept, cli = query(m, REFUSED_OUTPUT, code=1)
        self.assertEqual(set(result), set(ids))
        for data in result.values():
            self.assertEqual(data["status"], m.FAILED)
            self.assertIsNone(data["code"])
            self.assertIn("T3_CH_CERN_HelixNebula_REHA", str(data["error"]))
        slept.assert_not_called()
        self.assertEqual(len(cli.commands), 1)
        self.assertNotIn("/proj", m._unreadable)
        self.assertEqual(m._refused_projects, {"/proj"})

    def test_a_failing_exit_without_a_refusal_is_still_retried(self):
        """The must-not-break side: a non-zero exit of any other kind is a slow task."""
        m = manager()
        result, _, slept, cli = query(m, UNREADABLE_OUTPUT, code=1)
        for data in result.values():
            self.assertEqual(data["status"], m.PENDING)
        self.assertEqual(slept.call_count, m.query_retries)
        self.assertEqual(len(cli.commands), m.query_retries + 1)
        self.assertEqual(m._unreadable["/proj"], 1)
        self.assertEqual(m._refused_projects, set())


class WhenTheServerHasNotScheduledItYet(unittest.TestCase):
    """Every task now starts in WAITING, which law 0.1.20 does not accept."""

    def test_its_jobs_are_pending_not_failed(self):
        m = manager()
        result, ids, _, _ = query(m, WAITING_OUTPUT)
        self.assertEqual(set(result), set(ids))
        for data in result.values():
            self.assertEqual(data["status"], m.PENDING)

    def test_it_costs_neither_a_retry_delay_nor_a_step_towards_the_ceiling(self):
        m = manager()
        _, _, slept, cli = query(m, WAITING_OUTPUT)
        slept.assert_not_called()
        self.assertEqual(len(cli.commands), 1)
        self.assertNotIn("/proj", m._unreadable)

    def test_but_it_is_still_reported_once(self):
        m = manager()
        with mock.patch("builtins.print") as printed:
            query(m, WAITING_OUTPUT)
            query(m, WAITING_OUTPUT)
        said = printed_text(printed)
        self.assertEqual(len([s for s in said if "WAITING" in s]), 1)


class ASlowTaskIsStillTreatedAsSlow(unittest.TestCase):
    """The behaviour that must not regress: an unreadable response of any other kind."""

    def test_an_unreadable_response_keeps_the_old_path(self):
        m = manager()
        result, _, slept, cli = query(m, UNREADABLE_OUTPUT)
        for data in result.values():
            self.assertEqual(data["status"], m.PENDING)
        self.assertEqual(m._unreadable["/proj"], 1)
        self.assertEqual(slept.call_count, m.query_retries)
        self.assertEqual(len(cli.commands), m.query_retries + 1)


class ARefusalInheritedFromAnEarlierRun(unittest.TestCase):
    """The DSProd 2026-09-13 restart: three tasks refused *before* the fix was deployed were
    re-polled by the corrected run, counted as its own refusals, and stopped it on the first poll
    -- before the 5000 branches they held could be submitted again with the corrected whitelist.
    """

    def query(self, m, proj_dir):
        result, ids, _, _ = query(m, REFUSED_OUTPUT, proj_dir=proj_dir)
        return result, ids

    def test_old_refusals_do_not_stop_a_corrected_run(self):
        m = manager()
        for proj in ("/old-1", "/old-2", "/old-3"):
            self.query(m, proj)
        self.assertIsNone(m.stop_reason)

    def test_but_their_jobs_are_still_failed_so_the_branches_come_back(self):
        """This is the recovery: law resubmits them into a task built with the corrected list."""
        m = manager()
        result, ids = self.query(m, "/old-1")
        self.assertEqual(set(result), set(ids))
        self.assertTrue(all(d["status"] == m.FAILED for d in result.values()))

    def test_the_report_says_it_was_not_this_run_that_sent_it(self):
        m = manager()
        with mock.patch("builtins.print") as printed:
            self.query(m, "/old-1")
        self.assertIn("earlier run", "\n".join(printed_text(printed)))

    def test_a_refusal_of_this_runs_own_submission_still_counts(self):
        m = manager()
        self.query(m, "/old-1")
        self.query(m, "/old-2")
        m._submitted_projects.update({"/new-1", "/new-2"})
        self.query(m, "/new-1")
        self.assertIsNone(m.stop_reason)
        self.query(m, "/new-2")
        self.assertIn("made by this run", m.stop_reason)

    def test_submitting_is_what_marks_a_task_as_this_runs(self):
        m = manager()
        job_id = m.JobId(1, "task", "/fresh")
        with mock.patch.object(law.cms.CrabJobManager, "submit", return_value=[job_id]):
            m.submit("job.jdl")
        self.assertIn("/fresh", m._submitted_projects)


class AnUnscheduledTaskCannotStallForever(unittest.TestCase):
    """Accepting WAITING must not remove the guard against a task that never leaves it."""

    def test_a_long_wait_is_repeated_in_the_log_rather_than_said_once(self):
        m = manager(unscheduled_report_every=3)
        with mock.patch("builtins.print") as printed:
            for _ in range(7):
                query(m, WAITING_OUTPUT)
        said = printed_text(printed)
        self.assertEqual(len([x for x in said if "WAITING" in x]), 3)
        self.assertIn("polls", said[-1])

    def test_a_task_that_never_reaches_a_scheduler_stops_the_run(self):
        m = manager(max_unscheduled_polls=4)
        for _ in range(4):
            query(m, WAITING_OUTPUT)
        self.assertIsNone(m.stop_reason)
        result, _, _, _ = query(m, WAITING_OUTPUT)
        self.assertIn("WAITING", m.stop_reason)
        self.assertIn("TaskWorker", m.stop_reason)
        # recorded, not raised: the jobs of the poll that stops the run are still reported
        self.assertTrue(all(d["status"] == m.PENDING for d in result.values()))

    def test_a_task_that_gets_scheduled_forgets_the_wait(self):
        m = manager(max_unscheduled_polls=4)
        for _ in range(3):
            query(m, WAITING_OUTPUT)
        result, _, _, _ = query(m, SCHEDULED_OUTPUT)
        self.assertTrue(all(d["status"] == m.RUNNING for d in result.values()))
        self.assertNotIn("/proj", m._unscheduled)


class AFreshlySubmittedTaskIsLawsOwnBusiness(unittest.TestCase):
    """`SUBMITTED` with no job data yet is accepted by law itself, and must stay that way."""

    def test_law_reports_its_jobs_pending_without_flaf_intervening(self):
        result = lc.FLAFCrabJobManager.parse_query_output(
            SUBMITTED_OUTPUT, "/proj", [1, 2]
        )
        self.assertEqual(
            [d["status"] for d in result.values()],
            [lc.FLAFCrabJobManager.PENDING] * 2,
        )


class NothingToDegradeTo(unittest.TestCase):
    """Without job ids and without a readable crab.log there is no job to report pending, so the
    error that made the response unreadable is raised rather than replaced by a crash.
    """

    def test_the_query_error_is_raised(self):
        m = manager()
        with tempfile.TemporaryDirectory() as proj_dir:
            with self.assertRaises(Exception) as caught:
                query(m, UNREADABLE_OUTPUT, proj_dir=proj_dir, ids=False)
        self.assertIn("first lines of what crab returned", str(caught.exception))
        self.assertIn("Some unexpected output", str(caught.exception))

    def test_all_pending_raises_the_error_it_was_given(self):
        m = manager()
        error = RuntimeError("the status that could not be read")
        with tempfile.TemporaryDirectory() as proj_dir:
            with self.assertRaises(RuntimeError) as caught:
                m._all_pending(proj_dir, None, error)
        self.assertIs(caught.exception, error)

    def test_a_readable_crab_log_still_gives_the_jobs(self):
        """The must-not-break side: job ids are read from crab.log when the caller has none."""
        m = manager()
        with tempfile.TemporaryDirectory() as proj_dir:
            with open(os.path.join(proj_dir, "crab.log"), "w") as f:
                f.write(
                    "config.Data.totalUnits = 2\n"
                    f"DEBUG 2026-09-12 05:40:21 Task name:\t{TASK_NAME}\n"
                )
            result = m._all_pending(proj_dir, None, RuntimeError("unused"))
        self.assertEqual(sorted(j.crab_num for j in result), [1, 2])
        self.assertTrue(all(d["status"] == m.PENDING for d in result.values()))


class AnUnreadableTaskStopsTheRunFromTheCallback(unittest.TestCase):
    """FLAF deviation from DSProd, which raised from `query()` past `max_unreadable_polls`.

    law swallows that raise (see `StoppingTheRunActuallyStopsIt`), counts it as one more failed
    poll and skips the poll callback on it. FLAF records the stop instead and keeps reporting the
    jobs pending, so the callback is reached on the very poll that crosses the ceiling.
    """

    def unreadable_polls(self, m, n):
        results = []
        with mock.patch("builtins.print"):
            for _ in range(n):
                results.append(query(m, UNREADABLE_OUTPUT)[0])
        return results

    def test_up_to_the_ceiling_nothing_is_recorded(self):
        m = manager(max_unreadable_polls=3)
        self.unreadable_polls(m, 3)
        self.assertIsNone(m.stop_reason)
        self.assertEqual(m._unreadable["/proj"], 3)

    def test_the_poll_past_the_ceiling_returns_pending_and_records_the_stop(self):
        m = manager(max_unreadable_polls=3)
        self.unreadable_polls(m, 3)
        result = self.unreadable_polls(m, 1)[0]
        self.assertEqual(len(result), 3)
        self.assertTrue(all(d["status"] == m.PENDING for d in result.values()))
        self.assertIn("unreadable for 4 consecutive polls", m.stop_reason)
        self.assertIn("first lines of what crab returned", m.stop_reason)

    def test_the_poll_callback_raises_it(self):
        m = manager(max_unreadable_polls=1)
        self.unreadable_polls(m, 2)
        workflow = mock.Mock(spec=lc.CrabWorkflow)
        workflow._flaf_crab_job_manager = m
        with self.assertRaises(RuntimeError) as caught:
            lc.CrabWorkflow.crab_poll_callback(workflow, mock.Mock())
        self.assertIn("unreadable for 2 consecutive polls", str(caught.exception))

    def test_a_readable_poll_resets_the_count(self):
        m = manager(max_unreadable_polls=3)
        self.unreadable_polls(m, 3)
        query(m, SCHEDULED_OUTPUT)
        self.unreadable_polls(m, 3)
        self.assertIsNone(m.stop_reason)


class _RefusedSubmissionTestCrabWorkflow(lc.CrabWorkflow):
    """lc.CrabWorkflow with its abstract hooks filled, so that an instance can exist. Never
    __init__'d: constructing a luigi task wants its parameters and a scheduler."""

    def create_branch_map(self):
        return {0: None, 1: None}

    def run(self):
        pass

    def ana_data_path(self):
        return self._data_dir


class PollLoopNotStopped(AssertionError):
    pass


class RealPollLoop:
    """law's real `poll()` of a FLAF CRAB workflow proxy over scripted `crab status` answers.

    The proxy is never __init__'d (as in DSProd's tests); its state is what law's __init__ would
    set, and its job manager is built by the real `create_job_manager` -> FLAF
    `crab_create_job_manager` chain. Replaced: the `crab` subprocess, the sandbox env, the site
    record and the watchdog (storage-backed), the Kerberos renewal, and `submit`/`dump_job_data`,
    which nothing here may reach.
    """

    #: more polls than any test here needs to see the loop end
    max_iterations = 20

    def __init__(self, case, out, n_jobs=2, retries=0, poll_fails=2, **manager_attrs):
        data_dir = tempfile.mkdtemp(prefix="flaf_refused_test_")
        case.addCleanup(shutil.rmtree, data_dir, True)

        patchers, self.cli = crab_answers(out)
        patchers += [
            mock.patch.object(lc.time, "sleep"),
            mock.patch.object(lc.SiteStats, "shared", return_value=None),
            mock.patch.object(
                lc.CrabWorkflow,
                "job_watchdog",
                return_value=types.SimpleNamespace(enabled=False),
            ),
            mock.patch.object(lc, "update_kinit"),
            mock.patch.object(lc, "require_fresh_negatives"),
        ]
        for patcher in patchers:
            patcher.start()
            case.addCleanup(patcher.stop)

        wf = object.__new__(_RefusedSubmissionTestCrabWorkflow)
        wf._data_dir = data_dir
        wf.global_params = {}
        for name, value in dict(
            poll_interval=0.0,
            walltime=NO_FLOAT,
            acceptance=1.0,
            tolerance=0.0,
            retries=retries,
            poll_fails=poll_fails,
            no_poll=False,
            clear_logs=False,
            check_unreachable_acceptance=False,
            scheduler_messages=None,
            task_id="_RefusedSubmissionTestCrabWorkflow__refused_test",
        ).items():
            setattr(wf, name, value)
        self.messages = []
        wf.publish_message = lambda msg, *a, **k: self.messages.append(str(msg))
        wf.publish_progress = lambda *a, **k: None
        # law calls this at the top of every iteration: a loop that the code under test fails
        # to stop ends here, as a test failure rather than a hang
        self.iterations = 0
        wf._handle_scheduler_messages = self._count_iteration
        self.workflow = wf

        proxy = object.__new__(lc._FLAFCrabWorkflowProxy)
        proxy.task = wf
        proxy.dashboard = NoJobDashboard()
        proxy.poll_data = PollData(
            n_parallel=None, n_finished_min=-1, n_failed_max=-1, n_active=0
        )
        proxy.job_data = JobData()
        proxy.job_manager = proxy.create_job_manager(sandbox_name=SANDBOX, threads=1)
        proxy.job_manager.status_diff_styles["unsubmitted"] = ({}, {}, {})
        proxy._job_manager_setup_kwargs = {}
        proxy._skip_jobs = {}
        proxy._job_retries = collections.defaultdict(int)
        proxy._submitted = False
        proxy._existing_branches = set()
        proxy._tracking_url = None
        proxy._printed_first_log = False
        proxy._initial_process_resources = None
        # every job in flight already, so law's loop never reaches submit()
        proxy._set_parallel_jobs(n_jobs)
        proxy.submit = mock.Mock(name="submit")
        proxy.dump_job_data = mock.Mock(name="dump_job_data")
        self.proxy = proxy

        self.manager = proxy.job_manager
        for key, value in manager_attrs.items():
            setattr(self.manager, key, value)
        for job_num, job_id in enumerate(job_ids(self.manager, n_jobs), 1):
            proxy.job_data.jobs[job_num] = JobData.job_data(
                job_id=job_id, branches=[job_num - 1]
            )

    def _count_iteration(self):
        self.iterations += 1
        if self.iterations > self.max_iterations:
            raise PollLoopNotStopped(
                f"law's poll loop was still running after {self.max_iterations} polls"
            )

    def run(self):
        with mock.patch("builtins.print"):
            return self.proxy.poll()

    @property
    def polls(self):
        return len(self.cli.commands) // (self.manager.query_retries + 1)


class StoppingTheRunActuallyStopsIt(unittest.TestCase):
    """Where the stop is raised decides whether it stops anything at all.

    law queries through a thread pool and `get_async_result_silent` turns an exception there into
    the *result*, so raising from `query` would be counted as one more unreadable poll: the refused
    task's jobs would stay unfailed and every other project would lose that poll's status. The poll
    callback is the one hook the loop calls outside its own error handling.
    """

    def test_law_would_swallow_a_raise_from_query(self):
        """The premise, pinned against the installed law rather than assumed."""

        class Boom:
            def get(self, timeout=None):
                raise RuntimeError("stop the run")

        self.assertIsInstance(get_async_result_silent(Boom()), RuntimeError)

    def test_laws_thread_pool_turns_a_raise_from_query_into_the_result(self):
        """The same premise one level up, through the call law's poll loop makes."""
        m = manager()
        ids = job_ids(m)
        with mock.patch.object(
            lc.FLAFCrabJobManager, "query", side_effect=RuntimeError("stop the run")
        ):
            result = m.query_group(ids, **QUERY_KWARGS)
        self.assertEqual(set(result), set(ids))
        self.assertTrue(all(isinstance(v, RuntimeError) for v in result.values()))

    def test_the_poll_callback_raises_what_the_manager_recorded(self):
        workflow = mock.Mock(spec=lc.CrabWorkflow)
        workflow._flaf_crab_job_manager = manager(stop_reason="two refused submissions")
        with self.assertRaises(RuntimeError) as caught:
            lc.CrabWorkflow.crab_poll_callback(workflow, mock.Mock())
        self.assertIn("two refused submissions", str(caught.exception))

    def test_the_manager_that_polls_is_the_one_the_callback_reads(self):
        """The single link in the stop chain: without this assignment nothing is ever raised."""
        workflow = object.__new__(_RefusedSubmissionTestCrabWorkflow)
        workflow._data_dir = "/data"
        workflow.global_params = {}
        built = manager()
        stats = object()
        watchdog = types.SimpleNamespace(enabled=False)
        with mock.patch.object(
            law.cms.CrabWorkflow, "crab_create_job_manager", return_value=built
        ), mock.patch.object(
            lc.SiteStats, "shared", return_value=stats
        ) as shared, mock.patch.object(
            lc.CrabWorkflow, "job_watchdog", return_value=watchdog
        ):
            returned = workflow.crab_create_job_manager()
        self.assertIs(returned, built)
        self.assertIs(workflow._flaf_crab_job_manager, built)
        self.assertIs(built.site_stats, stats)
        self.assertIs(built.watchdog, watchdog)
        self.assertEqual(built.site_cache_path, "/data/cms_psn_sites.json")
        self.assertEqual(shared.call_args.args[0], "/data/crab_site_stats.json")

    def test_it_does_nothing_but_renew_when_there_is_nothing_to_report(self):
        workflow = mock.Mock(spec=lc.CrabWorkflow)
        workflow._flaf_crab_job_manager = manager()
        workflow._crab_kinit_update = mock.Mock()
        workflow._watchdog_refresh = None
        workflow.job_watchdog.return_value = types.SimpleNamespace(enabled=False)
        self.assertTrue(lc.CrabWorkflow.crab_poll_callback(workflow, mock.Mock()))
        workflow._crab_kinit_update.assert_called_once_with()

    def test_an_unreadable_task_ends_laws_real_poll_loop_on_the_poll_past_the_ceiling(
        self,
    ):
        loop = RealPollLoop(self, UNREADABLE_OUTPUT, max_unreadable_polls=2)
        self.assertIs(loop.workflow._flaf_crab_job_manager, loop.manager)
        with self.assertRaises(RuntimeError) as caught:
            loop.run()
        self.assertIn("unreadable for 3 consecutive polls", str(caught.exception))
        self.assertEqual(loop.polls, 3)
        loop.proxy.submit.assert_not_called()
        # every poll reached the status line: none of them was a failed query to law
        self.assertEqual(len(loop.messages), 3)

    def test_a_second_refused_submission_ends_laws_real_poll_loop(self):
        # one retry each, so that law's tolerance check is not what ends the loop
        loop = RealPollLoop(self, REFUSED_OUTPUT, n_jobs=2, retries=1)
        loop.manager._submitted_projects.update({"/proj"})
        loop.manager._refused_projects.add("/earlier-submission-of-this-run")
        with self.assertRaises(RuntimeError) as caught:
            loop.run()
        self.assertIn("made by this run", str(caught.exception))
        self.assertEqual(len(loop.cli.commands), 1)
        # the jobs were failed -- and law booked the attempt -- before the run stopped, so
        # law's record is consistent whichever way the run ends
        for job_num, data in loop.proxy.job_data.jobs.items():
            self.assertEqual(data["status"], loop.manager.RETRY)
            self.assertIn("SUBMITREFUSED", str(data["error"]))
            self.assertEqual(loop.proxy._job_retries[job_num], 1)

    def test_a_waiting_task_keeps_laws_real_poll_loop_going(self):
        """The must-not-break side: nothing stops a run whose task is merely unscheduled."""
        loop = RealPollLoop(self, WAITING_OUTPUT, max_unscheduled_polls=3)
        with self.assertRaises(RuntimeError) as caught:
            loop.run()
        self.assertIn("TaskWorker", str(caught.exception))
        # three polls tolerated, the fourth crosses the ceiling: never a retry, never a sleep
        self.assertEqual(len(loop.cli.commands), 4)
        self.assertEqual(len(loop.messages), 4)


if __name__ == "__main__":
    unittest.main()
