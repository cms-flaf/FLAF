#!/usr/bin/env python3
"""Why a CRAB job failed, in the job's own words.

CRAB's exit code is a label, not a diagnosis: every one of the 4197 failures of one DSProd
production carried exit 5, "Error while running CMSSW", and law repeats exactly that. The
FLAF CI CRAB chain is no different -- AnaTupleFileTask job 4 of ci_crab_20260829_024357
failed at T2_US_Nebraska with `"Error": [5, "Error while running CMSSW:\\n", {}]`. What says
what happened is the exception the payload raised, and it sits in the job's stdout on the
schedd, whose URL law already records as `extra["log_file"]`.

These tests pin reading the payload's error out of a job's stdout (on DSProd's and on real
FLAF CRAB output), fetching only the tail of that stdout without holding up the poll, and
reporting each failed attempt once -- both on hand-made job data and on the job data law
0.1.21 itself parses out of a `crab status --json` response.

Ported from DSProd test/test_failure_reporting.py (the merge-group message tests are
DSProd-specific and are not ported).
"""

import importlib.util
import itertools
import json
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

# law_customizations imports FLAF.Common.Setup, which imports ROOT at module level. The
# unit-test venv of the GitHub Actions workflow has no ROOT and nothing exercised here
# touches it, so an empty module stands in for this import only (an attribute used at
# import time would fail loudly rather than be absorbed by a mock) and is withdrawn
# afterwards: a later test that needs ROOT must still fail to import it.
_root_stubbed = "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None
if _root_stubbed:
    sys.modules["ROOT"] = types.ModuleType("ROOT")
try:
    import law  # noqa: F401
    from FLAF.run_tools import law_customizations as lc
finally:
    if _root_stubbed:
        del sys.modules["ROOT"]

Manager = lc.FLAFCrabJobManager

#: the tail of one of the three DSProd job logs of 2026-09-15, trimmed
REAL_LOG = """\
== CMSSW: Begin processing
/srv/gWMS-CMSRunAnalysis.sh: line 45: ./submit_env.sh: No such file or directory
== CMSSW: Traceback (most recent call last):
== CMSSW:   File "<string>", line 1, in <module>
== CMSSW: ImportError: No module named FWCore.ParameterSet.Config
== CMSSW: branch=65
== CMSSW: Traceback (most recent call last):
== CMSSW:   File "/srv/dsprod/tasks.py", line 1270, in run
== CMSSW:     raise RuntimeError(
== CMSSW: RuntimeError: 1 of 50 staged nano files of this merge group are gone -- seeds 45, e.g. \
root://cmseos.fnal.gov//eos/uscms/store/user/x/nano_v12_45.root -- although their seeds are \
recorded as produced.
Error executing application in CMSSW environment.
"""

#: a real FLAF CRAB job stdout, trimmed: AnaTupleFileTask branch 0 of the CI CRAB chain
#: ci_crab_20260812_161958 (the stdall.txt law tees from the job's stdout, staged to EOS).
#: Every FLAF job that relocates the CMSSW bundle leaves a harmless scram FileExistsError
#: behind during bootstrap; the payload's own failure comes after it.
FLAF_REAL_LOG = """\
bootstrap: relocating CMSSW CMSSW_16_0_6 -> /srv/job_Wd4YqAOF1qHD/bundle/soft/CMSSW_16_0_6
Traceback (most recent call last):
  File "/cvmfs/cms.cern.ch/share/lcg/SCRAMV1/V3_00_95/bin/scram.py", line 114, in <module>
    symlink(relobj, locobj)
FileExistsError: [Errno 17] File exists: '/cvmfs/cms.cern.ch/el9_amd64_gcc13/cms/cmssw/\
CMSSW_16_0_6/objs/el9_amd64_gcc13' -> '/srv/job_Wd4YqAOF1qHD/bundle/soft/CMSSW_16_0_6/external/\
el9_amd64_gcc13/objs-base'
bootstrap: WARNING: scram ProjectRename failed; applying path sed fallback

-- run task branch 0 -------------------------------------------------------------------------------

Traceback (most recent call last):
  File "/srv/job_Wd4YqAOF1qHD/bundle/AnaProd/anaTupleDef.py", line 224, in Initialize
    ROOT.gROOT.ProcessLine(
cppyy.gbl.cms.Exception: long TROOT::ProcessLine(const char* line, Int_t* error = nullptr) =>
    Exception: An exception of category 'InvalidMetaGraphDef' occurred.
Exception Message:
error while loading metaGraphDef from 'HHbtag_v3_par_0': NOT_FOUND

Traceback (most recent call last):
  File "/srv/job_Wd4YqAOF1qHD/bundle/FLAF/RunKit/run_tools.py", line 137, in ps_call
    raise PsCallError(cmd_str, proc.returncode)
FLAF.RunKit.run_tools.PsCallError: Error while running "python3 -u /srv/job_Wd4YqAOF1qHD/\
bundle/FLAF/AnaProd/anaTupleProducer.py --period Run3_2022EE". Error code: 1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/srv/job_Wd4YqAOF1qHD/bundle/FLAF/AnaProd/tasks.py", line 279, in run
    raise RuntimeError("anaTupleProducer failed.")
RuntimeError: anaTupleProducer failed.
INFO: Informed scheduler that task   AnaTupleFileTask_0__False_b42dcddd8f   has status   FAILED
===== Luigi Execution Summary =====
anaTupleProducer failed: Error while running "python3 -u /srv/job_Wd4YqAOF1qHD/bundle/FLAF/\
AnaProd/anaTupleProducer.py --period Run3_2022EE". Error code: 1
task exit code: 40
execution of branch 0 failed (exit code 40), stop job
"""

#: a second real FLAF failure (ci_crab_20260808_222110), whose message runs onto a second line
FLAF_LFN_LOG = """\
  File "/srv/job_8R9m6tlWbghT/bundle/FLAF/RunKit/grid_tools.py", line 521, in lfn_to_pfn
    raise RuntimeError(
RuntimeError: lfn_to_pfn: unable to resolve PFN for T2_CH_CERN:/store/group/phys_higgs/\
HLepRare/skim_2025_v1/Run3_2022EE and no cached value is available. Rucio may be unavailable \
(CannotAuthenticate: Cannot authenticate.
Details: Cannot authenticate to account cmsplt01 with given credentials).
INFO: Informed scheduler that task   AnaTupleFileTask_0__False_bf8d030524   has status   FAILED
task exit code: 40
"""


def cmssw_prefixed(text):
    """The same stream as CRAB's job_out carries it, behind the CMSSW wrapper's prefix."""
    return "".join(f"== CMSSW: {line}\n" for line in text.splitlines())


def manager(**attrs):
    m = Manager(sandbox_name="cmssw::CMSSW_14_0_0::arch=el9_amd64_gcc12")
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def failed(crab_num, log="http://x/job_out.1.0.txt", code=5, site="T2_CH_CERN"):
    job_id = Manager.JobId(crab_num, "260915:task", "/proj")
    extra = {"site_history": [site] if site else []}
    if log:
        extra["log_file"] = log
    return job_id, {
        "status": Manager.FAILED,
        "code": code,
        "error": "Error while running CMSSW:",
        "extra": extra,
    }


def result(*jobs):
    return dict(jobs)


class ReadingThePayloadsError(unittest.TestCase):
    def test_the_last_exception_is_the_one_that_ended_the_job(self):
        """A job log carries harmless earlier exceptions -- the probe that imports FWCore
        before the release is set up always leaves an ImportError behind."""
        found = lc.payload_error(REAL_LOG)
        self.assertTrue(
            found.startswith("RuntimeError: 1 of 50 staged nano files"), found
        )

    def test_a_log_without_an_exception_says_so_rather_than_inventing_one(self):
        self.assertIsNone(lc.payload_error("all fine\nnothing to see\n"))

    def test_a_very_long_message_is_trimmed(self):
        found = lc.payload_error("RuntimeError: " + "x" * 5000)
        self.assertLess(len(found), 1000)
        self.assertTrue(found.endswith("..."))

    def test_an_exception_with_a_dotted_name_is_recognised(self):
        self.assertEqual(
            lc.payload_error("== CMSSW: law.target.RemoteFileError: gone"),
            "law.target.RemoteFileError: gone",
        )

    def test_flaf_bootstrap_noise_does_not_hide_the_payload_error(self):
        """The scram FileExistsError of the bundle relocation comes first in every FLAF
        CRAB job; the error that ended this one is the task's own, raised last."""
        for text in (FLAF_REAL_LOG, cmssw_prefixed(FLAF_REAL_LOG)):
            self.assertEqual(
                lc.payload_error(text), "RuntimeError: anaTupleProducer failed."
            )

    def test_a_flaf_error_is_read_on_its_first_line(self):
        """The continuation of a multi-line message is not an exception line; the reason
        is kept whole from the line that names the exception."""
        found = lc.payload_error(cmssw_prefixed(FLAF_LFN_LOG))
        self.assertTrue(
            found.startswith("RuntimeError: lfn_to_pfn: unable to resolve PFN"), found
        )
        self.assertTrue(found.endswith("Cannot authenticate."), found)

    def test_a_flaf_framework_exception_keeps_its_module_path(self):
        found = lc.payload_error(FLAF_REAL_LOG.split("During handling")[0])
        self.assertTrue(
            found.startswith("FLAF.RunKit.run_tools.PsCallError: Error while running"),
            found,
        )


class FetchingTheLogCannotHoldUpThePoll(unittest.TestCase):
    """`query()` runs in law's thread pool and `get_async_result_silent` waits on it without
    a timeout of its own, so every CRAB task of the run waits for whatever this does.
    """

    class Response:
        def __init__(self, status, chunks):
            self.status = status
            self._chunks = list(chunks)

        def read(self, n):
            return self._chunks.pop(0) if self._chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fetch(self, response, **kwargs):
        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "ssl.create_default_context"
        ) as context, mock.patch.dict(os.environ, {"X509_USER_PROXY": "/tmp/proxy"}):
            with mock.patch("urllib.request.urlopen", return_value=response) as opened:
                text = lc.fetch_job_stdout("http://x/job_out.1.0.txt", **kwargs)
        return text, opened, context.return_value

    def test_the_tail_is_asked_for_by_range(self):
        text, opened, _ = self.fetch(self.Response(206, [b"the tail\n"]))
        self.assertEqual(text, "the tail\n")
        request = opened.call_args.args[0]
        self.assertIn("bytes=-", request.headers.get("Range", ""))

    def test_the_run_proxy_is_the_client_certificate(self):
        """The schedd answers 401 without a client certificate; the proxy the submission
        already holds is that certificate, and the request must go out over it."""
        _, opened, context = self.fetch(self.Response(206, [b"tail\n"]))
        context.load_cert_chain.assert_called_once_with("/tmp/proxy", "/tmp/proxy")
        self.assertIs(opened.call_args.kwargs.get("context"), context)
        self.assertIsNotNone(opened.call_args.kwargs.get("timeout"))

    def test_a_server_that_ignores_the_range_is_not_followed_forever(self):
        """The runaway log a tail is for: without a bound this reads gigabytes into a poll."""
        endless = self.Response(200, [b"x" * 4096] * 100000)
        with self.assertRaises(RuntimeError) as caught:
            self.fetch(endless, max_bytes=4096, deadline=60.0)
        self.assertIn("still arriving", str(caught.exception))

    def test_a_whole_log_sent_without_a_range_is_cut_to_its_tail(self):
        """A server that ignores the range sends the log from the front; what is kept is
        its end, where the exception that ended the job is."""
        chunks = [
            b"OSError: early\n".ljust(4096, b"a"),
            b"b" * 4096,
            b"\nRuntimeError: the end\n".rjust(4096, b"c"),
        ]
        text, _, _ = self.fetch(self.Response(200, chunks), max_bytes=4096)
        self.assertEqual(text, chunks[-1].decode())
        self.assertEqual(lc.payload_error(text), "RuntimeError: the end")

    def test_a_ranged_response_that_trickles_is_cut_off_too(self):
        """The bytes of a 206 are already bounded -- the tail is all it sends -- but the
        time is not, and `timeout` bounds one socket read rather than a transfer that
        dribbles.

        The clock is injected rather than waited on: asking whether real time passed
        between two in-memory reads is a test that passes or fails by luck.
        """
        ticks = itertools.count(0.0, 100.0)
        with mock.patch.object(lc.time, "monotonic", side_effect=lambda: next(ticks)):
            with self.assertRaises(RuntimeError) as caught:
                self.fetch(self.Response(206, [b"x" * 4096] * 10), deadline=60.0)
        self.assertIn("still arriving", str(caught.exception))

    def test_a_log_that_fits_is_returned_whole(self):
        text, _, _ = self.fetch(self.Response(200, [b"short log\n"]), max_bytes=4096)
        self.assertEqual(text, "short log\n")

    def test_without_a_proxy_it_says_so_instead_of_trying(self):
        with mock.patch.dict(os.environ, {"X509_USER_PROXY": ""}):
            with mock.patch("urllib.request.urlopen") as opened:
                with self.assertRaises(RuntimeError) as caught:
                    lc.fetch_job_stdout("http://x/job_out.1.0.txt")
        self.assertIn("X509_USER_PROXY", str(caught.exception))
        opened.assert_not_called()

    def test_a_proxy_path_that_does_not_exist_is_no_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            gone = os.path.join(tmp, "x509up_gone")
            with mock.patch.dict(os.environ, {"X509_USER_PROXY": gone}):
                with mock.patch("urllib.request.urlopen") as opened:
                    with self.assertRaises(RuntimeError) as caught:
                        lc.fetch_job_stdout("http://x/job_out.1.0.txt")
        self.assertIn("X509_USER_PROXY", str(caught.exception))
        opened.assert_not_called()


class ReportingAFailedJob(unittest.TestCase):
    def report(self, m, res):
        with mock.patch.object(lc, "fetch_job_stdout", return_value=REAL_LOG) as fetch:
            with mock.patch("builtins.print") as printed:
                m.report_failures(res)
        return fetch, [c.args[0] for c in printed.call_args_list]

    def test_the_reason_the_site_and_the_code_are_printed(self):
        m = manager()
        _, lines = self.report(m, result(failed(66)))
        self.assertEqual(len(lines), 1)
        self.assertIn("RuntimeError: 1 of 50 staged nano files", lines[0])
        self.assertIn("T2_CH_CERN", lines[0])
        self.assertIn("exit code 5", lines[0])
        self.assertIn("66", lines[0])

    def test_a_poll_that_repeats_does_not_report_again(self):
        """The same failure comes back on every poll until the job is resubmitted."""
        m = manager()
        res = result(failed(66))
        fetch, lines = self.report(m, res)
        self.assertEqual(len(lines), 1)
        self.assertEqual(fetch.call_count, 1)
        fetch2, lines2 = self.report(m, res)
        self.assertEqual(lines2, [])
        self.assertEqual(fetch2.call_count, 0)

    def test_a_new_attempt_is_reported_again(self):
        """The log URL carries the attempt, so the next try's failure is news."""
        m = manager()
        self.report(m, result(failed(66, log="http://x/job_out.1.0.txt")))
        _, lines = self.report(m, result(failed(66, log="http://x/job_out.1.1.txt")))
        self.assertEqual(len(lines), 1)

    def test_law_bookkeeping_is_not_a_payload_failure(self):
        """A failure without a job-level code is a kill, a watchdog verdict or law's own
        resync -- there is no payload stdout to read, and `_harvest_site_stats` ignores it
        for the same reason.
        """
        m = manager()
        fetch, lines = self.report(m, result(failed(66, code=None)))
        self.assertEqual(lines, [])
        self.assertEqual(fetch.call_count, 0)

    def test_a_failure_with_no_log_url_is_skipped_rather_than_raising(self):
        """law records no `log_file` until it has parsed a scheduler id out of the status."""
        m = manager()
        fetch, lines = self.report(m, result(failed(66, log=None)))
        self.assertEqual(lines, [])
        self.assertEqual(fetch.call_count, 0)

    def test_an_entry_that_is_not_a_status_dict_is_skipped(self):
        """law's query data is not guaranteed to carry only dicts, and this runs in the
        poll."""
        m = manager()
        job_id = Manager.JobId(66, "260915:task", "/proj")
        with mock.patch.object(lc, "fetch_job_stdout", return_value=REAL_LOG):
            with mock.patch("builtins.print") as printed:
                m.report_failures({job_id: None, "x": "not a dict"})
        self.assertEqual(printed.call_args_list, [])

    def test_a_job_id_that_cannot_be_described_is_still_pointed_at(self):
        """A failed entry under a key that is not a crab JobId cannot be named, but its
        stdout URL is still worth a line -- and the poll must not pay for it."""
        m = manager()
        _, data = failed(66)
        with mock.patch.object(lc, "fetch_job_stdout", return_value=REAL_LOG):
            with mock.patch("builtins.print") as printed:
                m.report_failures({"not-a-job-id": data})
        lines = [c.args[0] for c in printed.call_args_list]
        self.assertEqual(len(lines), 1)
        self.assertIn("could not report a failed job", lines[0])
        self.assertIn("http://x/job_out.1.0.txt", lines[0])

    def test_a_finished_job_is_not_reported(self):
        m = manager()
        job_id, data = failed(66)
        data["status"] = Manager.FINISHED
        fetch, lines = self.report(m, {job_id: data})
        self.assertEqual(lines, [])
        self.assertEqual(fetch.call_count, 0)

    def test_a_wave_of_failures_is_capped_and_says_that_it_capped(self):
        """A production that fails by the hundred fails for a handful of reasons, and the
        poll must not wait on one HTTP fetch per job."""
        m = manager(max_failure_reports=2)
        jobs = [failed(n, log=f"http://x/job_out.{n}.0.txt") for n in range(1, 8)]
        fetch, lines = self.report(m, result(*jobs))
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(len(lines), 3)
        self.assertIn("5 more failed job(s)", lines[-1])
        self.assertIn("max_failure_reports=2", lines[-1])

    def test_the_default_cap_is_five(self):
        m = manager()
        jobs = [failed(n, log=f"http://x/job_out.{n}.0.txt") for n in range(1, 9)]
        fetch, lines = self.report(m, result(*jobs))
        self.assertEqual(fetch.call_count, 5)
        self.assertIn("3 more failed job(s)", lines[-1])

    def test_a_log_that_cannot_be_read_is_said_out_loud(self):
        """The one thing a diagnostic must not do is fail quietly."""
        m = manager()
        with mock.patch.object(
            lc, "fetch_job_stdout", side_effect=RuntimeError("401 unauthorized")
        ):
            with mock.patch("builtins.print") as printed:
                m.report_failures(result(failed(66)))
        line = printed.call_args_list[0].args[0]
        self.assertIn("could not read its stdout", line)
        self.assertIn("401 unauthorized", line)
        self.assertIn("http://x/job_out.1.0.txt", line)

    def test_a_log_without_an_exception_is_also_said_out_loud(self):
        m = manager()
        with mock.patch.object(lc, "fetch_job_stdout", return_value="nothing here"):
            with mock.patch("builtins.print") as printed:
                m.report_failures(result(failed(66)))
        self.assertIn("carries no exception", printed.call_args_list[0].args[0])

    def test_reporting_never_raises_into_the_poll(self):
        """It runs inside `query()`: an exception here would cost the whole poll -- every
        other task's status with it -- to print a message."""
        m = manager()
        with mock.patch.object(lc, "fetch_job_stdout", side_effect=KeyboardInterrupt):
            with mock.patch("builtins.print"):
                with self.assertRaises(KeyboardInterrupt):
                    m.report_failures(result(failed(66)))
        # anything short of a BaseException is swallowed into the message
        for boom in (OSError("down"), ValueError("weird"), Exception("x")):
            with mock.patch.object(lc, "fetch_job_stdout", side_effect=boom):
                with mock.patch("builtins.print"):
                    m.report_failures(result(failed(70, log=f"http://x/{boom}.txt")))

    def test_the_log_is_read_with_the_managers_byte_bound(self):
        m = manager(max_log_bytes=12345)
        fetch, _ = self.report(m, result(failed(66)))
        self.assertEqual(fetch.call_args.kwargs.get("max_bytes"), 12345)


#: the CRAB task of the failed FLAF CI job: AnaTupleFileTask, ci_crab_20260829_024357
TASK = (
    "260829_011605:kandroso_crab_AnaTupleFileTask_ci_crab_20260829_024357_Run3_2022EE_"
    "fe7c2005"
)
LOG_URL = (
    "https://cmsweb.cern.ch:8443/scheddmon/0199/kandroso/" + TASK + "/job_out.{}.{}.txt"
)


def crab_log(n_jobs):
    """The lines of a crab.log that law's `_parse_log_file` reads."""
    return (
        "config.JobType.disableAutomaticOutputCollection = True\n"
        f"config.Data.totalUnits = {n_jobs}\n"
        f"INFO 2026-08-29 01:16:05.166 UTC: \t Task name: {TASK}\n"
    )


def status_output(jobs):
    """A `crab status --json` response in the shape the CRAB client printed it for TASK."""
    return "\n".join(
        [
            f"CRAB project directory:\t\t/proj/crab_{TASK.split(':kandroso_crab_')[1]}",
            f"Task name:\t\t\t{TASK}",
            "Grid scheduler - Task Worker:\tcrab3@vocms0199.cern.ch - crab-prod-tw01",
            "Status on the CRAB server:\tSUBMITTED",
            "Dashboard monitoring URL:\thttps://monit-grafana.cern.ch/d/cmsTMDetail/x",
            "Status on the scheduler:\tSUBMITTED",
            json.dumps(jobs),
            "",
        ]
    )


def job_entry(state, retries=0, site="T2_US_Nebraska", error=None):
    entry = {
        "Retries": retries,
        "Restarts": 0,
        "SiteHistory": ["Unknown", site, site],
        "State": state,
        "RecordedSite": True,
    }
    if error is not None:
        entry["Error"] = error
    return entry


class TheJobDataLawParsesReachesTheReport(unittest.TestCase):
    """The real law 0.1.21 `CrabJobManager.query` and `parse_query_output`, fed the status
    response of the failed FLAF CI job -- only the `crab` subprocess and the stdout fetch
    are faked. This is what ties `report_failures` to the keys law actually fills:
    `code` from `Error[0]`, `extra["log_file"]` with the attempt from `Retries`, and the
    last entry of `site_history`.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.proj_dir = os.path.join(
            self._tmp.name, "crab_AnaTupleFileTask_ci_crab_20260829_024357_Run3_2022EE"
        )
        os.makedirs(self.proj_dir)
        self.m = manager(query_retry_delay=0.0)

    def tearDown(self):
        self._tmp.cleanup()

    def write_crab_log(self, n_jobs):
        with open(os.path.join(self.proj_dir, "crab.log"), "w") as f:
            f.write(crab_log(n_jobs))

    def poll(self, jobs, n_jobs=None, stdout=FLAF_REAL_LOG, fetch_error=None):
        self.write_crab_log(n_jobs or len(jobs))
        out = status_output(jobs)
        fetch_kwargs = (
            {"side_effect": fetch_error} if fetch_error else {"return_value": stdout}
        )
        with mock.patch.object(
            Manager, "cmssw_env", new_callable=mock.PropertyMock, return_value={}
        ), mock.patch(
            "law.contrib.cms.job.interruptable_popen", return_value=(0, out, None)
        ) as popen, mock.patch.object(
            lc, "fetch_job_stdout", **fetch_kwargs
        ) as fetch, mock.patch(
            "builtins.print"
        ) as printed:
            res = self.m.query(self.proj_dir)
        self.assertEqual(popen.call_count, 1, "the status must be read exactly once")
        return res, fetch, [c.args[0] for c in printed.call_args_list]

    def status_of(self, res):
        return {job_id.crab_num: data["status"] for job_id, data in res.items()}

    def standard_jobs(self, retries=0):
        return {
            "4": job_entry(
                "failed", retries, error=[5, "Error while running CMSSW:\n", {}]
            ),
            "1": job_entry("running", site="T1_DE_KIT"),
            "2": job_entry("finished", site="T1_DE_KIT", error=[0, "OK", {}]),
            "3": job_entry("idle", site="T1_DE_KIT"),
        }

    def test_the_failed_job_is_reported_from_the_url_law_built(self):
        res, fetch, lines = self.poll(self.standard_jobs())
        self.assertEqual(
            self.status_of(res),
            {
                4: Manager.FAILED,
                1: Manager.RUNNING,
                2: Manager.FINISHED,
                3: Manager.PENDING,
            },
        )
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0], LOG_URL.format(4, 0))
        self.assertEqual(len(lines), 1, lines)
        line = lines[0]
        self.assertIn("crab job 4 of", line)
        self.assertIn(os.path.basename(self.proj_dir), line)
        # the last site of the history is where the attempt ran
        self.assertIn("at T2_US_Nebraska", line)
        self.assertIn("exit code 5", line)
        self.assertIn("RuntimeError: anaTupleProducer failed.", line)

    def test_the_next_poll_of_the_same_failure_is_silent(self):
        self.poll(self.standard_jobs())
        _, fetch, lines = self.poll(self.standard_jobs())
        self.assertEqual(fetch.call_count, 0)
        self.assertEqual(lines, [])

    def test_the_retried_attempt_is_reported_from_its_own_log(self):
        self.poll(self.standard_jobs(retries=0))
        _, fetch, lines = self.poll(self.standard_jobs(retries=1))
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0], LOG_URL.format(4, 1))
        self.assertEqual(len(lines), 1)

    def test_a_job_missing_from_the_response_is_not_fetched(self):
        """law marks a job it finds no entry for failed, without a code and without a log
        URL: that is law's bookkeeping, and there is no stdout to read."""
        res, fetch, lines = self.poll(self.standard_jobs(), n_jobs=5)
        missing = [d for j, d in res.items() if j.crab_num == 5][0]
        self.assertEqual(missing["status"], Manager.FAILED)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0], LOG_URL.format(4, 0))
        self.assertEqual(len(lines), 1)

    def test_a_killed_or_held_job_is_not_fetched(self):
        """law maps `killing` and `held` to failed; without an `Error` entry there is no
        job-level code and no payload failure to explain."""
        jobs = {
            "1": job_entry("killing"),
            "2": job_entry("held"),
        }
        res, fetch, lines = self.poll(jobs)
        self.assertEqual(set(self.status_of(res).values()), {Manager.FAILED})
        self.assertEqual(fetch.call_count, 0)
        self.assertEqual(lines, [])

    def test_an_unreadable_log_costs_the_poll_nothing(self):
        res, fetch, lines = self.poll(
            self.standard_jobs(), fetch_error=RuntimeError("401 unauthorized")
        )
        self.assertEqual(
            self.status_of(res),
            {
                4: Manager.FAILED,
                1: Manager.RUNNING,
                2: Manager.FINISHED,
                3: Manager.PENDING,
            },
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("could not read its stdout (401 unauthorized)", lines[0])
        self.assertIn(LOG_URL.format(4, 0), lines[0])


if __name__ == "__main__":
    unittest.main()
