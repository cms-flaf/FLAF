#!/usr/bin/env python3
"""A CRAB job is finished when its products exist, not when CRAB says so.

CRAB parks a job in `transferring` between the payload exiting and the post-job classifying it,
and it parks a payload that exited non-zero there too. law maps that state to FINISHED whenever
transfers are skipped -- which FLAF pins, since its jobs stage their own products -- so a poll
landing inside that window reads a failed job as finished, writes it off with `dummy_job_id` and
never queries it again (DSProd, 2026-09-18: one poll booked 113 such jobs as `finished: 113` for
a production that had written nothing).

`crab_check_job_completeness()` closes that door, and the tests here hold it shut from every
side: a job whose products are missing is demoted and retried; a job whose products are there
is checked once for the whole run; and a job whose product was written after its directory was
listed is NOT demoted. That last one has a FLAF-specific edge: `exists()` answers an unknown
file from a listing marker that the path-cache server shares between processes, a CRAB worker
cannot reach that server, so the marker of a listing taken before the worker wrote its product
says "absent" in every process. Clearing the driver's own cache does not help -- the next
lookup falls through to the server -- which is why the check bumps the fresh-negatives epoch.
A driver that resumes a workflow meets the same marker for every product written while no
driver polled, before its first poll: the CRAB proxy's run() gathers the existing outputs again
on fresh listings (`AResumedRun`, driven through law's whole run()).

Nothing about completeness is replaced. law's real poll loop runs the real FLAF job manager's
`query()`, law's real `crab status` parsing with the query kwargs FLAF pins, the branch's own
`complete()`, and the real `GFALFileInterface` with its real local and remote caches. Only what
does not exist on a test runner is faked: the `crab` executable, `gfal-ls`, the path-cache
server, the analysis configuration, the credentials and the submission itself.
"""

import contextlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import uuid
from collections import OrderedDict
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

# FLAF.Common.Setup imports ROOT at module level; nothing here touches it, and the unit-test
# workflow has no ROOT.
try:
    import ROOT  # noqa: F401
except ImportError:
    sys.modules["ROOT"] = mock.MagicMock()

import law  # noqa: E402
import law.contrib.cms.job  # noqa: E402
import luigi.task_register  # noqa: E402
from law.job.dashboard import NoJobDashboard  # noqa: E402

from FLAF.run_tools import crab_watchdog  # noqa: E402
from FLAF.run_tools import law_customizations as lc  # noqa: E402
from FLAF.RunKit import law_gfal  # noqa: E402
from FLAF.RunKit.law_gfal import GFALFileInterface, PathCache  # noqa: E402
from FLAF.RunKit.law_wlcg import WLCGFileSystem  # noqa: E402

BASE = "davs://storage.example:1094/store/user/flaf"
CACHE_HOST = "path-cache.example"
CACHE_PORT = 12345
PERIOD = "Run3_2022"

_data_dir = None
_data_before = None


def setUpModule():
    # the job manager keeps its site record and law its job data under $ANALYSIS_DATA_PATH;
    # nothing written here may land in a real analysis area
    global _data_dir, _data_before
    _data_before = os.environ.get("ANALYSIS_DATA_PATH")
    _data_dir = tempfile.mkdtemp(prefix="flaf_completeness_test_")
    os.environ["ANALYSIS_DATA_PATH"] = _data_dir


def tearDownModule():
    if _data_before is None:
        os.environ.pop("ANALYSIS_DATA_PATH", None)
    else:
        os.environ["ANALYSIS_DATA_PATH"] = _data_before
    shutil.rmtree(_data_dir, ignore_errors=True)


class ProductTask(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    """Built like every FLAF production task: two branches writing into one directory."""

    bundle_flavours = ["core"]

    def create_branch_map(self):
        return {0: "a", 1: "b"}

    def output(self):
        return self.remote_target(
            self.version,
            self.__class__.__name__,
            self.period,
            f"product_{self.branch}.root",
        )

    def run(self):
        raise AssertionError("nothing in these tests runs a branch")


class ResumedProductTask(ProductTask):
    """Six branches over two directories, so that what a run lists is visible per directory."""

    def create_branch_map(self):
        return {branch: f"part_{branch % 2}" for branch in range(6)}

    def output(self):
        return self.remote_target(
            self.version,
            self.__class__.__name__,
            self.period,
            self.branch_data,
            f"product_{self.branch}.root",
        )


def merged_marker(product):
    """What a downstream merge leaves in place of a product it consumed (issue #229)."""
    return product.sibling(product.basename + ".merged", type="f")


class MergedProductTask(ResumedProductTask):
    """Complete by its own rule as well, as HistFromNtupleProducerTask is: a branch whose
    product a downstream merge removed and replaced by its marker is done."""

    def complete(self):
        if not self.is_branch():
            return super().complete()
        product = self.output()
        return product.exists() or merged_marker(product).exists()


class Entry:
    """The fields of grid_tools.FileInfo that law_gfal reads off a listing."""

    def __init__(self, name, is_dir=False):
        self.name = name
        self.size = 0
        self.is_dir = is_dir


class FakeStorage:
    """What the endpoint would answer to `gfal-ls`, keyed by directory URI.

    Only the listing call is faked. The cache, the "unknown file in a listed directory is
    absent" shortcut and the epoch check above it are the real ones, because they are what
    decides the verdict.
    """

    def __init__(self):
        self.dirs = {BASE: set()}
        self.listings = []

    def add(self, uri, is_dir=False):
        """Create a file (or directory) and every directory above it, up to BASE."""
        if is_dir:
            self.dirs.setdefault(uri, set())
        while uri != BASE:
            parent, _, name = uri.rpartition("/")
            self.dirs.setdefault(parent, set()).add(name)
            uri = parent

    def remove(self, uri):
        parent, _, name = uri.rpartition("/")
        self.dirs[parent].discard(name)

    def ls(self, uri, **kwargs):
        self.listings.append(uri)
        if uri not in self.dirs:
            return None  # gfal_ls_checked's "no such file or directory"
        return [
            Entry(name, is_dir=f"{uri}/{name}" in self.dirs)
            for name in sorted(self.dirs[uri])
        ]


class FakeCacheServer:
    """In-memory stand-in for pathCacheServer, shared by every RemotePathCache client."""

    def __init__(self):
        self.entries = {}

    def set_status(self, entries, *args, **kwargs):
        for path, exists in entries:
            self.entries[path] = exists

    def get_status(self, path, *args, **kwargs):
        return self.entries.get(path)

    def get_status_many(self, paths, *args, **kwargs):
        return {path: self.entries.get(path) for path in paths}


def crab_status_output(states):
    """What `crab status --json` prints for a task whose jobs are in `states`, a mapping of
    the CRAB job number to its CRAB state."""
    jobs = {str(crab_num): {"State": state} for crab_num, state in states.items()}
    return (
        "CRAB project directory:\t\t/fake/proj\n"
        "Status on the CRAB server:\tSUBMITTED\n"
        "Status on the scheduler:\tSUBMITTED\n"
        f"{json.dumps(jobs)}\n"
    )


class Harness:
    """The pieces every test needs: storage, cache server, file system and a task."""

    def __init__(self, case, workflow="crab", task_cls=ProductTask, **task_params):
        self.case = case
        self.storage = FakeStorage()
        self.server = FakeCacheServer()
        patchers = [
            mock.patch.multiple(
                law_gfal,
                gfal_ls_checked=self.storage.ls,
                set_remote_cache_status=self.server.set_status,
                get_remote_cache_status=self.server.get_status,
                get_remote_cache_status_many=self.server.get_status_many,
                # a runner has no grid proxy
                get_voms_proxy_info=lambda: {"path": None},
            ),
            # the epoch is process-wide state: never let one test's poll leak into another
            mock.patch.object(GFALFileInterface, "negatives_valid_after", 0.0),
            # FLAF reads the analysis configuration through Setup, and a FLAF checkout alone
            # has none; every task built while the harness lives (branches included) sees
            # this one
            mock.patch.object(lc.Setup, "getGlobal", side_effect=self._setup),
            # law runs `crab` in a CMSSW sandbox whose environment is built from cvmfs; the
            # `crab` call itself is faked wherever a test reaches it
            mock.patch.object(
                law.cms.CrabJobManager,
                "cmssw_env",
                property(lambda manager: {"PATH": os.environ.get("PATH", "")}),
            ),
            # FLAF moves the CRAB client's home under the temporary directory: keep it in
            # this test's area
            mock.patch.object(tempfile, "tempdir", _data_dir),
            # a query that fails is retried after this long; fail fast instead
            mock.patch.object(lc.FLAFCrabJobManager, "query_retry_delay", 0.0),
        ]
        for patcher in patchers:
            patcher.start()
            case.addCleanup(patcher.stop)

        self.fs = self.new_fs()
        self.setup = types.SimpleNamespace(
            global_params={"crab": {}},
            fs_dict={"default": self.fs},
            get_fs=lambda name, custom_paths=None: self.fs,
        )
        # a fresh version per harness: luigi caches task instances by their parameters
        self.task_cls = task_cls
        self.task_params = dict(
            version=f"v_{uuid.uuid4().hex[:8]}",
            period=PERIOD,
            workflow=workflow,
            poll_interval=0.0,
            **task_params,
        )
        self.task = task_cls(**self.task_params)
        self.product_dir = os.path.dirname(self.product_uri(0))

    def _setup(self, *args, **kwargs):
        return self.setup

    def new_fs(self):
        """Another process's view of the storage: its own local cache, the shared server."""
        return WLCGFileSystem(
            BASE,
            local_path_cache_validity_period=600,
            path_cache_host=CACHE_HOST,
            path_cache_port=CACHE_PORT,
        )

    def product_path(self, branch):
        return self.task.as_branch(branch).output().path

    def product_uri(self, branch):
        return self.fs.file_interface.uri(self.product_path(branch))

    def produce(self, branch):
        """Put a branch's product on the storage, as a CRAB job does: without telling the
        path-cache server, which it cannot reach."""
        self.storage.add(self.product_uri(branch))


class PollHarness(Harness):
    """Drives law's real poll loop over scripted batch-system responses.

    For CRAB, the FLAF job manager's own `query()` runs, and through it law's: the `crab`
    executable is the only fake on that path. For HTCondor, the job manager's `query()` (the
    `condor_q` call) is the fake. Both workflows' own poll callbacks run, with `kinit` and the
    watchdog's listing of the heartbeat directory faked (no flags written yet). The credential
    setup (a runner has no VOMS proxy or MyProxy credential), the submission of retries and
    luigi's scheduler messaging are doubles.
    """

    def __init__(
        self,
        case,
        responses,
        before_query=None,
        max_polls=None,
        workflow="crab",
        **task_params,
    ):
        super().__init__(case, workflow=workflow, **task_params)
        self.workflow = workflow
        self.script(responses, before_query=before_query, max_polls=max_polls)
        self._job_nums = {}

        self.proxy = self.task.workflow_proxy
        proxy_cls = {
            "crab": lc._FLAFCrabWorkflowProxy,
            "htcondor": lc._BundleAwareHTCondorWorkflowProxy,
        }[workflow]
        case.assertIsInstance(self.proxy, proxy_cls)
        # set by law's run(), which is not what is under test
        self.proxy.dashboard = self.task.create_job_dashboard() or NoJobDashboard()
        # a fresh submission, as a production is: law has a separate rule for the first poll
        # of a RESUMED one (`_submitted and i == 0`), where a finished job with no outputs is
        # retried as "initially missing task outputs" -- a guard that covers one iteration of
        # one case and is why the incident happened on a freshly submitted task
        self.proxy._submitted = False
        # what law found at the start of the run: nothing (law ORs accepted branches in)
        self.proxy._existing_branches = set()
        # law checks that it exists before it runs `crab status`
        self.proj_dir = os.path.join(_data_dir, f"crab_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.proj_dir)

    def script(self, responses, before_query=None, max_polls=None):
        """Arm the batch-system responses of one poll loop, and clear its records."""
        #: stop the loop after this many polls, the way law lets a poll callback stop it
        self.max_polls = max_polls
        #: per poll: job_num -> the CRAB state (crab) or law's status (htcondor)
        self.responses = list(responses)
        #: poll index -> what happened on the storage since the previous poll. It fires
        #: BEFORE that poll's status is read, which is the order the grid works in: a job
        #: runs and writes between two polls, and the next poll reports it finished
        self.before_query = dict(before_query or {})
        self.queries = 0
        self.iterations = 0
        self.crab_commands = []
        #: branch -> how often law asked whether it is complete
        self.complete_calls = {}
        #: retry generations law handed to submit(), in order (law also calls submit()
        #: without one, to fill free slots from the unsubmitted backlog)
        self.offered = []
        #: job_num -> (status, error) after each poll, as the poll callback saw it
        self.after_poll = []
        #: the fresh-negatives epoch after each poll
        self.epochs = []
        #: at the top of each iteration of the loop: the storage listings taken so far, and
        #: what the cache server knows
        self.listings_at_iteration = []
        self.server_at_iteration = []
        self.messages = []

    def job(self, job_num, branch):
        if self.workflow == "crab":
            job_id = self.proxy.job_manager.JobId(
                job_num, "fake_crab_task", self.proj_dir
            )
        else:
            job_id = f"{1000 + job_num}.0"
        self._job_nums[job_id] = job_num
        self.proxy.job_data.jobs[job_num] = self.proxy.job_data_cls.job_data(
            branches=[branch], job_id=job_id
        )

    def _next_iteration(self, *args, **kwargs):
        """Called at the top of every iteration of law's loop, outside its error handling.

        A loop that has run out of scripted polls is stopped here: law polls pending jobs
        for ever, and with no poll interval that would hang the test instead of failing it.
        """
        self.iterations += 1
        self.listings_at_iteration.append(list(self.storage.listings))
        self.server_at_iteration.append(dict(self.server.entries))
        if self.iterations > len(self.responses) + 3:
            raise AssertionError(
                f"the poll loop did not end after {self.iterations - 1} polls: "
                f"{self.after_poll[-1:]}"
            )

    def _next_response(self):
        action = self.before_query.get(self.queries)
        if action is not None:
            action()
        response = self.responses[min(self.queries, len(self.responses) - 1)]
        self.queries += 1
        return response

    def _crab(self, cmd, *args, **kwargs):
        self.crab_commands.append(cmd)
        return 0, crab_status_output(self._next_response()), ""

    def _condor_q(self, manager, job_id, **kwargs):
        response = self._next_response()

        def state(_id):
            status = response[self._job_nums[_id]]
            return manager.job_status_dict(job_id=_id, status=status, code=0)

        if isinstance(job_id, (list, tuple)):
            return {_id: state(_id) for _id in job_id}
        return state(job_id)

    def _poll_callback(self, task_self, poll_data):
        self.after_poll.append(
            {
                num: (data["status"], data["error"])
                for num, data in self.proxy.job_data.jobs.items()
            }
        )
        self.epochs.append(GFALFileInterface.negatives_valid_after)
        real = getattr(
            lc.CrabWorkflow if self.workflow == "crab" else lc.HTCondorWorkflow,
            f"{self.workflow}_poll_callback",
        )
        if real(task_self, poll_data) is False:
            return False
        return self.max_polls is None or len(self.after_poll) < self.max_polls

    def _submit(self, proxy_self, retry_jobs=None):
        if retry_jobs:
            self.offered.append(dict(retry_jobs))
        return OrderedDict()

    def _patches(self):
        """The doubles of every poll loop: the batch system and the task's environment."""
        real_complete = ProductTask.complete

        def spy(task_self):
            branch = getattr(task_self, "branch", None)
            if branch is not None and branch >= 0:
                self.complete_calls[branch] = self.complete_calls.get(branch, 0) + 1
            return real_complete(task_self)

        proxy_cls = type(self.proxy)
        patches = [
            mock.patch.object(
                ProductTask,
                f"{self.workflow}_poll_callback",
                autospec=True,
                side_effect=self._poll_callback,
            ),
            mock.patch.object(ProductTask, "complete", autospec=True, side_effect=spy),
            # luigi attaches `scheduler_messages` only while a worker runs the task, and the
            # loop reads it once per iteration
            mock.patch.object(
                ProductTask,
                "_handle_scheduler_messages",
                side_effect=self._next_iteration,
            ),
            mock.patch.object(
                ProductTask, "publish_message", side_effect=self.messages.append
            ),
            mock.patch.object(lc, "update_kinit"),
            # the watchdog's one listing per interval: no job has written a flag yet
            mock.patch.object(crab_watchdog, "gfal_ls_safe", return_value=None),
        ]
        if self.workflow == "crab":
            patches += [
                mock.patch.object(
                    law.contrib.cms.job, "interruptable_popen", side_effect=self._crab
                ),
                mock.patch.object(proxy_cls, "setup_job_manager", return_value={}),
            ]
        else:
            patches.append(
                mock.patch.object(
                    type(self.proxy.job_manager),
                    "query",
                    autospec=True,
                    side_effect=self._condor_q,
                )
            )
        return patches

    def run(self):
        patches = self._patches() + [
            mock.patch.object(
                type(self.proxy), "submit", autospec=True, side_effect=self._submit
            )
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            self.proxy.poll()

    def status(self, job_num):
        return self.proxy.job_data.jobs[job_num]["status"]

    def error(self, job_num):
        return self.proxy.job_data.jobs[job_num]["error"]


class RunHarness(PollHarness):
    """Drives whole CRAB driver runs through the workflow's run(), as luigi's worker does.

    Around PollHarness's poll loop everything is real here as well: the FLAF CRAB proxy's
    run(), law's `_run_impl()` (the submission file, the branches whose outputs exist, what to
    submit) and law's submit() behind FLAF's (the mass-lost-outputs brake, the job-source
    probe, the wave gate). Only `_submit_group`, which writes the CRAB job file and runs `crab
    submit`, is a double.

    Each driver is a process of its own: a new task instance with its own local caches and a
    fresh-negatives epoch of zero, sharing the storage, the cache server and the submission
    file with the drivers before it.
    """

    def __init__(self, case, task_cls=ResumedProductTask, **task_params):
        super().__init__(case, responses=[], task_cls=task_cls, **task_params)
        self.new_driver()

    def new_driver(self):
        # luigi hands out one task instance per parameter set and process
        luigi.task_register.Register.clear_instance_cache()
        GFALFileInterface.negatives_valid_after = 0.0
        self.fs = self.new_fs()
        self.task = self.task_cls(**self.task_params)
        self.proxy = self.task.workflow_proxy
        self.case.assertIsInstance(self.proxy, lc._FLAFCrabWorkflowProxy)
        #: the jobs this driver handed to `crab submit`, per submission round
        self.submitted = []

    def product_dirs(self):
        return sorted({os.path.dirname(self.product_uri(b)) for b in range(6)})

    def _submit_group(self, proxy_self, submit_jobs, **kwargs):
        self.submitted.append(dict(submit_jobs))
        # one CRAB task holds every job, numbered as law numbers them
        manager = proxy_self.job_manager
        job_ids = [
            manager.JobId(job_num, "fake_crab_task", self.proj_dir)
            for job_num in submit_jobs
        ]
        return job_ids, {
            job_num: {"job": None, "config": None, "log": None}
            for job_num in submit_jobs
        }

    def drive(self, responses, before_query=None, max_polls=None):
        """Schedule the workflow and run it."""
        self.script(responses, before_query=before_query, max_polls=max_polls)
        # what luigi asks while it schedules the workflow: law gathers the branches whose
        # outputs exist, and its per-job skip verdicts, right here (`process_resources`)
        self.task.complete()
        self.task.process_resources()
        self.storage.listings.clear()
        patches = self._patches() + [
            mock.patch.object(
                lc._FLAFCrabWorkflowProxy,
                "_submit_group",
                autospec=True,
                side_effect=self._submit_group,
            ),
            # the job-source probe in front of every submission round reads this tree
            mock.patch.dict(os.environ, {"FLAF_PATH": flaf_repo}),
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            self.task.run()


class TheProductsDecide(unittest.TestCase):
    def test_a_finished_job_with_no_products_is_refused_and_retried(self):
        """The 2026-09-18 incident: without the check, job 1 is booked as finished and
        written off. CRAB reports both jobs `transferring`; only job 2's branch produced
        anything. Job 1's retry then writes its product and is accepted."""
        h = PollHarness(
            self,
            responses=[
                {1: "transferring", 2: "transferring"},
                {1: "transferring", 2: "transferring"},
            ],
            before_query={1: lambda: h.produce(0)},
            acceptance=1.0,
            tolerance=0.0,
            retries=1,
        )
        h.job(1, 0)
        h.job(2, 1)
        h.produce(1)
        h.run()

        manager = h.proxy.job_manager
        first = h.after_poll[0]
        self.assertEqual(
            first[1][0],
            manager.RETRY,
            f"job 1 was not demoted in the first poll: {first}",
        )
        self.assertIn("missing outputs", first[1][1])
        self.assertEqual(first[2][0], manager.FINISHED)
        # the demoted job is handed back for resubmission, the accepted one is not
        self.assertEqual(h.offered[0], {1: [0]})
        self.assertEqual(h.proxy.job_data.attempts.get(1), 1)
        self.assertNotIn(2, h.proxy.job_data.attempts)
        # the accepted job is written off by job id; the demoted one kept it for the retry
        # and is accepted only once its product exists
        self.assertEqual(h.status(1), manager.FINISHED)
        self.assertEqual(h.status(2), manager.FINISHED)
        self.assertEqual(h.queries, 2)

    def test_a_poll_lists_a_directory_once_for_all_its_jobs(self):
        """The cost bound of the fresh negatives: within one poll, the listing taken for one
        job answers every other job in that directory, the absent ones included."""
        h = PollHarness(
            self,
            responses=[{1: "transferring", 2: "transferring"}],
            acceptance=0.5,
            tolerance=1.0,
            retries=0,
        )
        h.job(1, 0)
        h.job(2, 1)
        h.produce(0)  # job 2's branch wrote nothing
        h.run()

        manager = h.proxy.job_manager
        self.assertEqual(h.status(1), manager.FINISHED)
        self.assertEqual(h.status(2), manager.FAILED)
        self.assertIn("missing outputs", h.error(2))
        self.assertEqual(h.storage.listings, [h.product_dir])

    def test_a_record_written_since_the_last_listing_is_still_seen(self):
        """The mirror failure, and the reason the check bumps the epoch every poll.

        Both branches share a directory. Accepting job 1 in the first poll caches that
        directory's listing; job 2's product lands afterwards. Answering job 2 from that
        listing would demote a job that did the work.
        """
        h = PollHarness(
            self,
            responses=[
                {1: "transferring", 2: "running"},
                {1: "transferring", 2: "transferring"},
            ],
            before_query={1: lambda: h.produce(1)},
            acceptance=1.0,
            tolerance=1.0,
        )
        h.job(1, 0)
        h.job(2, 1)
        h.produce(0)
        h.run()

        manager = h.proxy.job_manager
        self.assertEqual(
            h.status(2),
            manager.FINISHED,
            f"a job that produced its output was demoted: {h.error(2)}",
        )
        self.assertEqual(h.status(1), manager.FINISHED)
        self.assertEqual(h.offered, [], "nothing should have been retried")
        # the price of seeing it: one listing per poll per directory, not one per job
        self.assertEqual(
            h.storage.listings,
            [h.product_dir, h.product_dir],
            "the check is one listing per poll per directory",
        )

    def test_a_finished_job_is_checked_once_for_the_whole_run(self):
        """What the check must NOT become: a storage round trip per job per poll.

        Job 1 finishes in the first poll and job 2 only in the third. law keeps accepted
        jobs in `finished_jobs` and skips them at the top of every later iteration, so
        branch 0's completeness is asked for exactly once.
        """
        h = PollHarness(
            self,
            responses=[
                {1: "transferring", 2: "running"},
                {1: "transferring", 2: "running"},
                {1: "transferring", 2: "transferring"},
            ],
            acceptance=1.0,
            tolerance=1.0,
        )
        h.job(1, 0)
        h.job(2, 1)
        h.produce(0)
        h.produce(1)
        h.run()

        manager = h.proxy.job_manager
        self.assertEqual(h.queries, 3, "the loop must really have polled three times")
        for cmd in h.crab_commands:
            self.assertIn("crab status --dir", cmd)
        self.assertEqual(h.status(1), manager.FINISHED)
        self.assertEqual(h.status(2), manager.FINISHED)
        self.assertEqual(
            h.complete_calls.get(0),
            1,
            f"branch 0 was re-checked on every poll: {h.complete_calls}",
        )
        self.assertEqual(h.complete_calls.get(1), 1)
        # once accepted, a job is no longer queried at all
        self.assertEqual(
            h.proxy.job_data.jobs[1]["job_id"], h.proxy.job_data.dummy_job_id
        )


class TheSharedListingTrap(unittest.TestCase):
    """FLAF-specific: a listing published to the path-cache server before the CRAB worker
    wrote its product says "absent" for it in every process, and the worker never corrects
    it. The epoch bump in `crab_check_job_completeness` is what gets past it."""

    def stale_listing_scenario(self, lister):
        """A listing of the product directory is on the server; then the worker writes."""
        h = PollHarness(
            self,
            responses=[{1: "transferring"}],
            max_polls=1,
            acceptance=1.0,
            tolerance=1.0,
            retries=1,
        )
        h.job(1, 0)
        h.storage.add(h.product_dir, is_dir=True)
        # the submit-time status check lists the directory and publishes the listing
        fs = h.fs if lister == "driver" else h.new_fs()
        self.assertFalse(fs.exists(h.product_path(0)))
        marker = law_gfal.listing_marker(h.product_dir)
        self.assertIs(h.server.entries.get(marker), True)
        # the CRAB worker writes its product and cannot tell the server
        h.produce(0)
        self.assertIsNone(h.server.entries.get(h.product_uri(0)))
        h.storage.listings.clear()
        return h

    def test_the_job_is_accepted_and_the_file_republished(self):
        for lister in ("another process", "driver"):
            with self.subTest(listed_by=lister):
                h = self.stale_listing_scenario(lister)
                h.run()

                self.assertEqual(
                    h.status(1),
                    h.proxy.job_manager.FINISHED,
                    f"a job whose product exists was demoted: {h.error(1)}",
                )
                self.assertEqual(h.offered, [])
                self.assertEqual(h.storage.listings, [h.product_dir])
                # the fresh listing republished the directory: every other process now
                # finds the product without listing again
                self.assertIs(h.server.entries.get(h.product_uri(0)), True)
                n_listed = len(h.storage.listings)
                other = h.new_fs()
                self.assertTrue(other.exists(h.product_path(0)))
                self.assertEqual(len(h.storage.listings), n_listed)

    def test_clearing_only_the_local_cache_demotes_it(self):
        """The trap the epoch exists for: a check that flushes the driver's in-process cache
        instead still reads the server's stale marker, and demotes a job that did the work.
        """

        for lister in ("another process", "driver"):
            with self.subTest(listed_by=lister):
                h = self.stale_listing_scenario(lister)
                path_cache = h.fs.file_interface.path_cache

                def clear_local_cache():
                    path_cache.local_cache = PathCache(600)

                with mock.patch.object(
                    lc, "require_fresh_negatives", side_effect=clear_local_cache
                ):
                    h.run()

                self.assertEqual(h.status(1), h.proxy.job_manager.RETRY)
                self.assertIn("missing outputs", h.error(1))
                self.assertEqual(
                    h.storage.listings, [], "the stale server marker answered"
                )


class AResumedRun(unittest.TestCase):
    """A driver that picks a CRAB workflow up from its submission file.

    Jobs go on finishing while no driver polls -- the driver was restarted, or the workflow
    waited for its turn -- and a CRAB worker cannot tell the path-cache server what it wrote,
    so the listing the previous driver published still answers "absent" for those products.
    law gathers the existing outputs while luigi schedules the workflow, and on the first
    poll of a resumed run it sends a job reported finished whose outputs it did not gather
    back to the grid ("initially missing task outputs"). The FLAF proxy's run() gathers them
    again first, with every "absent" resting on a listing taken from there on.
    """

    finished_while_away = (2, 3, 4)

    def resumable(self, task_cls=ResumedProductTask):
        return RunHarness(
            self, task_cls=task_cls, acceptance=1.0, tolerance=0.0, retries=1
        )

    def old_driver_then_away(self):
        """The old driver submits and polls once; then three jobs finish while none polls."""
        h = self.resumable()
        # produced by an earlier submission: the old driver books jobs 1 and 2 as finished
        # without submitting them, and lists both directories doing so
        h.produce(0)
        h.produce(1)
        h.drive([{job: "running" for job in (3, 4, 5, 6)}], max_polls=1)

        self.assertEqual(h.submitted, [{3: [2], 4: [3], 5: [4], 6: [5]}])
        finished, running = h.proxy.job_manager.FINISHED, h.proxy.job_manager.RUNNING
        self.assertEqual(
            {num: status for num, (status, _) in h.after_poll[-1].items()},
            {1: finished, 2: finished, 3: running, 4: running, 5: running, 6: running},
        )
        for directory in h.product_dirs():
            marker = law_gfal.listing_marker(directory)
            self.assertIs(h.server.entries.get(marker), True)
        for branch in self.finished_while_away:
            h.produce(branch)
            self.assertIsNone(h.server.entries.get(h.product_uri(branch)))
        h.new_driver()
        return h

    def resume(self, h):
        """The new driver: CRAB reports jobs 3-5 finished, and job 6 finishes a poll later."""
        h.drive(
            [
                {3: "finished", 4: "finished", 5: "finished", 6: "running"},
                {3: "finished", 4: "finished", 5: "finished", 6: "finished"},
            ],
            before_query={1: lambda: h.produce(5)},
        )

    def test_jobs_that_finished_meanwhile_are_accepted(self):
        h = self.old_driver_then_away()
        self.resume(h)

        manager = h.proxy.job_manager
        self.assertEqual(h.queries, 2)
        for poll in h.after_poll:
            for job_num, (status, error) in poll.items():
                self.assertNotIn(
                    status,
                    (manager.RETRY, manager.FAILED),
                    f"job {job_num} was sent back: {error}",
                )
        self.assertEqual(
            {num: status for num, (status, _) in h.after_poll[0].items()},
            {**{job: manager.FINISHED for job in range(1, 6)}, 6: manager.RUNNING},
        )
        for job_num in range(1, 7):
            self.assertEqual(h.status(job_num), manager.FINISHED)
        self.assertEqual(h.submitted, [], "a finished job was submitted again")
        self.assertEqual(dict(h.proxy.job_data.attempts), {})

    def test_the_resync_lists_each_directory_once(self):
        """Six products in two directories, three of them unknown to the cache server and
        one still missing: two listings, before the first poll."""
        h = self.old_driver_then_away()
        self.resume(h)

        self.assertEqual(sorted(h.listings_at_iteration[0]), h.product_dirs())

    def test_the_resync_republishes_what_the_workers_wrote(self):
        h = self.old_driver_then_away()
        self.resume(h)

        # before the first poll, not only once something later lists the directory again
        for branch in self.finished_while_away:
            uri = h.product_uri(branch)
            self.assertIs(h.server_at_iteration[0].get(uri), True)
        # another process now finds every product without listing anything
        n_listings = len(h.storage.listings)
        other = h.new_fs()
        for branch in range(6):
            self.assertTrue(other.exists(h.product_path(branch)))
        self.assertEqual(len(h.storage.listings), n_listings)

    def test_without_the_resync_they_go_back_to_the_grid(self):
        """The proxy's run() with its resync, or a part of it, left out. Each of the three
        steps is needed on its own: the epoch, so that the stale "absent" is not believed;
        the existing branches and the skip verdicts, both gathered while luigi scheduled the
        workflow, so that law does not reuse them."""
        law_run = lc._FLAFCrabWorkflowProxyBase.run
        steps = ("fresh negatives", "existing branches", "skip verdicts")

        def run_without(dropped):
            def run(proxy):
                if "fresh negatives" not in dropped:
                    lc.require_fresh_negatives()
                if "existing branches" not in dropped:
                    proxy._existing_branches = None
                if "skip verdicts" not in dropped:
                    proxy._skip_jobs.clear()
                return law_run(proxy)

            return run

        for dropped in [steps] + [(step,) for step in steps]:
            with self.subTest(dropped=dropped):
                h = self.old_driver_then_away()
                with mock.patch.object(
                    lc._FLAFCrabWorkflowProxy,
                    "run",
                    autospec=True,
                    side_effect=run_without(dropped),
                ):
                    self.resume(h)

                manager = h.proxy.job_manager
                for job_num in (3, 4, 5):
                    self.assertEqual(
                        h.after_poll[0][job_num],
                        (manager.RETRY, "initially missing task outputs"),
                    )

    @staticmethod
    def merge(h, branch):
        """A downstream merge consumes a product and leaves its marker. It runs where the
        cache server is reachable, so it publishes both, as GFALFileInterface's remove() and
        filecopy() do."""
        product_uri = h.product_uri(branch)
        marker = merged_marker(h.task.as_branch(branch).output())
        marker_uri = h.fs.file_interface.uri(marker.path)
        h.storage.remove(product_uri)
        h.storage.add(marker_uri)
        h.server.set_status([(product_uri, False), (marker_uri, True)])

    # The brake recounts candidates by the branch task's own complete(); one found complete
    # is marked skippable for law, whose cached verdict (from the output collection) would
    # otherwise send it back to the grid.
    def test_branches_done_by_their_own_rule_are_not_sent_back(self):
        """Products consumed by a downstream merge and replaced by its markers, as
        HistMergerTask does with `remove_merged_inputs`: the branches are complete, the
        brake's fresh look says so, and the jobs must not go back to the grid."""
        h = self.resumable(MergedProductTask)
        for directory in h.product_dirs():
            h.storage.add(directory, is_dir=True)
        h.drive(
            [{**{job: "finished" for job in range(1, 6)}, 6: "running"}],
            before_query={0: lambda: [h.produce(branch) for branch in range(5)]},
            max_polls=1,
        )
        for job_num in range(1, 6):
            self.assertEqual(h.status(job_num), h.proxy.job_manager.FINISHED)
        for branch in range(5):
            self.merge(h, branch)
        h.new_driver()

        # a resubmitted job is reported running in the second poll
        h.drive(
            [
                {6: "running"},
                {**{job: "running" for job in range(1, 6)}, 6: "finished"},
            ],
            before_query={1: lambda: h.produce(5)},
            max_polls=2,
        )
        self.assertEqual(h.submitted, [], "branches that are done were resubmitted")


class TheSwitchItself(unittest.TestCase):
    def test_crab_workflows_check_completeness(self):
        h = Harness(self, workflow="crab")
        proxy = h.task.workflow_proxy
        self.assertIsInstance(proxy, lc._FLAFCrabWorkflowProxy)
        self.assertEqual(GFALFileInterface.negatives_valid_after, 0.0)
        # law's own lookup of the hook, as its poll loop does it
        self.assertTrue(
            proxy._get_task_attribute("check_job_completeness")(),
            "law then accepts CRAB's FINISHED without looking at fs_default",
        )
        first_epoch = GFALFileInterface.negatives_valid_after
        self.assertGreater(first_epoch, 0.0)
        # every poll takes a new epoch: a listing from the previous poll does not count
        self.assertTrue(h.task.crab_check_job_completeness())
        self.assertGreater(GFALFileInterface.negatives_valid_after, first_epoch)

    def test_htcondor_workflows_are_unaffected(self):
        """HTCondor jobs publish what they write to the cache server, so their workflows
        keep law's default: no completeness check and no fresh-negatives epoch."""
        h = Harness(self, workflow="htcondor")
        proxy = h.task.workflow_proxy
        self.assertIsInstance(proxy, lc._BundleAwareHTCondorWorkflowProxy)
        self.assertFalse(proxy._get_task_attribute("check_job_completeness")())
        self.assertFalse(h.task.htcondor_check_job_completeness())
        self.assertEqual(GFALFileInterface.negatives_valid_after, 0.0)

    def test_an_htcondor_poll_takes_the_batch_system_at_its_word(self):
        """The same loop over HTCondor: a job reported finished is accepted as it stands,
        without a storage round trip, and no poll moves the epoch."""
        h = PollHarness(
            self,
            responses=[
                {
                    1: law.job.base.BaseJobManager.FINISHED,
                    2: law.job.base.BaseJobManager.RUNNING,
                },
                {2: law.job.base.BaseJobManager.FINISHED},
            ],
            workflow="htcondor",
            acceptance=1.0,
            tolerance=1.0,
        )
        h.job(1, 0)
        h.job(2, 1)
        h.run()

        manager = h.proxy.job_manager
        self.assertEqual(h.queries, 2)
        self.assertEqual(h.status(1), manager.FINISHED)
        self.assertEqual(h.status(2), manager.FINISHED)
        self.assertEqual(h.complete_calls, {}, "an HTCondor job was checked on storage")
        self.assertEqual(h.storage.listings, [])
        self.assertEqual(h.epochs, [0.0])
        self.assertEqual(GFALFileInterface.negatives_valid_after, 0.0)


class TestSkipTransfersIsPinned(unittest.TestCase):
    """A job whose stageout is disabled must not be polled until the workflow gives up on it.

    CRAB reports a job that finished its payload as `transferring`/`transferred` until it has
    staged its outputs out. FLAF disables CRAB stageout -- it writes every product itself --
    so those states are the end of the job. law decides that per poll from
    `config.JobType.disableAutomaticOutputCollection` in the project's `crab.log`: a log that
    is missing makes the query raise before `crab` even runs, and a log without that line
    reads as False, at which point those jobs are polled as running for ever.
    `crab_job_kwargs_query` pins it instead.
    """

    def test_law_passes_it_to_every_status_query(self):
        h = Harness(self, workflow="crab")
        # law's own assembly of the query kwargs, from the task attribute
        self.assertEqual(
            h.task.workflow_proxy._get_job_kwargs("query"), {"skip_transfers": True}
        )

    def test_it_is_what_makes_a_transferring_job_finished(self):
        manager = lc.FLAFCrabJobManager
        for status in ("transferring", "transferred"):
            self.assertEqual(
                manager.map_status(status, skip_transfers=True), manager.FINISHED
            )
            # the reading a missing or rewritten crab.log produces
            self.assertEqual(manager.map_status(status), manager.RUNNING)

    def query_without_crab_log(self, query, **query_kwargs):
        """Run `query` on the real job manager for a project whose crab.log is missing.

        Only the `crab` executable is faked; it reports the job `transferring`.
        """
        h = Harness(self, workflow="crab")
        manager = h.task.workflow_proxy.job_manager
        self.assertIsInstance(manager, lc.FLAFCrabJobManager)
        crab = mock.Mock(return_value=(0, crab_status_output({1: "transferring"}), ""))
        with tempfile.TemporaryDirectory() as proj_dir, mock.patch.object(
            law.contrib.cms.job, "interruptable_popen", crab
        ):
            self.assertFalse(os.path.exists(os.path.join(proj_dir, "crab.log")))
            job_ids = [manager.JobId(1, "fake_crab_task", proj_dir)]
            try:
                result = query(manager, proj_dir, job_ids, **query_kwargs)
            except Exception as exc:
                result = exc
        return manager, job_ids, crab, result

    def test_a_missing_crab_log_no_longer_stops_the_query(self):
        h = Harness(self, workflow="crab")
        kwargs = h.task.workflow_proxy._get_job_kwargs("query")
        manager, job_ids, crab, result = self.query_without_crab_log(
            lc.FLAFCrabJobManager.query, **kwargs
        )
        self.assertNotIsInstance(result, Exception, f"the query raised: {result!r}")
        self.assertEqual(crab.call_count, 1, "crab status must have run")
        self.assertIn("crab status", crab.call_args[0][0])
        self.assertEqual(result[job_ids[0]]["status"], manager.FINISHED)

    def test_without_the_pin_crab_never_runs(self):
        """What the pin replaces, on the real classes: law's own query raises before it runs
        `crab`, and FLAF's retries can only turn that into jobs that stay pending."""
        _, _, crab, result = self.query_without_crab_log(law.cms.CrabJobManager.query)
        self.assertIsInstance(result, AttributeError)
        self.assertEqual(crab.call_count, 0)

        manager, job_ids, crab, result = self.query_without_crab_log(
            lc.FLAFCrabJobManager.query
        )
        self.assertNotIsInstance(result, Exception, f"the query raised: {result!r}")
        self.assertEqual(crab.call_count, 0)
        self.assertEqual(result[job_ids[0]]["status"], manager.PENDING)


class TestWhyItCannotBeLeftToTheLog(unittest.TestCase):
    """law's fallback, exercised on the real class, on the two logs seen in a work area."""

    def test_a_project_without_a_log_says_nothing_at_all(self):
        self.assertIsNone(
            lc.FLAFCrabJobManager._parse_log_file("/does/not/exist/crab.log")
        )

    def test_a_log_that_does_not_mention_the_setting_says_nothing_either(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "crab.log")
            with open(path, "w") as f:
                f.write("config.General.requestName = 'ProductTask_v1_Run3_2022_abc'\n")
            log_data = lc.FLAFCrabJobManager._parse_log_file(path) or {}
        self.assertIsNone(log_data.get("disable_output_collection"))


if __name__ == "__main__":
    unittest.main()
