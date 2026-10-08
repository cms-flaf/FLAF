#!/usr/bin/env python3
"""What every FLAF submission round checks before law is handed control.

Both remote proxies (`_BundleAwareHTCondorWorkflowProxy`, `_FLAFCrabWorkflowProxy`) share
`SubmissionGuards`, ported from the DSProd CRAB production (cms-flaf/DSProd):

* law resolves the files it ships with every job through `law.util.rel_path`, which treats a
  module file as a directory as soon as one stat of the software tree fails, and the job file
  is built inside law's submit(), where nothing catches the error: one blink of the storage
  the tree lives on ended a 16000-branch production twice. FLAF rebinds `rel_path`, probes the
  job sources before a round, skips the round (parking the offered retries) while they cannot
  be read, and refuses to build a job file against a tree that went away mid-round.
* A resumed workflow whose jobs come back in large numbers for missing outputs (storage
  unreachable during the check, or outputs removed after use) would resubmit most of a
  production (DSProd: 8300 jobs); the run stops instead -- counting only the jobs whose
  branches are still incomplete on a fresh look -- and leaves the submission file as it found
  it, so that the next run judges the same jobs again.
* A batch job must not rebuild an upstream product inline in a slot sized for another task.

Where law is involved, law's real code runs: the proxies are built with their real
constructors over a stand-in workflow task, law's `run()` / `poll()` / `submit()` drive them,
and only the batch system (the job manager's `query`, the proxy's `_submit_group`) is scripted.
"""

import ast
import contextlib
import errno
import importlib.util
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

import law  # noqa: E402
import law.util  # noqa: E402
import luigi  # noqa: E402
import luigi.cmdline_parser  # noqa: E402
from law.job.base import BaseJobManager  # noqa: E402
from law.job.dashboard import NoJobDashboard  # noqa: E402
from law.workflow.remote import BaseRemoteWorkflowProxy, JobData  # noqa: E402

# The unit-test runner has no ROOT, and law_customizations reaches FLAF.Common.Utilities,
# which imports it at module level without using it on import. A placeholder stands in for
# that import only and is removed again, so that every other test module still sees ROOT as
# absent; one already registered by another module is left alone (find_spec raises on it).
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
from FLAF.RunKit.law_gfal import GFALFileInterface  # noqa: E402

# loaded by law_customizations (law.contrib.load("htcondor"), law.contrib.load("cms"))
import law.contrib.cms.job as cms_job  # noqa: E402
import law.contrib.htcondor.workflow as htcondor_workflow  # noqa: E402

#: the batch system's job states, as law's job managers name them
RUNNING = BaseJobManager.RUNNING
FINISHED = BaseJobManager.FINISHED
FAILED = BaseJobManager.FAILED
RETRY = BaseJobManager.RETRY

#: what `job_source_error` answers for a path that does not exist (POSIX ENOENT)
ENOENT = f"[Errno {errno.ENOENT}]"


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/bin/sh\n")


def make_flaf_tree(root):
    """FLAF's own job sources under `root`, so that FLAF_PATH=root is a readable tree."""
    for parts in lc._FLAF_JOB_SOURCES:
        _touch(os.path.join(root, *parts))


class FakeClock:
    """Stands in for the `time` module as seen by law_customizations only.

    The probe's retry delay and the skip budget are read from it, so neither a test nor law's
    own poll loop waits, and the budget can be moved by minutes. Everything else is the real
    `time` module: patching `time.sleep` globally would also turn the idle wait of
    multiprocessing's ThreadPool (used by law's query_batch) into a busy loop.
    """

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)

    def monotonic(self):
        return self.now

    def advance(self, minutes):
        self.now += minutes * 60

    def __getattr__(self, name):
        return getattr(time, name)


class ScriptedJobManager(BaseJobManager):
    """law's job-manager base with the batch system replaced by a script of job states.

    `script[job_id]` lists the states the batch system answers with, one per query; the last
    one repeats. An id not in the script (a job submitted during the test) answers FINISHED.
    `on_query(job_id, status)`, when set, runs before each answer -- where a job writes its
    outputs before it is seen in the state it reports.
    """

    # as for law's HTCondor and CRAB managers: law submits through the proxy's _submit_group
    job_grouping_submit = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.script = {}
        self.on_query = None

    def query(self, job_id, **kwargs):
        states = self.script.get(job_id) or [FINISHED]
        status = states.pop(0) if len(states) > 1 else states[0]
        if self.on_query is not None:
            self.on_query(job_id, status)
        code = {FINISHED: 0, FAILED: 1}.get(status)
        error = "payload exited with 1" if status == FAILED else None
        return self.job_status_dict(
            job_id=job_id, status=status, code=code, error=error
        )

    def submit(self, *args, **kwargs):
        raise AssertionError("jobs reach the batch system through _submit_group only")

    def cancel(self, *args, **kwargs):
        raise AssertionError("nothing is cancelled here")

    def cleanup(self, *args, **kwargs):
        raise AssertionError("nothing is cleaned up here")


class FakeBranchTask:
    """A branch of `FakeWorkflowTask`, as the workflow proxy asks it whether it is complete.

    Complete by the rule of a task whose outputs may be merged away downstream (as
    HistFromNtupleProducerTask with `remove_merged_inputs`): its output exists, or the marker
    left in its place does. law's own view of the workflow is the output collection alone,
    which knows nothing of markers.
    """

    def __init__(self, workflow, branch):
        self.workflow = workflow
        self.branch = branch

    def complete(self):
        self.workflow.events.append(("complete", self.branch))
        return os.path.exists(self.workflow.output_path(self.branch)) or os.path.exists(
            self.workflow.marker_path(self.branch)
        )


class FakeWorkflowTask:
    """The workflow task behind a remote proxy, as law's proxy, run() and poll() read it.

    Parameter values are law's defaults (BaseRemoteWorkflow), except `poll_interval`: the poll
    loop must not wait between iterations here. Outputs are real local files, one per branch,
    collected in a real `law.TargetCollection`.
    """

    #: law's default for both workflow types: the job data is dumped on every poll iteration
    dump_intermediate_job_data = True

    tasks_per_job = 1
    submission_threads = 1
    parallel_jobs = law.NO_INT
    poll_interval = 0.0
    poll_fails = 5
    walltime = law.NO_FLOAT
    acceptance = 1.0
    tolerance = 0.0
    retries = 5
    no_poll = False
    shuffle_jobs = False
    append_retry_jobs = False
    clear_logs = False
    ignore_submission = False
    cancel_jobs = False
    cleanup_jobs = False
    align_polling_status_line = False
    check_unreachable_acceptance = False
    reset_branch_map_before_run = False
    cache_branch_map = True
    htcondor_pool = law.NO_STR
    htcondor_scheduler = law.NO_STR
    htcondor_job_kwargs_submit = {}
    htcondor_job_kwargs_query = {}
    crab_job_kwargs_submit = {}
    crab_job_kwargs_query = {}

    #: a poll loop that runs longer than this is a test failure, not a hang
    max_polls = 25

    def __init__(self, workdir, n_branches):
        self.task_id = f"FakeWorkflowTask__{os.path.basename(workdir)}"
        self.workdir = workdir
        self.branch_map = OrderedDict((b, b) for b in range(n_branches))
        self.manager = None
        self.messages = []
        self.n_polls = 0
        self.on_poll = None
        #: ("complete", branch) for each branch asked whether it is complete
        self.events = []
        self.jobs_file = law.LocalFileTarget(os.path.join(workdir, "jobs.json"))
        self.collection = law.TargetCollection(
            OrderedDict(
                (b, law.LocalFileTarget(self.output_path(b))) for b in self.branch_map
            )
        )

    def output_path(self, branch):
        return os.path.join(self.workdir, "outputs", f"branch_{branch}.root")

    def marker_path(self, branch):
        return self.output_path(branch) + ".merged"

    def produce(self, branches):
        for b in branches:
            _touch(self.output_path(b))

    def merge_away(self, branches):
        """What a downstream merge with `remove_merged_inputs` leaves: a marker, no output."""
        for b in branches:
            _touch(self.marker_path(b))
            os.remove(self.output_path(b))

    def as_branch(self, branch):
        return FakeBranchTask(self, branch)

    def outputs(self):
        return {"jobs": self.jobs_file, "collection": self.collection}

    def is_controlling_remote_jobs(self):
        return False

    def get_task_family(self):
        return "FakeWorkflowTask"

    def _crab_cfg(self):
        return {}

    def publish_message(self, msg, *args, **kwargs):
        self.messages.append(str(msg))

    def publish_progress(self, *args, **kwargs):
        pass

    def decrease_running_resources(self, *args, **kwargs):
        pass

    def _handle_scheduler_messages(self):
        pass

    def forward_dashboard_event(self, *args, **kwargs):
        pass

    def modify_polling_status_line(self, status_line):
        return status_line

    def set_tracking_url(self, url):
        pass

    def create_job_dashboard(self):
        return None

    def _poll_callback(self, poll_data):
        self.n_polls += 1
        if self.n_polls > self.max_polls:
            raise AssertionError(f"the poll loop did not end in {self.max_polls} polls")
        if self.on_poll is not None:
            self.on_poll(self.n_polls - 1)
        return True

    def _create_job_manager(self, **kwargs):
        self.manager = ScriptedJobManager(**kwargs)
        return self.manager

    def _factory_dir(self):
        path = os.path.join(self.workdir, "job_files")
        os.makedirs(path, exist_ok=True)
        return path

    # what the proxies look up as `<workflow_type>_<name>` on the task

    def htcondor_create_job_manager(self, **kwargs):
        return self._create_job_manager(**kwargs)

    def crab_create_job_manager(self, **kwargs):
        return self._create_job_manager(**kwargs)

    def htcondor_create_job_file_factory(self, **kwargs):
        return lc.CERNHTCondorJobFileFactory(dir=self._factory_dir())

    def crab_create_job_file_factory(self, **kwargs):
        return lc.FLAFCrabJobFileFactory(dir=self._factory_dir())

    def htcondor_workflow_run_context(self):
        return contextlib.nullcontext()

    crab_workflow_run_context = htcondor_workflow_run_context

    def htcondor_dump_intermediate_job_data(self):
        return self.dump_intermediate_job_data

    crab_dump_intermediate_job_data = htcondor_dump_intermediate_job_data

    def htcondor_job_resources(self, job_num, branches):
        return {}

    crab_job_resources = htcondor_job_resources

    def htcondor_post_submit_delay(self):
        return 0

    crab_post_submit_delay = htcondor_post_submit_delay

    def htcondor_check_job_completeness(self):
        return False

    crab_check_job_completeness = htcondor_check_job_completeness

    def htcondor_check_job_completeness_delay(self):
        return 0

    crab_check_job_completeness_delay = htcondor_check_job_completeness_delay

    def htcondor_poll_callback(self, poll_data):
        return self._poll_callback(poll_data)

    crab_poll_callback = htcondor_poll_callback

    def htcondor_destination_info(self, info):
        return info

    crab_destination_info = htcondor_destination_info


def build_proxy(proxy_cls, workdir, n_branches):
    """A proxy built by its real constructor, with only the batch system scripted.

    `setup_job_manager` is settled up front: for CRAB it needs a CMSSW sandbox, a VOMS proxy and
    a MyProxy credential, none of which a unit-test runner has, and none of which is under test.
    """
    task = FakeWorkflowTask(workdir, n_branches)
    proxy = proxy_cls(task=task)
    # what law's run() sets before it submits or polls, for tests that call submit() directly
    proxy.dashboard = NoJobDashboard()
    proxy._cached_output = task.outputs()
    proxy._job_manager_setup_kwargs = {}

    # one list of job numbers per round that reached the batch system
    proxy.submitted = []

    def submit_group(submit_jobs, **kwargs):
        nums = list(submit_jobs)
        proxy.submitted.append(nums)
        ids = [f"{num}.{len(proxy.submitted)}" for num in nums]
        data = OrderedDict(
            (num, {"job": "job.jdl", "config": None, "log": None}) for num in nums
        )
        return ids, data

    proxy._submit_group = submit_group
    return proxy


@contextlib.contextmanager
def law_submit_spy():
    """Record every call of law's own BaseRemoteWorkflowProxy.submit, and still run it."""
    calls = []
    real_submit = BaseRemoteWorkflowProxy.submit

    def spy(self, retry_jobs=None):
        calls.append(retry_jobs)
        return real_submit(self, retry_jobs=retry_jobs)

    with mock.patch.object(BaseRemoteWorkflowProxy, "submit", spy):
        yield calls


def job(num, status, job_id=JobData.dummy_job_id, error=None):
    """law's job entry for job `num`, which covers branch num - 1 (law numbers jobs from 1)."""
    code = {FINISHED: 0, FAILED: 1}.get(status)
    return JobData.job_data(
        job_id=job_id, branches=[num - 1], status=status, code=code, error=error
    )


class _ProxyCase(unittest.TestCase):
    """A scratch area, FLAF_PATH on a readable tree under it, and the fake clock.

    The scenarios are mixins over this case, instantiated once per proxy class below.
    """

    proxy_cls = None

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="flaf_submission_guards_")
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)
        self.tree = os.path.join(self.workdir, "FLAF")
        os.makedirs(self.tree)
        make_flaf_tree(self.tree)
        env = mock.patch.dict(os.environ, {"FLAF_PATH": self.tree})
        env.start()
        self.addCleanup(env.stop)
        self.clock = FakeClock()
        clock = mock.patch.object(lc, "time", self.clock)
        clock.start()
        self.addCleanup(clock.stop)
        #: what the proxies do in order: ("fresh",) for each require_fresh_negatives(), and
        #: what the tasks and job managers built by `proxy()` add
        self.events = []
        real_require_fresh_negatives = lc.require_fresh_negatives

        def require_fresh_negatives():
            self.events.append(("fresh",))
            real_require_fresh_negatives()

        for patch in (
            mock.patch.object(lc, "require_fresh_negatives", require_fresh_negatives),
            # the process-wide epoch that require_fresh_negatives() moves is put back
            mock.patch.object(
                GFALFileInterface,
                "negatives_valid_after",
                GFALFileInterface.negatives_valid_after,
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def break_tree(self):
        """The storage under the software tree blinks: FLAF's bootstrap.sh cannot be read."""
        os.remove(self.missing_path())

    def restore_tree(self):
        make_flaf_tree(self.tree)

    def missing_path(self):
        return os.path.join(self.tree, *lc._FLAF_JOB_SOURCES[0])

    def proxy(self, n_branches):
        proxy = build_proxy(self.proxy_cls, self.workdir, n_branches)
        proxy.task.events = self.events
        return proxy

    def skip_messages(self, proxy):
        return [m for m in proxy.task.messages if "skipping this submission round" in m]


# ---------------------------------------------------------------------------------------------
# (1) law's own paths survive a failed stat of the software tree
# ---------------------------------------------------------------------------------------------


class LawsPathsWithoutAStat(unittest.TestCase):
    """law resolves its job sources as `rel_path(__file__, ...)`; a module is never a directory."""

    @contextlib.contextmanager
    def tree_unreadable(self):
        with mock.patch("os.path.exists", return_value=False), mock.patch(
            "os.path.isfile", return_value=False
        ):
            yield

    def test_the_crab_wrapper_resolves_to_the_real_file(self):
        with self.tree_unreadable():
            wrapper = cms_job.rel_path(cms_job.__file__, "crab", "crab_wrapper.sh")
            pset = cms_job.rel_path(cms_job.__file__, "crab", "PSet.py")
        self.assertTrue(os.path.isfile(wrapper), wrapper)
        self.assertTrue(os.path.isfile(pset), pset)

    def test_law_job_sh_resolves_to_the_real_file(self):
        with self.tree_unreadable():
            path = law.util.law_src_path("job", "law_job.sh")
        self.assertTrue(os.path.isfile(path), path)

    def test_the_htcondor_wrapper_resolves_to_the_real_file(self):
        with self.tree_unreadable():
            path = htcondor_workflow.rel_path(
                htcondor_workflow.__file__, "htcondor_wrapper.sh"
            )
        self.assertTrue(os.path.isfile(path), path)

    def test_a_directory_anchor_is_still_joined(self):
        # the strict rule must not strip a real directory given as the anchor
        law_dir = os.path.dirname(os.path.abspath(law.__file__))
        path = law.util.rel_path(law_dir, "job", "law_job.sh")
        self.assertTrue(os.path.isfile(path), path)

    def test_every_law_module_binding_is_replaced(self):
        original = lc._law_rel_path
        self.assertIsNot(original, lc._strict_rel_path)
        self.assertEqual(original.__module__, "law.util")
        still_original = sorted(
            name
            for name, module in list(sys.modules.items())
            if name.split(".")[0] == "law"
            and getattr(module, "rel_path", None) is original
        )
        self.assertEqual(still_original, [], "law modules still on law's rel_path")
        for module in (law.util, cms_job, htcondor_workflow):
            self.assertIs(module.rel_path, lc._strict_rel_path, module.__name__)


# ---------------------------------------------------------------------------------------------
# (2) the probe of the job sources
# ---------------------------------------------------------------------------------------------


class TheJobSources(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="flaf_job_sources_")
        self.addCleanup(shutil.rmtree, self.workdir, ignore_errors=True)
        self.clock = FakeClock()
        clock = mock.patch.object(lc, "time", self.clock)
        clock.start()
        self.addCleanup(clock.stop)

    def flaf_path(self, path):
        env = mock.patch.dict(os.environ, {"FLAF_PATH": path})
        env.start()
        self.addCleanup(env.stop)

    def test_every_source_exists_in_the_installed_law_and_in_flaf(self):
        self.flaf_path(flaf_repo)
        paths = lc.job_source_paths()
        missing = [p for p in paths if not os.path.isfile(p)]
        self.assertEqual(missing, [], "job sources that the installed law/FLAF lack")

    def test_the_list_covers_what_law_and_flaf_ship_with_a_job(self):
        """Pinned against how law itself and FLAF's hooks resolve the files they ship."""
        self.flaf_path(flaf_repo)
        paths = set(lc.job_source_paths())
        shipped = {
            law.util.law_src_path("job", "law_job.sh"),
            cms_job.rel_path(cms_job.__file__, "crab", "crab_wrapper.sh"),
            cms_job.rel_path(cms_job.__file__, "crab", "PSet.py"),
            htcondor_workflow.rel_path(
                htcondor_workflow.__file__, "htcondor_wrapper.sh"
            ),
        }
        workflow = types.SimpleNamespace(_flaf_root=lc.flaf_root)
        shipped.add(lc.HTCondorWorkflow.htcondor_bootstrap_file(workflow))
        shipped.add(lc.HTCondorWorkflow.htcondor_stageout_file(workflow))
        shipped.add(lc.CrabWorkflow.crab_bootstrap_file(workflow).path)
        shipped.add(lc.CrabWorkflow.crab_stageout_file(workflow).path)
        self.assertEqual(shipped - paths, set(), "shipped but never probed")

    def test_nothing_is_missing_in_a_readable_tree(self):
        self.flaf_path(flaf_repo)
        self.assertIsNone(lc.missing_job_source(retries=0))

    def test_it_names_the_first_unreadable_source(self):
        self.flaf_path(self.workdir)
        first, second = (os.path.join(self.workdir, *p) for p in lc._FLAF_JOB_SOURCES)
        self.assertEqual(lc.missing_job_source(retries=0), first)
        _touch(first)
        self.assertEqual(lc.missing_job_source(retries=0), second)
        _touch(second)
        self.assertIsNone(lc.missing_job_source(retries=0))

    def test_it_does_not_wait_when_asked_not_to(self):
        self.flaf_path(self.workdir)
        lc.missing_job_source(retries=0)
        self.assertEqual(self.clock.slept, [])

    def test_a_blink_shorter_than_the_retries_is_ridden_out(self):
        self.flaf_path(flaf_repo)
        real_isfile = os.path.isfile
        first = lc.job_source_paths()[0]
        answers = {first: [False, False]}

        def blinking_isfile(path):
            pending = answers.get(path)
            if pending:
                return pending.pop(0)
            return real_isfile(path)

        with mock.patch.object(lc.os.path, "isfile", side_effect=blinking_isfile):
            self.assertIsNone(lc.missing_job_source(retries=2, delay=0.5))
        self.assertEqual(self.clock.slept, [0.5, 0.5])

    def test_the_errno_of_a_missing_path_is_reported(self):
        reason = lc.job_source_error(
            os.path.join(self.workdir, "no", "such", "file.sh")
        )
        self.assertIn(ENOENT, reason)

    def test_a_directory_is_not_reported_as_a_storage_error(self):
        reason = lc.job_source_error(self.workdir)
        self.assertNotIn("Errno", reason)
        self.assertIn("not a regular file", reason)

    def test_the_last_resort_guard_raises_with_the_path_and_errno(self):
        self.flaf_path(self.workdir)
        with self.assertRaises(RuntimeError) as caught:
            lc.wait_for_job_sources()
        msg = str(caught.exception)
        self.assertIn(os.path.join(self.workdir, *lc._FLAF_JOB_SOURCES[0]), msg)
        self.assertIn(ENOENT, msg)

    def test_the_last_resort_guard_passes_a_readable_tree(self):
        self.flaf_path(flaf_repo)
        lc.wait_for_job_sources()


class _JobFileFactoryProbe:
    """A job file must not be built against a tree that went away inside one submission."""

    factory_cls = None
    law_factory_cls = None

    def make_factory(self):
        workdir = tempfile.mkdtemp(prefix="flaf_job_factory_")
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        self.job_file = os.path.join(workdir, "job.cfg")
        with open(self.job_file, "w") as f:
            f.write("universe = vanilla\n")
        return self.factory_cls(dir=workdir)

    def run_create(self, probe):
        factory = self.make_factory()
        order = []

        def law_create(_self, **kwargs):
            order.append("build")
            return self.job_file, types.SimpleNamespace()

        def probe_call():
            order.append("probe")
            return probe()

        with mock.patch.object(
            lc, "wait_for_job_sources", side_effect=probe_call
        ), mock.patch.object(self.law_factory_cls, "create", law_create):
            try:
                factory.create()
            finally:
                self.order = order
        return order

    def test_an_unreadable_tree_stops_the_build(self):
        def unreadable():
            raise RuntimeError("tree gone")

        with self.assertRaises(RuntimeError):
            self.run_create(unreadable)
        self.assertEqual(self.order, ["probe"], "law built the job file anyway")


class TheHTCondorJobFileFactory(_JobFileFactoryProbe, unittest.TestCase):
    factory_cls = lc.CERNHTCondorJobFileFactory
    law_factory_cls = law.htcondor.HTCondorJobFileFactory

    def test_a_readable_tree_is_probed_then_built(self):
        self.assertEqual(self.run_create(lambda: None), ["probe", "build"])


class TheCrabJobFileFactory(_JobFileFactoryProbe, unittest.TestCase):
    factory_cls = lc.FLAFCrabJobFileFactory
    law_factory_cls = law.cms.CrabJobFileFactory


# ---------------------------------------------------------------------------------------------
# (3) law_job.sh without the dependency print-out
# ---------------------------------------------------------------------------------------------


class TheNoPrintDepsJobScript(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.mkdtemp(prefix="flaf_law_job_")
        self.addCleanup(shutil.rmtree, self.data, ignore_errors=True)
        env = mock.patch.dict(os.environ, {"ANALYSIS_DATA_PATH": self.data})
        env.start()
        self.addCleanup(env.stop)
        self.original = law.util.law_src_path("job", "law_job.sh")
        self.custom = os.path.join(self.data, "law_job_no_print_deps.sh")

    def write_custom(self, content, older_than_original):
        with open(self.custom, "w") as f:
            f.write(content)
        mtime = os.path.getmtime(self.original) + (
            -3600 if older_than_original else 3600
        )
        os.utime(self.custom, (mtime, mtime))

    def read_custom(self):
        with open(self.custom) as f:
            return f.read()

    def test_it_is_generated_with_deps_depth_zero(self):
        with open(self.original) as f:
            self.assertRegex(f.read(), r'deps_depth="[1-9]')
        self.assertEqual(lc.law_job_no_print_deps(), self.custom)
        content = self.read_custom()
        self.assertIn('deps_depth="0"', content)
        self.assertNotRegex(content, r'deps_depth="[1-9]')
        self.assertTrue(os.access(self.custom, os.X_OK))

    def test_it_is_regenerated_when_laws_copy_is_newer(self):
        self.write_custom("stale copy\n", older_than_original=True)
        lc.law_job_no_print_deps()
        content = self.read_custom()
        self.assertNotIn("stale copy", content)
        self.assertIn('deps_depth="0"', content)

    def test_an_up_to_date_copy_is_reused(self):
        self.write_custom("current copy\n", older_than_original=False)
        self.assertEqual(lc.law_job_no_print_deps(), self.custom)
        self.assertEqual(self.read_custom(), "current copy\n")

    def test_a_failed_stat_of_laws_tree_reuses_the_existing_copy(self):
        # older than law's copy, so only the failed stat can explain keeping it
        self.write_custom("existing copy\n", older_than_original=True)
        real_getmtime = os.path.getmtime

        def getmtime(path):
            if os.path.abspath(path) == os.path.abspath(self.original):
                raise OSError(errno.EACCES, "Permission denied", path)
            return real_getmtime(path)

        with mock.patch.object(lc.os.path, "getmtime", side_effect=getmtime):
            self.assertEqual(lc.law_job_no_print_deps(), self.custom)
        self.assertEqual(self.read_custom(), "existing copy\n")


# ---------------------------------------------------------------------------------------------
# (4) a submission round while the job sources cannot be read
# ---------------------------------------------------------------------------------------------


class _WhenTheJobSourcesAreUnreadable:
    """Port of DSProd's WhenLawsTreeIsUnreachable, on a real proxy over law's real submit."""

    def backlog_proxy(self):
        """A fresh workflow about to submit its first round: jobs 1 and 2 waiting."""
        proxy = self.proxy(2)
        proxy.job_data.unsubmitted_jobs = OrderedDict([(1, [0]), (2, [1])])
        return proxy

    def running_proxy(self):
        """A workflow in flight: job 1 running, job 2 failed and offered as a retry, job 3
        waiting in the backlog."""
        proxy = self.proxy(3)
        proxy.job_data.jobs[1] = job(1, RUNNING, job_id="1.0")
        proxy.job_data.jobs[2] = job(2, FAILED, job_id="2.0")
        proxy.job_data.attempts[2] = 1
        proxy.job_data.unsubmitted_jobs[3] = [2]
        return proxy

    def test_the_round_is_abandoned_before_law_is_reached(self):
        proxy = self.backlog_proxy()
        self.break_tree()
        with law_submit_spy() as law_calls:
            result = proxy.submit()
        self.assertEqual(law_calls, [], "law's submit() was handed the round")
        self.assertEqual(proxy.submitted, [])
        self.assertEqual(result, OrderedDict())

    def test_no_job_leaves_the_backlog(self):
        proxy = self.backlog_proxy()
        self.break_tree()
        proxy.submit()
        self.assertEqual(list(proxy.job_data.unsubmitted_jobs), [1, 2])
        self.assertEqual(proxy.job_data.jobs, {})

    def test_offered_retries_are_parked_in_front_and_dumped(self):
        proxy = self.running_proxy()
        n_jobs = len(proxy.job_data)
        self.break_tree()
        proxy.submit({2: [1]})
        self.assertEqual(list(proxy.job_data.unsubmitted_jobs), [2, 3])
        self.assertNotIn(2, proxy.job_data.jobs)
        self.assertIn(1, proxy.job_data.jobs)
        # law's poll loop snapshots len(job_data) once; it must not move
        self.assertEqual(len(proxy.job_data), n_jobs)
        self.assertEqual(proxy.job_data.attempts, {2: 1}, "an attempt was spent")
        # a killed driver must find the parked retry again
        dumped = proxy.task.jobs_file.load(formatter="json")
        self.assertEqual(list(dumped["unsubmitted_jobs"]), ["2", "3"])

    def test_the_reason_carries_the_path_and_what_the_storage_said(self):
        proxy = self.backlog_proxy()
        self.break_tree()
        proxy.submit()
        msgs = self.skip_messages(proxy)
        self.assertEqual(len(msgs), 1, proxy.task.messages)
        self.assertIn(self.missing_path(), msgs[0])
        self.assertIn(ENOENT, msgs[0])
        self.assertIn("nothing is lost", msgs[0])
        self.assertIn("next poll", msgs[0])

    def test_the_probe_does_not_wait_the_tree_out(self):
        # it runs inside the poll loop: one short re-probe at most, never the 15 s of the
        # last-resort guard per poll
        proxy = self.backlog_proxy()
        self.break_tree()
        proxy.submit()
        self.assertLessEqual(sum(self.clock.slept), 1.0)

    def test_a_readable_tree_does_not_skip_the_round(self):
        proxy = self.backlog_proxy()
        with law_submit_spy() as law_calls:
            result = proxy.submit()
        self.assertEqual(len(law_calls), 1, "law's submit() was not reached")
        self.assertEqual(proxy.submitted, [[1, 2]])
        self.assertEqual(list(result), [1, 2])
        self.assertEqual(self.skip_messages(proxy), [])

    def test_no_poll_raises_instead_of_skipping(self):
        # a --no-poll invocation submits once and returns: nothing would submit this later
        proxy = self.backlog_proxy()
        proxy.task.no_poll = True
        self.break_tree()
        with self.assertRaises(RuntimeError) as caught:
            proxy.submit()
        self.assertIn(self.missing_path(), str(caught.exception))
        self.assertIn(ENOENT, str(caught.exception))
        self.assertEqual(proxy.submitted, [])
        self.assertEqual(list(proxy.job_data.unsubmitted_jobs), [1, 2])

    def test_skipping_for_longer_than_the_budget_raises(self):
        budget = lc.SubmissionGuards.max_skip_minutes
        proxy = self.running_proxy()
        self.break_tree()
        proxy.submit({2: [1]})
        self.clock.advance(budget - 1)
        proxy.submit()
        self.assertEqual(len(self.skip_messages(proxy)), 2)
        self.clock.advance(2)
        with self.assertRaises(RuntimeError) as caught:
            proxy.submit()
        self.assertIn(self.missing_path(), str(caught.exception))
        self.assertIn(f"{budget:.0f} minutes", str(caught.exception))
        self.assertEqual(proxy.submitted, [])

    def test_a_readable_round_restarts_the_budget(self):
        budget = lc.SubmissionGuards.max_skip_minutes
        proxy = self.running_proxy()
        self.break_tree()
        proxy.submit({2: [1]})
        self.clock.advance(10)
        self.restore_tree()
        proxy.submit()
        self.assertEqual(proxy.submitted, [[2, 3]])
        self.clock.advance(budget)
        self.break_tree()
        proxy.job_data.jobs[1]["status"] = FAILED
        proxy.submit({1: [0]})
        self.assertEqual(len(self.skip_messages(proxy)), 2)
        self.assertEqual(list(proxy.job_data.unsubmitted_jobs), [1])

    def test_nothing_waiting_does_not_probe(self):
        proxy = self.proxy(1)
        proxy.job_data.jobs[1] = job(1, RUNNING, job_id="1.0")
        with mock.patch.object(
            lc, "missing_job_source", wraps=lc.missing_job_source
        ) as probe:
            proxy.submit({})
            proxy.submit(None)
        probe.assert_not_called()
        self.assertEqual(proxy.submitted, [])

    def test_attempts_are_not_spent_while_rounds_are_skipped(self):
        """law's real poll loop: job 2 fails while the tree is unreadable for three polls.

        A failed job left in law's `jobs` is queried again on the next poll, reads FAILED again
        and spends another attempt; parked in the backlog it waits without being polled, and
        goes out with the attempt its one real failure cost once the tree is back.
        """
        proxy = self.proxy(2)
        proxy.job_data.jobs[1] = job(1, RUNNING, job_id="1.0")
        proxy.job_data.jobs[2] = job(2, RUNNING, job_id="2.0")
        proxy.task.manager.script = {
            "1.0": [RUNNING, RUNNING, RUNNING, FINISHED],
            "2.0": [FAILED],
        }
        self.break_tree()

        def on_poll(iteration):
            if iteration == 3:
                self.restore_tree()

        proxy.task.on_poll = on_poll
        proxy.poll()

        self.assertEqual(len(self.skip_messages(proxy)), 3, proxy.task.messages)
        self.assertEqual(proxy.job_data.attempts, {2: 1}, "attempts spent on skips")
        self.assertEqual(dict(proxy._job_retries), {2: 1})
        self.assertEqual(proxy.submitted, [[2]])
        self.assertEqual(proxy.job_data.unsubmitted_jobs, {})
        self.assertEqual(
            {num: data["status"] for num, data in proxy.job_data.jobs.items()},
            {1: FINISHED, 2: FINISHED},
        )


class HTCondorWhenTheJobSourcesAreUnreadable(
    _WhenTheJobSourcesAreUnreadable, _ProxyCase
):
    proxy_cls = lc._BundleAwareHTCondorWorkflowProxy


class CrabWhenTheJobSourcesAreUnreadable(_WhenTheJobSourcesAreUnreadable, _ProxyCase):
    proxy_cls = lc._FLAFCrabWorkflowProxy


# ---------------------------------------------------------------------------------------------
# (5) a resumed workflow whose jobs come back in large numbers for missing outputs
# ---------------------------------------------------------------------------------------------


class _MassLostOutputsBrake:
    """law's real run() over a resumed submission file, 20 jobs of one branch each.

    law re-checks every job it had recorded as finished: one whose outputs are gone carries no
    job id any more and comes back as "unknown job id"; one that was still live and is now
    reported finished without outputs comes back as "initially missing task outputs". Both are
    retried in the first poll -- which, for a storage outage during the check or outputs removed
    after use, means redoing the production. law judges from the output collection as it
    gathered it; the brake counts a job only when its branches are still incomplete when the
    task's own complete() is asked again.
    """

    N_JOBS = 20

    def resumed(self, finished=(), running=(), outputs=(), script=None, attempts=None):
        """Write the submission file of a previous run and return a proxy that will resume it.

        `finished` jobs were recorded FINISHED (dummy id), `running` ones RUNNING with id
        r<num>; only the jobs in `outputs` have their output on disk. `attempts` is the retry
        counter earlier runs left in the file.
        """
        proxy = self.proxy(self.N_JOBS)
        data = JobData()
        for num in finished:
            data.jobs[num] = job(num, FINISHED)
        for num in running:
            data.jobs[num] = job(num, RUNNING, job_id=f"r{num}")
        data.attempts.update(attempts or {})
        self.assertEqual(len(data.jobs), self.N_JOBS)
        proxy.task.jobs_file.dump(data, formatter="json", indent=4)
        proxy.task.produce(num - 1 for num in outputs)
        proxy.task.manager.script = dict(script or {})
        return proxy

    def all_finished(self, lost=()):
        jobs = range(1, self.N_JOBS + 1)
        return self.resumed(
            finished=jobs, outputs=[num for num in jobs if num not in lost]
        )

    def stop_message(self, n_lost):
        return f"{n_lost} of the {self.N_JOBS} jobs of this resumed workflow"

    def asked_complete(self):
        """The branches whose complete() was asked, in order."""
        return [event[1] for event in self.events if event[0] == "complete"]

    def test_mass_lost_outputs_stop_the_run_before_anything_is_submitted(self):
        proxy = self.all_finished(lost=range(1, 6))
        with self.assertRaises(RuntimeError) as caught:
            proxy.run()
        msg = str(caught.exception)
        self.assertIn(self.stop_message(5), msg)
        self.assertIn("still missing on a fresh look", msg)
        self.assertIn("--ignore-submission", msg)
        self.assertEqual(proxy.submitted, [])
        # the verdict rests on the task's own rule, asked after absence was made to rest on a
        # fresh listing
        self.assertEqual(sorted(self.asked_complete()), [0, 1, 2, 3, 4])
        first_complete = self.events.index(("complete", self.asked_complete()[0]))
        self.assertIn(("fresh",), self.events[:first_complete])

    def test_a_single_lost_job_is_retried(self):
        proxy = self.all_finished(lost=[7])
        proxy.run()
        self.assertEqual(proxy.submitted, [[7]])
        self.assertEqual(proxy.job_data.attempts, {7: 1})

    def test_lost_outputs_up_to_the_fraction_are_retried(self):
        allowed = int(lc.SubmissionGuards.max_lost_fraction * self.N_JOBS)
        proxy = self.all_finished(lost=range(1, allowed + 1))
        proxy.run()
        self.assertEqual(proxy.submitted, [list(range(1, allowed + 1))])
        # within the fraction, nothing is looked at again
        self.assertEqual(self.asked_complete(), [])

    def test_lost_outputs_beyond_the_fraction_stop_the_run(self):
        allowed = int(lc.SubmissionGuards.max_lost_fraction * self.N_JOBS)
        proxy = self.all_finished(lost=range(1, allowed + 2))
        with self.assertRaises(RuntimeError) as caught:
            proxy.run()
        self.assertIn(self.stop_message(allowed + 1), str(caught.exception))
        self.assertEqual(proxy.submitted, [])

    def test_initially_missing_outputs_are_counted_with_the_lost_ones(self):
        # 2 recorded-finished jobs lost, and 1 of the live jobs reported finished without its
        # output: each alone is within the fraction, together they are not
        proxy = self.resumed(
            finished=range(1, 11),
            running=range(11, 21),
            outputs=[num for num in range(3, 21) if num != 15],
            script={f"r{num}": [FINISHED] for num in range(11, 21)},
        )
        with self.assertRaises(RuntimeError) as caught:
            proxy.run()
        self.assertIn(self.stop_message(3), str(caught.exception))
        self.assertEqual(proxy.submitted, [])

    def test_genuine_failures_beside_the_lost_outputs_are_not_counted(self):
        # first poll: 2 lost outputs (within the fraction) and 8 live jobs that genuinely fail
        script = {f"r{num}": [FAILED] for num in range(11, 19)}
        script.update({f"r{num}": [RUNNING, FINISHED] for num in (19, 20)})
        proxy = self.resumed(
            finished=range(1, 11),
            running=range(11, 21),
            outputs=range(3, 11),
            script=script,
        )
        proxy.run()
        self.assertEqual(len(proxy.submitted), 1)
        self.assertEqual(sorted(proxy.submitted[0]), [1, 2] + list(range(11, 19)))

    def test_genuine_later_failures_are_not_counted(self):
        # the 2 lost jobs are retried and fail genuinely, as do 8 live jobs, on the next poll
        script = {f"r{num}": [RUNNING, FAILED] for num in range(11, 19)}
        script.update({f"r{num}": [RUNNING, RUNNING, FINISHED] for num in (19, 20)})
        script.update({"1.1": [FAILED], "2.1": [FAILED]})
        proxy = self.resumed(
            finished=range(1, 11),
            running=range(11, 21),
            outputs=range(3, 11),
            script=script,
        )
        proxy.run()
        self.assertEqual(len(proxy.submitted), 2)
        self.assertEqual(proxy.submitted[0], [1, 2])
        self.assertEqual(sorted(proxy.submitted[1]), [1, 2] + list(range(11, 19)))

    def test_the_verdict_is_given_once(self):
        """After the first retry generation, recorded-finished jobs no longer count."""
        proxy = self.proxy(self.N_JOBS)
        for num in range(1, self.N_JOBS + 1):
            proxy.job_data.jobs[num] = job(num, FINISHED)
        proxy._submitted = True
        proxy._snapshot_resumed_jobs()
        proxy._stop_on_mass_lost_outputs({1: [0]})
        proxy._stop_on_mass_lost_outputs(
            {num: [num - 1] for num in range(1, self.N_JOBS + 1)}
        )

    def test_a_fresh_run_never_stops(self):
        # no submission file: 8 of 20 first submissions fail, and all of them are retried
        proxy = self.proxy(self.N_JOBS)
        proxy.task.manager.script = {f"{num}.1": [FAILED] for num in range(1, 9)}
        proxy.run()
        self.assertEqual(len(proxy.submitted), 2)
        self.assertEqual(proxy.submitted[0], list(range(1, self.N_JOBS + 1)))
        self.assertEqual(sorted(proxy.submitted[1]), list(range(1, 9)))

    # A resumed run in which no job had been recorded FINISHED still counts its "initially
    # missing task outputs" jobs: only a fresh run skips the brake.
    def test_initially_missing_outputs_alone_stop_the_run(self):
        proxy = self.resumed(
            running=range(1, 21),
            outputs=range(6, 21),
            script={f"r{num}": [FINISHED] for num in range(1, 21)},
        )
        with self.assertRaises(RuntimeError) as caught:
            proxy.run()
        self.assertIn(self.stop_message(5), str(caught.exception))
        self.assertEqual(proxy.submitted, [])

    # A skipped round parks the offered generation in `unsubmitted_jobs`, where law would
    # submit it on a later poll without offering it as retries again, so the brake has to
    # judge it before the round is skipped.
    def test_a_skipped_first_round_does_not_bypass_the_brake(self):
        proxy = self.all_finished(lost=range(1, 11))
        self.break_tree()

        def on_poll(iteration):
            if iteration == 1:
                self.restore_tree()

        proxy.task.on_poll = on_poll
        with self.assertRaises(RuntimeError):
            proxy.run()
        self.assertEqual(proxy.submitted, [])

    # -- what a stop leaves behind --------------------------------------------------------------

    @staticmethod
    def lost_and_live_script():
        """Live jobs 11 and 12 report finished; 13-20 are still running."""
        script = {f"r{num}": [FINISHED] for num in (11, 12)}
        script.update({f"r{num}": [RUNNING] for num in range(13, 21)})
        return script

    def lost_and_live(self):
        """5 of 20 jobs come back: recorded-finished 1-3 lost their outputs, and live 11 and 12
        are reported finished without theirs. Jobs 2 and 11 had been retried before."""
        return self.resumed(
            finished=range(1, 11),
            running=range(11, 21),
            outputs=range(4, 11),
            script=self.lost_and_live_script(),
            attempts={2: 1, 11: 2},
        )

    def test_a_stop_leaves_the_submission_file_as_it_found_it(self):
        """law dumps the job data on every poll by default -- after it has rewritten the jobs
        that came back as retries and counted their attempts, before it offers them to
        submit(). Left like that, the next run would find them as retries, not as finished
        or live jobs, and resubmit them without a verdict."""
        proxy = self.lost_and_live()
        before = proxy.task.jobs_file.load(formatter="json")
        dumped = []
        dump = proxy.dump_job_data

        def recorded_dump():
            dumped.append(
                (
                    {num: data["status"] for num, data in proxy.job_data.jobs.items()},
                    dict(proxy.job_data.attempts),
                )
            )
            dump()

        proxy.dump_job_data = recorded_dump
        with self.assertRaises(RuntimeError) as caught:
            proxy.run()
        self.assertIn(self.stop_message(5), str(caught.exception))
        self.assertEqual(proxy.submitted, [])
        # law's own dump of the first poll, then the brake's
        self.assertEqual(len(dumped), 2, "law's intermediate dump did not happen")
        statuses, attempts = dumped[0]
        self.assertEqual(
            {num for num, status in statuses.items() if status == RETRY},
            {1, 2, 3, 11, 12},
        )
        self.assertEqual(attempts, {1: 1, 2: 2, 3: 1, 11: 3, 12: 1})
        after = proxy.task.jobs_file.load(formatter="json")
        self.assertEqual(after["jobs"], before["jobs"])
        self.assertEqual(after["attempts"], {"2": 1, "11": 2})
        self.assertEqual(after["unsubmitted_jobs"], {})

    def test_the_next_resumed_run_stops_again(self):
        with self.assertRaises(RuntimeError):
            self.lost_and_live().run()
        again = self.proxy(self.N_JOBS)
        again.task.manager.script = self.lost_and_live_script()
        with self.assertRaises(RuntimeError) as caught:
            again.run()
        self.assertIn(self.stop_message(5), str(caught.exception))
        self.assertEqual(again.submitted, [])
        self.assertEqual(again.job_data.attempts, {2: 1, 11: 2})

    # law's intermediate dump of the first poll comes before submit(), where the brake
    # judges; until then the jobs that came back for missing outputs are written as they were
    # loaded, so a driver ending in between leaves the next run something to judge.
    def test_a_run_ended_before_the_verdict_leaves_it_to_the_next_run(self):
        proxy = self.all_finished(lost=range(1, 6))

        def driver_ends(iteration):
            raise KeyboardInterrupt

        proxy.task.on_poll = driver_ends
        with self.assertRaises(KeyboardInterrupt):
            proxy.run()
        self.assertEqual(proxy.submitted, [])
        again = self.proxy(self.N_JOBS)
        with self.assertRaises(RuntimeError) as caught:
            again.run()
        self.assertIn(self.stop_message(5), str(caught.exception))
        self.assertEqual(again.submitted, [])

    # -- what does not count as lost --------------------------------------------------------------

    def test_live_jobs_that_finished_during_startup_are_not_counted(self):
        """20 live jobs; 3 of them finish, outputs written, after law has gathered the existing
        outputs (luigi's scheduling, then run()) and before its first status query. law reads
        them as "initially missing task outputs" -- 3 of 20, above the fraction -- but their
        outputs are there when asked again, so the run goes on."""
        early = (1, 2, 3)
        script = {f"r{num}": [FINISHED] for num in early}
        script.update({f"r{num}": [RUNNING, FINISHED] for num in range(4, 21)})
        proxy = self.resumed(running=range(1, 21), script=script)
        task = proxy.task

        def job_writes_its_output(job_id, status):
            self.events.append(("query", job_id))
            if status == FINISHED and job_id.startswith("r"):
                task.produce([int(job_id[1:]) - 1])

        task.manager.on_query = job_writes_its_output
        # luigi's scheduling: law gathers the existing outputs and caches its verdicts
        proxy.process_resources()
        proxy.run()

        self.assertEqual(
            {num: data["status"] for num, data in proxy.job_data.jobs.items()},
            {num: FINISHED for num in range(1, self.N_JOBS + 1)},
        )
        # law resubmits them still, on the verdict it cached; what is at stake here is the stop
        self.assertLessEqual(
            {num for nums in proxy.submitted for num in nums}, set(early)
        )
        self.assertEqual(sorted(self.asked_complete()), [0, 1, 2])
        # asked after the jobs were seen finished, with absence resting on a fresh listing
        first_query = next(i for i, e in enumerate(self.events) if e[0] == "query")
        first_complete = self.events.index(("complete", self.asked_complete()[0]))
        self.assertIn(("fresh",), self.events[first_query:first_complete])

    def test_branches_complete_by_their_own_rule_are_not_counted(self):
        """5 of 20 recorded-finished jobs had their outputs merged away downstream, markers left
        in their place. law, which reads the output collection only, retries them as "unknown
        job id"; the task's own complete() counts them done."""
        proxy = self.all_finished()
        proxy.task.merge_away(range(5))
        proxy.run()
        self.assertEqual(sorted(self.asked_complete()), [0, 1, 2, 3, 4])
        self.assertEqual(
            {num: data["status"] for num, data in proxy.job_data.jobs.items()},
            {num: FINISHED for num in range(1, self.N_JOBS + 1)},
        )

    def test_only_branches_still_incomplete_are_counted(self):
        # 5 merged away and 3 genuinely lost: 8 come back, 3 count -- still above the fraction
        proxy = self.all_finished(lost=range(6, 9))
        proxy.task.merge_away(range(5))
        with self.assertRaises(RuntimeError) as caught:
            proxy.run()
        self.assertIn(self.stop_message(3), str(caught.exception))
        self.assertEqual(proxy.submitted, [])
        self.assertEqual(sorted(self.asked_complete()), list(range(8)))


class HTCondorMassLostOutputsBrake(_MassLostOutputsBrake, _ProxyCase):
    proxy_cls = lc._BundleAwareHTCondorWorkflowProxy


class CrabMassLostOutputsBrake(_MassLostOutputsBrake, _ProxyCase):
    proxy_cls = lc._FLAFCrabWorkflowProxy


class TheJobDataIsDumpedOnEveryPoll(unittest.TestCase):
    """What the brake has to undo depends on law dumping the job data on every poll, as the
    stand-in task does: pinned against law and against FLAF's workflows, which keep it.
    """

    def test_law_and_flaf_dump_on_every_poll(self):
        for workflow, law_workflow, name in (
            (
                lc.HTCondorWorkflow,
                law.htcondor.HTCondorWorkflow,
                "htcondor_dump_intermediate_job_data",
            ),
            (lc.CrabWorkflow, law.cms.CrabWorkflow, "crab_dump_intermediate_job_data"),
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(workflow, name), getattr(law_workflow, name))
                self.assertIs(getattr(law_workflow, name)(None), True)
                self.assertIs(FakeWorkflowTask.dump_intermediate_job_data, True)


class CrabResumedRunGathersOutputsAgain(_ProxyCase):
    """A resumed CRAB run judges which outputs exist from what it gathers when it starts to
    run, not from what luigi's scheduling gathered: a CRAB worker cannot publish to the
    path-cache server, and an output written while no driver was polling must not send its
    branch back to the grid."""

    proxy_cls = lc._FLAFCrabWorkflowProxy

    def resumed_live(self, n_jobs, finished_early=()):
        """A resumed file of `n_jobs` live jobs; those in `finished_early` report finished from
        the first status query on, the others one query later. A job writes its output when it
        is first seen finished."""
        proxy = self.proxy(n_jobs)
        data = JobData()
        for num in range(1, n_jobs + 1):
            data.jobs[num] = job(num, RUNNING, job_id=f"r{num}")
        proxy.task.jobs_file.dump(data, formatter="json", indent=4)
        task = proxy.task
        task.manager.script = {
            f"r{num}": [FINISHED] if num in finished_early else [RUNNING, FINISHED]
            for num in range(1, n_jobs + 1)
        }

        def job_writes_its_output(job_id, status):
            self.events.append(("query", job_id))
            if status == FINISHED and job_id.startswith("r"):
                task.produce([int(job_id[1:]) - 1])

        task.manager.on_query = job_writes_its_output
        return proxy

    def test_outputs_written_after_scheduling_are_found(self):
        proxy = self.resumed_live(4, finished_early=(1, 2))
        # luigi's scheduling: law gathers the existing outputs and caches its verdicts
        proxy.process_resources()
        # jobs 1 and 2 finish, outputs written, while luigi is still scheduling
        proxy.task.produce([0, 1])
        proxy.run()
        self.assertEqual(proxy.submitted, [])
        self.assertEqual(
            {num: data["status"] for num, data in proxy.job_data.jobs.items()},
            {num: FINISHED for num in range(1, 5)},
        )
        # finished on sight: never queried, never offered as retries, never judged
        self.assertEqual({e[1] for e in self.events if e[0] == "query"}, {"r3", "r4"})
        self.assertEqual([e for e in self.events if e[0] == "complete"], [])

    def test_absence_rests_on_listings_taken_from_run_on(self):
        proxy = self.resumed_live(2)
        gather = proxy._get_existing_branches

        def recorded_gather(*args, **kwargs):
            self.events.append(("gather",))
            return gather(*args, **kwargs)

        proxy._get_existing_branches = recorded_gather
        proxy.process_resources()
        n_before_run = len(self.events)
        proxy.run()
        after_run = self.events[n_before_run:]
        self.assertIn(("gather",), after_run)
        self.assertIn(("fresh",), after_run[: after_run.index(("gather",))])


# ---------------------------------------------------------------------------------------------
# (6) a batch job does not build an upstream product inline
# ---------------------------------------------------------------------------------------------


class SubmissionGuardsProducer(luigi.Task):
    """Stands in for the producer whose run() a job would reach inline."""

    branch = luigi.IntParameter(default=7)


class SubmissionGuardsJobRoot(luigi.Task):
    """Stands in for the task a batch job was submitted to run."""

    branch = luigi.IntParameter(default=0)


@contextlib.contextmanager
def job_command_line(args):
    """luigi's parser for `law run <args>`, as on a worker."""
    with luigi.cmdline_parser.CmdlineParser.global_instance(
        list(args), allow_override=True
    ) as parser:
        yield parser


class NoInlineBuildOnAWorker(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"LAW_JOB_HOME": "/srv/job"})
        env.start()
        self.addCleanup(env.stop)

    def refuse(self, producer=None):
        lc.HTCondorWorkflow._refuse_inline_on_worker(
            producer or SubmissionGuardsProducer(branch=7)
        )

    def test_the_submitted_family_is_read_from_the_command_line(self):
        with job_command_line(["SubmissionGuardsJobRoot", "--branch", "3"]):
            self.assertEqual(lc.submitted_task_family(), "SubmissionGuardsJobRoot")

    def test_another_tasks_job_refuses(self):
        with job_command_line(["SubmissionGuardsJobRoot", "--branch", "3"]):
            with self.assertRaises(RuntimeError) as caught:
                self.refuse()
        msg = str(caught.exception)
        self.assertIn("SubmissionGuardsProducer", msg)
        self.assertIn("SubmissionGuardsJobRoot", msg)

    def test_its_own_job_runs(self):
        with job_command_line(["SubmissionGuardsProducer", "--branch", "7"]):
            self.refuse()

    def test_off_a_batch_node_it_runs(self):
        os.environ.pop("LAW_JOB_HOME")
        with job_command_line(["SubmissionGuardsJobRoot", "--branch", "3"]):
            self.refuse()

    def test_an_unknown_root_task_lets_it_run(self):
        with mock.patch.object(luigi.cmdline_parser.CmdlineParser, "_instance", None):
            self.assertIsNone(lc.submitted_task_family())
            self.refuse()


#: the producers that must refuse before doing anything, by source file
PRODUCERS = {
    os.path.join("AnaProd", "tasks.py"): ("AnaTupleFileTask", "AnaTupleMergeTask"),
    os.path.join("Analysis", "tasks.py"): (
        "AnalysisCacheTask",
        "HistTupleProducerTask",
        "HistFromNtupleProducerTask",
        "HistMergerTask",
    ),
}


def read_source(rel_path):
    with open(os.path.join(flaf_repo, rel_path)) as f:
        return f.read()


def run_guard_problems(source, class_names):
    """What is wrong with the run() of each of `class_names` in `source`, as text.

    Read from the source rather than imported: the task modules need ROOT and awkward, which
    the unit-test runner does not have.
    """
    classes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
    }
    problems = []
    for name in class_names:
        cls = classes.get(name)
        if cls is None:
            problems.append(f"{name}: class not found")
            continue
        bases = {ast.unparse(base) for base in cls.bases}
        if "HTCondorWorkflow" not in bases:
            problems.append(f"{name}: does not derive from HTCondorWorkflow")
        run = next(
            (
                node
                for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "run"
            ),
            None,
        )
        if run is None:
            problems.append(f"{name}: no run()")
            continue
        body = list(run.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        first = ast.unparse(body[0]) if body else ""
        if first != "self._refuse_inline_on_worker()":
            problems.append(f"{name}.run() starts with {first!r}")
    return problems


class TheProducersRefuseFirst(unittest.TestCase):
    def test_each_producer_run_starts_with_the_guard(self):
        problems = []
        for rel_path, class_names in PRODUCERS.items():
            problems += run_guard_problems(read_source(rel_path), class_names)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
