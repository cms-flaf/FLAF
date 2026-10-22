#!/usr/bin/env python3
"""Per-era submission state of a FLAF remote workflow over several eras.

A workflow with era branches over several eras (`--era A,B`) runs one poll loop, but every
era's jobs live in the job file a run over that era alone reads and writes, and every era's jobs
are laid out, built and submitted by a proxy of that era's own workflow. What is proven here:

- EraJobFiles: each era's file is the single-era file (path, name, local job and branch
  numbers, its own tasks_per_job, byte-identical JSON); load() merges with the era-id shift;
  dump() writes each era back in its own numbers; parent.touch(), remove() and exists() cover
  every era; growing one era renames that era's file only (I4); from the same fresh state each
  era's file holds what the run over that era alone writes, in flight and finished.
- The layout of the merged job data through the real proxies: fresh eras chunked with the
  root's --tasks-per-job or packed by each era's own costs as the run over that era alone packs
  them, a resumed era keeps its file and has its unsubmitted jobs packed again as the resumed
  run over that era alone does, a complete era is left out with its file untouched, the job
  count law's poll() snapshots is the final one, and nothing is laid out while law's dynamic
  workflow condition leaves the root with its placeholder map.
- --cancel-jobs and --cleanup-jobs hand the job manager every era's jobs and lay out nothing; a
  round in which one era's jobs fail to build records the ids of the eras submitted before it,
  and the next run (also with --retries 0) submits every era as the runs over each era alone
  with the same failure do.
- Every job equals the job a run over that era alone builds with the same flags: command line,
  local branches, chunks, HTCondor and CRAB resources, worker-file, log and heartbeat paths,
  bundles, the CRAB request-name prefix and the staged path cache, also as the job itself sees
  it with LAW_CRAB_JOB_NUMBER set (I8); retry escalation, no leak into the era's own workflow,
  submission order, routed intermediate dumps that carry each job's id from the dump after
  its submission, the per-era lost-outputs guard, no job resources booked at scheduling, the
  CRAB watchdog composite and the shared job manager.
- The per-era luigi resource token: two instances of a stage that share an era never run at
  once, instances over other eras or of other producers do (C7, I17(13)).
- Cross-run compatibility (I22): drivers killed with os._exit and restarted over other era
  sets, on HTCondor and CRAB fakes, submit for each era exactly what the single-era run in the
  same state submits.
- A workflow over several eras without era branches (StatInference's card stages), through
  luigi's worker: the plain proxy, law's own branch numbers, its requirements and its branch
  tasks and jobs over every era of the run, no map built on a single-era instance, one job file,
  one luigi token and one remote log directory (given its own store_parts and
  remote_log_dir_target).

Every invariant has a negative control: a mutation of the implementation under which the test
fails (see the report of the change set). Only what a unit-test runner lacks is faked: the batch
systems (an HTCondor job manager and the `crab` executable over one shared state file), the
grid storage listing, the CMSSW sandbox, the credentials and kinit.
"""

import base64
import contextlib
import datetime
import fcntl
import glob
import gzip
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter, OrderedDict
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import law  # noqa: E402
import luigi  # noqa: E402
import luigi.cmdline_parser  # noqa: E402
import luigi.interface  # noqa: E402

# law_customizations imports ROOT and Corrections at module level, which the unit-test runner
# does not have and nothing tested here uses: stand-ins for these imports alone.
_HEAVY_MODULES = [
    "ROOT",
    "numpy",
    "awkward",
    "uproot",
    "Corrections",
    "Corrections.Corrections",
    "Corrections.CorrectionsCore",
    "Corrections.lumi",
]


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_replaced = {
    name: sys.modules.get(name) for name in _HEAVY_MODULES if not _importable(name)
}
sys.modules.update({name: mock.MagicMock() for name in _replaced})
try:
    from FLAF.Common import Setup as setup_module
    from FLAF.Common.Setup import ERA_BRANCH_STRIDE, ERA_IDS, Setup
    from FLAF.run_tools import law_customizations as lc
finally:
    for _name, _module in _replaced.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

import law.contrib.cms.job as cms_job  # noqa: E402
from law.job.dashboard import NoJobDashboard  # noqa: E402
from law.workflow.remote import BaseRemoteWorkflowProxy, JobData  # noqa: E402

from FLAF.RunKit import law_gfal  # noqa: E402
from FLAF.RunKit.law_gfal import GFALFileInterface  # noqa: E402
from FLAF.run_tools import crab_watchdog  # noqa: E402
from FLAF.run_tools.crab_sites import SiteStats  # noqa: E402

A, B, C = "Run3_2022", "Run3_2022EE", "Run3_2023"
STRIDE = ERA_BRANCH_STRIDE
BASE = "davs://storage.example:1094/store/user/flaf"

FILES = {
    "config/global.yaml": f"""
phys_model: M
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
fs_default: {BASE}
bundles:
  core:
    patterns: [ "config/{{era}}" ]
    hashed: true
crab:
  whitelist: [ T2_CH_CERN ]
  watchdog:
    enabled: true
    interval_minutes: 10
""",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT ]\n",
    "config/processes.yaml": "TT:\n  datasets: [ TTto2L2Nu ]\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
}
for _era in (A, B, C):
    FILES[f"config/{_era}/datasets.yaml"] = "TTto2L2Nu: {}\n"
    # the era's own bundle content
    FILES[f"config/{_era}/notes.txt"] = f"{_era}\n"


def root_dir():
    """The scratch area of this module: the analysis, its data area, the fake batch system,
    the fake storage and the outputs, shared by the driver processes through the environment.
    """
    return os.environ["ERA_JOBS_ROOT"]


def make_scratch_analysis(root):
    ana = os.path.join(root, "ana")
    os.makedirs(ana)
    os.symlink(flaf_repo, os.path.join(ana, "FLAF"))
    for path, text in FILES.items():
        full = os.path.join(ana, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(text)
    os.makedirs(os.path.join(root, "data"))
    os.makedirs(os.path.join(root, "tmp"))
    return {
        "ERA_JOBS_ROOT": root,
        "ANALYSIS_PATH": ana,
        "ANALYSIS_DATA_PATH": os.path.join(root, "data"),
        "FLAF_PATH": flaf_repo,
        "FLAF_ENVIRONMENT_PATH": os.path.join(root, "no_env"),
        "LAW_JOB_FILE_DIR": os.path.join(root, "job_files"),
    }


_module_env = None


def setUpModule():
    global _module_env
    root = tempfile.mkdtemp(prefix="flaf_era_jobs_")
    _module_env = mock.patch.dict(os.environ, make_scratch_analysis(root))
    _module_env.start()
    # a driver here is never a batch job, and has no grid proxy
    for name in ("LAW_CRAB_JOB_NUMBER", "LAW_JOB_HOME", "X509_USER_PROXY", "CRAB_Id"):
        os.environ.pop(name, None)
    # The drivers run as processes of their own (_CrossRuns) share a path-cache server, as
    # on a submit machine: one of this module's (short: a socket path holds 107 bytes).
    os.environ["FLAF_PATH_CACHE_DIR"] = os.path.join(
        tempfile.mkdtemp(prefix="fpc", dir="/tmp"), "cache"
    )


def tearDownModule():
    root = os.environ.get("ERA_JOBS_ROOT")
    server_dir = os.environ.get("FLAF_PATH_CACHE_DIR")
    _module_env.stop()
    if root:
        shutil.rmtree(root, ignore_errors=True)
    if server_dir:
        # the server outlives the drivers that started it, by design
        subprocess.run(["pkill", "-KILL", "-f", "--", f"--socket {server_dir}/"])
        shutil.rmtree(os.path.dirname(server_dir), ignore_errors=True)


# --------------------------------------------------------------------------------------------
# what a runner does not have: the grid storage, the batch systems, the CMSSW sandbox
# --------------------------------------------------------------------------------------------


class Entry:
    """The fields of a gfal listing entry that law_gfal reads."""

    def __init__(self, name, is_dir=False):
        self.name = name
        self.size = 0
        self.is_dir = is_dir


class FakeStorage:
    """What the storage endpoint answers to `gfal-ls`, keyed by directory URI. Only the listing
    is faked; the path caches above it are law_gfal's own."""

    def __init__(self):
        self.dirs = {BASE: set()}
        self.listings = []

    def add(self, uri):
        while uri != BASE:
            parent, _, name = uri.rpartition("/")
            self.dirs.setdefault(parent, set()).add(name)
            uri = parent

    def ls(self, uri, **kwargs):
        self.listings.append(uri)
        if uri not in self.dirs:
            return None
        return [
            Entry(name, is_dir=f"{uri}/{name}" in self.dirs)
            for name in sorted(self.dirs[uri])
        ]


def out_path(version, family, era, branch):
    """Where branch `branch` (its era's own number) of a test stage writes its output."""
    return os.path.join(root_dir(), "out", version, family, era, f"{branch}.txt")


WRAPPER_ARG_RE = re.compile(r"\['(\d+)'\]=\"([^\"]*)\"")


def wrapper_jobs(wrapper_path):
    """The per-job argument lines law rendered into a job wrapper, in job order."""
    with open(wrapper_path) as f:
        return [args for _, args in WRAPPER_ARG_RE.findall(f.read())]


def decode_job_args(args):
    """(module, family, task parameters as a dict, branches) of one law job argument line."""
    parts = args.split(" ")
    params = base64.b64decode(parts[2]).decode()
    branches = [int(b) for b in base64.b64decode(parts[3]).decode().split()]
    return parts[0], parts[1], params, param_values(params), branches


def param_values(params):
    """The task parameters of a job command line: law's ProxyCommand writes --key=value."""
    return OrderedDict(token.split("=", 1) for token in shlex.split(params))


class Batch:
    """The batch system both backends submit to: one state file shared by every driver
    process, so the jobs one driver submits keep running for the next. A job finishes on the
    `finish_after`-th status query and writes its branches' outputs on the query before that.
    """

    def __init__(self, root=None):
        self.root = root or root_dir()
        self.path = os.path.join(self.root, "batch.json")

    @contextlib.contextmanager
    def state(self):
        with open(self.path + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                st = {"n": 0, "jobs": {}}
                if os.path.exists(self.path):
                    with open(self.path) as f:
                        st = json.load(f)
                yield st
                tmp = f"{self.path}.tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(st, f, indent=1)
                os.replace(tmp, self.path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def jobs(self):
        with self.state() as st:
            return dict(st["jobs"])

    def submit(self, backend, arg_lines, key=None):
        """Register one job per argument line; returns the batch's job keys."""
        finish_after = int(os.environ.get("ERA_JOBS_FINISH_AFTER", "1"))
        keys = []
        with self.state() as st:
            st["n"] += 1
            cluster = st["n"]
            for i, args in enumerate(arg_lines):
                _, family, _, values, branches = decode_job_args(args)
                job_key = key(cluster, i) if key else f"{cluster}.{i}"
                st["jobs"][job_key] = {
                    "backend": backend,
                    "family": family,
                    "version": values["--version"],
                    "era": values["--era"],
                    "branches": branches,
                    "polls": 0,
                    "finish_after": finish_after,
                    "pid": os.getpid(),
                }
                keys.append(job_key)
        return keys

    def query(self, job_key):
        """'pending', 'running', 'finished' or 'failed', advancing the job by one query."""
        with self.state() as st:
            job = st["jobs"].get(job_key)
            if job is None:
                return None
            job["polls"] += 1
            if job["polls"] >= job["finish_after"] - 1 and not job.get("wrote"):
                for b in job["branches"]:
                    path = out_path(job["version"], job["family"], job["era"], b)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as f:
                        f.write(f"job {job_key}\n")
                job["wrote"] = True
            if job["polls"] >= job["finish_after"]:
                return "finished"
            return "running" if job["polls"] > 1 else "pending"

    def forget(self, version):
        """Drop the jobs of `version`, which a test left in flight: a later test reads every
        job of the batch as one of its own."""
        with self.state() as st:
            st["jobs"] = {
                key: job for key, job in st["jobs"].items() if job["version"] != version
            }

    def submissions(self, version):
        """(era, local branch) -> how often it was submitted, over every driver of `version`."""
        counts = Counter()
        for job in self.jobs().values():
            if job["version"] == version:
                for b in job["branches"]:
                    counts[(job["era"], b)] += 1
        return counts


class FakeCondorManager(law.htcondor.HTCondorJobManager):
    """law's HTCondor job manager with condor_submit / condor_q replaced by `Batch`.

    `submit` reads what the job will run from the files law wrote: the JDL names the wrapper,
    and the wrapper carries one argument line per job."""

    job_grouping_submit = True
    chunk_size_query = 0

    def submit(self, job_file, job_files=None, **kwargs):
        with open(job_file) as f:
            jdl = f.read()
        if "arguments from (" in jdl:
            # a group: the arguments are rendered into the wrapper
            executable = re.search(r"^executable\s*=\s*(\S+)\s*$", jdl, re.M).group(1)
            if not os.path.isabs(executable):
                executable = os.path.join(os.path.dirname(job_file), executable)
            lines = wrapper_jobs(executable)
        else:
            lines = [
                args.strip().strip('"')
                for args in re.findall(r"^arguments\s*=\s*(.*)$", jdl, re.M)
            ]
        ids = Batch().submit("htcondor", lines)
        # law's own batched submission of one file answers with its one id
        return ids if job_files is not None else ids[0]

    def query(self, job_id, **kwargs):
        batch = Batch()
        status_map = {
            "pending": self.PENDING,
            "running": self.RUNNING,
            "finished": self.FINISHED,
            None: self.FAILED,
        }

        def one(_id):
            status = status_map[batch.query(_id)]
            return self.job_status_dict(
                job_id=_id, status=status, code=0 if status == self.FINISHED else None
            )

        if isinstance(job_id, (list, tuple)):
            return {_id: one(_id) for _id in job_id}
        return one(job_id)

    def cancel(self, *args, **kwargs):
        return {}

    def cleanup(self, *args, **kwargs):
        return {}


CRAB_STATES = {"pending": "idle", "running": "running", "finished": "finished"}


def fake_crab(cmd, *args, **kwargs):
    """The `crab` executable over `Batch`: `submit` reads the config law and FLAF wrote and
    creates the project directory with its crab.log; `status` answers from the batch state.
    """
    argv = shlex.split(cmd)
    verb = argv[1]
    if verb == "submit":
        cwd = kwargs.get("cwd") or os.getcwd()
        config_file = os.path.join(cwd, argv[argv.index("--config") + 1])
        with open(config_file) as f:
            text = f.read()

        def value(name):
            return re.search(rf"\.{name}\s*=\s*['\"]([^'\"]+)['\"]", text).group(1)

        request, work_area = value("requestName"), value("workArea")
        script = os.path.join(cwd, value("scriptExe"))
        proj_dir = os.path.join(work_area, f"crab_{request}")
        task_name = f"261018_120000:tester_crab_{request}"
        lines = wrapper_jobs(script)
        os.makedirs(proj_dir, exist_ok=True)
        Batch().submit("crab", lines, key=lambda cluster, i: f"{proj_dir}#{i + 1}")
        log = os.path.join(proj_dir, "crab.log")
        with open(log, "w") as f:
            f.write(f"config.Data.totalUnits = {len(lines)}\n")
            f.write("config.JobType.disableAutomaticOutputCollection = True\n")
            f.write(f"DEBUG 2026-10-18 Task name : {task_name}\n")
        return 0, f"Task name : {task_name}\nLog file is {log}\n", ""
    if verb == "status":
        proj_dir = argv[argv.index("--dir") + 1]
        batch = Batch()
        n_jobs = 0
        with open(os.path.join(proj_dir, "crab.log")) as f:
            n_jobs = int(re.search(r"totalUnits = (\d+)", f.read()).group(1))
        states = {
            str(i): {"State": CRAB_STATES[batch.query(f"{proj_dir}#{i}")]}
            for i in range(1, n_jobs + 1)
        }
        out = (
            f"CRAB project directory:\t\t{proj_dir}\n"
            "Status on the CRAB server:\tSUBMITTED\n"
            "Status on the scheduler:\tSUBMITTED\n"
            f"{json.dumps(states)}\n"
        )
        return 0, out, ""
    raise AssertionError(f"unexpected crab command {cmd}")


def no_real_batch_system(*args, **kwargs):
    raise AssertionError(f"a real batch-system command was about to run: {args}")


def batch_patches(storage=None):
    """Everything a driver needs faked, as patchers to enter.

    The HTCondor job manager of every workflow is the fake, and the command runner of law's
    real one raises: no test may ever reach a real batch system."""
    storage = storage or FakeStorage()
    return [
        mock.patch.object(
            law.htcondor.HTCondorWorkflow,
            "htcondor_create_job_manager",
            lambda task, **kwargs: FakeCondorManager(**kwargs),
        ),
        mock.patch(
            "law.contrib.htcondor.job.interruptable_popen",
            side_effect=no_real_batch_system,
        ),
        mock.patch(
            "law.contrib.htcondor.job.get_htcondor_version", return_value=(10, 0, 0)
        ),
        mock.patch.multiple(
            law_gfal,
            gfal_ls_checked=storage.ls,
            get_voms_proxy_info=lambda: {"path": None},
        ),
        mock.patch.object(crab_watchdog, "gfal_ls_safe", return_value=None),
        mock.patch.object(lc, "update_kinit"),
        mock.patch.object(cms_job, "interruptable_popen", side_effect=fake_crab),
        mock.patch.object(
            law.cms.CrabJobManager,
            "cmssw_env",
            property(lambda manager: {"PATH": os.environ.get("PATH", "")}),
        ),
        mock.patch.object(
            lc._FLAFCrabWorkflowProxy, "setup_job_manager", return_value={}
        ),
        mock.patch.object(lc.FLAFCrabJobManager, "query_retry_delay", 0.0),
        mock.patch.object(tempfile, "tempdir", os.path.join(root_dir(), "tmp")),
    ]


# --------------------------------------------------------------------------------------------
# FLAF stages, built like the production ones: a dynamic workflow condition on a local gate
# workflow, branch maps through cached_branch_map, era branches
# --------------------------------------------------------------------------------------------


def n_branches_path(version):
    return os.path.join(root_dir(), "n", f"{version}.json")


def set_n(version, **n):
    """How many branches each era's map has (what an era's input lists would say)."""
    path = n_branches_path(version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    current = {}
    if os.path.exists(path):
        with open(path) as f:
            current = json.load(f)
    current.update(n)
    with open(path, "w") as f:
        json.dump(current, f)


def gate_path(version, era):
    return os.path.join(root_dir(), "gate", version, f"{era}.json")


class EraJobsGate(lc.Task, law.LocalWorkflow):
    """The local workflow whose outputs gate the maps downstream, as InputFileTask does."""

    allow_multi_era = True
    era_branches = True

    def create_branch_map(self):
        return self.cached_branch_map(lambda: {0: self.only_era})

    def output(self):
        return law.LocalFileTarget(gate_path(self.version, self.only_era))

    def run(self):
        with open(n_branches_path(self.version)) as f:
            n = json.load(f)[self.only_era]
        self.output().dump({"n": n}, formatter="json")


class _JobFileDirs:
    """law's job-file factories write under the module's scratch area."""

    @property
    def htcondor_job_file_factory_defaults(self):
        return {"dir": os.environ["LAW_JOB_FILE_DIR"]}

    @property
    def crab_job_file_factory_defaults(self):
        return {"dir": os.environ["LAW_JOB_FILE_DIR"]}


class EraJobsUp(
    _JobFileDirs, lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow
):
    """A remote FLAF stage over the gate, one branch per input of its era."""

    allow_multi_era = True
    era_branches = True
    poll_interval = lc.copy_param(lc.HTCondorWorkflow.poll_interval, 0.0002)
    # no bundle is configured under this name: the scenario drivers need none built
    bundle_flavours = ["scenario"]

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return EraJobsGate.req(self, branch=-1, branches=()).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        with open(gate_path(self.version, self.only_era)) as f:
            n = json.load(f)["n"]
        return {b: f"input_{b}" for b in range(n)}

    def stage_workflow_requires(self):
        reqs = super().stage_workflow_requires()
        reqs["gate"] = EraJobsGate.req(self, branches=())
        return reqs

    def requires(self):
        return EraJobsGate.req(self, branch=0)

    @workflow_condition.output
    def output(self):
        return law.LocalFileTarget(
            out_path(self.version, self.task_family, self.only_era, self.branch)
        )

    def run(self):
        self.output().dump(f"local {self.branch}", formatter="text")

    def _killed_after_polls(self):
        n = int(os.environ.get("ERA_JOBS_KILL_AFTER", "0") or 0)
        self._polls = getattr(self, "_polls", 0) + 1
        if n and self._polls >= n:
            sys.stdout.flush()
            os._exit(3)

    def htcondor_poll_callback(self, poll_data):
        self._killed_after_polls()
        return super().htcondor_poll_callback(poll_data)

    def crab_poll_callback(self, poll_data):
        self._killed_after_polls()
        return super().crab_poll_callback(poll_data)


class EraJobsSrc(lc.Task, law.LocalWorkflow):
    """What EraJobsRenderUp's branches read: one file per branch on the grid storage."""

    allow_multi_era = True
    era_branches = True

    def create_branch_map(self):
        return self.cached_branch_map(lambda: {b: b for b in range(4)})

    def output(self):
        return self.remote_target(
            self.version, "src", self.only_era, f"src_{self.branch}.root"
        )

    def run(self):
        raise AssertionError("the inputs are on the storage already")


class EraJobsRenderUp(EraJobsUp):
    """EraJobsUp with what makes a job era-specific: a configured bundle, requirements on
    the grid storage (the staged path cache), and an input that the job-file factory shares
    between the jobs it builds, written with era-specific content to an era-less path.
    """

    bundle_flavours = ["core"]
    # not spooled: the worker files are shipped through the storage, under the era's path
    htcondor_spool = lc.copy_param(lc.HTCondorWorkflow.htcondor_spool, False)

    def requires(self):
        return {
            "gate": EraJobsGate.req(self, branch=0),
            "src": EraJobsSrc.req(self, branch=self.branch),
        }

    def _era_note(self, config):
        from law.job.base import JobInputFile

        path = os.path.join(os.environ["ANALYSIS_DATA_PATH"], "era_note.txt")
        with open(path, "w") as f:
            f.write(self.only_era)
        config.input_files["era_note"] = JobInputFile(
            path=path, copy=True, share=True, render=False
        )

    def htcondor_job_config(self, config, job_num, branches):
        config = super().htcondor_job_config(config, job_num, branches)
        self._era_note(config)
        return config

    def crab_job_config(self, config, job_nums, branches=None):
        config = super().crab_job_config(config, job_nums, branches)
        self._era_note(config)
        return config


class EraJobsCostUp(EraJobsUp):
    """EraJobsUp that packs by cost and escalates retries, as AnaTupleFileTask does."""

    def branch_cost_map(self):
        return {b: (600.0, "job") for b in self.branch_map}

    def cost_params(self):
        return {
            "retry_runtime_factor": 1.5,
            "retry_memory_factor": 1.25,
            "retry_max_factor": 4.0,
            "request_memory_mb": 2000,
            "default_file_seconds": 600.0,
            "max_units_per_job": 1,
            "tier_safety": {"job": 1.0},
            "parallel_jobs": 0,
        }

    def cost_capacity_seconds(self):
        return 3600.0


#: the estimated seconds of each branch of EraJobsCostPackUp, by era
PACK_COSTS = {
    A: (3000.0, 400.0, 400.0, 3000.0, 300.0, 200.0),
    B: (100.0, 3500.0, 100.0),
}


class EraJobsCostPackUp(EraJobsCostUp):
    """EraJobsCostUp whose branches cost what an era's inputs would: unequal within an era and
    different between eras, so that packing by cost groups them otherwise than chunking.
    """

    def branch_cost_map(self):
        costs = PACK_COSTS[self.only_era]
        return {b: (costs[b], "job") for b in self.branch_map}

    def cost_params(self):
        return dict(super().cost_params(), max_units_per_job=4)


class EraJobsTokenUp(_JobFileDirs, lc.Task, lc.HTCondorWorkflow, law.LocalWorkflow):
    """A remote stage without requirements, to watch luigi schedule its instances."""

    allow_multi_era = True
    era_branches = True

    def create_branch_map(self):
        return self.cached_branch_map(lambda: {0: 0, 1: 1})

    def output(self):
        return law.LocalFileTarget(
            out_path(self.version, self.task_family, self.only_era, self.branch)
        )

    def run(self):
        raise AssertionError("only the workflow runs here, and its run is stubbed")


class EraJobsTokenCache(EraJobsTokenUp):
    """Like AnalysisCacheTask: one job-file directory per producer."""

    producer_to_run = luigi.Parameter()

    def htcondor_output_directory(self):
        return law.LocalDirectoryTarget(self.local_path(self.producer_to_run))

    def output(self):
        return law.LocalFileTarget(
            out_path(
                self.version,
                f"{self.task_family}_{self.producer_to_run}",
                self.only_era,
                self.branch,
            )
        )


class EraCardsInput(lc.Task, law.LocalWorkflow):
    """What EraCardsUp's cards read: a local workflow over the same eras without era branches."""

    allow_multi_era = True

    def create_branch_map(self):
        return {0: "input"}

    def store_parts(self):
        return (self.version, self.__class__.__name__)

    def output(self):
        return law.LocalFileTarget(
            out_path(self.version, self.task_family, lc.era_csv(self.era), self.branch)
        )

    def run(self):
        self.output().dump("input", formatter="text")


class EraCardsUp(_JobFileDirs, lc.Task, lc.HTCondorWorkflow, law.LocalWorkflow):
    """Shaped like StatInference's card stages: a remote workflow over several eras without era
    branches (one branch per card a configuration lists, each reading several eras), whose job
    state and logs are keyed on the version alone. As a card needs all of its eras, the map is
    refused over fewer, so nothing may build it on a single-era instance."""

    allow_multi_era = True
    poll_interval = lc.copy_param(lc.HTCondorWorkflow.poll_interval, 0.0002)

    def create_branch_map(self):
        missing = {A, B} - set(self.era)
        if missing:
            raise ValueError(f"the cards need {', '.join(sorted(missing))} in --era")
        return {0: "card_0", 1: "card_1"}

    def stage_workflow_requires(self):
        reqs = super().stage_workflow_requires()
        reqs["cards_input"] = EraCardsInput.req(self, branches=())
        return reqs

    def requires(self):
        return EraCardsInput.req(self, branch=0)

    def store_parts(self):
        return (self.version, self.__class__.__name__)

    def remote_log_dir_target(self):
        return self.remote_dir_target(self.version, "logs", self.__class__.__name__)

    def output(self):
        # where the fake batch system writes a job's outputs: under its --era value
        return law.LocalFileTarget(
            out_path(self.version, self.task_family, lc.era_csv(self.era), self.branch)
        )

    def run(self):
        raise AssertionError("the jobs run on the fake batch system")


# --------------------------------------------------------------------------------------------
# helpers: one process per driver, as luigi has it
# --------------------------------------------------------------------------------------------


def fresh_process():
    """Forget what a driver process keeps between runs: luigi's task instances, the parsed
    command line, the branch maps, the Setups (and the path caches of their file systems).
    """
    luigi.task_register.Register.clear_instance_cache()
    law.parser._reset()
    lc.Task._req_prefer_cli_drop_cache.clear()
    lc.Task._branch_map_cache.clear()
    ana = os.environ.get("ANALYSIS_PATH")
    for key in [k for k in Setup._global_instances if k[0] == ana]:
        del Setup._global_instances[key]
    lc.BundleTask._source_hash_cache.clear()
    data = os.environ.get("ANALYSIS_DATA_PATH") or "/nonexistent"
    for key in [k for k in SiteStats._shared if str(k).startswith(data)]:
        SiteStats._shared.pop(key, None)
    GFALFileInterface.negatives_valid_after = 0.0
    lc._era_cli_checked = False


def write_gates(version, **n):
    """The gate's outputs, as if it had run for these eras."""
    set_n(version, **n)
    for era, count in n.items():
        path = gate_path(version, era)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"n": count}, f)


def data_path(*parts):
    return os.path.join(os.environ["ANALYSIS_DATA_PATH"], *parts)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def job_file_names(version, family="EraJobsUp", kind=None):
    """era -> sorted job-file names under the single-era output directories."""
    out = {}
    for path in sorted(glob.glob(data_path(version, family, "*", "*_jobs*.json"))):
        if kind and not os.path.basename(path).startswith(kind):
            continue
        out.setdefault(os.path.basename(os.path.dirname(path)), []).append(
            os.path.basename(path)
        )
    return out


def outputs_of(version, family="EraJobsUp"):
    base = os.path.join(root_dir(), "out", version, family)
    return sorted(
        os.path.relpath(p, base) for p in glob.glob(os.path.join(base, "*", "*.txt"))
    )


def wipe(version):
    """Remove everything a run of `version` left on disk (job files, outputs)."""
    for path in (data_path(version), os.path.join(root_dir(), "out", version)):
        shutil.rmtree(path, ignore_errors=True)


LOG_ID_RE = re.compile(r"_\d+\.\d+\.txt$")


def stored_batch_key(job_id):
    """The fake batch system's key of a job id as a job file stores it: an HTCondor id is the
    key, a CRAB JobId is written as [crab_num, task_name, proj_dir]."""
    if isinstance(job_id, list):
        crab_num, _, proj_dir = job_id
        return f"{proj_dir}#{crab_num}"
    return job_id


def normalised_job_file(path, version):
    """An era's job file with what differs between two runs of the same work replaced: each
    job id, and the id in the name of a job's log, by the era and branches the batch system
    runs under that id; the version by a placeholder."""
    batch = Batch().jobs()

    def submitted(key):
        job = batch[key]
        return f"<{job['era']}:{job['branches']}>"

    data = load_json(path)
    for job in data["jobs"].values():
        if job["job_id"] != JobData.dummy_job_id:
            job["job_id"] = submitted(stored_batch_key(job["job_id"]))
        log = (job.get("extra") or {}).get("log")
        if log:
            match = re.search(r"_(\d+\.\d+)\.txt$", log)
            if match:
                log = f"{log[: match.start()]}_{submitted(match.group(1))}.txt"
            job["extra"]["log"] = log
    return json.loads(json.dumps(data).replace(version, "<version>"))


class Render:
    """What each job file a run builds carries, recorded at the job-file factory: the
    factory is called with everything FLAF's job-config hooks decided."""

    def __init__(self):
        self.files = []

    @staticmethod
    def _summary(kwargs, c):
        jobs = []
        for args in c.arguments:
            _, family, params, values, branches = decode_job_args(args)
            jobs.append({"family": family, "params": params, "branches": branches})
        rv = kwargs.get("render_variables") or {}
        summary = {
            "eras": sorted({decode_job_args(a)[3]["--era"] for a in c.arguments}),
            "jobs": jobs,
            "bundle_list": rv.get("bundle_list"),
            "log_remote_base_url": rv.get("log_remote_base_url"),
            "analysis_path": rv.get("analysis_path"),
            "run_token_server_host": rv.get("run_token_server_host"),
        }
        worker_files = kwargs.get("_worker_files_remote_dir")
        if worker_files is not None:
            summary["worker_files"] = worker_files.uri()
        if "custom_content" in kwargs and kwargs["custom_content"] is not None:
            summary["custom"] = {
                key: value
                for key, value in (tuple(item) for item in kwargs["custom_content"])
                if key in ("+MaxRuntime", "RequestCpus", "RequestMemory")
            }
        crab = kwargs.get("crab")
        if crab is not None:
            summary["crab"] = {
                "numCores": crab.JobType.numCores,
                "maxMemoryMB": crab.JobType.maxMemoryMB,
                "maxJobRuntimeMin": crab.JobType.maxJobRuntimeMin,
                "request_prefix": c.request_name.rsplit("_", 1)[0],
            }
        files = c.input_files
        if "path_cache" in files:
            with gzip.open(files["path_cache"].path_sub_abs, "rt") as f:
                summary["path_cache"] = sorted(
                    (e["path"], e["exists"]) for e in json.load(f)["entries"]
                )
        if "era_note" in files:
            with open(files["era_note"].path_sub_abs) as f:
                summary["era_note"] = f.read()
        return summary

    @contextlib.contextmanager
    def recording(self):
        def spy(original):
            def create(factory, *args, **kwargs):
                job_file, c = original(factory, *args, **kwargs)
                self.files.append(self._summary(kwargs, c))
                return job_file, c

            return create

        def worker_files(original):
            # the upload to the grid storage is all that is left out
            def stage(job_file, worker_files_dir=None):
                return original(job_file, None)

            return stage

        patches = [
            mock.patch.object(
                cls,
                "create",
                autospec=True,
                side_effect=spy(cls.create),
            )
            for cls in (lc.CERNHTCondorJobFileFactory, lc.FLAFCrabJobFileFactory)
        ] + [
            mock.patch.object(
                lc.CERNHTCondorJobFileFactory,
                "_stage_and_update_jdl",
                staticmethod(
                    worker_files(lc.CERNHTCondorJobFileFactory._stage_and_update_jdl)
                ),
            )
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            yield self

    def by_era(self):
        out = {}
        for summary in self.files:
            (era,) = summary["eras"]
            out.setdefault(era, []).append(summary)
        return out


class _Case(unittest.TestCase):
    """A fresh driver process over the module's scratch analysis, with the batch systems, the
    storage listing, the sandbox and the credentials faked."""

    backend = "htcondor"

    def setUp(self):
        self.version = f"v{os.getpid()}_{time.monotonic_ns() % 10**9}"
        self.storage = FakeStorage()
        fresh_process()
        self.addCleanup(fresh_process)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for patch in batch_patches(self.storage):
            stack.enter_context(patch)
        stack.enter_context(
            mock.patch.object(GFALFileInterface, "negatives_valid_after", 0.0)
        )
        stack.enter_context(mock.patch.object(lc, "_era_cli_checked", False))
        # each driver run here is a process of its own: none shares a path-cache server
        stack.enter_context(
            mock.patch.object(setup_module, "_local_path_cache_server", [None])
        )
        self.addCleanup(wipe, self.version)

    @contextlib.contextmanager
    def cli(self, eras, *args, family="EraJobsUp", workflow=None):
        """The root task of `law run <family> --era <eras> ...`, inside its command line."""
        argv = [
            family,
            "--version",
            self.version,
            "--era",
            ",".join(eras),
            "--workflow",
            workflow or self.backend,
        ] + [str(a) for a in args]
        with luigi.cmdline_parser.CmdlineParser.global_instance(argv) as cp:
            yield cp.get_task_obj()

    def run_root(self, task):
        """What luigi's worker does with the root workflow: run() it."""
        task.scheduler_messages = None
        return task.run()

    def drive(self, eras, *args, family="EraJobsUp", workflow=None):
        """One driver process: a fresh process, the command line, the run."""
        fresh_process()
        with self.cli(eras, *args, family=family, workflow=workflow) as task:
            self.run_root(task)
            return task

    def jobs_file(self, era, name, family="EraJobsUp"):
        return data_path(self.version, family, era, name)

    def kind(self):
        return f"{self.backend}_jobs"

    def submissions(self):
        return Batch().submissions(self.version)

    def batch_keys(self):
        """(era, local branches) -> the key of the batch job submitted for them."""
        return {
            (job["era"], tuple(job["branches"])): key
            for key, job in Batch().jobs().items()
            if job["version"] == self.version
        }


# --------------------------------------------------------------------------------------------
# I4: each era's job file is the single-era file
# --------------------------------------------------------------------------------------------


class _EachErasFileIsTheSingleEraFile:
    def test_each_era_file_sits_at_the_single_era_path_and_name(self):
        write_gates(self.version, **{A: 3, B: 2})
        with self.cli((A, B), "--tasks-per-job", 2) as root:
            files = root.workflow_proxy.output()["jobs"]
            self.assertIsInstance(files, lc.EraJobFiles)
            merged = {era: t.path for era, t in files.targets().items()}
        singles = {}
        for era in (A, B):
            fresh_process()
            with self.cli((era,), "--tasks-per-job", 2) as single:
                singles[era] = single.workflow_proxy.output()["jobs"].path
        self.assertEqual(merged, singles)
        self.assertEqual(
            merged,
            {
                A: self.jobs_file(A, f"{self.kind()}_0To3.json"),
                B: self.jobs_file(B, f"{self.kind()}_0To2.json"),
            },
        )

    def test_a_run_writes_each_era_in_its_own_numbers_as_the_single_era_proxy_does(
        self,
    ):
        write_gates(self.version, **{A: 3, B: 2})
        self.drive((A, B), "--tasks-per-job", 2)
        written = {
            A: self.jobs_file(A, f"{self.kind()}_0To3.json"),
            B: self.jobs_file(B, f"{self.kind()}_0To2.json"),
        }
        for era, path in written.items():
            with self.subTest(era=era):
                raw = read_bytes(path)
                data = json.loads(raw)
                # local job numbers, local branches, the era's tasks_per_job
                self.assertEqual(data["tasks_per_job"], 2)
                self.assertTrue(all(int(n) < STRIDE for n in data["jobs"]))
                branches = sorted(
                    b for j in data["jobs"].values() for b in j["branches"]
                )
                self.assertEqual(branches, list(range(3 if era == A else 2)))
                self.assertEqual(
                    sorted(data["jobs"]), ["1", "2"] if era == A else ["1"]
                )
                # the single-era proxy dumps the same job data to the same bytes
                fresh_process()
                with self.cli((era,), "--tasks-per-job", 2) as single:
                    proxy = single.workflow_proxy
                    proxy.dashboard = NoJobDashboard()
                    proxy.job_data.update(json.loads(raw))
                    os.remove(path)
                    proxy.dump_job_data()
                self.assertEqual(read_bytes(path), raw)

    def test_a_fresh_run_writes_each_era_as_the_run_over_that_era_alone(self):
        """From the same fresh state, each era's file of a run over several eras holds what
        the run over that era alone writes -- jobs, ids, statuses, logs and every other field
        -- with its jobs in flight and once they finished."""
        base = self.version
        try:
            for state, flags in (("in_flight", ("--no-poll",)), ("finished", ())):
                written = {}
                for variant, runs in (("together", [(A, B)]), ("alone", [(A,), (B,)])):
                    self.version = f"{base}_{state}_{variant}"
                    self.addCleanup(wipe, self.version)
                    self.addCleanup(Batch().forget, self.version)
                    write_gates(self.version, **{A: 3, B: 2})
                    with in_flight_jobs():
                        for eras in runs:
                            self.drive(eras, "--tasks-per-job", 2, *flags)
                    written[variant] = {
                        era: [
                            (
                                name,
                                normalised_job_file(
                                    self.jobs_file(era, name), self.version
                                ),
                            )
                            for name in names
                        ]
                        for era, names in job_file_names(self.version).items()
                    }
                with self.subTest(state=state):
                    self.assertEqual(sorted(written["together"]), [A, B])
                    self.assertEqual(written["together"], written["alone"])
                    # the ids were compared: in flight every job carries one
                    ids = [
                        job["job_id"]
                        for files in written["together"].values()
                        for _, data in files
                        for job in data["jobs"].values()
                    ]
                    self.assertEqual(len(ids), 3)
                    if state == "in_flight":
                        self.assertNotIn(JobData.dummy_job_id, ids)
        finally:
            self.version = base

    def test_load_merges_the_era_files_with_the_era_id_shift(self):
        write_gates(self.version, **{A: 3, B: 2})
        self.drive((A, B), "--tasks-per-job", 2)
        fresh_process()
        with self.cli((A, B), "--tasks-per-job", 3) as root:
            files = root.workflow_proxy.output()["jobs"]
            data = files.load(formatter="json")
        a, b = ERA_IDS[A] * STRIDE, ERA_IDS[B] * STRIDE
        self.assertEqual(sorted(data["jobs"]), [a + 1, a + 2, b + 1])
        self.assertEqual(
            {n: j["branches"] for n, j in data["jobs"].items()},
            {a + 1: [a, a + 1], a + 2: [a + 2], b + 1: [b, b + 1]},
        )
        self.assertEqual(files.tasks_per_job, {A: 2, B: 2})
        # the merged count law sees is the root's; each file keeps its own
        self.assertEqual(data["tasks_per_job"], 3)

    def test_exists_parent_touch_and_remove_cover_every_era(self):
        write_gates(self.version, **{A: 3, B: 2})
        with self.cli((A, B)) as root:
            files = root.workflow_proxy.output()["jobs"]
            targets = files.targets()
            self.assertFalse(files.exists())
            files.parent.touch()
            for target in targets.values():
                self.assertTrue(os.path.isdir(os.path.dirname(target.path)))
            # one era's file is enough for law to resume
            with open(targets[B].path, "w") as f:
                json.dump({"jobs": {}}, f)
            self.assertTrue(files.exists())
            with open(targets[A].path, "w") as f:
                json.dump({"jobs": {}}, f)
            files.remove()
            self.assertFalse(any(os.path.exists(t.path) for t in targets.values()))

    def test_laws_output_removal_removes_every_eras_file(self):
        # `--remove-output 0,a`: law's own removal, which passes keyword arguments
        write_gates(self.version, **{A: 3, B: 2})
        with self.cli((A, B)) as root:
            targets = root.workflow_proxy.output()["jobs"].targets()
            for target in targets.values():
                target.parent.touch()
                with open(target.path, "w") as f:
                    json.dump({"jobs": {}}, f)
            law.task.interactive.remove_task_output(root, "0", mode="a")
            self.assertFalse(any(os.path.exists(t.path) for t in targets.values()))

    def test_a_local_job_number_beyond_the_stride_is_refused(self):
        write_gates(self.version, **{A: 3, B: 2})
        with self.cli((A, B)) as root:
            files = root.workflow_proxy.output()["jobs"]
            target = files.targets()[A]
            target.parent.touch()
            data = JobData(tasks_per_job=1)
            data.jobs[STRIDE] = JobData.job_data(branches=[0])
            target.dump(data, formatter="json")
            with self.assertRaises(RuntimeError) as caught:
                files.load(formatter="json")
        self.assertIn(f"job {STRIDE} is outside", str(caught.exception))

    def test_growing_one_era_renames_that_eras_file_only(self):
        write_gates(self.version, **{A: 3, B: 2})
        self.drive((A, B))
        before = job_file_names(self.version)
        self.assertEqual(
            before,
            {A: [f"{self.kind()}_0To3.json"], B: [f"{self.kind()}_0To2.json"]},
        )
        b_file = self.jobs_file(B, f"{self.kind()}_0To2.json")
        b_bytes, b_mtime = read_bytes(b_file), os.stat(b_file).st_mtime_ns
        write_gates(self.version, **{A: 4})
        self.drive((A, B))
        self.assertEqual(
            job_file_names(self.version),
            {
                A: [f"{self.kind()}_0To3.json", f"{self.kind()}_0To4.json"],
                B: [f"{self.kind()}_0To2.json"],
            },
        )
        # B was complete: left out, its file untouched (S5)
        self.assertEqual(read_bytes(b_file), b_bytes)
        self.assertEqual(os.stat(b_file).st_mtime_ns, b_mtime)
        # only A's new branch was submitted
        self.assertEqual(
            self.submissions(),
            Counter({(A, 0): 1, (A, 1): 1, (A, 2): 1, (A, 3): 1, (B, 0): 1, (B, 1): 1}),
        )


class HTCondorEachErasFileIsTheSingleEraFile(_EachErasFileIsTheSingleEraFile, _Case):
    backend = "htcondor"


class CrabEachErasFileIsTheSingleEraFile(_EachErasFileIsTheSingleEraFile, _Case):
    backend = "crab"


# --------------------------------------------------------------------------------------------
# the per-era layout, through the real proxies
# --------------------------------------------------------------------------------------------


@contextlib.contextmanager
def in_flight_jobs(finish_after=3):
    """Jobs submitted meanwhile finish only on their `finish_after`-th status query."""
    with mock.patch.dict(os.environ, {"ERA_JOBS_FINISH_AFTER": str(finish_after)}):
        yield


@contextlib.contextmanager
def poll_entries(record):
    """Record law's job count, and the jobs it counts, as its poll() starts."""
    original = BaseRemoteWorkflowProxy.poll

    def poll(proxy):
        record.append(
            (
                len(proxy.job_data),
                sorted(proxy.job_data.jobs),
                sorted(proxy.job_data.unsubmitted_jobs),
            )
        )
        return original(proxy)

    with mock.patch.object(
        BaseRemoteWorkflowProxy, "poll", autospec=True, side_effect=poll
    ):
        yield


class _TheLayout:
    def batch_chunks(self):
        return sorted(
            (job["era"], job["branches"])
            for job in Batch().jobs().values()
            if job["version"] == self.version
        )

    def test_fresh_eras_are_chunked_with_the_roots_tasks_per_job(self):
        write_gates(self.version, **{A: 3, B: 2})
        self.drive((A, B), "--tasks-per-job", 2)
        self.assertEqual(self.batch_chunks(), [(A, [0, 1]), (A, [2]), (B, [0, 1])])

    def test_a_resumed_era_keeps_its_jobs_and_its_tasks_per_job(self):
        write_gates(self.version, **{A: 3, B: 2})
        with in_flight_jobs():
            self.drive((A,), "--no-poll")
        a_file = self.jobs_file(A, f"{self.kind()}_0To3.json")
        first = load_json(a_file)
        self.assertEqual(first["tasks_per_job"], 1)
        in_flight = [j["job_id"] for j in first["jobs"].values()]
        self.assertEqual(len(in_flight), 3)
        self.drive((A, B), "--tasks-per-job", 2)
        # A's in-flight jobs were polled to their end, not submitted again
        self.assertEqual(
            self.submissions(),
            Counter({(A, 0): 1, (A, 1): 1, (A, 2): 1, (B, 0): 1, (B, 1): 1}),
        )
        self.assertTrue(all(job.get("wrote") for job in Batch().jobs().values()))
        after = load_json(a_file)
        self.assertEqual(after["tasks_per_job"], 1)
        self.assertEqual(
            {n: j["branches"] for n, j in after["jobs"].items()},
            {n: j["branches"] for n, j in first["jobs"].items()},
        )
        b = load_json(self.jobs_file(B, f"{self.kind()}_0To2.json"))
        self.assertEqual(b["tasks_per_job"], 2)
        self.assertEqual([j["branches"] for j in b["jobs"].values()], [[0, 1]])

    def test_a_complete_era_is_left_out_and_its_file_untouched(self):
        write_gates(self.version, **{A: 2, B: 2})
        self.drive((A,))
        a_file = self.jobs_file(A, f"{self.kind()}_0To2.json")
        a_bytes, a_mtime = read_bytes(a_file), os.stat(a_file).st_mtime_ns
        time.sleep(0.01)
        record = []
        with poll_entries(record):
            self.drive((A, B))
        b0 = ERA_IDS[B] * STRIDE
        # A resumed from its file and complete: only B, unsubmitted, when law counts
        self.assertEqual(record[0], (2, [], [b0 + 1, b0 + 2]))
        self.assertEqual(read_bytes(a_file), a_bytes)
        self.assertEqual(os.stat(a_file).st_mtime_ns, a_mtime)
        self.assertEqual(
            self.submissions(), Counter({(A, 0): 1, (A, 1): 1, (B, 0): 1, (B, 1): 1})
        )

    def test_the_job_count_is_final_before_laws_poll_takes_it(self):
        """law's poll() takes the job count once, at its start: a resumed run never submits
        before polling, so a fresh era's jobs must be laid out by then."""
        write_gates(self.version, **{A: 3, B: 2})
        with in_flight_jobs():
            self.drive((A,), "--no-poll")
        record = []
        with poll_entries(record):
            self.drive((A, B))
        a0, b0 = ERA_IDS[A] * STRIDE, ERA_IDS[B] * STRIDE
        self.assertEqual(record[0], (5, [a0 + 1, a0 + 2, a0 + 3], [b0 + 1, b0 + 2]))
        self.assertEqual(
            outputs_of(self.version),
            [f"{A}/0.txt", f"{A}/1.txt", f"{A}/2.txt", f"{B}/0.txt", f"{B}/1.txt"],
        )
        b = load_json(self.jobs_file(B, f"{self.kind()}_0To2.json"))
        self.assertEqual(
            sorted(j["status"] for j in b["jobs"].values()), ["finished", "finished"]
        )

    def test_nothing_is_laid_out_before_the_branch_map_exists(self):
        """While law's dynamic workflow condition is unmet -- no era's gate, or one era's
        alone -- the root's map is law's placeholder, which holds no branch of any era: a
        layout from it would hold no job, and the run would end with nothing submitted.
        """
        set_n(self.version, **{A: 2, B: 2})
        for gates in ((), (A,)):
            for era in gates:
                write_gates(self.version, **{era: 2})
            with self.subTest(gates=gates):
                fresh_process()
                with self.cli((A, B)) as root:
                    self.assertFalse(root.workflow_condition())
                    # law's run: a fresh run lays out in submit()
                    with self.assertRaises(RuntimeError) as caught:
                        self.run_root(root)
                    self.assertIn(
                        "jobs cannot be laid out before the branch map exists",
                        str(caught.exception),
                    )
                fresh_process()
                with self.cli((A, B)) as root:
                    # a resumed run lays out in poll()
                    with self.assertRaises(RuntimeError) as caught:
                        root.workflow_proxy.poll()
                    self.assertIn(
                        "jobs cannot be laid out before the branch map exists",
                        str(caught.exception),
                    )
                self.assertEqual(job_file_names(self.version), {})
                self.assertEqual(sum(self.submissions().values()), 0)


class HTCondorTheLayout(_TheLayout, _Case):
    backend = "htcondor"

    def test_a_job_number_beyond_the_stride_is_refused(self):
        """Cost packing numbers new jobs above every number used so far, and a long-lived
        file can climb that high: a number of 10**6 would land in the next era's block.
        """
        write_gates(self.version, **{A: 2, B: 1})
        files = None
        with self.cli((A,), family="EraJobsCostUp") as single:
            files = single.workflow_proxy.output()["jobs"]
        # a finished job under the last number one era can take, one branch left to submit
        data = JobData(tasks_per_job=1)
        data.jobs[STRIDE - 1] = JobData.job_data(branches=[0], status="finished")
        data.unsubmitted_jobs[1] = [1]
        path = out_path(self.version, "EraJobsCostUp", A, 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        files.parent.touch()
        files.dump(data, formatter="json", indent=4)
        with self.assertRaises(RuntimeError) as caught:
            self.drive((A, B), family="EraJobsCostUp")
        self.assertIn(f"job {STRIDE} is outside", str(caught.exception))
        self.assertEqual(sum(self.submissions().values()), 0)


class CrabTheLayout(_TheLayout, _Case):
    backend = "crab"


class EachEraIsPackedByItsOwnCosts(_Case):
    """Cost-aware packing (HTCondor) runs per era, with that era's costs: the jobs of each era
    of a run over several eras are the jobs the run over that era alone packs."""

    family = "EraJobsCostPackUp"

    def test_each_eras_jobs_are_packed_as_in_a_run_over_that_era_alone(self):
        base = self.version
        layouts, submitted = {}, {}
        try:
            for variant, runs in (("together", [(A, B)]), ("alone", [(A,), (B,)])):
                self.version = f"{base}_{variant}"
                self.addCleanup(wipe, self.version)
                write_gates(
                    self.version, **{A: len(PACK_COSTS[A]), B: len(PACK_COSTS[B])}
                )
                for eras in runs:
                    self.drive(eras, family=self.family)
                layouts[variant] = {
                    era: {
                        name: {
                            n: j["branches"]
                            for n, j in load_json(
                                self.jobs_file(era, name, family=self.family)
                            )["jobs"].items()
                        }
                        for name in names
                    }
                    for era, names in job_file_names(
                        self.version, family=self.family
                    ).items()
                }
                submitted[variant] = sorted(
                    (job["era"], job["branches"])
                    for job in Batch().jobs().values()
                    if job["version"] == self.version
                )
        finally:
            self.version = base
        self.assertEqual(layouts["together"], layouts["alone"])
        self.assertEqual(submitted["together"], submitted["alone"])
        # packed by cost, not chunked by --tasks-per-job: fewer jobs than branches in each era
        for era in (A, B):
            with self.subTest(era=era):
                (jobs,) = layouts["together"][era].values()
                self.assertLess(len(jobs), len(PACK_COSTS[era]))
                self.assertEqual(
                    sorted(b for branches in jobs.values() for b in branches),
                    list(range(len(PACK_COSTS[era]))),
                )

    def test_a_resumed_eras_jobs_are_packed_again_as_in_a_run_over_that_era_alone(self):
        """A resumed era's jobs that were never submitted are packed again with the costs and
        settings of this run (an earlier run packed them with others, as when the cost model
        learned in between), as the resumed run over that era alone packs them."""
        base = self.version
        name = f"{self.kind()}_0To{len(PACK_COSTS[A])}.json"
        layouts, submitted, earlier = {}, {}, {}
        try:
            for variant, runs in (("together", [(A, B)]), ("alone", [(A,), (B,)])):
                self.version = f"{base}_resumed_{variant}"
                self.addCleanup(wipe, self.version)
                write_gates(
                    self.version, **{A: len(PACK_COSTS[A]), B: len(PACK_COSTS[B])}
                )
                # the same earlier run over A, packed one branch per job: one job in
                # flight, the others unsubmitted
                with in_flight_jobs(), mock.patch.object(
                    EraJobsCostPackUp, "cost_params", EraJobsCostUp.cost_params
                ):
                    self.drive(
                        (A,), "--no-poll", "--parallel-jobs", 1, family=self.family
                    )
                data = load_json(self.jobs_file(A, name, family=self.family))
                earlier[variant] = {
                    "jobs": {n: j["branches"] for n, j in data["jobs"].items()},
                    "unsubmitted_jobs": data["unsubmitted_jobs"],
                }
                for eras in runs:
                    self.drive(eras, family=self.family)
                layouts[variant] = {
                    era: {
                        file_name: {
                            n: j["branches"]
                            for n, j in load_json(
                                self.jobs_file(era, file_name, family=self.family)
                            )["jobs"].items()
                        }
                        for file_name in names
                    }
                    for era, names in job_file_names(
                        self.version, family=self.family
                    ).items()
                }
                submitted[variant] = sorted(
                    (job["era"], job["branches"])
                    for job in Batch().jobs().values()
                    if job["version"] == self.version
                )
        finally:
            self.version = base
        self.assertEqual(earlier["together"], earlier["alone"])
        self.assertEqual(layouts["together"], layouts["alone"])
        self.assertEqual(submitted["together"], submitted["alone"])
        # discriminating: A's unsubmitted jobs were packed again, into fewer jobs
        before = earlier["together"]
        self.assertEqual(len(before["jobs"]), 1)
        self.assertEqual(len(before["unsubmitted_jobs"]), len(PACK_COSTS[A]) - 1)
        after = layouts["together"][A][name]
        self.assertLess(len(after), len(PACK_COSTS[A]))
        self.assertEqual(
            sorted(b for branches in after.values() for b in branches),
            list(range(len(PACK_COSTS[A]))),
        )


# --------------------------------------------------------------------------------------------
# I8: every job equals the job of a run over its era alone
# --------------------------------------------------------------------------------------------

#: the per-task values law hands between a workflow and its branches, and FLAF excludes from
#: what one task hands another; all non-default
FLAGS = (
    "--tasks-per-job",
    2,
    "--max-runtime",
    48,
    "--n-cpus",
    4,
    "--crab-memory",
    5000,
)


class _EveryJobIsTheSingleEraJob:
    family = "EraJobsRenderUp"
    extra = ()

    def setUp(self):
        super().setUp()
        for era in (A, B):
            for b in range(4):
                self.storage.add(f"{BASE}/{self.version}/src/{era}/src_{b}.root")

    def render(self, eras, *flags):
        wipe(self.version)
        write_gates(self.version, **{A: 3, B: 2})
        render = Render()
        with render.recording():
            task = self.drive(eras, *flags, *self.extra, family=self.family)
        return render, task

    def test_every_job_equals_the_job_of_a_run_over_its_era_alone(self):
        merged, _ = self.render((A, B), *FLAGS)
        by_era = merged.by_era()
        self.assertEqual(sorted(by_era), [A, B])
        for era in (A, B):
            with self.subTest(era=era):
                single, _ = self.render((era,), *FLAGS)
                self.assertEqual(by_era[era], single.by_era()[era])

    def test_the_root_over_several_eras_builds_no_job_itself(self):
        """Every job is built by an era's own workflow; a hook reached on the root instead
        raises rather than build a job for one of its eras."""
        write_gates(self.version, **{A: 3, B: 2})
        a0 = ERA_IDS[A] * STRIDE
        with self.cli((A, B), *FLAGS, *self.extra, family=self.family) as root:
            factory = root.workflow_proxy.create_job_file_factory()
            config = factory.get_config()
            config.input_files = {}
            config.output_files = {} if self.backend == "htcondor" else []
            config.render_variables = {}
            config.custom_content = []
            config.request_name = "request"
            hook = getattr(root, f"{self.backend}_job_config")
            with self.assertRaises(lc.EraScopeError):
                hook(config, [a0 + 1], [[a0, a0 + 1]])

    def test_what_the_jobs_carry(self):
        merged, _ = self.render((A, B), *FLAGS)
        chunks = {
            era: [job["branches"] for s in summaries for job in s["jobs"]]
            for era, summaries in merged.by_era().items()
        }
        self.assertEqual(chunks, {A: [[0, 1], [2]], B: [[0, 1]]})
        for era, summaries in merged.by_era().items():
            for summary in summaries:
                with self.subTest(era=era):
                    for job in summary["jobs"]:
                        values = param_values(job["params"])
                        self.assertEqual(values["--era"], era)
                        removed = "--period"  # noqa: era-rename
                        self.assertNotIn(removed, job["params"])
                        self.assertEqual(values["--max-runtime"], "2-00:00:00")
                        self.assertEqual(values["--n-cpus"], "4")
                    self.assertIn(
                        f"/{self.version}/bundles/{era}/core_", summary["bundle_list"]
                    )
                    self.assertEqual(
                        summary["log_remote_base_url"],
                        f"{BASE}/{self.version}/logs/{self.family}/{era}",
                    )
                    self.assertEqual(summary["era_note"], era)
                    self.check_backend(era, summary)

    def check_snapshot(self, era, summary):
        # the job finds its requirements in the snapshot: those of its era, nothing else
        paths = [path for path, _ in summary["path_cache"]]
        self.assertTrue(paths)
        prefix = f"{BASE}/{self.version}/src/"
        self.assertEqual(
            {p[len(prefix) :].split("/")[0] for p in paths if p.startswith(prefix)},
            {era},
        )
        self.assertTrue(all(p.startswith(prefix) for p in paths), paths)


class HTCondorEveryJobIsTheSingleEraJob(_EveryJobIsTheSingleEraJob, _Case):
    backend = "htcondor"
    # bundles and worker files are shipped only with --bundle and without spooling
    extra = ("--bundle",)

    def check_backend(self, era, summary):
        self.assertEqual(
            summary["custom"], {"+MaxRuntime": 48 * 3600 - 1, "RequestCpus": 4}
        )
        self.assertEqual(
            summary["worker_files"], f"{BASE}/{self.version}/worker_files/{era}"
        )
        self.check_snapshot(era, summary)


class CrabEveryJobIsTheSingleEraJob(_EveryJobIsTheSingleEraJob, _Case):
    backend = "crab"

    def check_backend(self, era, summary):
        self.assertEqual(
            summary["crab"],
            {
                "numCores": 4,
                "maxMemoryMB": 5000,
                "maxJobRuntimeMin": 48 * 60,
                "request_prefix": f"{self.family}_{self.version}_{era}",
            },
        )
        self.check_snapshot(era, summary)

    def test_the_job_as_it_runs_on_a_crab_worker(self):
        """What a CRAB job builds from its command line, with LAW_CRAB_JOB_NUMBER set: the
        branch of its era's own workflow, writing the flag the driver watches."""
        merged, root = self.render((A, B), *FLAGS)
        watched = {
            era: root.job_watchdog().parts[ERA_IDS[era]].flag_dir for era in (A, B)
        }
        logs = {era: s[0]["log_remote_base_url"] for era, s in merged.by_era().items()}
        for era, summaries in merged.by_era().items():
            for job in summaries[0]["jobs"]:
                for branch in job["branches"]:
                    with self.subTest(era=era, branch=branch):
                        fresh_process()
                        argv = (
                            [job["family"]]
                            + shlex.split(job["params"])
                            + ["--branch", str(branch)]
                        )
                        with mock.patch.dict(
                            os.environ, {"LAW_CRAB_JOB_NUMBER": "1"}
                        ), luigi.cmdline_parser.CmdlineParser.global_instance(
                            argv
                        ) as cp:
                            task = cp.get_task_obj()
                            self.assertEqual((task.era, task.branch), ((era,), branch))
                            self.assertEqual(
                                (
                                    task.output()["src"].path
                                    if isinstance(task.output(), dict)
                                    else task.output().path
                                ),
                                out_path(self.version, self.family, era, branch),
                            )
                            self.assertEqual(
                                task.crab_heartbeat().uri, f"{watched[era]}/{branch}"
                            )
                            self.assertEqual(
                                task.remote_log_dir_target().uri(), logs[era]
                            )


# --------------------------------------------------------------------------------------------
# retry escalation, no leak, order, routed dumps
# --------------------------------------------------------------------------------------------


class RetriesEscalateAsInARunOverOneEra(_Case):
    """`_retry_resource_factors` reads the attempts through the workflow's proxy: the era's
    proxy must be the one installed while it builds the job file."""

    family = "EraJobsCostUp"

    def custom(self, render):
        return [s["custom"] for s in render.files]

    def test_a_second_attempt_asks_for_what_it_asks_for_over_one_era(self):
        write_gates(self.version, **{A: 2, B: 2})
        a0 = ERA_IDS[A] * STRIDE
        merged = Render()
        with self.cli((A, B), "--max-runtime", 4, family=self.family) as root:
            proxy = root.workflow_proxy
            proxy.dashboard = NoJobDashboard()
            # law's submit() books the jobs before it hands them to _submit_group
            for job_num in (a0 + 1, a0 + 2):
                proxy.job_data.jobs[job_num] = JobData.job_data(branches=[job_num - 1])
            proxy.job_data.attempts[a0 + 1] = 2
            with merged.recording():
                proxy._submit_group(OrderedDict([(a0 + 1, [a0])]))
                proxy._submit_group(OrderedDict([(a0 + 2, [a0 + 1])]))
        fresh_process()
        single = Render()
        with self.cli((A,), "--max-runtime", 4, family=self.family) as root:
            proxy = root.workflow_proxy
            proxy.dashboard = NoJobDashboard()
            proxy.job_file_factory = proxy.create_job_file_factory()
            for job_num, attempts in ((1, 2), (2, 0)):
                proxy.job_data.jobs[job_num] = JobData.job_data(branches=[job_num - 1])
                proxy.job_data.attempts[job_num] = attempts
            with single.recording():
                proxy._submit_group(OrderedDict([(1, [0])]))
                proxy._submit_group(OrderedDict([(2, [1])]))
        self.assertEqual(self.custom(merged), self.custom(single))
        self.assertEqual(
            self.custom(merged),
            [
                # 1.5 ** 2 on the runtime, 1.25 ** 2 on the memory
                {"+MaxRuntime": 32399, "RequestMemory": 3125, "RequestCpus": 1},
                {"+MaxRuntime": 14399, "RequestMemory": 2000, "RequestCpus": 1},
            ],
        )


def batch_key(job_id):
    """The fake batch system's key of a job id law got back from a submission."""
    if isinstance(job_id, tuple) and hasattr(job_id, "proj_dir"):
        return f"{job_id.proj_dir}#{job_id.crab_num}"
    return job_id


class _TheEraWorkflowsAreLeftAlone:
    def test_the_era_workflows_own_proxy_is_left_alone(self):
        write_gates(self.version, **{A: 3, B: 2})
        fresh_process()
        with self.cli((A, B), "--tasks-per-job", 2) as root:
            own = {era: root.single(era).workflow_proxy for era in (A, B)}

            def state(proxy):
                return (
                    proxy.job_manager,
                    proxy.job_file_factory,
                    proxy.dashboard,
                    proxy._submitted,
                    json.dumps(proxy.job_data, sort_keys=True, default=str),
                )

            before = {era: state(proxy) for era, proxy in own.items()}
            self.run_root(root)
            subs = root.workflow_proxy.era_subs()
            for era in (A, B):
                with self.subTest(era=era):
                    self.assertIs(root.single(era)._workflow_proxy, own[era])
                    self.assertEqual(state(own[era]), before[era])
                    self.assertIsNot(subs[era], own[era])
                    self.assertTrue(subs[era].job_data.jobs)
        self.assertEqual(sum(self.submissions().values()), 5)

    def test_interleaved_submissions_come_back_in_the_order_law_asked(self):
        write_gates(self.version, **{A: 2, B: 2})
        a0, b0 = ERA_IDS[A] * STRIDE, ERA_IDS[B] * STRIDE
        submit_jobs = OrderedDict(
            [(a0 + 1, [a0]), (b0 + 1, [b0]), (a0 + 2, [a0 + 1]), (b0 + 2, [b0 + 1])]
        )
        with self.cli((A, B)) as root:
            proxy = root.workflow_proxy
            proxy.dashboard = NoJobDashboard()
            for job_num, branches in submit_jobs.items():
                proxy.job_data.jobs[job_num] = JobData.job_data(branches=branches)
            job_ids, data = proxy._submit_group(submit_jobs)
        self.assertEqual(list(data), list(submit_jobs))
        batch = Batch().jobs()
        self.assertEqual(
            [
                (batch[batch_key(j)]["era"], batch[batch_key(j)]["branches"])
                for j in job_ids
            ],
            [(A, [0]), (B, [0]), (A, [1]), (B, [1])],
        )


class HTCondorTheEraWorkflowsAreLeftAlone(_TheEraWorkflowsAreLeftAlone, _Case):
    backend = "htcondor"


class CrabTheEraWorkflowsAreLeftAlone(_TheEraWorkflowsAreLeftAlone, _Case):
    backend = "crab"


class FakeBatchedCondorManager(FakeCondorManager):
    """An HTCondor that takes one job file per job: law submits through `_submit_batch`."""

    job_grouping_submit = False


class IntermediateDumpsKeepEveryJobOfTheEra(_Case):
    """law dumps the job data while it submits a batch, from the proxy that submits: an era
    proxy's own dump would write that era's file with the jobs of this round alone."""

    @contextlib.contextmanager
    def one_file_per_job(self, paths):
        """law submits one job file per job and dumps the job data after each job; yields
        path -> the content of every write of that file, for the files in `paths`."""
        writes = {path: [] for path in paths}
        original = law.LocalFileTarget.dump

        def dump(target, *args, **kwargs):
            result = original(target, *args, **kwargs)
            if target.path in writes:
                writes[target.path].append(load_json(target.path))
            return result

        with mock.patch.object(
            law.htcondor.HTCondorWorkflow,
            "htcondor_create_job_manager",
            lambda task, **kwargs: FakeBatchedCondorManager(**kwargs),
        ), mock.patch.object(
            EraJobsUp, "htcondor_dump_intermediate_job_data", lambda task: 1
        ), mock.patch.object(
            law.LocalFileTarget, "dump", autospec=True, side_effect=dump
        ):
            yield writes

    def assertIdsWrittenOneByOne(self, writes, expected):
        """Every write of an era's file carries each job of `expected` (job number -> the
        batch key it was submitted under) with that id or with none, and the jobs of one
        batch get their ids one write at a time: law dumps after each job it submits."""
        seen, firsts = set(), []
        for data in writes:
            real = set()
            for num, key in expected.items():
                job = data["jobs"].get(num)
                if job is None or job["job_id"] == JobData.dummy_job_id:
                    continue
                self.assertEqual(stored_batch_key(job["job_id"]), key, data)
                real.add(num)
            if real - seen:
                firsts.append(sorted(real - seen))
            seen |= real
        self.assertEqual([len(new) for new in firsts], [1] * len(expected), firsts)
        self.assertEqual(seen, set(expected))

    def test_no_dump_drops_the_jobs_an_earlier_run_left_in_the_eras_file(self):
        write_gates(self.version, **{A: 4, B: 1})
        self.drive((A,), "--no-poll", "--parallel-jobs", 2)
        a_file = self.jobs_file(A, f"{self.kind()}_0To4.json")
        earlier = load_json(a_file)
        self.assertEqual(sorted(earlier["jobs"]), ["1", "2"])
        self.assertEqual(sorted(earlier["unsubmitted_jobs"]), ["3", "4"])
        with self.one_file_per_job([a_file]) as written:
            self.drive((A, B))
        writes = written[a_file]
        self.assertGreater(len(writes), 2)
        for data in writes:
            jobs, waiting = set(data["jobs"]), set(data["unsubmitted_jobs"])
            # every job once: a job both submitted and waiting is submitted again by the
            # next driver if this one ends here
            self.assertEqual(jobs & waiting, set(), data)
            self.assertEqual(jobs | waiting, {"1", "2", "3", "4"})
            self.assertTrue({"1", "2"} <= jobs, sorted(jobs))
        self.assertEqual(
            self.submissions(),
            Counter({(A, 0): 1, (A, 1): 1, (A, 2): 1, (A, 3): 1, (B, 0): 1}),
        )
        # the ids: the earlier run's jobs keep theirs (law may drop one it records finished),
        # and each job submitted now is written with its id by the dump right after it
        for data in writes:
            for num in ("1", "2"):
                job = data["jobs"][num]
                if job["job_id"] == JobData.dummy_job_id:
                    self.assertEqual(job["status"], "finished", data)
                else:
                    self.assertEqual(job["job_id"], earlier["jobs"][num]["job_id"])
        keys = self.batch_keys()
        self.assertIdsWrittenOneByOne(
            writes, {"3": keys[(A, (2,))], "4": keys[(A, (3,))]}
        )

    def test_each_dump_of_a_round_writes_the_ids_known_so_far_in_every_era(self):
        """Both eras submitted in one round, one job file per job: each era's file gets the id
        of each of its jobs from the dump that follows that job, not at the end of the round.
        """
        write_gates(self.version, **{A: 2, B: 2})
        self.addCleanup(Batch().forget, self.version)
        files = {era: self.jobs_file(era, f"{self.kind()}_0To2.json") for era in (A, B)}
        with self.one_file_per_job(files.values()) as written:
            self.drive((A, B), "--no-poll")
        self.assertEqual(
            self.submissions(), Counter({(A, 0): 1, (A, 1): 1, (B, 0): 1, (B, 1): 1})
        )
        keys = self.batch_keys()
        for era, path in files.items():
            with self.subTest(era=era):
                self.assertIdsWrittenOneByOne(
                    written[path], {"1": keys[(era, (0,))], "2": keys[(era, (1,))]}
                )


# --------------------------------------------------------------------------------------------
# --cancel-jobs / --cleanup-jobs, and a round whose submission fails part-way
# --------------------------------------------------------------------------------------------


class _CancelAndCleanupReachEveryEra:
    """--cancel-jobs and --cleanup-jobs over several eras hand the job manager the jobs of
    every era's file -- also of an era whose outputs all exist, as the run over that era
    alone does -- and lay out, submit and write nothing."""

    @contextlib.contextmanager
    def manager_calls(self, action, calls):
        """The job ids the job manager is asked to cancel or clean up, as batch keys."""
        manager = (
            FakeCondorManager if self.backend == "htcondor" else lc.FLAFCrabJobManager
        )

        def record(_manager, job_ids, *args, **kwargs):
            calls.extend(batch_key(job_id) for job_id in job_ids)
            return []

        with mock.patch.object(manager, f"{action}_batch", record), mock.patch.object(
            manager, f"{action}_group", record
        ):
            yield

    def test_every_eras_jobs_are_handed_to_the_manager_and_nothing_is_laid_out(self):
        write_gates(self.version, **{A: 2, B: 2})
        self.addCleanup(Batch().forget, self.version)
        with in_flight_jobs(finish_after=5):
            self.drive((A, B), "--no-poll")
        # A's outputs all exist while its jobs still run
        for branch in (0, 1):
            path = out_path(self.version, "EraJobsUp", A, branch)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
        files = {era: self.jobs_file(era, f"{self.kind()}_0To2.json") for era in (A, B)}
        before = {era: read_bytes(path) for era, path in files.items()}
        in_files = sorted(
            stored_batch_key(job["job_id"])
            for path in files.values()
            for job in load_json(path)["jobs"].values()
        )
        self.assertEqual(in_files, sorted(self.batch_keys().values()))
        self.assertEqual(len(in_files), 4)
        for flag, action in (
            ("--cancel-jobs", "cancel"),
            ("--cleanup-jobs", "cleanup"),
        ):
            with self.subTest(flag=flag):
                calls = {}
                for eras in ((A, B), (A,), (B,)):
                    calls[eras] = []
                    with self.manager_calls(action, calls[eras]), mock.patch.object(
                        lc.EraMergedProxy,
                        "_lay_out_eras",
                        autospec=True,
                        side_effect=lc.EraMergedProxy._lay_out_eras,
                    ) as lay_out:
                        self.drive(eras, flag)
                    self.assertEqual(lay_out.call_count, 0)
                self.assertEqual(sorted(calls[(A, B)]), in_files)
                self.assertEqual(
                    sorted(calls[(A, B)]), sorted(calls[(A,)] + calls[(B,)])
                )
                self.assertEqual(
                    {era: read_bytes(path) for era, path in files.items()}, before
                )
                self.assertEqual(sum(self.submissions().values()), 4)


class HTCondorCancelAndCleanupReachEveryEra(_CancelAndCleanupReachEveryEra, _Case):
    backend = "htcondor"


class CrabCancelAndCleanupReachEveryEra(_CancelAndCleanupReachEveryEra, _Case):
    backend = "crab"


class _ARoundThatFailsPartWay:
    """An era's jobs fail to build after another era's jobs were submitted in the same
    round: those jobs run, so their ids are recorded before the error ends the run, and the
    next run polls them and submits every branch once."""

    @contextlib.contextmanager
    def failing_build(self, era):
        """The job-config hook raises for the jobs of `era`; jobs submitted meanwhile stay in
        flight."""
        hook = f"{self.backend}_job_config"
        build = getattr(EraJobsUp, hook)

        def failing(task, config, *args, **kwargs):
            if task.only_era == era:
                raise RuntimeError(f"cannot build the job file of {era}")
            return build(task, config, *args, **kwargs)

        with in_flight_jobs(), mock.patch.object(EraJobsUp, hook, failing):
            yield

    def test_the_jobs_submitted_before_the_failure_are_recorded(self):
        write_gates(self.version, **{A: 2, B: 2})
        with self.failing_build(B):
            with self.assertRaises(RuntimeError) as caught:
                self.drive((A, B), "--no-poll")
        self.assertIn(f"cannot build the job file of {B}", str(caught.exception))
        self.assertEqual(self.submissions(), Counter({(A, 0): 1, (A, 1): 1}))
        a_file = self.jobs_file(A, f"{self.kind()}_0To2.json")
        self.assertTrue(os.path.exists(a_file))
        batch = Batch().jobs()
        recorded = {}
        for num, job in load_json(a_file)["jobs"].items():
            self.assertNotEqual(job["job_id"], JobData.dummy_job_id, num)
            submitted = batch[stored_batch_key(job["job_id"])]
            recorded[num] = (submitted["era"], submitted["branches"])
        self.assertEqual(recorded, {"1": (A, [0]), "2": (A, [1])})
        # no job of B was submitted: none is recorded with an id
        for name in job_file_names(self.version).get(B, []):
            b = load_json(self.jobs_file(B, name))
            self.assertEqual(
                {job["job_id"] for job in b["jobs"].values()} - {JobData.dummy_job_id},
                set(),
            )
        self.drive((A, B))
        self.assertEqual(
            self.submissions(), Counter({(A, 0): 1, (A, 1): 1, (B, 0): 1, (B, 1): 1})
        )
        self.assertEqual(
            outputs_of(self.version),
            [f"{A}/0.txt", f"{A}/1.txt", f"{B}/0.txt", f"{B}/1.txt"],
        )

    def test_the_jobs_never_submitted_are_first_attempts_in_the_next_run(self):
        """The next run submits the jobs of the era whose build failed as the run over that
        era alone with the same failure does: as first attempts. Counted as retries of jobs
        that never ran, they would use up --retries (with --retries 0, fail unsubmitted) and,
        for a task packed by cost, ask for escalated resources."""
        base = self.version
        attempts = {}
        try:
            for variant, eras in (("together", (A, B)), ("alone", (B,))):
                self.version = f"{base}_{variant}"
                self.addCleanup(wipe, self.version)
                write_gates(self.version, **{A: 2, B: 2})
                with self.failing_build(B):
                    with self.assertRaises(RuntimeError):
                        self.drive(eras, "--no-poll")
                self.drive(eras)
                self.assertEqual(self.submissions()[(B, 0)], 1)
                b_file = self.jobs_file(B, f"{self.kind()}_0To2.json")
                attempts[variant] = load_json(b_file)["attempts"]
        finally:
            self.version = base
        self.assertEqual(attempts["alone"], {})
        self.assertEqual(attempts["together"], attempts["alone"])

    def test_without_retries_every_era_reruns_as_the_run_over_it_alone(self):
        """Whichever era's build fails -- the first of the round, before anything was
        submitted, or a later one -- a rerun with --retries 0 (the HelloWorld CI's value)
        submits every era as the runs over each era alone, with the same failure, do: no job
        that never ran may count as a failed attempt."""
        base = self.version
        for failing in (A, B):
            outcome = {}
            for variant, runs in (("together", [(A, B)]), ("alone", [(A,), (B,)])):
                self.version = f"{base}_{failing}_{variant}"
                self.addCleanup(wipe, self.version)
                self.addCleanup(Batch().forget, self.version)
                write_gates(self.version, **{A: 2, B: 2})
                with self.failing_build(failing):
                    for eras in runs:
                        try:
                            self.drive(eras, "--no-poll")
                        except RuntimeError as e:
                            self.assertIn(
                                f"cannot build the job file of {failing}", str(e)
                            )
                raised = []
                for eras in runs:
                    try:
                        self.drive(eras, "--retries", 0)
                    except Exception as e:
                        raised.append(f"{','.join(eras)}: {e}")
                outcome[variant] = (raised, self.submissions())
            self.version = base
            with self.subTest(failing=failing):
                self.assertEqual(outcome["alone"][0], [])
                self.assertEqual(
                    sorted(outcome["alone"][1].items()),
                    [((era, b), 1) for era in (A, B) for b in (0, 1)],
                )
                self.assertEqual(outcome["together"], outcome["alone"])


class HTCondorARoundThatFailsPartWay(_ARoundThatFailsPartWay, _Case):
    backend = "htcondor"


class CrabARoundThatFailsPartWay(_ARoundThatFailsPartWay, _Case):
    backend = "crab"


class HTCondorABatchInterruptedPartWay(_Case):
    """A batch submission (one job file per job) interrupted after an era's first job: law's
    intermediate dump has recorded that job's id, which the run keeps as a run over the era
    alone keeps it, so the next run submits each branch once. A later era's job-file factory
    failing leaves the earlier eras' ids recorded the same way."""

    backend = "htcondor"

    @contextlib.contextmanager
    def batched(self, interrupt_era=None, dump_every=1):
        real = FakeBatchedCondorManager.submit_batch

        def submit_batch(manager, job_files, **kwargs):
            with open(job_files[0]) as f:
                jdl = f.read()
            args = re.findall(r"^arguments\s*=\s*(.*)$", jdl, re.M)[0]
            if decode_job_args(args.strip().strip('"'))[3]["--era"] != interrupt_era:
                return real(manager, job_files, **kwargs)
            ids = real(manager, job_files[:1], **kwargs)
            raise KeyboardInterrupt(f"Ctrl-C after {ids}")

        with mock.patch.object(
            law.htcondor.HTCondorWorkflow,
            "htcondor_create_job_manager",
            lambda task, **kwargs: FakeBatchedCondorManager(**kwargs),
        ), mock.patch.object(
            EraJobsUp, "htcondor_dump_intermediate_job_data", lambda task: dump_every
        ), mock.patch.object(
            FakeBatchedCondorManager, "submit_batch", submit_batch
        ):
            yield

    def test_the_interrupted_eras_submitted_job_keeps_its_id(self):
        base = self.version
        counts = {}
        for variant, eras in (("together", (A, B)), ("alone", (B,))):
            self.version = f"{base}_{variant}"
            self.addCleanup(wipe, self.version)
            self.addCleanup(Batch().forget, self.version)
            write_gates(self.version, **{A: 2, B: 2})
            with in_flight_jobs(finish_after=5), self.batched(B):
                with self.assertRaises(KeyboardInterrupt):
                    self.drive(eras, "--no-poll")
            interrupted = load_json(self.jobs_file(B, "htcondor_jobs_0To2.json"))
            self.assertNotIn(
                interrupted["jobs"]["1"]["job_id"], (None, JobData.dummy_job_id)
            )
            self.drive(eras, "--no-poll")
            counts[variant] = self.submissions()[(B, 0)]
        self.version = base
        self.assertEqual(counts, {"together": 1, "alone": 1})

    def test_an_id_not_dumped_yet_is_kept_too(self):
        # law dumps every 50 jobs here, so the interrupted era's submitted job is known to its
        # era's proxy alone when the round ends
        write_gates(self.version, **{A: 2, B: 2})
        with in_flight_jobs(finish_after=5), self.batched(B, dump_every=50):
            with self.assertRaises(KeyboardInterrupt):
                self.drive((A, B), "--no-poll")
        interrupted = load_json(self.jobs_file(B, "htcondor_jobs_0To2.json"))
        self.assertNotIn(
            interrupted["jobs"]["1"]["job_id"], (None, JobData.dummy_job_id)
        )
        self.assertEqual(dict(interrupted["unsubmitted_jobs"]), {"2": [1]})
        self.drive((A, B), "--no-poll")
        self.assertEqual(self.submissions()[(B, 0)], 1)

    def test_a_later_eras_factory_failure_keeps_the_earlier_eras_ids(self):
        write_gates(self.version, **{A: 2, B: 2})
        create = law.htcondor.HTCondorWorkflow.htcondor_create_job_file_factory

        def factory(task, **kwargs):
            # law also makes one for the workflow over both eras, which builds no job
            if task.era == (B,):
                raise RuntimeError(f"no job file factory for {B}")
            return create(task, **kwargs)

        # law's own rate (a dump every 50 jobs), so A's ids reach its file through the
        # failure handling alone
        with in_flight_jobs(), self.batched(dump_every=50), mock.patch.object(
            EraJobsUp, "htcondor_create_job_file_factory", factory
        ):
            with self.assertRaises(RuntimeError):
                self.drive((A, B), "--no-poll")
        recorded = load_json(self.jobs_file(A, "htcondor_jobs_0To2.json"))["jobs"]
        self.assertEqual(len(recorded), 2)
        for job in recorded.values():
            self.assertNotEqual(job["job_id"], JobData.dummy_job_id)
        self.drive((A, B), "--no-poll")
        self.assertEqual(
            self.submissions(), Counter({(A, 0): 1, (A, 1): 1, (B, 0): 1, (B, 1): 1})
        )


# --------------------------------------------------------------------------------------------
# the lost-outputs guard of a resumed run, per era (I11's guard on a run over several eras)
# --------------------------------------------------------------------------------------------


class _LostOutputsAreJudgedPerEra:
    def test_one_eras_mass_loss_stops_the_run_whatever_the_other_eras_backlog(self):
        write_gates(self.version, **{A: 10, B: 60})
        self.drive((A,))
        for branch in range(5):
            os.remove(out_path(self.version, "EraJobsUp", A, branch))
        submitted = sum(self.submissions().values())
        with self.assertRaises(RuntimeError) as caught:
            self.drive((A, B))
        self.assertIn(f"5 of the 10 jobs of {A}", str(caught.exception))
        self.assertEqual(sum(self.submissions().values()), submitted)

    def test_a_loss_within_the_fraction_is_resubmitted(self):
        write_gates(self.version, **{A: 10, B: 2})
        self.drive((A,))
        os.remove(out_path(self.version, "EraJobsUp", A, 3))
        self.drive((A, B))
        counts = self.submissions()
        self.assertEqual(counts[(A, 3)], 2)
        self.assertEqual(sum(counts.values()), 10 + 1 + 2)


class HTCondorLostOutputsAreJudgedPerEra(_LostOutputsAreJudgedPerEra, _Case):
    backend = "htcondor"


class CrabLostOutputsAreJudgedPerEra(_LostOutputsAreJudgedPerEra, _Case):
    backend = "crab"


# --------------------------------------------------------------------------------------------
# scheduling: no job resources booked, a token per era
# --------------------------------------------------------------------------------------------


class _NothingIsBookedAtScheduling:
    def test_the_merged_proxy_books_no_job_while_luigi_schedules(self):
        write_gates(self.version, **{A: 3, B: 2})
        for branch in (0, 1):
            path = out_path(self.version, "EraJobsUp", A, branch)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
        with self.cli((A, B), "--tasks-per-job", 2) as root:
            resources = root.process_resources()
            proxy = root.workflow_proxy
            self.assertEqual(proxy.process_resources(), {})
            self.assertEqual(dict(proxy.job_data.jobs), {})
            self.assertEqual(dict(proxy.job_data.unsubmitted_jobs), {})
        self.assertEqual(
            resources,
            {
                f"flaf_jobs:{data_path(self.version, 'EraJobsUp', era)}": 1
                for era in (A, B)
            },
        )


class HTCondorNothingIsBookedAtScheduling(_NothingIsBookedAtScheduling, _Case):
    backend = "htcondor"


class CrabNothingIsBookedAtScheduling(_NothingIsBookedAtScheduling, _Case):
    backend = "crab"


class _AResumedEraIsJudgedOnAFreshLook:
    def test_a_job_that_wrote_its_outputs_while_no_driver_polled_is_not_resubmitted(
        self,
    ):
        """law reports a resumed job that finishes in the first status query as having no
        outputs (they did not exist when it gathered them); FLAF looks again, through the
        canonical branch tasks of the run over several eras."""
        write_gates(self.version, **{A: 3, B: 1})
        self.drive((A,), "--no-poll")
        self.assertFalse(outputs_of(self.version))
        self.drive((A, B))
        self.assertEqual(
            self.submissions(), Counter({(A, 0): 1, (A, 1): 1, (A, 2): 1, (B, 0): 1})
        )


class HTCondorAResumedEraIsJudgedOnAFreshLook(_AResumedEraIsJudgedOnAFreshLook, _Case):
    backend = "htcondor"


class CrabAResumedEraIsJudgedOnAFreshLook(_AResumedEraIsJudgedOnAFreshLook, _Case):
    backend = "crab"


# --------------------------------------------------------------------------------------------
# CRAB: the stall watchdog and the job manager of a run over several eras
# --------------------------------------------------------------------------------------------


class Flag:
    """One entry of the heartbeat directory listing."""

    def __init__(self, name, date):
        self.name = str(name)
        self.date = date
        self.is_dir = False


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def minutes(n):
    return datetime.timedelta(minutes=n)


class TheCrabWatchdogOverSeveralEras(_Case):
    backend = "crab"

    def setUp(self):
        super().setUp()
        write_gates(self.version, **{A: 3, B: 2})

    def test_each_era_is_watched_by_its_own_watchdog(self):
        with self.cli((A, B)) as root:
            watchdog = root.job_watchdog()
            self.assertIsInstance(watchdog, lc.EraWatchdogs)
            self.assertIs(root.job_watchdog(), watchdog)
            self.assertIs(root.workflow_proxy.job_manager.watchdog, watchdog)
            for era in (A, B):
                part = watchdog.parts[ERA_IDS[era]]
                self.assertIs(part, root.single(era).job_watchdog())
                self.assertEqual(
                    part.flag_dir,
                    f"{BASE}/{self.version}/{crab_watchdog.HEARTBEAT_DIR}/EraJobsUp/{era}",
                )

    def test_jobs_are_fed_to_their_eras_watchdog_and_the_verdicts_joined(self):
        """A's job 1 and B's job 1 stalled, B's job 2 healthy: the flags are named by the
        era's own branch numbers."""
        a0, b0 = ERA_IDS[A] * STRIDE, ERA_IDS[B] * STRIDE
        now = utcnow()
        with self.cli((A, B)) as root:
            watchdog = root.job_watchdog()
            manager = root.workflow_proxy.job_manager
            flag_dirs = {era: watchdog.parts[ERA_IDS[era]].flag_dir for era in (A, B)}
            flags = {
                flag_dirs[A]: [Flag(0, now - minutes(99)), Flag(1, now - minutes(99))],
                flag_dirs[B]: [Flag(0, now - minutes(99)), Flag(1, now - minutes(1))],
            }
            ids = {
                "A1": manager.JobId(1, "taskA", "/projA"),
                "B1": manager.JobId(1, "taskB", "/projB"),
                "B2": manager.JobId(2, "taskB", "/projB"),
            }
            watchdog.set_jobs(
                {
                    a0 + 1: {"job_id": list(ids["A1"]), "branches": [a0, a0 + 1]},
                    b0 + 1: {"job_id": list(ids["B1"]), "branches": [b0]},
                    b0 + 2: {"job_id": list(ids["B2"]), "branches": [b0 + 1]},
                }
            )
            self.assertEqual(
                watchdog.parts[ERA_IDS[A]]._by_id, {(1, "taskA"): (1, [0, 1])}
            )
            self.assertEqual(
                watchdog.parts[ERA_IDS[B]]._by_id,
                {(1, "taskB"): (1, [0]), (2, "taskB"): (2, [1])},
            )
            with mock.patch.object(
                crab_watchdog,
                "gfal_ls_safe",
                side_effect=lambda uri, **kwargs: flags.get(uri),
            ):
                self.assertTrue(watchdog.refresh())
            result = {
                job_id: manager.job_status_dict(job_id=job_id, status=manager.RUNNING)
                for job_id in ids.values()
            }
            # first seen running long ago: the grace for a first beat is over
            self.assertEqual(watchdog.verdicts(result, now=now - minutes(999)), {})
            verdicts = watchdog.verdicts(result)
        self.assertEqual(sorted(verdicts), sorted([ids["A1"], ids["B1"]]))
        for job_id in (ids["A1"], ids["B1"]):
            self.assertIn("heartbeat 99 min old", verdicts[job_id])
        # why the jobs are split: one era's watchdog fed the numbers of the run over both
        # eras finds none of its flags, and condemns B's healthy job for having none
        plain = crab_watchdog.StallWatchdog(
            flag_dirs[B],
            crab_watchdog.watchdog_config(
                {"watchdog": {"enabled": True, "interval_minutes": 10}}
            ),
        )
        plain.set_jobs(
            {
                b0 + 1: {"job_id": list(ids["B1"]), "branches": [b0]},
                b0 + 2: {"job_id": list(ids["B2"]), "branches": [b0 + 1]},
            }
        )
        with mock.patch.object(
            crab_watchdog,
            "gfal_ls_safe",
            side_effect=lambda uri, **kwargs: flags.get(uri),
        ):
            self.assertTrue(plain.refresh())
        self.assertEqual(plain.verdicts(result, now=now - minutes(999)), {})
        control = plain.verdicts(result)
        self.assertIn(ids["B2"], control)
        self.assertIn("no heartbeat", control[ids["B2"]])

    def test_a_replaced_job_is_forgotten_by_every_eras_watchdog(self):
        with self.cli((A, B)) as root:
            watchdog = root.job_watchdog()
            job_id = root.workflow_proxy.job_manager.JobId(1, "task", "/proj")
            with contextlib.ExitStack() as stack:
                spies = [
                    stack.enter_context(
                        mock.patch.object(part, "forget", wraps=part.forget)
                    )
                    for part in watchdog.parts.values()
                ]
                watchdog.forget(job_id)
        for spy in spies:
            spy.assert_called_once_with(job_id)

    def test_the_era_proxies_submit_through_the_roots_job_manager(self):
        fresh_process()
        with self.cli((A, B)) as root:
            self.run_root(root)
            manager = root.workflow_proxy.job_manager
            subs = root.workflow_proxy.era_subs()
            self.assertEqual(sorted(subs), [A, B])
            for sub in subs.values():
                self.assertIs(sub.job_manager, manager)
            projects = {
                key.split("#")[0]
                for key, job in Batch().jobs().items()
                if job["version"] == self.version
            }
            self.assertEqual(len(projects), 2)
            self.assertEqual(manager._submitted_projects, projects)

    def test_the_poll_callback_stops_on_what_the_shared_manager_recorded(self):
        with self.cli((A, B)) as root:
            sub = root.workflow_proxy.era_subs()[B]
            # a refusal recorded while era B's jobs were submitted
            sub.job_manager.stop_reason = "two refused submissions"
            with self.assertRaises(RuntimeError) as caught:
                root.crab_poll_callback(mock.Mock())
        self.assertIn("two refused submissions", str(caught.exception))


# --------------------------------------------------------------------------------------------
# C7: the per-era luigi resource token
# --------------------------------------------------------------------------------------------


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


class TheEraTokenSerialisesInstancesThatShareAnEra(_Case):
    """luigi gives a resource it has no configuration for a capacity of one: two instances of
    a stage that drive the same era's job file never run at once, whatever --workers is.
    """

    #: how long each stubbed workflow run lasts, in seconds
    run_seconds = 1.0

    def build(self, tasks):
        log = os.path.join(root_dir(), f"runs_{self.version}.jsonl")
        seconds = self.run_seconds

        def run_impl(proxy):
            start = time.time()
            time.sleep(seconds)
            with open(log, "a") as f:
                f.write(json.dumps([proxy.task.task_id, start, time.time()]) + "\n")

        with mock.patch.object(
            BaseRemoteWorkflowProxy, "_run_impl", autospec=True, side_effect=run_impl
        ):
            ok = luigi.build(tasks, workers=3, local_scheduler=True, log_level="ERROR")
        self.assertTrue(ok)
        spans = {}
        with open(log) as f:
            for line in f:
                task_id, start, end = json.loads(line)
                spans[task_id] = (start, end)
        self.assertEqual(sorted(spans), sorted(t.task_id for t in tasks))
        return spans

    def test_instances_sharing_an_era_never_overlap(self):
        tasks = {
            eras: EraJobsTokenUp(version=self.version, era=eras, workflow="htcondor")
            for eras in ((B,), (A, B), (C,))
        }
        for task in tasks.values():
            self.assertTrue(
                any(k.startswith("flaf_jobs:") for k in task.process_resources())
            )
        spans = self.build(list(tasks.values()))
        b, ab, c = (spans[tasks[e].task_id] for e in ((B,), (A, B), (C,)))
        self.assertFalse(overlap(b, ab), (b, ab))
        self.assertTrue(overlap(c, b) or overlap(c, ab), (b, ab, c))

    def test_producers_of_one_era_in_their_own_directories_overlap(self):
        tasks = [
            EraJobsTokenCache(
                version=self.version, era=(A,), workflow="htcondor", producer_to_run=p
            )
            for p in ("p1", "p2")
        ]
        spans = self.build(tasks)
        self.assertTrue(overlap(*spans.values()), spans)


class EraJobsResourcesUp(EraJobsUp):
    """EraJobsUp whose jobs claim a resource, so that law reports finished jobs to luigi."""

    def htcondor_job_resources(self, job_num, branches):
        resources = dict(super().htcondor_job_resources(job_num, branches) or {})
        resources["slots"] = resources.get("slots", 0) + 1
        return resources


class LawNeverReleasesTheToken(_Case):
    def test_finished_jobs_release_their_own_resources_only(self):
        write_gates(self.version, **{A: 2, B: 2})
        decreased = []
        with mock.patch.object(
            EraJobsResourcesUp,
            "decrease_running_resources",
            create=True,
            side_effect=lambda resources: decreased.append(dict(resources)),
        ):
            for eras in ((A,), (A, B)):
                fresh_process()
                with self.cli(eras, family="EraJobsResourcesUp") as root:
                    # what luigi's worker asks while it schedules the workflow
                    tokens = [
                        k
                        for k in root.process_resources()
                        if k.startswith("flaf_jobs:")
                    ]
                    self.assertEqual(len(tokens), len(eras))
                    self.run_root(root)
        self.assertTrue(decreased)
        self.assertEqual(
            [k for d in decreased for k in d if k.startswith("flaf_jobs:")], []
        )


# --------------------------------------------------------------------------------------------
# S10: the branch tasks of a run over several eras are its eras' own
# --------------------------------------------------------------------------------------------


class TheBranchTasksAreTheErasOwn(_Case):
    def test_a_global_branch_is_the_branch_of_its_eras_own_workflow(self):
        write_gates(self.version, **{A: 2, B: 1})
        with self.cli((A, B), workflow="local") as root:
            branches = root.get_branch_tasks()
            self.assertEqual(
                [(k, t.era, t.branch) for k, t in branches.items()],
                [
                    (ERA_IDS[A] * STRIDE, (A,), 0),
                    (ERA_IDS[A] * STRIDE + 1, (A,), 1),
                    (ERA_IDS[B] * STRIDE, (B,), 0),
                ],
            )
            for k, task in branches.items():
                era, local = root.era_of(k), root.local_branch(k)
                self.assertIs(task, root.single(era).req_branch(local))
                self.assertEqual(
                    task.output().path, out_path(self.version, "EraJobsUp", era, local)
                )
            for bad in (-1, 0, ERA_IDS[C] * STRIDE):
                with self.assertRaises(ValueError):
                    root.era_of(bad)


# --------------------------------------------------------------------------------------------
# I22: cross-run compatibility, one driver process per run, killed where the scenario says
# --------------------------------------------------------------------------------------------


def _drive(spec):
    """A driver process: `law run EraJobsUp ...` through luigi's command line, with the batch
    system, the storage and the credentials faked; ERA_JOBS_KILL_AFTER ends it with
    os._exit(3) after that many polls."""
    argv = [
        "EraJobsUp",
        "--version",
        spec["version"],
        "--era",
        spec["eras"],
        "--workflow",
        spec["workflow"],
        "--local-scheduler",
        "--workers",
        "1",
        "--log-level",
        "ERROR",
    ] + [str(a) for a in spec["args"]]
    with contextlib.ExitStack() as stack:
        for patch in batch_patches():
            stack.enter_context(patch)
        ok = luigi.interface.run(argv)
        print("BUILD_OK" if ok else "BUILD_FAILED")
        fresh_process()
        with luigi.cmdline_parser.CmdlineParser.global_instance(argv) as cp:
            print(f"ROOT_COMPLETE {cp.get_task_obj().complete()}")
    sys.stdout.flush()


def file_digest(path):
    import hashlib

    return hashlib.sha256(read_bytes(path)).hexdigest()


class _CrossRuns:
    """The rerun patterns of the design (section 3.5), each run a driver process of its own.

    Oracle: for every era, a run over several eras submits what the run over that era alone
    would submit in the same state -- an in-flight job of an earlier run is polled to its end,
    never submitted again, and every branch is submitted once.
    """

    def run_driver(self, eras, *args, kill=0, finish_after=3):
        spec = {
            "version": self.version,
            "eras": ",".join(eras),
            "workflow": self.workflow,
            "args": list(args),
        }
        env = dict(
            os.environ,
            ERA_JOBS_KILL_AFTER=str(kill),
            ERA_JOBS_FINISH_AFTER=str(finish_after),
        )
        cmd = [sys.executable, os.path.abspath(__file__), "_drive", json.dumps(spec)]
        with tempfile.TemporaryFile("w+") as log:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                rc = None
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            log.seek(0)
            out = log.read()
        if rc == 3:
            return "KILLED", out
        if rc == 0 and "BUILD_OK" in out:
            return "OK", out
        return "FAILED", out

    def assertRun(self, result, expected="OK"):
        status, out = result
        self.assertEqual(status, expected, out[-3000:])
        return out

    def files(self):
        return job_file_names(self.version, kind=f"{self.workflow}_jobs")

    def file(self, era, name):
        return self.jobs_file(era, f"{self.workflow}_jobs_{name}.json")

    def eras_of_jobs(self):
        return {
            job["era"]
            for job in Batch().jobs().values()
            if job["version"] == self.version
        }

    def test_S1_a_stopped_single_era_run_resumed_over_two_eras(self):
        set_n(self.version, **{A: 4, B: 2})
        self.assertRun(self.run_driver((A,), "--parallel-jobs", 2, kill=1), "KILLED")
        first = load_json(self.file(A, "0To4"))
        in_flight = [
            j["job_id"] for j in first["jobs"].values() if j["job_id"] != "dummy_job_id"
        ]
        self.assertEqual(len(in_flight), 2)
        self.assertEqual(len(first["unsubmitted_jobs"]), 2)
        self.assertRun(self.run_driver((A, B)))
        self.assertEqual(
            self.files(),
            {
                A: [f"{self.workflow}_jobs_0To4.json"],
                B: [f"{self.workflow}_jobs_0To2.json"],
            },
        )
        counts = self.submissions()
        self.assertEqual(
            counts,
            Counter({(A, b): 1 for b in range(4)} | {(B, b): 1 for b in range(2)}),
        )
        self.assertEqual(self.eras_of_jobs(), {A, B})
        self.assertEqual(
            outputs_of(self.version),
            [f"{A}/{b}.txt" for b in range(4)] + [f"{B}/{b}.txt" for b in range(2)],
        )

    def test_S2_S3_a_stopped_two_era_run_resumed_over_one_era_then_another_set(self):
        set_n(self.version, **{A: 3, B: 3, C: 2})
        self.assertRun(self.run_driver((A, B), "--parallel-jobs", 4, kill=1), "KILLED")
        a = load_json(self.file(A, "0To3"))
        b = load_json(self.file(B, "0To3"))
        self.assertEqual((len(a["jobs"]), len(a["unsubmitted_jobs"])), (3, 0))
        self.assertEqual((len(b["jobs"]), len(b["unsubmitted_jobs"])), (1, 2))
        b_digest = file_digest(self.file(B, "0To3"))
        # S2: the run over A alone resumes A's file, and leaves B's alone
        self.assertRun(self.run_driver((A,)))
        self.assertEqual(
            self.submissions(),
            Counter({(A, 0): 1, (A, 1): 1, (A, 2): 1, (B, 0): 1}),
        )
        self.assertEqual(file_digest(self.file(B, "0To3")), b_digest)
        a_digest = file_digest(self.file(A, "0To3"))
        # S3: over B and C: B resumed, C fresh, A untouched
        self.assertRun(self.run_driver((B, C)))
        self.assertEqual(
            self.submissions(),
            Counter(
                {(A, b): 1 for b in range(3)}
                | {(B, b): 1 for b in range(3)}
                | {(C, b): 1 for b in range(2)}
            ),
        )
        self.assertEqual(file_digest(self.file(A, "0To3")), a_digest)
        self.assertEqual(
            self.files(),
            {
                A: [f"{self.workflow}_jobs_0To3.json"],
                B: [f"{self.workflow}_jobs_0To3.json"],
                C: [f"{self.workflow}_jobs_0To2.json"],
            },
        )

    def test_S4_an_era_whose_map_grew_while_its_jobs_were_in_flight(self):
        """The r3 oracle, literally: from the same state, the run over both eras submits for
        each era what the runs over each era alone submit. A's file is named by its new
        map, and every A branch lacking an output is submitted again (today's single-era
        rule); B resumes its file."""
        base = self.version
        submitted, files = {}, {}
        for variant, runs in (("together", [(A, B)]), ("alone", [(A,), (B,)])):
            self.version = f"{base}_{variant}"
            set_n(self.version, **{A: 3, B: 2})
            self.assertRun(self.run_driver((A, B), kill=1), "KILLED")
            write_gates(self.version, **{A: 4})
            for eras in runs:
                self.assertRun(self.run_driver(eras))
            submitted[variant], files[variant] = self.submissions(), self.files()
        self.version = base
        self.assertEqual(submitted["together"], submitted["alone"])
        self.assertEqual(files["together"], files["alone"])
        self.assertEqual(
            files["together"],
            {
                A: [
                    f"{self.workflow}_jobs_0To3.json",
                    f"{self.workflow}_jobs_0To4.json",
                ],
                B: [f"{self.workflow}_jobs_0To2.json"],
            },
        )
        self.assertEqual(
            submitted["together"],
            Counter({(A, 0): 2, (A, 1): 2, (A, 2): 2, (A, 3): 1, (B, 0): 1, (B, 1): 1}),
        )

    def test_S5_a_finished_era_is_pruned(self):
        set_n(self.version, **{A: 2, B: 2})
        self.assertRun(self.run_driver((A,)))
        a_digest = file_digest(self.file(A, "0To2"))
        self.assertRun(self.run_driver((A, B)))
        self.assertEqual(
            self.submissions(),
            Counter({(A, 0): 1, (A, 1): 1, (B, 0): 1, (B, 1): 1}),
        )
        self.assertEqual(file_digest(self.file(A, "0To2")), a_digest)

    def test_S6_repeated_restarts(self):
        set_n(self.version, **{A: 3, B: 3})
        for _ in range(3):
            self.assertRun(
                self.run_driver((A, B), "--parallel-jobs", 1, kill=1), "KILLED"
            )
        self.assertRun(self.run_driver((A, B), "--parallel-jobs", 1))
        self.assertEqual(
            self.submissions(),
            Counter({(A, b): 1 for b in range(3)} | {(B, b): 1 for b in range(3)}),
        )
        self.assertEqual(
            self.files(),
            {
                A: [f"{self.workflow}_jobs_0To3.json"],
                B: [f"{self.workflow}_jobs_0To3.json"],
            },
        )

    def test_S7_branch_selections_given_before_the_maps_exist(self):
        set_n(self.version, **{A: 4, B: 3})
        a0, b0 = ERA_IDS[A] * STRIDE, ERA_IDS[B] * STRIDE
        out = self.assertRun(
            self.run_driver((A, B), "--branches", f"{a0 + 1},{a0 + 2},{b0}")
        )
        self.assertIn("ROOT_COMPLETE True", out)
        self.assertEqual(self.submissions(), Counter({(A, 1): 1, (A, 2): 1, (B, 0): 1}))
        # law names a selection by its ranges (law/workflow/base.py get_branches_repr)
        selected = {
            A: [f"{self.workflow}_jobs_1To4.json"],
            B: [f"{self.workflow}_jobs_0.json"],
        }
        self.assertEqual(self.files(), selected)
        # the run over A alone with the same selection is the same work, in the same file
        out = self.assertRun(self.run_driver((A,), "--branches", "1,2"))
        self.assertIn("ROOT_COMPLETE True", out)
        self.assertEqual(sum(self.submissions().values()), 3)
        self.assertEqual(self.files(), selected)
        self.assertRun(self.run_driver((A, B)))
        self.assertEqual(
            self.submissions(),
            Counter({(A, b): 1 for b in range(4)} | {(B, b): 1 for b in range(3)}),
        )

    def test_S7_a_single_era_selection_given_before_its_map_exists(self):
        set_n(self.version, **{A: 4})
        out = self.assertRun(self.run_driver((A,), "--branches", "1,2"))
        self.assertIn("ROOT_COMPLETE True", out)
        self.assertEqual(self.submissions(), Counter({(A, 1): 1, (A, 2): 1}))
        self.assertEqual(self.files(), {A: [f"{self.workflow}_jobs_1To4.json"]})

    def test_S8_a_local_run_shares_the_canonical_branches(self):
        set_n(self.version, **{A: 2, B: 2})
        self.workflow, remote = "local", self.workflow
        try:
            self.assertRun(self.run_driver((A, B)))
        finally:
            self.workflow = remote
        self.assertEqual(
            outputs_of(self.version),
            [f"{A}/0.txt", f"{A}/1.txt", f"{B}/0.txt", f"{B}/1.txt"],
        )
        self.assertRun(self.run_driver((A,)))
        self.assertEqual(sum(self.submissions().values()), 0)


class HTCondorCrossRuns(_CrossRuns, _Case):
    workflow = "htcondor"


class CrabCrossRuns(_CrossRuns, _Case):
    workflow = "crab"


# --------------------------------------------------------------------------------------------
# a workflow over several eras without era branches (StatInference's card stages)
# --------------------------------------------------------------------------------------------


class EraCardWorkflow(_Case):
    """One workflow whose branches each read several eras: law's own branches, its requirements
    over all eras of the run, no single-era map built, every job over all eras, one job file,
    one luigi token and one log directory per version."""

    def test_one_workflow_with_law_branches(self):
        with self.cli((A, B), family="EraCardsUp") as task:
            self.assertIs(
                type(task.workflow_proxy), lc._BundleAwareHTCondorWorkflowProxy
            )
            self.assertEqual(task.get_branch_map(), {0: "card_0", 1: "card_1"})
            branch_tasks = task.get_branch_tasks()
            self.assertEqual(sorted(branch_tasks), [0, 1])
            for branch, branch_task in branch_tasks.items():
                self.assertEqual(branch_task.era, (A, B))
                self.assertEqual(branch_task.branch, branch)
            directory = data_path(self.version, "EraCardsUp")
            self.assertEqual(task.htcondor_output_directory().abspath, directory)
            self.assertEqual(task.process_resources(), {f"flaf_jobs:{directory}": 1})

    def test_requirements_over_all_eras(self):
        with self.cli((A, B), family="EraCardsUp") as task:
            reqs = task.workflow_requires()
            # no bundles: the stage ships none
            self.assertEqual(list(reqs), ["cards_input"])
            self.assertEqual(reqs["cards_input"].era, (A, B))
            self.assertEqual(
                reqs["cards_input"].task_id,
                EraCardsInput.req(task, branches=()).task_id,
            )

    def test_jobs_run_over_every_era(self):
        """Through luigi's worker, which asks for the requirements, the completeness and the
        resources before it runs the workflow (inside the command line, which law's job
        command reads)."""
        render = Render()
        with render.recording(), self.cli((A, B), family="EraCardsUp") as task:
            self.assertTrue(
                luigi.build([task], local_scheduler=True, log_level="ERROR")
            )
        directory = data_path(self.version, "EraCardsUp")
        self.assertEqual(
            [n for n in os.listdir(directory) if n.startswith("htcondor_jobs")],
            ["htcondor_jobs_0To2.json"],
        )
        self.assertEqual(
            [
                n
                for n in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, n))
            ],
            [],
        )
        logs = f"{BASE}/{self.version}/logs/EraCardsUp/"
        jobs = load_json(os.path.join(directory, "htcondor_jobs_0To2.json"))["jobs"]
        self.assertTrue(jobs)
        for job in jobs.values():
            self.assertTrue(job["extra"]["log"].startswith(logs), job["extra"]["log"])
        self.assertTrue(render.files)
        branches = []
        for summary in render.files:
            self.assertEqual(summary["eras"], [f"{A},{B}"])
            self.assertEqual(summary["log_remote_base_url"], logs.rstrip("/"))
            branches += [b for job in summary["jobs"] for b in job["branches"]]
        self.assertEqual(sorted(branches), [0, 1])
        self.assertEqual(
            self.submissions(), Counter({(f"{A},{B}", 0): 1, (f"{A},{B}", 1): 1})
        )
        # the requirement ran over both eras
        self.assertTrue(
            os.path.exists(out_path(self.version, "EraCardsInput", f"{A},{B}", 0))
        )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "_drive":
        _drive(json.loads(sys.argv[2]))
    else:
        unittest.main()
