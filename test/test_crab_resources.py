#!/usr/bin/env python3
"""How many cores, how much memory and how much runtime a CRAB job asks for.

CRAB sells memory only in per-core units -- the client refuses any task above
`max(MAX_MEMORY_SINGLE_CORE, numCores * MAX_MEMORY_PER_CORE)` and accepts only 1, 2, 4 or 8
cores, with a PSet declaring exactly that many threads -- so cores and memory cannot be fully
independent. What they can be, and what these tests pin down, is: with no request a job asks
for the most CRAB grants for its cores; an explicit `--crab-memory` is honoured exactly,
never silently raised and never silently lowered (on CRAB the number is a kill threshold, so a
shrunk request is a dead branch and an unsatisfiable one has to be an error at submit time);
a request larger than the task's own cores can hold buys the cores it needs; and the request
belongs to the task it was given to -- it never travels through req() to what that task
requires, never reaches the worker command line, and never changes an HTCondor submission.

Ported from DSProd test/test_crab_resources.py; everything here drives the real FLAF and law
code, with only the analysis Setup faked.
"""

import ast
import contextlib
import importlib.util
import itertools
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

# FLAF.Common.Utilities (imported through Setup) imports ROOT at module level, but nothing
# exercised here touches it, and the unit-test CI has no ROOT. The stand-in (a MagicMock, as
# in test_setup_loading.py) is removed from sys.modules right after the import, so that an
# `import ROOT` in another test module of the same process still fails as it should.
_root_stub = None
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    _root_stub = sys.modules["ROOT"] = mock.MagicMock()

import law  # noqa: E402
import luigi  # noqa: E402

from FLAF.run_tools import law_customizations as lc  # noqa: E402

if _root_stub is not None and sys.modules.get("ROOT") is _root_stub:
    del sys.modules["ROOT"]

FAMILY = "SomeTask"

#: the current production CRAB client, resolved the way /cvmfs/cms.cern.ch/common/crab does
CRAB_CVMFS_BASE = "/cvmfs/cms.cern.ch/share/cms"
CRAB_CLIENT_LATEST = os.path.join(
    CRAB_CVMFS_BASE, "crab", "1.0", "etc", "crab-prod.latest"
)


def ceiling(n_cores):
    """The CRAB client's own acceptance rule (CRABClient/Commands/submit.py), written out
    independently of `lc.crab_memory_ceiling` so that a change there is caught here."""
    return max(lc.CRAB_MB_SINGLE_CORE, n_cores * lc.CRAB_MB_PER_CORE)


class CrabResProducer(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    """A FLAF task with the base classes, in the order, of AnaTupleMergeTask / HistMergerTask."""

    bundle_flavours = ["core"]

    def create_branch_map(self):
        return {0: 0}

    def run(self):
        pass


class CrabResConsumer(CrabResProducer):
    """A FLAF task that requires CrabResProducer (through CrabResProducer.req(self))."""


class CrabResCostedProducer(CrabResProducer):
    """A task whose HTCondor memory request comes from the cost model, like AnaTupleFileTask."""

    def cost_params(self):
        return {
            "request_memory_mb": 4000,
            "retry_max_factor": 4.0,
            "retry_runtime_factor": 2.0,
            "retry_memory_factor": 2.0,
        }


_versions = itertools.count()


def unique_version():
    # luigi caches task instances by parameter values; a cached instance would keep the
    # Setup (and so the `crab:` block) of whichever test created it first
    return f"vres{next(_versions)}"


@contextlib.contextmanager
def flaf_env(crab_cfg=None):
    """The analysis Setup faked, ANALYSIS_DATA_PATH in a temporary directory, submit node."""
    setup = types.SimpleNamespace(
        global_params={"crab": dict(crab_cfg or {})},
        get_fs=lambda name: None,
    )
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
        lc.Setup, "getGlobal", return_value=setup
    ), mock.patch.dict(os.environ, {"ANALYSIS_DATA_PATH": tmp, "ANALYSIS_PATH": tmp}):
        for var in ("LAW_JOB_HOME", "LAW_CRAB_JOB_NUMBER", "X509_USER_PROXY"):
            os.environ.pop(var, None)
        yield tmp


def make_task(cls=CrabResProducer, **params):
    params.setdefault("version", unique_version())
    params.setdefault("period", "Run3_2022")
    params.setdefault("workflow", "local")
    return cls(**params)


def crab_config(task, tmp):
    """Run the real `crab_job_config` on law's own CRAB config object, filled the way
    law's CrabWorkflowProxy.create_job_file fills it before calling the hook."""
    factory = lc.FLAFCrabJobFileFactory(dir=tmp, mkdtemp=False, cleanup=False)
    config = factory.get_config()
    config.input_files = {}
    config.output_files = []
    config.render_variables = {}
    config.custom_content = []
    config.request_name = task.crab_request_name({0: [0]}).replace(".", "_")
    return task.crab_job_config(config, [0], [[0]])


def htcondor_config(task, tmp):
    """Run the real `htcondor_job_config` on law's own HTCondor config object."""
    factory = lc.CERNHTCondorJobFileFactory(dir=tmp, mkdtemp=False, cleanup=False)
    config = factory.get_config()
    config.input_files = {}
    config.output_files = {}
    config.render_variables = {}
    config.custom_content = []
    return task.htcondor_job_config(config, 0, [0])


def submit_crab(crab_cfg=None, **params):
    """(numCores, maxMemoryMB, config) of a CRAB submission of a fresh task."""
    with flaf_env(crab_cfg) as tmp:
        config = crab_config(make_task(**params), tmp)
        return config.crab.JobType.numCores, config.crab.JobType.maxMemoryMB, config


def pset_threads(path):
    """The `process.options.numberOfThreads` a PSet declares, read without CMSSW."""
    with open(path) as f:
        tree = ast.parse(f.read())
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "numberOfThreads"
    ]
    if len(values) != 1:
        raise AssertionError(f"{path}: {len(values)} numberOfThreads settings")
    call = values[0]
    # the CRAB client refuses a numberOfThreads that is not a uint32 (CMSSWConfig.py)
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "uint32"
        and len(call.args) == 1
    ):
        raise AssertionError(
            f"{path}: numberOfThreads is not a uint32: {ast.dump(call)}"
        )
    return ast.literal_eval(call.args[0])


class TheCrabLimits(unittest.TestCase):
    def test_the_limits_are_the_crab_clients(self):
        """CRABClient ServerUtilities.MAX_MEMORY_PER_CORE / MAX_MEMORY_SINGLE_CORE and the
        numCores values JobType/CMSSWConfig.py accepts; checked against the client itself in
        TheCrabClientItself where cvmfs is mounted. Every other expectation in this file is
        derived from these constants."""
        self.assertEqual(lc.CRAB_MB_PER_CORE, 2500)
        self.assertEqual(lc.CRAB_MB_SINGLE_CORE, 3000)
        self.assertEqual(tuple(lc.CRAB_ALLOWED_CORES), (1, 2, 4, 8))

    def test_the_ceiling_is_the_clients_rule(self):
        for n_cores in lc.CRAB_ALLOWED_CORES:
            self.assertEqual(lc.crab_memory_ceiling(n_cores), ceiling(n_cores))


class WhatTheJobAsksFor(unittest.TestCase):
    def test_an_unset_request_gets_the_most_crab_grants_for_the_cores(self):
        for n_cpus in lc.CRAB_ALLOWED_CORES:
            with self.subTest(n_cpus=n_cpus):
                self.assertEqual(
                    lc.crab_resources(FAMILY, n_cpus, 0), (n_cpus, ceiling(n_cpus))
                )
        # the single-core job gets CRAB's single-core allowance, not one core's share
        self.assertEqual(lc.crab_resources(FAMILY, 1, 0), (1, lc.CRAB_MB_SINGLE_CORE))
        self.assertEqual(lc.crab_resources(FAMILY, 2, 0), (2, 2 * lc.CRAB_MB_PER_CORE))
        self.assertEqual(lc.crab_resources(FAMILY, 4, 0), (4, 4 * lc.CRAB_MB_PER_CORE))

    def test_zero_and_minus_one_both_mean_unset(self):
        for unset in (0, -1, None):
            with self.subTest(unset=unset):
                self.assertEqual(lc.crab_resources(FAMILY, 4, unset), (4, ceiling(4)))

    def test_core_counts_crab_rejects_are_never_submitted(self):
        """CRAB accepts only 1, 2, 4, 8; a computed 3 or 6 is refused at submit."""
        self.assertEqual(lc.crab_resources(FAMILY, 3, 0), (4, ceiling(4)))
        for n_cpus in (5, 6, 7, 8):
            with self.subTest(n_cpus=n_cpus):
                self.assertEqual(lc.crab_resources(FAMILY, n_cpus, 0), (8, ceiling(8)))

    def test_more_threads_than_crab_has_cores_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            lc.crab_resources(FAMILY, lc.CRAB_ALLOWED_CORES[-1] + 1, 0)
        msg = str(caught.exception)
        self.assertIn(FAMILY, msg)
        self.assertIn(f"n_cpus={lc.CRAB_ALLOWED_CORES[-1] + 1}", msg)

    def test_a_smaller_explicit_request_is_honoured(self):
        """The DSProd regression: the old floor raised every request back to n * 2500."""
        self.assertEqual(lc.crab_resources(FAMILY, 4, 6000), (4, 6000))
        self.assertEqual(lc.crab_resources(FAMILY, 1, 1000), (1, 1000))

    def test_an_explicit_request_is_honoured_exactly_at_every_core_count(self):
        for n_cores in lc.CRAB_ALLOWED_CORES:
            for memory in (1000, ceiling(n_cores) - 1, ceiling(n_cores)):
                with self.subTest(n_cores=n_cores, memory=memory):
                    self.assertEqual(
                        lc.crab_resources(FAMILY, n_cores, memory), (n_cores, memory)
                    )

    def test_memory_buys_the_cores_it_needs(self):
        """A single-threaded task asking 5000 MB: CRAB has no 1-core 5000 MB job."""
        self.assertEqual(lc.crab_resources(FAMILY, 1, 5000), (2, 5000))
        self.assertEqual(
            lc.crab_resources(FAMILY, 1, ceiling(2) + 1), (4, ceiling(2) + 1)
        )
        self.assertEqual(
            lc.crab_resources(FAMILY, 2, ceiling(4) + 1), (8, ceiling(4) + 1)
        )

    def test_the_single_core_allowance_does_not_buy_a_second_core(self):
        """DSProd bought cores in 2500 MB steps, so a 1-core 3000 MB job became 2 cores."""
        single = lc.CRAB_MB_SINGLE_CORE
        self.assertEqual(lc.crab_resources(FAMILY, 1, single), (1, single))
        self.assertEqual(lc.crab_resources(FAMILY, 1, single + 1), (2, single + 1))

    def test_the_largest_grantable_request_takes_eight_cores(self):
        top = lc.CRAB_ALLOWED_CORES[-1]
        self.assertEqual(
            lc.crab_resources(FAMILY, 1, ceiling(top)), (top, ceiling(top))
        )

    def test_an_unsatisfiable_request_is_an_error_not_a_clamp(self):
        """Silently shrinking it would turn a kill threshold into a dead branch."""
        too_much = ceiling(lc.CRAB_ALLOWED_CORES[-1]) + 1
        for n_cpus in (1, 4, 8):
            with self.subTest(n_cpus=n_cpus):
                with self.assertRaises(ValueError) as caught:
                    lc.crab_resources(FAMILY, n_cpus, too_much)
                msg = str(caught.exception)
                self.assertIn(FAMILY, msg)
                self.assertIn(str(too_much), msg)

    def test_a_value_that_cannot_be_megabytes_is_refused(self):
        for memory in (1, 10, 999):
            with self.subTest(memory=memory):
                with self.assertRaises(ValueError) as caught:
                    lc.crab_resources(FAMILY, 4, memory)
                self.assertIn(FAMILY, str(caught.exception))
                self.assertIn("MB", str(caught.exception))
        self.assertEqual(lc.crab_resources(FAMILY, 1, 1000), (1, 1000))

    def test_every_case_is_one_crab_accepts(self):
        """Every thread count CRAB can run, every request from 'unset' to one MB above the
        top: the answer is accepted by the client's rule, honours the request, gives the
        payload at least its threads, and buys no core it does not need."""
        top = ceiling(lc.CRAB_ALLOWED_CORES[-1])
        failures = []
        n_checked = 0
        for n_cpus in range(1, lc.CRAB_ALLOWED_CORES[-1] + 1):
            for memory in itertools.chain((0, -1), range(1000, top + 2)):
                n_checked += 1
                case = (n_cpus, memory)
                if memory > top:
                    try:
                        got = lc.crab_resources(FAMILY, n_cpus, memory)
                    except ValueError:
                        continue
                    failures.append(f"{case}: {got} instead of an error")
                    continue
                cores, granted = lc.crab_resources(FAMILY, n_cpus, memory)
                fit = [
                    c
                    for c in lc.CRAB_ALLOWED_CORES
                    if c >= n_cpus and (memory <= 0 or memory <= ceiling(c))
                ]
                expected_granted = ceiling(cores) if memory <= 0 else memory
                if cores not in lc.CRAB_ALLOWED_CORES:
                    failures.append(f"{case}: {cores} cores are refused by CRAB")
                elif granted > ceiling(cores):
                    failures.append(
                        f"{case}: {granted} MB above CRAB's {ceiling(cores)}"
                    )
                elif cores < n_cpus:
                    failures.append(f"{case}: {cores} cores for {n_cpus} threads")
                elif granted != expected_granted:
                    failures.append(
                        f"{case}: {granted} MB, expected {expected_granted}"
                    )
                elif cores != fit[0]:
                    failures.append(f"{case}: {cores} cores, {fit[0]} suffice")
        self.assertGreater(n_checked, 8 * (top - 1000))
        self.assertEqual(failures[:10], [], f"{len(failures)} of {n_checked} cases")


class ThePsetMatchesTheCores(unittest.TestCase):
    def test_the_pset_declares_exactly_the_cores_requested(self):
        """The client refuses a task whose PSet threads differ from numCores."""
        for n_cpus, crab_memory in ((1, 0), (1, 5000), (3, 0), (2, ceiling(4) + 1)):
            with self.subTest(n_cpus=n_cpus, crab_memory=crab_memory):
                with flaf_env() as tmp:
                    task = make_task(n_cpus=n_cpus, crab_memory=crab_memory)
                    config = crab_config(task, tmp)
                    job = config.crab.JobType
                    self.assertEqual(
                        (job.numCores, job.maxMemoryMB),
                        lc.crab_resources(task.task_family, n_cpus, crab_memory),
                    )
                    self.assertTrue(job.psetName.startswith(tmp))
                    self.assertEqual(pset_threads(job.psetName), job.numCores)


class TheCrabJobConfig(unittest.TestCase):
    def test_crab_memory_reaches_the_crab_request(self):
        self.assertEqual(submit_crab(n_cpus=1, crab_memory=5000)[:2], (2, 5000))
        self.assertEqual(submit_crab(n_cpus=4, crab_memory=6000)[:2], (4, 6000))

    def test_an_unset_crab_memory_gets_the_ceiling(self):
        self.assertEqual(submit_crab(n_cpus=1)[:2], (1, lc.CRAB_MB_SINGLE_CORE))
        self.assertEqual(submit_crab(n_cpus=4)[:2], (4, ceiling(4)))

    def test_an_unsatisfiable_request_fails_the_submission_naming_the_task(self):
        too_much = ceiling(lc.CRAB_ALLOWED_CORES[-1]) + 1
        with self.assertRaises(ValueError) as caught:
            submit_crab(n_cpus=1, crab_memory=too_much)
        self.assertIn(CrabResProducer.get_task_family(), str(caught.exception))

    def test_the_task_is_not_changed_by_its_submission(self):
        """numCores is snapped to what CRAB accepts; the task keeps its own values."""
        with flaf_env() as tmp:
            task = make_task(n_cpus=3, crab_memory=6000)
            before = (task.n_cpus, task.crab_memory, task.task_id)
            config = crab_config(task, tmp)
            self.assertEqual(config.crab.JobType.numCores, 4)
            self.assertEqual((task.n_cpus, task.crab_memory, task.task_id), before)

    def test_the_retired_memory_key_fails_closed(self):
        """`crab.memory_mb_per_cpu` used to scale the request; read now it would mean
        nothing, and a silently ignored key is a production with the wrong memory."""
        with flaf_env({"memory_mb_per_cpu": 2500}) as tmp:
            task = make_task()
            with self.assertRaises(RuntimeError) as caught:
                task._crab_cfg()
            self.assertIn("memory_mb_per_cpu", str(caught.exception))
            with self.assertRaises(RuntimeError) as caught:
                crab_config(task, tmp)
            self.assertIn("memory_mb_per_cpu", str(caught.exception))

    def test_a_runtime_floor_that_does_not_parse_is_refused(self):
        """Without maxJobRuntimeMin CRAB kills every job at its own default."""
        for bad in ("sixty", "60.5", "1h", [60], None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    submit_crab({"min_runtime_min": bad}, max_runtime=2.0)
                self.assertIn("min_runtime_min", str(caught.exception))

    def test_the_runtime_floor_lifts_short_tasks(self):
        # 0.1 h is 6 min, too short to download and unpack the bundles
        self.assertEqual(
            submit_crab(max_runtime=0.1)[2].crab.JobType.maxJobRuntimeMin, 60
        )
        self.assertEqual(
            submit_crab({"min_runtime_min": 90}, max_runtime=0.1)[
                2
            ].crab.JobType.maxJobRuntimeMin,
            90,
        )
        self.assertEqual(
            submit_crab({"min_runtime_min": "45"}, max_runtime=0.1)[
                2
            ].crab.JobType.maxJobRuntimeMin,
            45,
        )

    def test_the_max_job_runtime_is_the_larger_of_floor_and_task_runtime(self):
        for max_runtime, floor in (
            (0.1, 60),
            (0.5, 60),
            (2.5, 60),
            (12.0, 60),
            (2.0, 200),
        ):
            with self.subTest(max_runtime=max_runtime, floor=floor):
                config = submit_crab(
                    {"min_runtime_min": floor}, max_runtime=max_runtime
                )[2]
                self.assertEqual(
                    config.crab.JobType.maxJobRuntimeMin,
                    max(int(max_runtime * 60), floor),
                )

    def test_the_log_tag_is_the_request_names_unique_suffix(self):
        """CRAB numbers the jobs of every CRAB task from 1; the waves and retries of one
        production stage their logs into one directory, so the tag tells them apart."""
        with flaf_env() as tmp:
            task = make_task()
            tags = set()
            for _ in range(3):
                config = crab_config(task, tmp)
                tag = config.render_variables["crab_log_tag"]
                self.assertTrue(config.request_name.endswith("_" + tag))
                self.assertEqual(tag, config.request_name.rsplit("_", 1)[-1])
                self.assertRegex(tag, r"^[0-9a-f]{8}$")
                tags.add(tag)
            self.assertEqual(len(tags), 3)

    def test_the_log_tag_stays_unique_for_a_long_version(self):
        """CRAB request names are limited to 100 characters. Truncated from the end, a long
        task family + version + period would cut through the uuid, and crab_log_tag (the
        text after the last `_`) would become the period's tail, shared by every CRAB task
        of the production -- one job's staged log would then overwrite another's."""
        with flaf_env() as tmp:
            task = make_task(version="v2610_" + "x" * 75)
            tags = set()
            for _ in range(3):
                config = crab_config(task, tmp)
                self.assertLessEqual(len(config.request_name), 100)
                tags.add(config.render_variables["crab_log_tag"])
            self.assertEqual(len(tags), 3)
            for tag in tags:
                self.assertRegex(tag, r"^[0-9a-f]{8}$")


class TheRequestBelongsToItsTask(unittest.TestCase):
    """--crab-memory is a per-task resource request, like max_runtime and n_cpus."""

    def test_crab_memory_is_excluded_from_req_and_from_branches(self):
        for cls in (lc.CrabWorkflow, CrabResProducer, CrabResConsumer):
            with self.subTest(cls=cls.__name__):
                self.assertIn("crab_memory", cls.exclude_params_req)
                self.assertIn("crab_memory", cls.exclude_params_branch)
        # law unites exclude_params_* across bases: CrabWorkflow's set must not have
        # replaced what HTCondorWorkflow and Task exclude
        self.assertLessEqual(
            {"max_runtime", "n_cpus", "tasks_per_job", "crab_memory"},
            CrabResConsumer.exclude_params_req,
        )
        self.assertIn("crab_memory", dict(CrabResProducer.get_params()))
        self.assertFalse(CrabResProducer.crab_memory.significant)

    def test_a_requiring_tasks_request_does_not_leak_through_req(self):
        with flaf_env() as tmp:
            consumer = make_task(CrabResConsumer, n_cpus=1, crab_memory=9000)
            self.assertNotIn("crab_memory", CrabResProducer.req_params(consumer))
            producer = CrabResProducer.req(consumer)
            self.assertEqual(producer.crab_memory, 0)
            config = crab_config(producer, tmp)
            self.assertEqual(
                (config.crab.JobType.numCores, config.crab.JobType.maxMemoryMB),
                (1, lc.CRAB_MB_SINGLE_CORE),
            )

    def test_an_explicit_pin_through_req_still_works(self):
        with flaf_env():
            consumer = make_task(CrabResConsumer, crab_memory=9000)
            self.assertEqual(
                CrabResProducer.req(consumer, crab_memory=4000).crab_memory, 4000
            )

    def cli(self, *extra):
        return [
            CrabResConsumer.get_task_family(),
            "--version",
            unique_version(),
            "--period",
            "Run3_2022",
            "--workflow",
            "local",
            *extra,
        ]

    def test_a_task_prefixed_cli_value_reaches_that_task(self):
        family = CrabResProducer.get_task_family()
        with flaf_env(), luigi.cmdline_parser.CmdlineParser.global_instance(
            self.cli(f"--{family}-crab-memory", "7000", "--crab-memory", "6000")
        ) as parser:
            root = parser.get_task_obj()
            self.assertEqual(root.crab_memory, 6000)
            self.assertEqual(CrabResProducer.req(root).crab_memory, 7000)

    def test_the_root_tasks_cli_value_stays_with_the_root_task(self):
        with flaf_env(), luigi.cmdline_parser.CmdlineParser.global_instance(
            self.cli("--crab-memory", "6000")
        ) as parser:
            root = parser.get_task_obj()
            self.assertEqual(root.crab_memory, 6000)
            self.assertEqual(CrabResProducer.req(root).crab_memory, 0)

    def test_the_worker_command_line_does_not_carry_it(self):
        """Only the submitting workflow reads it; a branch on a worker has no use for it."""
        with flaf_env():
            workflow = make_task(CrabResConsumer, crab_memory=9000)
            self.assertEqual(workflow.cli_args().get("--crab-memory"), "9000")
            branch = workflow.as_branch()
            self.assertNotIn("--crab-memory", branch.cli_args())
            self.assertNotIn("--crab-memory", workflow.req_branch(0).cli_args())


class TheSharedHTCondorPath(unittest.TestCase):
    """crab_memory and its CRAB validation must not reach an HTCondor submission."""

    @staticmethod
    def submit(tmp, crab_memory):
        task = make_task(CrabResCostedProducer, n_cpus=2, crab_memory=crab_memory)
        config = htcondor_config(task, tmp)
        return list(config.custom_content), dict(config.render_variables)

    def test_htcondor_job_config_ignores_crab_memory(self):
        with flaf_env() as tmp:
            unset = self.submit(tmp, 0)
            self.assertIn(("RequestMemory", 4000), unset[0])
            self.assertIn(("RequestCpus", 2), unset[0])
            for crab_memory in (5000, ceiling(lc.CRAB_ALLOWED_CORES[-1]) + 1, 10):
                with self.subTest(crab_memory=crab_memory):
                    # an unsatisfiable or nonsensical CRAB request is no HTCondor error
                    self.assertEqual(self.submit(tmp, crab_memory), unset)


class TheCrabClientItself(unittest.TestCase):
    """The constants against the CRAB client on cvmfs (skipped where it is not mounted)."""

    @classmethod
    def setUpClass(cls):
        try:
            with open(CRAB_CLIENT_LATEST) as f:
                version = f.read().strip()
        except OSError:
            raise unittest.SkipTest(
                f"no CRAB client ({CRAB_CLIENT_LATEST}): the CRAB_* constants are checked "
                "against their literal values only"
            )
        cls.lib = os.path.join(CRAB_CVMFS_BASE, "crab-prod", version, "lib")

    def parse(self, *path):
        with open(os.path.join(self.lib, *path)) as f:
            return ast.parse(f.read())

    def test_the_memory_constants(self):
        values = {}
        for node in self.parse("ServerUtilities.py").body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], "id", None)
                if name in ("MAX_MEMORY_PER_CORE", "MAX_MEMORY_SINGLE_CORE"):
                    values[name] = ast.literal_eval(node.value)
        self.assertEqual(
            values,
            {
                "MAX_MEMORY_PER_CORE": lc.CRAB_MB_PER_CORE,
                "MAX_MEMORY_SINGLE_CORE": lc.CRAB_MB_SINGLE_CORE,
            },
        )

    def test_the_accepted_core_counts(self):
        accepted = None
        for node in ast.walk(self.parse("CRABClient", "JobType", "CMSSWConfig.py")):
            if (
                isinstance(node, ast.Compare)
                and getattr(node.left, "id", None) == "numPSetCores"
                and isinstance(node.ops[0], ast.NotIn)
            ):
                accepted = ast.literal_eval(node.comparators[0])
        self.assertIsNotNone(accepted, "the client's numCores check was not found")
        self.assertEqual(
            tuple(v for v in accepted if v is not None), tuple(lc.CRAB_ALLOWED_CORES)
        )

    def test_the_ceiling_is_the_expression_the_client_evaluates(self):
        expr = None
        for node in ast.walk(self.parse("CRABClient", "Commands", "submit.py")):
            if (
                isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "absMaxMemory"
            ):
                expr = compile(ast.Expression(body=node.value), "submit.py", "eval")
        self.assertIsNotNone(expr, "the client's memory check was not found")
        for n_cores in lc.CRAB_ALLOWED_CORES:
            names = {
                "__builtins__": {"max": max},
                "MAX_MEMORY_SINGLE_CORE": lc.CRAB_MB_SINGLE_CORE,
                "MAX_MEMORY_PER_CORE": lc.CRAB_MB_PER_CORE,
                "nCores": n_cores,
            }
            with self.subTest(n_cores=n_cores):
                self.assertEqual(eval(expr, names), lc.crab_memory_ceiling(n_cores))


if __name__ == "__main__":
    unittest.main()
