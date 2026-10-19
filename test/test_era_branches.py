#!/usr/bin/env python3
"""The era as a branch dimension of FLAF workflows: flags, numbering, scope, maps, selections.

What is proven here, on the real FLAF and law classes over a scratch analysis with three eras
(Run3_2022, Run3_2023 and Run3_2022EE, each with a data dataset of its own and the shared MC
dataset TTto2L2Nu):

- class flags: a class takes several eras only with `allow_multi_era`, `era_branches` needs it,
  and every shipped stage class has both, `BundleTask` and the plain `Task` neither;
- numbering: `era id * ERA_BRANCH_STRIDE + local`, its inverse, its bounds, `era_of`, and the
  `python -m FLAF.Common.Setup --era-branch` command;
- era scope: on a workflow over two eras every per-era accessor raises `EraScopeError` (also
  through `getattr(task, name, None)`), `fs_default` and `cmssw_env` are the first era's, and a
  task over one era keeps today's values;
- settings that a run over several eras uses once must agree between its eras: construction,
  the cost parameters, the HTCondor proxy and the producer resources refuse them otherwise,
  naming the key and each era's value, while output file systems may differ per era;
- the dataset helpers raise on every call over two eras, in any order;
- `EraScopeError` passes through the excepts of the HTCondor proxy that swallow other errors;
- `--branches` encoding (one era: local numbers, and a number of no known era is beyond them;
  several: global numbers, closed ranges inside one era, every era selected, and at least one
  number in each era's map), with the decoded number in the message, and the refusal of
  `--branch` and `--<Task>-branch` with several eras on the command line;
- branch maps: the union of the single-era maps shifted by era id, unchanged for one era, AND
  over the eras' dynamic conditions with law's placeholder before (told by the condition, not by
  the map's content), and a selection given before the condition holds kept, for one era and
  for several;
- canonical branch tasks: the branch task of a multi-era workflow is the single-era branch
  task, the same luigi instance in every era set, and a local workflow yields each once; a
  workflow without era branches keeps law's branches, however large its own numbers;
- `with_eras`/`single` derive each instance once per root;
- per-task flags (`--tasks-per-job`, `--max-runtime`, `--n-cpus`, `--crab-memory`) reach each
  era's own workflow, its job chunking and its branches as in a run over that era alone;
- the AnaTuple cost store: one model per store and process, so two eras' workflows never drop
  each other's measurements, a second era with other scheduling settings is refused, and a
  multi-era workflow records each era's samples in its numbers;
- cost probes: each era's probe map is its block of the map over any era set.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import law  # noqa: E402
import luigi  # noqa: E402
import luigi.cmdline_parser  # noqa: E402
import luigi.task_register  # noqa: E402
import law.parser  # noqa: E402
from law.workflow.remote import JobData  # noqa: E402

# The task modules import ROOT, numpy, awkward, uproot and Corrections at module level, which
# the unit-test runner does not have and nothing tested here uses: stand-ins for these imports
# alone, and the previous entries are put back afterwards.
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
    from FLAF.Common.Setup import (
        ERA_BRANCH_STRIDE,
        ERA_BY_ID,
        ERA_IDS,
        UnknownEraError,
        era_branch,
        split_era_branch,
    )
    from FLAF.run_tools import law_customizations as lc
    from FLAF.run_tools.crab_watchdog import HEARTBEAT_DIR
    from FLAF.AnaProd import tasks as ana_prod
    from FLAF.AnaProd.CostModel import merged_params
    from FLAF.Analysis import tasks as analysis
    from FLAF.test.hello_world_task import HelloWorldTask
finally:
    for _name, _module in _replaced.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

STRIDE = ERA_BRANCH_STRIDE
ERA_A, ERA_B, ERA_C = "Run3_2022", "Run3_2023", "Run3_2022EE"
BASE_A, BASE_B, BASE_C = (ERA_IDS[e] * STRIDE for e in (ERA_A, ERA_B, ERA_C))

FILES = {
    "config/global.yaml": """
phys_model: M
channelSelection: [ eE ]
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
fs_default: @FS@
payload_producers:
  P:
    n_cpus: 2
    max_runtime: 3.0
    crab_memory: 2000
    save_as: root
    cmssw_env: false
    dependencies: null
""",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT ]\n  data: [ data ]\n",
    "config/processes.yaml": "TT:\n  datasets: [ TTto2L2Nu ]\n",
    f"config/{ERA_A}/processes.yaml": "data:\n  datasets: [ Data_A ]\n",
    f"config/{ERA_B}/processes.yaml": "data:\n  datasets: [ Data_B ]\n",
    f"config/{ERA_C}/processes.yaml": "data:\n  datasets: [ Data_C ]\n",
    f"config/{ERA_A}/datasets.yaml": "Data_A: {}\nTTto2L2Nu: {}\n",
    f"config/{ERA_B}/datasets.yaml": "Data_B: {}\nTTto2L2Nu: {}\n",
    f"config/{ERA_C}/datasets.yaml": "Data_C: {}\nTTto2L2Nu: {}\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
}

#: input files per (era, dataset), with what InputFileTask records of each
INPUT_FILES = {
    (ERA_A, "Data_A"): ["a0.root", "a1.root"],
    (ERA_A, "TTto2L2Nu"): ["tt0.root"],
    (ERA_B, "Data_B"): ["b0.root"],
    (ERA_B, "TTto2L2Nu"): ["tt0.root", "tt1.root", "tt2.root"],
    (ERA_C, "Data_C"): ["c0.root"],
    (ERA_C, "TTto2L2Nu"): ["tt0.root"],
}


class EraBrOneEra(lc.Task):
    def run(self):
        pass


class EraBrMultiOnly(lc.Task):
    allow_multi_era = True

    def run(self):
        pass


class EraBrBranchesOnly(lc.Task, law.LocalWorkflow):
    era_branches = True

    def create_branch_map(self):
        return {0: 0}

    def run(self):
        pass


class EraBrStage(lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow):
    """A stage whose branches carry the era: four branches per era."""

    allow_multi_era = True
    era_branches = True

    def create_branch_map(self):
        return self.cached_branch_map(
            lambda: {i: f"{self.only_era}:{i}" for i in range(4)}
        )

    def output(self):
        return self.local_target(f"out_{self.branch}.txt")

    def run(self):
        pass


class EraBrOtherStage(EraBrStage):
    pass


class EraBrMapOwner(lc.Task, law.LocalWorkflow):
    """Several eras, one map of its own (the StatInference card shape): no era branches."""

    allow_multi_era = True

    def create_branch_map(self):
        return {0: "A", 1: "B"}

    def output(self):
        return law.LocalFileTarget(
            os.path.join(os.environ["ANALYSIS_DATA_PATH"], "owner", str(self.branch))
        )

    def run(self):
        pass


class EraBrBigMapOwner(lc.Task, law.LocalWorkflow):
    """Several eras, one map of its own whose numbers reach past ERA_BRANCH_STRIDE: they are
    the map's own numbers, not era branch numbers."""

    allow_multi_era = True

    def create_branch_map(self):
        return {0: "small", BASE_A: "big", BASE_B + 3: "bigger"}

    def output(self):
        return law.LocalFileTarget(
            os.path.join(os.environ["ANALYSIS_DATA_PATH"], "big", str(self.branch))
        )

    def run(self):
        pass


class EraBrNoneData(lc.Task, law.LocalWorkflow):
    """A real one-branch map whose branch data is None, without a dynamic condition: what law
    returns as its placeholder, built by the class itself."""

    allow_multi_era = True
    era_branches = True

    def create_branch_map(self):
        return self.cached_branch_map(lambda: {0: None})

    def output(self):
        return self.local_target("done.txt")

    def run(self):
        pass


class EraBrGated(lc.Task, law.LocalWorkflow):
    """The same map behind a dynamic condition: a gate file per era, all of them required."""

    allow_multi_era = True
    era_branches = True

    def gate(self, era):
        return os.path.join(os.environ["ANALYSIS_DATA_PATH"], self.version, "gate", era)

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return all(os.path.exists(self.gate(era)) for era in self.era)

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(lambda: {0: None})

    @workflow_condition.output
    def output(self):
        return self.local_target("done.txt")

    def run(self):
        pass


class EraBrCostProbe(lc.Task, lc.HTCondorWorkflow, law.LocalWorkflow):
    """A single-era workflow that packs its jobs by cost, as AnaTupleFileTask does."""

    def create_branch_map(self):
        return {0: "a", 1: "b"}

    def output(self):
        return self.local_target(f"out_{self.branch}.txt")

    def cost_params(self):
        return merged_params({"parallel_jobs": 0})

    def branch_cost_map(self):
        return {0: (10.0, "job"), 1: (10.0, "job")}

    def cost_capacity_seconds(self):
        return 3600.0

    def record_job_durations(self, samples):
        return False

    def run(self):
        pass


#: every FLAF stage whose branches carry the era
SHIPPED_STAGES = [
    ana_prod.InputFileTask,
    ana_prod.AnaTupleFileTask,
    ana_prod.AnaTupleCostProbeTask,
    ana_prod.AnaTupleFileListBuilderTask,
    ana_prod.AnaTupleFileListTask,
    ana_prod.AnaTupleMergeTask,
    analysis.HistTupleProducerTask,
    analysis.PreHistTupleProductionTask,
    analysis.HistFromNtupleProducerTask,
    analysis.HistMergerTask,
    analysis.AnalysisCacheTask,
    analysis.HistPlotTask,
    analysis.AnalysisCacheAggregationTask,
    HelloWorldTask,
]


class _ScratchAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_era_branches_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ana = os.path.join(self.tmp, "ana")
        self.data = os.path.join(self.tmp, "data")
        self.fs = os.path.join(self.tmp, "fs")
        os.makedirs(self.ana)
        os.symlink(flaf_repo, os.path.join(self.ana, "FLAF"))
        for path, text in FILES.items():
            self.write(os.path.join(self.ana, path), text.replace("@FS@", self.fs))
        env = mock.patch.dict(
            os.environ, {"ANALYSIS_PATH": self.ana, "ANALYSIS_DATA_PATH": self.data}
        )
        env.start()
        self.addCleanup(env.stop)
        for name in ("LAW_JOB_HOME", "LAW_CRAB_JOB_NUMBER", "CRAB_Id"):
            os.environ.pop(name, None)
        # the command line is checked once per process
        checked = mock.patch.object(lc, "_era_cli_checked", False)
        checked.start()
        self.addCleanup(checked.stop)
        # law caches the parsed command line per process, FLAF its prefer-cli drop set per
        # parser identity and luigi task instances by their parameters: a real run has one
        # command line per process, this module many
        self.reset_cli_state()
        self.addCleanup(self.reset_cli_state)
        # task ids, the branch-map cache and the instance cache are process-wide: a version
        # of its own keeps this test apart from the others
        self.version = f"v_{self.id().rsplit('.', 1)[-1]}"

    @staticmethod
    def write(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def era_config(self, era, text):
        """Settings of one era, in the analysis' own `config/<era>/global.yaml`."""
        self.write(os.path.join(self.ana, "config", era, "global.yaml"), text)

    @staticmethod
    def reset_cli_state():
        law.parser._reset()
        lc.Task._req_prefer_cli_drop_cache.clear()
        luigi.task_register.Register.clear_instance_cache()

    def command_line(self, *args):
        """The command line of `law run`, as the process-wide parser sees it."""
        law.parser._reset()
        return luigi.cmdline_parser.CmdlineParser.global_instance(list(args))

    def from_command_line(self, *args):
        """The root task the command line builds, as `law run` does."""
        with self.command_line(*args) as cp:
            return cp.get_task_obj()

    def write_input_lists(self, era, version=None, file_info=None):
        ref = ana_prod.InputFileTask(version=version or self.version, era=era)
        for dataset_id, dataset_name in ref.iter_datasets():
            path = (
                ana_prod.InputFileTask.req(ref, branch=dataset_id, branches=())
                .output()
                .abspath
            )
            files = INPUT_FILES[(era, dataset_name)]
            content = {"input_files": files}
            if file_info is not None:
                content["file_info"] = {f: dict(file_info) for f in files}
            self.write(path, json.dumps(content))


# --------------------------------------------------------------------------------------
# T1: class flags
# --------------------------------------------------------------------------------------


class ClassFlags(_ScratchAnalysis):
    def test_a_class_without_allow_multi_era_takes_one_era(self):
        with self.assertRaises(ValueError) as caught:
            EraBrOneEra(version=self.version, era=f"{ERA_B},{ERA_A}")
        self.assertIn("EraBrOneEra takes one era", str(caught.exception))
        self.assertIn(f"{ERA_A},{ERA_B}", str(caught.exception))
        self.assertEqual(EraBrOneEra(version=self.version, era=ERA_A).era, (ERA_A,))

    def test_allow_multi_era_takes_several_eras(self):
        task = EraBrMultiOnly(version=self.version, era=f"{ERA_B},{ERA_A}")
        self.assertEqual(task.era, (ERA_A, ERA_B))
        other = EraBrMultiOnly(version=self.version, era=(ERA_A, ERA_B))
        self.assertEqual(task.task_id, other.task_id)
        self.assertIs(task, other)
        self.assertEqual(task.cli_args()["--era"], f"{ERA_A},{ERA_B}")

    def test_era_branches_needs_allow_multi_era(self):
        for era in (ERA_A, f"{ERA_A},{ERA_B}"):
            with self.subTest(era=era):
                with self.assertRaises(TypeError) as caught:
                    EraBrBranchesOnly(version=self.version, era=era)
                self.assertIn(
                    "era_branches needs allow_multi_era", str(caught.exception)
                )

    def test_every_shipped_stage_has_both_flags(self):
        for cls in SHIPPED_STAGES:
            with self.subTest(cls=cls.__name__):
                self.assertIs(cls.allow_multi_era, True)
                self.assertIs(cls.era_branches, True)

    def test_the_errors_are_not_attribute_errors(self):
        # getattr(task, name, default) and hasattr would swallow an AttributeError
        self.assertTrue(issubclass(lc.EraScopeError, RuntimeError))
        self.assertFalse(issubclass(lc.EraScopeError, AttributeError))
        self.assertTrue(issubclass(lc.EraDependentValueError, lc.EraScopeError))
        self.assertTrue(issubclass(lc.BranchSelectionError, ValueError))

    def test_bundles_and_the_plain_task_have_neither(self):
        for cls in (lc.BundleTask, lc.Task):
            with self.subTest(cls=cls.__name__):
                self.assertIs(cls.allow_multi_era, False)
                self.assertIs(cls.era_branches, False)
        with self.assertRaises(ValueError) as caught:
            lc.BundleTask(version=self.version, era=(ERA_A, ERA_B), flavour="core")
        self.assertIn("BundleTask takes one era", str(caught.exception))


# --------------------------------------------------------------------------------------
# numbering
# --------------------------------------------------------------------------------------


class Numbering(_ScratchAnalysis):
    def test_round_trip(self):
        for era in (ERA_A, ERA_C, ERA_B, "Run3_2026"):
            for local in (0, 1, 37000, STRIDE - 1):
                with self.subTest(era=era, local=local):
                    number = era_branch(era, local)
                    self.assertEqual(number, ERA_IDS[era] * STRIDE + local)
                    self.assertEqual(split_era_branch(number), (era, local))
        self.assertEqual(era_branch(ERA_B, 4), 7_000_004)

    def test_bounds(self):
        for local in (-1, STRIDE, STRIDE + 3):
            with self.subTest(local=local):
                with self.assertRaises(ValueError):
                    era_branch(ERA_A, local)
        with self.assertRaises(UnknownEraError):
            era_branch("Run3_2099", 0)
        self.assertEqual(split_era_branch(-1), (None, -1))
        self.assertEqual(split_era_branch(5), (None, 5))
        self.assertEqual(split_era_branch(12 * STRIDE + 1), (None, 1))

    def test_ids_and_eras_invert(self):
        self.assertEqual(len(ERA_BY_ID), len(ERA_IDS))
        for era, era_id in ERA_IDS.items():
            self.assertEqual(ERA_BY_ID[era_id], era)

    def test_era_of_and_local_branch(self):
        task = HelloWorldTask(version=self.version, era=(ERA_A, ERA_B))
        self.assertEqual(task.era_of(BASE_A), ERA_A)
        self.assertEqual(task.era_of(BASE_B + 3), ERA_B)
        self.assertEqual(task.local_branch(BASE_B + 3), 3)
        for number in (
            -1,
            0,
            3,
            ERA_IDS["Run2_2016_HIPM"] * STRIDE,
            BASE_C,
            12 * STRIDE,
        ):
            with self.subTest(number=number):
                with self.assertRaises(ValueError) as caught:
                    task.era_of(number)
                self.assertIn(f"{number} is not a branch number", str(caught.exception))

    def test_the_era_branch_command(self):
        """`python -m FLAF.Common.Setup --era-branch`, in a fresh interpreter. FLAF.Common
        imports ROOT at module level: where the runner has none, an empty module stands in
        for that import, as for the imports of this file."""
        env = dict(os.environ)
        paths = [flaf_parent]
        if not _importable("ROOT"):
            stub = os.path.join(self.tmp, "stub")
            self.write(os.path.join(stub, "ROOT.py"), "")
            paths.insert(0, stub)
        env["PYTHONPATH"] = os.pathsep.join(
            paths + [p for p in [env.get("PYTHONPATH")] if p]
        )

        def run(*args):
            return subprocess.run(
                [sys.executable, "-m", "FLAF.Common.Setup", *args],
                cwd=self.tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )

        done = run("--era-branch", ERA_B, "4")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip().splitlines()[-1], "7000004")
        for args in ((ERA_B, str(STRIDE)), ("Run3_2099", "1")):
            with self.subTest(args=args):
                refused = run("--era-branch", *args)
                self.assertNotEqual(refused.returncode, 0)


# --------------------------------------------------------------------------------------
# T2 / I13: era scope
# --------------------------------------------------------------------------------------

#: per-era reads of a task, by name, as a call on the task
PER_ERA_READS = {
    "only_era": lambda t: t.only_era,
    "setup": lambda t: t.setup,
    "datasets": lambda t: t.datasets,
    "global_params": lambda t: t.global_params,
    "fs_nanoAOD": lambda t: t.fs_nanoAOD,
    "fs_anaCache": lambda t: t.fs_anaCache,
    "fs_anaTuple": lambda t: t.fs_anaTuple,
    "fs_HistTuple": lambda t: t.fs_HistTuple,
    "fs_anaCacheTuple": lambda t: t.fs_anaCacheTuple,
    "fs_nnCacheTuple": lambda t: t.fs_nnCacheTuple,
    "fs_histograms": lambda t: t.fs_histograms,
    "fs_plots": lambda t: t.fs_plots,
    "iter_datasets": lambda t: list(t.iter_datasets()),
    "get_dataset_id": lambda t: t.get_dataset_id("TTto2L2Nu"),
    "get_dataset_name": lambda t: t.get_dataset_name(0),
    "get_nano_version": lambda t: t.get_nano_version("TTto2L2Nu"),
    "get_fs_nanoAOD": lambda t: t.get_fs_nanoAOD("TTto2L2Nu"),
    "store_parts": lambda t: t.store_parts(),
    "local_path": lambda t: t.local_path(),
    "local_target": lambda t: t.local_target("x"),
    "remote_log_dir_target": lambda t: t.remote_log_dir_target(),
    "law_job_home": lambda t: t.law_job_home(),
    "htcondor_output_directory": lambda t: t.htcondor_output_directory(),
    "crab_output_directory": lambda t: t.crab_output_directory(),
    "crab_request_name": lambda t: t.crab_request_name([]),
    "_heartbeat_dir_parts": lambda t: t._heartbeat_dir_parts(),
    "heartbeat_dir_target": lambda t: t.heartbeat_dir_target(),
    "heartbeat_target": lambda t: t.heartbeat_target(0),
}

#: the per-era reads that are attributes, read as `getattr(task, name, None)` does
PER_ERA_PROPERTIES = [
    "only_era",
    "setup",
    "datasets",
    "global_params",
    "fs_nanoAOD",
    "fs_anaCache",
    "fs_anaTuple",
    "fs_HistTuple",
    "fs_anaCacheTuple",
    "fs_nnCacheTuple",
    "fs_histograms",
    "fs_plots",
]


class EraScope(_ScratchAnalysis):
    def two_eras(self, **kwargs):
        return HelloWorldTask(version=self.version, era=(ERA_A, ERA_B), **kwargs)

    def test_every_per_era_read_raises_on_two_eras(self):
        for name, read in PER_ERA_READS.items():
            with self.subTest(read=name):
                # an instance of its own: what one read leaves behind must not answer another
                task = HelloWorldTask(
                    version=f"{self.version}_{name}", era=(ERA_A, ERA_B)
                )
                with self.assertRaises(lc.EraScopeError) as caught:
                    read(task)
                self.assertNotIsInstance(caught.exception, AttributeError)
                self.assertIn(
                    f"HelloWorldTask(era={ERA_A},{ERA_B})", str(caught.exception)
                )

    def test_a_second_read_raises_again(self):
        """The dataset helpers keep their first read: a failed one must not be kept."""
        for name in ("iter_datasets", "get_dataset_id", "get_dataset_name"):
            with self.subTest(read=name):
                task = HelloWorldTask(
                    version=f"{self.version}_{name}", era=(ERA_A, ERA_B)
                )
                for attempt in range(2):
                    with self.assertRaises(lc.EraScopeError, msg=f"read {attempt}"):
                        PER_ERA_READS[name](task)

    def test_every_dataset_helper_call_raises(self):
        """The three helpers share one table: a failed read by one of them must leave nothing
        that another one answers from, in any order and however often they are called.
        """
        names = ("iter_datasets", "get_dataset_id", "get_dataset_name")
        for first in names:
            with self.subTest(first=first):
                task = HelloWorldTask(
                    version=f"{self.version}_{first}", era=(ERA_A, ERA_B)
                )
                order = (first,) + tuple(n for n in names if n != first)
                for attempt in range(3):
                    for name in order:
                        with self.assertRaises(
                            lc.EraScopeError, msg=f"{name}, round {attempt}"
                        ):
                            PER_ERA_READS[name](task)
                self.assertIsNone(task._dataset_id_name_list)
                self.assertIsNone(task._dataset_id_name_dict)
                self.assertIsNone(task._dataset_name_id_dict)
                # each era's own workflow reads its own datasets
                self.assertEqual(
                    list(task.single(ERA_B).iter_datasets()),
                    [(0, "Data_B"), (1, "TTto2L2Nu")],
                )

    def test_getattr_with_a_default_does_not_hide_it(self):
        task = self.two_eras()
        for name in PER_ERA_PROPERTIES:
            with self.subTest(name=name):
                with self.assertRaises(lc.EraScopeError):
                    getattr(task, name, None)
                with self.assertRaises(lc.EraScopeError):
                    hasattr(task, name)

    def job_config(self):
        """A job config as law's HTCondor proxy starts it (create_job_file)."""
        config = lc.CERNHTCondorJobFileFactory(dir=self.tmp).get_config()
        config.input_files = {}
        config.output_files = {}
        config.render_variables = {}
        config.custom_content = []
        return config

    def test_the_worker_files_directory_raises_on_two_eras(self):
        """The directory HTCondor job inputs are staged to without spooling."""
        task = self.two_eras(bundle=True, htcondor_spool=False, workflow="htcondor")
        config = self.job_config()
        # bundles need a remote fs_default; what is under test is the path after them
        with mock.patch.object(
            HelloWorldTask, "_apply_bundle_render_variables", lambda self, c: None
        ):
            with self.assertRaises(lc.EraScopeError):
                task.htcondor_job_config(config, 1, [0])
        one = HelloWorldTask(
            version=self.version,
            era=ERA_A,
            bundle=True,
            htcondor_spool=False,
            workflow="htcondor",
        )
        # a remote directory is a WLCG target: its parts are what is compared
        with mock.patch.object(
            HelloWorldTask, "_apply_bundle_render_variables", lambda self, c: None
        ), mock.patch.object(
            HelloWorldTask, "remote_dir_target", lambda self, *parts, fs=None: parts
        ):
            config = one.htcondor_job_config(self.job_config(), 1, [0])
        self.assertEqual(
            config._worker_files_remote_dir, (self.version, "worker_files", ERA_A)
        )

    def test_fs_default_and_cmssw_env_are_the_first_eras(self):
        task = self.two_eras()
        first, second = task.setup_for(ERA_A), task.setup_for(ERA_B)
        self.assertIsNot(first, second)
        self.assertIs(task.fs_default, first.get_fs("default"))
        self.assertIsNot(task.fs_default, second.get_fs("default"))
        first.cmssw_env_ = {"era": ERA_A}
        second.cmssw_env_ = {"era": ERA_B}
        self.assertEqual(task.cmssw_env, {"era": ERA_A})

    def test_setup_for_names_the_era(self):
        task = self.two_eras()
        for era in (ERA_A, ERA_B):
            self.assertEqual(task.setup_for(era).global_params["era"], era)
        self.assertEqual(
            sorted(task.setup_for(ERA_B).datasets), ["Data_B", "TTto2L2Nu"]
        )
        with self.assertRaises(ValueError):
            task.setup_for(ERA_C)

    def test_one_era_keeps_todays_values(self):
        task = HelloWorldTask(version=self.version, era=ERA_A)
        v, name = self.version, "HelloWorldTask"
        local = os.path.join(self.data, v, name, ERA_A)
        self.assertEqual(task.only_era, ERA_A)
        self.assertEqual(task.store_parts(), (v, name, ERA_A))
        self.assertEqual(task.local_path(), local)
        self.assertEqual(task.local_target("x").abspath, os.path.join(local, "x"))
        self.assertEqual(task.htcondor_output_directory().abspath, local)
        self.assertEqual(task.crab_output_directory().abspath, local)
        # remote directories are WLCG targets: their parts are what is compared
        with mock.patch.object(
            HelloWorldTask, "remote_dir_target", lambda self, *parts, fs=None: parts
        ):
            self.assertEqual(task.remote_log_dir_target(), (v, "logs", name, ERA_A))
            self.assertEqual(
                task.heartbeat_dir_target(), (v, HEARTBEAT_DIR, name, ERA_A)
            )
        self.assertEqual(task._heartbeat_dir_parts(), [v, HEARTBEAT_DIR, name, ERA_A])
        self.assertRegex(
            task.crab_request_name([]), rf"^{name}_{v}_{ERA_A}_[0-9a-f]{{8}}$"
        )
        home, remove = task.law_job_home()
        self.assertTrue(remove)
        self.assertEqual(os.path.dirname(home), local)
        self.assertEqual(task.setup.global_params["era"], ERA_A)
        self.assertIs(task.setup, task.setup_for(ERA_A))
        self.assertIs(task.global_params, task.setup.global_params)
        self.assertEqual(sorted(task.datasets), ["Data_A", "TTto2L2Nu"])
        self.assertEqual(list(task.iter_datasets()), [(0, "Data_A"), (1, "TTto2L2Nu")])
        self.assertEqual(task.get_dataset_id("TTto2L2Nu"), 1)
        self.assertEqual(task.get_dataset_name(0), "Data_A")
        self.assertIs(task.fs_default, task.setup.get_fs("default"))
        self.assertEqual(task.fs_anaTuple.base, self.fs)
        self.assertEqual(
            task.as_branch(0).output().abspath,
            os.path.join(self.fs, v, name, ERA_A, "hello_world_done.txt"),
        )


# --------------------------------------------------------------------------------------
# T3 / I6: settings used once by a run over several eras
# --------------------------------------------------------------------------------------


class SettingsAgreement(_ScratchAnalysis):
    def assert_names(self, caught, key, *values):
        message = str(caught.exception)
        self.assertIsInstance(caught.exception, lc.EraScopeError)
        self.assertIn(key, message)
        for value in values:
            self.assertIn(str(value), message)
        for era in (ERA_A, ERA_B):
            self.assertIn(era, message)

    def test_a_differing_fs_default_is_refused_at_construction(self):
        other = os.path.join(self.tmp, "fs_b")
        self.era_config(ERA_B, f"fs_default: {other}\n")
        with self.assertRaises(lc.EraDependentValueError) as caught:
            HelloWorldTask(version=self.version, era=(ERA_A, ERA_B))
        self.assert_names(caught, "fs_default", self.fs, other)
        # each era alone is fine
        self.assertEqual(
            HelloWorldTask(version=self.version, era=ERA_B).fs_default.base, other
        )

    def test_differing_workflow_settings_are_refused_at_construction(self):
        cases = {
            "crab": ("{whitelist: [T2_CH_CERN]}", "{whitelist: [T2_DE_DESY]}"),
            "runTokenServer": (
                "{host: a.cern.ch, port: 1}",
                "{host: b.cern.ch, port: 1}",
            ),
            "WLCGFileSystem": (
                "{remotePathCacheHost: a.cern.ch, remotePathCachePort: 1}",
                "{remotePathCacheHost: b.cern.ch, remotePathCachePort: 1}",
            ),
        }
        values = {
            "crab": ("T2_CH_CERN", "T2_DE_DESY"),
            "runTokenServer": ("a.cern.ch", "b.cern.ch"),
            "WLCGFileSystem": ("a.cern.ch", "b.cern.ch"),
        }
        names = {
            "crab": "crab",
            "runTokenServer": "runTokenServer",
            "WLCGFileSystem": "WLCGFileSystem.remotePathCacheHost/Port",
        }
        for key, (a, b) in cases.items():
            with self.subTest(key=key):
                version = f"{self.version}_{key}"
                self.era_config(ERA_A, f"{key}: {a}\n")
                self.era_config(ERA_B, f"{key}: {b}\n")
                with self.assertRaises(lc.EraDependentValueError) as caught:
                    HelloWorldTask(version=version, era=(ERA_A, ERA_B))
                self.assert_names(caught, names[key], *values[key])
                HelloWorldTask(version=version, era=ERA_A)
                HelloWorldTask(version=version, era=ERA_B)

    def test_a_differing_path_cache_port_alone_is_refused(self):
        for era, port in ((ERA_A, 9100), (ERA_B, 9200)):
            self.era_config(
                era,
                "WLCGFileSystem: "
                f"{{remotePathCacheHost: a.cern.ch, remotePathCachePort: {port}}}\n",
            )
        with self.assertRaises(lc.EraDependentValueError) as caught:
            HelloWorldTask(version=self.version, era=(ERA_A, ERA_B))
        self.assert_names(
            caught, "WLCGFileSystem.remotePathCacheHost/Port", "9100", "9200"
        )
        HelloWorldTask(version=self.version, era=ERA_A)
        HelloWorldTask(version=self.version, era=ERA_B)

    def test_equal_settings_are_read_once(self):
        for era in (ERA_A, ERA_B):
            self.era_config(
                era,
                "crab: {whitelist: [T2_CH_CERN]}\n"
                "runTokenServer: {host: a.cern.ch, port: 1}\n",
            )
        task = HelloWorldTask(version=self.version, era=(ERA_A, ERA_B))
        self.assertEqual(task._crab_cfg(), {"whitelist": ["T2_CH_CERN"]})
        self.assertEqual(
            task.common(lambda s: s.global_params["runTokenServer"]["host"], "host"),
            "a.cern.ch",
        )

    def test_output_file_systems_may_differ(self):
        for key in ("fs_HistTuple", "fs_anaTuple"):
            with self.subTest(key=key):
                version = f"{self.version}_{key}"
                other = os.path.join(self.tmp, f"{key}_b")
                self.era_config(ERA_B, f"{key}: {other}\n")
                task = HelloWorldTask(version=version, era=(ERA_A, ERA_B))
                name = key[len("fs_") :]
                self.assertEqual(task.setup_for(ERA_A).get_fs(name).base, self.fs)
                self.assertEqual(task.setup_for(ERA_B).get_fs(name).base, other)
                self.assertEqual(getattr(task.single(ERA_A), key).base, self.fs)
                self.assertEqual(getattr(task.single(ERA_B), key).base, other)
                ana_prod.AnaTupleFileTask(version=version, era=(ERA_A, ERA_B))

    def test_differing_anatuple_scheduling_is_refused_where_it_is_read(self):
        self.era_config(ERA_B, "anaTuple_scheduling: {parallel_jobs: 10}\n")
        task = ana_prod.AnaTupleFileTask(
            version=self.version, era=(ERA_A, ERA_B), workflow="htcondor"
        )
        with self.assertRaises(lc.EraDependentValueError) as caught:
            task.cost_params()
        self.assert_names(caught, "anaTuple_scheduling", "parallel_jobs': 10")
        with self.assertRaises(lc.EraDependentValueError) as caught:
            task.workflow_proxy
        self.assert_names(caught, "anaTuple_scheduling")
        # one era reads its own
        single = ana_prod.AnaTupleFileTask(version=self.version, era=ERA_B)
        self.assertEqual(single.cost_params()["parallel_jobs"], 10)

    def test_a_differing_cmssw_env_flag_is_refused_where_it_is_read(self):
        self.era_config(ERA_B, "use_cmssw_env_AnaTupleProduction: true\n")
        task = ana_prod.AnaTupleFileTask(version=self.version, era=(ERA_A, ERA_B))
        with self.assertRaises(lc.EraDependentValueError) as caught:
            task.bundle_flavours
        self.assert_names(caught, "use_cmssw_env_AnaTupleProduction")

    def test_differing_producer_settings_are_refused_at_construction(self):
        base = {
            "n_cpus": 2,
            "max_runtime": 3.0,
            "crab_memory": 2000,
            "save_as": "root",
            "cmssw_env": False,
        }
        changed = {
            "n_cpus": 4,
            "max_runtime": 5.0,
            "crab_memory": 4000,
            "save_as": "json",
        }
        for key, value in changed.items():
            with self.subTest(key=key):
                version = f"{self.version}_{key}"
                settings = dict(base, dependencies=None)
                settings[key] = value
                self.era_config(
                    ERA_B, "payload_producers:\n  P: " + json.dumps(settings) + "\n"
                )
                with self.assertRaises(lc.EraDependentValueError) as caught:
                    analysis.AnalysisCacheTask(
                        version=version, era=(ERA_A, ERA_B), producer_to_run="P"
                    )
                self.assert_names(caught, f"payload_producers.P.{key}")
                self.assertIn(repr(base[key]), str(caught.exception))
                self.assertIn(repr(value), str(caught.exception))

    def test_a_differing_producer_cmssw_env_is_refused_where_it_is_read(self):
        self.era_config(
            ERA_B,
            "payload_producers:\n  P: "
            + json.dumps(
                {
                    "n_cpus": 2,
                    "max_runtime": 3.0,
                    "crab_memory": 2000,
                    "save_as": "root",
                    "cmssw_env": True,
                    "dependencies": None,
                }
            )
            + "\n",
        )
        task = analysis.AnalysisCacheTask(
            version=self.version, era=(ERA_A, ERA_B), producer_to_run="P"
        )
        with self.assertRaises(lc.EraDependentValueError) as caught:
            task.bundle_flavours
        self.assert_names(caught, "payload_producers.P.cmssw_env", False, True)

    def test_equal_producer_settings_construct(self):
        task = analysis.AnalysisCacheTask(
            version=self.version, era=(ERA_A, ERA_B), producer_to_run="P"
        )
        self.assertEqual(
            (task.n_cpus, task.max_runtime, task.crab_memory), (2, 3.0, 2000)
        )
        self.assertEqual(task.output_file_extension, "root")


# --------------------------------------------------------------------------------------
# T6: EraScopeError passes through the excepts that swallow other errors
# --------------------------------------------------------------------------------------


class ScopeErrorsPropagate(_ScratchAnalysis):
    def setUp(self):
        super().setUp()
        self.task = EraBrCostProbe(version=self.version, era=ERA_A, workflow="htcondor")
        self.proxy = self.task.workflow_proxy
        self.assertIs(type(self.proxy), lc._BundleAwareHTCondorWorkflowProxy)

    def test_cost_repacking(self):
        self.proxy.job_data.unsubmitted_jobs = OrderedDict([(1, [0]), (2, [1])])
        with mock.patch.object(
            self.task, "branch_cost_map", side_effect=lc.EraScopeError("costs")
        ):
            with self.assertRaises(lc.EraScopeError):
                self.proxy._cost_repack_unsubmitted()
        # any other error keeps the current grouping, as before
        with mock.patch.object(
            self.task, "branch_cost_map", side_effect=RuntimeError("costs")
        ):
            self.proxy._cost_repack_unsubmitted()
        self.assertEqual(
            self.proxy.job_data.unsubmitted_jobs, OrderedDict([(1, [0]), (2, [1])])
        )

    def finished_sample(self):
        manager = self.proxy.job_manager
        self.proxy._cost_own_jobs.add(1)
        self.proxy.job_data.jobs[1] = JobData.job_data(
            branches=[0], status=manager.RUNNING
        )
        self.proxy.harvest_job_durations()
        self.proxy.job_data.jobs[1]["status"] = manager.FINISHED

    def test_harvesting_job_durations(self):
        self.finished_sample()
        with mock.patch.object(
            self.task, "record_job_durations", side_effect=lc.EraScopeError("record")
        ) as record:
            with self.assertRaises(lc.EraScopeError):
                self.proxy.harvest_job_durations()
        record.assert_called_once()
        self.proxy._job_harvested.clear()
        self.finished_sample()
        with mock.patch.object(
            self.task, "record_job_durations", side_effect=RuntimeError("record")
        ):
            self.proxy.harvest_job_durations()

    def test_the_remote_log_location_of_a_submitted_group(self):
        submitted = (
            ["1.0"],
            OrderedDict([(1, {"log": "/afs/stdall.txt", "config": None})]),
        )
        with mock.patch.object(
            lc, "WLCGFileSystem", law.LocalFileSystem
        ), mock.patch.object(
            lc.BundleAwareHTCondorWorkflowProxyBase,
            "_submit_group",
            return_value=submitted,
        ):
            with mock.patch.object(
                self.task,
                "remote_log_dir_target",
                side_effect=lc.EraScopeError("logs"),
            ):
                with self.assertRaises(lc.EraScopeError):
                    self.proxy._submit_group(OrderedDict([(1, [0])]))
            with mock.patch.object(
                self.task, "remote_log_dir_target", side_effect=RuntimeError("logs")
            ):
                _, data = self.proxy._submit_group(OrderedDict([(1, [0])]))
        self.assertEqual(data[1]["log"], "/afs/stdall.txt")


# --------------------------------------------------------------------------------------
# I18: --branches encoding; T7: --branch on the command line
# --------------------------------------------------------------------------------------


class BranchSelection(_ScratchAnalysis):
    def stage(self, era, branches):
        return EraBrStage(version=self.version, era=era, branches=branches)

    def assert_refused(self, era, branches, *fragments):
        with self.assertRaises(lc.BranchSelectionError) as caught:
            self.stage(era, branches)
        self.assertIsInstance(caught.exception, ValueError)
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))

    def test_one_era_takes_its_own_numbers(self):
        self.assert_refused(
            ERA_A,
            (BASE_A + 1,),
            f"branch {BASE_A + 1} is branch 1 of {ERA_A}",
            f"--era {ERA_A} --branches 1",
        )
        self.assert_refused(
            ERA_A, ((0, BASE_B + 3),), f"branch {BASE_B + 2} is branch 2 of {ERA_B}"
        )
        self.assert_refused(ERA_A, ((BASE_A, None),), f"branch {BASE_A}")
        self.assertEqual(self.stage(ERA_A, (1, 3)).get_branch_map().keys(), {1, 3})

    def test_a_number_of_no_known_era_is_beyond_a_run_over_one_era(self):
        # 12 is no era id; 1 is Run2_2016_HIPM, which FLAF does not configure
        for number in (12 * STRIDE + 1, ERA_IDS["Run2_2016_HIPM"] * STRIDE + 1):
            with self.subTest(number=number):
                self.assert_refused(
                    ERA_A,
                    (number,),
                    f"branch {number} is beyond the {STRIDE} numbers of a run over "
                    "one era",
                )
        self.assert_refused(
            ERA_A,
            (BASE_C + 1,),
            f"branch {BASE_C + 1} is branch 1 of {ERA_C}",
            f"--era {ERA_C} --branches 1",
        )

    def test_several_eras_take_global_numbers(self):
        two = (ERA_A, ERA_B)
        self.assert_refused(two, (1,), "branch 1 is branch 1 of no era")
        self.assert_refused(two, ((BASE_A, None),), "open range")
        self.assert_refused(two, ((None, BASE_A + 2), (BASE_B,)), "open range")
        self.assert_refused(two, ((BASE_A, BASE_B + 2),), f"spans {ERA_A} and {ERA_B}")
        self.assert_refused(
            two,
            (BASE_A + 1, BASE_C + 1, BASE_B),
            f"branch {BASE_C + 1} is branch 1 of {ERA_C}, not of an era of this run",
        )
        self.assert_refused(two, (BASE_A + 1,), f"selects no branch of {ERA_B}")
        task = self.stage(two, ((BASE_A, BASE_A + 2), BASE_B + 3))
        self.assertEqual(task.get_branch_map().keys(), {BASE_A, BASE_A + 1, BASE_B + 3})
        self.assertEqual(task.local_selection(ERA_A), ((0, 2),))
        self.assertEqual(task.local_selection(ERA_B), ((3,),))

    def test_a_selection_naming_no_branch_of_an_eras_map_is_refused(self):
        """Every number decodes to an era of the run, so construction takes it; the map, four
        branches per era, holds none of one era's numbers, which law would widen to that
        era's whole map."""
        two = (ERA_A, ERA_B)
        for branches, era in (
            ((BASE_A + 100, BASE_B), ERA_A),
            ((BASE_A, BASE_B + 100), ERA_B),
            (((BASE_A + 10, BASE_A + 20), BASE_B + 1), ERA_A),
        ):
            with self.subTest(branches=branches):
                task = self.stage(two, branches)
                with self.assertRaises(lc.BranchSelectionError) as caught:
                    task.get_branch_map()
                self.assertIn(
                    f"--branches selects no branch of the 4 of {era}",
                    str(caught.exception),
                )

    def test_a_selection_with_numbers_beyond_an_eras_map_keeps_the_others(self):
        two = (ERA_A, ERA_B)
        for branches, expected in (
            ((BASE_A + 1, BASE_A + 100, BASE_B), {BASE_A + 1, BASE_B}),
            (
                ((BASE_A + 2, BASE_A + 10), BASE_B + 3),
                {BASE_A + 2, BASE_A + 3, BASE_B + 3},
            ),
        ):
            with self.subTest(branches=branches):
                task = self.stage(two, branches)
                self.assertEqual(task.get_branch_map().keys(), expected)
                self.assertEqual(task.get_branch_tasks().keys(), expected)

    def test_a_single_era_open_range_behaves_as_today(self):
        task = EraBrStage(version=self.version, era=ERA_A, branches="2:")
        self.assertEqual(task.branches, ((2, None),))
        self.assertEqual(task.get_branch_map().keys(), {2, 3})
        task = EraBrStage(version=self.version, era=ERA_A, branches=":2")
        self.assertEqual(task.get_branch_map().keys(), {0, 1})

    def test_a_selection_copied_into_a_workflow_of_the_same_numbering(self):
        two = self.stage((ERA_A, ERA_B), (BASE_A + 1, BASE_B))
        copied = EraBrOtherStage.req(two)
        self.assertEqual(copied.get_branch_map().keys(), {BASE_A + 1, BASE_B})
        one = self.stage(ERA_A, (1,))
        self.assertEqual(EraBrOtherStage.req(one).get_branch_map().keys(), {1})

    def test_without_era_branches_the_numbers_are_the_maps_own(self):
        task = EraBrMapOwner(version=self.version, era=(ERA_A, ERA_B), branches=(1,))
        self.assertEqual(task.get_branch_map(), {1: "B"})
        with self.assertRaises(lc.BranchSelectionError) as caught:
            EraBrMapOwner(version=self.version, era=(ERA_A, ERA_B), branches=(BASE_A,))
        # not the advice for a stage with era branches (`--era <era> --branches <local>`):
        # this workflow's branches need all of its eras
        self.assertIn("numbers its branches as its own map does", str(caught.exception))
        self.assertNotIn("--branches", str(caught.exception))


class BranchOnTheCommandLine(_ScratchAnalysis):
    def test_a_branch_with_several_eras_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.from_command_line(
                "EraBrStage",
                "--version",
                self.version,
                "--era",
                f"{ERA_A},{ERA_B}",
                "--branch",
                "3",
            )
        self.assertIn("a branch belongs to one era", str(caught.exception))
        self.assertIn(f"--era {ERA_A},{ERA_B}", str(caught.exception))

    def test_the_root_tasks_own_branch_flag_with_several_eras_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.from_command_line(
                "EraBrStage",
                "--version",
                self.version,
                "--era",
                f"{ERA_A},{ERA_B}",
                "--EraBrStage-branch",
                "3",
            )
        self.assertIn(
            f"--branch 3 with --era {ERA_A},{ERA_B}: a branch belongs to one era",
            str(caught.exception),
        )

    def test_the_root_tasks_own_branch_flag_of_one_era_is_taken(self):
        task = self.from_command_line(
            "EraBrStage",
            "--version",
            self.version,
            "--era",
            ERA_A,
            "--EraBrStage-branch",
            "3",
        )
        self.assertEqual((task.era, task.branch), ((ERA_A,), 3))
        self.assertEqual(task.branch_data, f"{ERA_A}:3")

    def test_a_branch_of_one_era_is_taken(self):
        task = self.from_command_line(
            "EraBrStage", "--version", self.version, "--era", ERA_A, "--branch", "3"
        )
        self.assertEqual((task.era, task.branch), ((ERA_A,), 3))
        self.assertEqual(task.branch_data, f"{ERA_A}:3")

    def test_several_eras_without_a_branch_are_taken(self):
        task = self.from_command_line(
            "EraBrStage", "--version", self.version, "--era", f"{ERA_A},{ERA_B}"
        )
        self.assertEqual((task.era, task.branch), ((ERA_A, ERA_B), -1))

    def test_a_branch_of_a_workflow_without_era_branches_is_taken(self):
        task = self.from_command_line(
            "EraBrMapOwner",
            "--version",
            self.version,
            "--era",
            f"{ERA_A},{ERA_B}",
            "--branch",
            "1",
        )
        self.assertEqual((task.era, task.branch), ((ERA_A, ERA_B), 1))
        self.assertEqual(task.branch_data, "B")


# --------------------------------------------------------------------------------------
# maps, the AND gate and selections given before it holds (I5)
# --------------------------------------------------------------------------------------


class BranchMaps(_ScratchAnalysis):
    def test_the_union_of_the_single_era_maps(self):
        task = HelloWorldTask(version=self.version, era=(ERA_A, ERA_B))
        self.assertEqual(task.get_branch_map(), {BASE_A: "hello", BASE_B: "hello"})
        stage = EraBrStage(version=self.version, era=(ERA_A, ERA_B))
        expected = {BASE_A + i: f"{ERA_A}:{i}" for i in range(4)}
        expected.update({BASE_B + i: f"{ERA_B}:{i}" for i in range(4)})
        self.assertEqual(stage.get_branch_map(), expected)
        inputs = ana_prod.InputFileTask(version=self.version, era=(ERA_A, ERA_B))
        self.assertEqual(
            inputs.get_branch_map(),
            {
                BASE_A: "Data_A",
                BASE_A + 1: "TTto2L2Nu",
                BASE_B: "Data_B",
                BASE_B + 1: "TTto2L2Nu",
            },
        )

    def test_one_era_is_the_builders_map(self):
        self.assertEqual(
            HelloWorldTask(version=self.version, era=ERA_A).get_branch_map(),
            {0: "hello"},
        )
        for cls in (ana_prod.InputFileTask, ana_prod.AnaTupleCostProbeTask):
            with self.subTest(cls=cls.__name__):
                task = cls(version=self.version, era=ERA_A)
                built = task._build_branch_map()
                self.assertEqual(built, {0: "Data_A", 1: "TTto2L2Nu"})
                self.assertEqual(task.get_branch_map(), built)
                self.assertEqual(
                    list(task.get_branch_map().items()), list(built.items())
                )

    def test_several_eras_without_era_branches_build_no_union(self):
        class EraBrUnflagged(lc.Task, law.LocalWorkflow):
            allow_multi_era = True

            def create_branch_map(self):
                return self.cached_branch_map(lambda: {0: 0})

            def run(self):
                pass

        with self.assertRaises(lc.EraScopeError) as caught:
            EraBrUnflagged(version=self.version, era=(ERA_A, ERA_B)).get_branch_map()
        self.assertIn("needs era_branches", str(caught.exception))


class DynamicConditions(_ScratchAnalysis):
    """AnaTupleFileTask's map is gated on the InputFileTask lists of its eras."""

    def anatuple(self, era, **kwargs):
        return ana_prod.AnaTupleFileTask(version=self.version, era=era, **kwargs)

    def produce(self, era, branch):
        outputs = self.anatuple(era, branch=branch).output()
        for target in outputs.values():
            self.write(target.abspath, "{}")

    def test_an_unmet_era_holds_back_the_whole_map(self):
        self.write_input_lists(ERA_A)
        self.produce(ERA_A, 0)
        self.assertTrue(self.anatuple(ERA_A, branch=0).complete())
        task = self.anatuple((ERA_A, ERA_B))
        self.assertEqual(task.get_branch_map(), {0: None})
        self.assertFalse(task.complete())
        placeholder = task.as_branch(0)
        self.assertEqual((placeholder.era, placeholder.branch), ((ERA_A, ERA_B), 0))
        self.assertIs(task.get_branch_tasks()[0], placeholder)
        target = placeholder.output()
        self.assertIsInstance(target, law.LocalFileTarget)
        self.assertTrue(target.path.endswith(".DYNAMIC_WORKFLOW_PLACEHOLDER"))
        self.assertFalse(target.exists())
        with self.assertRaises(lc.EraScopeError):
            placeholder.setup

        self.write_input_lists(ERA_B)
        self.assertEqual(
            task.get_branch_map().keys(),
            {
                BASE_A,
                BASE_A + 1,
                BASE_A + 2,
                BASE_B,
                BASE_B + 1,
                BASE_B + 2,
                BASE_B + 3,
            },
        )
        self.assertEqual(
            task.get_branch_map()[BASE_B + 3],
            ("TTto2L2Nu", "tt2.root", "anaTupleFile_2"),
        )

    def test_a_map_over_an_era_that_cannot_be_built_is_refused(self):
        self.write_input_lists(ERA_A)
        task = self.anatuple((ERA_A, ERA_B))
        # a condition taken as met over both eras while one era's is not
        task._dynamic_workflow_condition_met = True
        with self.assertRaises(RuntimeError) as caught:
            task.get_branch_map()
        self.assertIn(
            f"the branch map of {ERA_B} cannot be built yet", str(caught.exception)
        )

    def test_a_selection_given_before_the_condition_holds_is_kept(self):
        task = self.anatuple((ERA_A, ERA_B), branches=(BASE_A + 1, BASE_B))
        self.assertEqual(task.get_branch_map(), {0: None})
        self.write_input_lists(ERA_A)
        self.write_input_lists(ERA_B)
        self.assertEqual(task.get_branch_map().keys(), {BASE_A + 1, BASE_B})
        self.assertEqual(task.get_branch_tasks().keys(), {BASE_A + 1, BASE_B})

    def test_a_single_era_selection_given_before_the_condition_holds_is_kept(self):
        task = self.anatuple(ERA_A, branches=(1,))
        self.assertEqual(task.get_branch_map(), {0: None})
        self.write_input_lists(ERA_A)
        self.assertEqual(task.get_branch_map().keys(), {1})


class PlaceholderByCondition(_ScratchAnalysis):
    """law's placeholder is told by the class' dynamic condition, never by the map's content."""

    def test_a_real_none_map_without_a_condition_builds_the_union(self):
        self.assertEqual(
            EraBrNoneData(version=self.version, era=ERA_A).get_branch_map(), {0: None}
        )
        task = EraBrNoneData(version=self.version, era=(ERA_A, ERA_B))
        self.assertEqual(task.get_branch_map(), {BASE_A: None, BASE_B: None})
        for era in (ERA_A, ERA_B):
            base = ERA_IDS[era] * STRIDE
            self.assertIs(
                task.get_branch_tasks()[base],
                EraBrNoneData(version=self.version, era=era, branch=0),
            )

    def test_a_conditioned_none_map_is_laws_placeholder_until_every_gate_exists(self):
        task = EraBrGated(version=self.version, era=(ERA_A, ERA_B))
        self.write(task.gate(ERA_A), "")
        single_a = EraBrGated(version=self.version, era=ERA_A)
        self.assertEqual(single_a.get_branch_map(), {0: None})
        self.assertTrue(lc._branch_map_ready(single_a))
        self.assertEqual(task.get_branch_map(), {0: None})
        self.assertFalse(lc._branch_map_ready(task))
        placeholder = task.as_branch(0)
        self.assertEqual((placeholder.era, placeholder.branch), ((ERA_A, ERA_B), 0))
        self.assertTrue(
            placeholder.output().path.endswith(".DYNAMIC_WORKFLOW_PLACEHOLDER")
        )
        self.write(task.gate(ERA_B), "")
        self.assertTrue(lc._branch_map_ready(task))
        self.assertEqual(task.get_branch_map(), {BASE_A: None, BASE_B: None})


# --------------------------------------------------------------------------------------
# S10: canonical branch tasks
# --------------------------------------------------------------------------------------


class CanonicalBranches(_ScratchAnalysis):
    def test_the_branch_task_is_the_single_era_branch_task(self):
        ab = HelloWorldTask(version=self.version, era=(ERA_A, ERA_B))
        ac = HelloWorldTask(version=self.version, era=(ERA_A, ERA_C))
        a0 = HelloWorldTask(version=self.version, era=ERA_A, branch=0)
        b0 = HelloWorldTask(version=self.version, era=ERA_B, branch=0)
        c0 = HelloWorldTask(version=self.version, era=ERA_C, branch=0)
        self.assertIs(ab.get_branch_tasks()[BASE_A], a0)
        self.assertIs(ab.get_branch_tasks()[BASE_B], b0)
        self.assertIs(ac.get_branch_tasks()[BASE_A], a0)
        self.assertIs(ac.get_branch_tasks()[BASE_C], c0)
        self.assertIs(ab.as_branch(BASE_B), b0)
        self.assertIs(ab.req_branch(BASE_B), b0)
        self.assertEqual(b0.setup.global_params["era"], ERA_B)
        self.assertEqual(
            b0.output().abspath,
            os.path.join(
                self.fs, self.version, "HelloWorldTask", ERA_B, "hello_world_done.txt"
            ),
        )
        stage = EraBrStage(version=self.version, era=(ERA_A, ERA_B))
        for number, branch in stage.get_branch_tasks().items():
            era, local = split_era_branch(number)
            self.assertIs(
                branch, EraBrStage(version=self.version, era=era, branch=local)
            )
            self.assertEqual(branch.branch_data, f"{era}:{local}")

    def test_canonical_branches_carry_one_era_and_no_renamed_flag(self):
        for task in (
            HelloWorldTask(version=self.version, era=(ERA_A, ERA_B)),
            EraBrStage(version=self.version, era=(ERA_A, ERA_C)),
        ):
            for number, branch in task.get_branch_tasks().items():
                with self.subTest(task=task.task_family, number=number):
                    era, local = split_era_branch(number)
                    self.assertEqual((branch.era, branch.branch), ((era,), local))
                    # the removed parameter, read where luigi keeps its value
                    self.assertIsNone(branch.param_kwargs["period"])
                    args = branch.cli_args()
                    self.assertEqual(args["--era"], era)
                    self.assertEqual(args["--branch"], str(local))
                    self.assertFalse(
                        [name for name in args if "period" in name], msg=args
                    )

    def yielded_branches(self, task):
        """What law's local workflow yields when it runs (law/workflow/local.py): its branch
        tasks, as luigi's dynamic requirements."""
        proxy = task.workflow_proxy
        self.assertIs(type(proxy), law.workflow.local.LocalWorkflowProxy)
        # law records per process which workflows have yielded
        self.addCleanup(setattr, proxy, "_local_workflow_has_yielded", False)
        yielded = []
        for item in proxy.run():
            self.assertIsInstance(item, luigi.DynamicRequirements)
            yielded.extend(item.requirements)
        return yielded

    def test_a_local_workflow_yields_each_canonical_branch_once(self):
        ab = EraBrStage(version=self.version, era=(ERA_A, ERA_B), workflow="local")
        # the branches a run over each era alone yields
        expected = [
            EraBrStage(version=self.version, era=era, branch=local, workflow="local")
            for era in (ERA_A, ERA_B)
            for local in range(4)
        ]
        yielded = self.yielded_branches(ab)
        self.assertEqual(len(yielded), len(expected))
        self.assertEqual(
            sorted(t.task_id for t in yielded), sorted(t.task_id for t in expected)
        )
        for task in expected:
            self.assertEqual(sum(t is task for t in yielded), 1, msg=task.task_id)
        for task in yielded:
            self.assertIsNone(task.param_kwargs["period"])
            self.assertEqual(len(task.era), 1)
        # a second run of the same workflow yields nothing more
        self.assertEqual(self.yielded_branches(ab), [])
        # another era set yields the very same branches of the era it shares
        ac = EraBrStage(version=self.version, era=(ERA_A, ERA_C), workflow="local")
        yielded_ac = self.yielded_branches(ac)
        self.assertEqual(len(yielded_ac), 8)
        shared = [t for t in yielded_ac if t.era == (ERA_A,)]
        self.assertEqual(len(shared), 4)
        for task in shared:
            self.assertEqual(sum(t is task for t in expected), 1, msg=task.task_id)

    def test_a_local_workflow_yields_the_selected_canonical_branches(self):
        task = EraBrStage(
            version=self.version,
            era=(ERA_A, ERA_B),
            branches=(BASE_A + 1, BASE_B + 2),
            workflow="local",
        )
        yielded = self.yielded_branches(task)
        self.assertEqual(len(yielded), 2)
        for era, local in ((ERA_A, 1), (ERA_B, 2)):
            branch = EraBrStage(
                version=self.version, era=era, branch=local, workflow="local"
            )
            self.assertEqual(sum(t is branch for t in yielded), 1, msg=era)

    def test_a_number_below_the_stride_goes_to_law(self):
        ab = HelloWorldTask(version=self.version, era=(ERA_A, ERA_B))
        own = ab.req_branch(0)
        self.assertEqual((own.era, own.branch), ((ERA_A, ERA_B), 0))
        with self.assertRaises(lc.EraScopeError):
            own.setup


# --------------------------------------------------------------------------------------
# T4: per-task flags reach each era's own workflow
# --------------------------------------------------------------------------------------

FLAGS = [
    "--tasks-per-job",
    "2",
    "--max-runtime",
    "48",
    "--n-cpus",
    "4",
    "--crab-memory",
    "5000",
]


def per_task(task):
    return (task.tasks_per_job, task.max_runtime, task.n_cpus, task.crab_memory)


class PerTaskFlags(_ScratchAnalysis):
    def check(self, *flags, expected):
        version = ["--version", self.version]
        with self.command_line(
            "EraBrStage", *version, "--era", f"{ERA_A},{ERA_B}", *flags
        ) as cp:
            root = cp.get_task_obj()
            self.assertEqual(per_task(root), expected)
            single = root.single(ERA_A)
            self.assertEqual(per_task(single), expected)
            self.assertEqual(single.era, (ERA_A,))
            branch = root.get_branch_tasks()[BASE_A]
            self.assertEqual((branch.max_runtime, branch.n_cpus), expected[1:3])
            self.assertIs(branch, single.req_branch(0))
            proxy = root.workflow_proxy
            proxy._lay_out_eras()
            chunks = {
                era: list(sub.job_data.unsubmitted_jobs.values())
                for era, sub in proxy.era_subs().items()
            }
        with self.command_line("EraBrStage", *version, "--era", ERA_A, *flags) as cp:
            alone = cp.get_task_obj()
        self.assertIs(single, alone)
        return chunks

    def test_the_flags_of_the_command_line(self):
        chunks = self.check(*FLAGS, expected=(2, 48.0, 4, 5000))
        self.assertEqual(chunks, {ERA_A: [[0, 1], [2, 3]], ERA_B: [[0, 1], [2, 3]]})

    def test_a_per_family_tasks_per_job(self):
        flags = FLAGS[2:] + ["--EraBrStage-tasks-per-job", "3"]
        chunks = self.check(*flags, expected=(3, 48.0, 4, 5000))
        self.assertEqual(chunks, {ERA_A: [[0, 1, 2], [3]], ERA_B: [[0, 1, 2], [3]]})

    def test_the_merged_layout_numbers(self):
        with self.command_line(
            "EraBrStage", "--version", self.version, "--era", f"{ERA_A},{ERA_B}", *FLAGS
        ) as cp:
            proxy = cp.get_task_obj().workflow_proxy
            proxy._lay_out_eras()
        self.assertEqual(
            dict(proxy.job_data.unsubmitted_jobs),
            {
                BASE_A + 1: [BASE_A, BASE_A + 1],
                BASE_A + 2: [BASE_A + 2, BASE_A + 3],
                BASE_B + 1: [BASE_B, BASE_B + 1],
                BASE_B + 2: [BASE_B + 2, BASE_B + 3],
            },
        )


class SameFamilyInstances(_ScratchAnalysis):
    def test_single_and_with_eras_are_memoised_and_for_workflows_only(self):
        root = EraBrStage(version=self.version, era=(ERA_A, ERA_B))
        # luigi's instance cache alone gives equal arguments the same instance, so what
        # tells the memo is how often the instance is derived (law's full req each time)
        derive = lc.Task.req.__func__
        derived = []

        def counting_req(cls, inst, **kwargs):
            derived.append((kwargs.get("era"), kwargs.get("branches")))
            return derive(cls, inst, **kwargs)

        with mock.patch.object(EraBrStage, "req", classmethod(counting_req)):
            first = root.single(ERA_A)
            self.assertIs(root.single(ERA_A), first)
            self.assertIs(root.with_eras((ERA_A,)), first)
            selected = root.single(ERA_B, ((1, 3),))
            self.assertIs(root.with_eras((ERA_B,), ((1, 3),)), selected)
            self.assertIsNot(root.single(ERA_B), selected)
        self.assertEqual(
            derived, [((ERA_A,), ()), ((ERA_B,), ((1, 3),)), ((ERA_B,), ())]
        )
        self.assertIs(root.with_eras((ERA_B, ERA_A)), root)
        self.assertIs(
            root.single(ERA_B, ((1, 3),)),
            EraBrStage(version=self.version, era=ERA_B, branches=((1, 3),)),
        )
        with self.assertRaises(RuntimeError):
            root.get_branch_tasks()[BASE_A].single(ERA_A)
        # the memo is the root's own: it outlives luigi's instance cache
        luigi.task_register.Register.clear_instance_cache()
        self.assertIs(root.single(ERA_A), first)
        self.assertIsNot(EraBrStage(version=self.version, era=ERA_A), first)


class FlagGating(_ScratchAnalysis):
    def test_a_workflow_without_era_branches_keeps_laws_branches(self):
        task = EraBrMapOwner(version=self.version, era=(ERA_A, ERA_B))
        branch = task.req_branch(1)
        self.assertEqual((branch.era, branch.branch), ((ERA_A, ERA_B), 1))
        self.assertEqual(branch.branch_data, "B")
        self.assertEqual(task.get_branch_map(), {0: "A", 1: "B"})
        selected = EraBrMapOwner(
            version=self.version, era=(ERA_A, ERA_B), branches=(1,)
        )
        self.assertEqual(selected.get_branch_map(), {1: "B"})

    def test_a_number_past_the_stride_in_a_map_of_its_own_stays_laws(self):
        """Without era_branches a number is the map's own, however large: it names no era."""
        task = EraBrBigMapOwner(version=self.version, era=(ERA_A, ERA_B))
        branch_map = task.get_branch_map()
        self.assertEqual(branch_map, {0: "small", BASE_A: "big", BASE_B + 3: "bigger"})
        branches = task.get_branch_tasks()
        for number, data in branch_map.items():
            with self.subTest(number=number):
                branch = task.req_branch(number)
                self.assertEqual((branch.era, branch.branch), ((ERA_A, ERA_B), number))
                self.assertEqual(branch.branch_data, data)
                self.assertIs(branches[number], branch)
                self.assertIs(task.as_branch(number), branch)


# --------------------------------------------------------------------------------------
# T10 / D9: one cost model per store and process
# --------------------------------------------------------------------------------------

#: what InputFileTask records of every input file: enough events for a measurement
FILE_INFO = {"n_events": 10000, "size": 1000000}


class CostStorePerProcess(_ScratchAnalysis):
    def cost_entries(self, version):
        path = os.path.join(self.data, version, "AnaTupleCost", "cost_model.json")
        with open(path) as f:
            return {key.split("|")[1] for key in json.load(f)["entries"]}

    def test_two_eras_of_one_version_keep_each_others_measurements(self):
        for order in ((ERA_A, ERA_B), (ERA_B, ERA_A)):
            with self.subTest(order=order):
                version = f"{self.version}_{order[0]}"
                for era in order:
                    self.write_input_lists(era, version=version, file_info=FILE_INFO)
                tasks = [
                    ana_prod.AnaTupleFileTask(version=version, era=era) for era in order
                ]
                # both read the store before either writes, as both proxies do while
                # luigi schedules them
                models = [task.load_cost_model() for task in tasks]
                self.assertIs(models[0], models[1])
                for task in tasks:
                    self.assertTrue(task.record_job_durations([([0], 1000.0)]))
                self.assertEqual(self.cost_entries(version), {"Data_A", "Data_B"})

    def test_equal_settings_given_in_each_era_share_the_model(self):
        for era in (ERA_A, ERA_B):
            self.era_config(era, "anaTuple_scheduling: {default_sec_per_event: 0.5}\n")
            self.write_input_lists(era, file_info=FILE_INFO)
        tasks = [
            ana_prod.AnaTupleFileTask(version=self.version, era=era)
            for era in (ERA_A, ERA_B)
        ]
        models = [task.load_cost_model() for task in tasks]
        self.assertIs(models[0], models[1])
        self.assertEqual(models[0].params["default_sec_per_event"], 0.5)
        for task in tasks:
            self.assertTrue(task.record_job_durations([([0], 1000.0)]))
        self.assertEqual(self.cost_entries(self.version), {"Data_A", "Data_B"})

    def test_two_eras_with_other_scheduling_settings_are_refused(self):
        """One model per store is loaded with one era's settings: another era that asks for
        other ones would pack its jobs with the first era's."""
        self.era_config(ERA_B, "anaTuple_scheduling: {default_sec_per_event: 0.5}\n")
        default = merged_params()["default_sec_per_event"]
        self.assertNotEqual(default, 0.5)
        for order in ((ERA_A, ERA_B), (ERA_B, ERA_A)):
            with self.subTest(order=order):
                version = f"{self.version}_{order[0]}"
                for era in order:
                    self.write_input_lists(era, version=version, file_info=FILE_INFO)
                first, second = (
                    ana_prod.AnaTupleFileTask(version=version, era=era) for era in order
                )
                model = first.load_cost_model()
                # an era-scope error, which the proxies' cost-packing fallbacks re-raise
                with self.assertRaises(lc.EraScopeError) as caught:
                    second.load_cost_model()
                message = str(caught.exception)
                self.assertIn(first.cost_model_path(), message)
                self.assertIn("two anaTuple_scheduling settings", message)
                self.assertIn(f"'default_sec_per_event': {default!r}", message)
                self.assertIn("'default_sec_per_event': 0.5", message)
                self.assertIs(first.load_cost_model(), model)

    def test_several_eras_record_each_eras_samples_in_its_numbers(self):
        for era in (ERA_A, ERA_B):
            self.write_input_lists(era, file_info=FILE_INFO)
        task = ana_prod.AnaTupleFileTask(version=self.version, era=(ERA_A, ERA_B))
        spies = {}
        for era in (ERA_A, ERA_B):
            single = task.single(era)
            spies[era] = mock.Mock(wraps=single.record_job_durations)
            single.record_job_durations = spies[era]
        changed = task.record_job_durations(
            [
                ([BASE_A], 1000.0),
                ([BASE_B + 1, BASE_B + 2], 2000.0),
                ([BASE_A + 2, BASE_B], 1000.0),
            ]
        )
        self.assertTrue(changed)
        spies[ERA_A].assert_called_once_with([([0], 1000.0)])
        spies[ERA_B].assert_called_once_with([([1, 2], 2000.0)])
        self.assertEqual(self.cost_entries(self.version), {"Data_A", "TTto2L2Nu"})


# --------------------------------------------------------------------------------------
# I12: cost probes
# --------------------------------------------------------------------------------------


class CostProbeMaps(_ScratchAnalysis):
    def test_each_eras_map_is_its_block_in_every_era_set(self):
        probe = ana_prod.AnaTupleCostProbeTask
        singles = {
            era: probe(version=self.version, era=era).get_branch_map()
            for era in (ERA_A, ERA_B, ERA_C)
        }
        for eras in ((ERA_A, ERA_B), (ERA_A, ERA_C)):
            with self.subTest(eras=eras):
                union = probe(version=self.version, era=eras).get_branch_map()
                for era in eras:
                    base = ERA_IDS[era] * STRIDE
                    block = {
                        number - base: data
                        for number, data in union.items()
                        if split_era_branch(number)[0] == era
                    }
                    self.assertEqual(block, singles[era])
                self.assertEqual(len(union), sum(len(singles[e]) for e in eras))

    def test_a_shared_dataset_has_one_probe_target(self):
        task = ana_prod.AnaTupleCostProbeTask(version=self.version, era=(ERA_A, ERA_B))
        branches = task.get_branch_tasks()
        a, b = branches[BASE_A + 1], branches[BASE_B + 1]
        self.assertEqual((a.branch_data, b.branch_data), ("TTto2L2Nu", "TTto2L2Nu"))
        self.assertEqual(a.output().abspath, b.output().abspath)
        self.assertNotEqual(
            branches[BASE_A].output().abspath, branches[BASE_B].output().abspath
        )


if __name__ == "__main__":
    unittest.main()
