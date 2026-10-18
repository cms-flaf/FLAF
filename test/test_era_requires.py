#!/usr/bin/env python3
"""Workflow requirements of the FLAF stages over one era and over several.

`Task.workflow_requires()` is a dispatcher: over one era (or for a class without era
branches) it returns the stage's renamed body, `stage_workflow_requires()`; over several eras
it asks each era's single-era instance whether it is complete and lifts the bodies of the
incomplete ones into one requirement structure. Proven here, on the real FLAF and law classes
over a scratch analysis with Run3_2022 and Run3_2023:

* Over one era every stage with a body (AnaTupleFileTask ... AnalysisCacheAggregationTask)
  requires exactly what the base code required: the same task_ids under the same keys, for
  `--workflow local`, `htcondor`, `htcondor --bundle` and `crab`, with no input lists or merge
  plans (dynamic conditions not met) and with both. The base result is the JSON fixture
  `era_requires_base.json`, recorded with FLAF e8e42ed (the parent of the change) by copying
  this file into that checkout's test/ and running
  `python test/test_era_requires.py --write-fixture test/era_requires_base.json` there; the
  tests only read it.
* A class outside FLAF that overrides `workflow_requires()` and calls super() (the shape of
  StatInference's card tasks) still gets the bundles; AnaTupleFileListTask still has none.
* The lift drops nothing a single era requires (branch-task dicts keyed by local branch, the
  per-era bundles); workflows of one stage with the same other parameters become one
  instance over the eras that keeps the requirement's own per-task values; a one-era lift is
  the requirement itself.
* Selections are classified from the constructor's record, never from the selection law
  rewrites in place: a strict one against law's placeholder map is kept, one covering the
  whole map or none of it is whole, strict ones of two eras are one instance in the
  numbering over both (era id * 1 000 000 + the era's own number).
* An era whose part of a stage is complete requires nothing, not even the intermediates its
  outputs were made from, in both condition phases, also with a selection given before the
  gate; bundles come only for the pending eras.
* Over two eras a stage body reads per-era state and raises EraScopeError; the CRAB hook of a
  two-era root asks for nothing, over one era for that era's bundles.
* I3: what luigi schedules (`Worker.add` against a local scheduler, nothing run) for
  `--era A,B` is the union of what it schedules for `--era A` and for `--era B`, on a probe
  chain shaped like FLAF's (a gate, a dynamically conditioned upstream, a consumer of all of
  it and one of a strict subset), fresh, add-an-era before and after the gate, partially
  complete, one pending era with a strict subset, a selection given before the gate, and two
  pending eras with strict subsets; with the process caches cleared before each of the three
  (as separate processes would see them) and one after the other on shared instances.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import law  # noqa: E402
import luigi  # noqa: E402
import luigi.scheduler  # noqa: E402
import luigi.task_register  # noqa: E402
import luigi.worker  # noqa: E402

# The task modules import ROOT, numpy, awkward, uproot and Corrections at module level, and
# HistHelper (read by HistMergerTask's map) imports ROOT and numpy. The unit-test runner has
# none of them, and nothing tested here uses them: a stand-in is registered, for these
# imports alone, for each one that cannot be imported for real, and the previous entries are
# put back afterwards.
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
    from FLAF.run_tools import law_customizations as lc
    from FLAF.AnaProd import tasks as ana_prod
    from FLAF.Analysis import tasks as analysis
    from FLAF.test.hello_world_task import HelloWorldTask
finally:
    for _name, _module in _replaced.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


def _load_hist_helper():
    """FLAF.Common.HistHelper, whose findBinEntry HistMergerTask's map reads, loaded with a
    stand-in ROOT (it declares C++ to ROOT on import) and kept out of sys.modules: the tests
    that build such maps install it for their duration only."""
    name = "FLAF.Common.HistHelper"
    stand_ins = {"ROOT": mock.MagicMock()}
    if not _importable("numpy"):
        stand_ins["numpy"] = mock.MagicMock()
    saved = {key: sys.modules.get(key) for key in list(stand_ins) + [name]}
    sys.modules.update(stand_ins)
    try:
        with mock.patch.dict(os.environ, {"ANALYSIS_PATH": flaf_parent}):
            return importlib.import_module(name)
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module
        package = sys.modules["FLAF.Common"]
        if saved[name] is None and hasattr(package, "HistHelper"):
            delattr(package, "HistHelper")


_hist_helper = _load_hist_helper()

ERA_A = "Run3_2022"
ERA_B = "Run3_2023"

#: the analysis configuration; {store} is the scratch directory that fs_default points to
FILES = {
    "config/global.yaml": """
phys_model: M
channelSelection: [ eE ]
categories: [ res ]
custom_regions: regions
regions: [ OS ]
fs_default: {store}
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: [ v1, P1_c1, { name: v2, plot_task: false } ]
    fullResolution_variables: []
anaTuple_scheduling:
  probe_enabled: true
payload_producers:
  P1:
    columns: [ c1 ]
    dependencies: null
    n_cpus: 2
  P2:
    columns: [ n2 ]
    is_global: true
    needs_aggregation: true
    dependencies: [ P1 ]
corrections:
  norm:
    normCacheProducer: P2
bundles:
  core: {}
  soft: {}
  inputFileList: {}
  AnaTupleFileList: {}
""",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT, DY ]\n  data: [ data ]\n",
    "config/processes.yaml": "TT:\n  datasets: [ TTto2L2Nu ]\nDY:\n  datasets: [ DYto2L ]\n",
    f"config/{ERA_A}/processes.yaml": "data:\n  datasets: [ Data_A ]\n",
    f"config/{ERA_B}/processes.yaml": "data:\n  datasets: [ Data_B ]\n",
    f"config/{ERA_A}/datasets.yaml": "Data_A: {}\nTTto2L2Nu: {}\nDYto2L: {}\n",
    f"config/{ERA_B}/datasets.yaml": "Data_B: {}\nTTto2L2Nu: {}\nDYto2L: {}\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
    "config/plot/histograms.yaml": "v1: {}\nv2: {}\nP1_c1: {}\n",
}

#: NanoAOD input files per (era, dataset)
INPUT_FILES = {
    (ERA_A, "Data_A"): ["a0.root", "a1.root"],
    (ERA_A, "TTto2L2Nu"): ["tt0.root"],
    (ERA_A, "DYto2L"): ["dy0.root", "dy1.root"],
    (ERA_B, "Data_B"): ["b0.root"],
    (ERA_B, "TTto2L2Nu"): ["tt0.root", "tt1.root", "tt2.root"],
    (ERA_B, "DYto2L"): ["dy0.root"],
}

#: merge plans per (era, AnaTupleFileList dataset): (anaTuple inputs, outputs, events) per
#: merged file. Run3_2023 has an empty merged file, which no later stage reads.
MERGE_PLANS = {
    (ERA_A, "data"): [(["Data_A/anaTupleFile_0", "Data_A/anaTupleFile_1"], 1, 10)],
    (ERA_A, "TTto2L2Nu"): [(["TTto2L2Nu/anaTupleFile_0"], 1, 10)],
    (ERA_A, "DYto2L"): [(["DYto2L/anaTupleFile_0", "DYto2L/anaTupleFile_1"], 2, 20)],
    (ERA_B, "data"): [(["Data_B/anaTupleFile_0"], 1, 10)],
    (ERA_B, "TTto2L2Nu"): [
        (["TTto2L2Nu/anaTupleFile_0", "TTto2L2Nu/anaTupleFile_1"], 1, 10),
        (["TTto2L2Nu/anaTupleFile_2"], 1, 0),
    ],
    (ERA_B, "DYto2L"): [(["DYto2L/anaTupleFile_0"], 1, 10)],
}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def make_analysis(tmp):
    """The scratch analysis under `tmp`; returns its path. FLAF is the checkout under test."""
    ana = os.path.join(tmp, "ana")
    os.makedirs(ana)
    os.symlink(flaf_repo, os.path.join(ana, "FLAF"))
    for path, text in FILES.items():
        write(
            os.path.join(ana, path), text.replace("{store}", os.path.join(tmp, "store"))
        )
    return ana


def write_input_lists(version, era):
    """The InputFileTask lists of `era`, where InputFileTask writes them."""
    ref = ana_prod.InputFileTask(version=version, era=era)
    for dataset_id, dataset_name in ref.iter_datasets():
        target = ana_prod.InputFileTask.req(
            ref, branch=dataset_id, branches=()
        ).output()
        write(
            target.abspath,
            json.dumps({"input_files": INPUT_FILES[(era, dataset_name)]}),
        )


def write_merge_plans(version, era):
    """The AnaTupleFileListTask outputs (merge plans) of `era`."""
    ref = ana_prod.AnaTupleFileListTask(version=version, era=era)
    for branch, (dataset_name, _) in ref.get_branch_map().items():
        target = ana_prod.AnaTupleFileListTask.req(
            ref, branch=branch, branches=()
        ).output()
        plan = []
        n_out = 0
        for inputs, n_outputs, n_events in MERGE_PLANS[(era, dataset_name)]:
            outputs = [f"anaTuple_{n_out + i}.root" for i in range(n_outputs)]
            n_out += n_outputs
            plan.append(
                {"inputs": inputs, "outputs": outputs, "n_events": n_events, "runs": []}
            )
        write(target.abspath, json.dumps(plan))


def clear_process_caches():
    """Forget what earlier tasks of this process cached: instances, maps, list contents."""
    luigi.task_register.Register.clear_instance_cache()
    lc.Task._branch_map_cache.clear()
    ana_prod.InputFileTask.input_file_cache.clear()
    ana_prod.InputFileTask._WF_complete_ids.clear()
    analysis._anaTuple_outputs_cache.clear()


# ----------------------------------------------------------------------------- rename identity

#: (class name, module, extra parameters) of every stage with a requirement body
STAGES = [
    ("AnaTupleFileTask", ana_prod, {}),
    ("AnaTupleCostProbeTask", ana_prod, {}),
    ("AnaTupleFileListBuilderTask", ana_prod, {}),
    ("AnaTupleFileListTask", ana_prod, {}),
    ("AnaTupleMergeTask", ana_prod, {}),
    ("HistTupleProducerTask", analysis, {}),
    ("HistFromNtupleProducerTask", analysis, {}),
    ("HistMergerTask", analysis, {}),
    ("AnalysisCacheTask", analysis, {"producer_to_run": "P2"}),
    ("HistPlotTask", analysis, {}),
    ("AnalysisCacheAggregationTask", analysis, {"producer_to_aggregate": "P2"}),
]

FLAVOURS = {
    "local": {"workflow": "local"},
    "htcondor": {"workflow": "htcondor"},
    "bundle": {"workflow": "htcondor", "bundle": True},
    "crab": {"workflow": "crab"},
}

#: version per condition phase: placeholder (no input lists, no merge plans) and met (both)
PHASE_VERSIONS = {"placeholder": "v_requires_placeholder", "met": "v_requires_met"}


def prepare_phase(phase):
    version = PHASE_VERSIONS[phase]
    if phase == "met":
        for era in (ERA_A, ERA_B):
            write_input_lists(version, era)
            write_merge_plans(version, era)
    return version


def requirement_ids(reqs):
    """{key: sorted task_ids} of a workflow_requires() structure."""
    return {
        key: sorted(task.task_id for task in luigi.task.flatten(value))
        for key, value in reqs.items()
    }


def one_era_requirements(phase):
    """{stage/flavour: requirement_ids} of every stage over Run3_2022 in `phase`."""
    version = prepare_phase(phase)
    result = {}
    for stage, module, extra in STAGES:
        for flavour, params in FLAVOURS.items():
            task = getattr(module, stage)(version=version, era=ERA_A, **extra, **params)
            result[f"{stage}/{flavour}"] = requirement_ids(task.workflow_requires())
    return result


FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "era_requires_base.json"
)


def write_fixture(path):
    """Record one_era_requirements() of both phases with the FLAF this file sits in."""
    tmp = tempfile.mkdtemp(prefix="flaf_era_requires_fixture_")
    try:
        os.environ["ANALYSIS_PATH"] = make_analysis(tmp)
        os.environ["ANALYSIS_DATA_PATH"] = os.path.join(tmp, "data")
        sys.modules["FLAF.Common.HistHelper"] = _hist_helper
        clear_process_caches()
        fixture = {phase: one_era_requirements(phase) for phase in PHASE_VERSIONS}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(path, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


# ----------------------------------------------------------------------------- numbering

#: the design's numbering (era id = the C++ Period value), written out as the oracle
ERA_ID = {ERA_A: 5, ERA_B: 7}
STRIDE = 1_000_000


def global_number(era, local):
    return ERA_ID[era] * STRIDE + local


def numbers(branches):
    """The branch numbers a `branches` value selects (closed ranges only)."""
    out = set()
    for item in branches:
        if isinstance(item, int):
            out.add(item)
        elif len(item) == 1:
            out.add(item[0])
        else:
            out.update(range(item[0], item[1]))
    return out


def local_numbers(task, era):
    """The selection of `era` in `task`'s constructor selection, in the era's own numbers;
    the empty set stands for the whole map."""
    selected = numbers(task._initial_branches)
    if len(task.era) == 1:
        return selected
    base = ERA_ID[era] * STRIDE
    return {b - base for b in selected if base <= b < base + STRIDE}


def flat(reqs):
    return [task for value in reqs.values() for task in luigi.task.flatten(value)]


def requirement_key(task):
    """What a requirement asks for: a workflow by family, eras, other significant parameters
    and the branches it selects (as a selection while its map is law's placeholder)."""
    if not isinstance(task, law.BaseWorkflow) or not task.is_workflow():
        return (task.task_id,)
    branch_map = task.get_branch_map()
    if list(branch_map.items()) == [(0, None)]:
        selected = ("*",) + tuple(sorted(numbers(task._initial_branches)))
    else:
        selected = tuple(sorted(branch_map))
    return (task.get_task_family(), task.era, selected, sig_params(task))


# ----------------------------------------------------------------------------- probe chain
#
# A chain shaped like FLAF's: a gate (InputFileTask/AnaTupleFileListTask: one branch per era,
# its output says how large the downstream maps are), a dynamically conditioned upstream
# (AnaTupleMergeTask) whose maps follow the gate, a consumer of all of it (HistMergerTask /
# HistFromNtupleProducerTask) and one of a strict subset (HistPlotTask with plot_task: False).
# The consumers require the upstream with a parameter of its own (as producer_to_run), so a
# requirement built from the requirer instead of the requirement is visible in the work.

#: upstream branches per era (the gate's content)
N_UP = {ERA_A: 3, ERA_B: 4}


class _ReqProbe(lc.Task, law.LocalWorkflow):
    allow_multi_era = True
    era_branches = True

    def run(self):
        raise RuntimeError("scheduled only")

    def gate(self, **kwargs):
        return ReqGate.req(self, branches=(), **kwargs)


class ReqGate(_ReqProbe):
    def create_branch_map(self):
        return self.cached_branch_map(lambda: {0: self.only_era})

    def output(self):
        return self.local_target("gate.json")


class ReqUp(_ReqProbe):
    level = luigi.Parameter(default="raw")

    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return self.gate(branch=-1).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def _build_branch_map(self):
        with open(self.gate(branch=0).output().abspath) as f:
            return {i: i for i in range(json.load(f)["n"])}

    def stage_workflow_requires(self):
        reqs = super().stage_workflow_requires()
        reqs["gate"] = self.gate()
        return reqs

    @workflow_condition.output
    def output(self):
        return self.local_target(self.level, f"{self.branch}.txt")


class _ReqConsumer(_ReqProbe):
    @law.dynamic_workflow_condition
    def workflow_condition(self):
        return self.gate(branch=-1).complete()

    @workflow_condition.create_branch_map
    def create_branch_map(self):
        return self.cached_branch_map(self._build_branch_map)

    def up(self, **kwargs):
        return ReqUp.req(self, level="cooked", **kwargs)

    def stage_workflow_requires(self):
        reqs = super().stage_workflow_requires()
        if not self.gate(branch=-1).complete():
            reqs["gate"] = self.gate()
            reqs["up"] = self.up(branches=())
            return reqs
        reqs["up"] = self.up(branches=tuple(sorted(set(self.branch_map.values()))))
        return reqs

    @workflow_condition.output
    def output(self):
        return self.local_target(f"{self.branch}.txt")


class ReqDown(_ReqConsumer):
    def _build_branch_map(self):
        return dict(self.up(branch=-1, branches=()).get_branch_map())


class ReqPlot(_ReqConsumer):
    def _build_branch_map(self):
        up = self.up(branch=-1, branches=()).get_branch_map()
        return dict(enumerate(i for i in sorted(up) if i % 2 == 0))


class ReqRoot(lc.Task):
    """Two consumers of the upstream in one graph."""

    allow_multi_era = True

    def requires(self):
        return [ReqDown.req(self), ReqPlot.req(self)]

    def complete(self):
        return False

    def run(self):
        raise RuntimeError("scheduled only")


def sig_params(task):
    return tuple(
        sorted(
            (name, value)
            for name, value in task.to_str_params(only_significant=True).items()
            if name not in ("era", "branch", "branches")
        )
    )


def work_of(task):
    """What a pending task stands for: (family, era, local branch, other parameters) per
    incomplete selected branch, and (family, era, '*<selection>', ...) per era whose map is
    still law's placeholder (whole selection: '*')."""
    if not isinstance(task, law.BaseWorkflow) or not task.is_workflow():
        return set()
    family = task.get_task_family()
    work = set()
    branch_map = task.get_branch_map()
    if list(branch_map.items()) != [(0, None)]:
        for branch_task in task.get_branch_tasks().values():
            if not branch_task.complete():
                work.add(
                    (
                        family,
                        branch_task.era[0],
                        branch_task.branch,
                        sig_params(branch_task),
                    )
                )
        return work
    for era in task.era:
        selection = local_numbers(task, era)
        # the oracle builds the era's own instance without the code under test
        one = type(task).req(  # noqa: era-scope
            task,
            era=(era,),
            branch=-1,
            branches=tuple(sorted(selection)),
            _skip_task_excludes=True,
        )
        if list(one.get_branch_map().items()) == [(0, None)]:
            label = "*" + ",".join(map(str, sorted(selection)))
            work.add((family, era, label, sig_params(one)))
        else:
            work |= work_of(one)
    return work


def scheduled_work(roots):
    """The work luigi schedules for `roots`: `Worker.add` against a local scheduler, which
    checks completeness and expands the requirements of every incomplete task without
    running anything (luigi/worker.py)."""
    scheduler = luigi.scheduler.Scheduler()
    worker = luigi.worker.Worker(scheduler=scheduler, worker_processes=1)
    for root in roots:
        if not worker.add(root):
            raise RuntimeError(f"scheduling {root} failed (see the log above)")
    work = set()
    # the task objects luigi scheduled, by task_id: the scheduler keeps only their names
    for task_id in scheduler.task_list("PENDING", ""):
        work |= work_of(worker._scheduled_tasks[task_id])
    return work


class _Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_era_requires_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ana = make_analysis(self.tmp)
        env = mock.patch.dict(
            os.environ,
            {
                "ANALYSIS_PATH": self.ana,
                "ANALYSIS_DATA_PATH": os.path.join(self.tmp, "data"),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        # the command line is checked once per process
        checked = mock.patch.object(lc, "_era_cli_checked", False)
        checked.start()
        self.addCleanup(checked.stop)
        previous = sys.modules.get("FLAF.Common.HistHelper")
        sys.modules["FLAF.Common.HistHelper"] = _hist_helper
        self.addCleanup(self._restore_module, previous)
        # task instances, branch maps and list contents are cached per process, keyed by
        # parameters that do not name the scratch directory
        clear_process_caches()
        self.addCleanup(clear_process_caches)
        self.version = f"v_{self.id().rsplit('.', 1)[-1]}"

    @staticmethod
    def _restore_module(previous):
        if previous is None:
            sys.modules.pop("FLAF.Common.HistHelper", None)
        else:
            sys.modules["FLAF.Common.HistHelper"] = previous


# ----------------------------------------------------------------------------- tests


class OneEraRequirementsAreTheBaseCodes(_Scratch):
    """Over one era, `workflow_requires()` (now a dispatcher over the renamed stage bodies)
    returns what the base code returned, for every stage, workflow and condition phase.
    """

    def test_the_requirements_of_every_stage(self):
        with open(FIXTURE) as f:
            base = json.load(f)
        for phase in PHASE_VERSIONS:
            clear_process_caches()
            now = one_era_requirements(phase)
            self.assertEqual(sorted(now), sorted(base[phase]))
            for case, expected in base[phase].items():
                with self.subTest(phase=phase, case=case):
                    self.assertEqual(now[case], expected)

    def test_the_fixture_covers_bundles_and_both_phases(self):
        # what makes the comparison above able to see lost bundles at all
        with open(FIXTURE) as f:
            base = json.load(f)
        for phase in PHASE_VERSIONS:
            for stage, _, _ in STAGES:
                with self.subTest(phase=phase, stage=stage):
                    has_bundles = stage != "AnaTupleFileListTask"
                    for flavour in ("bundle", "crab"):
                        self.assertEqual(
                            "bundles" in base[phase][f"{stage}/{flavour}"], has_bundles
                        )
                    for flavour in ("local", "htcondor"):
                        self.assertNotIn("bundles", base[phase][f"{stage}/{flavour}"])
        self.assertNotEqual(
            base["placeholder"]["HistPlotTask/local"], base["met"]["HistPlotTask/local"]
        )


class CardShapedProbe(lc.Task, lc.HTCondorWorkflow, law.LocalWorkflow):
    """StatInference's CreateDatacardsTask shape: its own workflow_requires() on top of
    FLAF's, which carries the bundles."""

    bundle_flavours = ["core"]

    def create_branch_map(self):
        return {0: "card"}

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["hists"] = ReqGate.req(self, branches=())
        return reqs

    def output(self):
        return self.local_target("card.txt")

    def run(self):
        raise RuntimeError("not run here")


class CrabCardShapedProbe(
    lc.Task, lc.HTCondorWorkflow, lc.CrabWorkflow, law.LocalWorkflow
):
    bundle_flavours = ["core"]
    allow_multi_era = True

    def create_branch_map(self):
        return {0: "card"}

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["hists"] = ReqGate.req(self, branches=())
        return reqs

    def output(self):
        return self.local_target("card.txt")

    def run(self):
        raise RuntimeError("not run here")


class ExternalOverridesKeepTheirBundles(_Scratch):
    """A class outside FLAF that overrides workflow_requires() and calls super() still gets
    FLAF's bundles; AnaTupleFileListTask, which never had any, still has none."""

    def assert_bundles(self, task):
        reqs = task.workflow_requires()
        self.assertIn("hists", reqs)
        self.assertEqual(
            [(b.flavour, b.era) for b in reqs["bundles"]], [("core", (ERA_A,))]
        )

    def test_htcondor_with_bundles(self):
        self.assert_bundles(
            CardShapedProbe(
                version=self.version, era=ERA_A, workflow="htcondor", bundle=True
            )
        )

    def test_crab(self):
        task = CrabCardShapedProbe(version=self.version, era=ERA_A, workflow="crab")
        self.assert_bundles(task)
        self.assertEqual(
            [b.flavour for b in task.crab_workflow_requires()["bundles"]], ["core"]
        )

    def test_crab_with_bundles_on_htcondor(self):
        self.assert_bundles(
            CrabCardShapedProbe(
                version=self.version, era=ERA_A, workflow="htcondor", bundle=True
            )
        )

    def test_the_file_list_has_none(self):
        for params in (FLAVOURS["bundle"], FLAVOURS["crab"]):
            with self.subTest(**params):
                task = ana_prod.AnaTupleFileListTask(
                    version=self.version, era=ERA_A, **params
                )
                reqs = task.workflow_requires()
                self.assertEqual(list(reqs), ["AnaTupleFileListBuilderTask"])
                self.assertEqual(task.effective_workflow, "local")


class _TwoEras(_Scratch):
    def met(self, *eras):
        for era in eras:
            write_input_lists(self.version, era)
            write_merge_plans(self.version, era)
        clear_process_caches()

    def complete_hist_merger(self, era, skip=()):
        """HistMergerTask of `era` done (every variable merged but `skip`), its inputs gone."""
        task = analysis.HistMergerTask(version=self.version, era=era)
        for branch, branch_task in task.get_branch_tasks().items():
            if branch not in skip:
                write(branch_task.output().abspath, "merged")
        clear_process_caches()

    def root(self, cls=analysis.HistMergerTask, eras=(ERA_A, ERA_B), **kwargs):
        kwargs.setdefault("workflow", "local")
        return cls(version=self.version, era=eras, **kwargs)


class LiftKeepsEveryRequirement(_TwoEras):
    """`_lift` over two eras: nothing a single era requires is dropped, workflows of a stage
    with era branches become one instance over the eras that keeps the requirement's own
    values, a requirement of one era is that very instance."""

    def per_era(self, root):
        return {era: root.single(era).stage_workflow_requires() for era in root.era}

    def test_analysis_cache_containers(self):
        # {key: {local branch: branch task}} under two keys and a bundles list per era
        self.met(ERA_A, ERA_B)
        root = self.root(
            analysis.AnalysisCacheTask,
            producer_to_run="P2",
            workflow="htcondor",
            bundle=True,
        )
        per_era = self.per_era(root)
        lifted = root.workflow_requires()
        self.assertEqual(sorted(lifted), ["P1", "anaTuple", "bundles"])
        for key in lifted:
            with self.subTest(key=key):
                expected = [
                    t for era in per_era for t in luigi.task.flatten(per_era[era][key])
                ]
                self.assertEqual(len(lifted[key]), len(expected))
                self.assertGreater(len(expected), 4)
                for task in expected:
                    self.assertTrue(any(t is task for t in lifted[key]), task)
        self.assertEqual(
            sorted(b.era[0] for b in lifted["bundles"]), [ERA_A] * 3 + [ERA_B] * 3
        )

    def test_a_body_with_branch_tasks_a_list_and_a_workflow(self):
        root = self.root(analysis.AnalysisCacheTask, producer_to_run="P2")
        merge = {
            era: ana_prod.AnaTupleMergeTask(
                version=self.version, era=era, workflow="local"
            )
            for era in root.era
        }
        per_era = {
            era: {
                "anaTuple": {
                    0: merge[era].req_branch(0),
                    1: merge[era].req_branch(1),
                },
                "bundles": [
                    lc.BundleTask(version=self.version, era=era, flavour=f)
                    for f in ("core", "soft")
                ],
                "merge": merge[era],
            }
            for era in root.era
        }
        lifted = root._lift(per_era)
        self.assertEqual(len(lifted["anaTuple"]), 4)
        for era in root.era:
            for task in per_era[era]["anaTuple"].values():
                self.assertTrue(any(t is task for t in lifted["anaTuple"]))
        self.assertEqual(len(lifted["bundles"]), 4)
        self.assertEqual(len(lifted["merge"]), 1)
        (merged,) = lifted["merge"]
        self.assertEqual(merged.era, (ERA_A, ERA_B))
        self.assertEqual(merged.branches, ())
        self.assertEqual(sig_params(merged), sig_params(merge[ERA_A]))

    def test_a_stage_with_the_same_parameters_is_one_instance(self):
        self.met(ERA_A, ERA_B)
        root = self.root(analysis.HistTupleProducerTask)
        lifted = root.workflow_requires()
        (cache,) = lifted["analysisCache"]
        self.assertEqual((cache.era, cache.producer_to_run), ((ERA_A, ERA_B), "P1"))
        self.assertEqual(cache.branches, ())
        (aggregate,) = lifted["aggregateCache"]
        self.assertEqual(aggregate.era, (ERA_A, ERA_B))
        # Run3_2023 skips its empty merged file: a strict selection beside a whole one
        merges = sorted(lifted["anaTuple"], key=lambda t: t.era)
        self.assertEqual([t.era for t in merges], [(ERA_A,), (ERA_B,)])
        # Run3_2022's requirement covers its whole map, and over one era it is kept as it is
        self.assertEqual(
            set(merges[0].get_branch_map()),
            set(merges[0].req(merges[0], branches=()).get_branch_map()),
        )
        single_b = root.single(ERA_B).stage_workflow_requires()["anaTuple"]
        self.assertEqual(
            numbers(merges[1]._initial_branches), numbers(single_b.branches)
        )
        self.assertLess(
            len(numbers(single_b.branches)),
            len(single_b.req(single_b, branches=()).get_branch_map()),
        )

    def test_different_parameters_stay_apart(self):
        root = self.root(analysis.HistMergerTask)
        per_era = {
            ERA_A: {
                "hfn": analysis.HistFromNtupleProducerTask(
                    version=self.version,
                    era=ERA_A,
                    workflow="local",
                    n_files_per_job=20,
                )
            },
            ERA_B: {
                "hfn": analysis.HistFromNtupleProducerTask(
                    version=self.version, era=ERA_B, workflow="local", n_files_per_job=5
                )
            },
        }
        lifted = root._lift(per_era)
        self.assertEqual(
            sorted((t.era, t.n_files_per_job) for t in lifted["hfn"]),
            [((ERA_A,), 20), ((ERA_B,), 5)],
        )

    def per_task_requirement(self, era, **kwargs):
        return analysis.HistMergerTask(
            version=self.version,
            era=era,
            workflow="htcondor",
            tasks_per_job=3,
            max_runtime=7.0,
            n_cpus=5,
            **kwargs,
        )

    def test_a_two_era_lift_keeps_the_requirements_values(self):
        root = self.root(analysis.HistPlotTask, workflow="htcondor")
        lifted = root._lift(
            {era: {"merge": self.per_task_requirement(era)} for era in root.era}
        )
        (merged,) = lifted["merge"]
        self.assertEqual(merged.era, (ERA_A, ERA_B))
        self.assertEqual(
            (merged.tasks_per_job, merged.max_runtime, merged.n_cpus), (3, 7.0, 5)
        )

    def test_a_one_era_lift_is_the_requirement(self):
        self.met(ERA_A, ERA_B)
        root = self.root(analysis.HistPlotTask, workflow="htcondor")
        cases = {
            "whole": self.per_task_requirement(ERA_B),
            "range": self.per_task_requirement(ERA_B, branches=((0, 2),)),
            # the form the stage bodies build, tuple(set of branch numbers): HistPlotTask's
            # strict subset, HistMergerTask's whole map
            "numbers": self.per_task_requirement(ERA_B, branches=(0, 1)),
            "all numbers": self.per_task_requirement(ERA_B, branches=(0, 1, 2)),
        }
        for case, requirement in cases.items():
            with self.subTest(case=case):
                (lifted,) = root._lift({ERA_B: {"merge": requirement}})["merge"]
                self.assertEqual(lifted.task_id, requirement.task_id)
                self.assertIs(lifted, requirement)

    def test_the_run_over_two_eras_requires_what_one_era_requires(self):
        # HistPlotTask's strict HistMergerTask requirement (plot_task: False), Run3_2022 done
        self.met(ERA_A, ERA_B)
        plot_a = analysis.HistPlotTask(
            version=self.version, era=ERA_A, workflow="local"
        )
        for branch_task in plot_a.get_branch_tasks().values():
            for target in branch_task.output().values():
                write(target.abspath, "plot")
        clear_process_caches()
        alone = analysis.HistPlotTask(version=self.version, era=ERA_B, workflow="local")
        expected = alone.workflow_requires()["merge"]
        (lifted,) = self.root(analysis.HistPlotTask).workflow_requires()["merge"]
        self.assertEqual(lifted.task_id, expected.task_id)


class LiftClassifiesSelections(_TwoEras):
    """Whole or strict, from the selection the requirement was constructed with: a strict
    selection against a map that cannot be built yet is kept as given; one that covers the
    whole map, or none of it (law then takes the whole map), counts as whole; strict
    selections of two eras become one instance in the numbering over both."""

    def lift_one(self, *requirements):
        root = self.root()
        lifted = root._lift({r.era[0]: {"merge": r} for r in requirements})["merge"]
        return lifted

    def merger(self, era, branches):
        return analysis.HistMergerTask(
            version=self.version, era=era, workflow="local", branches=branches
        )

    def test_a_strict_selection_against_the_placeholder_is_kept(self):
        requirement = self.merger(ERA_B, (1, 2))
        # law has rewritten the live selection while the map was its placeholder
        self.assertEqual(requirement.get_branch_map(), {0: None})
        self.assertEqual(requirement.branches, ())
        (lifted,) = self.lift_one(requirement)
        self.assertEqual(lifted.era, (ERA_B,))
        self.assertEqual(numbers(lifted._initial_branches), {1, 2})

    def test_a_selection_of_the_whole_map_or_of_nothing_is_whole(self):
        # over one era the requirement is kept as it is, so the reading shows where it
        # decides something: with another era that needs the whole stage
        self.met(ERA_A, ERA_B)
        whole = set(self.merger(ERA_A, ()).get_branch_map())
        self.assertEqual(whole, {0, 1, 2})
        nothing = self.merger(ERA_A, (7,))
        # law's own reading of a selection that matches no branch
        self.assertEqual(set(nothing.get_branch_map()), whole)
        for branches in ((0, 1, 2), ((0, 3),), (7,)):
            with self.subTest(branches=branches):
                (lifted,) = self.lift_one(
                    self.merger(ERA_A, branches), self.merger(ERA_B, ())
                )
                self.assertEqual((lifted.era, lifted.branches), ((ERA_A, ERA_B), ()))

    def test_strict_selections_of_two_eras_are_one_instance(self):
        self.met(ERA_A, ERA_B)
        (lifted,) = self.lift_one(self.merger(ERA_A, (0, 1)), self.merger(ERA_B, (2,)))
        self.assertEqual(lifted.era, (ERA_A, ERA_B))
        self.assertEqual(
            numbers(lifted._initial_branches),
            {global_number(ERA_A, 0), global_number(ERA_A, 1), global_number(ERA_B, 2)},
        )
        tasks = lifted.get_branch_tasks()
        self.assertEqual(
            sorted((t.era, t.branch) for t in tasks.values()),
            [((ERA_A,), 0), ((ERA_A,), 1), ((ERA_B,), 2)],
        )

    def test_a_whole_and_a_strict_era_are_two_instances(self):
        self.met(ERA_A, ERA_B)
        lifted = self.lift_one(self.merger(ERA_A, ()), self.merger(ERA_B, (2,)))
        self.assertEqual(
            sorted(
                (t.era, tuple(sorted(numbers(t._initial_branches)))) for t in lifted
            ),
            [((ERA_A,), ()), ((ERA_B,), (2,))],
        )


class CompleteErasArePruned(_TwoEras):
    """An era whose part of a stage is complete requires nothing, in both condition phases:
    the HistFromNtupleProducerTask outputs it was merged from, removed since, are never
    requested; neither are its bundles."""

    def assert_only(self, reqs, era):
        self.assertTrue(flat(reqs))
        self.assertEqual({t.era for t in flat(reqs)}, {(era,)})

    def check(self, phase, params):
        self.met(ERA_A)
        if phase == "met":
            self.met(ERA_B)
        self.complete_hist_merger(ERA_A)
        # what Run3_2022's part would require if it were not pruned is gone
        self.assertFalse(
            analysis.HistFromNtupleProducerTask(
                version=self.version, era=ERA_A, workflow="local"
            ).complete()
        )
        root = analysis.HistMergerTask(
            version=self.version, era=(ERA_A, ERA_B), **params
        )
        self.assertFalse(root.complete())
        reqs = root.workflow_requires()
        self.assert_only(reqs, ERA_B)
        alone = analysis.HistMergerTask(version=self.version, era=ERA_B, **params)
        self.assertEqual(
            sorted(map(requirement_key, flat(reqs))),
            sorted(map(requirement_key, flat(alone.workflow_requires()))),
        )
        return reqs

    def test_both_phases_and_workflows(self):
        for phase in ("placeholder", "met"):
            for flavour in ("local", "bundle", "crab"):
                with self.subTest(phase=phase, flavour=flavour):
                    self.version = f"v_pruned_{phase}_{flavour}"
                    clear_process_caches()
                    reqs = self.check(phase, FLAVOURS[flavour])
                    self.assertEqual(
                        "bundles" in reqs, flavour in ("bundle", "crab"), sorted(reqs)
                    )

    def test_a_selection_given_before_the_gate(self):
        # Run3_2022 is complete for the selected variables only; luigi asks the root whether
        # it is complete first, and law then rewrites its live selection to the whole map
        self.met(ERA_A)
        self.complete_hist_merger(ERA_A, skip=(2,))
        root = self.root(
            branches=(
                global_number(ERA_A, 0),
                global_number(ERA_A, 1),
                global_number(ERA_B, 0),
            )
        )
        self.assertFalse(root.complete())
        self.assertEqual(root.branches, ())
        self.assert_only(root.workflow_requires(), ERA_B)


STAGE_CLASSES = [getattr(module, stage) for stage, module, _ in STAGES]
BUNDLED_STAGES = [cls for cls in STAGE_CLASSES if cls.bundle_flavours != []]


class TheRequirementPathOverTwoEras(_TwoEras):
    """Over two eras, the stage bodies read per-era state and so raise; the CRAB hook of the
    root asks for nothing (the lifted requirements carry each pending era's bundles)."""

    def extra(self, cls):
        return dict(next(e for s, m, e in STAGES if getattr(m, s) is cls))

    def test_a_stage_body_over_two_eras_raises(self):
        for phase in ("placeholder", "met"):
            if phase == "met":
                self.met(ERA_A, ERA_B)
            for cls in BUNDLED_STAGES:
                with self.subTest(phase=phase, stage=cls.__name__):
                    root = cls(
                        version=self.version,
                        era=(ERA_A, ERA_B),
                        workflow="htcondor",
                        bundle=True,
                        **self.extra(cls),
                    )
                    with self.assertRaises(lc.EraScopeError):
                        root.stage_workflow_requires()
        with self.subTest(stage="HistTupleProducerTask", bundle=False):
            clear_process_caches()
            self.version += "_placeholder"
            root = self.root(analysis.HistTupleProducerTask)
            with self.assertRaises(lc.EraScopeError):
                root.stage_workflow_requires()

    def test_the_crab_hook_of_a_two_era_root_asks_for_nothing(self):
        for cls in BUNDLED_STAGES:
            with self.subTest(stage=cls.__name__):
                root = cls(
                    version=self.version,
                    era=(ERA_A, ERA_B),
                    workflow="crab",
                    **self.extra(cls),
                )
                self.assertEqual(dict(root.crab_workflow_requires()), {})
                # over one era it still asks for that era's bundles
                one = cls(
                    version=self.version, era=ERA_A, workflow="crab", **self.extra(cls)
                )
                bundles = one.crab_workflow_requires()["bundles"]
                self.assertTrue(bundles)
                self.assertEqual({b.era for b in bundles}, {(ERA_A,)})

    def test_bundles_only_for_the_pending_eras(self):
        for flavour in ("bundle", "crab"):
            with self.subTest(flavour=flavour):
                params = FLAVOURS[flavour]
                both = HelloWorldTask(
                    version=self.version, era=(ERA_A, ERA_B), **params
                )
                self.assertEqual(
                    sorted(b.era for b in both.workflow_requires()["bundles"]),
                    [(ERA_A,), (ERA_B,)],
                )
        done = HelloWorldTask(version=self.version, era=ERA_A, branch=0)
        write(done.output().abspath, "hello")
        for flavour in ("bundle", "crab"):
            with self.subTest(flavour=flavour, done=ERA_A):
                clear_process_caches()
                root = HelloWorldTask(
                    version=self.version, era=(ERA_A, ERA_B), **FLAVOURS[flavour]
                )
                self.assertEqual(
                    [(b.flavour, b.era) for b in root.workflow_requires()["bundles"]],
                    [("core", (ERA_B,))],
                )
                proxy_reqs = root.workflow_proxy.requires()
                self.assertEqual([b.era for b in proxy_reqs["bundles"]], [(ERA_B,)])


class SchedulingIsTheUnionOfTheEras(_Scratch):
    """I3: the work luigi schedules for `--era A,B` is the union of the work for `--era A`
    and for `--era B`, after completeness pruning, in every state of an add-an-era or
    partial production, whether the three are scheduled as separate processes would (caches
    cleared) or one after the other in one process (shared instances, branch maps and law's
    rewritten selections).

    Work is (family, era, local branch, other significant parameters) per incomplete
    selected branch, and (family, era, '*<selection>', ...) per era whose map is law's
    placeholder."""

    def gate(self, era):
        task = ReqGate(version=self.version, era=era, branch=0)
        write(task.output().abspath, json.dumps({"n": N_UP[era]}))

    def produce(self, cls, era, skip=(), **kwargs):
        workflow = cls(version=self.version, era=era, **kwargs)
        for branch, task in workflow.get_branch_tasks().items():
            if branch not in skip:
                write(task.output().abspath, "done")

    def consume(self, era):
        shutil.rmtree(
            os.path.dirname(
                ReqUp(version=self.version, era=era, level="cooked", branch=0)
                .output()
                .abspath
            )
        )

    def roots(self, cls, eras, branches=None):
        kwargs = {} if branches is None else {"branches": branches}
        return [cls(version=self.version, era=eras, **kwargs)]

    def assert_union(self, cls, branches=None, expect=None):
        """`branches`: {era: local selection} of the roots (None: whole)."""

        def selection(eras):
            if branches is None:
                return None
            if len(eras) == 1:
                return tuple(sorted(branches[eras[0]]))
            return tuple(sorted(global_number(e, b) for e in eras for b in branches[e]))

        def work(*eras_list, fresh=True):
            out = set()
            for eras in eras_list:
                if fresh:
                    clear_process_caches()
                out |= scheduled_work(self.roots(cls, eras, selection(eras)))
            return out

        union = work((ERA_A,), (ERA_B,))
        both = work((ERA_A, ERA_B))
        self.assertEqual(both, union)
        # one process: the eras alone first, then both, on the same luigi instances
        clear_process_caches()
        work((ERA_A,), (ERA_B,), fresh=False)
        self.assertEqual(work((ERA_A, ERA_B), fresh=False), union)
        if expect is not None:
            self.assertEqual(
                {(family, era) for family, era, *_ in union}, expect, sorted(union)
            )
        return union

    def test_i_fresh(self):
        for cls in (ReqDown, ReqPlot, ReqRoot):
            with self.subTest(root=cls.__name__):
                self.assert_union(cls)

    def a_done_and_consumed(self):
        self.gate(ERA_A)
        self.produce(ReqUp, ERA_A, level="cooked")
        for cls in (ReqDown, ReqPlot):
            self.produce(cls, ERA_A)
        self.consume(ERA_A)

    def test_ii_placeholder_add_an_era(self):
        self.a_done_and_consumed()
        work = self.assert_union(
            ReqDown,
            expect={("ReqDown", ERA_B), ("ReqGate", ERA_B), ("ReqUp", ERA_B)},
        )
        self.assertIn(("ReqDown", ERA_B, "*"), {w[:3] for w in work})
        self.assert_union(ReqPlot)
        self.assert_union(ReqRoot)

    def test_iii_met_add_an_era(self):
        self.a_done_and_consumed()
        self.gate(ERA_B)
        self.assert_union(ReqDown, expect={("ReqDown", ERA_B), ("ReqUp", ERA_B)})
        self.assert_union(ReqPlot, expect={("ReqPlot", ERA_B), ("ReqUp", ERA_B)})
        self.assert_union(ReqRoot)

    def test_iv_one_era_partially_complete(self):
        self.gate(ERA_A)
        self.gate(ERA_B)
        self.produce(ReqUp, ERA_A, level="cooked")
        self.produce(ReqDown, ERA_A, skip=(2,))
        work = self.assert_union(ReqDown)
        self.assertEqual(
            {w[:3] for w in work if w[1] == ERA_A}, {("ReqDown", ERA_A, 2)}
        )
        self.assert_union(ReqRoot)

    def test_v_one_pending_era_with_a_strict_subset(self):
        self.gate(ERA_A)
        self.produce(ReqUp, ERA_A, level="cooked")
        self.produce(ReqPlot, ERA_A)
        self.consume(ERA_A)
        self.gate(ERA_B)
        work = self.assert_union(ReqPlot)
        self.assertEqual(
            {w[:3] for w in work if w[0] == "ReqUp"},
            {("ReqUp", ERA_B, 0), ("ReqUp", ERA_B, 2)},
        )

    def test_v_a_selection_given_before_the_gate(self):
        # Run3_2022 is complete for its selected branches only, Run3_2023 has no gate yet
        self.gate(ERA_A)
        self.produce(ReqUp, ERA_A, level="cooked")
        self.produce(ReqDown, ERA_A, skip=(2,))
        self.consume(ERA_A)
        work = self.assert_union(ReqDown, branches={ERA_A: (0, 1), ERA_B: (0, 1)})
        self.assertIn(("ReqDown", ERA_B, "*0,1"), {w[:3] for w in work})
        self.assertFalse([w for w in work if w[1] == ERA_A])

    def test_vi_two_pending_eras_with_strict_subsets(self):
        self.gate(ERA_A)
        self.gate(ERA_B)
        work = self.assert_union(ReqPlot)
        self.assertEqual(
            {w[:3] for w in work if w[0] == "ReqUp"},
            {
                ("ReqUp", ERA_A, 0),
                ("ReqUp", ERA_A, 2),
                ("ReqUp", ERA_B, 0),
                ("ReqUp", ERA_B, 2),
            },
        )
        self.assert_union(ReqRoot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-fixture", metavar="PATH")
    args, rest = parser.parse_known_args()
    if args.write_fixture:
        write_fixture(args.write_fixture)
    else:
        unittest.main(argv=[sys.argv[0]] + rest)
