#!/usr/bin/env python3
"""Every FLAF stage of an analysis, constructed over several of its eras at once.

A run over several eras (`law run <Stage> --era A,B,...`) is one workflow per stage, which reads
the settings it uses once for all its eras through `Task.common()`: construction raises when
`fs_default`, `crab`, the run token server, the path-cache host, a producer's resources or
`anaTuple_scheduling` differ between the eras. This check constructs each stage over the given
eras on the analysis's own configuration, with every workflow type it supports, and takes it
through what the scheduler does before anything runs (workflow proxy, resources, requirements,
outputs, completeness), so that an era configuration that cannot be run together with the
others, or a value read for one era on a task over several, shows up in the analysis's CI rather
than in a production.

No grid in the CI: the storage of every `fs_*` is redirected to a local directory through a
user_custom file, so `fs_default` is compared on the configuration as written, separately.
ROOT and Corrections are stand-ins; nothing here runs a payload.

Usage, from the analysis repository: python FLAF/test/test_era_dimension.py ERA ERA [ERA ...]
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import traceback
from unittest import mock

ana_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ana_path not in sys.path:
    sys.path.insert(0, ana_path)

_STAND_INS = [
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


for _name in _STAND_INS:
    if _name.split(".")[0] in ("ROOT", "Corrections") or not _importable(_name):
        sys.modules[_name] = mock.MagicMock()


def local_storage_user_custom(directory, setup):
    """A user_custom file sending every fs_* of the configuration to `directory`."""
    keys = sorted(
        {k for k in setup.global_params.keys() if k.startswith("fs_")} | {"fs_default"}
    )
    path = os.path.join(directory, "user_custom.yaml")
    with open(path, "w") as f:
        for key in keys:
            f.write(f"{key}: {os.path.join(directory, 'store', key)}\n")
    return path


def stages(lc, setup):
    """(label, class, extra parameters) of every FLAF stage."""
    from FLAF.AnaProd import tasks as anaprod
    from FLAF.Analysis import tasks as analysis
    from FLAF.test.hello_world_task import HelloWorldTask

    out = [
        (cls.__name__, cls, {})
        for cls in (
            anaprod.InputFileTask,
            anaprod.AnaTupleCostProbeTask,
            anaprod.AnaTupleFileTask,
            anaprod.AnaTupleFileListBuilderTask,
            anaprod.AnaTupleFileListTask,
            anaprod.AnaTupleMergeTask,
            analysis.HistTupleProducerTask,
            analysis.PreHistTupleProductionTask,
            analysis.HistFromNtupleProducerTask,
            analysis.HistMergerTask,
            analysis.HistPlotTask,
            HelloWorldTask,
        )
    ]
    for name, cfg in (setup.global_params.get("payload_producers") or {}).items():
        out.append(
            (
                f"AnalysisCacheTask[{name}]",
                analysis.AnalysisCacheTask,
                {"producer_to_run": name},
            )
        )
        if (cfg or {}).get("needs_aggregation", False):
            out.append(
                (
                    f"AnalysisCacheAggregationTask[{name}]",
                    analysis.AnalysisCacheAggregationTask,
                    {"producer_to_aggregate": name},
                )
            )
    return out


def check_stage(lc, label, cls, extra, eras, user_custom):
    """Problems of one stage over `eras`, for each workflow type it supports."""
    problems = []
    workflows = ["local"]
    if issubclass(cls, lc.HTCondorWorkflow):
        workflows.append("htcondor")
    if issubclass(cls, lc.CrabWorkflow):
        workflows.append("crab")
    for workflow in workflows:
        what = f"{label} --workflow {workflow}"
        try:
            task = cls(
                version="era_dimension",
                era=",".join(eras),
                workflow=workflow,
                user_custom=user_custom,
                **extra,
            )
            if task.effective_workflow != workflow:
                continue  # a stage that always runs locally
            proxy = task.workflow_proxy
            if workflow != "local" and not isinstance(proxy, lc.EraMergedProxy):
                problems.append(f"{what}: proxy {type(proxy).__name__} is not per era")
            task.process_resources()
            proxy.requires()
            proxy.output()
            if task.complete():
                problems.append(f"{what}: complete although nothing was produced")
            print(f"  OK: {what}")
        except Exception as e:
            traceback.print_exc()
            problems.append(f"{what}: {type(e).__name__}: {e}")
    return problems


def test_era_dimension(eras):
    from FLAF.Common.Setup import Setup
    from FLAF.run_tools import law_customizations as lc

    eras = sorted(set(eras))
    if len(eras) < 2:
        print("Give at least two eras", file=sys.stderr)
        return 1
    print(f"Analysis path: {ana_path}; eras: {', '.join(eras)}")
    problems = []
    setups = {
        era: Setup(ana_path=ana_path, era=era, law_run_version="t") for era in eras
    }
    for what, getter in lc.Task.common_settings:
        values = {era: getter(setup) for era, setup in setups.items()}
        if len(set(map(repr, values.values()))) > 1:
            problems.append(f"{what} differs between the eras: {values}")

    tmp = tempfile.mkdtemp(prefix="flaf_era_dimension_")
    try:
        os.environ["ANALYSIS_PATH"] = ana_path
        os.environ["ANALYSIS_DATA_PATH"] = os.path.join(tmp, "data")
        # as env.sh sets it: a bundle workflow checks that one of its bundles packs it
        os.environ["FLAF_ENVIRONMENT_PATH"] = os.path.join(ana_path, "soft", "flaf_env")
        user_custom = local_storage_user_custom(tmp, setups[eras[0]])
        for label, cls, extra in stages(lc, setups[eras[0]]):
            problems += check_stage(lc, label, cls, extra, eras, user_custom)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 80)
    if problems:
        print(f"FAILED: {len(problems)} problem(s) over {', '.join(eras)}:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"SUCCESS: every stage runs over {', '.join(eras)}")
    return 0


if __name__ == "__main__":
    sys.exit(test_era_dimension(sys.argv[1:]))
