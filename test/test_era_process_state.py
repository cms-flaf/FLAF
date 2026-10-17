#!/usr/bin/env python3
"""Process-level state that a law process handling several eras, versions or user_custom
files must keep apart (cms-flaf/FLAF#338).

A scratch analysis with two eras, Run3_2022 and Run3_2023, on top of FLAF's own configuration:
both have the MC dataset TTto2L2Nu, and each has its own data dataset (Data_A, Data_B). The
input-file lists are written where InputFileTask writes them, and the real task classes build
their branch maps from them.
"""

import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

# The task modules import ROOT, numpy, awkward, uproot and Corrections at module level. The
# unit-test runner has none of them (FLAF is checked out alone there), and nothing tested here
# uses them: a stand-in is registered, for these imports alone, for each one that cannot be
# imported for real (another test module may have left a placeholder ROOT), and the previous
# entries are put back afterwards.
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
    from FLAF.Common.Setup import ERA_IDS, Setup, UnknownEraError, known_eras
    from FLAF.AnaProd import tasks as ana_prod
    from FLAF.Analysis import tasks as analysis
finally:
    for _name, _module in _replaced.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

ERA_A = "Run3_2022"
ERA_B = "Run3_2023"

FILES = {
    "config/global.yaml": """
phys_model: M
channelSelection: [ eE ]
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
""",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT ]\n  data: [ data ]\n",
    "config/processes.yaml": """
TT:
  datasets: [ TTto2L2Nu ]
  processors:
    - name: counter
      module: era_process_state_processor
      class: Counter
      stages: [ AnaTupleMerge ]
""",
    f"config/{ERA_A}/processes.yaml": "data:\n  datasets: [ Data_A ]\n",
    f"config/{ERA_B}/processes.yaml": "data:\n  datasets: [ Data_B ]\n",
    f"config/{ERA_A}/datasets.yaml": "Data_A: {}\nTTto2L2Nu: {}\n",
    f"config/{ERA_B}/datasets.yaml": "Data_B: {}\nTTto2L2Nu: {}\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
}

PROCESSOR = """
class Counter:
    def __init__(self, global_params):
        self.era = global_params["era"]
"""

#: input files per (era, dataset): the two eras differ in every dataset
INPUT_FILES = {
    (ERA_A, "Data_A"): ["a0.root", "a1.root"],
    (ERA_A, "TTto2L2Nu"): ["tt0.root"],
    (ERA_B, "Data_B"): ["b0.root"],
    (ERA_B, "TTto2L2Nu"): ["tt0.root", "tt1.root", "tt2.root"],
}


class _ScratchAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_era_process_state_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ana = os.path.join(self.tmp, "ana")
        os.makedirs(self.ana)
        os.symlink(flaf_repo, os.path.join(self.ana, "FLAF"))
        for path, text in FILES.items():
            self.write(os.path.join(self.ana, path), text)
        modules = os.path.join(self.tmp, "modules")
        self.write(os.path.join(modules, "era_process_state_processor.py"), PROCESSOR)
        sys.path.insert(0, modules)
        self.addCleanup(sys.path.remove, modules)
        env = mock.patch.dict(
            os.environ,
            {
                "ANALYSIS_PATH": self.ana,
                "ANALYSIS_DATA_PATH": os.path.join(self.tmp, "data"),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        # task_ids, the branch-map cache and the instance cache are process-wide: a version
        # of its own keeps this test apart from the others
        self.version = f"v_{self.id().rsplit('.', 1)[-1]}"

    @staticmethod
    def write(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    def write_input_lists(self, era):
        ref = ana_prod.InputFileTask(version=self.version, era=era)
        for dataset_id, dataset_name in ref.iter_datasets():
            path = (
                ana_prod.InputFileTask.req(ref, branch=dataset_id, branches=())
                .output()
                .abspath
            )
            self.write(
                path, json.dumps({"input_files": INPUT_FILES[(era, dataset_name)]})
            )


class InputFileListsPerEra(_ScratchAnalysis):
    def test_each_era_has_its_own_verdict(self):
        self.write_input_lists(ERA_A)
        task_a = ana_prod.AnaTupleFileTask(version=self.version, era=ERA_A)
        task_b = ana_prod.AnaTupleFileTask(version=self.version, era=ERA_B)
        self.assertTrue(ana_prod.InputFileTask.WF_complete(task_a))
        self.assertFalse(ana_prod.InputFileTask.WF_complete(task_b))
        self.write_input_lists(ERA_B)
        self.assertTrue(ana_prod.InputFileTask.WF_complete(task_b))

    def test_each_version_has_its_own_verdict(self):
        self.write_input_lists(ERA_A)
        task = ana_prod.AnaTupleFileTask(version=self.version, era=ERA_A)
        other = ana_prod.AnaTupleFileTask(version=self.version + "_other", era=ERA_A)
        self.assertTrue(ana_prod.InputFileTask.WF_complete(task))
        self.assertFalse(ana_prod.InputFileTask.WF_complete(other))


class BuilderBranchSetsPerEra(_ScratchAnalysis):
    def builder_sets(self, era):
        builder = ana_prod.AnaTupleFileListBuilderTask(version=self.version, era=era)
        anaTuple_map = builder._get_anaTuple_map()
        return {
            dataset: builder._get_branch_set_for_dataset(dataset, group, anaTuple_map)
            for dataset, group in builder.create_branch_map().values()
        }

    def test_each_era_selects_from_its_own_map(self):
        self.write_input_lists(ERA_A)
        self.write_input_lists(ERA_B)
        expected_a = {"data": {0, 1}, "TTto2L2Nu": {2}}
        expected_b = {"data": {0}, "TTto2L2Nu": {1, 2, 3}}
        self.assertEqual(self.builder_sets(ERA_A), expected_a)
        self.assertEqual(self.builder_sets(ERA_B), expected_b)
        # and the first era is not disturbed by the second
        self.assertEqual(self.builder_sets(ERA_A), expected_a)

    def test_the_anatuple_map_is_the_eras_own(self):
        self.write_input_lists(ERA_A)
        self.write_input_lists(ERA_B)
        for era in (ERA_A, ERA_B):
            builder = ana_prod.AnaTupleFileListBuilderTask(
                version=self.version, era=era
            )
            files = [entry[1] for entry in builder._get_anaTuple_map().values()]
            datasets = sorted(
                ds for (e, ds) in INPUT_FILES if e == era and ds != "TTto2L2Nu"
            ) + ["TTto2L2Nu"]
            expected = [f for ds in datasets for f in INPUT_FILES[(era, ds)]]
            self.assertEqual(files, expected, era)

    def test_the_placeholder_is_never_cached(self):
        builder = ana_prod.AnaTupleFileListBuilderTask(
            version=self.version, era=ERA_A
        )
        self.assertNotIsInstance(builder._get_anaTuple_map(), dict)
        self.write_input_lists(ERA_A)
        self.assertEqual(self.builder_sets(ERA_A), {"data": {0, 1}, "TTto2L2Nu": {2}})


class AnaTupleOutputsPerUserCustom(_ScratchAnalysis):
    def test_each_user_custom_has_its_own_outputs(self):
        first = os.path.join(self.tmp, "first.yaml")
        second = os.path.join(self.tmp, "second.yaml")
        self.write(first, "# first\n")
        self.write(second, "# second\n")

        def all_branch_outputs(task):
            return {0: [task.user_custom]}

        with mock.patch.object(
            ana_prod.AnaTupleMergeTask, "all_branch_outputs", all_branch_outputs
        ):
            for user_custom in (first, second, first):
                ref = ana_prod.AnaTupleMergeTask(
                    version=self.version, era=ERA_A, user_custom=user_custom
                )
                self.assertEqual(
                    analysis._anaTuple_outputs(ref), {0: [user_custom]}, user_custom
                )


class ProcessorInstancesPerCall(_ScratchAnalysis):
    def test_every_call_gets_new_instances(self):
        setup = Setup(self.ana, ERA_A, self.version)
        cfg, first = setup.get_processors(
            "TT", stage="AnaTupleMerge", create_instances=True
        )
        self.assertEqual([entry["name"] for entry in cfg], ["counter"])
        # what mergeAnaTuples does with the dict it is given
        first["ds"] = object()
        _, second = setup.get_processors(
            "TT", stage="AnaTupleMerge", create_instances=True
        )
        self.assertEqual(sorted(second), ["counter"])
        self.assertIsNot(second["counter"], first["counter"])
        self.assertEqual(second["counter"].era, ERA_A)

    def test_the_entries_alone_are_unchanged(self):
        setup = Setup(self.ana, ERA_A, self.version)
        entries = setup.get_processors("TT", stage="AnaTupleMerge")
        self.assertIs(setup.get_processors("TT", stage="AnaTupleMerge"), entries)
        self.assertEqual(setup.get_processors("TT", stage="AnaTuple"), [])


class EraValidation(_ScratchAnalysis):
    def test_an_unknown_era_is_named_with_the_known_ones(self):
        for era in ("Run3_2099", "Processors", f"{ERA_A},{ERA_B}"):
            with self.subTest(era=era):
                with self.assertRaises(UnknownEraError) as caught:
                    Setup(self.ana, era, self.version)
                self.assertIn(f"unknown era '{era}'", str(caught.exception))
                self.assertIn(ERA_A, str(caught.exception))

    def test_a_user_custom_cannot_change_the_era(self):
        user_custom = os.path.join(self.tmp, "user_custom.yaml")
        self.write(user_custom, f"era: {ERA_B}\n")
        with self.assertRaises(RuntimeError) as caught:
            Setup(self.ana, ERA_A, self.version, user_custom_file=user_custom)
        self.assertIn(f"era: '{ERA_B}'", str(caught.exception))

    def test_a_customisation_cannot_change_the_era(self):
        with self.assertRaises(RuntimeError) as caught:
            Setup(self.ana, ERA_A, self.version, customisations=f"era={ERA_B}")
        self.assertIn(f"era: '{ERA_B}'", str(caught.exception))

    def test_a_known_era_loads(self):
        setup = Setup(self.ana, ERA_B, self.version)
        self.assertEqual(setup.global_params["era"], ERA_B)
        self.assertEqual(sorted(setup.datasets), ["Data_B", "TTto2L2Nu"])


class EraIds(unittest.TestCase):
    def test_the_ids_are_the_cpp_enum_values(self):
        with open(os.path.join(flaf_repo, "include", "AnalysisTools.h")) as f:
            header = f.read()
        body = re.search(r"enum class Period : int \{(.*?)\};", header, re.S).group(1)
        enum = {
            name: int(value) for name, value in re.findall(r"(\w+)\s*=\s*(\d+)", body)
        }
        self.assertEqual(ERA_IDS, enum)

    def test_every_configured_era_has_an_id(self):
        eras = known_eras(flaf_parent)
        self.assertEqual(
            eras,
            [
                "Run3_2022",
                "Run3_2022EE",
                "Run3_2023",
                "Run3_2023BPix",
                "Run3_2024",
                "Run3_2025",
                "Run3_2026",
            ],
        )
        self.assertLessEqual(set(eras), set(ERA_IDS))


if __name__ == "__main__":
    unittest.main()
