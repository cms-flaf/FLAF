#!/usr/bin/env python3
"""Setup navigates from a process to the entry the physics model lists.

A meta-process is replaced in the physics model by its expanded members, and a group is
listed in place of its sub_processes, so the dataset's own process is often not what
phys_models.yaml names. process_parent / process_ancestors / original_process walk back up
that hierarchy, and PhysicsModel.listed_process_type gives the role of the listed entry.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

sys.modules["ROOT"] = mock.MagicMock()

from FLAF.Common.Setup import Setup

GLOBAL = """
phys_model: M
channelSelection: [ eE ]
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
"""

PROCESSES = """
Signal:
  is_meta_process: true
  meta_setup:
    dataset_name_pattern: Sig_M_(\\d+)
    parameters: [ MASS ]
    process_name: Signal_${MASS}
    name_pattern: Signal ${MASS}
    to_plot: []
    plot_color: []
  datasets: [ Sig_M_300, Sig_M_500 ]
H:
  sub_processes: [ VH, ggH ]
VH:
  sub_processes: [ ZH ]
ZH:
  datasets: [ ZHto2Tau ]
ggH:
  datasets: [ GluGluHto2Tau ]
TT:
  datasets: [ TTto2L2Nu ]
"""

DATASETS = "".join(
    f"{name}:\n  crossSection: 1pb\n"
    for name in ["Sig_M_300", "Sig_M_500", "ZHto2Tau", "GluGluHto2Tau", "TTto2L2Nu"]
)


class TestProcessHierarchy(unittest.TestCase):
    def make_setup(
        self,
        processes=PROCESSES,
        model="M:\n  backgrounds: [ H, TT ]\n  signals: [ Signal ]\n",
    ):
        files = {
            "config/global.yaml": GLOBAL,
            "config/phys_models.yaml": model,
            "config/processes.yaml": processes,
            "config/Run3_2024/datasets.yaml": DATASETS,
            "config/weights.yaml": "norm: {}\nshape: {}\n",
        }
        for path, text in files.items():
            full = os.path.join(self.ana, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(text)
        return Setup(self.ana, "Run3_2024", "test")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ana = self.tmp.name
        os.symlink(flaf_repo, os.path.join(self.ana, "FLAF"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_expanded_member_leads_to_its_meta_process(self):
        setup = self.make_setup()
        process = setup.datasets["Sig_M_500"]["process_name"]
        self.assertEqual(process, "Signal_500")
        self.assertEqual(setup.process_parent(process), "Signal")
        self.assertEqual(setup.process_ancestors(process), ["Signal_500", "Signal"])
        self.assertEqual(setup.original_process(process), "Signal")
        self.assertEqual(setup.phys_model.listed_process_type("Signal"), "signals")
        # the expanded member is what the model holds now, not the meta-process
        self.assertNotIn("Signal", setup.phys_model.processes())

    def test_nested_sub_process_leads_to_the_listed_group(self):
        setup = self.make_setup()
        process = setup.datasets["ZHto2Tau"]["process_name"]
        self.assertEqual(setup.process_ancestors(process), ["ZH", "VH", "H"])
        self.assertEqual(setup.original_process(process), "H")
        self.assertEqual(setup.phys_model.listed_process_type("H"), "backgrounds")

    def test_listed_process_is_its_own_original(self):
        setup = self.make_setup()
        self.assertIsNone(setup.process_parent("TT"))
        self.assertEqual(setup.process_ancestors("TT"), ["TT"])
        self.assertEqual(setup.original_process("TT"), "TT")

    def test_original_role_matches_process_group(self):
        setup = self.make_setup()
        for dataset in setup.datasets.values():
            original = setup.original_process(dataset["process_name"])
            self.assertEqual(
                setup.phys_model.listed_process_type(original),
                dataset["process_group"],
            )

    def test_unlisted_process_has_no_role(self):
        setup = self.make_setup()
        with self.assertRaises(RuntimeError):
            setup.phys_model.listed_process_type("ZH")

    def test_sub_process_of_two_groups_is_refused(self):
        processes = PROCESSES + "Other:\n  sub_processes: [ ggH ]\n"
        model = "M:\n  backgrounds: [ H, Other, TT ]\n  signals: [ Signal ]\n"
        with self.assertRaisesRegex(RuntimeError, "sub_process of both"):
            self.make_setup(processes, model)


if __name__ == "__main__":
    unittest.main()
