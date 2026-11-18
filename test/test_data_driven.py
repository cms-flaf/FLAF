#!/usr/bin/env python3
"""Common/DataDriven.py: the declaration of a data-driven process (processes.yaml) and the
names of its estimator's nuisances. Plain Python, no ROOT."""

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common.DataDriven import (  # noqa: E402
    nuisance_hist_name,
    validate_data_driven,
)

# FLAF.Common.Utilities (imported through Setup) imports ROOT at module level, which nothing here
# touches. Where ROOT is missing, a stand-in serves the import only and is withdrawn again.
_root_stand_in = None
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    _root_stand_in = sys.modules["ROOT"] = mock.MagicMock()

from FLAF.Common.Setup import Setup  # noqa: E402

if _root_stand_in is not None and sys.modules.get("ROOT") is _root_stand_in:
    del sys.modules["ROOT"]

GOOD = {
    "name": "Fakes",
    "color": "kGray+1",
    "data_driven": {
        "estimator": "Analysis.SL_fakes.estimate",
        "nuisances": {"FakesStat": "CMS_fakes_stat_{}"},
        "min_tt": 1.0,
    },
}


def with_changes(**changes):
    process = dict(GOOD, data_driven=dict(GOOD["data_driven"]))
    for key, value in changes.items():
        if key in ("estimator", "nuisances"):
            process["data_driven"][key] = value
        elif value is None:
            del process[key]
        else:
            process[key] = value
    return process


class Declaration(unittest.TestCase):
    def test_a_good_declaration_with_its_parameters(self):
        cfg = validate_data_driven("Fakes", GOOD, "backgrounds")
        self.assertEqual(cfg["min_tt"], 1.0)
        no_nuisances = dict(GOOD, data_driven={"estimator": "a.b"})
        validate_data_driven("Fakes", no_nuisances, "backgrounds")

    def test_refused(self):
        cases = [
            ("a background", GOOD, "signals"),
            ("no 'datasets'", with_changes(datasets=["TT"]), "backgrounds"),
            ("no 'sub_processes'", with_changes(sub_processes=["TT"]), "backgrounds"),
            ("a 'color'", with_changes(color=None), "backgrounds"),
            ("<module>.<function>", with_changes(estimator="estimate"), "backgrounds"),
            ("<module>.<function>", with_changes(estimator="a."), "backgrounds"),
            ("<module>.<function>", with_changes(estimator=3), "backgrounds"),
            ("ending with '_{}'", with_changes(nuisances={"S": "stat"}), "backgrounds"),
            (
                "ending with '_{}'",
                with_changes(nuisances={"S": "s_{}_{}"}),
                "backgrounds",
            ),
            ("ending with '_{}'", with_changes(nuisances=["S"]), "backgrounds"),
            ("ending with '_{}'", with_changes(nuisances={"S": "s{}"}), "backgrounds"),
            ("ending with '_{}'", with_changes(nuisances={"S": "{}"}), "backgrounds"),
            (
                "ending with '_{}'",
                with_changes(nuisances={"S": "s_{mass}_{}"}),
                "backgrounds",
            ),
            (
                "ending with '_{}'",
                with_changes(nuisances={"S": "s_{0}"}),
                "backgrounds",
            ),
            ("ending with '_{}'", with_changes(nuisances={"S": "s_{"}), "backgrounds"),
            (
                "ending with '_{}'",
                with_changes(nuisances={"S": "s_{}_{era}"}),
                "backgrounds",
            ),
            (
                "ending with '_{}'",
                with_changes(nuisances={"S": "s_{}_x"}),
                "backgrounds",
            ),
            (
                "ending with '_{}'",
                with_changes(nuisances={"S": "s_{:>3}"}),
                "backgrounds",
            ),
            (
                "ending with '_{}'",
                with_changes(nuisances={"S": "s_{era!r}_{}"}),
                "backgrounds",
            ),
            (
                "not 'Central'",
                with_changes(nuisances={"Central": "s_{}"}),
                "backgrounds",
            ),
            ("must be a mapping", dict(GOOD, data_driven="a.b"), "backgrounds"),
        ]
        for msg, process, kind in cases:
            with self.subTest(msg=msg, process=process, kind=kind):
                with self.assertRaisesRegex(RuntimeError, msg):
                    validate_data_driven("Fakes", process, kind)

    def test_nuisance_names(self):
        key = ("e", "SR_SL_HMESub_m300", "boosted")
        self.assertEqual(
            nuisance_hist_name("Fakes", "CMS_fakes_stat_{}", "Up", key, "Run3_2022"),
            "Fakes_CMS_fakes_stat_Up",
        )
        template = "CMS_fakes_{channel}_{category}_{era}_{region}_{}"
        validate_data_driven(
            "Fakes", with_changes(nuisances={"S": template}), "backgrounds"
        )
        self.assertEqual(
            nuisance_hist_name("Fakes", template, "Down", key, "Run3_2022"),
            "Fakes_CMS_fakes_e_boosted_Run3_2022_SR_SL_HMESub_m300_Down",
        )


SETUP_GLOBAL = """
phys_model: M
channelSelection: [ eE ]
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
"""

SETUP_PROCESSES = """
TT:
  datasets: [ TTto2L2Nu ]
Fakes:
  name: Fakes
  color: kGray+1
  data_driven:
    estimator: Analysis.SL_fakes.estimate
    nuisances: { FakesStat: "FakesStat_{}" }
    tt_process: TT
"""


class InSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ana = self.tmp.name
        os.symlink(flaf_repo, os.path.join(self.ana, "FLAF"))

    def make_setup(self, processes=SETUP_PROCESSES, model=None, **kw):
        model = model or "M:\n  backgrounds: [ TT, Fakes ]\n"
        files = {
            "config/global.yaml": SETUP_GLOBAL,
            "config/phys_models.yaml": model,
            "config/processes.yaml": processes,
            "config/Run3_2024/datasets.yaml": "TTto2L2Nu:\n  crossSection: 1pb\n",
            "config/weights.yaml": "norm: {}\nshape: {}\n",
        }
        for path, text in files.items():
            full = os.path.join(self.ana, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(text)
        return Setup(self.ana, "Run3_2024", "test", **kw)

    def test_a_data_driven_background_without_datasets(self):
        setup = self.make_setup()
        self.assertEqual(list(setup.data_driven_processes), ["Fakes"])
        self.assertEqual(setup.data_driven_processes["Fakes"]["tt_process"], "TT")
        self.assertEqual(list(setup.datasets), ["TTto2L2Nu"])
        self.assertIn("Fakes", setup.phys_model.processes("backgrounds"))
        # the plotter reads its name and colour from here
        self.assertEqual(setup.parent_processes["Fakes"]["color"], "kGray+1")

    def test_not_in_the_model_or_not_selected(self):
        model = "M:\n  backgrounds: [ TT ]\n"
        self.assertEqual(self.make_setup(model=model).data_driven_processes, {})
        selected = self.make_setup(custom_process_selection="TT")
        self.assertEqual(selected.data_driven_processes, {})

    def test_a_data_driven_sub_process_is_refused(self):
        model = "M:\n  backgrounds: [ TT, Other ]\n"
        processes = SETUP_PROCESSES + "Other:\n  sub_processes: [ Fakes ]\n"
        with self.assertRaisesRegex(RuntimeError, "not as a sub_process of 'Other'"):
            self.make_setup(processes=processes, model=model)
        # a data-driven group deeper down
        nested = SETUP_PROCESSES.replace("Fakes:", "Mid:").replace(
            "  data_driven:", "  sub_processes: [ TT2 ]\n  data_driven:"
        )
        nested += "TT2:\n  datasets: [ TTto2L2Nu ]\nOther:\n  sub_processes: [ Mid ]\n"
        with self.assertRaisesRegex(RuntimeError, "not as a sub_process of 'Other'"):
            self.make_setup(processes=nested, model=model)

    def test_a_bad_declaration_stops_the_setup(self):
        with self.assertRaisesRegex(RuntimeError, "is a background"):
            self.make_setup(model="M:\n  backgrounds: [ TT ]\n  signals: [ Fakes ]\n")
        bad = SETUP_PROCESSES.replace("Analysis.SL_fakes.estimate", "estimate")
        with self.assertRaisesRegex(RuntimeError, "<module>.<function>"):
            self.make_setup(processes=bad)


if __name__ == "__main__":
    unittest.main()
