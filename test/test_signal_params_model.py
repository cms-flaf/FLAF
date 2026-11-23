#!/usr/bin/env python3
"""`signal_params` and the physics model: a histograms.yaml entry with `signal_params` is filled only for the
datasets matching one of its entries (Analysis/tasks.py `_variable_applies_to_dataset`), unless the physics
model sets `apply_signal_params: false`, which fills every variable for every dataset (a model for studies,
e.g. every signal in every signal-specific variable); the key defaults to true and must be a boolean
(Common/Setup.py PhysicsModel).
"""

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

# The task modules import ROOT, numpy, awkward, uproot and Corrections at module level, which the
# unit-test runner does not have and nothing tested here uses: stand-ins for the missing ones while
# importing, withdrawn afterwards. The filter imports findBinEntry from HistHelper when called, which
# needs ROOT and ANALYSIS_PATH: a stand-in module with the exact-name lookup serves the calls.
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


_missing = [name for name in _HEAVY_MODULES if not _importable(name)]
sys.modules.update({name: mock.MagicMock() for name in _missing})
try:
    from FLAF.Analysis.tasks import _variable_applies_to_dataset  # noqa: E402
    from FLAF.Common.Setup import PhysicsModel  # noqa: E402
finally:
    for name in _missing:
        sys.modules.pop(name, None)

HISTS = {
    "score_M400": {"signal_params": {"mass": 400}},
    "score_M400_spin0": {"signal_params": [{"mass": 400, "spin": 0}]},
    "lep_pt": {},
}
DATASETS = {
    "sig_M300_spin0": {"mass": 300, "spin": 0},
    "sig_M400_spin0": {"mass": 400, "spin": 0},
    "sig_M400_spin2": {"mass": 400, "spin": 2},
    "TT": {},
}


def setup(apply_signal_params=None):
    model = {"signals": ["sig"], "backgrounds": ["TT"]}
    if apply_signal_params is not None:
        model["apply_signal_params"] = apply_signal_params
    return types.SimpleNamespace(
        hists=HISTS, datasets=DATASETS, phys_model=PhysicsModel("M", model)
    )


HIST_HELPER = types.SimpleNamespace(findBinEntry=lambda hists, var: var)


def applies(s, var, ds):
    with mock.patch.dict(sys.modules, {"FLAF.Common.HistHelper": HIST_HELPER}):
        return _variable_applies_to_dataset(s, var, ds)


class PhysicsModelKey(unittest.TestCase):
    def test_default_is_true(self):
        self.assertIs(PhysicsModel("M", {"signals": ["s"]}).apply_signal_params, True)

    def test_boolean_values(self):
        for value in (True, False):
            with self.subTest(value=value):
                model = PhysicsModel(
                    "M", {"signals": ["s"], "apply_signal_params": value}
                )
                self.assertIs(model.apply_signal_params, value)

    def test_non_boolean_raises(self):
        for value in ("false", 0, 1, None, [False]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "apply_signal_params"):
                    PhysicsModel("M", {"signals": ["s"], "apply_signal_params": value})

    def test_other_unknown_keys_still_raise(self):
        with self.assertRaisesRegex(RuntimeError, "invalid process types"):
            PhysicsModel("M", {"signals": ["s"], "apply_signal_param": False})


class SignalParamsFilter(unittest.TestCase):
    def test_applied_by_default(self):
        s = setup()
        self.assertTrue(applies(s, "score_M400", "sig_M400_spin0"))
        self.assertTrue(applies(s, "score_M400", "sig_M400_spin2"))
        self.assertFalse(applies(s, "score_M400", "sig_M300_spin0"))
        self.assertFalse(applies(s, "score_M400_spin0", "sig_M400_spin2"))
        # a dataset without the key matches, a variable without signal_params goes everywhere
        self.assertTrue(applies(s, "score_M400", "TT"))
        self.assertTrue(applies(s, "lep_pt", "sig_M300_spin0"))
        self.assertEqual(applies(setup(True), "score_M400", "sig_M300_spin0"), False)

    def test_not_applied_with_the_model_key_false(self):
        s = setup(False)
        for var in HISTS:
            for ds in DATASETS:
                with self.subTest(var=var, ds=ds):
                    self.assertTrue(applies(s, var, ds))


if __name__ == "__main__":
    unittest.main()
