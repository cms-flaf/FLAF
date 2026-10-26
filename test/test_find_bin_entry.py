#!/usr/bin/env python3
"""HistHelper.findBinEntry: the histograms.yaml entry of a variable, matched by regex.

HistFromNtupleProducer asks it once per booked histogram, so the answer is cached per
(config, variable): the cache must return what the regex scan returns, keep configs apart, and
not cache a failure. ROOT and numpy are not needed (HistHelper only uses them at import, mocked
here; the CI runner has neither).
"""

import os
import sys
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)
os.environ.setdefault("ANALYSIS_PATH", flaf_parent)

# Stand-ins only for this import, whatever other test modules left in sys.modules (a bare
# ROOT module has no gROOT), and removed afterwards so that they do not leak into them.
with mock.patch.dict(
    sys.modules, {"ROOT": mock.MagicMock(), "numpy": mock.MagicMock()}
):
    import FLAF.Common.HistHelper as HistHelper

CONFIG = {
    "DeepHME_mass": {"x_bins": "78|0:1"},
    "DNN_M[0-9]+_Signal": {"x_bins": "20|0:1"},
    "DNN_m[0-9]+_vs_HME": {"var_list": ["a", "b"]},
}


class FindBinEntry(unittest.TestCase):
    def setUp(self):
        HistHelper._bin_entry_cache.clear()

    def test_matches_the_entry_whose_pattern_fits(self):
        self.assertEqual(
            HistHelper.findBinEntry(CONFIG, "DeepHME_mass"), "DeepHME_mass"
        )
        self.assertEqual(
            HistHelper.findBinEntry(CONFIG, "DNN_M300_Signal"), "DNN_M[0-9]+_Signal"
        )
        self.assertEqual(
            HistHelper.findBinEntry(CONFIG, "DNN_m1000_vs_HME"), "DNN_m[0-9]+_vs_HME"
        )

    def test_a_repeated_lookup_does_not_scan_again(self):
        first = HistHelper.findBinEntry(CONFIG, "DNN_M300_Signal")
        with mock.patch.object(HistHelper.re, "fullmatch") as fullmatch:
            for _ in range(1000):
                self.assertEqual(
                    HistHelper.findBinEntry(CONFIG, "DNN_M300_Signal"), first
                )
        fullmatch.assert_not_called()

    def test_configs_are_kept_apart(self):
        other = {"DNN_M[0-9]+_.*": {"x_bins": "10|0:1"}}
        self.assertEqual(
            HistHelper.findBinEntry(CONFIG, "DNN_M300_Signal"), "DNN_M[0-9]+_Signal"
        )
        self.assertEqual(
            HistHelper.findBinEntry(other, "DNN_M300_Signal"), "DNN_M[0-9]+_.*"
        )

    def test_a_failure_is_raised_every_time_and_not_cached(self):
        ambiguous = dict(CONFIG, **{"DNN_M300_.*": {}})
        for _ in range(2):
            with self.assertRaises(KeyError):
                HistHelper.findBinEntry(CONFIG, "no_such_variable")
            with self.assertRaises(RuntimeError):
                HistHelper.findBinEntry(ambiguous, "DNN_M300_Signal")
        self.assertEqual(HistHelper._bin_entry_cache, {})


if __name__ == "__main__":
    unittest.main()
