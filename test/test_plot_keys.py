#!/usr/bin/env python3
"""Common/PlotKeys.py: the value of a histograms.yaml setting by channel, category and region, shared by
x_rebin (HistPlotter findNewBins) and blind_ranges (PlotBlinding). Plain Python."""

import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common.PlotKeys import OTHER, leaves, resolve  # noqa: E402

# the shape of HH_bbtautau's x_rebin of tau1_met_mt: region names at the top, channel overrides below
X_REBIN = {
    "res1b_cat3_masswindow": [0, 10, 300],
    "baseline": [0, 20, 300],
    "other": [0, 50, 300],
    "muTau": {"baseline_masswindow": [0, 30, 300]},
    "tauTau": {"baseline": [0, 40, 300], "other": [0, 60, 300]},
}


def r(spec, channel=None, category=None, region=None):
    return resolve(spec, "test", channel=channel, category=category, region=region)


class ResolveTest(unittest.TestCase):
    def test_a_value_is_the_same_for_every_plot(self):
        self.assertEqual(r([1, 2], "e", "res2b", "SR"), [1, 2])
        self.assertEqual(r("10|0:1"), "10|0:1")

    def test_any_key_at_any_level(self):
        self.assertEqual(
            r(X_REBIN, "eTau", "inclusive", "res1b_cat3_masswindow"), [0, 10, 300]
        )
        self.assertEqual(
            r(X_REBIN, "eTau", "res1b_cat3_masswindow", "OS_Iso"), [0, 10, 300]
        )
        self.assertEqual(
            r(X_REBIN, "muTau", "baseline_masswindow", "OS_Iso"), [0, 30, 300]
        )
        self.assertEqual(r(X_REBIN, "tauTau", "baseline", "OS_Iso"), [0, 40, 300])

    def test_other(self):
        self.assertEqual(r(X_REBIN, "eTau", "inclusive", "OS_Iso"), [0, 50, 300])
        self.assertEqual(r(X_REBIN, "tauTau", "inclusive", "OS_Iso"), [0, 60, 300])

    def test_no_fallback_below_a_matching_key(self):
        # muTau names only baseline_masswindow: its other plots have no value, as in the search
        # findNewBins had before PlotKeys; neither the top-level baseline nor "other" is taken
        for category in ("baseline", "inclusive"):
            with self.subTest(category=category):
                with self.assertRaisesRegex(ValueError, "at muTau: no key matches"):
                    r(X_REBIN, "muTau", category, "OS_Iso")

    def test_other_is_searched_too(self):
        spec = {"SR": [1], OTHER: {"e": [2], OTHER: [3]}}
        self.assertEqual(r(spec, "e", "res2b", "CR"), [2])
        self.assertEqual(r(spec, "mu", "res2b", "CR"), [3])
        self.assertEqual(r(spec, "e", "res2b", "SR"), [1])

    def test_first_of_channel_category_region(self):
        spec = {"e": [1], "res2b": [2], "SR": [3]}
        self.assertEqual(r(spec, "e", "res2b", "SR"), [1])
        self.assertEqual(r(spec, "mu", "res2b", "SR"), [2])
        self.assertEqual(r(spec, "mu", "boosted", "SR"), [3])

    def test_none_is_not_looked_up(self):
        spec = {"SR": [1], OTHER: [2]}
        self.assertEqual(r(spec, region="SR"), [1])
        self.assertEqual(r(spec), [2])

    def test_no_value_raises(self):
        with self.assertRaisesRegex(ValueError, "no value"):
            r({"SR": [1]}, "e", "res2b", "CR")
        with self.assertRaisesRegex(ValueError, "no value"):
            r({"e": {"SR": [1]}}, "e", "res2b", "CR")
        with self.assertRaisesRegex(ValueError, "no value"):
            r({})

    def test_null_raises(self):
        with self.assertRaisesRegex(ValueError, "null"):
            r({"SR": None, OTHER: [1]}, "e", "res2b", "SR")


class LeavesTest(unittest.TestCase):
    def test_every_value_with_its_path(self):
        spec = {"SR": {"e": [1], OTHER: []}, "CR": [2], "x": {}}
        self.assertEqual(
            list(leaves(spec)),
            [(("SR", "e"), [1]), (("SR", OTHER), []), (("CR",), [2]), (("x",), {})],
        )
        self.assertEqual(list(leaves([1, 2])), [((), [1, 2])])
        self.assertEqual(list(leaves({})), [((), {})])


if __name__ == "__main__":
    unittest.main()
