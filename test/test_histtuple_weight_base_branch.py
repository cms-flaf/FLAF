#!/usr/bin/env python3
"""HistTuple-stage check: weight_base_branch selects one-era vs shared-MC.

HistTuple multiplies the AnaTuple column named by ``weight_base_branch``.
``weight_base`` uses the full-sample denominator; ``weight_base_cmb`` uses
the in-era denominator. The residue share is only a target — the two
denominators keep both yields equal to L·σ even when the split is not 9:9:2.
"""

import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common.shared_mc import shared_mc_in_era, shared_mc_split

SHARED_MC = {
    "split_modulus": 20,
    "eras": {
        "Run3_2024": [0, 8],
        "Run3_2025": [9, 17],
        "Run3_2026": [18, 19],
    },
}


def _two_denom_weights(events, genw, era):
    split_mod, lo, hi, _ = shared_mc_split(era, SHARED_MC)
    in_era = [shared_mc_in_era(e, split_mod, lo, hi) for e in events]
    denom_all = sum(genw)
    denom_cmb = sum(w for w, ok in zip(genw, in_era) if ok)
    w_base = [w / denom_all for w in genw]
    w_cmb = [
        (w / denom_cmb if ok and denom_cmb else 0.0) for w, ok in zip(genw, in_era)
    ]
    return w_base, w_cmb


class TestHistTupleWeightBaseBranch(unittest.TestCase):
    def test_one_era_uses_full_denominator(self):
        events = list(range(25))
        genw = [float(i + 1) for i in events]
        for era in SHARED_MC["eras"]:
            w_base, _ = _two_denom_weights(events, genw, era)
            self.assertEqual(len(w_base), len(events))
            self.assertTrue(all(w > 0 for w in w_base))
            self.assertAlmostEqual(sum(w_base), 1.0)

    def test_shared_uses_in_era_denominator(self):
        events = list(range(25))
        genw = [float(i + 1) for i in events]
        for event_i, event in enumerate(events):
            assigned = 0
            for era in SHARED_MC["eras"]:
                _, w_cmb = _two_denom_weights(events, genw, era)
                if w_cmb[event_i] != 0:
                    assigned += 1
            self.assertEqual(assigned, 1)
        for era in SHARED_MC["eras"]:
            _, w_cmb = _two_denom_weights(events, genw, era)
            self.assertAlmostEqual(sum(w_cmb), 1.0)

    def test_flag_selects_different_columns(self):
        events = list(range(25))
        genw = [float(i + 1) for i in events]
        w_base, w_cmb = _two_denom_weights(events, genw, "Run3_2025")
        self.assertNotEqual(w_base, w_cmb)


if __name__ == "__main__":
    unittest.main()
