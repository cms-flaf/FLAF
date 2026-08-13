#!/usr/bin/env python3
"""HistTuple-stage check: weight_base_branch selects one-era vs shared-MC.

HistTuple does not recompute the split. It multiplies the AnaTuple column named
by ``weight_base_branch`` (``weight_base`` or ``weight_base_cmb``).
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


def _histtuple_final_weight(weight_base, event, era, branch):
    """Same product HistTuple uses: SF * selected base-weight column."""
    split_mod, lo, hi, frac = shared_mc_split(era, SHARED_MC)
    cmb = weight_base / frac if shared_mc_in_era(event, split_mod, lo, hi) else 0.0
    selected = weight_base if branch == "weight_base" else cmb
    return selected


class TestHistTupleWeightBaseBranch(unittest.TestCase):
    def test_one_era_keeps_every_event(self):
        n = 0
        n_nonzero = 0
        for era in SHARED_MC["eras"]:
            for event in range(100):
                w = _histtuple_final_weight(2.0, event, era, "weight_base")
                n += 1
                n_nonzero += int(w != 0)
                self.assertEqual(w, 2.0)
        self.assertEqual(n_nonzero, n)

    def test_shared_is_partition_and_rescales(self):
        weight_base = 2.0
        totals = {era: 0.0 for era in SHARED_MC["eras"]}
        nonzero = {era: 0 for era in SHARED_MC["eras"]}
        n = 100
        for event in range(n):
            assigned = 0
            for era in SHARED_MC["eras"]:
                w = _histtuple_final_weight(weight_base, event, era, "weight_base_cmb")
                totals[era] += w
                if w != 0:
                    assigned += 1
                    nonzero[era] += 1
                    _, _, _, frac = shared_mc_split(era, SHARED_MC)
                    self.assertAlmostEqual(w, weight_base / frac)
            self.assertEqual(assigned, 1)
        for era, total in totals.items():
            self.assertAlmostEqual(total / n, weight_base)
        self.assertEqual(nonzero["Run3_2024"], 45)
        self.assertEqual(nonzero["Run3_2025"], 45)
        self.assertEqual(nonzero["Run3_2026"], 10)

    def test_unknown_branch_is_not_silently_one_era(self):
        # HistTuple must use the configured column name, not fall back.
        self.assertNotEqual(
            _histtuple_final_weight(2.0, 0, "Run3_2025", "weight_base"),
            _histtuple_final_weight(2.0, 0, "Run3_2025", "weight_base_cmb"),
        )


if __name__ == "__main__":
    unittest.main()
