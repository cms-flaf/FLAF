#!/usr/bin/env python3
"""Unit tests for the shared-MC residue split (2024/2025/2026)."""

import os
import sys
import unittest

# Parent of the FLAF repo so `import FLAF.Common...` resolves.
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


class TestSharedMcSplit(unittest.TestCase):
    def test_fractions_match_residue_share(self):
        _, _, _, frac_24 = shared_mc_split("Run3_2024", SHARED_MC)
        _, _, _, frac_25 = shared_mc_split("Run3_2025", SHARED_MC)
        _, _, _, frac_26 = shared_mc_split("Run3_2026", SHARED_MC)
        self.assertAlmostEqual(frac_24, 9 / 20)
        self.assertAlmostEqual(frac_25, 9 / 20)
        self.assertAlmostEqual(frac_26, 2 / 20)
        self.assertAlmostEqual(frac_24 + frac_25 + frac_26, 1.0)

    def test_unknown_era(self):
        with self.assertRaises(RuntimeError):
            shared_mc_split("Run3_2023", SHARED_MC)

    def test_assignment_is_partition(self):
        split_mod, lo24, hi24, _ = shared_mc_split("Run3_2024", SHARED_MC)
        _, lo25, hi25, _ = shared_mc_split("Run3_2025", SHARED_MC)
        _, lo26, hi26, _ = shared_mc_split("Run3_2026", SHARED_MC)
        n24 = n25 = n26 = 0
        for event in range(split_mod):
            in24 = shared_mc_in_era(event, split_mod, lo24, hi24)
            in25 = shared_mc_in_era(event, split_mod, lo25, hi25)
            in26 = shared_mc_in_era(event, split_mod, lo26, hi26)
            self.assertEqual(int(in24) + int(in25) + int(in26), 1)
            n24 += int(in24)
            n25 += int(in25)
            n26 += int(in26)
        self.assertEqual(n24, 9)
        self.assertEqual(n25, 9)
        self.assertEqual(n26, 2)

    def test_two_denominators_match_when_split_is_not_9_9_2(self):
        # 25 events: residue counts are 14:9:2, not 9:9:2. Weights are not flat.
        events = list(range(25))
        genw = [float(i + 1) for i in events]
        denom_all = sum(genw)
        lumi_xs = 100.0
        sum_base = 0.0
        sums_cmb = {}
        for era in SHARED_MC["eras"]:
            split_mod, lo, hi, frac = shared_mc_split(era, SHARED_MC)
            in_era = [shared_mc_in_era(e, split_mod, lo, hi) for e in events]
            denom_cmb = sum(w for w, ok in zip(genw, in_era) if ok)
            actual_frac = denom_cmb / denom_all
            self.assertNotAlmostEqual(actual_frac, frac)
            self.assertGreater(denom_cmb, 0)
            sum_cmb = 0.0
            for w, ok in zip(genw, in_era):
                sum_base += lumi_xs * w / denom_all
                sum_cmb += lumi_xs * w / denom_cmb if ok else 0.0
            sums_cmb[era] = sum_cmb
            self.assertAlmostEqual(sum_cmb, lumi_xs)
            # Scaling the single-year weight by the residue fraction is biased.
            biased = sum(
                (lumi_xs * w / denom_all) / frac if ok else 0.0
                for w, ok in zip(genw, in_era)
            )
            self.assertNotAlmostEqual(biased, lumi_xs)
        # sum_base was accumulated once per era
        self.assertAlmostEqual(sum_base / 3, lumi_xs)


if __name__ == "__main__":
    unittest.main()
