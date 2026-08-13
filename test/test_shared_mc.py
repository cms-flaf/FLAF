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

    def test_cmb_zero_on_other_era_and_rescales(self):
        split_mod, lo24, hi24, frac_24 = shared_mc_split("Run3_2024", SHARED_MC)
        _, lo25, hi25, frac_25 = shared_mc_split("Run3_2025", SHARED_MC)
        _, lo26, hi26, frac_26 = shared_mc_split("Run3_2026", SHARED_MC)
        weight_base = 2.5
        event_24 = lo24
        event_25 = lo25
        event_26 = lo26
        self.assertTrue(shared_mc_in_era(event_24, split_mod, lo24, hi24))
        self.assertTrue(shared_mc_in_era(event_25, split_mod, lo25, hi25))
        self.assertTrue(shared_mc_in_era(event_26, split_mod, lo26, hi26))
        self.assertAlmostEqual((weight_base / frac_24) * frac_24, weight_base)
        self.assertAlmostEqual((weight_base / frac_25) * frac_25, weight_base)
        self.assertAlmostEqual((weight_base / frac_26) * frac_26, weight_base)

    def test_cmb_expectation_matches_single_year(self):
        split_mod, lo24, hi24, frac_24 = shared_mc_split("Run3_2024", SHARED_MC)
        _, lo25, hi25, frac_25 = shared_mc_split("Run3_2025", SHARED_MC)
        _, lo26, hi26, frac_26 = shared_mc_split("Run3_2026", SHARED_MC)
        weight_base = 1.0
        sum_24 = sum_25 = sum_26 = 0.0
        n = split_mod * 5
        for event in range(n):
            if shared_mc_in_era(event, split_mod, lo24, hi24):
                sum_24 += weight_base / frac_24
            elif shared_mc_in_era(event, split_mod, lo25, hi25):
                sum_25 += weight_base / frac_25
            else:
                self.assertTrue(shared_mc_in_era(event, split_mod, lo26, hi26))
                sum_26 += weight_base / frac_26
        self.assertAlmostEqual(sum_24 / n, weight_base)
        self.assertAlmostEqual(sum_25 / n, weight_base)
        self.assertAlmostEqual(sum_26 / n, weight_base)


if __name__ == "__main__":
    unittest.main()
