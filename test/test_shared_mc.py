#!/usr/bin/env python3
"""Unit tests for the 2024/2025 shared-MC luminosity split."""

import os
import sys
import unittest

# Parent of the FLAF repo so `import FLAF.Common...` resolves.
flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common.shared_mc import shared_mc_in_24, shared_mc_split

SHARED_MC = {
    "split_modulus": 1000000,
    "years": {
        "24": {"luminosity": 109948.18},
        "25": {"luminosity": 110730.86},
    },
}


class TestSharedMcSplit(unittest.TestCase):
    def test_year_from_era(self):
        y24, _, _, _ = shared_mc_split("Run3_2024", SHARED_MC)
        y25, _, _, _ = shared_mc_split("Run3_2025", SHARED_MC)
        self.assertEqual(y24, "24")
        self.assertEqual(y25, "25")

    def test_fractions_sum_to_one(self):
        _, _, _, frac_24 = shared_mc_split("Run3_2024", SHARED_MC)
        _, _, _, frac_25 = shared_mc_split("Run3_2025", SHARED_MC)
        self.assertAlmostEqual(frac_24 + frac_25, 1.0, places=9)
        self.assertGreater(frac_24, 0)
        self.assertGreater(frac_25, 0)

    def test_unknown_era_year(self):
        with self.assertRaises(RuntimeError):
            shared_mc_split("Run3_2023", SHARED_MC)

    def test_assignment_is_complementary(self):
        _, split_mod, thresh_24, _ = shared_mc_split("Run3_2024", SHARED_MC)
        n24 = n25 = 0
        for event in range(0, split_mod, 17):
            if shared_mc_in_24(event, split_mod, thresh_24):
                n24 += 1
            else:
                n25 += 1
        self.assertGreater(n24, 0)
        self.assertGreater(n25, 0)
        self.assertEqual(n24 + n25, len(range(0, split_mod, 17)))

    def test_cmb_zero_on_other_year_and_rescales(self):
        _, split_mod, thresh_24, frac_24 = shared_mc_split("Run3_2024", SHARED_MC)
        _, _, _, frac_25 = shared_mc_split("Run3_2025", SHARED_MC)
        weight_base = 2.5
        event_24 = thresh_24 - 1
        event_25 = thresh_24
        self.assertTrue(shared_mc_in_24(event_24, split_mod, thresh_24))
        self.assertFalse(shared_mc_in_24(event_25, split_mod, thresh_24))
        cmb_24_on_24 = weight_base / frac_24
        cmb_24_on_25 = 0.0
        cmb_25_on_24 = 0.0
        cmb_25_on_25 = weight_base / frac_25
        self.assertAlmostEqual(cmb_24_on_24 * frac_24, weight_base)
        self.assertAlmostEqual(cmb_25_on_25 * frac_25, weight_base)
        self.assertEqual(cmb_24_on_25, 0.0)
        self.assertEqual(cmb_25_on_24, 0.0)

    def test_cmb_expectation_matches_single_year(self):
        _, split_mod, thresh_24, frac_24 = shared_mc_split("Run3_2024", SHARED_MC)
        _, _, _, frac_25 = shared_mc_split("Run3_2025", SHARED_MC)
        weight_base = 1.0
        sum_24 = sum_25 = 0.0
        step = 50
        events = range(0, split_mod, step)
        n = len(events)
        for event in events:
            if shared_mc_in_24(event, split_mod, thresh_24):
                sum_24 += weight_base / frac_24
            else:
                sum_25 += weight_base / frac_25
        self.assertAlmostEqual(sum_24 / n, weight_base, places=2)
        self.assertAlmostEqual(sum_25 / n, weight_base, places=2)


if __name__ == "__main__":
    unittest.main()
