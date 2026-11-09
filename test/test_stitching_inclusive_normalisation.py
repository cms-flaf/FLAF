#!/usr/bin/env python3
"""Stitching denominators of shape-weight variations, normalised over the whole stitched sample.

Dividing every bin by its own sum under a variation would pin each bin to its nominal cross
section. The stitcher instead keeps the nominal per-bin sums and shares one factor C_k between
the bins, so the bins may change relative to each other while the total stays nominal.
Needs neither ROOT nor a config file.
"""

import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Processors.MCStitching import (
    CENTRAL,
    MCStitcher,
    inclusivelyNormalisedDenominators,
)

# Two bins: 0 jets (sigma 80) and 2 jets (sigma 20, generated with fewer events). Under the
# variation the summed weights are 1025 and 425: the 0-jet bin gains 2.5%, the 2-jet bin
# loses 15%, and the varied total would be 80 * 1.025 + 20 * 0.85 = 99.
SIGMA = {"0j": 80.0, "2j": 20.0}
NOMINAL = {"0j": 1000.0, "2j": 500.0}
VARIED = {"0j": 1025.0, "2j": 425.0}
# A pileup-like variation: the same 0.2% change in every bin.
UNIFORM = {name: 1.002 * NOMINAL[name] for name in NOMINAL}


def bin_yields(denominators, sums):
    return {name: SIGMA[name] * sums[name] / denominators[name] for name in SIGMA}


def make_stitcher():
    stitcher = MCStitcher.__new__(MCStitcher)
    stitcher.bins = [
        {"name": name, "selection": f"bin == {i}", "crossSectionValue": SIGMA[name]}
        for i, name in enumerate(SIGMA)
    ]
    return stitcher


def make_ana_caches():
    # Two datasets sharing the bins, as the inclusive and the binned samples do.
    def entry(sums, share):
        return {"Stitcher": {name: sums[name] * share for name in sums}}

    caches = {}
    for ds_name, share in [("inclusive", 0.25), ("binned", 0.75)]:
        caches[ds_name] = {
            "denominator": {
                CENTRAL: {CENTRAL: entry(NOMINAL, share)},
                "qcd_scale": {"_muR2_muF2": entry(VARIED, share)},
                "pu": {"Up": entry(UNIFORM, share)},
            }
        }
    return caches


class FakeDataFrame:
    def __init__(self):
        self.defined = {}

    def Define(self, name, expression):
        self.defined[name] = expression
        return self


class TestInclusivelyNormalisedDenominators(unittest.TestCase):
    def test_factor_is_the_relative_change_of_the_total(self):
        _, norm = inclusivelyNormalisedDenominators(SIGMA, VARIED, NOMINAL)
        self.assertAlmostEqual(norm, 0.99)

    def test_total_stays_nominal_and_bins_keep_their_change(self):
        denom, _ = inclusivelyNormalisedDenominators(SIGMA, VARIED, NOMINAL)
        yields = bin_yields(denom, VARIED)
        self.assertAlmostEqual(sum(yields.values()), sum(SIGMA.values()))
        self.assertAlmostEqual(yields["0j"], 80.0 * 1.025 / 0.99)
        self.assertAlmostEqual(yields["2j"], 20.0 * 0.85 / 0.99)
        # The 2-jet share falls from 20% to 17.2%.
        self.assertAlmostEqual(yields["2j"] / 100.0, 0.17171717, places=6)

    def test_per_bin_division_pins_every_bin(self):
        # Dividing each bin by its own varied sum N_k,b, for comparison.
        yields = bin_yields(VARIED, VARIED)
        self.assertEqual(yields, SIGMA)

    def test_uniform_variation_is_removed_entirely(self):
        uniform = {name: 1.1 * NOMINAL[name] for name in NOMINAL}
        denom, norm = inclusivelyNormalisedDenominators(SIGMA, uniform, NOMINAL)
        self.assertAlmostEqual(norm, 1.1)
        for name, value in bin_yields(denom, uniform).items():
            self.assertAlmostEqual(value, SIGMA[name])

    def test_empty_bin_keeps_its_varied_sum(self):
        sigma = dict(SIGMA, empty=5.0)
        nominal = dict(NOMINAL, empty=0.0)
        varied = dict(VARIED, empty=0.0)
        denom, norm = inclusivelyNormalisedDenominators(sigma, varied, nominal)
        self.assertAlmostEqual(norm, 0.99)
        self.assertEqual(denom["empty"], 0.0)

    def test_all_bins_empty_raises(self):
        zeros = {name: 0.0 for name in SIGMA}
        with self.assertRaises(RuntimeError):
            inclusivelyNormalisedDenominators(SIGMA, zeros, zeros)


class TestStitcherBinDenominators(unittest.TestCase):
    def test_variation_is_normalised_over_the_sample(self):
        stitcher = make_stitcher()
        denom = stitcher.binDenominators(
            make_ana_caches(), "Stitcher", "qcd_scale", "_muR2_muF2"
        )
        self.assertAlmostEqual(denom["0j"], NOMINAL["0j"] * 0.99)
        self.assertAlmostEqual(denom["2j"], NOMINAL["2j"] * 0.99)

    def test_uniform_variation_matches_per_bin_sums(self):
        # A variation that changes every bin alike (pileup) comes out as before.
        stitcher = make_stitcher()
        denom = stitcher.binDenominators(make_ana_caches(), "Stitcher", "pu", "Up")
        for name in UNIFORM:
            self.assertAlmostEqual(denom[name], UNIFORM[name])

    def test_nominal_is_unchanged(self):
        stitcher = make_stitcher()
        denom = stitcher.binDenominators(
            make_ana_caches(), "Stitcher", CENTRAL, CENTRAL
        )
        for name in NOMINAL:
            self.assertAlmostEqual(denom[name], NOMINAL[name])

    def test_single_bin_is_unchanged(self):
        stitcher = make_stitcher()
        stitcher.bins = stitcher.bins[:1]
        denom = stitcher.binDenominators(
            make_ana_caches(), "Stitcher", "qcd_scale", "_muR2_muF2"
        )
        self.assertAlmostEqual(denom["0j"], VARIED["0j"])

    def test_denominator_expression_uses_the_new_values(self):
        stitcher = make_stitcher()
        df = stitcher.onAnaTuple_defineDenominator(
            FakeDataFrame(),
            "__denom",
            "Stitcher",
            "inclusive",
            "qcd_scale",
            "_muR2_muF2",
            make_ana_caches(),
        )
        expression = df.defined["__denom"]
        self.assertIn(f"if(bin == 0) return {NOMINAL['0j'] * 0.99};", expression)
        self.assertIn(f"if(bin == 1) return {NOMINAL['2j'] * 0.99};", expression)


if __name__ == "__main__":
    unittest.main()
