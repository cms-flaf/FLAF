#!/usr/bin/env python3
"""PDF members as an indexed shape-weight source.

PDF4LHC15 (arXiv:1510.03865) takes the spread across members of each bin, so each member
needs its own denominator and its own weight_base_pdf<k>_rel -- the same grid pileup uses
for Up/Down, with the scales being member indices instead. These tests cover the scale
axis, the registry cross product, the per-member weights, and the property the whole
design rests on: summed with the nominal weight, every member reproduces the nominal
yield, so only the acceptance survives.
"""

import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import ROOT

from Corrections.CorrectionsCore import (
    ShapeWeightRegistry,
    central,
    down,
    getScales,
    getSystName,
    registerSourceScales,
    splitSystName,
    up,
)
from Corrections.pdf import pdfWeightProducer

ROOT.gROOT.SetBatch(True)

N_MEMBERS = 5


def pu_branch(source, scale):
    return f"weight_pu_{scale}"


def frame(n_events, members):
    return (
        ROOT.RDataFrame(n_events)
        .Define("LHEPdfWeight", members)
        .Define("weight_gen", "float(rdfentry_ + 1)")
    )


class PdfScaleAxisTest(unittest.TestCase):
    def setUp(self):
        registerSourceScales("pdf", pdfWeightProducer.scales({"n_members": N_MEMBERS}))

    def test_scales_are_member_indices(self):
        self.assertEqual(getScales("pdf"), ["0", "1", "2", "3", "4"])
        # Sources that did not declare anything keep Up/Down.
        self.assertEqual(getScales("pu"), [up, down])
        self.assertEqual(getScales(central), [central])

    def test_syst_names_round_trip(self):
        self.assertEqual(getSystName("pdf", "3"), "pdf3")
        self.assertEqual(splitSystName("pdf3"), ("pdf", "3"))
        self.assertEqual(getSystName("pu", up), "puUp")
        self.assertEqual(splitSystName("puUp"), ("pu", up))
        with self.assertRaises(RuntimeError):
            getSystName("pdf", str(N_MEMBERS))

    def test_registry_cross_product(self):
        reg = ShapeWeightRegistry()
        reg.register("pu", ["pu"], pu_branch)
        reg.register("pdf", pdfWeightProducer.uncSource, pdfWeightProducer.branchName)
        # A PDF member carries the central pileup weight, and a pileup variation carries
        # the central PDF weight.
        self.assertEqual(
            reg.branches("pdf", "3"), ["weight_pu_Central", "weight_pdf_3"]
        )
        self.assertEqual(reg.branches("pu", up), ["weight_pu_Up", "weight_pdf_Central"])
        keys = list(reg.asDict().keys())
        self.assertIn(("pdf", "4"), keys)
        self.assertEqual(len(keys), 1 + 2 + N_MEMBERS)


class PdfWeightTest(unittest.TestCase):
    def setUp(self):
        self.producer = pdfWeightProducer("LHEPdfWeight", N_MEMBERS)

    def test_member_weights(self):
        # Three members present, two configured members past the end.
        df = frame(4, "ROOT::RVecF{2.f, 4.f, 1.f}")
        df, branches = self.producer.getWeight(df, return_list_of_branches=True)
        self.assertEqual(
            branches,
            ["weight_pdf_Central"] + [f"weight_pdf_{k}" for k in range(N_MEMBERS)],
        )
        cols = df.AsNumpy(branches)
        self.assertEqual(set(cols["weight_pdf_Central"]), {1.0})
        self.assertEqual(set(cols["weight_pdf_0"]), {1.0})
        self.assertEqual(set(cols["weight_pdf_1"]), {2.0})
        self.assertEqual(set(cols["weight_pdf_2"]), {0.5})
        # Absent members are no-ops.
        self.assertEqual(set(cols["weight_pdf_3"]), {1.0})
        self.assertEqual(set(cols["weight_pdf_4"]), {1.0})

    def test_unusable_nominal_is_a_no_op(self):
        df = frame(3, "rdfentry_ == 1 ? ROOT::RVecF{0.f, 5.f} : ROOT::RVecF{2.f, 4.f}")
        df, branches = self.producer.getWeight(df, return_list_of_branches=True)
        cols = df.AsNumpy(["weight_pdf_1"])
        self.assertEqual(list(cols["weight_pdf_1"]), [2.0, 1.0, 2.0])

    def test_missing_branch_gives_ones(self):
        df = ROOT.RDataFrame(3).Define("weight_gen", "1.f")
        df, branches = self.producer.getWeight(df, return_list_of_branches=True)
        cols = df.AsNumpy(branches)
        for name in branches:
            self.assertEqual(set(cols[name]), {1.0})

    def test_already_built_raises(self):
        df = frame(2, "ROOT::RVecF{1.f, 2.f}").Define("weight_pdf_Central", "1.f")
        with self.assertRaises(RuntimeError):
            self.producer.getWeight(df, return_list_of_branches=True)


class PdfShapeOnlyTest(unittest.TestCase):
    """Emulates what the `base` block does with the denominators the anaCache stores."""

    def test_every_member_keeps_the_nominal_yield(self):
        registerSourceScales("pdf", pdfWeightProducer.scales({"n_members": N_MEMBERS}))
        producer = pdfWeightProducer("LHEPdfWeight", N_MEMBERS)
        members = "ROOT::RVecF{2.f, 2.f + float(rdfentry_), 1.f}"
        df, branches = producer.getWeight(
            frame(6, members), return_list_of_branches=True
        )
        cols = df.AsNumpy(branches + ["weight_gen"])
        gen = [float(w) for w in cols["weight_gen"]]

        # The anaCache denominator of each member: gen weight times that member's weight,
        # summed over every generated event.
        denom = {
            name: sum(g * float(w) for g, w in zip(gen, cols[name]))
            for name in branches
        }
        nominal = sum(gen)
        self.assertAlmostEqual(denom["weight_pdf_Central"], nominal, places=9)
        # Member 0 is the nominal, so its denominator is the Central one.
        self.assertAlmostEqual(denom["weight_pdf_0"], nominal, places=9)

        for k in range(N_MEMBERS):
            name = f"weight_pdf_{k}"
            # weight_base_pdf<k>_rel = (numer_k / D_k) / (numer_C / D_C)
            rel = [
                (float(w) / denom[name]) / (1.0 / denom["weight_pdf_Central"])
                for w in cols[name]
            ]
            total = sum(g * r for g, r in zip(gen, rel))
            self.assertAlmostEqual(total / nominal, 1.0, places=9)
        # ...and member 0's relative weight is exactly 1 for every event.
        rel0 = [
            (float(w) / denom["weight_pdf_0"]) / (1.0 / denom["weight_pdf_Central"])
            for w in cols["weight_pdf_0"]
        ]
        for value in rel0:
            self.assertAlmostEqual(value, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
