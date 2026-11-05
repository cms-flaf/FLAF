#!/usr/bin/env python3
"""Theory uncertainties stored as members (Common/TheoryUncertainties.py).

The members of a PDF or QCD-scale uncertainty are reduced to Up/Down per bin once the datasets of a
process are summed: the 7-point envelope for the QCD scale, the Hessian sum (+ alpha_s) for the PDF.
An entry's `variables` restricts it to the fit variables, and `process_groups` lets processes share
one nuisance name. The reduction itself needs no ROOT; the histogram tests do.
"""

import math
import os
import sys
import unittest

import pytest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import FLAF.Common.TheoryUncertainties as TU  # noqa: E402

QCD = {
    "expression": "weight_base_qcd_scale{scale}_rel * final_weight",
    "name": "QCDscale_{process}_{}",
    "reduction": "envelope",
    "variables": ["^DNN_m[0-9]+_vs_HME$"],
    "process_groups": {"DY": ["DY_M_10to50", "DY_M_50"]},
}
PDF = {
    "expression": "weight_base_pdf{scale}_rel * final_weight",
    "name": "pdf_{process}_{}",
    "reduction": "hessian",
    "variables": ["^DNN_m[0-9]+_vs_HME$"],
}
QCD_SCALES = [
    "_muR0p5_muF0p5",
    "_muR0p5_muF1",
    "_muR0p5_muF2",
    "_muR1_muF0p5",
    "_muR1_muF2",
    "_muR2_muF0p5",
    "_muR2_muF1",
    "_muR2_muF2",
]
PDF_SCALES = [str(k) for k in range(103)]


class ConfigTest(unittest.TestCase):
    def test_reduction(self):
        self.assertEqual(TU.reduction(QCD), "envelope")
        self.assertEqual(TU.reduction(PDF), "hessian")
        self.assertIsNone(TU.reduction({"name": "ps_isr_{}"}))
        with self.assertRaises(ValueError):
            TU.reduction({"reduction": "replicas"})

    def test_variables(self):
        self.assertTrue(TU.applies_to_variable(QCD, "DNN_m300_vs_HME"))
        self.assertFalse(TU.applies_to_variable(QCD, "lep1_pt"))
        self.assertFalse(TU.applies_to_variable(QCD, "DNN_M300_Signal"))
        self.assertTrue(TU.applies_to_variable({"name": "x_{}"}, "lep1_pt"))
        self.assertTrue(TU.applies_to_any_variable(QCD, ["lep1_pt", "DNN_m500_vs_HME"]))
        self.assertFalse(TU.applies_to_any_variable(QCD, ["lep1_pt"]))

    def test_names(self):
        self.assertEqual(TU.format_name(QCD, "Up", "TT"), "QCDscale_TT_Up")
        self.assertEqual(TU.format_name(QCD, "Down", "DY_M_10to50"), "QCDscale_DY_Down")
        self.assertEqual(TU.format_name(QCD, "Up", "DY_M_50"), "QCDscale_DY_Up")
        # an ordinary entry, without {process}, is formatted as before
        self.assertEqual(TU.format_name({"name": "ps_isr_{}"}, "Up", "TT"), "ps_isr_Up")

    def test_regex_groups(self):
        entry = dict(
            QCD,
            process_groups={
                "GluGluToRadion_bbWW_2L": ["^GluGluToRadion_bbWW_2L_[0-9]+$"],
                "DY": ["DY_M_10to50", "DY_M_50"],
            },
        )
        self.assertEqual(
            TU.format_name(entry, "Up", "GluGluToRadion_bbWW_2L_500"),
            "QCDscale_GluGluToRadion_bbWW_2L_Up",
        )
        self.assertEqual(
            TU.format_name(entry, "Up", "GluGluToRadion_bbWW_1L_500"),
            "QCDscale_GluGluToRadion_bbWW_1L_500_Up",
        )
        # a literal member is not a pattern: "DY" alone is not in the group
        self.assertEqual(TU.process_name(entry, "DY"), "DY")

    def test_members_used(self):
        used = TU.members_used(QCD, QCD_SCALES)
        self.assertEqual(len(used), 6)
        self.assertNotIn("_muR0p5_muF2", used)
        self.assertNotIn("_muR2_muF0p5", used)
        self.assertEqual(TU.members_used(PDF, PDF_SCALES), PDF_SCALES[1:])


class ReduceCellsTest(unittest.TestCase):
    def test_envelope_is_the_7_point_range(self):
        nominal = [10.0, 20.0]
        members = {s: [10.0, 20.0] for s in QCD_SCALES}
        members["_muR2_muF2"] = [8.0, 21.0]
        members["_muR0p5_muF0p5"] = [12.5, 19.0]
        # the excluded corners are outside the envelope and must not count
        members["_muR0p5_muF2"] = [100.0, 0.0]
        members["_muR2_muF0p5"] = [0.0, 100.0]
        up, down = TU.reduce_cells(QCD, nominal, members)
        self.assertEqual(up, [12.5, 21.0])
        self.assertEqual(down, [8.0, 19.0])

    def test_envelope_contains_the_nominal(self):
        nominal = [10.0]
        members = {s: [11.0] for s in QCD_SCALES}
        up, down = TU.reduce_cells(QCD, nominal, members)
        self.assertEqual((up, down), ([11.0], [10.0]))

    def test_hessian_with_alphas(self):
        nominal = [100.0, 50.0]
        members = {s: [100.0, 50.0] for s in PDF_SCALES}
        members["1"] = [103.0, 50.0]
        members["2"] = [96.0, 51.0]
        members["101"] = [98.0, 50.0]  # alpha_s down
        members["102"] = [104.0, 50.0]  # alpha_s up
        up, down = TU.reduce_cells(PDF, nominal, members)
        sigma0 = math.sqrt(
            3.0**2 + 4.0**2 + 3.0**2
        )  # eigenvectors 3, -4; alpha_s (104-98)/2
        sigma1 = 1.0
        self.assertAlmostEqual(up[0], 100.0 + sigma0)
        self.assertAlmostEqual(down[0], 100.0 - sigma0)
        self.assertAlmostEqual(up[1], 50.0 + sigma1)
        self.assertAlmostEqual(down[1], 50.0 - sigma1)

    def test_hessian_settings(self):
        nominal = [100.0]
        members = {s: [100.0] for s in PDF_SCALES}
        members["101"], members["102"] = [98.0], [104.0]
        entry = dict(PDF, alphas_scale=0.75)
        up, _ = TU.reduce_cells(entry, nominal, members)
        self.assertAlmostEqual(up[0], 100.0 + 3.0 * 0.75)
        up, _ = TU.reduce_cells(dict(PDF, alphas=None), nominal, members)
        self.assertAlmostEqual(up[0], 100.0)

    def test_hessian_without_finite_alphas_uses_the_eigenvectors(self):
        # the X->HH signal samples carry no alpha_s members: they are NaN in the anaTuples
        nominal = [100.0]
        members = {s: [100.0] for s in PDF_SCALES}
        members["1"] = [103.0]
        members["101"], members["102"] = [float("nan")], [float("nan")]
        self.assertFalse(TU.alphas_available(PDF, members))
        up, down = TU.reduce_cells(PDF, nominal, members)
        self.assertAlmostEqual(up[0], 103.0)
        self.assertAlmostEqual(down[0], 97.0)

    def test_hessian_non_finite_eigenvector_raises(self):
        members = {s: [1.0] for s in PDF_SCALES}
        members["7"] = [float("nan")]
        with self.assertRaises(ValueError):
            TU.reduce_cells(PDF, [1.0], members)

    def test_hessian_missing_member_raises(self):
        members = {s: [1.0] for s in PDF_SCALES if s != "57"}
        with self.assertRaises(ValueError):
            TU.reduce_cells(PDF, [1.0], members)


@pytest.mark.needs_root
class ReduceHistDictTest(unittest.TestCase):
    """The step HistMergerFromHists runs once every dataset is summed into its process."""

    @classmethod
    def setUpClass(cls):
        global ROOT
        import ROOT

        ROOT.gROOT.SetBatch(True)

    def hist(self, name, values, dim=1):
        if dim == 1:
            h = ROOT.TH1D(name, "", len(values), 0, len(values))
            for i, v in enumerate(values, start=1):
                h.SetBinContent(i, v)
                h.SetBinError(i, 0.1 * v)
        else:
            h = ROOT.TH2D(name, "", 2, 0, 2, 2, 0, 2)
            for i, v in enumerate(values):
                h.SetBinContent(h.GetBin(i % 2 + 1, i // 2 + 1), v)
        h.SetDirectory(0)
        return h

    def test_members_replaced_by_up_down(self):
        key = ("eE", "SR", "res2b")
        nominal = self.hist("nom", [10.0, 20.0])
        hists = {(key, ("Central", "Central")): nominal}
        for s in QCD_SCALES:
            hists[(key, ("qcd_scale", s))] = self.hist(s, [10.0, 20.0])
        hists[(key, ("qcd_scale", "_muR2_muF2"))] = self.hist("a", [9.0, 22.0])
        hists[(key, ("ps_isr", "Up"))] = self.hist("isr", [11.0, 21.0])
        data = {(key, ("Central", "Central")): self.hist("data", [30.0, 40.0])}
        all_hists = {"TT": hists, "data": data}
        TU.reduce_in_hist_dict(all_hists, "qcd_scale", QCD, QCD_SCALES)
        remaining = sorted(u for (_, u) in hists)
        self.assertEqual(
            remaining,
            sorted(
                [
                    ("Central", "Central"),
                    ("ps_isr", "Up"),
                    ("qcd_scale", "Up"),
                    ("qcd_scale", "Down"),
                ]
            ),
        )
        up, down = (
            hists[(key, ("qcd_scale", "Up"))],
            hists[(key, ("qcd_scale", "Down"))],
        )
        self.assertEqual([up.GetBinContent(i) for i in (1, 2)], [10.0, 22.0])
        self.assertEqual([down.GetBinContent(i) for i in (1, 2)], [9.0, 20.0])
        # the nominal keeps its contents and its errors carry over to Up/Down
        self.assertEqual(nominal.GetBinContent(1), 10.0)
        self.assertAlmostEqual(up.GetBinError(2), 2.0)
        self.assertEqual(list(data), [(key, ("Central", "Central"))])

    def test_2d_and_overflow(self):
        key = ("muMu", "SR", "boosted")
        nominal = self.hist("nom2", [1.0, 2.0, 3.0, 4.0], dim=2)
        nominal.SetBinContent(0, 5.0)  # underflow cell
        hists = {(key, ("Central", "Central")): nominal}
        for s in PDF_SCALES:
            hists[(key, ("pdf", s))] = nominal.Clone(f"m{s}")
        shifted = nominal.Clone("m1")
        shifted.SetBinContent(0, 7.0)
        hists[(key, ("pdf", "1"))] = shifted
        TU.reduce_in_hist_dict({"DY_M_50": hists}, "pdf", PDF, PDF_SCALES)
        up = hists[(key, ("pdf", "Up"))]
        self.assertIsInstance(up, ROOT.TH2D)
        self.assertAlmostEqual(up.GetBinContent(0), 7.0)
        self.assertAlmostEqual(up.GetBinContent(up.GetBin(2, 2)), 4.0)

    def test_missing_member_raises(self):
        key = ("eE", "SR", "res2b")
        hists = {(key, ("Central", "Central")): self.hist("n", [1.0])}
        for s in QCD_SCALES[:-1]:
            hists[(key, ("qcd_scale", s))] = self.hist(s, [1.0])
        with self.assertRaises(RuntimeError):
            TU.reduce_in_hist_dict({"TT": hists}, "qcd_scale", QCD, QCD_SCALES)


@pytest.mark.needs_root
class WeightsByVariablesTest(unittest.TestCase):
    """HistProducerFromNTuple fills a theory weight only for the variables it lists."""

    def test_groups(self):
        try:
            import FLAF.Analysis.HistProducerFromNTuple as producer
        except ImportError as e:
            raise unittest.SkipTest(
                f"HistProducerFromNTuple needs an analysis environment: {e}"
            )
        unc_cfg = {
            "norm": {"qcd_scale": QCD, "ps_isr": {"name": "ps_isr_{}"}},
            "shape": {},
        }
        weights = [
            ("Central", "Central", "weight_Central"),
            ("ps_isr", "Up", "weight_ps_isr_Up"),
            ("qcd_scale", "_muR2_muF2", "weight_qcd_scale__muR2_muF2"),
        ]
        groups = producer.weights_by_variables(
            weights, unc_cfg, ["lep1_pt", "DNN_m300_vs_HME"]
        )
        self.assertEqual(
            groups,
            [
                (["lep1_pt", "DNN_m300_vs_HME"], weights[:2]),
                (["DNN_m300_vs_HME"], weights[2:]),
            ],
        )
        # no fit variable in the job: the theory weight is not filled at all
        groups = producer.weights_by_variables(weights, unc_cfg, ["lep1_pt"])
        self.assertEqual(groups, [(["lep1_pt"], weights[:2])])


if __name__ == "__main__":
    unittest.main()
