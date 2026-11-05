#!/usr/bin/env python3
"""Theory uncertainties stored as members (Common/TheoryUncertainties.py).

HistMerger writes every member of a PDF or QCD-scale uncertainty as its own histogram,
``<process>_<name>`` with the member in place of ``{}``; reducing them to Up/Down is left to the
datacard step. The meta-era sum treats a member as a variation of its process: an era without
it enters with its central histogram, and a process name ending like a member (``..._MY_100``)
is not mistaken for one.
"""

import os
import sys
import tempfile
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
}
PDF = {
    "expression": "weight_base_pdf{scale}_rel * final_weight",
    "name": "pdf_{process}_{}",
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
UP_DOWN = ["Up", "Down"]


class NamesTest(unittest.TestCase):
    def test_has_members(self):
        self.assertTrue(TU.has_members(QCD_SCALES, UP_DOWN))
        self.assertTrue(TU.has_members(PDF_SCALES, UP_DOWN))
        self.assertFalse(TU.has_members(["Up", "Down"], UP_DOWN))

    def test_member_names(self):
        # the member's leading underscore is dropped
        self.assertEqual(
            TU.format_name(QCD, "_muR2_muF1", "TT"), "QCDscale_TT_muR2_muF1"
        )
        self.assertEqual(TU.format_name(PDF, "57", "TT"), "pdf_TT_57")

    def test_process_names(self):
        # every process of the physics model has its own
        self.assertEqual(TU.format_name(QCD, "Up", "DY_M_50"), "QCDscale_DY_M_50_Up")
        self.assertEqual(
            TU.format_name(PDF, "0", "GluGluToRadion_bbWW_2L_500"),
            "pdf_GluGluToRadion_bbWW_2L_500_0",
        )
        # an ordinary entry, without {process}, is formatted as before
        self.assertEqual(TU.format_name({"name": "ps_isr_{}"}, "Up", "TT"), "ps_isr_Up")


@pytest.mark.needs_root
class SumEraMembersTest(unittest.TestCase):
    """SumEraFiles with member_names: members are variations of their process."""

    PROCESSES = ["TT", "XtoYH_MX_1000_MY_100"]

    def setUp(self):
        try:
            import ROOT
        except ImportError as e:
            raise unittest.SkipTest(f"needs ROOT: {e}")
        import FLAF.Common.HistHelper as HistHelper

        ROOT.gROOT.SetBatch(True)
        self.ROOT, self.HistHelper = ROOT, HistHelper
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def member_names(self, process):
        return [
            f"{process}_{TU.format_name(PDF, m, process)}" for m in ("0", "1", "100")
        ]

    def write_era(self, name, contents):
        """One merged file: eE/SR/res2b with one-bin histograms {name: value}."""
        path = os.path.join(self.tmp.name, name)
        out = self.ROOT.TFile(path, "RECREATE")
        d = self.HistHelper.mkdir_recursive(out, "eE/SR/res2b")
        for hist_name, value in contents.items():
            h = self.ROOT.TH1D(hist_name, hist_name, 1, 0.0, 1.0)
            h.SetDirectory(0)
            h.SetBinContent(1, value)
            d.WriteTObject(h, hist_name)
        out.Close()
        return path

    def summed(self, path):
        f = self.ROOT.TFile.Open(path)
        d = f.Get("eE/SR/res2b")
        values = {
            k.GetName(): d.Get(k.GetName()).GetBinContent(1) for k in d.GetListOfKeys()
        }
        f.Close()
        return values

    def test_members_summed_and_missing_member_takes_central(self):
        era_a = self.write_era(
            "a.root",
            {
                "TT": 10.0,
                "TT_pdf_TT_0": 11.0,
                "TT_pdf_TT_1": 12.0,
                "TT_pdf_TT_100": 13.0,
                # a process whose name ends like a member
                "XtoYH_MX_1000_MY_100": 1.0,
                "XtoYH_MX_1000_MY_100_pdf_XtoYH_MX_1000_MY_100_1": 2.0,
            },
        )
        # era b misses TT's member 100 and the signal's member 1
        era_b = self.write_era(
            "b.root",
            {
                "TT": 20.0,
                "TT_pdf_TT_0": 21.0,
                "TT_pdf_TT_1": 22.0,
                "XtoYH_MX_1000_MY_100": 3.0,
            },
        )
        out = os.path.join(self.tmp.name, "sum.root")
        self.HistHelper.SumEraFiles([era_a, era_b], out, UP_DOWN, self.member_names)
        values = self.summed(out)
        self.assertEqual(values["TT"], 30.0)
        self.assertEqual(values["TT_pdf_TT_0"], 32.0)
        self.assertEqual(values["TT_pdf_TT_1"], 34.0)
        # era b has no member 100: its central TT enters instead
        self.assertEqual(values["TT_pdf_TT_100"], 33.0)
        self.assertEqual(values["XtoYH_MX_1000_MY_100"], 4.0)
        self.assertEqual(values["XtoYH_MX_1000_MY_100_pdf_XtoYH_MX_1000_MY_100_1"], 5.0)


if __name__ == "__main__":
    unittest.main()
