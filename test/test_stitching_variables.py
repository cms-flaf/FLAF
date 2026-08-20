#!/usr/bin/env python3
"""Stitching variables must survive the anaTuple.

A stitcher derives its bin variables from nanoAOD collections (GenPart/LHEPart) that the
anaTuple does not keep, and the merge stage evaluates the very same bin selections to pick
the cross-section and the denominator. It therefore has to read the gen-level information
back from branches the analysis stored (see FLAF/docs/concepts/stitching.md); falling back
to the nanoAOD collections is only possible while they are still around.
"""

import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import ROOT

from FLAF.Processors.MCStitchingDYMll import DYMllStitcher
from FLAF.Processors.MCStitchingDYtautau import DYtautauStitcher
from FLAF.Processors.MCStitchingTT import TTStitcher

ROOT.gROOT.SetBatch(True)

# statusFlags bits used by the gen-level identification: isHardProcess (7), isLastCopy (13).
HARD_LAST_COPY = (1 << 7) | (1 << 13)
LAST_COPY = 1 << 13

# Z -> tau tau, both taus decaying hadronically: tau (pt 50) -> nu_tau (pt 20) + hadrons,
# so each visible pt is 30 > 20 and the gen filter accepts the event.
TAUTAU_NANOAOD = {
    "GenPart_pdgId": f"ROOT::RVecI{{15, -15, 16, -16}}",
    "GenPart_statusFlags": f"ROOT::RVecI{{{HARD_LAST_COPY}, {HARD_LAST_COPY}, 0, 0}}",
    "GenPart_genPartIdxMother": "ROOT::RVecI{-1, -1, 0, 1}",
    "GenPart_pt": "ROOT::RVecF{50.f, 50.f, 20.f, 20.f}",
    "GenPart_eta": "ROOT::RVecF{0.f, 0.f, 0.f, 0.f}",
    "GenPart_phi": "ROOT::RVecF{0.f, 3.14159f, 0.f, 3.14159f}",
    "GenPart_mass": "ROOT::RVecF{1.777f, 1.777f, 0.f, 0.f}",
}

# t -> W(-> mu nu) b, tbar -> W(-> u dbar) bbar: exactly one leptonically decaying W.
TT_NANOAOD = {
    "GenPart_pdgId": "ROOT::RVecI{6, -6, 24, 5, -24, -5, -13, 14, 2, -1}",
    "GenPart_statusFlags": (
        f"ROOT::RVecI{{{LAST_COPY}, {LAST_COPY}, {LAST_COPY}, 0, {LAST_COPY}, "
        "0, 0, 0, 0, 0}"
    ),
    "GenPart_genPartIdxMother": "ROOT::RVecI{-1, -1, 0, 0, 1, 1, 2, 2, 4, 4}",
}

# Drell-Yan to two muons at LHE level, back to back with pt 40 each.
DY_NANOAOD = {
    "LHEPart_pdgId": "ROOT::RVecI{13, -13}",
    "LHEPart_status": "ROOT::RVecI{1, 1}",
    "LHEPart_pt": "ROOT::RVecF{40.f, 40.f}",
    "LHEPart_eta": "ROOT::RVecF{0.f, 0.f}",
    "LHEPart_phi": "ROOT::RVecF{0.f, 3.14159f}",
    "LHEPart_mass": "ROOT::RVecF{0.105f, 0.105f}",
}


def make_stitcher(cls):
    """A stitcher that skips the cross-section configuration: only defineVariables is used."""
    return cls(
        global_params={},
        processor_entry={"useDatasetCrossSection": True},
        stage="AnaTuple",
    )


def make_df(columns):
    df = ROOT.RDataFrame(4)
    for name, expression in columns.items():
        df = df.Define(name, expression)
    return df


def values(df, column):
    return list(df.Take[df.GetColumnType(column)](column).GetValue())


class TestStitchingVariables(unittest.TestCase):
    def test_defined_from_stored_branches_without_nanoaod_collections(self):
        # The regression: the merge stage reads an anaTuple, where GenPart/LHEPart are gone.
        # Every stitcher must be satisfied by the stored gen-process information alone.
        df = make_df(
            {
                "TauTauInfo_passFilter": "rdfentry_ % 2 == 0",
                "TTInfo_nLeptonicW": "static_cast<int>(rdfentry_ % 3)",
                "DYInfo_flavor": "13",
                "DYInfo_mll": "91.f",
            }
        )
        df = make_stitcher(DYtautauStitcher).defineVariables(df)
        df = make_stitcher(TTStitcher).defineVariables(df)
        df = make_stitcher(DYMllStitcher).defineVariables(df)

        self.assertEqual(values(df, "DY_tautau_filter"), [1, 0, 1, 0])
        self.assertEqual(values(df, "TT_n_leptonic_W"), [0, 1, 2, 0])
        self.assertEqual(values(df, "LHE_dilep_flavor"), [13] * 4)
        self.assertEqual(values(df, "LHE_mll"), [91.0] * 4)

    def test_defined_from_nanoaod_collections_when_not_stored(self):
        # The anaTuple production stage, where the collections are still available.
        columns = dict(TAUTAU_NANOAOD)
        columns.update(DY_NANOAOD)
        df = make_df(columns)
        df = make_stitcher(DYtautauStitcher).defineVariables(df)
        df = make_stitcher(DYMllStitcher).defineVariables(df)

        self.assertEqual(values(df, "DY_tautau_filter"), [1] * 4)
        self.assertEqual(values(df, "LHE_dilep_flavor"), [13] * 4)
        self.assertAlmostEqual(values(df, "LHE_mll")[0], 80.0, delta=1.0)

        df_tt = make_stitcher(TTStitcher).defineVariables(make_df(TT_NANOAOD))
        self.assertEqual(values(df_tt, "TT_n_leptonic_W"), [1] * 4)

    def test_stored_value_wins_over_the_collections(self):
        # Both available: the stored branch is what the denominators were built from, so it
        # must be the one that is used.
        columns = dict(TAUTAU_NANOAOD)
        columns["TauTauInfo_passFilter"] = "false"
        df = make_stitcher(DYtautauStitcher).defineVariables(make_df(columns))
        self.assertEqual(values(df, "DY_tautau_filter"), [0] * 4)

    def test_already_defined_variable_is_kept(self):
        df = make_df({"DY_tautau_filter": "7", "TauTauInfo_passFilter": "true"})
        df = make_stitcher(DYtautauStitcher).defineVariables(df)
        self.assertEqual(values(df, "DY_tautau_filter"), [7] * 4)


if __name__ == "__main__":
    unittest.main()
