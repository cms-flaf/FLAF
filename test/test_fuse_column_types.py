#!/usr/bin/env python3
"""FuseAnaTuples must not change the type of any column.

Columns sharing the text before their first underscore are stored as one collection with one
counter. A scalar stored next to arrays of the same prefix would silently become an array (one
copy per entry), as HH_bbWW's TTInfo_nLeptonicW did next to the per-top TTInfo_* arrays; that
has to stop the production so the names can be fixed in the anaTuple definition.

Needs ROOT, uproot and awkward: run inside an analysis environment (source env.sh).
"""

import os
import sys
import tempfile
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import awkward as ak
import ROOT
import uproot

from FLAF.AnaProd.FuseAnaTuples import checkColumnTypes, columnType, fuseAnaTuples
from FLAF.Common.TupleHelpers import defineColumnGrouping

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine(f".include {flaf_repo}")
ROOT.gInterpreter.Declare('#include "include/Utilities.h"')

N_EVENTS = 8
SELECTED = {
    ("Central", "Central"): [0, 2, 3, 4, 5, 6],
    ("JES", "Up"): [0, 1, 3, 4, 5, 6, 7],
}
# one column of every type an anaTuple uses
COLUMNS = {
    "FullEventId": "static_cast<ULong64_t>(1000 + rdfentry_)",
    "isData": "rdfentry_ % 2 == 0",
    "LHE_NpNLO": "static_cast<UChar_t>(rdfentry_ % 3)",
    "channelId": "static_cast<int>(rdfentry_)",
    "event": "static_cast<ULong64_t>(rdfentry_)",
    "weight_gen": "static_cast<float>(1.5 * rdfentry_)",
    "PV_z": "static_cast<double>(0.25 * rdfentry_)",
    "centralJet_pt": "ROOT::RVecF(rdfentry_ % 3, 30.f + rdfentry_)",
    "centralJet_partonFlavour": "ROOT::RVecI(rdfentry_ % 3, 5)",
    "centralJet_passId": "ROOT::RVecB(rdfentry_ % 3, true)",
    "centralJet_hadronFlavour": "ROOT::RVec<UChar_t>(rdfentry_ % 3, 5)",
    "centralJet_jetId": "ROOT::RVec<Short_t>(rdfentry_ % 3, 6)",
    "BeamSpot_type": "static_cast<Char_t>(rdfentry_ % 2)",
    "run": "static_cast<UInt_t>(355000 + rdfentry_)",
}


def write_input(path, entries, columns):
    selection = " || ".join(f"rdfentry_ == {e}" for e in entries) or "false"
    df = ROOT.RDataFrame(N_EVENTS).Filter(selection)
    for name, expression in columns.items():
        df = df.Define(name, expression)
    df.Snapshot("Events", path, list(columns))


def run_fuse(work_dir, columns):
    raw_dir = os.path.join(work_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    reference = os.path.join(raw_dir, "reference.root")
    write_input(
        reference, list(range(N_EVENTS)), {"FullEventId": COLUMNS["FullEventId"]}
    )
    output_files = []
    for (source, scale), entries in SELECTED.items():
        path = os.path.join(raw_dir, f"{source}_{scale}.root")
        write_input(path, entries, columns)
        output_files.append(
            {"unc_source": source, "unc_scale": scale, "file_name": path}
        )
    config = {
        "output_files": output_files,
        "reference_file": reference,
        "tree_name": "Events",
        "full_event_id_column": "FullEventId",
    }
    fused_dir = os.path.join(work_dir, "fused")
    os.makedirs(fused_dir, exist_ok=True)
    fuseAnaTuples(
        config=config, work_dir=fused_dir, tuple_output="anaTuple.root", verbose=0
    )
    return os.path.join(fused_dir, "anaTuple.root"), output_files


class TestCollectionGrouping(unittest.TestCase):
    def test_scalars_and_arrays_of_one_prefix_are_refused(self):
        arrays = {
            "TTInfo_nLeptonicW": ak.Array([2, 1]),
            "TTInfo_top_pt": ak.Array([[1.0, 2.0], [3.0, 4.0]]),
        }
        with self.assertRaisesRegex(RuntimeError, r"TTInfo.*\['nLeptonicW'\]"):
            defineColumnGrouping(arrays, list(arrays), verbose=0)

    def test_arrays_of_different_lengths_are_refused(self):
        arrays = {
            "Jet_pt": ak.Array([[1.0, 2.0], [3.0]]),
            "Jet_eta": ak.Array([[1.0], [3.0]]),
        }
        with self.assertRaisesRegex(RuntimeError, "different lengths"):
            defineColumnGrouping(arrays, list(arrays), verbose=0)

    def test_scalar_or_array_collections_are_kept(self):
        arrays = {
            "lep1_pt": ak.Array([1.0, 2.0]),
            "lep1_charge": ak.Array([1, -1]),
            "Jet_pt": ak.Array([[1.0, 2.0], [3.0]]),
            "Jet_eta": ak.Array([[0.5, 1.5], [2.5]]),
            "nJet": ak.Array([2, 1]),
        }
        grouped = defineColumnGrouping(arrays, list(arrays), verbose=0)
        self.assertEqual(sorted(grouped), ["Jet", "lep1"])


class TestFuseColumnTypes(unittest.TestCase):
    def test_every_type_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, output_files = run_fuse(tmp, COLUMNS)
            with uproot.open(path) as f:
                central = {b.name: columnType(b.typename) for b in f["Events"].branches}
                shifted = {
                    b.name: columnType(b.typename)
                    for b in f["Events__JES__Up"].branches
                }
            with uproot.open(output_files[0]["file_name"]) as f:
                raw = {b.name: columnType(b.typename) for b in f["Events"].branches}
            for name, raw_type in raw.items():
                with self.subTest(column=name):
                    self.assertEqual(central[name], raw_type)
                    if name != "FullEventId":
                        self.assertEqual(shifted[f"{name}__delta"], raw_type)

    def test_a_scalar_sharing_a_prefix_with_arrays_stops_the_fusion(self):
        columns = dict(COLUMNS)
        columns["TTInfo_nLeptonicW"] = "static_cast<int>(rdfentry_ % 3)"
        columns["TTInfo_top_pt"] = "ROOT::RVecF(2, 100.f)"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "TTInfo.*nLeptonicW"):
                run_fuse(tmp, columns)

    def test_a_type_change_is_reported(self):
        # the last line of defence: whatever the reason, a column that comes out with another
        # type than it went in stops the fusion
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "raw.root")
            write_input(raw, list(range(N_EVENTS)), COLUMNS)
            fused = os.path.join(tmp, "fused.root")
            with uproot.recreate(fused) as f:
                f["Events"] = {
                    "FullEventId": ak.Array(list(range(N_EVENTS))),
                    "channelId": ak.Array([[1]] * N_EVENTS),
                }
            inputs = {("Central", "Central"): {"file_name": raw}}
            with self.assertRaisesRegex(
                RuntimeError, r"Events/channelId.*\(False, 'int32_t'\) -> \(True"
            ):
                checkColumnTypes(inputs, fused, "Events", ["valid", "FullEventId"])


if __name__ == "__main__":
    unittest.main()
