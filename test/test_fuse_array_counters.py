#!/usr/bin/env python3
"""FuseAnaTuples with array collections whose size changes with the shift.

A ROOT tree reads an array of a friend tree with the counter of the same name in the main tree,
if there is one. A shifted tree that stored its collection sizes under the central names would
therefore give Central.<array> the shifted size, both where FuseAnaTuples computes the deltas
and where a reader adds them back to the central values. The shifted trees store their counters
as n<collection>_shifted, and the deltas are computed against the central elements only.

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

import ROOT
import uproot

from FLAF.AnaProd.FuseAnaTuples import fuseAnaTuples
from FLAF.Common.Utilities import CreateDataFrame

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine(f".include {flaf_repo}")
ROOT.gInterpreter.Declare('#include "include/Utilities.h"')

# The number of jets of each event in each variation: shifted collections longer, shorter and
# as long as the central one, and empty on either side. A central row with more jets than the
# next one leaves values behind that a read with the wrong size picks up.
N_JETS = {
    ("Central", "Central"): [3, 1, 2, 0, 2, 3, 1, 2, 0, 1],
    ("JES", "Up"): [4, 2, 1, 2, 2, 0, 3, 2, 1, 1],
    ("JES", "Down"): [2, 0, 3, 1, 2, 3, 0, 4, 1, 1],
}
N_EVENTS = 10
# event 8 is selected only in the shifted variations, event 9 only in Central and Up
SELECTED = {
    ("Central", "Central"): [0, 1, 2, 3, 4, 5, 6, 7, 9],
    ("JES", "Up"): list(range(N_EVENTS)),
    ("JES", "Down"): list(range(9)),
}
SHIFTS = {("Central", "Central"): 0, ("JES", "Up"): 1, ("JES", "Down"): -1}
COLUMNS = ["centralJet_pt", "centralJet_hadronFlavour", "ncentralJet", "met_pt"]


def raw_values(variation, event):
    """What anaTupleProducer stores for an event; every difference is exact in float."""
    shift = SHIFTS[variation]
    n = N_JETS[variation][event]
    return {
        "centralJet_pt": [20.0 + 10 * i + event + 0.5 * shift for i in range(n)],
        "centralJet_hadronFlavour": [(event + i + shift) % 6 for i in range(n)],
        "ncentralJet": n,
        "met_pt": 50.0 + event + 0.5 * shift,
    }


def write_input(path, variation):
    shift = SHIFTS[variation]
    entries = SELECTED[variation]
    n_jets = ", ".join(str(n) for n in N_JETS[variation])
    selection = " || ".join(f"rdfentry_ == {e}" for e in entries) or "false"
    df = ROOT.RDataFrame(N_EVENTS).Filter(selection)
    df = df.Define("FullEventId", "static_cast<ULong64_t>(1000 + rdfentry_)")
    df = df.Define("n_jets", f"std::vector<int>{{{n_jets}}}.at(rdfentry_)")
    df = df.Define(
        "centralJet_pt",
        "ROOT::RVecF res(n_jets); for (int i = 0; i < n_jets; ++i)"
        f" res[i] = 20.f + 10 * i + rdfentry_ + 0.5f * {shift}; return res;",
    )
    df = df.Define(
        "centralJet_hadronFlavour",
        "ROOT::RVecI res(n_jets); for (int i = 0; i < n_jets; ++i)"
        f" res[i] = (static_cast<int>(rdfentry_) + i + {shift} + 6) % 6; return res;",
    )
    df = df.Define("met_pt", f"50.f + rdfentry_ + 0.5f * {shift}")
    df.Snapshot(
        "Events",
        path,
        ["FullEventId", "centralJet_pt", "centralJet_hadronFlavour", "met_pt"],
    )


def run_fuse(work_dir):
    raw_dir = os.path.join(work_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    reference = os.path.join(raw_dir, "reference.root")
    ROOT.RDataFrame(N_EVENTS).Define(
        "FullEventId", "static_cast<ULong64_t>(1000 + rdfentry_)"
    ).Snapshot("Events", reference, ["FullEventId"])
    output_files = []
    for variation in N_JETS:
        path = os.path.join(raw_dir, f"{variation[0]}_{variation[1]}.root")
        write_input(path, variation)
        output_files.append(
            {"unc_source": variation[0], "unc_scale": variation[1], "file_name": path}
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
    return os.path.join(fused_dir, "anaTuple.root")


def merge_like(fused, merged):
    """MergeAnaTuples copies every tree with an RDataFrame snapshot of all its columns."""
    options = ROOT.RDF.RSnapshotOptions()
    options.fMode = "RECREATE"
    for tree_name in ["Events", "Events__JES__Up", "Events__JES__Down"]:
        df = ROOT.RDataFrame(tree_name, fused)
        df.Snapshot(
            tree_name, merged, sorted(str(c) for c in df.GetColumnNames()), options
        )
        options.fMode = "UPDATE"


def read_tree(path, tree_name):
    """What a reader sees: CreateDataFrame with the central tree as friend."""
    files = {}
    _, _, central_tree, _ = CreateDataFrame(
        treeName="Events", fileName=path, caches={}, files=files
    )
    _, df, _, _ = CreateDataFrame(
        treeName=tree_name,
        fileName=path,
        caches={},
        files=files,
        centralTree=central_tree if tree_name != "Events" else None,
        filter_valid=False,
    )
    columns = ["FullEventId", "valid"] + COLUMNS
    if tree_name != "Events":
        df = df.Define("central_pt", "Central.centralJet_pt")
        df = df.Define("central_valid", "Central.valid")
        columns += ["central_pt", "central_valid"]
    df = df.Redefine("valid", "static_cast<int>(valid)")
    arrays = df.AsNumpy(columns)
    rows = {}
    for row, event_id in enumerate(arrays["FullEventId"]):
        rows[int(event_id) - 1000] = {
            c: (
                list(arrays[c][row])
                if arrays[c].dtype == object
                else arrays[c][row].item()
            )
            for c in columns
        }
    return rows


class FuseArrayCountersTest:
    """The checks, run on the output of FuseAnaTuples and on a merge-like copy of it."""

    def test_shifted_trees_use_their_own_counters(self):
        with uproot.open(self.path) as f:
            central = f["Events"]
            self.assertEqual(central["centralJet_pt"].count_branch.name, "ncentralJet")
            for variation in ["Up", "Down"]:
                tree = f[f"Events__JES__{variation}"]
                self.assertNotIn("ncentralJet", tree.keys())
                for column in [
                    "centralJet_pt__delta",
                    "centralJet_hadronFlavour__delta",
                ]:
                    self.assertEqual(
                        tree[column].count_branch.name, "ncentralJet_shifted"
                    )

    def test_stored_deltas(self):
        # read without friends: independent of how ROOT resolves friend counters
        with uproot.open(self.path) as f:
            central = f["Events"].arrays(library="np")
            central_rows = dict(zip(central["FullEventId"] - 1000, range(N_EVENTS)))
            for variation in [("JES", "Up"), ("JES", "Down")]:
                tree = f[f"Events__{variation[0]}__{variation[1]}"]
                stored = tree.arrays(library="np")
                for row, event in enumerate(stored["FullEventId"] - 1000):
                    event = int(event)
                    shifted_valid = event in SELECTED[variation]
                    central_valid = event in SELECTED[("Central", "Central")]
                    shifted = raw_values(variation, event) if shifted_valid else None
                    nominal = raw_values(("Central", "Central"), event)
                    self.assertEqual(bool(stored["valid"][row]), shifted_valid)
                    self.assertEqual(
                        bool(central["valid"][central_rows[event]]), central_valid
                    )
                    for column in ["centralJet_pt", "centralJet_hadronFlavour"]:
                        value = list(stored[f"{column}__delta"][row])
                        if not shifted_valid:
                            expected = []
                        elif not central_valid:
                            expected = shifted[column]
                        else:
                            n_common = min(len(shifted[column]), len(nominal[column]))
                            expected = [
                                s - c
                                for s, c in zip(
                                    shifted[column][:n_common],
                                    nominal[column][:n_common],
                                )
                            ] + shifted[column][n_common:]
                        with self.subTest(
                            variation=variation, event=event, column=column
                        ):
                            self.assertEqual(value, expected)

    def test_readers_see_the_stored_values(self):
        for variation in N_JETS:
            tree_name = (
                "Events"
                if variation[0] == "Central"
                else f"Events__{variation[0]}__{variation[1]}"
            )
            rows = read_tree(self.path, tree_name)
            for event in range(N_EVENTS):
                row = rows[event]
                if event not in SELECTED[variation]:
                    self.assertEqual(row["valid"], 0)
                    continue
                self.assertEqual(row["valid"], 1)
                expected = raw_values(variation, event)
                for column in COLUMNS:
                    with self.subTest(tree=tree_name, event=event, column=column):
                        self.assertEqual(row[column], expected[column])
                if tree_name != "Events" and event in SELECTED[("Central", "Central")]:
                    # the central collection seen from a shifted tree keeps its size
                    with self.subTest(tree=tree_name, event=event, column="Central"):
                        self.assertEqual(
                            row["central_pt"],
                            raw_values(("Central", "Central"), event)["centralJet_pt"],
                        )


class TestFusedFile(FuseArrayCountersTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = run_fuse(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestMergedFile(FuseArrayCountersTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "merged.root")
        merge_like(run_fuse(cls.tmp.name), cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
