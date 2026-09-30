#!/usr/bin/env python3
"""FuseAnaTuples with anaTuple_shift_invariant_columns.

Columns declared shift-invariant are stored in the central tree only, and a placeholder row of
the central tree (an event selected only in a shifted variation) gets them from the variation.
A reader of a shifted tree, which takes a column it lacks from the central tree, must then see
exactly what it saw when the column was stored as <name>__delta.

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
import numpy as np
import ROOT
import uproot

from FLAF.AnaProd.FuseAnaTuples import fuseAnaTuples
from FLAF.Common.Utilities import CreateDataFrame

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine(f".include {flaf_repo}")
ROOT.gInterpreter.Declare('#include "include/Utilities.h"')

N_EVENTS = 12
# Events selected in each variation. Event 1 is selected only in Up, event 2 only in Down,
# event 3 in Up and Down but not in the central selection, event 4 nowhere.
SELECTED = {
    ("Central", "Central"): [0, 5, 6, 7, 8, 9, 10, 11],
    ("JES", "Up"): [0, 1, 3, 5, 6, 7, 8, 9, 10, 11],
    ("JES", "Down"): [0, 2, 3, 5, 6, 7, 8, 9, 10],
}
GEN_PATTERNS = ["^weight_gen$", "^LHE_", "^LHEPart_"]


def write_input(path, entries, variation, gen_offset=None):
    """A per-variation anaTuple as anaTupleProducer writes it: RDF snapshot, RVec arrays."""
    selection = " || ".join(f"rdfentry_ == {e}" for e in entries) or "false"
    shift = (
        0.5 if variation == ("JES", "Up") else -0.5 if variation[0] == "JES" else 0.0
    )
    df = ROOT.RDataFrame(N_EVENTS).Filter(selection)
    df = df.Define("FullEventId", "static_cast<ULong64_t>(1000 + rdfentry_)")
    df = df.Define("weight_gen", "static_cast<float>(rdfentry_ % 2 == 0 ? 1 : -1)")
    df = df.Define("LHE_Vpt", "static_cast<float>(10.5 * rdfentry_)")
    df = df.Define("LHE_NpNLO", "static_cast<UChar_t>(rdfentry_ % 3)")
    df = df.Define(
        "LHEPart_pt",
        "ROOT::RVecF(rdfentry_ % 3 + 1, static_cast<float>(rdfentry_) + 0.25f)",
    )
    df = df.Define("LHEPart_pdgId", "ROOT::RVecI(rdfentry_ % 3 + 1, 11)")
    # reco quantities: change with the shift; same number of jets in every variation
    df = df.Define("centralJet_pt", f"ROOT::RVecF{{30.f + {shift}f, 40.f + {shift}f}}")
    df = df.Define("met_pt", f"static_cast<float>(50. + rdfentry_ + {shift})")
    if gen_offset is not None:
        event, value = gen_offset
        df = df.Redefine("LHE_Vpt", f"rdfentry_ == {event} ? {value}f : LHE_Vpt")
    columns = [
        "FullEventId",
        "weight_gen",
        "LHE_Vpt",
        "LHE_NpNLO",
        "LHEPart_pt",
        "LHEPart_pdgId",
        "centralJet_pt",
        "met_pt",
    ]
    df.Snapshot("Events", path, columns)


def write_reference(path):
    df = ROOT.RDataFrame(N_EVENTS).Define(
        "FullEventId", "static_cast<ULong64_t>(1000 + rdfentry_)"
    )
    df.Snapshot("Events", path, ["FullEventId"])


def run_fuse(work_dir, patterns, selected=SELECTED, gen_offsets=None):
    gen_offsets = gen_offsets or {}
    raw_dir = os.path.join(work_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    reference = os.path.join(raw_dir, "reference.root")
    write_reference(reference)
    output_files = []
    for variation, entries in selected.items():
        path = os.path.join(raw_dir, f"{variation[0]}_{variation[1]}.root")
        write_input(path, entries, variation, gen_offsets.get(variation))
        output_files.append(
            {"unc_source": variation[0], "unc_scale": variation[1], "file_name": path}
        )
    config = {
        "output_files": output_files,
        "reference_file": reference,
        "tree_name": "Events",
        "full_event_id_column": "FullEventId",
        "shift_invariant_columns": patterns,
    }
    fused_dir = os.path.join(work_dir, "fused")
    os.makedirs(fused_dir, exist_ok=True)
    fuseAnaTuples(
        config=config, work_dir=fused_dir, tuple_output="anaTuple.root", verbose=0
    )
    return os.path.join(fused_dir, "anaTuple.root")


def read_tree(path, tree_name, columns):
    """What a reader sees in a tree: CreateDataFrame with the central tree as friend."""
    files = {}
    central_tree = None
    if tree_name != "Events":
        _, _, central_tree, _ = CreateDataFrame(
            treeName="Events", fileName=path, caches={}, files=files
        )
    _, df, _, _ = CreateDataFrame(
        treeName=tree_name,
        fileName=path,
        caches={},
        files=files,
        centralTree=central_tree,
        filter_valid=False,
    )
    # AsNumpy turns bool and unsigned char into Python bools: compare them as int instead
    read = {}
    for column in ["valid", "FullEventId"] + columns:
        if df.GetColumnType(column) in ["bool", "Bool_t", "UChar_t", "unsigned char"]:
            df = df.Define(f"_int_{column}", f"static_cast<int>({column})")
            read[column] = f"_int_{column}"
        else:
            read[column] = column
    arrays = df.AsNumpy(list(read.values()))
    result = {}
    for column, name in read.items():
        values = arrays[name]
        if values.dtype == object:
            values = np.array([list(v) for v in values], dtype=object)
        result[column] = values
    return result


def same(a, b):
    if a.dtype == object:
        return all(list(x) == list(y) for x, y in zip(a, b))
    return np.array_equal(a, b)


class TestFuseShiftInvariantColumns(unittest.TestCase):
    COLUMNS = [
        "weight_gen",
        "LHE_Vpt",
        "LHE_NpNLO",
        "LHEPart_pt",
        "LHEPart_pdgId",
        "centralJet_pt",
        "met_pt",
    ]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.plain = run_fuse(os.path.join(cls.tmp.name, "plain"), [])
        cls.declared = run_fuse(os.path.join(cls.tmp.name, "declared"), GEN_PATTERNS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def branches(self, path, tree_name):
        with uproot.open(path) as f:
            return set(f[tree_name].keys())

    def test_declared_columns_leave_the_shifted_trees(self):
        for tree_name in ["Events__JES__Up", "Events__JES__Down"]:
            plain = self.branches(self.plain, tree_name)
            declared = self.branches(self.declared, tree_name)
            removed = plain - declared
            # the counter of the fully declared LHEPart collection goes too
            self.assertEqual(
                removed,
                {
                    "weight_gen__delta",
                    "LHE_Vpt__delta",
                    "LHE_NpNLO__delta",
                    "LHEPart_pt__delta",
                    "LHEPart_pdgId__delta",
                    "nLHEPart",
                    "nLHEPart__delta",
                },
            )
            self.assertEqual(declared - plain, set())
        self.assertEqual(
            self.branches(self.plain, "Events"), self.branches(self.declared, "Events")
        )

    def test_readers_see_the_same_values(self):
        # Every row a reader keeps (valid in that tree), including the events selected
        # only by a shift, reads exactly what it read before.
        for tree_name in ["Events", "Events__JES__Up", "Events__JES__Down"]:
            plain = read_tree(self.plain, tree_name, self.COLUMNS)
            declared = read_tree(self.declared, tree_name, self.COLUMNS)
            self.assertTrue(np.array_equal(plain["valid"], declared["valid"]))
            valid = plain["valid"].astype(bool)
            for column in self.COLUMNS:
                with self.subTest(tree=tree_name, column=column):
                    self.assertTrue(same(plain[column][valid], declared[column][valid]))

    def test_placeholder_rows_get_the_declared_columns(self):
        central = read_tree(self.declared, "Events", self.COLUMNS)
        up = read_tree(self.plain, "Events__JES__Up", self.COLUMNS)
        down = read_tree(self.plain, "Events__JES__Down", self.COLUMNS)
        ids = list(central["FullEventId"])
        for event, source in [(1, up), (2, down), (3, up)]:
            row = ids.index(1000 + event)
            self.assertFalse(central["valid"][row])
            for column in ["weight_gen", "LHE_Vpt", "LHE_NpNLO", "LHEPart_pt"]:
                with self.subTest(event=event, column=column):
                    self.assertTrue(same(central[column][[row]], source[column][[row]]))
                    self.assertNotEqual(central["LHE_Vpt"][row], 0)
            # reco columns keep the placeholder default
            self.assertEqual(central["met_pt"][row], 0)
            self.assertEqual(len(central["centralJet_pt"][row]), 0)
        self.assertNotIn(1004, ids)

    def test_a_changing_declared_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "LHE_Vpt.*JES/Up"):
                run_fuse(tmp, GEN_PATTERNS, gen_offsets={("JES", "Up"): (5, 999.0)})

    def test_disagreeing_variations_on_a_placeholder_row_raise(self):
        # event 3 is a placeholder: Up fills it, Down must agree
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "LHE_Vpt.*JES/Down"):
                run_fuse(tmp, GEN_PATTERNS, gen_offsets={("JES", "Down"): (3, 999.0)})

    def test_a_partly_declared_array_collection_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "LHEPart.*only in part"):
                run_fuse(tmp, ["^LHEPart_pt$"])

    def test_a_reco_column_declared_by_mistake_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "met_pt.*JES/Up"):
                run_fuse(tmp, ["^met_pt$"])

    def test_reserved_columns_cannot_be_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "Reserved"):
                run_fuse(tmp, ["^valid$"])

    def test_no_selected_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = {variation: [] for variation in SELECTED}
            path = run_fuse(tmp, GEN_PATTERNS, selected=empty)
            branches = self.branches(path, "Events__JES__Up")
            self.assertNotIn("LHE_Vpt__delta", branches)
            self.assertIn("met_pt__delta", branches)

    def test_central_only_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_like = {("Central", "Central"): SELECTED[("Central", "Central")]}
            path = run_fuse(tmp, GEN_PATTERNS, selected=data_like)
            with uproot.open(path) as f:
                trees = [
                    k.split(";")[0] for k, c in f.classnames().items() if c == "TTree"
                ]
            self.assertEqual(trees, ["Events"])


class TestMergeRefusesMixedLists(unittest.TestCase):
    def test_different_lists_raise(self):
        from FLAF.AnaProd.MergeAnaTuples import checkShiftInvariantColumns

        same = {"ds": [{"shift_invariant_columns": GEN_PATTERNS}] * 2}
        checkShiftInvariantColumns(same)
        # anaTuples produced before the key existed have no entry: an empty list
        old = {"ds": [{"trees": []}, {"shift_invariant_columns": []}]}
        checkShiftInvariantColumns(old)
        for mixed in [
            {"ds": [{"shift_invariant_columns": GEN_PATTERNS}, {"trees": []}]},
            {
                "a": [{"shift_invariant_columns": GEN_PATTERNS}],
                "b": [{"shift_invariant_columns": GEN_PATTERNS[:1]}],
            },
        ]:
            with self.assertRaisesRegex(RuntimeError, "different"):
                checkShiftInvariantColumns(mixed)


if __name__ == "__main__":
    unittest.main()
