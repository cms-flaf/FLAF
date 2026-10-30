#!/usr/bin/env python3
"""include/HistFiller.h fills what one RDataFrame histogram per (key, weight, variable) fills.

HistProducerFromNTuple used to book a unit-bin HistoND per (selection key, weight, variable) on a
filtered RDataFrame; it now fills them all in one loop with HistFiller. On a synthetic dataset with
1D, 2D and 3D variables, scalar and vector bin columns (vector x vector, scalar x vector, empty
vectors), under- and overflows, overlapping keys and several weights, every cell, error and entry
count must be the same as RDataFrame's, bit for bit, single-threaded (in event order). Under implicit
MT every thread slot fills its own copy, summed by Write: the entries stay exact and the sums differ by
the order of the additions only. FillTree, the producer's driver, is checked against RDataFrame
histograms with the real binning, by the names it writes. Needs ROOT (and Corrections, for FillTree).

Run by hand in an analysis environment: ``python -m pytest -m needs_root test/test_hist_filler.py``.
"""

import math
import os
import sys
import unittest

import pytest

pytestmark = pytest.mark.needs_root

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import ROOT  # noqa: E402

ROOT.gROOT.SetBatch(True)
assert ROOT.gInterpreter.Declare(
    f'#include "{os.path.join(flaf_repo, "include", "HistFiller.h")}"'
)

N_EVENTS = 5000
KEYS = ["x % 3 == 0", "x % 2 == 0", "y > 2", "true"]  # overlapping
WEIGHTS = ["w0", "w1", "1.0"]
# name: (bin columns, bins per dimension)
VARIABLES = {
    "a": (["xb"], [6]),
    "ab": (["xb", "yb"], [6, 4]),
    "abc": (["xb", "yb", "zb"], [6, 4, 3]),
    "v": (["vb"], [5]),
    "vv": (["vb", "ub"], [5, 3]),
    "sv": (["xb", "vb"], [6, 5]),
}
SCALAR_COLS = ["xb", "yb", "zb"]
VECTOR_COLS = ["vb", "ub"]


def dataset(n_events=N_EVENTS):
    rdf = ROOT.RDataFrame(n_events)
    return (
        rdf.Define("x", "int((rdfentry_ * 2654435761ULL) % 1000)")
        .Define("y", "int((rdfentry_ * 40503ULL) % 7)")
        # bins from -2 to n+2: under- and overflows on every axis
        .Define("xb", "x % 10 - 2")
        .Define("yb", "y - 1")
        .Define("zb", "(x / 7) % 6 - 1")
        .Define("n_elem", "int(x % 4)")  # 0 to 3 elements, some events empty
        .Define(
            "vb",
            "ROOT::RVec<int> r(n_elem); for (int i = 0; i < n_elem; ++i) r[i] = (x + 3 * i) % 8 - 1; return r;",
        )
        .Define(
            "ub",
            "ROOT::RVec<short> r(n_elem); for (int i = 0; i < n_elem; ++i) r[i] = short((y + i) % 5 - 1); return r;",
        )
        .Define("w0", "0.37 + 0.01 * (x % 13) - 0.2 * (y == 3)")
        .Define("w1", "float(1.5 - 0.003 * x)")
        .Define("wi", "double(1 + x % 3)")  # integer weights: sums exact in any order
    )


def unit_model(nbins):
    args = []
    for n in nbins:
        args += [n, -0.5, n - 0.5]
    cls = {1: ROOT.RDF.TH1DModel, 2: ROOT.RDF.TH2DModel, 3: ROOT.RDF.TH3DModel}[
        len(nbins)
    ]
    return cls("", "", *args)


def cells(hist):
    return [
        (hist.GetBinContent(i), hist.GetBinError(i)) for i in range(hist.GetNcells())
    ]


def reference_hists(rdf, weights):
    """One RDataFrame histogram action per (key, weight, variable), as the producer used to book them."""
    reference = {}
    for k, key in enumerate(KEYS):
        filtered = rdf.Filter(key)
        for j, weight in enumerate(weights):
            node = filtered.Define("_w", weight)
            for var, (cols, nbins) in VARIABLES.items():
                book = getattr(node, f"Histo{len(nbins)}D")
                reference[(k, j, var)] = book(unit_model(nbins), *cols, "_w")
    return reference


def fill(rdf, weights, n_slots):
    """Fill every (key, weight, variable) with one HistFiller over n_slots copies; {(k, j, var): hist}."""
    filler = ROOT.analysis.HistFiller()
    for var, (cols, nbins) in VARIABLES.items():
        is_vector = [1 if c in VECTOR_COLS else 0 for c in cols]
        index = [
            VECTOR_COLS.index(c) if c in VECTOR_COLS else SCALAR_COLS.index(c)
            for c in cols
        ]
        filler.AddVariable(nbins, is_vector, index)
    filler.Allocate(len(KEYS), len(weights), n_slots)
    loop = (
        rdf.Define(
            "_s",
            "ROOT::RVec<int>{"
            + ", ".join(f"static_cast<int>({c})" for c in SCALAR_COLS)
            + "}",
        )
        .Define(
            "_v",
            "ROOT::RVec<ROOT::RVec<int>>{"
            + ", ".join(f"analysis::HistFillerBins({c})" for c in VECTOR_COLS)
            + "}",
        )
        .Define(
            "_w",
            "ROOT::RVec<double>{"
            + ", ".join(f"static_cast<double>({w})" for w in weights)
            + "}",
        )
        .Define(
            "_p",
            "ROOT::RVec<char>{"
            + ", ".join(f"analysis::HistFillerKey(({k}))" for k in KEYS)
            + "}",
        )
        .Define(
            "_f",
            f"analysis::HistFillerFill({ROOT.addressof(filler)}, rdfslot_, _p, _s, _v, _w)",
        )
    )
    loop.Sum("_f").GetValue()

    out = ROOT.TMemFile("hist_filler_test.root", "RECREATE")
    hists = ROOT.std.vector["TH1*"]()
    dirs = ROOT.std.vector["TDirectory*"]()
    names = ROOT.std.vector["std::string"]()
    keep = []
    # one directory per key, shared by the variables, as in a producer's file
    key_dirs = [out.mkdir(f"k{k}") for k in range(len(KEYS))]
    for var, (cols, nbins) in VARIABLES.items():
        hist = unit_model(nbins).GetHistogram()
        keep.append(hist)
        hists.push_back(hist)
        for key_dir in key_dirs:
            dirs.push_back(key_dir)
        for j in range(len(weights)):
            # names of the variable's own, as the producer writes them
            names.push_back(f"{var}_w{j}")
    filler.Write(hists, dirs, names)
    filler.Release()
    got = {}
    for k in range(len(KEYS)):
        for j in range(len(weights)):
            for var in VARIABLES:
                hist = out.Get(f"k{k}/{var}_w{j}")
                assert hist, (k, j, var)
                hist.SetDirectory(0)
                got[(k, j, var)] = hist
    out.Close()
    return got


class HistFillerMatchesRDataFrame(unittest.TestCase):
    def test_every_histogram_is_identical(self):
        rdf = dataset()
        reference = reference_hists(rdf, WEIGHTS)
        got = fill(rdf, WEIGHTS, 1)
        self.assertEqual(len(got), len(KEYS) * len(WEIGHTS) * len(VARIABLES))
        for key, ref in reference.items():
            self.assertEqual(cells(got[key]), cells(ref.GetValue()), key)
            self.assertEqual(got[key].GetEntries(), ref.GetEntries(), key)

    def test_vector_columns_of_different_sizes_are_refused(self):
        filler = ROOT.analysis.HistFiller()
        filler.AddVariable([4, 4], [1, 1], [0, 1])
        filler.Allocate(1, 1, 1)
        bins = ROOT.RVec["ROOT::RVec<int>"]()
        bins.push_back(ROOT.RVec["int"]([1, 2]))
        bins.push_back(ROOT.RVec["int"]([1]))
        with self.assertRaises(Exception):
            filler.Fill(
                0,
                ROOT.RVec["char"]([1]),
                ROOT.RVec["int"](),
                bins,
                ROOT.RVec["double"]([1.0]),
            )


def direct_fill(filler, slot, bin_value, weight):
    """One event of one key and one scalar 1D variable, on ``slot``."""
    return filler.Fill(
        slot,
        ROOT.RVec["char"]([1]),
        ROOT.RVec["int"]([bin_value]),
        ROOT.RVec["ROOT::RVec<int>"](),
        ROOT.RVec["double"]([weight]),
    )


class HistFillerSlots(unittest.TestCase):
    """Every slot fills its own copy, and Write sums all of them."""

    def write_one(self, filler, n_bins):
        out = ROOT.TMemFile("hist_filler_slots.root", "RECREATE")
        hist = unit_model([n_bins]).GetHistogram()
        hists, dirs, names = (
            ROOT.std.vector["TH1*"](),
            ROOT.std.vector["TDirectory*"](),
            ROOT.std.vector["std::string"](),
        )
        hists.push_back(hist)
        dirs.push_back(out)
        names.push_back("h")
        filler.Write(hists, dirs, names)
        got = out.Get("h")
        got.SetDirectory(0)
        out.Close()
        return got

    def test_write_sums_every_slot(self):
        filler = ROOT.analysis.HistFiller()
        filler.AddVariable([3], [0], [0])
        filler.Allocate(1, 1, 3)
        # slot s fills 2**s events of weight 1 into bin 1
        for slot in range(3):
            for _ in range(2**slot):
                self.assertEqual(direct_fill(filler, slot, 1, 1.0), 1)
        got = self.write_one(filler, 3)
        self.assertEqual(got.GetBinContent(2), 7.0)
        self.assertEqual(got.GetBinError(2), math.sqrt(7.0))
        self.assertEqual(got.GetEntries(), 7)

    def test_a_slot_out_of_range_is_refused(self):
        filler = ROOT.analysis.HistFiller()
        filler.AddVariable([3], [0], [0])
        filler.Allocate(1, 1, 2)
        with self.assertRaises(Exception):
            direct_fill(filler, 2, 1, 1.0)
        # also for an event in no key
        with self.assertRaises(Exception):
            filler.Fill(
                5,
                ROOT.RVec["char"]([0]),
                ROOT.RVec["int"]([1]),
                ROOT.RVec["ROOT::RVec<int>"](),
                ROOT.RVec["double"]([1.0]),
            )

    def test_no_slot_is_refused(self):
        filler = ROOT.analysis.HistFiller()
        filler.AddVariable([3], [0], [0])
        with self.assertRaises(Exception):
            filler.Allocate(1, 1, 0)


class HistFillerMultiThread(unittest.TestCase):
    """Under implicit MT: the same histograms as single-threaded, up to the order of the additions."""

    N_THREADS = 4

    def setUp(self):
        ROOT.EnableImplicitMT(self.N_THREADS)

    def tearDown(self):
        ROOT.DisableImplicitMT()

    def test_integer_weights_are_identical(self):
        rdf = dataset(200000)
        n_slots = rdf.GetNSlots()
        self.assertGreater(n_slots, 1)
        got = fill(rdf, ["wi", "1.0"], n_slots)
        ROOT.DisableImplicitMT()
        reference = reference_hists(dataset(200000), ["wi", "1.0"])
        for key, ref in reference.items():
            self.assertEqual(cells(got[key]), cells(ref.GetValue()), key)
            self.assertEqual(got[key].GetEntries(), ref.GetEntries(), key)

    def test_fractional_weights_differ_by_rounding_only(self):
        rdf = dataset(200000)
        got = fill(rdf, WEIGHTS, rdf.GetNSlots())
        ROOT.DisableImplicitMT()
        reference = reference_hists(dataset(200000), WEIGHTS)
        for key, ref in reference.items():
            ref = ref.GetValue()
            self.assertEqual(got[key].GetEntries(), ref.GetEntries(), key)
            scale = max(abs(c) for c, _ in cells(ref)) or 1.0
            for (c, e), (rc, re) in zip(cells(got[key]), cells(ref)):
                self.assertLessEqual(abs(c - rc), 1e-12 * scale, key)
                self.assertLessEqual(abs(e - re), 1e-12 * scale, key)


class KeysAndBinColumns(unittest.TestCase):
    """A key filter is a bool, and a bin column holds integers that fit an int, as RDataFrame's
    Filter and Histo would take them; anything else is refused when the loop is compiled.
    """

    def declares(self, code):
        return bool(ROOT.gInterpreter.Declare(code))

    def test_a_bool_key_is_accepted(self):
        self.assertTrue(
            self.declares(
                "char hf_test_key_ok(int bits) { return analysis::HistFillerKey((bits & 256) != 0); }"
            )
        )

    def test_a_key_that_is_not_a_bool_is_refused(self):
        # bits & 256 would keep its low 8 bits, 0, through a char cast
        self.assertFalse(
            self.declares(
                "char hf_test_key_int(int bits) { return analysis::HistFillerKey(bits & 256); }"
            )
        )
        self.assertFalse(
            self.declares(
                "char hf_test_key_float(float f) { return analysis::HistFillerKey(f); }"
            )
        )

    def test_narrow_vector_bin_columns_are_accepted(self):
        for i, t in enumerate(["int", "short", "unsigned char", "bool"]):
            self.assertTrue(
                self.declares(
                    f"ROOT::RVec<int> hf_test_bins_ok{i}(const ROOT::RVec<{t}>& b) "
                    "{ return analysis::HistFillerBins(b); }"
                ),
                t,
            )

    def test_wide_vector_bin_columns_are_refused(self):
        for i, t in enumerate(["Long64_t", "unsigned int", "float"]):
            self.assertFalse(
                self.declares(
                    f"ROOT::RVec<int> hf_test_bins_bad{i}(const ROOT::RVec<{t}>& b) "
                    "{ return analysis::HistFillerBins(b); }"
                ),
                t,
            )


def import_producer():
    try:
        import FLAF.Analysis.HistProducerFromNTuple as producer
    except ImportError as e:
        raise unittest.SkipTest(
            f"HistProducerFromNTuple needs an analysis environment: {e}"
        )
    return producer


class ProducerColumnTypes(unittest.TestCase):
    def test_bin_column_types(self):
        producer = import_producer()
        accepted = [
            "int",
            "Int_t",
            "short",
            "UShort_t",
            "bool",
            "Bool_t",
            "ROOT::VecOps::RVec<int>",
            "ROOT::RVec<short>",
            "vector<int>",
            "std::vector<UChar_t>",
            "const int&",
        ]
        refused = [
            "unsigned int",
            "UInt_t",
            "Long64_t",
            "ULong64_t",
            "unsigned long",
            "long",
            "float",
            "double",
            "ROOT::VecOps::RVec<Long64_t>",
            "vector<float>",
        ]
        for t in accepted:
            self.assertTrue(producer.IsBinColumnType(t), t)
        for t in refused:
            self.assertFalse(producer.IsBinColumnType(t), t)


# bin edges of the FillTree variables: real, non-unit binnings
EDGES = {
    "p": [0.0, 1.0, 2.5, 4.0, 7.0, 10.0],
    "q": [-1.0, 0.0, 0.5, 2.0],
}


class FillTreeMatchesRDataFrame(unittest.TestCase):
    """FillTree, as the producer runs it: real binnings, a further cut, a shape tree, a budget that
    takes several loops and splits a variable over its weights. Every histogram is read back by the name
    the producer gives it and compared with an RDataFrame histogram of the real binning filled with the
    raw values."""

    KEYS = [(("ch1", "SR"), "x % 3 == 0"), (("ch2", "SR"), "x % 2 == 0")]
    CUT = "y > 1"
    WEIGHTS = [
        ("Central", "Central", "weight_Central"),
        ("NormA", "Up", "weight_NormA_Up"),
        ("NormA", "Down", "weight_NormA_Down"),
    ]

    @classmethod
    def setUpClass(cls):
        cls.producer = import_producer()
        ROOT.gInterpreter.Declare("""
            int hf_test_bin(double v, const std::vector<double>& edges) {
                TAxis axis(edges.size() - 1, edges.data());
                return axis.FindFixBin(v) - 1;
            }
            """)

    def tree(self, shift):
        rdf = ROOT.RDataFrame(4000)
        edges_p = "{" + ",".join(map(str, EDGES["p"])) + "}"
        edges_q = "{" + ",".join(map(str, EDGES["q"])) + "}"
        return (
            rdf.Define("x", "int((rdfentry_ * 2654435761ULL) % 1000)")
            .Define("y", "int((rdfentry_ * 40503ULL) % 7)")
            .Define("p", f"0.011 * x - 0.5 + {shift}")  # under- and overflows
            .Define("q", "0.45 * y - 1.2")
            .Define("p_bin", f"hf_test_bin(p, std::vector<double>{edges_p})")
            .Define("q_bin", f"hf_test_bin(q, std::vector<double>{edges_q})")
            .Define("weight_Central", "0.5 + 0.001 * x")
            .Define("weight_NormA_Up", "weight_Central * 1.1")
            .Define("weight_NormA_Down", "weight_Central * (0.9 - 0.0001 * y)")
            .Define(self.CUT_NAME, self.CUT)
        )

    CUT_NAME = "cutY"

    def models(self):
        p_edges = ROOT.std.vector["double"](EDGES["p"])
        q_edges = ROOT.std.vector["double"](EDGES["q"])
        self._keep = (p_edges, q_edges)
        return {
            "p": (
                ROOT.RDF.TH1DModel("", "", len(EDGES["p"]) - 1, p_edges.data()),
                ["p_bin"],
            ),
            "pq": (
                ROOT.RDF.TH2DModel(
                    "",
                    "",
                    len(EDGES["p"]) - 1,
                    p_edges.data(),
                    len(EDGES["q"]) - 1,
                    q_edges.data(),
                ),
                ["p_bin", "q_bin"],
            ),
        }

    def run_fill(self, max_cells):
        keys = [
            (key + (self.CUT_NAME,), f"{flt} && {self.CUT_NAME}")
            for key, flt in self.KEYS
        ]
        var_models = self.models()
        out = ROOT.TMemFile("fill_tree_test.root", "RECREATE")
        var_files = {var: out.mkdir(var) for var in var_models}
        self.producer.FillTree(
            self.tree(0.0),
            "Events",
            self.WEIGHTS,
            keys,
            list(var_models),
            var_models,
            var_files,
            max_cells,
        )
        self.producer.FillTree(
            self.tree(0.37),
            "Events__ShapeB__Up",
            [("ShapeB", "Up", "weight_Central")],
            keys,
            list(var_models),
            var_models,
            var_files,
            max_cells,
        )
        return out, keys

    def reference(self, rdf, flt, weight, var):
        node = rdf.Filter(flt)
        model, _ = self.models()[var]
        if var == "p":
            return node.Histo1D(model, "p", weight)
        return node.Histo2D(model, "p", "q", weight)

    def check(self, max_cells):
        out, keys = self.run_fill(max_cells)
        trees = {"Events": self.tree(0.0), "Events__ShapeB__Up": self.tree(0.37)}
        expected = []
        for key, flt in keys:
            for var in ("p", "pq"):
                for unc, scale, weight in self.WEIGHTS:
                    name = var if unc == "Central" else f"{var}_{unc}_{scale}"
                    expected.append(("Events", key, flt, var, name, weight))
                expected.append(
                    (
                        "Events__ShapeB__Up",
                        key,
                        flt,
                        var,
                        f"{var}_ShapeB_Up",
                        "weight_Central",
                    )
                )
        refs = [
            (tree, key, var, name, self.reference(trees[tree], flt, weight, var))
            for tree, key, flt, var, name, weight in expected
        ]
        for tree, key, var, name, ref in refs:
            path = f"{var}/{'/'.join(key)}/{name}"
            got = out.Get(path)
            self.assertTrue(got, path)
            ref = ref.GetValue()
            self.assertEqual(cells(got), cells(ref), path)
            self.assertEqual(got.GetEntries(), ref.GetEntries(), path)
            for axis in ("GetXaxis", "GetYaxis"):
                a, b = getattr(got, axis)(), getattr(ref, axis)()
                self.assertEqual(
                    [a.GetBinLowEdge(i) for i in range(1, a.GetNbins() + 2)],
                    [b.GetBinLowEdge(i) for i in range(1, b.GetNbins() + 2)],
                    path,
                )
        # nothing else was written
        n_written = 0
        for var in ("p", "pq"):
            for key, _ in keys:
                n_written += len(out.Get(f"{var}/{'/'.join(key)}").GetListOfKeys())
        self.assertEqual(n_written, len(expected))
        out.Close()

    def test_one_loop(self):
        self.check(0)

    def test_several_loops_and_a_split_over_the_weights(self):
        # p: 7 cells, pq: 7 x 5 = 35 cells; 2 keys. 80 cells: p alone (42), then pq over its 3 weights
        # (70 each) in loops of one weight each.
        groups = self.producer.fill_groups({"p": 7, "pq": 35}, 2, 3, 80)
        self.assertEqual(
            groups,
            [(["p"], (0, 3)), (["pq"], (0, 1)), (["pq"], (1, 2)), (["pq"], (2, 3))],
        )
        self.check(80)


if __name__ == "__main__":
    unittest.main()
