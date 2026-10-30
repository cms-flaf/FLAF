#!/usr/bin/env python3
"""include/HistFiller.h fills what one RDataFrame histogram per (key, weight, variable) fills.

HistProducerFromNTuple used to book a unit-bin HistoND per (selection key, weight, variable) on a
filtered RDataFrame; it now fills them all in one loop with HistFiller. On a synthetic dataset with
1D, 2D and 3D variables, scalar and vector bin columns (vector x vector, scalar x vector, empty
vectors), under- and overflows, overlapping keys and several weights, every cell, error and entry
count must be the same as RDataFrame's, bit for bit (both single-threaded, in event order). Needs ROOT.
"""

import os
import sys
import unittest

import pytest

pytestmark = pytest.mark.needs_root

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ROOT  # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.gInterpreter.Declare(
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


def dataset():
    rdf = ROOT.RDataFrame(N_EVENTS)
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


class HistFillerMatchesRDataFrame(unittest.TestCase):
    def test_every_histogram_is_identical(self):
        rdf = dataset()
        # reference: one histogram action per (key, weight, variable)
        reference = {}
        for k, key in enumerate(KEYS):
            filtered = rdf.Filter(key)
            for j, weight in enumerate(WEIGHTS):
                node = filtered.Define("_w", weight)
                for var, (cols, nbins) in VARIABLES.items():
                    book = getattr(node, f"Histo{len(nbins)}D")
                    reference[(k, j, var)] = book(unit_model(nbins), *cols, "_w")

        scalar_cols = ["xb", "yb", "zb"]
        vector_cols = ["vb", "ub"]
        filler = ROOT.analysis.HistFiller()
        for var, (cols, nbins) in VARIABLES.items():
            is_vector = [1 if c in vector_cols else 0 for c in cols]
            index = [
                vector_cols.index(c) if c in vector_cols else scalar_cols.index(c)
                for c in cols
            ]
            filler.AddVariable(nbins, is_vector, index)
        filler.Allocate(len(KEYS), len(WEIGHTS))
        loop = (
            rdf.Define(
                "_s",
                "ROOT::RVec<int>{"
                + ", ".join(f"static_cast<int>({c})" for c in scalar_cols)
                + "}",
            )
            .Define(
                "_v",
                "ROOT::RVec<ROOT::RVec<int>>{"
                + ", ".join(f"analysis::HistFillerBins({c})" for c in vector_cols)
                + "}",
            )
            .Define(
                "_w",
                "ROOT::RVec<double>{"
                + ", ".join(f"static_cast<double>({w})" for w in WEIGHTS)
                + "}",
            )
            .Define(
                "_p",
                "ROOT::RVec<char>{"
                + ", ".join(f"static_cast<char>(({k}))" for k in KEYS)
                + "}",
            )
            .Define(
                "_f",
                f"analysis::HistFillerFill({ROOT.addressof(filler)}, _p, _s, _v, _w)",
            )
        )
        loop.Sum("_f").GetValue()

        out = ROOT.TMemFile("hist_filler_test.root", "RECREATE")
        hists = ROOT.std.vector["TH1*"]()
        dirs = ROOT.std.vector["TDirectory*"]()
        names = ROOT.std.vector["std::string"]()
        keep = []
        for var, (cols, nbins) in VARIABLES.items():
            hist = unit_model(nbins).GetHistogram()
            keep.append(hist)
            hists.push_back(hist)
            for k in range(len(KEYS)):
                dirs.push_back(out.mkdir(f"{var}_k{k}"))
            for j in range(len(WEIGHTS)):
                names.push_back(f"w{j}")
        filler.Write(hists, dirs, names)

        n_checked = 0
        for (k, j, var), ref in reference.items():
            got = out.Get(f"{var}_k{k}/w{j}")
            self.assertTrue(got, (k, j, var))
            self.assertEqual(cells(got), cells(ref.GetValue()), (k, j, var))
            self.assertEqual(got.GetEntries(), ref.GetEntries(), (k, j, var))
            n_checked += 1
        self.assertEqual(n_checked, len(KEYS) * len(WEIGHTS) * len(VARIABLES))
        filler.Release()

    def test_vector_columns_of_different_sizes_are_refused(self):
        filler = ROOT.analysis.HistFiller()
        filler.AddVariable([4, 4], [1, 1], [0, 1])
        filler.Allocate(1, 1)
        bins = ROOT.RVec["ROOT::RVec<int>"]()
        bins.push_back(ROOT.RVec["int"]([1, 2]))
        bins.push_back(ROOT.RVec["int"]([1]))
        with self.assertRaises(Exception):
            filler.Fill(
                ROOT.RVec["char"]([1]),
                ROOT.RVec["int"](),
                bins,
                ROOT.RVec["double"]([1.0]),
            )


if __name__ == "__main__":
    unittest.main()
