#!/usr/bin/env python3
"""The signal trees of the histogram files: concatenated, never silently incomplete.

HistProducerFromNTuple writes ``SignalTrees/<variation>`` next to the histograms of a signal
chunk; HistMergerFromHists concatenates them per process into ``SignalTrees/<process>/``, and
SumEraFiles concatenates the eras with the rule of the histograms (an era without a variation
enters with its central rows). The rows must be every row of the inputs, in order, with the
branches of the first tree: a tree with other branches, or an era with the histograms of a
process but not its trees, raises; so does a process with trees in some of its chunks only
(signal_tree_inputs, the merger's check).
"""

import os
import re
import sys
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.needs_root

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)
os.environ.setdefault("ANALYSIS_PATH", flaf_parent)

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

import FLAF.Common.HistHelper as HistHelper

SCALES = ["Up", "Down"]
SEL = "sel__eTau__OS_Iso__res2b"
# the columns of a varied tree; a central one has the norm-variation weights in addition
COLUMNS = {
    "x": "static_cast<float>(0.5 * event)",
    "x_bin": "static_cast<int>(event % 3)",
    "weight_Central": "1.0 + 0.01 * event",
    SEL: "event % 2 == 0",
}
NORM_WEIGHTS = {"weight_lumi_Up": "1.1", "weight_lumi_Down": "0.9"}
HIST_DIRS = ["eTau/OS_Iso/res2b", "muTau/OS_Iso/res2b"]
N_BINS = 3


def write_tree(path, name, events, columns):
    """Snapshot one row per ``events`` entry into ``path`` (created or updated) as ``name``."""
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fMode = "UPDATE"
    first = events[0] if events else 0
    df = ROOT.RDataFrame(len(events)).Define(
        "event", f"static_cast<ULong64_t>(rdfentry_ + {first})"
    )
    for column, expression in columns.items():
        df = df.Define(column, expression)
    df.Snapshot(name, path, ["event"] + list(columns), opts)


def read_rows(path, name):
    return {
        column: np.asarray(values)
        for column, values in ROOT.RDataFrame(name, path).AsNumpy().items()
    }


def branch_names(path, name):
    root_file = ROOT.TFile.Open(path, "READ")
    names = {branch.GetName() for branch in root_file.Get(name).GetListOfBranches()}
    root_file.Close()
    return names


def hist_values(era, dir_index, name):
    """The bin contents of histogram ``name`` of an era: distinct for every input."""
    offset = 100000 * era + 10000 * dir_index + 10 * sum(map(ord, name))
    return [offset + b for b in range(N_BINS)]


def write_hists(path, era, names):
    root_file = ROOT.TFile(path, "UPDATE")
    for dir_index, dir_path in enumerate(HIST_DIRS):
        directory = HistHelper.mkdir_recursive(root_file, dir_path)
        for name in names:
            hist = ROOT.TH1D(name, name, N_BINS, 0, N_BINS)
            hist.SetDirectory(0)
            for b, value in enumerate(hist_values(era, dir_index, name), start=1):
                hist.SetBinContent(b, value)
            directory.WriteTObject(hist, name)
    root_file.Close()


def era_events(era, variation, n):
    """Event numbers of the rows of a tree: distinct for every era and variation."""
    first = 10000 * era + 100 * len(variation)
    return list(range(first, first + n))


def write_era(path, era, hist_names, tree_variations, n_rows=4, process="sig"):
    """A merged file of one era: histograms, and ``SignalTrees/<process>/<variation>``."""
    write_hists(path, era, hist_names)
    for variation in tree_variations:
        columns = dict(COLUMNS)
        if variation == "Central":
            columns.update(NORM_WEIGHTS)
        write_tree(
            path,
            f"{HistHelper.SIGNAL_TREES_DIR}/{process}/{variation}",
            era_events(era, variation, n_rows),
            columns,
        )


def read_hist(path, name):
    root_file = ROOT.TFile.Open(path, "READ")
    hist = root_file.Get(name)
    values = [hist.GetBinContent(b) for b in range(1, N_BINS + 1)]
    root_file.Close()
    return values


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)


class ConcatTrees(TempDirTest):
    def concat(self, inputs, **kwargs):
        """concat_trees over the ``Events`` tree of each of ``inputs`` into out.root."""
        in_files = [ROOT.TFile.Open(path, "READ") for path in inputs]
        out_file = ROOT.TFile(self.path("out.root"), "RECREATE")
        try:
            HistHelper.concat_trees(
                out_file, "Events", [f.Get("Events") for f in in_files], **kwargs
            )
        finally:
            out_file.Close()
            for in_file in in_files:
                in_file.Close()
        return read_rows(self.path("out.root"), "Events")

    def chunk(self, name, events, columns=COLUMNS):
        path = self.path(name)
        write_tree(path, "Events", events, columns)
        return path

    def assert_rows(self, rows, chunks):
        expected = [read_rows(path, "Events") for path in chunks]
        self.assertEqual(set(rows), set(expected[0]))
        for column, values in rows.items():
            np.testing.assert_array_equal(
                values, np.concatenate([e[column] for e in expected]), err_msg=column
            )

    def test_two_chunks(self):
        chunks = [
            self.chunk("a.root", list(range(5))),
            self.chunk("b.root", list(range(100, 107))),
        ]
        rows = self.concat(chunks)
        self.assertEqual(len(rows["event"]), 12)
        self.assert_rows(rows, chunks)

    def test_empty_first_tree(self):
        chunks = [
            self.chunk("a.root", []),
            self.chunk("b.root", list(range(100, 107))),
        ]
        rows = self.concat(chunks)
        self.assertEqual(len(rows["event"]), 7)
        self.assert_rows(rows, chunks)

    def test_later_tree_missing_a_branch(self):
        columns = {k: v for k, v in COLUMNS.items() if k != "x"}
        chunks = [
            self.chunk("a.root", list(range(5))),
            self.chunk("b.root", list(range(100, 107)), columns),
        ]
        for kwargs in ({}, {"extra_branches_from": 1}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(RuntimeError, r"\['x'\]"):
                    self.concat(chunks, **kwargs)

    def test_later_tree_with_an_extra_branch(self):
        chunks = [
            self.chunk("a.root", list(range(5))),
            self.chunk("b.root", list(range(100, 107)), {**COLUMNS, **NORM_WEIGHTS}),
        ]
        with self.assertRaisesRegex(RuntimeError, "weight_lumi_Down"):
            self.concat(chunks)
        # from index 2 on only: the second tree is index 1
        with self.assertRaisesRegex(RuntimeError, "weight_lumi_Down"):
            self.concat(chunks, extra_branches_from=2)
        rows = self.concat(chunks, extra_branches_from=1)
        self.assertEqual(set(rows), {"event"} | set(COLUMNS))
        expected = [read_rows(path, "Events") for path in chunks]
        for column in rows:
            np.testing.assert_array_equal(
                rows[column],
                np.concatenate([e[column] for e in expected]),
                err_msg=column,
            )

    def test_first_tree_with_an_extra_branch(self):
        chunks = [
            self.chunk("a.root", list(range(5)), {**COLUMNS, **NORM_WEIGHTS}),
            self.chunk("b.root", list(range(100, 107))),
        ]
        with self.assertRaisesRegex(RuntimeError, "weight_lumi_Down"):
            self.concat(chunks, extra_branches_from=1)

    def test_branch_of_another_type(self):
        # CopyEntries only warns and copies nothing: the entry check stops it
        columns = {**COLUMNS, "x": "0.5 * event"}  # double, not float
        chunks = [
            self.chunk("a.root", list(range(5))),
            self.chunk("b.root", list(range(100, 107)), columns),
        ]
        with self.assertRaisesRegex(RuntimeError, "5 of 12 entries copied"):
            self.concat(chunks)

    def test_no_or_missing_tree(self):
        out_file = ROOT.TFile(self.path("out.root"), "RECREATE")
        try:
            with self.assertRaisesRegex(RuntimeError, "0 input tree"):
                HistHelper.concat_trees(out_file, "Events", [])
            in_file = ROOT.TFile.Open(self.chunk("a.root", list(range(5))), "READ")
            trees = [in_file.Get("Events"), in_file.Get("NotThere")]
            with self.assertRaisesRegex(RuntimeError, "2 input tree.*1 missing"):
                HistHelper.concat_trees(out_file, "Events", trees)
            in_file.Close()
        finally:
            out_file.Close()


class SumEraFilesSignalTrees(TempDirTest):
    CENTRAL_HISTS = ["sig", "bkg"]
    SHAPE = ["CMS_scale_j_Up", "CMS_scale_j_Down"]

    def full_era(self, name, era):
        path = self.path(name)
        write_era(
            path,
            era,
            self.CENTRAL_HISTS + [f"sig_{v}" for v in self.SHAPE],
            ["Central"] + self.SHAPE,
        )
        return path

    def assert_tree(self, out, variation, parts):
        """``SignalTrees/sig/<variation>`` of ``out`` is the concatenation of ``parts``,
        (path, variation) pairs, with the branches of the first."""
        name = f"{HistHelper.SIGNAL_TREES_DIR}/sig/{variation}"
        rows = read_rows(out, name)
        first_path, first_variation = parts[0]
        branches = branch_names(
            first_path, f"{HistHelper.SIGNAL_TREES_DIR}/sig/{first_variation}"
        )
        self.assertEqual(branch_names(out, name), branches)
        expected = [
            read_rows(path, f"{HistHelper.SIGNAL_TREES_DIR}/sig/{v}")
            for path, v in parts
        ]
        for column in branches:
            np.testing.assert_array_equal(
                rows[column],
                np.concatenate([e[column] for e in expected]),
                err_msg=f"{variation} {column}",
            )

    def assert_hist(self, out, name, parts):
        """Every ``<dir>/<name>`` of ``out`` is the sum of ``parts``, (era, name) pairs."""
        for dir_index, dir_path in enumerate(HIST_DIRS):
            expected = np.sum(
                [hist_values(era, dir_index, part) for era, part in parts], axis=0
            )
            self.assertEqual(
                read_hist(out, f"{dir_path}/{name}"), list(expected), msg=name
            )

    def test_two_eras(self):
        era_a = self.full_era("a.root", 1)
        era_b = self.full_era("b.root", 2)
        out = self.path("sum.root")
        HistHelper.SumEraFiles([era_a, era_b], out, SCALES)
        for variation in ["Central"] + self.SHAPE:
            self.assert_tree(out, variation, [(era_a, variation), (era_b, variation)])
        self.assertEqual(
            len(read_rows(out, f"{HistHelper.SIGNAL_TREES_DIR}/sig/Central")["event"]),
            8,
        )
        for name in self.CENTRAL_HISTS + [f"sig_{v}" for v in self.SHAPE]:
            self.assert_hist(out, name, [(1, name), (2, name)])

    def test_era_without_a_variation(self):
        # Run3_2023 has no CMS_scale_j_2022_Up: its central rows and histograms stand in,
        # after the rows of the era that has it, although it is listed first.
        decorrelated = "CMS_scale_j_2022_Up"
        era_a = self.full_era("a.root", 1)
        write_era(era_a, 1, [f"sig_{decorrelated}"], [decorrelated])
        era_b = self.full_era("b.root", 2)
        out = self.path("sum.root")
        HistHelper.SumEraFiles([era_b, era_a], out, SCALES)
        self.assert_tree(out, decorrelated, [(era_a, decorrelated), (era_b, "Central")])
        self.assertNotIn(
            "weight_lumi_Up",
            branch_names(out, f"{HistHelper.SIGNAL_TREES_DIR}/sig/{decorrelated}"),
        )
        self.assert_tree(out, "Central", [(era_b, "Central"), (era_a, "Central")])
        self.assert_hist(
            out, f"sig_{decorrelated}", [(2, "sig"), (1, f"sig_{decorrelated}")]
        )
        self.assert_hist(out, "sig", [(2, "sig"), (1, "sig")])

    def test_era_without_the_process(self):
        era_a = self.full_era("a.root", 1)
        era_b = self.path("b.root")
        write_era(era_b, 2, ["bkg"], [])
        out = self.path("sum.root")
        HistHelper.SumEraFiles([era_a, era_b], out, SCALES)
        self.assert_tree(out, "Central", [(era_a, "Central")])
        self.assert_hist(out, "sig", [(1, "sig")])
        self.assert_hist(out, "bkg", [(1, "bkg"), (2, "bkg")])

    def test_era_with_histograms_but_no_trees(self):
        era_a = self.full_era("a.root", 1)
        names = self.CENTRAL_HISTS + [f"sig_{v}" for v in self.SHAPE]
        # no SignalTrees at all, and the trees of another process only
        for trees_of in [None, "other"]:
            with self.subTest(trees_of=trees_of):
                era_b = self.path(f"b_{trees_of}.root")
                write_hists(era_b, 2, names)
                if trees_of:
                    write_tree(
                        era_b,
                        f"{HistHelper.SIGNAL_TREES_DIR}/{trees_of}/Central",
                        era_events(2, "Central", 4),
                        {**COLUMNS, **NORM_WEIGHTS},
                    )
                out = self.path(f"sum_{trees_of}.root")
                for in_paths in ([era_a, era_b], [era_b, era_a]):
                    with self.assertRaisesRegex(
                        RuntimeError, re.escape(f"{era_b} has histograms of sig but no")
                    ):
                        HistHelper.SumEraFiles(in_paths, out, SCALES)
                    # raised before the output file is opened
                    self.assertFalse(os.path.exists(out))


class SignalTreeInputs(unittest.TestCase):
    def test_all_none_and_mixed(self):
        chunks = {
            "sig": [("sig_a", "a.root", True), ("sig_b", "b.root", True)],
            "bkg": [("bkg_a", "c.root", False), ("bkg_b", "d.root", False)],
        }
        self.assertEqual(
            HistHelper.signal_tree_inputs(chunks), {"sig": ["a.root", "b.root"]}
        )
        for order in [[True, False], [False, True]]:
            mixed = {
                "sig": [(f"sig_{i}", f"{i}.root", has) for i, has in enumerate(order)]
            }
            with self.assertRaisesRegex(RuntimeError, "SignalTrees missing in"):
                HistHelper.signal_tree_inputs(mixed)


class GetAllItemsRecursive(TempDirTest):
    def test_signal_trees_are_not_read(self):
        # the layout of a producer output: histograms, and SignalTrees/<variation>
        path = self.path("chunk.root")
        names = ["sig", "sig_CMS_scale_j_Up"]
        write_hists(path, 1, names)
        write_tree(
            path,
            f"{HistHelper.SIGNAL_TREES_DIR}/Central",
            era_events(1, "Central", 4),
            COLUMNS,
        )
        root_file = ROOT.TFile.Open(path, "READ")
        items = HistHelper.get_all_items_recursive(root_file)
        self.assertEqual(
            {key: sorted(hists) for key, hists in items.items()},
            {tuple(d.split("/")): sorted(names) for d in HIST_DIRS},
        )
        for (key, hists), dir_index in zip(sorted(items.items()), [0, 1]):
            for name, hist in hists.items():
                self.assertEqual(
                    [hist.GetBinContent(b) for b in range(1, N_BINS + 1)],
                    hist_values(1, dir_index, name),
                )
        # ReadObj would have attached the tree to its directory
        trees_dir = root_file.GetDirectory(HistHelper.SIGNAL_TREES_DIR)
        self.assertFalse(trees_dir.GetList().FindObject("Central"))
        root_file.Close()


if __name__ == "__main__":
    unittest.main()
