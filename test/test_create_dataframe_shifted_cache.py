#!/usr/bin/env python3
"""CreateDataFrame must give a shifted tree the values of the shifted cache tree (cms-flaf/FLAF#402).

A reader of a shifted anaTuple tree (HistTupleProducer, AnalysisCacheProducer) first opens the
central tree with its caches, then each shifted tree with the central tree and central caches as
friends. The central tree object carries its caches as friends under their plain names; if that
object is friended to the shifted tree, "<cache>.<column>" resolves through it and the shifted tree
reads the central cache. The cache values differ per tree here, so the test sees which one is read.

Needs ROOT: run inside an analysis environment (source env.sh).
"""

import os
import sys
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.needs_root

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import numpy as np
import ROOT

from FLAF.Common.Utilities import CreateDataFrame

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine(f".include {flaf_repo}")
ROOT.gInterpreter.Declare('#include "include/Utilities.h"')

N_EVENTS = 10
SHIFTED = "Events__JES__Up"
# valid rows: event 1 selected only in the shift, event 2 only in the central selection
VALID = {"Events": [0, 2, 3, 4, 5, 6, 7, 8, 9], SHIFTED: [0, 1, 3, 4, 5, 6, 7, 8, 9]}
# cache values: different in every tree, so the tree they come from is identifiable
SCORE = {"Events": "100.f + rdfentry_", SHIFTED: "200.f + rdfentry_"}


def write_files(work_dir):
    ana = os.path.join(work_dir, "anaTuple.root")
    cache = os.path.join(work_dir, "cache.root")
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fMode = "UPDATE"
    for i, tree in enumerate(["Events", SHIFTED]):
        valid = " || ".join(f"rdfentry_ == {e}" for e in VALID[tree])
        df = ROOT.RDataFrame(N_EVENTS)
        df = df.Define("FullEventId", "static_cast<ULong64_t>(1000 + rdfentry_)")
        df = df.Define("valid", f"static_cast<bool>({valid})")
        if tree == "Events":
            df = df.Define("x", "static_cast<float>(rdfentry_)")
            columns = ["FullEventId", "valid", "x"]
        else:
            df = df.Define("x__delta", "0.5f")
            columns = ["FullEventId", "valid", "x__delta"]
        if i == 0:
            df.Snapshot(tree, ana, columns)
        else:
            df.Snapshot(tree, ana, columns, opts)
        cdf = ROOT.RDataFrame(N_EVENTS)
        cdf = cdf.Define("FullEventId", "static_cast<ULong64_t>(1000 + rdfentry_)")
        cdf = cdf.Define("score", SCORE[tree])
        if i == 0:
            cdf.Snapshot(tree, cache, ["FullEventId", "score"])
        else:
            cdf.Snapshot(tree, cache, ["FullEventId", "score"], opts)
    return ana, cache


class TestCreateDataFrameShiftedCache(unittest.TestCase):
    def test_shifted_tree_reads_its_own_cache(self):
        with tempfile.TemporaryDirectory() as work_dir:
            ana, cache = write_files(work_dir)
            files = {}
            caches = {"DNN": cache}
            # as HistTupleProducer: the central call first, its tree and caches passed on
            _, df_c, central_tree, central_caches = CreateDataFrame(
                treeName="Events", fileName=ana, caches=caches, files=files
            )
            _, df_s, _, _ = CreateDataFrame(
                treeName=SHIFTED,
                fileName=ana,
                caches=caches,
                files=files,
                centralTree=central_tree,
                centralCaches=central_caches,
            )
            c = df_c.AsNumpy(["FullEventId", "score"])
            s = df_s.AsNumpy(["FullEventId", "score", "x"])
            ids_s = s["FullEventId"] - 1000
            # every selected row of the shifted tree carries the shifted cache value,
            # including the event selected only in the shift
            np.testing.assert_array_equal(s["score"], 200.0 + ids_s)
            self.assertIn(1, list(ids_s))
            # the central tree keeps reading its own cache
            np.testing.assert_array_equal(c["score"], 100.0 + (c["FullEventId"] - 1000))
            # delta columns are still rebuilt from the central tree
            expected_x = np.where(ids_s == 1, 0.5, ids_s + 0.5)
            np.testing.assert_allclose(s["x"], expected_x)
            # the central cache stays reachable under <cache>__Central
            central = df_s.Define("central_score", "DNN__Central.score").AsNumpy(
                ["central_score"]
            )
            np.testing.assert_array_equal(central["central_score"], 100.0 + ids_s)


if __name__ == "__main__":
    unittest.main()
