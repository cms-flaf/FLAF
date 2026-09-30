#!/usr/bin/env python3
"""TupleHelpers.writeTree writes a TTree, as `file[name] = data` did up to uproot 5.6.

Since uproot 5.7 that assignment writes an RNTuple, which FLAF's readers (TChain, friends,
RDataFrame over trees) do not accept.

Needs uproot, awkward and numpy: run inside an analysis environment (source env.sh).
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
import uproot

from FLAF.Common.TupleHelpers import defineColumnGrouping, writeTree


def write_and_read(data, name="Events"):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.root")
        with uproot.recreate(path) as f:
            writeTree(f, name, data)
        with uproot.open(path) as f:
            classes = {k.split(";")[0]: c for k, c in f.classnames().items()}
            tree = f[name]
            return (
                classes,
                tree.arrays(library="ak"),
                {b.name: b.typename for b in tree.branches},
            )


class TestWriteTree(unittest.TestCase):
    def test_grouped_anatuple_columns(self):
        # what FuseAnaTuples writes: scalars and zipped collections
        columns = {
            "FullEventId": np.arange(4, dtype=np.uint64),
            "valid": np.array([True, False, True, True]),
            "met_pt": ak.Array([1.5, 0.0, 2.5, 3.5]),
            "centralJet_pt": ak.Array([[30.0, 20.0], [], [40.0], [50.0, 45.0, 35.0]]),
            "centralJet_btag": ak.Array([[1, 0], [], [1], [0, 0, 1]]),
        }
        grouped = defineColumnGrouping(columns, list(columns), verbose=0)
        classes, arrays, types = write_and_read(grouped)
        self.assertEqual(classes, {"Events": "TTree"})
        self.assertEqual(types["centralJet_pt"], "double[]")
        self.assertEqual(
            ak.to_list(arrays["centralJet_pt"]), ak.to_list(columns["centralJet_pt"])
        )
        self.assertEqual(ak.to_list(arrays["ncentralJet"]), [2, 0, 1, 3])
        self.assertEqual(ak.to_list(arrays["valid"]), [True, False, True, True])
        self.assertEqual(types["FullEventId"], "uint64_t")

    def test_record_array(self):
        # what AnalysisCacheProducer writes: the producer's record array
        record = ak.zip(
            {
                "FullEventId": np.arange(3, dtype=np.int64),
                "dnn_score": np.array([0.1, 0.5, 0.9], dtype=np.float32),
            }
        )
        classes, arrays, types = write_and_read(record)
        self.assertEqual(classes, {"Events": "TTree"})
        self.assertEqual(sorted(types), ["FullEventId", "dnn_score"])
        self.assertEqual(types["dnn_score"], "float")
        self.assertEqual(ak.to_list(arrays["FullEventId"]), [0, 1, 2])

    def test_empty_and_fixed_size_numpy(self):
        data = {
            "FullEventId": np.array([], dtype=np.int64),
            "x": np.array([], dtype=np.float32),
        }
        classes, arrays, types = write_and_read(data)
        self.assertEqual(classes, {"Events": "TTree"})
        self.assertEqual(len(arrays), 0)
        classes, arrays, types = write_and_read({"v": np.ones((2, 3))})
        self.assertEqual(types["v"], "double[3]")

    def test_tree_in_a_directory(self):
        classes, arrays, _ = write_and_read({"x": np.arange(2)}, name="dir/tree")
        self.assertEqual(classes, {"dir": "TDirectory", "dir/tree": "TTree"})


if __name__ == "__main__":
    unittest.main()
