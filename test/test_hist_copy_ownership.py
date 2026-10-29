#!/usr/bin/env python3
"""The histograms that TupleHelpers.copyFileContent (AnaProd's merge and fuse steps) and
skim_tree.copy_histograms read must be freed.

PyROOT does not own what TKey::ReadObj and TH1::Clone return, and SetDirectory(0) alone left
the histogram owned by nobody: every input histogram stayed allocated until the job ended.
The live histograms are counted with ROOT's object table, which counts only the objects made
after it is switched on, so the calls run in a child process that switches it on first; the
pytest process is left as it is.

Needs ROOT and uproot: run inside an analysis environment (source env.sh).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.needs_root

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)

CHILD = """
import gc, json, os, sys
sys.path.insert(0, sys.argv[1])
import ROOT
ROOT.TObject.SetObjectStat(True)
from FLAF.Common.TupleHelpers import copyFileContent
from FLAF.RunKit.skim_tree import copy_histograms

def live():
    gc.collect()  # copyFileContent's recursive closure is a reference cycle
    ROOT.gObjectTable.UpdateInstCount()
    return ROOT.TH1D.Class().GetInstanceCount()

# In a function, so that no name keeps the proxy of a histogram that Close() deleted: cppyy
# would hand that proxy out again for a histogram made at the same address.
def write_input(path, weight):
    f = ROOT.TFile(path, "RECREATE")
    h = ROOT.TH1D("h", "", 4, 0.0, 4.0)
    h.Fill(1.5, weight)
    f.Write()
    f.Close()

tmp = sys.argv[2]
inputs = [os.path.join(tmp, f"in_{i}.root") for i in range(3)]
for i, path in enumerate(inputs):
    write_input(path, i + 1.0)
counts = {"start": live()}
copyFileContent(inputs, os.path.join(tmp, "copied.root"), copyTrees=False, verbose=0)
counts["copyFileContent"] = live()
copy_histograms(
    inputs, {"h": inputs}, ["h"], os.path.join(tmp, "skimmed.root"),
    ROOT.RDF.RSnapshotOptions(),
)
counts["copy_histograms"] = live()
print(json.dumps(counts))
"""


class TestHistCopyOwnership(unittest.TestCase):
    def test_input_histograms_are_freed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", CHILD, flaf_parent, tmp],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            counts = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(counts["start"], 0)
            self.assertEqual(
                counts["copyFileContent"],
                0,
                "copyFileContent left input histograms owned by nobody",
            )
            self.assertEqual(
                counts["copy_histograms"],
                0,
                "copy_histograms left a histogram owned by nobody",
            )

            import ROOT

            for name in ("copied.root", "skimmed.root"):
                f = ROOT.TFile.Open(os.path.join(tmp, name), "READ")
                try:
                    self.assertEqual(f.Get("h").GetBinContent(2), 6.0, name)
                finally:
                    f.Close()


if __name__ == "__main__":
    unittest.main()
