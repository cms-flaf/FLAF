#!/usr/bin/env python3
"""SaveHist must not hand out a histogram that nobody owns.

``SaveHist`` used to write ``model.GetHistogram().Clone()``. With the output file open,
``TH1::Clone`` registers the copy in that file, so PyROOT cedes ownership; the following
``SetDirectory(0)`` then removed the file's ownership as well and the object was left
owned by nobody -- one fully-allocated histogram leaked per saved histogram, tens of GB
over a production job. The regression is invisible in bin contents, so it is pinned here
by the ownership of the object that actually reaches ``WriteTObject``.
"""

import os
import sys
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)
os.environ.setdefault("ANALYSIS_PATH", flaf_parent)

import ROOT

ROOT.gROOT.SetBatch(True)

from FLAF.Analysis.HistProducerFromNTuple import SaveHist


class _RecordingDir:
    """Delegates to a real TDirectory while keeping the objects written through it."""

    def __init__(self, real_dir):
        self._real_dir = real_dir
        self.written = []

    def WriteTObject(self, obj, name, option=""):
        self.written.append(obj)
        return self._real_dir.WriteTObject(obj, name, option)


class TestSaveHistOwnership(unittest.TestCase):
    KEY = ("e", "OS_Iso", "boosted")

    def setUp(self):
        self.path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"save_hist_{os.getpid()}.root"
        )
        self.out_file = ROOT.TFile(self.path, "RECREATE")
        # HistHelper.GetModel builds unnamed models, which is what makes an
        # auto-registered histogram a *stray unnamed* object in the output file.
        self.model = ROOT.RDF.TH2DModel("", "", 4, 0.0, 1.0, 3, 0.0, 300.0)
        self.unit_hist = ROOT.TH2D("unit", "unit", 4, 0.0, 1.0, 3, 0.0, 300.0)
        self.unit_hist.SetDirectory(0)
        for i in range(self.unit_hist.GetNcells()):
            self.unit_hist.SetBinContent(i, 0.5 * i)
            self.unit_hist.SetBinError(i, 0.25 * i)
        self.unit_hist.SetEntries(17)

    def tearDown(self):
        if self.out_file and self.out_file.IsOpen():
            self.out_file.Close()
        if os.path.exists(self.path):
            os.remove(self.path)

    def _save(self, hist_name="var"):
        """Run the real SaveHist and return the object it wrote.

        ``rdf`` is only read when ``verbose`` is set, which it is not here.
        """
        recorder = None
        real_mkdir = sys.modules["FLAF.Common.Utilities"].mkdir

        def recording_mkdir(file, path):
            nonlocal recorder
            recorder = _RecordingDir(real_mkdir(file, path))
            return recorder

        with mock.patch("FLAF.Common.Utilities.mkdir", side_effect=recording_mkdir):
            SaveHist(
                self.KEY,
                self.out_file,
                [(self.model, self.unit_hist, None)],
                hist_name,
                "Central",
                "Central",
            )
        self.assertEqual(len(recorder.written), 1)
        return recorder.written[0]

    def _assert_freed_after_use(self, written, where=""):
        """Python must own it, and it must belong to no TDirectory.

        Both halves are needed. A bare ``Clone()`` leaves ROOT as the owner via the open
        output file, which frees the histogram only when that file is closed and so still
        accumulates one per save; a ``Clone()`` detached with ``SetDirectory(0)`` leaves it
        owned by nobody at all.
        """
        self.assertTrue(
            written.__python_owns__,
            f"{where}SaveHist wrote a histogram that Python does not own, "
            "so it outlives the call",
        )
        self.assertFalse(
            written.GetDirectory(),
            f"{where}SaveHist wrote a histogram registered in a TDirectory, "
            "so it is retained until that directory is closed",
        )

    def test_written_histogram_is_freed_after_the_call(self):
        self._assert_freed_after_use(self._save())

    def test_repeated_saves_stay_freeable(self):
        for i in range(5):
            self._assert_freed_after_use(self._save(hist_name=f"var{i}"), f"save {i}: ")

    def test_written_contents_match_the_unit_histogram(self):
        written = self._save()
        self.assertEqual(written.GetNcells(), self.unit_hist.GetNcells())
        for i in range(self.unit_hist.GetNcells()):
            self.assertAlmostEqual(written.GetBinContent(i), 0.5 * i)
            self.assertAlmostEqual(written.GetBinError(i), 0.25 * i)
        self.assertEqual(written.GetEntries(), 17)

    def test_output_file_holds_only_the_named_histogram(self):
        self._save()
        self.out_file.Close()
        with ROOT.TFile.Open(self.path) as read_back:
            self.assertEqual(
                [k.GetName() for k in read_back.GetListOfKeys()], [self.KEY[0]]
            )
            stored = read_back.Get("/".join(self.KEY) + "/var")
            self.assertTrue(stored)
            self.assertEqual(stored.GetEntries(), 17)


if __name__ == "__main__":
    unittest.main()
