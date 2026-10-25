#!/usr/bin/env python3
"""Data blinding in HistPlotter's plots (Common/PlotBlinding.py): the histograms.yaml keys
`blind` and `blind_ranges`, and the data points hidden inside blind_ranges windows.

A hidden bin has NaN as content and error. The ROOT-free tests use a stand-in for ROOT's TH1
and for HistPlotter's rebinning (each stored bin goes to the plotted bin that holds its centre,
as TH1::Rebin does); RootRebinTest runs the real TH1 and HistHelper.RebinHisto, and
RenderedPlotTest the real PlotKit rendering of both panels, where ROOT or PlotKit can be
imported.
"""

import math
import os
import sys
import tempfile
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common.PlotBlinding import (  # noqa: E402
    BLIND_KEYS,
    blind_data,
    blind_windows,
    check_data_style,
    overlapping_bins,
)


class Hist:
    """The bin accessors of a ROOT TH1 that blind_data uses. `values` are the bins 1..n;
    `under` and `over` the under- and overflow."""

    def __init__(self, edges, values=None, under=0.0, over=0.0, name="data"):
        self.edges = [float(e) for e in edges]
        n = len(self.edges) - 1
        values = [0.0] * n if values is None else [float(v) for v in values]
        assert len(values) == n
        self.values = [float(under)] + values + [float(over)]
        self.errors = [math.sqrt(abs(v)) for v in self.values]
        self.name = name

    def GetName(self):
        return self.name

    def GetNbinsX(self):
        return len(self.edges) - 1

    def GetBinLowEdge(self, i):
        if i == 0:
            return -math.inf
        return self.edges[i - 1]

    def GetBinContent(self, i):
        return self.values[i]

    def GetBinError(self, i):
        return self.errors[i]

    def SetBinContent(self, i, value):
        self.values[i] = value

    def SetBinError(self, i, error):
        self.errors[i] = error

    def Clone(self, name):
        clone = Hist(self.edges, name=name)
        clone.values = list(self.values)
        clone.errors = list(self.errors)
        return clone

    def SetDirectory(self, directory):
        pass

    def Reset(self):
        self.values = [0.0] * len(self.values)
        self.errors = [0.0] * len(self.errors)


def rebinner(new_edges, merge_overflow=False):
    """HistPlotter's apply_binning on the stand-in: every bin, under- and overflow included,
    goes to the new bin that holds its centre (centre <= upper edge), or to the under- or
    overflow; merge_overflow adds the overflow to the last bin (wantOverflow)."""

    def rebin(hist, name):
        out = Hist(new_edges, name=name)
        n_new = out.GetNbinsX()
        for i, value in enumerate(hist.values):
            if i == 0:
                target = 0
            elif i == len(hist.values) - 1:
                target = n_new + 1
            else:
                centre = 0.5 * (hist.edges[i - 1] + hist.edges[i])
                if centre < new_edges[0]:
                    target = 0
                elif centre > new_edges[-1]:
                    target = n_new + 1
                else:
                    target = next(
                        j for j in range(1, n_new + 1) if centre <= new_edges[j]
                    )
            out.values[target] += value
        if merge_overflow:
            out.values[n_new] += out.values[n_new + 1]
        return out

    return rebin


def no_rebin(hist, name):
    return hist


def hidden(hist):
    """The bins 1..n whose content is NaN, and a check that their errors are NaN too."""
    bins = [i for i in range(1, hist.GetNbinsX() + 1) if math.isnan(hist.values[i])]
    for i in bins:
        assert math.isnan(hist.errors[i]), (i, hist.errors[i])
    return bins


EDGES = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class BlindWindowsTest(unittest.TestCase):
    def test_no_key_no_window(self):
        self.assertEqual(blind_windows("x", "x", {"x_bins": "10|0:1"}), [])
        self.assertEqual(blind_windows("x", "x", {"blind": True}), [])

    def test_one_window(self):
        entry = {"x_bins": "10|0:1", "blind_ranges": [115, 135]}
        self.assertEqual(blind_windows("x", "x", entry), [(115.0, 135.0)])

    def test_list_of_windows(self):
        entry = {"blind_ranges": [[110, 115], [120.5, 130]]}
        self.assertEqual(
            blind_windows("x", "x", entry), [(110.0, 115.0), (120.5, 130.0)]
        )
        self.assertEqual(blind_windows("x", "x", {"blind_ranges": [[1, 2]]}), [(1, 2)])

    def test_infinite_bound(self):
        entry = {"blind_ranges": [250, math.inf]}
        self.assertEqual(blind_windows("x", "x", entry), [(250.0, math.inf)])

    def test_invalid_windows_raise(self):
        for ranges in (
            [135, 115],
            [115, 115],
            [115],
            [115, 125, 135],
            [],
            [[]],
            [[115, 135], 140],
            [[115, 135], [140]],
            [[135, 115]],
            ["115", "135"],
            [True, 135],
            [115, math.nan],
            "115:135",
            {"lo": 115, "hi": 135},
            None,
            115,
        ):
            with self.subTest(ranges=ranges):
                with self.assertRaisesRegex(ValueError, "blind_ranges"):
                    blind_windows("m_mumu", "m_mumu", {"blind_ranges": ranges})

    def test_unknown_blind_key_raises(self):
        for key in ("blind_region", "blind_range", "blinding", "Blind", "BLIND_RANGES"):
            with self.subTest(key=key):
                entry = {"x_bins": "10|0:1", key: [115, 135]}
                with self.assertRaises(ValueError) as caught:
                    blind_windows("m_mumu", "m_mumu(_.*)?", entry)
                message = str(caught.exception)
                self.assertIn(repr(key), message)
                self.assertIn("m_mumu(_.*)?", message)
                self.assertIn("m_mumu)", message)

    def test_known_keys_and_others_pass(self):
        self.assertEqual(BLIND_KEYS, ("blind", "blind_ranges"))
        entry = {"blind": True, "blind_ranges": [1, 2], "unblinded_note": 1, 7: "x"}
        self.assertEqual(blind_windows("x", "x", entry), [(1.0, 2.0)])

    def test_more_than_one_dimension_raises(self):
        for entry in (
            {"var_list": ["a", "b"], "blind_ranges": [1, 2]},
            {"2d": {"combined_bins": []}, "blind_ranges": [1, 2]},
        ):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "not supported"):
                    blind_windows("v", "v", entry)
        # the blinding of the whole plot stays available to them
        self.assertEqual(blind_windows("v", "v", {"var_list": ["a", "b"]}), [])
        self.assertEqual(
            blind_windows("v", "v", {"var_list": ["v"], "blind_ranges": [1, 2]}),
            [(1.0, 2.0)],
        )


class CheckDataStyleTest(unittest.TestCase):
    def test_data_hist_blind_raises(self):
        for key in ("blind", "blind_ranges", "Blind"):
            for which in ("cms_stacked.yaml", "Run3_2022.yaml"):
                with self.subTest(key=key, which=which):
                    cfgs = {
                        "cms_stacked.yaml": {"data_hist": {"draw_opt": "0pe"}},
                        "Run3_2022.yaml": {"lumi_text": {}},
                    }
                    cfgs[which]["data_hist"] = {"marker_style": 20, key: True}
                    with self.assertRaises(ValueError) as caught:
                        check_data_style(cfgs)
                    self.assertIn(which, str(caught.exception))
                    self.assertIn(repr(key), str(caught.exception))

    def test_without_blind_passes(self):
        check_data_style(
            {
                "cms_stacked.yaml": {"data_hist": {"draw_opt": "0pe"}},
                "Run3_2022.yaml": {"data_hist": None},
                "other.yaml": None,
            }
        )


class OverlapTest(unittest.TestCase):
    def test_boundary_rule(self):
        # a bin ending at lo or starting at hi is shown; partial overlaps are hidden
        self.assertEqual(overlapping_bins(EDGES, [(115, 135)]), [3, 4, 5, 6])
        self.assertEqual(overlapping_bins(EDGES, [(117, 133)]), [3, 4, 5, 6])
        self.assertEqual(overlapping_bins(EDGES, [(114.9, 135.1)]), [2, 3, 4, 5, 6, 7])
        self.assertEqual(overlapping_bins(EDGES, [(60, 70)]), [])
        self.assertEqual(overlapping_bins(EDGES, [(150, 160)]), [])
        self.assertEqual(overlapping_bins(EDGES, [(-math.inf, 101)]), [0])

    def test_flow_bins(self):
        flow_edges = [-math.inf] + EDGES + [math.inf]
        self.assertEqual(overlapping_bins(flow_edges, [(140, 200)]), [9, 10, 11])
        self.assertEqual(overlapping_bins(flow_edges, [(0, 100)]), [0])


class BlindDataTest(unittest.TestCase):
    def test_window_hides_its_bins_only(self):
        stored = Hist(EDGES, VALUES, under=5, over=7)
        plotted = stored.Clone("data")
        before = (list(plotted.values), list(plotted.errors))
        self.assertEqual(
            blind_data(plotted, stored, [(115, 135)], no_rebin), [4, 5, 6, 7]
        )
        self.assertEqual(hidden(plotted), [4, 5, 6, 7])
        for i in (0, 1, 2, 3, 8, 9, 10, 11):
            self.assertEqual(plotted.values[i], before[0][i])
            self.assertEqual(plotted.errors[i], before[1][i])

    def test_partial_overlap_hides_the_bin(self):
        stored = Hist(EDGES, VALUES)
        plotted = stored.Clone("data")
        blind_data(plotted, stored, [(117, 118)], no_rebin)
        self.assertEqual(hidden(plotted), [4])
        plotted = stored.Clone("data")
        blind_data(plotted, stored, [(114.9, 120)], no_rebin)
        self.assertEqual(hidden(plotted), [3, 4])

    def test_no_window_no_effect(self):
        stored = Hist(EDGES, VALUES, under=5, over=7)
        plotted = stored.Clone("data")
        self.assertEqual(blind_data(plotted, stored, [], rebinner(EDGES, True)), [])
        self.assertEqual(plotted.values, stored.values)
        self.assertEqual(plotted.errors, stored.errors)

    def test_list_of_windows(self):
        stored = Hist(EDGES, VALUES)
        plotted = stored.Clone("data")
        blind_data(plotted, stored, [(100, 105), (127, 141)], no_rebin)
        self.assertEqual(hidden(plotted), [1, 6, 7, 8, 9])

    def test_same_object_as_stored(self):
        stored = Hist(EDGES, VALUES)
        blind_data(stored, stored, [(115, 135)], no_rebin)
        self.assertEqual(hidden(stored), [4, 5, 6, 7])

    def test_window_on_rebinned_histogram(self):
        fine = [100 + i for i in range(51)]
        stored = Hist(fine, [1.0] * 50)
        rebin = rebinner([100, 110, 114, 136, 140, 150])
        plotted = rebin(stored, "data")
        blind_data(plotted, stored, [(115, 135)], rebin)
        self.assertEqual(hidden(plotted), [3])
        self.assertEqual([plotted.values[i] for i in (1, 2, 4, 5)], [10, 4, 4, 10])

    def test_events_moved_out_of_a_window_by_rebinning_are_hidden(self):
        # the stored bin [134, 137) holds events of the window [120, 135); its centre, 135.5,
        # puts it in the plotted bin [135, 140), which does not overlap the window
        stored = Hist([100, 120, 134, 137, 140, 150], [5, 5, 5, 5, 5])
        rebin = rebinner([100, 120, 135, 140, 150])
        plotted = rebin(stored, "data")
        self.assertEqual(plotted.values[3], 10)
        self.assertEqual(overlapping_bins(plotted.edges, [(120, 135)]), [1])
        blind_data(plotted, stored, [(120, 135)], rebin)
        self.assertEqual(hidden(plotted), [2, 3])
        self.assertEqual([plotted.values[1], plotted.values[4]], [5, 5])

    def test_overflow_merged_into_last_bin_is_hidden(self):
        stored = Hist(EDGES, VALUES, over=7)
        rebin = rebinner(EDGES, merge_overflow=True)
        plotted = rebin(stored, "data")
        blind_data(plotted, stored, [(160, 200)], rebin)
        self.assertEqual(hidden(plotted), [10])
        # without the merge the overflow is not drawn, so nothing is hidden
        rebin = rebinner(EDGES)
        plotted = rebin(stored, "data")
        blind_data(plotted, stored, [(160, 200)], rebin)
        self.assertEqual(hidden(plotted), [])

    def test_plotted_range_beyond_stored_range(self):
        # plotted bins past the stored range overlap the window but receive no events: they
        # are hidden, so no marker at zero is drawn inside the window
        stored = Hist([0, 250, 500, 750, 1000], [1, 2, 3, 4])
        rebin = rebinner([0, 250, 1000, 2000, 3000])
        plotted = rebin(stored, "data")
        blind_data(plotted, stored, [(250, 3500)], rebin)
        self.assertEqual(hidden(plotted), [2, 3, 4])
        self.assertEqual(plotted.values[1], 1)


try:
    import ROOT  # noqa: F401
except ImportError:
    ROOT = None


# HistHelper includes the analysis' headers at import
@unittest.skipIf(
    ROOT is None or "ANALYSIS_PATH" not in os.environ,
    "needs ROOT and an analysis environment",
)
class RootRebinTest(unittest.TestCase):
    """The real TH1 and HistHelper.RebinHisto, as HistPlotter's apply_binning calls it."""

    def setUp(self):
        import array

        from FLAF.Common.HistHelper import RebinHisto

        self.array = array
        self.RebinHisto = RebinHisto

    def make(self, edges, values, over=0.0):
        hist = ROOT.TH1D("stored", "", len(edges) - 1, self.array.array("d", edges))
        hist.SetDirectory(0)
        for i, value in enumerate(values, start=1):
            hist.SetBinContent(i, value)
            hist.SetBinError(i, math.sqrt(value))
        hist.SetBinContent(len(edges), over)
        return hist

    def apply_binning(self, new_edges, want_overflow):
        def rebin(hist, key):
            return self.RebinHisto(hist, new_edges, key, wantOverflow=want_overflow)

        return rebin

    def test_events_moved_out_of_a_window_by_rebinning_are_hidden(self):
        stored = self.make([100, 120, 134, 137, 140, 150], [5, 5, 5, 5, 5])
        rebin = self.apply_binning([100, 120, 135, 140, 150], False)
        plotted = rebin(stored, "data")
        self.assertEqual(plotted.GetBinContent(3), 10)
        self.assertEqual(blind_data(plotted, stored, [(120, 135)], rebin), [2, 3])
        self.assertEqual(plotted.GetBinContent(1), 5)
        self.assertEqual(plotted.GetBinContent(4), 5)
        for i in (2, 3):
            self.assertTrue(math.isnan(plotted.GetBinContent(i)))
            self.assertTrue(math.isnan(plotted.GetBinError(i)))

    def test_overflow_merged_into_last_bin_is_hidden(self):
        stored = self.make(EDGES, VALUES, over=7)
        for want_overflow, expected in ((True, [10]), (False, [])):
            with self.subTest(want_overflow=want_overflow):
                rebin = self.apply_binning(EDGES, want_overflow)
                plotted = rebin(stored, "data")
                self.assertEqual(
                    blind_data(plotted, stored, [(160, 200)], rebin), expected
                )

    def test_aligned_window(self):
        fine = [100 + 0.25 * i for i in range(201)]
        stored = self.make(fine, [1.0] * 200)
        rebin = self.apply_binning([100 + i for i in range(51)], False)
        plotted = rebin(stored, "data")
        self.assertEqual(
            blind_data(plotted, stored, [(115, 135)], rebin), list(range(16, 36))
        )
        self.assertEqual(plotted.GetBinContent(15), 4)
        self.assertEqual(plotted.GetBinContent(36), 4)


try:
    import FLAF.PlotKit.Plotter as Plotter
except ImportError:
    Plotter = None


@unittest.skipIf(Plotter is None, "needs PlotKit and its dependencies")
class RenderedPlotTest(unittest.TestCase):
    """The data markers PlotKit's default (matplotlib) backend draws in the plot and in the
    ratio panel, for a data histogram blinded by blind_data."""

    def markers(self, data):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures = []
        savefig = matplotlib.figure.Figure.savefig

        def keep(fig, *args, **kwargs):
            figures.append(fig)

        bkg = Hist(EDGES, [100] * 10, name="bkg")
        plotter = Plotter.Plotter(
            page_cfg={"page_setup": {"backend": "mplhep"}},
            hist_cfg={"x": {"x_title": "x"}},
        )
        matplotlib.figure.Figure.savefig = keep
        close = plt.close
        plt.close = lambda *args: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                plotter.plot(
                    "x",
                    {
                        "bkg": (bkg, "Bkg", "kAzure-9", "backgrounds"),
                        "data": (data, "Data", "kBlack", "data"),
                    },
                    os.path.join(tmp, "x.png"),
                    want_data=True,
                    custom={},
                )
        finally:
            matplotlib.figure.Figure.savefig = savefig
            plt.close = close
        (fig,) = figures
        ax, rax = fig.axes
        drawn = []
        for axis in (ax, rax):
            (container,) = axis.containers
            x, y = container.lines[0].get_data()
            drawn.append(
                sorted(round(float(xi)) for xi, yi in zip(x, y) if math.isfinite(yi))
            )
        plt.close(fig)
        return drawn

    def test_window_has_no_marker_in_plot_and_ratio(self):
        centres = [round(0.5 * (a + b)) for a, b in zip(EDGES, EDGES[1:])]
        data = Hist(EDGES, [100] * 10)
        self.assertEqual(self.markers(data), [centres, centres])
        blind_data(data, data, [(117, 133)], no_rebin)
        shown = [c for c in centres if not 115 < c < 135]
        self.assertEqual(self.markers(data), [shown, shown])


if __name__ == "__main__":
    unittest.main()
