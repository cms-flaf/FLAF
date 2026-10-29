#!/usr/bin/env python3
"""Plot groups (global.yaml's plot_process_groups) and control regions (plot_control_regions),
as HistPlotter applies them.

Processes are summed into one legend entry per group, or per set of parameter values for a
group whose patterns name parameters: masses (GluGluToRadion_bbWW_2L_${MASS}) or couplings
(GluGluToHHto2B2Tau_${kl}_${kt}_${c2}, with values such as 1p74 and m0p758). Entries come in
increasing order of their values, numbers by value; a mistake in a group raises instead of
giving a plot that looks right but is not.
"""

import os
import sys
import tempfile
import unittest
import warnings

import pytest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common.PlotGroups import (  # noqa: E402
    PLOT_SETTINGS,
    GroupProcesses,
    _group_instances,
    _param_value_key,
    control_regions,
    drop_empty_entries,
    plot_settings,
    want_data,
    want_signals,
)


class Hist:
    """The part of a ROOT TH1 that GroupProcesses and drop_empty_entries use: bin contents
    (the cells, under- and overflow taken as included) and their errors."""

    def GetNcells(self):
        return len(self.values)

    def GetBinContent(self, i):
        return self.values[i]

    def __init__(self, values, errors=None, name=""):
        self.values = list(values)
        self.errors = list(errors) if errors else [0.0] * len(self.values)
        self.name = name

    def Clone(self, name):
        return Hist(self.values, self.errors, name)

    def SetDirectory(self, directory):
        pass

    def Add(self, other):
        self.values = [a + b for a, b in zip(self.values, other.values)]
        self.errors = [(a**2 + b**2) ** 0.5 for a, b in zip(self.errors, other.errors)]


def entry(kind, value=1.0, color="kBlack", errors=None):
    return (Hist([value, 2 * value], errors), "plot name", color, kind)


def names(instances):
    return [[dict(values), members] for values, members in instances.items()]


class Grouping(unittest.TestCase):
    def test_without_groups_nothing_changes(self):
        hists = {"TT": entry("backgrounds"), "data": entry("data")}
        self.assertEqual(GroupProcesses(hists, []), hists)

    def test_members_are_summed_bin_by_bin_with_errors(self):
        hists = {
            "TT": entry("backgrounds", 10.0, "kOrange", errors=[3.0, 0.0]),
            "ttV": entry("backgrounds", 1.0, errors=[4.0, 0.0]),
            "DY": entry("backgrounds", 5.0),
        }
        grouped = GroupProcesses(hists, [{"name": "tt", "processes": ["TT", "ttV"]}])
        self.assertEqual(list(grouped), ["TT+ttV", "DY"])
        total, label, color, kind = grouped["TT+ttV"]
        self.assertEqual(total.values, [11.0, 22.0])
        self.assertEqual(total.errors, [5.0, 0.0])
        self.assertEqual((label, color, kind), ("tt", "kOrange", "backgrounds"))
        # the inputs are not modified
        self.assertEqual(hists["TT"][0].values, [10.0, 20.0])

    def test_masses_make_one_entry_each_in_numeric_order(self):
        hists = {
            f"GluGluToRadion_bbWW_{ch}_{m}": entry("signals")
            for m in ("1000", "400", "4000", "700")
            for ch in ("2L", "1L")
        }
        group = {
            "name": "X M=${MASS}",
            "processes": [
                "GluGluToRadion_bbWW_2L_${MASS}",
                "GluGluToRadion_bbWW_1L_${MASS}",
            ],
        }
        grouped = GroupProcesses(hists, [group])
        self.assertEqual(
            [label for _, label, _, _ in grouped.values()],
            ["X M=400", "X M=700", "X M=1000", "X M=4000"],
        )
        self.assertEqual(
            list(grouped)[0], "GluGluToRadion_bbWW_2L_400+GluGluToRadion_bbWW_1L_400"
        )

    def test_couplings_in_the_cms_p_m_form_sort_by_value(self):
        # HH_bbtautau's VBF signals: plain integers, 1p74 = 1.74, m0p758 = -0.758
        hists = {
            f"VBFHHto2B2Tau_{v}": entry("signals")
            for v in ("1_1_1", "1p74_1p37_14p4", "m0p758_1p44_m19p3", "1_0_1")
        }
        group = {
            "name": "VBF CV=${CV} C2V=${C2V} C3=${C3}",
            "processes": ["VBFHHto2B2Tau_${CV}_${C2V}_${C3}"],
        }
        self.assertEqual(
            [values for values, _ in names(_group_instances(group, hists, set()))],
            [
                {"CV": "m0p758", "C2V": "1p44", "C3": "m19p3"},
                {"CV": "1", "C2V": "0", "C3": "1"},
                {"CV": "1", "C2V": "1", "C3": "1"},
                {"CV": "1p74", "C2V": "1p37", "C3": "14p4"},
            ],
        )

    def test_parameters_compare_in_the_order_the_pattern_names_them(self):
        # HH_bbtautau's ggF signals: kl first, then kt, then c2 (not alphabetical: c2, kl)
        hists = {
            f"GluGluToHHto2B2Tau_{v}": entry("signals")
            for v in ("1p00_1p00_0p00", "0p00_1p00_1p00", "2p45_1p00_0p00")
        }
        group = {
            "name": "ggHH kl=${kl} kt=${kt} c2=${c2}",
            "processes": ["GluGluToHHto2B2Tau_${kl}_${kt}_${c2}"],
        }
        self.assertEqual(
            [members for _, members in names(_group_instances(group, hists, set()))],
            [
                ["GluGluToHHto2B2Tau_0p00_1p00_1p00"],
                ["GluGluToHHto2B2Tau_1p00_1p00_0p00"],
                ["GluGluToHHto2B2Tau_2p45_1p00_0p00"],
            ],
        )

    def test_any_value_sorts_without_raising(self):
        # numbers by value first (m2 = -2), then the rest as text; mixing used to raise
        # TypeError (int < str), and '²' ValueError ('²'.isdigit() but int('²') fails)
        values = ["400", "M400", "0p00_1p00", "alt", "²", "1.5", "M-400", "m2", "10"]
        self.assertEqual(
            sorted(values, key=_param_value_key),
            ["m2", "1.5", "10", "400", "0p00_1p00", "M-400", "M400", "alt", "²"],
        )

    def test_values_may_hold_dots_and_dashes(self):
        hists = {"S_1.5": entry("signals"), "S_M-400": entry("signals")}
        group = {"name": "S ${P}", "processes": ["S_${P}"]}
        self.assertEqual(len(_group_instances(group, hists, set())), 2)

    def test_a_process_taken_by_an_earlier_group_is_not_taken_again(self):
        hists = {"TT": entry("backgrounds"), "ttV": entry("backgrounds")}
        grouped = GroupProcesses(
            hists,
            [
                {"name": "ttV", "processes": ["ttV"]},
                {"name": "tt", "processes": ["TT", "ttV"]},
            ],
        )
        self.assertEqual(list(grouped), ["ttV", "TT"])


class Mistakes(unittest.TestCase):
    hists = {
        "TT": entry("backgrounds", 10.0),
        "ttV": entry("backgrounds", 1.0),
        "H": entry("backgrounds"),
        "S_1L_400": entry("signals"),
        "S_2L_400": entry("signals"),
        "S_400": entry("signals"),
        "data": entry("data"),
    }

    def assertRaisesFor(self, group, message, error=ValueError):
        with self.assertRaisesRegex(error, message):
            GroupProcesses(self.hists, [group])

    def test_a_process_listed_twice_raises(self):
        # was stacked twice: TT + TT + ttV = 21 from an input of 11
        self.assertRaisesFor(
            {"name": "tt", "processes": ["TT", "TT", "ttV"]}, "more than once"
        )

    def test_a_histogram_matched_by_two_patterns_raises(self):
        self.assertRaisesFor(
            {"name": "S ${M}", "processes": ["S_1L_${M}", "S_${M}"]},
            "counted twice",
        )

    def test_processes_must_be_a_list_of_names(self):
        for processes in ("TT", [], ["TT", 1], None):
            with self.subTest(processes=processes):
                self.assertRaisesFor(
                    {"name": "tt", "processes": processes}, "list of process names"
                )

    def test_every_process_must_name_the_same_parameters(self):
        # [H, S_2L_${M}] would bypass the background / signal check per instance
        self.assertRaisesFor(
            {"name": "H ${M}", "processes": ["H", "S_2L_${M}"]}, "same parameters"
        )

    def test_the_name_must_name_the_parameters(self):
        self.assertRaisesFor(
            {"name": "S", "processes": ["S_2L_${M}"]}, "parameters of its processes"
        )
        self.assertRaisesFor(
            {"name": "S ${M} ${SPIN}", "processes": ["S_2L_${M}"]},
            "parameters of its processes",
        )

    def test_an_unknown_key_raises(self):
        for key in ("colour", "procesess"):
            with self.subTest(key=key):
                self.assertRaisesFor(
                    {"name": "tt", "processes": ["TT", "ttV"], key: "kRed"},
                    rf"'tt' has the unknown keys \['{key}'\]",
                )

    def test_a_parameter_must_be_a_name(self):
        for param in ("1", "M-X", ""):
            with self.subTest(param=param):
                self.assertRaisesFor(
                    {"name": f"S ${{{param}}}", "processes": [f"S_${{{param}}}"]},
                    r"plot group 'S .*': .* is not a parameter",
                )

    def test_backgrounds_signals_and_data_do_not_mix(self):
        self.assertRaisesFor(
            {"name": "x", "processes": ["TT", "S_400"]}, "mixes", RuntimeError
        )
        self.assertRaisesFor(
            {"name": "x", "processes": ["TT", "data"]}, "mixes", RuntimeError
        )


class EmptyEntries(unittest.TestCase):
    """drop_empty_entries: an entry without events is not drawn, a group keeps its name."""

    groups = [{"name": "tt + ttV", "processes": ["TT", "ttV"]}]

    def drawn(self, hists):
        return {
            key: (entry[1], entry[0].values)
            for key, entry in drop_empty_entries(
                GroupProcesses(hists, self.groups)
            ).items()
        }

    def test_an_empty_member_leaves_the_group_and_its_name(self):
        drawn = self.drawn({"TT": entry("backgrounds"), "ttV": entry("backgrounds", 0)})
        self.assertEqual(drawn, {"TT+ttV": ("tt + ttV", [1.0, 2.0])})

    def test_an_empty_group_or_process_is_not_drawn(self):
        hists = {
            "TT": entry("backgrounds", 0),
            "ttV": entry("backgrounds", 0),
            "DY": entry("backgrounds"),
            "VV": entry("backgrounds", 0),
            "S_400": entry("signals", 0),
        }
        self.assertEqual(list(self.drawn(hists)), ["DY"])

    def test_cancelling_bins_are_not_empty(self):
        hists = {"W": (Hist([1.0, -1.0]), "W", "kRed", "backgrounds")}
        self.assertEqual(list(self.drawn(hists)), ["W"])

    def test_data_without_events_is_drawn(self):
        self.assertEqual(list(self.drawn({"data": entry("data", 0)})), ["data"])


class ControlRegions(unittest.TestCase):
    """plot_control_regions: where blinded variables show data (and no signal is drawn)."""

    regions = ["OS_Iso", "SR", "TT_CR", "DY_CR", "SR_SL_M300"]

    def test_a_blinded_variable_shows_data_only_in_the_control_regions(self):
        crs = control_regions(["TT_CR", "DY_CR"], self.regions)
        for region, blind, expected in [
            ("SR", True, False),
            ("OS_Iso", True, False),
            ("SR_SL_M300", True, False),
            ("TT_CR", True, True),
            ("DY_CR", True, True),
            ("SR", False, True),
            ("TT_CR", False, True),
        ]:
            with self.subTest(region=region, blind=blind):
                self.assertEqual(want_data(True, blind, region, crs), expected)

    def test_signals_are_drawn_when_wanted_and_never_in_a_control_region(self):
        crs = control_regions(["TT_CR", "DY_CR"], self.regions)
        for region, want, expected in [
            ("SR", True, True),
            ("OS_Iso", True, True),
            ("TT_CR", True, False),
            ("DY_CR", True, False),
            # plot_wantSignals: false -- the switch #183 had left without effect
            ("SR", False, False),
            ("TT_CR", False, False),
        ]:
            with self.subTest(region=region, want=want):
                self.assertEqual(want_signals(want, region, crs), expected)
        with self.assertRaises(TypeError):
            want_signals(True, "SR", "TT_CR")

    def test_no_data_when_it_is_not_wanted(self):
        crs = control_regions(["TT_CR"], self.regions)
        for region in ("SR", "TT_CR"):
            for blind in (True, False):
                self.assertFalse(want_data(False, blind, region, crs))

    def test_without_the_key_every_blinded_region_stays_blinded(self):
        crs = control_regions(None, self.regions)
        self.assertEqual(crs, [])
        self.assertFalse(want_data(True, True, "TT_CR", crs))
        self.assertTrue(want_data(True, False, "TT_CR", crs))

    def test_a_string_is_refused(self):
        # "SR" in "SR_SL_M300_CR" is True for a string: it would unblind the SR
        for value in ("TT_CR, DY_CR", "SR_SL_M300_CR", {"TT_CR": 1}, ["TT_CR", 1]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "list of region names"):
                    control_regions(value, self.regions)
        with self.assertRaises(TypeError):
            want_data(True, True, "SR", "SR_SL_M300_CR")

    def test_an_unknown_region_is_warned_about_and_stays_blinded(self):
        with self.assertWarnsRegex(UserWarning, r"\['TT_CRR'\]"):
            crs = control_regions(["TT_CRR", "DY_CR"], self.regions)
        self.assertFalse(want_data(True, True, "TT_CR", crs))
        # without the analysis' region list there is nothing to compare with
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(control_regions(["TT_CRR"]), ["TT_CRR"])


class Settings(unittest.TestCase):
    """plot_settings: what a plot shows, from the merged Setup.global_params of its eras."""

    params = {
        "plot_process_groups": [{"name": "tt", "processes": ["TT", "ttV"]}],
        "plot_control_regions": ["TT_CR", "DY_CR"],
        "signal_plot_scale": "bkg",
        "custom_regions": "QCDRegions",
        "QCDRegions": ["OS_Iso", "SR", "TT_CR", "DY_CR"],
    }

    def test_one_era(self):
        settings = plot_settings({"Run3_2022": dict(self.params)})
        self.assertEqual(settings["plot_control_regions"], ["TT_CR", "DY_CR"])
        self.assertEqual(settings["signal_plot_scale"], "bkg")
        self.assertEqual(settings["regions"], ["OS_Iso", "SR", "TT_CR", "DY_CR"])

    def test_unset_settings_are_none(self):
        settings = plot_settings({"Run3_2022": {}})
        self.assertEqual(
            [settings[k] for k in PLOT_SETTINGS + ("regions",)], [None] * 4
        )

    def test_the_eras_of_a_meta_era_must_agree(self):
        eras = {"Run3_2022": dict(self.params), "Run3_2023": dict(self.params)}
        self.assertEqual(plot_settings(eras)["signal_plot_scale"], "bkg")
        eras["Run3_2023"]["plot_control_regions"] = ["TT_CR"]
        with self.assertRaisesRegex(ValueError, "plot_control_regions differs"):
            plot_settings(eras)

    def test_regions_given_as_a_mapping_count_by_their_names(self):
        params = dict(
            self.params,
            custom_regions="MuMuMassRegions",
            MuMuMassRegions={"Z_sideband": "m > 70", "Signal_Fit": "m > 110"},
        )
        settings = plot_settings({"Run2_2018": params})
        self.assertEqual(settings["regions"], ["Z_sideband", "Signal_Fit"])
        with self.assertWarnsRegex(UserWarning, r"\['Z_sidebnd'\]"):
            control_regions(["Z_sidebnd"], settings["regions"])

    def test_the_regions_of_all_eras(self):
        other = dict(self.params, QCDRegions=["SR", "W_CR"])
        settings = plot_settings({"Run3_2022": dict(self.params), "Run3_2023": other})
        self.assertEqual(
            settings["regions"], ["OS_Iso", "SR", "TT_CR", "DY_CR", "W_CR"]
        )


try:
    import ROOT  # noqa: F401
except ImportError:
    ROOT = None


@pytest.mark.needs_root
@unittest.skipIf(ROOT is None, "needs ROOT")
class RootOwnership(unittest.TestCase):
    """The group totals must be freed after the plot. PyROOT does not own what Clone() returns,
    so SetDirectory(0) alone left each total owned by nobody, one histogram leaked per group and
    plot. The tests run with an input file open, as HistPlotter calls GroupProcesses.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, "merged.root")
        ROOT.TFile(path, "RECREATE").Close()
        # the current directory while GroupProcesses runs, as in HistPlotter
        self.in_file = ROOT.TFile.Open(path, "READ")

    def tearDown(self):
        self.in_file.Close()
        self.tmp.cleanup()

    def hist(self, name, value):
        """A histogram as HistPlotter's _detached leaves it: in no directory, owned by
        Python."""
        hist = ROOT.TH1D(name, "", 2, 0.0, 2.0)
        hist.SetDirectory(0)
        ROOT.SetOwnership(hist, True)
        hist.SetBinContent(1, value)
        return hist

    def test_fixture_reproduces_the_leak_condition(self):
        clone = self.hist("probe", 1.0).Clone("clone")
        self.assertTrue(clone.GetDirectory())
        self.assertFalse(clone.__python_owns__)
        clone.SetDirectory(0)
        ROOT.SetOwnership(clone, True)

    def test_group_totals_are_owned_by_python(self):
        hists = {
            p: (self.hist(p, value), p, "kBlack", "backgrounds")
            for p, value in (("TT", 10.0), ("ttV", 1.0), ("W", 5.0))
        }
        groups = [
            {"name": "tt", "processes": ["TT", "ttV"]},
            {"name": "W", "processes": ["W"]},
        ]
        grouped = GroupProcesses(hists, groups)
        self.assertEqual(grouped["TT+ttV"][0].GetBinContent(1), 11.0)
        for key in ("TT+ttV", "W"):
            total = grouped[key][0]
            self.assertTrue(
                total.__python_owns__,
                f"the total {key} is not owned by Python, so it outlives the plot",
            )
            self.assertFalse(total.GetDirectory())


if __name__ == "__main__":
    unittest.main()
