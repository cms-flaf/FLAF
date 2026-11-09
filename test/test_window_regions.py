#!/usr/bin/env python3
"""Common/WindowRegions.py: window sets, the window regions a flavour expands to, their guards, and the
keys of a HistFromNtuple job. Plain Python: FillTree with window blocks is checked in test_hist_filler.py.
"""

import os
import sys
import tempfile
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common import WindowRegions as WR  # noqa: E402

# a 10 GeV grid from 200 to 400 and one bin below and above, as the HME grid starts
HME_EDGES = [0.0] + [200.0 + 10.0 * i for i in range(21)] + [500.0]
# 5 GeV bins from 0 to 100, then one to 200
LL_EDGES = [5.0 * i for i in range(21)] + [200.0]
EDGES = {"DeepHME_mass": HME_EDGES, "ll_mass": LL_EDGES}

DEFINITIONS = """
windows:
  300:
    res2b: { DeepHME_mass: [ 250, 350 ] }
    boosted: { DeepHME_mass: [ 0, 400 ] }
  400:
    res2b: { DeepHME_mass: [ 300, 500 ] }
    boosted: { DeepHME_mass: [ 350, 500 ] }
"""

FLAVOUR_VARS = [
    "DeepHME_mass",
    "ll_mass",
    "DNN_M300_Signal",
    "DNN_M400_Signal",
    "lep1_pt",
    {"name": "DNN_m300_vs_HME", "vars": ["DNN_M300_Signal", "DeepHME_mass"]},
]


class WindowRegionsTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "config"))
        with open(os.path.join(self.tmp.name, "config", "hme.yaml"), "w") as f:
            f.write(DEFINITIONS)
        self.global_params = {
            "windows": {
                "HME": {
                    "variables": ["DeepHME_mass"],
                    "definitions": "config/hme.yaml",
                }
            }
        }

    def write_definitions(self, text, name="hme.yaml"):
        with open(os.path.join(self.tmp.name, "config", name), "w") as f:
            f.write(text)

    def sets(self):
        return WR.load_window_sets(self.global_params, self.tmp.name)

    def expand(self, entries, flavour_vars=FLAVOUR_VARS, active=None, regions=None):
        names = {v["name"] if isinstance(v, dict) else v for v in flavour_vars}
        return WR.expand_window_regions(
            {"window_regions": entries},
            self.sets(),
            regions or ["SR", "TT_CR", "DY_CR"],
            ["res2b", "recovery", "boosted"],
            flavour_vars,
            names if active is None else set(active),
            lambda var: EDGES[var],
        )


class Expansion(WindowRegionsTestBase):
    def test_each_mass_holds_only_its_own_variable(self):
        regions = self.expand(
            [
                {
                    "window": "HME",
                    "regions": ["SR", "TT_CR"],
                    "masses": "all",
                    "vars": ["DNN_M{MX}_Signal"],
                }
            ]
        )
        self.assertEqual(
            [(r["name"], r["base_region"], r["mass"], r["vars"]) for r in regions],
            [
                ("SR_HMEWin_m300", "SR", 300, ["DNN_M300_Signal"]),
                ("TT_CR_HMEWin_m300", "TT_CR", 300, ["DNN_M300_Signal"]),
                ("SR_HMEWin_m400", "SR", 400, ["DNN_M400_Signal"]),
                ("TT_CR_HMEWin_m400", "TT_CR", 400, ["DNN_M400_Signal"]),
            ],
        )

    def test_cuts_on_the_bin_index_with_the_flows_at_the_grid_ends(self):
        r300, r400 = self.expand(
            [{"window": "HME", "regions": ["SR"], "masses": "all", "vars": ["lep1_pt"]}]
        )
        # 250 is edge 6, 350 edge 16: bins 6..15
        self.assertEqual(
            r300["cuts"]["res2b"], "DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 16"
        )
        # from the first edge: the underflow too
        self.assertEqual(r300["cuts"]["boosted"], "DeepHME_mass_bin < 21")
        # to the last edge: the overflow too
        self.assertEqual(r400["cuts"]["res2b"], "DeepHME_mass_bin >= 11")
        self.assertNotIn("recovery", r300["cuts"])

    def test_a_listed_mass_and_a_restricted_variable_list(self):
        regions = self.expand(
            [
                {
                    "window": "HME",
                    "regions": ["SR"],
                    "masses": [400],
                    "vars": ["DNN_M{MX}_Signal", "lep1_pt"],
                }
            ],
            active={"DeepHME_mass", "DNN_M400_Signal"},
        )
        # a user_custom `variables:` that leaves lep1_pt out drops it, without an error
        self.assertEqual(
            [(r["name"], r["vars"]) for r in regions],
            [("SR_HMEWin_m400", ["DNN_M400_Signal"])],
        )

    def test_no_entries_no_regions(self):
        self.assertEqual(self.expand([]), [])

    def test_region_name_pattern(self):
        self.global_params["windows"]["HME"]["region_name"] = "{region}_HME{MX}"
        regions = self.expand(
            [{"window": "HME", "regions": ["SR"], "masses": [300], "vars": ["lep1_pt"]}]
        )
        self.assertEqual(regions[0]["name"], "SR_HME300")


class Guards(WindowRegionsTestBase):
    ENTRY = {
        "window": "HME",
        "regions": ["SR"],
        "masses": "all",
        "vars": ["DNN_M{MX}_Signal"],
    }

    def refused(self, msg, **changes):
        entry = dict(self.ENTRY, **changes)
        with self.assertRaisesRegex(WR.WindowRegionError, msg):
            self.expand([entry])

    def test_unknown_window_set(self):
        self.refused("window set 'mbb' is not defined", window="mbb")

    def test_unknown_region(self):
        self.refused("region 'CR_X' is not among the regions", regions=["CR_X"])

    def test_mass_without_window(self):
        self.refused("mass 500 has no window", masses=[500])

    def test_variable_not_histogrammed(self):
        self.refused("'mbb' is not a variable of the flavour", vars=["mbb"])

    def test_mass_tagged_variable_without_placeholder(self):
        self.refused("names mass 300 but has no \\{MX\\}", vars=["DNN_M300_Signal"])

    def test_unknown_placeholder(self):
        self.refused("unknown placeholder", vars=["DNN_M{MASS}_Signal"])

    def test_unknown_key(self):
        self.refused("unknown keys \\['mass'\\]", mass=300)

    def test_window_variable_not_histogrammed(self):
        with self.assertRaisesRegex(
            WR.WindowRegionError, "DeepHME_mass is not histogrammed"
        ):
            self.expand(
                [self.ENTRY], flavour_vars=["DNN_M300_Signal", "DNN_M400_Signal"]
            )

    def test_edge_off_the_binning(self):
        self.write_definitions(
            "windows:\n  300: { res2b: { DeepHME_mass: [ 255, 350 ] } }\n"
        )
        self.refused("255.0 is not a bin edge", masses=[300])

    def test_category_unknown(self):
        self.write_definitions(
            "windows:\n  300: { res3b: { DeepHME_mass: [ 250, 350 ] } }\n"
        )
        self.refused("category 'res3b'", masses=[300])

    def test_box_variable_not_in_the_set(self):
        self.write_definitions("windows:\n  300: { res2b: { ll_mass: [ 10, 80 ] } }\n")
        with self.assertRaisesRegex(
            WR.WindowRegionError, "\\['ll_mass'\\] not among the set's variables"
        ):
            self.sets()

    def test_a_window_is_a_box_not_a_list(self):
        self.write_definitions("windows:\n  300: { res2b: [ 250, 350 ] }\n")
        with self.assertRaisesRegex(WR.WindowRegionError, "expected a box"):
            self.sets()

    def test_the_old_single_variable_key_is_refused(self):
        self.global_params["windows"]["HME"] = {
            "variable": "DeepHME_mass",
            "definitions": "config/hme.yaml",
        }
        with self.assertRaisesRegex(WR.WindowRegionError, "needs 'variables'"):
            self.sets()

    def test_region_defined_twice(self):
        with self.assertRaisesRegex(WR.WindowRegionError, "defined twice"):
            self.expand([self.ENTRY, self.ENTRY])

    def test_definitions_file_missing(self):
        self.global_params["windows"]["HME"]["definitions"] = "config/none.yaml"
        with self.assertRaisesRegex(WR.WindowRegionError, "not found"):
            self.sets()


class MultiVariableWindows(WindowRegionsTestBase):
    """A window set of two variables: every variable of a box is cut on, a variable left out of a
    box is not, and the edges of each variable are checked against its own binning."""

    def setUp(self):
        super().setUp()
        self.write_definitions(
            """
windows:
  300:
    res2b: { DeepHME_mass: [ 250, 350 ], ll_mass: [ 10, 80 ] }
    recovery: { ll_mass: [ 0, 200 ] }
    boosted: { DeepHME_mass: [ 260, 400 ] }
""",
            "hme_mll.yaml",
        )
        self.global_params["windows"]["HME_mll"] = {
            "variables": ["DeepHME_mass", "ll_mass"],
            "definitions": "config/hme_mll.yaml",
        }

    def regions(self):
        return self.expand(
            [
                {
                    "window": "HME_mll",
                    "regions": ["SR"],
                    "masses": [300],
                    "vars": ["DNN_M{MX}_Signal"],
                }
            ]
        )

    def test_every_variable_of_the_box_is_cut_on(self):
        (r,) = self.regions()
        self.assertEqual(r["name"], "SR_HME_mllWin_m300")
        # HME bins 6..15 and ll_mass 10..80 = bins 2..15
        self.assertEqual(
            r["cuts"]["res2b"],
            "DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 16 "
            "&& ll_mass_bin >= 2 && ll_mass_bin < 16",
        )
        # a variable left out of the box is not cut on
        self.assertEqual(
            r["cuts"]["boosted"], "DeepHME_mass_bin >= 7 && DeepHME_mass_bin < 21"
        )
        # a box over the whole binning of its variable (flows included) cuts nothing
        self.assertEqual(r["cuts"]["recovery"], "true")

    def test_edges_are_checked_per_variable(self):
        self.write_definitions(
            "windows:\n  300: { res2b: { DeepHME_mass: [ 250, 350 ], ll_mass: [ 12, 80 ] } }\n",
            "hme_mll.yaml",
        )
        with self.assertRaisesRegex(
            WR.WindowRegionError, "HME_mll m=300 res2b ll_mass low edge: 12.0"
        ):
            self.regions()

    def test_every_variable_of_the_set_must_be_histogrammed(self):
        with self.assertRaisesRegex(
            WR.WindowRegionError, "ll_mass is not histogrammed"
        ):
            self.expand(
                [
                    {
                        "window": "HME_mll",
                        "regions": ["SR"],
                        "masses": [300],
                        "vars": ["DNN_M{MX}_Signal"],
                    }
                ],
                flavour_vars=["DeepHME_mass", "DNN_M300_Signal"],
            )


class JobBlocks(WindowRegionsTestBase):
    def test_keys_per_base_key_with_a_window(self):
        regions = self.expand(
            [
                {
                    "window": "HME",
                    "regions": ["SR"],
                    "masses": [300],
                    "vars": ["DNN_M{MX}_Signal"],
                }
            ]
        )
        base_keys = [
            (("eE", "SR", "res2b"), "f0"),
            (("eE", "SR", "recovery"), "f1"),  # no window in this category
            (("eE", "TT_CR", "res2b"), "f2"),  # region not listed
            (("muMu", "SR", "boosted"), "f3"),
        ]
        blocks = WR.window_blocks(
            base_keys, regions, ["DeepHME_mass", "DNN_M300_Signal"]
        )
        self.assertEqual(len(blocks), 1)
        keys, block_vars = blocks[0]
        self.assertEqual(block_vars, ["DNN_M300_Signal"])
        self.assertEqual(
            [(k, i) for k, i, _ in keys],
            [
                (("eE", "SR_HMEWin_m300", "res2b"), 0),
                (("muMu", "SR_HMEWin_m300", "boosted"), 3),
            ],
        )
        self.assertEqual(WR.count_window_hists(blocks, 10), 20)

    def test_a_job_without_the_window_variables_books_nothing(self):
        regions = self.expand(
            [
                {
                    "window": "HME",
                    "regions": ["SR"],
                    "masses": "all",
                    "vars": ["DNN_M{MX}_Signal"],
                }
            ]
        )
        self.assertEqual(
            WR.window_blocks([(("eE", "SR", "res2b"), "f")], regions, ["lep1_pt"]), []
        )


class Axis(unittest.TestCase):
    def test_uniform_and_listed_edges(self):
        self.assertEqual(WR.axis_edges("4|0:2"), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(WR.axis_edges([0, 1, 5]), [0.0, 1.0, 5.0])

    def test_a_window_spanning_everything_is_no_cut(self):
        self.assertEqual(WR.bin_cut("v_bin", [0.0, 1.0, 2.0], 0, 2, "w"), "true")


if __name__ == "__main__":
    unittest.main()
