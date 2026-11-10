#!/usr/bin/env python3
"""Common/SubRegions.py: the sub-regions a flavour expands to (bin-form and C++ cuts), their guards, the
keys of a HistFromNtuple job, and the flavour's regions (Setup). Plain Python, except the Setup tests,
which are skipped where Setup cannot be imported; FillTree with sub-region blocks is checked in
test_hist_filler.py."""

import os
import sys
import tempfile
import types
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common import SubRegions as SR  # noqa: E402

# a 10 GeV grid from 200 to 400 and one bin below and above, as the HME grid starts
HME_EDGES = [0.0] + [200.0 + 10.0 * i for i in range(21)] + [500.0]
# 5 GeV bins from 0 to 100, then one to 200
LL_EDGES = [5.0 * i for i in range(21)] + [200.0]
EDGES = {"DeepHME_mass": HME_EDGES, "ll_mass": LL_EDGES}

CUTS = """
fits: { note: other keys of the file are left alone }
cuts:
  300:
    res2b: { DeepHME_mass: [ 250, 350 ] }
    boosted: { DeepHME_mass: [ 0, 400 ] }
  400:
    res2b: { DeepHME_mass: [ 300, 500 ] }
    boosted: { DeepHME_mass: [ 350, 500 ] }
"""

FLAVOUR = {
    "variables": [
        "DeepHME_mass",
        "ll_mass",
        "DNN_M300_Signal",
        "DNN_M400_Signal",
        "lep1_pt",
        {"name": "DNN_m300_vs_HME", "vars": ["DNN_M300_Signal", "DeepHME_mass"]},
    ],
    "fullResolution_variables": ["DeepHME_mass", "ll_mass"],
}

ENTRY = {
    "name": "HMESub",
    "cuts": "config/cuts.yaml",
    "regions": ["SR"],
    "masses": "all",
    "vars": ["DNN_M{MX}_Signal"],
}


class SubRegionsTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "config"))
        self.write_cuts(CUTS)

    def write_cuts(self, text, name="cuts.yaml"):
        with open(os.path.join(self.tmp.name, "config", name), "w") as f:
            f.write(text)

    def expand(self, entries, flavour=None, active=None, regions=None):
        cfg = dict(FLAVOUR if flavour is None else flavour)
        cfg["sub_regions"] = entries
        names = {v["name"] if isinstance(v, dict) else v for v in cfg["variables"]}
        return SR.expand_sub_regions(
            "TestFlavour",
            cfg,
            self.tmp.name,
            regions or ["SR", "TT_CR", "DY_CR"],
            ["res2b", "recovery", "boosted"],
            names if active is None else set(active),
            lambda var: EDGES[var],
        )

    def one(self, cut_text, **entry):
        """The cuts of mass 300 from a file whose mass 300 is `cut_text`."""
        self.write_cuts("cuts:\n  300:\n" + cut_text)
        (r,) = self.expand([dict(ENTRY, masses=[300], **entry)])
        return r


class Expansion(SubRegionsTestBase):
    def test_each_mass_holds_only_its_own_variable(self):
        regions = self.expand([dict(ENTRY, regions=["SR", "TT_CR"])])
        self.assertEqual(
            [(r["name"], r["base_region"], r["mass"], r["vars"]) for r in regions],
            [
                ("SR_HMESub_m300", "SR", 300, ["DNN_M300_Signal"]),
                ("TT_CR_HMESub_m300", "TT_CR", 300, ["DNN_M300_Signal"]),
                ("SR_HMESub_m400", "SR", 400, ["DNN_M400_Signal"]),
                ("TT_CR_HMESub_m400", "TT_CR", 400, ["DNN_M400_Signal"]),
            ],
        )
        self.assertEqual({r["sub_region"] for r in regions}, {"HMESub"})

    def test_a_listed_mass_and_a_restricted_variable_list(self):
        regions = self.expand(
            [dict(ENTRY, masses=[400], vars=["DNN_M{MX}_Signal", "lep1_pt"])],
            active={"DeepHME_mass", "DNN_M400_Signal"},
        )
        # a user_custom `variables:` that leaves lep1_pt out drops it, without an error
        self.assertEqual(
            [(r["name"], r["vars"]) for r in regions],
            [("SR_HMESub_m400", ["DNN_M400_Signal"])],
        )

    def test_no_entries_no_sub_regions(self):
        self.assertEqual(self.expand([]), [])

    def test_region_name_pattern(self):
        (r,) = self.expand(
            [dict(ENTRY, masses=[300], region_name="{region}_{name}{MX}")]
        )
        self.assertEqual(r["name"], "SR_HMESub300")


class BinFormCuts(SubRegionsTestBase):
    def test_bin_index_cuts_with_the_flows_at_the_grid_ends(self):
        r300, r400 = self.expand([ENTRY])
        # 250 is edge 6, 350 edge 16: bins 6..15
        self.assertEqual(
            r300["cuts"]["res2b"], "DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 16"
        )
        # from the first edge: the underflow too
        self.assertEqual(r300["cuts"]["boosted"], "DeepHME_mass_bin < 21")
        # to the last edge: the overflow too
        self.assertEqual(r400["cuts"]["res2b"], "DeepHME_mass_bin >= 11")
        self.assertNotIn("recovery", r300["cuts"])
        self.assertEqual(r300["columns"]["res2b"], ["DeepHME_mass_bin"])

    def test_intervals_of_one_variable_are_ored(self):
        r = self.one("    res2b: { DeepHME_mass: [ [ 250, 300 ], [ 350, 400 ] ] }\n")
        self.assertEqual(
            r["cuts"]["res2b"],
            "((DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 11) || "
            "(DeepHME_mass_bin >= 16 && DeepHME_mass_bin < 21))",
        )

    def test_variables_are_anded(self):
        r = self.one(
            "    res2b: { DeepHME_mass: [ [ 250, 300 ], [ 350, 400 ] ], ll_mass: [ 10, 80 ] }\n"
        )
        self.assertEqual(
            r["cuts"]["res2b"],
            "((DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 11) || "
            "(DeepHME_mass_bin >= 16 && DeepHME_mass_bin < 21)) "
            "&& ll_mass_bin >= 2 && ll_mass_bin < 16",
        )
        self.assertEqual(r["columns"]["res2b"], ["DeepHME_mass_bin", "ll_mass_bin"])

    def test_the_whole_range_cuts_nothing(self):
        r = self.one(
            "    res2b: { ll_mass: [ 0, 200 ] }\n"
            "    boosted: { ll_mass: [ [ 0, 50 ], [ 0, 200 ] ], DeepHME_mass: [ 250, 350 ] }\n"
        )
        self.assertEqual(r["cuts"]["res2b"], "true")
        self.assertEqual(r["columns"]["res2b"], [])
        self.assertEqual(
            r["cuts"]["boosted"], "DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 16"
        )

    def test_edge_off_the_binning(self):
        with self.assertRaisesRegex(SR.SubRegionError, "255.0 is not a bin edge"):
            self.one("    res2b: { DeepHME_mass: [ 255, 350 ] }\n")

    def test_empty_interval(self):
        with self.assertRaisesRegex(SR.SubRegionError, "empty interval"):
            self.one("    res2b: { DeepHME_mass: [ 350, 250 ] }\n")

    def test_malformed_interval(self):
        with self.assertRaisesRegex(SR.SubRegionError, "expected \\[low, high\\]"):
            self.one("    res2b: { DeepHME_mass: [ 250, 300, 350 ] }\n")

    def test_variable_not_binned(self):
        flavour = dict(FLAVOUR, variables=["ll_mass", "DNN_M300_Signal"])
        self.write_cuts("cuts:\n  300:\n    res2b: { DeepHME_mass: [ 250, 350 ] }\n")
        with self.assertRaisesRegex(
            SR.SubRegionError, "DeepHME_mass is not a binned variable"
        ):
            self.expand([dict(ENTRY, masses=[300])], flavour=flavour)


class CppCuts(SubRegionsTestBase):
    def test_used_as_written_with_its_columns(self):
        r = self.one('    res2b: "DeepHME_mass >= 255.5 && ll_mass < 80"\n')
        self.assertEqual(r["cuts"]["res2b"], "DeepHME_mass >= 255.5 && ll_mass < 80")
        self.assertEqual(r["columns"]["res2b"], ["DeepHME_mass", "ll_mass"])

    def test_the_mass_placeholder(self):
        r = self.one(
            '    res2b: "DeepHME_mass > 0.5 * {MX} && DeepHME_mass < 1.4 * {MX}"\n'
        )
        self.assertEqual(
            r["cuts"]["res2b"], "DeepHME_mass > 0.5 * 300 && DeepHME_mass < 1.4 * 300"
        )

    def test_functions_literals_and_namespaces_are_not_variables(self):
        r = self.one(
            '    res2b: "std::abs(DeepHME_mass - 300.f) < 1e2 && abs(ll_mass) > .5 && true"\n'
        )
        self.assertEqual(r["columns"]["res2b"], ["DeepHME_mass", "ll_mass"])

    def test_a_variable_not_at_full_resolution_is_refused(self):
        flavour = dict(FLAVOUR, fullResolution_variables=["ll_mass"])
        self.write_cuts('cuts:\n  300:\n    res2b: "DeepHME_mass > 250"\n')
        with self.assertRaisesRegex(
            SR.SubRegionError,
            "uses DeepHME_mass, not in fullResolution_variables of flavour TestFlavour; add it "
            "there \\(it is not added automatically\\)",
        ):
            self.expand([dict(ENTRY, masses=[300])], flavour=flavour)

    def test_a_bin_column_or_a_flag_is_not_a_full_resolution_variable(self):
        for expr in ("DeepHME_mass_bin >= 6", "res2b && DeepHME_mass > 250"):
            with self.subTest(expr=expr):
                with self.assertRaisesRegex(
                    SR.SubRegionError, "not in fullResolution_variables"
                ):
                    self.one(f'    res2b: "{expr}"\n')

    def test_a_cut_without_variable(self):
        with self.assertRaisesRegex(SR.SubRegionError, "uses no variable"):
            self.one('    res2b: "1 > 0"\n')

    def test_both_forms_in_one_file(self):
        r = self.one(
            "    res2b: { DeepHME_mass: [ 250, 350 ] }\n"
            '    boosted: "DeepHME_mass >= 260 && DeepHME_mass < 470"\n'
        )
        self.assertEqual(
            r["cuts"],
            {
                "res2b": "DeepHME_mass_bin >= 6 && DeepHME_mass_bin < 16",
                "boosted": "DeepHME_mass >= 260 && DeepHME_mass < 470",
            },
        )


class Guards(SubRegionsTestBase):
    def refused(self, msg, **changes):
        with self.assertRaisesRegex(SR.SubRegionError, msg):
            self.expand([dict(ENTRY, **changes)])

    def test_unknown_region(self):
        self.refused(
            "region 'CR_X' is not among the flavour's regions", regions=["CR_X"]
        )

    def test_mass_without_cut(self):
        self.refused("mass 500 has no cut", masses=[500])

    def test_variable_not_of_the_flavour(self):
        self.refused("'mbb' is not a variable of the flavour", vars=["mbb"])

    def test_mass_tagged_variable_without_placeholder(self):
        self.refused("names mass 300 but has no \\{MX\\}", vars=["DNN_M300_Signal"])

    def test_unknown_placeholder(self):
        self.refused("unknown placeholder", vars=["DNN_M{MASS}_Signal"])

    def test_unknown_key(self):
        self.refused("unknown keys \\['mass'\\]", mass=300)

    def test_name_must_be_a_word(self):
        self.refused("name must be a word", name="HME Sub")

    def test_cuts_file_missing(self):
        self.refused("not found", cuts="config/none.yaml")

    def test_no_cuts_mapping(self):
        self.write_cuts("windows: {}\n")
        self.refused("has no 'cuts' mapping")

    def test_a_cut_is_a_dict_or_a_string(self):
        self.write_cuts("cuts:\n  300:\n    res2b: [ 250, 350 ]\n")
        self.refused(
            "a cut is a \\{variable: \\[low, high\\]\\} dict or a C\\+\\+ string"
        )

    def test_category_unknown(self):
        self.write_cuts("cuts:\n  300:\n    res3b: { DeepHME_mass: [ 250, 350 ] }\n")
        self.refused("category 'res3b'", masses=[300])

    def test_region_defined_twice(self):
        with self.assertRaisesRegex(SR.SubRegionError, "defined twice"):
            self.expand([ENTRY, ENTRY])


class JobBlocks(SubRegionsTestBase):
    def test_keys_per_base_key_with_a_cut(self):
        regions = self.expand([dict(ENTRY, masses=[300])])
        base_keys = [
            (("eE", "SR", "res2b"), "f0"),
            (("eE", "SR", "recovery"), "f1"),  # no cut in this category
            (("eE", "TT_CR", "res2b"), "f2"),  # region not listed
            (("muMu", "SR", "boosted"), "f3"),
        ]
        blocks = SR.sub_region_blocks(
            base_keys, regions, ["DeepHME_mass", "DNN_M300_Signal"]
        )
        self.assertEqual(len(blocks), 1)
        keys, block_vars = blocks[0]
        self.assertEqual(block_vars, ["DNN_M300_Signal"])
        self.assertEqual(
            [(k, i, cols) for k, i, _, cols in keys],
            [
                (("eE", "SR_HMESub_m300", "res2b"), 0, ["DeepHME_mass_bin"]),
                (("muMu", "SR_HMESub_m300", "boosted"), 3, ["DeepHME_mass_bin"]),
            ],
        )
        self.assertEqual(SR.count_sub_region_pairs(blocks), 2)

    def test_a_job_without_the_sub_region_variables_books_nothing(self):
        regions = self.expand([ENTRY])
        self.assertEqual(
            SR.sub_region_blocks([(("eE", "SR", "res2b"), "f")], regions, ["lep1_pt"]),
            [],
        )


class CppVariables(unittest.TestCase):
    def test_identifiers(self):
        self.assertEqual(
            SR.cpp_variables("x > 1.5e-3 && f(y) < 2.f && ROOT::VecOps::Max(z) || !w"),
            ["x", "y", "z", "w"],
        )


class Axis(unittest.TestCase):
    def test_uniform_and_listed_edges(self):
        self.assertEqual(SR.axis_edges("4|0:2"), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(SR.axis_edges([0, 1, 5]), [0.0, 1.0, 5.0])

    def test_an_interval_spanning_everything_is_no_cut(self):
        self.assertEqual(SR.bin_cut("v_bin", [0.0, 1.0, 2.0], 0, 2, "w"), "true")


class FlavourRegions(unittest.TestCase):
    """Setup._resolve_regions: the flavour's `regions`, else the list `custom_regions` names."""

    def resolve(self, flavour_cfg, global_params):
        try:
            from FLAF.Common.Setup import Setup
        except Exception as e:  # an environment without the analysis packages
            self.skipTest(f"Setup cannot be imported: {e}")
        stub = types.SimpleNamespace(global_params=global_params, histTuple_flavor="F")
        return Setup._resolve_regions(stub, flavour_cfg)

    def test_the_flavour_regions(self):
        self.assertEqual(
            self.resolve({"regions": ["SR", "TT_CR"]}, {}), ["SR", "TT_CR"]
        )

    def test_the_custom_regions_fallback(self):
        gp = {"custom_regions": "QCDRegions", "QCDRegions": ["OS_Iso", "SR"]}
        self.assertEqual(self.resolve({}, gp), ["OS_Iso", "SR"])

    def test_a_region_cut_mapping(self):
        gp = {"custom_regions": "Regions", "Regions": {"Z": "m > 70", "H": "m > 110"}}
        self.assertEqual(self.resolve({}, gp), {"Z": "m > 70", "H": "m > 110"})

    def test_both_is_an_error(self):
        gp = {"custom_regions": "QCDRegions", "QCDRegions": ["SR"]}
        with self.assertRaisesRegex(RuntimeError, "regions belong to one place"):
            self.resolve({"regions": ["SR"]}, gp)

    def test_none_at_all(self):
        self.assertEqual(self.resolve({}, {}), [])


if __name__ == "__main__":
    unittest.main()
