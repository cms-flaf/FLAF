#!/usr/bin/env python3
"""Common/HistBooking.py: loading an analysis' booking plugin, the regions it fills in, and the checks of
the blocks it gives a HistFromNtuple job. Plain Python with a stand-in plugin; FillTree with several blocks
is checked in test_hist_filler.py."""

import os
import sys
import types
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Common import HistBooking as HB  # noqa: E402

PLUGIN = "hist_booking_test_plugin"
KEYS = [(("eE", "SR", "res2b"), "f0"), (("eE", "SR", "boosted"), "f1")]
VARS = ["x", "y"]


class PluginTestBase(unittest.TestCase):
    def install(self, regions=lambda setup: ["SR"], booking=None):
        module = types.ModuleType(PLUGIN)
        module.regions = regions
        if booking is not None:
            module.booking = booking
        sys.modules[PLUGIN] = module
        self.addCleanup(sys.modules.pop, PLUGIN, None)
        return module

    def blocks(self, raw):
        module = self.install(booking=lambda **kw: raw)
        return HB.plugin_blocks(module, None, KEYS, VARS)


class Loading(PluginTestBase):
    def test_no_plugin(self):
        self.assertIsNone(HB.load_plugin({}))

    def test_a_plugin(self):
        module = self.install(booking=lambda **kw: [])
        self.assertIs(HB.load_plugin({"hist_booking_module": PLUGIN}), module)

    def test_a_plugin_without_booking(self):
        self.install()
        with self.assertRaisesRegex(HB.HistBookingError, "no booking"):
            HB.load_plugin({"hist_booking_module": PLUGIN})


class Regions(PluginTestBase):
    def setup_with(self, **gp):
        return types.SimpleNamespace(global_params=dict(gp))

    def test_the_list_custom_regions_names_is_filled(self):
        module = self.install(
            regions=lambda s: ["SR", "TT_CR"], booking=lambda **kw: []
        )
        setup = self.setup_with(custom_regions="QCDRegions", QCDRegions=["old"])
        HB.fill_regions(module, setup)
        self.assertEqual(setup.global_params["QCDRegions"], ["SR", "TT_CR"])

    def test_without_custom_regions(self):
        module = self.install(
            regions=lambda s: {"Z": "m > 70"}, booking=lambda **kw: []
        )
        setup = self.setup_with()
        HB.fill_regions(module, setup)
        self.assertEqual(setup.global_params["custom_regions"], "histTuple_regions")
        self.assertEqual(setup.global_params["histTuple_regions"], {"Z": "m > 70"})

    def test_anything_else_is_an_error(self):
        module = self.install(regions=lambda s: "SR", booking=lambda **kw: [])
        with self.assertRaisesRegex(HB.HistBookingError, "expected a list"):
            HB.fill_regions(module, self.setup_with())


class Blocks(PluginTestBase):
    def test_the_default_booking(self):
        self.assertEqual(HB.default_blocks(KEYS, VARS), [(KEYS, VARS, [])])
        self.assertEqual(HB.count_pairs(HB.default_blocks(KEYS, VARS)), 4)

    def test_the_plugin_blocks(self):
        blocks = self.blocks(
            [
                {"keys": KEYS[:1], "variables": ["x", "y"]},
                {
                    "keys": [(("eE", "SR_sub", "res2b"), "f0 && c")],
                    "variables": ["x"],
                    "columns": ["c_bin"],
                },
            ]
        )
        self.assertEqual(
            blocks,
            [
                ([KEYS[0]], ["x", "y"], []),
                ([(("eE", "SR_sub", "res2b"), "f0 && c")], ["x"], ["c_bin"]),
            ],
        )
        self.assertEqual(HB.count_pairs(blocks), 3)

    def test_unknown_variables(self):
        with self.assertRaisesRegex(HB.HistBookingError, "\\['z'\\] are not variables"):
            self.blocks([{"keys": KEYS, "variables": ["x", "y", "z"]}])

    def test_every_variable_needs_a_key(self):
        with self.assertRaisesRegex(
            HB.HistBookingError, "\\['y'\\] are booked in no key"
        ):
            self.blocks([{"keys": KEYS, "variables": ["x"]}])

    def test_a_pair_booked_twice(self):
        with self.assertRaisesRegex(
            HB.HistBookingError, "x booked twice in eE/SR/res2b"
        ):
            self.blocks(
                [
                    {"keys": KEYS, "variables": VARS},
                    {"keys": KEYS[:1], "variables": ["x"]},
                ]
            )

    def test_malformed_blocks(self):
        for raw in (
            [{"variables": VARS}],
            [{"keys": KEYS}],
            [{"keys": [("eE", "SR", "res2b")], "variables": VARS}],
            ["not a block"],
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    HB.HistBookingError, f"{PLUGIN}.booking: expected blocks"
                ):
                    self.blocks(raw)

    def test_the_plugin_gets_the_job(self):
        seen = {}

        def booking(setup, keys, variables):
            seen.update(setup=setup, keys=keys, variables=variables)
            return [{"keys": keys, "variables": variables}]

        module = self.install(booking=booking)
        HB.plugin_blocks(module, "the setup", KEYS, VARS)
        self.assertEqual(seen, {"setup": "the setup", "keys": KEYS, "variables": VARS})


if __name__ == "__main__":
    unittest.main()
