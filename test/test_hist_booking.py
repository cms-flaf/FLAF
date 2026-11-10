#!/usr/bin/env python3
"""Common/HistBooking.py: loading an analysis' booking plugin, its regions, and the checks of the
booking blocks it gives a HistFromNtuple job. Plain Python, with a stand-in plugin module; FillTree with
several blocks is checked in test_hist_filler.py."""

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


def plugin(regions=None, booking=None, **extra):
    module = types.ModuleType(PLUGIN)
    if regions is not None:
        module.regions = regions
    if booking is not None:
        module.booking = booking
    for name, value in extra.items():
        setattr(module, name, value)
    return module


class PluginTestBase(unittest.TestCase):
    def install(self, module):
        sys.modules[PLUGIN] = module
        self.addCleanup(sys.modules.pop, PLUGIN, None)
        return HB.load_plugin({"hist_booking_module": PLUGIN})

    def blocks(self, raw):
        module = self.install(plugin(regions=lambda s: [], booking=lambda **kw: raw))
        return HB.plugin_blocks(module, None, KEYS, VARS)


class Loading(PluginTestBase):
    def test_no_plugin(self):
        self.assertIsNone(HB.load_plugin({}))
        self.assertIsNone(HB.load_plugin({"hist_booking_module": None}))

    def test_a_plugin(self):
        module = self.install(plugin(regions=lambda s: ["SR"], booking=lambda **kw: []))
        self.assertIs(module, sys.modules[PLUGIN])

    def test_a_module_that_cannot_be_imported(self):
        with self.assertRaisesRegex(HB.HistBookingError, "cannot import it"):
            HB.load_plugin({"hist_booking_module": "no_such_booking_module_xyz"})

    def test_a_module_without_the_functions(self):
        with self.assertRaisesRegex(HB.HistBookingError, "defines no booking"):
            self.install(plugin(regions=lambda s: []))

    def test_a_name_that_is_not_a_string(self):
        with self.assertRaisesRegex(HB.HistBookingError, "must be a module name"):
            HB.load_plugin({"hist_booking_module": ["a"]})


class Regions(PluginTestBase):
    def test_a_list_or_a_mapping(self):
        for value in (["SR", "TT_CR"], {"Z": "m > 70"}):
            module = self.install(
                plugin(regions=lambda s, v=value: v, booking=lambda **kw: [])
            )
            self.assertEqual(HB.plugin_regions(module, None), value)

    def test_anything_else_is_an_error(self):
        module = self.install(plugin(regions=lambda s: "SR", booking=lambda **kw: []))
        with self.assertRaisesRegex(HB.HistBookingError, "list of region names"):
            HB.plugin_regions(module, None)


class Blocks(PluginTestBase):
    def test_the_default_booking(self):
        self.assertEqual(HB.default_blocks(KEYS, VARS), [(KEYS, VARS, [])])
        self.assertEqual(HB.count_pairs(HB.default_blocks(KEYS, VARS)), 4)

    def test_blocks_as_lists_or_mappings_of_keys(self):
        blocks = self.blocks(
            [
                {"keys": KEYS[:1], "variables": ["x", "y"]},
                {
                    "keys": {("eE", "SR_sub", "res2b"): "f0 && c"},
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

    def test_a_block_without_keys_is_dropped(self):
        blocks = self.blocks(
            [{"keys": KEYS, "variables": VARS}, {"keys": [], "variables": ["x"]}]
        )
        self.assertEqual(len(blocks), 1)

    def refused(self, msg, raw):
        with self.assertRaisesRegex(HB.HistBookingError, msg):
            self.blocks(raw)

    def test_not_a_list(self):
        self.refused("expected a list of blocks", {"keys": KEYS})

    def test_a_block_is_a_mapping_with_keys_and_variables(self):
        self.refused("a block is a mapping", [KEYS])
        self.refused("needs 'keys' and 'variables'", [{"keys": KEYS}])
        self.refused(
            "needs 'keys' and 'variables'", [{"keys": KEYS, "variables": VARS, "x": 1}]
        )

    def test_a_key_is_a_tuple_and_a_filter(self):
        self.refused(
            "a key is \\(key_tuple, filter\\)",
            [{"keys": [KEYS[0][0]], "variables": VARS}],
        )
        self.refused(
            "a key tuple is a tuple of names",
            [{"keys": [(("eE", ""), "f")], "variables": VARS}],
        )

    def test_unknown_variables(self):
        self.refused(
            "\\['z'\\] are not variables of the job",
            [{"keys": KEYS, "variables": ["z"]}],
        )

    def test_every_variable_needs_a_block(self):
        self.refused("\\['y'\\] booked in no key", [{"keys": KEYS, "variables": ["x"]}])

    def test_a_pair_booked_twice(self):
        self.refused(
            "x is booked twice in eE/SR/res2b",
            [{"keys": KEYS, "variables": VARS}, {"keys": KEYS[:1], "variables": ["x"]}],
        )

    def test_columns_are_names(self):
        self.refused(
            "'columns' must be a list of names",
            [{"keys": KEYS, "variables": VARS, "columns": "c"}],
        )

    def test_the_plugin_gets_the_job(self):
        seen = {}

        def booking(setup, keys, variables):
            seen.update(setup=setup, keys=keys, variables=variables)
            return [{"keys": keys, "variables": variables}]

        module = self.install(plugin(regions=lambda s: [], booking=booking))
        HB.plugin_blocks(module, "the setup", KEYS, VARS)
        self.assertEqual(seen, {"setup": "the setup", "keys": KEYS, "variables": VARS})


if __name__ == "__main__":
    unittest.main()
