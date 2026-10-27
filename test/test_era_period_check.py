#!/usr/bin/env python3
"""The event-level era check of the merge, HistTuple and cache producers.

Every AnaTuple event carries its era in the column `period` (the C++ `Period` value). The check
counts, in the event loop the producer runs anyway, the events of another era, skipping the
placeholder rows (valid == false) whose columns are all zero, and stops the producer when it
finds any. Needs ROOT, so it runs on lxplus, not in the GitHub CI.
"""

import os
import sys
import unittest

import pytest

pytestmark = pytest.mark.needs_root

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import ROOT  # noqa: E402

from FLAF.Common.Setup import ERA_IDS  # noqa: E402
from FLAF.Common.Utilities import book_era_check, check_era  # noqa: E402


def frame(era_ids, valid=None):
    """An RDataFrame with one event per entry of `era_ids` (and of `valid`, when given)."""
    df = ROOT.RDataFrame(len(era_ids))
    values = ",".join(str(p) for p in era_ids)
    df = df.Define("period", f"std::vector<int> v{{{values}}}; return v[rdfentry_];")
    if valid is not None:
        flags = ",".join("true" if v else "false" for v in valid)
        df = df.Define(
            "valid", f"std::vector<bool> v{{{flags}}}; return bool(v[rdfentry_]);"
        )
    return df


def run_loop(df, foreign_count):
    """Run the event loop through another action, as the producers' snapshots do."""
    total = df.Count()
    total.GetValue()
    return foreign_count


class EraCheck(unittest.TestCase):
    def test_events_of_the_era_pass(self):
        df = frame([ERA_IDS["Run3_2022"]] * 3)
        check_era(run_loop(df, book_era_check(df, "Run3_2022")), "Run3_2022", "x")

    def test_an_event_of_another_era_stops_the_producer(self):
        df = frame([ERA_IDS["Run3_2022"], ERA_IDS["Run3_2022EE"]])
        count = run_loop(df, book_era_check(df, "Run3_2022"))
        with self.assertRaises(RuntimeError) as caught:
            check_era(count, "Run3_2022", "input.root")
        self.assertIn("input.root: 1 event(s)", str(caught.exception))
        self.assertIn("another era than Run3_2022", str(caught.exception))

    def test_placeholder_rows_are_not_counted(self):
        df = frame([ERA_IDS["Run3_2023"], 0], valid=[True, False])
        check_era(run_loop(df, book_era_check(df, "Run3_2023")), "Run3_2023", "x")

    def test_a_valid_row_of_another_era_is_counted(self):
        df = frame([ERA_IDS["Run3_2023"], ERA_IDS["Run3_2022"]], valid=[True, True])
        with self.assertRaises(RuntimeError):
            check_era(run_loop(df, book_era_check(df, "Run3_2023")), "Run3_2023", "x")

    def test_the_count_shares_the_event_loop(self):
        df = frame([ERA_IDS["Run3_2022"]] * 4)
        count = book_era_check(df, "Run3_2022")
        df.Count().GetValue()
        self.assertEqual(df.GetNRuns(), 1)
        check_era(count, "Run3_2022", "x")
        self.assertEqual(df.GetNRuns(), 1, "reading the count ran another event loop")


if __name__ == "__main__":
    unittest.main()
