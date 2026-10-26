import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Analysis.histFromNtupleBatch import (
    BYTES_PER_CELL,
    cells_budget,
    count_booked_hists,
    iter_hist_batches,
    n_cut_slots,
    unc_scale_pairs,
)


def _cover(batches, var_cells):
    """(histograms, cells) of every batch, and every (var, key, cut, unc, scale) booked."""
    sizes, booked = [], []
    for b_uncs, b_keys, b_cuts, b_vars in batches:
        cuts = list(b_cuts) or [None]
        pairs = unc_scale_pairs(b_uncs)
        items = [
            (v, k, c, p) for v in b_vars for k in b_keys for c in cuts for p in pairs
        ]
        booked.extend(items)
        sizes.append((len(items), sum(var_cells.get(v, 1) for v, *_ in items)))
    return sizes, booked


class TestHistFromNtupleBatch(unittest.TestCase):
    def test_count_includes_up_down(self):
        uncs = {
            "Central": ["Central"],
            "JER": ["Up", "Down"],
            "JES_Total": ["Up", "Down"],
        }
        self.assertEqual(len(unc_scale_pairs(uncs)), 5)
        # 10 vars × 4 keys × 1 cut × 5 scales
        self.assertEqual(count_booked_hists(10, 4, 1, 5), 200)
        self.assertEqual(n_cut_slots({}), 1)
        self.assertEqual(n_cut_slots({"a": 1, "b": 2}), 2)

    def test_under_threshold_is_one_batch(self):
        uncs = {"Central": ["Central"], "JER": ["Up", "Down"]}
        keys = {("e", "SR"): "e && SR", ("mu", "SR"): "mu && SR"}
        vars_ = ["lep1_pt", "lep1_eta"]
        batches = list(iter_hist_batches(uncs, keys, {}, vars_, max_hists=100))
        self.assertEqual(len(batches), 1)
        b_uncs, b_keys, b_cuts, b_vars = batches[0]
        self.assertEqual(b_uncs, uncs)
        self.assertEqual(b_keys, keys)
        self.assertEqual(b_vars, vars_)

    def test_disabled_threshold_is_one_batch(self):
        uncs = {"Central": ["Central"]}
        keys = {("e",): "e"}
        vars_ = [f"v{i}" for i in range(50)]
        batches = list(iter_hist_batches(uncs, keys, {}, vars_, max_hists=0))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][3], vars_)

    def test_split_by_variables(self):
        uncs = {"Central": ["Central"], "JER": ["Up", "Down"]}  # 3 scales
        keys = {("e",): "e", ("mu",): "mu"}  # 2 keys
        # 3 × 2 × 1 = 6 hists/var; max 10 → 1 var per batch
        vars_ = ["a", "b", "c"]
        batches = list(iter_hist_batches(uncs, keys, {}, vars_, max_hists=10))
        self.assertEqual(len(batches), 3)
        seen = []
        for b_uncs, b_keys, _, b_vars in batches:
            self.assertEqual(b_uncs, uncs)
            self.assertEqual(set(b_keys), set(keys))
            self.assertEqual(len(b_vars), 1)
            seen.extend(b_vars)
        self.assertEqual(seen, vars_)

    def test_one_var_over_budget_splits_keys(self):
        uncs = {"Central": ["Central"]}
        keys = {("k%d" % i,): "c" for i in range(10)}
        vars_ = ["only"]
        # 1 × 10 × 1 × 1 = 10; max 3 → 4 key-batches (3+3+3+1)
        batches = list(iter_hist_batches(uncs, keys, {}, vars_, max_hists=3))
        self.assertGreater(len(batches), 1)
        seen_keys = []
        for _, b_keys, _, b_vars in batches:
            self.assertEqual(b_vars, ["only"])
            self.assertLessEqual(len(b_keys), 3)
            seen_keys.extend(b_keys)
        self.assertEqual(sorted(seen_keys), sorted(keys))

    def test_each_batch_respects_budget(self):
        uncs = {
            "Central": ["Central"],
            "JER": ["Up", "Down"],
            "EleES": ["Up", "Down"],
        }  # 5
        keys = {("c%d" % i,): "x" for i in range(8)}  # 8
        cuts = {"cutA": "a", "cutB": "b"}  # 2
        vars_ = ["v1", "v2", "v3"]  # 3
        # total = 3*8*2*5 = 240
        max_hists = 20
        batches = list(iter_hist_batches(uncs, keys, cuts, vars_, max_hists))
        self.assertGreater(len(batches), 1)
        covered = 0
        for b_uncs, b_keys, b_cuts, b_vars in batches:
            n = count_booked_hists(
                len(b_vars),
                max(1, len(b_keys)),
                n_cut_slots(b_cuts),
                max(1, len(unc_scale_pairs(b_uncs))),
            )
            self.assertLessEqual(n, max_hists)
            covered += n
        self.assertEqual(covered, 240)


class TestHistFromNtupleCellBudget(unittest.TestCase):
    """The cell budget: batches of mixed 1D / 2D variables within both limits."""

    UNCS = {"Central": ["Central"], "JER": ["Up", "Down"], "PU": ["Up", "Down"]}  # 5
    KEYS = {(f"ch{i}", "SR", f"cat{j}"): "x" for i in range(3) for j in range(3)}  # 9
    VARS = ["pt1", "pt2", "m_2d_a", "eta", "m_2d_b", "m_2d_c"]
    CELLS = {
        "pt1": 102,
        "pt2": 102,
        "eta": 52,
        "m_2d_a": 8160,
        "m_2d_b": 8160,
        "m_2d_c": 8160,
    }

    def test_cells_budget(self):
        # 1500 MB over 4 threads + the merged copy, 16 B per cell
        self.assertEqual(cells_budget(1500, 4), int(1500e6 / (BYTES_PER_CELL * 5)))
        self.assertEqual(cells_budget(0, 4), 0)
        self.assertEqual(cells_budget(None, 4), 0)
        self.assertLess(cells_budget(1500, 8), cells_budget(1500, 4))

    def test_no_cells_is_the_count_alone(self):
        for max_hists in (0, 20, 100, 4000):
            old = list(
                iter_hist_batches(self.UNCS, self.KEYS, {}, self.VARS, max_hists)
            )
            new = list(
                iter_hist_batches(
                    self.UNCS,
                    self.KEYS,
                    {},
                    self.VARS,
                    max_hists,
                    var_cells=None,
                    max_cells=0,
                )
            )
            self.assertEqual(old, new, max_hists)

    def test_every_histogram_once_within_both_budgets(self):
        n_total = len(self.VARS) * len(self.KEYS) * 5
        for max_hists, max_cells in (
            (4000, 400_000),
            (100, 400_000),
            (4000, 50_000),
            (30, 9_000),
            (0, 100_000),
        ):
            batches = list(
                iter_hist_batches(
                    self.UNCS,
                    self.KEYS,
                    {},
                    self.VARS,
                    max_hists,
                    var_cells=self.CELLS,
                    max_cells=max_cells,
                )
            )
            sizes, booked = _cover(batches, self.CELLS)
            self.assertEqual(len(booked), n_total, (max_hists, max_cells))
            self.assertEqual(len(set(booked)), n_total, (max_hists, max_cells))
            for n, cells in sizes:
                if max_hists:
                    self.assertLessEqual(n, max_hists, (max_hists, max_cells))
                # a single histogram larger than the budget is the one allowed overshoot
                self.assertTrue(
                    cells <= max_cells or n == 1, (max_hists, max_cells, n, cells)
                )

    def test_2d_variables_get_smaller_batches(self):
        # 45 histograms per variable: 1D 2.3-4.6k cells, 2D 367k cells. Within 400k cells the
        # first 2D variable still fits with the 1D ones around it; each further one needs its own.
        # The count (4000) alone would book all six in one pass.
        batches = list(
            iter_hist_batches(
                self.UNCS,
                self.KEYS,
                {},
                self.VARS,
                4000,
                var_cells=self.CELLS,
                max_cells=400_000,
            )
        )
        self.assertEqual(
            [b[3] for b in batches],
            [["pt1", "pt2", "m_2d_a", "eta"], ["m_2d_b"], ["m_2d_c"]],
        )
        self.assertEqual(
            len(list(iter_hist_batches(self.UNCS, self.KEYS, {}, self.VARS, 4000))), 1
        )

    def test_2d_variable_over_budget_is_split(self):
        # one 2D variable is 45 x 8160 = 367k cells; 100k cells fit 12 of its histograms per batch
        batches = list(
            iter_hist_batches(
                self.UNCS,
                self.KEYS,
                {},
                ["m_2d_a"],
                4000,
                var_cells=self.CELLS,
                max_cells=100_000,
            )
        )
        sizes, booked = _cover(batches, self.CELLS)
        self.assertGreater(len(batches), 1)
        self.assertEqual(len(set(booked)), 45)
        self.assertTrue(all(cells <= 100_000 for _, cells in sizes), sizes)

    def test_empty_variable_list(self):
        batches = list(
            iter_hist_batches(
                self.UNCS, self.KEYS, {}, [], 4000, var_cells=self.CELLS, max_cells=1000
            )
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][3], [])


if __name__ == "__main__":
    unittest.main()
