import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Analysis.histFromNtupleBatch import (
    BYTES_PER_BOOKED_HIST,
    BYTES_PER_CELL,
    count_booked_hists,
    hist_bytes,
    iter_hist_batches,
    memory_budget,
    n_cut_slots,
    unc_scale_pairs,
)


def _cover(batches, var_bytes):
    """(histograms, bytes) of every batch, and every (var, key, cut, unc, scale) booked."""
    sizes, booked = [], []
    for b_uncs, b_keys, b_cuts, b_vars in batches:
        cuts = list(b_cuts) or [None]
        pairs = unc_scale_pairs(b_uncs)
        items = [
            (v, k, c, p) for v in b_vars for k in b_keys for c in cuts for p in pairs
        ]
        booked.extend(items)
        sizes.append((len(items), sum(var_bytes.get(v, 1) for v, *_ in items)))
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


class TestHistFromNtupleMemoryBudget(unittest.TestCase):
    """The memory budget: batches of mixed 1D / 2D variables within both limits."""

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
    # one thread slot: 1D 7.6 kB / 6.8 kB per histogram, 2D 136.6 kB
    BYTES = {var: hist_bytes(cells, 1) for var, cells in CELLS.items()}

    def test_hist_bytes_and_budget(self):
        # one copy per thread slot (none extra for the merge), plus the fixed cost per action
        self.assertEqual(
            hist_bytes(8160, 1), BYTES_PER_BOOKED_HIST + 8160 * BYTES_PER_CELL
        )
        self.assertEqual(
            hist_bytes(8160, 4), BYTES_PER_BOOKED_HIST + 4 * 8160 * BYTES_PER_CELL
        )
        self.assertEqual(hist_bytes(100, 0), hist_bytes(100, 1))
        self.assertEqual(memory_budget(1500), 1_500_000_000)
        self.assertEqual(memory_budget(0), 0)
        self.assertEqual(memory_budget(None), 0)

    def test_without_a_memory_budget_the_batches_are_the_counts(self):
        # literal batches: 45 histograms per variable, so 2 variables per 100
        expected = {
            0: [self.VARS],
            100: [["pt1", "pt2"], ["m_2d_a", "eta"], ["m_2d_b", "m_2d_c"]],
            4000: [self.VARS],
        }
        for max_hists, var_lists in expected.items():
            for kwargs in ({}, {"var_bytes": self.BYTES, "max_bytes": 0}):
                batches = list(
                    iter_hist_batches(
                        self.UNCS, self.KEYS, {}, self.VARS, max_hists, **kwargs
                    )
                )
                self.assertEqual(
                    [b[3] for b in batches], var_lists, (max_hists, kwargs)
                )
                for b_uncs, b_keys, b_cuts, _ in batches:
                    self.assertEqual(
                        (b_uncs, b_keys, b_cuts), (self.UNCS, self.KEYS, {})
                    )

    def test_every_histogram_once_within_both_budgets(self):
        n_total = len(self.VARS) * len(self.KEYS) * 5
        for max_hists, max_bytes in (
            (4000, 13_000_000),
            (100, 13_000_000),
            (4000, 2_000_000),
            (30, 300_000),
            (0, 4_000_000),
        ):
            batches = list(
                iter_hist_batches(
                    self.UNCS,
                    self.KEYS,
                    {},
                    self.VARS,
                    max_hists,
                    var_bytes=self.BYTES,
                    max_bytes=max_bytes,
                )
            )
            sizes, booked = _cover(batches, self.BYTES)
            self.assertEqual(len(booked), n_total, (max_hists, max_bytes))
            self.assertEqual(len(set(booked)), n_total, (max_hists, max_bytes))
            for n, size in sizes:
                if max_hists:
                    self.assertLessEqual(n, max_hists, (max_hists, max_bytes))
                # a single histogram larger than the budget is the one allowed overshoot
                self.assertTrue(
                    size <= max_bytes or n == 1, (max_hists, max_bytes, n, size)
                )

    def test_largest_variables_are_packed_first(self):
        # in input order: d1..d8 (8 MB), then m2a (6 MB) and m2b (6 MB) each start a pass -> 3
        # passes; first-fit-decreasing puts each 2D with four 1D: 2 passes of 10 MB
        uncs, keys = {"Central": ["Central"]}, {("e",): "e"}
        var_bytes = {f"d{i}": 1_000_000 for i in range(1, 9)}
        var_bytes.update({"m2a": 6_000_000, "m2b": 6_000_000})
        batches = list(
            iter_hist_batches(
                uncs,
                keys,
                {},
                list(var_bytes),
                0,
                var_bytes=var_bytes,
                max_bytes=10_000_000,
            )
        )
        self.assertEqual(
            [b[3] for b in batches],
            [["d1", "d2", "d3", "d4", "m2a"], ["d5", "d6", "d7", "d8", "m2b"]],
        )

    def test_over_budget_variable_is_chunked_by_what_fits(self):
        # one key, 21 (unc, scale) pairs of 1 MB, 5 MB per batch: 5 batches, not 21
        uncs = {"Central": ["Central"]}
        uncs.update({f"u{i}": ["Up", "Down"] for i in range(10)})
        batches = list(
            iter_hist_batches(
                uncs,
                {("e",): "e"},
                {},
                ["v"],
                0,
                var_bytes={"v": 1_000_000},
                max_bytes=5_000_000,
            )
        )
        self.assertEqual([len(unc_scale_pairs(b[0])) for b in batches], [5, 5, 5, 5, 1])
        sizes, booked = _cover(batches, {"v": 1_000_000})
        self.assertEqual(len(set(booked)), 21)

    def test_further_cuts_count_in_the_memory(self):
        # with 2 cuts a 2D variable is 90 histograms, 12.3 MB: one per 13 MB batch. Without the
        # cut factor two would be packed together, 24.6 MB.
        cuts = {"cutA": ("a", "a"), "cutB": ("b", "b")}
        batches = list(
            iter_hist_batches(
                self.UNCS,
                self.KEYS,
                cuts,
                ["m_2d_a", "m_2d_b", "pt1"],
                0,
                var_bytes=self.BYTES,
                max_bytes=13_000_000,
            )
        )
        sizes, booked = _cover(batches, self.BYTES)
        self.assertEqual(len(set(booked)), 3 * 9 * 2 * 5)
        self.assertTrue(all(size <= 13_000_000 for _, size in sizes), sizes)
        self.assertEqual([b[3] for b in batches], [["m_2d_a", "pt1"], ["m_2d_b"]])

    def test_empty_variable_list(self):
        batches = list(
            iter_hist_batches(
                self.UNCS, self.KEYS, {}, [], 4000, var_bytes=self.BYTES, max_bytes=1000
            )
        )
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][3], [])


if __name__ == "__main__":
    unittest.main()
