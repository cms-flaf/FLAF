import os
import sys
import unittest

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

from FLAF.Analysis.histFromNtupleBatch import (
    count_booked_hists,
    iter_hist_batches,
    n_cut_slots,
    unc_scale_pairs,
)


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


if __name__ == "__main__":
    unittest.main()
