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
    fill_groups,
)


def _cover(groups, var_cells, n_keys, n_slots=1):
    """Cells of every group, all slots' copies, and every (variable, weight) filled."""
    sizes, filled = [], []
    for group_vars, (first, end) in groups:
        filled.extend((v, w) for v in group_vars for w in range(first, end))
        sizes.append(
            sum(var_cells[v] for v in group_vars) * n_keys * (end - first) * n_slots
        )
    return sizes, filled


# The HH_bbWW CR layout of a HistFromNtuple job: DeepHME_mass (80 cells), 100 DNN 1D variables (22 cells)
# and 25 2D DNN_m*_vs_HME (8160 cells), 27 keys, 49 weights on the central tree.
CR_LAYOUT = {"DeepHME_mass": 80}
CR_LAYOUT.update({f"DNN_M{m}_Signal": 22 for m in range(100)})
CR_LAYOUT.update({f"DNN_m{m}_vs_HME": 8160 for m in range(25)})


class TestFillGroups(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(BYTES_PER_CELL, 16)
        self.assertEqual(cells_budget(1500), 1500 * 10**6 // 16)
        self.assertEqual(cells_budget(0), 0)
        self.assertEqual(cells_budget(None), 0)

    def test_budget_is_an_int_for_a_float(self):
        # --max-hist-memory-mb and the config value reach cells_budget as floats
        for mb in (100.0, 1500.0, 1.5):
            self.assertIsInstance(cells_budget(mb), int)
        # below 1 MB the cap is off
        self.assertEqual(cells_budget(0.5), 0)

    def test_a_small_budget_splits_a_cr_2d_variable_over_its_weights(self):
        var_cells = {"DNN_m300_vs_HME": 8160}
        groups = fill_groups(var_cells, 27, 49, cells_budget(100.0))
        self.assertEqual(
            groups, [(["DNN_m300_vs_HME"], (0, 28)), (["DNN_m300_vs_HME"], (28, 49))]
        )

    def test_the_budget_counts_every_slot(self):
        max_cells = cells_budget(1500.0)
        n_loops = {}
        for n_slots in (1, 2, 4, 8):
            groups = fill_groups(CR_LAYOUT, 27, 49, max_cells, n_slots)
            sizes, filled = _cover(groups, CR_LAYOUT, 27, n_slots)
            self.assertEqual(
                sorted(filled),
                sorted((v, w) for v in CR_LAYOUT for w in range(49)),
                n_slots,
            )
            self.assertTrue(all(s <= max_cells for s in sizes), (n_slots, sizes))
            n_loops[n_slots] = len(groups)
        # more slots, more copies: never fewer loops
        self.assertEqual(n_loops[1], len(fill_groups(CR_LAYOUT, 27, 49, max_cells)))
        self.assertTrue(n_loops[1] <= n_loops[2] <= n_loops[4] <= n_loops[8], n_loops)

    def test_the_open_group_shares_the_loops_of_a_variable_over_the_budget(self):
        # a 100 x 100 x 80 3D variable between 1D ones: its weight chunks take the 1D variables before it
        # instead of a loop of their own
        var_cells = {"a": 52, "b": 52, "c": 52, "big": 102 * 102 * 82, "d": 52}
        max_cells = cells_budget(1500.0)
        groups = fill_groups(var_cells, 27, 49, max_cells)
        big = [g for g in groups if "big" in g[0]]
        self.assertTrue(all(g[0] == ["a", "b", "c", "big"] for g in big), groups)
        self.assertEqual(groups[-1], (["d"], (0, 49)))
        self.assertEqual(len(groups), len(big) + 1)
        sizes, filled = _cover(groups, var_cells, 27)
        self.assertEqual(
            sorted(filled), sorted((v, w) for v in var_cells for w in range(49))
        )
        self.assertTrue(all(s <= max_cells for s in sizes))
        # the variables keep their order
        first_seen = list(dict.fromkeys(v for g in groups for v in g[0]))
        self.assertEqual(first_seen, list(var_cells))

    def test_no_budget_is_one_group(self):
        var_cells = {"a": 10, "b": 8160}
        self.assertEqual(fill_groups(var_cells, 27, 49, 0), [(["a", "b"], (0, 49))])

    def test_everything_fits(self):
        var_cells = {"a": 10, "b": 20}
        self.assertEqual(fill_groups(var_cells, 2, 3, 1000), [(["a", "b"], (0, 3))])

    def test_every_variable_and_weight_once_within_the_budget(self):
        var_cells = {
            f"v{i}": c for i, c in enumerate([102, 80, 8160, 8160, 50, 8160, 30])
        }
        n_keys, n_weights, max_cells = 27, 49, 3 * 8160 * 27 * 49
        groups = fill_groups(var_cells, n_keys, n_weights, max_cells)
        sizes, filled = _cover(groups, var_cells, n_keys)
        self.assertEqual(
            sorted(filled), sorted((v, w) for v in var_cells for w in range(n_weights))
        )
        self.assertTrue(all(s <= max_cells for s in sizes))
        # variables keep their order
        order = [v for group_vars, _ in groups for v in group_vars]
        self.assertEqual(order, list(var_cells))

    def test_a_variable_larger_than_the_budget_is_split_over_its_weights(self):
        var_cells = {"small": 10, "big": 1000, "after": 10}
        n_keys, n_weights = 2, 10
        max_cells = 1000 * n_keys * 3  # three weights of "big" per group
        groups = fill_groups(var_cells, n_keys, n_weights, max_cells)
        self.assertEqual(groups[0], (["small"], (0, 10)))
        self.assertEqual(
            [g for g in groups if g[0] == ["big"]],
            [
                (["big"], (0, 3)),
                (["big"], (3, 6)),
                (["big"], (6, 9)),
                (["big"], (9, 10)),
            ],
        )
        self.assertEqual(groups[-1], (["after"], (0, 10)))
        sizes, filled = _cover(groups, var_cells, n_keys)
        self.assertEqual(len(filled), len(set(filled)))
        self.assertEqual(len(filled), 3 * n_weights)
        self.assertTrue(all(s <= max_cells for s in sizes))

    def test_one_weight_above_the_budget_still_gets_a_group(self):
        groups = fill_groups({"huge": 10**6}, 4, 2, 100)
        self.assertEqual(groups, [(["huge"], (0, 1)), (["huge"], (1, 2))])

    def test_no_variables(self):
        self.assertEqual(fill_groups({}, 27, 49, 1000), [])


if __name__ == "__main__":
    unittest.main()
