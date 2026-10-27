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


def _cover(groups, var_cells, n_keys):
    """Cells of every group, and every (variable, weight) filled."""
    sizes, filled = [], []
    for group_vars, (first, end) in groups:
        filled.extend((v, w) for v in group_vars for w in range(first, end))
        sizes.append(sum(var_cells[v] for v in group_vars) * n_keys * (end - first))
    return sizes, filled


class TestFillGroups(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(BYTES_PER_CELL, 16)
        self.assertEqual(cells_budget(1500), 1500 * 10**6 // 16)
        self.assertEqual(cells_budget(0), 0)
        self.assertEqual(cells_budget(None), 0)

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
