"""Split the histograms of one HistFromNtuple event loop so that their arrays fit a memory budget.

HistProducerFromNTuple fills, per tree, every (selection key, weight, variable) histogram in one event
loop (include/HistFiller.h): ``n_keys x n_weights`` histograms per variable, each of the variable's cells
held as a sum and a sum of squared weights. The variables are packed in order into groups of at most
``max_cells`` cells in all; a variable that alone exceeds it is split over its weights. Each group is one
event loop over the tree.
"""

BYTES_PER_CELL = 16  # a cell: sum of weights and sum of squared weights, as doubles


def cells_budget(memory_mb):
    """Cells one event loop may hold within ``memory_mb``; ``memory_mb < 1`` disables the cap (0)."""
    if not memory_mb or memory_mb < 1:
        return 0
    return int(memory_mb * 1e6 / BYTES_PER_CELL)


def fill_groups(var_cells, n_keys, n_weights, max_cells):
    """``[(variables, (first_weight, end_weight))]``, the event loops of one tree.

    ``var_cells``: {variable: cells of one of its histograms}, in the order to fill them. Every
    (variable, weight) is in exactly one group, and a group holds at most ``max_cells`` cells unless one
    (variable, weight) alone does. ``max_cells < 1`` gives a single group.
    """
    variables = list(var_cells)
    if not variables:
        return []
    if not max_cells or max_cells < 1:
        return [(variables, (0, n_weights))]
    groups = []
    current, current_cells = [], 0
    for var in variables:
        per_weight = var_cells[var] * n_keys
        cells = per_weight * n_weights
        if cells > max_cells:
            if current:
                groups.append((current, (0, n_weights)))
                current, current_cells = [], 0
            step = max(1, max_cells // per_weight)
            for first in range(0, n_weights, step):
                groups.append(([var], (first, min(first + step, n_weights))))
            continue
        if current and current_cells + cells > max_cells:
            groups.append((current, (0, n_weights)))
            current, current_cells = [], 0
        current.append(var)
        current_cells += cells
    if current:
        groups.append((current, (0, n_weights)))
    return groups
