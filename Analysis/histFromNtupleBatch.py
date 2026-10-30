"""Split the histograms of one HistFromNtuple event loop so that their arrays fit a memory budget.

HistProducerFromNTuple fills, per tree, every (selection key, weight, variable) histogram in one event
loop (include/HistFiller.h): ``n_keys x n_weights`` histograms per variable, each of the variable's cells
held as a sum and a sum of squared weights, and every thread slot of the loop fills its own copy of them.
The variables are packed in order into groups of at most ``max_cells`` cells in all, every slot's copy
counted; a variable that alone exceeds it is split over its weights, and the variables of the open group
join its weight chunks when that needs no more loops. Each group is one event loop over the tree.
"""

BYTES_PER_CELL = 16  # a cell: sum of weights and sum of squared weights, as doubles


def cells_budget(memory_mb):
    """Cells one event loop may hold within ``memory_mb``, an int; ``memory_mb < 1`` disables the cap (0)."""
    if not memory_mb or memory_mb < 1:
        return 0
    return int(memory_mb * 1e6 / BYTES_PER_CELL)


def _weight_chunks(n_weights, step):
    return [
        (first, min(first + step, n_weights)) for first in range(0, n_weights, step)
    ]


def fill_groups(var_cells, n_keys, n_weights, max_cells, n_slots=1):
    """``[(variables, (first_weight, end_weight))]``, the event loops of one tree.

    ``var_cells``: {variable: cells of one of its histograms}, in the order to fill them; ``n_slots``:
    the thread slots of the loop, one copy of its cells each. Every (variable, weight) is in exactly one
    group, and a group holds at most ``max_cells`` cells in its ``n_slots`` copies unless one (variable,
    weight) alone does. ``max_cells < 1`` gives a single group. The variables keep their order.
    """
    variables = list(var_cells)
    if not variables:
        return []
    if not max_cells or max_cells < 1:
        return [(variables, (0, n_weights))]
    n_slots = max(1, int(n_slots))
    groups = []
    current, current_cells = [], 0  # current_cells: per weight
    for var in variables:
        per_weight = var_cells[var] * n_keys * n_slots
        if per_weight * n_weights > max_cells:
            chunks = _weight_chunks(n_weights, max(1, max_cells // per_weight))
            if current:
                joint_step = max_cells // (per_weight + current_cells)
                joint = _weight_chunks(n_weights, joint_step) if joint_step else None
                if joint is not None and len(joint) <= len(chunks):
                    # the open group shares the variable's loops instead of a loop of its own
                    groups.extend((current + [var], chunk) for chunk in joint)
                    current, current_cells = [], 0
                    continue
                groups.append((current, (0, n_weights)))
                current, current_cells = [], 0
            groups.extend(([var], chunk) for chunk in chunks)
            continue
        if current and (current_cells + per_weight) * n_weights > max_cells:
            groups.append((current, (0, n_weights)))
            current, current_cells = [], 0
        current.append(var)
        current_cells += per_weight
    if current:
        groups.append((current, (0, n_weights)))
    return groups
