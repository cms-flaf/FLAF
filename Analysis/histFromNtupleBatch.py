"""Split HistFromNtuple histogram booking so one RDataFrame pass stays within a hist budget.

The booked-histogram count is
``n_vars × n_keys × n_cuts × n_scale_slots``, where ``n_scale_slots`` counts
every (uncertainty, scale) pair (Central plus each Up/Down). When that product
exceeds ``max_hists``, the work is partitioned so each batch books at most
``max_hists`` histograms. Axes are split in this order: variables, then extra
cuts, then selection keys, then (unc, scale).

A count alone does not bound memory: RDataFrame keeps a copy of every booked histogram per
thread, and a 2D histogram can have a hundred times the cells of a 1D one. Given each
variable's cell count (``var_cells``), ``max_cells`` additionally caps the cells booked in one
batch (summed over its histograms); see ``cells_budget`` for the budget from a memory size.
"""

BYTES_PER_CELL = (
    16  # a TH1D/TH2D/TH3D cell: content + sum of weights squared, as doubles
)


def cells_budget(memory_mb, n_threads):
    """Cells one batch may book so its histograms stay within ``memory_mb``.

    RDataFrame books one copy of each histogram per thread and merges them into one more, so a
    batch of C cells holds about C x BYTES_PER_CELL x (n_threads + 1) bytes. ``memory_mb < 1``
    disables the cap (returns 0).
    """
    if not memory_mb or memory_mb < 1:
        return 0
    return int(memory_mb * 1e6 / (BYTES_PER_CELL * (max(1, n_threads) + 1)))


def n_cut_slots(further_cuts):
    return max(1, len(further_cuts) if further_cuts else 0)


def unc_scale_pairs(uncs_to_compute):
    pairs = []
    for unc, scales in uncs_to_compute.items():
        for scale in scales:
            pairs.append((unc, scale))
    return pairs


def pairs_to_uncs(pairs):
    out = {}
    for unc, scale in pairs:
        out.setdefault(unc, []).append(scale)
    return out


def count_booked_hists(n_vars, n_keys, n_cuts, n_scales):
    return n_vars * n_keys * n_cuts * n_scales


def _chunks(items, size):
    size = max(1, int(size))
    if not items:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


def iter_hist_batches(
    uncs_to_compute,
    key_filter_dict,
    further_cuts,
    vars_to_process,
    max_hists,
    var_cells=None,
    max_cells=0,
):
    """Yield ``(uncs_dict, keys_dict, cuts_dict, vars_list)`` batches.

    Each batch books at most ``max_hists`` histograms. ``max_hists < 1`` disables
    splitting (one batch with everything). With ``var_cells`` ({variable: cells of one of its
    histograms}) and ``max_cells >= 1``, each batch also books at most ``max_cells`` cells;
    without them the batches are exactly those of the count alone.
    """
    pairs = unc_scale_pairs(uncs_to_compute)
    keys = list(key_filter_dict.items())
    cuts = list(further_cuts.items()) if further_cuts else []
    vars_list = list(vars_to_process)
    if var_cells and max_cells and max_cells >= 1:
        for b_pairs, b_keys, b_cuts, b_vars in _split_by_cells(
            pairs, keys, cuts, vars_list, max_hists, var_cells, max_cells
        ):
            yield pairs_to_uncs(b_pairs), dict(b_keys), dict(b_cuts), b_vars
        return
    n_total = count_booked_hists(
        max(1, len(vars_list)),
        max(1, len(keys)),
        n_cut_slots(further_cuts),
        max(1, len(pairs)),
    )
    if max_hists < 1 or n_total <= max_hists:
        yield uncs_to_compute, key_filter_dict, further_cuts, vars_list
        return

    for b_pairs, b_keys, b_cuts, b_vars in _split_axes(
        pairs, keys, cuts, vars_list, max_hists
    ):
        yield pairs_to_uncs(b_pairs), dict(b_keys), dict(b_cuts), b_vars


def _split_by_cells(pairs, keys, cuts, vars_list, max_hists, var_cells, max_cells):
    """Variables packed in order into batches within both ``max_hists`` histograms and
    ``max_cells`` cells; a variable that alone exceeds either is split over its cuts, keys and
    (unc, scale) pairs, with as many histograms per batch as both budgets allow."""
    if not vars_list:
        yield pairs, keys, cuts, vars_list
        return
    per_var = _n_hists(pairs, keys, cuts, [None])
    count_cap = max_hists if max_hists and max_hists >= 1 else None
    batch, n_batch, cells_batch = [], 0, 0
    for var in vars_list:
        cells = max(1, int(var_cells.get(var, 1)))
        cells_var = per_var * cells
        too_big = cells_var > max_cells or (count_cap and per_var > count_cap)
        fits = (count_cap is None or n_batch + per_var <= count_cap) and (
            cells_batch + cells_var <= max_cells
        )
        if batch and (too_big or not fits):
            yield pairs, keys, cuts, batch
            batch, n_batch, cells_batch = [], 0, 0
        if too_big:
            limit = max(1, max_cells // cells)
            if count_cap:
                limit = min(limit, count_cap)
            yield from _split_axes(pairs, keys, cuts, [var], limit)
            continue
        batch.append(var)
        n_batch += per_var
        cells_batch += cells_var
    if batch:
        yield pairs, keys, cuts, batch


def _n_hists(pairs, keys, cuts, vars_list):
    return count_booked_hists(
        max(1, len(vars_list)),
        max(1, len(keys)),
        max(1, len(cuts) if cuts else 1),
        max(1, len(pairs)),
    )


def _split_axes(pairs, keys, cuts, vars_list, max_hists):
    if _n_hists(pairs, keys, cuts, vars_list) <= max_hists:
        yield pairs, keys, cuts, vars_list
        return

    per_var = _n_hists(pairs, keys, cuts, [None])
    if per_var <= max_hists:
        for d in _chunks(vars_list, max_hists // per_var):
            yield pairs, keys, cuts, d
        return

    for d in _chunks(vars_list, 1):
        if cuts:
            per_cut = _n_hists(pairs, keys, [None], d)
            if per_cut <= max_hists:
                for c in _chunks(cuts, max_hists // per_cut):
                    yield pairs, keys, c, d
                continue
            for c in _chunks(cuts, 1):
                yield from _split_keys_and_scales(pairs, keys, c, d, max_hists)
        else:
            yield from _split_keys_and_scales(pairs, keys, cuts, d, max_hists)


def _split_keys_and_scales(pairs, keys, cuts, vars_list, max_hists):
    if _n_hists(pairs, keys, cuts, vars_list) <= max_hists:
        yield pairs, keys, cuts, vars_list
        return
    per_key = _n_hists(pairs, [None], cuts, vars_list)
    if per_key <= max_hists:
        for b in _chunks(keys, max_hists // per_key):
            yield pairs, b, cuts, vars_list
        return
    for b in _chunks(keys, 1):
        if _n_hists(pairs, b, cuts, vars_list) <= max_hists:
            yield pairs, b, cuts, vars_list
            continue
        for a in _chunks(pairs, 1):
            yield a, b, cuts, vars_list
