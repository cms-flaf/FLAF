"""Split HistFromNtuple histogram booking so one RDataFrame pass stays within a hist budget.

The booked-histogram count is
``n_vars × n_keys × n_cuts × n_scale_slots``, where ``n_scale_slots`` counts
every (uncertainty, scale) pair (Central plus each Up/Down). When that product
exceeds ``max_hists``, the work is partitioned so each batch books at most
``max_hists`` histograms. Axes are split in this order: variables, then extra
cuts, then selection keys, then (unc, scale).

A count alone does not bound memory: a 2D histogram can have a hundred times the cells of a
1D one, and RDataFrame keeps one copy of every booked histogram per thread slot. Given the
memory of one histogram of each variable (``var_bytes``, see ``hist_bytes``), ``max_bytes``
additionally caps the memory booked in one batch.
"""

BYTES_PER_CELL = (
    16  # a TH1D/TH2D/TH3D cell: content + sum of weights squared, as doubles
)
# Memory of one booked histogram action apart from its cells: the TH1 object, the models kept
# by the booking, the RDataFrame action and its column register. Measured: 4.5-6 kB at nMT = 1.
BYTES_PER_BOOKED_HIST = 6000


def hist_bytes(cells, n_slots):
    """Memory of one booked histogram of ``cells`` cells (flows included).

    A histogram booked from a model is filled per thread slot: slot 0's own object is the result,
    and the other slots' copies are merged into it, so there are ``n_slots`` copies of the cells
    (1 without implicit MT), plus the fixed cost of the booked action.
    """
    return BYTES_PER_BOOKED_HIST + max(1, n_slots) * cells * BYTES_PER_CELL


def memory_budget(memory_mb):
    """Bytes one batch may book; ``memory_mb < 1`` disables the cap (returns 0)."""
    if not memory_mb or memory_mb < 1:
        return 0
    return int(memory_mb * 1e6)


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
    var_bytes=None,
    max_bytes=0,
):
    """Yield ``(uncs_dict, keys_dict, cuts_dict, vars_list)`` batches.

    Each batch books at most ``max_hists`` histograms; ``max_hists < 1`` removes the count cap.
    With ``var_bytes`` ({variable: memory of one of its booked histograms}) and
    ``max_bytes >= 1``, each batch also books at most ``max_bytes``; a single pass needs both
    caps off. Without them the batches are exactly those of the count alone.
    """
    pairs = unc_scale_pairs(uncs_to_compute)
    keys = list(key_filter_dict.items())
    cuts = list(further_cuts.items()) if further_cuts else []
    vars_list = list(vars_to_process)
    if var_bytes and max_bytes and max_bytes >= 1:
        for b_pairs, b_keys, b_cuts, b_vars in _split_by_memory(
            pairs, keys, cuts, vars_list, max_hists, var_bytes, max_bytes
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


def _split_by_memory(pairs, keys, cuts, vars_list, max_hists, var_bytes, max_bytes):
    """Batches within both ``max_hists`` histograms and ``max_bytes``.

    The variables that fit a batch are packed first-fit-decreasing (largest first, each into
    the first batch with room), so large 2D/3D variables do not each start a nearly empty pass.
    A variable that alone exceeds either cap is split over its cuts, keys and (unc, scale)
    pairs, with as many histograms per batch as both caps allow. Within a batch the variables
    keep their input order.
    """
    if not vars_list:
        yield pairs, keys, cuts, vars_list
        return
    per_var = _n_hists(pairs, keys, cuts, [None])
    count_cap = max_hists if max_hists and max_hists >= 1 else None
    order = {var: i for i, var in enumerate(vars_list)}
    hist_size = {var: max(1, int(var_bytes.get(var, 1))) for var in vars_list}

    fitting = []
    for var in vars_list:
        too_big = per_var * hist_size[var] > max_bytes or (
            count_cap and per_var > count_cap
        )
        if too_big:
            limit = max(1, max_bytes // hist_size[var])
            if count_cap:
                limit = min(limit, count_cap)
            yield from _split_axes(pairs, keys, cuts, [var], limit)
        else:
            fitting.append(var)

    batches = []  # [variables, histograms, bytes]
    for var in sorted(fitting, key=lambda v: (-hist_size[v], order[v])):
        var_total = per_var * hist_size[var]
        for batch in batches:
            if (count_cap is None or batch[1] + per_var <= count_cap) and (
                batch[2] + var_total <= max_bytes
            ):
                batch[0].append(var)
                batch[1] += per_var
                batch[2] += var_total
                break
        else:
            batches.append([[var], per_var, var_total])
    for batch_vars, _, _ in batches:
        yield pairs, keys, cuts, sorted(batch_vars, key=order.get)


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
    # one key does not fit: as many (unc, scale) pairs per batch as do
    per_pair = _n_hists([None], [None], cuts, vars_list)
    for b in _chunks(keys, 1):
        for a in _chunks(pairs, max_hists // per_pair):
            yield a, b, cuts, vars_list
