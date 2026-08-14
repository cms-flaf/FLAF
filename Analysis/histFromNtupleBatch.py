"""Split HistFromNtuple histogram booking so one RDataFrame pass stays within a hist budget.

The booked-histogram count is
``n_vars × n_keys × n_cuts × n_scale_slots``, where ``n_scale_slots`` counts
every (uncertainty, scale) pair (Central plus each Up/Down). When that product
exceeds ``max_hists``, the work is partitioned so each batch books at most
``max_hists`` histograms. Axes are split in this order: variables, then extra
cuts, then selection keys, then (unc, scale).
"""


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
    uncs_to_compute, key_filter_dict, further_cuts, vars_to_process, max_hists
):
    """Yield ``(uncs_dict, keys_dict, cuts_dict, vars_list)`` batches.

    Each batch books at most ``max_hists`` histograms. ``max_hists < 1`` disables
    splitting (one batch with everything).
    """
    pairs = unc_scale_pairs(uncs_to_compute)
    keys = list(key_filter_dict.items())
    cuts = list(further_cuts.items()) if further_cuts else []
    vars_list = list(vars_to_process)
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
