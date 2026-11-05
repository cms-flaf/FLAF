"""Theory uncertainties stored as a family of members (PDF members, QCD-scale points).

A normalisation uncertainty of `weights.yaml` normally has two scales, Up and Down. A theory
uncertainty instead has one weight per member: the 103 PDF members, the 8 (muR, muF) points of
the QCD scale. Its members are only meaningful together, bin by bin, once the datasets of a
process are summed, so HistMerger fills one histogram per member and then reduces them to the
Up/Down pair every later step uses. The entry in `weights.yaml` says how:

    norm:
      qcd_scale:
        expression: weight_base_qcd_scale{scale}_rel * final_weight
        name: QCDscale_{process}_{}
        reduction: envelope

- ``reduction``: ``envelope`` -- Up/Down are the largest/smallest of the nominal and the
  members, bin by bin; ``exclude`` lists members left out (default: the two points where muR and
  muF move in opposite directions, the 7-point envelope). ``hessian`` -- Up/Down are the
  nominal +- sqrt(sum_k (m_k - m_0)^2 + alpha_s^2) over the ``members`` [first, last] (default
  [1, 100]), with alpha_s = (m_up - m_down) / 2 * ``alphas_scale`` from the ``alphas`` pair
  (default [101, 102], scale 1); m_0 is the nominal. An entry without ``reduction`` is an
  ordinary Up/Down uncertainty.

Like every other uncertainty, it is filled for every variable when the uncertainties are
computed. The reduction is done per process of the physics model, and ``{process}`` in ``name``
is that process: each process (each DY mass range, each signal) has its own nuisance.
"""

import math

ENVELOPE = "envelope"
HESSIAN = "hessian"
REDUCTIONS = (ENVELOPE, HESSIAN)
# The 7-point QCD-scale envelope leaves out the points where muR and muF move apart.
DEFAULT_ENVELOPE_EXCLUDE = ("_muR0p5_muF2", "_muR2_muF0p5")


def reduction(unc_entry):
    """The reduction of a weights.yaml entry, or None for an ordinary Up/Down uncertainty."""
    method = (unc_entry or {}).get("reduction")
    if method is None:
        return None
    if method not in REDUCTIONS:
        raise ValueError(
            f"unknown theory-uncertainty reduction {method!r}; expected one of "
            + ", ".join(REDUCTIONS)
        )
    return method


def format_name(unc_entry, scale, process):
    """The histogram suffix of one scale of an uncertainty, e.g. QCDscale_DY_M_50_Up."""
    return unc_entry["name"].format(scale, process=process)


def _hessian_settings(unc_entry):
    first, last = unc_entry.get("members", [1, 100])
    alphas = unc_entry.get("alphas", [101, 102])
    if alphas is not None and len(alphas) != 2:
        raise ValueError("hessian 'alphas' must be a pair [down member, up member]")
    return (
        [str(k) for k in range(int(first), int(last) + 1)],
        None if alphas is None else [str(a) for a in alphas],
        float(unc_entry.get("alphas_scale", 1.0)),
    )


def members_used(unc_entry, scales):
    """The members of `scales` a reduction reads."""
    method = reduction(unc_entry)
    if method == ENVELOPE:
        exclude = set(unc_entry.get("exclude", DEFAULT_ENVELOPE_EXCLUDE))
        return [s for s in scales if s not in exclude]
    members, alphas, _ = _hessian_settings(unc_entry)
    return members + (alphas or [])


def alphas_available(unc_entry, members):
    """Whether a hessian reduction can use its alpha_s pair for these members.

    A sample whose PDF weights stop at the eigenvectors (the X->HH signals, the four-flavour
    samples) has NaN in place of the alpha_s members, and so has a process summing it. The
    reduction then uses the eigenvectors alone. `members` maps scale -> cell contents.
    """
    _, alphas, _ = _hessian_settings(unc_entry)
    if alphas is None:
        return False
    return all(
        a in members and all(math.isfinite(x) for x in members[a]) for a in alphas
    )


def reduce_cells(unc_entry, nominal, members):
    """Up and Down cell contents from the nominal and `members` ({scale: contents}).

    `nominal` and each member are sequences over the same cells; returns (up, down) lists.
    """
    method = reduction(unc_entry)
    n = len(nominal)
    if method == ENVELOPE:
        exclude = set(unc_entry.get("exclude", DEFAULT_ENVELOPE_EXCLUDE))
        used = [m for s, m in members.items() if s not in exclude]
        if not used:
            raise ValueError("envelope reduction: no member left after 'exclude'")
        up = [max([nominal[i]] + [m[i] for m in used]) for i in range(n)]
        down = [min([nominal[i]] + [m[i] for m in used]) for i in range(n)]
        return up, down
    member_names, alphas, alphas_scale = _hessian_settings(unc_entry)
    missing = [k for k in member_names + (alphas or []) if k not in members]
    if missing:
        raise ValueError(f"hessian reduction: members {missing} are missing")
    bad = [k for k in member_names if not all(math.isfinite(x) for x in members[k])]
    if bad:
        raise ValueError(f"hessian reduction: members {bad} are not finite")
    if not alphas_available(unc_entry, members):
        alphas = None
    up, down = [], []
    for i in range(n):
        var = sum((members[k][i] - nominal[i]) ** 2 for k in member_names)
        if alphas is not None:
            a_down, a_up = alphas
            var += ((members[a_up][i] - members[a_down][i]) / 2 * alphas_scale) ** 2
        sigma = math.sqrt(var)
        up.append(nominal[i] + sigma)
        down.append(nominal[i] - sigma)  # a negative bin is left to the datacard maker
    return up, down


def _cells(hist):
    return [hist.GetBinContent(i) for i in range(hist.GetNcells())]


def reduce_histograms(unc_entry, nominal, members, unc_name):
    """Up/Down histograms (clones of `nominal`) from member histograms ({scale: TH1})."""
    contents = {s: _cells(h) for s, h in members.items()}
    up_cells, down_cells = reduce_cells(unc_entry, _cells(nominal), contents)
    out = []
    for label, cells in (("Up", up_cells), ("Down", down_cells)):
        hist = nominal.Clone(f"{nominal.GetName()}_{unc_name}{label}")
        hist.SetDirectory(0)
        for i, value in enumerate(cells):
            hist.SetBinContent(i, value)
        out.append(hist)
    return tuple(out)


def reduce_in_hist_dict(all_hists_dict, unc_name, unc_entry, scales, data_type="data"):
    """Replace the member histograms of `unc_name` by its Up/Down pair, in HistMerger's
    ``{process: {(key_tuple, (unc, scale)): hist}}``.

    Run after the datasets of every process are summed. A process without the members (data)
    is left alone; a process with some members but not all raises.
    """
    used = members_used(unc_entry, scales)
    hessian = reduction(unc_entry) == HESSIAN
    for process, hists in all_hists_dict.items():
        if process == data_type:
            continue
        key_tuples = {kt for (kt, (unc, _)) in hists if unc == unc_name}
        without_alphas = 0
        for key_tuple in key_tuples:
            nominal = hists.get((key_tuple, ("Central", "Central")))
            if nominal is None:
                raise RuntimeError(
                    f"{process} {'/'.join(key_tuple)}: {unc_name} members without a "
                    "nominal histogram"
                )
            members = {s: hists.get((key_tuple, (unc_name, s))) for s in used}
            missing = [s for s, h in members.items() if h is None]
            if missing:
                raise RuntimeError(
                    f"{process} {'/'.join(key_tuple)}: {unc_name} misses members {missing}"
                )
            if hessian and not alphas_available(
                unc_entry, {s: _cells(h) for s, h in members.items()}
            ):
                without_alphas += 1
            up, down = reduce_histograms(unc_entry, nominal, members, unc_name)
            for s in scales:
                hists.pop((key_tuple, (unc_name, s)), None)
            hists[(key_tuple, (unc_name, "Up"))] = up
            hists[(key_tuple, (unc_name, "Down"))] = down
        if without_alphas:
            print(
                f"{unc_name}: {process} has no finite alpha_s members (its samples do not "
                f"carry them); the eigenvectors alone are used in {without_alphas} of "
                f"{len(key_tuples)} directories"
            )
