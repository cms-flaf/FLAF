"""Blinding of the data in HistPlotter's plots, decided without ROOT so it is unit-tested in CI.

A variable's entry in config/plot/histograms.yaml takes two blinding keys:

- ``blind: true``: no data on the plots of the variable;
- ``blind_ranges``: no data point inside windows of the x axis, given as one window ``[lo, hi]``
  or a list of windows ``[[lo, hi], ...]``, in every region.

Any other key starting with "blind", and a blinding key in the page-level data style
(``data_hist``), raises: a key that no code reads must never pass for blinding.
"""

import math

BLIND_KEYS = ("blind", "blind_ranges")

_RANGES_FORM = (
    "blind_ranges takes one window [lo, hi] or a list of windows [[lo, hi], ...], "
    "with numbers lo < hi"
)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_blind_key(key):
    return isinstance(key, str) and key.lower().startswith("blind")


def blind_windows(var, entry_name, entry):
    """The blinding windows of variable `var` as a list of (lo, hi): the blind_ranges of its
    histograms.yaml entry (`entry`, under the key `entry_name`, which may be a pattern), or []
    without it. Raises on a key starting with "blind" that is not one of BLIND_KEYS, on
    blind_ranges that are not one window or a list of windows of two numbers lo < hi, and on
    blind_ranges for a variable of more than one dimension, whose plotted x axis is not the
    variable's."""
    where = f"histograms.yaml entry {entry_name!r}"
    if entry_name != var:
        where += f" (variable {var})"
    unknown = [key for key in entry if _is_blind_key(key) and key not in BLIND_KEYS]
    if unknown:
        raise ValueError(
            f"{where} has the blinding key(s) {unknown}, which no code reads; the blinding "
            f"keys are {list(BLIND_KEYS)}"
        )
    if "blind_ranges" not in entry:
        return []
    # the dimension as HistProducerFromNTuple counts it; a "2d" entry is plotted unrolled
    if len(entry.get("var_list") or [var]) > 1 or "2d" in entry:
        raise ValueError(
            f"{where}: blind_ranges is not supported for a variable of more than one "
            "dimension"
        )
    ranges = entry["blind_ranges"]
    if isinstance(ranges, list) and ranges and all(_is_number(x) for x in ranges):
        windows = [ranges]
    elif (
        isinstance(ranges, list) and ranges and all(isinstance(w, list) for w in ranges)
    ):
        windows = ranges
    else:
        raise ValueError(f"{where}: blind_ranges = {ranges!r}; {_RANGES_FORM}")
    for window in windows:
        if (
            len(window) != 2
            or not all(_is_number(x) for x in window)
            or not window[0] < window[1]
        ):
            raise ValueError(f"{where}: blind_ranges window {window!r}; {_RANGES_FORM}")
    return [(float(lo), float(hi)) for lo, hi in windows]


def check_data_style(page_cfgs):
    """Raise on a blinding key in the page-level data style, ``data_hist`` of each page
    configuration in `page_cfgs` ({file name: loaded YAML}): blinding is set per variable in
    histograms.yaml, and such a key is read by no code."""
    for name, cfg in page_cfgs.items():
        keys = [key for key in (cfg or {}).get("data_hist") or {} if _is_blind_key(key)]
        if keys:
            raise ValueError(
                f"{name}: data_hist has the key(s) {keys}, which no code reads; data is "
                "blinded per variable in config/plot/histograms.yaml, with "
                f"{list(BLIND_KEYS)}"
            )


def overlapping_bins(edges, windows):
    """The indices i of the bins [edges[i], edges[i + 1]) that share more than a point with a
    window (lo, hi): edges[i] < hi and lo < edges[i + 1]. A bin that overlaps a window in
    part is one of them; a bin that ends at lo or starts at hi is not. ROOT fills a value on a
    bin edge into the bin above it, so the values lo <= x < hi are always in such a bin.
    """
    return [
        i
        for i in range(len(edges) - 1)
        if any(edges[i] < hi and lo < edges[i + 1] for lo, hi in windows)
    ]


def _edges(hist):
    return [hist.GetBinLowEdge(i) for i in range(1, hist.GetNbinsX() + 2)]


def blind_data(hist, stored, windows, rebin):
    """Hide the data of `hist`, the plotted data histogram, inside the `windows`; returns the
    numbers of the hidden bins.

    A hidden bin gets NaN as content and error, which PlotKit's matplotlib backend draws as no
    marker in the plot and no point in the ratio panel, as mplhep's own blinding does. Hidden
    are the bins of `hist` that overlap a window (overlapping_bins), and every bin that holds
    events of a bin of `stored`, the histogram `hist` was made from by `rebin`
    (``rebin(hist, name)``, HistPlotter's apply_binning), that overlaps a window: an indicator
    of those bins, under- and overflow included, goes through the same `rebin`, so that no
    rebinning or overflow merging can move an event from a window into a bin that is drawn.
    `hist` and `stored` are ROOT TH1s, or anything with the same bin accessors; they may be
    the same object."""
    if not windows:
        return []
    indicator = stored.Clone(f"{stored.GetName()}_blind_indicator")
    indicator.SetDirectory(0)
    indicator.Reset()
    # with -inf and +inf added, the index of a bin is its ROOT number, 0 being the underflow
    flow_edges = [-math.inf] + _edges(stored) + [math.inf]
    for i in overlapping_bins(flow_edges, windows):
        indicator.SetBinContent(i, 1.0)
    indicator = rebin(indicator, indicator.GetName())
    hidden = {i + 1 for i in overlapping_bins(_edges(hist), windows)}
    hidden.update(
        i for i in range(1, hist.GetNbinsX() + 1) if indicator.GetBinContent(i) != 0
    )
    for i in sorted(hidden):
        hist.SetBinContent(i, math.nan)
        hist.SetBinError(i, math.nan)
    return sorted(hidden)
