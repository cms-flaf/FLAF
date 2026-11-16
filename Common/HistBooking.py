"""Histogram booking: what HistFromNtupleProducer fills, optionally decided by an analysis plugin.

By default every variable of a job is filled in every selection key that the analysis'
``createKeyFilterDict`` (and the global ``further_cuts``) give. An analysis can instead name a module in
its ``global.yaml`` (``hist_booking_module: Analysis.hist_booking``) defining

- ``regions(setup)``: the regions, a list of names or a ``{region: cut}`` mapping, which fill the entry
  ``custom_regions`` names (``histTuple_regions`` if it names none), so that everything reading that
  entry gets them;
- ``booking(setup, keys, variables)``: for a job's selection keys ``[(key_tuple, filter)]`` and
  variables, a list of blocks ``{"keys": [(key_tuple, filter)], "variables": [...], "columns": [...]}``
  (``columns`` optional: input columns the filters read). Each block's variables are filled in its keys.

and optionally

- ``histtuple_columns(setup)``: the variables the booking's filters need in the histTuples beyond the
  flavour's own lists, ``{"binned": [...], "fullResolution": [...]}`` (both keys optional). A binned
  variable (binning from histograms.yaml) gets its ``<var>_bin`` column stored, a full-resolution one its
  raw column. They join ``setup.histTuple_vars``, so the HistTuple task requires the AnalysisCache
  producers that deliver them, but they are not histogrammed or merged (HistFromNtuple and HistMerger
  stay on ``setup.histTuple_plot_vars``). Typically the variables of bin-form sub-region cuts that are
  not plotted themselves.

FLAF only checks what protects its outputs: every variable of the job booked, no (key, variable) twice,
and the declared columns present in the input (in FillTree).
"""

import importlib


class HistBookingError(RuntimeError):
    pass


def load_plugin(global_params):
    """The module named by ``hist_booking_module``, or None."""
    name = global_params.get("hist_booking_module")
    if not name:
        return None
    module = importlib.import_module(name)
    for function in ("regions", "booking"):
        if not callable(getattr(module, function, None)):
            raise HistBookingError(f"hist_booking_module {name}: no {function}(...)")
    return module


def fill_regions(module, setup):
    """Put ``module.regions(setup)`` (a list of names or a ``{region: cut}`` mapping) in the entry
    ``custom_regions`` names."""
    regions = module.regions(setup)
    if not isinstance(regions, (list, dict)):
        raise HistBookingError(
            f"{module.__name__}.regions: expected a list (or a region -> cut mapping), got "
            f"{regions!r}"
        )
    name = setup.global_params.get("custom_regions") or "histTuple_regions"
    setup.global_params["custom_regions"] = name
    setup.global_params[name] = regions


HISTTUPLE_COLUMN_KINDS = ("binned", "fullResolution")


def _var_names(variables):
    """The column names of a variable list: names, ``{"name": ...}`` entries, or the ``vars`` of a
    multi-dimensional entry."""
    names = []
    for var in variables:
        if isinstance(var, dict) and "vars" in var:
            names.extend(var["vars"])
        elif isinstance(var, dict):
            names.append(var["name"])
        else:
            names.append(var)
    return names


def add_histtuple_columns(module, setup):
    """Store ``module.histtuple_columns(setup)`` (if the plugin defines it) on the setup:
    ``histTuple_extra_binned_vars`` and ``histTuple_extra_fullres_vars``, and their names added to
    ``histTuple_vars`` (which decides the AnalysisCache producers a HistTuple task requires).
    """
    setup.histTuple_extra_binned_vars = []
    setup.histTuple_extra_fullres_vars = []
    function = getattr(module, "histtuple_columns", None) if module else None
    if function is None:
        return
    if not callable(function):
        raise HistBookingError(f"{module.__name__}.histtuple_columns is not callable")
    columns = function(setup)
    if columns is None:
        return
    if not isinstance(columns, dict) or set(columns) - set(HISTTUPLE_COLUMN_KINDS):
        raise HistBookingError(
            f"{module.__name__}.histtuple_columns: expected "
            f'{{"binned": [...], "fullResolution": [...]}}, got {columns!r}'
        )
    for kind, attr in zip(
        HISTTUPLE_COLUMN_KINDS,
        ("histTuple_extra_binned_vars", "histTuple_extra_fullres_vars"),
    ):
        names = columns.get(kind) or []
        if not isinstance(names, (list, tuple)) or not all(
            isinstance(n, str) and n for n in names
        ):
            raise HistBookingError(
                f"{module.__name__}.histtuple_columns: {kind} must be a list of variable names, "
                f"got {names!r}"
            )
        setattr(setup, attr, list(dict.fromkeys(names)))
        for name in names:
            if name not in setup.histTuple_vars:
                setup.histTuple_vars.append(name)


def histtuple_binned_vars(setup):
    """The variables whose ``<var>_bin`` column the histTuples store: the flavour's plotted variables
    (flattened) and the plugin's extra binned ones, without duplicates."""
    names = _var_names(setup.histTuple_plot_vars)
    names += getattr(setup, "histTuple_extra_binned_vars", [])
    return list(dict.fromkeys(names))


def histtuple_fullres_vars(setup):
    """The variables the histTuples store at full resolution: the flavour's and the plugin's extra
    ones, without duplicates."""
    names = list(setup.histTuple_fullres_vars)
    names += getattr(setup, "histTuple_extra_fullres_vars", [])
    return list(dict.fromkeys(names))


def default_blocks(keys, variables):
    """Every variable in every key: the booking without a plugin."""
    return [(list(keys), list(variables), [])]


def plugin_blocks(module, setup, keys, variables):
    """The plugin's blocks for a job, as ``[(keys, variables, columns)]``."""
    blocks, booked = [], set()
    for block in module.booking(
        setup=setup, keys=list(keys), variables=list(variables)
    ):
        try:
            block_keys = [(tuple(k), flt) for k, flt in block["keys"]]
            block_vars = list(block["variables"])
        except (KeyError, TypeError, ValueError) as e:
            raise HistBookingError(
                f"{module.__name__}.booking: expected blocks "
                '{"keys": [(key_tuple, filter)], "variables": [...], "columns": [...]}, '
                f"got {block!r} ({type(e).__name__}: {e})"
            ) from e
        unknown = [v for v in block_vars if v not in variables]
        if unknown:
            raise HistBookingError(f"booking: {unknown} are not variables of the job")
        for key_tuple, _ in block_keys:
            for var in block_vars:
                if (key_tuple, var) in booked:
                    raise HistBookingError(
                        f"booking: {var} booked twice in {'/'.join(key_tuple)}"
                    )
                booked.add((key_tuple, var))
        if block_keys and block_vars:
            blocks.append((block_keys, block_vars, list(block.get("columns") or [])))
    unbooked = [v for v in variables if not any(v in b[1] for b in blocks)]
    if unbooked:
        raise HistBookingError(f"booking: {unbooked} are booked in no key")
    return blocks


def count_pairs(blocks):
    """(key, variable) pairs of a job's blocks."""
    return sum(len(keys) * len(block_vars) for keys, block_vars, _ in blocks)
