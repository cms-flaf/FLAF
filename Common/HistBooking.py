"""Histogram booking: what HistFromNtupleProducer fills, optionally decided by an analysis plugin.

By default every variable of a job is filled in every selection key that the analysis'
``createKeyFilterDict`` (and the global ``further_cuts``) give. An analysis can instead name a module in
its ``global.yaml`` (``hist_booking_module: Analysis.hist_booking``) defining

- ``regions(setup)``: the regions, which fill the list ``custom_regions`` names (``histTuple_regions``
  if it names none), so that everything reading that list gets them;
- ``booking(setup, keys, variables)``: for a job's selection keys ``[(key_tuple, filter)]`` and
  variables, a list of blocks ``{"keys": [(key_tuple, filter)], "variables": [...], "columns": [...]}``
  (``columns`` optional: input columns the filters read). Each block's variables are filled in its keys.

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
    """Put ``module.regions(setup)`` in the list ``custom_regions`` names."""
    regions = module.regions(setup)
    if not isinstance(regions, (list, dict)):
        raise HistBookingError(
            f"{module.__name__}.regions: expected a list (or a region -> cut mapping), got "
            f"{regions!r}"
        )
    name = setup.global_params.get("custom_regions") or "histTuple_regions"
    setup.global_params["custom_regions"] = name
    setup.global_params[name] = regions


def default_blocks(keys, variables):
    """Every variable in every key: the booking without a plugin."""
    return [(list(keys), list(variables), [])]


def plugin_blocks(module, setup, keys, variables):
    """The plugin's blocks for a job, as ``[(keys, variables, columns)]``."""
    blocks, booked = [], set()
    for block in module.booking(
        setup=setup, keys=list(keys), variables=list(variables)
    ):
        block_keys = [(tuple(k), flt) for k, flt in block["keys"]]
        block_vars = list(block["variables"])
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
