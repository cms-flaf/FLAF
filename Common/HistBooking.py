"""Histogram booking: what HistFromNtupleProducer fills, optionally decided by an analysis plugin.

By default every variable of a job is filled in every selection key (channel, region, category[, ...])
that the analysis' ``createKeyFilterDict`` (and the global ``further_cuts``) give: one booking block, as
it has always been.

An analysis can instead name a plugin module in its ``global.yaml``::

    hist_booking_module: Analysis.hist_booking

The module defines two functions:

- ``regions(setup)``: the regions of the analysis, a list (or a region -> cut mapping). FLAF publishes
  them as the global parameter ``histTuple_regions`` in place of the list ``custom_regions`` names: the
  analysis' ``createKeyFilterDict``, the ``histTuple_selectors`` and the plot settings read them.
- ``booking(setup, keys, variables)``: the booking blocks of a HistFromNtupleProducer job. ``keys`` are
  the job's selection keys ``[(key_tuple, filter)]``, ``variables`` its variables. It returns a list of
  blocks, each a mapping::

      {"keys": [(key_tuple, filter), ...],     # or {key_tuple: filter}
       "variables": [variable, ...],           # filled in every key of the block
       "columns": [column, ...]}               # optional: the input columns the filters read

  A block's keys are its own: a plugin can book a variable in some keys only, or in keys of its own
  (e.g. a region restricted by a further cut), and leave keys out. Every variable of the job must be
  booked in at least one block, no (key, variable) twice, and the job stops before filling if its input
  lacks a column a block declares.

Without ``hist_booking_module`` nothing changes.
"""

import importlib

PLUGIN_FUNCTIONS = ("regions", "booking")


class HistBookingError(RuntimeError):
    pass


def load_plugin(global_params):
    """The analysis' booking plugin module named by ``hist_booking_module``, or None."""
    name = global_params.get("hist_booking_module")
    if not name:
        return None
    if not isinstance(name, str):
        raise HistBookingError(
            f"hist_booking_module must be a module name, got {name!r}"
        )
    try:
        module = importlib.import_module(name)
    except ImportError as e:
        raise HistBookingError(
            f"hist_booking_module {name}: cannot import it ({e})"
        ) from e
    missing = [f for f in PLUGIN_FUNCTIONS if not callable(getattr(module, f, None))]
    if missing:
        raise HistBookingError(
            f"hist_booking_module {name}: defines no {', '.join(missing)}(...)"
        )
    return module


def plugin_regions(module, setup):
    """``module.regions(setup)``: a list of region names or a region -> cut mapping."""
    regions = module.regions(setup)
    if not isinstance(regions, (list, dict)) or not all(
        isinstance(r, str) for r in regions
    ):
        raise HistBookingError(
            f"{module.__name__}.regions: expected a list of region names or a region -> cut "
            f"mapping, got {regions!r}"
        )
    return regions.copy()


def default_blocks(keys, variables):
    """Every variable in every key: the booking without a plugin."""
    return [(list(keys), list(variables), [])]


def plugin_blocks(module, setup, keys, variables):
    """The plugin's blocks for a job, checked: ``[(keys, variables, columns)]`` with keys a list of
    ``(key_tuple, filter)``."""
    raw = module.booking(setup=setup, keys=list(keys), variables=list(variables))
    what = f"{module.__name__}.booking"
    if not isinstance(raw, list):
        raise HistBookingError(f"{what}: expected a list of blocks, got {raw!r}")
    job_vars = list(variables)
    booked_pairs = set()
    booked_vars = set()
    blocks = []
    for i, block in enumerate(raw):
        where = f"{what}: block {i}"
        if not isinstance(block, dict):
            raise HistBookingError(f"{where}: a block is a mapping, got {block!r}")
        unknown = set(block) - {"keys", "variables", "columns"}
        if unknown or "keys" not in block or "variables" not in block:
            raise HistBookingError(
                f"{where}: needs 'keys' and 'variables' (and optionally 'columns'), got "
                f"{sorted(block)}"
            )
        block_keys = block["keys"]
        if isinstance(block_keys, dict):
            block_keys = list(block_keys.items())
        if not isinstance(block_keys, list):
            raise HistBookingError(f"{where}: 'keys' must be a list or a mapping")
        keys_out = []
        for item in block_keys:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[1], str)
            ):
                raise HistBookingError(
                    f"{where}: a key is (key_tuple, filter), got {item!r}"
                )
            key_tuple = tuple(item[0])
            if not key_tuple or not all(isinstance(k, str) and k for k in key_tuple):
                raise HistBookingError(
                    f"{where}: a key tuple is a tuple of names, got {item[0]!r}"
                )
            keys_out.append((key_tuple, item[1]))
        block_vars = block["variables"]
        if not isinstance(block_vars, list) or not block_vars:
            raise HistBookingError(f"{where}: 'variables' must be a non-empty list")
        unknown_vars = [v for v in block_vars if v not in job_vars]
        if unknown_vars:
            raise HistBookingError(
                f"{where}: {unknown_vars} are not variables of the job ({job_vars})"
            )
        columns = block.get("columns") or []
        if not isinstance(columns, list) or not all(
            isinstance(c, str) for c in columns
        ):
            raise HistBookingError(f"{where}: 'columns' must be a list of names")
        for key_tuple, _ in keys_out:
            for var in block_vars:
                pair = (key_tuple, var)
                if pair in booked_pairs:
                    raise HistBookingError(
                        f"{where}: {var} is booked twice in {'/'.join(key_tuple)}"
                    )
                booked_pairs.add(pair)
        if keys_out:
            booked_vars.update(block_vars)
            blocks.append((keys_out, list(block_vars), list(columns)))
    unbooked = [v for v in job_vars if v not in booked_vars]
    if unbooked:
        raise HistBookingError(
            f"{what}: {unbooked} booked in no key: every variable of the job needs a block"
        )
    return blocks


def count_pairs(blocks):
    """(key, variable) pairs of a job's blocks."""
    return sum(len(keys) * len(block_vars) for keys, block_vars, _ in blocks)
