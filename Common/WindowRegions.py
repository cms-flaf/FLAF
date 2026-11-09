"""Window regions: regions restricted to a window that depends on a model parameter.

A resonance search keeps, per mass hypothesis, only the events whose reconstructed mass lies in a window
around it, a different window per category. Instead of carrying the full 2D (discriminant x mass)
histograms to apply that cut later, the cut can be made at histogramming time, as regions of their own:
``SR_HMEWin_m300`` is ``SR`` with the HME window of MX = 300, and it holds only the variables asked for
in it.

A window is a box in one or more variables: per mass and category, a ``[low, high]`` for each of the
variables it cuts on, all of them required at once.

Two blocks of configuration:

``windows`` (top level of global.yaml) names window sets, each its variables and a definitions file::

    windows:
      HME:
        variables: [ DeepHME_mass ]           # histogrammed variables of the flavour (their bin columns)
        definitions: config/windows/hme.yaml
        parameter: MX                         # optional, the placeholder name (default MX)

    # config/windows/hme.yaml: per mass and category, a [low, high] per variable of the set; a variable
    # left out of a box is not cut on there
    windows:
      300:
        res2b: { DeepHME_mass: [ 250, 400 ] }
        boosted: { DeepHME_mass: [ 260, 470 ] }

``window_regions`` (in a histTuple flavour) lists what to histogram in which windows::

    window_regions:
      - window: HME
        regions: [ SR, TT_CR, DY_CR ]        # base regions of the flavour's region list
        masses: all                          # or a list of masses of the definitions
        vars: [ "DNN_M{MX}_Signal" ]         # {MX} -> the window's own mass

Each entry and mass makes regions ``<region>_<window>Win_m<mass>`` (``region_name`` overrides the
pattern), with a key per base key of a listed region whose category the definitions give a window for:
the base key's selection and ``low <= variable < high`` for every variable of the box. Nothing else is
booked in them: a window region has only its entry's variables, which must already be histogrammed by
the flavour (so their bin columns exist and the base regions keep them).

The cut is made on each variable's stored bin index, so every window edge must be an edge of that
variable's binning (checked); a window starting at the first edge includes the underflow and one ending
at the last edge the overflow, as a flow-inclusive projection of the full histogram would. The window
histograms are then exactly the window slices of the full ones.

Every ambiguity is an error: an unknown window set, mass, region, category or box variable, an edge off
the binning, a variable not histogrammed, a mass-tagged variable without the placeholder, a name
collision.
"""

import os
import re

import yaml

DEFAULT_PARAMETER = "MX"


class WindowRegionError(RuntimeError):
    pass


def _var_name(var):
    return var["name"] if isinstance(var, dict) else var


def _var_components(var):
    """The 1D variables a flavour variable is made of: itself, or a 2D variable's ``vars``."""
    if isinstance(var, dict) and "vars" in var:
        return list(var["vars"])
    return [_var_name(var)]


def axis_edges(x_bins):
    """The bin edges of a histograms.yaml ``x_bins``: a list of edges or ``"n|low:high"``."""
    if isinstance(x_bins, list):
        return [float(e) for e in x_bins]
    n_bins, bin_range = str(x_bins).split("|")
    low, high = (float(v) for v in bin_range.split(":"))
    n_bins = int(n_bins)
    return [low + (high - low) * i / n_bins for i in range(n_bins + 1)]


def edge_index(edges, value, what):
    """The index of ``value`` among ``edges``; an error if it is not one of them."""
    for i, edge in enumerate(edges):
        if abs(edge - float(value)) <= 1e-6 * max(1.0, abs(edge)):
            return i
    raise WindowRegionError(
        f"{what}: {value} is not a bin edge of the variable's binning; a window must sit on "
        f"the stored edges, or the cut on the bin index could not reproduce it"
    )


def bin_cut(bin_column, edges, low, high, what):
    """The cut ``low <= x < high`` on the bin index column (``FindFixBin - 1``: -1 is the underflow,
    ``n_bins`` the overflow). A window from the first edge takes the underflow too, one to the last
    edge the overflow."""
    i_low = edge_index(edges, low, f"{what} low edge")
    i_high = edge_index(edges, high, f"{what} high edge")
    if i_high <= i_low:
        raise WindowRegionError(f"{what}: empty window [{low}, {high}]")
    n_bins = len(edges) - 1
    cuts = []
    if i_low > 0:
        cuts.append(f"{bin_column} >= {i_low}")
    if i_high < n_bins:
        cuts.append(f"{bin_column} < {i_high}")
    return " && ".join(cuts) if cuts else "true"


def box_cut(box, edges, what):
    """The cut of a box ``{variable: (low, high)}``: every variable's ``bin_cut``, in the order of the
    box, all required; ``"true"`` if none cuts anything."""
    cuts = [
        bin_cut(f"{var}_bin", edges[var], low, high, f"{what} {var}")
        for var, (low, high) in box.items()
    ]
    cuts = [c for c in cuts if c != "true"]
    return " && ".join(cuts) if cuts else "true"


def load_window_sets(global_params, ana_path):
    """``{name: {"variables", "parameter", "region_name", "definitions",
    "windows": {mass: {category: {variable: (low, high)}}}}}`` of the top-level ``windows`` block.
    """
    cfg = global_params.get("windows") or {}
    if not isinstance(cfg, dict):
        raise WindowRegionError(
            f"windows must map a window set name to its definition, got {cfg!r}"
        )
    sets = {}
    for name, entry in cfg.items():
        if (
            not isinstance(entry, dict)
            or "variables" not in entry
            or "definitions" not in entry
        ):
            raise WindowRegionError(
                f"window set {name}: needs 'variables' and 'definitions'"
            )
        unknown = set(entry) - {
            "variables",
            "definitions",
            "parameter",
            "region_name",
        }
        if unknown:
            raise WindowRegionError(
                f"window set {name}: unknown keys {sorted(unknown)}"
            )
        variables = entry["variables"]
        if (
            not isinstance(variables, list)
            or not variables
            or not all(isinstance(v, str) for v in variables)
            or len(set(variables)) != len(variables)
        ):
            raise WindowRegionError(
                f"window set {name}: 'variables' must be a non-empty list of distinct names, "
                f"got {variables!r}"
            )
        path = entry["definitions"]
        if not os.path.isabs(path):
            path = os.path.join(ana_path, path)
        if not os.path.exists(path):
            raise WindowRegionError(
                f"window set {name}: definitions file {path} not found"
            )
        with open(path) as f:
            definitions = yaml.safe_load(f) or {}
        windows_raw = definitions.get("windows")
        if not isinstance(windows_raw, dict) or not windows_raw:
            raise WindowRegionError(
                f"window set {name}: {path} has no 'windows' mapping"
            )
        windows = {}
        for mass, per_cat in windows_raw.items():
            if not isinstance(per_cat, dict) or not per_cat:
                raise WindowRegionError(
                    f"window set {name}: mass {mass} has no category windows"
                )
            windows[int(mass)] = {}
            for cat, box in per_cat.items():
                what = f"window set {name}: mass {mass} {cat}"
                if not isinstance(box, dict) or not box:
                    raise WindowRegionError(
                        f"{what}: expected a box {{<variable>: [low, high], ...}}, got {box!r}"
                    )
                unknown_vars = sorted(set(box) - set(variables))
                if unknown_vars:
                    raise WindowRegionError(
                        f"{what}: {unknown_vars} not among the set's variables {variables}"
                    )
                windows[int(mass)][str(cat)] = {}
                for var, bounds in box.items():
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                        raise WindowRegionError(
                            f"{what} {var}: expected [low, high], got {bounds!r}"
                        )
                    windows[int(mass)][str(cat)][var] = (
                        float(bounds[0]),
                        float(bounds[1]),
                    )
        parameter = entry.get("parameter", DEFAULT_PARAMETER)
        sets[name] = {
            "variables": list(variables),
            "parameter": parameter,
            "region_name": entry.get(
                "region_name", "{region}_{window}Win_m{" + parameter + "}"
            ),
            "windows": windows,
            "definitions": path,
        }
    return sets


def _format(template, parameter, value, what):
    placeholders = set(re.findall(r"\{(\w+)\}", template))
    unknown = placeholders - {parameter, "region", "window"}
    if unknown:
        raise WindowRegionError(
            f"{what}: unknown placeholder(s) {sorted(unknown)} in '{template}'"
        )
    return template.replace("{" + parameter + "}", str(value))


def expand_window_regions(
    flavour_cfg,
    window_sets,
    base_regions,
    categories,
    flavour_vars,
    active_var_names,
    hist_edges,
):
    """The window regions of a flavour, a list of dicts:
    ``{"name", "base_region", "window", "mass", "vars", "cuts": {category: cut expression}}``.

    ``flavour_vars``: the flavour's ``variables`` as configured; ``active_var_names``: those histogrammed
    in this setup (a user_custom ``variables:`` may restrict them; window variables it leaves out are
    dropped, not refused); ``hist_edges(var)``: the bin edges of a 1D variable.
    """
    entries = flavour_cfg.get("window_regions") or []
    if not entries:
        return []
    if not isinstance(entries, list):
        raise WindowRegionError("window_regions must be a list of entries")
    flavour_names = {_var_name(v) for v in flavour_vars}
    binned_components = set()
    for v in flavour_vars:
        binned_components.update(_var_components(v))
    base_regions = list(base_regions)
    regions_out = []
    names_seen = set(base_regions)
    for i_entry, entry in enumerate(entries):
        what = f"window_regions[{i_entry}]"
        if not isinstance(entry, dict):
            raise WindowRegionError(f"{what}: an entry is a mapping, got {entry!r}")
        unknown = set(entry) - {"window", "regions", "masses", "vars"}
        missing = {"window", "regions", "masses", "vars"} - set(entry)
        if unknown or missing:
            raise WindowRegionError(
                f"{what}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
            )
        set_name = entry["window"]
        if set_name not in window_sets:
            raise WindowRegionError(
                f"{what}: window set '{set_name}' is not defined in 'windows' ({sorted(window_sets)})"
            )
        wset = window_sets[set_name]
        parameter = wset["parameter"]
        edges = {}
        for wvar in wset["variables"]:
            if wvar not in binned_components:
                raise WindowRegionError(
                    f"{what}: the window variable {wvar} is not histogrammed by the flavour, so "
                    "it has no bin column to cut on"
                )
            edges[wvar] = hist_edges(wvar)

        regions = entry["regions"]
        if not isinstance(regions, list) or not regions:
            raise WindowRegionError(f"{what}: regions must be a non-empty list")
        for reg in regions:
            if reg not in base_regions:
                raise WindowRegionError(
                    f"{what}: region '{reg}' is not among the regions histogrammed ({base_regions})"
                )

        masses = entry["masses"]
        if masses == "all":
            masses = sorted(wset["windows"])
        elif isinstance(masses, list) and masses:
            masses = [int(m) for m in masses]
            for m in masses:
                if m not in wset["windows"]:
                    raise WindowRegionError(
                        f"{what}: mass {m} has no window in {wset['definitions']}"
                    )
        else:
            raise WindowRegionError(f"{what}: masses must be 'all' or a non-empty list")

        var_templates = entry["vars"]
        if not isinstance(var_templates, list) or not var_templates:
            raise WindowRegionError(f"{what}: vars must be a non-empty list")
        all_masses = sorted(wset["windows"])
        for tmpl in var_templates:
            if "{" + parameter + "}" in tmpl:
                continue
            tagged = [m for m in all_masses if re.search(rf"(?<!\d){m}(?!\d)", tmpl)]
            if tagged:
                raise WindowRegionError(
                    f"{what}: variable '{tmpl}' names mass {tagged[0]} but has no {{{parameter}}}: "
                    "it would be filled in the windows of every mass"
                )

        for mass in masses:
            per_cat = wset["windows"][mass]
            for cat in per_cat:
                if cat not in categories:
                    raise WindowRegionError(
                        f"{what}: category '{cat}' of {wset['definitions']} (mass {mass}) is not "
                        f"a configured category ({categories})"
                    )
            cuts = {
                cat: box_cut(box, edges, f"{set_name} m={mass} {cat}")
                for cat, box in per_cat.items()
            }
            mass_vars = []
            for tmpl in var_templates:
                var = _format(tmpl, parameter, mass, what)
                if var not in flavour_names:
                    raise WindowRegionError(
                        f"{what}: variable '{var}' is not a variable of the flavour; a window region "
                        "only holds variables the flavour already histograms"
                    )
                if var in active_var_names and var not in mass_vars:
                    mass_vars.append(var)
            for reg in regions:
                name = _format(wset["region_name"], parameter, mass, what)
                name = name.replace("{region}", reg).replace("{window}", set_name)
                if name in names_seen:
                    raise WindowRegionError(
                        f"{what}: region name '{name}' is defined twice"
                    )
                names_seen.add(name)
                regions_out.append(
                    {
                        "name": name,
                        "base_region": reg,
                        "window": set_name,
                        "mass": mass,
                        "vars": list(mass_vars),
                        "cuts": dict(cuts),
                    }
                )
    return regions_out


def window_blocks(base_keys, window_regions, vars_to_process):
    """The window part of a HistFromNtuple job: ``[(keys, vars)]``, one block per window region with
    variables in this job, ``keys`` a list of ``(key_tuple, base_key_index, cut)``.

    ``base_keys``: the job's ``[(key_tuple, filter)]``, key tuples ``(channel, region, category, ...)``.
    A window key is the base key with the region renamed, for each base key of the window region's base
    region whose category has a window.
    """
    blocks = []
    job_vars = list(vars_to_process)
    for wreg in window_regions:
        block_vars = [v for v in job_vars if v in wreg["vars"]]
        if not block_vars:
            continue
        keys = []
        for index, (key_tuple, _) in enumerate(base_keys):
            if len(key_tuple) < 3 or key_tuple[1] != wreg["base_region"]:
                continue
            cut = wreg["cuts"].get(key_tuple[2])
            if cut is None:
                continue
            new_key = (key_tuple[0], wreg["name"]) + tuple(key_tuple[2:])
            keys.append((new_key, index, cut))
        if keys:
            blocks.append((keys, block_vars))
    return blocks


def count_window_hists(blocks, n_weights):
    return sum(len(keys) * len(block_vars) for keys, block_vars in blocks) * n_weights
