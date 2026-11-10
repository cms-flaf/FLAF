"""Sub-regions: regions restricted by a cut that depends on a model parameter.

A resonance search keeps, per mass hypothesis, only the events in a mass-dependent part of a region, e.g.
the reconstructed mass near the hypothesis, with a different cut per category. Instead of carrying full 2D
(discriminant x mass) histograms to apply that cut later, the cut can be made at histogramming time, as
regions of their own: ``SR_HMESub_m300`` is ``SR`` with the HMESub cut of MX = 300, and it holds only the
variables asked for in it.

A histTuple flavour lists them::

    histTuple_flavors:
      Signal_Extraction_DL_window:
        regions: [ SR, TT_CR, DY_CR ]
        sub_regions:
          - name: HMESub                              # -> SR_HMESub_m300
            cuts: config/sub_regions/hme_DL.yaml      # the cut per mass and category
            regions: [ SR, TT_CR, DY_CR ]             # base regions among the flavour's regions
            masses: all                               # or a list of masses of the cuts file
            vars: [ "DNN_M{MX}_Signal" ]              # {MX} -> the sub-region's own mass
            # parameter: MX                           # optional: the placeholder name (default MX)
            # region_name: "{region}_{name}_m{MX}"    # optional: the region name pattern (this default)

The cuts file maps mass -> category -> cut, a cut in one of two forms::

    cuts:
      300:
        # bin form: per variable one [low, high] or a list of them (OR); the variables are AND'ed
        res2b:    { DeepHME_mass: [ 250, 400 ] }
        recovery: { DeepHME_mass: [ [ 250, 400 ], [ 500, 600 ] ], ll_mass: [ 12, 80 ] }
        # C++ form: an expression on full-resolution variables ({MX} is replaced by the mass)
        boosted:  "DeepHME_mass >= 260 && DeepHME_mass < 470"

- Bin form: the cut is made on each variable's stored bin index (``<var>_bin``), so each variable must be
  a binned variable of the flavour and every edge a bin edge of its binning; an interval from the first
  edge takes the underflow too, one to the last edge the overflow. A sub-region histogram is then exactly
  the slice of the full histogram.
- C++ form: the expression is used as written. Every variable it names must be in the flavour's
  ``fullResolution_variables`` (stored raw in the histTuples); nothing is added to them automatically,
  so a variable that is not there is an error. Function names (``abs(``), namespaced names
  (``std::abs``), numeric literals and ``true``/``false`` are not variables.

Keys: each entry and mass makes regions ``<region>_<name>_m<mass>``, with a key per base key of a listed
region whose category has a cut: the base key's selection and the cut. Nothing else is booked in them: a
sub-region has only its entry's variables, which must already be variables of the flavour.

Every ambiguity is an error: an unknown region, mass, category or cut variable, an edge off the binning, a
variable the flavour does not histogram, a mass-tagged variable without the placeholder, a region name
made twice.
"""

import os
import re

import yaml

DEFAULT_PARAMETER = "MX"
ENTRY_KEYS = {"name", "cuts", "regions", "masses", "vars"}
OPTIONAL_ENTRY_KEYS = {"parameter", "region_name"}
# words of a C++ cut that are not column names
CPP_WORDS = {
    "true",
    "false",
    "and",
    "or",
    "not",
    "static_cast",
    "double",
    "float",
    "int",
    "bool",
    "unsigned",
    "long",
    "const",
}


class SubRegionError(RuntimeError):
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
    raise SubRegionError(
        f"{what}: {value} is not a bin edge of the variable's binning; a bin-form cut must sit on "
        "the stored edges, or the cut on the bin index could not reproduce it (use the C++ form, "
        "on a full-resolution variable, for other edges)"
    )


def bin_cut(bin_column, edges, low, high, what):
    """The cut ``low <= x < high`` on the bin index column (``FindFixBin - 1``: -1 is the underflow,
    ``n_bins`` the overflow). An interval from the first edge takes the underflow too, one to the last
    edge the overflow."""
    i_low = edge_index(edges, low, f"{what} low edge")
    i_high = edge_index(edges, high, f"{what} high edge")
    if i_high <= i_low:
        raise SubRegionError(f"{what}: empty interval [{low}, {high}]")
    n_bins = len(edges) - 1
    cuts = []
    if i_low > 0:
        cuts.append(f"{bin_column} >= {i_low}")
    if i_high < n_bins:
        cuts.append(f"{bin_column} < {i_high}")
    return " && ".join(cuts) if cuts else "true"


def _intervals(value, what):
    """``[low, high]`` or a non-empty list of them, as a list of (low, high)."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        return [(float(value[0]), float(value[1]))]
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, (list, tuple)) for v in value)
    ):
        out = []
        for v in value:
            out.extend(_intervals(v, what))
        return out
    raise SubRegionError(
        f"{what}: expected [low, high] or a list of [low, high], got {value!r}"
    )


def dict_cut(cut, edges_of, binned, what):
    """The bin-index cut of a bin-form cut ``{variable: intervals}`` and its columns."""
    parts, columns = [], []
    for var, value in cut.items():
        if var not in binned:
            raise SubRegionError(
                f"{what}: {var} is not a binned variable of the flavour, so it has no bin column "
                "to cut on"
            )
        edges = edges_of(var)
        terms = [
            bin_cut(f"{var}_bin", edges, low, high, f"{what} {var}")
            for low, high in _intervals(cut[var], f"{what} {var}")
        ]
        if "true" in terms:
            continue  # the whole range: no cut, no column needed
        columns.append(f"{var}_bin")
        if len(terms) == 1:
            parts.append(terms[0])
        else:
            parts.append("(" + " || ".join(f"({t})" for t in terms) + ")")
    return (" && ".join(parts) if parts else "true"), columns


def cpp_variables(expr):
    """The column names a C++ cut uses: its identifiers, without numeric literals, function names
    (followed by ``(``; their arguments are columns), namespaced names (``std::abs``) and C++ words.
    A ``<`` after a name is a comparison, not a template."""
    text = re.sub(r'"[^"]*"', " ", expr)
    text = re.sub(
        r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlLuU]*(?![\w.])", " ", text
    )
    names = []
    for m in re.finditer(r"[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*", text):
        word = m.group(0)
        following = text[m.end() :].lstrip()[:1]
        if "::" in word or following == "(" or word in CPP_WORDS:
            continue
        if word not in names:
            names.append(word)
    return names


def cpp_cut(expr, fullres, what, flavour):
    """A C++ cut and its columns; every variable it names must be a full-resolution variable."""
    if not expr.strip():
        raise SubRegionError(f"{what}: empty cut")
    names = cpp_variables(expr)
    if not names:
        raise SubRegionError(f"{what}: the cut '{expr}' uses no variable")
    missing = [n for n in names if n not in fullres]
    if missing:
        raise SubRegionError(
            f"{what}: the cut '{expr}' uses {', '.join(missing)}, not in fullResolution_variables "
            f"of flavour {flavour}; add {'it' if len(missing) == 1 else 'them'} there (it is not "
            "added automatically)"
        )
    return expr, names


def load_cuts(path, ana_path, what):
    """``{mass: {category: cut}}`` of a cuts file, a cut a dict (bin form) or a string (C++ form)."""
    if not os.path.isabs(path):
        path = os.path.join(ana_path, path)
    if not os.path.exists(path):
        raise SubRegionError(f"{what}: cuts file {path} not found")
    with open(path) as f:
        content = yaml.safe_load(f) or {}
    raw = content.get("cuts") if isinstance(content, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise SubRegionError(f"{what}: {path} has no 'cuts' mapping")
    cuts = {}
    for mass, per_cat in raw.items():
        if not isinstance(per_cat, dict) or not per_cat:
            raise SubRegionError(f"{what}: {path} mass {mass} has no category cuts")
        cuts[int(mass)] = {}
        for cat, cut in per_cat.items():
            if isinstance(cut, dict):
                if not cut:
                    raise SubRegionError(f"{what}: {path} mass {mass} {cat}: empty cut")
            elif not isinstance(cut, str):
                raise SubRegionError(
                    f"{what}: {path} mass {mass} {cat}: a cut is a {{variable: [low, high]}} dict "
                    f"or a C++ string, got {cut!r}"
                )
            cuts[int(mass)][str(cat)] = cut
    return cuts, path


def _format(template, parameter, value, what, allowed=("region", "name")):
    placeholders = set(re.findall(r"\{(\w+)\}", template))
    unknown = placeholders - {parameter, *allowed}
    if unknown:
        raise SubRegionError(
            f"{what}: unknown placeholder(s) {sorted(unknown)} in '{template}'"
        )
    return template.replace("{" + parameter + "}", str(value))


def expand_sub_regions(
    flavour_name,
    flavour_cfg,
    ana_path,
    base_regions,
    categories,
    active_var_names,
    hist_edges,
):
    """The sub-regions of a flavour, a list of dicts ``{"name", "sub_region", "base_region", "mass",
    "vars", "cuts": {category: expression}, "columns": {category: [columns]}}``.

    ``active_var_names``: the variables histogrammed in this setup (a user_custom ``variables:`` may
    restrict them; sub-region variables it leaves out are dropped, not refused); ``hist_edges(var)``: the
    bin edges of a 1D variable.
    """
    entries = flavour_cfg.get("sub_regions") or []
    if not entries:
        return []
    if not isinstance(entries, list):
        raise SubRegionError(f"flavour {flavour_name}: sub_regions must be a list")
    flavour_vars = flavour_cfg.get("variables") or []
    flavour_names = {_var_name(v) for v in flavour_vars}
    binned = set()
    for v in flavour_vars:
        binned.update(_var_components(v))
    fullres = set(flavour_cfg.get("fullResolution_variables") or [])
    base_regions = list(base_regions)
    out = []
    names_seen = set(base_regions)
    for i_entry, entry in enumerate(entries):
        what = f"flavour {flavour_name} sub_regions[{i_entry}]"
        if not isinstance(entry, dict):
            raise SubRegionError(f"{what}: an entry is a mapping, got {entry!r}")
        unknown = set(entry) - ENTRY_KEYS - OPTIONAL_ENTRY_KEYS
        missing = ENTRY_KEYS - set(entry)
        if unknown or missing:
            raise SubRegionError(
                f"{what}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
            )
        name = entry["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SubRegionError(f"{what}: name must be a word, got {name!r}")
        what = f"{what} ({name})"
        parameter = entry.get("parameter", DEFAULT_PARAMETER)
        region_name = entry.get("region_name", "{region}_{name}_m{" + parameter + "}")
        cuts, cuts_path = load_cuts(entry["cuts"], ana_path, what)

        regions = entry["regions"]
        if not isinstance(regions, list) or not regions:
            raise SubRegionError(f"{what}: regions must be a non-empty list")
        for reg in regions:
            if reg not in base_regions:
                raise SubRegionError(
                    f"{what}: region '{reg}' is not among the flavour's regions ({base_regions})"
                )

        masses = entry["masses"]
        if masses == "all":
            masses = sorted(cuts)
        elif isinstance(masses, list) and masses:
            masses = [int(m) for m in masses]
            for m in masses:
                if m not in cuts:
                    raise SubRegionError(f"{what}: mass {m} has no cut in {cuts_path}")
        else:
            raise SubRegionError(f"{what}: masses must be 'all' or a non-empty list")

        var_templates = entry["vars"]
        if not isinstance(var_templates, list) or not var_templates:
            raise SubRegionError(f"{what}: vars must be a non-empty list")
        all_masses = sorted(cuts)
        for tmpl in var_templates:
            if "{" + parameter + "}" in tmpl:
                continue
            tagged = [m for m in all_masses if re.search(rf"(?<!\d){m}(?!\d)", tmpl)]
            if tagged:
                raise SubRegionError(
                    f"{what}: variable '{tmpl}' names mass {tagged[0]} but has no {{{parameter}}}: "
                    "it would be filled in the sub-regions of every mass"
                )

        for mass in masses:
            mass_cuts, mass_columns = {}, {}
            for cat, cut in cuts[mass].items():
                if cat not in categories:
                    raise SubRegionError(
                        f"{what}: category '{cat}' of {cuts_path} (mass {mass}) is not a configured "
                        f"category ({categories})"
                    )
                cut_what = f"{what} m={mass} {cat}"
                if isinstance(cut, dict):
                    expr, columns = dict_cut(cut, hist_edges, binned, cut_what)
                else:
                    text = _format(cut, parameter, mass, cut_what, allowed=())
                    expr, columns = cpp_cut(text, fullres, cut_what, flavour_name)
                mass_cuts[cat] = expr
                mass_columns[cat] = columns
            mass_vars = []
            for tmpl in var_templates:
                var = _format(tmpl, parameter, mass, what, allowed=())
                if var not in flavour_names:
                    raise SubRegionError(
                        f"{what}: variable '{var}' is not a variable of the flavour; a sub-region "
                        "only holds variables the flavour already histograms"
                    )
                if var in active_var_names and var not in mass_vars:
                    mass_vars.append(var)
            for reg in regions:
                reg_name = _format(region_name, parameter, mass, what)
                reg_name = reg_name.replace("{region}", reg).replace("{name}", name)
                if reg_name in names_seen:
                    raise SubRegionError(
                        f"{what}: region name '{reg_name}' is defined twice"
                    )
                names_seen.add(reg_name)
                out.append(
                    {
                        "name": reg_name,
                        "sub_region": name,
                        "base_region": reg,
                        "mass": mass,
                        "vars": list(mass_vars),
                        "cuts": dict(mass_cuts),
                        "columns": {c: list(v) for c, v in mass_columns.items()},
                    }
                )
    return out


def sub_region_blocks(base_keys, sub_regions, vars_to_process):
    """The sub-region part of a HistFromNtuple job: ``[(keys, vars)]``, one block per sub-region with
    variables in this job, ``keys`` a list of ``(key_tuple, base_key_index, cut, columns)``.

    ``base_keys``: the job's ``[(key_tuple, filter)]``, key tuples ``(channel, region, category, ...)``.
    A sub-region key is the base key with the region renamed, for each base key of the sub-region's base
    region whose category has a cut.
    """
    blocks = []
    job_vars = list(vars_to_process)
    for sreg in sub_regions:
        block_vars = [v for v in job_vars if v in sreg["vars"]]
        if not block_vars:
            continue
        keys = []
        for index, (key_tuple, _) in enumerate(base_keys):
            if len(key_tuple) < 3 or key_tuple[1] != sreg["base_region"]:
                continue
            cut = sreg["cuts"].get(key_tuple[2])
            if cut is None:
                continue
            new_key = (key_tuple[0], sreg["name"]) + tuple(key_tuple[2:])
            keys.append((new_key, index, cut, list(sreg["columns"][key_tuple[2]])))
        if keys:
            blocks.append((keys, block_vars))
    return blocks


def count_sub_region_pairs(blocks):
    """(key, variable) pairs of the blocks of a job."""
    return sum(len(keys) * len(block_vars) for keys, block_vars in blocks)
