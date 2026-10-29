"""What HistPlotter draws, decided without ROOT so it is unit-tested in CI: plot groups,
processes summed into one legend entry (global.yaml's plot_process_groups; a histogram only
needs Clone, SetDirectory and Add), and the control regions, where blinded variables show data
and no signal is drawn (plot_control_regions)."""

import re
import warnings

# The global settings of what a plot shows; one plot has one value of each.
PLOT_SETTINGS = ("plot_process_groups", "plot_control_regions", "signal_plot_scale")


def plot_settings(global_params):
    """The plot settings (PLOT_SETTINGS, and the analysis' regions) of a plot, from the merged
    Setup.global_params of each era it holds -- so user_custom and era configurations apply,
    as for every other task. ``global_params``: {era: Setup.global_params}, one era, or the
    eras of a meta-era. A setting the eras disagree on raises: one plot cannot group or blind
    two ways."""
    eras = list(global_params)
    settings = {}
    for key in PLOT_SETTINGS:
        values = {era: global_params[era].get(key) for era in eras}
        if any(v != values[eras[0]] for v in values.values()):
            raise ValueError(
                f"{key} differs between the eras of the plot, {values}: set it once, in "
                "config/global.yaml or a user_custom, for all of them"
            )
        settings[key] = values[eras[0]]
    # the regions histogrammed: custom_regions names the global list of them (QCDRegions),
    # or a mapping of region to cut (H_mumu), which counts by its keys as HistPlotTask does
    regions = []
    for era in eras:
        params = global_params[era]
        era_regions = params.get(params.get("custom_regions") or "")
        if isinstance(era_regions, dict):
            era_regions = list(era_regions)
        if isinstance(era_regions, list):
            regions += [r for r in era_regions if r not in regions]
    settings["regions"] = regions or None
    return settings


def control_regions(value, regions=None):
    """global.yaml's plot_control_regions as a list of region names.

    It must be a YAML list: written without brackets (``TT_CR, DY_CR``) it is one string, and a
    region test on a string matches substrings -- "SR" in "SR_SL_M300_CR" -- which would show
    data in a signal region of a blinded variable. A name that is not one of the analysis'
    `regions` (a typo leaves that region blinded) is warned about; it is not an error, since a
    configuration can narrow the regions it runs (e.g. CI)."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(r, str) for r in value):
        raise ValueError(
            f"plot_control_regions = {value!r} must be a list of region names, "
            "e.g. [ TT_CR, DY_CR ]"
        )
    if isinstance(regions, list):
        unknown = [r for r in value if r not in regions]
        if unknown:
            warnings.warn(
                f"plot_control_regions names {unknown}, which are not among the regions "
                f"{regions}; those regions stay blinded",
                UserWarning,
            )
    return list(value)


def want_signals(want, region, control_regions):
    """Whether a plot draws the signals: when they are wanted (HistPlotTask's
    plot_wantSignals, default true, passed as --wantSignals), never in a control region --
    a control region holds next to no signal, which normalised to the background is noise.
    """
    if not isinstance(control_regions, list):
        raise TypeError(f"control_regions = {control_regions!r} must be a list")
    return bool(want) and region not in control_regions


def want_data(want, blind, region, control_regions):
    """Whether a plot shows data: when it is wanted, unless the variable is blinded and the
    region is not a control region. `control_regions` is the list control_regions() gives.
    """
    if not isinstance(control_regions, list):
        raise TypeError(f"control_regions = {control_regions!r} must be a list")
    return bool(want) and not (blind and region not in control_regions)


# A ${PARAM} in a plot group's process pattern or name, and a ${...} of any content, to
# refuse one that is not a parameter name.
_GROUP_PARAM = re.compile(r"\$\{(\w+)\}")
_GROUP_BRACES = re.compile(r"\$\{([^}]*)\}")
# The keys a plot group may have.
_GROUP_KEYS = ("name", "processes", "color")
# A parameter value: 400, 1p74, m0p758, 1.5, M-400, ... -- word characters, '.' and '-'.
_GROUP_VALUE = r"[\w.\-]+?"
# A number as written in a value: plain (400, 1.5, -2) or in the CMS form (1p74, m0p758).
_NUMBER = re.compile(r"(?P<sign>-|m)?(?P<int>[0-9]+)(?:[.p](?P<frac>[0-9]+))?")


def _param_value_key(value):
    """Sort key of a parameter value: a number -- written plainly or in the CMS p/m form, so
    1p74 = 1.74 and m0p758 = -0.758 -- by its value, before any other text, which sorts as
    text. Never raises, whatever the values of a group."""
    number = _NUMBER.fullmatch(value)
    if number is None:
        return (1, 0.0, value)
    text = number["int"] + ("." + number["frac"] if number["frac"] else "")
    return (0, -float(text) if number["sign"] else float(text), value)


def _check_group(group):
    """The parameters of a plot group, in the order its first process pattern names them,
    after checking the group: a name and a non-empty list of distinct process names, all of
    which name the same parameters, as the name does, and no key but name, processes and
    color. A mistake raises a ValueError naming the group instead of giving a plot that looks
    right but is not."""
    name = group.get("name") if isinstance(group, dict) else None
    processes = group.get("processes") if isinstance(group, dict) else None
    if (
        not isinstance(name, str)
        or not isinstance(processes, list)
        or not processes
        or not all(isinstance(p, str) for p in processes)
    ):
        raise ValueError(
            f"plot_process_groups entry {group!r} needs a name and a list of process "
            "names, e.g. {name: Diboson, processes: [ VV, VVV ]}"
        )
    unknown = sorted(set(group) - set(_GROUP_KEYS))
    if unknown:
        raise ValueError(
            f"plot group {name!r} has the unknown keys {unknown}; a group takes "
            f"{list(_GROUP_KEYS)}"
        )
    bad = sorted(
        {
            "${" + p + "}"
            for text in [name] + processes
            for p in _GROUP_BRACES.findall(text)
            if not p.isidentifier()
        }
    )
    if bad:
        raise ValueError(
            f"plot group {name!r}: {', '.join(bad)} is not a parameter; a parameter is a "
            "name of letters, digits and _ not starting with a digit, e.g. ${MASS}"
        )
    duplicates = sorted({p for p in processes if processes.count(p) > 1})
    if duplicates:
        raise ValueError(f"plot group {name!r} lists {duplicates} more than once")
    params = list(dict.fromkeys(_GROUP_PARAM.findall(processes[0])))
    for pattern in processes[1:]:
        if set(_GROUP_PARAM.findall(pattern)) != set(params):
            raise ValueError(
                f"plot group {name!r}: every process must name the same parameters, but "
                f"{processes[0]!r} names {params} and {pattern!r} names "
                f"{sorted(set(_GROUP_PARAM.findall(pattern)))}"
            )
    if set(_GROUP_PARAM.findall(name)) != set(params):
        raise ValueError(
            f"plot group {name!r} must name the parameters of its processes, {params}, so "
            "that each entry has its own label"
        )
    return params


def _pattern_regex(pattern):
    """A process pattern as a regex: each ${PARAM} captures a value, a repeated one must
    repeat it."""
    parts, seen, pos = [], set(), 0
    for m in _GROUP_PARAM.finditer(pattern):
        parts.append(re.escape(pattern[pos : m.start()]))
        param = m.group(1)
        parts.append(
            f"(?P={param})" if param in seen else f"(?P<{param}>{_GROUP_VALUE})"
        )
        seen.add(param)
        pos = m.end()
    parts.append(re.escape(pattern[pos:]))
    return re.compile("".join(parts))


def _group_instances(group, hists, used):
    """{parameter values: member processes} of one plot group. A process of the group may
    name parameters, e.g. GluGluToRadion_bbWW_2L_${MASS}: the group then stands for one entry
    per set of values found in `hists`, in increasing order (numbers by value, the parameters
    in the order the pattern names them). A process that an earlier group took is not taken
    again; one that two patterns of this group match raises."""
    params = _check_group(group)
    patterns = [(p, _pattern_regex(p)) for p in group["processes"]]
    instances, matched_by = {}, {}
    for pattern, regex in patterns:
        for name in hists:
            match = regex.fullmatch(name)
            if not match or name in used:
                continue
            if name in matched_by:
                raise ValueError(
                    f"plot group {group['name']!r}: {name!r} matches both "
                    f"{matched_by[name]!r} and {pattern!r}, and would be counted twice"
                )
            matched_by[name] = pattern
            values = tuple((p, match[p]) for p in params)
            instances.setdefault(values, []).append(name)

    def order(values):
        return tuple(_param_value_key(v) for _, v in values)

    return {values: instances[values] for values in sorted(instances, key=order)}


def GroupProcesses(hists, groups):
    """The entries of `hists` summed into the plot groups of `groups` (global.yaml's
    plot_process_groups: [{name, color, processes}]): backgrounds into stack entries, stacked
    in that order from the bottom, and signals into one curve each. A group whose processes
    carry parameters (${MASS}) makes an entry per set of values, its name formatted with
    them. A group with none of its processes is skipped, and every entry no group takes is
    kept as it is, after the groups. The members of a group are all backgrounds or all
    signals. ``hists``: {process: (hist, plot name, color, process group)}."""
    grouped, used = {}, set()
    for group in groups:
        instances = _group_instances(group, hists, used)
        kinds = {hists[p][3] for members in instances.values() for p in members}
        if len(kinds) > 1 or kinds & {"data"}:
            raise RuntimeError(
                f"plot group {group['name']!r} mixes {sorted(kinds)}: "
                f"{[p for members in instances.values() for p in members]}"
            )
        for values, members in instances.items():
            key = "+".join(members)
            total = hists[members[0]][0].Clone(key)
            total.SetDirectory(0)
            # Clone() gives the copy to the current directory (in HistPlotter the open input
            # file), so PyROOT does not own it, and SetDirectory(0) leaves it owned by nobody,
            # never freed. This is ROOT.SetOwnership(total, True) without importing ROOT.
            total.__python_owns__ = True
            for p in members[1:]:
                total.Add(hists[p][0])
            name = group["name"]
            for param, value in values:
                name = name.replace("${" + param + "}", value)
            color = group.get("color", hists[members[0]][2])
            grouped[key] = (total, name, color, hists[members[0]][3])
            used.update(members)
    grouped.update((k, v) for k, v in hists.items() if k not in used)
    return grouped


def drop_empty_entries(hists):
    """The entries of `hists` (as GroupProcesses gives them) that are drawn: a background or
    signal entry whose every bin, under- and overflow included, is zero has nothing to show
    and gets neither a slice nor a legend entry. A group of which only some members are
    empty is drawn under its own name -- tt + ttV stays tt + ttV when ttV has no events.
    Data is always kept: no events observed is a measurement."""
    return {
        key: entry
        for key, entry in hists.items()
        if entry[3] == "data"
        or any(entry[0].GetBinContent(i) != 0 for i in range(entry[0].GetNcells()))
    }
