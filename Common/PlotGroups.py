"""Plot groups: processes summed into one legend entry of a plot (global.yaml's
plot_process_groups). ROOT is not imported here -- a histogram only needs Clone, SetDirectory
and Add -- so the logic is unit-tested without ROOT."""

import re

# A ${PARAM} in a plot group's process pattern or name.
_GROUP_PARAM = re.compile(r"\$\{(\w+)\}")
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
    which name the same parameters, as the name does. A mistake raises a ValueError naming the
    group instead of giving a plot that looks right but is not."""
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
