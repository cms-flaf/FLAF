"""The value of a histograms.yaml setting that can differ by channel, category and region: ``x_rebin``
(Analysis/HistPlotter.py findNewBins) and ``blind_ranges`` (Common/PlotBlinding.py). Without ROOT, so it is
unit-tested in CI.

The setting is either a value, the same for every plot, or a mapping whose keys are channel, category or
region names, in any order and nesting, with the key ``other`` for the plots no key names. At each level the
first of the plot's channel, category and region that is a key is followed, with the remaining two, or else
``other``, which is searched in the same way; a level with neither is an error (there is no backtracking: a
branch that has no value for the plot does not fall back to another key or to an outer ``other``). Any value
that is not a mapping is the result.
"""

OTHER = "other"
_KEY_NAMES = ("channel", "category", "region")


def _search(node, keys, what, path):
    if not isinstance(node, dict):
        return node
    for name, value in keys.items():
        if value in node:
            rest = {k: v for k, v in keys.items() if k != name}
            return _search(node[value], rest, what, path + (value,))
    if OTHER in node:
        return _search(node[OTHER], keys, what, path + (OTHER,))
    at = "/".join(map(str, path)) or "the top level"
    raise ValueError(
        f"{what}: no value for {keys or 'a plot without channel, category and region'}"
        f' at {at}: no key matches, and there is no "{OTHER}"'
    )


def resolve(spec, what, channel=None, category=None, region=None):
    """The value of `spec` (a setting of histograms.yaml, described by `what` in errors) for the plot
    of `channel`, `category` and `region`; a None among them is not looked up. Raises when a level on
    the way has no matching key and no ``other``, and when the value found is null."""
    given = dict(zip(_KEY_NAMES, (channel, category, region)))
    keys = {name: value for name, value in given.items() if value is not None}
    found = _search(spec, keys, what, ())
    if found is None:
        raise ValueError(f"{what}: the value for {keys} is null")
    return found


def leaves(spec, path=()):
    """(path of keys, value) of every value of `spec`, for checks that cover the keys no plot reaches;
    an empty mapping is yielded as a value."""
    if not isinstance(spec, dict) or not spec:
        yield path, spec
        return
    for key, value in spec.items():
        yield from leaves(value, path + (key,))
