"""Data-driven backgrounds: processes whose histograms an analysis estimator computes, at the end
of HistMergerTask, from the merged histograms of the other processes (data minus simulation in
control regions, transfer factors, ...). Such a process is declared in processes.yaml without
datasets and listed among the backgrounds of the physics model like any other one::

    Fakes:
      name: Fakes
      color: kGray+1
      data_driven:
        estimator: Analysis.SL_fakes.estimate   # <module>.<function>, imported from ANALYSIS_PATH
        nuisances:                              # optional: the estimator's own uncertainty sources,
          FakesStat: CMS_bbww_fakes_stat_{channel}_{category}_{}   # {source: name template}
        # any other key: a parameter of the estimator (ctx.config)

HistMergerFromHists calls ``estimate(ctx)`` once per data-driven process and uncertainty source,
after every process is merged, on the merged file of one variable of one era, when data was
merged and the merge is not restricted to a subset of the processes or datasets. The estimator
returns ``{((channel, region, category), (unc, scale)): hist}``: for the variations of the
current source, and, at the nominal pass, Up/Down of its own nuisances, named by their template:
``{}`` is Up/Down and comes last (the era sum and the plots recognise a variation by its name's
ending), and ``{channel}``, ``{region}``, ``{category}`` and ``{era}`` are the fields of
the histogram (to decorrelate a nuisance, e.g. a statistical one per channel and category). They
are written as that process, so the era sum, the plots and the datacards treat it as any other
background. This module does not import ROOT: Setup validates the declarations.
"""

import importlib
import string

TEMPLATE_FIELDS = ("channel", "region", "category", "era")


def validate_data_driven(process_name, process, process_type):
    """The `data_driven` block of a process of the physics model, checked."""
    cfg = process["data_driven"]
    where = f"process '{process_name}': data_driven"
    if not isinstance(cfg, dict):
        raise RuntimeError(f"{where} must be a mapping")
    if process_type != "backgrounds":
        raise RuntimeError(
            f"{where}: a data-driven process is a background; the physics model lists it "
            f"under '{process_type}'"
        )
    for key in ("datasets", "sub_processes"):
        if key in process:
            raise RuntimeError(f"{where}: a data-driven process has no '{key}'")
    if "color" not in process:
        raise RuntimeError(f"process '{process_name}': a 'color' is needed to plot it")
    estimator = cfg.get("estimator")
    if (
        not isinstance(estimator, str)
        or "." not in estimator
        or not all(estimator.split("."))
    ):
        raise RuntimeError(
            f"{where}: 'estimator' must name a function as <module>.<function>, "
            f"got {estimator!r}"
        )
    nuisances = cfg.get("nuisances", {})
    if not isinstance(nuisances, dict) or not all(
        isinstance(k, str) and k != "Central" and _template_ok(v)
        for k, v in nuisances.items()
    ):
        raise RuntimeError(
            f"{where}: 'nuisances' must map a source (not 'Central') to a histogram-name "
            f"template ending with '{{}}' (Up/Down), optionally with the fields "
            f"{list(TEMPLATE_FIELDS)}, got {nuisances!r}"
        )
    return cfg


def _template_ok(template):
    """One bare '{}', at the end, and otherwise only the bare fields of TEMPLATE_FIELDS."""
    if not isinstance(template, str):
        return False
    try:
        segments = list(string.Formatter().parse(template))
    except ValueError:
        return False
    fields = [(f, spec, conv) for _, f, spec, conv in segments if f is not None]
    if not segments or segments[-1][1] != "" or segments[-1][2:] != ("", None):
        return False
    return [f for f, _, _ in fields].count("") == 1 and all(
        f in ("",) + TEMPLATE_FIELDS and spec == "" and conv is None
        for f, spec, conv in fields
    )


def load_estimator(path):
    """The estimator function named <module>.<function>."""
    module_name, function_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), function_name)


def nuisance_hist_name(process, template, scale, key_dir, era):
    """The name of an estimator nuisance's histogram: <process>_<template>, with Up/Down and the
    fields of the histogram at ``key_dir`` (channel, region, category) of ``era``."""
    channel, region, category = key_dir[:3]
    fields = dict(channel=channel, region=region, category=category, era=era)
    return f"{process}_{template.format(scale, **fields)}"
