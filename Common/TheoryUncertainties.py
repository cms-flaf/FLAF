"""Uncertainties stored as a family of members (PDF members, QCD-scale points).

A normalisation uncertainty of `weights.yaml` normally has two scales, Up and Down. A theory
uncertainty instead has one weight per member: the 103 PDF members, the 8 (muR, muF) points of
the QCD scale, as the corrections configuration registers them (``getScales``). Like every other
uncertainty, its members are filled for every variable when the uncertainties are computed, and
HistMerger writes the summed histogram of every member of every process. They are only
meaningful together, bin by bin, so reducing them to Up/Down (envelope, Hessian sum) is left to
the datacard step, which reads the members of the era (or era group) it fits.

    norm:
      qcd_scale:
        expression: weight_base_qcd_scale{scale}_rel * final_weight
        name: QCDscale_{process}_{}

``{process}`` in ``name`` is the process of the physics model, so each process (each DY mass
range, each signal) has its own. A member histogram is ``<process>_<name>`` with the member in
place of ``{}`` and its leading underscore dropped: ``TT_QCDscale_TT_muR2_muF1``,
``TT_pdf_TT_57``.
"""


def has_members(unc_scales, updown_scales):
    """Whether an uncertainty has members rather than the ordinary Up/Down scales."""
    return list(unc_scales) != list(updown_scales)


def format_name(unc_entry, scale, process):
    """The histogram suffix of one scale of an uncertainty: QCDscale_DY_M_50_Up,
    QCDscale_TT_muR2_muF1, pdf_TT_57."""
    return unc_entry["name"].format(scale.lstrip("_"), process=process)
