import os

from FLAF.Processors.MCStitching import MCStitcher


def _declare_helpers():
    from FLAF.Common.Utilities import DeclareHeader

    flaf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DeclareHeader(os.path.join(flaf_dir, "include", "GenProcess", "DY.h"))


class DYMllStitcher(MCStitcher):
    """MCStitcher that adds dilepton flavor and mass axes to the DY Vpt/NpNLO stitching,
    so the whole DY background can be handled by a single stitched process: the flavor
    axis splits the inclusive (all-flavor) DY into e/mu/tau, and a mass-window sample
    (e.g. DYto2Mu_MLL_105to160) is stitched into the mu-mu bins via the mass axis.

    Defines ``LHE_dilep_flavor`` (11/13/15) and ``LHE_mll`` from the strict LHE-level DY
    identification in ``FLAF/include/GenProcess/DY.h``.
    """

    def defineVariables(self, df):
        _declare_helpers()
        if "_dy_lhe_info" not in df.GetColumnNames():
            df = df.Define(
                "_dy_lhe_info",
                "gen_process::dy::identifyLHE(LHEPart_pt, LHEPart_eta, LHEPart_phi, "
                "LHEPart_mass, LHEPart_pdgId, LHEPart_status)",
            )
        if "LHE_dilep_flavor" not in df.GetColumnNames():
            df = df.Define("LHE_dilep_flavor", "_dy_lhe_info.flavor")
        if "LHE_mll" not in df.GetColumnNames():
            df = df.Define("LHE_mll", "_dy_lhe_info.mll")
        return super().defineVariables(df)
