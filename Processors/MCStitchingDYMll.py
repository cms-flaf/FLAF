import os

from FLAF.Processors.MCStitching import MCStitcher


def _declare_helpers():
    from FLAF.Common.Utilities import DeclareHeader

    flaf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DeclareHeader(os.path.join(flaf_dir, "include", "GenProcess.h"))


class DYMllStitcher(MCStitcher):
    """MCStitcher that adds dilepton flavor and mass axes to the DY Vpt/NpNLO stitching,
    so the whole DY background can be handled by a single stitched process: the flavor
    axis splits the inclusive (all-flavor) DY into e/mu/tau, and a mass-window sample
    (e.g. DYto2Mu_MLL_105to160) is stitched into the mu-mu bins via the mass axis.

    Defines ``LHE_dilep_flavor`` (11/13/15) and ``LHE_mll`` which the bins select on.
    """

    def defineVariables(self, df):
        _declare_helpers()
        if "LHE_dilep_flavor" not in df.GetColumnNames():
            df = df.Define(
                "LHE_dilep_flavor",
                "gen_process::LHEDileptonFlavor(LHEPart_pdgId, LHEPart_status)",
            )
        if "LHE_mll" not in df.GetColumnNames():
            df = df.Define(
                "LHE_mll",
                "gen_process::LHEDileptonMass(LHEPart_pt, LHEPart_eta, LHEPart_phi, "
                "LHEPart_mass, LHEPart_pdgId, LHEPart_status)",
            )
        return super().defineVariables(df)
