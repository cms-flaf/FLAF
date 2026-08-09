import os

from FLAF.Processors.MCStitching import MCStitcher


def _declare_helpers():
    from FLAF.Common.Utilities import DeclareHeader

    flaf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DeclareHeader(os.path.join(flaf_dir, "include", "GenProcess", "DY.h"))


class DYtautauStitcher(MCStitcher):
    """MCStitcher that adds a Z->tautau gen-filter axis to the DY Vpt/NpNLO stitching, so
    the tau-tau filtered sample can be stitched in on top of the inclusive DY (for
    HH_bbtautau).

    Defines ``DY_tautau_filter`` (1 if the event passes the gen-level tau-tau filter,
    else 0) from the strict tau-tau identification in ``FLAF/include/GenProcess/DY.h``.
    """

    def defineVariables(self, df):
        _declare_helpers()
        if "DY_tautau_filter" not in df.GetColumnNames():
            df = df.Define(
                "DY_tautau_filter",
                "gen_process::dy::identifyTauTau(GenPart_pt, GenPart_eta, GenPart_phi, "
                "GenPart_mass, GenPart_pdgId, GenPart_statusFlags, "
                "GenPart_genPartIdxMother).passFilter() ? 1 : 0",
            )
        return super().defineVariables(df)
