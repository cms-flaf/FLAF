import os

from FLAF.Processors.MCStitching import MCStitcher


def _declare_helpers():
    from FLAF.Common.Utilities import DeclareHeader

    flaf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DeclareHeader(os.path.join(flaf_dir, "include", "GenProcess.h"))


class TTStitcher(MCStitcher):
    """Stitch the inclusive ttbar sample with the decay-channel samples
    (2L2Nu / LNu2Q / 4Q) by the number of leptonically decaying W bosons.

    Defines ``TT_n_leptonic_W`` (0, 1 or 2) from the generator-level W -> lepton
    decays, which the stitching bins select on.
    """

    def defineVariables(self, df):
        _declare_helpers()
        if "TT_n_leptonic_W" not in df.GetColumnNames():
            df = df.Define(
                "TT_n_leptonic_W",
                "gen_process::nLeptonicW(GenPart_pdgId, GenPart_genPartIdxMother)",
            )
        return super().defineVariables(df)
