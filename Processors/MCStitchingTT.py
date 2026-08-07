import os

from FLAF.Processors.MCStitching import MCStitcher


def _declare_helpers():
    from FLAF.Common.Utilities import DeclareHeader

    flaf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DeclareHeader(os.path.join(flaf_dir, "include", "GenProcess", "TT.h"))


class TTStitcher(MCStitcher):
    """Stitch the inclusive ttbar sample with the decay-channel samples
    (2L2Nu / LNu2Q / 4Q) by the number of leptonically decaying W bosons.

    Defines ``TT_n_leptonic_W`` (0, 1 or 2) from the strict gen-level ttbar
    identification in ``FLAF/include/GenProcess/TT.h``.
    """

    def defineVariables(self, df):
        _declare_helpers()
        if "TT_n_leptonic_W" not in df.GetColumnNames():
            df = df.Define(
                "TT_n_leptonic_W",
                "gen_process::tt::identify(GenPart_pdgId, GenPart_statusFlags, "
                "GenPart_genPartIdxMother).nLeptonicW()",
            )
        return super().defineVariables(df)
