import os

from FLAF.Processors.MCStitching import MCStitcher, defineFromStoredOrExpression


def _declare_helpers():
    from FLAF.Common.Utilities import DeclareHeader

    flaf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DeclareHeader(os.path.join(flaf_dir, "include", "GenProcess", "TT.h"))


def _prepare(df):
    _declare_helpers()
    return df


class TTStitcher(MCStitcher):
    """Stitch the inclusive ttbar sample with the decay-channel samples
    (2L2Nu / LNu2Q / 4Q) by the number of leptonically decaying W bosons.

    Defines ``TT_n_leptonic_W`` (0, 1 or 2) from ``TTInfo_nLeptonicW`` when the anaTuple
    stores it, otherwise from the strict gen-level ttbar identification in
    ``FLAF/include/GenProcess/TT.h``.
    """

    def defineVariables(self, df):
        df = defineFromStoredOrExpression(
            df,
            "TT_n_leptonic_W",
            stored="TTInfo_nLeptonicW",
            expression="gen_process::tt::identify(GenPart_pdgId, GenPart_statusFlags, "
            "GenPart_genPartIdxMother).nLeptonicW()",
            prepare=_prepare,
        )
        return super().defineVariables(df)
