import ROOT

from FLAF.Processors.MCStitching import MCStitcher

_helpers_declared = False


def _declare_helpers():
    global _helpers_declared
    if _helpers_declared:
        return
    # Templated on the vector element types because nanoAOD stores
    # GenPart_genPartIdxMother as Short_t in recent versions and Int_t in older ones.
    ok = ROOT.gInterpreter.Declare("""
    #ifndef FLAF_TT_STITCH_HELPERS
    #define FLAF_TT_STITCH_HELPERS
    namespace flaf_stitch {
    // Count W bosons that decay to a charged lepton (e/mu/tau), i.e. the number of
    // leptonically decaying W's in the event. Only the charged lepton whose direct
    // mother is the W is counted, so radiated lepton copies are not double counted.
    template <typename VecId, typename VecMother>
    int nLeptonicW(const VecId& GenPart_pdgId, const VecMother& GenPart_genPartIdxMother) {
        int n = 0;
        for (size_t i = 0; i < GenPart_pdgId.size(); ++i) {
            const int apdg = std::abs(static_cast<int>(GenPart_pdgId[i]));
            if (apdg == 11 || apdg == 13 || apdg == 15) {
                const int m = static_cast<int>(GenPart_genPartIdxMother[i]);
                if (m >= 0 && std::abs(static_cast<int>(GenPart_pdgId[m])) == 24)
                    ++n;
            }
        }
        return n;
    }
    }  // namespace flaf_stitch
    #endif
    """)
    if not ok:
        raise RuntimeError(
            "TTStitcher: failed to declare C++ helper flaf_stitch::nLeptonicW"
        )
    _helpers_declared = True


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
                "flaf_stitch::nLeptonicW(GenPart_pdgId, GenPart_genPartIdxMother)",
            )
        return super().defineVariables(df)
