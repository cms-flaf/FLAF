import ROOT

from FLAF.Processors.MCStitching import MCStitcher

_helpers_declared = False


def _declare_helpers():
    global _helpers_declared
    if _helpers_declared:
        return
    ok = ROOT.gInterpreter.Declare("""
    #ifndef FLAF_DY_MLL_STITCH_HELPERS
    #define FLAF_DY_MLL_STITCH_HELPERS
    #include "Math/Vector4D.h"
    namespace flaf_stitch {
    template <typename VecF, typename VecId>
    float LHEDileptonMass(const VecF& LHEPart_pt, const VecF& LHEPart_eta,
                          const VecF& LHEPart_phi, const VecF& LHEPart_mass,
                          const VecId& LHEPart_pdgId, const VecId& LHEPart_status) {
        ROOT::Math::PtEtaPhiMVector p4;
        int n = 0;
        for (size_t i = 0; i < LHEPart_pdgId.size(); ++i) {
            const int apdg = std::abs(static_cast<int>(LHEPart_pdgId[i]));
            if ((apdg == 11 || apdg == 13 || apdg == 15)
                    && static_cast<int>(LHEPart_status[i]) == 1) {
                p4 += ROOT::Math::PtEtaPhiMVector(LHEPart_pt[i], LHEPart_eta[i],
                                                  LHEPart_phi[i], LHEPart_mass[i]);
                ++n;
            }
        }
        return n >= 2 ? static_cast<float>(p4.M()) : -1.f;
    }
    }  // namespace flaf_stitch
    #endif
    """)
    if not ok:
        raise RuntimeError(
            "DYMllStitcher: failed to declare C++ helper flaf_stitch::LHEDileptonMass"
        )
    _helpers_declared = True


class DYMllStitcher(MCStitcher):
    """MCStitcher that adds a dilepton-mass axis to the DY Vpt/NpNLO stitching, so a
    mass-window sample (e.g. DYto2Mu_MLL_105to160) can be stitched in on top of the
    inclusive DY.

    Defines ``LHE_mll`` (the invariant mass of the two outgoing LHE charged leptons)
    which the stitching bins select on.
    """

    def defineVariables(self, df):
        _declare_helpers()
        df = df.Define(
            "LHE_mll",
            "flaf_stitch::LHEDileptonMass(LHEPart_pt, LHEPart_eta, LHEPart_phi, "
            "LHEPart_mass, LHEPart_pdgId, LHEPart_status)",
        )
        return super().defineVariables(df)
