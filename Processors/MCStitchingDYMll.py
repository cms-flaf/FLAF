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
    // Flavor of the DY dilepton pair: |pdgId| of the outgoing LHE charged lepton
    // (11 = e, 13 = mu, 15 = tau), or 0 if none is found.
    template <typename VecId>
    int LHEDileptonFlavor(const VecId& LHEPart_pdgId, const VecId& LHEPart_status) {
        for (size_t i = 0; i < LHEPart_pdgId.size(); ++i) {
            const int apdg = std::abs(static_cast<int>(LHEPart_pdgId[i]));
            if ((apdg == 11 || apdg == 13 || apdg == 15)
                    && static_cast<int>(LHEPart_status[i]) == 1)
                return apdg;
        }
        return 0;
    }
    }  // namespace flaf_stitch
    #endif
    """)
    if not ok:
        raise RuntimeError(
            "DYMllStitcher: failed to declare C++ helpers flaf_stitch::LHEDilepton*"
        )
    _helpers_declared = True


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
                "flaf_stitch::LHEDileptonFlavor(LHEPart_pdgId, LHEPart_status)",
            )
        if "LHE_mll" not in df.GetColumnNames():
            df = df.Define(
                "LHE_mll",
                "flaf_stitch::LHEDileptonMass(LHEPart_pt, LHEPart_eta, LHEPart_phi, "
                "LHEPart_mass, LHEPart_pdgId, LHEPart_status)",
            )
        return super().defineVariables(df)
