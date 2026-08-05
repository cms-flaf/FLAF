import ROOT

from FLAF.Processors.MCStitching import MCStitcher

_helpers_declared = False


def _declare_helpers():
    global _helpers_declared
    if _helpers_declared:
        return
    # Reproduces the generator-level DY->tautau filter used to produce the filtered
    # sample. Kept final states: ElHad, ElMu, HadHad, MuHad (ElEl and MuMu are excluded,
    # matching the sample's Final_States). Cuts are on the visible tau-decay products;
    # |eta| < 3.0 is required for every visible object.
    ok = ROOT.gInterpreter.Declare("""
    #ifndef FLAF_DY_TAUTAU_STITCH_HELPERS
    #define FLAF_DY_TAUTAU_STITCH_HELPERS
    namespace flaf_stitch {
    template <typename VecF, typename VecId, typename VecMother, typename VecFlags>
    bool passDYtautauFilter(const VecF& GenPart_pt, const VecF& GenPart_eta,
                            const VecF& GenPart_phi, const VecF& GenPart_mass,
                            const VecMother& GenPart_genPartIdxMother,
                            const VecId& GenPart_pdgId, const VecFlags& GenPart_statusFlags,
                            unsigned long long event) {
        using GenLepton = reco_tau::gen_truth::GenLepton;
        auto genLeptons = GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta, GenPart_phi,
                                                 GenPart_mass, GenPart_genPartIdxMother,
                                                 GenPart_pdgId, GenPart_statusFlags, event);
        // Visible tau decays, encoded as kind (0 = e, 1 = mu, 2 = hadrons), pt, |eta|.
        std::vector<std::tuple<int, double, double>> taus;
        for (const auto& gl : genLeptons) {
            int kind;
            switch (gl.kind()) {
                case GenLepton::Kind::TauDecayedToElectron: kind = 0; break;
                case GenLepton::Kind::TauDecayedToMuon:     kind = 1; break;
                case GenLepton::Kind::TauDecayedToHadrons:  kind = 2; break;
                default: continue;
            }
            const auto& p4 = gl.visibleP4();
            taus.emplace_back(kind, p4.pt(), std::abs(p4.eta()));
        }
        if (taus.size() != 2) return false;
        const int k1 = std::get<0>(taus[0]), k2 = std::get<0>(taus[1]);
        const double pt1 = std::get<1>(taus[0]), pt2 = std::get<1>(taus[1]);
        if (std::get<2>(taus[0]) >= 3.0 || std::get<2>(taus[1]) >= 3.0) return false;

        auto ptOf = [&](int kind) {
            return k1 == kind ? pt1 : pt2;
        };
        const int kmin = std::min(k1, k2), kmax = std::max(k1, k2);
        if (kmin == 0 && kmax == 1)  // ElMu
            return ptOf(1) > 8 && ptOf(0) > 11;
        if (kmin == 0 && kmax == 2)  // ElHad
            return ptOf(0) > 22 && ptOf(2) > 16;
        if (kmin == 1 && kmax == 2)  // MuHad
            return ptOf(1) > 19 && ptOf(2) > 16;
        if (k1 == 2 && k2 == 2)      // HadHad
            return pt1 > 20 && pt2 > 20;
        return false;  // ElEl, MuMu -> not in Final_States
    }
    }  // namespace flaf_stitch
    #endif
    """)
    if not ok:
        raise RuntimeError(
            "DYtautauStitcher: failed to declare C++ helper flaf_stitch::passDYtautauFilter"
        )
    _helpers_declared = True


class DYtautauStitcher(MCStitcher):
    """MCStitcher that adds a Z->tautau gen-filter axis to the DY Vpt/NpNLO stitching, so
    the tau-tau filtered sample can be stitched in on top of the inclusive DY (for
    HH_bbtautau).

    Defines ``DY_tautau_filter`` (1 if the event passes the gen-level tau-tau filter,
    else 0) which the stitching bins select on.
    """

    def defineVariables(self, df):
        _declare_helpers()
        if "DY_tautau_filter" not in df.GetColumnNames():
            df = df.Define(
                "DY_tautau_filter",
                "flaf_stitch::passDYtautauFilter(GenPart_pt, GenPart_eta, GenPart_phi, "
                "GenPart_mass, GenPart_genPartIdxMother, GenPart_pdgId, "
                "GenPart_statusFlags, event) ? 1 : 0",
            )
        return super().defineVariables(df)
