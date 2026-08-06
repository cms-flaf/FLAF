/*! Generator-level process / decay classification helpers.

Small, reusable functions that classify the generated process from the nanoAOD
`GenPart` / `LHEPart` collections (decay modes, dilepton flavor/mass, gen filters).
Originally factored out of the MC stitching processors (FLAF issue #169) so they can
be reused elsewhere. All functions are templated on the vector element types because
nanoAOD stores e.g. `GenPart_genPartIdxMother` as `Short_t` in recent versions and
`Int_t` in older ones.
*/

#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"

// `passDYtautauFilter` uses `reco_tau::gen_truth::GenLepton`; include "GenLepton.h"
// before this header (the FLAF producer already declares it via InitializeCorrections).

namespace gen_process {

    //! Number of W bosons that decay to a charged lepton (e/mu/tau), i.e. the number of
    //! leptonically decaying W's in the event. Only the charged lepton whose direct mother
    //! is the W is counted, so radiated lepton copies are not double counted.
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

    //! Invariant mass of the two outgoing LHE charged leptons (e.g. the DY dilepton mass),
    //! or -1 if fewer than two are found.
    template <typename VecF, typename VecId>
    float LHEDileptonMass(const VecF& LHEPart_pt,
                          const VecF& LHEPart_eta,
                          const VecF& LHEPart_phi,
                          const VecF& LHEPart_mass,
                          const VecId& LHEPart_pdgId,
                          const VecId& LHEPart_status) {
        ROOT::Math::PtEtaPhiMVector p4;
        int n = 0;
        for (size_t i = 0; i < LHEPart_pdgId.size(); ++i) {
            const int apdg = std::abs(static_cast<int>(LHEPart_pdgId[i]));
            if ((apdg == 11 || apdg == 13 || apdg == 15) && static_cast<int>(LHEPart_status[i]) == 1) {
                p4 += ROOT::Math::PtEtaPhiMVector(LHEPart_pt[i], LHEPart_eta[i], LHEPart_phi[i], LHEPart_mass[i]);
                ++n;
            }
        }
        return n >= 2 ? static_cast<float>(p4.M()) : -1.f;
    }

    //! Flavor of the outgoing LHE dilepton pair: |pdgId| of the LHE charged lepton
    //! (11 = e, 13 = mu, 15 = tau), or 0 if none is found.
    template <typename VecId>
    int LHEDileptonFlavor(const VecId& LHEPart_pdgId, const VecId& LHEPart_status) {
        for (size_t i = 0; i < LHEPart_pdgId.size(); ++i) {
            const int apdg = std::abs(static_cast<int>(LHEPart_pdgId[i]));
            if ((apdg == 11 || apdg == 13 || apdg == 15) && static_cast<int>(LHEPart_status[i]) == 1)
                return apdg;
        }
        return 0;
    }

    //! Reproduces the generator-level DY->tautau filter used for the filtered sample.
    //! Kept final states: ElHad, ElMu, HadHad, MuHad (ElEl and MuMu are excluded); every
    //! visible tau-decay product must satisfy |eta| < 3.0. Requires GenLepton.h.
    template <typename VecF, typename VecId, typename VecMother, typename VecFlags>
    bool passDYtautauFilter(const VecF& GenPart_pt,
                            const VecF& GenPart_eta,
                            const VecF& GenPart_phi,
                            const VecF& GenPart_mass,
                            const VecMother& GenPart_genPartIdxMother,
                            const VecId& GenPart_pdgId,
                            const VecFlags& GenPart_statusFlags,
                            unsigned long long event) {
        using GenLepton = reco_tau::gen_truth::GenLepton;
        auto genLeptons = GenLepton::fromNanoAOD(GenPart_pt,
                                                 GenPart_eta,
                                                 GenPart_phi,
                                                 GenPart_mass,
                                                 GenPart_genPartIdxMother,
                                                 GenPart_pdgId,
                                                 GenPart_statusFlags,
                                                 event);
        // Visible tau decays, encoded as kind (0 = e, 1 = mu, 2 = hadrons), pt, |eta|.
        std::vector<std::tuple<int, double, double>> taus;
        for (const auto& gl : genLeptons) {
            int kind;
            switch (gl.kind()) {
                case GenLepton::Kind::TauDecayedToElectron:
                    kind = 0;
                    break;
                case GenLepton::Kind::TauDecayedToMuon:
                    kind = 1;
                    break;
                case GenLepton::Kind::TauDecayedToHadrons:
                    kind = 2;
                    break;
                default:
                    continue;
            }
            const auto& p4 = gl.visibleP4();
            taus.emplace_back(kind, p4.pt(), std::abs(p4.eta()));
        }
        if (taus.size() != 2)
            return false;
        const int k1 = std::get<0>(taus[0]), k2 = std::get<0>(taus[1]);
        const double pt1 = std::get<1>(taus[0]), pt2 = std::get<1>(taus[1]);
        if (std::get<2>(taus[0]) >= 3.0 || std::get<2>(taus[1]) >= 3.0)
            return false;

        auto ptOf = [&](int kind) { return k1 == kind ? pt1 : pt2; };
        const int kmin = std::min(k1, k2), kmax = std::max(k1, k2);
        if (kmin == 0 && kmax == 1)  // ElMu
            return ptOf(1) > 8 && ptOf(0) > 11;
        if (kmin == 0 && kmax == 2)  // ElHad
            return ptOf(0) > 22 && ptOf(2) > 16;
        if (kmin == 1 && kmax == 2)  // MuHad
            return ptOf(1) > 19 && ptOf(2) > 16;
        if (k1 == 2 && k2 == 2)  // HadHad
            return pt1 > 20 && pt2 > 20;
        return false;  // ElEl, MuMu -> not in Final_States
    }

}  // namespace gen_process
