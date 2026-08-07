/*! Strict generator-level identification of the Drell-Yan process.

Two independent, strict views of the event, matching the two DY stitching axes:
  - identifyLHE(): the outgoing LHE dilepton pair -> flavor (e/mu/tau) and invariant mass
    (used by the flavor and m_ll stitching axes; matches the sample generation-level cuts);
  - identifyTauTau(): the two generator-level taus and their visible decays (used by the
    Z->tautau filter axis).

Any deviation from the expected topology throws std::runtime_error. Self-contained and
templated on the vector element types.
*/

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "Math/Vector4D.h"

namespace gen_process {
    namespace dy {

        // ---- LHE-level dilepton flavor and mass ---------------------------------------

        struct DYInfo {
            int flavor;  //!< |pdgId| of the dilepton pair: 11 (e), 13 (mu) or 15 (tau).
            float mll;   //!< invariant mass of the outgoing LHE dilepton pair.
        };

        //! Identify the DY dilepton pair from the LHE record: exactly two outgoing
        //! (status == 1) charged leptons of the same flavor are required.
        template <typename VecF, typename VecId>
        DYInfo identifyLHE(const VecF& LHEPart_pt,
                           const VecF& LHEPart_eta,
                           const VecF& LHEPart_phi,
                           const VecF& LHEPart_mass,
                           const VecId& LHEPart_pdgId,
                           const VecId& LHEPart_status) {
            ROOT::Math::PtEtaPhiMVector p4;
            int flavor = 0, n = 0;
            for (std::size_t i = 0; i < LHEPart_pdgId.size(); ++i) {
                const int a = std::abs(static_cast<int>(LHEPart_pdgId[i]));
                if ((a == 11 || a == 13 || a == 15) && static_cast<int>(LHEPart_status[i]) == 1) {
                    if (n == 0)
                        flavor = a;
                    else if (a != flavor)
                        throw std::runtime_error("gen_process::dy: outgoing LHE leptons have mixed flavor");
                    p4 += ROOT::Math::PtEtaPhiMVector(LHEPart_pt[i], LHEPart_eta[i], LHEPart_phi[i], LHEPart_mass[i]);
                    ++n;
                }
            }
            if (n != 2)
                throw std::runtime_error("gen_process::dy: expected exactly 2 outgoing LHE charged leptons, found " +
                                         std::to_string(n));
            return DYInfo{flavor, static_cast<float>(p4.M())};
        }

        // ---- Generator-level tau-tau decays and Z->tautau filter ----------------------

        enum class TauVis : int { Electron = 0, Muon = 1, Hadrons = 2 };

        struct TauTauInfo {
            std::array<TauVis, 2> vis_type{{TauVis::Hadrons, TauVis::Hadrons}};
            std::array<double, 2> vis_pt{{0., 0.}};
            std::array<double, 2> vis_abseta{{0., 0.}};

            //! DY->tautau gen filter: kept final states ElHad / ElMu / HadHad / MuHad
            //! (ElEl and MuMu excluded); every visible product must have |eta| < 3.0.
            bool passFilter() const {
                if (vis_abseta[0] >= 3.0 || vis_abseta[1] >= 3.0)
                    return false;
                const int k1 = static_cast<int>(vis_type[0]), k2 = static_cast<int>(vis_type[1]);
                const auto ptOf = [&](int kind) { return k1 == kind ? vis_pt[0] : vis_pt[1]; };
                const int kmin = std::min(k1, k2), kmax = std::max(k1, k2);
                if (kmin == 0 && kmax == 1)  // ElMu
                    return ptOf(1) > 8 && ptOf(0) > 11;
                if (kmin == 0 && kmax == 2)  // ElHad
                    return ptOf(0) > 22 && ptOf(2) > 16;
                if (kmin == 1 && kmax == 2)  // MuHad
                    return ptOf(1) > 19 && ptOf(2) > 16;
                if (k1 == 2 && k2 == 2)  // HadHad
                    return vis_pt[0] > 20 && vis_pt[1] > 20;
                return false;  // ElEl, MuMu
            }
        };

        namespace detail {
            inline bool isLastCopy(int status_flags) { return (status_flags >> 13) & 1; }
            inline bool isNeutrino(int apdg) { return apdg == 12 || apdg == 14 || apdg == 16; }

            template <typename VecMother>
            std::vector<std::vector<int>> daughterMap(const VecMother& mother) {
                std::vector<std::vector<int>> daughters(mother.size());
                for (int i = 0; i < static_cast<int>(mother.size()); ++i) {
                    const int m = static_cast<int>(mother[i]);
                    if (m >= 0)
                        daughters[m].push_back(i);
                }
                return daughters;
            }

            //! Collect the final-state (no-daughter) descendants of a particle.
            inline void finalStates(int p, const std::vector<std::vector<int>>& daughters, std::vector<int>& out) {
                if (daughters[p].empty()) {
                    out.push_back(p);
                    return;
                }
                for (const int d : daughters[p])
                    finalStates(d, daughters, out);
            }
        }  // namespace detail

        //! Identify the two generator-level taus of a Z->tautau event and their visible
        //! decays. Exactly two last-copy taus are required; each tau must decay either to a
        //! single charged lepton (+ neutrinos) or to hadrons.
        template <typename VecF, typename VecId, typename VecFlags, typename VecMother>
        TauTauInfo identifyTauTau(const VecF& GenPart_pt,
                                  const VecF& GenPart_eta,
                                  const VecF& GenPart_phi,
                                  const VecF& GenPart_mass,
                                  const VecId& GenPart_pdgId,
                                  const VecFlags& GenPart_statusFlags,
                                  const VecMother& GenPart_genPartIdxMother) {
            const auto daughters = detail::daughterMap(GenPart_genPartIdxMother);
            const std::size_t n = GenPart_pdgId.size();
            const auto apdg = [&](int i) { return std::abs(static_cast<int>(GenPart_pdgId[i])); };

            std::vector<int> taus;
            for (std::size_t i = 0; i < n; ++i)
                if (apdg(static_cast<int>(i)) == 15 && detail::isLastCopy(static_cast<int>(GenPart_statusFlags[i])))
                    taus.push_back(static_cast<int>(i));
            if (taus.size() != 2)
                throw std::runtime_error("gen_process::dy: expected exactly 2 last-copy taus, found " +
                                         std::to_string(taus.size()));

            TauTauInfo info;
            for (int k = 0; k < 2; ++k) {
                std::vector<int> fs;
                detail::finalStates(taus[k], daughters, fs);
                int n_e = 0, n_mu = 0;
                ROOT::Math::PtEtaPhiMVector vis;
                for (const int p : fs) {
                    const int a = apdg(p);
                    if (a == 11)
                        ++n_e;
                    else if (a == 13)
                        ++n_mu;
                    if (!detail::isNeutrino(a))
                        vis +=
                            ROOT::Math::PtEtaPhiMVector(GenPart_pt[p], GenPart_eta[p], GenPart_phi[p], GenPart_mass[p]);
                }
                if (n_e + n_mu > 1)
                    throw std::runtime_error("gen_process::dy: tau decay with more than one charged lepton");
                if (n_e == 1)
                    info.vis_type[k] = TauVis::Electron;
                else if (n_mu == 1)
                    info.vis_type[k] = TauVis::Muon;
                else
                    info.vis_type[k] = TauVis::Hadrons;
                info.vis_pt[k] = vis.pt();
                info.vis_abseta[k] = std::abs(vis.eta());
            }
            return info;
        }

    }  // namespace dy
}  // namespace gen_process
