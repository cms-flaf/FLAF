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
            inline bool isHardProcess(int status_flags) { return (status_flags >> 7) & 1; }
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
        //! decays. This reproduces the CMSSW DY->tautau generator filter
        //! EmbeddingHepMCFilter (GeneratorInterface/Core/src/EmbeddingHepMCFilter.cc) on
        //! the stored GenPart collection:
        //!   - the two taus are the hard-process taus. CMSSW selects the taus whose direct
        //!     mother is the Z (pdgId 23), but that mother link is absent for ~5% of events
        //!     in BOTH central nanoAOD and the full-gen HLepRare skims (the amcatnlo ME
        //!     proceeds via gamma*/Z), so the hard-process flag is used instead -- it yields
        //!     exactly two taus for every event tested. This matches the filter's
        //!     IncludeDY=true fallback (isFirstCopy && fromHardProcess);
        //!   - the decay mode is taken from the tau's last copy: a direct charged-lepton
        //!     daughter -> e/mu, otherwise hadronic;
        //!   - the visible momentum is the hard-process tau's own 4-momentum minus its
        //!     neutrinos. By momentum conservation this equals the FSR-dressed sum of the
        //!     visible decay products (decay_and_sump4Vis's status==1 non-neutrino sum), but
        //!     is robust to how the showered decay products are stored: an explicit sum over
        //!     final-state descendants silently under-counts hadronic products on the
        //!     full-gen HLepRare record (and mildly over-counts on pruned central nanoAOD).
        //!
        //! Validated on the DYto2Tau *_Filtered samples (events that passed the production
        //! filter): 99.2% are recovered, identically on central nanoAOD and on the full-gen
        //! HLepRare skims. The residual ~0.8% is irreducible from any stored gen record --
        //! the filter ran on the full HepMC GenEvent at generation time, and the soft
        //! products it summed were already dropped by the gen pruning applied before miniAOD
        //! (hence central nanoAOD and the fuller HLepRare skims agree to <0.01%).
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
            const auto flags = [&](int i) { return static_cast<int>(GenPart_statusFlags[i]); };

            std::vector<int> taus;
            for (std::size_t i = 0; i < n; ++i)
                if (apdg(static_cast<int>(i)) == 15 && detail::isHardProcess(flags(static_cast<int>(i))))
                    taus.push_back(static_cast<int>(i));
            if (taus.size() != 2)
                throw std::runtime_error("gen_process::dy: expected exactly 2 hard-process taus, found " +
                                         std::to_string(taus.size()));

            TauTauInfo info;
            for (int k = 0; k < 2; ++k) {
                // Resolve the hard-process tau to its last copy (walk down tau copies).
                int tau = taus[k];
                while (!detail::isLastCopy(flags(tau))) {
                    int next = -1;
                    for (const int d : daughters[tau])
                        if (apdg(d) == 15) {
                            next = d;
                            break;
                        }
                    if (next < 0)
                        throw std::runtime_error("gen_process::dy: could not resolve the tau last copy");
                    tau = next;
                }

                // Decay mode from the last copy's direct daughters (conversions deeper in
                // the chain are ignored).
                int lepton = -1;
                for (const int d : daughters[tau]) {
                    const int a = apdg(d);
                    if (a == 11 || a == 13) {
                        if (lepton >= 0)
                            throw std::runtime_error("gen_process::dy: tau with two charged-lepton daughters");
                        lepton = d;
                    }
                }
                // Visible p4 = the hard-process tau's own momentum minus its neutrinos.
                // This is robust to how the showered decay products are stored (see the
                // note above); by momentum conservation it is the FSR-dressed visible.
                std::vector<int> fs;
                detail::finalStates(taus[k], daughters, fs);
                ROOT::Math::PtEtaPhiMVector nu;
                for (const int p : fs)
                    if (detail::isNeutrino(apdg(p)))
                        nu +=
                            ROOT::Math::PtEtaPhiMVector(GenPart_pt[p], GenPart_eta[p], GenPart_phi[p], GenPart_mass[p]);
                const ROOT::Math::PtEtaPhiMVector hard(
                    GenPart_pt[taus[k]], GenPart_eta[taus[k]], GenPart_phi[taus[k]], GenPart_mass[taus[k]]);
                const ROOT::Math::PtEtaPhiMVector vis = hard - nu;

                if (lepton < 0)
                    info.vis_type[k] = TauVis::Hadrons;
                else
                    info.vis_type[k] = apdg(lepton) == 11 ? TauVis::Electron : TauVis::Muon;
                info.vis_pt[k] = vis.pt();
                info.vis_abseta[k] = std::abs(vis.eta());
            }
            return info;
        }

    }  // namespace dy
}  // namespace gen_process
