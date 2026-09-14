/*! Strict generator-level identification of the ttbar process.

Verifies the expected topology (exactly two last-copy tops, one t and one tbar; each
top -> W b; each W -> l nu or q q') and returns a struct describing the two W decays. When
the GenPart kinematics are passed as well, the struct also carries the four-momenta of the
two last-copy tops and of their b quarks; the GenPart index of each leptonic W's charged
lepton is always filled.
Any deviation from the expected topology throws std::runtime_error so unexpected cases
are surfaced rather than silently mis-classified.

Self-contained and templated on the vector element types (nanoAOD stores
GenPart_genPartIdxMother/statusFlags as Short_t/UShort_t in recent versions, Int_t in
older ones).
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
namespace tt {

    //! Decay of a W boson: to a charged lepton (value = |pdgId| of the lepton) or hadrons.
    enum class WDecay : int { ToElectron = 11, ToMuon = 13, ToTau = 15, ToHadrons = 0 };

    struct TTInfo {
        //! Decay of each of the two W bosons.
        std::array<WDecay, 2> w_decay{{WDecay::ToHadrons, WDecay::ToHadrons}};

        // The members below are indexed by charge rather than by GenPart position: [0] is
        // always the top and what it decays to, [1] the anti-top and what it decays to.

        //! Four-momentum of the last-copy top and anti-top -- after radiation and before
        //! decay, which is what the top pT reweighting is defined on. Zero unless identify()
        //! is given the GenPart kinematics.
        std::array<ROOT::Math::PtEtaPhiMVector, 2> top_p4{};

        //! Four-momentum of the b quark each top decays to, as the direct top daughter (before
        //! FSR). Zero unless identify() is given the GenPart kinematics.
        std::array<ROOT::Math::PtEtaPhiMVector, 2> b_p4{};

        //! GenPart index of the charged lepton (e, mu or tau) from each top's W: the direct
        //! daughter of the W's last copy, -1 when that W decays to hadrons. Resolving it to a
        //! gen lepton, e.g. with reco_tau::gen_truth::findLeptonByIndex, is left to the caller.
        std::array<int, 2> lep_index{{-1, -1}};

        //! Number of leptonically decaying W's (0, 1 or 2); tau counts as leptonic.
        int nLeptonicW() const {
            int n = 0;
            for (const auto d : w_decay)
                if (d != WDecay::ToHadrons)
                    ++n;
            return n;
        }
    };

    namespace detail {
        inline bool isLastCopy(int status_flags) {
            return (status_flags >> 13) & 1;  // kIsLastCopy
        }

        template <typename VecMother>
        std::vector<std::vector<int>> daughterMap(const VecMother& GenPart_genPartIdxMother) {
            std::vector<std::vector<int>> daughters(GenPart_genPartIdxMother.size());
            for (int i = 0; i < static_cast<int>(GenPart_genPartIdxMother.size()); ++i) {
                const int m = static_cast<int>(GenPart_genPartIdxMother[i]);
                if (m >= 0)
                    daughters[m].push_back(i);
            }
            return daughters;
        }

        //! The GenPart kinematic columns, bundled so identify() can take them optionally.
        template <typename VecF>
        struct Kinematics {
            const VecF& pt;
            const VecF& eta;
            const VecF& phi;
            const VecF& mass;

            ROOT::Math::PtEtaPhiMVector p4(int i) const { return {pt[i], eta[i], phi[i], mass[i]}; }
        };
    }  // namespace detail

    template <typename VecId, typename VecFlags, typename VecMother, typename VecF = std::vector<float>>
    TTInfo identify(const VecId& GenPart_pdgId,
                    const VecFlags& GenPart_statusFlags,
                    const VecMother& GenPart_genPartIdxMother,
                    const detail::Kinematics<VecF>* kinematics = nullptr) {
        const auto daughters = detail::daughterMap(GenPart_genPartIdxMother);
        const std::size_t n = GenPart_pdgId.size();
        const auto apdg = [&](int i) { return std::abs(static_cast<int>(GenPart_pdgId[i])); };
        const auto flags = [&](int i) { return static_cast<int>(GenPart_statusFlags[i]); };

        // 1. Exactly two last-copy tops, one top and one anti-top.
        std::vector<int> tops;
        for (std::size_t i = 0; i < n; ++i)
            if (apdg(static_cast<int>(i)) == 6 && detail::isLastCopy(flags(static_cast<int>(i))))
                tops.push_back(static_cast<int>(i));
        if (tops.size() != 2)
            throw std::runtime_error("gen_process::tt: expected exactly 2 last-copy tops, found "
                                     + std::to_string(tops.size()));
        if (static_cast<int>(GenPart_pdgId[tops[0]]) * static_cast<int>(GenPart_pdgId[tops[1]]) >= 0)
            throw std::runtime_error("gen_process::tt: expected one top and one anti-top");

        TTInfo info;
        for (int k = 0; k < 2; ++k) {
            const int top = tops[k];
            const std::size_t slot = static_cast<int>(GenPart_pdgId[top]) > 0 ? 0 : 1;
            if (kinematics != nullptr)
                info.top_p4[slot] = kinematics->p4(top);

            // 2. top -> W b (ISR/FSR gluons and photons allowed).
            int w = -1, b = -1;
            for (const int d : daughters[top]) {
                const int a = apdg(d);
                if (a == 24) {
                    if (w >= 0)
                        throw std::runtime_error("gen_process::tt: top with two W daughters");
                    w = d;
                } else if (a == 5) {
                    if (b >= 0)
                        throw std::runtime_error("gen_process::tt: top with two b daughters");
                    b = d;
                } else if (a != 21 && a != 22) {
                    throw std::runtime_error("gen_process::tt: unexpected top daughter pdgId "
                                             + std::to_string(static_cast<int>(GenPart_pdgId[d])));
                }
            }
            if (w < 0)
                throw std::runtime_error("gen_process::tt: top without a W daughter");
            if (b < 0)
                throw std::runtime_error("gen_process::tt: top without a b daughter");
            if (kinematics != nullptr)
                info.b_p4[slot] = kinematics->p4(b);

            // 3. Resolve the W to its last copy, then require W -> l nu or W -> q q'.
            int w_last = w;
            while (!detail::isLastCopy(flags(w_last))) {
                int next = -1;
                for (const int d : daughters[w_last]) {
                    if (apdg(d) == 24) {
                        if (next >= 0)
                            throw std::runtime_error("gen_process::tt: W with multiple W copies");
                        next = d;
                    }
                }
                if (next < 0)
                    throw std::runtime_error("gen_process::tt: could not resolve the W last copy");
                w_last = next;
            }

            int lepton = -1, neutrino = -1, n_quarks = 0;
            for (const int d : daughters[w_last]) {
                const int a = apdg(d);
                if (a == 11 || a == 13 || a == 15) {
                    if (lepton >= 0)
                        throw std::runtime_error("gen_process::tt: W with two charged leptons");
                    lepton = d;
                } else if (a == 12 || a == 14 || a == 16) {
                    if (neutrino >= 0)
                        throw std::runtime_error("gen_process::tt: W with two neutrinos");
                    neutrino = d;
                } else if (a >= 1 && a <= 6) {
                    ++n_quarks;
                } else if (a != 21 && a != 22) {
                    throw std::runtime_error("gen_process::tt: unexpected W daughter pdgId "
                                             + std::to_string(static_cast<int>(GenPart_pdgId[d])));
                }
            }
            if (lepton >= 0 && neutrino >= 0 && n_quarks == 0) {
                info.w_decay[k] = static_cast<WDecay>(apdg(lepton));
                info.lep_index[slot] = lepton;
            } else if (lepton < 0 && neutrino < 0 && n_quarks == 2) {
                info.w_decay[k] = WDecay::ToHadrons;
            } else {
                throw std::runtime_error("gen_process::tt: W decay is neither l+nu nor q+q'");
            }
        }
        return info;
    }

    //! As above, and also fill TTInfo::top_p4 and TTInfo::b_p4 from the GenPart kinematics --
    //! the form an RDataFrame expression naming the columns uses.
    template <typename VecId, typename VecFlags, typename VecMother, typename VecF>
    TTInfo identify(const VecId& GenPart_pdgId,
                    const VecFlags& GenPart_statusFlags,
                    const VecMother& GenPart_genPartIdxMother,
                    const VecF& GenPart_pt,
                    const VecF& GenPart_eta,
                    const VecF& GenPart_phi,
                    const VecF& GenPart_mass) {
        const detail::Kinematics<VecF> kinematics{GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass};
        return identify(GenPart_pdgId, GenPart_statusFlags, GenPart_genPartIdxMother, &kinematics);
    }

}  // namespace tt
}  // namespace gen_process
