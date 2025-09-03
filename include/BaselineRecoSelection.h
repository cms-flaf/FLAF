#pragma once
#include "AnalysisTools.h"
#include "HHCore.h"
#include <optional>

template <size_t N>
void FillHTTCandidates(Channel refChannel,
                       ROOT::VecOps::RVec<HTTCand<N>> &HttCandidates,
                       const HTTCand<N> &refHttCand,
                       double dR2_thr,
                       const std::vector<Leg> &leg_types,
                       const std::map<Leg, std::set<size_t>> &otherLegs,
                       const RVecLV &otherLegs_p4,
                       size_t leg_index) {
    if (refHttCand.channel() != refChannel)
        throw analysis::exception(
            "ERROR: FillHTTCandidates: HttCandidate has an "
            "unexpected channel: %1% != %2%") %
            static_cast<int>(refHttCand.channel()) % static_cast<int>(refChannel);
    HttCandidates.push_back(refHttCand);
}

template <size_t N, typename... Args>
void FillHTTCandidates(Channel refChannel,
                       ROOT::VecOps::RVec<HTTCand<N>> &HttCandidates,
                       const HTTCand<N> &refHttCand,
                       double dR2_thr,
                       const std::vector<Leg> &leg_types,
                       const std::map<Leg, std::set<size_t>> &otherLegs,
                       const RVecLV &otherLegs_p4,
                       size_t leg_index,
                       const RVecB &leg_sel,
                       const RVecLV &leg_p4,
                       const RVecF &leg_rawIso,
                       const RVecI &leg_charge,
                       const RVecI &leg_genMatchIdx,
                       Args &&...leg_info) {
    if (leg_index >= leg_types.size())
        throw analysis::exception("ERROR: FillHTTCandidates: too many arguments for channel %1%.") %
            static_cast<int>(refChannel);
    const Leg leg_type = leg_types.at(leg_index);
    for (size_t leg_idx = 0; leg_idx < leg_sel.size(); ++leg_idx) {
        if (!leg_sel[leg_idx] || (otherLegs.count(leg_type) && otherLegs.at(leg_type).count(leg_idx)))
            continue;
        bool pass_dR = true;
        for (size_t other_idx = 0; other_idx < otherLegs_p4.size(); ++other_idx) {
            if (ROOT::Math::VectorUtil::DeltaR2(leg_p4.at(leg_idx), otherLegs_p4.at(other_idx)) <= dR2_thr) {
                pass_dR = false;
                break;
            }
        }
        if (!pass_dR)
            continue;
        auto newHttCand = refHttCand;
        newHttCand.leg_type[leg_index] = leg_type;
        newHttCand.leg_index[leg_index] = leg_idx;
        newHttCand.leg_p4[leg_index] = leg_p4.at(leg_idx);
        newHttCand.leg_charge[leg_index] = leg_charge.at(leg_idx);
        newHttCand.leg_rawIso[leg_index] = leg_rawIso.at(leg_idx);
        newHttCand.leg_genMatchIdx[leg_index] = leg_genMatchIdx.at(leg_idx);

        auto newOtherLegs = otherLegs;
        newOtherLegs[leg_type].insert(leg_idx);

        auto newOtherLegs_p4 = otherLegs_p4;
        newOtherLegs_p4.push_back(leg_p4.at(leg_idx));

        FillHTTCandidates(refChannel,
                          HttCandidates,
                          newHttCand,
                          dR2_thr,
                          leg_types,
                          newOtherLegs,
                          newOtherLegs_p4,
                          leg_index + 1,
                          std::forward<Args>(leg_info)...);
    }
}
// template<size_t N>
void FillHWWCandidates(Channel refChannel,
                       ROOT::VecOps::RVec<HWWCand> &HwwCandidates,
                       const HWWCand &refHwwCand,
                       double dR2_thr,
                       const std::vector<Leg> &leg_types,
                       const std::map<Leg, std::set<size_t>> &otherLegs,
                       const RVecLV &otherLegs_p4,
                       size_t leg_index) {
    if (refHwwCand.channel() != refChannel)
        throw analysis::exception(
            "ERROR: FillHWWCandidates: HwwCandidate has an "
            "unexpected channel: %1% != %2%") %
            static_cast<int>(refHwwCand.channel()) % static_cast<int>(refChannel);
    HwwCandidates.push_back(refHwwCand);
}

// template<size_t N, typename ...Args>
template <typename... Args>
void FillHWWCandidates(Channel refChannel,
                       ROOT::VecOps::RVec<HWWCand> &HwwCandidates,
                       const HWWCand &refHwwCand,
                       double dR2_thr,
                       const std::vector<Leg> &leg_types,
                       const std::map<Leg, std::set<size_t>> &otherLegs,
                       const RVecLV &otherLegs_p4,
                       size_t leg_index,
                       const RVecB &leg_sel,
                       const RVecLV &leg_p4,
                       const RVecF &leg_rawIso,
                       const RVecI &leg_charge,
                       const RVecI &leg_genMatchIdx,
                       Args &&...leg_info) {
    if (leg_index >= leg_types.size())
        throw analysis::exception("ERROR: FillHWWCandidates: too many arguments for channel %1%.") %
            static_cast<int>(refChannel);
    const Leg leg_type = leg_types.at(leg_index);
    for (size_t leg_idx = 0; leg_idx < leg_sel.size(); ++leg_idx) {
        if (!leg_sel[leg_idx] || (otherLegs.count(leg_type) && otherLegs.at(leg_type).count(leg_idx)))
            continue;
        bool pass_dR = true;
        for (size_t other_idx = 0; other_idx < otherLegs_p4.size(); ++other_idx) {
            if (ROOT::Math::VectorUtil::DeltaR2(leg_p4.at(leg_idx), otherLegs_p4.at(other_idx)) <= dR2_thr) {
                pass_dR = false;
                break;
            }
        }
        if (!pass_dR)
            continue;
        auto newHwwCand = refHwwCand;
        newHwwCand.leg_type[leg_index] = leg_type;
        newHwwCand.leg_index[leg_index] = leg_idx;
        newHwwCand.leg_p4[leg_index] = leg_p4.at(leg_idx);
        newHwwCand.leg_charge[leg_index] = leg_charge.at(leg_idx);
        newHwwCand.leg_rawIso[leg_index] = leg_rawIso.at(leg_idx);
        newHwwCand.leg_genMatchIdx[leg_index] = leg_genMatchIdx.at(leg_idx);
        auto newOtherLegs = otherLegs;
        newOtherLegs[leg_type].insert(leg_idx);

        auto newOtherLegs_p4 = otherLegs_p4;
        newOtherLegs_p4.push_back(leg_p4.at(leg_idx));
        FillHWWCandidates(refChannel,
                          HwwCandidates,
                          newHwwCand,
                          dR2_thr,
                          leg_types,
                          newOtherLegs,
                          newOtherLegs_p4,
                          leg_index + 1,
                          std::forward<Args>(leg_info)...);
    }
}

template <size_t N, typename... Args>
ROOT::VecOps::RVec<HTTCand<N>> GetHTTCandidates(Channel channel, double dR_thr, Args &&...leg_info) {
    const double dR2_thr = std::pow(dR_thr, 2);
    ROOT::VecOps::RVec<HTTCand<N>> HttCandidates;
    const auto leg_types = ChannelToLegs(channel);
    if (leg_types.empty())
        throw analysis::exception("ERROR: no legs are expected for channel %1%") % static_cast<int>(channel);
    HTTCand<N> refHttCand;
    std::map<Leg, std::set<size_t>> otherLegs;
    RVecLV otherLegs_p4;
    try {
        FillHTTCandidates(channel,
                          HttCandidates,
                          refHttCand,
                          dR2_thr,
                          leg_types,
                          otherLegs,
                          otherLegs_p4,
                          0,
                          std::forward<Args>(leg_info)...);
    } catch (analysis::exception &e) {
        std::cerr << "ERROR: GetHTTCandidates: target channel = " << static_cast<int>(channel) << '\n'
                  << e.what() << std::endl;
        std::cerr << "Expected leg types: ";
        for (auto leg : leg_types)
            std::cerr << static_cast<int>(leg) << ' ';
        std::cerr << std::endl;
        throw;
    } catch (std::out_of_range &e) {
        std::cerr << "ERROR: GetHTTCandidates: target channel = " << static_cast<int>(channel) << '\n'
                  << e.what() << std::endl;
        throw;
    }
    return HttCandidates;
}
// template<size_t N, typename ...Args>
template <typename... Args>
ROOT::VecOps::RVec<HWWCand> GetHWWCandidates(Channel channel, double dR_thr, Args &&...leg_info) {
    const double dR2_thr = std::pow(dR_thr, 2);
    ROOT::VecOps::RVec<HWWCand> HwwCandidates;
    const auto leg_types = ChannelToLegs(channel);
    const size_t n_legs = leg_types.size();
    if (leg_types.empty())
        throw analysis::exception("ERROR: no legs are expected for channel %1%") % static_cast<int>(channel);
    HWWCand refHwwCand(n_legs);
    std::map<Leg, std::set<size_t>> otherLegs;
    RVecLV otherLegs_p4;
    try {
        FillHWWCandidates(channel,
                          HwwCandidates,
                          refHwwCand,
                          dR2_thr,
                          leg_types,
                          otherLegs,
                          otherLegs_p4,
                          0,
                          std::forward<Args>(leg_info)...);
    } catch (analysis::exception &e) {
        std::cerr << "ERROR: GetHWWCandidates: target channel = " << static_cast<int>(channel) << '\n'
                  << e.what() << std::endl;
        std::cerr << "Expected leg types: ";
        for (auto leg : leg_types)
            std::cerr << static_cast<int>(leg) << ' ';
        std::cerr << std::endl;
        throw;
    } catch (std::out_of_range &e) {
        std::cerr << "ERROR: GetHWWCandidates: target channel = " << static_cast<int>(channel) << '\n'
                  << e.what() << std::endl;
        throw;
    }
    return HwwCandidates;
}
template <size_t N>
HTTCand<N> GetBestHTTCandidate(const std::vector<const ROOT::VecOps::RVec<HTTCand<N>> *> HttCandidates,
                               unsigned long long event) {
    const auto &comparitor = [&](const HTTCand<N> &cand1, const HTTCand<N> &cand2) -> bool {
        if (cand1 == cand2)
            return false;
        if (cand1.channel() != cand2.channel()) {
            throw analysis::exception(
                "ERROR: different channels considered for HTT "
                "candiate choice!! %1% VS %2%") %
                static_cast<int>(cand1.channel()) % static_cast<int>(cand2.channel());
        }
        for (size_t idx = 0; idx < cand1.leg_index.size(); ++idx) {
            if (cand1.leg_type[idx] != cand2.leg_type[idx]) {
                throw analysis::exception(
                    "ERROR: different leg types considered for "
                    "HTT candiate choice!! %1% VS %2%") %
                    static_cast<int>(cand1.leg_type[idx]) % static_cast<int>(cand2.leg_type[idx]);
            }
            if (cand1.leg_type[idx] == Leg::none)
                continue;
            if (cand1.leg_rawIso[idx] != cand2.leg_rawIso[idx])
                return cand1.leg_rawIso[idx] < cand2.leg_rawIso[idx];
            if (cand1.leg_p4[idx].pt() != cand2.leg_p4[idx].pt())
                return cand1.leg_p4[idx].pt() > cand2.leg_p4[idx].pt();
            if (std::abs(cand1.leg_p4[idx].eta()) != std::abs(cand2.leg_p4[idx].eta()))
                return std::abs(cand1.leg_p4[idx].eta()) < std::abs(cand2.leg_p4[idx].eta());
        }
        throw analysis::exception(
            "ERROR: criteria for best tau pair selection is "
            "not found in channel %1% and event %2%") %
            static_cast<int>(cand1.channel()) % event;
    };

    for (auto cands : HttCandidates) {
        if (!cands->empty())
            return *std::min_element(cands->begin(), cands->end(), comparitor);
    }

    throw analysis::exception("ERROR: no suitable HTT candidate ");
}
// template<size_t N>
HWWCand GetBestHWWCandidate(const std::vector<const ROOT::VecOps::RVec<HWWCand> *> HwwCandidates,
                            unsigned long long event) {
    const auto &comparitor = [&](const HWWCand &cand1, const HWWCand &cand2) -> bool {
        if (cand1 == cand2)
            return false;
        if (cand1.channel() != cand2.channel()) {
            throw analysis::exception(
                "ERROR: different channels considered for HWW "
                "candiate choice!! %1% VS %2%") %
                static_cast<int>(cand1.channel()) % static_cast<int>(cand2.channel());
        }
        for (size_t idx = 0; idx < cand1.leg_index.size(); ++idx) {
            if (cand1.leg_type[idx] != cand2.leg_type[idx]) {
                throw analysis::exception(
                    "ERROR: different leg types considered for "
                    "HWW candiate choice!! %1% VS %2%") %
                    static_cast<int>(cand1.leg_type[idx]) % static_cast<int>(cand2.leg_type[idx]);
            }
            if (cand1.leg_type[idx] == Leg::none)
                continue;
            if (cand1.leg_rawIso[idx] != cand2.leg_rawIso[idx])
                return cand1.leg_rawIso[idx] < cand2.leg_rawIso[idx];
            if (cand1.leg_p4[idx].pt() != cand2.leg_p4[idx].pt())
                return cand1.leg_p4[idx].pt() > cand2.leg_p4[idx].pt();
            if (std::abs(cand1.leg_p4[idx].eta()) != std::abs(cand2.leg_p4[idx].eta()))
                return std::abs(cand1.leg_p4[idx].eta()) < std::abs(cand2.leg_p4[idx].eta());
        }
        throw analysis::exception(
            "ERROR: criteria for best W pair selection is "
            "not found in channel %1% and event %2%") %
            static_cast<int>(cand1.channel()) % event;
    };
    for (auto cands : HwwCandidates) {
        if (!cands->empty())
            return *std::min_element(cands->begin(), cands->end(), comparitor);
    }

    throw analysis::exception("ERROR: no suitable HWW candidate ");
}

template <size_t N>
bool GenRecoMatching(const HTTCand<N> &genHttCandidate, const HTTCand<N> &recoHttCandidate, double dR_thr) {
    const double dR2_thr = std::pow(dR_thr, 2);
    std::vector<bool> matching(HTTCand<N>::n_legs * 2, false);
    for (size_t gen_idx = 0; gen_idx < HTTCand<N>::n_legs; ++gen_idx) {
        for (size_t reco_idx = 0; reco_idx < HTTCand<N>::n_legs; ++reco_idx) {
            if (genHttCandidate.leg_type[gen_idx] == recoHttCandidate.leg_type[reco_idx] &&
                genHttCandidate.leg_charge[gen_idx] == recoHttCandidate.leg_charge[reco_idx]) {
                const double dR2 =
                    ROOT::Math::VectorUtil::DeltaR2(genHttCandidate.leg_p4[gen_idx], recoHttCandidate.leg_p4[reco_idx]);
                if (dR2 < dR2_thr)
                    matching[gen_idx * HTTCand<N>::n_legs + reco_idx] = true;
            }
        }
    }
    return (matching[0] && matching[3]) || (matching[1] && matching[2]);
}

RVecI GenRecoJetMatching(int event,
                         const RVecI &Jet_idx,
                         const RVecI &GenJet_idx,
                         const RVecB &Jet_sel,
                         const RVecB &GenJet_sel,
                         const RVecLV &GenJet_p4,
                         const RVecLV &Jet_p4,
                         float DeltaR_thr) {
    RVecI recoJetMatched(Jet_idx.size(), -1);
    std::set<size_t> taken_jets;
    for (size_t gen_idx = 0; gen_idx < GenJet_p4.size(); ++gen_idx) {
        if (GenJet_sel[gen_idx] != 1)
            continue;
        size_t best_jet_idx = Jet_p4.size();
        float deltaR_min = std::numeric_limits<float>::infinity();
        for (size_t reco_idx = 0; reco_idx < Jet_p4.size(); ++reco_idx) {
            if (Jet_sel[reco_idx] != 1 || taken_jets.count(reco_idx))
                continue;
            auto deltaR = ROOT::Math::VectorUtil::DeltaR(Jet_p4[reco_idx], GenJet_p4[gen_idx]);
            if (deltaR < deltaR_min && deltaR < DeltaR_thr) {
                best_jet_idx = reco_idx;
                deltaR_min = deltaR;
            }
        }
        if (best_jet_idx < Jet_p4.size()) {
            taken_jets.insert(best_jet_idx);
            recoJetMatched.at(best_jet_idx) = gen_idx;
        }
    }
    return recoJetMatched;
}

std::optional<HbbCand> GetHbbCandidate(const RVecF &HHbTagScores,
                                       const RVecB &JetSel,
                                       const RVecLV &Jet_p4,
                                       const RVecI &Jet_idx) {
    RVecI JetIdxOrdered = ReorderObjects(HHbTagScores, Jet_idx);
    HbbCand HbbCandidate;
    for (int j = 0; j < HbbCandidate.n_legs; j++) {
        HbbCandidate.leg_index[j] = -1;
    }
    int leg_idx = 0;
    for (int i = 0; i < Jet_idx.size(); i++) {
        int jet_idx = JetIdxOrdered[i];
        if (!JetSel[jet_idx])
            continue;
        HbbCandidate.leg_index[leg_idx] = jet_idx;
        HbbCandidate.leg_p4[leg_idx] = Jet_p4.at(jet_idx);
        leg_idx++;
        if (leg_idx == HbbCandidate.n_legs)
            break;
    }
    if (HbbCandidate.leg_index[0] >= 0 && HbbCandidate.leg_index[1] >= 0) {
        return HbbCandidate;
    }
    return std::nullopt;
}
template <size_t N>
LorentzVectorM GetMetNoMu(const HTTCand<N> &HttCandidate, const LorentzVectorM &Met_p4) {
    LorentzVectorM Muon_p4_sig(0., 0., 0., 0.);
    for (size_t leg_idx = 0; leg_idx < HttCandidate.n_legs; leg_idx++) {
        if (HttCandidate.leg_type[leg_idx] == Leg::mu) {
            Muon_p4_sig += HttCandidate.leg_p4[leg_idx];
        }
    }
    return Met_p4 + Muon_p4_sig;
}

using LegIndexPair = std::pair<Leg, size_t>;
using LegMatching = std::pair<Leg, RVecSetInt>;
using RVecMatching = ROOT::VecOps::RVec<LegMatching>;

bool _HasOOMatching(const RVecMatching &legVector,
                    size_t legIndex,
                    std::set<int> &onlineSelected,
                    std::set<LegIndexPair> &offlineSelected) {
    if (legIndex >= legVector.size())
        return true;
    for (size_t offlineIndex = 0; offlineIndex < legVector[legIndex].second.size(); ++offlineIndex) {
        const LegIndexPair offlinePair(legVector[legIndex].first, offlineIndex);
        if (offlineSelected.count(offlinePair))
            continue;
        offlineSelected.insert(offlinePair);
        for (int onlineIndex : legVector[legIndex].second[offlineIndex]) {
            if (onlineSelected.count(onlineIndex))
                continue;
            onlineSelected.insert(onlineIndex);
            // std::cout<< onlineIndex<<std::endl;
            if (_HasOOMatching(legVector, legIndex + 1, onlineSelected, offlineSelected))
                return true;
            onlineSelected.erase(onlineIndex);
        }
        offlineSelected.erase(offlinePair);
    }
    return false;
}

std::pair<bool, std::set<LegIndexPair>> HasOOMatching(const RVecMatching &legVector) {
    std::set<int> onlineSelected;
    std::set<LegIndexPair> offlineSelected;

    const bool hasMatching = _HasOOMatching(legVector, 0, onlineSelected, offlineSelected);
    return std::make_pair(hasMatching, offlineSelected);
}
