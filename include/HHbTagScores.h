#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#include "../../HHbtag/interface/HH_BTag.h"
#pragma clang diagnostic pop

#include "AnalysisTools.h"
#include "HHCore.h"
#include <map>

inline int PeriodToHHbTagInput(int version, Period period) {
    static const std::map<std::pair<int, Period>, int> periodHHBtag{
        // v1
        {{1, Period::Run2_2016_HIPM}, 2016},
        {{1, Period::Run2_2016}, 2016},
        {{1, Period::Run2_2017}, 2017},
        {{1, Period::Run2_2018}, 2018},
        {{1, Period::Run3_2022}, 2018},
        {{1, Period::Run3_2022EE}, 2018},
        {{1, Period::Run3_2023}, 2018},
        {{1, Period::Run3_2023BPix}, 2018},
        // v2
        {{2, Period::Run2_2016_HIPM}, 2016},
        {{2, Period::Run2_2016}, 2016},
        {{2, Period::Run2_2017}, 2017},
        {{2, Period::Run2_2018}, 2018},
        {{2, Period::Run3_2022}, 2018},
        {{2, Period::Run3_2022EE}, 2018},
        {{2, Period::Run3_2023}, 2018},
        {{2, Period::Run3_2023BPix}, 2018},
        // v3
        {{3, Period::Run2_2016_HIPM}, 0},
        {{3, Period::Run2_2016}, 0},
        {{3, Period::Run2_2017}, 0},
        {{3, Period::Run2_2018}, 0},
        {{3, Period::Run3_2022}, 0},
        {{3, Period::Run3_2022EE}, 1},
        {{3, Period::Run3_2023}, 2},
        {{3, Period::Run3_2023BPix}, 3},
    };
    auto iter = periodHHBtag.find(std::make_pair(version, period));
    if (iter == periodHHBtag.end()) {
        throw analysis::exception("Period correspondence not found");
    }
    return iter->second;
}

inline int ChannelToHHbTagInput(int version, Channel channel) {
    static const std::map<std::pair<int, Channel>, int> channelHHBtag{
        // v1
        {{1, Channel::eE}, -1},
        {{1, Channel::eMu}, -1},
        {{1, Channel::muMu}, -1},
        {{1, Channel::eTau}, 0},
        {{1, Channel::muTau}, 1},
        {{1, Channel::tauTau}, 2},
        // v2
        {{2, Channel::eE}, -1},
        {{2, Channel::eMu}, -1},
        {{2, Channel::muMu}, -1},
        {{2, Channel::eTau}, 0},
        {{2, Channel::muTau}, 1},
        {{2, Channel::tauTau}, 2},
        // v3
        {{3, Channel::muTau}, 0},
        {{3, Channel::eTau}, 1},
        {{3, Channel::tauTau}, 2},
        {{3, Channel::muMu}, 3},
        {{3, Channel::eE}, 4},
        {{3, Channel::eMu}, 5},
    };
    auto iter = channelHHBtag.find(std::make_pair(version, channel));
    if (iter == channelHHBtag.end()) {
        throw analysis::exception("Channel correspondance not found");
    }
    return iter->second;
}

struct HHBtagWrapper {
    struct Handle {
        std::unique_ptr<hh_btag::HH_BTag> tagger;
        int version{-1};
    };

    static void Initialize(const std::string &path, int version) {
        std::array<std::string, 2> models;
        for (size_t n = 0; n < 2; ++n) {
            std::ostringstream ss_model;
            ss_model << path + "HHbtag_v" << version << "_par_" << n;
            models.at(n) = ss_model.str();
        }
        _Get().tagger = std::make_unique<hh_btag::HH_BTag>(models);
        _Get().version = version;
    }
    static const Handle &Get() {
        auto &hh_btag = HHBtagWrapper::_Get();
        if (!hh_btag.tagger)
            throw std::runtime_error("HHBtag is not initialized.");
        return hh_btag;
    }

  private:
    static Handle &_Get() {
        static Handle hh_btag;
        return hh_btag;
    }
};

RVecF GetHHBtagScore(const RVecB &Jet_sel,
                     const RVecI &Jet_idx,
                     const RVecLV &jet_p4,
                     const RVecF &Jet_deepFlavour,
                     const float &met_pt,
                     const float &met_phi,
                     const HTTCand<2> &HTT_Cand,
                     const int &period,
                     int event) {
    const ULong64_t parity = event % 2;
    RVecI JetIdxOrdered = ReorderObjects(Jet_deepFlavour, Jet_idx);
    const auto &handle = HHBtagWrapper::Get();
    int channelId = ChannelToHHbTagInput(handle.version, HTT_Cand.channel());
    RVecF all_scores(JetIdxOrdered.size(), -1.);
    std::vector<float> jet_pt;
    std::vector<float> jet_eta;
    std::vector<float> jet_deepFlavour;
    std::vector<float> rel_jet_M_pt;
    std::vector<float> rel_jet_E_pt;
    std::vector<float> jet_htt_deta;
    std::vector<float> jet_htt_dphi;
    std::vector<int> goodjet_idx_ordered;

    LorentzVectorM hTT_p4 = HTT_Cand.leg_p4[0] + HTT_Cand.leg_p4[1];
    LorentzVectorM MET_p4(met_pt, 0, met_phi, 0);
    float htt_pt = hTT_p4.Pt();
    float htt_eta = hTT_p4.Eta();
    float htt_met_dphi = ROOT::Math::VectorUtil::DeltaPhi(hTT_p4, MET_p4);
    float htt_scalar_pt = HTT_Cand.leg_p4[0].Pt() + HTT_Cand.leg_p4[1].Pt();
    float rel_met_pt_htt_pt = met_pt / htt_scalar_pt;

    // select good jets only

    auto hhBtag_period = PeriodToHHbTagInput(handle.version, static_cast<Period>(period));
    for (size_t jet_idx = 0; jet_idx < jet_p4.size(); jet_idx++) {
        int jet_idx_ordered = JetIdxOrdered[jet_idx];
        if (!Jet_sel[jet_idx_ordered])
            continue;
        goodjet_idx_ordered.push_back(jet_idx_ordered);
        jet_pt.push_back(jet_p4.at(jet_idx_ordered).Pt());
        jet_eta.push_back(jet_p4.at(jet_idx_ordered).Eta());
        jet_deepFlavour.push_back(Jet_deepFlavour.at(jet_idx_ordered));
        rel_jet_M_pt.push_back(jet_p4.at(jet_idx_ordered).M() / jet_p4.at(jet_idx_ordered).Pt());
        rel_jet_E_pt.push_back(jet_p4.at(jet_idx_ordered).E() / jet_p4.at(jet_idx_ordered).Pt());
        jet_htt_deta.push_back(static_cast<float>(hTT_p4.Eta() - jet_p4.at(jet_idx_ordered).Eta()));
        jet_htt_dphi.push_back(ROOT::Math::VectorUtil::DeltaPhi(hTT_p4, jet_p4.at(jet_idx_ordered)));
    }

    RVecF goodJet_scores = handle.tagger->GetScore(jet_pt,
                                                   jet_eta,
                                                   rel_jet_M_pt,
                                                   rel_jet_E_pt,
                                                   jet_htt_deta,
                                                   jet_deepFlavour,
                                                   jet_htt_dphi,
                                                   hhBtag_period,
                                                   channelId,
                                                   htt_pt,
                                                   htt_eta,
                                                   htt_met_dphi,
                                                   rel_met_pt_htt_pt,
                                                   htt_scalar_pt,
                                                   parity);

    for (size_t jet_idx = 0; jet_idx < goodjet_idx_ordered.size(); jet_idx++) {
        int jet_idx_ordered = goodjet_idx_ordered[jet_idx];
        all_scores[jet_idx_ordered] = goodJet_scores[jet_idx];
    }
    return all_scores;
}
