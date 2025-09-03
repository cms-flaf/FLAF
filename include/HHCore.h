#pragma once

#include "AnalysisTools.h"
#include "GenTools.h"

enum class Vleg {
    PromptElectron = 1,
    PromptMuon = 2,
    TauDecayedToElectron = 3,
    TauDecayedToMuon = 4,
    TauDecayedToHadrons = 5,
    Nu = 6,
    Jet = 7
};

struct VCand {
    static constexpr size_t n_legs = 2;
    int index;
    std::array<Vleg, n_legs> leg_kind;
    std::array<int, n_legs> leg_index;
    std::array<LorentzVectorM, n_legs> leg_p4;      // p4 of quark from W->qq or lep p4
    std::array<LorentzVectorM, n_legs> leg_vis_p4;  // p4 of jet matched to quark; if no match = 0
    LorentzVectorM cand_p4;
};

struct HVVCand {
    static constexpr size_t n_legs = 2;
    std::array<VCand, n_legs> legs;
    LorentzVectorM cand_p4;
};

struct HBBCand {
    static constexpr size_t n_legs = 2;
    std::array<int, n_legs> leg_index;
    std::array<LorentzVectorM, n_legs> leg_p4;      // p4 of quark from H->bb
    std::array<LorentzVectorM, n_legs> leg_vis_p4;  // p4 of jet matched to quark; if no match = 0
    LorentzVectorM cand_p4;
};

template <size_t N>
struct HTTCand {
    static constexpr size_t n_legs = N;
    std::array<Leg, n_legs> leg_type;
    std::array<int, n_legs> leg_index;
    std::array<LorentzVectorM, n_legs> leg_p4;
    std::array<int, n_legs> leg_charge;
    std::array<float, n_legs> leg_rawIso;
    std::array<int, n_legs> leg_genMatchIdx;

    HTTCand() { leg_type.fill(Leg::none); }

    Channel channel() const { return _channel(std::make_index_sequence<n_legs>{}); }

    bool operator==(const HTTCand &other) const {
        for (size_t idx = 0; idx < n_legs; ++idx) {
            if (leg_type[idx] != other.leg_type[idx] || leg_index[idx] != other.leg_index[idx])
                return false;
        }
        return true;
    }
    bool isLeg(int obj_index, Leg leg) const {
        for (size_t idx = 0; idx < n_legs; idx++) {
            if (leg_type[idx] == leg && leg_index[idx] == obj_index) {
                return true;
            }
        }
        return false;
    }
    RVecB isLeg(const RVecI &obj_vec, Leg leg) const {
        RVecB isLeg_vector(obj_vec.size(), false);
        for (size_t obj_idx = 0; obj_idx < obj_vec.size(); obj_idx++) {
            isLeg_vector[obj_idx] = isLeg(obj_vec[obj_idx], leg);
        }
        return isLeg_vector;
    }

    std::vector<LorentzVectorM> getLegP4s() const {
        std::vector<LorentzVectorM> leg_p4s;
        for (size_t idx = 0; idx < n_legs; idx++) {
            if (leg_type[idx] != Leg::none) {
                leg_p4s.push_back(leg_p4[idx]);
            }
        }
        return leg_p4s;
    }

    size_t nValidLegs() const {
        size_t n_valid_legs = 0;
        for (size_t idx = 0; idx < n_legs; idx++) {
            if (leg_type[idx] != Leg::none) {
                ++n_valid_legs;
            }
        }
        return n_valid_legs;
    }

  private:
    template <size_t... Idx>
    Channel _channel(std::index_sequence<Idx...>) const {
        return LegsToChannel(leg_type[Idx]...);
    }
};

// template<size_t N>
struct HWWCand {
    size_t n_legs;
    std::vector<Leg> leg_type;
    std::vector<int> leg_index;
    std::vector<LorentzVectorM> leg_p4;
    std::vector<int> leg_charge;
    std::vector<float> leg_rawIso;
    std::vector<int> leg_genMatchIdx;
    HWWCand(size_t num_legs = 0) {
        n_legs = num_legs;
        leg_type.resize(num_legs, Leg::none);
        leg_index.resize(num_legs);
        leg_p4.resize(num_legs);
        leg_charge.resize(num_legs);
        leg_charge.resize(num_legs);
        leg_rawIso.resize(num_legs);
        leg_genMatchIdx.resize(num_legs);
    }
    Channel channel() const {
        if (n_legs == 1) {
            return LegsToChannel(leg_type.at(0));
        }
        if (n_legs == 2) {
            return LegsToChannel(leg_type.at(0), leg_type.at(1));
        } else {
            throw std::runtime_error("wrong number legs");
        }
    }
    bool operator==(const HWWCand &other) const {
        for (size_t idx = 0; idx < n_legs; ++idx) {
            if (leg_type.at(idx) != other.leg_type.at(idx) || leg_index.at(idx) != other.leg_index.at(idx))
                return false;
        }
        return true;
    }
    bool isLeg(int obj_index, Leg leg) const {
        for (size_t idx = 0; idx < n_legs; idx++) {
            if (leg_type.at(idx) == leg && leg_index.at(idx) == obj_index) {
                return true;
            }
        }
        return false;
    }
    RVecB isLeg(const RVecI &obj_vec, Leg leg) const {
        RVecB isLeg_vector(obj_vec.size(), false);
        for (size_t obj_idx = 0; obj_idx < obj_vec.size(); obj_idx++) {
            isLeg_vector[obj_idx] = isLeg(obj_vec[obj_idx], leg);
        }
        return isLeg_vector;
    }
    std::vector<LorentzVectorM> getLegP4s() const {
        std::vector<LorentzVectorM> leg_p4s;
        for (size_t idx = 0; idx < n_legs; idx++) {
            if (leg_type.at(idx) != Leg::none) {
                leg_p4s.push_back(leg_p4.at(idx));
            }
        }
        return leg_p4s;
    }

    size_t nValidLegs() const {
        size_t n_valid_legs = 0;
        for (size_t idx = 0; idx < n_legs; idx++) {
            if (leg_type[idx] != Leg::none) {
                ++n_valid_legs;
            }
        }
        return n_valid_legs;
    }

  private:
    template <size_t... Idx>
    Channel _channel(std::index_sequence<Idx...>) const {
        return LegsToChannel(leg_type[Idx]...);
    }
};

struct HbbCand {
    static constexpr size_t n_legs = 2;
    std::array<int, n_legs> leg_index;
    std::array<LorentzVectorM, n_legs> leg_p4;
};
template <size_t N>
std::ostream &operator<<(std::ostream &os, const HTTCand<N> &cand) {
    for (size_t n = 0; n < HTTCand<N>::n_legs; ++n) {
        if (cand.leg_type[n] == Leg::none)
            continue;
        os << "leg " << (n + 1) << ": " << " type = " << static_cast<int>(cand.leg_type[n])
           << " index = " << cand.leg_index[n] << " (pt,eta,phi,m) = " << cand.leg_p4[n]
           << " charge=" << cand.leg_charge[n] << " rawIso=" << cand.leg_rawIso[n] << "\n";
    }
    return os;
}