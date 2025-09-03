#pragma once
#include "AnalysisTools.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

// see AN-13-178
template <typename LVector1, typename LVector2>
double Calculate_MT(const LVector1 &lepton_p4, const LVector2 &met_p4) {
    const double delta_phi = TVector2::Phi_mpi_pi(lepton_p4.Phi() - met_p4.Phi());
    return std::sqrt(2.0 * lepton_p4.Pt() * met_p4.Pt() * (1.0 - std::cos(delta_phi)));
}

template <typename LVector1, typename LVector2, typename LVector3>
double Calculate_TotalMT(const LVector1 &lepton1_p4, const LVector2 &lepton2_p4, const LVector3 &met_p4) {
    const double mt_1 = Calculate_MT(lepton1_p4, met_p4);
    const double mt_2 = Calculate_MT(lepton2_p4, met_p4);
    const double mt_ll = Calculate_MT(lepton1_p4, lepton2_p4);
    return std::sqrt(std::pow(mt_1, 2) + std::pow(mt_2, 2) + std::pow(mt_ll, 2));
}

namespace analysis {
    template <typename LVector1, typename LVector2>
    double Calculate_CosDTheta(const LVector1 &obj1_p4, const LVector2 &obj2_p4) {
        const ROOT::Math::DisplacementVector3D boost_vec = (obj1_p4 + obj2_p4).BoostToCM();
        const ROOT::Math::Boost b(boost_vec);
        const ROOT::Math::LorentzVector obj1 = b(obj1_p4);
        const ROOT::Math::LorentzVector obj2 = b(obj2_p4);

        const double dTheta = obj1.Theta() - obj2.Theta();
        const double cosDTheta = cos(dTheta);
        return cosDTheta;
    }
}  // namespace analysis