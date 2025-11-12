#pragma once

#include "AnalysisTools.h"
#include "GenLepton.h"
#include "GenStatusFlags.h"
#include "TextIO.h"
#include "exception.h"

struct ParticleInfo {
    int pdgId;
    float charge;
    std::string name;
    std::string type;
    float mass{-1.};
};

class ParticleDB {
  public:
    static void Initialize(const std::string_view inputFile) {
        std::ifstream file(std::string(inputFile).c_str(), ios::in);
        std::string line;
        while (getline(file, line)) {
            auto values = analysis::SplitValueList(line, true, ",", true);
            if (values.size() != 4 && values.size() != 5) {
                throw analysis::exception("invalid line %1%") % line;
            }
            ParticleInfo currentInfo;
            currentInfo.pdgId = analysis::Parse<int>(values.at(0));
            currentInfo.name = values.at(1);
            currentInfo.type = values.at(2);
            currentInfo.charge = analysis::Parse<float>(values.at(3));
            if (values.size() > 4) {
                currentInfo.mass = analysis::Parse<float>(values.at(4));
            }
            particles()[currentInfo.pdgId] = currentInfo;
        }
    }

    static const ParticleInfo &GetParticleInfo(int pdgId) {
        if (particles().empty())
            throw analysis::exception("ParticleDB is not initialized.");
        auto iter = particles().find(pdgId);
        if (iter == particles().end()) {
            throw analysis::exception("ParticleInfo not found for particle ID");
        }
        return iter->second;
    }
    static const ParticleInfo &GetParticleInfo(std::string_view particle_name) {
        for (const auto &info : particles()) {
            if (info.second.name == particle_name)
                return info.second;
        }
        throw analysis::exception("Particle with name '%1%' not found.") % particle_name;
    }

    static float GetMass(int pdgId, float mass) {
        static const std::set<int> pdgId_noMass{11, 12, 13, 14, 15, 16, 22, 111, 211, 311, 321, 421, 411};
        if (pdgId_noMass.count(std::abs(pdgId)))
            return GetParticleInfo(pdgId).mass;
        return mass;
    }

  private:
    static std::map<int, ParticleInfo> &particles() {
        static std::map<int, ParticleInfo> p;
        return p;
    }
};

struct PdG {
    static int e() {
        static const int pdg = ParticleDB::GetParticleInfo("e-").pdgId;
        return pdg;
    }
    static int mu() {
        static const int pdg = ParticleDB::GetParticleInfo("mu-").pdgId;
        return pdg;
    }
    static int tau() {
        static const int pdg = ParticleDB::GetParticleInfo("tau-").pdgId;
        return pdg;
    }
    static int Higgs() {
        static const int pdg = ParticleDB::GetParticleInfo("h0").pdgId;
        return pdg;
    }
    static int nu_e() {
        static const int pdg = ParticleDB::GetParticleInfo("nu_e").pdgId;
        return pdg;
    }
    static int nu_mu() {
        static const int pdg = ParticleDB::GetParticleInfo("nu_mu").pdgId;
        return pdg;
    }
    static int nu_tau() {
        static const int pdg = ParticleDB::GetParticleInfo("nu_tau").pdgId;
        return pdg;
    }
    static int b() {
        static const int pdg = ParticleDB::GetParticleInfo("b").pdgId;
        return pdg;
    }
    static int bbar() {
        static const int pdg = ParticleDB::GetParticleInfo("bbar").pdgId;
        return pdg;
    }
    static int Wplus() {
        static const int pdg = ParticleDB::GetParticleInfo("W+").pdgId;
        return pdg;
    }
    static int Wminus() {
        static const int pdg = ParticleDB::GetParticleInfo("W-").pdgId;
        return pdg;
    }
    static int Z() {
        static const int pdg = ParticleDB::GetParticleInfo("Z0").pdgId;
        return pdg;
    }
    static int HeavyHiggs() {
        static const int pdg = ParticleDB::GetParticleInfo("H0").pdgId;
        return pdg;
    }

    static bool isNeutrino(int pdg) {
        static const std::set<int> neutrinos = {nu_e(), nu_mu(), nu_tau()};
        return neutrinos.count(std::abs(pdg)) > 0;
    }

    static bool isChargedLepton(int pdg) {
        static const std::set<int> leptons = {e(), mu(), tau()};
        return leptons.count(std::abs(pdg)) > 0;
    }
};

inline Leg PdGToLeg(int pdg) {
    static const std::map<int, Leg> pdg_to_leg = {
        {PdG::e(), Leg::e},
        {PdG::mu(), Leg::mu},
        {PdG::tau(), Leg::tau},
    };
    const auto iter = pdg_to_leg.find(std::abs(pdg));
    if (iter == pdg_to_leg.end())
        throw analysis::exception("Unknown leg type. leg_pdg = %1%.") % pdg;
    return iter->second;
}

bool isRelated(int potential_mother, int particle_idx, const RVecI &GenPart_genPartIdxMother) {
    if (potential_mother == particle_idx)
        return true;
    const int mother_idx = GenPart_genPartIdxMother.at(particle_idx);
    if (mother_idx < 0)
        return false;
    return isRelated(potential_mother, mother_idx, GenPart_genPartIdxMother);
}

int GetLastCopy(int genPart,
                const RVecI &GenPart_pdgId,
                const RVecI &GenPart_statusFlags,
                const RVecVecI &GenPart_daughters) {
    GenStatusFlags pFlags(GenPart_statusFlags.at(genPart));
    int genPart_copy = genPart;
    while (true) {
        GenStatusFlags flags(GenPart_statusFlags.at(genPart_copy));
        if (flags.isLastCopy())
            return genPart_copy;
        std::vector<int> copies;
        for (int daughter : GenPart_daughters.at(genPart_copy)) {
            GenStatusFlags daughter_flags(GenPart_statusFlags.at(daughter));
            if (GenPart_pdgId.at(daughter) == GenPart_pdgId.at(genPart_copy) && daughter_flags.fromHardProcess())
                copies.push_back(daughter);
        }
        if (copies.size() != 1)
            break;
        genPart_copy = copies[0];
    }
    throw analysis::exception("Last copy not found.");
}

LorentzVectorM GetVisibleP4(int genPart,
                            const RVecI &GenPart_pdgId,
                            const RVecVecI &GenPart_daughters,
                            const RVecF &GenPart_pt,
                            const RVecF &GenPart_eta,
                            const RVecF &GenPart_phi,
                            const RVecF &GenPart_mass) {
    LorentzVectorXYZ sum(0., 0., 0., 0.);
    std::function<void(int)> addParticle;
    addParticle = [&](int p) {
        if (GenPart_daughters.at(p).empty() && !PdG::isNeutrino(p)) {
            sum += LorentzVectorM(
                GenPart_pt.at(p), GenPart_eta.at(p), GenPart_phi.at(p), ParticleDB::GetMass(p, GenPart_mass.at(p)));
        }
        for (int daughter : GenPart_daughters.at(p))
            addParticle(daughter);
    };
    addParticle(genPart);
    return LorentzVectorM(sum);
}

ROOT::VecOps::RVec<RVecI> GetDaughters(const RVecI &GenPart_genPartIdxMother) {
    ROOT::VecOps::RVec<RVecI> daughters(GenPart_genPartIdxMother.size());
    for (int part_idx = 0; part_idx < GenPart_genPartIdxMother.size(); part_idx++) {
        if (GenPart_genPartIdxMother[part_idx] >= 0) {
            daughters.at(GenPart_genPartIdxMother[part_idx]).push_back(part_idx);
        }
    }
    return daughters;
}
RVecI GetMothers(const int &part_idx, const RVecI &GenPart_genPartIdxMother) {
    RVecI mothers;
    int new_idx = part_idx;
    int mother_idx = GenPart_genPartIdxMother[new_idx];
    while (mother_idx >= 0) {
        mothers.push_back(mother_idx);
        new_idx = mother_idx;
        mother_idx = GenPart_genPartIdxMother[new_idx];
    }
    return mothers;
}

RVecI GetLastHadrons(const RVecI &GenPart_pdgId,
                     const RVecI &GenPart_genPartIdxMother,
                     const ROOT::VecOps::RVec<RVecI> &GenPart_daughters) {
    RVecI lastHadrons;
    // if(evt!=905) return lastHadrons;

    for (int part_idx = 0; part_idx < GenPart_pdgId.size(); part_idx++) {
        RVecI mothers = GetMothers(part_idx, GenPart_genPartIdxMother);
        RVecI daughters = GenPart_daughters.at(part_idx);
        bool comesFrom_b = false;
        bool comesFrom_H = false;
        bool hasHadronsDaughters = false;

        for (auto &mother : mothers) {
            if (abs(GenPart_pdgId[mother]) == 5)
                comesFrom_b = true;
            if (abs(GenPart_pdgId[mother]) == 25)
                comesFrom_H = true;
        }
        for (auto &daughter : daughters) {
            ParticleInfo daughterInformation = ParticleDB::GetParticleInfo(GenPart_pdgId[daughter]);
            if (daughterInformation.type == "baryon" || daughterInformation.type == "meson") {
                hasHadronsDaughters = true;
            }
        }
        if (comesFrom_b && comesFrom_H && !hasHadronsDaughters) {
            ParticleInfo lastHadronInformation = ParticleDB::GetParticleInfo(GenPart_pdgId[part_idx]);
            if (lastHadronInformation.type == "baryon" || lastHadronInformation.type == "meson") {
                lastHadrons.push_back(part_idx);
            }
        }
    }
    return lastHadrons;
}

void PrintDecayChainParticle(ULong64_t evt,
                             int genPart_idx,
                             const RVecI &GenPart_pdgId,
                             const RVecI &GenPart_genPartIdxMother,
                             const RVecI &GenPart_statusFlags,
                             const RVecF &GenPart_pt,
                             const RVecF &GenPart_eta,
                             const RVecF &GenPart_phi,
                             const RVecF &GenPart_mass,
                             const RVecI &GenPart_status,
                             const std::string pre,
                             const ROOT::VecOps::RVec<RVecI> &GenPart_daughters,
                             std::ostream &os) {
    const ParticleInfo &particle_information = ParticleDB::GetParticleInfo(GenPart_pdgId[genPart_idx]);
    const float particleMass = ParticleDB::GetMass(GenPart_pdgId[genPart_idx], GenPart_mass[genPart_idx]);
    const LorentzVectorM genParticle_momentum(
        GenPart_pt[genPart_idx], GenPart_eta[genPart_idx], GenPart_phi[genPart_idx], particleMass);

    os << particle_information.name << " <" << GenPart_pdgId[genPart_idx] << '>' << " pt=" << genParticle_momentum.pt()
       << " eta=" << genParticle_momentum.eta() << " phi=" << genParticle_momentum.phi()
       << " E=" << genParticle_momentum.energy() << " m=" << genParticle_momentum.mass() << " index=" << genPart_idx
       << " flags=" << GenStatusFlags(GenPart_statusFlags[genPart_idx]).toString()
       << " status=" << GenPart_status[genPart_idx] << " charge=" << particle_information.charge
       << " type=" << particle_information.type << '\n';

    const RVecI &daughters = GenPart_daughters.at(genPart_idx);
    for (int d_idx = 0; d_idx < daughters.size(); ++d_idx) {
        const int n = daughters[d_idx];
        os << pre << "+-> ";
        const char pre_first = d_idx == daughters.size() - 1 ? ' ' : '|';
        const std::string pre_d = pre + pre_first;
        PrintDecayChainParticle(evt,
                                n,
                                GenPart_pdgId,
                                GenPart_genPartIdxMother,
                                GenPart_statusFlags,
                                GenPart_pt,
                                GenPart_eta,
                                GenPart_phi,
                                GenPart_mass,
                                GenPart_status,
                                pre_d,
                                GenPart_daughters,
                                os);
    }
}

int PrintDecayChain(ULong64_t evt,
                    const RVecI &GenPart_pdgId,
                    const RVecI &GenPart_genPartIdxMother,
                    const RVecI &GenPart_statusFlags,
                    const RVecF &GenPart_pt,
                    const RVecF &GenPart_eta,
                    const RVecF &GenPart_phi,
                    const RVecF &GenPart_mass,
                    const RVecI &GenPart_status,
                    const RVecVecI &GenPart_daughters,
                    const std::string &outFile) {
    std::ofstream out_file(outFile, std::ios_base::app);
    out_file << "event=" << evt << '\n';
    for (int genPart_idx = 0; genPart_idx < GenPart_pdgId.size(); ++genPart_idx) {
        if (GenPart_genPartIdxMother[genPart_idx] == -1)
            PrintDecayChainParticle(evt,
                                    genPart_idx,
                                    GenPart_pdgId,
                                    GenPart_genPartIdxMother,
                                    GenPart_statusFlags,
                                    GenPart_pt,
                                    GenPart_eta,
                                    GenPart_phi,
                                    GenPart_mass,
                                    GenPart_status,
                                    "",
                                    GenPart_daughters,
                                    out_file);
    }
    return 0;
}

int MatchGenLepton(const LorentzVectorM &obj_p4,
                   const std::vector<reco_tau::gen_truth::GenLepton> &genLeptons,
                   float dR_thr) {
    int best_idx = -1;
    float dR_min = dR_thr;
    for (int genLep_idx = 0; genLep_idx < genLeptons.size(); genLep_idx++) {
        auto dR_objGenLep = ROOT::Math::VectorUtil::DeltaR(obj_p4, genLeptons.at(genLep_idx).visibleP4());
        if (dR_objGenLep < dR_min) {
            dR_objGenLep = dR_min;
            best_idx = genLep_idx;
        }
    }
    return best_idx;
}

RVecI MatchGenLepton(const RVecLV &obj_p4,
                     const std::vector<reco_tau::gen_truth::GenLepton> &genLeptons,
                     float dR_thr) {
    RVecI best_indices(obj_p4.size(), -1);
    for (int obj_idx = 0; obj_idx < obj_p4.size(); obj_idx++) {
        best_indices[obj_idx] = MatchGenLepton(obj_p4.at(obj_idx), genLeptons, dR_thr);
    }
    return best_indices;
}

RVecI GetGenLeptonMatch(const RVecI &obj_genMatchIdx, const std::vector<reco_tau::gen_truth::GenLepton> &genLeptons) {
    using Kind = reco_tau::gen_truth::GenLepton::Kind;
    RVecI kind(obj_genMatchIdx.size(), static_cast<int>(GenLeptonMatch::NoMatch));
    for (size_t obj_idx = 0; obj_idx < obj_genMatchIdx.size(); ++obj_idx) {
        const int lep_idx = obj_genMatchIdx.at(obj_idx);
        if (lep_idx >= 0) {
            const auto &genLep = genLeptons.at(lep_idx);
            const double pt_thr = genLep.kind() == Kind::TauDecayedToHadrons ? 15 : 8;
            if (genLep.visibleP4().pt() > pt_thr)
                kind[obj_idx] = static_cast<int>(genLep.kind());
        }
    }
    return kind;
}
