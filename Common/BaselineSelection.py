import ROOT
import os
from .Utilities import *

initialized = False

ana_reco_object_collections = {
    "v12": [
        "Electron",
        "Muon",
        "Tau",
        "Jet",
        "FatJet",
        "MET",
        "PuppiMET",
        "DeepMETResponseTune",
        "DeepMETResolutionTune",
        "SubJet",
    ],
    "v14": [
        "Electron",
        "Muon",
        "Tau",
        "Jet",
        "FatJet",
        "SubJet",
        "PFMET",
        "PuppiMET",
        "DeepMETResponseTune",
        "DeepMETResolutionTune",
    ],
}
deepTauVersions = {"2p1": "2017", "2p5": "2018"}


def Initialize(loadTF=False, loadHHBtag=False):
    global initialized
    if not initialized:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        ROOT.gROOT.ProcessLine(".include " + os.environ["ANALYSIS_PATH"])
        header_path_RootExt = "include/RootExt.h"
        header_path_GenLepton = "include/GenLepton.h"
        header_path_Gen = "include/BaselineGenSelection.h"
        header_path_Reco = "include/BaselineRecoSelection.h"
        header_path_HHbTag = "include/HHbTagScores.h"
        header_path_AnalysisMath = "include/AnalysisMath.h"
        ROOT.gInterpreter.Declare(f'#include "{header_path_RootExt}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_GenLepton}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_Gen}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_Reco}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_AnalysisMath}"')
        for wpcl in [
            WorkingPointsTauVSe,
            WorkingPointsTauVSmu,
            WorkingPointsTauVSjet,
            WorkingPointsbTag,
            WorkingPointsMuonID,
        ]:
            ROOT.gInterpreter.Declare(f"{generate_enum_class(wpcl)}")
        if loadTF:
            import FLAF.RunKit.includeCMSSWlibs as IncludeLibs

            IncludeLibs.includeLibTool("tensorflow")
        if loadHHBtag:
            lib_path = os.path.join(
                os.environ["FLAF_CMSSW_BASE"],
                "lib",
                os.environ["FLAF_CMSSW_ARCH"],
                "libHHToolsHHbtag.so",
            )
            load_result = ROOT.gSystem.Load(lib_path)
            if load_result != 0:
                raise RuntimeError(
                    f"HHBtagWrapper failed to load with status {load_result}"
                )
            ROOT.gInterpreter.Declare(f'#include "{header_path_HHbTag}"')
            ROOT.gROOT.ProcessLine(
                f'HHBtagWrapper::Initialize("{os.environ["CMSSW_BASE"]}/src/HHTools/HHbtag/models/", 3)'
            )

        initialized = True


def applyMETFlags(df, MET_flags, badMET_flag_runs, isData):
    if badMET_flag_runs:
        df = applyBadMETfilter(df, badMET_flag_runs, isData)
    MET_flags_string = " && ".join(MET_flags)
    return df.Filter(MET_flags_string, "MET filters")


def applyBadMETfilter(df, badMET_flag_runs, isData):
    if not isData:
        return df
    else:
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2#ECal_BadCalibration_Filter_Flag
        df = df.Define(
            f"Flag_badMET_calib",
            f""" !( PuppiMET_p4.pt()>100 &&
                                                Any(v_ops::pt(Jet_p4) > 50
                                                && v_ops::eta(Jet_p4) >= -0.5 && v_ops::eta(Jet_p4) <= -0.1
                                                && v_ops::phi(Jet_p4) >= -2.1 && v_ops::phi(Jet_p4) <= -1.8
                                                && abs(PuppiMET_p4.phi() - v_ops::phi(Jet_p4)) > 2.9
                                                && (Jet_neEmEF > 0.9 || Jet_chEmEF > 0.9)
                                                ) )""",
        )

        df = df.Redefine(
            f"Flag_ecalBadCalibFilter",
            f" ( run >= {badMET_flag_runs[0]} && run <= {badMET_flag_runs[1]} ) ? Flag_badMET_calib : Flag_ecalBadCalibFilter",
        )
        return df


def DefineGenObjects(
    df, isData=False, isHH=False, Hbb_AK4mass_mpv=125.0, p4_suffix="nano"
):
    if isData:
        df = df.Define("genLeptons", "std::vector<reco_tau::gen_truth::GenLepton>()")
    else:
        df = df.Define("GenPart_daughters", "GetDaughters(GenPart_genPartIdxMother)")
        df = df.Define(
            "genLeptons",
            """reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta,
                                        GenPart_phi, GenPart_mass, GenPart_genPartIdxMother, GenPart_pdgId,
                                        GenPart_statusFlags, event)""",
        )

    for lep in ["Electron", "Muon", "Tau"]:
        dR_matching = 0.2 if lep != "Muon" else 0.1
        df = df.Define(
            f"{lep}_genMatchIdx",
            f"MatchGenLepton({lep}_p4_{p4_suffix}, genLeptons, {dR_matching})",
        )
        df = df.Define(
            f"{lep}_genMatch", f"GetGenLeptonMatch({lep}_genMatchIdx, genLeptons)"
        )
    if isData:
        return df

    if isHH:
        df = df.Define(
            "genHttCandidate",
            """GetGenHTTCandidate(event, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, false)""",
        )
        df = df.Define(
            "genHbbIdx",
            """GetGenHBBIndex(event, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags)""",
        )
        df = df.Define("genHbb_isBoosted", "GenPart_pt[genHbbIdx]>550")
    for var in ["GenJet", "GenJetAK8", "SubGenJetAK8"]:
        df = df.Define(f"{var}_idx", f"CreateIndexes({var}_pt.size())")
        df = df.Define(
            f"{var}_p4", f"GetP4({var}_pt,{var}_eta,{var}_phi,{var}_mass, {var}_idx)"
        )

    df = df.Define("GenJet_b_PF", "abs(GenJet_partonFlavour)==5")
    df = df.Define("GenJetAK8_b_PF", "abs(GenJetAK8_partonFlavour)==5")
    df = df.Define(
        "GenJet_Hbb",
        f"FindTwoJetsClosestToMPV({Hbb_AK4mass_mpv}, GenJet_p4, GenJet_b_PF)",
    )
    df = df.Define("GenJetAK8_Hbb", "FindGenJetAK8(GenJetAK8_mass, GenJetAK8_b_PF)")

    return df


def SelectRecoP4(df, syst_name="nano", nano_version="v12"):
    for obj in ana_reco_object_collections[nano_version]:
        df = df.Define(f"{obj}_p4", f"{obj}_p4_{syst_name}")
    return df


def CreateRecoP4(df, suffix="nano", nano_version="v12"):
    if len(suffix) > 0:
        suffix = "_" + suffix
    if "TrigObj_pt" in df.GetColumnNames():
        df = df.Define(f"TrigObj_idx", f"CreateIndexes(TrigObj_pt.size())")
        if "TrigObj_mass" not in df.GetColumnNames():
            df = df.Define("TrigObj_mass", "RVecF(TrigObj_pt.size(), 0.f)")
        df = df.Define(
            f"TrigObj_p4",
            f"GetP4(TrigObj_pt,TrigObj_eta,TrigObj_phi, TrigObj_mass, TrigObj_idx)",
        )
    for obj in ana_reco_object_collections[nano_version]:
        if "MET" in obj:
            df = df.Define(
                f"{obj}_p4{suffix}", f"LorentzVectorM({obj}_pt, 0., {obj}_phi, 0.)"
            )
        else:
            df = df.Define(f"{obj}_idx", f"CreateIndexes({obj}_pt.size())")
            df = df.Define(
                f"{obj}_p4{suffix}",
                f"GetP4({obj}_pt, {obj}_eta, {obj}_phi, {obj}_mass, {obj}_idx)",
            )
    return df


# From JERC recommendation:
# veto events if ANY jet with a loose selection lies in the veto regions. The nominal “loose selection” would be:
# jet pT > 15 GeV
# tight jet ID
# (jet charged EM fraction + jet neutral EM fraction) < 0.9
# jets that don’t overlap with PF muon (dR < 0.2)

# From json file:
# Non-zero value for (eta, phi) indicates that the region is vetoed.


def ApplyJetVetoMap(df, apply_filter=True):
    df = df.Define(
        f"Jet_vetoMapLooseRegion_presel",
        "Jet_pt > 15 && ( Jet_jetId & 2 ) && Jet_chHEF + Jet_neHEF < 0.9 && Jet_isInsideVetoRegion",
    )  #  (Jet_puId > 0 || Jet_pt >50) &&  for CHS jets
    df = df.Define(f"Muon_p4_pfCand", "Muon_p4[Muon_isPFcand]")
    df = df.Define(
        f"Jet_vetoMap",
        " RemoveOverlaps(Jet_p4, Jet_vetoMapLooseRegion_presel, Muon_p4_pfCand, 0.2)",
    )
    if apply_filter:
        return df.Filter(f"Jet_p4[Jet_vetoMap].size()==0", "Jet Veto Map filter")
    return df
