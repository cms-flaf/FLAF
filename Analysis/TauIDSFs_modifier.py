import os
import ROOT

header_path_RootExt = os.path.join(
    os.environ["ANALYSIS_PATH"], "PlotKit/include/RootExt.h"
)
ROOT.gInterpreter.Declare(f'#include "{header_path_RootExt}"')
ROOT.gInterpreter.Declare(
    """
    static std::map<std::string, std::shared_ptr<TH1>> histMap ;
    void FillhistMap(std::string variable,  std::string fileName, std::string histName){
        if(!(histMap.find(variable)!= histMap.end())){
            auto inputFile = root_ext::OpenRootFile(fileName);
            auto hist = root_ext::ReadCloneObject<TH1>(*inputFile, histName, "", true);
            histMap[variable] = std::shared_ptr<TH1>(hist);
        }
    }

    float GetSFValue(float value, std::string variable){
        return histMap.at(variable)->GetBinContent(histMap.at(variable)->FindFixBin(value));
        }
    static std::map<int,float> deepTauV2p5SFs= {
        {0, 0.8861654},
        {1, 0.9226725},
        {10, 0.9329327},
        {11, 0.7653489},
    };
    """
)


def GetNewSFs_Pt(df, weights_to_apply, deepTauVersion="v2p1"):
    hist_path = os.path.join(
        os.environ["ANALYSIS_PATH"],
        "Corrections/data/TAU/2018_UL/TauID_SF_pt_DeepTau2017v2p1VSjet_VSjetMedium_VSeleVVLoose_Mar07.root",
    )
    ROOT.FillhistMap("pt", hist_path, "DMinclusive_2018_hist")
    for tau_idx in [1, 2]:
        df = df.Define(
            f"weight_tau{tau_idx}ID_Central_new",
            f"""tau{tau_idx}_gen_kind == 5 ? GetSFValue(tau{tau_idx}_pt,"pt"):1.f;""",
        )
        if "weight_tau{tau_idx}ID_Central_new" not in weights_to_apply:
            weights_to_apply.append(f"weight_tau{tau_idx}ID_Central_new")
    return df


def GetNewSFs_DM(df, weights_to_apply, deepTauVersion="v2p1"):
    decayModes = ["0", "1", "10", "11"]
    histNames_map_string = {}
    if deepTauVersion == "v2p1":
        hist_path = os.path.join(
            os.environ["ANALYSIS_PATH"],
            "Corrections/data/TAU/2018_UL/TauID_SF_dm_DeepTau2017v2p1VSjet_VSjetMedium_VSeleVVLoose_Mar07.root",
        )
        infile = ROOT.TFile.Open(hist_path, "READ")
        for decayMode in decayModes:
            ROOT.FillhistMap(decayMode, hist_path, f"DM{decayMode}_2018_hist")
        for tau_idx in [1, 2]:
            df = df.Define(
                f"weight_tau{tau_idx}ID_Central_new_DM",
                f"""tau{tau_idx}_gen_kind == 5 ? GetSFValue(tau{tau_idx}_pt,std::to_string(tau{tau_idx}_decayMode)): 1.f;""",
            )
            if f"weight_tau{tau_idx}ID_Central_new_DM" not in weights_to_apply:
                weights_to_apply.append(f"weight_tau{tau_idx}ID_Central_new_DM")
    elif deepTauVersion == "v2p5":

        for tau_idx in [1, 2]:
            df = df.Define(
                f"weight_tau{tau_idx}ID_Central_new_DM",
                f"""tau{tau_idx}_gen_kind == 5 ? deepTauV2p5SFs.at(tau{tau_idx}_decayMode) : 1.f;""",
            )
            weights_to_apply.append(f"weight_tau{tau_idx}ID_Central_new_DM")
            if f"weight_tau{tau_idx}ID_Central_new_DM" not in weights_to_apply:
                weights_to_apply.append(f"weight_tau{tau_idx}ID_Central_new_DM")
    else:
        raise RuntimeError("Error: unknown deepTauVersion")
    return df
