import ROOT
import sys
import os
import math
from FLAF.RunKit.run_tools import ps_call

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from Common.HistHelper import *

unc_to_not_consider_boosted = [
    "PUJetID",
    "JER",
    "JES_FlavorQCD",
    "JES_RelativeBal",
    "JES_HF",
    "JES_BBEC1",
    "JES_EC2",
    "JES_Absolute",
    "JES_Total",
    "JES_BBEC1_2018",
    "JES_Absolute_2018",
    "JES_EC2_2018",
    "JES_HF_2018",
    "JES_RelativeSample_2018",
    "bTagSF_Loose_btagSFbc_correlated",
    "bTagSF_Loose_btagSFbc_uncorrelated",
    "bTagSF_Loose_btagSFlight_correlated",
    "bTagSF_Loose_btagSFlight_uncorrelated",
    "bTagSF_Medium_btagSFbc_correlated",
    "bTagSF_Medium_btagSFbc_uncorrelated",
    "bTagSF_Medium_btagSFlight_correlated",
    "bTagSF_Medium_btagSFlight_uncorrelated",
    "bTagSF_Tight_btagSFbc_correlated",
    "bTagSF_Tight_btagSFbc_uncorrelated",
    "bTagSF_Tight_btagSFlight_correlated",
    "bTagSF_Tight_btagSFlight_uncorrelated",
    "bTagShapeSF_lf",
    "bTagShapeSF_hf",
    "bTagShapeSF_lfstats1",
    "bTagShapeSF_lfstats2",
    "bTagShapeSF_hfstats1",
    "bTagShapeSF_hfstats2",
    "bTagShapeSF_cferr1",
    "bTagShapeSF_cferr2",
]


def GetHisto(channel, category, inFileName, inDir, sample_name, uncSource, scale):
    inFile = ROOT.TFile(os.path.join(inDir, inFileName), "READ")
    dir_0 = inFile.Get(channel)
    dir_1 = dir_0.Get(category)
    # print(channel, category)
    total_histName = sample_name
    # print([str(key.GetName()) for key in inFile.GetListOfKeys()])
    # print([str(key.GetName()) for key in dir_0.GetListOfKeys()])
    # print([str(key.GetName()) for key in dir_1.GetListOfKeys()])
    if uncSource != "Central":
        total_histName += f"_{uncSource}{scale}"
    for key in dir_1.GetListOfKeys():
        key_name = key.GetName()
        # print(key_name, total_histName)
        if key_name != total_histName:
            continue
        obj = key.ReadObj()
        if obj.IsA().InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)
            inFile.Close()
            return obj
    return


def GetShiftedRatios(
    channel, category, inFileName_Central, inDir, sample_name, uncSource
):
    hist_central = GetHisto(
        channel, category, inFileName_Central, inDir, sample_name, "Central", "-"
    )
    # print(hist_central.GetNbinsX())
    hist_up = GetHisto(
        channel, category, inFileName, inDir, sample_name, uncSource, "Up"
    )
    # print(hist_up.GetNbinsX())
    hist_up_ratio = hist_up.Clone("hist_ratio_up")
    hist_up_ratio.Divide(hist_central)
    hist_down = GetHisto(
        channel, category, inFileName, inDir, sample_name, uncSource, "Down"
    )
    hist_down_ratio = hist_down.Clone("hist_ratio_down")
    hist_down_ratio.Divide(hist_central)
    return hist_central, hist_up_ratio, hist_up, hist_down_ratio, hist_down


def fit_function(x, par):
    return par[0] + par[1] * x[0]


def constant_function(x, par):
    return par[0]


def GetChi2(histogram):
    fit_func = ROOT.TF1("fit_func", constant_function, 0, 10, 1)
    fit_func.SetParameter(0, 1.0)
    histogram.Fit(fit_func, "q")
    chi2 = fit_func.GetChisquare()
    ndf = fit_func.GetNDF()
    p_value = ROOT.TMath.Prob(chi2, ndf)
    fit_param = fit_func.GetParameter(0)
    fit_param_error = fit_func.GetParError(0)
    return chi2, p_value, fit_param, fit_param_error


def GetChi2Method(
    hist_central,
    hist_up_ratio,
    hist_up,
    hist_down_ratio,
    hist_down,
    sample,
    ch,
    cat,
    unc,
    unc_dict,
):
    # print(sample,ch,cat,unc)
    chi2_up, p_value_up, fit_param_up, fit_param_error_up = GetChi2(hist_up_ratio)
    chi2_down, p_value_down, fit_param_down, fit_param_error_down = GetChi2(
        hist_down_ratio
    )
    # if p_value_up < 0.05 or p_value_down < 0.05:
    # Stampare i parametri del fit e il chi-quadro ridotto
    hist_integral_central = hist_central.Integral(0, hist_central.GetNbinsX())
    hist_integral_up = hist_up.Integral(0, hist_up.GetNbinsX())
    hist_integral_down = hist_down.Integral(0, hist_down.GetNbinsX())
    hist_ratio_integral_up = hist_up_ratio.Integral(0, hist_up_ratio.GetNbinsX())
    hist_ratio_integral_down = hist_down_ratio.Integral(0, hist_down_ratio.GetNbinsX())
    if sample not in unc_dict.keys():
        unc_dict[sample] = {}

    if unc not in unc_dict[sample].keys():
        unc_dict[sample][unc] = {}
    if ch not in unc_dict[sample][unc].keys():
        unc_dict[sample][unc][ch] = {}

    if cat not in unc_dict[sample][unc][ch].keys():
        unc_dict[sample][unc][ch][cat] = {}
    # print(unc_dict[sample][unc][ch][cat])
    unc_dict[sample][unc][ch][cat] = {
        "Up": {
            "number_bins_central": hist_central.GetNbinsX(),
            "number_bins": hist_up.GetNbinsX(),
            "number_bins_ratio": hist_up_ratio.GetNbinsX(),
            "p_value": p_value_up,
            "p_value_interestng": p_value_up < 0.05,
            "chi2": chi2_up,
            "integral": hist_integral_up,
            "integral_central": hist_integral_central,
            "integral_ratio": hist_ratio_integral_up,
            "intercept": fit_param_up,
            "intercept_error": fit_param_error_up,
        },
        "Down": {
            "number_bins_central": hist_central.GetNbinsX(),
            "number_bins": hist_down.GetNbinsX(),
            "number_bins_ratio": hist_down_ratio.GetNbinsX(),
            "p_value": p_value_down,
            "p_value_interestng": p_value_down < 0.05,
            "chi2": chi2_down,
            "integral": hist_integral_down,
            "integral_central": hist_integral_central,
            "integral_ratio": hist_ratio_integral_down,
            "intercept": fit_param_down,
            "intercept_error": fit_param_error_down,
        },
    }
    # print(unc_dict)

    """
    canvas = ROOT.TCanvas("canvas", "Fit Plot")
    histogram.Draw()
    fit_func.Draw("same")
    canvas.Update()
    canvas.SaveAs(f"output/histograms/fit_chi2/fit_plot_{sample}_{ch}_{cat}_{unc}_{updown}.png")
    """


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--histDir", required=True)
    parser.add_argument("--outDir", required=True)
    parser.add_argument("--inFileName", required=True)
    parser.add_argument("--mass", required=False, type=int, default=1250)
    parser.add_argument("--sampleConfig", required=True, type=str)
    parser.add_argument("--bckgConfig", required=True, type=str)
    parser.add_argument("--uncConfig", required=True, type=str)
    parser.add_argument("--wantBTag", required=False, type=bool, default=False)
    parser.add_argument("--suffix", required=False, type=str, default="")
    parser.add_argument("--var", required=False, type=str, default="tau1_pt")
    args = parser.parse_args()
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptStat(0)

    hist_cfg = os.path.join(os.environ["ANALYSIS_PATH"], "config/plot/histograms.yaml")
    with open(hist_cfg, "r") as f:
        hist_cfg_dict = yaml.safe_load(f)

    with open(args.sampleConfig, "r") as f:
        sample_cfg_dict = yaml.safe_load(f)

    with open(args.bckgConfig, "r") as f:
        bckg_cfg_dict = yaml.safe_load(f)
    signals = list(sample_cfg_dict["GLOBAL"]["signal_types"])
    all_samples_list, all_samples_types = GetSamplesStuff(
        bckg_cfg_dict.keys(), sample_cfg_dict, args.histDir, True, True, False
    )
    all_histlist = {}
    histNamesDict = {}
    # all_vars = list(hist_cfg_dict.keys())
    unc_cfg_dict = {}
    with open(args.uncConfig, "r") as f:
        unc_cfg_dict = yaml.safe_load(f)
    all_uncertainties = list(unc_cfg_dict["norm"].keys())
    all_uncertainties.extend(unc_cfg_dict["shape"].keys())
    # print(all_uncertainties)

    categories = list(sample_cfg_dict["GLOBAL"]["categories"])
    # print(categories)
    btag_dir = "bTag_WP" if args.wantBTag else "bTag_shape"
    # print(btag_dir)
    # print(args.histDir,all_samples_types)
    unc_dict = {}
    for sample in all_samples_list:
        if sample == "data":
            continue
        for uncSource in all_uncertainties:
            # print(uncSource)
            inFileName = f"{args.inFileName}_{args.var}_{uncSource}{args.suffix}.root"
            # print(inFileName)
            inDir = os.path.join(args.histDir, "all_histograms", args.var, btag_dir)
            # print(inDir)
            inFileName_Central = (
                f"{args.inFileName}_{args.var}_Central{args.suffix}.root"
            )
            # print(inFileName_Central)
            # print(os.path.join(inDir,inFileName))
            if not os.path.exists(os.path.join(inDir, inFileName)):
                continue
            # inFileName_Central=f'{args.inFileName}_{args.var}_Central{args.suffix}.root'
            sample_name = sample
            for channel in ["eTau", "muTau", "tauTau"]:
                for category in ["inclusive", "res2b", "res1b", "boosted"]:
                    # if category=='boosted': continue
                    if (
                        category == "boosted"
                        and uncSource in unc_to_not_consider_boosted
                    ):
                        continue
                    # print(channel, category, uncSource, sample_name)

                    hist_central, hist_up_ratio, hist_up, hist_down_ratio, hist_down = (
                        GetShiftedRatios(
                            channel,
                            category,
                            inFileName_Central,
                            inDir,
                            sample_name,
                            uncSource,
                        )
                    )
                    GetChi2Method(
                        hist_central,
                        hist_up_ratio,
                        hist_up,
                        hist_down_ratio,
                        hist_down,
                        sample_name,
                        channel,
                        category,
                        uncSource,
                        unc_dict,
                    )
    outDir = os.path.join(args.outDir, args.var, btag_dir)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    with open(f"{outDir}/slopeInfo.json", "w") as f:
        json.dump(unc_dict, f, indent=4)
