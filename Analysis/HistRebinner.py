import ROOT
import sys
import os
import math
import array
import time

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


def GetHisto(channel, category, inFile, hist_name, uncSource, scale):
    dir_0 = inFile.Get(channel)
    dir_1 = dir_0.Get(category)
    # print(channel, category)
    # print([str(key.GetName()) for key in inFile.GetListOfKeys()])
    # print([str(key.GetName()) for key in dir_0.GetListOfKeys()])
    # print([str(key.GetName()) for key in dir_1.GetListOfKeys()])
    # if uncSource != 'Central':
    #    total_histName += f'_{uncSource}{scale}'
    for key in dir_1.GetListOfKeys():
        key_name = key.GetName()
        # print(key_name, hist_name)
        if key_name != hist_name:
            continue
        obj = key.ReadObj()
        if obj.IsA().InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)
            return obj
    return


def RebinHisto(hist_initial, new_binning):
    new_binning_array = array.array("d", new_binning)
    new_hist = hist_initial.Rebin(len(new_binning) - 1, "new_hist", new_binning_array)
    return new_hist


def getNewBins(bins):
    if type(bins) == list:
        final_bins = bins
    else:
        n_bins, bin_range = bins.split("|")
        start, stop = bin_range.split(":")
        bin_width = (int(stop) - int(start)) / int(n_bins)
        final_bins = []
        bin_center = int(start)
        while bin_center >= int(start) and bin_center <= int(stop):
            final_bins.append(bin_center)
            bin_center = bin_center + bin_width
    return final_bins


def SaveHists(hist_rebinned_dict, outfile):
    for key_1, hist_dict in hist_rebinned_dict.items():
        ch, cat, uncSource, uncScale = key_1
        key_dir = ch, cat
        # if cat == 'btag_shape': continue
        dir_name = "/".join(key_dir)
        dir_ptr = mkdir(outfile, dir_name)
        dir_ptr.WriteTObject(hist_dict["histogram"], hist_dict["name"], "Overwrite")


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--inFile", required=True)
    parser.add_argument("--outFile", required=True)
    parser.add_argument("--sampleConfig", required=True, type=str)
    parser.add_argument("--bckgConfig", required=True, type=str)
    parser.add_argument("--uncConfig", required=True, type=str)
    parser.add_argument("--histConfig", required=True, type=str)
    parser.add_argument("--suffix", required=False, type=str, default="")
    parser.add_argument("--uncSource", required=False, type=str, default="")
    parser.add_argument("--var", required=False, type=str, default="kinFit_m")
    parser.add_argument("--category", required=False, type=str, default="")
    parser.add_argument("--channel", required=False, type=str, default="")
    parser.add_argument("--wantBTag", required=False, type=bool, default=False)
    args = parser.parse_args()
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptStat(0)
    startTime = time.time()
    # python3 /afs/cern.ch/work/v/vdamante/hhbbTauTauRes/prod/Framework/Analysis/HistRebinner.py --histDir /eos/home-v/vdamante/HH_bbtautau_resonant_Run2/histograms/Run2_2018/v9_deepTau2p1 --inFileName all_histograms --sampleConfig /afs/cern.ch/work/v/vdamante/hhbbTauTauRes/prod/Framework/config/samples_Run2_2018.yaml --var kinFit_m --uncConfig /afs/cern.ch/work/v/vdamante/hhbbTauTauRes/prod/Framework/config/weight_definition.yaml --histConfig /afs/cern.ch/work/v/vdamante/hhbbTauTauRes/prod/Framework/config/plots/histograms.yaml --uncSource Central

    with open(args.sampleConfig, "r") as f:
        sample_cfg_dict = yaml.safe_load(f)

    with open(args.bckgConfig, "r") as f:
        bckg_cfg_dict = yaml.safe_load(f)

    signals = list(sample_cfg_dict["GLOBAL"]["signal_types"])
    wantSignals = False
    wantAllMasses = False
    wantOneMass = False
    all_samples_list, all_samples_types = GetSamplesStuff(
        bckg_cfg_dict.keys(), sample_cfg_dict, wantSignals, wantAllMasses, wantOneMass
    )
    unc_cfg_dict = {}
    with open(args.uncConfig, "r") as f:
        unc_cfg_dict = yaml.safe_load(f)
    hist_cfg_dict = {}

    with open(args.histConfig, "r") as f:
        hist_cfg_dict = yaml.safe_load(f)
    all_uncertainties = list(unc_cfg_dict["norm"].keys())
    all_uncertainties.extend(unc_cfg_dict["shape"].keys())

    categories = list(sample_cfg_dict["GLOBAL"]["categories"])
    btag_dir = "bTag_WP" if args.wantBTag else "bTag_shape"
    unc_dict = {}

    outfile = ROOT.TFile(args.outFile, "RECREATE")
    scales_to_consider = scales if args.uncSource != "Central" else ["Central"]
    # print(all_samples_list)
    print(f"{args.inFile}")
    if not os.path.exists(args.inFile):
        print(f"{args.inFile} does not exist")
    else:
        inFile = ROOT.TFile(args.inFile, "READ")
        for sample_type in all_samples_list + ["QCD", "data"]:
            # print(sample_type)
            hist_rebinned_dict = {}
            for channel in sample_cfg_dict["GLOBAL"]["channelSelection"]:
                if args.channel != "" and channel != args.channel:
                    continue
                for category in sample_cfg_dict["GLOBAL"]["categories"]:
                    # if category == 'boosted' and 'b#Tag' in args.uncSource: continue
                    if (
                        category == "boosted"
                        and args.uncSource in unc_to_not_consider_boosted
                    ):
                        continue
                    if args.category != "" and category != args.category:
                        continue
                    bins_to_compute = (
                        hist_cfg_dict[args.var]["x_rebin"][channel][category]
                        if "x_rebin" in hist_cfg_dict[args.var].keys()
                        else hist_cfg_dict[args.var]["x_bins"]
                    )
                    new_bins = getNewBins(bins_to_compute)
                    total_histName = sample_type
                    for uncScale in scales_to_consider:
                        hist_name = sample_type
                        # print(uncScale)
                        if args.uncSource != "Central":
                            if sample_type == "data":
                                continue
                            if uncScale == "Central":
                                continue
                            hist_name += f"_{args.uncSource}{uncScale}"
                        uncScale_str = "-" if args.uncSource == "Central" else uncScale
                        # print(channel, category, uncScale, sample_type, uncScale_str)
                        print(f"hist name = {hist_name}")

                        hist_initial = GetHisto(
                            channel,
                            category,
                            inFile,
                            hist_name,
                            args.uncSource,
                            uncScale_str,
                        )
                        # print(f'hist initial entries = {hist_initial.GetEntries()}')
                        # print(channel, category, sample_type, hist_initial.Integral(0, hist_initial.GetNbinsX()+1))
                        if hist_initial.Integral(0, hist_initial.GetNbinsX() + 1) == 0:
                            hist_final = ROOT.TH1D(
                                hist_initial.GetName(),
                                hist_initial.GetTitle(),
                                len(new_bins) - 1,
                                array.array("d", new_bins),
                            )
                        else:
                            hist_final = RebinHisto(hist_initial, new_bins)
                            if sample_type == "QCD":
                                if not FixNegativeContributions(hist_final):
                                    print(
                                        f"for {channel} {category} QCD has negative contributions"
                                    )

                        hist_rebinned_dict[
                            (channel, category, args.uncSource, uncScale)
                        ] = {"name": hist_name, "histogram": hist_final}
                    SaveHists(hist_rebinned_dict, outfile)
        inFile.Close()
        outfile.Close()

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
