import ROOT
import sys
import os
import math
import shutil
import time

ROOT.EnableThreadSafety()

from FLAF.RunKit.run_tools import ps_call

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
from FLAF.Common.HistHelper import *
from FLAF.Analysis.hh_bbtautau import *


def createCacheQuantities(dfWrapped_cache, cache_map_name):
    df_cache = dfWrapped_cache.df
    map_creator_cache = ROOT.analysis.CacheCreator(*dfWrapped_cache.colTypes)()
    df_cache = map_creator_cache.processCache(
        ROOT.RDF.AsRNode(df_cache),
        Utilities.ListToVector(dfWrapped_cache.colNames),
        cache_map_name,
    )
    return df_cache


# def clean_map_placeholder(dfWrapped_cache):
#    map_creator_cache = ROOT.analysis.CacheCreator(*dfWrapped_cache.colTypes)()
#    map_creator_cache.clean_map()


def AddCacheColumnsInDf(
    dfWrapped_central, dfWrapped_cache, cache_map_name="cache_map_placeholder"
):
    col_names_cache = dfWrapped_cache.colNames
    col_tpyes_cache = dfWrapped_cache.colTypes
    # if "kinFit_result" in col_names_cache:
    #    col_names_cache.remove("kinFit_result")
    dfWrapped_cache.df = createCacheQuantities(dfWrapped_cache, cache_map_name)
    if dfWrapped_cache.df.Filter(f"{cache_map_name} > 0").Count().GetValue() <= 0:
        raise RuntimeError("no events passed map placeolder")
    dfWrapped_central.AddCacheColumns(col_names_cache, col_tpyes_cache)
    # ROOT.gInterpreter.ProcessLine("""delete analysis::GetEntriesMap();""")


def SaveHists(histograms, out_file):
    for key_1, hist in histograms.items():
        print(key_1)
        ch, reg, cat = key_1
        dir_name = "/".join(key_1)
        dir_ptr = mkdir(out_file, dir_name)
        hist_name = sample_type
        dir_ptr.WriteTObject(hist.GetValue(), hist_name, "Overwrite")
        print("written")


def GetHistogramDictFromDataframes(
    var,
    df_central,
    key_filter_dict,
    hist_cfg_dict,
    wantBTag=False,
    want2D=False,
    furtherCut="",
):

    histograms = {}
    # print("A")
    for key_1, key_cut in key_filter_dict.items():
        ch, reg, cat = key_1
        if key_1 in histograms.keys():
            continue
        if cat == "boosted" and var in bjet_vars:
            continue
        if cat != "boosted" and var in var_to_add_boosted:
            continue
        # print(ch, reg, cat)
        total_weight_expression = GetWeight(ch, cat) if sample_type != "data" else "1"
        # print(df_central.GetColumnNames())
        df_central_new = df_central.Filter(key_cut)
        df_central_new = df_central_new.Define(
            f"final_weight_0_{ch}_{cat}_{reg}", f"{total_weight_expression}"
        )
        # print(f"final_weight_0_{ch}_{cat}_{reg} expression = {total_weight_expression}")
        final_string_weight = (
            ApplyBTagWeight(
                cat,
                applyBtag=wantBTag,
                finalWeight_name=f"final_weight_0_{ch}_{cat}_{reg}",
            )
            if sample_type != "data"
            else "1"
        )
        df_central_new = df_central_new.Filter(f"{cat}")
        weight_name = "final_weight"
        # print("B")
        if cat == "btag_shape":
            final_string_weight = f"final_weight_0_{ch}_{cat}_{reg}"
        # print(ch, cat, final_string_weight)
        histograms[key_1] = df_central_new.Define(
            "weight_for_hists", f"{final_string_weight}"
        ).Histo1D(GetModel(hist_cfg_dict, var), var, "weight_for_hists")
        # print("C")
        if want2D and not wantBTag:
            # if 'nBJets' not in dataframe.GetColumnNames():
            #    dataframe = dataframe.Define("nBJets", "ExtraJet_pt[abs(ExtraJet_eta) < 2.5].size()")
            histograms[key_1] = df_central_new.Define(
                "weight_for_hists", f"{final_string_weight}"
            ).Histo2D(Get2DModel(hist_cfg_dict, var), var, "nBJets", "weight_for_hists")
    # print(histograms)
    return histograms


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--inDir", required=True, type=str)
    parser.add_argument("--cacheDir", required=False, type=str)
    parser.add_argument("--outDir", required=False, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--deepTauVersion", required=False, type=str, default="v2p1")
    parser.add_argument("--histConfig", required=True, type=str)
    parser.add_argument("--var", required=True, type=str)
    parser.add_argument("--sampleConfig", required=True, type=str)
    parser.add_argument("--furtherCut", required=False, type=str, default="")
    parser.add_argument("--wantBTag", required=False, type=bool, default=False)
    parser.add_argument("--want2D", required=False, type=bool, default=False)
    args = parser.parse_args()

    startTime = time.time()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/KinFitNamespace.h"')
    ROOT.gInterpreter.Declare(f'#include "include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')
    ROOT.gROOT.ProcessLine('#include "include/AnalysisTools.h"')
    if not os.path.isdir(args.outDir):
        os.makedirs(args.outDir)
    print(f"further cut = {args.furtherCut}")
    hist_cfg_dict = {}
    with open(args.histConfig, "r") as f:
        hist_cfg_dict = yaml.safe_load(f)
    sample_cfg_dict = {}
    with open(args.sampleConfig, "r") as f:
        sample_cfg_dict = yaml.safe_load(f)

    btag_dir = "bTag_WP" if args.wantBTag else "bTag_shape"

    finalDir = os.path.join(args.outDir, args.var, btag_dir)
    if not os.path.isdir(finalDir):
        os.makedirs(finalDir)
    finalFileName = (
        f"{finalDir}/{args.var}2D_onlyCentral.root"
        if args.want2D
        else f"{finalDir}/{args.var}_onlyCentral.root"
    )
    print(finalFileName)
    outfile = ROOT.TFile(finalFileName, "RECREATE")

    sample_type = (
        sample_cfg_dict[args.dataset]["sampleType"]
        if args.dataset != "data"
        else "data"
    )
    key_filter_dict = createKeyFilterDict(sample_cfg_dict["GLOBAL"])

    dfWrapped_central = DataFrameBuilder(
        ROOT.RDataFrame("Events", f"{args.inDir}/*.root"), args.deepTauVersion
    )
    # print("df wrapped central created")
    # print(dfWrapped_central.df.Count().GetValue())

    if args.cacheDir:
        dfWrapped_cache = DataFrameBuilder(
            ROOT.RDataFrame("Events", f"{args.cacheDir}/*.root"), args.deepTauVersion
        )
        # print("df wrapped cache created")
        # print(dfWrapped_cache.df.Count().GetValue())
        AddCacheColumnsInDf(dfWrapped_central, dfWrapped_cache, "cache_map_Central")
        # print("cache columns added")

    central_histograms = GetHistogramDictFromDataframes(
        args.var,
        PrepareDfWrapped(dfWrapped_central).df,
        key_filter_dict,
        hist_cfg_dict,
        args.wantBTag,
        args.want2D,
        args.furtherCut,
    )
    # central quantities definition
    print("got central histograms")
    # save histograms
    SaveHists(central_histograms, outfile)
    print("hist saved")
    outfile.Close()

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
