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

import importlib

# Import correct analysis
# from Analysis.hh_bbtautau import *
# import Analysis.hh_bbww as analysis


def createCacheQuantities(dfWrapped_cache, cache_map_name, cache_entry_name):
    df_cache = dfWrapped_cache.df
    map_creator_cache = ROOT.analysis.CacheCreator(*dfWrapped_cache.colTypes)()
    df_cache = map_creator_cache.processCache(
        ROOT.RDF.AsRNode(df_cache),
        Utilities.ListToVector(dfWrapped_cache.colNames),
        cache_map_name,
        cache_entry_name,
    )
    return df_cache


def AddCacheColumnsInDf(
    dfWrapped_central,
    dfWrapped_cache,
    cache_map_name="cache_map_placeholder",
    cache_entry_name="_cache_entry",
):
    col_names_cache = dfWrapped_cache.colNames
    col_types_cache = dfWrapped_cache.colTypes
    # print(col_names_cache)
    # if "kinFit_result" in col_names_cache:
    #    col_names_cache.remove("kinFit_result")
    dfWrapped_cache.df = createCacheQuantities(
        dfWrapped_cache, cache_map_name, cache_entry_name
    )
    if dfWrapped_cache.df.Filter(f"{cache_map_name} > 0").Count().GetValue() <= 0:
        raise RuntimeError("no events passed map placeolder")
    dfWrapped_central.AddCacheColumns(col_names_cache, col_types_cache, cache_map_name)


def createCentralQuantities(df_central, central_col_types, central_columns):
    map_creator = ROOT.analysis.MapCreator(*central_col_types)()
    df_central = map_creator.processCentral(
        ROOT.RDF.AsRNode(df_central), Utilities.ListToVector(central_columns), 1
    )
    # df_central = map_creator.getEventIdxFromShifted(ROOT.RDF.AsRNode(df_central))
    return df_central


def SaveHists(histograms, out_file, categories_to_save):
    for key_tuple, hist_list in histograms.items():
        (key_1, key_2) = key_tuple
        ch, reg, cat = key_1
        if cat not in categories_to_save:
            continue
        sample_type, uncName, scale = key_2
        dir_name = "/".join(key_1)
        dir_ptr = Utilities.mkdir(out_file, dir_name)
        merged_hist = hist_list[0].GetValue()
        for hist in hist_list[1:]:
            merged_hist.Add(hist.GetValue())
        isCentral = "Central" in key_2
        hist_name = sample_type
        if not isCentral:
            hist_name += f"_{uncName}{scale}"
        # print(dir_name, hist_name)
        dir_ptr.WriteTObject(merged_hist, hist_name, "Overwrite")


def GetHistogramDictFromDataframes(
    var,
    all_dataframes,
    key_2,
    key_filter_dict,
    unc_cfg_dict,
    hist_cfg_dict,
    global_cfg_dict,
    furtherCut="",
    verbose=False,
):
    dataframes = all_dataframes[key_2]
    sample_type, uncName, scale = key_2
    isCentral = "Central" in key_2
    # print(f"key2 is {key_2}")
    histograms = {}
    boosted_categories = global_cfg_dict.get("boosted_categories", [])
    boosted_variables = global_cfg_dict.get("var_only_boosted", [])
    categories = global_cfg_dict["categories"]
    all_categories = categories + boosted_categories
    if args.var in boosted_variables:
        all_categories = boosted_categories
    if args.var.startswith("b1") or args.var.startswith("b2"):
        all_categories = categories
    unc_to_not_consider_boosted = global_cfg_dict.get("unc_to_not_consider_boosted", [])
    # if not isCentral:
    #     print(unc_to_not_consider_boosted)

    for key_1, key_cut in key_filter_dict.items():
        ch, reg, cat = key_1
        if cat not in all_categories:
            continue
        if ch not in global_cfg_dict["channels_to_consider"]:
            continue
        if (key_1, key_2) in histograms.keys():
            continue

        if var in boosted_variables and uncName in unc_to_not_consider_boosted:
            continue
        total_weight_expression = (
            analysis.GetWeight(ch, cat, boosted_categories)
            if sample_type != "data"
            else "1"
        )

        weight_name = "final_weight"
        if not isCentral:
            if type(unc_cfg_dict) == dict:
                if (
                    uncName in unc_cfg_dict.keys()
                    and "expression" in unc_cfg_dict[uncName].keys()
                ):
                    weight_name = unc_cfg_dict[uncName]["expression"].format(
                        scale=scale
                    )
        if (key_1, key_2) not in histograms.keys():
            histograms[(key_1, key_2)] = []

        for dataframe in dataframes:
            if furtherCut != "":
                key_cut += f" && {furtherCut}"
            dataframe_new = dataframe.Filter(key_cut)
            btag_weight = (
                analysis.GetBTagWeight(global_cfg_dict, cat, applyBtag=False)
                if sample_type != "data"
                else "1"
            )
            total_weight_expression = "*".join([total_weight_expression, btag_weight])
            # dataframe_new = dataframe_new.Define(f"final_weight_0_{ch}_{cat}_{reg}", f"{total_weight_expression}") # no need to define it twice
            dataframe_new = dataframe_new.Filter(f"{cat}")
            histograms[(key_1, key_2)].append(
                dataframe_new.Define("final_weight", total_weight_expression)
                .Define("weight_for_hists", f"{weight_name}")
                .Histo1D(GetModel(hist_cfg_dict, var), var, "weight_for_hists")
            )
    return histograms


def GetShapeDataFrameDict(
    all_dataframes,
    global_cfg_dict,
    key,
    key_central,
    file_keys,
    inFile,
    inFileCache,
    compute_variations,
    period,
    colNames,
    colTypes,
    hasCache=True,
    **kwargset,
):
    sample_type, uncName, scale = key
    if compute_variations and key != key_central and sample_type != "data":
        kwargset["isCentral"] = False
        if key not in all_dataframes.keys():
            all_dataframes[key] = []
        treeName = f"Events_{uncName}{scale}"
        treeName_noDiff = f"{treeName}_noDiff"

        if treeName_noDiff in file_keys:
            # print(treeName_noDiff)
            dfWrapped_noDiff = analysis.DataFrameBuilderForHistograms(
                ROOT.RDataFrame(treeName_noDiff, inFile),
                global_cfg_dict,
                period,
                **kwargset,
            )
            if hasCache:
                dfWrapped_cache_noDiff = analysis.DataFrameBuilderForHistograms(
                    ROOT.RDataFrame(treeName_noDiff, inFileCache),
                    global_cfg_dict,
                    period,
                    **kwargset,
                )
                AddCacheColumnsInDf(
                    dfWrapped_noDiff,
                    dfWrapped_cache_noDiff,
                    f"cache_map_{uncName}{scale}_noDiff",
                )
            dfWrapped_noDiff.AddMissingColumns(colNames, colTypes)
            dfWrapped_noDiff = analysis.PrepareDfForHistograms(dfWrapped_noDiff)
            all_dataframes[key].append(dfWrapped_noDiff.df)
        treeName_Valid = f"{treeName}_Valid"
        if treeName_Valid in file_keys:
            # print(treeName_Valid)
            dfWrapped_Valid = analysis.DataFrameBuilderForHistograms(
                ROOT.RDataFrame(treeName_Valid, inFile),
                global_cfg_dict,
                period,
                **kwargset,
            )
            if hasCache:
                dfWrapped_cache_Valid = analysis.DataFrameBuilderForHistograms(
                    ROOT.RDataFrame(treeName_Valid, inFileCache),
                    global_cfg_dict,
                    period,
                    **kwargset,
                )
                AddCacheColumnsInDf(
                    dfWrapped_Valid,
                    dfWrapped_cache_Valid,
                    f"cache_map_{uncName}{scale}_Valid",
                )
            dfWrapped_Valid.CreateFromDelta(colNames, colTypes)
            dfWrapped_Valid = analysis.PrepareDfForHistograms(dfWrapped_Valid)
            dfWrapped_Valid.AddMissingColumns(colNames, colTypes)
            all_dataframes[key].append(dfWrapped_Valid.df)
        treeName_nonValid = f"{treeName}_nonValid"
        if treeName_nonValid in file_keys:
            # print(treeName_nonValid)
            dfWrapped_nonValid = analysis.DataFrameBuilderForHistograms(
                ROOT.RDataFrame(treeName_nonValid, inFile),
                global_cfg_dict,
                period,
                **kwargset,
            )
            if hasCache:
                dfWrapped_cache_nonValid = analysis.DataFrameBuilderForHistograms(
                    ROOT.RDataFrame(treeName_nonValid, inFileCache),
                    global_cfg_dict,
                    period,
                    **kwargset,
                )
                AddCacheColumnsInDf(
                    dfWrapped_nonValid,
                    dfWrapped_cache_nonValid,
                    f"cache_map_{uncName}{scale}_nonValid",
                )
            dfWrapped_nonValid.AddMissingColumns(colNames, colTypes)
            dfWrapped_nonValid = analysis.PrepareDfForHistograms(dfWrapped_nonValid)
            all_dataframes[key].append(dfWrapped_nonValid.df)
        if not all_dataframes[key]:
            all_dataframes.pop(key)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--cacheFile", required=False, type=str, default="")
    parser.add_argument("--outFileName", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--sampleType", required=True, type=str)
    parser.add_argument("--deepTauVersion", required=False, type=str, default="v2p1")
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compute_rel_weights", type=bool, default=False)
    parser.add_argument("--histConfig", required=True, type=str)
    parser.add_argument("--globalConfig", required=True, type=str)
    parser.add_argument("--uncConfig", required=True, type=str)
    parser.add_argument("--var", required=True, type=str)
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--furtherCut", required=False, type=str, default="")
    parser.add_argument("--region", required=False, type=str, default="SR")
    parser.add_argument(
        "--channels", required=False, type=str, default="eTau,muTau,tauTau"
    )
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()

    startTime = time.time()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
    ROOT.gInterpreter.Declare(f'#include "FLAF/include/pnetSF.h"')
    ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
    ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')
    ROOT.gROOT.ProcessLine(
        f'#include "FLAF/include/MT2.h"'
    )  # Include due to DNN using these
    ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')
    # if not os.path.isdir(args.outDir):
    #    os.makedirs(args.outDir)
    # if args.furtherCut:
    #     print(f"further cut = {args.furtherCut}")

    hist_cfg_dict = {}
    with open(args.histConfig, "r") as f:
        hist_cfg_dict = yaml.safe_load(f)
    unc_cfg_dict = {}
    with open(args.uncConfig, "r") as f:
        unc_cfg_dict = yaml.safe_load(f)
    global_cfg_dict = {}
    with open(args.globalConfig, "r") as f:
        global_cfg_dict = yaml.safe_load(f)

    analysis_import = global_cfg_dict["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")

    # Only import KinFit for bbtautau analysis
    if os.path.exists(
        os.path.join(os.environ["ANALYSIS_PATH"], "include/KinFitNamespace.h")
    ):
        ROOT.gInterpreter.Declare(f'#include "include/KinFitNamespace.h"')

    global_cfg_dict["channels_to_consider"] = args.channels.split(",")
    # print(global_cfg_dict['channels_to_consider'])
    # central hist definition
    create_new_hist = False
    key_not_exist = False
    df_empty = False
    inFile_root = ROOT.TFile.Open(args.inFile, "READ")
    inFile_keys = [k.GetName() for k in inFile_root.GetListOfKeys()]
    # print(inFile_keys)
    if "Events" not in inFile_keys:
        key_not_exist = True
    inFile_root.Close()
    if (
        not key_not_exist
        and ROOT.RDataFrame("Events", args.inFile).Count().GetValue() == 0
    ):
        df_empty = True
    isData = args.dataset == "data"
    scales = global_cfg_dict["scales"]
    categories = global_cfg_dict["categories"]
    boosted_categories = global_cfg_dict.get("boosted_categories", [])
    boosted_variables = global_cfg_dict.get("boosted_variables", [])
    all_categories = categories + boosted_categories
    if args.var in boosted_variables:
        all_categories = boosted_categories
    if args.var.startswith("b1") or args.var.startswith("b2") or args.var == "kinFit_m":
        all_categories = categories
    datasetType = 3
    if args.sampleType == "TT":
        datasetType = 1
    if args.sampleType == "DY":
        datasetType = 2
    # if args.sampleType in global_cfg_dict['signal_types']:
    #     datasetType = 0
    # print(f"datasetType is {datasetType}")
    # print()
    create_new_hist = key_not_exist or df_empty
    all_dataframes = {}
    all_dataframes_shape = {}
    all_histograms = {}
    compute_rel_weights_not_data = args.compute_rel_weights and args.dataset != "data"
    if not create_new_hist:
        if args.deepTauVersion == "":
            args.deepTauVersion = "PlaceHolder"
        # Put kwargset into config later
        kwargset = {}
        if analysis_import == "Analysis.hh_bbww":
            kwargset = {}
        if analysis_import == "Analysis.hh_bbtautau":
            kwargset["deepTauVersion"] = args.deepTauVersion
            kwargset["bTagWPString"] = "Medium"
            kwargset["pNetWPstring"] = "Loose"
            kwargset["region"] = args.region
            kwargset["isData"] = isData
            kwargset["isCentral"] = True
            kwargset["wantTriggerSFErrors"] = compute_rel_weights_not_data
            kwargset["whichType"] = datasetType
        if analysis_import == "Analysis.H_mumu":
            ROOT.gInterpreter.Declare(
                f'#include "include/Helper.h"'
            )  # not related to FullEvtId definition but needed for analysis specific purpose. At a certain point it will be moved to analysis specific section.
            kwargset["isData"] = isData
            kwargset["isCentral"] = True
            kwargset["wantTriggerSFErrors"] = compute_rel_weights_not_data

        all_dataframes = {}
        all_histograms = {}

        dfWrapped_central = analysis.DataFrameBuilderForHistograms(
            ROOT.RDataFrame("Events", args.inFile),
            global_cfg_dict,
            args.period,
            **kwargset,
        )

        key_central = (args.dataset, "Central", "Central")
        key_filter_dict = analysis.createKeyFilterDict(global_cfg_dict, args.period)

        outfile = ROOT.TFile(args.outFileName, "RECREATE")

        # print(col_names_central)

        hasCache = args.cacheFile != ""
        if hasCache:
            dfWrapped_cache_central = analysis.DataFrameBuilderForHistograms(
                ROOT.RDataFrame("Events", args.cacheFile),
                global_cfg_dict,
                args.period,
                **kwargset,
            )
            # dfWrapped_cache_central = analysis.DataFrameBuilderForHistograms(ROOT.RDataFrame('Events',args.cacheFile),global_cfg_dict, args.period, deepTauVersion=args.deepTauVersion, bTagWPString = "Medium",pNetWPstring="Loose",region=args.region,isData=isData,isCentral=True, wantTriggerSFErrors=compute_rel_weights_not_data,whichType=datasetType)

            AddCacheColumnsInDf(
                dfWrapped_central, dfWrapped_cache_central, "cache_map_Central"
            )

        col_names_central = dfWrapped_central.colNames
        col_types_central = dfWrapped_central.colTypes
        new_dfWrapped_Central = analysis.PrepareDfForHistograms(dfWrapped_central)
        if key_central not in all_dataframes:

            all_dataframes[key_central] = [new_dfWrapped_Central.df]
        central_histograms = GetHistogramDictFromDataframes(
            args.var,
            all_dataframes,
            key_central,
            key_filter_dict,
            unc_cfg_dict["norm"],
            hist_cfg_dict,
            global_cfg_dict,
            args.furtherCut,
            False,
        )

        # central quantities definition
        compute_variations = (
            args.compute_unc_variations or args.compute_rel_weights
        ) and args.dataset != "data"
        if compute_variations:
            all_dataframes[key_central][0] = createCentralQuantities(
                all_dataframes[key_central][0], col_types_central, col_names_central
            )
            # If original count is already 0, then you don't need to raise error
            if (
                all_dataframes[key_central][0]
                .Filter("map_placeholder > 0")
                .Count()
                .GetValue()
                <= 0
            ):
                raise RuntimeError("no events passed map placeolder")
        # norm weight histograms

        if compute_rel_weights_not_data:
            for uncName in unc_cfg_dict["norm"].keys():
                for scale in scales:
                    # print(uncName, scale)
                    key_2 = (args.dataset, uncName, scale)
                    if key_2 not in all_dataframes.keys():
                        all_dataframes[key_2] = []
                    all_dataframes[key_2] = [all_dataframes[key_central][0]]
                    kwargset["isCentral"] = "Central" in key_2
                    norm_histograms = GetHistogramDictFromDataframes(
                        args.var,
                        all_dataframes,
                        key_2,
                        key_filter_dict,
                        unc_cfg_dict["norm"],
                        hist_cfg_dict,
                        global_cfg_dict,
                        args.furtherCut,
                        False,
                    )
                    central_histograms.update(norm_histograms)

        # save histograms
        SaveHists(central_histograms, outfile, all_categories)

        # shape weight  histograms
        all_dataframes_shape[key_central] = [all_dataframes[key_central][0]]
        compute_shape = args.compute_unc_variations and args.dataset != "data"
        if compute_shape:
            for uncName in unc_cfg_dict["shape"]:
                print(f"uncname is {uncName}")
                for scale in scales:
                    key_2 = (args.dataset, uncName, scale)
                    print(f" key 2 is {key_2}")
                    GetShapeDataFrameDict(
                        all_dataframes_shape,
                        global_cfg_dict,
                        key_2,
                        key_central,
                        inFile_keys,
                        args.inFile,
                        args.cacheFile,
                        compute_shape,
                        args.period,
                        col_names_central,
                        col_types_central,
                        hasCache,
                        **kwargset,
                    )
                    if key_2 not in all_dataframes_shape.keys():
                        continue
                    if all_dataframes_shape[key_2] == []:
                        print("empty list")
                        continue
                    shape_histograms = GetHistogramDictFromDataframes(
                        args.var,
                        all_dataframes_shape,
                        key_2,
                        key_filter_dict,
                        unc_cfg_dict["shape"],
                        hist_cfg_dict,
                        global_cfg_dict,
                        args.furtherCut,
                        True,
                    )
                    SaveHists(shape_histograms, outfile, all_categories)
        outfile.Close()

    else:
        print(f"NO HISTOGRAM CREATED!!!! dataset: {args.dataset} ")
        createVoidHist(args.outFileName, hist_cfg_dict[args.var])

    # finally:
    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
