import ROOT
import sys
import os
import time
import importlib

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
import FLAF.Common.Setup as Setup
from FLAF.Common.HistHelper import *


def checkFile(inFileRoot, channels, qcdRegions, categories):
    keys_channels = [str(key.GetName()) for key in inFileRoot.GetListOfKeys()]
    for channel in channels:
        if channel not in keys_channels:
            return False
    for channel in channels:
        dir_0 = inFileRoot.Get(channel)
        keys_qcdRegions = [str(key.GetName()) for key in dir_0.GetListOfKeys()]
        if not all(element in keys_qcdRegions for element in qcdRegions):
            print("check list not worked for qcdRegions")
            return False
        for qcdRegion in qcdRegions:
            dir_1 = dir_0.Get(qcdRegion)
            keys_categories = [str(key.GetName()) for key in dir_1.GetListOfKeys()]
            if not all(element in keys_categories for element in categories):
                print("check list not worked for categories")
                return False
            for cat in categories:
                dir_2 = dir_1.Get(cat)
                keys_histograms = [str(key.GetName()) for key in dir_2.GetListOfKeys()]
                if not keys_histograms:
                    return False
    return True


def fill_hists(
    items_dict,
    all_hist_dict,
    dataset_type,
    var_input,
    unc_sources,
    data_type="data",
):
    """Accumulate the histograms of one input file for all requested unc sources in a
    single pass. Data has no per-uncertainty variations: only its nominal histogram is
    accumulated here; alias_data_variations() links it under the varied keys afterwards
    (QCD estimation looks data up per (unc, scale))."""
    effective_unc_sources = unc_sources if dataset_type != data_type else []
    hists = all_hist_dict.setdefault(dataset_type, {})
    for key_tuple, hist_map in items_dict.items():
        for var, var_hist in hist_map.items():
            unc_scale = resolve_unc_scale(var, var_input, effective_unc_sources)
            if unc_scale is None:
                continue
            final_key = (key_tuple, unc_scale)
            if final_key not in hists:
                var_hist.SetDirectory(0)
                hists[final_key] = var_hist
            else:
                hists[final_key].Add(var_hist)


def alias_data_variations(all_hist_dict, data_type, unc_sources):
    """Expose the accumulated nominal data histogram under every varied (unc, scale) key.
    Downstream consumers (QCD estimation) only read/clone these entries, so sharing the
    same object is safe; the output-writing loop skips data for non-Central sources."""
    data_hists = all_hist_dict.get(data_type)
    if not data_hists:
        return
    central = {
        key_tuple: hist
        for (key_tuple, unc_scale), hist in data_hists.items()
        if unc_scale == ("Central", "Central")
    }
    for unc_source in unc_sources:
        if unc_source == "Central":
            continue
        for scale in ["Up", "Down"]:
            for key_tuple, hist in central.items():
                data_hists.setdefault((key_tuple, (unc_source, scale)), hist)


def resolve_merged_hist_name(
    dataset_type, uncName, uncScale, unc_sources, all_unc_dict
):
    """Derive the merged output's histogram/tree name for one (dataset_type, uncName,
    uncScale) combination, or None if that combination should be skipped -- the exact
    naming and skip rules the histogram-writing loop uses, factored out so the
    signal-tree writing pass agrees on the same name for the same key."""
    if uncName not in unc_sources:
        return None
    hist_name = dataset_type
    if uncName != "Central":
        if dataset_type == "data":
            return None
        if uncScale == "Central":
            return None
        hist_name += f"""_{all_unc_dict[uncName]["name"].format(uncScale)}"""
    else:
        if uncScale != "Central":
            return None
    return hist_name


def GetBTagWeightDict(
    var, all_hists_dict, categories, boosted_categories, boosted_variables
):
    all_hists_dict_1D = {}
    for dataset_type in all_hists_dict.keys():
        all_hists_dict_1D[dataset_type] = {}
        for key_name, histogram in all_hists_dict[dataset_type].items():
            key_1, key_2 = key_name

            if var not in boosted_variables:
                ch, reg, cat = key_1
                uncName, scale = key_2
                key_tuple_num = ((ch, reg, "btag_shape"), key_2)
                key_tuple_den = ((ch, reg, "inclusive"), key_2)
                ratio_num_hist = (
                    all_hists_dict[dataset_type][key_tuple_num]
                    if key_tuple_num in all_hists_dict[dataset_type].keys()
                    else None
                )
                ratio_den_hist = (
                    all_hists_dict[dataset_type][key_tuple_den]
                    if key_tuple_den in all_hists_dict[dataset_type].keys()
                    else None
                )
                num = ratio_num_hist.Integral(0, ratio_num_hist.GetNbinsX() + 1)
                den = ratio_den_hist.Integral(0, ratio_den_hist.GetNbinsX() + 1)
                ratio = 0.0
                if ratio_den_hist.Integral(0, ratio_den_hist.GetNbinsX() + 1) != 0:
                    ratio = ratio_num_hist.Integral(
                        0, ratio_num_hist.GetNbinsX() + 1
                    ) / ratio_den_hist.Integral(0, ratio_den_hist.GetNbinsX() + 1)
                if (
                    cat in boosted_categories
                    or cat.startswith("btag_shape")
                    or cat.startswith("baseline")
                ):
                    ratio = 1
                histogram.Scale(ratio)
            else:
                print(
                    f"for var {var} no ratio is considered and the histogram is directly saved"
                )

            all_hists_dict_1D[dataset_type][key_name] = histogram
    return all_hists_dict_1D


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("inFiles", nargs="+", type=str)
    parser.add_argument("--dataset_names", required=True, type=str)
    parser.add_argument("--var", required=True, type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--era", required=True, type=str)
    parser.add_argument(
        "--uncSources",
        required=False,
        type=str,
        default="Central",
        help="comma-separated list of uncertainty sources to merge in one pass",
    )
    parser.add_argument("--channels", required=False, type=str, default="")
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    parser.add_argument("--user-custom", type=str, default=None)

    args = parser.parse_args()
    startTime = time.time()

    setup = Setup.Setup(
        os.environ["ANALYSIS_PATH"],
        args.era,
        args.LAWrunVersion,
        user_custom_file=args.user_custom,
    )

    global_cfg_dict = setup.global_params
    unc_cfg_dict = setup.weights_config

    analysis_import = global_cfg_dict["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")
    estimateQCD = global_cfg_dict.get("plot_wantQCD", False)

    # ----> this part is analysis dependent. Need to be put in proper place <-----

    # boosted categories and QCD regions --> e.g. for hmm no boosted categories and no QCD regions but muMu mass regions
    # instead, better to define custom categories/regions
    # boosted_categories = list(
    #     global_cfg_dict.get("boosted_categories", [])
    # )  # list(global_cfg_dict['boosted_categories'])
    # Controlregions = list(global_cfg_dict['ControlRegions']) #Later maybe we want to separate Controls from QCDs

    # Regions def
    regions_name = global_cfg_dict.get(
        "regions", None
    )  # can be extended to list of names, if for example adding QCD regions + other control regions
    regions = []
    if regions_name:
        regions = list(global_cfg_dict.get(regions_name, []))
        if not regions:
            print("No custom regions found")

    # Categories def
    categories = list(global_cfg_dict["categories"])
    # custom_categories_name = global_cfg_dict.get(
    #     "custom_categories", None
    # )  # can be extended to list of names
    # custom_categories = []
    # if custom_categories_name:
    #     custom_categories = list(global_cfg_dict.get(custom_categories_name, []))
    #     if not custom_categories:
    #         print("No custom categories found")
    all_categories = categories  # + custom_categories

    # Channels def
    setup.global_params["channels_to_consider"] = (
        args.channels.split(",")
        if args.channels
        else setup.global_params["channelSelection"]
    )
    channels = setup.global_params["channels_to_consider"]

    # Variables exception def
    custom_variables = global_cfg_dict.get(
        "var_only_custom", {}
    )  # e.g. var only boosted. Will be constructed as:
    # { "cat == boosted" : [particleNet.. ], "cat != boosted" : [b1_.. ]  }
    # replacing this part:
    # if args.var.startswith("b1") or args.var.startswith("b2"):
    #     all_categories = categories

    # Uncertainties
    uncNameTypes = GetUncNameTypes(unc_cfg_dict)
    scales = list(global_cfg_dict["scales"])
    unc_sources = [s.strip() for s in args.uncSources.split(",") if s.strip()]
    unknown_sources = [
        s for s in unc_sources if s != "Central" and s not in uncNameTypes
    ]
    if unknown_sources:
        raise ValueError(f"Unknown uncertainty source(s): {', '.join(unknown_sources)}")
    # Uncertainties exception
    unc_exception = global_cfg_dict.get(
        "unc_exception", {}
    )  # e.g. boosted categories with unc list to not consider
    # { "cat == boosted" : [JER, JES] }
    # unc_to_not_consider_boosted = list(
    #     global_cfg_dict.get("unc_to_not_consider_boosted", [])
    # )

    # file structure : channel - region - category - varName_unc (if not central, else only varName)

    # Datasets
    dataset_cfg_dict = setup.datasets
    data_process_name = "data"
    data_processes = setup.phys_model.processes(process_type="data")
    if len(data_processes) > 0:
        data_base_processes = setup.phys_model.base_processes(data_processes[0])
        data_process_name = data_base_processes[0]
    dataset_cfg_dict["data"] = {
        "process_name": data_process_name
    }  # Data isn't actually in config dict, but just add it here to keep working format
    # print(dataset_cfg_dict)

    all_hists_dict = {}
    all_signal_hist_dict = {}
    signal_tree_sources = {}
    all_datasets = args.dataset_names.split(",")
    for dataset_name, inFile_path in zip(all_datasets, args.inFiles):
        if not os.path.exists(inFile_path):
            print(
                f"input file for dataset {dataset_name} (with path= {inFile_path}) does not exist, skipping"
            )
            continue

        base_process_name = dataset_cfg_dict[dataset_name]["process_name"]
        dataset_type = setup.base_processes[base_process_name]["parent_process"]
        print(dataset_type)
        if dataset_type not in all_hists_dict.keys():
            all_hists_dict[dataset_type] = {}

        with ROOT.TFile.Open(inFile_path, "READ") as inFile:
            # check that the file is ok
            if inFile.IsZombie():
                raise RuntimeError(f"{inFile_path} is zombie")
            if not checkFile(inFile, channels, regions, all_categories):
                raise RuntimeError(f"{dataset_name} has void file")
            all_items = get_all_items_recursive(inFile)
            fill_hists(
                all_items,
                all_hists_dict,
                dataset_type,
                args.var,
                unc_sources,
                data_process_name,
            )

            signal_trees_in_file = get_all_signal_trees_recursive(inFile)
            for key_tuple, tree_map in signal_trees_in_file.items():
                for name, tree_obj in tree_map.items():
                    unc_scale = resolve_unc_scale(name, args.var, unc_sources)
                    if unc_scale is None:
                        continue
                    bucket = signal_tree_sources.setdefault(dataset_type, {})
                    bucket.setdefault((key_tuple, unc_scale), []).append(
                        (inFile_path, "/".join(key_tuple) + "/Trees/" + name)
                    )
                    hist = unwrap_hist_from_tree(tree_obj)
                    fill_hists(
                        {key_tuple: {name: hist}},
                        all_signal_hist_dict,
                        dataset_type,
                        args.var,
                        unc_sources,
                        data_process_name,
                    )
    alias_data_variations(all_hists_dict, data_process_name, unc_sources)

    # here there should be the custom applications - e.g. GetBTagWeightDict, AddQCDInHistDict, etc.
    # analysis.ApplyMergeCustomisations() # --> here go the QCD and bTag functions
    """
    if global_cfg_dict["ApplyBweight"] == True:
        all_hists_dict_1D = GetBTagWeightDict(
            args.var, all_hists_dict, categories, boosted_categories, boosted_variables
        )
    else:
        all_hists_dict_1D = all_hists_dict
    """
    if len(data_processes) > 0:
        data_processes = data_processes[0]
    if not data_processes:
        data_processes = None
    if (
        analysis_import == "Analysis.hh_bbtautau"
        and estimateQCD
        and data_processes != None
        and data_processes in all_hists_dict.keys()
    ):
        from Analysis.QCD_estimation import AddQCDInHistDict

        fixNegativeContributions = False
        # Snapshot the sample list before the loop: AddQCDInHistDict adds a "QCD" entry
        # to all_hists_dict, which must not appear in all_samples_list for later sources.
        dataset_types = list(all_hists_dict.keys())
        for unc_source in unc_sources:
            error_on_qcdnorm, error_on_qcdnorm_varied = AddQCDInHistDict(
                args.var,
                all_hists_dict,
                channels,
                all_categories,
                unc_source,
                dataset_types,
                scales,
                data_process_name=data_processes,
                wantNegativeContributions=False,
            )

    all_unc_dict = unc_cfg_dict["norm"].copy()
    all_unc_dict.update(unc_cfg_dict["shape"])

    # Signal Trees/<hist_name>: concatenate every contributing chunk's rows first
    # (uproot needs to create the file fresh), reusing the exact same hist_name
    # derivation the plain-histogram loop below uses, so both agree on the name for
    # the same (dataset_type, uncName, uncScale). Sum each one's embedded histogram
    # along the way and hold the result for the attach pass after that loop.
    signal_merged_hists = {}  # (key_dir, hist_name) -> TH1
    if signal_tree_sources:
        import uproot

        created_tree_paths = set()
        with uproot.recreate(args.outFile) as uproot_out:
            for dataset_type, bucket in signal_tree_sources.items():
                for (key_dir, unc_scale), sources in bucket.items():
                    uncName, uncScale = unc_scale
                    hist_name = resolve_merged_hist_name(
                        dataset_type, uncName, uncScale, unc_sources, all_unc_dict
                    )
                    if hist_name is None:
                        continue
                    out_tree_path = "/".join(key_dir) + "/Trees/" + hist_name
                    for file_path, internal_path in sources:
                        with uproot.open(f"{file_path}:{internal_path}") as src_tree:
                            for arrays in src_tree.iterate():
                                if out_tree_path in created_tree_paths:
                                    uproot_out[out_tree_path].extend(arrays)
                                else:
                                    uproot_out[out_tree_path] = arrays
                                    created_tree_paths.add(out_tree_path)
                    hist = all_signal_hist_dict.get(dataset_type, {}).get(
                        (key_dir, unc_scale)
                    )
                    if hist is not None:
                        signal_merged_hists[(key_dir, hist_name)] = hist

    # 209 = LZMA level 9, the standard compression for merged histogram files
    outFile = ROOT.TFile(
        args.outFile, "UPDATE" if signal_tree_sources else "RECREATE", "", 209
    )
    for dataset_type in all_hists_dict.keys():
        for key in all_hists_dict[dataset_type].keys():
            key_dir, (uncName, uncScale) = key
            # here there can be some custom requirements - e.g. regions / categories to not merge, datasets to ignore
            hist = all_hists_dict[dataset_type][key]
            hist_name = resolve_merged_hist_name(
                dataset_type, uncName, uncScale, unc_sources, all_unc_dict
            )
            if hist_name is None:
                continue

            dir_name = "/".join(key_dir)
            dir_ptr = Utilities.mkdir(outFile, dir_name)
            hist.SetTitle(hist_name)
            hist.SetName(hist_name)
            dir_ptr.WriteTObject(hist, hist_name, "Overwrite")
    outFile.Close()

    if signal_merged_hists:
        outFile = ROOT.TFile(args.outFile, "UPDATE", "", 209)
        for (key_dir, hist_name), hist in signal_merged_hists.items():
            tree_path = "/".join(key_dir) + "/Trees/" + hist_name
            tree = outFile.Get(tree_path)
            wrap_hist_in_tree(hist, hist_name, base_tree=tree)
            tree.Write("", ROOT.TObject.kOverwrite)
        outFile.Close()

    executionTime = time.time() - startTime

    print("Execution time in seconds: " + str(executionTime))
