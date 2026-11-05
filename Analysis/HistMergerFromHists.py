import ROOT
import sys
import os
import time
import importlib

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
import FLAF.Common.Setup as Setup
import FLAF.Common.TheoryUncertainties as TheoryUncertainties
from FLAF.Common.HistHelper import *
from Corrections.CorrectionsCore import getScales


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
    target_keys = {var_input: ("Central", "Central")}
    if dataset_type != data_type:
        for unc_source in unc_sources:
            if unc_source == "Central":
                continue
            # Up/Down, or every member of a theory uncertainty (reduced after the sum)
            for scale in getScales(unc_source):
                target_keys[f"{var_input}_{unc_source}_{scale}"] = (unc_source, scale)
    hists = all_hist_dict.setdefault(dataset_type, {})
    for key_tuple, hist_map in items_dict.items():
        for var, var_hist in hist_map.items():
            unc_scale = target_keys.get(var)
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
    all_unc_dict = unc_cfg_dict["norm"].copy()
    all_unc_dict.update(unc_cfg_dict["shape"])
    # The members of a theory uncertainty (PDF, QCD scale) are known from the corrections
    # configuration, and it is merged only for the variables its entry lists.
    from Corrections.Corrections import Corrections

    Corrections.registerConfiguredSourceScales(global_cfg_dict)
    unc_sources = [
        s
        for s in unc_sources
        if s == "Central"
        or TheoryUncertainties.applies_to_variable(all_unc_dict[s], args.var)
    ]
    theory_uncs = {
        s: all_unc_dict[s]
        for s in unc_sources
        if s != "Central" and TheoryUncertainties.reduction(all_unc_dict[s])
    }
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
    chunk_has_trees = {}

    def process_of(dataset_name):
        base_process_name = dataset_cfg_dict[dataset_name]["process_name"]
        return setup.base_processes[base_process_name]["parent_process"]

    # The datasets of a process are read one after the other, so that its theory-uncertainty
    # members can be reduced as soon as its last dataset is summed: only one process holds its
    # members at a time (~0.4 GB per process for a 2D DNN_m*_vs_HME, 15 processes per variable).
    inputs = list(zip(args.dataset_names.split(","), args.inFiles))
    process_order = list(dict.fromkeys(process_of(name) for name, _ in inputs))
    inputs.sort(key=lambda item: process_order.index(process_of(item[0])))
    last_input_of = {process_of(name): i for i, (name, _) in enumerate(inputs)}

    def reduce_theory_uncertainties(process):
        for unc_name, unc_entry in theory_uncs.items():
            TheoryUncertainties.reduce_in_hist_dict(
                {process: all_hists_dict.get(process, {})},
                unc_name,
                unc_entry,
                getScales(unc_name),
                data_type=data_process_name,
            )

    for i, (dataset_name, inFile_path) in enumerate(inputs):
        dataset_type = process_of(dataset_name)
        if not os.path.exists(inFile_path):
            print(
                f"input file for dataset {dataset_name} (with path= {inFile_path}) does not exist, skipping"
            )
            if last_input_of[dataset_type] == i:
                reduce_theory_uncertainties(dataset_type)
            continue

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
            has_trees = bool(inFile.GetDirectory(SIGNAL_TREES_DIR))
            chunk_has_trees.setdefault(dataset_type, []).append(
                (dataset_name, inFile_path, has_trees)
            )
        del all_items
        # Every dataset of the process is summed now: reduce each theory uncertainty's members
        # to its Up/Down pair, so that everything below sees a plain Up/Down uncertainty.
        if last_input_of[dataset_type] == i:
            reduce_theory_uncertainties(dataset_type)
    signal_tree_files = signal_tree_inputs(chunk_has_trees)
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

    # 209 = LZMA level 9, the standard compression for merged histogram files
    outFile = ROOT.TFile(args.outFile, "RECREATE", "", 209)
    for dataset_type in all_hists_dict.keys():
        for key in all_hists_dict[dataset_type].keys():
            key_dir, (uncName, uncScale) = key
            # here there can be some custom requirements - e.g. regions / categories to not merge, datasets to ignore
            hist = all_hists_dict[dataset_type][key]
            hist_name = dataset_type
            if uncName not in unc_sources:
                continue
            if uncName != "Central":
                if dataset_type == "data":
                    continue
                if uncScale == "Central":
                    continue
                hist_name += "_" + TheoryUncertainties.format_name(
                    all_unc_dict[uncName], uncScale, dataset_type
                )
            else:
                if uncScale != "Central":
                    continue

            dir_name = "/".join(key_dir)
            dir_ptr = Utilities.mkdir(outFile, dir_name)
            hist.SetTitle(hist_name)
            hist.SetName(hist_name)
            dir_ptr.WriteTObject(hist, hist_name, "Overwrite")

    # The signal trees of the chunks, concatenated per process, one variation at a time;
    # a shape variation takes the name its histograms have.
    tree_variations = {"Central": "Central"}
    for uncName in unc_sources:
        if uncName in unc_cfg_dict["shape"].keys():
            for uncScale in scales:
                tree_variations[f"{uncName}_{uncScale}"] = all_unc_dict[uncName][
                    "name"
                ].format(uncScale)
    for dataset_type, paths in signal_tree_files.items():
        out_dir = Utilities.mkdir(outFile, f"{SIGNAL_TREES_DIR}/{dataset_type}")
        in_files = [ROOT.TFile.Open(path, "READ") for path in paths]
        for tree_name, variation in tree_variations.items():
            trees = [
                in_file.Get(f"{SIGNAL_TREES_DIR}/{tree_name}") for in_file in in_files
            ]
            if not all(trees):
                raise RuntimeError(
                    f"{SIGNAL_TREES_DIR}/{tree_name} missing in a file of {dataset_type}"
                )
            concat_trees(out_dir, variation, trees)
        for in_file in in_files:
            in_file.Close()
    outFile.Close()
    executionTime = time.time() - startTime

    print("Execution time in seconds: " + str(executionTime))
