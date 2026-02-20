import ROOT
import sys
import os
import array

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Setup as Setup
from FLAF.Common.HistHelper import *


def GetHistName(dataset_name, dataset_type, uncName, unc_scale, global_cfg_dict):
    hist_names = []
    dataset_namehist = dataset_name
    onlyCentral = dataset_name == "data" or uncName == "Central"
    scales = ["Central"] if onlyCentral else global_cfg_dict["scales"]
    for scale in scales:
        histKey = (dataset_namehist, uncName, scale)
        histName = dataset_namehist
        if not onlyCentral:
            histName = f"{dataset_namehist}_{uncName}{scale}"
    return histName


def findNewBins(hist_cfg_dict, var, **keys):
    cfg = hist_cfg_dict.get(var, {})

    if "2d" in cfg:
        return cfg["2d"]

    if "x_rebin" not in cfg:
        if "x_bins" not in cfg:
            raise RuntimeError(f'bins definition not found for "{var}"')
        return cfg["x_bins"]

    x_rebin = cfg["x_rebin"]

    if isinstance(x_rebin, list):
        return x_rebin

    def recursive_search(d, remaining_keys):
        if isinstance(d, list):
            return d
        if not remaining_keys and isinstance(d, dict) and "other" in d:
            return d["other"]
        if not isinstance(d, dict):
            return None

        for k_name, k_value in remaining_keys.items():
            if k_value in d:
                found = recursive_search(
                    d[k_value],
                    {kk: vv for kk, vv in remaining_keys.items() if kk != k_name},
                )
                if found is not None:
                    return found

        if "other" in d:
            return d["other"]
        raise RuntimeError(f'Unable to find correct rebin for "{var}"')

    result = recursive_search(x_rebin, {k: v for k, v in keys.items() if v is not None})

    if result is None:
        raise RuntimeError(f'Unable to find correct rebin for "{var}"')
    return result


if __name__ == "__main__":
    import argparse
    import FLAF.PlotKit.Plotter as Plotter
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--all_outFiles", required=True)
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--var", required=False, type=str, default="tau1_pt")
    parser.add_argument("--globalConfig", required=True, type=str)
    parser.add_argument(
        "--all_keys", required=False, type=str, default="tauTau:inclusive:OS_Iso"
    )
    parser.add_argument("--wantData", required=False, action="store_true")
    parser.add_argument("--wantSignals", required=False, action="store_true")
    parser.add_argument("--wantQCD", required=False, type=bool, default=False)
    parser.add_argument("--wantOverflow", required=False, type=bool, default=False)
    parser.add_argument("--wantLogScale", required=False, type=str, default="")
    parser.add_argument("--uncSource", required=False, type=str, default="Central")
    parser.add_argument("--year", required=False, type=str, default="Run2_2018")
    parser.add_argument("--rebin", required=False, type=bool, default=False)
    parser.add_argument("--analysis", required=False, type=str, default="")
    parser.add_argument("--ana_path", required=True, type=str)
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--LAWrunVersion", required=True, type=str)

    args = parser.parse_args()

    page_cfg = os.path.join(
        os.environ["ANALYSIS_PATH"], "config", "plot/cms_stacked.yaml"
    )
    page_cfg_custom = os.path.join(
        os.environ["ANALYSIS_PATH"], f"config", f"plot/{args.year}.yaml"
    )  # to be fixed!!
    hist_cfg = os.path.join(
        os.environ["ANALYSIS_PATH"], "config", "plot/histograms.yaml"
    )

    setup = Setup.Setup(args.ana_path, args.period, args.LAWrunVersion)

    #### config opening ####
    with open(hist_cfg, "r") as f:
        hist_cfg_dict = yaml.safe_load(f)
    with open(page_cfg, "r") as f:
        page_cfg_dict = yaml.safe_load(f)
    with open(page_cfg_custom, "r") as f:
        page_cfg_custom_dict = yaml.safe_load(f)
    inputs_cfg = os.path.join(
        os.environ["ANALYSIS_PATH"], "config", "plot", "inputs.yaml"
    )

    with open(args.globalConfig, "r") as f:
        global_cfg_dict = yaml.safe_load(f)

    keys = args.all_keys.split(",")
    outFiles = args.all_outFiles.split(",")
    print(f"Running HistPlot for var {args.var} making {len(outFiles)} plots")
    for key, outFile in zip(keys, outFiles):
        print(f"Plotting key {key}")
        channel, category, custom_region = key.split(":")

        all_histlist = {}

        all_datasets_dict = {}
        data_processes = setup.phys_model.processes(process_type="data")
        if len(data_processes) > 0:
            data_process = data_processes[0]
            all_datasets_dict[data_process] = {
                "process_name": data_process,
                "process_group": "data",
                "plot_name": "data",
                "plot_color": "kBlack",
            }

        if args.wantQCD:
            all_datasets_dict["QCD"] = {
                "process_name": "QCD",
                "process_group": "backgrounds",
                "plot_name": "QCD",
                "plot_color": "kGray+2",
            }

        for dataset_name in setup.datasets.keys():
            base_process_name = setup.datasets[dataset_name]["process_name"]
            process_group = setup.datasets[dataset_name]["process_group"]
            if process_group == "data":
                continue

            process_name = setup.base_processes[base_process_name]["parent_process"]
            process = setup.parent_processes[process_name]
            all_datasets_key = process_name

            if process.get("to_plot", True):
                if process_group == "signals":
                    if "channels" in process.keys():
                        if channel not in process["channels"]:
                            continue
                all_datasets_dict[all_datasets_key] = {}
                all_datasets_dict[all_datasets_key]["process_name"] = process_name
                all_datasets_dict[all_datasets_key]["process_group"] = process_group
                all_datasets_dict[all_datasets_key]["plot_name"] = process.get(
                    "name", process_name
                )
                all_datasets_dict[all_datasets_key]["plot_color"] = process["color"]

        plotter = Plotter.Plotter(
            page_cfg=page_cfg,
            page_cfg_custom=page_cfg_custom,
            hist_cfg=hist_cfg_dict,
        )
        cat_txt = category.replace("_masswindow", "")
        cat_txt = cat_txt.replace("_cat2", "")
        cat_txt = cat_txt.replace("_cat3", "")
        custom_region_text = (
            ""
            if custom_region not in page_cfg_custom_dict["customregion_text"].keys()
            else page_cfg_custom_dict["customregion_text"][custom_region]
        )
        custom1 = {
            "cat_text": cat_txt,
            "ch_text": page_cfg_custom_dict["channel_text"][channel],
            "customreg_text": custom_region_text,
            "datasim_text": "CMS " + page_cfg_dict["scope_text"]["text"],
            "scope_text": "",
        }
        var_entry = findBinEntry(hist_cfg_dict, args.var)

        blind_check = hist_cfg_dict[var_entry].get("blind", False)
        args.wantData = args.wantData and (not blind_check)
        if args.wantData == False:
            custom1["datasim_text"] = "CMS simulation"
        inFile_root = ROOT.TFile.Open(args.inFile, "READ")
        dir_0 = inFile_root.Get(channel)
        keys_0 = [str(k) for k in dir_0.GetListOfKeys()]
        dir_0p1 = dir_0.Get(custom_region)
        keys_0p1 = [str(k) for k in dir_0p1.GetListOfKeys()]
        dir_1 = dir_0p1.Get(category)
        keys_1 = [str(k) for k in dir_1.GetListOfKeys()]

        hists_to_plot_unbinned = {}
        if args.wantLogScale == "y":
            hist_cfg_dict[var_entry]["use_log_y"] = True
            hist_cfg_dict[var_entry]["max_y_sf"] = 2000.2
        if args.wantLogScale == "x":
            hist_cfg_dict[var_entry]["use_log_x"] = True
        if args.wantLogScale == "xy":
            hist_cfg_dict[var_entry]["use_log_y"] = True
            hist_cfg_dict[var_entry]["max_y_sf"] = 2000.2
            hist_cfg_dict[var_entry]["use_log_x"] = True

        rebin_condition = args.rebin and "x_rebin" in hist_cfg_dict[var_entry].keys()
        bins_to_compute = (
            hist_cfg_dict[var_entry]["x_bins"] if not rebin_condition else None
        )

        if rebin_condition:
            bins_to_compute = findNewBins(
                hist_cfg_dict,
                var_entry,
                channel=channel,
                category=category,
                region=custom_region,
            )
        new_bins = GetBinVec(bins_to_compute)

        for dataset_name, dataset_content in all_datasets_dict.items():
            dataset_process_name = dataset_content["process_name"]
            dataset_process_group = dataset_content["process_group"]
            dataset_plot_name = dataset_content["plot_name"]
            dataset_plot_color = dataset_content["plot_color"]

            if dataset_process_group == "data" and not args.wantData:
                continue

            if args.uncSource != "Central":
                continue  # to be fixed

            dataset_histname = GetHistName(
                dataset_process_name,
                dataset_process_group,
                "Central",
                "Central",
                global_cfg_dict,
            )
            if dataset_histname not in dir_1.GetListOfKeys():
                print(f"ERRORE: {dataset_histname} non è nelle keys")
                continue
            obj = dir_1.Get(dataset_histname)
            if not obj.IsA().InheritsFrom(ROOT.TH1.Class()):
                print(f"ERRORE: {dataset_histname} non è un istogramma")
            obj.SetDirectory(0)

            if dataset_process_name in hists_to_plot_unbinned.keys():
                print(hists_to_plot_unbinned[dataset_process_name])

            if dataset_process_name not in hists_to_plot_unbinned.keys():
                hists_to_plot_unbinned[dataset_process_name] = (
                    obj,
                    dataset_plot_name,
                    dataset_plot_color,
                    dataset_process_group,
                )
            else:
                hists_to_plot_unbinned[dataset_process_name][0].Add(
                    hists_to_plot_unbinned[dataset_process_name][0], obj
                )
        hists_to_plot_binned = {}
        for hist_key, (
            hist_unbinned,
            plot_name,
            plot_color,
            dataset_process_group,
        ) in hists_to_plot_unbinned.items():
            old_hist = hist_unbinned
            new_hist = RebinHisto(
                old_hist,
                new_bins,
                hist_key,
                wantOverflow=args.wantOverflow,
                verbose=False,
            )
            hists_to_plot_binned[hist_key] = (
                (new_hist, plot_name, plot_color, dataset_process_group)
                if rebin_condition
                else (old_hist, plot_name, plot_color, dataset_process_group)
            )

        scale = global_cfg_dict.get("signal_plot_scale", 1.0)
        plotter.plot(
            var_entry,
            hists_to_plot_binned,
            outFile,
            want_data=args.wantData,
            custom=custom1,
            scale=scale,
        )
        inFile_root.Close()
        print(outFile)

    print(f"HistPlotter: all plots are produced.", file=sys.stderr)
