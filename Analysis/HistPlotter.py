import ROOT
import sys
import os
import importlib
import yaml

from FLAF.Common.HistHelper import *
from FLAF.Common.Setup import Setup


def get_histograms_from_dir(
    directory, hist_name, sample_plot_name, hist_dict, key_to_save_list
):
    keys = [k.GetName() for k in directory.GetListOfKeys()]

    if hist_name in keys:
        obj = directory.Get(hist_name)
        if obj.InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)

            path = directory.GetPath().split(":")[-1].strip("/")
            key_to_save = "/".join(key_to_save_list)
            if path == key_to_save:
                hist_dict.setdefault(sample_plot_name, obj).Add(obj)

    for key in keys:
        sub_dir = directory.Get(key)
        if sub_dir.InheritsFrom(ROOT.TDirectory.Class()):
            get_histograms_from_dir(
                sub_dir, hist_name, sample_plot_name, hist_dict, key_to_save_list
            )


def GetHistName(sample_name, sample_type, uncName, unc_scale, global_cfg_dict):
    sample_hist_name = (
        sample_type
        if sample_type in global_cfg_dict["sample_types_to_merge"]
        or global_cfg_dict["signal_types"]
        else sample_name
    )

    only_central = sample_name == "data" or uncName == "Central"
    scales = ["Central"] if only_central else global_cfg_dict["scales"]

    return (
        f"{sample_hist_name}_{uncName}{scales[0]}"
        if not only_central
        else sample_hist_name
    )


def findNewBins(
    hist_cfg_dict, var, channel, category
):  # to be fixed for regions and subregion inclusion eventually
    cfg = hist_cfg_dict[var]
    if "x_rebin" not in cfg:
        return cfg["x_bins"]

    rebin_cfg = cfg["x_rebin"]

    if isinstance(rebin_cfg, list):
        return rebin_cfg

    # Nested dict lookups with fallbacks
    for first, second in [(channel, category), (category, channel)]:
        if first in rebin_cfg:
            if isinstance(rebin_cfg[first], list):
                return rebin_cfg[first]
            if isinstance(rebin_cfg[first], dict) and second in rebin_cfg[first]:
                return rebin_cfg[first][second]

    return rebin_cfg.get("other", cfg["x_bins"])


def filter_inputs(inputs_cfg_dict, args):
    return [
        d
        for d in inputs_cfg_dict
        if not (
            (d.get("type") == "signal" and not args.wantSignals)
            or (d.get("name") == "data" and not args.wantData)
            or (d.get("name") == "QCD" and not args.wantQCD)
        )
    ]


def build_all_samples_types(
    global_cfg_dict, bckg_cfg_dict, sig_cfg_dict, inputs_cfg_dict, wantQCD
):
    all_samples_types = {"data": {"type": "data", "plot": "data"}}
    if wantQCD:
        all_samples_types["QCD"] = {"type": "QCD", "plot": "QCD"}

    # Add backgrounds
    for sample, cfg in bckg_cfg_dict.items():
        if "sampleType" not in cfg:
            continue
        s_type = cfg["sampleType"]
        s_name = (
            s_type if s_type in global_cfg_dict["sample_types_to_merge"] else sample
        )
        if s_name in all_samples_types:
            continue

        plot_name = next(
            (d["name"] for d in inputs_cfg_dict if s_type in d.get("types", [])),
            "Other",
        )
        all_samples_types[s_name] = {"type": s_type, "plot": plot_name}

    # Add signals
    for sample, cfg in sig_cfg_dict.items():
        if cfg.get("sampleType") in global_cfg_dict["signal_types"]:
            if any(d["name"] == sample for d in inputs_cfg_dict):
                all_samples_types[sample] = {"type": cfg["sampleType"], "plot": sample}

    return all_samples_types


if __name__ == "__main__":
    import argparse
    import FLAF.PlotKit.Plotter as Plotter

    parser = argparse.ArgumentParser()
    parser.add_argument("--outFile", required=True)
    parser.add_argument("--inFile", required=True)
    parser.add_argument("--var", default="tau1_pt")
    parser.add_argument("--channel", default="tauTau")
    parser.add_argument("--region", default="OS_Iso")
    parser.add_argument("--category", default="inclusive")
    parser.add_argument("--subregion", default="inclusive")
    parser.add_argument("--wantData", action="store_true")
    parser.add_argument("--wantSignals", action="store_true")
    parser.add_argument("--wantQCD", action="store_true")
    parser.add_argument("--rebin", action="store_true")
    parser.add_argument("--wantOverflow", type=bool, default=False)
    parser.add_argument("--wantLogScale", default="")
    parser.add_argument("--uncSource", default="Central")
    parser.add_argument("--period", default="Run2_2018")
    args = parser.parse_args()

    setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], args.period, None)
    analysis = importlib.import_module(setup.global_params["analysis_import"])

    global_cfg_dict = setup.global_params
    hist_cfg_dict = setup.hists
    bckg_cfg_dict = setup.bckg_config
    sig_cfg_dict = setup.signal_config

    # Load plotting configs
    def load_yaml(path):
        with open(path) as f:
            return yaml.safe_load(f)

    page_cfg = os.path.join(
        os.environ["ANALYSIS_PATH"], "config", "plot", "cms_stacked.yaml"
    )
    page_cfg_custom = os.path.join(
        os.environ["ANALYSIS_PATH"], "config", "plot", f"{args.period}.yaml"
    )
    inputs_cfg = os.path.join(
        os.environ["ANALYSIS_PATH"], "config", "plot", "inputs.yaml"
    )

    page_cfg_dict = load_yaml(page_cfg)
    page_cfg_custom_dict = load_yaml(page_cfg_custom)
    inputs_cfg_dict = filter_inputs(load_yaml(inputs_cfg), args)

    # Build samples dictionary
    all_samples_types = build_all_samples_types(
        global_cfg_dict, bckg_cfg_dict, sig_cfg_dict, inputs_cfg_dict, args.wantQCD
    )

    # Plotter setup
    plotter = Plotter.Plotter(page_cfg, page_cfg_custom, hist_cfg_dict, inputs_cfg_dict)

    # Custom CMS text
    cat_txt = (
        args.category.replace("_masswindow", "")
        .replace("_cat2", "")
        .replace("_cat3", "")
    )
    custom1 = {
        "cat_text": cat_txt,
        "ch_text": page_cfg_custom_dict["channel_text"][args.channel],
        "datasim_text": "CMS data" if args.wantData else "CMS simulation",
        "scope_text": "",
    }

    # Open ROOT input file
    inFile_root = ROOT.TFile.Open(args.inFile, "READ")

    # Apply log scale options
    if "y" in args.wantLogScale:
        hist_cfg_dict[args.var]["use_log_y"] = True
        hist_cfg_dict[args.var]["max_y_sf"] = 2000.2
    if "x" in args.wantLogScale:
        hist_cfg_dict[args.var]["use_log_x"] = True

    # Determine binning
    bins_to_compute = (
        findNewBins(hist_cfg_dict, args.var, args.channel, args.category)
        if args.rebin and "x_rebin" in hist_cfg_dict[args.var]
        else hist_cfg_dict[args.var]["x_bins"]
    )
    new_bins = getNewBins(bins_to_compute)

    # Collect histograms
    hists_unbinned = {}
    for sample, cfg in all_samples_types.items():
        if args.uncSource != "Central":
            continue  # TODO: implement systematics
        sample_histname = GetHistName(
            sample, cfg["type"], "Central", "Central", global_cfg_dict
        )
        values = [args.channel, args.region, args.category, args.subregion]
        get_histograms_from_dir(
            inFile_root, sample_histname, cfg["plot"], hists_unbinned, values
        )

    # Rebin if required
    hists_binned = {
        name: (
            RebinHisto(h, new_bins, name, wantOverflow=args.wantOverflow)
            if args.rebin
            else h
        )
        for name, h in hists_unbinned.items()
    }

    # Make the plot
    plotter.plot(
        args.var, hists_binned, args.outFile, want_data=args.wantData, custom=custom1
    )

    inFile_root.Close()
    print("Saved:", args.outFile)
