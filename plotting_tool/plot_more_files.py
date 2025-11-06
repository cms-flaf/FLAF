import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import ROOT
import matplotlib.ticker as ticker
import yaml
import re
import matplotlib.colors as mcolors
import argparse
import os
import sys
import importlib


from drawer_functions import *
from HelpersForHistograms import *

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a specific contribution for one or different regions."
    )
    parser.add_argument(
        "--outFile", required=True, help="Output file name for the plot (e.g., my_plot)"
    )
    parser.add_argument(
        "--inFiles", nargs="+", required=True, help="Input ROOT file(s) path."
    )
    parser.add_argument(
        "--vars",
        required=True,
        type=str,
        help="Comma-separated variables to plot (e.g., m_mumu,p_T_mu1)",
    )
    parser.add_argument(
        "--channel",
        required=False,
        type=str,
        default="muMu",
        help="Analysis channel (common to all regions)",
    )
    parser.add_argument(
        "--period", required=False, type=str, default="Run3_2022", help="Run era"
    )
    parser.add_argument(
        "--mass_region",
        required=False,
        type=str,
        default="Z_sideband",
        help="muMu mass region",
    )
    parser.add_argument(
        "--category",
        required=False,
        type=str,
        default="ggH",
        help="Analysis category (base for qcdregions, or compared if category type)",
    )
    parser.add_argument(
        "--sub_region",
        required=False,
        type=str,
        help="Specific sub-region to plot within the category",
    )
    parser.add_argument(
        "--wantLogY", action="store_true", help="Apply log scale to y-axis"
    )
    parser.add_argument("--wantData", action="store_true", help="Include data")
    parser.add_argument("--wantSignal", action="store_true", help="Include signal")
    parser.add_argument(
        "--contribution",
        required=False,
        type=str,
        help="Specific contribution to plot (e.g., 'DY,TT').",
        default="all",
    )
    parser.add_argument(
        "--rebin",
        action="store_true",
        help="Enable rebinning based on histogram configuration",
    )
    parser.add_argument(
        "--unstacked", action="store_true", help="Plot contributions unstacked."
    )
    parser.add_argument(
        "--compare_vars",
        action="store_true",
        help="Compare different variables in one plot.",
    )

    args = parser.parse_args()

    setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], args.period, None)

    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")

    global_cfg_dict = setup.global_params
    sample_cfg_dict = setup.samples
    hist_cfg_dict = setup.hists
    bckg_cfg_dict = setup.bckg_config
    sig_cfg_dict = setup.signal_config
    unc_cfg_dict = setup.weights_config

    page_cfg = os.path.join(
        os.environ["ANALYSIS_PATH"], f"config", f"plot", f"cms_stacked.yaml"
    )
    with open(page_cfg, "r") as f:
        page_cfg_dict = yaml.safe_load(f)
    page_cfg_custom = os.path.join(
        os.environ["ANALYSIS_PATH"], f"config", f"plot", f"{args.period}.yaml"
    )
    with open(page_cfg_custom, "r") as f:
        page_cfg_custom_dict = yaml.safe_load(f)
    inputs_cfg = os.path.join(
        os.environ["ANALYSIS_PATH"], "config", "plot", "inputs.yaml"
    )
    with open(inputs_cfg, "r") as f:
        raw_inputs_cfg_dict = yaml.safe_load(f)

    all_samples_types = {}
    if args.wantData:
        all_samples_types = {"data": {"type": "data", "plot": "data"}}

    for sample_name in bckg_cfg_dict.keys():
        if "sampleType" not in bckg_cfg_dict[sample_name].keys():
            continue
        bckg_sample_type = bckg_cfg_dict[sample_name]["sampleType"]
        bckg_sample_name = (
            bckg_sample_type
            if bckg_sample_type in global_cfg_dict["sample_types_to_merge"]
            else sample_name
        )
        if bckg_sample_name in all_samples_types.keys():
            continue
        all_samples_types[bckg_sample_name] = {"type": bckg_sample_type}

        for sample_for_plot_dict in raw_inputs_cfg_dict:
            plot_types = sample_for_plot_dict["types"]
            if bckg_sample_type in plot_types:
                all_samples_types[bckg_sample_name]["plot"] = sample_for_plot_dict[
                    "name"
                ]
        if "plot" not in all_samples_types[bckg_sample_name].keys():
            all_samples_types[bckg_sample_name]["plot"] = "Other"

    if args.wantSignal:
        for sig_sample_name in sig_cfg_dict.keys():
            if "sampleType" not in sig_cfg_dict[sig_sample_name].keys():
                continue
            sig_sample_type = sig_cfg_dict[sig_sample_name]["sampleType"]
            if sig_sample_type not in global_cfg_dict["signal_types"]:
                continue
            for sample_for_plot_dict in raw_inputs_cfg_dict:
                if sample_for_plot_dict["name"] == sig_sample_name:
                    all_samples_types[sig_sample_type] = {
                        "type": sig_sample_type,
                        "plot": sig_sample_type,
                    }

    contributions_to_plot = []
    if args.contribution == "all":
        for sample_for_plot_dict in raw_inputs_cfg_dict:
            plot_name = sample_for_plot_dict.get("name")
            plot_types = sample_for_plot_dict.get("types", [])
            if (
                "data" in plot_types
                and args.wantData
                and "data" not in contributions_to_plot
            ):
                contributions_to_plot.append("data")
            for sample_name, sample_content in all_samples_types.items():
                if (
                    sample_content["plot"] == plot_name
                    and sample_name not in contributions_to_plot
                ):
                    contributions_to_plot.append(sample_name)
        contributions_to_plot.reverse()
    else:
        contributions_to_plot = args.contribution.split(",")

    if args.wantData and "data" not in contributions_to_plot:
        contributions_to_plot.append("data")

    print(f"Contributions = {contributions_to_plot}")

    variables_to_plot = args.vars.split(",")

    base_path = f"{args.channel}/{args.mass_region}/{args.category}"
    if args.sub_region:
        base_path += f"/{args.sub_region}"

    hists_to_plot = {}
    if args.compare_vars:
        hists_by_var = {}
        for var in variables_to_plot:
            hists_by_contrib = {}
            for file_path in args.inFiles:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}. Skipping.")
                    continue
                if var not in file_path.split("/"):
                    continue
                inFile_root = ROOT.TFile.Open(file_path, "READ")
                for sample_name in contributions_to_plot:
                    sample_content = all_samples_types.get(sample_name)
                    if not sample_content:
                        continue
                    sample_plot_name = sample_content.get("plot")
                    hist_path = f"{base_path}/{sample_plot_name}"
                    hist = inFile_root.Get(hist_path)
                    if hist and not hist.IsZombie():
                        if sample_plot_name not in hists_by_contrib:
                            hists_by_contrib[sample_plot_name] = hist.Clone()
                            hists_by_contrib[sample_plot_name].SetDirectory(0)
                        else:
                            hists_by_contrib[sample_plot_name].Add(hist)
                    else:
                        print(f"Warning: Histogram not found at path: {hist_path}")
                inFile_root.Close()

            if hists_by_contrib:
                total_hist = None
                for contrib_hist in hists_by_contrib.values():
                    if total_hist is None:
                        total_hist = contrib_hist
                    else:
                        total_hist.Add(contrib_hist)
                hists_by_var[var] = total_hist

        if args.rebin and "x_rebin" in hist_cfg_dict.get(variables_to_plot[0], {}):
            hists_rebinned = {}
            for var, hist in hists_by_var.items():
                if hist:
                    bins_to_compute = findNewBins(
                        hist_cfg_dict,
                        var,
                        channel=args.channel,
                        category=args.category,
                        region=args.mass_region,
                    )
                    new_bins = getNewBins(bins_to_compute)
                    hists_rebinned[var] = RebinHisto(
                        hist, new_bins, var, wantOverflow=False, verbose=False
                    )
            hists_to_plot = hists_rebinned
        else:
            hists_to_plot = hists_by_var

    else:
        # stacked/unstacked
        for var in variables_to_plot:
            hists_to_plot[var] = {}
            for file_path in args.inFiles:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}. Skipping.")
                    continue

                inFile_root = ROOT.TFile.Open(file_path, "READ")
                for sample_name in contributions_to_plot:
                    sample_content = all_samples_types.get(sample_name)
                    if not sample_content:
                        continue
                    sample_plot_name = sample_content.get("plot")
                    hist_path = f"{base_path}/{sample_plot_name}"
                    hist = inFile_root.Get(hist_path)
                    if hist and not hist.IsZombie():
                        if sample_plot_name not in hists_to_plot[var]:
                            hists_to_plot[var][sample_plot_name] = hist.Clone()
                            hists_to_plot[var][sample_plot_name].SetDirectory(0)
                        else:
                            hists_to_plot[var][sample_plot_name].Add(hist)
                    else:
                        print(f"Warning: Histogram not found at path: {hist_path}")
                inFile_root.Close()

        hists_rebinned = {}
        for var, hist_dict in hists_to_plot.items():
            hists_rebinned[var] = {}
            rebin_condition = (
                args.rebin and "x_rebin" in hist_cfg_dict.get(var, {}).keys()
            )
            new_bins = None
            if rebin_condition:
                bins_to_compute = findNewBins(
                    hist_cfg_dict, var, args.channel, args.category
                )
                new_bins = getNewBins(bins_to_compute)

            for hist_key, hist in hist_dict.items():
                if rebin_condition:
                    new_hist = RebinHisto(
                        hist, new_bins, hist_key, wantOverflow=False, verbose=False
                    )
                    hists_rebinned[var][hist_key] = new_hist
                else:
                    hists_rebinned[var][hist_key] = hist
        hists_to_plot = hists_rebinned

    if args.compare_vars:
        if hists_to_plot:
            print(
                f"comparing the following variables: {variables_to_plot} found in {base_path}"
            )
            for var in variables_to_plot:
                plot_histogram_from_config(
                    variable=var,
                    histograms_dict=hists_to_plot,
                    inputs_cfg=raw_inputs_cfg_dict,
                    axes_cfg_dict=hist_cfg_dict,
                    page_cfg_dict=page_cfg_dict,
                    page_cfg_custom_dict=page_cfg_custom_dict,
                    filename_base=f"{args.outFile}_var_comparison",
                    period=args.period,
                    stacked=False,
                    compare_mode=False,
                    compare_vars_mode=True,
                    wantLogX=False,
                    wantLogY=args.wantLogY,
                    wantData=False,
                    wantSignal=False,
                    wantRatio=False,
                    category=args.category,
                    channel=args.channel,
                    group_minor_contributions=False,
                )
        else:
            print(f"{variables_to_plot} not found in {base_path}.")
    else:
        for var in variables_to_plot:
            main_region_data = hists_to_plot.get(var)
            if main_region_data:
                print(f"plotting {var} found in {base_path}")
                plot_histogram_from_config(
                    variable=var,
                    histograms_dict=main_region_data,
                    inputs_cfg=raw_inputs_cfg_dict,
                    axes_cfg_dict=hist_cfg_dict,
                    page_cfg_dict=page_cfg_dict,
                    page_cfg_custom_dict=page_cfg_custom_dict,
                    filename_base=f"{args.outFile}_{var}_stacked",
                    period=args.period,
                    stacked=not args.unstacked,
                    compare_mode=False,
                    compare_vars_mode=False,
                    wantLogY=args.wantLogY,
                    wantData=args.wantData,
                    wantSignal=args.wantSignal,
                    category=args.category,
                    channel=args.channel,
                )
            else:
                print(f"{var} not found in {base_path}.")
