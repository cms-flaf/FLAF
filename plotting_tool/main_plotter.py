import ROOT
import sys
import os
import yaml
import importlib


from drawer_functions import *
from HelpersForHistograms import *

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a specific contribution for one or different regions.")
    parser.add_argument('--outFile', required=True, help="Output file name for the plot (e.g., my_plot)")
    parser.add_argument('--inFile', required=True, type=str, help="Input ROOT file containing histograms")
    parser.add_argument('--period', required=True, type=str, default='Run3_2022', help="Run era")
    parser.add_argument('--var', required=True, type=str, default='m_mumu', help="Variable to plot")
    parser.add_argument('--channel', required=False, type=str, default='muMu', help="Analysis channel")
    parser.add_argument('--region', required=False, type=str, default='OS_Iso', help="region (QCD, Mass..)")
    parser.add_argument('--category', required=False, type=str, default='inclusive', help="Analysis category (res1/2b, boosted, ggH/VBF,..)")
    parser.add_argument('--wantLogY', action='store_true', help="Apply log scale to y-axis")
    parser.add_argument('--wantData', action='store_true', help="Include data")
    parser.add_argument('--wantRatio', action='store_true', help="Include data")
    parser.add_argument('--wantSignal', action='store_true', help="Include data")
    parser.add_argument('--contribution', required=False, type=str, help="Specific contribution to plot (e.g., 'DY,TT').", default="all")
    parser.add_argument('--pre_path', required=False, type=str, help="specific path for histograms", default=None) # when paths are different
    parser.add_argument('--rebin', action='store_true', help="Enable rebinning based on histogram configuration")
    parser.add_argument('--compare_list', required=False, type=str, help="comma-separated subcategory to compare")
    parser.add_argument('--unstacked', action='store_true', help="Plot contributions unstacked.")



    args = parser.parse_args()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"], args.period, None
    )

    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")

    phys_model_cfg_dict = setup.phys_model
    global_cfg_dict = setup.global_params
    hist_cfg_dict = setup.hists
    unc_cfg_dict = setup.weights_config
    processes_cfg_dict = setup.processes

    page_cfg = os.path.join(os.environ["ANALYSIS_PATH"], f'config', f'plot',f"cms_stacked.yaml")
    with open(page_cfg, 'r') as f:
        page_cfg_dict = yaml.safe_load(f)
    page_cfg_custom = os.path.join(os.environ["ANALYSIS_PATH"], f'config', f'plot',f"{args.period}.yaml")
    with open(page_cfg_custom, 'r') as f:
        page_cfg_custom_dict = yaml.safe_load(f)

    signals = phys_model_cfg_dict['signals']
    backgrounds = phys_model_cfg_dict['backgrounds']
    all_contributions = backgrounds
    if args.wantSignal:
        all_contributions+=signals
    if args.wantData:
        all_contributions+=['data_obs']
    if args.contribution != 'all':
        all_contributions = args.contribution.split(",")

    hists_to_plot = {}

    rebin_condition = args.rebin and "x_rebin" in hist_cfg_dict[args.var].keys()
    new_bins = None
    if rebin_condition:
        bins_to_compute = findNewBins(hist_cfg_dict, args.var, channel=args.channel, category=args.category, region=args.region)
        new_bins = getNewBins(bins_to_compute)


    inFile_root = ROOT.TFile.Open(args.inFile, "READ")
    hists_to_plot = {}
    for sample_type in all_contributions:
        get_histograms_from_dir(inFile_root, sample_type, hists_to_plot)

    hists_to_plot_binned = {}
    for path, hist_dict in hists_to_plot.items():
        if path not in hists_to_plot_binned:
            hists_to_plot_binned[path] = {}

        for hist_key, hist in hist_dict.items():
            if rebin_condition:
                new_hist = RebinHisto(hist, new_bins, hist_key, wantOverflow=False, verbose=False)
                hists_to_plot_binned[path][hist_key] = new_hist
            else:
                hists_to_plot_binned[path][hist_key] = hist

    pre_path = f"{args.channel}/{args.region}/{args.category}" if args.pre_path is None else args.pre_path # patch for current bbtt shapes

    regions_to_compare = args.compare_list.split(",") if args.compare_list else []

    if regions_to_compare:
        comparison_paths = [f"{pre_path}/{reg}" for reg in regions_to_compare] # structure has the sub-regions to compare as final folders containing the histos: e.g. channel-region-category-subregion(s)
        if all(path in hists_to_plot_binned for path in comparison_paths):
            comparison_data = {path: hists_to_plot_binned[path] for path in comparison_paths}
            print(f"plotting regions: {', '.join(comparison_paths)} comparison for the following contributions: {', '.join(contributions_to_plot)}")
            plot_histogram_from_config(
                variable=args.var,
                histograms_dict=comparison_data,
                inputs_cfg=raw_inputs_cfg_dict,
                axes_cfg_dict=hist_cfg_dict,
                page_cfg_dict=page_cfg_dict,
                page_cfg_custom_dict=page_cfg_custom_dict,
                filename_base=f"{args.outFile}_regions_comparison",
                period=args.period,
                stacked=False,
                compare_mode=True,
                wantLogX=False,
                wantLogY=args.wantLogY,
                wantData=args.wantData,
                wantSignal=args.wantSignal,
                wantRatio = args.wantRatio,
                category=args.category,
                channel=args.channel
            )
        else:
            print("Regions not found.")

    elif pre_path in hists_to_plot_binned:
        main_region_data = {contrib: hists_to_plot_binned[pre_path][contrib] for contrib in all_contributions if contrib in hists_to_plot_binned[pre_path]}

        if len(all_contributions) >= 1:
            print(f"Plotting histograms found at this path: {pre_path}")
            plot_histogram_from_config(
                variable=args.var,
                histograms_dict=main_region_data,
                phys_model_dict=phys_model_cfg_dict,
                processes_dict=processes_cfg_dict,
                axes_cfg_dict=hist_cfg_dict,
                page_cfg_dict=page_cfg_dict,
                page_cfg_custom_dict=page_cfg_custom_dict,
                filename_base=f"{args.outFile}_stacked",
                period=args.period,
                stacked=True,
                wantLogX=False,
                wantLogY=args.wantLogY,
                wantData=args.wantData,
                wantSignal=args.wantSignal,
                wantRatio=args.wantRatio,
                category=args.category,
                channel=args.channel
            )
        else:
            print("No contributions found")
    else:
        print(f"{pre_path} not found")
    inFile_root.Close()