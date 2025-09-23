import argparse
import os
import sys
import importlib
import ROOT
import time

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *
import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
from FLAF.RunKit.run_tools import ps_call


def find_keys(inFiles_list):
    unique_keys = set()
    for infile in inFiles_list:
        rf = ROOT.TFile.Open(infile)
        if not rf or rf.IsZombie():
            raise RuntimeError(f"Unable to open {infile}")
        for key in rf.GetListOfKeys():
            unique_keys.add(key.GetName())
        rf.Close()
    return sorted(unique_keys)


def SaveHist(key_tuple, outFile, hist_list, hist_name, unc, scale):
    dir_name = "/".join(key_tuple)
    dir_ptr = Utilities.mkdir(outFile, dir_name)
    model, unit_hist = hist_list[0]
    merged_hist = model.GetHistogram().Clone()
    for i in range(0, unit_hist.GetNbinsX() + 2):
        bin_content = unit_hist.GetBinContent(i)
        bin_error = unit_hist.GetBinError(i)
        merged_hist.SetBinContent(i, bin_content)
        merged_hist.SetBinError(i, bin_error)

    nentries = unit_hist.GetEntries()
    if len(hist_list) > 1:
        for model, unit_hist in hist_list[1:]:
            hist = model.GetHistogram()
            for i in range(0, unit_hist.GetNbinsX() + 2):
                bin_content = unit_hist.GetBinContent(i)
                bin_error = unit_hist.GetBinError(i)
                hist.SetBinContent(i, bin_content)
                hist.SetBinError(i, bin_error)
                nentries += unit_hist.GetEntries()
            merged_hist.Add(hist)

    merged_hist.SetEntries(nentries)
    isCentral = unc == "Central"
    final_hist_name = hist_name if isCentral else f"{hist_name}_{unc}_{scale}"
    dir_ptr.WriteTObject(merged_hist, final_hist_name, "Overwrite")


def GetUnitBinHist(rdf, var, filter_to_apply, weight_name, unc, scale):
    model, unit_bin_model = GetModel(hist_cfg_dict, var, return_unit_bin_model=True)
    unit_hist = rdf.Filter(filter_to_apply).Histo1D(
        unit_bin_model, f"{var}_bin", weight_name
    )
    return model, unit_hist


def SaveSingleHistSet(
    all_trees,
    var,
    filter_expr,
    unc,
    scale,
    key,
    outFile,
    is_shift_unc,
    treeName,
    further_cut_name=None,
):
    hist_list = []
    if is_shift_unc:
        tree_prefix = f"Events_{unc}{scale}"
        shifts = ["noDiff", "Valid", "nonValid"]
        for shift in shifts:
            tree_name_full = f"{tree_prefix}_{shift}"
            if tree_name_full not in all_trees:
                continue
            rdf_shift = all_trees[tree_name_full]
            model, unit_hist = GetUnitBinHist(
                rdf_shift, var, filter_expr, "weight_Central", unc, scale
            )
            hist_list.append((model, unit_hist))
    else:
        weight_name = f"weight_{unc}_{scale}" if unc != "Central" else "weight_Central"
        rdf_central = all_trees[treeName]
        model, unit_hist = GetUnitBinHist(
            rdf_central, var, filter_expr, weight_name, unc, scale
        )
        hist_list.append((model, unit_hist))

    if hist_list:
        key_tuple = key
        if further_cut_name:
            key_tuple = key + (further_cut_name,)
        SaveHist(key_tuple, outFile, hist_list, var, unc, scale)


def SaveTmpFileUnc(
    tmp_files,
    uncs_to_compute,
    unc_cfg_dict,
    all_trees,
    var,
    key_filter_dict,
    further_cuts,
    treeName,
):
    for unc, scales in uncs_to_compute.items():
        tmp_file = f"tmp_{var}_{unc}.root"
        tmp_file_root = ROOT.TFile(tmp_file, "RECREATE")
        is_shift_unc = unc in unc_cfg_dict["shape"]

        for scale in scales:
            for key, filter_to_apply_base in key_filter_dict.items():
                filter_to_apply_final = filter_to_apply_base
                if further_cuts:
                    for further_cut_name in further_cuts.keys():
                        filter_to_apply_final = (
                            f"{filter_to_apply_base} && {further_cut_name}"
                        )
                        SaveSingleHistSet(
                            all_trees,
                            var,
                            filter_to_apply_final,
                            unc,
                            scale,
                            key,
                            tmp_file_root,
                            is_shift_unc,
                            treeName,
                            further_cut_name,
                        )
                else:
                    SaveSingleHistSet(
                        all_trees,
                        var,
                        filter_to_apply_final,
                        unc,
                        scale,
                        key,
                        tmp_file_root,
                        is_shift_unc,
                        treeName,
                    )
        tmp_file_root.Close()
        tmp_files.append(tmp_file)


def CreateFakeStructure(outFile, setup, var, key_filter_dict, further_cuts):
    hist_cfg_dict = setup.hists
    channels = setup.global_params["channels_to_consider"]

    for filter_key in key_filter_dict.keys():
        print(filter_key)
        for further_cut_name in [None] + list(further_cuts.keys()):
            model, unit_bin_model = GetModel(
                hist_cfg_dict, var, return_unit_bin_model=True
            )
            nbins = unit_bin_model.fNbinsX
            xmin = -0.5
            xmax = unit_bin_model.fNbinsX - 0.5
            empty_hist = ROOT.TH1F(var, var, nbins, xmin, xmax)
            empty_hist.Sumw2()
            key_tuple = filter_key
            if further_cut_name:
                key_tuple += (further_cut_name,)
            SaveHist(
                key_tuple, outFile, [(model, empty_hist, 0)], var, "Central", "Central"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFiles", nargs="+", type=str)
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--var", type=str, default=None)
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compute_rel_weights", type=bool, default=False)
    parser.add_argument("--furtherCut", type=str, default=None)
    args = parser.parse_args()

    start = time.time()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"], args.period, args.customisations
    )
    unc_cfg_dict = setup.weights_config
    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")

    treeName = setup.global_params["treeName"]
    all_infiles = [fileName for fileName in args.inputFiles]
    unique_keys = find_keys(all_infiles)
    inFiles = Utilities.ListToVector(all_infiles)
    base_rdfs = {}

    hist_cfg_dict = setup.hists

    channels = (
        args.channels.split(",")
        if args.channels
        else setup.global_params["channelSelection"]
    )
    setup.global_params["channels_to_consider"] = channels

    for key in unique_keys:
        if not key.startswith(treeName):
            continue
        valid_files = []
        has_entries = False
        for f in all_infiles:
            rf = ROOT.TFile.Open(f)
            if rf and rf.Get(key):
                tree = rf.Get(key)
                if tree and tree.GetEntries() > 0:
                    has_entries = True
                    valid_files.append(f)
            rf.Close()

        if valid_files and has_entries:
            base_rdfs[key] = ROOT.RDataFrame(key, Utilities.ListToVector(valid_files))
        else:
            print(f"[INFO] Tree {key} non trovato o senza entries: creo fake structure")
            outFile_root = ROOT.TFile(args.outFile, "UPDATE")
            key_filter_dict = analysis.createKeyFilterDict(
                setup.global_params, setup.global_params["era"]
            )
            further_cuts = {}
            if args.furtherCut:
                further_cuts = {f: (f, f) for f in args.furtherCut.split(",")}
            if (
                "further_cuts" in setup.global_params
                and setup.global_params["further_cuts"]
            ):
                further_cuts.update(setup.global_params["further_cuts"])
            CreateFakeStructure(
                outFile_root, setup, args.var, key_filter_dict, further_cuts
            )
            outFile_root.Close()
            continue

    further_cuts = {}
    if args.furtherCut:
        further_cuts = {f: (f, f) for f in args.furtherCut.split(",")}
    if "further_cuts" in setup.global_params and setup.global_params["further_cuts"]:
        further_cuts.update(setup.global_params["further_cuts"])
    print(further_cuts)
    key_filter_dict = analysis.createKeyFilterDict(
        setup.global_params, setup.global_params["era"]
    )

    variables = setup.global_params["variables"]
    vars_needed = set(variables)
    for further_cut_name, (var_for_cut, _) in further_cuts.items():
        if var_for_cut:
            vars_needed.add(var_for_cut)

    all_trees = {}
    for tree_name, rdf in base_rdfs.items():
        for var in vars_needed:
            if var not in rdf.GetColumnNames():
                print(f"attenzione, {var} not in column names")
        for further_cut_name, (var_for_cut, cut_expr) in further_cuts.items():
            if further_cut_name not in rdf.GetColumnNames():
                rdf = rdf.Define(further_cut_name, cut_expr)
        all_trees[tree_name] = rdf

    uncs_to_compute = {}
    if args.compute_rel_weights:
        uncs_to_compute.update(
            {key: setup.global_params["scales"] for key in unc_cfg_dict["norm"].keys()}
        )
    if args.compute_unc_variations:
        uncs_to_compute.update(
            {key: setup.global_params["scales"] for key in unc_cfg_dict["shape"]}
        )
    uncs_to_compute["Central"] = ["Central"]

    tmp_files = []
    if all_trees:
        SaveTmpFileUnc(
            tmp_files,
            uncs_to_compute,
            unc_cfg_dict,
            all_trees,
            args.var,
            key_filter_dict,
            further_cuts,
            treeName,
        )

    if tmp_files:
        hadd_str = f"hadd -f -j -O {args.outFile} " + " ".join(tmp_files)
        ps_call([hadd_str], True)

    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)
    time_elapsed = time.time() - start
    print(f"execution time = {time_elapsed} ")
