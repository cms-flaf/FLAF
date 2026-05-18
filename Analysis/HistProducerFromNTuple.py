import argparse
import os
import sys
import importlib
import ROOT
import time

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.HistHelper as HistHelper
import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
from FLAF.RunKit.run_tools import ps_call

ROOT.EnableThreadSafety()
ROOT.EnableImplicitMT()
num_threads = ROOT.GetThreadPoolSize()

print(f"ROOT is currently using {num_threads} threads.")


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


def SaveHist(key_tuple, outFile, hist_list, hist_name, unc, scale, verbose=0):
    model, unit_hist = hist_list[0]
    if verbose > 0:
        print(f"Saving hist for key: {key_tuple}, unc: {unc}, scale: {scale}")
    dir_name = "/".join(key_tuple)
    dir_ptr = Utilities.mkdir(outFile, dir_name)

    merged_hist = model.GetHistogram().Clone()
    # If we use the THnD then we have 'GetNbins' function, else use 'GetNcells'
    N_bins = (
        unit_hist.GetNbins()
        if hasattr(unit_hist, "GetNbins")
        else unit_hist.GetNcells()
    )

    # This can be a loop over many bins, several times. Can be improved to be ran in c++ instead
    for i in range(0, N_bins):
        bin_content = unit_hist.GetBinContent(i)
        bin_error = unit_hist.GetBinError(i)
        merged_hist.SetBinContent(i, bin_content)
        merged_hist.SetBinError(i, bin_error)

    nentries = unit_hist.GetEntries()
    if len(hist_list) > 1:
        for model, unit_hist in hist_list[1:]:
            hist = model.GetHistogram()
            for i in range(0, N_bins):
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
    import ast

    var_is_list = False
    var_list = None
    injected_entry = None
    var_entry = var
    if isinstance(var, list):
        var_is_list = True
    if isinstance(var, str) and var.startswith("[") and var.endswith("]"):
        var_list = ast.literal_eval(var)
        if isinstance(var_list, list) and len(var_list) >= 2:
            var_is_list = True

    if var_is_list:
        dims = len(var_list)
        first_var = var_list[0]
        var_entry_x = HistHelper.findBinEntry(hist_cfg_dict, first_var)

        x_bins = hist_cfg_dict[var_entry_x]["x_bins"]

        injected_entry = f"{'_'.join(var_list)}_2d"
        hist_cfg_dict[injected_entry] = {"var_list": var_list}

        for i, v in enumerate(var_list):
            var_entry_v = HistHelper.findBinEntry(hist_cfg_dict, v)
            bin_key = f"{v}_bins"
            hist_cfg_dict[injected_entry][bin_key] = hist_cfg_dict[var_entry_v][
                "x_bins"
            ]

        var_entry = injected_entry

    else:
        var_entry = HistHelper.findBinEntry(hist_cfg_dict, var)
        dims = (
            1
            if not hist_cfg_dict[var_entry].get("var_list", False)
            else len(hist_cfg_dict[var_entry]["var_list"])
        )
        var_list = hist_cfg_dict[var_entry].get("var_list", False)

    model, unit_bin_model = HistHelper.GetModel(
        hist_cfg_dict, var_entry, dims, return_unit_bin_model=True
    )

    if injected_entry:
        del hist_cfg_dict[injected_entry]

    var_bin_list = [f"{v}_bin" for v in var_list] if dims > 1 else [f"{var}_bin"]

    rdf_filtered = rdf.Filter(filter_to_apply)
    if dims >= 1 and dims <= 3:
        mkhist_fn = getattr(rdf_filtered, f"Histo{dims}D")
        unit_hist = mkhist_fn(unit_bin_model, *var_bin_list, weight_name)
    else:
        raise RuntimeError("Only 1D, 2D and 3D histograms are supported")
    # Materialize and clone histogram to free RDF memory
    h = unit_hist.GetValue().Clone()
    h.SetDirectory(0)
    return model, h


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
        tree_prefix = f"Events__{unc}__{scale}"
        rdf_shift = all_trees[tree_prefix]
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

    def save_fn():
        if hist_list:
            key_tuple = key
            if further_cut_name:
                key_tuple = key + (further_cut_name,)
            SaveHist(key_tuple, outFile, hist_list, var, unc, scale)

    return save_fn


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
    tmp_file = f"tmp_{var}.root"
    var_list = []
    import ast

    if isinstance(var, str) and var.startswith("[") and var.endswith("]"):
        var_list = ast.literal_eval(var)
    tmp_file = "tmp_"
    tmp_file += "_".join(v for v in var_list)
    tmp_file += ".root"

    tmp_file_root = ROOT.TFile(tmp_file, "RECREATE")
    for unc, scales in uncs_to_compute.items():
        is_shift_unc = unc in unc_cfg_dict["shape"].keys()
        for scale in scales:
            print(f"considering {unc}, {scale}")
            for key, filter_to_apply_base in key_filter_dict.items():
                if further_cuts:
                    for further_cut_name in further_cuts.keys():
                        filter_to_apply_final = (
                            f"{filter_to_apply_base} && {further_cut_name}"
                        )
                        key_tuple = key + (further_cut_name,)
                        # Determine base RDF and weight name
                        if is_shift_unc:
                            tree_prefix = f"Events__{unc}__{scale}"
                            rdf_base = all_trees[tree_prefix]
                            weight_name = "weight_Central"
                        else:
                            weight_name = (
                                f"weight_{unc}_{scale}"
                                if unc != "Central"
                                else "weight_Central"
                            )
                            rdf_base = all_trees[treeName]
                        model, unit_hist = GetUnitBinHist(
                            rdf_base,
                            var,
                            filter_to_apply_final,
                            weight_name,
                            unc,
                            scale,
                        )
                        hist_list = [(model, unit_hist)]
                        SaveHist(key_tuple, tmp_file_root, hist_list, var, unc, scale)
                else:
                    filter_to_apply_final = filter_to_apply_base
                    key_tuple = key
                    if is_shift_unc:
                        tree_prefix = f"Events__{unc}__{scale}"
                        rdf_base = all_trees[tree_prefix]
                        weight_name = "weight_Central"
                    else:
                        weight_name = (
                            f"weight_{unc}_{scale}"
                            if unc != "Central"
                            else "weight_Central"
                        )
                        rdf_base = all_trees[treeName]
                    model, unit_hist = GetUnitBinHist(
                        rdf_base, var, filter_to_apply_final, weight_name, unc, scale
                    )
                    hist_list = [(model, unit_hist)]
                    SaveHist(key_tuple, tmp_file_root, hist_list, var, unc, scale)
    tmp_file_root.Close()
    tmp_files.append(tmp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFiles", nargs="+", type=str)
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument(
        "--var",
        type=str,
        default=None,
        help="Variable to plot. For 2D histograms, use format '[var_x,var_y]'",
    )
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compute_rel_weights", type=bool, default=False)
    parser.add_argument("--furtherCut", type=str, default=None)
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    args = parser.parse_args()

    start = time.time()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"],
        args.period,
        args.LAWrunVersion,
        customisations=args.customisations,
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

        base_rdfs[key] = ROOT.RDataFrame(key, Utilities.ListToVector(all_infiles))
        ROOT.RDF.Experimental.AddProgressBar(base_rdfs[key])

    further_cuts = {}
    if args.furtherCut:
        further_cuts = {f: (f, f) for f in args.furtherCut.split(",")}
    if "further_cuts" in setup.global_params and setup.global_params["further_cuts"]:
        further_cuts.update(setup.global_params["further_cuts"])
        print(f"further cuts = {further_cuts}")
    key_filter_dict = analysis.createKeyFilterDict(
        setup.global_params, setup.global_params["era"]
    )

    variables = setup.global_params["variables"]
    vars_needed = set()
    for var in variables:
        if isinstance(var, (list, tuple)):
            for v in var:
                if "(" in v or ")" in v or "?" in v or "*" in v or "[" in v or "{" in v:
                    continue
                vars_needed.add(v)
        elif isinstance(var, dict):
            if "vars" in var:
                for v in var["vars"]:
                    if (
                        "(" in v
                        or ")" in v
                        or "?" in v
                        or "*" in v
                        or "[" in v
                        or "{" in v
                    ):
                        continue
                    vars_needed.add(v)
            elif "name" in var:
                v = var["name"]
                if (
                    "(" not in v
                    and ")" not in v
                    and "?" not in v
                    and "*" not in v
                    and "[" not in v
                    and "{" not in v
                ):
                    vars_needed.add(v)
        else:
            if (
                "(" not in var
                and ")" not in var
                and "?" not in var
                and "*" not in var
                and "[" not in var
                and "{" not in var
            ):
                vars_needed.add(var)
    for further_cut_name, (vars_for_cut, _) in further_cuts.items():
        for var_for_cut in vars_for_cut:
            if var_for_cut:
                vars_needed.add(var_for_cut)

    all_trees = {}
    for tree_name, rdf in base_rdfs.items():
        for further_cut_name, (vars_for_cut, cut_expr) in further_cuts.items():
            if further_cut_name not in rdf.GetColumnNames():
                rdf = rdf.Define(further_cut_name, cut_expr)
        all_trees[tree_name] = rdf

    uncs_to_compute = {}
    uncs_to_compute["Central"] = ["Central"]
    if args.dataset_name != "data":
        if args.compute_rel_weights:
            uncs_to_compute.update(
                {
                    key: setup.global_params["scales"]
                    for key in unc_cfg_dict["norm"].keys()
                }
            )
        if args.compute_unc_variations:
            uncs_to_compute.update(
                {
                    key: setup.global_params["scales"]
                    for key in unc_cfg_dict["shape"].keys()
                }
            )
            print(f"uncs to compute = {uncs_to_compute}")

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
        hadd_str = f"hadd -f209 -j -O {args.outFile} " + " ".join(tmp_files)
        ps_call([hadd_str], True)

    for f in tmp_files:
        print(f"processing file {f}")
        if os.path.exists(f):
            os.remove(f)
    time_elapsed = time.time() - start
    print(f"execution time = {time_elapsed} ")
