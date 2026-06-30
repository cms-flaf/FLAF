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
from FLAF.Analysis.HistTupleProducer import DefineBinnedColumn


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
    model, unit_hist, rdf = hist_list[0]
    if verbose > 0:
        print(
            f"Saving hist for key: {key_tuple}, unc: {unc}, scale: {scale}. Number of RDF runs: {rdf.GetNRuns()}"
        )
    dir_name = "/".join(key_tuple)
    dir_ptr = Utilities.mkdir(outFile, dir_name)

    merged_hist = model.GetHistogram().Clone()
    # Detach from the current ROOT directory: the histogram is persisted explicitly via
    # WriteTObject below, so it must not also be auto-flushed into the output file's root
    # (which would leave one stray, unnamed histogram per call when writing directly).
    merged_hist.SetDirectory(0)
    N_bins = (
        unit_hist.GetNbins()
        if hasattr(unit_hist, "GetNbins")
        else unit_hist.GetNcells()
    )
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


def BookUnitHist(rdf_filtered, var, weight_name):
    """Book the unit-bin histogram for ``var`` on an ALREADY-filtered RDataFrame node.

    The selection filter is applied once by the caller and the resulting node is shared
    across all variables, so the (compound) channel/region/category cut is evaluated once
    per event instead of once per variable.
    """
    var_entry = HistHelper.findBinEntry(hist_cfg_dict, var)
    dims = (
        1
        if not hist_cfg_dict[var_entry].get("var_list", False)
        else len(hist_cfg_dict[var_entry]["var_list"])
    )
    if dims < 1 or dims > 3:
        raise RuntimeError("Only 1D, 2D and 3D histograms are supported")
    model, unit_bin_model = HistHelper.GetModel(
        hist_cfg_dict, var, dims, return_unit_bin_model=True
    )
    var_bin_list = (
        [f"{v}_bin" for v in hist_cfg_dict[var_entry]["var_list"]]
        if dims > 1
        else [f"{var}_bin"]
    )
    mkhist_fn = getattr(rdf_filtered, f"Histo{dims}D")
    unit_hist = mkhist_fn(unit_bin_model, *var_bin_list, weight_name)
    return model, unit_hist


def _make_save_fn(key_tuple, outFile, model, unit_hist, rdf, var, unc, scale):
    def save_fn():
        SaveHist(key_tuple, outFile, [(model, unit_hist, rdf)], var, unc, scale)

    return save_fn


def BuildAllHistActions(
    uncs_to_compute,
    unc_cfg_dict,
    all_trees,
    vars_to_process,
    key_filter_dict,
    further_cuts,
    treeName,
    var_tmp_files,
):
    """Register histogram actions for every variable, sharing one filtered RDataFrame node
    per (uncertainty, scale, selection-key, further-cut) across ALL variables.

    The dominant per-event cost is evaluating the compound channel/region/category filter,
    not filling the (precomputed unit-bin) histogram. Booking each variable on its own
    ``rdf.Filter(...)`` re-evaluated the identical selection once per variable; sharing the
    filtered node collapses ~Nvar redundant filter passes into a single one. Histograms are
    identical -- only the RDF graph is smaller and the single event loop does far less work.

    Returns a list of save callables; invoke them after booking so ROOT runs one event loop.
    """
    save_fns = []
    cut_names = list(further_cuts.keys()) if further_cuts else [None]
    for unc, scales in uncs_to_compute.items():
        is_shift_unc = unc in unc_cfg_dict["shape"].keys()
        for scale in scales:
            if is_shift_unc:
                rdf_base = all_trees[f"Events__{unc}__{scale}"]
                weight_name = "weight_Central"
            else:
                rdf_base = all_trees[treeName]
                weight_name = (
                    f"weight_{unc}_{scale}" if unc != "Central" else "weight_Central"
                )
            for key, filter_to_apply_base in key_filter_dict.items():
                for further_cut_name in cut_names:
                    filter_to_apply_final = (
                        f"{filter_to_apply_base} && {further_cut_name}"
                        if further_cut_name
                        else filter_to_apply_base
                    )
                    rdf_filtered = rdf_base.Filter(filter_to_apply_final)
                    key_tuple = key + (further_cut_name,) if further_cut_name else key
                    for var in vars_to_process:
                        model, unit_hist = BookUnitHist(rdf_filtered, var, weight_name)
                        _, tmp_root_file = var_tmp_files[var]
                        save_fns.append(
                            _make_save_fn(
                                key_tuple,
                                tmp_root_file,
                                model,
                                unit_hist,
                                rdf_filtered,
                                var,
                                unc,
                                scale,
                            )
                        )
    return save_fns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFiles", nargs="+", type=str)
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--outDir", required=True, type=str)
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--vars", required=True, type=str)
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compute_rel_weights", type=bool, default=False)
    parser.add_argument("--furtherCut", type=str, default=None)
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    parser.add_argument("--nMT", type=int, default=8)
    parser.add_argument("--user-custom", type=str, default=None)
    args = parser.parse_args()

    ROOT.EnableImplicitMT(args.nMT)

    start = time.time()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"],
        args.period,
        args.LAWrunVersion,
        customisations=args.customisations,
        user_custom_file=args.user_custom,
    )
    unc_cfg_dict = setup.weights_config
    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")

    treeName = setup.global_params["treeName"]
    all_infiles = [fileName for fileName in args.inputFiles]
    unique_keys = find_keys(all_infiles)

    hist_cfg_dict = setup.hists

    channels = (
        args.channels.split(",")
        if args.channels
        else setup.global_params["channelSelection"]
    )
    setup.global_params["channels_to_consider"] = channels

    base_rdfs = {}
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
    print(further_cuts)

    key_filter_dict = analysis.createKeyFilterDict(
        setup.global_params, setup.global_params["era"]
    )

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
    print(uncs_to_compute)

    vars_to_process = [v.strip() for v in args.vars.split(",") if v.strip()]
    os.makedirs(args.outDir, exist_ok=True)

    for var in vars_to_process:
        var_entry = HistHelper.findBinEntry(hist_cfg_dict, var)
        sub_vars = hist_cfg_dict[var_entry].get("var_list") or [var]
        for v in sub_vars:
            v_entry = HistHelper.findBinEntry(hist_cfg_dict, v)
            if not hist_cfg_dict[v_entry].get("x_bins"):
                continue
            binned_def_created = False
            col_name = f"{v}_bin"
            for tree_name in list(all_trees.keys()):
                if col_name not in list(all_trees[tree_name].GetColumnNames()):
                    if not binned_def_created:
                        DefineBinnedColumn(hist_cfg_dict, v)
                        binned_def_created = True
                    all_trees[tree_name] = all_trees[tree_name].Define(
                        col_name, f"get_{v}_bin({v})"
                    )

    if all_trees:
        # Open a tmp ROOT file per variable, then register all histogram actions sharing one
        # filtered RDataFrame node per selection across variables (see BuildAllHistActions).
        # Collecting every action before triggering lets ROOT execute them in a single
        # event-loop pass over the input files.
        # Write each variable's histograms directly into its final, compressed output file.
        # SaveHist persists objects via WriteTObject as the actions run, so once the single
        # event loop has executed we just close the files -- no per-variable hadd recompress
        # pass (209 == LZMA level 9, matching the previous `hadd -f209` output compression).
        var_tmp_files = {}
        for var in vars_to_process:
            out_path = os.path.join(args.outDir, f"{var}.root")
            out_root_file = ROOT.TFile(out_path, "RECREATE", "", 209)
            var_tmp_files[var] = (out_path, out_root_file)

        all_save_fns = BuildAllHistActions(
            uncs_to_compute,
            unc_cfg_dict,
            all_trees,
            vars_to_process,
            key_filter_dict,
            further_cuts,
            treeName,
            var_tmp_files,
        )

        for fn in all_save_fns:
            fn()

        for var in vars_to_process:
            _, out_root_file = var_tmp_files[var]
            out_root_file.Close()

    time_elapsed = time.time() - start
    print(f"execution time = {time_elapsed} ")
