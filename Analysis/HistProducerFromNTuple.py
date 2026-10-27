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
from FLAF.Analysis.histFromNtupleBatch import (
    count_booked_hists,
    iter_hist_batches,
    n_cut_slots,
    unc_scale_pairs,
)


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

    # The model hands out a fresh, directory-less histogram on every call, and Python owns it.
    merged_hist = model.GetHistogram()
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


def BuildSignalTreeExtraSnapshots(
    key_filter_dict,
    further_cuts,
    all_trees,
    treeName,
    uncs_to_compute,
    unc_cfg_dict,
    extra_vars,
    tmp_dir,
):
    """Snapshot ``extra_vars`` once per selection key for Central, reused for every
    norm-type systematic (they share Central's tree and selection, so there is
    nothing to re-derive), and once more per shape-type systematic (which filters a
    genuinely different tree, e.g. ``Events__JES__Up``) -- mirroring
    ``BuildAllHistActions``'s own ``rdf_base`` choice exactly, so the same
    ``(unc, scale)`` resolves to the same source tree in both places.

    Returns ``{(uncName, uncScale): {key_tuple: (tmp_path, tmp_name, snap)}}``;
    ``("Central", "Central")`` and every norm-type entry share the exact same inner
    dict object.
    """
    cut_names = list(further_cuts.keys()) if further_cuts else [None]
    col_vec = Utilities.ListToVector(list(extra_vars))
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fLazy = True

    def _snapshot_base(rdf_base, bucket_name):
        snap_info = {}
        for key, filter_base in key_filter_dict.items():
            for further_cut_name in cut_names:
                filt = (
                    f"{filter_base} && {further_cut_name}"
                    if further_cut_name
                    else filter_base
                )
                key_tuple = key + (further_cut_name,) if further_cut_name else key
                rdf_filtered = rdf_base.Filter(filt)
                tmp_name = f"sigtree_{bucket_name}_" + "_".join(key_tuple)
                tmp_path = os.path.join(tmp_dir, f"{tmp_name}.root")
                snap = rdf_filtered.Snapshot(tmp_name, tmp_path, col_vec, opts)
                snap_info[key_tuple] = (tmp_path, tmp_name, snap)
        return snap_info

    central_snap_info = _snapshot_base(all_trees[treeName], "central")
    by_unc_scale = {("Central", "Central"): central_snap_info}
    for unc, scales in uncs_to_compute.items():
        if unc == "Central":
            continue
        is_shift_unc = unc in unc_cfg_dict["shape"].keys()
        for scale in scales:
            if is_shift_unc:
                rdf_base = all_trees[f"Events__{unc}__{scale}"]
                by_unc_scale[(unc, scale)] = _snapshot_base(rdf_base, f"{unc}_{scale}")
            else:
                by_unc_scale[(unc, scale)] = central_snap_info
    return by_unc_scale


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
    parser.add_argument("--era", required=True, type=str)
    parser.add_argument("--outDir", required=True, type=str)
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--vars", required=True, type=str)
    parser.add_argument(
        "--compute_unc_variations", type=Utilities.str_to_bool, default=False
    )
    parser.add_argument(
        "--compute_rel_weights", type=Utilities.str_to_bool, default=False
    )
    parser.add_argument("--furtherCut", type=str, default=None)
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    parser.add_argument("--nMT", type=int, default=8)
    parser.add_argument("--user-custom", type=str, default=None)
    parser.add_argument(
        "--max-hists",
        type=int,
        default=None,
        help="Max histograms booked in one RDataFrame pass, counting every "
        "(variable, selection, unc, scale) including Up/Down. "
        "0 disables batching. Default: hist_from_ntuple_max_hists from config, else 4000.",
    )
    args = parser.parse_args()

    ROOT.EnableImplicitMT(args.nMT)

    start = time.time()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"],
        args.era,
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

    signal_tree_cfg = setup.global_params.get("signal_tree_hists_ELReweighter", {})
    store_signal_trees = signal_tree_cfg.get("store_signal_trees", False)
    signal_tree_vars = set(signal_tree_cfg.get("vars_to_store", []))
    signal_extra_vars = list(signal_tree_cfg.get("extra_vars_to_store", []))
    is_signal = (
        setup.datasets.get(args.dataset_name, {}).get("process_group") == "signals"
    )
    write_signal_ELR_trees = bool(store_signal_trees and is_signal and signal_tree_vars)

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

    if write_signal_ELR_trees and signal_extra_vars:
        central_cols = set(str(c) for c in all_trees[treeName].GetColumnNames())
        missing = [v for v in signal_extra_vars if v not in central_cols]
        if missing:
            raise ValueError(
                f"signal_tree_hists_ELReweighter.extra_vars_to_store references "
                f"column(s) {missing} not present in HistTuple tree {treeName!r}."
            )
        for unc, scales in uncs_to_compute.items():
            if unc == "Central" or unc not in unc_cfg_dict["shape"]:
                continue
            for scale in scales:
                shift_tree_name = f"Events__{unc}__{scale}"
                shift_cols = set(
                    str(c) for c in all_trees[shift_tree_name].GetColumnNames()
                )
                missing_shift = [v for v in signal_extra_vars if v not in shift_cols]
                if missing_shift:
                    raise ValueError(
                        f"signal_tree_hists_ELReweighter.extra_vars_to_store "
                        f"references column(s) {missing_shift} not present in "
                        f"{shift_tree_name!r}."
                    )

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
        # Open one compressed output file per variable. SaveHist writes into it as
        # each batch's event loop runs (209 == LZMA level 9).
        var_tmp_files = {}
        for var in vars_to_process:
            out_path = os.path.join(args.outDir, f"{var}.root")
            out_root_file = ROOT.TFile(out_path, "RECREATE", "", 209)
            var_tmp_files[var] = (out_path, out_root_file)

        if args.max_hists is not None:
            max_hists = args.max_hists
        else:
            max_hists = int(setup.global_params.get("hist_from_ntuple_max_hists", 4000))
        n_total = count_booked_hists(
            max(1, len(vars_to_process)),
            max(1, len(key_filter_dict)),
            n_cut_slots(further_cuts),
            max(1, len(unc_scale_pairs(uncs_to_compute))),
        )
        batches = list(
            iter_hist_batches(
                uncs_to_compute,
                key_filter_dict,
                further_cuts,
                vars_to_process,
                max_hists,
            )
        )
        print(
            f"Booking {n_total} histograms "
            f"({len(vars_to_process)} vars × {len(key_filter_dict)} keys × "
            f"{n_cut_slots(further_cuts)} cuts × "
            f"{len(unc_scale_pairs(uncs_to_compute))} unc/scales); "
            f"max_hists={max_hists} → {len(batches)} batch(es)"
        )

        signal_extra_snap_info = {}
        if write_signal_ELR_trees and signal_extra_vars:
            signal_tmp_dir = os.path.join(args.outDir, "signal_tree_tmp")
            os.makedirs(signal_tmp_dir, exist_ok=True)
            signal_extra_snap_info = BuildSignalTreeExtraSnapshots(
                key_filter_dict,
                further_cuts,
                all_trees,
                treeName,
                uncs_to_compute,
                unc_cfg_dict,
                signal_extra_vars,
                signal_tmp_dir,
            )

        for batch_idx, (b_uncs, b_keys, b_cuts, b_vars) in enumerate(batches, start=1):
            n_batch = count_booked_hists(
                max(1, len(b_vars)),
                max(1, len(b_keys)),
                n_cut_slots(b_cuts),
                max(1, len(unc_scale_pairs(b_uncs))),
            )
            print(f"Histogram batch {batch_idx}/{len(batches)}: {n_batch} hists")
            save_fns = BuildAllHistActions(
                b_uncs,
                unc_cfg_dict,
                all_trees,
                b_vars,
                b_keys,
                b_cuts,
                treeName,
                var_tmp_files,
            )
            for fn in save_fns:
                fn()
            del save_fns

        if write_signal_ELR_trees and signal_extra_vars:
            cut_names = list(further_cuts.keys()) if further_cuts else [None]
            key_tuples_full = []
            for key in key_filter_dict.keys():
                for further_cut_name in cut_names:
                    key_tuple = key + (further_cut_name,) if further_cut_name else key
                    key_tuples_full.append(key_tuple)

            signal_vars_present = signal_tree_vars & set(vars_to_process)
            # Loop key_tuple (category) outermost and close every snapshot file opened
            # for it before moving to the next one, instead of holding one handle open
            # per (var, unc_scale, key_tuple) for the whole run -- with many categories
            # and systematics that quickly exceeds the process's open-file limit.
            # Cache by tmp_path (not unc_scale): Central and every norm-type systematic
            # share the exact same underlying snapshot file for a given key_tuple, so
            # this also avoids opening that one shared file once per systematic.
            for key_tuple_full in key_tuples_full:
                dir_name = "/".join(key_tuple_full)
                open_by_path = {}  # tmp_path -> (TFile, TTree)
                for var in signal_vars_present:
                    _, out_root_file = var_tmp_files[var]
                    cat_dir = out_root_file.GetDirectory(dir_name)
                    if not cat_dir:
                        continue
                    tree_dir = Utilities.mkdir(out_root_file, dir_name + "/Trees")
                    for key in cat_dir.GetListOfKeys():
                        # The "Trees" subdirectory we just created is itself a key in
                        # cat_dir now; ReadObj() on a directory key that's still empty
                        # on disk corrupts later writes into it when the file closes
                        # -- skip it by name before ever reading it, not just by type.
                        if key.GetName() == "Trees":
                            continue
                        hist = key.ReadObj()
                        if not hist.InheritsFrom("TH1"):
                            continue
                        hist.SetDirectory(0)
                        hist_key_name = key.GetName()
                        unc_scale = HistHelper.resolve_unc_scale(
                            hist_key_name, var, uncs_to_compute.keys()
                        ) or ("Central", "Central")
                        bucket = signal_extra_snap_info.get(unc_scale)
                        if bucket is None or key_tuple_full not in bucket:
                            continue
                        tmp_path, tmp_name, _snap = bucket[key_tuple_full]
                        if tmp_path not in open_by_path:
                            src_file = ROOT.TFile.Open(tmp_path, "READ")
                            open_by_path[tmp_path] = (src_file, src_file.Get(tmp_name))
                        _, src_tree = open_by_path[tmp_path]
                        # TFile.Open() above (on cache miss) leaves gDirectory pointed
                        # at that read-only snapshot file; CloneTree() writes into
                        # whatever directory is "current", so cd() into the writable
                        # output directory immediately before cloning, every time.
                        tree_dir.cd()
                        cloned_tree = src_tree.CloneTree(-1, "fast")
                        tree = HistHelper.wrap_hist_in_tree(
                            hist, hist_key_name, base_tree=cloned_tree
                        )
                        tree_dir.WriteTObject(tree, hist_key_name, "Overwrite")
                for src_file, _ in open_by_path.values():
                    src_file.Close()

            seen_tmp_paths = set()
            for bucket in signal_extra_snap_info.values():
                for tmp_path, _tmp_name, _snap in bucket.values():
                    if tmp_path not in seen_tmp_paths:
                        seen_tmp_paths.add(tmp_path)
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

        for var in vars_to_process:
            _, out_root_file = var_tmp_files[var]
            out_root_file.Close()

    time_elapsed = time.time() - start
    print(f"execution time = {time_elapsed} ")
