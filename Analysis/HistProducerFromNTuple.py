import argparse
import os
import sys
import importlib
import re
import ROOT
import time

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.HistHelper as HistHelper
import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
from Corrections.CorrectionsCore import getScales
from FLAF.Analysis.HistTupleProducer import DefineBinnedColumn
from FLAF.Analysis.histFromNtupleBatch import cells_budget, fill_groups
from FLAF.Common import WindowRegions

# Memory (MB) the histograms of one event loop may take, every thread slot's copy counted, unless
# hist_from_ntuple_max_hist_memory_mb or --max-hist-memory-mb says otherwise.
DEFAULT_MAX_HIST_MEMORY_MB = 1500

# Window regions book only what a flavour lists, but a list can still be too wide (every variable in
# every window): a job refuses more (key, variable) pairs in window regions than this, unless
# window_regions_max_key_vars says otherwise (0 disables the check). 25 masses x 3 regions x 3
# categories x 5 channels x 1 variable = 1125.
DEFAULT_MAX_WINDOW_KEY_VARS = 5000

_HIST_FILLER_H = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "include",
    "HistFiller.h",
)
if not ROOT.gInterpreter.Declare(f'#include "{_HIST_FILLER_H}"'):
    raise RuntimeError(f"Failed to include {_HIST_FILLER_H}")


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


def VarModel(hist_cfg, var):
    """The output histogram model of ``var`` and the bin column of each of its dimensions."""
    var_entry = HistHelper.findBinEntry(hist_cfg, var)
    var_list = hist_cfg[var_entry].get("var_list")
    dims = len(var_list) if var_list else 1
    if dims < 1 or dims > 3:
        raise RuntimeError("Only 1D, 2D and 3D histograms are supported")
    model = HistHelper.GetModel(hist_cfg, var, dims)
    components = list(var_list) if var_list else [var]
    return model, [f"{v}_bin" for v in components]


def ModelBins(model):
    """Bins per dimension of a TH1DModel / TH2DModel / TH3DModel."""
    if isinstance(model, ROOT.RDF.TH3DModel):
        return [model.fNbinsX, model.fNbinsY, model.fNbinsZ]
    if isinstance(model, ROOT.RDF.TH2DModel):
        return [model.fNbinsX, model.fNbinsY]
    return [model.fNbinsX]


def ModelCells(model):
    """Cells of one histogram of a model, under- and overflows included."""
    cells = 1
    for n in ModelBins(model):
        cells *= n + 2
    return cells


def IsVectorType(type_name):
    return "RVec" in type_name or "vector" in type_name


# Element types of a bin column: integers that fit an int (include/HistFiller.h narrows them to int, so a
# wider or an unsigned int type could land in another bin).
BIN_COLUMN_TYPES = {
    "bool",
    "Bool_t",
    "char",
    "Char_t",
    "signed char",
    "unsigned char",
    "UChar_t",
    "short",
    "Short_t",
    "unsigned short",
    "UShort_t",
    "int",
    "Int_t",
}


def ElementType(type_name):
    """The element type of a column type: ``int`` for ``int``, ``ROOT::VecOps::RVec<int>`` or
    ``vector<int>``."""
    base = type_name.replace("const", "").replace("&", "").strip()
    match = re.fullmatch(
        r"(?:ROOT::(?:VecOps::)?RVec|(?:std::)?vector)\s*<\s*(.+?)\s*>", base
    )
    return match.group(1) if match else base


def IsBinColumnType(type_name):
    return ElementType(type_name) in BIN_COLUMN_TYPES


def TreeFills(uncs_to_compute, unc_cfg_dict, treeName):
    """``{tree: [(unc, scale, weight column)]}``: the histograms of each tree, as one (unc, scale) each.
    A shape uncertainty is its own tree filled with the central weight; the central tree also takes
    every normalisation uncertainty as a weight."""
    fills = {}
    for unc, scales in uncs_to_compute.items():
        is_shift_unc = unc in unc_cfg_dict["shape"].keys()
        for scale in scales:
            if is_shift_unc:
                tree, weight = f"Events__{unc}__{scale}", "weight_Central"
            else:
                tree = treeName
                weight = (
                    f"weight_{unc}_{scale}" if unc != "Central" else "weight_Central"
                )
            fills.setdefault(tree, []).append((unc, scale, weight))
    return fills


def FillTree(
    rdf,
    tree_name,
    weights,
    keys,
    vars_to_process,
    var_models,
    var_files,
    max_cells,
    window_blocks=(),
):
    """Fill and write every (key, weight, variable) histogram of one tree, in event loops of at most
    ``max_cells`` cells, every thread slot's copy counted (include/HistFiller.h).

    ``window_blocks`` (Common/WindowRegions.window_blocks): further ``(keys, vars)`` blocks, each key a
    ``(key_tuple, base_key_index, cut)`` selecting the events of base key ``base_key_index`` that pass
    ``cut``. A block fills only its own variables in its own keys, with a filler of its own; the blocks
    of a loop fill in the same pass over the tree. Without blocks the loops are as before.
    """
    columns = {str(c): str(rdf.GetColumnType(c)) for c in rdf.GetColumnNames()}
    for _, _, weight in weights:
        if IsVectorType(columns[weight]):
            raise RuntimeError(f"{tree_name}: weight column {weight} is a vector")
    # a key filter must be a bool, as for RDataFrame's Filter (HistFillerKey refuses anything else)
    pass_expr = (
        "ROOT::RVec<char>{"
        + ", ".join(f"analysis::HistFillerKey(({flt}))" for _, flt in keys)
        + "}"
    )
    # blocks: (key tuples, pass column definition, variables); the base block's pass is _hf_pass
    blocks = [([key_tuple for key_tuple, _ in keys], None, list(vars_to_process))]
    for block_keys, block_vars in window_blocks:
        for _, _, cut in block_keys:
            for col in re.findall(r"\b(\w+_bin)\b", cut):
                if col not in columns:
                    raise RuntimeError(
                        f"{tree_name}: window cut '{cut}' needs the bin column {col}, absent from "
                        "the input: the window variable must be histogrammed by the flavour"
                    )
        pass_def = (
            "ROOT::RVec<char>{"
            + ", ".join(
                f"static_cast<char>(_hf_pass[{index}] && "
                f"analysis::HistFillerKey(({cut})))"
                for _, index, cut in block_keys
            )
            + "}"
        )
        blocks.append(([k for k, _, _ in block_keys], pass_def, list(block_vars)))
    # the unit of the memory budget: one variable of one block, in all the block's keys
    unit_cells = {}
    for b, (block_keys, _, block_vars) in enumerate(blocks):
        for var in block_vars:
            unit_cells[(b, var)] = ModelCells(var_models[var][0]) * len(block_keys)
    n_slots = rdf.GetNSlots()  # each fills its own copy of a loop's histograms
    groups = fill_groups(unit_cells, 1, len(weights), max_cells, n_slots)
    for group_units, (w_first, w_end) in groups:
        group_weights = weights[w_first:w_end]
        loop = rdf.Define(
            "_hf_weights",
            "ROOT::RVec<double>{"
            + ", ".join(f"static_cast<double>({w})" for _, _, w in group_weights)
            + "}",
        ).Define("_hf_pass", pass_expr)
        fillers = []  # (filler, key tuples, variables), one per block in this loop
        filled_cols = []
        for b, (block_keys, pass_def, _) in enumerate(blocks):
            group_vars = [var for unit_block, var in group_units if unit_block == b]
            if not group_vars:
                continue
            suffix = "" if b == 0 else f"_w{b}"
            pass_col = f"_hf_pass{suffix}"
            if pass_def is not None:
                loop = loop.Define(pass_col, pass_def)
            scalar_cols, vector_cols = [], []
            filler = ROOT.analysis.HistFiller()
            for var in group_vars:
                model, bin_cols = var_models[var]
                is_vector, index = [], []
                for col in bin_cols:
                    col_type = columns[col]
                    if not IsBinColumnType(col_type):
                        raise RuntimeError(
                            f"{tree_name}: bin column {col} is {col_type}; a bin column holds "
                            f"integers that fit an int ({', '.join(sorted(BIN_COLUMN_TYPES))})"
                        )
                    cols = vector_cols if IsVectorType(col_type) else scalar_cols
                    if col not in cols:
                        cols.append(col)
                    is_vector.append(1 if cols is vector_cols else 0)
                    index.append(cols.index(col))
                filler.AddVariable(ModelBins(model), is_vector, index)
            filler.Allocate(len(block_keys), len(group_weights), n_slots)
            address = ROOT.addressof(filler)
            loop = (
                loop.Define(
                    f"_hf_scalar_bins{suffix}",
                    "ROOT::RVec<int>{"
                    + ", ".join(f"static_cast<int>({c})" for c in scalar_cols)
                    + "}",
                )
                .Define(
                    f"_hf_vector_bins{suffix}",
                    "ROOT::RVec<ROOT::RVec<int>>{"
                    + ", ".join(f"analysis::HistFillerBins({c})" for c in vector_cols)
                    + "}",
                )
                .Define(
                    f"_hf_filled{suffix}",
                    f"analysis::HistFillerFill({address}, rdfslot_, {pass_col}, "
                    f"_hf_scalar_bins{suffix}, _hf_vector_bins{suffix}, _hf_weights)",
                )
            )
            fillers.append((filler, block_keys, group_vars))
            filled_cols.append(f"_hf_filled{suffix}")
        if len(filled_cols) == 1:
            sum_col = filled_cols[0]
        else:
            # every block's fill must run; an event counts once if it is in any key
            sum_col = "_hf_filled_any"
            loop = loop.Define(
                sum_col,
                "static_cast<int>("
                + " || ".join(f"({c} != 0)" for c in filled_cols)
                + ")",
            )
        t_start = time.time()
        n_filled = loop.Sum(sum_col).GetValue()
        t_loop = time.time() - t_start

        n_cells = 0
        n_hists = 0
        for filler, block_keys, group_vars in fillers:
            hists = ROOT.std.vector["TH1*"]()
            dirs = ROOT.std.vector["TDirectory*"]()
            names = ROOT.std.vector["std::string"]()
            keep_alive = []
            for var in group_vars:
                hist = var_models[var][0].GetHistogram()
                keep_alive.append(hist)
                hists.push_back(hist)
                for key_tuple in block_keys:
                    dirs.push_back(Utilities.mkdir(var_files[var], "/".join(key_tuple)))
                for unc, scale, _ in group_weights:
                    names.push_back(var if unc == "Central" else f"{var}_{unc}_{scale}")
            filler.Write(hists, dirs, names)
            n_cells += filler.NumberOfCells()
            n_hists += len(group_vars) * len(block_keys) * len(group_weights)
            filler.Release()
        print(
            f"{tree_name}: {len(fillers)} block(s), {len(group_units)} (block, var) units, "
            f"{len(group_weights)} weights, {n_hists} hists, {n_cells} cells × {n_slots} thread "
            f"slot(s) (~{n_cells * n_slots * 16 / 1e6:.0f} MB), "
            f"{n_filled:.0f} events in a key; "
            f"loop {t_loop:.1f} s, write {time.time() - t_start - t_loop:.1f} s",
            flush=True,
        )
        del fillers


def BookSignalTreeSnapshots(
    uncs_to_compute,
    unc_cfg_dict,
    all_trees,
    key_filter_dict,
    further_cuts,
    treeName,
    columns,
    tmp_dir,
):
    """Book one lazy Snapshot per tree the histograms read -- the central one and each
    shape-varied one -- so that it runs in their event loop.

    It keeps the events that pass any selection key, with ``columns``, a boolean
    ``sel__<channel>__<region>__<category>[__<further cut>]`` per key holding the filter of
    its histograms and, in the central tree, the weight of every norm variation: each
    histogram can be rebuilt from these rows. Returns ``{variation: (tmp_path, snapshot)}``
    with ``variation`` ``"Central"`` or ``"<unc>_<scale>"``.
    """
    cut_names = list(further_cuts.keys()) if further_cuts else [None]
    sel_columns = {}
    for key, filter_to_apply_base in key_filter_dict.items():
        for further_cut_name in cut_names:
            key_tuple = key + (further_cut_name,) if further_cut_name else key
            sel_columns["sel__" + "__".join(key_tuple)] = (
                f"{filter_to_apply_base} && {further_cut_name}"
                if further_cut_name
                else filter_to_apply_base
            )
    sources = {"Central": (treeName, [])}
    for unc, scales in uncs_to_compute.items():
        if unc == "Central":
            continue
        for scale in scales:
            if unc in unc_cfg_dict["shape"].keys():
                sources[f"{unc}_{scale}"] = (f"Events__{unc}__{scale}", [])
            else:
                sources["Central"][1].append(f"weight_{unc}_{scale}")
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fLazy = True
    snapshots = {}
    for variation, (tree_name, weight_columns) in sources.items():
        rdf = all_trees[tree_name]
        available = set(str(c) for c in rdf.GetColumnNames())
        missing = [c for c in columns + weight_columns if c not in available]
        if missing:
            raise ValueError(
                f"signal_tree_hists_ELReweighter: column(s) {missing} not present in "
                f"HistTuple tree {tree_name!r}."
            )
        for sel_column, filter_to_apply in sel_columns.items():
            rdf = rdf.Define(sel_column, filter_to_apply)
        rdf = rdf.Filter(" || ".join(sel_columns))
        tmp_path = os.path.join(tmp_dir, f"{variation}.root")
        snapshot_columns = list(dict.fromkeys(columns + weight_columns))
        snapshot_columns += list(sel_columns)
        snapshots[variation] = (
            tmp_path,
            rdf.Snapshot(
                "Events", tmp_path, Utilities.ListToVector(snapshot_columns), opts
            ),
        )
    return snapshots


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
    parser.add_argument(
        "--nMT",
        type=int,
        default=1,
        help="Threads of the event loop: 1 (default) runs without implicit MT, 0 uses all "
        "cores (EnableImplicitMT(0)). Every thread fills its own copy of the histograms.",
    )
    parser.add_argument("--user-custom", type=str, default=None)
    parser.add_argument(
        "--max-hist-memory-mb",
        type=float,
        default=None,
        help="Max memory (MB) of the histograms filled in one event loop, every thread "
        "slot's copy of them; a tree is filled in as many loops as its variables need. "
        "Below 1 it is disabled. Default: hist_from_ntuple_max_hist_memory_mb from config, "
        f"else {DEFAULT_MAX_HIST_MEMORY_MB}.",
    )
    args = parser.parse_args()

    if args.nMT != 1:
        ROOT.EnableImplicitMT(max(0, args.nMT))

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

    signal_tree_cfg = setup.global_params.get("signal_tree_hists_ELReweighter") or {}
    is_signal = (
        setup.datasets.get(args.dataset_name, {}).get("process_group") == "signals"
    )

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

    # A norm uncertainty is Up/Down, or the members of a theory uncertainty (PDF, QCD scale)
    # as the corrections configuration declares them (see Common/TheoryUncertainties.py).
    from Corrections.Corrections import Corrections

    Corrections.registerConfiguredSourceScales(setup.global_params)
    uncs_to_compute = {}
    uncs_to_compute["Central"] = ["Central"]
    if args.dataset_name != "data":
        if args.compute_rel_weights:
            uncs_to_compute.update(
                {key: getScales(key) for key in unc_cfg_dict["norm"].keys()}
            )
        if args.compute_unc_variations:
            uncs_to_compute.update(
                {
                    key: setup.global_params["scales"]
                    for key in unc_cfg_dict["shape"].keys()
                }
            )
    print(
        {
            unc: scales if len(scales) <= 3 else f"{len(scales)} members"
            for unc, scales in uncs_to_compute.items()
        }
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
        tree_fills = TreeFills(uncs_to_compute, unc_cfg_dict, treeName)
        missing_trees = sorted(set(tree_fills) - set(all_trees))
        if missing_trees:
            raise RuntimeError(
                f"Trees missing from every input file: {missing_trees}; present: "
                f"{sorted(all_trees)}"
            )
        # One compressed output file per variable (209 == LZMA level 9).
        var_files = {}
        for var in vars_to_process:
            out_path = os.path.join(args.outDir, f"{var}.root")
            var_files[var] = ROOT.TFile(out_path, "RECREATE", "", 209)

        if args.max_hist_memory_mb is not None:
            max_memory_mb = args.max_hist_memory_mb
        else:
            max_memory_mb = float(
                setup.global_params.get(
                    "hist_from_ntuple_max_hist_memory_mb", DEFAULT_MAX_HIST_MEMORY_MB
                )
            )
        max_cells = cells_budget(max_memory_mb)
        cut_names = list(further_cuts.keys()) if further_cuts else [None]
        keys = [
            (
                key + (cut_name,) if cut_name else key,
                f"{key_filter} && {cut_name}" if cut_name else key_filter,
            )
            for key, key_filter in key_filter_dict.items()
            for cut_name in cut_names
        ]
        var_models = {var: VarModel(hist_cfg_dict, var) for var in vars_to_process}
        n_hists = len(vars_to_process) * len(keys) * sum(map(len, tree_fills.values()))
        # Window regions (Common/WindowRegions.py): each fills only its own variables, in its own keys.
        window_blocks = WindowRegions.window_blocks(
            keys, setup.window_regions, vars_to_process
        )
        if window_blocks:
            n_window_pairs = WindowRegions.count_window_hists(window_blocks, 1)
            n_window_hists = n_window_pairs * sum(map(len, tree_fills.values()))
            max_window_pairs = int(
                setup.global_params.get(
                    "window_regions_max_key_vars", DEFAULT_MAX_WINDOW_KEY_VARS
                )
            )
            print(
                f"Window regions: {len(window_blocks)} blocks, "
                f"{sum(len(k) for k, _ in window_blocks)} keys, {n_window_pairs} "
                f"(key, variable) pairs, {n_window_hists} histograms",
                flush=True,
            )
            if 0 < max_window_pairs < n_window_pairs:
                raise RuntimeError(
                    f"The window regions book {n_window_pairs} (key, variable) pairs in this "
                    f"job, above window_regions_max_key_vars = {max_window_pairs}: check the "
                    "flavour's window_regions (masses, regions, vars), or raise the limit on "
                    "purpose"
                )
            n_hists += n_window_hists
        print(
            f"Filling {n_hists} histograms ({len(vars_to_process)} vars × {len(keys)} keys × "
            f"{sum(map(len, tree_fills.values()))} unc/scales) from {len(tree_fills)} trees, "
            f"max hist memory {max_memory_mb:g} MB per event loop (every thread slot's copy)",
            flush=True,
        )

        signal_tree_vars = [
            var
            for var in signal_tree_cfg.get("vars_to_store", [])
            if var in vars_to_process
        ]
        signal_tree_snapshots = {}
        signal_tree_tmp_created = False
        if signal_tree_cfg.get("store_signal_trees") and is_signal and signal_tree_vars:
            # each observable as its histograms are filled (<v>_bin) and, when the
            # HistTuple flavour keeps it at full resolution, its value
            central_columns = set(str(c) for c in all_trees[treeName].GetColumnNames())
            signal_tree_columns = list(signal_tree_cfg.get("extra_vars_to_store", []))
            signal_tree_columns.append("weight_Central")
            for var in signal_tree_vars:
                var_entry = HistHelper.findBinEntry(hist_cfg_dict, var)
                for v in hist_cfg_dict[var_entry].get("var_list") or [var]:
                    signal_tree_columns.append(f"{v}_bin")
                    if v in central_columns:
                        signal_tree_columns.append(v)
            signal_tree_tmp_dir = os.path.join(args.outDir, "signal_tree_tmp")
            signal_tree_tmp_created = not os.path.isdir(signal_tree_tmp_dir)
            os.makedirs(signal_tree_tmp_dir, exist_ok=True)
            # Booked lazily before the fill loops, on the same trees, so each runs in its
            # tree's first event loop rather than in a loop of its own.
            signal_tree_snapshots = BookSignalTreeSnapshots(
                uncs_to_compute,
                unc_cfg_dict,
                all_trees,
                key_filter_dict,
                further_cuts,
                treeName,
                list(dict.fromkeys(signal_tree_columns)),
                signal_tree_tmp_dir,
            )

        for tree_name, weights in tree_fills.items():
            FillTree(
                all_trees[tree_name],
                tree_name,
                weights,
                keys,
                vars_to_process,
                var_models,
                var_files,
                max_cells,
                window_blocks,
            )

        # A snapshot not already run by its tree's first fill loop runs here.
        for variation, (tmp_path, snapshot) in signal_tree_snapshots.items():
            snapshot.GetValue()
            tmp_file = ROOT.TFile.Open(tmp_path, "READ")
            for var in signal_tree_vars:
                HistHelper.concat_trees(
                    Utilities.mkdir(var_files[var], HistHelper.SIGNAL_TREES_DIR),
                    variation,
                    [tmp_file.Get("Events")],
                )
            tmp_file.Close()
            os.remove(tmp_path)
        signal_tree_snapshots.clear()
        if signal_tree_tmp_created:
            os.rmdir(signal_tree_tmp_dir)

        for out_file in var_files.values():
            out_file.Close()

    time_elapsed = time.time() - start
    print(f"execution time = {time_elapsed} ")
