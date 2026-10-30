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
from FLAF.Analysis.HistTupleProducer import DefineBinnedColumn
from FLAF.Analysis.histFromNtupleBatch import cells_budget, fill_groups

# Memory (MB) the histograms of one event loop may take, every thread slot's copy counted, unless
# hist_from_ntuple_max_hist_memory_mb or --max-hist-memory-mb says otherwise.
DEFAULT_MAX_HIST_MEMORY_MB = 1500

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
):
    """Fill and write every (key, weight, variable) histogram of one tree, in event loops of at most
    ``max_cells`` cells, every thread slot's copy counted (include/HistFiller.h)."""
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
    var_cells = {var: ModelCells(var_models[var][0]) for var in vars_to_process}
    n_slots = rdf.GetNSlots()  # each fills its own copy of a loop's histograms
    groups = fill_groups(var_cells, len(keys), len(weights), max_cells, n_slots)
    for group_vars, (w_first, w_end) in groups:
        group_weights = weights[w_first:w_end]
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
        filler.Allocate(len(keys), len(group_weights), n_slots)
        address = ROOT.addressof(filler)
        loop = (
            rdf.Define(
                "_hf_scalar_bins",
                "ROOT::RVec<int>{"
                + ", ".join(f"static_cast<int>({c})" for c in scalar_cols)
                + "}",
            )
            .Define(
                "_hf_vector_bins",
                "ROOT::RVec<ROOT::RVec<int>>{"
                + ", ".join(f"analysis::HistFillerBins({c})" for c in vector_cols)
                + "}",
            )
            .Define(
                "_hf_weights",
                "ROOT::RVec<double>{"
                + ", ".join(f"static_cast<double>({w})" for _, _, w in group_weights)
                + "}",
            )
            .Define("_hf_pass", pass_expr)
            .Define(
                "_hf_filled",
                f"analysis::HistFillerFill({address}, rdfslot_, _hf_pass, "
                "_hf_scalar_bins, _hf_vector_bins, _hf_weights)",
            )
        )
        t_start = time.time()
        n_filled = loop.Sum("_hf_filled").GetValue()
        t_loop = time.time() - t_start

        hists = ROOT.std.vector["TH1*"]()
        dirs = ROOT.std.vector["TDirectory*"]()
        names = ROOT.std.vector["std::string"]()
        keep_alive = []
        for var in group_vars:
            hist = var_models[var][0].GetHistogram()
            keep_alive.append(hist)
            hists.push_back(hist)
            for key_tuple, _ in keys:
                dirs.push_back(Utilities.mkdir(var_files[var], "/".join(key_tuple)))
            for unc, scale, _ in group_weights:
                names.push_back(var if unc == "Central" else f"{var}_{unc}_{scale}")
        filler.Write(hists, dirs, names)
        filler.Release()
        print(
            f"{tree_name}: {len(group_vars)} vars × {len(keys)} keys × "
            f"{len(group_weights)} weights, {filler.NumberOfCells()} cells × {n_slots} thread "
            f"slot(s) (~{filler.NumberOfCells() * n_slots * 16 / 1e6:.0f} MB), "
            f"{n_filled:.0f} events in a key; "
            f"loop {t_loop:.1f} s, write {time.time() - t_start - t_loop:.1f} s",
            flush=True,
        )
        del filler


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
        print(
            f"Filling {n_hists} histograms ({len(vars_to_process)} vars × {len(keys)} keys × "
            f"{sum(map(len, tree_fills.values()))} unc/scales) from {len(tree_fills)} trees, "
            f"max hist memory {max_memory_mb:g} MB per event loop (every thread slot's copy)",
            flush=True,
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
            )

        for out_file in var_files.values():
            out_file.Close()

    time_elapsed = time.time() - start
    print(f"execution time = {time_elapsed} ")
