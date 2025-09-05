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


def filter_files_with_tree(file_list, treeName):
    valid_files = []
    for f in file_list:
        rf = ROOT.TFile.Open(f)
        if not rf or rf.IsZombie():
            rf.Close()
            continue
        if rf.Get(treeName):
            valid_files.append(f)
        rf.Close()
    return valid_files


def find_keys(inFiles_list):
    """Trova e restituisce tutte le chiavi (tree) uniche dai file di input."""
    unique_keys = set()
    for infile in inFiles_list:
        rf = ROOT.TFile.Open(infile)
        if not rf or rf.IsZombie():
            continue
        for key in rf.GetListOfKeys():
            unique_keys.add(key.GetName())
        rf.Close()
    return sorted(unique_keys)


ROOT.gInterpreter.Declare(
    """
#include <vector>
#include <cmath>
#include <algorithm>
#include "ROOT/RVec.hxx"

template <typename T>
float GetBinValue(const T& bin, const std::vector<float>& edges) {
    int ibin = static_cast<int>(bin);
    float max_val = *std::max_element(edges.begin(), edges.end());
    if (std::abs(ibin) >= static_cast<int>(edges.size()))
        return std::copysign(max_val, bin);
    else if (ibin <= 0)
        return 0.f;
    else
        return edges.at(ibin);
}

template <typename T>
ROOT::VecOps::RVec<float> GetBinValue(const ROOT::VecOps::RVec<T>& bins, const std::vector<float>& edges) {
    ROOT::VecOps::RVec<float> result;
    float max_val = *std::max_element(edges.begin(), edges.end());
    for (const auto& bin : bins) {
        int ibin = static_cast<int>(bin);
        if (std::abs(ibin) >= static_cast<int>(edges.size()))
            result.push_back(std::copysign(max_val, bin));
        else if (ibin <= 0)
            continue;
        else
            result.push_back(edges.at(ibin));
    }
    return result;
}
"""
)


def SaveHist(key_tuple, outFile, hist_list, hist_name, unc, scale):
    """Salva gli istogrammi uniti in un file ROOT nella directory corrispondente."""
    dir_name = "/".join(key_tuple)
    dir_ptr = Utilities.mkdir(outFile, dir_name)
    merged_hist = hist_list[0].GetValue()
    if len(hist_list) > 1:
        for hist in hist_list[1:]:
            merged_hist.Add(hist.GetValue())

    isCentral = unc == "Central"
    final_hist_name = hist_name if isCentral else f"{hist_name}_{unc}_{scale}"
    dir_ptr.WriteTObject(merged_hist, final_hist_name, "Overwrite")


def GetBinValues(rdf, hist_cfg_dict, var):
    """Aggiunge le colonne con i valori dei bin all'RDataFrame."""
    edges_vector = GetBinVec(hist_cfg_dict, var)
    rdf = rdf.Define(
        f"{var}_edges_vector",
        f"""std::vector<float> edges_vector({edges_vector}); return edges_vector;""",
    ).Define(f"{var}", f"GetBinValue({var}_bin, {var}_edges_vector)")
    return rdf


def GetHist(rdf, var, filter_to_apply, weight_name, unc, scale):
    histo = rdf.Filter(filter_to_apply).Histo1D(
        GetModel(hist_cfg_dict, f"{var}"), f"{var}", weight_name
    )
    histo.SetName(f"{var}_{unc}_{scale}")
    histo.SetTitle(f"{var}_{unc}_{scale}")
    return histo


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
            hist_list.append(
                GetHist(rdf_shift, var, filter_expr, "weight_Central", unc, scale)
            )
    else:
        weight_name = f"weight_{unc}_{scale}" if unc != "Central" else "weight_Central"
        rdf_central = all_trees[treeName]
        hist_list.append(
            GetHist(rdf_central, var, filter_expr, weight_name, unc, scale)
        )

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
                # only with further cuts
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


# -------------------- MAIN --------------------
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
    for key in unique_keys:
        if not key.startswith(treeName):
            continue
        valid_files = []
        for f in all_infiles:
            rf = ROOT.TFile.Open(f)
            if rf and rf.Get(key):
                valid_files.append(f)
            rf.Close()

        if valid_files:
            base_rdfs[key] = ROOT.RDataFrame(key, Utilities.ListToVector(valid_files))

    hist_cfg_dict = setup.hists

    channels = (
        args.channels.split(",")
        if args.channels
        else setup.global_params["channelSelection"]
    )
    setup.global_params["channels_to_consider"] = channels

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
                rdf = GetBinValues(rdf, hist_cfg_dict, var)

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
