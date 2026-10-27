import sys
import math
import ROOT
import os
import numpy as np
import array
import re

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities

ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
ROOT.gROOT.ProcessLine(f'#include "FLAF/include/HistHelper.h"')


def findBinEntry(hist_cfg_dict, var_name):
    """
    Match variable name against regex-based histogram config entries.
    """

    matches = []

    for pattern in hist_cfg_dict.keys():
        if re.fullmatch(pattern, var_name):
            matches.append(pattern)

    if not matches:
        raise KeyError(f"No histogram config pattern matches variable '{var_name}'")

    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous histogram config for '{var_name}': {matches}")

    return matches[0]


def get_all_items_recursive(root_dir, path=()):
    items_dict = {}
    local_items = {}

    for key in root_dir.GetListOfKeys():
        obj = key.ReadObj()

        if obj.InheritsFrom("TDirectory"):
            items_dict.update(get_all_items_recursive(obj, path + (key.GetName(),)))
        elif obj.InheritsFrom("TH1"):
            # owned here, not by the file: closing a file that owns ~20k histograms takes
            # minutes, and would delete the ones returned. Owned by Python, so a histogram is
            # freed once nothing refers to it: a merge that only Add()s an input into its
            # running sum would otherwise keep every input histogram until the job ends
            # (all ~170k of a variable; 7-12 GB for the 2D DNN_m*_vs_HME).
            obj.SetDirectory(0)
            ROOT.SetOwnership(obj, True)
            local_items[key.GetName()] = obj

    if local_items:
        items_dict[path] = local_items

    return items_dict


def wrap_hist_in_tree(hist, name, base_tree=None):
    """Attach a clone of ``hist`` to a TTree's ``GetUserInfo()`` list, named to match
    the tree, so the histogram rides through the tree's Write/Read cycle without
    becoming a branch. If ``base_tree`` is given (e.g. a cloned Snapshot tree already
    holding real rows), it is reused and renamed; otherwise a fresh empty tree is
    created."""
    tree = base_tree if base_tree is not None else ROOT.TTree(name, name)
    tree.SetName(name)
    tree.SetTitle(name)
    hist_clone = hist.Clone()
    hist_clone.SetName(name)
    hist_clone.SetTitle(name)
    hist_clone.SetDirectory(0)
    tree.GetUserInfo().Add(hist_clone)
    return tree


def unwrap_hist_from_tree(tree):
    """Return the histogram clone embedded in ``tree.GetUserInfo()`` (see
    ``wrap_hist_in_tree``), detached from any file."""
    hist = tree.GetUserInfo().At(0)
    hist.SetDirectory(0)
    return hist


def get_all_signal_trees_recursive(root_dir, path=()):
    """Collect every TTree directly inside a directory literally named ``"Trees"``
    anywhere under ``root_dir``, keyed by the channel/region/category path leading to
    it (with the ``"Trees"`` segment itself stripped). Only ever looks one level
    inside a ``Trees`` directory -- this feature's trees live there and nowhere
    else, so there is no need for a fully generic tree walker."""
    items_dict = {}
    for key in root_dir.GetListOfKeys():
        obj = key.ReadObj()
        if not obj.InheritsFrom("TDirectory"):
            continue
        if key.GetName() == "Trees":
            local_items = {}
            for tree_key in obj.GetListOfKeys():
                tree_obj = tree_key.ReadObj()
                if tree_obj.InheritsFrom("TTree"):
                    local_items[tree_key.GetName()] = tree_obj
            if local_items:
                items_dict[path] = local_items
        else:
            items_dict.update(
                get_all_signal_trees_recursive(obj, path + (key.GetName(),))
            )
    return items_dict


def resolve_unc_scale(name, var_input, unc_sources):
    """Match a stored histogram/tree name like ``"<var>_<unc>_<scale>"`` (or exactly
    ``"<var>"`` for Central) back to its ``(uncName, uncScale)`` pair, or ``None`` if
    it doesn't belong to ``var_input`` at all."""
    if name == var_input:
        return ("Central", "Central")
    for unc_source in unc_sources:
        if unc_source == "Central":
            continue
        for scale in ("Up", "Down"):
            if name == f"{var_input}_{unc_source}_{scale}":
                return (unc_source, scale)
    return None


def load_all_items(file_path):
    inFile = ROOT.TFile.Open(file_path, "READ")
    if not inFile or inFile.IsZombie():
        raise RuntimeError(f"Errore apertura file {file_path}")
    items = get_all_items_recursive(inFile)
    inFile.Close()
    return items


def save_items_to_root(items_dict, out_file_path):
    outFile = ROOT.TFile(out_file_path, "RECREATE")
    for path_tuple, obj in items_dict.items():
        dir_path = "/".join(path_tuple[:-1])  # tutto tranne l'ultimo (nome oggetto)
        hist_name = path_tuple[-1]
        # crea la directory se non esiste
        dir_ptr = mkdir_recursive(outFile, dir_path)
        obj.SetDirectory(0)  # evita che ROOT lo chiuda prematuramente
        dir_ptr.WriteTObject(obj, hist_name, "Overwrite")
    outFile.Close()


def mkdir_recursive(root_file, dir_path):
    if dir_path == "":
        return root_file
    current = root_file
    for folder in dir_path.split("/"):
        if not current.GetDirectory(folder):
            current.mkdir(folder)
        current = current.GetDirectory(folder)
    return current


def GetUncNameTypes(unc_cfg_dict):
    uncNames = []
    uncNames.extend(list(unc_cfg_dict["norm"].keys()))
    uncNames.extend([unc for unc in unc_cfg_dict["shape"].keys()])
    return uncNames


def createVoidHist(outFileName, hist_cfg_dict, var):
    var_entry = findBinEntry(hist_cfg_dict, var)
    x_bins = hist_cfg_dict[var_entry]["x_bins"]
    if isinstance(hist_cfg_dict["x_bins"], list):
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        hvoid = ROOT.TH1F("", "", x_bins_vec.size() - 1, x_bins_vec.data())
    else:
        n_bins, bin_range = x_bins.split("|")
        start, stop = bin_range.split(":")
        hvoid = ROOT.TH1F("", "", int(n_bins), float(start), float(stop))
    outFile = ROOT.TFile(outFileName, "RECREATE")
    hvoid.Write()
    outFile.Close()


def RenormalizeHistogram(histogram, norm, include_overflows=True):
    integral = (
        histogram.Integral(0, histogram.GetNbinsX() + 1)
        if include_overflows
        else histogram.Integral()
    )
    if integral != 0:
        histogram.Scale(norm / integral)


def FixNegativeContributions(histogram):
    correction_factor = 0.0

    ss_debug = ""
    ss_negative = ""

    original_Integral = histogram.Integral(0, histogram.GetNbinsX() + 1)
    ss_debug += "\nSubtracted hist for '{}'.\n".format(histogram.GetName())
    ss_debug += "Integral after bkg subtraction: {}.\n".format(original_Integral)
    if original_Integral < 0:
        print(ss_debug)
        print(
            "Integral after bkg subtraction is negative for histogram '{}'".format(
                histogram.GetName()
            )
        )
        return False, ss_debug, ss_negative

    for n in range(1, histogram.GetNbinsX() + 1):
        if histogram.GetBinContent(n) >= 0:
            continue
        prefix = (
            "WARNING"
            if histogram.GetBinContent(n) + histogram.GetBinError(n) >= 0
            else "ERROR"
        )

        ss_negative += (
            "{}: {} Bin {}, content = {}, error = {}, bin limits=[{},{}].\n".format(
                prefix,
                histogram.GetName(),
                n,
                histogram.GetBinContent(n),
                histogram.GetBinError(n),
                histogram.GetBinLowEdge(n),
                histogram.GetBinLowEdge(n + 1),
            )
        )

        error = correction_factor - histogram.GetBinContent(n)
        new_error = math.sqrt(
            math.pow(error, 2) + math.pow(histogram.GetBinError(n), 2)
        )
        histogram.SetBinContent(n, correction_factor)
        histogram.SetBinError(n, new_error)

    RenormalizeHistogram(histogram, original_Integral, True)
    return True, ss_debug, ss_negative


def RebinHisto(hist_initial, new_binning, sample, wantOverflow=True, verbose=False):
    if isinstance(new_binning, dict):
        # Prepare data structures for C++ function
        y_bin_ranges = ROOT.std.vector("std::pair<float,float>")()
        output_bin_edges_vec = ROOT.std.vector("std::vector<float>")()

        for combined_bin in new_binning["combined_bins"]:
            # Parse y_bin range
            y_min, y_max = combined_bin["y_bin"]
            y_bin_ranges.push_back(ROOT.std.pair("float", "float")(y_min, y_max))

            # Parse x_bins spec (can be string "nbins|min:max" or list of bin edges)
            out_spec = combined_bin["x_bins"]
            out_edges = ROOT.std.vector("float")()
            if isinstance(out_spec, list):
                for edge in out_spec:
                    out_edges.push_back(float(edge))
            else:
                n_out_bins, out_range = out_spec.split("|")
                out_min, out_max = map(float, out_range.split(":"))
                n_out_bins = int(n_out_bins)
                # Create uniform bins
                step = (out_max - out_min) / n_out_bins
                for i in range(n_out_bins + 1):
                    out_edges.push_back(out_min + i * step)
            output_bin_edges_vec.push_back(out_edges)

        # Create ROOT vectors
        # Call the C++ function which returns a new histogram
        new_hist = ROOT.analysis.rebinHistogramDict(
            hist_initial, y_bin_ranges, output_bin_edges_vec
        )
        new_hist.SetName(sample)

    else:
        new_binning_array = array.array("d", new_binning)
        new_hist = hist_initial.Rebin(len(new_binning) - 1, sample, new_binning_array)

    if sample == "data":
        new_hist.SetBinErrorOption(ROOT.TH1.kPoisson)

    if wantOverflow:
        # Merge overflow into the last bin
        n_bins = new_hist.GetNbinsX()
        n_finalbin = new_hist.GetBinContent(n_bins)
        n_overflow = new_hist.GetBinContent(n_bins + 1)
        new_hist.SetBinContent(n_bins, n_finalbin + n_overflow)

        err_finalbin = new_hist.GetBinError(n_bins)
        err_overflow = new_hist.GetBinError(n_bins + 1)
        new_hist.SetBinError(n_bins, math.sqrt(err_finalbin**2 + err_overflow**2))

    if verbose:
        for nbin in range(len(new_binning)):
            print(
                f"bin {nbin}, content = {new_hist.GetBinContent(nbin)}, error = {new_hist.GetBinError(nbin)}"
            )

    # Fix possible negative bins
    fix_ok, debug_info, negative_bins = FixNegativeContributions(new_hist)
    if not fix_ok:
        print("Negative bins not fixed:", debug_info, negative_bins)
        for nbin in range(new_hist.GetNbinsX() + 1):
            if new_hist.GetBinContent(nbin) < 0:
                print(
                    f"{sample}, bin {nbin} content is < 0: {new_hist.GetBinContent(nbin)}"
                )

    return new_hist


def GetBinVec(x_bins):
    if isinstance(x_bins, dict):
        return x_bins

    x_bins_vec = None
    if not isinstance(x_bins, list):
        n_bins, bin_range = x_bins.split("|")
        start, stop = bin_range.split(":")
        x_bins = np.linspace(float(start), float(stop), int(n_bins) + 1).tolist()
    x_bins_vec = Utilities.ListToVector(x_bins, "float")
    return x_bins_vec


def GetModel(hist_cfg, var, dims, return_unit_bin_model=False):
    THModel_Inputs = []
    unit_bin_Inputs = []
    var_entry = findBinEntry(hist_cfg, var)
    if dims == 1:
        x_bins_vec = GetBinVec(hist_cfg[var_entry]["x_bins"])
        THModel_Inputs.append(x_bins_vec.size() - 1)
        THModel_Inputs.append(x_bins_vec.data())
        model = ROOT.RDF.TH1DModel("", "", *THModel_Inputs)
        if not return_unit_bin_model:
            return model
        unit_bin_Inputs = [model.fNbinsX, -0.5, model.fNbinsX - 0.5]
        unit_bin_model = ROOT.RDF.TH1DModel("", "", *unit_bin_Inputs)

    elif (dims == 2) or (dims == 3):
        list_var_bins_vec = []
        for var_nD in hist_cfg[var_entry]["var_list"]:
            var_bin_name = f"{var_nD}_bins"
            var_bins = (
                hist_cfg[var_entry][var_bin_name]
                if var_bin_name in hist_cfg[var_entry]
                else hist_cfg[var_nD]["x_bins"]
            )
            var_bins_vec = GetBinVec(var_bins)
            list_var_bins_vec.append(var_bins_vec)
            THModel_Inputs.append(var_bins_vec.size() - 1)
            THModel_Inputs.append(var_bins_vec.data())
        if dims == 2:
            model = ROOT.RDF.TH2DModel("", "", *THModel_Inputs)
            if not return_unit_bin_model:
                return model
            unit_bin_Inputs = [
                model.fNbinsX,
                -0.5,
                model.fNbinsX - 0.5,
                model.fNbinsY,
                -0.5,
                model.fNbinsY - 0.5,
            ]
            unit_bin_model = ROOT.RDF.TH2DModel("", "", *unit_bin_Inputs)
        if dims == 3:
            model = ROOT.RDF.TH3DModel("", "", *THModel_Inputs)
            if not return_unit_bin_model:
                return model
            unit_bin_Inputs = [
                model.fNbinsX,
                -0.5,
                model.fNbinsX - 0.5,
                model.fNbinsY,
                -0.5,
                model.fNbinsY - 0.5,
                model.fNbinsZ,
                -0.5,
                model.fNbinsZ - 0.5,
            ]
            unit_bin_model = ROOT.RDF.TH3DModel("", "", *unit_bin_Inputs)
    else:
        raise RuntimeError("nD histogram not implemented yet")
        # model = ROOT.RDF.THnDModel("", "", )

    return model, unit_bin_model


def SumEraFiles(in_paths, out_path, scales):
    """Sum the merged histogram files of several eras into one, bin by bin.

    A variation that an era does not have -- a source decorrelated between the eras, such
    as ``TT_CMS_scale_j_2022_Up`` in Run3_2023 -- enters the sum as that era's central
    histogram of the process: the source does not move that era's events. A plain hadd
    would leave the varied histogram holding the events of its own era alone.
    """
    in_files = [ROOT.TFile.Open(path, "READ") for path in in_paths]
    eras = [get_all_items_recursive(in_file) for in_file in in_files]
    signal_tree_eras = [get_all_signal_trees_recursive(in_file) for in_file in in_files]

    # Signal Trees/<name>: concatenate rows across eras first, since uproot needs to
    # create the file fresh; sum each name's embedded histogram along the way and
    # hold the result for the attach pass after the plain-histogram loop below. The
    # plain-histogram loop reopens in "UPDATE" mode when this ran, so it appends
    # rather than wiping what was just written -- same ordering the merger uses.
    signal_tree_dir_paths = list(
        dict.fromkeys(path for era in signal_tree_eras for path in era)
    )
    merged_signal_hists = {}  # (dir_path, name) -> TH1
    if signal_tree_dir_paths:
        import uproot

        with uproot.recreate(out_path) as uproot_out:
            for dir_path in signal_tree_dir_paths:
                tree_maps = [era.get(dir_path, {}) for era in signal_tree_eras]
                names = list(dict.fromkeys(name for tm in tree_maps for name in tm))
                centrals = [n for n in names if not n.endswith(tuple(scales))]
                out_tree_path_base = "/".join(dir_path) + "/Trees/"
                for name in names:
                    # the process of a variation: the longest central name it extends
                    process = None
                    if name not in centrals:
                        process = max(
                            (c for c in centrals if name.startswith(c + "_")),
                            key=len,
                            default=None,
                        )
                    out_tree_path = out_tree_path_base + name
                    contributing_trees = []
                    for path, tm in zip(in_paths, tree_maps):
                        tree = tm.get(name, tm.get(process))
                        if tree is None:
                            continue
                        contributing_trees.append(tree)
                        internal_path = out_tree_path_base + tree.GetName()
                        with uproot.open(f"{path}:{internal_path}") as src_tree:
                            for arrays in src_tree.iterate():
                                if out_tree_path in uproot_out:
                                    uproot_out[out_tree_path].extend(arrays)
                                else:
                                    uproot_out[out_tree_path] = arrays
                    if contributing_trees:
                        hists = [unwrap_hist_from_tree(t) for t in contributing_trees]
                        total = hists[0].Clone(name)
                        total.SetDirectory(0)
                        for h in hists[1:]:
                            total.Add(h)
                        merged_signal_hists[(dir_path, name)] = total

    # 209 = LZMA level 9, as the merged files of one era
    out_file = ROOT.TFile(
        out_path, "UPDATE" if signal_tree_dir_paths else "RECREATE", "", 209
    )
    for dir_path in dict.fromkeys(path for era in eras for path in era):
        hists = [era.get(dir_path, {}) for era in eras]
        names = list(dict.fromkeys(name for era_hists in hists for name in era_hists))
        centrals = [name for name in names if not name.endswith(tuple(scales))]
        out_dir = mkdir_recursive(out_file, "/".join(dir_path))
        for name in names:
            # the process of a variation: the longest central name it extends
            process = None
            if name not in centrals:
                process = max(
                    (c for c in centrals if name.startswith(c + "_")),
                    key=len,
                    default=None,
                )
            parts = [h.get(name, h.get(process)) for h in hists]
            parts = [part for part in parts if part]
            total = parts[0].Clone(name)
            total.SetDirectory(0)
            for part in parts[1:]:
                total.Add(part)
            out_dir.WriteTObject(total, name, "Overwrite")
    out_file.Close()

    if merged_signal_hists:
        out_file = ROOT.TFile(out_path, "UPDATE", "", 209)
        for (dir_path, name), hist in merged_signal_hists.items():
            tree_path = "/".join(dir_path) + "/Trees/" + name
            tree = out_file.Get(tree_path)
            wrap_hist_in_tree(hist, name, base_tree=tree)
            tree.Write("", ROOT.TObject.kOverwrite)
        out_file.Close()

    for in_file in in_files:
        in_file.Close()
