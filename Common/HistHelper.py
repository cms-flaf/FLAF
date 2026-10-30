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


_bin_entry_cache = {}


def findBinEntry(hist_cfg_dict, var_name):
    """
    Match variable name against regex-based histogram config entries.

    The answer for a (config, variable) pair is cached: the producers ask it for the same
    variable once per booked histogram, and every miss runs one regex per config entry. An entry
    holds its config, so that config's id cannot be reused by another one, and a hit is only
    taken for that very config.
    """
    key = (id(hist_cfg_dict), var_name)
    cached = _bin_entry_cache.get(key)
    if cached is not None and cached[0] is hist_cfg_dict:
        return cached[1]

    matches = []

    for pattern in hist_cfg_dict.keys():
        if re.fullmatch(pattern, var_name):
            matches.append(pattern)

    if not matches:
        raise KeyError(f"No histogram config pattern matches variable '{var_name}'")

    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous histogram config for '{var_name}': {matches}")

    _bin_entry_cache[key] = (hist_cfg_dict, matches[0])
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
    created. ``base_tree``'s UserInfo is cleared first: CloneTree()/MergeTrees() carry
    the source tree's existing UserInfo along, so without this, merging a tree an
    earlier stage already wrapped would leave both the old and new histogram embedded
    instead of just the current one."""
    tree = base_tree if base_tree is not None else ROOT.TTree(name, name)
    tree.SetName(name)
    tree.SetTitle(name)
    tree.GetUserInfo().Clear()
    hist_clone = hist.Clone()
    hist_clone.SetName(name)
    hist_clone.SetTitle(name)
    hist_clone.SetDirectory(0)
    tree.GetUserInfo().Add(hist_clone)
    return tree


def unwrap_hist_from_tree(tree):
    """Return an independent clone of the histogram embedded in
    ``tree.GetUserInfo()`` (see ``wrap_hist_in_tree``). Must clone, not return the
    UserInfo entry directly: TTree owns that TList and deletes its contents when
    the tree itself is destroyed (e.g. when the source TFile closes), so holding
    onto the raw entry past that point is a dangling pointer -- SetDirectory(0)
    does not protect against this (it only affects TDirectory's separate
    auto-registration, not TList ownership)."""
    hist = tree.GetUserInfo().At(0).Clone()
    hist.SetDirectory(0)
    return hist


def _key_inherits_from(key, base_class_name):
    """Whether a TKey's stored class inherits from ``base_class_name``, checked via
    TClass's class hierarchy from the key's class-name metadata -- without calling
    TKey::ReadObj(), which would deserialize the (possibly large) object just to
    inspect its type."""
    cls = ROOT.TClass.GetClass(key.GetClassName())
    return bool(cls) and cls.InheritsFrom(base_class_name)


def get_all_signal_trees_recursive(root_dir, path=()):
    """Collect every TTree directly inside a directory literally named ``"Trees"``
    anywhere under ``root_dir``, keyed by the channel/region/category path leading to
    it (with the ``"Trees"`` segment itself stripped). Only ever looks one level
    inside a ``Trees`` directory -- this feature's trees live there and nowhere
    else, so there is no need for a fully generic tree walker. Every non-directory
    key (the overwhelming majority -- every plain histogram leaf) is skipped via its
    class-name metadata alone, never deserialized with ReadObj(): this function runs
    unconditionally on every merged file regardless of whether any analysis is even
    using this feature, so it must not pay the cost of reading every histogram just
    to discover it isn't a directory."""
    items_dict = {}
    for key in root_dir.GetListOfKeys():
        if not _key_inherits_from(key, "TDirectory"):
            continue
        obj = key.ReadObj()
        if key.GetName() == "Trees":
            local_items = {}
            for tree_key in obj.GetListOfKeys():
                if not _key_inherits_from(tree_key, "TTree"):
                    continue
                local_items[tree_key.GetName()] = tree_key.ReadObj()
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


def _hist_dir_paths(root_dir, path=()):
    """Paths of the directories holding histograms, in the order get_all_items_recursive finds
    them, from the keys' class names: no histogram is read."""
    paths = []
    has_hists = False
    for key in root_dir.GetListOfKeys():
        cls = ROOT.TClass.GetClass(key.GetClassName())
        if cls and cls.InheritsFrom("TDirectory"):
            paths.extend(
                _hist_dir_paths(
                    root_dir.GetDirectory(key.GetName()), path + (key.GetName(),)
                )
            )
        elif cls and cls.InheritsFrom("TH1"):
            has_hists = True
    if has_hists:
        paths.append(path)
    return paths


def _read_dir_hists(root_file, dir_path):
    """The histograms of one directory, owned by Python as in get_all_items_recursive."""
    directory = root_file.GetDirectory("/".join(dir_path)) if dir_path else root_file
    hists = {}
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            obj.SetDirectory(0)
            ROOT.SetOwnership(obj, True)
            hists[key.GetName()] = obj
    return hists


def _read_dir_signal_trees(root_file, dir_path):
    """The signal-tree feature's TTrees directly inside ``dir_path``'s ``"Trees"`` child,
    if any -- not detached from ``root_file`` (unlike ``_read_dir_hists``'s histograms):
    CloneTree()/MergeTrees() need the source TTree's file to stay open."""
    directory = root_file.GetDirectory("/".join(dir_path)) if dir_path else root_file
    trees_dir = directory.GetDirectory("Trees") if directory else None
    if not trees_dir:
        return {}
    trees = {}
    for key in trees_dir.GetListOfKeys():
        if _key_inherits_from(key, "TTree"):
            trees[key.GetName()] = key.ReadObj()
    return trees


def SumEraFiles(in_paths, out_path, scales):
    """Sum the merged histogram files of several eras into one, bin by bin.

    A variation that an era does not have -- a source decorrelated between the eras, such
    as ``TT_CMS_scale_j_2022_Up`` in Run3_2023 -- enters the sum as that era's central
    histogram of the process: the source does not move that era's events. A plain hadd
    would leave the varied histogram holding the events of its own era alone.

    One directory at a time: only that directory's histograms of the eras are in memory. Holding
    every histogram of every era at once took ~12 GB for a variable with control regions (four
    eras of ~400k histograms).

    Every input file is kept open for the whole call (there are only ever a handful of
    them -- eras, or HistMergerTask's own parallel merge batches -- unlike the many
    per-dataset files HistMergerFromHists.py streams through one at a time), so a
    signal-tree TTree found alongside a directory's histograms can be concatenated via
    CloneTree()/MergeTrees() directly, without re-reading anything from disk.
    """
    in_files = [ROOT.TFile.Open(path, "READ") for path in in_paths]
    era_dirs = [_hist_dir_paths(in_file) for in_file in in_files]
    # 209 = LZMA level 9, as the merged files of one era
    out_file = ROOT.TFile(out_path, "RECREATE", "", 209)
    for dir_path in dict.fromkeys(path for dirs in era_dirs for path in dirs):
        hists = [
            _read_dir_hists(in_file, dir_path) if dir_path in dirs else {}
            for in_file, dirs in zip(in_files, era_dirs)
        ]
        signal_trees = [
            _read_dir_signal_trees(in_file, dir_path) if dir_path in dirs else {}
            for in_file, dirs in zip(in_files, era_dirs)
        ]
        has_signal_trees = any(signal_trees)
        names = list(dict.fromkeys(name for era_hists in hists for name in era_hists))
        centrals = [name for name in names if not name.endswith(tuple(scales))]
        out_dir = mkdir_recursive(out_file, "/".join(dir_path))
        tree_dir = (
            mkdir_recursive(out_file, "/".join(dir_path) + "/Trees")
            if has_signal_trees
            else None
        )
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
            # Clone registers the copy in the open output file, which takes it from Python;
            # detached, it must be given back, or every summed histogram stays allocated.
            total.SetDirectory(0)
            ROOT.SetOwnership(total, True)
            for part in parts[1:]:
                total.Add(part)
            if has_signal_trees:
                # Same (uncName, uncScale) naming convention as the histogram above, so
                # the signal tree resolved via this name/process pair is this one's own
                # -- concatenate rather than sum, since rows (not bin contents) are rows.
                tree_parts = [st.get(name, st.get(process)) for st in signal_trees]
                tree_parts = [part for part in tree_parts if part]
                if tree_parts:
                    tree_dir.cd()
                    if len(tree_parts) == 1:
                        cloned_tree = tree_parts[0].CloneTree(-1, "fast")
                    else:
                        tlist = ROOT.TList()
                        for t in tree_parts:
                            tlist.Add(t)
                        cloned_tree = ROOT.TTree.MergeTrees(tlist)
                    cloned_tree.SetName(name)
                    cloned_tree.SetTitle(name)
                    wrap_hist_in_tree(total, name, base_tree=cloned_tree)
                    tree_dir.WriteTObject(cloned_tree, name, "Overwrite")
                    # Written to disk above; detach the in-memory tree the same way
                    # `total` already is, or it stays registered (and resident) in the
                    # still-open output file until out_file.Close() at the very end --
                    # across every category this run combines, that is the output-side
                    # twin of the input-side accumulation fixed just below.
                    cloned_tree.SetDirectory(0)
                    ROOT.SetOwnership(cloned_tree, True)
            out_dir.WriteTObject(total, name, "Overwrite")
        # Release this directory's signal trees before moving to the next one. Unlike
        # histograms (detached and Python-owned the moment they're read), these were
        # deliberately left attached to their source file above -- CloneTree()/
        # MergeTrees() need it open to read rows -- so without detaching them here,
        # every tree this call ever reads stays registered in its source TFile for the
        # whole run (the same "release only when everything finally closes" pattern
        # fixed in HistMergerFromHists.py's own per-dataset_type loop, reintroduced
        # here in a different spot).
        if has_signal_trees:
            for era_trees in signal_trees:
                for t in era_trees.values():
                    t.SetDirectory(0)
                    ROOT.SetOwnership(t, True)
    out_file.Close()
    for in_file in in_files:
        in_file.Close()
