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
            local_items[key.GetName()] = obj

    if local_items:
        items_dict[path] = local_items

    return items_dict


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


def getNewBins(bins):
    if isinstance(bins, list):
        return bins

    if isinstance(bins, dict):
        return bins

    n_bins, bin_range = bins.split("|")
    start, stop = map(float, bin_range.split(":"))
    step = (stop - start) / int(n_bins)

    return [start + i * step for i in range(int(n_bins) + 1)]


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
