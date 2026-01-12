import sys
import math
import ROOT
import os
import numpy as np
import array

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities


def get_all_items_recursive(root_dir, path=()):
    items_dict = {}
    local_items = {}

    for key in root_dir.GetListOfKeys():
        obj = key.ReadObj()

        if obj.InheritsFrom("TDirectory"):
            items_dict.update(get_all_items_recursive(obj, path + (key.GetName(),)))
        elif obj.InheritsFrom("TH1"):
            obj.SetDirectory(0)
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


def createVoidHist(outFileName, hist_cfg_dict):
    x_bins = hist_cfg_dict["x_bins"]
    if type(hist_cfg_dict["x_bins"]) == list:
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

    n_bins, bin_range = bins.split("|")
    start, stop = map(float, bin_range.split(":"))
    step = (stop - start) / int(n_bins)

    return [start + i * step for i in range(int(n_bins) + 1)]


def RebinHisto(hist_initial, new_binning, sample, wantOverflow=True, verbose=False):
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


def GetBinVec(hist_cfg, var):
    x_bins = hist_cfg[var]["x_bins"]
    x_bins_vec = None
    if type(hist_cfg[var]["x_bins"]) == list:
        x_bins_vec = Utilities.ListToVector(x_bins, "float")
    else:
        n_bins, bin_range = x_bins.split("|")
        start, stop = bin_range.split(":")
        edges = np.linspace(float(start), float(stop), int(n_bins)).tolist()
        # print(len(edges))
        x_bins_vec = Utilities.ListToVector(edges, "float")
    return x_bins_vec


def GetModel(hist_cfg, var, return_unit_bin_model=False):
    x_bins = hist_cfg[var]["x_bins"]
    if type(hist_cfg[var]["x_bins"]) == list:
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        model = ROOT.RDF.TH1DModel("", "", x_bins_vec.size() - 1, x_bins_vec.data())
    else:
        n_bins, bin_range = x_bins.split("|")
        start, stop = bin_range.split(":")
        model = ROOT.RDF.TH1DModel("", "", int(n_bins), float(start), float(stop))
    if not return_unit_bin_model:
        return model
    unit_bin_model = ROOT.RDF.TH1DModel(
        "", "", model.fNbinsX, -0.5, model.fNbinsX - 0.5
    )
    return model, unit_bin_model
