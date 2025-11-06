import ROOT
import array
import math
import numpy as np

period_dict = {
    "Run3_2022": "7.9804",
    "Run3_2022EE": "26.6717",
    "Run3_2023": "18.063",
    "Run3_2023BPix": "9.693",
}


def compute_stat_unc(histograms):
    """
    Computes the statistical uncertainty for a sum of histograms.
    Returns the total histogram content and the total statistical error per bin.
    """
    if not histograms:
        return np.array([]), np.array([])

    bin_counts = [
        np.array([h.GetBinContent(i + 1) for i in range(h.GetNbinsX())])
        for h in histograms.values()
    ]
    bin_errors_sq = [
        np.array([h.GetBinError(i + 1) ** 2 for i in range(h.GetNbinsX())])
        for h in histograms.values()
    ]

    total_content = np.sum(bin_counts, axis=0)
    total_error = np.sqrt(np.sum(bin_errors_sq, axis=0))

    return total_content, total_error


def extract_config_for_sample(contrib, input_cfg):
    """
    Finds the configuration for a given contribution type.
    """
    for group in input_cfg:
        if "name" in group.keys() and contrib == group["name"]:
            return group
    return {}


def resolve_text_positions(text_cfgs):
    """
    Resolves relative positions of text boxes from configuration.
    """
    resolved = {}
    resolving = set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(
                f"Cyclic reference detected in textbox positions at '{name}'"
            )
        if name not in text_cfgs:
            raise ValueError(f"Textbox '{name}' not found in config")

        resolving.add(name)
        cfg = text_cfgs[name]
        rel_pos = cfg.get("pos", [0, 0])
        ref = cfg.get("ref")

        if ref:
            ref_pos = resolve(ref)
            abs_pos = [ref_pos[0] + rel_pos[0], ref_pos[1] + rel_pos[1]]
        else:
            abs_pos = rel_pos

        resolved[name] = abs_pos
        resolving.remove(name)
        return abs_pos

    for name in text_cfgs:
        resolve(name)
    return resolved


def GetHistName(sample_name, sample_type, uncName, unc_scale, global_cfg_dict):
    sample_namehist = (
        sample_type
        if sample_type in global_cfg_dict["sample_types_to_merge"]
        else sample_name
    )
    onlyCentral = sample_name == "data" or uncName == "Central"
    histName = sample_namehist
    if not onlyCentral:
        histName = f"{sample_namehist}_{uncName}{unc_scale}"
    return histName


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


def RenormalizeHistogram(histogram, norm, include_overflows=True):
    integral = (
        histogram.Integral(0, histogram.GetNbinsX() + 1)
        if include_overflows
        else histogram.Integral()
    )
    if integral != 0:
        histogram.Scale(norm / integral)


def RebinHisto(hist_initial, new_binning, sample, wantOverflow=True, verbose=False):
    new_binning_array = array.array("d", new_binning)
    new_hist = hist_initial.Rebin(len(new_binning) - 1, sample, new_binning_array)
    if sample == "data":
        new_hist.SetBinErrorOption(ROOT.TH1.kPoisson)
    if wantOverflow:
        n_finalbin = new_hist.GetBinContent(new_hist.GetNbinsX())
        n_overflow = new_hist.GetBinContent(new_hist.GetNbinsX() + 1)
        new_hist.SetBinContent(new_hist.GetNbinsX(), n_finalbin + n_overflow)
        err_finalbin = new_hist.GetBinError(new_hist.GetNbinsX())
        err_overflow = new_hist.GetBinError(new_hist.GetNbinsX() + 1)
        new_hist.SetBinError(
            new_hist.GetNbinsX(),
            math.sqrt(err_finalbin * err_finalbin + err_overflow * err_overflow),
        )

    if verbose:
        for nbin in range(0, len(new_binning)):
            print(
                f"nbin = {nbin}, content = {new_hist.GetBinContent(nbin)}, error {new_hist.GetBinError(nbin)}"
            )
    fix_negative_contributions, debug_info, negative_bins_info = (
        FixNegativeContributions(new_hist)
    )

    return new_hist


def findNewBins(hist_cfg_dict, var, **keys):
    cfg = hist_cfg_dict.get(var, {})

    if "x_rebin" not in cfg:
        return cfg.get("x_bins", [])

    x_rebin = cfg["x_rebin"]
    if isinstance(x_rebin, list):
        return x_rebin

    def recursive_search(d, remaining_keys):
        if isinstance(d, list):
            return d
        if not remaining_keys and isinstance(d, dict) and "other" in d:
            return d["other"]
        if not isinstance(d, dict):
            return None
        for k_name, k_value in remaining_keys.items():
            if k_value in d:
                found = recursive_search(
                    d[k_value],
                    {kk: vv for kk, vv in remaining_keys.items() if kk != k_name},
                )
                if found is not None:
                    return found
        if "other" in d:
            return d["other"]
        return None

    result = recursive_search(x_rebin, {k: v for k, v in keys.items() if v is not None})
    return result if result is not None else cfg.get("x_bins", [])


def getNewBins(bins):
    if type(bins) == list:
        final_bins = bins
    else:  # Format like "10|0:100"
        n_bins_str, bin_range = bins.split("|")
        start_str, stop_str = bin_range.split(":")
        n_bins = int(n_bins_str)
        start = float(start_str)
        stop = float(stop_str)
        bin_width = (stop - start) / n_bins
        final_bins = []
        for i in range(n_bins + 1):
            final_bins.append(start + i * bin_width)
    return final_bins


def get_histograms_from_dir(directory, sample_type, hist_dict, pre_path=None):
    keys = [k.GetName() for k in directory.GetListOfKeys()]
    pre_path_list = pre_path.split("/") if pre_path else []
    if sample_type in keys:
        obj = directory.Get(sample_type)
        if obj.IsA().InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)

            path = directory.GetPath().split(":")[-1].strip("/")

            if path not in hist_dict:
                hist_dict[path] = {}

            if sample_type not in hist_dict[path]:
                hist_dict[path][sample_type] = obj
            else:
                hist_dict[path][sample_type].Add(obj)

    for key in keys:
        if pre_path_list and key not in pre_path_list:
            continue
        sub_dir = directory.Get(key)
        if sub_dir.IsA().InheritsFrom(ROOT.TDirectory.Class()):
            get_histograms_from_dir(sub_dir, sample_type, hist_dict)
