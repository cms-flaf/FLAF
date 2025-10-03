import sys
import math
import ROOT
import os

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities


def GetUncNameTypes(unc_cfg_dict):
    uncNames = []
    uncNames.extend(list(unc_cfg_dict["norm"].keys()))
    uncNames.extend([unc for unc in unc_cfg_dict["shape"]])
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


def createInvMass(df):

    df = df.Define("tautau_m_vis", "static_cast<float>((tau1_p4+tau2_p4).M())")
    particleNet_mass = (
        "particleNet_mass"
        if "SelectedFatJet_particleNet_mass_boosted" in df.GetColumnNames()
        else "particleNetLegacy_mass"
    )
    df = df.Define(
        "bb_m_vis_pnet",
        f"""
                   return static_cast<float>(SelectedFatJet_{particleNet_mass}_boosted);
                   """,
    )
    df = df.Define(
        "bb_m_vis_softdrop",
        f"""
                   return static_cast<float>(SelectedFatJet_msoftdrop_boosted);
                   """,
    )
    df = df.Define(
        "bb_m_vis_fj",
        f"""
                   return static_cast<float>(SelectedFatJet_mass_boosted);
                    """,
    )

    df = df.Define(
        "bb_m_vis",
        f""" if(b1_pt < 0. || b2_pt < 0.) return 0.f; return static_cast<float>((b1_p4+b2_p4).M());""",
    )
    df = df.Define(
        "bbtautau_mass_boosted",
        """return static_cast<float>((SelectedFatJet_p4_boosted+tau1_p4+tau2_p4).M());""",
    )
    df = df.Define(
        "bbtautau_mass",
        """if(b1_pt < 0. || b2_pt < 0.) return 0.f; return static_cast<float>((b1_p4+b2_p4+tau1_p4+tau2_p4).M());""",
    )
    df = df.Define("dR_tautau", "ROOT::Math::VectorUtil::DeltaR(tau1_p4, tau2_p4)")
    df = df.Define("dR_bb", "ROOT::Math::VectorUtil::DeltaR(b1_p4, b2_p4)")
    for tau_idx in [1, 2]:
        for met_var in ["met", "metnomu", "met_nano"]:
            df = df.Define(
                f"tau{tau_idx}_{met_var}_mt",
                f"static_cast<float>((tau{tau_idx}_p4+{met_var}_p4).Mt())",
            )
    return df


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


def GetModel(hist_cfg, var):
    x_bins = hist_cfg[var]["x_bins"]
    if type(hist_cfg[var]["x_bins"]) == list:
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        model = ROOT.RDF.TH1DModel("", "", x_bins_vec.size() - 1, x_bins_vec.data())
    else:
        n_bins, bin_range = x_bins.split("|")
        start, stop = bin_range.split(":")
        model = ROOT.RDF.TH1DModel("", "", int(n_bins), float(start), float(stop))
    return model


def Get2DModel(hist_cfg, var):
    x_bins = hist_cfg[var]["x_bins"]
    if type(hist_cfg[var]["x_bins"]) == list:
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        model = ROOT.RDF.TH2DModel(
            "", "", x_bins_vec.size() - 1, x_bins_vec.data(), 13, -0.5, 12.5
        )
    else:
        n_bins, bin_range = x_bins.split("|")
        start, stop = bin_range.split(":")
        model = ROOT.RDF.TH2DModel(
            "", "", int(n_bins), float(start), float(stop), 13, -0.5, 12.5
        )
    return model
