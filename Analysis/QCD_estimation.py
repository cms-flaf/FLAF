import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.HistHelper import *
from FLAF.Common.Utilities import *


def QCD_Estimation(
    histograms,
    all_samples_list,
    channel,
    category,
    uncName,
    scale,
    wantNegativeContributions,
):
    key_B = ((channel, "OS_AntiIso", category), (uncName, scale))
    key_C = ((channel, "SS_Iso", category), (uncName, scale))
    key_D = ((channel, "SS_AntiIso", category), (uncName, scale))
    hist_data = histograms["data"]
    hist_data_B = hist_data[key_B].Clone()
    hist_data_C = hist_data[key_C].Clone()
    hist_data_D = hist_data[key_D].Clone()
    n_data_B = hist_data_B.Integral(0, hist_data_B.GetNbinsX() + 1)
    n_data_C = hist_data_C.Integral(0, hist_data_C.GetNbinsX() + 1)
    n_data_D = hist_data_D.Integral(0, hist_data_D.GetNbinsX() + 1)
    print(f"Initially Yield for data in OS AntiIso region is {key_B} is {n_data_B}")
    print(f"Initially Yield for data in SS Iso region is{key_C} is {n_data_C}")
    print(f"Initially Yield for data in SS AntiIso region is{key_D} is {n_data_D}")
    for sample in all_samples_list:
        if (
            sample == "data"
            or "GluGluToBulkGraviton" in sample
            or "GluGluToRadion" in sample
            or "VBFToBulkGraviton" in sample
            or "VBFToRadion" in sample
            or sample == "QCD"
        ):
            # print(f"sample {sample} is not considered")
            continue
        # print(sample)
        hist_sample = histograms[sample]
        hist_sample_B = hist_sample[key_B].Clone()
        hist_sample_C = hist_sample[key_C].Clone()
        hist_sample_D = hist_sample[key_D].Clone()
        n_sample_B = hist_sample_B.Integral(0, hist_sample_B.GetNbinsX() + 1)
        n_data_B -= n_sample_B
        n_sample_C = hist_sample_C.Integral(0, hist_sample_C.GetNbinsX() + 1)
        n_data_C -= n_sample_C
        n_sample_D = hist_sample_D.Integral(0, hist_sample_D.GetNbinsX() + 1)
        n_data_D -= n_sample_D
        if n_data_B < 0:
            print(
                f"Yield for data in OS AntiIso region {key_B} after removing {sample} with yield {n_sample_B} is {n_data_B}"
            )
        if n_data_C < 0:
            print(
                f"Yield for data in SS Iso region {key_C} after removing {sample} with yield {n_sample_C} is {n_data_C}"
            )
        if n_data_D < 0:
            print(
                f"Yield for data in SS AntiIso region {key_D} after removing {sample} with yield {n_sample_D} is {n_data_D}"
            )
        hist_data_B.Add(hist_sample_B, -1)
        hist_data_C.Add(hist_sample_C, -1)
    if n_data_C <= 0 or n_data_D <= 0 or n_data_B <= 0:
        print(f"n_data_B = {n_data_B}")
        print(f"n_data_C = {n_data_C}")
        print(f"n_data_D = {n_data_D}")
    if n_data_B <= 0:
        n_data_B = 0.0
    if n_data_D <= 0:
        n_data_D = 0.0
    qcd_norm = n_data_B / n_data_D if n_data_D != 0.0 else 0.0
    # if qcd_norm<0 or n_data_C < 0:
    if n_data_C < 0:
        print(
            f"transfer factor <0, {category}, {channel}, {uncName}, {scale}, returning 0."
        )
        print(f"num {n_data_B}, den {n_data_D}")
        return (
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            0.0,
            0.0,
        )
    print(f"n_data_B = {n_data_B}")
    print(f"n_data_C = {n_data_C}")
    print(f"n_data_D = {n_data_D}")
    print(f"QCD norm = {qcd_norm}")
    hist_qcd_Central = hist_data_C.Clone()
    hist_qcd_Central.Scale(qcd_norm)
    n_data_D_abs = n_data_D
    if n_data_D < 0:
        n_data_D_abs = math.sqrt(n_data_D * n_data_D)
        print(f"attention, n_data_D < 0, {n_data_D}")
    n_data_B_abs = n_data_B
    if n_data_B < 0:
        n_data_B_abs = math.sqrt(n_data_B * n_data_B)
        print(f"attention, n_data_B < 0, {n_data_B}")
    error_on_qcdnorm = (
        math.sqrt(
            n_data_B_abs / (n_data_D_abs * n_data_D_abs)
            + (n_data_B_abs * n_data_B_abs * n_data_D_abs)
            / (n_data_D_abs * n_data_D_abs * n_data_D_abs * n_data_D_abs)
        )
        if n_data_D_abs != 0.0
        else 0.0
    )

    qcd_norm_shape = n_data_C / n_data_D if n_data_D != 0.0 else 0.0
    hist_qcd_Up = hist_data_B.Clone()
    hist_qcd_Up.Scale(qcd_norm_shape)
    hist_qcd_Down = hist_data_B.Clone()
    hist_qcd_Down.Scale(qcd_norm_shape)
    n_data_C_abs = n_data_C
    if n_data_C < 0:
        n_data_C_abs = math.sqrt(n_data_C * n_data_C)
        print(f"attention, n_data_C < 0, {n_data_C}")
    error_on_qcdnorm_varied = (
        math.sqrt(
            n_data_C_abs / (n_data_D_abs * n_data_D_abs)
            + (n_data_C_abs * n_data_C_abs * n_data_D_abs)
            / (n_data_D_abs * n_data_D_abs * n_data_D_abs * n_data_D_abs)
        )
        if n_data_D_abs != 0.0
        else 0.0
    )

    if wantNegativeContributions:
        fix_negative_contributions, debug_info, negative_bins_info = (
            FixNegativeContributions(hist_qcd_Central)
        )
        if not fix_negative_contributions:
            # return hist_data_B
            print(debug_info)
            print(negative_bins_info)
            print("Unable to estimate QCD")
            final_hist = ROOT.TH1D(
                "",
                "",
                hist_qcd_Central.GetNbinsX(),
                hist_qcd_Central.GetXaxis().GetBinLowEdge(1),
                hist_qcd_Central.GetXaxis().GetBinUpEdge(hist_qcd_Central.GetNbinsX()),
            ), ROOT.TH1D(
                "",
                "",
                hist_qcd_Central.GetNbinsX(),
                hist_qcd_Central.GetXaxis().GetBinLowEdge(1),
                hist_qcd_Central.GetXaxis().GetBinUpEdge(hist_qcd_Central.GetNbinsX()),
            )
            return final_hist, final_hist, final_hist, 0, 0
            # raise RuntimeError("Unable to estimate QCD")
    # if uncName == 'Central':
    #    return hist_qcd_Central,hist_qcd_Up,hist_qcd_Down
    return (
        hist_qcd_Central,
        hist_qcd_Up,
        hist_qcd_Down,
        error_on_qcdnorm,
        error_on_qcdnorm_varied,
    )


def QCD_Estimation_symm(
    histograms,
    all_samples_list,
    channel,
    category,
    uncName,
    scale,
    wantNegativeContributions,
):
    key_B = ((channel, "OS_AntiIso", category), (uncName, scale))
    key_C = ((channel, "SS_Iso", category), (uncName, scale))
    key_D = ((channel, "SS_AntiIso", category), (uncName, scale))
    hist_data = histograms["data"]
    hist_data_B = hist_data[key_B].Clone()
    hist_data_C = hist_data[key_C].Clone()
    hist_data_D = hist_data[key_D].Clone()
    n_data_B = hist_data_B.Integral(0, hist_data_B.GetNbinsX() + 1)
    n_data_C = hist_data_C.Integral(0, hist_data_C.GetNbinsX() + 1)
    n_data_D = hist_data_D.Integral(0, hist_data_D.GetNbinsX() + 1)
    print(f"Initially Yield for data in OS AntiIso region is {key_B} is {n_data_B}")
    print(f"Initially Yield for data in SS Iso region is{key_C} is {n_data_C}")
    print(f"Initially Yield for data in SS AntiIso region is{key_D} is {n_data_D}")
    for sample in all_samples_list:
        if (
            sample == "data"
            or "GluGluToBulkGraviton" in sample
            or "GluGluToRadion" in sample
            or "VBFToBulkGraviton" in sample
            or "VBFToRadion" in sample
            or sample == "QCD"
        ):
            ##print(f"sample {sample} is not considered")
            continue
        # print(sample)
        hist_sample = histograms[sample]
        hist_sample_B = hist_sample[key_B].Clone()
        hist_sample_C = hist_sample[key_C].Clone()
        hist_sample_D = hist_sample[key_D].Clone()
        n_sample_B = hist_sample_B.Integral(0, hist_sample_B.GetNbinsX() + 1)
        n_data_B -= n_sample_B
        n_sample_C = hist_sample_C.Integral(0, hist_sample_C.GetNbinsX() + 1)
        n_data_C -= n_sample_C
        n_sample_D = hist_sample_D.Integral(0, hist_sample_D.GetNbinsX() + 1)
        n_data_D -= n_sample_D
        if n_data_B < 0:
            print(
                f"Yield for data in OS AntiIso region {key_B} after removing {sample} with yield {n_sample_B} is {n_data_B}"
            )
        if n_data_C < 0:
            print(
                f"Yield for data in SS Iso region {key_C} after removing {sample} with yield {n_sample_C} is {n_data_C}"
            )
        if n_data_D < 0:
            print(
                f"Yield for data in SS AntiIso region {key_D} after removing {sample} with yield {n_sample_D} is {n_data_D}"
            )
        hist_data_B.Add(hist_sample_B, -1)
        hist_data_C.Add(hist_sample_C, -1)
    if n_data_C <= 0 or n_data_D <= 0:
        print(f"n_data_C = {n_data_C}")
        print(f"n_data_D = {n_data_D}")
    # for symmetrization uncomment those lines

    qcd_norm = n_data_B * n_data_C / n_data_D if n_data_D != 0 else 0
    if qcd_norm < 0 or n_data_C < 0:
        print(f"transfer factor <0, {category}, {channel}, {uncName}, {scale}")
        return (
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            0.0,
            0.0,
        )
        # raise  RuntimeError(f"transfer factor <=0 ! {qcd_norm}")
    # hist_data_B.Scale(kappa)
    if n_data_B != 0:
        hist_data_B.Scale(1 / n_data_B)
    if n_data_C != 0:
        hist_data_C.Scale(1 / n_data_C)

    hist_qcd_Up = hist_data_B.Clone()
    hist_qcd_Up.Scale(qcd_norm)
    hist_qcd_Down = hist_data_C.Clone()
    hist_qcd_Down.Scale(qcd_norm)
    hist_qcd_Central = hist_data_B.Clone()
    hist_qcd_Central.Add(hist_data_C)
    hist_qcd_Central.Scale(1.0 / 2.0)
    hist_qcd_Central.Scale(qcd_norm)
    """
    qcd_norm = n_data_B / n_data_D if n_data_D != 0. else 0.
    if qcd_norm<0:
        print(f"transfer factor <0, {category}, {channel}, {uncName}, {scale}, returning 0.")
        print(f"num {n_data_B}, den {n_data_D}")
        return ROOT.TH1D("","",hist_data_B.GetNbinsX(), hist_data_B.GetXaxis().GetBinLowEdge(1), hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX())),ROOT.TH1D("","",hist_data_B.GetNbinsX(), hist_data_B.GetXaxis().GetBinLowEdge(1), hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX())),ROOT.TH1D("","",hist_data_B.GetNbinsX(), hist_data_B.GetXaxis().GetBinLowEdge(1), hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX())),0.,0.
    hist_qcd_Central = hist_data_C.Clone()
    hist_qcd_Central.Scale(qcd_norm)
    n_data_D_abs = n_data_D
    if n_data_D < 0:
        n_data_D_abs = math.sqrt(n_data_D * n_data_D)
        print(f"attention, n_data_D < 0, {n_data_D}")
    n_data_B_abs = n_data_B
    if n_data_B < 0:
        n_data_B_abs = math.sqrt(n_data_B * n_data_B)
        print(f"attention, n_data_B < 0, {n_data_B}")
    error_on_qcdnorm = math.sqrt(n_data_B_abs/(n_data_D_abs * n_data_D_abs) + (n_data_B_abs*n_data_B_abs*n_data_D_abs)/(n_data_D_abs*n_data_D_abs*n_data_D_abs*n_data_D_abs)) if n_data_D_abs != 0. else 0.

    qcd_norm_shape = n_data_C / n_data_D if n_data_D != 0. else 0.
    hist_qcd_Up = hist_data_B.Clone()
    hist_qcd_Up.Scale(qcd_norm_shape)
    hist_qcd_Down = hist_data_B.Clone()
    hist_qcd_Down.Scale(qcd_norm_shape)
    n_data_C_abs = n_data_C
    if n_data_C < 0:
        n_data_C_abs = math.sqrt(n_data_C * n_data_C)
        print(f"attention, n_data_C < 0, {n_data_C}")
    error_on_qcdnorm_varied = math.sqrt(n_data_C_abs/(n_data_D_abs * n_data_D_abs) + (n_data_C_abs*n_data_C_abs*n_data_D_abs)/(n_data_D_abs*n_data_D_abs*n_data_D_abs*n_data_D_abs)) if n_data_D_abs != 0. else 0.
    """

    if wantNegativeContributions:
        fix_negative_contributions, debug_info, negative_bins_info = (
            FixNegativeContributions(hist_qcd_Central)
        )
        if not fix_negative_contributions:
            # return hist_data_B
            print(debug_info)
            print(negative_bins_info)
            print("Unable to estimate QCD")
            final_hist = ROOT.TH1D(
                "",
                "",
                hist_qcd_Central.GetNbinsX(),
                hist_qcd_Central.GetXaxis().GetBinLowEdge(1),
                hist_qcd_Central.GetXaxis().GetBinUpEdge(hist_qcd_Central.GetNbinsX()),
            ), ROOT.TH1D(
                "",
                "",
                hist_qcd_Central.GetNbinsX(),
                hist_qcd_Central.GetXaxis().GetBinLowEdge(1),
                hist_qcd_Central.GetXaxis().GetBinUpEdge(hist_qcd_Central.GetNbinsX()),
            )
            return final_hist, final_hist, final_hist, 0, 0
            # raise RuntimeError("Unable to estimate QCD")
    # if uncName == 'Central':
    #    return hist_qcd_Central,hist_qcd_Up,hist_qcd_Down

    return hist_qcd_Central, hist_qcd_Up, hist_qcd_Down, 1.0, 1.0


def QCD_Estimation_Inverted(
    histograms,
    all_samples_list,
    channel,
    category,
    uncName,
    scale,
    wantNegativeContributions,
):
    key_C = ((channel, "OS_AntiIso", category), (uncName, scale))
    key_B = ((channel, "SS_Iso", category), (uncName, scale))
    key_D = ((channel, "SS_AntiIso", category), (uncName, scale))
    hist_data = histograms["data"]
    hist_data_B = hist_data[key_B].Clone()
    hist_data_C = hist_data[key_C].Clone()
    hist_data_D = hist_data[key_D].Clone()
    n_data_B = hist_data_B.Integral(0, hist_data_B.GetNbinsX() + 1)
    n_data_C = hist_data_C.Integral(0, hist_data_C.GetNbinsX() + 1)
    n_data_D = hist_data_D.Integral(0, hist_data_D.GetNbinsX() + 1)
    print(f"Initially Yield for data in OS AntiIso region is {key_B} is {n_data_B}")
    print(f"Initially Yield for data in SS Iso region is{key_C} is {n_data_C}")
    print(f"Initially Yield for data in SS AntiIso region is{key_D} is {n_data_D}")
    for sample in all_samples_list:
        if (
            sample == "data"
            or "GluGluToBulkGraviton" in sample
            or "GluGluToRadion" in sample
            or "VBFToBulkGraviton" in sample
            or "VBFToRadion" in sample
            or sample == "QCD"
        ):
            ##print(f"sample {sample} is not considered")
            continue
        # print(sample)
        hist_sample = histograms[sample]
        hist_sample_B = hist_sample[key_B].Clone()
        hist_sample_C = hist_sample[key_C].Clone()
        hist_sample_D = hist_sample[key_D].Clone()
        n_sample_B = hist_sample_B.Integral(0, hist_sample_B.GetNbinsX() + 1)
        n_data_B -= n_sample_B
        n_sample_C = hist_sample_C.Integral(0, hist_sample_C.GetNbinsX() + 1)
        n_data_C -= n_sample_C
        n_sample_D = hist_sample_D.Integral(0, hist_sample_D.GetNbinsX() + 1)
        n_data_D -= n_sample_D
        if n_data_B < 0:
            print(
                f"Yield for data in OS AntiIso region {key_B} after removing {sample} with yield {n_sample_B} is {n_data_B}"
            )
        if n_data_C < 0:
            print(
                f"Yield for data in SS Iso region {key_C} after removing {sample} with yield {n_sample_C} is {n_data_C}"
            )
        if n_data_D < 0:
            print(
                f"Yield for data in SS AntiIso region {key_D} after removing {sample} with yield {n_sample_D} is {n_data_D}"
            )
        hist_data_B.Add(hist_sample_B, -1)
        hist_data_C.Add(hist_sample_C, -1)
    if n_data_C <= 0 or n_data_D <= 0:
        print(f"n_data_C = {n_data_C}")
        print(f"n_data_D = {n_data_D}")

    qcd_norm = n_data_B / n_data_D if n_data_D != 0.0 else 0.0
    if qcd_norm < 0 or n_data_C < 0:
        print(
            f"transfer factor <0, {category}, {channel}, {uncName}, {scale}, returning 0."
        )
        print(f"num {n_data_B}, den {n_data_D}")
        return (
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            ROOT.TH1D(
                "",
                "",
                hist_data_B.GetNbinsX(),
                hist_data_B.GetXaxis().GetBinLowEdge(1),
                hist_data_B.GetXaxis().GetBinUpEdge(hist_data_B.GetNbinsX()),
            ),
            0.0,
            0.0,
        )
    hist_qcd_Central = hist_data_C.Clone()
    hist_qcd_Central.Scale(qcd_norm)
    n_data_D_abs = n_data_D
    if n_data_D < 0:
        n_data_D_abs = math.sqrt(n_data_D * n_data_D)
        print(f"attention, n_data_D < 0, {n_data_D}")
    n_data_B_abs = n_data_B
    if n_data_B < 0:
        n_data_B_abs = math.sqrt(n_data_B * n_data_B)
        print(f"attention, n_data_B < 0, {n_data_B}")
    error_on_qcdnorm = (
        math.sqrt(
            n_data_B_abs / (n_data_D_abs * n_data_D_abs)
            + (n_data_B_abs * n_data_B_abs * n_data_D_abs)
            / (n_data_D_abs * n_data_D_abs * n_data_D_abs * n_data_D_abs)
        )
        if n_data_D_abs != 0.0
        else 0.0
    )

    qcd_norm_shape = n_data_C / n_data_D if n_data_D != 0.0 else 0.0
    hist_qcd_Up = hist_data_B.Clone()
    hist_qcd_Up.Scale(qcd_norm_shape)
    hist_qcd_Down = hist_data_B.Clone()
    hist_qcd_Down.Scale(qcd_norm_shape)
    n_data_C_abs = n_data_C
    if n_data_C < 0:
        n_data_C_abs = math.sqrt(n_data_C * n_data_C)
        print(f"attention, n_data_C < 0, {n_data_C}")
    error_on_qcdnorm_varied = (
        math.sqrt(
            n_data_C_abs / (n_data_D_abs * n_data_D_abs)
            + (n_data_C_abs * n_data_C_abs * n_data_D_abs)
            / (n_data_D_abs * n_data_D_abs * n_data_D_abs * n_data_D_abs)
        )
        if n_data_D_abs != 0.0
        else 0.0
    )

    if wantNegativeContributions:
        fix_negative_contributions, debug_info, negative_bins_info = (
            FixNegativeContributions(hist_qcd_Central)
        )
        if not fix_negative_contributions:
            # return hist_data_B
            print(debug_info)
            print(negative_bins_info)
            print("Unable to estimate QCD")
            final_hist = ROOT.TH1D(
                "",
                "",
                hist_qcd_Central.GetNbinsX(),
                hist_qcd_Central.GetXaxis().GetBinLowEdge(1),
                hist_qcd_Central.GetXaxis().GetBinUpEdge(hist_qcd_Central.GetNbinsX()),
            ), ROOT.TH1D(
                "",
                "",
                hist_qcd_Central.GetNbinsX(),
                hist_qcd_Central.GetXaxis().GetBinLowEdge(1),
                hist_qcd_Central.GetXaxis().GetBinUpEdge(hist_qcd_Central.GetNbinsX()),
            )
            return final_hist, final_hist, final_hist, 0, 0
            # raise RuntimeError("Unable to estimate QCD")
    # if uncName == 'Central':
    #    return hist_qcd_Central,hist_qcd_Up,hist_qcd_Down
    return (
        hist_qcd_Central,
        hist_qcd_Up,
        hist_qcd_Down,
        error_on_qcdnorm,
        error_on_qcdnorm_varied,
    )


def AddQCDInHistDict(
    var,
    all_histograms,
    channels,
    categories,
    uncName,
    all_samples_list,
    scales,
    wantNegativeContributions=False,
):
    if "QCD" not in all_histograms.keys():
        all_histograms["QCD"] = {}
    for channel in channels:
        for cat in categories:
            for scale in scales + ["Central"]:
                if uncName == "Central" and scale != "Central":
                    continue
                if uncName != "Central" and scale == "Central":
                    continue
                key = ((channel, "OS_Iso", cat), (uncName, scale))
                (
                    hist_qcd_Central,
                    hist_qcd_Up,
                    hist_qcd_Down,
                    error_on_qcdnorm,
                    error_on_qcdnorm_varied,
                ) = QCD_Estimation(
                    all_histograms, all_samples_list, channel, cat, uncName, scale, True
                )
                all_histograms["QCD"][key] = hist_qcd_Central
            if uncName == "QCDScale":
                keyQCD_up = ((channel, "OS_Iso", cat), ("QCDScale", "Up"))
                keyQCD_down = ((channel, "OS_Iso", cat), ("QCDScale", "Down"))
                all_histograms["QCD"][keyQCD_up] = hist_qcd_Up
                all_histograms["QCD"][keyQCD_down] = hist_qcd_Down
            if uncName == "Central":
                print(
                    f"for {channel}, {cat} and {var} the errors on QCD are {error_on_qcdnorm},{error_on_qcdnorm_varied}"
                )
    return error_on_qcdnorm, error_on_qcdnorm_varied


#### outdated ####
def CompareYields(histograms, all_samples_list, channel, category, uncName, scale):
    # print(channel, category)
    # print(histograms.keys())key_B_data = ((channel, 'OS_AntiIso', category), ('Central', 'Central'))
    key_A_data = ((channel, "OS_Iso", category), ("Central", "Central"))
    key_A = ((channel, "OS_Iso", category), (uncName, scale))
    key_B_data = ((channel, "OS_AntiIso", category), ("Central", "Central"))
    key_B = ((channel, "OS_AntiIso", category), (uncName, scale))
    key_C_data = ((channel, "SS_Iso", category), ("Central", "Central"))
    key_C = ((channel, "SS_Iso", category), (uncName, scale))
    key_D_data = ((channel, "SS_AntiIso", category), ("Central", "Central"))
    key_D = ((channel, "SS_AntiIso", category), (uncName, scale))
    hist_data = histograms["data"]
    # print(hist_data.keys())
    hist_data_A = hist_data[key_A_data]
    hist_data_B = hist_data[key_B_data]
    # if channel != 'tauTau' and category != 'inclusive': return hist_data_B
    hist_data_C = hist_data[key_C_data]
    hist_data_D = hist_data[key_D_data]
    n_data_A = hist_data_A.Integral(0, hist_data_A.GetNbinsX() + 1)
    n_data_B = hist_data_B.Integral(0, hist_data_B.GetNbinsX() + 1)
    n_data_C = hist_data_C.Integral(0, hist_data_C.GetNbinsX() + 1)
    n_data_D = hist_data_D.Integral(0, hist_data_D.GetNbinsX() + 1)
    print(f"data || {key_A_data} || {n_data_A}")
    print(f"data || {key_B_data} || {n_data_B}")
    print(f"data || {key_C_data} || {n_data_C}")
    print(f"data || {key_D_data} || {n_data_D}")
    for sample in all_samples_list:
        # print(sample)
        # find kappa value
        hist_sample = histograms[sample]
        # print(histograms[sample].keys())
        hist_sample_A = hist_sample[key_A]
        hist_sample_B = hist_sample[key_B]
        hist_sample_C = hist_sample[key_C]
        hist_sample_D = hist_sample[key_D]
        n_sample_A = hist_sample_A.Integral(0, hist_sample_A.GetNbinsX() + 1)
        n_sample_B = hist_sample_B.Integral(0, hist_sample_B.GetNbinsX() + 1)
        n_sample_C = hist_sample_C.Integral(0, hist_sample_C.GetNbinsX() + 1)
        n_sample_D = hist_sample_D.Integral(0, hist_sample_D.GetNbinsX() + 1)

        print(f"{sample} || {key_A} || {n_sample_A}")
        print(f"{sample} || {key_B} || {n_sample_B}")
        print(f"{sample} || {key_C} || {n_sample_C}")
        print(f"{sample} || {key_D} || {n_sample_D}")
