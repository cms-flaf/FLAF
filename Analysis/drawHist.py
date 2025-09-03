import ROOT
import sys
import math
import os

ROOT.gStyle.SetOptStat(0)
if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


import FLAF.Common.Utilities as Utilities

bTagWP = 2


def defineSelectionRegions(df):
    df = df.Define(
        "nSelBtag",
        f"int(b1_idbtagDeepFlavB >= {bTagWP}) + int(b2_idbtagDeepFlavB >= {bTagWP})",
    )
    df = df.Define("res1b", f"nSelBtag == 1")
    df = df.Define("res2b", f"nSelBtag == 2")
    df = df.Define("inclusive", f"return true;")
    return df


def defineChannels(df):
    df = df.Define("eTau", f"channelId==13")
    df = df.Define("muTau", f"channelId==23")
    df = df.Define("tauTau", f"channelId==33")
    return df


def defineQCDRegions(df):
    tau2_iso_var = f"tau2_idDeepTau{self.deepTauYear()}v{self.deepTauVersion}VSjet"
    df = df.Define("OS", "tau1_charge*tau2_charge < 0")
    df = df.Define(
        "Iso", f"{tau2_iso_var} >= {Utilities.WorkingPointsTauVSjet.Medium.value}"
    )
    df = df.Define(
        "AntiIso",
        f"{tau2_iso_var} >= {Utilities.WorkingPointsTauVSjet.VVVLoose.value} && !Iso",
    )
    df = df.Define("OS_Iso", f"OS && Iso")
    df = df.Define("SS_Iso", f"!OS && Iso")
    df = df.Define("OS_AntiIso", f"OS && AntiIso")
    df = df.Define("SS_AntiIso", f"!OS && AntiIso")
    return df


model = ROOT.RDF.TH1DModel("", "", 71, -0.5, 70.5)
anaTuple_df = ROOT.RDataFrame("Events", "output/TTToSemiLeptonic/anaTuple/nano_0.root")
"""
nanoAOD_Selected_df = ROOT.RDataFrame("Events", "output/TTToSemiLeptonic/nanoAOD/nano_0.root")
nanoAOD_notSelected_df = ROOT.RDataFrame("EventsNotSelected", "output/TTToSemiLeptonic/nanoAOD/nano_0.root")

anaTuple_hist = anaTuple_df.Histo1D(model,"Pileup_nTrueInt").GetValue()
anaTuple_hist.SetLineColor(ROOT.kBlue)
nanoAOD_Selected_hist = nanoAOD_Selected_df.Histo1D(model,"Pileup_nTrueInt").GetValue()
nanoAOD_Selected_hist.SetLineColor(ROOT.kRed)
nanoAOD_notSelected_hist = nanoAOD_notSelected_df.Histo1D(model,"Pileup_nTrueInt").GetValue()
nanoAOD_notSelected_hist.SetLineColor(ROOT.kGreen)
all_nano_hist = nanoAOD_notSelected_hist
all_nano_hist.Add(nanoAOD_Selected_hist,1)
all_nano_hist.Scale(1/all_nano_hist.Integral(0, all_nano_hist.GetNbinsX()))
all_nano_hist.SetLineColor(ROOT.kRed)

canvas = ROOT.TCanvas("Pileup_nTrueInt", "Pileup_nTrueInt", 800, 600)
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)  # Le coordinate sono relative al canvas (xmin, ymin, xmax, ymax)
legend.AddEntry(anaTuple_hist, "anaTuple", "l")
#legend.AddEntry(nanoAOD_Selected_hist, "nano selected", "l")
#legend.AddEntry(nanoAOD_notSelected_hist, "nano not selected", "l")
legend.AddEntry(all_nano_hist, "nano inclusive", "l")

anaTuple_hist.Scale(1/anaTuple_hist.Integral(0, anaTuple_hist.GetNbinsX()))
all_nano_hist.Scale(1/all_nano_hist.Integral(0, all_nano_hist.GetNbinsX()))
nanoAOD_notSelected_hist.Scale(1/nanoAOD_notSelected_hist.Integral(0, nanoAOD_notSelected_hist.GetNbinsX()))
nanoAOD_Selected_hist.Scale(1/nanoAOD_Selected_hist.Integral(0, nanoAOD_Selected_hist.GetNbinsX()))

#nanoAOD_Selected_hist.Draw("same")
all_nano_hist.Draw("same")
anaTuple_hist.Draw("same")
#nanoAOD_notSelected_hist.Draw("same")
legend.Draw()
canvas.Update()
canvas.SaveAs("Pileup_nTrueInt.png")
"""
anaTuple_df = defineSelectionRegions(anaTuple_df)
anaTuple_df = defineChannels(anaTuple_df)
anaTuple_df = defineQCDRegions(anaTuple_df)

sum_weight_total = (
    anaTuple_df.Filter("b1_pt>0&&b2_pt>0 && tauTau && HLT_ditau && SS_Iso && res2b")
    .Sum("weight_total")
    .GetValue()
)
error_weight_total = math.sqrt(
    anaTuple_df.Filter("tauTau && HLT_ditau && SS_Iso && res2b")
    .Define("weight_total_squared", "weight_total*weight_total")
    .Sum("weight_total_squared")
    .GetValue()
)
print(f"""the sum of weight_total is {sum_weight_total}""")
print(f"""the error on sum of weight_total is {error_weight_total}""")
sum_weight_puUp = (
    anaTuple_df.Filter("b1_pt>0&&b2_pt>0 && tauTau && HLT_ditau && SS_Iso && res2b")
    .Define("weight_puUp", "weight_total*weight_puUp_rel")
    .Sum("weight_puUp")
    .GetValue()
)
error_weight_puUp = math.sqrt(
    anaTuple_df.Filter("tauTau && HLT_ditau && SS_Iso && res2b")
    .Define(
        "weight_puUp_squared",
        "weight_total*weight_puUp_rel*weight_total*weight_puUp_rel",
    )
    .Sum("weight_puUp_squared")
    .GetValue()
)
print(f"""the sum of weight_total*weight_puUp_rel is {sum_weight_puUp}""")
print(
    f"""the difference w.r.t. weight_total sum is of weight_total*weight_puUp_rel is {abs(sum_weight_puUp-sum_weight_total)*100/sum_weight_total}"""
)
print(f"""the error of weight_total*weight_puUp_rel is {error_weight_puUp}""")

sum_weight_puDown = (
    anaTuple_df.Filter("b1_pt>0&&b2_pt>0 && tauTau && HLT_ditau && SS_Iso && res2b")
    .Define("weight_puDown", "weight_total*weight_puDown_rel")
    .Sum("weight_puDown")
    .GetValue()
)
error_weight_puDown = math.sqrt(
    anaTuple_df.Filter("tauTau && HLT_ditau && SS_Iso && res2b")
    .Define(
        "weight_puDown_squared",
        "weight_total*weight_puDown_rel*weight_total*weight_puDown_rel",
    )
    .Sum("weight_puDown_squared")
    .GetValue()
)
print(f"""the sum of weight_total*weight_puDown_rel is {sum_weight_puDown}""")
print(
    f"""the difference w.r.t. weight_total sum is of weight_total*weight_puDown_rel is {abs(sum_weight_puDown-sum_weight_total)*100/sum_weight_total}"""
)
print(f"""the error of weight_total*weight_puDown_rel is {error_weight_puDown}""")
