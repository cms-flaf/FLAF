import ROOT

# import statsmodels.stats.proportion as ssp
import os

ROOT.gROOT.ProcessLine(".include " + os.environ["ANALYSIS_PATH"])
hist_file = ROOT.TFile("output/bTagEff.root", "READ")  # take as argument


def GetKeyNames(self, dir=""):
    self.cd(dir)
    return [key.GetName() for key in ROOT.gDirectory.GetListOfKeys()]


ROOT.TFile.GetKeyNames = GetKeyNames
keyList = hist_file.GetKeyNames()
# $print( "\nKeys in file:", keyList)
nums = [
    "jet_pt_eta_0_Loose",
    "jet_pt_eta_0_Medium",
    "jet_pt_eta_0_Tight",
    "jet_pt_eta_4_Loose",
    "jet_pt_eta_4_Medium",
    "jet_pt_eta_4_Tight",
    "jet_pt_eta_5_Loose",
    "jet_pt_eta_5_Medium",
    "jet_pt_eta_5_Tight",
]
denums = ["jet_pt_eta_0", "jet_pt_eta_4", "jet_pt_eta_5"]
i = 0
for key in keyList:
    if i > 1:
        break
    hist = hist_file.Get(key)
    # print(f"""for key {key} the histogram has {hist.GetNbinsX()*hist.GetNbinsY()} bins
    # """)
    for x_bin_number in range(1, hist.GetNbinsX() - 1):
        for y_bin_number in range(1, hist.GetNbinsY() - 1):
            bin_number = hist.GetBin(x_bin_number, y_bin_number)
            bin_content = hist.GetBinContent(bin_number)
            if bin_content == 0:
                print(
                    f""" key = {key},
                    bin_number = {bin_number}, bin_content = {bin_content},
                    bin_center_X = {hist.GetXaxis().GetBinCenter(x_bin_number)},
                    bin_center_y = {hist.GetYaxis().GetBinCenter(y_bin_number)}"""
                )
prefix = "jet_pt_eta"
jet_flavors = [0, 4, 5]
wps = ["Loose", "Medium", "Tight"]
for jet_fl in jet_flavors:
    key_den = f"{prefix}_{jet_fl}"
    hist_den = hist_file.Get(key_den)
    for wp in wps:
        key_num = f"{prefix}_{jet_fl}_{wp}"
        hist_num = hist_file.Get(key_num)
        for x_bin_number in range(1, hist_den.GetNbinsX() - 1):
            for y_bin_number in range(1, hist_den.GetNbinsY() - 1):
                bin_number = hist.GetBin(x_bin_number, y_bin_number)
                num = hist_num.GetBinContent(bin_number)
                den = hist_den.GetBinContent(bin_number)
                c_low = ROOT.TEfficiency.ClopperPearson(den, num, 0.68, False)
                c_up = ROOT.TEfficiency.ClopperPearson(den, num, 0.68, True)
                # print(f"low = {c_low}, up = {c_up}")
                eff = num / den
                rel_cl_size = (c_up - c_low) / (2 * eff)
                if rel_cl_size > 0.01:
                    print(
                        f""" key_num = {key_num},
                          key_den={key_den},
                          bin_number = {bin_number},
                          bin_content_num = {hist_num.GetBinContent(bin_number)},
                          bin_content_den = {hist_den.GetBinContent(bin_number)},
                          bin_center_X = {hist_den.GetXaxis().GetBinCenter(x_bin_number)},
                          bin_center_y = {hist_den.GetYaxis().GetBinCenter(y_bin_number)},
                          c_low = {c_low},
                          c_up = {c_up},
                          eff = {eff}
                          rel_cl_size={rel_cl_size}"""
                    )
