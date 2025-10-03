import ROOT
import sys
import os
import math
import shutil
import json
import time
from FLAF.RunKit.run_tools import ps_call

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
from FLAF.Common.HistHelper import *
from FLAF.Analysis.hh_bbtautau import *


processes = [
    "GluGluToRadionToHHTo2B2Tau_M-250",
    "GluGluToRadionToHHTo2B2Tau_M-260",
    "GluGluToRadionToHHTo2B2Tau_M-270",
    "GluGluToRadionToHHTo2B2Tau_M-280",
    "GluGluToRadionToHHTo2B2Tau_M-300",
    "GluGluToRadionToHHTo2B2Tau_M-320",
    "GluGluToRadionToHHTo2B2Tau_M-350",
    "GluGluToRadionToHHTo2B2Tau_M-400",
    "GluGluToRadionToHHTo2B2Tau_M-450",
    "GluGluToRadionToHHTo2B2Tau_M-500",
    "GluGluToRadionToHHTo2B2Tau_M-550",
    "GluGluToRadionToHHTo2B2Tau_M-600",
    "GluGluToRadionToHHTo2B2Tau_M-650",
    "GluGluToRadionToHHTo2B2Tau_M-700",
    "GluGluToRadionToHHTo2B2Tau_M-750",
    "GluGluToRadionToHHTo2B2Tau_M-800",
    "GluGluToRadionToHHTo2B2Tau_M-850",
    "GluGluToRadionToHHTo2B2Tau_M-900",
    "GluGluToRadionToHHTo2B2Tau_M-1000",
    "GluGluToRadionToHHTo2B2Tau_M-1250",
    "GluGluToRadionToHHTo2B2Tau_M-1500",
    "GluGluToRadionToHHTo2B2Tau_M-1750",
    "GluGluToRadionToHHTo2B2Tau_M-2000",
    "GluGluToRadionToHHTo2B2Tau_M-2500",
    "GluGluToRadionToHHTo2B2Tau_M-3000",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-250",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-260",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-270",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-280",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-300",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-320",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-350",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-400",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-450",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-500",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-550",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-600",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-650",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-700",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-750",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-800",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-850",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-900",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-1000",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-1250",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-1500",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-1750",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-2000",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-2500",
    "GluGluToBulkGravitonToHHTo2B2Tau_M-3000",
    "data",
    "QCD",
    "W",
    "TT",
    "DY",
    "EWK_ZTo2L",
    "EWK_ZTo2Nu",
    "EWK_WplusToLNu",
    "EWK_WminusToLNu",
    "ggHToZZTo2L2Q",
    "GluGluHToTauTau_M125",
    "VBFHToTauTau_M125",
    "GluGluHToWWTo2L2Nu_M125",
    "VBFHToWWTo2L2Nu_M125",
    "WplusHToTauTau_M125",
    "WminusHToTauTau_M125",
    "ZHToTauTau_M125",
    "ZH_Hbb_Zll",
    "ZH_Hbb_Zqq",
    "HWplusJ_HToWW_M125",
    "HWminusJ_HToWW_M125",
    "HZJ_HToWW_M125",
    "WW",
    "WZ",
    "ZZ",
    "WWW_4F",
    "WWZ_4F",
    "WZZ",
    "ZZZ",
    "ST_t-channel_antitop_4f_InclusiveDecays",
    "ST_t-channel_top_4f_InclusiveDecays",
    "ST_tW_antitop_5f_InclusiveDecays",
    "ST_tW_top_5f_InclusiveDecays",
    "TTWW",
    "TTWH",
    "TTZH",
    "TTZZ",
    "TTWZ",
    "TTTT",
    "TTTW",
    "TTTJ",
    "TTGG",
    "TTGJets",
    "TT4b",
    "ttHTobb_M125",
    "ttHToTauTau_M125",
]


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--inFile", required=True)
    # parser.add_argument('--sigCfg', required=True)
    # parser.add_argument('--bckgCfg', required=True)
    # parser.add_argument('--remove-files', required=False, type=bool, default=False)
    args = parser.parse_args()

# print(all_histnames.keys())
inFile = ROOT.TFile.Open(args.inFile, "READ")

# with open(args.sigConfig, 'r') as f:
#     sig_cfg_dict = yaml.safe_load(f)

# with open(args.bckgConfig, 'r') as f:
#     bckg_cfg_dict = yaml.safe_load(f)
# all_samples = sig_cfg_dict.keys()+bckg_cfg_dict.keys()
# channels =[str(key.GetName()) for key in inFile.GetListOfKeys()]
channel = "eTau"
qcdReg = "OS_Iso"
cat = "res1b_cat3_masswindow"

dir_0 = inFile.Get(channel)
dir_1 = dir_0.Get(qcdReg)
dir_2 = dir_1.Get(cat)
keys2 = [str(key.GetName()) for key in dir_2.GetListOfKeys()]
for key_hist in dir_2.GetListOfKeys():
    key_name = key_hist.GetName()
    if key_name.startswith("WW"):
        print(key_name)

inFile.Close()


# /eos/user/v/vdamante/HH_bbtautau_resonant_Run2/histograms/v12_deepTau2p1_HTT_SC/Run2_2016/merged_SR/kinFit_m/tmp/all_histograms_kinFit_m_hadded.root
