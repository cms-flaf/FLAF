import ROOT
import sys
import os
import math
import shutil
import time

ROOT.EnableThreadSafety()

from FLAF.RunKit.run_tools import ps_call

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
from FLAF.Analysis.HistHelper import *
from Analysis.hh_bbtautau import *

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--outFileName", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--deepTauVersion", required=False, type=str, default="v2p1")
    parser.add_argument("--globalConfig", required=True, type=str)
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--region", required=False, type=str, default="SR")
    args = parser.parse_args()

    startTime = time.time()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/KinFitNamespace.h"')
    ROOT.gInterpreter.Declare(f'#include "include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')
    ROOT.gInterpreter.Declare(f'#include "include/pnetSF.h"')
    ROOT.gROOT.ProcessLine('#include "include/AnalysisTools.h"')
    # if not os.path.isdir(args.outDir):
    #    os.makedirs(args.outDir)

    global_cfg_dict = {}
    with open(args.globalConfig, "r") as f:
        global_cfg_dict = yaml.safe_load(f)

    if args.region == "DYCR":
        global_cfg_dict["channels_to_consider"] = ["muMu", "eE"]
        print(f"""considering {global_cfg_dict["channels_to_consider"]}""")

    isData = args.dataset == "data"

    # central hist definition
    create_new_hist = False
    key_not_exist = False
    df_empty = False
    inFile_root = ROOT.TFile.Open(args.inFile, "READ")
    inFile_keys = [k.GetName() for k in inFile_root.GetListOfKeys()]
    if "Events" not in inFile_keys:
        key_not_exist = True
    inFile_root.Close()
    if (
        not key_not_exist
        and ROOT.RDataFrame("Events", args.inFile).Count().GetValue() == 0
    ):
        df_empty = True

    create_new_hist = key_not_exist or df_empty

    if not create_new_hist:
        dfWrapped_central = DataFrameBuilderForHistograms(
            ROOT.RDataFrame("Events", args.inFile),
            global_cfg_dict,
            args.period,
            deepTauVersion=args.deepTauVersion,
            region=args.region,
            isData=isData,
        )
        all_dataframes = {}
        all_histograms = {}
        new_df = PrepareDfForHistograms(dfWrapped_central).df
        new_cols = []
        for col in new_df.GetColumnNames():
            if "p4" in col:
                continue
            new_cols.append(col)
        # print(new_cols)
        new_df.Snapshot("Events", args.outFileName, new_cols)
        # outfile.Close()

    # finally:
    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
