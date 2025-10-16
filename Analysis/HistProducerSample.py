import ROOT
import sys
import os
import math
import shutil
from FLAF.RunKit.run_tools import ps_call

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

# HistProducerSample.py --histDir my/hist/dir --outDir my/out/dir --hists m_tautau,tau1_pt --file-name-pattern 'nano_{id}.root' --file-ids '0-100'

import FLAF.Common.Utilities as Utilities
from FLAF.Common.HistHelper import *

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile", nargs="+", type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--test", required=False, type=bool, default=False)
    parser.add_argument("--remove-files", required=False, type=bool, default=False)

    args = parser.parse_args()

    # 1 list files :
    all_files = [fileName for fileName in args.inputFile]

    hadd_str = f"hadd -f209 -n 0 {args.outFile} "
    hadd_str += " ".join(f for f in all_files)
    if len(all_files) > 1:
        ps_call([hadd_str], True)
    else:
        shutil.copy(all_files[0], args.outFile)
    if os.path.exists(args.outFile) and args.remove_files:
        for histFile in all_files:
            if args.test:
                print(histFile)
            if histFile == args.outFile:
                continue
            os.remove(histFile)
