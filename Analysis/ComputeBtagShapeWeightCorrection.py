import ROOT
import sys
import numpy as np
import os
import math
import shutil
import json
from FLAF.RunKit.run_tools import ps_call
import re

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFiles", nargs="+", type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument(
        "--jetMultiplicities", nargs="+", default=[2, 3, 4, 5, 6, 7, 8], type=int
    )

    args = parser.parse_args()

    # this code will work on data: in that case all per-file jsons will contain ones
    # so it'll run the calculation that results in all ratios equals one

    # 1 list files :
    all_files = [fileName for fileName in args.inputFiles]

    dataset_btag_weight_dict = {}
    for this_json in all_files:
        with open(this_json, "r") as file:
            this_json_dict = json.load(file)

            for unc_src, unc_dict in this_json_dict.items():
                if unc_src not in dataset_btag_weight_dict.keys():
                    dataset_btag_weight_dict[unc_src] = {}

                this_unc_dict = dataset_btag_weight_dict[unc_src]
                for lep_cat, lep_cat_dict in unc_dict.items():
                    if lep_cat not in this_unc_dict.keys():
                        this_unc_dict[lep_cat] = {}

                    for key, val in lep_cat_dict.items():
                        if key not in this_unc_dict[lep_cat]:
                            this_unc_dict[lep_cat][key] = val
                        else:
                            this_unc_dict[lep_cat][key] += val

    # calculate ratio of integrals
    # structure: {category:{ratio_ncentralJet_k for k in range(2, 9)} for category in [e, mu, eE, eMu, muMu]}
    multiplicities = args.jetMultiplicities
    integral_ratio_dict = {}
    for unc_src, unc_src_dict in dataset_btag_weight_dict.items():
        if unc_src not in integral_ratio_dict.keys():
            integral_ratio_dict[unc_src] = {}

        this_unc_dict = integral_ratio_dict[unc_src]
        for lep_cat, lep_cat_dict in unc_src_dict.items():
            if lep_cat not in this_unc_dict.keys():
                this_unc_dict[lep_cat] = {}

            this_lep_cat_dict = this_unc_dict[lep_cat]
            weights_before = np.array(
                [lep_cat_dict[f"weight_noBtag_ncentralJet_{m}"] for m in multiplicities]
            )
            weights_after = np.array(
                [lep_cat_dict[f"weight_total_ncentralJet_{m}"] for m in multiplicities]
            )
            zero_division_mask = np.isclose(weights_after, 0)
            integral_ratios = np.ones_like(weights_before, dtype=float)
            np.divide(
                weights_before,
                weights_after,
                out=integral_ratios,
                where=~zero_division_mask,
            )
            this_lep_cat_dict = {
                f"ratio_ncentralJet_{m}": integral_ratios[idx]
                for idx, m in enumerate(multiplicities)
            }
            integral_ratio_dict[unc_src][lep_cat] = this_lep_cat_dict

    jsonName = args.outFile
    with open(jsonName, "w") as fp:
        json.dump(integral_ratio_dict, fp, indent=4)