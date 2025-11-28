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
    parser.add_argument("inputFile", nargs="+", type=str)
    parser.add_argument("--outFile", required=True, type=str)

    args = parser.parse_args()

    # this code will work on data: in that case all per-file jsons will contain ones
    # so it'll run the calculation that results in all ratios equals one

    # 1 list files :
    all_files = [fileName for fileName in args.inputFile]

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

    # merge all JER(Up|Down)_* -> JER(Up|Down), etc
    # Central does not have up/down, so it's an edge case
    base_sources = [
        "_".join(s.split("_")[:-1]) if s != "Central" else s
        for s in dataset_btag_weight_dict.keys()
    ]
    lepton_categories = ["e", "mu", "eE", "eMu", "muMu"]
    joint_dict = {src: {lc: {} for lc in lepton_categories} for src in base_sources}

    for unc_src, unc_dict in dataset_btag_weight_dict.items():
        for lep_cat, lep_cat_dict in unc_dict.items():
            matches = list({bs for bs in base_sources if bs in unc_src})
            if len(matches) != 1:
                raise RuntimeError(f"Unexpected number of matching base sources for unc_src={unc_src}. Expected 1, got {len(matches)}: {matches}")
            base_src = matches[0]

            for key, val in lep_cat_dict.items():
                if key not in joint_dict[base_src][lep_cat]:
                    joint_dict[base_src][lep_cat][key] = val
                else:
                    joint_dict[base_src][lep_cat][key] += val

    # calculate ratio of integrals
    # structure: {category:{ratio_ncentralJet_k for k in range(2, 9)} for category in [e, mu, eE, eMu, muMu]}
    integral_ratio_dict = {}
    for unc_src, unc_src_dict in joint_dict.items():
        if unc_src not in integral_ratio_dict.keys():
            integral_ratio_dict[unc_src] = {}

        this_unc_dict = integral_ratio_dict[unc_src]
        for lep_cat, lep_cat_dict in unc_src_dict.items():
            multiplicities = list(
                np.unique([int(key.split("_")[-1]) for key in lep_cat_dict.keys()])
            )

            if lep_cat not in this_unc_dict.keys():
                this_unc_dict[lep_cat] = {}

            this_lep_cat_dict = this_unc_dict[lep_cat]
            weights_before = np.array(
                [lep_cat_dict[f"weight_noBtag_ncentralJet_{m}"] for m in multiplicities]
            )
            weights_after = np.array(
                [lep_cat_dict[f"weight_total_ncentralJet_{m}"] for m in multiplicities]
            )
            zero_weight_mask = np.logical_or(
                weights_after == 0.0, weights_before == 0.0
            )
            integral_ratios = np.where(
                zero_weight_mask, 1.0, weights_before / weights_after
            )
            this_lep_cat_dict = {
                f"ratio_ncentralJet_{m}": integral_ratios[idx]
                for idx, m in enumerate(multiplicities)
            }
            integral_ratio_dict[unc_src][lep_cat] = this_lep_cat_dict

    jsonName = args.outFile
    with open(jsonName, "w") as fp:
        json.dump(integral_ratio_dict, fp, indent=4)
