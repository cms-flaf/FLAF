import sys
import os
import json
import argparse
import numpy as np

from FLAF.Common.Utilities import DeclareHeader
from FLAF.Common.Setup import Setup

def aggregate_caches(
    *,
    setup,
    producer_name,
    inputFiles,
    outFile,
):
    producer_cfg = producer_config = setup.global_params["payload_producers"][producer_name]
    save_as = producer_cfg.get("save_as")

    if save_as == "json":
        dataset_btag_weight_dict = {}
        for this_json in inputFiles:
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
        multiplicities = producer_cfg["jet_multiplicities"]
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
        
        with open(outFile, "w") as fp:
            json.dump(integral_ratio_dict, fp, indent=4)
    else:
        raise NotImplementedError(f"Aggregating caches in {save_as} format is not supported.")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFiles", nargs="+", type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--producer", required=True, type=str)
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    args = parser.parse_args()

    ana_path = os.environ["ANALYSIS_PATH"]
    sys.path.append(ana_path)
    headers = ["FLAF/include/HistHelper.h", "FLAF/include/Utilities.h"]
    for header in headers:
        DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)

    setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], args.period, args.LAWrunVersion)

    aggregate_caches(
        setup=setup,
        producer_name=args.producer,
        inputFiles=args.inputFiles,
        outFile=args.outFile,
    )