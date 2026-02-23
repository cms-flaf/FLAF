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
    producer_cfg = producer_config = setup.global_params["payload_producers"][
        producer_name
    ]
    save_as = producer_cfg.get("save_as")

    if save_as == "json":
        aggregated_dict = {}
        for this_json in inputFiles:
            with open(this_json, "r") as file:
                this_json_dict = json.load(file)

            for outer_key, inner_dict in this_json_dict.items():
                if outer_key not in aggregated_dict.keys():
                    aggregated_dict[outer_key] = inner_dict
                else:
                    for inner_key, inner_value in inner_dict.items():
                        aggregated_dict[outer_key][inner_key] += inner_value

        result_dict = {}
        bins = list(producer_cfg["bins"].keys())
        prefix = "weight_noBtag_"
        for outer_key, inner_dict in aggregated_dict.items():
            print(f"Computing btag shape norm correction for {outer_key}")
            noBtag_syst_keys = [key for key in inner_dict.keys() if "noBtag" in key]
            systs = {
                k[len(prefix) :].rsplit("_")[0]
                for k in noBtag_syst_keys
                if k.startswith(prefix)
            }
            result_dict[outer_key] = {}
            for bin_name in bins:
                weight_after = inner_dict[f"weight_total_{bin_name}"]
                for syst in systs:
                    weight_before = inner_dict[f"weight_noBtag_{syst}_{bin_name}"]
                    result_dict[outer_key][f"norm_{syst}_{bin_name}"] = (
                        weight_before / weight_after if weight_after != 0 else 1
                    )

        with open(outFile, "w") as fp:
            json.dump(result_dict, fp, indent=4)

    else:
        raise NotImplementedError(
            f"Aggregating caches in {save_as} format is not supported."
        )


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

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"], args.period, args.LAWrunVersion
    )

    aggregate_caches(
        setup=setup,
        producer_name=args.producer,
        inputFiles=args.inputFiles,
        outFile=args.outFile,
    )
