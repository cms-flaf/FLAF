import os
import json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile", nargs="+", type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--test", required=False, type=bool, default=False)
    parser.add_argument("--remove-files", required=False, type=bool, default=False)
    parser.add_argument("--nEventsPerFile", required=False, type=int, default=100_000)
    parser.add_argument("--isData", required=False, type=bool, default=False)
    parser.add_argument("--lumi", required=False, type=float, default=None)
    parser.add_argument(
        "--nPbPerFile", required=False, type=int, default=1_000
    )  # 1fb-1 per split data file

    args = parser.parse_args()

    # 1 list files :
    all_files = [fileName for fileName in args.inputFile]

    nEventsCounter = 0
    hadd_dict = {}
    nFileCounter = 0
    hadd_dict["merge_strategy"] = []
    input_file_list = []
    output_file_list = []
    # hadd_dict[f'anaTuple_{nFileCounter}.root'] = []
    nDataEventsCounter = 0
    for this_json in all_files:
        with open(this_json, "r") as file:
            data = json.load(file)

            nEvents = data["n_events"]
            nEventsCounter += nEvents
            dataset_name = data["dataset_name"]
            file_name = os.path.join(dataset_name, data["nano_file_name"])
            input_file_list.append(file_name)

            if nEventsCounter > args.nEventsPerFile and not args.isData:
                output_file_list.append(f"anaTuple_{nFileCounter}.root")
                hadd_dict["merge_strategy"].append(
                    {
                        "inputs": input_file_list,
                        "outputs": output_file_list,
                        "n_events": nEventsCounter,
                    }
                )
                nEventsCounter = 0
                nFileCounter += 1
                output_file_list = []
                input_file_list = []

            if args.isData:
                nDataEventsCounter += nEvents

    # Append whatever is leftover
    if len(input_file_list) > 0 and not args.isData:
        # Had leftover files, so we need to add them to the output
        output_file_list.append(f"anaTuple_{nFileCounter}.root")
        nFileCounter += 1
        hadd_dict["merge_strategy"].append(
            {
                "inputs": input_file_list,
                "outputs": output_file_list,
                "n_events": nEventsCounter,
            }
        )
        input_file_list = []
        output_file_list = []

    # If data, then just do the lumi look-up and calculate the nFiles for splitting
    if args.isData:
        if hasattr(args, "lumi") and hasattr(args, "nPbPerFile"):
            print("Inside the final data part")
            nPbPerFile = args.nPbPerFile
            lumi = args.lumi
            nFiles = (
                int(lumi / nPbPerFile) + 1
            )  # Need to add 1 since int will floor the division
            for nFileCounter in range(nFiles):
                output_file_list.append(f"anaTuple_{nFileCounter}.root")
            hadd_dict["merge_strategy"].append(
                {
                    "inputs": input_file_list,
                    "outputs": output_file_list,
                    "n_events": nDataEventsCounter,
                }
            )
        else:
            raise ValueError(
                "For data, you need to provide --lumi and --nPbPerFile arguments."
            )

    jsonName = args.outFile
    with open(jsonName, "w") as fp:
        json.dump(hadd_dict, fp)
