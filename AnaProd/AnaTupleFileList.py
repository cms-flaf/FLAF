import os
import json


def CreateMCMergeStrategy(input_reports, n_events_per_file):
    input_files = {}
    for report in input_reports:
        with open(report, "r") as file:
            data = json.load(file)
        file_path = os.path.join(data["dataset_name"], data["nano_file_name"])
        input_files[file_path] = data["n_events"]

    merge_strategy = []
    while len(input_files) > 0:
        file_idx = len(merge_strategy)
        out_file_name = f"anaTuple_{file_idx}.root"
        merge = {
            "inputs": [],
            "outputs": [out_file_name],
            "n_events": 0,
        }
        for file, n_events in input_files.items():
            # Edge case: if a single file has more events than the limit,
            # include it alone to avoid infinite loop
            if len(merge["inputs"]) == 0 and n_events > n_events_per_file:
                merge["inputs"].append(file)
                merge["n_events"] = n_events
                break
            elif merge["n_events"] + n_events <= n_events_per_file:
                merge["inputs"].append(file)
                merge["n_events"] += n_events

        for input_file in merge["inputs"]:
            del input_files[input_file]
        merge_strategy.append(merge)

    return merge_strategy


def CreateDataMergeStrategy(setup, input_reports, n_events_per_file):
    # First, group files by era
    era_files = {}
    for report in input_reports:
        with open(report, "r") as file:
            data = json.load(file)
        dataset_name = data["dataset_name"]
        file_path = os.path.join(dataset_name, data["nano_file_name"])
        dataset = setup.datasets[dataset_name]
        eraLetter = dataset["eraLetter"]
        eraVersion = dataset.get("eraVersion", "")
        era_key = f"{eraLetter}{eraVersion}"
        if era_key not in era_files:
            era_files[era_key] = []
        era_files[era_key].append({"path": file_path, "n_events": data["n_events"]})

    # Now create merge strategy with event limits per output file
    merge_strategy = []
    for era_key, files in era_files.items():
        file_idx = 0
        remaining_files = files.copy()

        while len(remaining_files) > 0:
            out_file_name = f"anaTuple_{era_key}_{file_idx}.root"
            merge = {
                "inputs": [],
                "outputs": [out_file_name],
                "n_events": 0,
            }

            files_to_remove = []
            for file_info in remaining_files:
                file_path = file_info["path"]
                n_events = file_info["n_events"]

                # Edge case: if a single file has more events than the limit,
                # include it alone to avoid infinite loop
                if len(merge["inputs"]) == 0 and n_events > n_events_per_file:
                    merge["inputs"].append(file_path)
                    merge["n_events"] = n_events
                    files_to_remove.append(file_info)
                    break
                elif merge["n_events"] + n_events <= n_events_per_file:
                    merge["inputs"].append(file_path)
                    merge["n_events"] += n_events
                    files_to_remove.append(file_info)

            for file_info in files_to_remove:
                remaining_files.remove(file_info)

            merge_strategy.append(merge)
            file_idx += 1

    return merge_strategy
