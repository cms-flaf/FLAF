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
    # First, group files by era (keeps original logic for duplicate filtering)
    input_files = {}
    for report in input_reports:
        with open(report, "r") as file:
            data = json.load(file)
        dataset_name = data["dataset_name"]
        file_path = os.path.join(dataset_name, data["nano_file_name"])
        dataset = setup.datasets[dataset_name]
        eraLetter = dataset["eraLetter"]
        eraVersion = dataset.get("eraVersion", "")
        output_name = f"anaTuple_{eraLetter}{eraVersion}.root"
        if output_name not in input_files:
            input_files[output_name] = {"files": [], "n_events": 0}
        input_files[output_name]["files"].append(file_path)
        input_files[output_name]["n_events"] += data["n_events"]

    # Now split into multiple output files if needed to satisfy n_events_per_file
    merge_strategy = []
    for output_file, inputs in input_files.items():
        # Extract era info from output_file name
        # Format: anaTuple_{eraLetter}{eraVersion}.root
        base_name = output_file.replace("anaTuple_", "").replace(".root", "")
        all_files = inputs["files"]

        # Build list of files with their event counts for splitting
        file_info_list = []
        for file_path in all_files:
            # Find the corresponding report to get n_events
            for report in input_reports:
                with open(report, "r") as file:
                    data = json.load(file)
                report_file_path = os.path.join(
                    data["dataset_name"], data["nano_file_name"]
                )
                if report_file_path == file_path:
                    file_info_list.append(
                        {"path": file_path, "n_events": data["n_events"]}
                    )
                    break

        # Split files into multiple outputs based on n_events_per_file
        out_idx = 0
        remaining_files = file_info_list.copy()

        while len(remaining_files) > 0:
            out_file_name = f"anaTuple_{base_name}_{out_idx}.root"
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
            out_idx += 1

    return merge_strategy
