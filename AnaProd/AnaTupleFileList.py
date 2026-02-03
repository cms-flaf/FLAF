import os
import json


def CreateMergeStrategy(setup, local_inputs, n_events_per_file, is_data):
    if is_data:
        return CreateDataMergeStrategy(setup, local_inputs, n_events_per_file)
    else:
        return CreateMCMergeStrategy(local_inputs, n_events_per_file)

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
            if merge["n_events"] + n_events <= n_events_per_file:
                merge["inputs"].append(file)
                merge["n_events"] += n_events
            elif len(merge["inputs"]) == 0 and n_events > n_events_per_file:
                merge["inputs"].append(file)
                merge["n_events"] += n_events
                break

        for input_file in merge["inputs"]:
            del input_files[input_file]
        merge_strategy.append(merge)

    return merge_strategy


def CreateDataMergeStrategy(setup, input_reports, n_events_per_file):
    input_files = {}
    for report in input_reports:
        with open(report, "r") as file:
            data = json.load(file)
        dataset_name = data["dataset_name"]
        file_path = os.path.join(dataset_name, data["nano_file_name"])
        dataset = setup.datasets[dataset_name]
        eraLetter = dataset["eraLetter"]
        eraVersion = dataset.get("eraVersion", "")
        output_label = f"{eraLetter}{eraVersion}"
        if output_label not in input_files:
            input_files[output_label] = {"files": [], "n_events": 0}
        input_files[output_label]["files"].append(file_path)
        input_files[output_label]["n_events"] += data["n_events"]

    merge_strategy = []
    for output_label, inputs in input_files.items():
        n_outputs = round(inputs["n_events"] / n_events_per_file)
        n_outputs = max(1, n_outputs)
        entry = {
                "inputs": inputs["files"],
                "outputs": [],
                "n_events": inputs["n_events"],
            }
        for i in range(n_outputs):
            output_file = f"anaTuple_{output_label}_{i}.root"
            entry["outputs"].append(output_file)
        merge_strategy.append(entry)
    return merge_strategy
