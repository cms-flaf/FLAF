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
            if merge["n_events"] + n_events <= n_events_per_file:
                merge["inputs"].append(file)
                merge["n_events"] += n_events

        for input_file in merge["inputs"]:
            del input_files[input_file]
        merge_strategy.append(merge)

    return merge_strategy


def CreateDataMergeStrategy(setup, input_reports):
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

    merge_strategy = []
    for output_file, inputs in input_files.items():
        merge_strategy.append(
            {
                "inputs": inputs["files"],
                "outputs": [output_file],
                "n_events": inputs["n_events"],
            }
        )
    return merge_strategy
