import os
import json


def CreateMergePlan(setup, local_inputs, n_events_per_file, is_data):
    """Create a merge plan for either data or MC.

    Args:
        setup (Setup): FLAF setup object
        local_inputs (list[str]): list of input report file paths
        n_events_per_file (int): an aproximate number of events per output file. The goal is to have output files with a number of events close to this value, so it is not guaranteed that the actual number of events is less or equal to this value.
        is_data (bool): data or MC

    Returns:
        dict: merge plan and combined reports
    """

    combined_reports = {}
    for report in local_inputs:
        with open(report, "r") as file:
            data = json.load(file)
        key = os.path.join(data["dataset_name"], data["anaTuple_file_name"])
        is_valid = data.get("valid", True)
        if not is_valid:
            print(f"{key}: is marked as invalid, skipping", file=os.sys.stderr)
            continue
        if key in combined_reports:
            raise ValueError(f"Duplicate report for file {key}")
        combined_reports[key] = data

    if is_data:
        plan = CreateDataMergePlan(setup, combined_reports, n_events_per_file)
    else:
        plan = CreateMCMergePlan(combined_reports, n_events_per_file)
    return {"plan": plan, "reports": combined_reports}


def CreateMCMergePlan(input_reports, n_events_per_file, oversize_tolerance=1.2):
    assert n_events_per_file > 0
    input_files = {}
    for file_path, data in input_reports.items():
        input_files[file_path] = data["n_events"]

    merge_plan = []
    while len(input_files) > 0:
        file_idx = len(merge_plan)
        out_file_name = f"anaTuple_{file_idx}.root"
        merge = {
            "inputs": [],
            "outputs": [out_file_name],
            "n_events": 0,
        }
        for file, n_events in input_files.items():
            n_old = merge["n_events"]
            n_new = n_old + n_events
            delta_old = abs(n_old - n_events_per_file)
            delta_new = abs(n_new - n_events_per_file)
            if (
                n_old == 0
                or n_events == 0
                or (
                    delta_new <= delta_old
                    and n_new <= n_events_per_file * oversize_tolerance
                )
            ):
                merge["inputs"].append(file)
                merge["n_events"] += n_events

        for input_file in merge["inputs"]:
            del input_files[input_file]
        merge_plan.append(merge)

    return merge_plan


def CreateDataMergePlan(setup, input_reports, n_events_per_file):
    assert n_events_per_file > 0
    input_files = {}
    for file_path, data in input_reports.items():
        dataset_name = data["dataset_name"]
        dataset = setup.datasets[dataset_name]
        eraLetter = dataset["eraLetter"]
        eraVersion = dataset.get("eraVersion", "")
        output_label = f"{eraLetter}{eraVersion}"
        if output_label not in input_files:
            input_files[output_label] = {"files": [], "n_events": 0}
        input_files[output_label]["files"].append(file_path)
        input_files[output_label]["n_events"] += data["n_events"]

    merge_plan = []
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
        merge_plan.append(entry)
    return merge_plan
