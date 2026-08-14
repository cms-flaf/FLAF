#!/usr/bin/env python3

import argparse
import json
import os
import sys

DATASET_TASKS = [
    "FLAF.Analysis.tasks.NanoAODProducerTask",
    "FLAF.Analysis.tasks.HistTupleProducerTask",
    "FLAF.Analysis.tasks.HistFromNtupleProducerTask",
]

AVAILABLE_ERAS = [
    "Run3_2022",
    "Run3_2022EE",
    "Run3_2023",
    "Run3_2023BPix",
    "Run3_2024",
    "Run3_2025",
]
ANALYSES = ["HH_bbWW", "HH_bbtautau", "H_mumu"]


def parse_eras(raw_value):
    if not raw_value or raw_value == "ALL":
        return ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]
    seen = []
    for era in raw_value.split():
        if era == "ALL":
            return ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]
        if era in AVAILABLE_ERAS and era not in seen:
            seen.append(era)
    return seen if seen else ["Run3_2022EE"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate GitHub Actions matrix definitions."
    )
    parser.add_argument(
        "--variables", default=None, help="JSON-serialized variables map."
    )
    args = parser.parse_args()

    raw_vars = args.variables or os.environ.get("VARIABLES", "{}")
    try:
        variables = json.loads(raw_vars)
    except Exception as e:
        print(f"Warning: Failed to parse VARIABLES JSON: {e}", file=sys.stderr)
        variables = {}

    analyses_matrix = []
    process_matrix = []
    era_matrix = []
    multi_era_matrix = []

    for ana in ANALYSES:
        if variables.get(f"{ana}_active") != "1":
            continue

        analyses_matrix.append(ana)
        eras = parse_eras(variables.get(f"{ana}_eras", "Run3_2022EE"))
        target_task = variables.get(f"{ana}_task", "FLAF.Analysis.tasks.HistPlotTask")
        task_args = variables.get(f"{ana}_args", "--test 1000")
        procs_str = variables.get(
            f"{ana}_processes", "custom_CI_Signal custom_CI_Background custom_CI_Data"
        )
        procs = procs_str.split() if procs_str else []

        dataset_task = ""
        era_task = ""
        multi_era_task = ""

        if target_task in DATASET_TASKS:
            dataset_task = target_task
        elif "Combine" in target_task or "Inference" in target_task:
            dataset_task = "FLAF.Analysis.tasks.HistFromNtupleProducerTask"
            era_task = "FLAF.Analysis.tasks.HistPlotTask"
            multi_era_task = target_task
        else:
            dataset_task = "FLAF.Analysis.tasks.HistFromNtupleProducerTask"
            era_task = target_task

        for era in eras:
            if dataset_task:
                for proc in procs:
                    proc_safe = proc.replace("/", "_").replace("-", "_")
                    process_matrix.append(
                        {
                            "analysis": ana,
                            "era": era,
                            "process": proc,
                            "proc_safe": proc_safe,
                            "task": dataset_task,
                            "args": f"{task_args} --process {proc}",
                        }
                    )
            if era_task:
                era_matrix.append(
                    {"analysis": ana, "era": era, "task": era_task, "args": task_args}
                )

        if multi_era_task:
            multi_era_matrix.append(
                {
                    "analysis": ana,
                    "task": multi_era_task,
                    "args": task_args,
                    "era": eras[0] if eras else "Run3_2022EE",
                }
            )

    if not analyses_matrix:
        raise ValueError(
            "No active analyses requested; nothing to run. Set '<analysis>_active' to '1' "
            f"for at least one of {', '.join(ANALYSES)} in the triggering variables."
        )

    output_file = os.environ.get("GITHUB_OUTPUT")
    results = {
        "analyses": analyses_matrix,
        "processes": process_matrix,
        "eras": era_matrix,
        "multi_eras": multi_era_matrix,
        "has_processes": len(process_matrix) > 0,
        "has_eras": len(era_matrix) > 0,
        "has_multi_eras": len(multi_era_matrix) > 0,
    }

    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"analyses={json.dumps(analyses_matrix)}\n")
            f.write(f"processes={json.dumps(process_matrix)}\n")
            f.write(f"eras={json.dumps(era_matrix)}\n")
            f.write(f"multi_eras={json.dumps(multi_era_matrix)}\n")
            f.write(f"has_processes={'true' if len(process_matrix) > 0 else 'false'}\n")
            f.write(f"has_eras={'true' if len(era_matrix) > 0 else 'false'}\n")
            f.write(
                f"has_multi_eras={'true' if len(multi_era_matrix) > 0 else 'false'}\n"
            )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
