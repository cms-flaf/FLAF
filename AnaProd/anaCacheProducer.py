import datetime
import json
import os
import sys
import yaml
import ROOT
from array import array
import importlib

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


def computeAnaCache(
    file_lists, global_params, generator, processors=None, event_range=None
):
    """
    Compute analysis cache. Dynamically loads and executes processors
    (e.g. Stitcher, PUWeight, etc.) defined in YAML.
    """
    from Corrections.Corrections import Corrections
    from Corrections.CorrectionsCore import central, getScales, getSystName
    from Corrections.pu import puWeightProducer

    start_time = datetime.datetime.now()
    # --- Initialize Corrections ---
    Corrections.initializeGlobal(
        global_params, sample_name=None, isData=False, load_corr_lib=True
    )
    anaCache = {"denominator": {}}
    # --- Dynamic Processor Loading ---
    processor_instances = []
    if processors:
        for p in processors:
            try:
                module = importlib.import_module(p["module"])
                cls = getattr(module, p["class"])
                instance = cls(global_params, config_path=p.get("config", None))
                processor_instances.append(instance)
                print(
                    f"[computeAnaCache] Loaded processor: {p['name']} ({p['class']})",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"[computeAnaCache] Failed to load processor {p}: {e}",
                    file=sys.stderr,
                )
    sources = [central]
    if "pu" in Corrections.getGlobal().to_apply:
        sources += puWeightProducer.uncSource
    for tree, file_list in file_lists.items():
        df = ROOT.RDataFrame(tree, file_list)
        if event_range is not None:
            df = df.Range(event_range)
        df, syst_names = Corrections.getGlobal().getDenominator(df, sources, generator)

        for source in sources:
            if source not in anaCache["denominator"]:
                anaCache["denominator"][source] = {}
            for scale in getScales(source):
                syst_name = getSystName(source, scale)
                anaCache["denominator"][source][scale] = (
                    anaCache["denominator"][source].get(scale, 0.0)
                    + df.Sum(f"weight_denom_{syst_name}").GetValue()
                )
        # --- Run processor hooks (per-processor logic) ---
        for proc in processor_instances:
            if hasattr(proc, "onAnaCacheTask"):
                try:
                    result = proc.onAnaCacheTask(df)
                    if result:
                        # Merge results into anaCache cumulatively
                        anaCache = proc.mergeAnaCache(anaCache, result)
                except Exception as e:
                    print(
                        f"[computeAnaCache] Processor {proc.__class__.__name__} failed: {e}",
                        file=sys.stderr,
                    )

    end_time = datetime.datetime.now()
    anaCache["runtime"] = (end_time - start_time).total_seconds()
    return anaCache


def create_filelists(input_files, keys=["Events", "EventsNotSelected"]):
    # def create_filelists(input_files, keys=["Events"]):

    file_lists = {}
    for input_file in input_files:
        with ROOT.TFile.Open(input_file) as tmp_file:
            for key in keys:
                if key in tmp_file.GetListOfKeys():
                    if key not in file_lists:
                        file_lists[key] = []
                    file_lists[key].append(input_file)
    return file_lists


def addAnaCaches(*anaCaches):
    """
    Combine multiple anaCaches into one.
    Merges denominators, runtimes, and any processor-provided sections (like DY_stitching).
    """
    anaCacheSum = {"denominator": {}, "runtime": 0.0}
    for idx, anaCache in enumerate(anaCaches):
        for source, source_entry in anaCache["denominator"].items():
            if source not in anaCacheSum["denominator"]:
                if idx == 0:
                    anaCacheSum["denominator"][source] = {}
                else:
                    raise RuntimeError(
                        f"addAnaCaches: source {source} not found in first cache"
                    )
            for scale, value in source_entry.items():
                if scale not in anaCacheSum["denominator"][source]:
                    if idx == 0:
                        anaCacheSum["denominator"][source][scale] = 0.0
                    else:
                        raise RuntimeError(
                            f"addAnaCaches: {source}/{scale} not found in first cache"
                        )
                anaCacheSum["denominator"][source][scale] += value
        # --- Merge processor-generated section ---
        for key, value in anaCache.items():
            if key in ["denominator", "runtime"]:
                continue  # handled separately

            if key not in anaCacheSum:
                # Create new section if not present
                anaCacheSum[key] = {}

            # Handle dict-like structures (e.g. DY_stitching, PU_weights, etc.)
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    # If this is an event-yield map (like evts_per_bin or bins dict)
                    if isinstance(subvalue, (int, float)):
                        anaCacheSum[key][subkey] = (
                            anaCacheSum[key].get(subkey, 0.0) + subvalue
                        )
                    elif isinstance(subvalue, dict):
                        # Handle nested dicts (e.g. "bins": {"bin1": {...}})
                        if subkey not in anaCacheSum[key]:
                            anaCacheSum[key][subkey] = {}
                        for k2, v2 in subvalue.items():
                            if isinstance(v2, (int, float)):
                                anaCacheSum[key][subkey][k2] = (
                                    anaCacheSum[key][subkey].get(k2, 0.0) + v2
                                )
                            else:
                                # Non-numeric values (config info, variables, etc.) just copy once
                                if k2 not in anaCacheSum[key][subkey]:
                                    anaCacheSum[key][subkey][k2] = v2
                    else:
                        # For scalar or list metadata
                        if subkey not in anaCacheSum[key]:
                            anaCacheSum[key][subkey] = subvalue

        anaCacheSum["runtime"] += anaCache["runtime"]
    return anaCacheSum


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", required=True, type=str)
    parser.add_argument("--output", required=False, default=None, type=str)
    parser.add_argument("--global-params", required=True, type=str)
    parser.add_argument("--processors", required=True, type=str)
    parser.add_argument("--generator-name", required=True, type=str)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    from FLAF.Common.Utilities import DeserializeObjectFromString

    input_files = args.input_files.split(",")
    global_params = DeserializeObjectFromString(args.global_params)
    processors_param = DeserializeObjectFromString(args.processors)

    file_lists = create_filelists(input_files)
    anaCache = computeAnaCache(
        file_lists,
        global_params,
        args.generator_name,
        processors_param,
        event_range=args.n_events,
    )
    if args.verbose > 0:
        print(json.dumps(anaCache))

    if args.output is not None:
        if os.path.exists(args.output):
            print(f"{args.output} already exist, removing it")
            os.remove(args.output)
        with open(args.output, "w") as file:
            yaml.dump(anaCache, file)
