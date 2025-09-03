import datetime
import json
import os
import sys
import yaml
import ROOT


if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


def computeAnaCache(file_lists, global_params, generator, range=None):
    from Corrections.Corrections import Corrections
    from Corrections.CorrectionsCore import central, getScales, getSystName
    from Corrections.pu import puWeightProducer

    start_time = datetime.datetime.now()
    Corrections.initializeGlobal(
        global_params, sample_name=None, isData=False, load_corr_lib=True
    )
    anaCache = {"denominator": {}}
    sources = [central]
    if "pu" in Corrections.getGlobal().to_apply:
        sources += puWeightProducer.uncSource

    for tree, file_list in file_lists.items():
        df = ROOT.RDataFrame(tree, file_list)
        if range is not None:
            df = df.Range(range)
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
    end_time = datetime.datetime.now()
    anaCache["runtime"] = (end_time - start_time).total_seconds()
    return anaCache


def create_filelists(input_files, keys=["Events", "EventsNotSelected"]):
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
        anaCacheSum["runtime"] += anaCache["runtime"]
    return anaCacheSum


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", required=True, type=str)
    parser.add_argument("--output", required=False, default=None, type=str)
    parser.add_argument("--global-params", required=True, type=str)
    parser.add_argument("--generator-name", required=True, type=str)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    from FLAF.Common.Utilities import DeserializeObjectFromString

    input_files = args.input_files.split(",")
    global_params = DeserializeObjectFromString(args.global_params)

    file_lists = create_filelists(input_files)
    anaCache = computeAnaCache(
        file_lists, global_params, args.generator_name, range=args.n_events
    )
    if args.verbose > 0:
        print(json.dumps(anaCache))

    if args.output is not None:
        if os.path.exists(args.output):
            print(f"{args.output} already exist, removing it")
            os.remove(args.output)
        with open(args.output, "w") as file:
            yaml.dump(anaCache, file)
