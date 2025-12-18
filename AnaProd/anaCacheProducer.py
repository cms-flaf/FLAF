import datetime
import json
import os
import sys
import yaml
import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from FLAF.Common.Utilities import create_processor_instances
from Corrections.Corrections import Corrections
from Corrections.CorrectionsCore import central, getScales, getSystName
from Corrections.pu import puWeightProducer


class DefaultAnaCacheProcessor:
    def onAnaCache_initializeDenomEntry(self):
        return []

    def onAnaCache_updateDenomEntry(
        self, entry, df, output_branch_name, weights_to_apply
    ):
        weight_formula = (
            "*".join(weights_to_apply) if len(weights_to_apply) > 0 else "1.0"
        )
        df = df.Define(output_branch_name, weight_formula)
        entry.append(df.Sum(output_branch_name))
        return entry

    def onAnaCache_materializeDenomEntry(self, entry):
        return [x.GetValue() if type(x) != float else x for x in entry]

    def onAnaCache_finalizeDenomEntry(self, entry):
        return sum(entry)

    def onAnaCache_combineAnaCaches(self, entries):
        return sum(entries)

    def onAnaTuple_defineCrossSection(
        self, df, crossSectionBranch, xs_db, dataset_name, dataset_entry
    ):
        xs_name = dataset_entry["crossSection"]
        xs_value = xs_db.getValue(xs_name)
        return df.Define(crossSectionBranch, str(xs_value))

    def onAnaTuple_defineDenominator(
        self,
        df,
        denomBranch,
        processor_name,
        dataset_name,
        source_name,
        scale_name,
        ana_caches,
    ):
        ana_cache = ana_caches[dataset_name]
        denom_value = ana_cache["denominator"][source_name][scale_name][processor_name]
        return df.Define(denomBranch, str(denom_value))


def createAnaCacheProcessorInstances(
    global_params, processor_entries, stage="AnaCache", verbose=0
):
    processor_instances = {"default": DefaultAnaCacheProcessor()}
    processor_instances.update(
        create_processor_instances(
            global_params, processor_entries, stage, verbose=verbose
        )
    )
    return processor_instances


def computeAnaCache(
    file_lists,
    global_params,
    use_genWeight_sign_only=True,
    processor_entries=None,
    event_range=None,
    verbose=0,
):
    """
    Compute analysis cache. Dynamically loads and executes processors
    (e.g. Stitcher, PUWeight, etc.) defined in YAML.
    """

    start_time = datetime.datetime.now()
    Corrections.initializeGlobal(
        global_params=global_params,
        stage="AnaCache",
        isData=False,
        load_corr_lib=True,
        dataset_name=None,
        dataset_cfg=None,
        process_name=None,
        process_cfg=None,
        processors=None,
        trigger_class=None,
    )

    processor_instances = createAnaCacheProcessorInstances(
        global_params, processor_entries, verbose=verbose
    )

    genWeight_def = (
        "std::copysign<double>(1., genWeight)"
        if use_genWeight_sign_only
        else "double(genWeight)"
    )
    sources = [central]
    if "pu" in Corrections.getGlobal().to_apply:
        sources += puWeightProducer.uncSource

    denominator = {}
    for source in sources:
        denominator[source] = {}
        for scale in getScales(source):
            denominator[source][scale] = {}
            for p_name, p_instance in processor_instances.items():
                denominator[source][scale][
                    p_name
                ] = p_instance.onAnaCache_initializeDenomEntry()

    for tree, file_list in file_lists.items():
        df = ROOT.RDataFrame(tree, file_list)
        if event_range is not None:
            df = df.Range(event_range)
        df = df.Define("genWeightD", genWeight_def)
        if "pu" in Corrections.getGlobal().to_apply:
            df = Corrections.getGlobal().pu.getWeight(df)
        for source in sources:
            for scale in getScales(source):
                syst_name = getSystName(source, scale)
                weights_to_apply = ["genWeightD"]
                if "pu" in Corrections.getGlobal().to_apply:
                    weights_to_apply.append(f"puWeight_{scale}")
                for p_name, p_instance in processor_instances.items():
                    output_branch_name = f"weight_denom_{p_name}_{syst_name}"
                    denominator[source][scale][p_name] = (
                        p_instance.onAnaCache_updateDenomEntry(
                            denominator[source][scale][p_name],
                            df,
                            output_branch_name,
                            weights_to_apply,
                        )
                    )

        for source in sources:
            for scale in getScales(source):
                for p_name, p_instance in processor_instances.items():
                    denominator[source][scale][p_name] = (
                        p_instance.onAnaCache_materializeDenomEntry(
                            denominator[source][scale][p_name]
                        )
                    )

    for source in sources:
        for scale in getScales(source):
            for p_name, p_instance in processor_instances.items():
                denominator[source][scale][p_name] = (
                    p_instance.onAnaCache_finalizeDenomEntry(
                        denominator[source][scale][p_name]
                    )
                )

    end_time = datetime.datetime.now()
    anaCache = {
        "denominator": denominator,
        "runtime": (end_time - start_time).total_seconds(),
    }
    return anaCache


def combineAnaCaches(anaCaches, processors):
    """
    Combine multiple anaCaches into one.
    Merges denominators, runtimes, and any processor-provided sections (like DY_stitching).
    """
    if len(anaCaches) == 0:
        raise RuntimeError("addAnaCaches: no anaCaches provided")
    denominator = {}
    anaCache_processors = set()
    for anaCache in anaCaches:
        for source, source_entry in anaCache["denominator"].items():
            if source not in denominator:
                denominator[source] = {}
            for scale in getScales(source):
                if scale not in denominator[source]:
                    denominator[source][scale] = {}
                anaCache_processors.update(source_entry.get(scale, {}).keys())
    for source in denominator.keys():
        for scale in getScales(source):
            for processor in anaCache_processors:
                if processor not in processors:
                    raise RuntimeError(
                        f"combineAnaCaches: processor {processor} not provided for combining anaCaches"
                    )
                entries = []
                for anaCache in anaCaches:
                    if (
                        source in anaCache["denominator"]
                        and scale in anaCache["denominator"][source]
                        and processor in anaCache["denominator"][source][scale]
                    ):
                        entries.append(
                            anaCache["denominator"][source][scale][processor]
                        )
                    else:
                        raise RuntimeError(
                            f"combineAnaCaches: missing entry for {source}/{scale}/{processor} in one of the caches"
                        )
                denominator[source][scale][processor] = processors[
                    processor
                ].onAnaCache_combineAnaCaches(entries)

    runtime = sum(anaCache["runtime"] for anaCache in anaCaches)
    anaCacheSum = {
        "denominator": denominator,
        "runtime": runtime,
    }
    return anaCacheSum


def create_filelists(input_files, keys=["Events", "EventsNotSelected"]):
    file_lists = {key: [] for key in keys}
    for input_file in input_files:
        with ROOT.TFile.Open(input_file) as tmp_file:
            for key in keys:
                if key in tmp_file.GetListOfKeys():
                    file_lists[key].append(input_file)
    file_lists = {
        key: file_list for key, file_list in file_lists.items() if len(file_list) > 0
    }
    return file_lists


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", required=True, type=str)
    parser.add_argument("--output", required=False, default=None, type=str)
    parser.add_argument("--global-params", required=True, type=str)
    parser.add_argument("--processors", required=False, type=str, default=None)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    from FLAF.Common.Utilities import DeserializeObjectFromString

    input_files = args.input_files.split(",")
    global_params = DeserializeObjectFromString(args.global_params)
    processors_entries = (
        DeserializeObjectFromString(args.processors) if args.processors else None
    )

    file_lists = create_filelists(input_files)
    anaCache = computeAnaCache(
        file_lists,
        global_params,
        processor_entries=processors_entries,
        event_range=args.n_events,
        verbose=args.verbose,
    )
    if args.verbose > 0:
        print(json.dumps(anaCache))

    if args.output is not None:
        if os.path.exists(args.output):
            print(f"{args.output} already exist, removing it", file=sys.stderr)
            os.remove(args.output)
        with open(args.output, "w") as file:
            yaml.dump(anaCache, file)
