import ROOT
import os
import sys
import yaml

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.Setup import Setup
from FLAF.Common.TupleHelpers import copyFileContent
from Corrections.Corrections import Corrections
from Corrections.CorrectionsCore import central, getSystName, getScales
from FLAF.AnaProd.anaTupleProducer import DefaultAnaCacheProcessor
from FLAF.Common.Utilities import DeserializeObjectFromString

ROOT.gInterpreter.Declare("""
    struct EventDuplicateFilter {
        using LumiEventMapType = std::map<unsigned int, std::set<unsigned long long>>;
        using EventMapType = std::map<unsigned int, LumiEventMapType>;

        EventDuplicateFilter() :
            eventMap(std::make_shared<EventMapType>()),
            eventMap_mutex(std::make_shared<std::mutex>())
        {
        }

        ROOT::RDF::RNode apply(ROOT::RDF::RNode df, const std::vector<std::string>& columns, const std::string& filter_name) {
            auto df_node = df.Filter([this](unsigned int run, unsigned int lumi, unsigned long long event){
                return saveEvent(run, lumi, event);
            }, columns, filter_name);
            return df_node;
        }

        bool saveEvent(unsigned int run, unsigned int lumi, unsigned long long event) {
            const std::lock_guard<std::mutex> lock(*eventMap_mutex);
            auto& events = (*eventMap)[run][lumi];
            if(events.find(event) != events.end())
                return false;
            events.insert(event);
            return true;
        }

        void clear() {
            const std::lock_guard<std::mutex> lock(*eventMap_mutex);
            eventMap->clear();
        }

        std::shared_ptr<EventMapType> eventMap;
        std::shared_ptr<std::mutex> eventMap_mutex;
    };
    """)


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

    anaCacheSum = {
        "denominator": denominator,
    }
    return anaCacheSum


def getTreeListFromReport(report):
    tree_list = []
    keys = set()
    trees = set()
    for entry in report["trees"]:
        unc_source = entry["unc_source"]
        unc_scale = entry["unc_scale"]
        tree_name = entry["tree_name"]
        key = (unc_source, unc_scale)
        if key in keys:
            raise RuntimeError(f"Duplicate tree entry for uncertainty {key} in report.")
        keys.add(key)
        if tree_name in trees:
            raise RuntimeError(f"Duplicate tree name {tree_name} in report.")
        trees.add(tree_name)
        tree_list.append((unc_source, unc_scale, tree_name))
    return sorted(tree_list)


def getColumns(df):
    all_columns = [str(c) for c in df.GetColumnNames()]
    simple_types = ["Int_t", "UInt_t", "Long64_t", "ULong64_t", "int", "long"]
    column_types = {c: str(df.GetColumnType(c)) for c in all_columns}
    all_columns = sorted(
        all_columns, key=lambda c: (column_types[c] not in simple_types, c)
    )
    return all_columns, column_types


def mergeAnaTuples(
    *,
    setup,
    dataset_name,
    is_data,
    work_dir,
    input_reports,
    input_roots,
    root_outputs,
    compression_algo="LZMA",
    compression_level=9,
):

    snapshot_options = ROOT.RDF.RSnapshotOptions()
    snapshot_options.fOverwriteIfExists = False
    snapshot_options.fLazy = False
    snapshot_options.fMode = "UPDATE"
    snapshot_options.fCompressionAlgorithm = getattr(
        ROOT.ROOT.RCompressionSetting.EAlgorithm, "k" + compression_algo
    )
    snapshot_options.fCompressionLevel = compression_level

    if not is_data:
        dataset_cfg = setup.datasets[dataset_name]
        process_name = dataset_cfg["process_name"]
        process = setup.base_processes[process_name]
        processors_cfg, processor_instances = setup.get_processors(
            process_name, stage="AnaTupleMerge", create_instances=True
        )
        if len(processors_cfg) == 0:
            processor_instances["default"] = DefaultAnaCacheProcessor()
        Corrections.initializeGlobal(
            setup=setup,
            stage="AnaTupleMerge",
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            process_name=process_name,
            process_cfg=process,
            processors=processor_instances,
            isData=is_data,
            load_corr_lib=True,
            trigger_class=None,
        )
        corrections = Corrections.getGlobal()

        ana_caches = {}
        for ds_name, reports in input_reports.items():
            ana_caches[ds_name] = combineAnaCaches(reports, processor_instances)

        tree_list = None
        for ds_name, reports in input_reports.items():
            for report in reports:
                report_tree_list = getTreeListFromReport(report)
                if tree_list is None:
                    tree_list = report_tree_list
                elif tree_list != report_tree_list:
                    raise RuntimeError(
                        f"Uncertainty list mismatch between reports for dataset {ds_name}."
                    )
    else:
        tree_list = [(central, central, "Events")]

    if len(root_outputs) > 1 and len(tree_list) > 1:
        raise NotImplementedError(
            "Cannot write multiple output files when there are multiple uncertainties."
        )
    tmp_central_file = os.path.join(work_dir, f"{dataset_name}_central_tmp.root")
    compute_unc_variations = setup.global_params.get("compute_unc_variations", False)

    for unc_source, unc_scale, tree_name in tree_list:
        syst_name = getSystName(unc_source, unc_scale)
        print(f"Merging {syst_name}")
        event_filter = None
        df = ROOT.RDataFrame(tree_name, input_roots)
        columns, _ = getColumns(df)
        if unc_source == central:
            if is_data:
                event_filter = ROOT.EventDuplicateFilter()
                df = event_filter.apply(
                    ROOT.RDF.AsRNode(df),
                    ["run", "luminosityBlock", "event"],
                    "EventDuplicateFilter",
                )
            else:
                for p_instance in processor_instances.values():
                    df = p_instance.onAnaCache_prepareDataFrame(df)

                df, weight_branches = corrections.getNormalisationCorrections(
                    df,
                    lepton_legs=None,
                    offline_legs=None,
                    trigger_names=None,
                    unc_source=unc_source,
                    unc_scale=unc_scale,
                    ana_caches=ana_caches,
                    return_variations=compute_unc_variations,
                    use_genWeight_sign_only=True,
                )
                columns += weight_branches
        output_file = (
            tmp_central_file
            if len(root_outputs) > 1 and unc_source == central
            else root_outputs[0]
        )
        df.Snapshot(tree_name, output_file, columns, snapshot_options)
        if event_filter is not None:
            event_filter.clear()
        snapshot_options.fMode = "UPDATE"

    if len(root_outputs) > 1:
        tree_name = tree_list[0][2]
        n_files = len(root_outputs)
        print(f"Splitting output into {n_files}")
        df = ROOT.RDataFrame(tree_name, tmp_central_file)
        n_events = df.Count().GetValue()
        # Add 1 to take care of int() always flooring
        n_events_per_file = int(n_events / n_files) + 1
        print(f"Going to make {n_events_per_file} events per file")
        range_start = 0
        for output in root_outputs:
            df_split = df.Range(range_start, range_start + n_events_per_file)
            snapshot_options.fMode = "UPDATE" if os.path.exists(output) else "RECREATE"
            df_split.Snapshot(tree_name, output, ".*", snapshot_options)
            range_start += n_events_per_file

    copyFileContent(
        input_roots,
        root_outputs[0],
        copyTrees=False,
        copyHistograms=True,
        appendIfExists=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--work-dir", required=True, type=str)
    parser.add_argument("--customisations", required=False, type=str, default=None)
    parser.add_argument("--input-reports", required=True, type=str)
    parser.add_argument("--input-roots", required=True, nargs="+", type=str)
    parser.add_argument("--root-outputs", required=False, nargs="+", type=str)
    parser.add_argument("--compression-level", type=int, default=9)
    parser.add_argument("--compression-algo", type=str, default="LZMA")
    parser.add_argument("--is-data", action="store_true")
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    args = parser.parse_args()

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"],
        args.period,
        args.LAWrunVersion,
        customisations=args.customisations,
    )

    report_files = DeserializeObjectFromString(args.input_reports)
    reports = {}
    for ds_name, report_files in report_files.items():
        reports[ds_name] = []
        for report_file in report_files:
            with open(report_file, "r") as f:
                reports[ds_name].append(yaml.safe_load(f))

    mergeAnaTuples(
        setup=setup,
        dataset_name=args.dataset,
        is_data=args.is_data,
        work_dir=args.work_dir,
        input_reports=reports,
        input_roots=args.input_roots,
        root_outputs=args.root_outputs,
        compression_algo=args.compression_algo,
        compression_level=args.compression_level,
    )
