import datetime
import os
import sys
import ROOT
import shutil
import json

# ROOT.EnableImplicitMT(1)
ROOT.EnableThreadSafety()

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.BaselineSelection as Baseline
import FLAF.Common.Utilities as Utilities
import FLAF.Common.ReportTools as ReportTools
import FLAF.Common.triggerSel as Triggers
from FLAF.Common.Setup import Setup
from FLAF.Common.shared_mc import shared_mc_in_era_expr, shared_mc_split
from FLAF.AnaProd.CostModel import chunk_bounds, scaled_bounds
from Corrections.Corrections import Corrections
from Corrections.lumi import LumiFilter
from Corrections.CorrectionsCore import (
    central,
    getScales,
    getSystName,
    ShapeWeightRegistry,
)


class DefaultAnaCacheProcessor:
    def __init__(self, default_denom_processor=True):
        self.default_denom_processor = default_denom_processor

    def onAnaCache_initializeDenomEntry(self):
        return []

    def onAnaCache_prepareDataFrame(self, df):
        return df

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

    def onAnaTuple_prepareDataFrame(self, df):
        return df

    def onAnaTuple_defineCrossSection(
        self, df, crossSectionBranch, xs_db, dataset_name, dataset_entry
    ):
        xs_name = dataset_entry["crossSection"]
        xs_value = xs_db.getValue(xs_name)
        return df.Define(crossSectionBranch, f"float({xs_value})")

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


def createAnatuple(
    *,
    inFile,
    inFileName,
    treeName,
    treeNameNotSelected,
    outDir,
    setup,
    dataset_name,
    snapshotOptions,
    range,
    evtIds,
    store_noncentral,
    compute_unc_variations,
    uncertainties,
    anaTupleDef,
    channels,
    outputName,
    reportOutput=None,
    use_genWeight_sign_only=True,
    chunk_index=0,
    n_chunks=1,
    max_scan_events=None,
):
    start_time = datetime.datetime.now()
    compression_settings = (
        snapshotOptions.fCompressionAlgorithm * 100 + snapshotOptions.fCompressionLevel
    )
    period = setup.global_params["era"]
    dataset_cfg = setup.datasets[dataset_name]
    mass = dataset_cfg.get("mass", -1)
    spin = dataset_cfg.get("spin", -100)
    isHH = mass > 0
    isData = dataset_cfg["process_group"] == "data"
    isSignal = dataset_cfg["process_group"] == "signals"
    loadTF = anaTupleDef.loadTF
    lepton_legs = anaTupleDef.lepton_legs
    offline_legs = anaTupleDef.offline_legs
    Baseline.Initialize(loadTF)
    if hasattr(anaTupleDef, "Initialize"):
        anaTupleDef.Initialize(setup, dataset_name)
    triggerFile = setup.global_params.get("triggerFile")
    trigger_class = None
    if triggerFile is not None:
        triggerFile = os.path.join(os.environ["ANALYSIS_PATH"], triggerFile)
        trigger_class = Triggers.Triggers(triggerFile)
    process_name = dataset_cfg["process_name"]
    process = setup.base_processes[process_name]
    processors_cfg, processor_instances = setup.get_processors(
        process_name, stage="AnaTuple", create_instances=True
    )
    if not isData:
        if "ds" in processor_instances:
            raise RuntimeError(
                "Processor name 'ds' is reserved for dataset-level cache, please rename the processor."
            )
        ds_processor_default = len(processors_cfg) == 0
        processor_instances["ds"] = DefaultAnaCacheProcessor(
            default_denom_processor=ds_processor_default
        )
    Corrections.initializeGlobal(
        setup=setup,
        stage="AnaTuple",
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        process_name=process_name,
        process_cfg=process,
        processors=processor_instances,
        isData=isData,
        load_corr_lib=True,
        trigger_class=trigger_class,
    )
    corrections = Corrections.getGlobal()
    root_file = ROOT.TFile.Open(inFile)
    tree = root_file.Get(treeName)
    df = ROOT.RDataFrame(tree)
    if treeNameNotSelected in root_file.GetListOfKeys():
        tree_not_selected = root_file.Get(treeNameNotSelected)
        df_not_selected = ROOT.RDataFrame(tree_not_selected)
    else:
        tree_not_selected = None
        df_not_selected = None

    nEventsInFile = df.Count().GetValue()
    # Attached to the unrestricted frame: the progress bar only accepts an RDataFrame or
    # an RNode, and it reports on the shared loop manager either way.
    ROOT.RDF.Experimental.AddProgressBar(df)

    # Restrict this job to its slice of the file.  This is applied here, before the
    # denominator and the snapshots are booked, so that every quantity the job reports
    # refers to the same slice and the sums over all chunks of a file reproduce the
    # whole-file values exactly.
    entry_begin, entry_end = chunk_bounds(nEventsInFile, chunk_index, n_chunks)
    if max_scan_events is not None and max_scan_events > 0:
        entry_end = min(entry_end, entry_begin + max_scan_events)
    n_scanned_events = max(entry_end - entry_begin, 0)
    if (entry_begin, entry_end) != (0, nEventsInFile):
        print(
            f"processing entries [{entry_begin}, {entry_end}) out of {nEventsInFile}"
            f" (chunk {chunk_index + 1}/{n_chunks})"
        )
        df = df.Range(entry_begin, entry_end)
        if df_not_selected is not None and nEventsInFile > 0:
            # Slice the not-selected tree by the same fraction.  Consecutive chunks share
            # a boundary by construction, so the slices are disjoint and cover the tree.
            ns_begin, ns_end = scaled_bounds(
                entry_begin, entry_end, nEventsInFile, tree_not_selected.GetEntries()
            )
            df_not_selected = ROOT.RDF.AsRNode(df_not_selected.Range(ns_begin, ns_end))
        # Erased back to RNode so the rest of the producer sees the same node type it
        # does for a whole file.
        df = ROOT.RDF.AsRNode(df)

    report = {}
    report["nano_file_name"] = inFileName
    report["anaTuple_file_name"] = outputName
    report["n_original_events"] = nEventsInFile
    report["n_scanned_events"] = n_scanned_events
    report["entry_range"] = [entry_begin, entry_end]
    report["dataset_name"] = dataset_name
    report["output_files"] = []

    runLumiTracker = ROOT.flaf.RunLumiTracker()
    df = df.Define("__runLumiTracker", runLumiTracker, ["run", "luminosityBlock"])
    runLumiTracker_sum = df.Sum("__runLumiTracker")
    handles_to_run = [runLumiTracker_sum]
    if df_not_selected is not None:
        df_not_selected = df_not_selected.Define(
            "__runLumiTracker", runLumiTracker, ["run", "luminosityBlock"]
        )
        handles_to_run.append(df_not_selected.Sum("__runLumiTracker"))

    # The denominators and the `base` numerator in Corrections are built from the same
    # registry, so they agree on which weights belong to each variation. Getting that
    # wrong is silent: a variation whose product is missing another producer's central
    # weight divides that weight out of the resulting _rel branch.
    shape_weight_registry = corrections.registerShapeWeights(
        ShapeWeightRegistry(), return_variations=compute_unc_variations
    )
    shape_sources = shape_weight_registry.sources

    shared_mc = None if isData else setup.global_params.get("shared_mc")
    shared_mc_expr = None
    if shared_mc:
        split_mod, lo, hi, _ = shared_mc_split(period, shared_mc)
        shared_mc_expr = shared_mc_in_era_expr(split_mod, lo, hi)

    def initializeDenomReport(key):
        report[key] = {}
        for shape_unc_source in shape_sources:
            report[key][shape_unc_source] = {}
            for shape_unc_scale in getScales(shape_unc_source):
                report[key][shape_unc_source][shape_unc_scale] = {}
                for p_name, p_instance in processor_instances.items():
                    report[key][shape_unc_source][shape_unc_scale][
                        p_name
                    ] = p_instance.onAnaCache_initializeDenomEntry()

    initializeDenomReport("denominator")
    if shared_mc_expr:
        initializeDenomReport("denominator_cmb")

    gen_weight_name = "weight_gen"

    def updateDenomEntry(rdf, report_key, branch_prefix):
        for p_instance in processor_instances.values():
            rdf = p_instance.onAnaCache_prepareDataFrame(rdf)

        for shape_unc_source in shape_sources:
            for shape_unc_scale in getScales(shape_unc_source):
                shape_unc_name = getSystName(shape_unc_source, shape_unc_scale)
                # Each shape producer contributes its varied branch only for the source
                # it owns, and its central branch otherwise. Keying off the scale alone
                # was correct only while pileup was the sole non-central source: with a
                # second source the pileup weight would be varied along with it.
                weights_to_apply = [gen_weight_name]
                weights_to_apply += shape_weight_registry.branches(
                    shape_unc_source, shape_unc_scale
                )
                for p_name, p_instance in processor_instances.items():
                    output_branch_name = f"{branch_prefix}_{p_name}_{shape_unc_name}"
                    report[report_key][shape_unc_source][shape_unc_scale][p_name] = (
                        p_instance.onAnaCache_updateDenomEntry(
                            report[report_key][shape_unc_source][shape_unc_scale][
                                p_name
                            ],
                            rdf,
                            output_branch_name,
                            weights_to_apply,
                        )
                    )
        return rdf

    if not isData:
        for data_frame in [df, df_not_selected]:
            if data_frame is None:
                continue
            genWeight_def = (
                "std::copysign<float>(1.f, genWeight)"
                if use_genWeight_sign_only
                else "genWeight"
            )
            data_frame = data_frame.Define(gen_weight_name, genWeight_def)
            # respect_enabled=False preserves the existing behaviour: this call site has
            # always defined the pileup weights regardless of the `enabled` config, so
            # the denominator carries them even where the numerator stage would not.
            data_frame, _ = corrections.defineShapeWeights(
                data_frame,
                return_variations=compute_unc_variations,
                respect_enabled=False,
            )
            updateDenomEntry(data_frame, "denominator", "weight_denom")
            if shared_mc_expr:
                data_frame = data_frame.Define("__shared_mc_in_era", shared_mc_expr)
                updateDenomEntry(
                    data_frame.Filter("__shared_mc_in_era"),
                    "denominator_cmb",
                    "weight_denom_cmb",
                )
    # if isData: json_dict_for_cache['RunLumi'] = unique_run_lumi

    if range is not None:
        df = df.Range(range)
    if len(evtIds) > 0:
        df = df.Filter(
            f"static const std::set<ULong64_t> evts = {{ {evtIds} }}; return evts.count(event) > 0;"
        )
    if isData and "lumiFile" in setup.global_params:
        lumiFile_path = setup.global_params["lumiFile"]
        if not lumiFile_path.startswith("/"):
            lumiFile_path = os.path.join(os.environ["ANALYSIS_PATH"], lumiFile_path)
        lumiFilter = LumiFilter(lumiFile_path)
        df = lumiFilter.filter(df)
    applyTriggerFilter = dataset_cfg.get("applyTriggerFilter", True)
    df = df.Define("period", f"static_cast<int>(Period::{period})")
    df = df.Define(
        "X_mass", f"static_cast<int>({mass})"
    )  # this has to be moved in specific analyses def
    df = df.Define(
        "X_spin", f"static_cast<int>({spin})"
    )  # this has to be moved in specific analyses def
    fullEventIdColumn = "FullEventId"
    df = df.Define(
        fullEventIdColumn,
        f"""eventId::encodeFullEventId({Utilities.crc16(dataset_name.encode())}, {Utilities.crc16(inFileName.encode())}, rdfentry_)""",
    )

    is_data = "true" if isData else "false"
    df = df.Define("isData", is_data)
    df = Baseline.CreateRecoP4(df, nano_version=setup.global_params["nano_version"])
    df = Baseline.DefineGenObjects(df, isData=isData, isHH=isHH)

    if isData:
        syst_dict = {"nano": "Central"}
        ana_reco_objects = Baseline.ana_reco_object_collections[
            setup.global_params["nano_version"]
        ]
        df, syst_dict = corrections.applyScaleUncertainties(df, ana_reco_objects)
    else:
        ana_reco_objects = Baseline.ana_reco_object_collections[
            setup.global_params["nano_version"]
        ]
        df, syst_dict = corrections.applyScaleUncertainties(df, ana_reco_objects)
    df_empty = df

    outfile_prefix = inFile.split("/")[-1]
    outfile_prefix = outfile_prefix.split(".")[0]
    outFileName = os.path.join(outDir, f"{outfile_prefix}_reference.root")
    report["reference_file"] = outFileName
    treeName = "Events"
    report["tree_name"] = treeName
    report["full_event_id_column"] = fullEventIdColumn
    outfilesNames = [outFileName]
    handles_to_run.append(
        df.Snapshot(treeName, outFileName, [fullEventIdColumn], snapshotOptions)
    )
    selection_reports = [df.Report()]

    print(f"syst_dict={syst_dict}")
    for syst_name, (unc_source, unc_scale) in syst_dict.items():
        if unc_source not in uncertainties and "all" not in uncertainties:
            continue
        is_central = syst_name in ["Central", "nano"]
        if not is_central and not compute_unc_variations:
            continue
        suffix = "" if is_central else f"_{syst_name}"
        if len(suffix) and not store_noncentral:
            continue
        columns_to_save = anaTupleDef.getDefaultColumnsToSave(isData)
        dfw = Utilities.DataFrameWrapper(df_empty, columns_to_save)
        dfw.Apply(
            Baseline.SelectRecoP4,
            syst_name,
            setup.global_params["nano_version"],
            setup.global_params["met_type"],
        )
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFilters#Analysis_Recommendations_for_any
        if "MET_flags" in setup.global_params:
            dfw.Apply(
                Baseline.applyMETFlags,
                setup.global_params["MET_flags"],
                setup.global_params.get("badMET_flag_runs", []),
                isData,
            )

        anaTupleDef.addAllVariables(
            dfw,
            syst_name,
            isData,
            trigger_class,
            lepton_legs,
            isSignal,
            applyTriggerFilter,
            setup.global_params,
            channels,
            dataset_cfg,
        )

        if not isData:
            triggers_to_use = set()
            for channel in channels:
                trigger_list = setup.global_params.get("triggers", {}).get(channel, [])
                for trigger in trigger_list:
                    if trigger not in trigger_class.trigger_dict.keys():
                        raise RuntimeError(
                            f"Trigger does not exist in triggers.yaml, {trigger}"
                        )
                    triggers_to_use.add(trigger)

            weight_branches = dfw.Apply(
                corrections.getNormalisationCorrections,
                lepton_legs=lepton_legs,
                offline_legs=offline_legs,
                trigger_names=triggers_to_use,
                unc_source=unc_source,
                unc_scale=unc_scale,
                ana_caches=None,
                return_variations=is_central and compute_unc_variations,
                use_genWeight_sign_only=use_genWeight_sign_only,
            )
            dfw.colToSave.extend(weight_branches)

        # Analysis anaTupleDef should define a legType as a leg obj
        # But to save with RDF, it needs to be converted to an int
        for leg_name in lepton_legs:
            branch_name = f"{leg_name}_legType"
            if branch_name in dfw.colToSave:
                dfw.Redefine(branch_name, f"static_cast<int>({branch_name})")
        varToSave = Utilities.ListToVector(dfw.colToSave)
        outfile_prefix = inFile.split("/")[-1]
        outfile_prefix = outfile_prefix.split(".")[0]
        outFileName = os.path.join(outDir, f"{outfile_prefix}{suffix}.root")
        outfilesNames.append(outFileName)
        report["output_files"].append(
            {
                "unc_source": unc_source,
                "unc_scale": unc_scale,
                "file_name": outFileName,
            }
        )
        selection_reports.append(dfw.df.Report())
        handles_to_run.append(
            dfw.df.Snapshot(treeName, outFileName, varToSave, snapshotOptions)
        )

    setup_end_time = datetime.datetime.now()
    ROOT.RDF.RunGraphs(handles_to_run)
    loop_end_time = datetime.datetime.now()
    # Separated because job-cost estimation extrapolates the event loop only: the setup
    # (JIT, corrections, the entry count) is a fixed cost that does not scale with events.
    report["setup_seconds"] = (setup_end_time - start_time).total_seconds()
    report["loop_seconds"] = (loop_end_time - setup_end_time).total_seconds()
    report["n_trees"] = len(report["output_files"])

    runLumiRanges_cpp = runLumiTracker.getRunLumiRanges()
    runLumiRanges = {}
    for run, lumi_ranges in runLumiRanges_cpp:
        run_str = str(run)
        if run_str not in runLumiRanges:
            runLumiRanges[run_str] = []
        for lumi_range in lumi_ranges:
            runLumiRanges[run_str].append([lumi_range.first, lumi_range.second])

    report["run_lumi_ranges"] = runLumiRanges

    denom_keys = ["denominator"]
    if "denominator_cmb" in report:
        denom_keys.append("denominator_cmb")
    for denom_key in denom_keys:
        for shape_unc_source in shape_sources:
            for shape_unc_scale in getScales(shape_unc_source):
                for p_name, p_instance in processor_instances.items():
                    report[denom_key][shape_unc_source][shape_unc_scale][p_name] = (
                        p_instance.onAnaCache_materializeDenomEntry(
                            report[denom_key][shape_unc_source][shape_unc_scale][p_name]
                        )
                    )
                    report[denom_key][shape_unc_source][shape_unc_scale][p_name] = (
                        p_instance.onAnaCache_finalizeDenomEntry(
                            report[denom_key][shape_unc_source][shape_unc_scale][p_name]
                        )
                    )

    hist_time = ROOT.TH1D(f"time", f"time", 1, 0, 1)
    end_time = datetime.datetime.now()
    hist_time.SetBinContent(1, (end_time - start_time).total_seconds())
    for index, fileName in enumerate(outfilesNames):
        outputRootFile = ROOT.TFile(fileName, "UPDATE", "", compression_settings)
        rep = ReportTools.SaveReport(
            selection_reports[index].GetValue(), reportName=f"Report"
        )
        outputRootFile.WriteTObject(rep, f"Report", "Overwrite")
        if index == 0:
            outputRootFile.WriteTObject(hist_time, f"runtime", "Overwrite")
        outputRootFile.Close()
        # if print_cutflow:
        #     report.Print()

    if reportOutput is not None:
        with open(reportOutput, "w") as f:
            json.dump(report, f)


if __name__ == "__main__":
    import argparse
    import os
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--outDir", required=True, type=str)
    parser.add_argument("--inFileName", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--anaTupleDef", required=True, type=str)
    parser.add_argument("--output-name", required=True, type=str)
    parser.add_argument(
        "--store-noncentral", action="store_true", help="Store ES variations."
    )
    parser.add_argument("--compute-unc-variations", action="store_true")
    parser.add_argument("--uncertainties", type=str, default="all")
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--treeName", required=False, type=str, default="Events")
    parser.add_argument(
        "--treeNameNotSelected", required=False, type=str, default="EventsNotSelected"
    )
    parser.add_argument(
        "--particleFile",
        type=str,
        default=f"{os.environ['FLAF_PATH']}/config/pdg_name_type_charge.txt",
    )
    parser.add_argument("--compressionLevel", type=int, default=4)
    parser.add_argument("--compressionAlgo", type=str, default="ZLIB")
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--nEvents", type=int, default=None)
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="index of the entry range to process (0-based)",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=1,
        help="number of equal entry ranges the input file is split into",
    )
    parser.add_argument(
        "--max-scan-events",
        type=int,
        default=None,
        help="stop after this many entries; used to time a dataset cheaply",
    )
    parser.add_argument("--evtIds", type=str, default="")
    parser.add_argument("--reportOutput", type=str, default=None)
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    parser.add_argument("--user-custom", type=str, default=None)

    args = parser.parse_args()

    ROOT.gROOT.ProcessLine(".include " + os.environ["FLAF_PATH"])
    ROOT.gROOT.ProcessLine('#include "include/RunLumiTracker.h"')
    ROOT.gROOT.ProcessLine('#include "include/GenTools.h"')
    ROOT.gInterpreter.ProcessLine(f'ParticleDB::Initialize("{args.particleFile}");')
    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"],
        args.period,
        args.LAWrunVersion,
        customisations=args.customisations,
        user_custom_file=args.user_custom,
    )

    channels = setup.global_params["channelSelection"]
    if args.channels:
        channels = (
            args.channels.split(",") if type(args.channels) == str else args.channels
        )
    anaTupleDef = Utilities.load_module(args.anaTupleDef)
    if os.path.isdir(args.outDir):
        shutil.rmtree(args.outDir)
    os.makedirs(args.outDir, exist_ok=True)
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = False
    snapshotOptions.fLazy = True
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(
        ROOT.ROOT.RCompressionSetting.EAlgorithm, "k" + args.compressionAlgo
    )

    snapshotOptions.fCompressionLevel = args.compressionLevel
    createAnatuple(
        inFile=args.inFile,
        inFileName=args.inFileName,
        treeName=args.treeName,
        treeNameNotSelected=args.treeNameNotSelected,
        outDir=args.outDir,
        setup=setup,
        dataset_name=args.dataset,
        snapshotOptions=snapshotOptions,
        range=args.nEvents,
        evtIds=args.evtIds,
        store_noncentral=args.store_noncentral,
        compute_unc_variations=args.compute_unc_variations,
        uncertainties=args.uncertainties.split(","),
        anaTupleDef=anaTupleDef,
        channels=channels,
        reportOutput=args.reportOutput,
        outputName=args.output_name,
        chunk_index=args.chunk_index,
        n_chunks=args.n_chunks,
        max_scan_events=args.max_scan_events,
    )
