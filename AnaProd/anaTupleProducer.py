import datetime
import os
import sys
import ROOT
import shutil
import json

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.BaselineSelection as Baseline
import FLAF.Common.Utilities as Utilities
import FLAF.Common.ReportTools as ReportTools
import FLAF.Common.triggerSel as Triggers
from FLAF.Common.Setup import Setup
from Corrections.Corrections import Corrections
from Corrections.lumi import LumiFilter
from Corrections.CorrectionsCore import central, getScales, getSystName
from Corrections.pu import puWeightProducer


class DefaultAnaCacheProcessor:
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


# ROOT.EnableImplicitMT(1)
ROOT.EnableThreadSafety()


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
    if len(processors_cfg) == 0:
        processor_instances["default"] = DefaultAnaCacheProcessor()
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
    report = {}
    nEventsInFile = (
        df.Count().GetValue()
    )  # If range exists, it only loads that number of events -- does this mean the same file could be loaded by multiple anaTuple jobs? This could be an issue for normalizing later
    # lumis = df.Take["unsigned int"]("luminosityBlock")
    # runs = df.Take["unsigned int"]("run")
    # lumis_val = lumis.GetValue()
    # runs_val = runs.GetValue()
    # run_lumi = [ f"{run}:{lumi}" for run,lumi in zip(runs_val,lumis_val) ]
    # unique_run_lumi = list(set(run_lumi))
    report["nano_file_name"] = inFileName
    report["anaTuple_file_name"] = outputName
    report["n_original_events"] = nEventsInFile
    report["dataset_name"] = dataset_name
    report["output_files"] = []

    shape_sources = [central]
    if "pu" in corrections.to_apply and compute_unc_variations:
        shape_sources += puWeightProducer.uncSource

    report["denominator"] = {}
    for shape_unc_source in shape_sources:
        report["denominator"][shape_unc_source] = {}
        for shape_unc_scale in getScales(shape_unc_source):
            report["denominator"][shape_unc_source][shape_unc_scale] = {}
            for p_name, p_instance in processor_instances.items():
                report["denominator"][shape_unc_source][shape_unc_scale][
                    p_name
                ] = p_instance.onAnaCache_initializeDenomEntry()

    gen_weight_name = "weight_gen"

    def updateDenomEntry(rdf):
        for p_instance in processor_instances.values():
            rdf = p_instance.onAnaCache_prepareDataFrame(rdf)

        for shape_unc_source in shape_sources:
            for shape_unc_scale in getScales(shape_unc_source):
                shape_unc_name = getSystName(shape_unc_source, shape_unc_scale)
                weights_to_apply = [gen_weight_name]
                if "pu" in corrections.to_apply:
                    weights_to_apply.append(f"weight_pu_{shape_unc_scale}")
                for p_name, p_instance in processor_instances.items():
                    output_branch_name = f"weight_denom_{p_name}_{shape_unc_name}"
                    report["denominator"][shape_unc_source][shape_unc_scale][p_name] = (
                        p_instance.onAnaCache_updateDenomEntry(
                            report["denominator"][shape_unc_source][shape_unc_scale][
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
            if "pu" in corrections.to_apply:
                data_frame = corrections.pu.getWeight(data_frame)
            updateDenomEntry(data_frame)
    # if isData: json_dict_for_cache['RunLumi'] = unique_run_lumi
    ROOT.RDF.Experimental.AddProgressBar(df)
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
    snaps = [df.Snapshot(treeName, outFileName, [fullEventIdColumn], snapshotOptions)]
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
        dfw.Apply(Baseline.SelectRecoP4, syst_name, setup.global_params["nano_version"])
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
        snaps.append(dfw.df.Snapshot(treeName, outFileName, varToSave, snapshotOptions))

    if snapshotOptions.fLazy == True:
        ROOT.RDF.RunGraphs(snaps)
    for shape_unc_source in shape_sources:
        for shape_unc_scale in getScales(shape_unc_source):
            for p_name, p_instance in processor_instances.items():
                report["denominator"][shape_unc_source][shape_unc_scale][p_name] = (
                    p_instance.onAnaCache_materializeDenomEntry(
                        report["denominator"][shape_unc_source][shape_unc_scale][p_name]
                    )
                )
                report["denominator"][shape_unc_source][shape_unc_scale][p_name] = (
                    p_instance.onAnaCache_finalizeDenomEntry(
                        report["denominator"][shape_unc_source][shape_unc_scale][p_name]
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
    parser.add_argument("--evtIds", type=str, default="")
    parser.add_argument("--reportOutput", type=str, default=None)
    parser.add_argument("--LAWrunVersion", required=True, type=str)

    args = parser.parse_args()

    ROOT.gROOT.ProcessLine(".include " + os.environ["FLAF_PATH"])
    ROOT.gROOT.ProcessLine('#include "include/GenTools.h"')
    ROOT.gInterpreter.ProcessLine(f'ParticleDB::Initialize("{args.particleFile}");')
    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"],
        args.period,
        args.LAWrunVersion,
        customisations=args.customisations,
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
    )
