import datetime
import os
import sys
import ROOT
import shutil
import zlib

# import fastcrc
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


# ROOT.EnableImplicitMT(1)
ROOT.EnableThreadSafety()


def SelectBTagShapeSF(df, weight_name):
    df = df.Define("weight_bTagShapeSF", weight_name)
    return df


def createAnatuple(
    inFile,
    inFileName,
    treeName,
    outDir,
    setup,
    sample_name,
    anaCache,
    snapshotOptions,
    range,
    evtIds,
    store_noncentral,
    compute_unc_variations,
    uncertainties,
    anaTupleDef,
    channels,
    jsonName=None,
):
    start_time = datetime.datetime.now()
    compression_settings = (
        snapshotOptions.fCompressionAlgorithm * 100 + snapshotOptions.fCompressionLevel
    )
    period = setup.global_params["era"]
    sample_config = setup.samples[sample_name]
    mass = -1 if "mass" not in sample_config else sample_config["mass"]
    spin = -100 if "spin" not in sample_config else sample_config["spin"]
    isHH = True if mass > 0 else False
    isData = sample_config["process_group"] == "data"
    isSignal = sample_config["process_group"] == "signals"
    loadTF = anaTupleDef.loadTF
    loadHHBtag = anaTupleDef.loadHHBtag
    lepton_legs = anaTupleDef.lepton_legs
    offline_legs = anaTupleDef.offline_legs
    Baseline.Initialize(loadTF, loadHHBtag)
    triggerFile = setup.global_params.get("triggerFile")
    trigger_class = None
    if triggerFile is not None:
        triggerFile = os.path.join(os.environ["ANALYSIS_PATH"], triggerFile)
        trigger_class = Triggers.Triggers(triggerFile)
    Corrections.initializeGlobal(
        setup.global_params,
        sample_name,
        sample_config["process_name"],
        isData=isData,
        load_corr_lib=True,
        trigger_class=trigger_class,
    )
    corrections = Corrections.getGlobal()
    df = ROOT.RDataFrame(treeName, inFile)
    json_dict_for_cache = {}
    nEventsInFile = (
        df.Count().GetValue()
    )  # If range exists, it only loads that number of events -- does this mean the same file could be loaded by multiple anaTuple jobs? This could be an issue for normalizing later
    # lumis = df.Take["unsigned int"]("luminosityBlock")
    # runs = df.Take["unsigned int"]("run")
    # lumis_val = lumis.GetValue()
    # runs_val = runs.GetValue()
    # run_lumi = [ f"{run}:{lumi}" for run,lumi in zip(runs_val,lumis_val) ]
    # unique_run_lumi = list(set(run_lumi))
    json_dict_for_cache["nano_file_name"] = inFileName
    json_dict_for_cache["nEvents"] = nEventsInFile
    json_dict_for_cache["sample_name"] = sample_name
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
    # isSignal = sample_type in setup.global_params["signal_types"]
    applyTriggerFilter = sample_config.get("applyTriggerFilter", True)
    df = df.Define("period", f"static_cast<int>(Period::{period})")
    df = df.Define(
        "X_mass", f"static_cast<int>({mass})"
    )  # this has to be moved in specific analyses def
    df = df.Define(
        "X_spin", f"static_cast<int>({spin})"
    )  # this has to be moved in specific analyses def
    df = df.Define(
        "FullEventId",
        f"""eventId::encodeFullEventId({Utilities.crc16(sample_name.encode())}, {Utilities.crc16(inFileName.encode())}, rdfentry_)""",
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
    snaps = []
    reports = []
    outfilesNames = []
    k = 0
    print(f"syst_dict={syst_dict}")
    for syst_name, source_name in syst_dict.items():
        if source_name not in uncertainties and "all" not in uncertainties:
            continue
        is_central = syst_name in ["Central", "nano"]
        if not is_central and not compute_unc_variations:
            continue
        suffix = "" if is_central else f"_{syst_name}"
        if len(suffix) and not store_noncentral:
            continue
        columns_to_save = anaTupleDef.getDefaultColumnsToSave(isData)
        if "sample_type" in columns_to_save:
            columns_to_save.remove("sample_type")
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
        )
        if setup.global_params["nano_version"] == "v12":
            dfw.DefineAndAppend("weight_L1PreFiring_Central", "L1PreFiringWeight_Nom")
            dfw.DefineAndAppend(
                "weight_L1PreFiring_ECAL_Central", "L1PreFiringWeight_ECAL_Nom"
            )
            dfw.DefineAndAppend(
                "weight_L1PreFiring_Muon_Central", "L1PreFiringWeight_Muon_Nom"
            )
            if is_central and compute_unc_variations:
                dfw.DefineAndAppend(
                    "weight_L1PreFiringDown_rel",
                    "L1PreFiringWeight_Dn/L1PreFiringWeight_Nom",
                )
                dfw.DefineAndAppend(
                    "weight_L1PreFiringUp_rel",
                    "L1PreFiringWeight_Up/L1PreFiringWeight_Nom",
                )
                dfw.DefineAndAppend(
                    "weight_L1PreFiring_ECALDown_rel",
                    "L1PreFiringWeight_ECAL_Dn/L1PreFiringWeight_ECAL_Nom",
                )
                dfw.DefineAndAppend(
                    "weight_L1PreFiring_ECALUp_rel",
                    "L1PreFiringWeight_ECAL_Up/L1PreFiringWeight_ECAL_Nom",
                )
                dfw.DefineAndAppend(
                    "weight_L1PreFiring_Muon_StatDown_rel",
                    "L1PreFiringWeight_Muon_StatDn/L1PreFiringWeight_Muon_Nom",
                )
                dfw.DefineAndAppend(
                    "weight_L1PreFiring_Muon_StatUp_rel",
                    "L1PreFiringWeight_Muon_StatUp/L1PreFiringWeight_Muon_Nom",
                )
                dfw.DefineAndAppend(
                    "weight_L1PreFiring_Muon_SystDown_rel",
                    "L1PreFiringWeight_Muon_SystDn/L1PreFiringWeight_Muon_Nom",
                )
                dfw.DefineAndAppend(
                    "weight_L1PreFiring_Muon_SystUp_rel",
                    "L1PreFiringWeight_Muon_SystUp/L1PreFiringWeight_Muon_Nom",
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
                setup.global_params,
                setup.samples,
                sample_name,
                lepton_legs,
                offline_legs,
                triggers_to_use,
                syst_name,
                source_name,
                return_variations=is_central and compute_unc_variations,
                isCentral=is_central,
                ana_cache=anaCache,
            )
            puIDbranches = [
                "weight_Jet_PUJetID_Central_tmp",
                "weight_Jet_PUJetID_effUp_rel_tmp",
                "weight_Jet_PUJetID_effDown_rel_tmp",
            ]
            for puIDbranch in puIDbranches:
                if puIDbranch in dfw.df.GetColumnNames():
                    new_branch_name = puIDbranch.strip("_tmp")
                    dfw.Define(
                        f"""ExtraJet_{new_branch_name}""", f"{puIDbranch}[ExtraJet_B1]"
                    )
                    if setup.global_params["storeExtraJets"]:
                        dfw.colToSave.append(f"""ExtraJet_{new_branch_name}""")
                    for bjet_idx in [1, 2]:
                        dfw.DefineAndAppend(
                            f"{new_branch_name}_b{bjet_idx}",
                            f"Hbb_isValid ? {puIDbranch}[b{bjet_idx}_idx] : -100.f",
                        )
                if puIDbranch in weight_branches:
                    weight_branches.remove(puIDbranch)
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
        reports.append(dfw.df.Report())
        snaps.append(
            dfw.df.Snapshot(f"Events", outFileName, varToSave, snapshotOptions)
        )

        if is_central:
            nEventsAfterFilter = dfw.df.Count()  # .GetValue()
    if snapshotOptions.fLazy == True:
        ROOT.RDF.RunGraphs(snaps)
    hist_time = ROOT.TH1D(f"time", f"time", 1, 0, 1)
    end_time = datetime.datetime.now()
    hist_time.SetBinContent(1, (end_time - start_time).total_seconds())
    for index, fileName in enumerate(outfilesNames):
        outputRootFile = ROOT.TFile(fileName, "UPDATE", "", compression_settings)
        rep = ReportTools.SaveReport(reports[index].GetValue(), reoprtName=f"Report")
        outputRootFile.WriteTObject(rep, f"Report", "Overwrite")
        if index == 0:
            outputRootFile.WriteTObject(hist_time, f"runtime", "Overwrite")
        outputRootFile.Close()
        # if print_cutflow:
        #     report.Print()

    # Dump
    if jsonName == None:
        jsonName = f"{inFileName.split('.')[0]}.json"
    jsonName = os.path.join(outDir, f"{jsonName}")

    # Move GetValue() to here so it only runs loop once (after the snaps list)
    json_dict_for_cache["nEvents_Filtered"] = nEventsAfterFilter.GetValue()
    with open(jsonName, "w") as fp:
        json.dump(json_dict_for_cache, fp)


if __name__ == "__main__":
    import argparse
    import os
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--outDir", required=True, type=str)
    parser.add_argument("--inFileName", required=True, type=str)
    parser.add_argument("--sample", required=True, type=str)
    parser.add_argument("--anaCache", required=True, type=str)
    parser.add_argument("--anaTupleDef", required=True, type=str)
    parser.add_argument(
        "--store-noncentral", action="store_true", help="Store ES variations."
    )
    parser.add_argument("--compute-unc-variations", action="store_true")
    parser.add_argument("--uncertainties", type=str, default="all")
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--treeName", required=False, type=str, default="Events")
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
    parser.add_argument("--jsonName", type=str, default=None)

    args = parser.parse_args()

    ROOT.gROOT.ProcessLine(".include " + os.environ["FLAF_PATH"])
    ROOT.gROOT.ProcessLine('#include "include/GenTools.h"')
    ROOT.gInterpreter.ProcessLine(f'ParticleDB::Initialize("{args.particleFile}");')
    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"], args.period, args.customisations
    )
    with open(args.anaCache, "r") as f:
        anaCache = yaml.safe_load(f)

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
        ROOT.ROOT, "k" + args.compressionAlgo
    )
    snapshotOptions.fCompressionLevel = args.compressionLevel
    createAnatuple(
        args.inFile,
        args.inFileName,
        args.treeName,
        args.outDir,
        setup,
        args.sample,
        anaCache,
        snapshotOptions,
        args.nEvents,
        args.evtIds,
        args.store_noncentral,
        args.compute_unc_variations,
        args.uncertainties.split(","),
        anaTupleDef,
        channels,
        jsonName=args.jsonName,
    )
