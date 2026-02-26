import os
import yaml
import ROOT
import time
import shutil
import importlib
import uproot
import awkward as ak
import numpy as np
import sys
import json

ROOT.EnableThreadSafety()

from FLAF.Common.Utilities import DeclareHeader
from FLAF.RunKit.run_tools import ps_call
import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
from Corrections.Corrections import Corrections

import FLAF.Common.triggerSel as Triggers
import FLAF.Common.BaselineSelection as Baseline
from Corrections.CorrectionsCore import getScales, central

defaultColToSave = ["FullEventId"]


def check_columns(expected_columns, columns_to_save, available_columns):
    if set(expected_columns) != set(columns_to_save):
        raise Exception(
            f"Mismatch between expected columns and save columns {expected_columns} : {columns_to_save}"
        )
    if not set(columns_to_save).issubset(set(available_columns)):
        raise Exception(
            f"Missing a column to save from available columns {columns_to_save} : {available_columns}"
        )


def run_producer(
    producer,
    dfw,
    setup,
    dataset,
    producer_config,
    outFileName,
    treeName,
    snapshotOptions,
    uprootCompression,
    workingDir,
):
    save_as = producer_config.get("save_as", "root")
    if "FullEventId" not in dfw.colToSave:
        dfw.colToSave.append("FullEventId")
    expected_columns = [
        f"{producer.payload_name}_{col}" for col in producer_config["columns"]
    ] + ["FullEventId"]
    if producer_config.get("awkward_based", False):
        vars_to_save = []
        if hasattr(producer, "prepare_dfw"):
            dfw = producer.prepare_dfw(dfw, dataset)
        vars_to_save = list(producer.vars_to_save)
        if "FullEventId" not in vars_to_save:
            vars_to_save.append("FullEventId")
        n_orig = dfw.df.Count()
        dfw.df.Snapshot(
            f"tmp", os.path.join(workingDir, "tmp.root"), vars_to_save, snapshotOptions
        )
        n_orig = n_orig.GetValue()
        final_array = None
        final_dict = None
        uproot_stepsize = producer_config.get("uproot_stepsize", "100MB")
        for array in uproot.iterate(
            f"{os.path.join(workingDir, 'tmp.root')}:tmp", step_size=uproot_stepsize
        ):  # For DNN 50MB translates to ~300_000 events
            new_array = producer.run(array)
            if save_as == "root":
                if len(new_array["FullEventId"]) != len(array["FullEventId"]):
                    raise Exception(
                        f"Mismatch in number of events between input and output {len(new_array['FullEventId'])} != {len(array['FullEventId'])}"
                    )
                if np.any(new_array["FullEventId"] != array["FullEventId"]):
                    print("orig FullEventId:", array["FullEventId"])
                    print("new FullEventId:", new_array["FullEventId"])
                    raise Exception("Mismatch in FullEventId between input and output")

                if final_array is None:
                    final_array = new_array
                else:
                    final_array = ak.concatenate([final_array, new_array])
            elif save_as == "json":
                final_dict = producer.combine(final_dict=final_dict, new_dict=new_array)
            else:
                raise RuntimeError(f"Illegal output format `{save_as}`.")

        if save_as == "root":
            check_columns(expected_columns, final_array.fields, final_array.fields)
            n_final = len(final_array["FullEventId"])
            with uproot.recreate(outFileName, compression=uprootCompression) as outfile:
                outfile[treeName] = final_array
        elif save_as == "json":
            with open(outFileName, "w") as f:
                json.dump(final_dict, f, indent=4)
    else:
        n_orig = dfw.df.Count()
        dfw = producer.run(dfw)
        n_final = dfw.df.Count()
        check_columns(expected_columns, dfw.colToSave, dfw.df.GetColumnNames())
        varToSave = Utilities.ListToVector(dfw.colToSave)
        dfw.df.Snapshot(treeName, outFileName, varToSave, snapshotOptions)
        n_orig = n_orig.GetValue()
        n_final = n_final.GetValue()
    if save_as == "root" and n_orig != n_final:
        # json doesn't save tree => cannot calculate n_final
        raise Exception(
            f"Mismatch in number of events before and after producer {n_orig} != {n_final}"
        )


def createAnalysisCache(
    *,
    setup,
    dataset_name,
    inFileName,
    period,
    cacheFileNames,
    snapshotOptions,
    producer_to_run,
    uprootCompression,
    workingDir,
    histTupleDef=None,
):
    treeName = setup.global_params.get("treeName", "Events")
    unc_cfg_dict = setup.weights_config
    isData = dataset_name == "data"

    Utilities.InitializeCorrections(setup, dataset_name, stage="AnalysisCache")
    scale_uncertainties = set()
    if setup.global_params["compute_unc_variations"]:
        scale_uncertainties.update(unc_cfg_dict["shape"].keys())
    print("Scale uncertainties to consider:", scale_uncertainties)

    producer_config = setup.global_params["payload_producers"][producer_to_run]
    producers_module_name = producer_config["producers_module_name"]
    producer_name = producer_config["producer_name"]
    producers_module = importlib.import_module(producers_module_name)
    producer_class = getattr(producers_module, producer_name)
    producer = producer_class(producer_config, producer_to_run, period)
    saveAs = producer_config.get("save_as", "root")

    tmp_fileNames = []
    centralTree = None
    centralCaches = None
    allRootFiles = {}
    for unc_source in [central] + list(scale_uncertainties):
        for unc_scale in getScales(unc_source):
            print(f"Processing events for: {unc_source} {unc_scale}")
            isCentral = unc_source == central
            fullTreeName = (
                treeName if isCentral else f"Events__{unc_source}__{unc_scale}"
            )
            df_orig, df, tree, cacheTrees = Utilities.CreateDataFrame(
                treeName=fullTreeName,
                fileName=inFileName,
                caches=cacheFileNames,
                files=allRootFiles,
                centralTree=centralTree,
                centralCaches=centralCaches,
                central=central,
                filter_valid=False,
            )
            if isCentral:
                centralTree = tree
                centralCaches = cacheTrees
            ROOT.RDF.Experimental.AddProgressBar(df_orig)

            if saveAs == "root":
                dfw = Utilities.DataFrameWrapper(df, defaultColToSave)
                tmp_fileName = f"{fullTreeName}.root"
            elif saveAs == "json":
                if hasattr(producer, "create_dfw"):
                    dfw = producer.create_dfw(
                        df=df,
                        dataset_name=dataset_name,
                        histTupleDef=histTupleDef,
                        isData=isData,
                        setup=setup,
                        uncName=unc_source,
                        uncScale=unc_scale,
                        unc_cfg_dict=unc_cfg_dict,
                        final_weight_name=f"weight_{unc_source}_{unc_scale}",
                        df_is_central=isCentral,
                    )

                tmp_fileName = (
                    f"{central}.json" if isCentral else f"{unc_source}_{unc_scale}.json"
                )
            else:
                raise NotImplementedError(f"Unsupported output format `{saveAs}`.")

            run_producer(
                producer,
                dfw,
                setup,
                dataset_name,
                producer_config,
                tmp_fileName,
                fullTreeName,
                snapshotOptions,
                uprootCompression,
                workingDir,
            )
            tmp_fileNames.append(tmp_fileName)

    return tmp_fileNames


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--cacheFiles", required=False, type=str, default=None)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--producer", type=str, default=None)
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compressionLevel", type=int, default=9)
    parser.add_argument("--compressionAlgo", type=str, default="LZMA")
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--workingDir", required=True, type=str)
    parser.add_argument("--histTupleDef", type=str)
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    args = parser.parse_args()

    startTime = time.time()

    ana_path = os.environ["ANALYSIS_PATH"]
    sys.path.append(ana_path)
    headers = ["FLAF/include/HistHelper.h", "FLAF/include/Utilities.h"]
    for header in headers:
        DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)

    setup = Setup.getGlobal(
        os.environ["ANALYSIS_PATH"], args.period, args.LAWrunVersion
    )
    producer_config = setup.global_params["payload_producers"][args.producer]
    histTupleDef = Utilities.load_module(args.histTupleDef)

    setup.global_params["channels_to_consider"] = (
        args.channels.split(",")
        if args.channels
        else setup.global_params["channelSelection"]
    )
    process_name = (
        setup.datasets[args.dataset]["process_name"]
        if args.dataset != "data"
        else "data"
    )
    setup.global_params["process_name"] = process_name
    process_group = (
        setup.datasets[args.dataset]["process_group"]
        if args.dataset != "data"
        else "data"
    )
    setup.global_params["process_group"] = process_group

    setup.global_params["compute_unc_variations"] = (
        args.compute_unc_variations and process_group != "data"
    )

    cacheFileNames = {}
    if args.cacheFiles:
        for entry in args.cacheFiles.split(","):
            name, file = entry.split(":")
            if name in cacheFileNames:
                raise RuntimeError(f"Cache file for {name} already specified.")
            cacheFileNames[name] = file

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = False
    snapshotOptions.fLazy = False
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(
        ROOT.ROOT.RCompressionSetting.EAlgorithm, "k" + args.compressionAlgo
    )
    snapshotOptions.fCompressionLevel = args.compressionLevel

    unc_cfg_dict = {}

    uprootCompression = getattr(uproot, args.compressionAlgo)
    uprootCompression = uprootCompression(args.compressionLevel)

    tmp_fileNames = createAnalysisCache(
        setup=setup,
        dataset_name=args.dataset,
        inFileName=args.inFile,
        period=args.period,
        cacheFileNames=cacheFileNames,
        snapshotOptions=snapshotOptions,
        producer_to_run=args.producer,
        uprootCompression=uprootCompression,
        workingDir=args.workingDir,
        histTupleDef=histTupleDef,
    )

    saveAs = producer_config.get("save_as", "root")
    if saveAs == "root":
        hadd_cmd = ["hadd", "-j", "-ff", args.outFile]
        hadd_cmd.extend(tmp_fileNames)
        ps_call(hadd_cmd, verbose=1)
        if os.path.exists(args.outFile) and len(tmp_fileNames) != 0:
            for file_syst in tmp_fileNames:
                if file_syst != args.outFile:
                    os.remove(file_syst)
    elif saveAs == "json":
        print(f"Saving json output to {args.outFile}")

        data = {}
        for file_name in tmp_fileNames:
            unc_name = file_name.removesuffix(".json")
            with open(file_name, "r") as json_file:
                file_data = json.load(json_file)
                data[unc_name] = file_data

        with open(args.outFile, "w") as final_output_file:
            json.dump(data, final_output_file, indent=4)

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
