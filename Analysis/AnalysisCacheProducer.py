import os
import sys
import yaml
import ROOT
import datetime
import time
import shutil
import importlib
import uproot
import awkward as ak
import json

ROOT.EnableThreadSafety()

from FLAF.Common.Utilities import DeclareHeader
from FLAF.RunKit.run_tools import ps_call
import FLAF.Common.LegacyVariables as LegacyVariables
import FLAF.Common.Utilities as Utilities
from FLAF.Analysis.HistProducerFile import AddCacheColumnsInDf

defaultColToSave = ["FullEventId"]
scales = ["Up", "Down"]


def getKeyNames(root_file_name):
    root_file = ROOT.TFile(root_file_name, "READ")
    key_names = [str(k.GetName()) for k in root_file.GetListOfKeys()]
    root_file.Close()
    return key_names


def applyLegacyVariables(dfw, global_cfg_dict, is_central=True):
    channels = global_cfg_dict["channelSelection"]
    trg_strings = {}
    for channel in channels:
        ch_value = global_cfg_dict["channelDefinition"][channel]
        dfw.df = dfw.df.Define(f"{channel}", f"channelId=={ch_value}")
        trigger_list = global_cfg_dict["triggers"][channel]
        trg_strings[channel] = "("
        trg_strings[channel] += " || ".join(f"HLT_{trg}" for trg in trigger_list)
        trg_strings[channel] += ")"
        for trigger in trigger_list:
            trigger_name = "HLT_" + trigger
            if trigger_name not in dfw.df.GetColumnNames():
                dfw.df = dfw.df.Define(trigger_name, "1")
    entryvalid_stri = "("
    entryvalid_stri += " || ".join(f"({ch} & {trg_strings[ch]})" for ch in channels)
    entryvalid_stri += ")"
    dfw.DefineAndAppend("entry_valid", entryvalid_stri)
    MT2Branches = dfw.Apply(LegacyVariables.GetMT2)
    dfw.colToSave.extend(MT2Branches)
    KinFitBranches = dfw.Apply(LegacyVariables.GetKinFit)
    dfw.colToSave.extend(KinFitBranches)
    if is_central:
        SVFitBranches = dfw.Apply(LegacyVariables.GetSVFit)
        dfw.colToSave.extend(SVFitBranches)


def createCentralQuantities(df_central, central_col_types, central_columns):
    map_creator = ROOT.analysis.MapCreator(*central_col_types)()
    df_central = map_creator.processCentral(
        ROOT.RDF.AsRNode(df_central), Utilities.ListToVector(central_columns)
    )
    return df_central


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
    producer_config,
    outFileName,
    treeName,
    snapshotOptions,
    uprootCompression,
    workingDir,
):
    if "FullEventId" not in dfw.colToSave:
        dfw.colToSave.append("FullEventId")
    expected_columns = [
        f"{producer.payload_name}_{col}" for col in producer_config["columns"]
    ] + ["FullEventId"]
    is_awkward_based = producer_config.get("awkward_based", False)
    if is_awkward_based:
        vars_to_save = []
        if hasattr(producer, "prepare_dfw"):
            dfw = producer.prepare_dfw(dfw)
        vars_to_save = list(producer.vars_to_save)
        if "FullEventId" not in vars_to_save:
            vars_to_save.append("FullEventId")
        dfw.df.Snapshot(
            f"tmp", os.path.join(workingDir, "tmp.root"), vars_to_save, snapshotOptions
        )
        # by default cache output is saved to root file
        # but we sometimes (BtagShapeProducer) want to save it as json
        save_as = producer_config.get("save_as", "root")
        final_array = None
        final_dict = None
        uproot_stepsize = producer_config.get("uproot_stepsize", "100MB")
        for array in uproot.iterate(
            f"{os.path.join(workingDir, 'tmp.root')}:tmp", step_size=uproot_stepsize
        ):  # For DNN 50MB translates to ~300_000 events
            new_array = producer.run(array)
            match save_as:
                case "root":
                    if final_array is None:
                        final_array = new_array
                    else:
                        final_array = ak.concatenate([final_array, new_array])
                case "json":
                    # if we are saving to json file, we need to 
                    # old_dict[key] += new_dict[key] for key in keys
                    if final_dict is None:
                        final_dict = {key:new_array[key] for key in new_array.keys()}
                    else:
                        for key in final_dict.keys():
                            final_dict[key] += new_array[key]
                case _:
                    raise RuntimeError(f'Illegal output format `{save_as}`.')
               
        match save_as:
            case "root":
                check_columns(expected_columns, final_array.fields, final_array.fields)
                with uproot.recreate(outFileName, compression=uprootCompression) as outfile:
                    outfile[treeName] = final_array
            case "json":
                with open(outFileName, "w") as f:
                    json.dump(final_dict, f, indent=4)
            case _:
                raise RuntimeError(f'Illegal output format `{save_as}`.')
    else:
        dfw = producer.run(dfw)
        check_columns(expected_columns, dfw.colToSave, dfw.df.GetColumnNames())
        varToSave = Utilities.ListToVector(dfw.colToSave)
        dfw.df.Snapshot(treeName, outFileName, varToSave, snapshotOptions)


def merge_cache_files(inFileName, cacheFileNames, treeName):
    dfw = Utilities.DataFrameBuilderBase(ROOT.RDataFrame(treeName, inFileName))
    if len(cacheFileNames) == 0:
        return dfw.df
    for i, cacheFileName in enumerate(cacheFileNames):
        cache = Utilities.DataFrameBuilderBase(ROOT.RDataFrame(treeName, cacheFileName))
        AddCacheColumnsInDf(dfw, cache, f"cache_map_Central_{i}", f"_cache_entry_{i}")
    return dfw.df


# add extra argument to this function - which payload producer to run
def createAnalysisCache(
    inFileName,
    outFileName,
    unc_cfg_dict,
    global_cfg_dict,
    snapshotOptions,
    compute_unc_variations,
    deepTauVersion,
    producer_to_run,
    uprootCompression,
    workingDir,
    cacheFileNames="",
):
    start_time = datetime.datetime.now()
    verbosity = ROOT.Experimental.RLogScopedVerbosity(
        ROOT.Detail.RDF.RDFLogChannel(), ROOT.Experimental.ELogLevel.kInfo
    )
    snaps = []
    all_files = []
    file_keys = getKeyNames(inFileName)

    df = merge_cache_files(inFileName, cacheFileNames, "Events")
    dfw = Utilities.DataFrameWrapper(df, defaultColToSave)

    # df = ROOT.RDataFrame('Events', inFileName)
    # df_begin = df
    # dfw = Utilities.DataFrameWrapper(df_begin,defaultColToSave)

    if not producer_to_run:
        raise RuntimeError("Producer must be specified to compute analysis cache")

    producer_config = global_cfg_dict["payload_producers"][producer_to_run]
    producers_module_name = producer_config["producers_module_name"]
    producer_name = producer_config["producer_name"]
    producers_module = importlib.import_module(producers_module_name)
    producer_class = getattr(producers_module, producer_name)
    producer = producer_class(producer_config, producer_to_run)
    output_file_extension = producer_config.get("save_as", "root")
    run_producer(
        producer,
        dfw,
        producer_config,
        f"{outFileName}_Central.{output_file_extension}",
        "Events",
        snapshotOptions,
        uprootCompression,
        workingDir,
    )
    all_files.append(f"{outFileName}_Central.{output_file_extension}")

    if compute_unc_variations:
        dfWrapped_central = Utilities.DataFrameBuilderBase(df)
        colNames = dfWrapped_central.colNames
        colTypes = dfWrapped_central.colTypes
        dfWrapped_central.df = createCentralQuantities(df, colTypes, colNames)
        if dfWrapped_central.df.Filter("map_placeholder > 0").Count().GetValue() <= 0:
            raise RuntimeError("no events passed map placeolder")
        print("finished defining central quantities")
        for uncName in unc_cfg_dict["shape"].keys():
            for scale in scales:
                treeName = f"Events_{uncName}{scale}"
                treeName_noDiff = f"{treeName}_noDiff"
                if treeName_noDiff in file_keys:
                    df_noDiff = merge_cache_files(
                        inFileName, cacheFileNames, treeName_noDiff
                    )
                    dfWrapped_noDiff = Utilities.DataFrameBuilderBase(df_noDiff)
                    dfWrapped_noDiff.CreateFromDelta(colNames, colTypes)
                    dfWrapped_noDiff.AddMissingColumns(colNames, colTypes)
                    dfW_noDiff = Utilities.DataFrameWrapper(
                        dfWrapped_noDiff.df, defaultColToSave
                    )
                    run_producer(
                        producer,
                        dfW_noDiff,
                        producer_config,
                        f"{outFileName}_{uncName}{scale}_noDiff.{output_file_extension}",
                        treeName_noDiff,
                        snapshotOptions,
                        uprootCompression,
                        workingDir,
                    )
                    all_files.append(f"{outFileName}_{uncName}{scale}_noDiff.{output_file_extension}")
                treeName_Valid = f"{treeName}_Valid"
                if treeName_Valid in file_keys:
                    df_Valid = merge_cache_files(
                        inFileName, cacheFileNames, treeName_Valid
                    )
                    dfWrapped_Valid = Utilities.DataFrameBuilderBase(df_Valid)
                    dfWrapped_Valid.CreateFromDelta(colNames, colTypes)
                    dfWrapped_Valid.AddMissingColumns(colNames, colTypes)
                    dfW_Valid = Utilities.DataFrameWrapper(
                        dfWrapped_Valid.df, defaultColToSave
                    )
                    run_producer(
                        producer,
                        dfW_Valid,
                        producer_config,
                        f"{outFileName}_{uncName}{scale}_Valid.{output_file_extension}",
                        treeName_Valid,
                        snapshotOptions,
                        uprootCompression,
                        workingDir,
                    )
                    all_files.append(f"{outFileName}_{uncName}{scale}_Valid.{output_file_extension}")
                treeName_nonValid = f"{treeName}_nonValid"
                if treeName_nonValid in file_keys:
                    df_nonValid = merge_cache_files(
                        inFileName, cacheFileNames, treeName_nonValid
                    )
                    dfWrapped_nonValid = Utilities.DataFrameBuilderBase(df_nonValid)
                    dfWrapped_nonValid.AddMissingColumns(colNames, colTypes)
                    dfW_nonValid = Utilities.DataFrameWrapper(
                        dfWrapped_nonValid.df, defaultColToSave
                    )
                    run_producer(
                        producer,
                        dfW_nonValid,
                        producer_config,
                        f"{outFileName}_{uncName}{scale}_nonValid.{output_file_extension}",
                        treeName_nonValid,
                        snapshotOptions,
                        uprootCompression,
                        workingDir,
                    )
                    all_files.append(f"{outFileName}_{uncName}{scale}_nonValid.{output_file_extension}")
    return all_files


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--inFileName", required=True, type=str)
    parser.add_argument("--outFileName", required=True, type=str)
    parser.add_argument("--uncConfig", required=True, type=str)
    parser.add_argument("--globalConfig", required=True, type=str)
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compressionLevel", type=int, default=4)
    parser.add_argument("--compressionAlgo", type=str, default="ZLIB")
    parser.add_argument("--deepTauVersion", type=str, default="v2p1")
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--producer", type=str, default=None)
    parser.add_argument("--workingDir", required=True, type=str)
    parser.add_argument("--cacheFileNames", required=False, type=str)
    args = parser.parse_args()

    ana_path = os.environ["ANALYSIS_PATH"]
    sys.path.append(ana_path)
    # headers = [ "FLAF/include/KinFitInterface.h", "FLAF/include/HistHelper.h", "FLAF/include/Utilities.h" ]
    headers = ["FLAF/include/HistHelper.h", "FLAF/include/Utilities.h", "FLAF/include/AnalysisTools.h"]
    for header in headers:
        DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = True
    snapshotOptions.fLazy = False
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(
        ROOT.ROOT, "k" + args.compressionAlgo
    )
    snapshotOptions.fCompressionLevel = args.compressionLevel
    unc_cfg_dict = {}

    uprootCompression = getattr(uproot, args.compressionAlgo)
    uprootCompression = uprootCompression(args.compressionLevel)

    with open(args.uncConfig, "r") as f:
        unc_cfg_dict = yaml.safe_load(f)

    global_cfg_dict = {}
    with open(args.globalConfig, "r") as f:
        global_cfg_dict = yaml.safe_load(f)

    producer_config = global_cfg_dict["payload_producers"][args.producer]
    save_as = producer_config.get("save_as", "root")

    startTime = time.time()
    if args.channels:
        global_cfg_dict["channelSelection"] = (
            args.channels.split(",") if type(args.channels) == str else args.channels
        )
    outFileNameFinal = f"{args.outFileName}.{save_as}" # outFileName comes from law task, I defined it without extention there
    cacheFileNames = args.cacheFileNames.split(",") if args.cacheFileNames else ""
    all_files = createAnalysisCache(
        args.inFileName,
        args.outFileName,
        unc_cfg_dict,
        global_cfg_dict,
        snapshotOptions,
        args.compute_unc_variations,
        args.deepTauVersion,
        args.producer,
        uprootCompression,
        args.workingDir,
        cacheFileNames,
    )
    match save_as:
        case "root":
            hadd_str = f"hadd -f209 -n10 {outFileNameFinal} "
            hadd_str += " ".join(f for f in all_files)
            if len(all_files) > 1:
                ps_call([hadd_str], True)
            else:
                shutil.copy(all_files[0], outFileNameFinal)
            if os.path.exists(outFileNameFinal):
                for histFile in all_files:
                    if histFile == outFileNameFinal:
                        continue
                    os.remove(histFile)
        case "json":
            # each root file has a bunch of uncertainty trees
            # I want to write event counts and weights from each unc tree to one big final json
            data = {}
            for file_name in all_files:
                # file_name = /afs/.../tmp_output_file_stuff.ext
                base_name = os.path.basename(file_name)
                # base_name = tmp_output_file_stuff.ext
                unc_name = '_'.join(base_name.split('_')[3:]) # extract stuff
                unc_name = unc_name.split('.')[0] # strip off file extension
                with open(file_name, 'r') as json_file:
                    file_data = json.load(json_file)
                    data[unc_name] = file_data

            print(f'About to write to {outFileNameFinal}')
            with open(outFileNameFinal, "w") as final_output_file:
                json.dump(data, final_output_file, indent=4)
        case _:
            raise RuntimeError(f'Illegal output format `{save_as}`.')
    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
