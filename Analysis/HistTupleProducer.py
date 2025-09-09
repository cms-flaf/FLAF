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

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
import importlib
from FLAF.RunKit.run_tools import ps_call
from FLAF.Common.HistHelper import *
from FLAF.Common.Utilities import getCustomisationSplit

# ROOT.EnableImplicitMT(1)
ROOT.EnableThreadSafety()


def create_file(file_name, times=None):
    with open(file_name, "w"):
        os.utime(file_name, times)


def DefineBinnedColumn(hist_cfg_dict, var):
    x_bins = hist_cfg_dict[var]["x_bins"]
    func_name = f"get_{var}_bin"
    axis_definition = ""

    if isinstance(x_bins, list):
        edges = x_bins
        n_bins = len(edges) - 1
        edges_cpp = "{" + ",".join(map(str, edges)) + "}"
        axis_definition = f"static const double bins[] = {edges_cpp}; static const TAxis axis({n_bins}, bins);"
    else:
        n_bins, bin_range = x_bins.split("|")
        start, stop = bin_range.split(":")
        axis_definition = f"static const TAxis axis({n_bins}, {start}, {stop});"

    ROOT.gInterpreter.Declare(
        f"""
        #include "ROOT/RVec.hxx"
        #include "TAxis.h"

        int {func_name}(double x) {{
            {axis_definition}
            return axis.FindFixBin(x) - 1;
        }}

        template<typename T>
        ROOT::VecOps::RVec<int> {func_name}(ROOT::VecOps::RVec<T> xvec) {{
            {axis_definition}
            ROOT::VecOps::RVec<int> out(xvec.size());
            for (size_t i = 0; i < xvec.size(); ++i) {{
                out[i] = axis.FindFixBin(xvec[i]) - 1;
            }}
            return out;
        }}
        """
    )


def createHistTuple(
    inFile,
    cacheFile,
    treeName,
    setup,
    hist_cfg_dict,
    unc_cfg_dict,
    snapshotOptions,
    range,
    evtIds,
    compute_unc_variations,
    compute_rel_weights,
    histTupleDef,
    inFile_keys,
):
    start_time = datetime.datetime.now()
    # compression_settings = snapshotOptions.fCompressionAlgorithm * 100 + snapshotOptions.fCompressionLevel
    histTupleDef.Initialize()
    histTupleDef.analysis_setup(setup)
    isData = True if setup.global_params["dataset"] == "data" else False
    isCentral = True
    sample_type = (
        "data"
        if isData
        else setup.samples[setup.global_params["dataset"]]["sampleType"]
    )
    snaps = []
    reports = []
    outfilesNames = []
    variables = []
    tmp_fileNames = []
    if treeName not in inFile_keys:
        print(f"ERRORE, {treeName} non esiste nel file, ritorno il nulla")
        return tmp_fileNames

    df_central = ROOT.RDataFrame(treeName, inFile)
    df_cache_central = None
    if cacheFile:
        df_cache_central = ROOT.RDataFrame(treeName, cacheFile)

    ROOT.RDF.Experimental.AddProgressBar(df_central)
    if range is not None:
        df_central = df_central.Range(range)
    if len(evtIds) > 0:
        df_central = df_central.Filter(
            f"static const std::set<ULong64_t> evts = {{ {evtIds} }}; return evts.count(event) > 0;"
        )

    # Central + weights shifting:

    if type(setup.global_params["variables"]) == list:
        variables = setup.global_params["variables"]
    elif type(setup.global_params["variables"]) == dict:
        variables = setup.global_params["variables"].keys()
    additional_vars = []
    if "additional_vars" in setup.global_params.keys():
        if type(setup.global_params["additional_vars"]) == list:
            additional_vars = setup.global_params["additional_vars"]
        elif type(setup.global_params["additional_vars"]) == dict:
            additional_vars = setup.global_params["additional_vars"].keys()

    dfw_central = histTupleDef.GetDfw(df_central, df_cache_central, setup.global_params)

    col_names_central = dfw_central.colNames
    col_types_central = dfw_central.colTypes

    all_rel_uncs_to_compute = []
    if compute_rel_weights:
        all_rel_uncs_to_compute.extend(unc_cfg_dict["norm"].keys())
    all_shifts_to_compute = []
    if compute_unc_variations:
        df_central = createCentralQuantities(
            df_central, col_types_central, col_names_central
        )
        # If original count is already 0, then you don't need to raise error
        if df_central.Filter("map_placeholder > 0").Count().GetValue() <= 0:
            raise RuntimeError("no events passed map placeolder")
        all_shifts_to_compute.extend(unc_cfg_dict["shape"])

    for unc in ["Central"] + all_rel_uncs_to_compute:
        scales = setup.global_params["scales"] if unc != "Central" else ["Central"]
        for scale in scales:
            final_weight_name = (
                f"weight_{unc}_{scale}" if unc != "Central" else "weight_Central"
            )
            histTupleDef.DefineWeightForHistograms(
                dfw_central,
                unc,
                scale,
                sample_type,
                unc_cfg_dict,
                hist_cfg_dict,
                setup.global_params,
                final_weight_name,
            )
            dfw_central.colToSave.append(final_weight_name)
    dfw_central.colToSave.extend(additional_vars)
    for var in variables:
        DefineBinnedColumn(hist_cfg_dict, var)
        dfw_central.df = dfw_central.df.Define(f"{var}_bin", f"get_{var}_bin({var})")
        dfw_central.colToSave.append(f"{var}_bin")

    varToSave = Utilities.ListToVector(dfw_central.colToSave)
    # reports.append(dfw_central.df.Report()) --> enable if we want reports
    tmp_fileName = f"{treeName}.root"
    tmp_fileNames.append(tmp_fileName)
    snaps.append(
        dfw_central.df.Snapshot(treeName, tmp_fileName, varToSave, snapshotOptions)
    )

    #### shifted trees

    for unc in all_shifts_to_compute:
        scales = setup.global_params["scales"]
        for scale in scales:
            treeName = f"Events_{unc}{scale}"
            # Lista delle variazioni
            shifts = ["noDiff", "Valid", "nonValid"]

            for shift in shifts:
                treeName_shift = f"{treeName}_{shift}"
                print(treeName_shift)

                if treeName_shift in inFile_keys:
                    # Cache dataframe se disponibile
                    df_shift_cache = None
                    if cacheFile:
                        df_shift_cache = ROOT.RDataFrame(treeName_shift, cacheFile)

                    # Costruzione del dataframe wrapper
                    dfw_shift = histTupleDef.GetDfw(
                        ROOT.RDataFrame(treeName_shift, inFile),
                        df_shift_cache,
                        setup.global_params,
                        shift,
                        col_names_central,
                        col_types_central,
                        f"cache_map_{unc}{scale}_{shift}",
                    )
                    final_weight_name = "weight_Central"

                    # Definizione pesi
                    histTupleDef.DefineWeightForHistograms(
                        dfw_shift,
                        unc,
                        scale,
                        sample_type,
                        unc_cfg_dict,
                        hist_cfg_dict,
                        setup.global_params,
                        final_weight_name,
                    )
                    dfw_shift.colToSave.append(final_weight_name)
                    # Aggiunta colonne
                    # dfw.colToSave.append("weight_Central")
                    for var in variables:
                        dfw_shift.df = dfw_shift.df.Define(
                            f"{var}_bin", f"get_{var}_bin({var})"
                        )
                        dfw_shift.colToSave.append(f"{var}_bin")
                    dfw_shift.colToSave.extend(additional_vars)

                    # Conversione lista → ROOT::VecOps::RVec
                    varToSave = Utilities.ListToVector(dfw_shift.colToSave)

                    # Output temporaneo
                    tmp_fileName = f"{treeName_shift}.root"
                    tmp_fileNames.append(tmp_fileName)

                    # Snapshot
                    snaps.append(
                        dfw_shift.df.Snapshot(
                            treeName_shift,
                            tmp_fileName,
                            varToSave,
                            snapshotOptions,
                        )
                    )

    if snapshotOptions.fLazy == True:
        ROOT.RDF.RunGraphs(snaps)
    end_time = datetime.datetime.now()
    return tmp_fileNames


if __name__ == "__main__":
    import argparse
    import os
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--cacheFile", required=False, type=str, default=None)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--histTupleDef", required=True, type=str)
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compute_rel_weights", type=bool, default=False)
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--compressionLevel", type=int, default=4)
    parser.add_argument("--compressionAlgo", type=str, default="ZLIB")
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--nEvents", type=int, default=None)
    parser.add_argument("--evtIds", type=str, default="")

    args = parser.parse_args()

    setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], args.period, "")

    treeName = setup.global_params[
        "treeName"
    ]  # treeName should be inside global params if not in customisations

    channels = setup.global_params["channelSelection"]
    setup.global_params["channels_to_consider"] = (
        args.channels.split(",")
        if args.channels
        else setup.global_params["channelSelection"]
    )
    setup.global_params["dataset"] = args.dataset
    setup.global_params["compute_rel_weights"] = (
        args.compute_rel_weights and args.dataset != "data"
    )
    histTupleDef = Utilities.load_module(args.histTupleDef)

    dont_create_HistTuple = False
    key_not_exist = False
    df_empty = False
    inFile_root = ROOT.TFile.Open(args.inFile, "READ")
    inFile_keys = [k.GetName() for k in inFile_root.GetListOfKeys()]
    if treeName not in inFile_keys:
        key_not_exist = True
    inFile_root.Close()
    if (
        not key_not_exist
        and ROOT.RDataFrame(treeName, args.inFile).Count().GetValue() == 0
    ):
        df_empty = True
    dont_create_HistTuple = key_not_exist or df_empty

    unc_cfg_dict = setup.weights_config
    hist_cfg_dict = setup.hists

    histTupleDef = Utilities.load_module(args.histTupleDef)
    if not dont_create_HistTuple:
        snapshotOptions = ROOT.RDF.RSnapshotOptions()
        snapshotOptions.fOverwriteIfExists = False
        snapshotOptions.fLazy = True
        snapshotOptions.fMode = "RECREATE"
        # snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, 'k' + args.compressionAlgo)
        # snapshotOptions.fCompressionLevel = args.compressionLevel
        tmp_fileNames = createHistTuple(
            args.inFile,
            args.cacheFile,
            treeName,
            setup,
            hist_cfg_dict,
            unc_cfg_dict,
            snapshotOptions,
            args.nEvents,
            args.evtIds,
            args.compute_unc_variations,
            args.compute_rel_weights,
            histTupleDef,
            inFile_keys,
        )
        if tmp_fileNames:
            hadd_str = f"hadd -f -j -O {args.outFile} "
            hadd_str += " ".join(f for f in tmp_fileNames)
            print(f"hadd_str is {hadd_str}")
            ps_call([hadd_str], True)
            if os.path.exists(args.outFile) and len(tmp_fileNames) != 0:
                for file_syst in tmp_fileNames:
                    if file_syst == args.outFile:
                        continue
                    os.remove(file_syst)
