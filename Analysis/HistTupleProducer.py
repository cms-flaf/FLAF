import time
import os
import sys
import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
from FLAF.RunKit.run_tools import ps_call

from FLAF.Common.HistHelper import findBinEntry
from Corrections.CorrectionsCore import getScales, central
from Corrections.Corrections import Corrections

import FLAF.Common.triggerSel as Triggers
import FLAF.Common.BaselineSelection as Baseline

# ROOT.EnableImplicitMT(1)
ROOT.EnableThreadSafety()


def DefineBinnedColumn(hist_cfg_dict, var):
    var_entry = findBinEntry(hist_cfg_dict, var)
    x_bins = hist_cfg_dict[var_entry]["x_bins"]
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

    ROOT.gInterpreter.Declare(f"""
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
        """)


def createHistTuple(
    *,
    setup,
    dataset_name,
    inFileName,
    cacheFileNames,
    snapshotOptions,
    range,
    evtIds,
    histTupleDef,
):
    treeName = setup.global_params.get("treeName", "Events")
    unc_cfg_dict = setup.weights_config
    hist_cfg_dict = setup.hists
    Utilities.InitializeCorrections(setup, dataset_name, stage="HistTuple")
    histTupleDef.Initialize()
    histTupleDef.analysis_setup(setup)
    isData = dataset_name == "data"

    if type(setup.global_params["variables"]) == list:
        variables = setup.global_params["variables"]
    elif type(setup.global_params["variables"]) == dict:
        variables = setup.global_params["variables"].keys()

    norm_uncertainties = set()
    if setup.global_params["compute_rel_weights"]:
        norm_uncertainties.update(unc_cfg_dict["norm"].keys())
    print("Norm uncertainties to consider:", norm_uncertainties)
    scale_uncertainties = set()
    if setup.global_params["compute_unc_variations"]:
        scale_uncertainties.update(unc_cfg_dict["shape"].keys())
    print("Scale uncertainties to consider:", scale_uncertainties)

    print("Defining binnings for variables")
    flatten_vars = set()
    for var in variables:
        if isinstance(var, dict) and "vars" in var:
            for v in var["vars"]:
                flatten_vars.add(v)
        else:
            flatten_vars.add(var)

    for var in flatten_vars:
        DefineBinnedColumn(hist_cfg_dict, var)

    snaps = []
    tmp_fileNames = []

    centralTree = None
    centralCaches = None
    allRootFiles = {}
    for unc_source in [central] + list(scale_uncertainties):
        for unc_scale in getScales(unc_source):
            print(f"Processing events for {unc_source} {unc_scale}")
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
                filter_valid=True,
            )
            if isCentral:
                centralTree = tree
                centralCaches = cacheTrees
            ROOT.RDF.Experimental.AddProgressBar(df_orig)

            if range is not None:
                df = df.Range(range)
            if evtIds and len(evtIds) > 0:
                df = df.Filter(
                    f"static const std::set<ULong64_t> evts = {{ {evtIds} }}; return evts.count(event) > 0;"
                )

            dfw = histTupleDef.GetDfw(df, setup, dataset_name)
            iter_descs = [
                {"source": unc_source, "scale": unc_scale, "weight": "weight_Central"}
            ]
            if isCentral:
                for unc_source_norm in norm_uncertainties:
                    for unc_scale_norm in getScales(unc_source_norm):
                        iter_descs.append(
                            {
                                "source": unc_source_norm,
                                "scale": unc_scale_norm,
                                "weight": f"weight_{unc_source_norm}_{unc_scale_norm}",
                            }
                        )
            for desc in iter_descs:
                print(f"Defining the final weight for {desc['source']} {desc['scale']}")
                histTupleDef.DefineWeightForHistograms(
                    dfw=dfw,
                    isData=isData,
                    uncName=desc["source"],
                    uncScale=desc["scale"],
                    unc_cfg_dict=unc_cfg_dict,
                    hist_cfg_dict=hist_cfg_dict,
                    global_params=setup.global_params,
                    final_weight_name=desc["weight"],
                    df_is_central=isCentral,
                )
                dfw.colToSave.append(desc["weight"])

            print("Defining binned columns")
            for var in flatten_vars:
                dfw.df = dfw.df.Define(f"{var}_bin", f"get_{var}_bin({var})")
                dfw.colToSave.append(f"{var}_bin")

            varToSave = Utilities.ListToVector(list(set(dfw.colToSave)))
            tmp_fileName = f"{fullTreeName}.root"
            tmp_fileNames.append(tmp_fileName)
            print("Creating snapshot")
            snaps.append(
                dfw.df.Snapshot(fullTreeName, tmp_fileName, varToSave, snapshotOptions)
            )

    if snapshotOptions.fLazy == True:
        ROOT.RDF.RunGraphs(snaps)

    return tmp_fileNames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True, type=str)
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--cacheFiles", required=False, type=str, default=None)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--histTupleDef", required=True, type=str)
    parser.add_argument("--compute_unc_variations", type=bool, default=False)
    parser.add_argument("--compute_rel_weights", type=bool, default=False)
    parser.add_argument("--customisations", type=str, default=None)
    parser.add_argument("--compressionLevel", type=int, default=9)
    parser.add_argument("--compressionAlgo", type=str, default="LZMA")
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--nEvents", type=int, default=None)
    parser.add_argument("--evtIds", type=str, default=None)

    args = parser.parse_args()
    startTime = time.time()

    ROOT.gROOT.ProcessLine(".include " + os.environ["FLAF_PATH"])
    ROOT.gROOT.ProcessLine('#include "include/Utilities.h"')

    setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], args.period)

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

    setup.global_params["compute_rel_weights"] = (
        args.compute_rel_weights and process_group != "data"
    )
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

    histTupleDef = Utilities.load_module(args.histTupleDef)

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = False
    snapshotOptions.fLazy = False
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(
        ROOT.ROOT.RCompressionSetting.EAlgorithm, "k" + args.compressionAlgo
    )
    snapshotOptions.fCompressionLevel = args.compressionLevel

    tmp_fileNames = createHistTuple(
        setup=setup,
        dataset_name=args.dataset,
        inFileName=args.inFile,
        cacheFileNames=cacheFileNames,
        snapshotOptions=snapshotOptions,
        range=args.nEvents,
        evtIds=args.evtIds,
        histTupleDef=histTupleDef,
    )
    hadd_cmd = ["hadd", "-j", "-ff", args.outFile]
    hadd_cmd.extend(tmp_fileNames)
    ps_call(hadd_cmd, verbose=1)
    if os.path.exists(args.outFile) and len(tmp_fileNames) != 0:
        for file_syst in tmp_fileNames:
            if file_syst != args.outFile:
                os.remove(file_syst)

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
