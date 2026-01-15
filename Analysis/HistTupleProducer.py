import time
import os
import sys
import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
from FLAF.RunKit.run_tools import ps_call
from FLAF.Common.HistHelper import *
from Corrections.Corrections import Corrections
import FLAF.Common.triggerSel as Triggers
import FLAF.Common.BaselineSelection as Baseline

# ROOT.EnableImplicitMT(1)
ROOT.EnableThreadSafety()


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
    *,
    setup,
    dataset_name,
    inFile,
    cacheFiles,
    treeName,
    hist_cfg_dict,
    unc_cfg_dict,
    snapshotOptions,
    range,
    evtIds,
    histTupleDef,
    inFile_keys,
):
    Baseline.Initialize(False, False)
    if dataset_name == "data":
        dataset_cfg = {}
        process_name = "data"
        process = {}
        isData = True
        processors_cfg = {}
        processor_instances = {}
    else:
        dataset_cfg = setup.datasets[dataset_name]
        process_name = dataset_cfg["process_name"]
        process = setup.base_processes[process_name]
        isData = dataset_cfg["process_group"] == "data"
        processors_cfg, processor_instances = setup.get_processors(
            process_name, stage="HistTuple", create_instances=True
        )
    triggerFile = setup.global_params.get("triggerFile")
    trigger_class = None
    if triggerFile is not None:
        triggerFile = os.path.join(os.environ["ANALYSIS_PATH"], triggerFile)
        trigger_class = Triggers.Triggers(triggerFile)

    Corrections.initializeGlobal(
        global_params=setup.global_params,
        stage="HistTuple",
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        process_name=process_name,
        process_cfg=process,
        processors=processor_instances,
        isData=isData,
        load_corr_lib=True,
        trigger_class=trigger_class,
    )

    histTupleDef.Initialize()
    histTupleDef.analysis_setup(setup)

    isCentral = True

    snaps = []
    outfilesNames = []
    variables = []
    tmp_fileNames = []
    if treeName not in inFile_keys:
        print(f"ERRORE, {treeName} non esiste nel file, ritorno il nulla")
        return tmp_fileNames

    df_central = ROOT.RDataFrame(treeName, inFile)
    df_cache_central = []
    if cacheFiles:
        for cacheFile in cacheFiles:
            df_cache_central.append(ROOT.RDataFrame(treeName, cacheFile))

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

    dfw_central = histTupleDef.GetDfw(df_central, df_cache_central, setup.global_params)

    col_names_central = dfw_central.colNames
    col_types_central = dfw_central.colTypes

    all_rel_uncs_to_compute = []
    if setup.global_params["compute_rel_weights"]:
        all_rel_uncs_to_compute.extend(unc_cfg_dict["norm"].keys())
    all_shifts_to_compute = []
    if setup.global_params["compute_unc_variations"]:
        df_central = createCentralQuantities(
            df_central, col_types_central, col_names_central
        )
        if df_central.Filter("map_placeholder > 0").Count().GetValue() <= 0:
            raise RuntimeError("no events passed map placeolder")
        all_shifts_to_compute.extend(unc_cfg_dict["shape"].keys())

    for unc in ["Central"] + all_rel_uncs_to_compute:
        scales = setup.global_params["scales"] if unc != "Central" else ["Central"]
        for scale in scales:
            final_weight_name = (
                f"weight_{unc}_{scale}" if unc != "Central" else "weight_Central"
            )
            histTupleDef.DefineWeightForHistograms(
                dfw=dfw_central,
                isData=isData,
                uncName=unc,
                uncScale=scale,
                unc_cfg_dict=unc_cfg_dict,
                hist_cfg_dict=hist_cfg_dict,
                global_params=setup.global_params,
                final_weight_name=final_weight_name,
                df_is_central=True,
            )
            dfw_central.colToSave.append(final_weight_name)

    # Return a flattened set of variables, the 2D happens later
    flatten_vars = set()
    for var in variables:
        if isinstance(var, dict) and "vars" in var:
            for v in var["vars"]:
                flatten_vars.add(v)
        else:
            flatten_vars.add(var)

    for var in flatten_vars:
        DefineBinnedColumn(hist_cfg_dict, var)
        dfw_central.df = dfw_central.df.Define(f"{var}_bin", f"get_{var}_bin({var})")
        dfw_central.colToSave.append(f"{var}_bin")

    varToSave = Utilities.ListToVector(list(set(dfw_central.colToSave)))
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
            shifts = ["noDiff", "Valid", "nonValid"]

            for shift in shifts:
                treeName_shift = f"{treeName}_{shift}"
                print(treeName_shift)

                if treeName_shift in inFile_keys:
                    df_shift_caches = []
                    if cacheFiles:
                        for cacheFile in cacheFiles:
                            df_shift_caches.append(
                                ROOT.RDataFrame(treeName_shift, cacheFile)
                            )

                    dfw_shift = histTupleDef.GetDfw(
                        ROOT.RDataFrame(treeName_shift, inFile),
                        df_shift_caches,
                        setup.global_params,
                        shift,
                        col_names_central,
                        col_types_central,
                        f"cache_map_{unc}{scale}_{shift}",
                    )
                    final_weight_name = "weight_Central"

                    histTupleDef.DefineWeightForHistograms(
                        dfw=dfw_shift,
                        isData=isData,
                        uncName=unc,
                        uncScale=scale,
                        unc_cfg_dict=unc_cfg_dict,
                        hist_cfg_dict=hist_cfg_dict,
                        global_params=setup.global_params,
                        final_weight_name=final_weight_name,
                        df_is_central=False,
                    )
                    dfw_shift.colToSave.append(final_weight_name)
                    for var in flatten_vars:
                        dfw_shift.df = dfw_shift.df.Define(
                            f"{var}_bin", f"get_{var}_bin({var})"
                        )
                        dfw_shift.colToSave.append(f"{var}_bin")

                    varToSave = Utilities.ListToVector(list(set(dfw_shift.colToSave)))

                    tmp_fileName = f"{treeName_shift}.root"
                    tmp_fileNames.append(tmp_fileName)

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
    return tmp_fileNames


def createVoidTree(file_name, tree_name):
    df = ROOT.RDataFrame(0)
    df = df.Define("test", "return true;")
    df.Snapshot(tree_name, file_name, {"test"})


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
    parser.add_argument("--compressionLevel", type=int, default=4)
    parser.add_argument("--compressionAlgo", type=str, default="ZLIB")
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--nEvents", type=int, default=None)
    parser.add_argument("--evtIds", type=str, default="")

    args = parser.parse_args()
    startTime = time.time()
    setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], args.period)

    treeName = setup.global_params[
        "treeName"
    ]  # treeName should be inside global params if not in customisations

    channels = setup.global_params["channelSelection"]
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
    cacheFiles = None
    if args.cacheFiles:
        cacheFiles = args.cacheFiles.split(",")
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
            setup=setup,
            dataset_name=args.dataset,
            inFile=args.inFile,
            cacheFiles=cacheFiles,
            treeName=treeName,
            hist_cfg_dict=hist_cfg_dict,
            unc_cfg_dict=unc_cfg_dict,
            snapshotOptions=snapshotOptions,
            range=args.nEvents,
            evtIds=args.evtIds,
            histTupleDef=histTupleDef,
            inFile_keys=inFile_keys,
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
    else:
        print(f"NO HISTOGRAM CREATED!!!! dataset: {args.dataset} ")
        createVoidTree(args.outFile, f"Events")

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
