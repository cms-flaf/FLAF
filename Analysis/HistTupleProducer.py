import time
import os
import sys
import ROOT
import json

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

cat_to_channelId = {"e": 1, "mu": 2, "eE": 11, "eMu": 12, "muMu": 22}


class BtagShapeWeightCorrector:
    def __init__(self, btag_integral_ratios):
        self.exisiting_srcScale_combs = [key for key in btag_integral_ratios.keys()]
        # if the btag_integral_ratios dictionary is not empty, do stuff
        if self.exisiting_srcScale_combs:
            ROOT.gInterpreter.Declare("#include <map>")

            for key in btag_integral_ratios.keys():
                # key in btag_integral_ratios has form f"{source}_{scale}", so function expects that
                # and creates a map and function to rescale btag weights for each f"{source}_{scale}" value
                self._declare_cpp_map_and_resc_func(btag_integral_ratios, key)

    def _declare_cpp_map_and_resc_func(self, btag_integral_ratios, unc_src_scale):
        correction_factors = btag_integral_ratios[unc_src_scale]

        # init c++ map
        cpp_map_entries = []
        for cat, multipl_dict in correction_factors.items():
            channelId = cat_to_channelId[cat]
            for key, ratio in multipl_dict.items():
                # key has structure f"ratio_ncetnralJet_{number}""
                num_jet = int(key.split("_")[-1])
                cpp_map_entries.append(f"{{{{{channelId}, {num_jet}}}, {ratio}}}")
        cpp_init = ", ".join(cpp_map_entries)

        ROOT.gInterpreter.Declare(
            f"""
            static const std::map<std::pair<int, int>, float> ratios_{unc_src_scale} = {{
                {cpp_init}
            }};

            float integral_correction_ratio_{unc_src_scale}(int ncentralJet, int channelId) {{
                std::pair<int, int> key{{channelId, ncentralJet}};
                try 
                {{
                    float ratio = ratios_{unc_src_scale}.at(key);
                    return ratio;
                }}
                catch (...)
                {{
                    return 1.0f;
                }}
            }}"""
        )

    def UpdateBtagWeight(self, dfw, unc_src="Central", unc_scale=None):
        # return original dfw if empty dict was passed to constructor
        if not self.exisiting_srcScale_combs:
            return dfw

        if unc_scale is None:
            unc_src_scale = unc_src
        else:
            unc_src_scale = f"{unc_src}_{unc_scale}"

        if unc_src_scale not in self.exisiting_srcScale_combs:
            raise RuntimeError(
                f"`BtagShapeWeightCorrection.json` does not contain key `{unc_src_scale}`."
            )

        dfw.df = dfw.df.Redefine(
            "weight_bTagShape_Central",
            f"""if (ncentralJet >= 2 && ncentralJet <= 8) 
                    return integral_correction_ratio_{unc_src_scale}(ncentralJet, channelId)*weight_bTagShape_Central;
                return weight_bTagShape_Central;""",
        )

        return dfw


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
    btagIntegralRatios,
    isData,
):
    treeName = setup.global_params.get("treeName", "Events")
    unc_cfg_dict = setup.weights_config
    hist_cfg_dict = setup.hists
    Utilities.InitializeCorrections(setup, dataset_name, stage="HistTuple")
    histTupleDef.Initialize()
    histTupleDef.analysis_setup(setup)
    isData = dataset_name == "data"

    # here correction to btag weights is applied to ensure that application of btag shape weights
    # does not modify the integral
    # if empty btagIntegralRatios passed, UpdateBtagWeight will do nothing
    weight_corrector = BtagShapeWeightCorrector(btagIntegralRatios)
    isMC = not isData

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

            # for now only evaluate central values
            if isCentral and isMC:
                print(
                    f"Calling weight_corrector.UpdateBtagWeight for unc_source={unc_source} unc_scale={unc_scale}"
                )
                weight_corrector.UpdateBtagWeight(dfw, unc_src=unc_source)

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
    parser.add_argument("--btagCorrectionsJson", type=str, default="")

    args = parser.parse_args()
    startTime = time.time()

    btagIntegralRatios = {}
    if args.btagCorrectionsJson:
        with open(args.btagCorrectionsJson, "r") as file:
            btagIntegralRatios = json.load(file)

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

    isData = process_group == "data"

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
        isData=isData,
        btagIntegralRatios=btagIntegralRatios,
    )
    hadd_cmd = ["hadd", "-j", args.outFile]
    hadd_cmd.extend(tmp_fileNames)
    ps_call(hadd_cmd, verbose=1)
    if os.path.exists(args.outFile) and len(tmp_fileNames) != 0:
        for file_syst in tmp_fileNames:
            if file_syst != args.outFile:
                os.remove(file_syst)

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
