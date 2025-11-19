import time
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

class BtagShapeWeightCorrector:
    def __init__(self, btag_integral_ratios):
        self.exisiting_srcScale_combs = [key for key in btag_integral_ratios.keys()]
        # if the btag_integral_ratios dictionary is not empty, do stuff
        if self.exisiting_srcScale_combs:
            ROOT.gInterpreter.Declare("#include <map>")

            for key in btag_integral_ratios.keys():
                # key in btag_integral_ratios has form f"{source}{scale}", so function expects that
                # and creates a map and function to rescale btag weights for each f"{source}{scale}" value
                self._declare_cpp_map_and_resc_func(btag_integral_ratios, key)

    def _declare_cpp_map_and_resc_func(self, btag_integral_ratios, unc_src_scale):
        correction_factors = btag_integral_ratios[unc_src_scale]
    
        # init c++ map
        cpp_map_entries = []
        for cat, multipl_dict in correction_factors.items():
            channelId = cat_to_channelId[cat]
            for key, ratio in multipl_dict.items():
                # key has structure f"ratio_ncetnralJet_{number}""
                num_jet = int(key.split('_')[-1])
                cpp_map_entries.append(f'{{{{{channelId}, {num_jet}}}, {ratio}}}')
        cpp_init = ", ".join(cpp_map_entries)

        ROOT.gInterpreter.Declare(f"""
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
            unc_scale = ""
        unc_src_scale = f"{unc_src}{unc_scale}"
        
        if unc_src_scale not in self.exisiting_srcScale_combs:
            raise RuntimeError(f"`BtagShapeWeight.json` does not contain key `{unc_src_scale}`.")
        
        dfw.df = (
            dfw.df
            .Filter("ncentralJet >= 2 && ncentralJet <= 8")
            .Redefine("weight_bTagShape_Central", f"return integral_correction_ratio_{unc_src_scale}(ncentralJet, channelId)*weight_bTagShape_Central;")
        )

        return dfw     

cat_to_channelId = {
    "e": 1,
    "mu": 2,
    "eE": 11,
    "eMu": 12,
    "muMu": 22 
}

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
    cacheFiles,
    treeName,
    setup,
    hist_cfg_dict,
    unc_cfg_dict,
    snapshotOptions,
    range,
    evtIds,
    histTupleDef,
    inFile_keys,
    btag_integral_ratios
):
    # compression_settings = snapshotOptions.fCompressionAlgorithm * 100 + snapshotOptions.fCompressionLevel
    histTupleDef.Initialize()
    histTupleDef.analysis_setup(setup)

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

    # here correction to btag weights is applied to ensure that application of btag shape weights
    # does not modify the integral
    # if empty btag_integral_ratios passed, UpdateBtagWeight will do nothing
    weight_corrector = BtagShapeWeightCorrector(btag_integral_ratios)
    dfw_central = weight_corrector.UpdateBtagWeight(dfw_central)

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
                dfw_central,
                unc,
                scale,
                unc_cfg_dict,
                hist_cfg_dict,
                setup.global_params,
                final_weight_name,
            )
            dfw_central.colToSave.append(final_weight_name)
    for var in variables:
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

                    dfw_shift = weight_corrector.UpdateBtagWeight(dfw_shift, unc_src=unc, unc_scale=scale)

                    histTupleDef.DefineWeightForHistograms(
                        dfw_shift,
                        unc,
                        scale,
                        unc_cfg_dict,
                        hist_cfg_dict,
                        setup.global_params,
                        final_weight_name,
                    )
                    dfw_shift.colToSave.append(final_weight_name)
                    for var in variables:
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
    import os
    import yaml

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
    parser.add_argument("--btag_json", type=str, default="")

    args = parser.parse_args()
    startTime = time.time()
    setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], args.period, "")

    btag_integral_ratios = {}
    if args.btag_json:
        with open(args.btag_json, 'r') as file:
            btag_integral_ratios = json.load(file)

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
        setup.samples[args.dataset]["process_name"]
        if args.dataset != "data"
        else "data"
    )
    setup.global_params["process_name"] = process_name
    process_group = (
        setup.samples[args.dataset]["process_group"]
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
    histTupleDef = Utilities.load_module(args.histTupleDef)
    cacheFiles = None
    if args.cacheFiles:
        cacheFiles = args.cacheFiles.split(",")
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
            cacheFiles,
            treeName,
            setup,
            hist_cfg_dict,
            unc_cfg_dict,
            snapshotOptions,
            args.nEvents,
            args.evtIds,
            histTupleDef,
            inFile_keys,
            btag_integral_ratios
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
