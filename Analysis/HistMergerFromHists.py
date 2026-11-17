import ROOT
import sys
import os
import time
import importlib
import json

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
import FLAF.Common.Setup as Setup
import FLAF.Common.TheoryUncertainties as TheoryUncertainties
from FLAF.Common.HistHelper import *
from Corrections.CorrectionsCore import getScales


def checkFile(inFileRoot, channels, qcdRegions, categories):
    keys_channels = [str(key.GetName()) for key in inFileRoot.GetListOfKeys()]
    for channel in channels:
        if channel not in keys_channels:
            return False
    for channel in channels:
        dir_0 = inFileRoot.Get(channel)
        keys_qcdRegions = [str(key.GetName()) for key in dir_0.GetListOfKeys()]
        if not all(element in keys_qcdRegions for element in qcdRegions):
            print("check list not worked for qcdRegions")
            return False
        for qcdRegion in qcdRegions:
            dir_1 = dir_0.Get(qcdRegion)
            keys_categories = [str(key.GetName()) for key in dir_1.GetListOfKeys()]
            if not all(element in keys_categories for element in categories):
                print("check list not worked for categories")
                return False
            for cat in categories:
                dir_2 = dir_1.Get(cat)
                keys_histograms = [str(key.GetName()) for key in dir_2.GetListOfKeys()]
                if not keys_histograms:
                    return False
    return True


def fill_hists(
    items_dict,
    all_hist_dict,
    dataset_type,
    var_input,
    unc_sources,
    data_type="data",
):
    """Accumulate the histograms of one input file for all requested unc sources in a
    single pass. Data has no per-uncertainty variations: only its nominal histogram is
    accumulated here; alias_data_variations() links it under the varied keys afterwards
    (QCD estimation looks data up per (unc, scale))."""
    target_keys = {var_input: ("Central", "Central")}
    if dataset_type != data_type:
        for unc_source in unc_sources:
            if unc_source == "Central":
                continue
            # Up/Down, or every member of a theory uncertainty (reduced after the sum)
            for scale in getScales(unc_source):
                target_keys[f"{var_input}_{unc_source}_{scale}"] = (unc_source, scale)
    hists = all_hist_dict.setdefault(dataset_type, {})
    for key_tuple, hist_map in items_dict.items():
        for var, var_hist in hist_map.items():
            unc_scale = target_keys.get(var)
            if unc_scale is None:
                continue
            final_key = (key_tuple, unc_scale)
            if final_key not in hists:
                var_hist.SetDirectory(0)
                hists[final_key] = var_hist
            else:
                hists[final_key].Add(var_hist)


def alias_data_variations(all_hist_dict, data_type, unc_sources):
    """Expose the accumulated nominal data histogram under every varied (unc, scale) key.
    Downstream consumers (QCD estimation) only read/clone these entries, so sharing the
    same object is safe; the output-writing loop skips data for non-Central sources."""
    data_hists = all_hist_dict.get(data_type)
    if not data_hists:
        return
    central = {
        key_tuple: hist
        for (key_tuple, unc_scale), hist in data_hists.items()
        if unc_scale == ("Central", "Central")
    }
    for unc_source in unc_sources:
        if unc_source == "Central":
            continue
        for scale in ["Up", "Down"]:
            for key_tuple, hist in central.items():
                data_hists.setdefault((key_tuple, (unc_source, scale)), hist)


def GetBTagWeightDict(
    var, all_hists_dict, categories, boosted_categories, boosted_variables
):
    all_hists_dict_1D = {}
    for dataset_type in all_hists_dict.keys():
        all_hists_dict_1D[dataset_type] = {}
        for key_name, histogram in all_hists_dict[dataset_type].items():
            key_1, key_2 = key_name

            if var not in boosted_variables:
                ch, reg, cat = key_1
                uncName, scale = key_2
                key_tuple_num = ((ch, reg, "btag_shape"), key_2)
                key_tuple_den = ((ch, reg, "inclusive"), key_2)
                ratio_num_hist = (
                    all_hists_dict[dataset_type][key_tuple_num]
                    if key_tuple_num in all_hists_dict[dataset_type].keys()
                    else None
                )
                ratio_den_hist = (
                    all_hists_dict[dataset_type][key_tuple_den]
                    if key_tuple_den in all_hists_dict[dataset_type].keys()
                    else None
                )
                num = ratio_num_hist.Integral(0, ratio_num_hist.GetNbinsX() + 1)
                den = ratio_den_hist.Integral(0, ratio_den_hist.GetNbinsX() + 1)
                ratio = 0.0
                if ratio_den_hist.Integral(0, ratio_den_hist.GetNbinsX() + 1) != 0:
                    ratio = ratio_num_hist.Integral(
                        0, ratio_num_hist.GetNbinsX() + 1
                    ) / ratio_den_hist.Integral(0, ratio_den_hist.GetNbinsX() + 1)
                if (
                    cat in boosted_categories
                    or cat.startswith("btag_shape")
                    or cat.startswith("baseline")
                ):
                    ratio = 1
                histogram.Scale(ratio)
            else:
                print(
                    f"for var {var} no ratio is considered and the histogram is directly saved"
                )

            all_hists_dict_1D[dataset_type][key_name] = histogram
    return all_hists_dict_1D


# 209 = LZMA level 9, the standard compression for merged histogram files
MERGED_COMPRESSION = 209


class MergeContext:
    """What every merge step of one variable needs from the configuration.

    The merge runs one process at a time (merge_process: the inputs of one process are summed
    and written into the output, then freed), followed by one post-processing step over the
    merged file (postprocess: the data-driven estimates, which need every process). Memory and
    local disk then hold one process at a time, whatever the number of processes.
    """

    def __init__(self, args):
        self.setup = Setup.Setup(
            os.environ["ANALYSIS_PATH"],
            args.era,
            args.LAWrunVersion,
            user_custom_file=args.user_custom,
        )
        setup = self.setup
        global_cfg_dict = setup.global_params
        unc_cfg_dict = setup.weights_config
        self.var = args.var
        self.analysis_import = global_cfg_dict["analysis_import"]
        importlib.import_module(f"{self.analysis_import}")
        self.estimateQCD = global_cfg_dict.get("plot_wantQCD", False)

        # Regions def: can be extended to a list of names, e.g. QCD regions + other control
        # regions
        regions_name = global_cfg_dict.get("regions", None)
        self.regions = []
        if regions_name:
            self.regions = list(global_cfg_dict.get(regions_name, []))
            if not self.regions:
                print("No custom regions found")
        self.categories = list(global_cfg_dict["categories"])

        setup.global_params["channels_to_consider"] = (
            args.channels.split(",")
            if args.channels
            else setup.global_params["channelSelection"]
        )
        self.channels = setup.global_params["channels_to_consider"]

        # Uncertainties
        uncNameTypes = GetUncNameTypes(unc_cfg_dict)
        self.scales = list(global_cfg_dict["scales"])
        self.unc_sources = [s.strip() for s in args.uncSources.split(",") if s.strip()]
        unknown_sources = [
            s for s in self.unc_sources if s != "Central" and s not in uncNameTypes
        ]
        if unknown_sources:
            raise ValueError(
                f"Unknown uncertainty source(s): {', '.join(unknown_sources)}"
            )
        self.unc_cfg_dict = unc_cfg_dict
        self.all_unc_dict = unc_cfg_dict["norm"].copy()
        self.all_unc_dict.update(unc_cfg_dict["shape"])
        # The members of a theory uncertainty (PDF, QCD scale) are known from the corrections
        # configuration. Every member is kept in the output; reducing them to Up/Down is left
        # to the datacard step.
        from Corrections.Corrections import Corrections

        Corrections.registerConfiguredSourceScales(global_cfg_dict)
        self.member_uncs = [
            s
            for s in self.unc_sources
            if s != "Central"
            and TheoryUncertainties.has_members(getScales(s), self.scales)
        ]

        # Datasets. Data isn't in the datasets configuration: it is added here to keep the
        # format.
        self.data_process_name = Setup.merge_data_process(setup)
        data_processes = setup.phys_model.processes(process_type="data")
        # the physics-model name of the data process, as the QCD estimate knows it
        self.qcd_data_process = data_processes[0] if data_processes else None

        # file structure: channel - region - category - <process>[_<variation>]

    def process_of(self, dataset_name):
        return Setup.merge_process_of(self.setup, dataset_name)

    def hist_name(self, dataset_type, key):
        """<process>[_<variation>]: the name of a summed histogram in the output."""
        _, (uncName, uncScale) = key
        if uncName == "Central":
            return dataset_type
        return (
            dataset_type
            + "_"
            + TheoryUncertainties.format_name(
                self.all_unc_dict[uncName], uncScale, dataset_type
            )
        )

    def to_write(self, dataset_type, key):
        """Whether the output gets this summed histogram: the requested sources, data only
        nominal, a variation only with its variations."""
        _, (uncName, uncScale) = key
        if uncName not in self.unc_sources:
            return False
        if uncName != "Central":
            return dataset_type != "data" and uncScale != "Central"
        return uncScale == "Central"

    def write_hist(self, out_file, dataset_type, key, hist):
        """Write one summed histogram as <process>[_<variation>] into its directory."""
        key_dir = key[0]
        hist_name = self.hist_name(dataset_type, key)
        dir_ptr = Utilities.mkdir(out_file, "/".join(key_dir))
        hist.SetTitle(hist_name)
        hist.SetName(hist_name)
        dir_ptr.WriteTObject(hist, hist_name, "Overwrite")

    def tree_variations(self):
        """{tree name in the chunks: name in the output}: a shape variation takes the name
        its histograms have."""
        variations = {"Central": "Central"}
        for uncName in self.unc_sources:
            if uncName in self.unc_cfg_dict["shape"].keys():
                for uncScale in self.scales:
                    variations[f"{uncName}_{uncScale}"] = self.all_unc_dict[uncName][
                        "name"
                    ].format(uncScale)
        return variations

    def needs_postprocess(self):
        """Whether a step after the merge needs every process: the data-driven QCD estimate
        of hh_bbtautau."""
        return self.analysis_import == "Analysis.hh_bbtautau" and self.estimateQCD


def open_merged(path):
    """The merged output, created by the first process and updated by the next ones."""
    out_file = ROOT.TFile(path, "UPDATE", "", MERGED_COMPRESSION)
    if not out_file or out_file.IsZombie():
        raise RuntimeError(f"cannot open {path} for update")
    return out_file


def merge_process(ctx, process, inputs, out_file):
    """Sum the inputs [(dataset_name, path)] of one process and write the result into the
    open ``out_file``: the histograms of the requested sources (theory members included) and
    the concatenated signal trees. Nothing of the process is kept in memory afterwards.
    """
    hists = {}
    chunks = []
    for dataset_name, inFile_path in inputs:
        if ctx.process_of(dataset_name) != process:
            raise RuntimeError(
                f"input {inFile_path} of dataset {dataset_name} is of process "
                f"{ctx.process_of(dataset_name)}, not {process}"
            )
        if not os.path.exists(inFile_path):
            print(
                f"input file for dataset {dataset_name} (with path= {inFile_path}) does not exist, skipping"
            )
            continue
        print(process)
        with ROOT.TFile.Open(inFile_path, "READ") as inFile:
            # check that the file is ok
            if inFile.IsZombie():
                raise RuntimeError(f"{inFile_path} is zombie")
            if not checkFile(inFile, ctx.channels, ctx.regions, ctx.categories):
                raise RuntimeError(f"{dataset_name} has void file")
            all_items = get_all_items_recursive(inFile)
            all_hists = {process: hists}
            fill_hists(
                all_items,
                all_hists,
                process,
                ctx.var,
                ctx.unc_sources,
                ctx.data_process_name,
            )
            has_trees = bool(inFile.GetDirectory(SIGNAL_TREES_DIR))
            chunks.append((dataset_name, inFile_path, has_trees))
        del all_items

    if process == ctx.data_process_name:
        # the nominal data histogram also under every variation name (written for a data
        # process not called "data")
        alias_data_variations({process: hists}, process, ctx.unc_sources)
    for key in list(hists):
        hist = hists.pop(key)
        if ctx.to_write(process, key):
            ctx.write_hist(out_file, process, key, hist)

    # the signal trees of the chunks, concatenated, one variation at a time
    for tree_process, paths in signal_tree_inputs({process: chunks}).items():
        out_dir = Utilities.mkdir(out_file, f"{SIGNAL_TREES_DIR}/{tree_process}")
        in_files = [ROOT.TFile.Open(path, "READ") for path in paths]
        for tree_name, variation in ctx.tree_variations().items():
            trees = [
                in_file.Get(f"{SIGNAL_TREES_DIR}/{tree_name}") for in_file in in_files
            ]
            if not all(trees):
                raise RuntimeError(
                    f"{SIGNAL_TREES_DIR}/{tree_name} missing in a file of {tree_process}"
                )
            concat_trees(out_dir, variation, trees)
        for in_file in in_files:
            in_file.Close()


def _read_hist(out_file, key, name):
    """A histogram of the merged file, detached from it and owned by Python (it is freed
    when dropped), or None."""
    hist = out_file.Get("/".join(key + (name,)))
    if not hist:
        return None
    hist.SetDirectory(ROOT.nullptr)
    ROOT.SetOwnership(hist, True)
    return hist


def _hist_dirs(directory, path=()):
    """The (channel, region, category) directories of the merged file, signal trees left
    out."""
    dirs = []
    for key in directory.GetListOfKeys():
        if not ROOT.TClass.GetClass(key.GetClassName()).InheritsFrom(
            ROOT.TDirectory.Class()
        ):
            continue
        name = key.GetName()
        if not path and name == SIGNAL_TREES_DIR:
            continue
        sub = directory.GetDirectory(name)
        if len(path) == 2:
            dirs.append(path + (name,))
        else:
            dirs.extend(_hist_dirs(sub, path + (name,)))
    return dirs


def postprocess(ctx, processes, out_file):
    """The steps after the merge that need every process, run on the merged file: the
    data-driven QCD estimate of hh_bbtautau. Each uncertainty source is estimated with only
    its histograms (and the nominal ones) read back, then written and freed.

    ``processes``: the merged processes, in merge order (the order the estimate subtracts
    them in, as when every process was held in memory)."""
    if not ctx.needs_postprocess():
        return
    if ctx.qcd_data_process is None:
        return
    from Analysis.QCD_estimation import AddQCDInHistDict

    key_dirs = _hist_dirs(out_file)
    for unc_source in ctx.unc_sources:
        # the members of a theory uncertainty: no QCD variation
        if unc_source in ctx.member_uncs:
            continue
        if unc_source == "Central":
            unc_keys = [("Central", "Central")]
        else:
            unc_keys = [(unc_source, scale) for scale in ctx.scales]
        all_hists_dict = {}
        for process in processes:
            # data: the nominal histogram, aliased under the variations below, as when it
            # was merged
            keys_of_process = (
                [("Central", "Central")]
                if process == ctx.data_process_name
                else unc_keys
            )
            hists = {}
            for key_dir in key_dirs:
                for unc_key in keys_of_process:
                    key = (key_dir, unc_key)
                    hist = _read_hist(out_file, key_dir, ctx.hist_name(process, key))
                    if hist is not None:
                        hists[key] = hist
            if hists:
                all_hists_dict[process] = hists
        if ctx.data_process_name in all_hists_dict:
            alias_data_variations(
                all_hists_dict, ctx.data_process_name, ctx.unc_sources
            )
        if ctx.qcd_data_process not in all_hists_dict:
            return
        # the processes the estimate subtracts: every merged one, "QCD" not among them
        dataset_types = list(all_hists_dict.keys())
        AddQCDInHistDict(
            ctx.var,
            all_hists_dict,
            ctx.channels,
            ctx.categories,
            unc_source,
            dataset_types,
            ctx.scales,
            data_process_name=ctx.qcd_data_process,
            wantNegativeContributions=False,
        )
        for key, hist in all_hists_dict.get("QCD", {}).items():
            if ctx.to_write("QCD", key):
                ctx.write_hist(out_file, "QCD", key, hist)
        del all_hists_dict


# Acknowledgement lines of --stream on stdout: "@@HistMerger merged <process>" once a process is
# in the output (its inputs can be deleted), "@@HistMerger postprocessed" after the last step.
STREAM_ACK = "@@HistMerger"


def stream(ctx, out_path, lines):
    """Merge into ``out_path`` (created) what ``lines`` asks for, one JSON message per line:
    {"process": <name>, "inputs": [[dataset_name, path], ...]} merges one process,
    {"postprocess": [dataset_name, ...]} runs the steps that need every process (the datasets
    in the merge order). Each step is acknowledged on stdout once written. The output stays
    open in between, and the configuration is read once, whatever the number of processes.
    """
    if os.path.exists(out_path):
        os.remove(out_path)
    out_file = open_merged(out_path)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if "process" in message:
            process = message["process"]
            merge_process(
                ctx, process, [tuple(item) for item in message["inputs"]], out_file
            )
            print(f"{STREAM_ACK} merged {process}", flush=True)
        elif "postprocess" in message:
            processes = list(
                dict.fromkeys(ctx.process_of(name) for name in message["postprocess"])
            )
            postprocess(ctx, processes, out_file)
            print(f"{STREAM_ACK} postprocessed", flush=True)
        else:
            raise ValueError(f"unknown message: {line}")
    out_file.Close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge the split histograms of one variable, one process at a time, then "
        "run the steps that need every process on the merged file: over the inputs given, or "
        "(--stream) over the processes sent on stdin as they are downloaded."
    )
    parser.add_argument("inFiles", nargs="*", type=str)
    parser.add_argument(
        "--dataset_names",
        type=str,
        default="",
        help="the datasets of the inputs, in their order (not with --stream)",
    )
    parser.add_argument("--var", required=True, type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--era", required=True, type=str)
    parser.add_argument(
        "--uncSources",
        required=False,
        type=str,
        default="Central",
        help="comma-separated list of uncertainty sources to merge in one pass",
    )
    parser.add_argument("--channels", required=False, type=str, default="")
    parser.add_argument("--LAWrunVersion", required=True, type=str)
    parser.add_argument("--user-custom", type=str, default=None)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="read the steps from stdin, one JSON message per line (see stream())",
    )
    args = parser.parse_args()

    startTime = time.time()
    ctx = MergeContext(args)
    if args.stream:
        if args.inFiles or args.dataset_names:
            parser.error("--stream takes its inputs from stdin")
        stream(ctx, args.outFile, sys.stdin)
    else:
        dataset_names = args.dataset_names.split(",") if args.dataset_names else []
        if not args.inFiles or len(dataset_names) != len(args.inFiles):
            parser.error(
                f"{len(args.inFiles)} input files for {len(dataset_names)} datasets"
            )
        inputs = list(zip(dataset_names, args.inFiles))
        process_order = list(
            dict.fromkeys(ctx.process_of(name) for name in dataset_names)
        )
        if os.path.exists(args.outFile):
            os.remove(args.outFile)
        out_file = open_merged(args.outFile)
        for process in process_order:
            merge_process(
                ctx,
                process,
                [item for item in inputs if ctx.process_of(item[0]) == process],
                out_file,
            )
        postprocess(ctx, process_order, out_file)
        out_file.Close()

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
