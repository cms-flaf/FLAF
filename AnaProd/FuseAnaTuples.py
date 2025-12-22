import awkward as ak
import json
import numpy as np
import os
import uproot
import ROOT

from FLAF.Common.TupleHelpers import defineColumnGrouping, copyFileContent
from Corrections.CorrectionsCore import central

default_values = {
    'uint32_t': np.uint32(0),
    'uint64_t': np.uint64(0),
    'int32_t': np.int32(0),
    'bool': False,
    'float': np.float32(0.0),
    'uint8_t': np.uint8(0),
}

def getDefaultValue(type_name):
    if type_name.startswith('RVec') or type_name.endswith('[]'):
        return ak.Array([])
    if type_name not in default_values:
        raise RuntimeError(f"No default value specified for type '{type_name}'")
    return default_values[type_name]

def extractCommonEvents(reference_file, inputs, tree_name, id_column):
    if len(inputs) == 0:
        raise RuntimeError("At least one input file is required to extract common events.")

    def loadIds(file):
        with uproot.open(file) as f:
            tree = f[tree_name]
            ids = tree.arrays([id_column], library="np")[id_column]
        return ids

    ref_ids = loadIds(reference_file)
    valid = np.zeros_like(ref_ids, dtype=bool)

    for key, input in inputs.items():
        input_ids = loadIds(input['file_name'])
        input_ids_valid = np.isin(ref_ids, input_ids, assume_unique=True, kind='sort')
        if not np.array_equal(input_ids, ref_ids[input_ids_valid]):
            raise RuntimeError("Event ID matching failed.")
        valid = valid | input_ids_valid
        input['valid'] = input_ids_valid

    for key, input in inputs.items():
        input['valid'] = input['valid'][valid]

    return len(ref_ids), ref_ids[valid]

def alignAnaTuple(*, input_file, ref_ids, input_valid, tree_name, output_file, id_column):
    input_tree = uproot.open(input_file)[tree_name]
    input_arrays = uproot.open(input_file)[tree_name].arrays()
    aligned_arrays = { id_column: ref_ids }
    if 'valid' in input_tree.keys():
        raise RuntimeError("Column name 'valid' is reserved. Please rename the column in the input file.")
    aligned_arrays['valid'] = input_valid
    indices = np.ones_like(ref_ids, dtype=np.int64) * -1
    n_valid = np.count_nonzero(input_valid)
    indices[input_valid] = np.arange(n_valid)
    indices = ak.where(input_valid, indices, np.nan)
    indices = ak.nan_to_none(indices)
    indices = ak.enforce_type(indices, "?int64")

    for branch in input_tree.branches:
        if branch.name == id_column:
            continue
        if '__' in branch.name:
            raise RuntimeError(f"Branch name '{branch.name}' contains reserved substring '__'. Please rename the branch in the input file.")
        default_value = getDefaultValue(branch.typename)
        input_array = input_arrays[branch.name]
        output_array = input_array[indices]
        output_array = ak.fill_none(output_array, default_value, axis=0)
        aligned_arrays[branch.name] = output_array

    aligned_arrays = defineColumnGrouping(aligned_arrays, aligned_arrays.keys(), verbose=0)
    with uproot.recreate(output_file, compression=uproot.ZLIB(4)) as out_file:
        out_file[tree_name] = aligned_arrays
    return n_valid


def fuseAnaTuples(*, config, work_dir, tuple_output, report_output=None, verbose=0):
    inputs = {}
    for input_desc in config["output_files"]:
        unc_source = input_desc["unc_source"]
        unc_scale = input_desc["unc_scale"]
        file_name = input_desc["file_name"]
        key = (unc_source, unc_scale)
        if key in inputs:
            raise RuntimeError(f"Multiple input files specified for uncertainty source '{unc_source}' scale '{unc_scale}'.")
        inputs[key] = { "file_name": file_name }

    central_key = (central, central)
    if central_key not in inputs:
        raise RuntimeError("Central input file is required.")
    central_input = inputs[(central, central)]
    shifted_inputs = { k: v for k, v in inputs.items() if k != central_key }

    reference_file = config["reference_file"]
    tree_name = config.get("tree_name", "Events")
    full_event_id_column = config.get("full_event_id_column", "FullEventId")

    n_events_total, reference_ids = extractCommonEvents(reference_file, inputs, tree_name, full_event_id_column)
    n_unique_events = len(reference_ids)
    if verbose > 0:
        print(f"The original number of events: {n_events_total}")
        print(f"The number of unique selected events: {n_unique_events}")
    for (unc_source, unc_scale), input in inputs.items():
        out_name = f"aligned_{unc_source}_{unc_scale}.root"
        out_full_name = os.path.join(work_dir, out_name)
        input['aligned_file'] = out_full_name
        n_events_valid = alignAnaTuple(
            input_file=input["file_name"],
            ref_ids=reference_ids,
            input_valid=input['valid'],
            tree_name=tree_name,
            output_file=out_full_name,
            id_column=full_event_id_column,
        )
        if verbose > 0:
            print(f"{unc_source}/{unc_scale}: aligned {n_events_valid} selected events to the superset of {n_unique_events} events.")

    special_columns = [ 'valid', full_event_id_column ]

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = True
    snapshotOptions.fLazy = False
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = (
        ROOT.ROOT.RCompressionSetting.EAlgorithm.kZLIB
    )
    snapshotOptions.fCompressionLevel = 4

    if verbose > 1:
        verbosity_keeper = ROOT.RLogScopedVerbosity(ROOT.Detail.RDF.RDFLogChannel(), 100)
    with ROOT.TFile.Open(central_input['aligned_file'], "READ") as central_file:
        central_tree = central_file.Get(tree_name)

        for (unc_source, unc_scale), input in shifted_inputs.items():
            if verbose > 0:
                print(f"{unc_source}/{unc_scale}: creating a snapshot with deltas... ")
            with ROOT.TFile.Open(input['aligned_file'], "READ") as shifted_file:
                shifted_tree = shifted_file.Get(tree_name)
                shifted_tree.AddFriend(central_tree, "central")
                df = ROOT.RDataFrame(shifted_tree)
                if verbose > 0:
                    ROOT.RDF.Experimental.AddProgressBar(df)

                columns_to_store = []
                for column in df.GetColumnNames():
                    if '.' in column:
                        continue
                    if column not in special_columns:
                        delta_column_name = f"{column}__delta"
                        central_valid = 'central.valid'
                        unc_valid = "valid"
                        central_column = f'central.{column}'
                        df = df.Define(delta_column_name, f"analysis::Delta({column}, {central_column}, {unc_valid}, {central_valid})")
                        column_to_store = delta_column_name
                    else:
                        column_to_store = column
                    columns_to_store.append(column_to_store)
                input['delta_file'] = os.path.join(work_dir, f"delta_{unc_source}_{unc_scale}.root")
                df.Snapshot(tree_name, input['delta_file'], columns_to_store, snapshotOptions)

    if verbose > 1:
        del verbosity_keeper

    output_file_path = os.path.join(work_dir, tuple_output)
    if verbose > 0:
        print(f"Collecting outputs into {output_file_path}... ", end="", flush=True)
    sources = []
    for (unc_source, unc_scale), input in inputs.items():
        suffix = f"__{unc_source}__{unc_scale}" if unc_source != central else ""
        sources.append({
            "file": input['file_name'],
            "name_suffix": suffix,
            "copyHistograms": True,
            "copyTrees": False
        })
        tree_file = input['aligned_file'] if unc_source == central else input['delta_file']
        sources.append({
            "file": tree_file,
            "name_suffix": suffix,
            "copyHistograms": False,
            "copyTrees": True
        })
    copyFileContent(sources, output_file_path, verbose=min(0, verbose-1))
    if verbose > 0:
        print("done.")

    if report_output is not None:
        report = { }
        for key, value in config.items():
            if key not in ["output_files", "reference_file"]:
                report[key] = value
        report["n_events"] = n_unique_events
        report_output_path = os.path.join(work_dir, report_output)
        with open(report_output_path, "w") as f:
            json.dump(report, f, indent=4)
        if verbose > 0:
            print(f"Report written to '{report_output_path}'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-config", required=True, type=str)
    parser.add_argument("--work-dir", required=True, type=str)
    parser.add_argument("--tuple-output", required=True, type=str)
    parser.add_argument("--report-output", required=False, type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    ROOT.gROOT.ProcessLine(f".include {os.environ['FLAF_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')

    with open(args.input_config, "r") as f:
        config = json.load(f)

    fuseAnaTuples(config=config, work_dir=args.work_dir, tuple_output=args.tuple_output,
                  report_output=args.report_output, verbose=args.verbose)
