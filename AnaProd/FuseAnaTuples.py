import awkward as ak
import numpy as np
import os
import uproot
import ROOT

from FLAF.Common.ConvertUproot import defineColumnGrouping, toUproot

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

def alignAnaTuple(input_file, ref_ids, tree_name, output_file):
    input_tree = uproot.open(input_file)[tree_name]
    input_arrays = uproot.open(input_file)[tree_name].arrays()
    aligned_arrays = { 'FullEventId': ref_ids }
    input_event_ids = ak.to_numpy(input_arrays["FullEventId"])
    if 'valid' in input_tree.keys():
        raise RuntimeError("Column name 'valid' is reserved. Please rename the column in the input file.")
    valid = np.isin(ref_ids, input_event_ids, assume_unique=True, kind='sort')
    aligned_arrays['valid'] = valid
    if not np.array_equal(input_event_ids, ref_ids[valid]):
        raise RuntimeError("Event ID matching failed.")
    indices = np.ones_like(ref_ids, dtype=np.int64) * -1
    indices[valid] = np.arange(len(input_event_ids))
    indices = ak.where(valid, indices, np.nan)
    indices = ak.nan_to_none(indices)
    indices = ak.enforce_type(indices, "?int64")

    n_valid = np.sum(aligned_arrays['valid'])
    for branch in input_tree.branches:
        if branch.name == "FullEventId":
            continue
        if '__' in branch.name:
            raise RuntimeError(f"Branch name '{branch.name}' contains reserved substring '__'. Please rename the branch in the input file.")
        default_value = getDefaultValue(branch.typename)
        input_array = input_arrays[branch.name]
        output_array = input_array[indices]
        output_array = ak.fill_none(output_array, default_value, axis=0)
        aligned_arrays[branch.name] = output_array

    aligned_arrays = defineColumnGrouping(aligned_arrays, aligned_arrays.keys(), verbose=0)
    with uproot.recreate(output_file) as out_file:
        out_file[tree_name] = aligned_arrays
    return n_valid


def fuseAnaTuples(work_dir, output_file, reference_file, inputs, tree_name="Events", verbose=0):
    ref_tree = uproot.open(reference_file)[tree_name]
    fullEventIds = ak.to_numpy(ref_tree.arrays(["FullEventId"])["FullEventId"])
    n_events_total = len(fullEventIds)
    split_outputs = {}
    for input in inputs:
        print(f"{input['unc_source']}/{input['unc_scale']}: processing...")
        out_name = f"aligned_{input['unc_source']}_{input['unc_scale']}.root"
        out_full_name = os.path.join(work_dir, out_name)
        split_outputs[(input['unc_source'], input['unc_scale'])] = out_full_name
        n_events_valid = alignAnaTuple(
            input["file"],
            ref_ids=fullEventIds,
            tree_name=tree_name,
            output_file=out_full_name,
        )
        print(f"{input['unc_source']}/{input['unc_scale']}: aligned {n_events_valid} selected events to the original {n_events_total} events.")

    central_file = ROOT.TFile.Open(split_outputs[("central", "central")], "READ")
    central_tree = central_file.Get(tree_name)
    combined_output_file_name = os.path.join(work_dir, "combined_output.root")
    for (unc_source, unc_scale), file in split_outputs.items():
        if unc_source == "central" and unc_scale == "central":
            continue
        print(f"Adding uncertainty source '{unc_source}' scale '{unc_scale}' from file '{file}'")
        unc_file = ROOT.TFile.Open(file, "READ")
        unc_tree = unc_file.Get(tree_name)
        central_tree.AddFriend(unc_tree, f"{unc_source}__{unc_scale}")
    df = ROOT.RDataFrame(central_tree)
    ROOT.RDF.Experimental.AddProgressBar(df)
    columns_to_store = []
    validity_indicators = []
    print(f"Defining new dataframe columns... ", end="", flush=True)
    for column in df.GetColumnNames():
        column_to_store = column
        is_validity_indicator = column == 'valid'
        if '.' in column:
            unc_name, column_name = column.split('.', 1)
            if column_name == "valid":
                new_column_name = f"{unc_name}__valid"
                df = df.Define(new_column_name, f"return {column}")
                column_to_store = new_column_name
                is_validity_indicator = True
            else:
                delta_column_name = f"{unc_name}__{column_name}__delta"
                central_valid = 'valid'
                unc_valid = f"{unc_name}.valid"
                df = df.Define(delta_column_name, f"analysis::Delta({column}, {column_name}, {unc_valid}, {central_valid})")
                column_to_store = delta_column_name
        columns_to_store.append(column_to_store)
        if is_validity_indicator:
            validity_indicators.append(str(column_to_store))
    print("done.")
    if len(validity_indicators) > 1:
        filter_str = " || ".join(validity_indicators)
        df = df.Filter(filter_str)
        print(f"Filter to keep only valid events: {filter_str}")
    print(f"Creating combined snapshot into '{combined_output_file_name}'... ")
    df.Snapshot(tree_name, combined_output_file_name, columns_to_store)
    output_file_path = os.path.join(work_dir, output_file)
    print(f"Creating the final output into {output_file_path}... ", end="", flush=True)
    toUproot(combined_output_file_name, output_file_path, verbose=verbose)
    print("done.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--work-dir", required=True, type=str)
    parser.add_argument("--output-file", required=True, type=str)
    parser.add_argument("--reference-file", required=True, type=str)
    parser.add_argument("--tree", required=False, type=str, default="Events")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("input", nargs="+", type=str)
    args = parser.parse_args()

    ROOT.gROOT.ProcessLine(f".include {os.environ['FLAF_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')

    inputs = []
    for input_desc in args.input:
        input_entries = input_desc.split(":")
        if len(input_entries) != 3:
            raise ValueError(f"Input description '{input_desc}' is invalid. Expected format 'file:unc_source:unc_scale")
        input_file, unc_source, unc_scale = input_entries
        inputs.append({ "file": os.path.join(args.input_dir, input_file), "unc_source": unc_source, "unc_scale": unc_scale })
    reference_file = os.path.join(args.input_dir, args.reference_file)

    fuseAnaTuples(args.work_dir, args.output_file, reference_file, inputs, tree_name=args.tree, verbose=args.verbose)
