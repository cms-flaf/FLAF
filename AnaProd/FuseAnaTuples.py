import awkward as ak
import json
import numpy as np
import os
import re
import uproot
import ROOT
import shutil

from FLAF.Common.TupleHelpers import (
    copyFileContent,
    defineColumnGrouping,
    parseColumnName,
)
from Corrections.CorrectionsCore import central

default_values = {
    "uint32_t": np.uint32(0),
    "uint64_t": np.uint64(0),
    "int32_t": np.int32(0),
    "bool": False,
    "float": np.float32(0.0),
    "double": np.float64(0.0),
    "uint8_t": np.uint8(0),
    "int8_t": np.int8(0),
    "int16_t": np.int16(0),
}


def isArrayType(type_name):
    return type_name.startswith("RVec") or type_name.endswith("[]")


def getDefaultValue(type_name):
    if isArrayType(type_name):
        return ak.Array([])
    if type_name not in default_values:
        raise RuntimeError(f"No default value specified for type '{type_name}'")
    return default_values[type_name]


def columnType(type_name):
    """(is array, element type) of a branch type as uproot reports it."""
    for prefix in ["RVec<", "ROOT::VecOps::RVec<", "std::vector<", "vector<"]:
        if type_name.startswith(prefix) and type_name.endswith(">"):
            return True, type_name[len(prefix) : -1]
    if type_name.endswith("[]"):
        return True, type_name[: -len("[]")]
    return False, type_name


def checkColumnTypes(inputs, output_file, tree_name, special_columns, central_only=()):
    """Every column keeps its type through the fusion: an array stays an array of the same
    element type, a scalar stays a scalar of the same type. Columns in central_only are not
    stored in the shifted trees."""
    with uproot.open(output_file) as output:
        mismatches = []
        for (unc_source, unc_scale), input in inputs.items():
            is_central = unc_source == central
            out_tree_name = (
                tree_name if is_central else f"{tree_name}__{unc_source}__{unc_scale}"
            )
            out_types = {
                branch.name: columnType(branch.typename)
                for branch in output[out_tree_name].branches
            }
            with uproot.open(input["file_name"]) as input_file:
                in_types = {
                    branch.name: columnType(branch.typename)
                    for branch in input_file[tree_name].branches
                }
            for name, in_type in in_types.items():
                if not is_central and name in central_only:
                    continue
                out_name = (
                    name if is_central or name in special_columns else f"{name}__delta"
                )
                out_type = out_types.get(out_name)
                if out_type != in_type:
                    mismatches.append(
                        f"{out_tree_name}/{out_name}: {in_type} -> {out_type}"
                    )
    if mismatches:
        raise RuntimeError(
            "Columns changed type while fusing (is array, element type):\n  "
            + "\n  ".join(mismatches)
        )


def shiftedCounterName(collection):
    # RDataFrame reads a friend's array with the main tree's counter of the same name, so a
    # reader of a shifted tree would give Central.<array> the size of the shifted collection.
    return f"n{collection}__shifted"


def arrayCounter(tree, column):
    """The name of the counter of an array branch, None for a scalar or an object branch."""
    branch = tree.GetBranch(column)
    leaf = branch.GetLeaf(column) if branch else None
    counter = leaf.GetLeafCount() if leaf else None
    return counter.GetName() if counter else None


def extractCommonEvents(reference_file, inputs, tree_name, id_column):
    if len(inputs) == 0:
        raise RuntimeError(
            "At least one input file is required to extract common events."
        )

    def loadIds(file):
        with uproot.open(file) as f:
            tree = f[tree_name]
            ids = tree.arrays([id_column], library="np")[id_column]
        return ids

    ref_ids = loadIds(reference_file)
    valid = np.zeros_like(ref_ids, dtype=bool)

    for key, input in inputs.items():
        input_ids = loadIds(input["file_name"])
        input_ids_valid = np.isin(ref_ids, input_ids, assume_unique=True, kind="sort")
        if not np.array_equal(input_ids, ref_ids[input_ids_valid]):
            raise RuntimeError("Event ID matching failed.")
        valid = valid | input_ids_valid
        input["valid"] = input_ids_valid

    for key, input in inputs.items():
        input["valid"] = input["valid"][valid]

    return len(ref_ids), ref_ids[valid]


def alignAnaTuple(
    *, input_file, ref_ids, input_valid, tree_name, output_file, id_column
):
    input_tree = uproot.open(input_file)[tree_name]
    input_arrays = uproot.open(input_file)[tree_name].arrays()
    aligned_arrays = {id_column: ref_ids}
    if "valid" in input_tree.keys():
        raise RuntimeError(
            "Column name 'valid' is reserved. Please rename the column in the input file."
        )
    aligned_arrays["valid"] = input_valid
    indices = np.ones_like(ref_ids, dtype=np.int64) * -1
    n_valid = np.count_nonzero(input_valid)
    indices[input_valid] = np.arange(n_valid)
    indices = ak.where(input_valid, indices, np.nan)
    indices = ak.nan_to_none(indices)
    indices = ak.enforce_type(indices, "?int64")

    for branch in input_tree.branches:
        if branch.name == id_column:
            continue
        if "__" in branch.name:
            raise RuntimeError(
                f"Branch name '{branch.name}' contains reserved substring '__'. Please rename the branch in the input file."
            )
        default_value = getDefaultValue(branch.typename)
        input_array = input_arrays[branch.name]
        output_array = input_array[indices]
        output_array = ak.fill_none(output_array, default_value, axis=0)
        aligned_arrays[branch.name] = output_array

    aligned_arrays = defineColumnGrouping(
        aligned_arrays, aligned_arrays.keys(), verbose=0
    )
    with uproot.recreate(output_file, compression=uproot.ZLIB(4)) as out_file:
        out_file[tree_name] = aligned_arrays
    return n_valid


def resolveShiftInvariantColumns(
    *, patterns, central_file, shifted_files, tree_name, reserved_columns
):
    """Match the analysis' shift-invariant column patterns against the per-variation inputs.

    Returns the declared columns that the shifted inputs carry, and the counters of the
    array collections that are declared as a whole.
    """

    def schema(file_name):
        with uproot.open(file_name) as f:
            return {branch.name: branch.typename for branch in f[tree_name].branches}

    def declared(name):
        return any(re.search(pattern, name) for pattern in patterns)

    reserved = sorted(name for name in reserved_columns if declared(name))
    if reserved:
        raise RuntimeError(f"Reserved columns {reserved} cannot be shift-invariant.")
    central_schema = schema(central_file)
    shifted_schemas = {key: schema(file) for key, file in shifted_files.items()}
    for key, shifted_schema in shifted_schemas.items():
        extra = sorted(
            name
            for name in shifted_schema
            if declared(name) and name not in central_schema
        )
        if extra:
            raise RuntimeError(
                f"Shift-invariant columns {extra} are in {key} but not in the central tree."
            )
    columns = set()
    for name in central_schema:
        if name in reserved_columns or not declared(name):
            continue
        missing_in = [
            key
            for key, shifted_schema in shifted_schemas.items()
            if name not in shifted_schema
        ]
        if len(missing_in) == len(shifted_schemas):
            continue  # stored for the central tree only, e.g. a weight variation
        if missing_in:
            raise RuntimeError(
                f"Shift-invariant column '{name}' is missing from {missing_in}."
            )
        columns.add(name)
    # A placeholder row can only get a whole collection: filling some of its arrays would
    # give them a length the others do not have.
    collections = {}
    for name, type_name in central_schema.items():
        if isArrayType(type_name):
            collection = parseColumnName(name)["full_collection_name"]
            collections.setdefault(collection, []).append(name)
    counters = set()
    for collection, names in collections.items():
        undeclared = sorted(name for name in names if name not in columns)
        if not undeclared:
            counters.add(f"n{collection}")
        elif len(undeclared) < len(names):
            raise RuntimeError(
                f"Array collection '{collection}' is shift-invariant only in part: "
                f"{undeclared} are not declared."
            )
    return columns, counters


def bitPattern(values):
    values = np.asarray(values)
    if values.dtype.kind == "f":
        return values.view(np.uint32 if values.itemsize == 4 else np.uint64)
    return values


def rowsEqual(x, y):
    """Per row, whether two aligned columns are bit-identical (arrays: also in length)."""
    if x.ndim == 1:
        return bitPattern(ak.to_numpy(x)) == bitPattern(ak.to_numpy(y))
    same_size = ak.to_numpy(ak.num(x) == ak.num(y))
    x, y = x[same_size], y[same_size]
    flat_equal = bitPattern(ak.flatten(x)) == bitPattern(ak.flatten(y))
    equal = np.zeros(len(same_size), dtype=bool)
    equal[same_size] = ak.to_numpy(ak.all(ak.unflatten(flat_equal, ak.num(x)), axis=1))
    return equal


def fillShiftInvariantColumns(
    *, columns, central_input, shifted_inputs, tree_name, id_column, verbose
):
    """Fill the declared columns of the central placeholder rows from a shifted input that
    selected the event, after checking that every input selecting an event agrees on them.
    """
    central_file = central_input["aligned_file"]
    with uproot.open(central_file) as f:
        central_arrays = f[tree_name].arrays()
    ids = ak.to_numpy(central_arrays[id_column])
    columns = sorted(columns)
    reference = {column: central_arrays[column] for column in columns}
    have_reference = ak.to_numpy(central_arrays["valid"]).copy()
    for (unc_source, unc_scale), input in shifted_inputs.items():
        with uproot.open(input["aligned_file"]) as f:
            shifted_arrays = f[tree_name].arrays(columns)
        compare = have_reference & input["valid"]
        for column in columns:
            equal = rowsEqual(
                reference[column][compare], shifted_arrays[column][compare]
            )
            if not np.all(equal):
                rows = np.nonzero(compare)[0][~equal]
                row = rows[0]
                raise RuntimeError(
                    f"Column '{column}' is declared shift-invariant but differs in"
                    f" {unc_source}/{unc_scale} for {len(rows)} of {np.count_nonzero(compare)}"
                    f" events, e.g. {id_column}={ids[row]}: {reference[column][row]} vs"
                    f" {shifted_arrays[column][row]}."
                )
        fill = ~have_reference & input["valid"]
        if np.any(fill):
            for column in columns:
                reference[column] = ak.where(
                    fill, shifted_arrays[column], reference[column]
                )
            have_reference |= fill
        if verbose > 0:
            print(
                f"{unc_source}/{unc_scale}: shift-invariant columns checked for"
                f" {np.count_nonzero(compare)} events, filled for {np.count_nonzero(fill)}."
            )
    for column in columns:
        central_arrays[column] = reference[column]
    fields = central_arrays.fields
    grouped = defineColumnGrouping(
        {field: central_arrays[field] for field in fields}, fields, verbose=0
    )
    with uproot.recreate(central_file, compression=uproot.ZLIB(4)) as out_file:
        out_file[tree_name] = grouped


def fuseAnaTuples(*, config, work_dir, tuple_output, report_output=None, verbose=0):
    inputs = {}
    for input_desc in config["output_files"]:
        unc_source = input_desc["unc_source"]
        unc_scale = input_desc["unc_scale"]
        file_name = input_desc["file_name"]
        key = (unc_source, unc_scale)
        if key in inputs:
            raise RuntimeError(
                f"Multiple input files specified for uncertainty source '{unc_source}' scale '{unc_scale}'."
            )
        inputs[key] = {"file_name": file_name}

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = True
    snapshotOptions.fLazy = False
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = (
        ROOT.ROOT.RCompressionSetting.EAlgorithm.kZLIB
    )
    snapshotOptions.fCompressionLevel = 4

    central_key = (central, central)
    if central_key not in inputs:
        raise RuntimeError("Central input file is required.")
    central_input = inputs[(central, central)]
    shifted_inputs = {k: v for k, v in inputs.items() if k != central_key}

    reference_file = config["reference_file"]
    tree_name = config.get("tree_name", "Events")
    full_event_id_column = config.get("full_event_id_column", "FullEventId")
    special_columns = ["valid", full_event_id_column]

    # Columns that no systematic variation changes are stored in the central tree only, and a
    # placeholder row (an event selected only in a variation) gets them from the variation.
    invariant_columns, invariant_counters = set(), set()
    invariant_patterns = config.get("shift_invariant_columns", [])
    if invariant_patterns and shifted_inputs:
        invariant_columns, invariant_counters = resolveShiftInvariantColumns(
            patterns=invariant_patterns,
            central_file=central_input["file_name"],
            shifted_files={
                key: input["file_name"] for key, input in shifted_inputs.items()
            },
            tree_name=tree_name,
            reserved_columns=special_columns,
        )
        if verbose > 0:
            print(
                f"Shift-invariant: {len(invariant_columns)} columns,"
                f" counters {sorted(invariant_counters)}."
            )

    n_events_total, reference_ids = extractCommonEvents(
        reference_file, inputs, tree_name, full_event_id_column
    )
    n_unique_events = len(reference_ids)
    if verbose > 0:
        print(f"The original number of events: {n_events_total}")
        print(f"The number of unique selected events: {n_unique_events}")
    for (unc_source, unc_scale), input in inputs.items():
        out_name = f"aligned_{unc_source}_{unc_scale}.root"
        out_full_name = os.path.join(work_dir, out_name)
        input["aligned_file"] = out_full_name
        if n_unique_events > 0:
            n_events_valid = alignAnaTuple(
                input_file=input["file_name"],
                ref_ids=reference_ids,
                input_valid=input["valid"],
                tree_name=tree_name,
                output_file=out_full_name,
                id_column=full_event_id_column,
            )
        else:
            n_events_valid = 0
            os.makedirs(os.path.dirname(out_full_name), exist_ok=True)
            df = ROOT.RDataFrame(tree_name, input["file_name"])
            df = df.Define("valid", "true")
            df.Snapshot(tree_name, out_full_name)
        if verbose > 0:
            print(
                f"{unc_source}/{unc_scale}: aligned {n_events_valid} selected events to the superset of {n_unique_events} events."
            )

    if invariant_columns and n_unique_events > 0:
        fillShiftInvariantColumns(
            columns=invariant_columns,
            central_input=central_input,
            shifted_inputs=shifted_inputs,
            tree_name=tree_name,
            id_column=full_event_id_column,
            verbose=verbose,
        )

    if verbose > 1:
        verbosity_keeper = ROOT.RLogScopedVerbosity(
            ROOT.Detail.RDF.RDFLogChannel(), 100
        )
    with ROOT.TFile.Open(central_input["aligned_file"], "READ") as central_file:
        central_tree = central_file.Get(tree_name)

        for (unc_source, unc_scale), input in shifted_inputs.items():
            if verbose > 0:
                print(f"{unc_source}/{unc_scale}: creating a snapshot with deltas... ")
            with ROOT.TFile.Open(input["aligned_file"], "READ") as shifted_file:
                shifted_tree = shifted_file.Get(tree_name)
                shifted_tree.AddFriend(central_tree, "central")
                df = ROOT.RDataFrame(shifted_tree)
                if verbose > 0:
                    ROOT.RDF.Experimental.AddProgressBar(df)

                columns_to_store = []
                for column in df.GetColumnNames():
                    if (
                        "." in column
                        or column in invariant_columns
                        or column in invariant_counters
                    ):
                        continue
                    if column not in special_columns:
                        delta_column_name = f"{column}__delta"
                        central_valid = "central.valid"
                        unc_valid = "valid"
                        central_column = f"central.{column}"
                        counter = arrayCounter(central_tree, str(column))
                        if counter is not None:
                            # central.<array> is read with the size of the shifted row:
                            # keep only the elements the central row has
                            central_column = (
                                f"ROOT::VecOps::Take(central.{column},"
                                f" std::min<long>(central.{column}.size(), central.{counter}))"
                            )
                        df = df.Define(
                            delta_column_name,
                            f"analysis::Delta({column}, {central_column}, {unc_valid}, {central_valid})",
                        )
                        column_to_store = delta_column_name
                    else:
                        column_to_store = column
                    columns_to_store.append(column_to_store)
                input["delta_file"] = os.path.join(
                    work_dir, f"delta_{unc_source}_{unc_scale}.root"
                )
                df.Snapshot(
                    tree_name, input["delta_file"], columns_to_store, snapshotOptions
                )

    if verbose > 1:
        del verbosity_keeper

    output_file_path = os.path.join(work_dir, tuple_output)
    if verbose > 0:
        print(f"Collecting outputs into {output_file_path}... ", end="", flush=True)
    sources = []
    for (unc_source, unc_scale), input in inputs.items():
        suffix = f"__{unc_source}__{unc_scale}" if unc_source != central else ""
        sources.append(
            {
                "file": input["file_name"],
                "name_suffix": suffix,
                "copyHistograms": True,
                "copyTrees": False,
            }
        )
        tree_file = (
            input["aligned_file"] if unc_source == central else input["delta_file"]
        )
        sources.append(
            {
                "file": tree_file,
                "name_suffix": suffix,
                "copyHistograms": False,
                "copyTrees": True,
                "counter_name": None if unc_source == central else shiftedCounterName,
            }
        )
    copyFileContent(sources, output_file_path, verbose=min(0, verbose - 1))
    checkColumnTypes(
        inputs,
        output_file_path,
        tree_name,
        special_columns,
        central_only=invariant_columns | invariant_counters,
    )
    if verbose > 0:
        print("done.")

    if report_output is not None:
        report = {}
        for key, value in config.items():
            if key not in ["output_files", "reference_file"]:
                report[key] = value
        report["n_events"] = n_unique_events
        report["trees"] = []
        for unc_source, unc_scale in inputs.keys():
            report["trees"].append(
                {
                    "unc_source": unc_source,
                    "unc_scale": unc_scale,
                    "tree_name": (
                        f"{tree_name}__{unc_source}__{unc_scale}"
                        if unc_source != central
                        else tree_name
                    ),
                }
            )
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

    fuseAnaTuples(
        config=config,
        work_dir=args.work_dir,
        tuple_output=args.tuple_output,
        report_output=args.report_output,
        verbose=args.verbose,
    )
