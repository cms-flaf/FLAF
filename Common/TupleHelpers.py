import uproot
import awkward as ak
import os
import re
import ROOT


def parseColumnName(column_name):
    if len(column_name) == 0:
        raise RuntimeError("Empty column name")
    meta_split = column_name.split("__")
    if len(meta_split) not in [1, 2, 3, 4] or any(
        len(part) == 0 for part in meta_split
    ):
        raise RuntimeError(f"Cannot parse column name: {column_name}")
    if len(meta_split) == 1:
        column_name_base = meta_split[0]
        unc_source = None
        unc_scale = None
        representation = None
    elif len(meta_split) == 2:
        column_name_base = meta_split[0]
        unc_source = None
        unc_scale = None
        representation = meta_split[1]
    else:
        unc_source = meta_split[0]
        unc_scale = meta_split[1]
        column_name_base = meta_split[2]
        representation = meta_split[3] if len(meta_split) >= 4 else None
    name_split = column_name_base.split("_", 1)
    if len(name_split) == 1:
        var_name = column_name_base
        full_var_name = column_name
        collection_name = None
        full_collection_name = None
    else:
        var_name = name_split[1]
        full_var_name = (
            f"{var_name}__{representation}" if representation is not None else var_name
        )
        collection_name = name_split[0]
        full_collection_name = (
            f"{unc_source}__{unc_scale}__{collection_name}"
            if unc_source is not None
            else collection_name
        )

    return {
        "var_name": var_name,
        "collection_name": collection_name,
        "full_var_name": full_var_name,
        "full_collection_name": full_collection_name,
        "unc_source": unc_source,
        "unc_scale": unc_scale,
        "representation": representation,
    }


def defineColumnGrouping(arrays, keys, verbose=1):
    groupped_arrays = {}
    collections = {}
    other_columns = []
    for key in keys:
        column_desc = parseColumnName(key)
        collection_name, column_name = (
            column_desc["full_collection_name"],
            column_desc["full_var_name"],
        )
        if collection_name is None:
            other_columns.append(key)
        else:
            if collection_name not in collections:
                collections[collection_name] = []
            collections[collection_name].append(column_name)

    for collection_name, columns in collections.items():
        if verbose > 1:
            print(f"  {collection_name}: {columns}")
        groupped_arrays[collection_name] = ak.zip(
            {column: arrays[collection_name + "_" + column] for column in columns}
        )
    counter_columns = ["n" + col_name for col_name in collections.keys()]
    other_columns = [col for col in other_columns if col not in counter_columns]
    if verbose > 1:
        print(f"  Other columns: {other_columns}")
    for column in other_columns:
        groupped_arrays[column] = arrays[column]

    return groupped_arrays


def copyFileContent(
    inputs,
    outputFile,
    *,
    copyTrees=True,
    copyHistograms=True,
    appendIfExists=False,
    verbose=1,
    step_size="100MB",
    compression_algorithm="LZMA",
    compression_level=9,
):
    compression = getattr(uproot, compression_algorithm)(compression_level)
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = False
    snapshotOptions.fLazy = False
    snapshotOptions.fMode = "UPDATE"
    snapshotOptions.fCompressionAlgorithm = getattr(
        ROOT.ROOT.RCompressionSetting.EAlgorithm, "k" + compression_algorithm
    )
    snapshotOptions.fCompressionLevel = compression_level
    if verbose > 0:
        print(f"copyFileContent: {inputs} -> {outputFile}")
        print(
            f"copyFileContent: copyTrees={copyTrees}, copyHistograms={copyHistograms}, appendIfExists={appendIfExists}, compression={compression}, step_size={step_size}"
        )

    def processInputs(original_inputs):
        processed_inputs = []
        if type(original_inputs) == str:
            original_inputs = [original_inputs]
        if type(original_inputs) != list:
            raise RuntimeError(f"Unknown input type: {type(original_inputs)}")
        for item in original_inputs:
            if type(item) is str:
                processed_inputs.append(
                    {
                        "file": item,
                        "name_prefix": "",
                        "name_suffix": "",
                        "copyTrees": copyTrees,
                        "copyHistograms": copyHistograms,
                    }
                )
            elif type(item) is dict:
                processed_inputs.append(
                    {
                        "file": item["file"],
                        "name_prefix": item.get("name_prefix", ""),
                        "name_suffix": item.get("name_suffix", ""),
                        "copyTrees": item.get("copyTrees", copyTrees),
                        "copyHistograms": item.get("copyHistograms", copyHistograms),
                    }
                )
            else:
                raise RuntimeError(f"Unknown input type: {type(item)}")
        return processed_inputs

    inputs = processInputs(inputs)

    known_types = [("TTree", "tree"), ("TH[1-9][0-9]*.", "histogram")]

    def get_obj_type(classname):
        for pattern, obj_type in known_types:
            if re.match(pattern, classname):
                return obj_type
        return None

    if appendIfExists and os.path.exists(outputFile):
        open_fn = uproot.update
        open_args = {}
    else:
        open_fn = uproot.recreate
        open_args = {"compression": compression}
    histograms = {}
    to_store_with_rdf = {}
    stored_with_uproot = set()
    with open_fn(outputFile, **open_args) as output_file:
        for input in inputs:
            copyInputTrees = input["copyTrees"]
            copyInputHistograms = input["copyHistograms"]
            with uproot.open(input["file"]) as input_file:
                for input_object_name in input_file.keys():
                    obj = input_file[input_object_name]
                    obj_type = get_obj_type(obj.classname)
                    if obj_type is None:
                        raise RuntimeError(
                            f"{input['file']}/{input_object_name}: unknown object type='{obj.classname}'"
                        )

                    is_tree = obj_type == "tree"
                    is_hist = obj_type == "histogram"
                    if not (
                        (copyInputTrees and is_tree)
                        or (copyInputHistograms and is_hist)
                    ):
                        if verbose > 1:
                            print(
                                f'Skipping object "{input_object_name}" of type "{obj.classname}"'
                            )
                        continue

                    out_name = f"{input['name_prefix']}{obj.name}{input['name_suffix']}"
                    if verbose > 0:
                        print(
                            f"{input['file']}/{input_object_name} -> {out_name} (type='{obj.classname}')"
                        )
                    if is_hist:
                        if out_name in histograms:
                            hist, is_converted = histograms[out_name]
                            if not is_converted:
                                hist = uproot.to_hist(hist)
                            new_hist = hist + uproot.to_hist(obj)
                            histograms[out_name] = (new_hist, True)
                        else:
                            histograms[out_name] = (obj, False)
                    else:
                        if obj.num_entries > 0:
                            for arrays in obj.iterate(step_size=step_size):
                                groupped_arrays = defineColumnGrouping(
                                    arrays, obj.keys(), verbose=min(0, verbose - 1)
                                )
                                if out_name in output_file:
                                    output_file[out_name].extend(groupped_arrays)
                                else:
                                    output_file[out_name] = groupped_arrays
                            stored_with_uproot.add(out_name)
                        else:
                            to_store_with_rdf[out_name] = (
                                input["file"],
                                input_object_name,
                            )

        for hist_name, (hist, _) in histograms.items():
            output_file[hist_name] = hist

    # If the file doesn't exist yet, change snapshot to RECREATE
    if not os.path.exists(outputFile):
        snapshotOptions.fMode = "RECREATE"
    for out_name, (input_file_name, input_name) in to_store_with_rdf.items():
        if out_name in stored_with_uproot:
            # Check if the empty tree was already handled by another file
            continue
        df = ROOT.RDataFrame(input_name, input_file_name)
        df.Snapshot(out_name, outputFile, ".*", snapshotOptions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert copy content of the input root files into the output. TTree columns are zipped."
    )
    parser.add_argument(
        "--outputFile", type=str, required=True, help="Output ROOT file"
    )
    parser.add_argument(
        "--no-copyTrees", action="store_true", help="Do not copy TTrees"
    )
    parser.add_argument(
        "--no-copyHistograms", action="store_true", help="Do not copy histograms"
    )
    parser.add_argument(
        "--appendIfExists",
        action="store_true",
        help="Append to the output file if it exists",
    )
    parser.add_argument(
        "--compression-algorithm",
        type=str,
        default="LZMA",
        help="Compression algorithm",
    )
    parser.add_argument(
        "--compression-level", type=int, default=9, help="Compression level"
    )
    parser.add_argument(
        "--step-size", type=str, default="100MB", help="Step size for reading TTrees"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("inFile", nargs="+", type=str, help="Input ROOT files")
    args = parser.parse_args()

    copyFileContent(
        args.inFile,
        args.outputFile,
        copyTrees=not args.no_copyTrees,
        copyHistograms=not args.no_copyHistograms,
        appendIfExists=args.appendIfExists,
        verbose=args.verbose,
        step_size=args.step_size,
        compression_algorithm=args.compression_algorithm,
        compression_level=args.compression_level,
    )
