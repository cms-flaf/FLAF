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
    compression_algorithm="LZMA",
    compression_level=9,
):
    if verbose > 0:
        print(f"copyFileContent: {inputs} -> {outputFile}")
        print(
            f"copyFileContent: copyTrees={copyTrees}, copyHistograms={copyHistograms}, "
            f"appendIfExists={appendIfExists}, "
            f"compression={compression_algorithm}({compression_level})"
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

    saved_dir = ROOT.gDirectory

    # First pass: read all input objects, recursing into TDirectoryFile.
    # Histograms are accumulated (added) across inputs with the same out_path.
    # Trees are collected as a list of (file_path, internal_path) sources for TChain.
    histograms = {}  # out_path -> ROOT TH1
    trees = {}  # out_path -> [(file_path, internal_path), ...]

    def collect_objects(directory, inp, dir_prefix=""):
        # Build a map of name -> latest-cycle key to avoid processing stale cycles.
        latest_keys = {}
        for key in directory.GetListOfKeys():
            name = key.GetName()
            if name not in latest_keys or key.GetCycle() > latest_keys[name].GetCycle():
                latest_keys[name] = key

        for obj_name, key in latest_keys.items():
            classname = key.GetClassName()
            internal_path = f"{dir_prefix}{obj_name}"

            if classname in ("TDirectory", "TDirectoryFile"):
                subdir = directory.Get(obj_name)
                if subdir:
                    collect_objects(subdir, inp, f"{internal_path}/")
                continue

            obj_type = get_obj_type(classname)
            if obj_type is None:
                raise RuntimeError(
                    f"{inp['file']}/{internal_path}: unknown object type='{classname}'"
                )
            is_tree = obj_type == "tree"
            is_hist = obj_type == "histogram"

            # Prefix/suffix apply to the leaf name only; directory path is preserved.
            out_path = f"{dir_prefix}{inp['name_prefix']}{obj_name}{inp['name_suffix']}"

            if not ((inp["copyTrees"] and is_tree) or (inp["copyHistograms"] and is_hist)):
                if verbose > 1:
                    print(f'Skipping object "{internal_path}" of type "{classname}"')
                continue

            if verbose > 0:
                print(f"{inp['file']}/{internal_path} -> {out_path} (type='{classname}')")

            if is_hist:
                obj = key.ReadObj()
                obj.SetDirectory(ROOT.nullptr)
                if out_path in histograms:
                    histograms[out_path].Add(obj)
                else:
                    histograms[out_path] = obj
            else:
                if out_path not in trees:
                    trees[out_path] = []
                trees[out_path].append((inp["file"], internal_path))

    for inp in inputs:
        input_file = ROOT.TFile.Open(inp["file"], "READ")
        if not input_file or input_file.IsZombie():
            raise RuntimeError(f"Cannot open input file: {inp['file']}")
        try:
            collect_objects(input_file, inp)
        finally:
            input_file.Close()

    def get_or_create_dir(root_file, path_parts):
        current = root_file
        for part in path_parts:
            d = current.Get(part)
            if not d:
                current.mkdir(part)
                d = current.Get(part)
            current = d
        return current

    if appendIfExists and os.path.exists(outputFile):
        comp_settings = ROOT.ROOT.CompressionSettings(
            getattr(ROOT.ROOT.RCompressionSetting.EAlgorithm, "k" + compression_algorithm),
            compression_level,
        )
        open_args = ("UPDATE", "", comp_settings)
    else:
        open_args = ("RECREATE",)
    output_file = ROOT.TFile.Open(outputFile, *open_args)
    if not output_file or output_file.IsZombie():
        raise RuntimeError(f"Cannot open output file: {outputFile}")
    try:
        for out_path, sources in trees.items():
            parts = out_path.split("/")
            target_dir = get_or_create_dir(output_file, parts[:-1])
            leaf = parts[-1]
            chain = ROOT.TChain(sources[0][1])
            for file_path, _ in sources:
                chain.Add(file_path)
            target_dir.cd()
            existing = target_dir.Get(leaf)
            if existing:
                existing.CopyEntries(chain)
                existing.Write("", ROOT.TObject.kOverwrite)
            else:
                new_tree = chain.CloneTree(-1, "fast")
                new_tree.SetName(leaf)
                new_tree.Write("", ROOT.TObject.kOverwrite)

        for out_path, hist in histograms.items():
            parts = out_path.split("/")
            target_dir = get_or_create_dir(output_file, parts[:-1])
            leaf = parts[-1]
            target_dir.cd()
            existing = target_dir.Get(leaf)
            if existing:
                existing.Add(hist)
                existing.Write(leaf, ROOT.TObject.kOverwrite)
            else:
                hist.Write(leaf, ROOT.TObject.kOverwrite)
    finally:
        output_file.Close()
        if saved_dir:
            saved_dir.cd()


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
        compression_algorithm=args.compression_algorithm,
        compression_level=args.compression_level,
    )
