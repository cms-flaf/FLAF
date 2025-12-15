import uproot
import awkward as ak
import os


def saveFile(outFile, out_tree_dict, histograms):
    if os.path.exists(outFile):
        os.remove(outFile)
    with uproot.recreate(outFile, compression=uproot.LZMA(9)) as out_file:
        for hist in histograms.keys():
            out_file[hist] = histograms[hist]
        for out_tree_key in out_tree_dict.keys():
            out_file[out_tree_key] = out_tree_dict[out_tree_key]

def parseColumnName(column_name):
    if len(column_name) == 0:
        raise RuntimeError("Empty column name")
    meta_split = column_name.split("__")
    if len(meta_split) not in [1, 3, 4] or any(len(part) == 0 for part in meta_split):
        raise RuntimeError(f"Cannot parse column name: {column_name}")
    if len(meta_split) == 1:
        column_name_base = meta_split[0]
        unc_source = None
        unc_scale = None
        representation = None
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
        full_var_name = f'{var_name}__{representation}' if representation is not None else var_name
        collection_name = name_split[0]
        full_collection_name = f'{unc_source}__{unc_scale}__{collection_name}' if unc_source is not None else collection_name

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
        collection_name, column_name = column_desc["full_collection_name"], column_desc["full_var_name"]
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

def toUproot(inFile, outFile, verbose=1):
    dfNames = []
    histograms = {}
    if verbose > 0:
        print("starting toUproot")
        print(inFile)

    input_file = uproot.open(inFile)

    object_names = input_file.keys()
    for object_name in object_names:
        obj = input_file[object_name]
        if verbose > 1:
            print(f'name="{obj.name}" type="{obj.classname}"')
        if obj.classname == "TTree":
            dfNames.append(obj.name)
        elif "TH1" in obj.classname:
            histograms[obj.name] = obj
        else:
            raise RuntimeError(f"Unknown object type: {obj.classname}")
    out_trees = {}

    for dfName in dfNames:
        if verbose > 1:
            print(f"Processing {dfName}")
        input_tree = input_file[dfName]
        keys = input_tree.keys()
        if verbose > 1:
            print(f"  n_entireties={input_tree.num_entries}")
            print(f"  n_keys={len(keys)}")
        out_tree = {}
        if len(keys) == 0 or input_tree.num_entries == 0:
            if verbose > 1:
                print(f"  Skipping empty tree")
            continue
        df = input_tree.arrays()
        out_trees[dfName] = defineColumnGrouping(df, keys, verbose=verbose)
    if verbose > 1 and len(histograms) > 0:
        print(f"Histograms: {histograms.keys()}")
    return saveFile(outFile, out_trees, histograms)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ROOT file with TTree to Uproot TTree format with zipped columns"
    )
    parser.add_argument("--inFile", type=str, required=True, help="Input ROOT file")
    parser.add_argument("--outFile", type=str, required=True, help="Output ROOT file")
    args = parser.parse_args()

    toUproot(args.inFile, args.outFile, verbose=2)
