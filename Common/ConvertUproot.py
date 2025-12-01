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
            raise RuntimeError(f'Unknown object type: {obj.classname}')
    out_trees = {}

    for dfName in dfNames:
        if verbose > 1:
            print(f'Processing {dfName}')
        input_tree = input_file[dfName]
        keys = input_tree.keys()
        if verbose > 1:
            print(f'  n_entireties={input_tree.num_entries}')
            print(f'  n_keys={len(keys)}')
        out_tree = {}
        if len(keys) == 0 or input_tree.num_entries == 0:
            if verbose > 1:
                print(f'  Skipping empty tree')
            continue
        df = input_tree.arrays()
        collections = {}
        other_columns = []
        for key in keys:
            parts = key.split("_", 1)
            if len(parts) == 1:
                other_columns.append(key)
            else:
                col_name, br_name = parts
                if not col_name in collections:
                    collections[col_name] = []
                collections[col_name].append(br_name)
        for col_name, columns in collections.items():
            if verbose > 1:
                print(f"  {col_name}: {columns}")
            out_tree[col_name] = ak.zip(
                {column: df[col_name + "_" + column] for column in columns}
            )
        counter_columns = ["n" + col_name for col_name in collections.keys()]
        other_columns = [ col for col in other_columns if col not in counter_columns ]
        if verbose > 1:
            print(f"  Other columns: {other_columns}")
        for column in other_columns:
            out_tree[column] = df[column]
        out_trees[dfName] = out_tree
    if verbose > 1 and len(histograms) > 0:
        print(f"Histograms: {histograms.keys()}")
    return saveFile(outFile, out_trees, histograms)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert ROOT file with TTree to Uproot TTree format with zipped columns"
    )
    parser.add_argument(
        "--inFile", type=str, required=True, help="Input ROOT file"
    )
    parser.add_argument(
        "--outFile", type=str, required=True, help="Output ROOT file"
    )
    args = parser.parse_args()

    toUproot(args.inFile, args.outFile, verbose=2)
