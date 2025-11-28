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
    return


def toUproot(inFile, outFile):
    dfNames = []
    histograms = {}
    print("starting toUproot")
    print(inFile)

    input_file = uproot.open(inFile)

    object_names = input_file.keys()
    # print(object_names)
    for object_name in object_names:
        obj = input_file[object_name]
        # print(f"Tipo: {obj.classname}, Nome: {obj.name}")
        if obj.classname == "TTree":
            dfNames.append(obj.name)
        elif "TH1" in obj.classname:
            histograms[obj.name] = obj
        else:
            print(f"oggetto di tipo {obj.classname}")
    # if len(dfNames)>1: print(f"len of dfNames is {len(dfNames)}")
    out_trees = {}

    for dfName in dfNames:
        # print(dfName)
        input_tree = input_file[dfName]
        keys = input_tree.keys()
        # print(keys)
        out_tree = {}
        if not keys:
            # with uproot.recreate(outFile, compression=uproot.LZMA(9)) as out_file:
            # out_file.close()
            continue
        df = input_tree.arrays()
        collections = {}
        other_columns = []
        for key in keys:
            parts = key.split("_", 1)
            if len(parts) == 1:
                # print(key)
                other_columns.append(key)
            else:
                col_name, br_name = parts
                # print(col_name, br_name)
                if not col_name in collections:
                    collections[col_name] = []
                collections[col_name].append(br_name)
        # print(collections.items())
        for col_name, columns in collections.items():
            out_tree[col_name] = ak.zip(
                {column: df[col_name + "_" + column] for column in columns}
            )
        n_columns = ["n" + col_name for col_name in collections.keys()]
        for column in other_columns:
            if column not in n_columns:
                out_tree[column] = df[column]
        out_trees[dfName] = out_tree
    # print(histograms)
    # print(out_trees)
    return saveFile(outFile, out_trees, histograms)
