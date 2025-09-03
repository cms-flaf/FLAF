import ROOT
import sys
import os
import math
import shutil
import time

ROOT.EnableThreadSafety()

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities

snapshotOptions = ROOT.RDF.RSnapshotOptions()
snapshotOptions.fOverwriteIfExists = False
snapshotOptions.fLazy = False
snapshotOptions.fMode = "RECREATE"
snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, "k" + "ZLIB")
snapshotOptions.fCompressionLevel = 4


def createCentralQuantities(df_central, central_col_types, central_columns):
    map_creator = ROOT.analysis.MapCreator(*central_col_types)()
    df_central = map_creator.processCentral(
        ROOT.RDF.AsRNode(df_central), Utilities.ListToVector(central_columns), 1
    )
    return df_central


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--inFile", required=True, type=str)
    parser.add_argument("--outDir", required=True, type=str)
    parser.add_argument("--uncConfig", required=True, type=str)
    args = parser.parse_args()

    startTime = time.time()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
    ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
    ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')

    unc_cfg_dict = {}
    with open(args.uncConfig, "r") as f:
        unc_cfg_dict = yaml.safe_load(f)

    inFile_root = ROOT.TFile.Open(args.inFile, "READ")
    inFile_keys = [k.GetName() for k in inFile_root.GetListOfKeys()]
    if "Events" not in inFile_keys:
        key_not_exist = True
    inFile_root.Close()

    dfWrapped_central = Utilities.DataFrameBuilderBase(
        ROOT.RDataFrame("Events", args.inFile)
    )

    snaps = []
    col_names_central = dfWrapped_central.colNames
    col_types_central = dfWrapped_central.colTypes

    # central quantities definition
    dfWrapped_central.df = createCentralQuantities(
        dfWrapped_central.df, col_types_central, col_names_central
    )
    if dfWrapped_central.df.Filter("map_placeholder > 0").Count().GetValue() <= 0:
        raise RuntimeError("no events passed map placeolder")

    # remove relative weights as they are not included in the varied ttrees (by construction)
    col_to_remove = []
    for col in col_names_central:
        if col.endswith("rel"):
            col_to_remove.append(col)
    for col_name in col_to_remove:
        col_idx = col_names_central.index(col_name)
        col_type = col_types_central[col_idx]
        col_names_central.pop(col_idx)
        col_types_central.pop(col_idx)

    for uncName in ["JER"]:  # unc_cfg_dict['shape']:
        files = {}
        # print(f"uncname is {uncName}")
        for scale in ["Up", "Down"]:
            if scale not in files.keys():
                files[scale] = []
            treeName = f"Events_{uncName}{scale}"

            treeName_noDiff = f"{treeName}_noDiff"
            if treeName_noDiff in inFile_keys:
                print(treeName_noDiff)
                dfWrapped_noDiff = Utilities.DataFrameBuilderBase(
                    ROOT.RDataFrame(treeName_noDiff, args.inFile)
                )
                dfWrapped_noDiff.AddMissingColumns(col_names_central, col_types_central)
                dfWrapped_noDiff.df.Snapshot(
                    "Events",
                    f"{args.outDir}{treeName_noDiff}.root",
                    Utilities.ListToVector(col_names_central),
                    snapshotOptions,
                )
                files[scale].append(f"{args.outDir}{treeName_noDiff}.root")

            treeName_Valid = f"{treeName}_Valid"
            if treeName_Valid in inFile_keys:
                print(treeName_Valid)
                dfWrapped_Valid = Utilities.DataFrameBuilderBase(
                    ROOT.RDataFrame(treeName_Valid, args.inFile)
                )
                dfWrapped_Valid.CreateFromDelta(col_names_central, col_types_central)
                dfWrapped_Valid.AddMissingColumns(col_names_central, col_types_central)
                dfWrapped_Valid.df.Snapshot(
                    "Events",
                    f"{args.outDir}{treeName_Valid}.root",
                    Utilities.ListToVector(col_names_central),
                    snapshotOptions,
                )
                files[scale].append(f"{args.outDir}{treeName_Valid}.root")

            treeName_nonValid = f"{treeName}_nonValid"
            if treeName_nonValid in inFile_keys:
                print(treeName_nonValid)
                dfWrapped_nonValid = Utilities.DataFrameBuilderBase(
                    ROOT.RDataFrame(treeName_nonValid, args.inFile)
                )
                for col in col_names_central:
                    col_idx = col_names_central.index(col)
                    col_type = col_types_central[col_idx]
                    dfWrapped_nonValid.df = dfWrapped_nonValid.df.Redefine(
                        col, f"static_cast<{col_type}>({col})"
                    )
                dfWrapped_nonValid.df.Snapshot(
                    "Events",
                    f"{args.outDir}{treeName_nonValid}.root",
                    Utilities.ListToVector(col_names_central),
                    snapshotOptions,
                )

                files[scale].append(f"{args.outDir}{treeName_nonValid}.root")

        for scale in ["Up", "Down"]:
            columns = {}
            for fileName in files[scale]:
                dfw = Utilities.DataFrameBuilderBase(
                    ROOT.RDataFrame("Events", fileName)
                )
                columns[fileName] = {}
                col_names_central = dfw.colNames
                col_types_central = dfw.colTypes
                columns[fileName]["col"] = col_names_central
                columns[fileName]["type"] = col_types_central
            # first element is the reference
            reference_name = next(iter(columns))  # reference name
            reference = columns[reference_name]
            ref_cols = reference["col"]
            ref_types = reference["type"]
            for fileName, entry in columns.items():
                if fileName == reference_name:
                    continue  # skip the reference file
                cols = entry["col"]
                types = entry["type"]
                # different columns check
                missing_cols = set(ref_cols) - set(cols)
                extra_cols = set(cols) - set(ref_cols)
                if missing_cols or extra_cols:
                    print(f"\n[file: {fileName}] different columns:")
                    if missing_cols:
                        print(f"  - missing columns: {missing_cols}")
                    if extra_cols:
                        print(f"  - extra columns: {extra_cols}")
                # type differences in columns check
                common_cols = set(ref_cols) & set(cols)
                type_mismatches = []
                for col in common_cols:
                    ref_index = ref_cols.index(col)
                    cur_index = cols.index(col)
                    if ref_types[ref_index] != types[cur_index]:
                        type_mismatches.append(
                            (col, ref_types[ref_index], types[cur_index])
                        )
                if type_mismatches:
                    print(f"\n[file: {fileName}] different types:")
                    for col, ref_t, cur_t in type_mismatches:
                        print(f"  - col '{col}': ref={ref_t}, current={cur_t}")
            df_total = ROOT.RDataFrame("Events", Utilities.ListToVector(files[scale]))
            df_total.Snapshot(
                "Events",
                f"{args.outDir}test_hAdded_{uncName}{scale}.root",
                Utilities.ListToVector(col_names_central),
                snapshotOptions,
            )

    executionTime = time.time() - startTime
    print("Execution time in seconds: " + str(executionTime))
