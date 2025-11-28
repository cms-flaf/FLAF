import ROOT
import os
import shutil
import sys

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()


if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import FLAF.Common.Utilities as Utilities
from FLAF.RunKit.run_tools import ps_call


def createVoidTree(file_name, tree_name):
    df = ROOT.RDataFrame(0)
    df = df.Define("test", "return true;")
    df.Snapshot(tree_name, file_name, {"test"})


def create_file(file_name, times=None):
    with open(file_name, "w"):
        os.utime(file_name, times)


def ListToVector(list, type="string"):
    vec = ROOT.std.vector(type)()
    for item in list:
        vec.push_back(item)
    return vec


col_type_dict = {
    "Float_t": "float",
    "Double_t": "double",
    "Bool_t": "bool",
    "Int_t": "int",
    "Short_t": "short",
    "ULong64_t": "unsigned long long",
    "ULong_t": "unsigned long",
    "Long_t": "long",
    "UChar_t": "unsigned char",
    "Char_t": "char",
    "UInt_t": "unsigned int",
    "ROOT::VecOps::RVec<float>": "ROOT::VecOps::RVec<float>",
    "ROOT::VecOps::RVec<Float_t>": "ROOT::VecOps::RVec<float>",
    "ROOT::VecOps::RVec<int>": "ROOT::VecOps::RVec<int>",
    "ROOT::VecOps::RVec<Int_t>": "ROOT::VecOps::RVec<int>",
    "ROOT::VecOps::RVec<unsigned char>": "ROOT::VecOps::RVec<unsigned char>",
    "ROOT::VecOps::RVec<UChar_t>": "ROOT::VecOps::RVec<unsigned char>",
    "ROOT::VecOps::RVec<char>": "ROOT::VecOps::RVec<char>",
    "ROOT::VecOps::RVec<Char_t>": "ROOT::VecOps::RVec<char>",
    "ROOT::VecOps::RVec<ULong64_t>": "ROOT::VecOps::RVec<unsigned long long>",
    "ROOT::VecOps::RVec<unsigned long long>": "ROOT::VecOps::RVec<unsigned long long>",
    "ROOT::VecOps::RVec<unsigned long>": "ROOT::VecOps::RVec<unsigned long>",
    "ROOT::VecOps::RVec<ULong_t>": "ROOT::VecOps::RVec<unsigned long>",
    "ROOT::VecOps::RVec<short>": "ROOT::VecOps::RVec<short>",
    "ROOT::VecOps::RVec<Short_t>": "ROOT::VecOps::RVec<short>",
    "ROOT::VecOps::RVec<UShort_t>": "ROOT::VecOps::RVec<unsigned short>",
    "ROOT::VecOps::RVec<double>": "ROOT::VecOps::RVec<double>",
    "ROOT::VecOps::RVec<Double_t>": "ROOT::VecOps::RVec<double>",
    "ROOT::VecOps::RVec<bool>": "ROOT::VecOps::RVec<bool>",
    "ROOT::VecOps::RVec<Bool_t>": "ROOT::VecOps::RVec<bool>",
}


def make_df(
    inputFileCentral,
    inputFileShifted,
    outDir,
    treeName,
    treeName_in="Events",
    treeName_central="Events",
):
    df_central = ROOT.RDataFrame(treeName_central, inputFileCentral)
    colNames_central = [str(c) for c in df_central.GetColumnNames()]
    if len(colNames_central) == 0:
        print(f"{treeName_central} central has no columns")
        createVoidTree(
            os.path.join(outDir, f"{treeName}_Valid.root"), f"{treeName}_Valid.root"
        )
        createVoidTree(
            os.path.join(outDir, f"{treeName}_nonValid.root"),
            f"{treeName}_nonValid.root",
        )
        createVoidTree(
            os.path.join(outDir, f"{treeName}_noDiff.root"), f"{treeName}_noDiff.root"
        )
        return

    df_out = ROOT.RDataFrame(treeName_in, inputFileShifted)
    colNames = [str(c) for c in df_out.GetColumnNames()]
    if len(colNames) == 0:
        print(f"{treeName} shifted has no columns")
        createVoidTree(
            os.path.join(outDir, f"{treeName}_Valid.root"), f"{treeName}_Valid.root"
        )
        createVoidTree(
            os.path.join(outDir, f"{treeName}_nonValid.root"),
            f"{treeName}_nonValid.root",
        )
        createVoidTree(
            os.path.join(outDir, f"{treeName}_noDiff.root"), f"{treeName}_noDiff.root"
        )
        return
    FullEventIdIdx = colNames.index("FullEventId")
    colNames[FullEventIdIdx], colNames[0] = colNames[0], colNames[FullEventIdIdx]
    col_types = [str(df_out.GetColumnType(c)) for c in colNames]
    tuple_maker = ROOT.analysis.TupleMaker(*col_types)(
        ROOT.RDataFrame(treeName_central, inputFileCentral), 4
    )
    print("tuplemaker created")
    df_out = tuple_maker.processOut(ROOT.RDF.AsRNode(df_out))
    print("tuplemaker processed")
    df_out = df_out.Define("isValid", "_entryCentral.use_count() > 0")
    df_unique = df_out.Filter("!isValid")
    df_out_valid = df_out.Filter("isValid")

    colToSave_diff = []

    colToNotToMakeDiff = ["FullEventId"]

    condition_noDiff_list = []
    condition_Valid_list = []

    for var_idx, var_name in enumerate(colNames):
        if var_name in colToNotToMakeDiff:
            continue
        condition_noDiff_list.append(
            f"analysis::IsSame({var_name}, _entryCentral->GetValue<{col_type_dict[col_types[var_idx]]}>({var_idx}))"
        )  # IS SAME TAKES (SHIFT, CENTRAL) !!
    condition_noDiff = " && ".join(condition_noDiff_list)
    df_out_valid = df_out_valid.Define("isSame", condition_noDiff)
    df_out_valid_noDiff = df_out_valid.Filter("isSame")
    df_out_valid_diff = df_out_valid.Filter("!isSame")
    for var_idx, var_name in enumerate(colNames):
        if var_name in colToNotToMakeDiff:
            continue
        df_out_valid_diff = df_out_valid_diff.Define(
            f"{var_name}Diff",
            f"""analysis::Delta({var_name},_entryCentral->GetValue<{col_type_dict[col_types[var_idx]]}>({var_idx}))""",
        )  # DELTA TAKES (SHIFT, CENTRAL) !!
        colToSave_diff.append(f"{var_name}Diff")

    snaps = []
    print("start making screenshot")
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = False
    snapshotOptions.fLazy = True
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = (
        ROOT.ROOT.RCompressionSetting.EAlgorithm.kZLIB
    )
    snapshotOptions.fCompressionLevel = 4
    colToSave_noDiff_v = ListToVector(colToNotToMakeDiff)
    colToSave_diff_v = ListToVector(colToSave_diff + colToNotToMakeDiff)
    colNames_v = ListToVector(colNames)
    outFile_Valid = os.path.join(outDir, f"{treeName}_Valid.root")
    outFile_nonValid = os.path.join(outDir, f"{treeName}_nonValid.root")
    outFile_Valid_noDiff = os.path.join(outDir, f"{treeName}_noDiff.root")
    if os.path.exists(outFile_Valid):
        os.remove(outFile_Valid)
    if os.path.exists(outFile_nonValid):
        os.remove(outFile_nonValid)
    if os.path.exists(outFile_Valid_noDiff):
        os.remove(outFile_Valid_noDiff)
    snaps.append(
        df_out_valid_noDiff.Snapshot(
            f"{treeName}_noDiff",
            outFile_Valid_noDiff,
            colToSave_noDiff_v,
            snapshotOptions,
        )
    )
    snaps.append(
        df_out_valid_diff.Snapshot(
            f"{treeName}_Valid", outFile_Valid, colToSave_diff_v, snapshotOptions
        )
    )
    snaps.append(
        df_unique.Snapshot(
            f"{treeName}_nonValid", outFile_nonValid, colNames_v, snapshotOptions
        )
    )
    tuple_maker.processIn(colNames_v)
    ROOT.RDF.RunGraphs(snaps)

    tuple_maker.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inFileCentral", required=True, type=str)
    parser.add_argument("--inFileShifted", required=True, type=str)
    parser.add_argument("--outDir", required=True, type=str)
    parser.add_argument("--treeName_out", required=False, type=str, default="Events")
    parser.add_argument("--treeName_in", required=False, type=str, default="Events")
    parser.add_argument(
        "--treeName_central", required=False, type=str, default="Events"
    )
    args = parser.parse_args()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['FLAF_PATH']}")
    # header_path_Skimmer = os.path.join(headers_dir, "include/SystSkimmer.h")
    ROOT.gInterpreter.Declare(f'#include "include/SystSkimmer.h"')
    make_df(
        args.inFileCentral,
        args.inFileShifted,
        args.outDir,
        args.treeName_out,
        args.treeName_in,
        args.treeName_central,
    )
    # try : make_df(args.inFileCentral,args.inFileShifted,args.outDir,args.treeName_out,args.treeName_in,args.treeName_central)
    # except:
    #  create_file(os.path.join(args.outDir, f"{args.treeName_out}_Valid.root"))
    #  create_file(os.path.join(args.outDir, f"{args.treeName_out}_nonValid.root"))
    #  create_file(os.path.join(args.outDir, f"{args.treeName_out}_noDiff.root"))
    """
  colNames_Central = [str(c) for c in ROOT.RDataFrame(args.treeName_in, args.inFileCentral).GetColumnNames()]
  colNames_Shifted = [str(c) for c in ROOT.RDataFrame(args.treeName_in, args.inFileShifted).GetColumnNames()]
  if len(colNames_Central)!=0 and len(colNames_Shifted)!=0:
    make_df(args.inFileCentral,args.inFileShifted,args.outDir,args.treeName_out,args.treeName_in,args.treeName_central)
  else:
    create_file(os.path.join(args.outDir, f"{args.treeName_out}_Valid.root"))
    create_file(os.path.join(args.outDir, f"{args.treeName_out}_nonValid.root"))
    create_file(os.path.join(args.outDir, f"{args.treeName_out}_noDiff.root"))
  """
