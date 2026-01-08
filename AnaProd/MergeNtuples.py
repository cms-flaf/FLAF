import ROOT
import os
import sys

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.TupleHelpers import copyFileContent

ROOT.gInterpreter.Declare(
    """
    using lumiEventMapType = std::map < unsigned int, std::set <unsigned long long > >;
    using eventMapType =  std::map < unsigned int ,  lumiEventMapType >;
    using RunLumiEventSet = std::set < std::tuple < unsigned int, unsigned int, unsigned long long > > ;
    bool saveEvent(unsigned int run, unsigned int lumi, unsigned long long event){
        static eventMapType eventMap;
        static std::mutex eventMap_mutex;
        const std::lock_guard<std::mutex> lock(eventMap_mutex);
        auto& events = eventMap[run][lumi];
        if(events.find(event) != events.end())
            return false;
        events.insert(event);
        return true;
        }
    """
)


class ObjDesc:
    def __init__(self, obj_type):
        self.obj_type = obj_type
        self.ref_obj = None
        self.objsToMerge = ROOT.TList()
        self.file_names = []


def merge_ntuples(df):
    if (
        "run" not in df.GetColumnNames()
        or "luminosityBlock" not in df.GetColumnNames()
        or "event" not in df.GetColumnNames()
    ):
        return df
    df = df.Filter("saveEvent(run, luminosityBlock, event)")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile", nargs="+", type=str)
    parser.add_argument("--outFile", required=True, type=str)
    parser.add_argument("--useUproot", type=bool, default=False)
    parser.add_argument("--apply-filter", required=False, type=int, default=0)
    parser.add_argument("--outFiles", required=False, nargs="+", type=str, default=None)
    args = parser.parse_args()
    headers_dir = os.path.dirname(os.path.abspath(__file__))

    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists = False
    snapshotOptions.fLazy = False
    snapshotOptions.fMode = "UPDATE"
    snapshotOptions.fCompressionAlgorithm = getattr(
        ROOT.ROOT.RCompressionSetting.EAlgorithm, "kLZMA"
    )
    snapshotOptions.fCompressionLevel = 9
    inputFiles = [
        (fileName, ROOT.TFile(fileName, "READ")) for fileName in args.inputFile
    ]
    objects = {}
    for fileName, file in inputFiles:
        for key in file.GetListOfKeys():
            key_name = key.GetName()
            obj = key.ReadObj()

            if obj.IsA().InheritsFrom(ROOT.TTree.Class()):
                objType = "TTree"
            elif obj.IsA().InheritsFrom(ROOT.TH1.Class()):
                objType = "TH1"
            else:
                raise RuntimeError(f"Object {obj} is not a TTree or TH1.")

            if key_name not in objects:
                objects[key_name] = ObjDesc(objType)
            obj_desc = objects[key_name]
            if objType != obj_desc.obj_type:
                raise RuntimeError(
                    f"Object {key_name} is of type {objType} but was previously found to be"
                    f"of type {obj_desc.obj_type}."
                )
            if objType == "TTree":
                if fileName in obj_desc.file_names:
                    continue  # When there were multiple cycles of a tree, without this catch you would append the same file multiple times
                obj_desc.file_names.append(fileName)
            else:
                if obj_desc.ref_obj is None:
                    obj_desc.ref_obj = obj
                else:
                    obj_desc.objsToMerge.Add(obj)

    tmpFileName = args.outFile
    if args.useUproot:
        tmpFileName += ".tmp.root"
    compression = ROOT.CompressionSettings(
        snapshotOptions.fCompressionAlgorithm, snapshotOptions.fCompressionLevel
    )
    # Set final output file name for the TH1's, use only first one now since TH1 can't be split into multiple files
    report_file_name = (
        args.outFiles[0]
        if args.outFiles is not None and len(args.outFiles) > 0
        else tmpFileName
    )
    outputFile = ROOT.TFile(report_file_name, "RECREATE", "", compression)
    for obj_name, obj_desc in objects.items():
        if obj_desc.obj_type != "TH1":
            continue
        obj_desc.ref_obj.Merge(obj_desc.objsToMerge)
        outputFile.WriteTObject(obj_desc.ref_obj, obj_name)
    outputFile.Close()
    for fileName, file in inputFiles:
        file.Close()

    # Delete tmp file if it exists
    outputFile = ROOT.TFile(tmpFileName, "RECREATE", "", compression)
    outputFile.Close()

    for obj_name, obj_desc in objects.items():
        print(f"At writing for {obj_name} {obj_desc}")
        if obj_desc.obj_type != "TTree":
            continue
        print(f"Loading {obj_name}, {obj_desc.file_names}")
        df = ROOT.RDataFrame(obj_name, obj_desc.file_names)
        print(f"initially there are {df.Count().GetValue()} events")
        if args.apply_filter:
            df = merge_ntuples(df)
        df.Snapshot(obj_name, tmpFileName, df.GetColumnNames(), snapshotOptions)
    # print(tmpFileName)
    if args.useUproot:
        copyFileContent(tmpFileName, args.outFile)
        # if os.path.exists(args.outFile):
        #    os.remove(tmpFileName)

    nTTrees = 0
    for obj_name, obj_desc in objects.items():
        if obj_desc.obj_type == "TTree":
            nTTrees += 1

    if args.outFiles != None and len(args.outFiles) > 0:
        if (
            len(args.outFiles) > 1 and nTTrees > 1
        ):  # Need this instead of objects, data was failing becuse it contained Report and Time
            for obj_name, obj_desc in objects.items():
                print(obj_name, obj_desc.obj_type)
            raise RuntimeError(
                "Cannot split multiple objects into multiple files, please use a single object or a single file."
            )
        for obj_name, obj_desc in objects.items():
            print(f"At splitting for {obj_name} {obj_desc}")
            # We want to make this general to also split MC later, but these object names could cause an issue?
            # Is there an edge case where a Events_JERUp event might go to file1, but the Event event might go to file0?
            # For now, this doesn't exist as MC is handle as one outFile per job -- only data has multiple outFiles
            if obj_desc.obj_type != "TTree":
                continue
            df = ROOT.RDataFrame(obj_name, tmpFileName)
            nFiles = len(args.outFiles)
            nEvents = df.Count().GetValue()
            nEventsPerFile = (
                int(nEvents / nFiles) + 1
            )  # Add 1 to take care of int() always flooring
            print(
                f"Going to make int({nEvents}/{nFiles}) {nEventsPerFile} events per file"
            )
            range_start = 0
            for outFileName in args.outFiles:
                df_split = df.Range(range_start, range_start + nEventsPerFile)
                df_split.Snapshot(
                    obj_name, outFileName, df_split.GetColumnNames(), snapshotOptions
                )
                range_start += nEventsPerFile
