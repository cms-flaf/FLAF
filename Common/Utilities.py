from enum import Enum
from inspect import signature
import ROOT
import base64
import copy
import importlib
import os
import pickle
import sys


class WorkingPointsTauVSmu(Enum):
    VLoose = 1
    Loose = 2
    Medium = 3
    Tight = 4


class WorkingPointsTauVSjet(Enum):
    VVVLoose = 1
    VVLoose = 2
    VLoose = 3
    Loose = 4
    Medium = 5
    Tight = 6
    VTight = 7
    VVTight = 8


class WorkingPointsTauVSe(Enum):
    VVVLoose = 1
    VVLoose = 2
    VLoose = 3
    Loose = 4
    Medium = 5
    Tight = 6
    VTight = 7
    VVTight = 8


class WorkingPointsBoostedTauVSjet(Enum):
    VVLoose = 1
    VLoose = 2
    Loose = 3
    Medium = 4
    Tight = 5
    VTight = 6
    VVTight = 7


class WorkingPointsbTag(Enum):
    Loose = 1
    Medium = 2
    Tight = 3


class WorkingPointsMuonID(Enum):
    HighPtID = 1
    LooseID = 2
    MediumID = 3
    MediumPromptID = 4
    SoftID = 5
    TightID = 6
    TrkHighPtID = 7


class MuonPfIsoID_WP(Enum):
    VeryLoose = 1
    Loose = 2
    Medium = 3
    Tight = 4
    VeryTight = 5
    VeryVeryTight = 6


deepTauVersions = {"2p1": "2017", "2p5": "2018"}


def defineP4(df, name):
    df = df.Define(
        f"{name}_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>({name}_pt,{name}_eta,{name}_phi,{name}_mass)",
    )
    return df


def mkdir(file, path):
    dir_names = path.split("/")
    current_dir = file
    for n, dir_name in enumerate(dir_names):
        dir_obj = current_dir.Get(dir_name)
        full_name = f"{file.GetPath()}" + "/".join(dir_names[:n])
        if dir_obj:
            if not dir_obj.IsA().InheritsFrom(ROOT.TDirectory.Class()):
                raise RuntimeError(
                    f"{dir_name} already exists in {full_name} and it is not a directory"
                )
        else:
            dir_obj = current_dir.mkdir(dir_name)
            if not dir_obj:

                raise RuntimeError(f"Failed to create {dir_name} in {full_name}")
        current_dir = dir_obj
    return current_dir


def ListToVector(list, type="string"):
    vec = ROOT.std.vector(type)()
    for item in list:
        vec.push_back(item)
    return vec


rootAnaPathSet = False


def DeclareHeader(header, verbose=0):
    global rootAnaPathSet
    if not rootAnaPathSet:
        if verbose > 0:
            print(f'Adding "{os.environ["ANALYSIS_PATH"]}" to the ROOT include path')
        ROOT.gROOT.ProcessLine(".include " + os.environ["ANALYSIS_PATH"])
        rootAnaPathSet = True
    if verbose > 0:
        print(f'Including "{header}"')
    if not os.path.exists(header):
        raise RuntimeError(f'"{header}" does not exist')
    if not ROOT.gInterpreter.Declare(f'#include "{header}"'):
        raise RuntimeError(f"Failed to include {header}")
    if verbose > 0:
        print(f'Successfully included "{header}"')


def generate_enum_class(cls):
    enum_string = "enum class {} : int {{\n".format(cls.__name__)
    for item in cls:
        enum_string += "    {} = {},\n".format(item.name, item.value)
    enum_string += "};"
    return enum_string


class DataFrameWrapper:
    def __init__(self, df, defaultColToSave=[]):
        self.df = df
        self.colToSave = copy.deepcopy(defaultColToSave)

    def Define(self, varToDefine, varToCall):
        self.df = self.df.Define(varToDefine, varToCall)

    def Redefine(self, varToDefine, varToCall):
        self.df = self.df.Redefine(varToDefine, varToCall)

    def Filter(self, filter_str, filter_name=""):
        self.df = self.df.Filter(filter_str, filter_name)

    def DefineAndAppend(self, varToDefine, varToCall):
        self.Define(varToDefine, varToCall)
        self.colToSave.append(varToDefine)

    def RedefineAndAppend(self, varToDefine, varToCall):
        self.Redefine(varToDefine, varToCall)
        self.colToSave.append(varToDefine)

    def Apply(self, func, *args, **kwargs):
        result = func(self.df, *args, **kwargs)
        if isinstance(result, tuple):
            self.df = result[0]
            if len(result) == 2:
                return result[1]
            return result[1:]
        else:
            self.df = result


class DataFrameBuilderBase:
    def __init__(self, df):
        self.df = df


def CreateDataFrame(
    *,
    treeName,
    fileName,
    caches,
    files,
    centralTree=None,
    centralCaches=None,
    central="Central",
    specialColumns=["FullEventId"],
    valid_column="valid",
    filter_valid=True,
):

    def GetFile(file_name):
        if file_name not in files:
            file = ROOT.TFile.Open(file_name)
            files[file_name] = file
        return files[file_name]

    def GetTree(treeName, fileName):
        file = GetFile(fileName)
        tree = file.Get(treeName)
        if tree is None or not tree or tree.IsZombie():
            raise RuntimeError(f"ERROR: tree {treeName} not found in file {fileName}")
        if type(tree) != ROOT.TTree:
            raise RuntimeError(
                f"ERROR: object {treeName} in file {fileName} has type {type(tree)}, while a TTree is expected."
            )
        return tree

    tree = GetTree(treeName, fileName)
    cacheTrees = {}
    for cacheName, cacheFileName in caches.items():
        cacheTree = GetTree(treeName, cacheFileName)
        tree.AddFriend(cacheTree, cacheName)
        cacheTrees[cacheName] = cacheTree
    if centralTree is not None:
        tree.AddFriend(centralTree, central)
    if centralCaches is not None:
        for cacheName, cacheTree in centralCaches.items():
            tree.AddFriend(cacheTree, f"{cacheName}__{central}")
    df_orig = ROOT.RDataFrame(tree)
    df = df_orig
    columns = [str(c) for c in df.GetColumnNames()]
    for column in columns:
        origin_split = column.split(".")
        if len(origin_split) == 2:
            origin = origin_split[0]
            full_name = origin_split[1]
        elif len(origin_split) == 1:
            origin = None
            full_name = origin_split[0]
        else:
            raise RuntimeError(
                f"Invalid column name: {column}. Unable to parse origin and name."
            )
        if origin is not None and (
            origin == central or origin.endswith(f"__{central}")
        ):
            continue
        name_split = full_name.split("__")
        if len(name_split) == 2:
            suffix = name_split[1]
            if suffix != "delta":
                raise RuntimeError(f"Unknown column suffix: {suffix}")
            column_name = name_split[0]
        elif len(name_split) == 1:
            column_name = name_split[0]
            suffix = None
        else:
            raise RuntimeError(
                f"Invalid column name: {column}. Unable to parse name and suffix."
            )
        if column_name in specialColumns or column_name == valid_column:
            continue
        if suffix is None:
            if origin is not None:
                df = df.Redefine(column_name, column)
        elif column_name not in columns:
            central_valid = f"{central}.{valid_column}"
            if origin is None:
                central_column = f"{central}.{column_name}"
            else:
                central_column = f"{origin}__{central}.{column_name}"
            for c in [central_column, valid_column, central_valid]:
                if c not in columns:
                    print("Available columns:", sorted(columns))
                    raise RuntimeError(
                        f"Column {c} needed to compute {column_name} from {column} is missing."
                    )
            df = df.Redefine(
                column_name,
                f"::analysis::FromDelta({column}, {central_column}, {valid_column}, {central_valid})",
            )
    if filter_valid:
        df = df.Filter(valid_column)
    return df_orig, df, tree, cacheTrees


def GetValues(collection):
    for key, value in collection.items():
        if isinstance(value, dict):
            GetValues(value)
        else:
            collection[key] = value.GetValue()
    return collection


def GetKeyNames(file, dir=""):
    if dir != "":
        file.cd(dir)
    return [str(key.GetName()) for key in ROOT.gDirectory.GetListOfKeys()]


def SerializeObjectToString(obj):
    obj_pkl = pickle.dumps(obj)
    return base64.b64encode(obj_pkl).decode()


def DeserializeObjectFromString(string):
    obj_pkl = base64.b64decode(string.encode())
    return pickle.loads(obj_pkl)


def load_module(module_path):
    module_file = os.path.basename(module_path)
    module_name, module_ext = os.path.splitext(module_file)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def getCustomisationSplit(customisations):
    customisation_dict = {}
    if customisations is None or len(customisations) == 0:
        return {}
    if type(customisations) == str:
        customisations = customisations.split(";")
    if type(customisations) != list:
        raise RuntimeError(f"Invalid type of customisations: {type(customisations)}")
    for customisation in customisations:
        substrings = customisation.split("=")
        if len(substrings) != 2:
            raise RuntimeError("len of substring is not 2!")
        customisation_dict[substrings[0]] = substrings[1]
    return customisation_dict


# generic function allowing to choose CRC type
# now chosen: CRC-16-CCITT (TRUE)
# needed temporarly until fastcrc is compatible with cmsEnv def or anatuple producer will be fully independent on cmsEnv.


def crc16(
    data: bytes, poly: int = 0x1021, init_val: int = 0xFFFF, reflect: bool = False
) -> int:
    crc = init_val
    for byte in data:
        if reflect:
            byte = int("{:08b}".format(byte)[::-1], 2)
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return crc & 0xFFFF


def load_processor(p_entry, stage, global_params, verbose=0):
    try:
        module = importlib.import_module(p_entry["module"])
        cls = getattr(module, p_entry["class"])
        init_sig = signature(cls.__init__)
        kwargs = {
            "global_params": global_params,
            "processor_entry": p_entry,
            "stage": stage,
            "verbose": verbose,
        }
        to_remove = []
        for key in kwargs:
            if key not in init_sig.parameters:
                to_remove.append(key)
        for key in to_remove:
            del kwargs[key]
        instance = cls(**kwargs)
        if verbose > 0:
            print(
                f"Loaded processor: {p_entry['name']}",
                file=sys.stderr,
            )
        return instance
    except Exception as e:
        print(
            f"Failed to load processor {p_entry}: {e}",
            file=sys.stderr,
        )
        raise


def create_processor_instances(global_params, processor_entries, stage, verbose=0):
    processor_instances = {}
    if processor_entries:
        for p_entry in processor_entries:
            p_name = p_entry["name"]
            if p_name in processor_instances:
                raise RuntimeError(
                    f"Processor {p_name} already exists in anaCache computation"
                )
            processor = load_processor(p_entry, stage, global_params, verbose=verbose)
            processor_instances[p_name] = processor
    return processor_instances


WPInit = False


def InitializeCorrections(setup, dataset_name, stage):
    from Corrections.Corrections import Corrections
    import FLAF.Common.triggerSel as Triggers

    global WPInit
    if not WPInit:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        for include_path in ["ANALYSIS_PATH", "FLAF_PATH"]:
            if include_path in os.environ:
                ROOT.gROOT.ProcessLine(".include " + os.environ[include_path])
        header_path_RootExt = "include/RootExt.h"
        header_path_GenLepton = "include/GenLepton.h"
        header_path_Gen = "include/BaselineGenSelection.h"
        header_path_Reco = "include/BaselineRecoSelection.h"
        header_path_AnalysisMath = "include/AnalysisMath.h"
        ROOT.gInterpreter.Declare(f'#include "{header_path_RootExt}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_GenLepton}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_Gen}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_Reco}"')
        ROOT.gInterpreter.Declare(f'#include "{header_path_AnalysisMath}"')
        for wpcl in [
            WorkingPointsTauVSe,
            WorkingPointsTauVSmu,
            WorkingPointsTauVSjet,
            WorkingPointsbTag,
            WorkingPointsMuonID,
        ]:
            ROOT.gInterpreter.Declare(f"{generate_enum_class(wpcl)}")
        WPInit = True
    isData = dataset_name == "data"
    dataset_cfg = {} if isData else setup.datasets[dataset_name]
    process_name = "data" if isData else dataset_cfg["process_name"]
    process = {} if isData else setup.base_processes[process_name]
    processors_cfg, processor_instances = (
        {},
        (
            {}
            if isData
            else setup.get_processors(process_name, stage=stage, create_instances=True)
        ),
    )

    triggerFile = setup.global_params.get("triggerFile")
    trigger_class = None
    if triggerFile is not None:
        triggerFile = os.path.join(os.environ["ANALYSIS_PATH"], triggerFile)
        trigger_class = Triggers.Triggers(triggerFile)
    if Corrections._global_instance is None:
        Corrections.initializeGlobal(
            setup=setup,
            stage=stage,
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            process_name=process_name,
            process_cfg=process,
            processors=processor_instances,
            isData=isData,
            load_corr_lib=True,
            trigger_class=trigger_class,
        )


class ServiceThread:
    def __init__(self):
        import threading
        from FLAF.RunKit.crabLaw import cond, update_kinit_thread

        self.cond = cond
        self.thread = threading.Thread(target=update_kinit_thread)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cond.acquire()
        self.cond.notify_all()
        self.cond.release()
        self.thread.join()
