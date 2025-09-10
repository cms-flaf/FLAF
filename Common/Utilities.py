from enum import Enum
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
    def CreateColumnTypes(self):
        colNames = [
            str(c) for c in self.df.GetColumnNames()
        ]  # if 'kinFit_result' not in str(c)]
        FullEventIdIdx = 0
        if "FullEventId" in colNames:
            FullEventIdIdx = colNames.index("FullEventId")
        if "entryIndex" in colNames:
            FullEventIdIdx = colNames.index("entryIndex")
        colNames[FullEventIdIdx], colNames[0] = colNames[0], colNames[FullEventIdIdx]
        self.colNames = colNames
        self.colTypes = [str(self.df.GetColumnType(c)) for c in self.colNames]

    def __init__(self, df, **kwargs):
        self.df = df
        self.colNames = []
        self.colTypes = []
        self.var_list = []
        self.CreateColumnTypes()

    def CreateFromDelta(self, central_columns, central_col_types):
        var_list = []
        for var_idx, var_name in enumerate(self.colNames):
            if not var_name.endswith("Diff"):
                continue
            var_name_forDelta = var_name.removesuffix("Diff")
            central_col_idx = central_columns.index(var_name_forDelta)
            if central_columns[central_col_idx] != var_name_forDelta:
                raise RuntimeError(
                    f"CreateFromDelta: {central_columns[central_col_idx]} != {var_name_forDelta}"
                )
            if var_name_forDelta in self.df.GetColumnNames():
                print(f"{var_name_forDelta} already exists, redefining it")
                if var_name_forDelta.startswith("n"):
                    self.df = self.df.Redefine(f"{var_name_forDelta}", f"""analysis::FromDelta({var_name}, analysis::GetEntriesMap()[FullEventId]->GetValue<{self.colTypes[var_idx]}>({central_col_idx}) )""")
                else: continue
            else:
                self.df = self.df.Define(
                    f"{var_name_forDelta}",
                    f"""analysis::FromDelta({var_name},
                                        analysis::GetEntriesMap()[FullEventId]->GetValue<{self.colTypes[var_idx]}>({central_col_idx}) )""",
                )
            var_list.append(f"{var_name_forDelta}")

    def AddMissingColumns(self, central_columns, central_col_types, verbose=0):
        for central_col_idx, central_col in enumerate(central_columns):
            if central_col in self.df.GetColumnNames():
                continue
            if verbose > 0:
                print(
                    f"Adding missing column {central_col} of type {central_col_types[central_col_idx]}"
                )
            self.df = self.df.Define(
                central_col,
                f"""analysis::GetEntriesMap()[FullEventId]->GetValue<{central_col_types[central_col_idx]}>({central_col_idx})""",
            )

    def AddCacheColumns(self, cache_cols, cache_col_types, cache_name):
        for cache_col_idx, cache_col in enumerate(cache_cols):
            if cache_col in self.df.GetColumnNames():
                continue
            if cache_col.replace(".", "_") in self.df.GetColumnNames():
                continue
            self.df = self.df.Define(
                cache_col.replace(".", "_"),
                f"""analysis::GetCacheEntriesMap("{cache_name}").at(FullEventId)->GetValue<{cache_col_types[cache_col_idx]}>({cache_col_idx})""",
            )


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


def create_file(file_name, times=None):
    with open(file_name, "w"):
        os.utime(file_name, times)


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
