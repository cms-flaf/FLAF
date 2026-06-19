from .run_tools import ps_call
import os
import re
import ROOT

pattern = "=|\n"


def includeLibTool(tool="", wantLib=False):
    command = ["scram", "tool", "info", tool]
    directory = os.environ["CMSSW_BASE"]
    returncode, output, err = ps_call(
        command, catch_stdout=True, cwd=directory, verbose=0
    )
    result = re.split(pattern, output)
    include_path = result[result.index("INCLUDE") + 1]
    ROOT.gInterpreter.AddIncludePath(include_path)
    if "LIBDIR" in result and wantLib:
        lib_path = result[result.index("LIBDIR") + 1]
        ROOT.gSystem.Load(f"{lib_path}/lib{tool}.so")
        # if(tool=="tensorflow"):
        # ROOT.gSystem.Load(f"{lib_path}/lib{tool}_framework.so")
        # ROOT.gSystem.Load(f"{lib_path}/lib{tool}_cc.so")
        # ROOT.gSystem.Load(f"{lib_path}/libtf2xla.so")
    if "ROOT_INCLUDE_PATH" in result:
        root_include_path = result[result.index("ROOT_INCLUDE_PATH") + 1]
        ROOT.gInterpreter.AddIncludePath(root_include_path)
