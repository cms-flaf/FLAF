import ROOT
import FLAF.Common.LegacyVariables as LegacyVariables
import FLAF.Common.Utilities as Utilities
import sys
import os


def applyLegacyVariables(df):
    sys.path.append(os.environ["ANALYSIS_PATH"])
    ROOT.gROOT.ProcessLine(".include " + os.environ["ANALYSIS_PATH"])
    defaultColToSave = [col for col in df.GetColumnNames()]
    LegacyVariables.Initialize()
    dfw = Utilities.DataFrameWrapper(df, defaultColToSave)
    MT2Branches = dfw.Apply(LegacyVariables.GetMT2)
    KinFitBranches = dfw.Apply(LegacyVariables.GetKinFit)
    SVFitBranches = dfw.Apply(LegacyVariables.GetSVFit)
    return dfw.df
