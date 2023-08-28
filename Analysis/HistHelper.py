import sys
import math
import ROOT
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

import Common.Utilities as Utilities

deepTauYears = {'v2p1':'2017','v2p5':'2018'}
QCDregions = ['OS_Iso', 'SS_Iso', 'OS_AntiIso', 'SS_AntiIso']
categories = ['res2b', 'res1b', 'inclusive']#, 'boosted']
#categories = ['res2b', 'res1b', 'inclusive', 'boosted']
channels = {'eTau':13, 'muTau':23, 'tauTau':33}
triggers = {'eTau':'HLT_singleEle', 'muTau':'HLT_singleMu', 'tauTau':"HLT_ditau"}
signals = ["VBFToRadionToHHTo2B2Tau_M-1000", "GluGluToRadionToHHTo2B2Tau_M-1000", "VBFToRadionToHHTo2B2Tau_M-1250", "GluGluToRadionToHHTo2B2Tau_M-1250", "GluGluToRadionToHHTo2B2Tau_M-1500", "VBFToRadionToHHTo2B2Tau_M-1500", "VBFToRadionToHHTo2B2Tau_M-1750", "GluGluToRadionToHHTo2B2Tau_M-1750", "VBFToRadionToHHTo2B2Tau_M-2000", "GluGluToRadionToHHTo2B2Tau_M-2000", "VBFToRadionToHHTo2B2Tau_M-250", "VBFToRadionToHHTo2B2Tau_M-2500", "GluGluToRadionToHHTo2B2Tau_M-250", "VBFToRadionToHHTo2B2Tau_M-260", "VBFToRadionToHHTo2B2Tau_M-270", "VBFToRadionToHHTo2B2Tau_M-280", "GluGluToRadionToHHTo2B2Tau_M-2500", "VBFToRadionToHHTo2B2Tau_M-300", "VBFToRadionToHHTo2B2Tau_M-3000", "GluGluToRadionToHHTo2B2Tau_M-260", "VBFToRadionToHHTo2B2Tau_M-320", "VBFToRadionToHHTo2B2Tau_M-350", "GluGluToRadionToHHTo2B2Tau_M-270", "VBFToRadionToHHTo2B2Tau_M-400", "VBFToRadionToHHTo2B2Tau_M-450", "GluGluToRadionToHHTo2B2Tau_M-280", "VBFToRadionToHHTo2B2Tau_M-500", "VBFToRadionToHHTo2B2Tau_M-550", "GluGluToRadionToHHTo2B2Tau_M-300", "VBFToRadionToHHTo2B2Tau_M-600", "GluGluToRadionToHHTo2B2Tau_M-3000", "VBFToRadionToHHTo2B2Tau_M-650", "GluGluToRadionToHHTo2B2Tau_M-320", "VBFToRadionToHHTo2B2Tau_M-700", "VBFToRadionToHHTo2B2Tau_M-750", "GluGluToRadionToHHTo2B2Tau_M-350", "VBFToRadionToHHTo2B2Tau_M-800", "GluGluToRadionToHHTo2B2Tau_M-400", "VBFToRadionToHHTo2B2Tau_M-850", "VBFToRadionToHHTo2B2Tau_M-900", "GluGluToRadionToHHTo2B2Tau_M-450", "GluGluToRadionToHHTo2B2Tau_M-500", "GluGluToRadionToHHTo2B2Tau_M-550", "GluGluToRadionToHHTo2B2Tau_M-600", "GluGluToRadionToHHTo2B2Tau_M-650", "GluGluToRadionToHHTo2B2Tau_M-700", "GluGluToRadionToHHTo2B2Tau_M-750", "GluGluToRadionToHHTo2B2Tau_M-800", "GluGluToRadionToHHTo2B2Tau_M-850", "GluGluToRadionToHHTo2B2Tau_M-900", "GluGluToBulkGravitonToHHTo2B2Tau_M-1000","GluGluToBulkGravitonToHHTo2B2Tau_M-1250","GluGluToBulkGravitonToHHTo2B2Tau_M-1500","GluGluToBulkGravitonToHHTo2B2Tau_M-1750","GluGluToBulkGravitonToHHTo2B2Tau_M-2000","GluGluToBulkGravitonToHHTo2B2Tau_M-250","VBFToBulkGravitonToHHTo2B2Tau_M-2500","GluGluToBulkGravitonToHHTo2B2Tau_M-2500","VBFToBulkGravitonToHHTo2B2Tau_M-260","VBFToBulkGravitonToHHTo2B2Tau_M-1250","GluGluToBulkGravitonToHHTo2B2Tau_M-260","VBFToBulkGravitonToHHTo2B2Tau_M-1500","GluGluToBulkGravitonToHHTo2B2Tau_M-270","VBFToBulkGravitonToHHTo2B2Tau_M-1750","VBFToBulkGravitonToHHTo2B2Tau_M-2000","VBFToBulkGravitonToHHTo2B2Tau_M-250","VBFToBulkGravitonToHHTo2B2Tau_M-1000","GluGluToBulkGravitonToHHTo2B2Tau_M-280","VBFToBulkGravitonToHHTo2B2Tau_M-270","GluGluToBulkGravitonToHHTo2B2Tau_M-300","VBFToBulkGravitonToHHTo2B2Tau_M-280","GluGluToBulkGravitonToHHTo2B2Tau_M-3000","VBFToBulkGravitonToHHTo2B2Tau_M-300","GluGluToBulkGravitonToHHTo2B2Tau_M-320","VBFToBulkGravitonToHHTo2B2Tau_M-3000","GluGluToBulkGravitonToHHTo2B2Tau_M-350","VBFToBulkGravitonToHHTo2B2Tau_M-320","GluGluToBulkGravitonToHHTo2B2Tau_M-400","VBFToBulkGravitonToHHTo2B2Tau_M-350","GluGluToBulkGravitonToHHTo2B2Tau_M-450","VBFToBulkGravitonToHHTo2B2Tau_M-400","GluGluToBulkGravitonToHHTo2B2Tau_M-500","VBFToBulkGravitonToHHTo2B2Tau_M-450","GluGluToBulkGravitonToHHTo2B2Tau_M-550","VBFToBulkGravitonToHHTo2B2Tau_M-500","VBFToBulkGravitonToHHTo2B2Tau_M-550","GluGluToBulkGravitonToHHTo2B2Tau_M-600","VBFToBulkGravitonToHHTo2B2Tau_M-600","GluGluToBulkGravitonToHHTo2B2Tau_M-650","VBFToBulkGravitonToHHTo2B2Tau_M-650","GluGluToBulkGravitonToHHTo2B2Tau_M-700","VBFToBulkGravitonToHHTo2B2Tau_M-700","GluGluToBulkGravitonToHHTo2B2Tau_M-750","VBFToBulkGravitonToHHTo2B2Tau_M-750","VBFToBulkGravitonToHHTo2B2Tau_M-800","GluGluToBulkGravitonToHHTo2B2Tau_M-800","VBFToBulkGravitonToHHTo2B2Tau_M-850","GluGluToBulkGravitonToHHTo2B2Tau_M-850","VBFToBulkGravitonToHHTo2B2Tau_M-900","GluGluToBulkGravitonToHHTo2B2Tau_M-900",]
col_type_dict = {
  'Float_t':'float',
  'Bool_t':'bool',
  'Int_t' :'int',
  'ULong64_t' :'unsigned long long',
  'Long64_t' : 'long long',
  'Long_t' :'long',
  'UInt_t' :'unsigned int',
  'Char_t' : 'char',
  'ROOT::VecOps::RVec<Float_t>':'ROOT::VecOps::RVec<float>',
  'ROOT::VecOps::RVec<Int_t>':'ROOT::VecOps::RVec<int>',
  'ROOT::VecOps::RVec<UChar_t>':'ROOT::VecOps::RVec<unsigned char>',
  'ROOT::VecOps::RVec<float>':'ROOT::VecOps::RVec<float>',
  'ROOT::VecOps::RVec<int>':'ROOT::VecOps::RVec<int>',
  'ROOT::VecOps::RVec<unsigned char>':'ROOT::VecOps::RVec<unsigned char>'
  }
def defineP4(df, name):
    df = df.Define(f"{name}_p4", f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>({name}_pt,{name}_eta,{name}_phi,{name}_mass)")
    return df

def defineAllP4(df):
    for idx in [0,1]:
        df = defineP4(df, f"tau{idx+1}")
        df = defineP4(df, f"b{idx+1}")
        #df = defineP4(df, f"tau{idx+1}_seedingJet")
    return df

def createInvMass(df):
    df = df.Define("tautau_m_vis", "(tau1_p4+tau2_p4).M()")
    df = df.Define("bb_m_vis", "(b1_p4+b2_p4).M()")
    df = df.Define("bbtautau_mass", "(b1_p4+b2_p4+tau1_p4+tau2_p4).M()")
    df = df.Define("dR_tautau", 'ROOT::Math::VectorUtil::DeltaR(tau1_p4, tau2_p4)')
    return df

def RenormalizeHistogram(histogram, norm, include_overflows=True):
    integral = histogram.Integral(0, histogram.GetNbinsX()+1) if include_overflows else histogram.Integral()
    histogram.Scale(norm / integral)

def FixNegativeContributions(histogram):
    correction_factor = 0.

    ss_debug = ""
    ss_negative = ""

    original_Integral = histogram.Integral(0, histogram.GetNbinsX()+1)
    ss_debug += "\nSubtracted hist for '{}'.\n".format(histogram.GetName())
    ss_debug += "Integral after bkg subtraction: {}.\n".format(original_Integral)
    if original_Integral < 0:
        print(ss_debug)
        print("Integral after bkg subtraction is negative for histogram '{}'".format(histogram.GetName()))
        return False,ss_debug, ss_negative

    for n in range(1, histogram.GetNbinsX()+1):
        if histogram.GetBinContent(n) >= 0:
            continue
        prefix = "WARNING" if histogram.GetBinContent(n) + histogram.GetBinError(n) >= 0 else "ERROR"

        ss_negative += "{}: {} Bin {}, content = {}, error = {}, bin limits=[{},{}].\n".format(
            prefix, histogram.GetName(), n, histogram.GetBinContent(n), histogram.GetBinError(n),
            histogram.GetBinLowEdge(n), histogram.GetBinLowEdge(n+1))

        error = correction_factor - histogram.GetBinContent(n)
        new_error = math.sqrt(math.pow(error, 2) + math.pow(histogram.GetBinError(n), 2))
        histogram.SetBinContent(n, correction_factor)
        histogram.SetBinError(n, new_error)

    RenormalizeHistogram(histogram, original_Integral, True)
    return True, ss_debug, ss_negative

class DataFrameBuilder:

    def defineSelectionRegions(self):
        self.df = self.df.Define("nSelBtag", f"int(b1_idbtagDeepFlavB >= {self.bTagWP}) + int(b2_idbtagDeepFlavB >= {self.bTagWP})")
        self.df = self.df.Define("res1b", f"nSelBtag == 1")
        self.df = self.df.Define("res2b", f"nSelBtag == 2")

    def defineQCDRegions(self):
        tau2_iso_var = f"tau2_idDeepTau{self.deepTauYear()}{self.deepTauVersion}VSjet"
        self.df = self.df.Define("OS", "tau1_charge*tau2_charge < 0")
        self.df = self.df.Define("Iso", f"{tau2_iso_var} >= {Utilities.WorkingPointsTauVSjet.Medium.value}")
        self.df = self.df.Define("AntiIso", f"{tau2_iso_var} >= {Utilities.WorkingPointsTauVSjet.VVVLoose.value} && !Iso")
        self.df = self.df.Define("OS_Iso", f"OS && Iso")
        self.df = self.df.Define("SS_Iso", f"!OS && Iso")
        self.df = self.df.Define("OS_AntiIso", f"OS && AntiIso")
        self.df = self.df.Define("SS_AntiIso", f"!OS && AntiIso")

    def deepTauYear(self):
        return deepTauYears[self.deepTauVersion]

    def selectTrigger(self, trigger):
        self.df = self.df.Filter(trigger)

    def CreateColumnTypes(self):
        colNames = [str(c) for c in self.df.GetColumnNames()]
        entryIndexIdx = colNames.index("entryIndex")
        colNames[entryIndexIdx], colNames[0] = colNames[0], colNames[entryIndexIdx]
        self.colNames = colNames
        self.colTypes = [str(self.df.GetColumnType(c)) for c in self.colNames]

    def __init__(self, df, deepTauVersion='v2p1'):
        self.df = df
        self.colNames=[]
        self.colTypes=[]
        self.deepTauVersion = deepTauVersion
        self.bTagWP = 2
        self.var_list = []
        self.CreateColumnTypes()


    def GetEventsFromShifted(self, df_central):
        df_final = df_central.Filter(""" std::find ( analysis::GetEntriesVec().begin(), analysis::GetEntriesVec().end(),
                                     entryIndex ) != analysis::GetEntriesVec().end()""")
        self.df=df_final

    def CreateFromDelta(self,var_list,central_columns,central_col_types):
        for var_idx,var_name in enumerate(self.colNames):
            if not var_name.endswith("Diff"):
                continue
            var_name_forDelta = var_name.removesuffix("Diff")
            central_col_idx = central_columns.index(var_name_forDelta)
            if central_columns[central_col_idx]!=var_name_forDelta:
                raise RuntimeError(f"CreateFromDelta: {central_columns[central_col_idx]} != {var_name_forDelta}")
            self.df = self.df.Define(f"{var_name_forDelta}", f"""analysis::FromDelta({var_name},
                                     analysis::GetEntriesMap()[entryIndex]->GetValue<{col_type_dict[self.colTypes[var_idx]]}>({central_col_idx}) )""")
            var_list.append(f"{var_name_forDelta}")
        for central_col_idx,central_col in enumerate(central_columns):
            if central_col in var_list or central_col in self.colNames: continue
            if central_col != 'channelId' : continue # this is for a bugfix that I still haven't figured out !!

            self.df = self.df.Define(central_col, f"""analysis::GetEntriesMap()[entryIndex]->GetValue<{central_col_types[central_col_idx]}>({central_col_idx})""")


def createModel(hist_cfg, var):
    hists = {}
    x_bins = hist_cfg[var]['x_bins']
    if type(hist_cfg[var]['x_bins'])==list:
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        model = ROOT.RDF.TH1DModel("", "", x_bins_vec.size()-1, x_bins_vec.data())
    else:
        n_bins, bin_range = x_bins.split('|')
        start,stop = bin_range.split(':')
        model = ROOT.RDF.TH1DModel("", "",int(n_bins), float(start), float(stop))
    return model




def PrepareDfWrapped(dfWrapped):
    dfWrapped.df = defineAllP4(dfWrapped.df)
    dfWrapped.df = createInvMass(dfWrapped.df)
    dfWrapped.defineQCDRegions()
    dfWrapped.defineSelectionRegions()
    return dfWrapped

def GetWeight(cat, channel, btag_wp):
    btag_weight = "1"
    if cat!='inclusive':
        btag_weight = f"weight_bTagSF_{btag_wp}_Central"
    trg_weights_dict = {
        'eTau':["weight_tau1_TrgSF_singleEle_Central","weight_tau2_TrgSF_singleEle_Central"],
        'muTau':["weight_tau1_TrgSF_singleMu_Central","weight_tau2_TrgSF_singleMu_Central"],
        'tauTau':["weight_tau1_TrgSF_ditau_Central","weight_tau2_TrgSF_ditau_Central"]
        }
    weights_to_apply = [ "weight_Jet_PUJetID_Central_b1", "weight_Jet_PUJetID_Central_b2", "weight_TauID_Central", btag_weight, "weight_tau1_EleidSF_Central", "weight_tau1_MuidSF_Central", "weight_tau2_EleidSF_Central", "weight_tau2_MuidSF_Central","weight_total"]
    #weights_to_apply = [  "weight_TauID_Central",btag_weight, "weight_tau1_EleidSF_Central", "weight_tau1_MuidSF_Central", "weight_tau2_EleidSF_Central", "weight_tau2_MuidSF_Central","weight_total"]
    #weights_to_apply = [ "weight_total"]

    weights_to_apply.extend(trg_weights_dict[channel])
    total_weight = '*'.join(weights_to_apply)
    return total_weight

def GetRelativeWeights(column_names):
    return [col for col in column_names if "weight" in col and col.endswith("rel")]
