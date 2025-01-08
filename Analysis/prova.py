import ROOT

def GetTauIDTotalWeight(df):
    prod_central_1 = "weight_tau1_TauID_SF_Medium_genuineElectron_barrelCentral * weight_tau1_TauID_SF_Medium_genuineElectron_endcapsCentral * weight_tau1_TauID_SF_Medium_genuineMuon_eta0p4to0p8Central * weight_tau1_TauID_SF_Medium_genuineMuon_eta0p8to1p2Central * weight_tau1_TauID_SF_Medium_genuineMuon_eta1p2to1p7Central * weight_tau1_TauID_SF_Medium_genuineMuon_etaGt1p7Central * weight_tau1_TauID_SF_Medium_genuineMuon_etaLt0p4Central * weight_tau1_TauID_SF_Medium_stat1_dm0Central * weight_tau1_TauID_SF_Medium_stat1_dm10Central * weight_tau1_TauID_SF_Medium_stat1_dm11Central * weight_tau1_TauID_SF_Medium_stat1_dm1Central * weight_tau1_TauID_SF_Medium_stat2_dm0Central * weight_tau1_TauID_SF_Medium_stat2_dm10Central * weight_tau1_TauID_SF_Medium_stat2_dm11Central * weight_tau1_TauID_SF_Medium_stat2_dm1Central * weight_tau1_TauID_SF_Medium_stat_highpT_bin1Central * weight_tau1_TauID_SF_Medium_stat_highpT_bin2Central * weight_tau1_TauID_SF_Medium_syst_allerasCentral * weight_tau1_TauID_SF_Medium_syst_highpTCentral * weight_tau1_TauID_SF_Medium_syst_highpT_bin1Central * weight_tau1_TauID_SF_Medium_syst_highpT_bin2Central * weight_tau1_TauID_SF_Medium_syst_highpT_extrapCentral * weight_tau1_TauID_SF_Medium_syst_yearCentral * weight_tau1_TauID_SF_Medium_syst_year_dm0Central * weight_tau1_TauID_SF_Medium_syst_year_dm10Central * weight_tau1_TauID_SF_Medium_syst_year_dm11Central * weight_tau1_TauID_SF_Medium_syst_year_dm1Central "
    prod_central_2 = "weight_tau2_TauID_SF_Medium_genuineElectron_barrelCentral * weight_tau2_TauID_SF_Medium_genuineElectron_endcapsCentral * weight_tau2_TauID_SF_Medium_genuineMuon_eta0p4to0p8Central * weight_tau2_TauID_SF_Medium_genuineMuon_eta0p8to1p2Central * weight_tau2_TauID_SF_Medium_genuineMuon_eta1p2to1p7Central * weight_tau2_TauID_SF_Medium_genuineMuon_etaGt1p7Central * weight_tau2_TauID_SF_Medium_genuineMuon_etaLt0p4Central * weight_tau2_TauID_SF_Medium_stat1_dm0Central * weight_tau2_TauID_SF_Medium_stat1_dm10Central * weight_tau2_TauID_SF_Medium_stat1_dm11Central * weight_tau2_TauID_SF_Medium_stat1_dm1Central * weight_tau2_TauID_SF_Medium_stat2_dm0Central * weight_tau2_TauID_SF_Medium_stat2_dm10Central * weight_tau2_TauID_SF_Medium_stat2_dm11Central * weight_tau2_TauID_SF_Medium_stat2_dm1Central * weight_tau2_TauID_SF_Medium_stat_highpT_bin1Central * weight_tau2_TauID_SF_Medium_stat_highpT_bin2Central * weight_tau2_TauID_SF_Medium_syst_allerasCentral * weight_tau2_TauID_SF_Medium_syst_highpTCentral * weight_tau2_TauID_SF_Medium_syst_highpT_bin1Central * weight_tau2_TauID_SF_Medium_syst_highpT_bin2Central * weight_tau2_TauID_SF_Medium_syst_highpT_extrapCentral * weight_tau2_TauID_SF_Medium_syst_yearCentral * weight_tau2_TauID_SF_Medium_syst_year_dm0Central * weight_tau2_TauID_SF_Medium_syst_year_dm10Central * weight_tau2_TauID_SF_Medium_syst_year_dm11Central * weight_tau2_TauID_SF_Medium_syst_year_dm1Central "
    if "weight_tau1_TauID_SF_Medium_Central_2" not in df.GetColumnNames():
        df = df.Define("weight_tau1_TauID_SF_Medium_Central_2", prod_central_1)
    if "weight_tau2_TauID_SF_Medium_Central_2" not in df.GetColumnNames():
        df = df.Define("weight_tau2_TauID_SF_Medium_Central_2", prod_central_2)
    return df
if __name__ == "__main__":
    inFileName = "/eos/user/a/aciocci/HHbbTauTauRes/anaTuples/v13_deepTau2p1_HTT/SC/Run2_2018/TTToSemiLeptonic/nanoHTT_0.root"
    rdf = ROOT.RDataFrame("Events",inFileName)
    rdf = GetTauIDTotalWeight(rdf)
    print("weight_tau1_TauID_SF_Medium_Central_2" in rdf.GetColumnNames())
    rdf.Filter("weight_tau1_TauID_SF_Medium_Central!=1").Display({"weight_tau1_TauID_SF_Medium_Central","weight_tau1_TauID_SF_Medium_Central_2"}).Print()