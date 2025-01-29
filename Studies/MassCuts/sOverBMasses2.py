import ROOT
import os
import sys
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

import argparse


if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

import Common.Utilities as Utilities
from Analysis.HistHelper import *
from Analysis.hh_bbtautau import *
from Studies.MassCuts.Square.GetSquareIntervals import *
from Studies.MassCuts.Square.SquarePlot import *


def createCacheQuantities(dfWrapped_cache, cache_map_name):
    df_cache = dfWrapped_cache.df
    map_creator_cache = ROOT.analysis.CacheCreator(*dfWrapped_cache.colTypes)()
    df_cache = map_creator_cache.processCache(ROOT.RDF.AsRNode(df_cache), Utilities.ListToVector(dfWrapped_cache.colNames), cache_map_name)
    return df_cache


def AddCacheColumnsInDf(dfWrapped_central, dfWrapped_cache,cache_map_name='cache_map_placeholder'):
    col_names_cache =  dfWrapped_cache.colNames
    col_types_cache =  dfWrapped_cache.colTypes
    dfWrapped_cache.df = createCacheQuantities(dfWrapped_cache, cache_map_name)
    if dfWrapped_cache.df.Filter(f"{cache_map_name} > 0").Count().GetValue() <= 0 : raise RuntimeError("no events passed map placeolder")
    dfWrapped_central.AddCacheColumns(col_names_cache,col_types_cache)


def FilterForbJets(cat,dfWrapper_s,dfWrapper_b):
    if cat == 'boosted':
        dfWrapper_s.df = dfWrapper_s.df.Define("FatJet_atLeast1BHadron",
        "SelectedFatJet_nBHadrons>0").Filter("SelectedFatJet_p4[FatJet_atLeast1BHadron].size()>0")
        dfWrapper_b.df = dfWrapper_b.df.Define("FatJet_atLeast1BHadron",
        "SelectedFatJet_nBHadrons>0").Filter("SelectedFatJet_p4[FatJet_atLeast1BHadron].size()>0")
    else:
        dfWrapper_s.df = dfWrapper_s.df.Filter("b1_hadronFlavour==5 && b2_hadronFlavour==5 ")
        dfWrapper_b.df = dfWrapper_b.df.Filter("b1_hadronFlavour==5 && b2_hadronFlavour==5 ")
    return dfWrapper_s,dfWrapper_b

def GetModel2D(x_bins, y_bins):#hist_cfg, var1, var2):
    #x_bins = hist_cfg[var1]['x_bins']
    #y_bins = hist_cfg[var2]['x_bins']
    if type(x_bins)==list:
        x_bins_vec = Utilities.ListToVector(x_bins, "double")
        if type(y_bins)==list:
            y_bins_vec = Utilities.ListToVector(y_bins, "double")
            model = ROOT.RDF.TH2DModel("", "", x_bins_vec.size()-1, x_bins_vec.data(), y_bins_vec.size()-1, y_bins_vec.data())
        else:
            n_y_bins, y_bin_range = y_bins.split('|')
            y_start,y_stop = y_bin_range.split(':')
            model = ROOT.RDF.TH2DModel("", "", x_bins_vec.size()-1, x_bins_vec.data(), int(n_y_bins), float(y_start), float(y_stop))
    else:
        n_x_bins, x_bin_range = x_bins.split('|')
        x_start,x_stop = x_bin_range.split(':')
        if type(y_bins)==list:
            y_bins_vec = Utilities.ListToVector(y_bins, "double")
            model = ROOT.RDF.TH2DModel("", "",int(n_x_bins), float(x_start), float(x_stop), y_bins_vec.size()-1, y_bins_vec.data())
        else:
            n_y_bins, y_bin_range = y_bins.split('|')
            y_start,y_stop = y_bin_range.split(':')
            model = ROOT.RDF.TH2DModel("", "",int(n_x_bins), float(x_start), float(x_stop), int(n_y_bins), float(y_start), float(y_stop))
    return model

def getDataFramesFromFile(infile, mass=None,resonance="Radion"):
    my_file = open(infile, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    new_data = []
    if mass != None:
        new_data = [line.strip() for line in data.splitlines() if f"M-{mass}" in line and resonance in line]
    else:
        new_data = data_into_list
    # print(new_data)
    inFiles = Utilities.ListToVector(new_data)
    df_initial = ROOT.RDataFrame("Events", inFiles)
    # print(df_initial.Count().GetValue())
    return df_initial


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=False, type=str, default='2018')
    parser.add_argument('--cat', required=False, type=str, default='res2b_cat3')
    parser.add_argument('--channel', required=False, type=str, default='tauTau')
    parser.add_argument('--wantPlots', required=False, type=bool, default=False)
    parser.add_argument('--resonance', required=False, type=str, default="Radion")
    parser.add_argument('--masses', required=False, type=str, default='2000')
    args = parser.parse_args()
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
    ROOT.gInterpreter.Declare(f'#include "include/KinFitNamespace.h"')
    ROOT.gInterpreter.Declare(f'#include "include/HistHelper.h"')
    ROOT.gInterpreter.Declare(f'#include "include/Utilities.h"')
    ROOT.gROOT.ProcessLine('#include "include/AnalysisTools.h"')
    ROOT.gInterpreter.Declare(f'#include "include/pnetSF.h"')

    signalFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/SignalSamples_{args.year}.txt"
    signalCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/SignalCaches_{args.year}.txt"
    TTFiles = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/TTSamples_{args.year}.txt"
    TTCaches = f"/afs/cern.ch/work/v/vdamante/FLAF/Studies/MassCuts/Inputs/TTCaches_{args.year}.txt"


    global_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/HH_bbtautau/global.yaml'
    global_cfg_dict = {}
    with open(global_cfg_file, 'r') as f:
        global_cfg_dict = yaml.safe_load(f)
    global_cfg_dict['channels_to_consider']=args.channel.split(',')

    hist_cfg_file = '/afs/cern.ch/work/v/vdamante/FLAF/config/plot/histograms.yaml'
    hist_cfg_dict = {}
    with open(hist_cfg_file, 'r') as f:
        hist_cfg_dict = yaml.safe_load(f)


    df_bckg = getDataFramesFromFile(TTFiles)
    df_bckg_cache = getDataFramesFromFile(TTCaches)
    dfWrapped_bckg = DataFrameBuilderForHistograms(df_bckg,global_cfg_dict, f"Run2_{args.year}")
    dfWrapped_bckg_cache =  DataFrameBuilderForHistograms(df_bckg_cache,global_cfg_dict, f"Run2_{args.year}")#(df_sig_cache,global_cfg_dict,args.year)
    AddCacheColumnsInDf(dfWrapped_bckg, dfWrapped_bckg_cache, "cache_map_Central")
    dfWrapped_bckg =  PrepareDfForHistograms(dfWrapped_bckg)




    filter_base = f"OS_Iso && {args.channel} && {args.cat}"
    old_linCut = f"( bb_m_vis > 40 && bb_m_vis < 270 ) && (tautau_m_vis > 15 && tautau_m_vis < 130 )"
    new_linCut = f"( bb_m_vis > 90 && bb_m_vis < 160 ) && (tautau_m_vis > 30 && tautau_m_vis < 100 )"
    svFit_ellyptic = "(((SVfit_m-116)*(SVfit_m-116)/(35*35)) + ((bb_m_vis-111)*(bb_m_vis-111)/(45*45))) < 1 "
    old_linCut_SVFit = f"( bb_m_vis > 40 && bb_m_vis < 270 ) && (SVfit_m > 15 && SVfit_m < 130 )"
    new_linCut_SVFit = f"( bb_m_vis > 90 && bb_m_vis < 160 ) && (SVfit_m > 30 && SVfit_m < 100 )"
    dfWrapped_bckg.df = dfWrapped_bckg.df.Filter(filter_base)
    dfWrapped_bckg.df = dfWrapped_bckg.df.Range(1000000)
    soverb_nocut = []
    soverb_old_linCut = []
    soverb_new_linCut = []
    soverb_svFit_ellyptic = []
    soverb_old_linCut_SVFit = []
    soverb_new_linCut_SVFit = []
    masses = args.masses.split(',')
    for mass in masses:

        df_sig = getDataFramesFromFile(signalFiles,mass,args.resonance)
        df_sig_cache = getDataFramesFromFile(signalCaches,mass,args.resonance)
        dfWrapped_sig = DataFrameBuilderForHistograms(df_sig,global_cfg_dict, f"Run2_{args.year}")
        dfWrapped_sig_cache =  DataFrameBuilderForHistograms(df_sig_cache,global_cfg_dict, f"Run2_{args.year}")#(df_sig_cache,global_cfg_dict,args.year)
        AddCacheColumnsInDf(dfWrapped_sig, dfWrapped_sig_cache, "cache_map_Central")
        dfWrapped_sig =  PrepareDfForHistograms(dfWrapped_sig)


        dfWrapped_sig.df = dfWrapped_sig.df.Filter(filter_base)
        dfWrapped_sig.df = dfWrapped_sig.df.Range(1000000)
        soverb_nocut.append( dfWrapped_sig.df.Filter(filter_base).Count().GetValue()/math.sqrt(dfWrapped_bckg.df.Filter(filter_base).Count().GetValue()))
        soverb_old_linCut.append(  dfWrapped_sig.df.Filter(old_linCut).Count().GetValue()/math.sqrt(dfWrapped_bckg.df.Filter(old_linCut).Count().GetValue()))
        soverb_new_linCut.append(  dfWrapped_sig.df.Filter(new_linCut).Count().GetValue()/math.sqrt(dfWrapped_bckg.df.Filter(new_linCut).Count().GetValue()))
        soverb_svFit_ellyptic.append(  dfWrapped_sig.df.Filter(svFit_ellyptic).Count().GetValue()/math.sqrt(dfWrapped_bckg.df.Filter(svFit_ellyptic).Count().GetValue()))

        # soverb_old_linCut_SVFit.append(  dfWrapped_sig.df.Filter(old_linCut_SVFit).Count().GetValue()/math.sqrt(dfWrapped_bckg.df.Filter(old_linCut_SVFit).Count().GetValue()))
        # soverb_new_linCut_SVFit.append(  dfWrapped_sig.df.Filter(new_linCut_SVFit).Count().GetValue()/math.sqrt(dfWrapped_bckg.df.Filter(new_linCut_SVFit).Count().GetValue()))
        # soverb_svFit_ellyptic.append(  dfWrapped_sig.df.Filter(svFit_ellyptic).Count().GetValue()/math.sqrt(dfWrapped_bckg.df.Filter(svFit_ellyptic).Count().GetValue()))
    print(args.year, args.cat, args.channel)
    print(masses)
    print("noCut")
    print(soverb_nocut)
    print("old linear cut")
    print(soverb_old_linCut)
    print("new linear cut")
    print(soverb_new_linCut)
    print("elliptic mass cut")
    print(soverb_svFit_ellyptic)

    print(f"old linear cut with svFit")
    print(soverb_old_linCut_SVFit)
    print(f"new linear cut with svFit")
    print(soverb_new_linCut_SVFit)
    # # Stile CMS (o altri disponibili)
    # hep.style.use("CMS")

    # # Dati di esempio
    # # Crea la figura
    # plt.figure(figsize=(8, 6))

    # # Disegna i grafici con marker e colori diversi
    # plt.plot(masses, soverb_noCut, label="No cut", color="red", marker="o", linestyle="-")
    # plt.plot(masses, soverb_old_linCut, label="Old cut", color="blue", marker="s", linestyle="--")
    # plt.plot(masses, soverb_new_linCut, label="New cut", color="green", marker="^", linestyle=":")
    # # plt.plot(masses, gr_svFit_ellyptic, label="Ellyptic", color="magenta", marker="d", linestyle="-.")

    # # Etichette degli assi
    # plt.xlabel("Mass")
    # plt.ylabel("s/\\sqrt{b}")

    # # Aggiungi legenda
    # plt.legend(loc="best")

    # # Titolo (opzionale)
    # plt.title("Overlay Graphs Example")

    # # Mostra il plot con stile HEP
    # hep.cms.text("Preliminary", loc=0)  # Puoi personalizzare la posizione
    # hep.cms.lumitext("Run 2, 13 TeV")   # Informazioni aggiuntive

    # # Mostra la figura
    # # plt.show()
    # filename='sqrtsbprova'
    # plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    # plt.savefig(f"{filename}.png", bbox_inches="tight")
    # print(soverb_nocut)
    # len_masses = len(masses)
    # canvas = ROOT.TCanvas("", "", 800, 600)
    # gr_noCut = ROOT.TGraph(len_masses, Utilities.ListToVector(masses), Utilities.ListToVector(soverb_noCut))
    # gr_old_linCut = ROOT.TGraph(len_masses, Utilities.ListToVector(masses), Utilities.ListToVector(soverb_old_linCut))
    # gr_new_linCut = ROOT.TGraph(len_masses, Utilities.ListToVector(masses), Utilities.ListToVector(soverb_new_linCut))
    # gr_svFit_ellyptic = ROOT.TGraph(len_masses, Utilities.ListToVector(masses), Utilities.ListToVector(soverb_svFit_ellyptic))

    # # Personalizza colori e marker
    # gr_noCut.SetLineColor(ROOT.kRed)
    # gr_noCut.SetMarkerColor(ROOT.kRed)
    # gr_noCut.SetMarkerStyle(20)

    # gr_old_linCut.SetLineColor(ROOT.kBlue)
    # gr_old_linCut.SetMarkerColor(ROOT.kBlue)
    # gr_old_linCut.SetMarkerStyle(21)

    # gr_new_linCut.SetLineColor(ROOT.kGreen)
    # gr_new_linCut.SetMarkerColor(ROOT.kGreen)
    # gr_new_linCut.SetMarkerStyle(22)

    # gr_svFit_ellyptic.SetLineColor(ROOT.kMagenta)
    # gr_svFit_ellyptic.SetMarkerColor(ROOT.kMagenta)
    # gr_svFit_ellyptic.SetMarkerStyle(23)

    # # Disegna i grafici sovrapposti
    # gr_noCut.Draw("AP")  # A: Asse, P: Marker, L: Linea
    # gr_old_linCut.Draw("P SAME")  # SAME per sovrapporre
    # gr_new_linCut.Draw("P SAME")
    # gr_svFit_ellyptic.Draw("P SAME")

    # # Aggiungi una legenda
    # legend = ROOT.TLegend(0.1, 0.7, 0.4, 0.9)
    # legend.AddEntry(gr_noCut, "No cut", "PL")
    # legend.AddEntry(gr_old_linCut, "Old cut", "PL")
    # legend.AddEntry(gr_new_linCut, "New cut", "PL")
    # legend.AddEntry(gr_svFit_ellyptic, "Ellyptic", "PL")
    # legend.Draw()

    # # Mostra la canvas
    # canvas.Draw()
