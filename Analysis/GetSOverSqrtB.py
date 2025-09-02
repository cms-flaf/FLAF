import ROOT
import matplotlib.pyplot as plt
import sys
import os
import math
import numpy as np
from scipy import stats
from FLAF.RunKit.run_tools import ps_call
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])
'''
for s/sqrt(b) it is more robust to take sum of all bins where err_b/b small enough (e.g. < 0.1 or 0.2). I.e.
significance = sqrt( sum_i ( s_i^2 / b_i ) ), where b_i > 0 && err_b_i / b_i < 0.2
'''
all_masses = [250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000]

from Analysis.HistHelper import *
def GetHisto(channel, category, inFile, sample_name, uncSource, scale):
    dir_0 = inFile.Get(channel)
    dir_1 = dir_0.Get(category)
    total_histName = sample_name
    if uncSource != 'Central':
        total_histName += f'_{uncSource}{scale}'
    for key in dir_1.GetListOfKeys():
        key_name = key.GetName()
        #print(key)
        if key_name != total_histName: continue
        obj = key.ReadObj()
        if obj.IsA().InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)
            return obj
    return

def GetBinValues(histogram):
    mean, sigma, integral = histogram.GetMean(),histogram.GetStdDev(),histogram.Integral(0,histogram.GetNbinsX())
    #maximum_bin = histogram.GetMaximumBin()
    #maximum = histogram.GetBinCenter(maximum_bin)
    for x_bin in range(0,histogram.GetNbinsX()):
        bin_center = histogram.GetXaxis().GetBinCenter(x_bin)
        bin_center_prev = histogram.GetXaxis().GetBinCenter(x_bin-1) if x_bin > 0 else 0
        bin_value = histogram.GetBinContent(x_bin)
        bin_value_prev = histogram.GetBinContent(x_bin-1) if x_bin > 0 else 0
        bin_width = (bin_center - bin_center_prev)/2
        #print(x_bin, bin_center,bin_center_prev,bin_width)
        if  sigma < bin_width:
            sigma = bin_width
        if mean <= bin_center and mean > bin_center_prev :
            break
    #print('mean=',mean)
    #print('sigma=',sigma)
    #print(mean-sigma/2, mean+sigma/2)
    sigma_norm = round(sigma/bin_width,0) * bin_width
    return mean, sigma_norm, mean-sigma_norm/2, mean+sigma_norm/2

def GetBinsForIntegral(histogram,lo_bin, up_bin):
    bins = []
    #print(histogram.GetNbinsX())
    #print(lo_bin, up_bin)
    for x_bin in range(0,histogram.GetNbinsX()+1):
        center = histogram.GetXaxis().GetBinCenter(x_bin)
        #print(center)
        if center >= lo_bin and center <= up_bin:
            bins.append(x_bin)
    return bins

def fit_function(x, par):
    return par[0] + par[1] * x[0]

def constant_function(x,par):
    return par[0]

def getSignificance(hist_signal, hist_bckg):
    bins_to_consider = []
    sigma2 = 0
    sigma = 0
    b= 0
    s2 = 0
    bins_errors = []
    bins_values = []
    bins_centers = []
    if hist_signal.GetNbinsX() != hist_bckg.GetNbinsX() :
        print("errore della vita")
        return 0
    for x_bin in range(0,hist_bckg.GetNbinsX()+1):
        bin_center = hist_bckg.GetXaxis().GetBinCenter(x_bin)
        bin_value = hist_bckg.GetBinContent(x_bin)
        bin_error = hist_bckg.GetBinError(x_bin)
        if bin_value!=0. and (bin_error/bin_value) < 0.2:
            bins_to_consider.append(x_bin)
            bins_centers.append(bin_center)
            bins_values.append(bin_value)
            bins_errors.append(bin_error)

    for bin_to_consider in bins_to_consider:
        s2 += hist_signal.GetBinContent(bin_to_consider)*hist_signal.GetBinContent(bin_to_consider)
        b += hist_bckg.GetBinContent(bin_to_consider)
        sigma2 += s2/b if b!=0. else 0.
    sigma = math.sqrt(sigma2) if sigma2 >=0. else 0.
    return bins_to_consider, bins_centers, bins_values, bins_errors, sigma2, sigma, math.sqrt(s2), math.sqrt(b)

if __name__ == "__main__":
    import argparse
    import json
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--histDir', required=True)
    parser.add_argument('--outDir', required=True)
    parser.add_argument('--plotDir', required=True)
    parser.add_argument('--inFileName', required=True)
    parser.add_argument('--bckgConfig', required=True, type=str)
    parser.add_argument('--sampleConfig', required=True, type=str)
    parser.add_argument('--uncConfig', required=True, type=str)
    parser.add_argument('--wantBTag', required=False, type=bool, default=False)
    parser.add_argument('--want2D', required=False, type=bool, default=False)
    parser.add_argument('--suffix', required=False, type=str, default='')
    parser.add_argument('--var', required=False, type=str, default='kinFit_m')
    args = parser.parse_args()
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptStat(0)


    btag_dir= "bTag_WP" if args.wantBTag else "bTag_shape"
    hist_cfg = os.path.join(os.environ['ANALYSIS_PATH'],"config/plot/histograms.yaml")
    with open(hist_cfg, 'r') as f:
        hist_cfg_dict = yaml.safe_load(f)

    with open(args.sampleConfig, 'r') as f:
        sample_cfg_dict = yaml.safe_load(f)

    with open(args.bckgConfig, 'r') as f:
        bckg_cfg_dict = yaml.safe_load(f)

    signals = list(sample_cfg_dict['GLOBAL']['signal_types'])
    all_samples_list,all_samples_types = GetSamplesStuff(bckg_cfg_dict.keys(),sample_cfg_dict, os.path.join(args.histDir,f'v9_deepTau2p1'),False)
    for deepTauVersion in ['2p1', '2p5']:
        inDir = os.path.join(args.histDir,f'v9_deepTau{deepTauVersion}', 'all_histograms',args.var,btag_dir)
        outDir = os.path.join(args.outDir,f'v9_deepTau{deepTauVersion}', 's_sqrt_b',args.var, btag_dir)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        TwoDTrue = args.want2D and not args.wantBTag
        inFileName_prefix =  f"all_histograms_{args.var}2D" if TwoDTrue else f"all_histograms_{args.var}"
        inFileName = os.path.join(inDir, f'{inFileName_prefix}{args.suffix}.root')
        #print(os.path.join(inDir,inFileName))
        inFile = ROOT.TFile(os.path.join(inDir, inFileName),"READ")
        if not os.path.exists(os.path.join(inDir,inFileName)):
            print(f'{os.path.join(inDir,inFileName)} does not exist')
            continue
        for signal in ['GluGluToRadion', 'GluGluToBulkGraviton']:
            mass_dict = {}
            mass_dict[signal] = {}
            for channel in ['eTau', 'muTau', 'tauTau']:
                mass_dict[signal][channel] = {}
                for category in ['inclusive','res2b','res1b']:
                    hist_bckg_total = GetHisto(channel, category, inFile, 'QCD', 'Central','-')
                    hists_to_merge = ROOT.TList()
                    mass_dict[signal][channel][category] = {}
                    s_sqrtb = []
                    mass_dict[signal][channel][category]= {}
                    for sample in all_samples_list:
                        sample_type = sample if not sample in sample_cfg_dict.keys() else sample_cfg_dict[sample]['sampleType']
                        #print(signals)
                        if sample_type in signals or sample_type=='data':
                            #print(f'ignoring {sample_type}')
                            continue
                        hists_to_merge.append(GetHisto(channel, category, inFile, sample, 'Central','-'))
                        integral_central = 0
                    hist_bckg_total.Merge(hists_to_merge)
                    for mass in all_masses:
                        hist_signal = GetHisto(channel, category, inFile, f'{signal}ToHHTo2B2Tau_M-{mass}', 'Central','-')
                        mass_dict[signal][channel][category][mass] = {}
                        bins_to_consider, bins_centers, bins_values, bins_errors, sigma2, sigma, s, sqrtb = getSignificance(hist_signal, hist_bckg_total)
                        mass_dict[signal][channel][category][mass]['bins_to_consider'] = bins_to_consider
                        mass_dict[signal][channel][category][mass]['bins_centers'] = bins_centers
                        mass_dict[signal][channel][category][mass]['bins_values'] = bins_values
                        mass_dict[signal][channel][category][mass]['bins_errors'] = bins_errors
                        mass_dict[signal][channel][category][mass]['sigma2'] = sigma2
                        mass_dict[signal][channel][category][mass]['sigma'] = sigma
                        mass_dict[signal][channel][category][mass]['s'] = s
                        mass_dict[signal][channel][category][mass]['sqrtb'] = sqrtb
            with open(f'{outDir}/sOverB_{signal}.json', 'w') as f:
                json.dump(mass_dict[signal], f, indent=4)
        inFile.Close()



    outDir_2p1 = os.path.join(args.outDir,f'v9_deepTau2p1',  's_sqrt_b',args.var, btag_dir)
    outDir_2p5= os.path.join(args.outDir,f'v9_deepTau2p5',  's_sqrt_b',args.var, btag_dir)
    pltDir = os.path.join(args.plotDir,'DeepTau2p1vsDeepTau2p5')
    plt_ratios_Dir = os.path.join(args.plotDir,'DeepTau2p1vsDeepTau2p5_ratios')
    if not os.path.exists(pltDir):
        os.makedirs(pltDir)
    if not os.path.exists(plt_ratios_Dir):
        os.makedirs(plt_ratios_Dir)

    print("\hline")
    print("\hline \\\\[-0.9em]")
    print(f"""signal & mass & category & percentage diff & version & $s/\sqrt{{b}}$ & s& $\sqrt{{b}}$ \\\\""") #& bins & bins values & bins centers & bins errors  \\\\""")
    print("\hline")
    print("\hline")
    k = 0
    j = 0
    for signal in ['GluGluToRadion', 'GluGluToBulkGraviton']:
        dict_2p1 = {}
        dict_2p5 = {}
        with open(f'{outDir_2p1}/sOverB_{signal}.json', 'r') as f:
            dict_2p1=json.load(f)
        with open(f'{outDir_2p5}/sOverB_{signal}.json', 'r') as f:
            dict_2p5=json.load(f)
        for channel in ['eTau', 'muTau', 'tauTau']:
            for category in ['inclusive','res2b','res1b']:
                all_sqrts_2p1=[]
                all_sqrts_2p5=[]
                all_ratios_2p5_2p1 = []
                all_masses_hist = []
                all_masses_ratios_hist = []
                for mass in all_masses:
                    j+=1
                    ratio_2p5_2p1 = dict_2p5[channel][category][f'{mass}']['sigma']/dict_2p1[channel][category][f'{mass}']['sigma'] if dict_2p1[channel][category][f'{mass}']['sigma'] != 0. else dict_2p5[channel][category][f'{mass}']['sigma']
                    if ratio_2p5_2p1!=0:
                        all_ratios_2p5_2p1.append(ratio_2p5_2p1)
                        all_masses_ratios_hist.append(mass)
                    ssqrtb_2p1 = round(dict_2p1[channel][category][f'{mass}']['sigma'],2)
                    ssqrtb_2p5 = round(dict_2p5[channel][category][f'{mass}']['sigma'],2)

                    if ssqrtb_2p1!=0 or ssqrtb_2p5!=0:
                        all_masses_hist.append(mass)
                        all_sqrts_2p1.append(dict_2p1[channel][category][f'{mass}']['sigma'])
                        all_sqrts_2p5.append(dict_2p5[channel][category][f'{mass}']['sigma'])

                    s_2p1 = round(dict_2p1[channel][category][f'{mass}']['s'],2)
                    s_2p5 = round(dict_2p5[channel][category][f'{mass}']['s'],2)

                    sqrtb_2p1 = round(dict_2p1[channel][category][f'{mass}']['sqrtb'],2)
                    sqrtb_2p5 = round(dict_2p5[channel][category][f'{mass}']['sqrtb'],2)

                    dict_mass_2p5 = dict_2p5[channel][category][f'{mass}']
                    dict_mass_2p1 = dict_2p1[channel][category][f'{mass}']

                    bins_centers_2p1 = dict_2p1[channel][category][f'{mass}']['bins_centers']
                    bins_centers_2p5 = dict_2p5[channel][category][f'{mass}']['bins_centers']
                    bins_values_2p1 = dict_2p1[channel][category][f'{mass}']['bins_values']
                    bins_values_2p5 = dict_2p5[channel][category][f'{mass}']['bins_values']
                    bins_errors_2p1 = dict_2p1[channel][category][f'{mass}']['bins_errors']
                    bins_errors_2p5 = dict_2p5[channel][category][f'{mass}']['bins_errors']
                    bins_to_consider_2p1 = dict_2p1[channel][category][f'{mass}']['bins_to_consider']
                    bins_to_consider_2p5 = dict_2p5[channel][category][f'{mass}']['bins_to_consider']
                    if ssqrtb_2p5 != 0. and (ssqrtb_2p1>ssqrtb_2p5):
                        percentage_diff = round(100*(ssqrtb_2p1-ssqrtb_2p5)/(ssqrtb_2p1+ssqrtb_2p5),2)
                        if percentage_diff > 5:
                            k+=1
                            print(f""" \multirow{{2}}{{*}}{{{signal}}}  &""")

                            print(f""" \multirow{{2}}{{*}}{{{mass}}}  &""")

                            print(f""" \multirow{{2}}{{*}}{{{category}}}  &""")

                            print(f""" \multirow{{2}}{{*}}{{{percentage_diff}}}  &""")

                            print(f"""2p1 & {ssqrtb_2p1} & {s_2p1} & {sqrtb_2p1} \\\\""") # & & {bins_centers_2p1}{bins_values_2p1} & {bins_values_2p1} & {bins_errors_2p1} \\\\ """)
                            print(f"""& & & & 2p5 & {ssqrtb_2p5} & {s_2p5} & {sqrtb_2p5} \\\\""") # & & {bins_centers_2p5}\\\\""") # & {bins_values_2p5} & {bins_values_2p5} & {bins_errors_2p5} \\\\ """)
                            print("\hline")
                            #print(f"""for signal {signal} mass {mass}, channel {channel}, category {category}:\n the ratio s/sqrt(b) for 2p1 is {ssqrtb_2p1}, and for 2p5 is {ssqrtb_2p5} \n for 2p1 the other values are: \n {dict_mass_2p1} \n for 2p5 the other values are: \n {dict_mass_2p5} """)
                            print()

                plt.title(f'{channel} {category} {args.var}')
                plt.xlabel('X_mass')
                plt.ylabel('S')
                #plt.plot(all_masses, all_ratios_2p5_2p1, "or", label="S(2p5)/S(2p1)")
                plt.plot(all_masses_hist, all_sqrts_2p1, "or", label="S(2p1)")
                plt.plot(all_masses_hist, all_sqrts_2p5, "ob", label="S(2p5)")
                plt.legend(loc="upper right")
                #plt.ylim(-1.5, 2.0)
                plt.savefig(f'{pltDir}/{channel}_{category}_{signal}_s_sqrtb.png')
                plt.clf()

                plt.title(f'{channel} {category} {args.var} ratios')
                plt.xlabel('X_mass')
                plt.ylabel("S(2p5)/S(2p1)")
                #plt.plot(all_masses, all_ratios_2p5_2p1, "or", label="S(2p5)/S(2p1)")
                plt.plot(all_masses_ratios_hist, all_ratios_2p5_2p1, "ob", label="S(2p5)/S(2p1)")
                #plt.legend(loc="upper right")
                #plt.ylim(-1.5, 2.0)
                plt.savefig(f'{plt_ratios_Dir}/{channel}_{category}_{signal}_S_ratios.png')
                plt.clf()
    print(f"times that 2p1 is better than 2p5 are: {k} over {j}")
