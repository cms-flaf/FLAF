import ROOT
import sys
import os
import math
import sys
import os
import math
import shutil
from RunKit.run_tools import ps_call
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

import Common.Utilities as Utilities
from Analysis.HistHelper import *
from Analysis.HistMerger import *

def GetHistograms(inFile, channel, category, uncSource, histNamesDict,all_histlist, wantData):
    inFile = ROOT.TFile(inFile,"READ")
    dir_0 = inFile.Get(channel)
    dir_1 = dir_0.Get(category)
    for key in dir_1.GetListOfKeys():
        obj = key.ReadObj()
        if obj.IsA().InheritsFrom(ROOT.TH1.Class()):
            obj.SetDirectory(0)
            key_name = key.GetName()
            if key_name not in histNamesDict.keys(): continue
            sample, uncName, scale = histNamesDict[key_name]
            sample_type = sample if not sample in sample_cfg_dict.keys() else sample_cfg_dict[sample]['sampleType']
            if sample_type in signals:
                sample= sample_type
            if (uncName, scale) not in all_histlist.keys():
                all_histlist[(uncName, scale)] = {}
            #if wantData == False and (sample == 'data' or sample_type == 'data'): continue
            all_histlist[(uncName, scale)][sample] = obj
    inFile.Close()

def GetSignalHistogram(inFileSig, channel, category, uncSource, histNamesDict,all_histlist, mass):
    inFileSignal = ROOT.TFile(inFileSig,"READ")
    dir_0Signal = inFileSignal.Get(channel)
    dir_qcdSignal = dir_0Signal.Get('OS_Iso')
    dir_1Signal = dir_qcdSignal.Get(category)
    for key in dir_1Signal.GetListOfKeys():
        objSignal = key.ReadObj()
        if objSignal.IsA().InheritsFrom(ROOT.TH1.Class()):
            objSignal.SetDirectory(0)
            key_name = key.GetName()
            key_name_split = key_name.split('_')
            if uncSource == 'Central' and len(key_name_split)>1 : continue
            else:
                key_name = key_name.split('_')[0]
            sample = key_name
            key_name += f'ToHHTo2B2Tau_M-{mass}'
            print(key_name)
            print(histNamesDict.keys())
            print(key_name in histNamesDict.keys())
            if key_name not in histNamesDict.keys(): continue
            sampleName, uncName, scale = histNamesDict[key_name]
            #sample_type = sample if not sample in sample_cfg_dict.keys() else sample_cfg_dict[sample]['sampleType']
            #if sample_type in signals:
            #    sample= sample_type
            if (uncName, scale) not in all_histlist.keys():
                all_histlist[(uncName, scale)] = {}
            #if args.wantData == False and (sample == 'data' or sample_type == 'data'): continue
            all_histlist[(uncName, scale)][sample] = objSignal
    inFileSignal.Close()

if __name__ == "__main__":
    import argparse
    import PlotKit.Plotter as Plotter
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--outFile', required=True)
    parser.add_argument('--inFile', required=True, type=str)
    parser.add_argument('--inFileRadion', required=True, type=str)
    parser.add_argument('--inFileGraviton', required=True, type=str)
    parser.add_argument('--var', required=False, type=str, default = 'tau1_pt')
    parser.add_argument('--mass', required=False, type=int, default=1250)
    parser.add_argument('--sampleConfig', required=True, type=str)
    parser.add_argument('--channel',required=False, type=str, default = 'tauTau')
    parser.add_argument('--category',required=False, type=str, default = 'inclusive')
    parser.add_argument('--wantData', required=False, type=bool, default=False)
    parser.add_argument('--uncSource', required=False, type=str,default='Central')
    parser.add_argument('--year', required=False, type=str,default='Run2_2018')
    args = parser.parse_args()

    page_cfg = os.path.join(os.environ['ANALYSIS_PATH'],"config/plot/cms_stacked.yaml")
    page_cfg_custom = os.path.join(os.environ['ANALYSIS_PATH'],"config/plot/Run2_2018.yaml")
    hist_cfg = os.path.join(os.environ['ANALYSIS_PATH'],"config/plot/histograms.yaml")
    with open(hist_cfg, 'r') as f:
        hist_cfg_dict = yaml.safe_load(f)
    if "x_rebin" in hist_cfg_dict[args.var].keys():
        hist_cfg_dict["x_bins"] = hist_cfg_dict[args.var]["x_rebin"][args.channel][args.category]
    with open(page_cfg_custom, 'r') as f:
        page_cfg_custom_dict = yaml.safe_load(f)
    inputs_cfg = os.path.join(os.environ['ANALYSIS_PATH'],"config/plot/inputs.yaml")
    with open(inputs_cfg, 'r') as f:
        inputs_cfg_dict = yaml.safe_load(f)

    #if args.wantData == False:
    #    for dicti in inputs_cfg_dict:
    #        if dicti['name'] == 'data':
    #            inputs_cfg_dict.remove(dicti)
        #print(inputs_cfg_dict)
    samples_list = [ s['name'] for s in inputs_cfg_dict]
    with open(args.sampleConfig, 'r') as f:
        sample_cfg_dict = yaml.safe_load(f)
    all_samples_separated = [k['name'] for k in inputs_cfg_dict]
    #print(all_samples_separated)
    all_histlist = {}

    plotter = Plotter.Plotter(page_cfg=page_cfg, page_cfg_custom=page_cfg_custom, hist_cfg=hist_cfg_dict, inputs_cfg=inputs_cfg_dict)

    wantSignals=True
    wantAllMasses=False
    wantOneMass=True

    #all_samples_list,all_samples_types = GetSamplesStuff(sample_cfg_dict,args.histDir,True,False,True,args.mass)
    all_samples_list,all_samples_types = GetSamplesStuff(sample_cfg_dict, wantSignals, wantAllMasses, wantOneMass, args.mass)
    all_samples_types.update({"QCD":"QCD"})
    #print(all_samples_types)
    histNamesDict = {}
    signals = list(sample_cfg_dict['GLOBAL']['signal_types'])
    CreateNamesDict(histNamesDict, all_samples_types, args.uncSource, scales, sample_cfg_dict)
    #print(histNamesDict)


    GetHistograms(args.inFile, args.channel, args.category, args.uncSource, histNamesDict,all_histlist, args.wantData)
    GetSignalHistogram(args.inFileRadion, args.channel, args.category, args.uncSource, histNamesDict,all_histlist, args.mass)
    GetSignalHistogram(args.inFileGraviton, args.channel, args.category, args.uncSource, histNamesDict,all_histlist, args.mass)

    #print(os.path.join(args.histDir, inFileName))

    print(all_histlist.items(), all_histlist.keys())

    hists_to_plot = {}

    print(all_samples_types.keys())
    for uncScale in scales+['Central']:
        if args.uncSource == 'Central' and uncScale != 'Central': continue
        if args.uncSource != 'Central' and uncScale == 'Central': continue
        key = ( args.uncSource, uncScale )
        if key not in all_histlist.keys(): continue
        samples_dict = all_histlist[key]
        obj_list = ROOT.TList()
        for sample,hist in samples_dict.items():
            #print(sample, hist.Integral(0, hist.GetNbinsX()+1))
            other_inputs = []
            if sample not in all_samples_separated+sample_cfg_dict['GLOBAL']['signal_types']:
                print(sample)
                if sample in other_inputs: continue
                if not other_inputs:
                    other_obj = hist
                other_inputs.append(sample)
                #sample = 'Other'
                #print(f'{sample} in other has {hist.Integral(0, hist.GetNbinsX()+1)} entries')
                obj_list.Add(hist)
                #obj_list=ROOT.TList()
            else:
                #print(f'{sample} has {hist.Integral(0, hist.GetNbinsX()+1)} entries')
                hists_to_plot[sample] = hist
        other_obj.Merge(obj_list)
        #print(f'other have {other_obj.GetEntries()} entries')
        hists_to_plot['Other'] = other_obj
        #print()
        #for sample,hist in hists_to_plot.items():
        #    print(sample, hist.Integral(0, hist.GetNbinsX()+1))
        #print()
        #print()
        #if args.wantData == True and 'data' not in hists_to_plot.keys():
        #    hists_to_plot['data'] = all_histlist[ ('Central','Central' )]['data']
        cat_txt = args.category if args.category !='inclusive' else 'incl'
        custom1= {'cat_text':f"{cat_txt} m_{{X}}={args.mass} GeV/c^{{2}}", 'ch_text':page_cfg_custom_dict['channel_text'][args.channel], 'datasim_text':'CMS data/simulation','scope_text':''}

        if args.wantData==False:
            custom1= {'cat_text':f"{cat_txt} m_{{X}}={args.mass} GeV/c^{{2}}", 'ch_text':page_cfg_custom_dict['channel_text'][args.channel], 'datasim_text':'CMS simulation', 'scope_text':''}

        plotter.plot(args.var, hists_to_plot, args.outFile, want_data = args.wantData, custom=custom1)
        print(args.outFile)

        '''
        outFile_prefix = f'{args.var}_{args.channel}_{args.category}'

        if(args.uncSource != 'Central'):
            outFile_prefix += f'_{args.uncSource}{uncScale}'
        tmpdir = 'from2D' if args.want2D else 'from1D'
        outDir = os.path.join(args.outDir,args.var,btag_dir,tmpdir)
        #print(outDir)
        #print(hists_to_plot)
        if not os.path.isdir(outDir):
            os.makedirs(outDir)
        outFileName = f"{outFile_prefix}_XMass{args.mass}.pdf"

        outFileFullPath = os.path.join(outDir,outFileName)
        plotter.plot(args.var, hists_to_plot, outFileFullPath, want_data = args.wantData, custom=custom1)
        print(outFileFullPath)
        '''